from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .joern import JoernRepositoryIndex, RepositoryCall, RepositoryMethod
from .semantics import Candidate, candidate_validation_error
from .source import FunctionSource, normalize_expression, parse_functions, source_language
from .standard_semantics import STANDARD_LEAF_CALLS, effects_for_call
from .symbol_resolution import ResolvedTarget, SymbolResolver


CANDIDATE_MANIFEST_VERSION = 3
DISCOVERY_POLICY_VERSION = 4


_CLEAR_SCALAR_TYPES = {
    "bool", "_Bool", "char", "signedchar", "unsignedchar",
    "short", "shortint", "signedshort", "signedshortint",
    "unsignedshort", "unsignedshortint", "int", "signed", "signedint",
    "unsigned", "unsignedint", "long", "longint", "signedlong",
    "signedlongint", "unsignedlong", "unsignedlongint", "longlong",
    "longlongint", "signedlonglong", "signedlonglongint",
    "unsignedlonglong", "unsignedlonglongint", "float", "double",
    "longdouble", "size_t", "ssize_t",
}


@dataclass(frozen=True)
class CandidateSelection:
    depth: int
    caller: str
    reason: str
    resolution: str


@dataclass(frozen=True)
class CandidateDiscovery:
    candidates: tuple[Candidate, ...]
    selections: dict[tuple[str, str, int, str], CandidateSelection]
    direct_candidates: int
    recursive_candidates: int
    expanded_methods: int
    unresolved_relevant_calls: int


@dataclass(frozen=True)
class BoundaryNeed:
    parameter_indices: tuple[int, ...] = ()
    return_value: bool = False


def candidate_source_fingerprint(candidate: Candidate) -> str:
    function = candidate.function
    payload = (
        function.path
        + "\0"
        + function.name
        + "\0"
        + function.text
        + "\0"
        + hashlib.sha256(function.translation_unit.encode()).hexdigest()
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def candidate_manifest_path(index: JoernRepositoryIndex) -> Path:
    return index.cache_dir / (
        f"{index.sample_key}-{index.index_fingerprint}.candidates.jsonl"
    )


def _type_definitely_pointer(type_text: str) -> bool:
    compact = "".join(type_text.split())
    return any(token in compact for token in ("*", "&", "["))


def _type_may_be_pointer(type_text: str) -> bool:
    compact = "".join(type_text.split())
    if _type_definitely_pointer(type_text):
        return True
    if compact in {"", "ANY", "<empty>"}:
        return True
    return compact not in _CLEAR_SCALAR_TYPES


def _method_can_produce_summary(method: RepositoryMethod) -> bool:
    return (
        method.return_type not in {"void", "<empty>"}
        or any(_type_may_be_pointer(type_text) for type_text in method.parameter_types)
    )


def _identifiers(expression: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_]\w*\b", normalize_expression(expression)))


def _relations_at(function: FunctionSource, line: int) -> dict[str, str]:
    return {
        normalize_expression(left): normalize_expression(right)
        for left, right in function.value_relations_before(line)
    }


def _expression_closure_at(
    function: FunctionSource,
    expressions: tuple[str, ...] | list[str],
    line: int,
) -> set[str]:
    relations = _relations_at(function, line)
    pending: list[str] = []
    for expression in expressions:
        pending.extend(_identifiers(expression))
    relevant: set[str] = set()
    while pending:
        name = pending.pop()
        if name in relevant:
            continue
        relevant.add(name)
        replacement = relations.get(name)
        if replacement:
            pending.extend(_identifiers(replacement) - relevant)
    return relevant


def _returns(function: FunctionSource) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for match in re.finditer(r"\breturn\s+([^;]+);", function.text):
        line = function.start_line + function.text[: match.start()].count("\n")
        result.append((line, match.group(1)))
    return result


def _memory_seed_groups(function: FunctionSource) -> list[tuple[int, tuple[str, ...]]]:
    groups: list[tuple[int, tuple[str, ...]]] = []
    for access in function.direct_memory_accesses():
        expressions = tuple(
            expression
            for expression in (access.buffer, access.extent)
            if expression and expression != "UNBOUNDED"
        )
        if expressions:
            groups.append((access.line, expressions))
    for call in function.calls():
        if call.indirect:
            continue
        for effect in effects_for_call(call):
            expressions = tuple(
                expression
                for expression in (effect.buffer, effect.extent)
                if expression and expression not in {"return", "UNBOUNDED"}
            )
            if expressions:
                groups.append((call.line, expressions))
    return groups


def _dependency_closure(function: FunctionSource) -> set[str]:
    """All values that feed a local memory operation or returned value.

    Reaching definitions are calculated at each real program point.  This is
    intentionally not a function-wide textual relation map: lexical reaching
    definitions outside the statement's enclosing block are not valid facts.
    """
    relevant: set[str] = set()
    for line, expressions in _memory_seed_groups(function):
        relevant.update(_expression_closure_at(function, expressions, line))
    for line, expression in _returns(function):
        relevant.update(_expression_closure_at(function, (expression,), line))
    return relevant


def _merge_need(left: BoundaryNeed | None, right: BoundaryNeed) -> BoundaryNeed:
    if left is None:
        return right
    return BoundaryNeed(
        parameter_indices=tuple(
            sorted(set(left.parameter_indices) | set(right.parameter_indices))
        ),
        return_value=left.return_value or right.return_value,
    )


def _scoped_dependency_closure(
    function: FunctionSource,
    need: BoundaryNeed,
) -> set[str]:
    boundary_names = {
        function.parameters[index]
        for index in need.parameter_indices
        if 0 <= index < len(function.parameters)
    }
    relevant = set(boundary_names)

    if need.return_value:
        for line, expression in _returns(function):
            relevant.update(_expression_closure_at(function, (expression,), line))

    memory_closures = [
        _expression_closure_at(function, expressions, line)
        for line, expressions in _memory_seed_groups(function)
    ]
    changed = True
    while changed:
        changed = False
        for closure in memory_closures:
            if closure & relevant and not closure <= relevant:
                relevant.update(closure)
                changed = True
    return relevant


def _source_call(function: FunctionSource, line: int, name: str):
    calls = function.calls()
    matches = [
        call for call in calls
        if not call.indirect and call.name == name and call.line == line
    ]
    if len(matches) == 1:
        return matches[0]
    named = [call for call in calls if not call.indirect and call.name == name]
    return named[0] if len(named) == 1 else None


def _repository_call_for_source(
    method: RepositoryMethod,
    source_call,
) -> RepositoryCall | None:
    if source_call is None:
        return None
    exact = [
        call for call in method.calls
        if call.name == source_call.name and call.line == source_call.line
    ]
    if len(exact) == 1:
        return exact[0]
    named = [call for call in method.calls if call.name == source_call.name]
    return named[0] if len(named) == 1 else None


def _result_flows_to_return(function: FunctionSource, result: str | None) -> bool:
    if not result:
        return False
    name = normalize_expression(result)
    if not re.fullmatch(r"[A-Za-z_]\w*", name):
        return False
    for line, expression in _returns(function):
        if name in _expression_closure_at(function, (expression,), line):
            return True
    return False


def _call_need(
    function: FunctionSource,
    source_call,
    callee: RepositoryMethod,
    relevant: set[str],
    parameter_scope: set[str],
    *,
    direct_layer: bool,
) -> tuple[str, BoundaryNeed] | None:
    if source_call is None:
        parameter_indices = tuple(
            index
            for index, type_text in enumerate(callee.parameter_types)
            if _type_may_be_pointer(type_text)
        )
        return_needed = callee.return_type not in {"void", "<empty>"}
        if not parameter_indices and not return_needed:
            return None
        return (
            "resolved custom callee retained with conservative boundary demand",
            BoundaryNeed(parameter_indices, return_needed),
        )

    result_name = normalize_expression(source_call.result or "")
    return_needed = bool(
        source_call.returned
        or _result_flows_to_return(function, source_call.result)
        or (result_name and result_name in relevant)
    )

    relevant_arguments: set[int] = set()
    pointer_arguments: set[int] = set()
    for index, argument in enumerate(source_call.arguments[: len(callee.parameter_types)]):
        argument_closure = _expression_closure_at(
            function, (argument,), source_call.line
        )
        if argument_closure & relevant:
            relevant_arguments.add(index)
        if _type_may_be_pointer(callee.parameter_types[index]):
            # A direct custom call can itself establish memory semantics on a
            # pointer argument, so the direct layer must not require that the
            # caller already has a modeled memory operation involving it.
            if direct_layer or argument_closure & parameter_scope or _identifiers(argument) & parameter_scope:
                pointer_arguments.add(index)

    parameter_indices = tuple(sorted(relevant_arguments | pointer_arguments))
    if not parameter_indices and not return_needed:
        return None

    if direct_layer and pointer_arguments:
        reason = "direct custom callee may define caller-visible memory semantics"
    elif source_call.returned or _result_flows_to_return(function, source_call.result):
        reason = "callee result contributes to caller return"
    elif result_name and result_name in relevant:
        reason = "callee result reaches a memory-relevant value"
    elif relevant_arguments:
        reason = "callee argument depends on a memory-relevant value"
    else:
        reason = "caller boundary value flows into summary-capable callee"
    return reason, BoundaryNeed(parameter_indices, return_needed)


def _candidate_key(source: FunctionSource) -> tuple[str, str, int, str]:
    return (source.path, source.name, source.start_line, source.language)


def _add_candidate(
    discovered: dict[tuple[str, str, int, str], Candidate],
    selections: dict[tuple[str, str, int, str], CandidateSelection],
    needs: dict[tuple[str, str, int, str], BoundaryNeed],
    *,
    sample_key: str,
    sources: list[FunctionSource],
    target: ResolvedTarget,
    call_line: int,
    depth: int,
    caller: str,
    reason: str,
    need: BoundaryNeed,
) -> None:
    variant_count = (
        len(sources)
        if len(sources) > 1
        and all(source.preprocessor_group is not None for source in sources)
        else 1
    )
    if target.resolution == "source-macro" and len(sources) > 1:
        variant_count = len(sources)

    for source in sources:
        key = _candidate_key(source)
        existing = discovered.get(key)
        lines = set(existing.call_lines if existing else ())
        lines.add(call_line)
        merged_need = _merge_need(needs.get(key), need)
        needs[key] = merged_need
        variant_group = (
            f"{source.path}:{source.name}:{source.preprocessor_group[0]}-"
            f"{source.preprocessor_group[1]}"
            if variant_count > 1 and source.preprocessor_group is not None
            else (
                f"{source.path}:{source.name}:source-variants"
                if variant_count > 1
                else None
            )
        )
        discovered[key] = Candidate(
            sample_key=sample_key,
            function=source,
            call_lines=tuple(sorted(lines)),
            method_full_name=target.method.full_name,
            variant_group=variant_group,
            variant_count=variant_count,
            required_parameters=merged_need.parameter_indices,
            require_return=merged_need.return_value,
            resolution=target.resolution,
        )
        current = selections.get(key)
        if current is None or depth < current.depth:
            selections[key] = CandidateSelection(
                depth, caller, reason, target.resolution
            )


def _iter_source_calls(function: FunctionSource):
    for call in function.calls():
        if call.indirect:
            continue
        if call.name.startswith("<operator>.") or call.name in STANDARD_LEAF_CALLS:
            continue
        yield call


def discover_relevant_candidates(
    sample_key: str,
    index: JoernRepositoryIndex,
    entry_method: RepositoryMethod,
    entry: FunctionSource,
) -> CandidateDiscovery:
    """Discover caller-observable custom semantics with bounded recursion.

    The entry layer favors recall: every uniquely resolved custom callee that
    can expose pointer or return semantics is considered. Recursive expansion
    is demand-driven and only follows values connected to the incoming caller
    boundary. This separates first-hop recall from deep call-graph explosion.
    """
    resolver = SymbolResolver(index)
    discovered: dict[tuple[str, str, int, str], Candidate] = {}
    selections: dict[tuple[str, str, int, str], CandidateSelection] = {}
    needs: dict[tuple[str, str, int, str], BoundaryNeed] = {}
    queue: list[
        tuple[RepositoryMethod, FunctionSource, str, int, BoundaryNeed]
    ] = []
    expanded_needs: dict[tuple[str, int, str], BoundaryNeed] = {}
    unresolved_relevant = 0

    entry_language = source_language(entry_method.path, entry.language)
    entry_relevant = _dependency_closure(entry)
    entry_parameter_scope = set(entry.parameters)

    for source_call in _iter_source_calls(entry):
        repository_call = _repository_call_for_source(entry_method, source_call)
        targets = resolver.resolve(
            repository_call, source_call, entry, entry_language
        )
        if not targets:
            # At the direct boundary every unresolved custom call with pointer-
            # like/value-bearing arguments is potentially memory-relevant. Keep
            # an explicit diagnostic rather than silently dropping it.
            if source_call.arguments:
                unresolved_relevant += 1
            continue
        for target in targets:
            callee = target.method
            if callee.full_name == entry_method.full_name or not _method_can_produce_summary(callee):
                continue
            decision = _call_need(
                entry,
                source_call,
                callee,
                entry_relevant,
                entry_parameter_scope,
                direct_layer=True,
            )
            if decision is None:
                continue
            reason, need = decision
            sources = resolver.sources_for_target(target, entry_language)
            if not sources:
                unresolved_relevant += 1
                continue
            _add_candidate(
                discovered,
                selections,
                needs,
                sample_key=sample_key,
                sources=sources,
                target=target,
                call_line=source_call.line,
                depth=1,
                caller=entry_method.full_name,
                reason=reason,
                need=need,
            )
            for source in sources:
                queue.append((callee, source, source.language, 1, need))

    while queue:
        caller, caller_source, caller_language, depth, incoming_need = queue.pop(0)
        expansion_key = (caller.full_name, caller_source.start_line, caller_source.path)
        merged_need = _merge_need(expanded_needs.get(expansion_key), incoming_need)
        if expanded_needs.get(expansion_key) == merged_need:
            continue
        expanded_needs[expansion_key] = merged_need

        caller_relevant = _scoped_dependency_closure(caller_source, merged_need)
        parameter_scope = {
            caller_source.parameters[index]
            for index in merged_need.parameter_indices
            if 0 <= index < len(caller_source.parameters)
        }

        for source_call in _iter_source_calls(caller_source):
            repository_call = _repository_call_for_source(caller, source_call)
            targets = resolver.resolve(
                repository_call, source_call, caller_source, caller_language
            )
            if not targets:
                argument_relevant = any(
                    _expression_closure_at(
                        caller_source, (argument,), source_call.line
                    ) & caller_relevant
                    for argument in source_call.arguments
                )
                result_relevant = bool(
                    source_call.result
                    and normalize_expression(source_call.result) in caller_relevant
                )
                if argument_relevant or result_relevant:
                    unresolved_relevant += 1
                continue

            for target in targets:
                callee = target.method
                if callee.full_name == caller.full_name or not _method_can_produce_summary(callee):
                    continue
                decision = _call_need(
                    caller_source,
                    source_call,
                    callee,
                    caller_relevant,
                    parameter_scope,
                    direct_layer=False,
                )
                if decision is None:
                    continue
                reason, need = decision
                sources = resolver.sources_for_target(target, caller_language)
                if not sources:
                    unresolved_relevant += 1
                    continue
                _add_candidate(
                    discovered,
                    selections,
                    needs,
                    sample_key=sample_key,
                    sources=sources,
                    target=target,
                    call_line=source_call.line,
                    depth=depth + 1,
                    caller=caller.full_name,
                    reason=reason,
                    need=need,
                )
                for source in sources:
                    queue.append(
                        (callee, source, source.language, depth + 1, need)
                    )

    candidates = tuple(
        sorted(discovered.values(), key=lambda item: _candidate_key(item.function))
    )
    direct = sum(
        selections[_candidate_key(candidate.function)].depth == 1
        for candidate in candidates
    )
    return CandidateDiscovery(
        candidates=candidates,
        selections=selections,
        direct_candidates=direct,
        recursive_candidates=len(candidates) - direct,
        expanded_methods=len(expanded_needs),
        unresolved_relevant_calls=unresolved_relevant,
    )


def write_candidate_manifest(index: JoernRepositoryIndex, discovery: CandidateDiscovery) -> Path:
    target = candidate_manifest_path(index)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "record_type": "manifest",
        "manifest_version": CANDIDATE_MANIFEST_VERSION,
        "discovery_policy_version": DISCOVERY_POLICY_VERSION,
        "sample_key": index.sample_key,
        "index_fingerprint": index.fingerprint,
        "direct_candidates": discovery.direct_candidates,
        "recursive_candidates": discovery.recursive_candidates,
        "expanded_methods": discovery.expanded_methods,
        "unresolved_relevant_calls": discovery.unresolved_relevant_calls,
        "candidate_count": len(discovery.candidates),
    }
    records: list[dict[str, object]] = [header]
    for candidate in discovery.candidates:
        function = candidate.function
        selection = discovery.selections[_candidate_key(function)]
        records.append({
            "record_type": "candidate",
            "sample_key": candidate.sample_key,
            "source_path": function.path,
            "function": function.name,
            "source_line": function.start_line,
            "end_line": function.end_line,
            "language": function.language,
            "method_full_name": candidate.method_full_name,
            "call_lines": list(candidate.call_lines),
            "variant_group": candidate.variant_group,
            "variant_count": candidate.variant_count,
            "required_parameters": list(candidate.required_parameters),
            "require_return": candidate.require_return,
            "resolution": candidate.resolution,
            "source_fingerprint": candidate_source_fingerprint(candidate),
            "skip_reason": candidate_validation_error(function),
            "depth": selection.depth,
            "caller": selection.caller,
            "selection_reason": selection.reason,
        })

    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(target)
    return target


def read_candidate_manifest(
    index: JoernRepositoryIndex,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    path = candidate_manifest_path(index)
    if not path.is_file():
        raise RuntimeError(
            f"{index.sample_key}: candidate manifest is missing; run preflight first"
        )
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not records or records[0].get("record_type") != "manifest":
        raise RuntimeError(
            f"{index.sample_key}: invalid candidate manifest: {path}"
        )
    header = records[0]
    if (
        header.get("manifest_version") != CANDIDATE_MANIFEST_VERSION
        or header.get("discovery_policy_version") != DISCOVERY_POLICY_VERSION
        or header.get("index_fingerprint") != index.fingerprint
    ):
        raise RuntimeError(
            f"{index.sample_key}: candidate manifest is stale; rerun preflight --refresh"
        )
    candidates = [
        record
        for record in records[1:]
        if record.get("record_type") == "candidate"
    ]
    if len(candidates) != int(header.get("candidate_count", -1)):
        raise RuntimeError(
            f"{index.sample_key}: incomplete candidate manifest: {path}"
        )
    return header, candidates


def _load_source_candidate(index, record, language: str) -> FunctionSource:
    path = str(record["source_path"])
    name = str(record["function"])
    source_line = int(record["source_line"])
    resolution = str(record.get("resolution", ""))
    text = index.repository.read_blob(path)

    if resolution == "source-macro":
        # Re-run the same resolver against a synthetic call so macro/source
        # definitions are reconstructed from repository source, not serialized
        # code fragments in the manifest.
        from .symbol_resolution import _macro_blocks, _macro_source

        for start, end, params, body, _raw in _macro_blocks(text, name):
            if start != source_line:
                continue
            source = _macro_source(
                path, text, name, start, end, params, body, language
            )
            if source is not None:
                return source
        raise RuntimeError(
            f"{index.sample_key}: source macro no longer resolves: {path}:{name}@{source_line}"
        )

    matches = [
        function
        for function in parse_functions(path, text, language_hint=language)
        if function.name == name and function.start_line == source_line
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{index.sample_key}: source function no longer resolves uniquely: "
            f"{path}:{name}@{source_line}"
        )
    return matches[0]


def load_manifest_candidate(
    index: JoernRepositoryIndex,
    record: dict[str, object],
    parse_cache: dict[tuple[str, str], list[FunctionSource]] | None = None,
) -> Candidate:
    resolution = str(record.get("resolution", "joern-exact"))
    method_full_name = str(record.get("method_full_name", ""))
    language = str(record.get("language") or source_language(str(record["source_path"])))

    if resolution.startswith("source-"):
        source = _load_source_candidate(index, record, language)
    else:
        method = index.methods().get(method_full_name)
        if method is None:
            raise RuntimeError(
                f"{index.sample_key}: candidate method disappeared from repository index: "
                f"{method_full_name}"
            )
        resolver = SymbolResolver(index)
        sources = resolver.sources_for_method(method, language)
        source_line = int(record["source_line"])
        matches = [source for source in sources if source.start_line == source_line]
        if len(matches) != 1:
            raise RuntimeError(
                f"{index.sample_key}: candidate source no longer resolves uniquely: "
                f"{record['source_path']}:{record['function']}@{source_line}; "
                "rerun preflight --refresh"
            )
        source = matches[0]

    candidate = Candidate(
        sample_key=index.sample_key,
        function=source,
        call_lines=tuple(int(line) for line in record.get("call_lines", [])),
        method_full_name=method_full_name,
        variant_group=(
            str(record["variant_group"])
            if record.get("variant_group") is not None
            else None
        ),
        variant_count=int(record.get("variant_count", 1)),
        required_parameters=tuple(
            int(index) for index in record.get("required_parameters", [])
        ),
        require_return=bool(record.get("require_return", False)),
        resolution=resolution,
    )
    if candidate_source_fingerprint(candidate) != str(record.get("source_fingerprint", "")):
        raise RuntimeError(
            f"{index.sample_key}: candidate source changed after preflight; "
            "rerun preflight --refresh"
        )
    return candidate
