from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .joern import JoernRepositoryIndex, RepositoryMethod
from .semantics import Candidate, candidate_validation_error
from .source import FunctionSource, normalize_expression, parse_functions, source_language
from .standard_semantics import STANDARD_LEAF_CALLS, standard_seed_expressions


CANDIDATE_MANIFEST_VERSION = 2
DISCOVERY_POLICY_VERSION = 2


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


@dataclass(frozen=True)
class CandidateDiscovery:
    candidates: tuple[Candidate, ...]
    selections: dict[tuple[str, str, int, str], CandidateSelection]
    direct_candidates: int
    recursive_candidates: int
    expanded_methods: int
    unresolved_relevant_calls: int


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


def _sources_for_method_cached(
    index: JoernRepositoryIndex,
    method: RepositoryMethod,
    language_hint: str,
    parse_cache: dict[tuple[str, str], list[FunctionSource]],
) -> list[FunctionSource]:
    base = index.repository.function_source(
        path=method.path,
        name=method.name,
        start_line=method.start_line,
        end_line=method.end_line,
        parameters=method.parameters,
        parameter_types=method.parameter_types,
        language_hint=language_hint,
    )
    if base.language != "c":
        return [base]

    cache_key = (method.path, base.language)
    if cache_key not in parse_cache:
        try:
            parse_cache[cache_key] = parse_functions(
                method.path,
                index.repository.read_blob(method.path),
                language_hint=base.language,
            )
        except (ValueError, UnicodeError):
            return [base]

    parsed = [
        function
        for function in parse_cache[cache_key]
        if function.name == method.name
        and len(function.parameters) == len(method.parameters)
    ]
    if len(parsed) <= 1:
        return [base]
    signatures = {function.parameter_signatures for function in parsed}
    groups = {function.preprocessor_group for function in parsed}
    branches = {function.preprocessor_branch for function in parsed}
    if (
        len(signatures) == 1
        and len(groups) == 1
        and None not in groups
        and None not in branches
        and len(branches) == len(parsed)
    ):
        return parsed
    return [base]


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


def _function_facts(function: FunctionSource, cache: dict[int, dict[str, object]]):
    key = id(function)
    if key not in cache:
        calls = function.calls()
        relations = function.value_relations_before(function.end_line + 1)
        relation_map = {
            normalize_expression(left): normalize_expression(right)
            for left, right in relations
        }
        seeds = list(standard_seed_expressions(function))
        for match in re.finditer(r"\breturn\s+([^;]+);", function.text):
            line = function.start_line + function.text[: match.start()].count("\n")
            seeds.append((line, match.group(1)))
        cache[key] = {
            "calls": calls,
            "relations": relation_map,
            "seeds": seeds,
        }
    return cache[key]


def _dependency_closure(
    function: FunctionSource,
    cache: dict[int, dict[str, object]],
) -> set[str]:
    facts = _function_facts(function, cache)
    relations = facts["relations"]
    pending: list[str] = []
    for _line, expression in facts["seeds"]:
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


def _source_call(function: FunctionSource, line: int, name: str, cache):
    calls = _function_facts(function, cache)["calls"]
    matches = [
        call for call in calls
        if not call.indirect and call.name == name and call.line == line
    ]
    if len(matches) == 1:
        return matches[0]
    named = [call for call in calls if not call.indirect and call.name == name]
    return named[0] if len(named) == 1 else None


def _result_flows_to_return(function: FunctionSource, result: str | None) -> bool:
    if not result:
        return False
    name = normalize_expression(result)
    if not re.fullmatch(r"[A-Za-z_]\w*", name):
        return False
    return re.search(rf"\breturn\s+[^;]*\b{re.escape(name)}\b", function.text) is not None


def _call_relevance(
    function: FunctionSource,
    source_call,
    callee: RepositoryMethod,
    cache: dict[int, dict[str, object]],
) -> str | None:
    if source_call is None:
        # A Joern-resolved edge is more trustworthy than guessed source coordinates.
        # This case occurs for preprocessed CPGs whose CALL line numbers refer to .i
        # coordinates. Keep the callee conservatively instead of silently losing it.
        return "Joern-resolved callee retained because source call coordinates are ambiguous"
    if source_call.returned or _result_flows_to_return(function, source_call.result):
        return "callee result contributes to caller return"

    relevant = _dependency_closure(function, cache)
    if source_call.result and normalize_expression(source_call.result) in relevant:
        return "callee result reaches a memory-relevant value"

    for argument in source_call.arguments:
        if _identifiers(argument) & relevant:
            return "callee argument depends on a memory-relevant value"

    pointer_positions = [
        index for index, type_text in enumerate(callee.parameter_types)
        if _type_may_be_pointer(type_text)
    ]
    for index in pointer_positions:
        if index >= len(source_call.arguments):
            continue
        argument = source_call.arguments[index]
        if _identifiers(argument) & (set(function.parameters) | relevant):
            return "caller pointer/value flows into summary-capable callee"
    return None


def _candidate_key(source: FunctionSource) -> tuple[str, str, int, str]:
    return (source.path, source.name, source.start_line, source.language)


def _add_candidate(
    discovered: dict[tuple[str, str, int, str], Candidate],
    selections: dict[tuple[str, str, int, str], CandidateSelection],
    *,
    sample_key: str,
    sources: list[FunctionSource],
    callee: RepositoryMethod,
    call_line: int,
    depth: int,
    caller: str,
    reason: str,
) -> None:
    variant_count = (
        len(sources)
        if len(sources) > 1
        and all(source.preprocessor_group is not None for source in sources)
        else 1
    )
    for source in sources:
        key = _candidate_key(source)
        existing = discovered.get(key)
        lines = set(existing.call_lines if existing else ())
        lines.add(call_line)
        variant_group = (
            f"{source.path}:{source.name}:"
            f"{source.preprocessor_group[0]}-{source.preprocessor_group[1]}"
            if variant_count > 1 and source.preprocessor_group is not None
            else None
        )
        discovered[key] = Candidate(
            sample_key=sample_key,
            function=source,
            call_lines=tuple(sorted(lines)),
            method_full_name=callee.full_name,
            variant_group=variant_group,
            variant_count=variant_count,
        )
        current = selections.get(key)
        if current is None or depth < current.depth:
            selections[key] = CandidateSelection(depth, caller, reason)


def discover_relevant_candidates(
    sample_key: str,
    index: JoernRepositoryIndex,
    entry_method: RepositoryMethod,
    entry: FunctionSource,
) -> CandidateDiscovery:
    """Follow only call edges justified by a memory-relevance backward slice."""
    parse_cache: dict[tuple[str, str], list[FunctionSource]] = {}
    fact_cache: dict[int, dict[str, object]] = {}
    discovered: dict[tuple[str, str, int, str], Candidate] = {}
    selections: dict[tuple[str, str, int, str], CandidateSelection] = {}
    queue: list[tuple[RepositoryMethod, FunctionSource, str, int]] = []
    expanded: set[tuple[str, int]] = set()
    unresolved_relevant = 0

    entry_language = source_language(entry_method.path, entry.language)
    for call in entry_method.calls:
        if call.name.startswith("<operator>.") or call.name in STANDARD_LEAF_CALLS:
            continue
        callees = index.callee_methods(call)
        if not callees:
            source_call = _source_call(entry, call.line, call.name, fact_cache)
            if source_call and any(
                _identifiers(arg) & _dependency_closure(entry, fact_cache)
                for arg in source_call.arguments
            ):
                unresolved_relevant += 1
            continue
        for callee in callees:
            if callee.full_name == entry_method.full_name or not _method_can_produce_summary(callee):
                continue
            source_call = _source_call(entry, call.line, call.name, fact_cache)
            reason = _call_relevance(entry, source_call, callee, fact_cache)
            if reason is None:
                continue
            callee_language = source_language(callee.path, entry_language)
            sources = _sources_for_method_cached(index, callee, callee_language, parse_cache)
            _add_candidate(
                discovered,
                selections,
                sample_key=sample_key,
                sources=sources,
                callee=callee,
                call_line=call.line,
                depth=1,
                caller=entry_method.full_name,
                reason=reason,
            )
            for source in sources:
                queue.append((callee, source, callee_language, 1))

    while queue:
        caller, caller_source, caller_language, depth = queue.pop(0)
        expansion_key = (caller.full_name, caller_source.start_line)
        if expansion_key in expanded:
            continue
        expanded.add(expansion_key)

        for call in caller.calls:
            if call.name.startswith("<operator>.") or call.name in STANDARD_LEAF_CALLS:
                continue
            callees = index.callee_methods(call)
            source_call = _source_call(caller_source, call.line, call.name, fact_cache)
            if not callees:
                if source_call and any(
                    _identifiers(arg) & _dependency_closure(caller_source, fact_cache)
                    for arg in source_call.arguments
                ):
                    unresolved_relevant += 1
                continue
            for callee in callees:
                if callee.full_name == caller.full_name or not _method_can_produce_summary(callee):
                    continue
                reason = _call_relevance(caller_source, source_call, callee, fact_cache)
                if reason is None:
                    continue
                callee_language = source_language(callee.path, caller_language)
                sources = _sources_for_method_cached(index, callee, callee_language, parse_cache)
                _add_candidate(
                    discovered,
                    selections,
                    sample_key=sample_key,
                    sources=sources,
                    callee=callee,
                    call_line=call.line,
                    depth=depth + 1,
                    caller=caller.full_name,
                    reason=reason,
                )
                for source in sources:
                    queue.append((callee, source, callee_language, depth + 1))

    candidates = tuple(sorted(discovered.values(), key=lambda item: _candidate_key(item.function)))
    direct = sum(selections[_candidate_key(candidate.function)].depth == 1 for candidate in candidates)
    return CandidateDiscovery(
        candidates=candidates,
        selections=selections,
        direct_candidates=direct,
        recursive_candidates=len(candidates) - direct,
        expanded_methods=len(expanded),
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


def read_candidate_manifest(index: JoernRepositoryIndex) -> tuple[dict[str, object], list[dict[str, object]]]:
    path = candidate_manifest_path(index)
    if not path.is_file():
        raise RuntimeError(f"{index.sample_key}: candidate manifest is missing; run preflight first")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records or records[0].get("record_type") != "manifest":
        raise RuntimeError(f"{index.sample_key}: invalid candidate manifest: {path}")
    header = records[0]
    if (
        header.get("manifest_version") != CANDIDATE_MANIFEST_VERSION
        or header.get("discovery_policy_version") != DISCOVERY_POLICY_VERSION
        or header.get("index_fingerprint") != index.fingerprint
    ):
        raise RuntimeError(f"{index.sample_key}: candidate manifest is stale; rerun preflight --refresh")
    candidates = [record for record in records[1:] if record.get("record_type") == "candidate"]
    if len(candidates) != int(header.get("candidate_count", -1)):
        raise RuntimeError(f"{index.sample_key}: incomplete candidate manifest: {path}")
    return header, candidates


def load_manifest_candidate(
    index: JoernRepositoryIndex,
    record: dict[str, object],
    parse_cache: dict[tuple[str, str], list[FunctionSource]] | None = None,
) -> Candidate:
    parse_cache = parse_cache if parse_cache is not None else {}
    method_full_name = str(record["method_full_name"])
    method = index.methods().get(method_full_name)
    if method is None:
        raise RuntimeError(
            f"{index.sample_key}: candidate method disappeared from Joern index: {method_full_name}"
        )
    language = str(record.get("language") or source_language(method.path))
    sources = _sources_for_method_cached(index, method, language, parse_cache)
    source_line = int(record["source_line"])
    matches = [source for source in sources if source.start_line == source_line]
    if len(matches) != 1:
        raise RuntimeError(
            f"{index.sample_key}: candidate source no longer resolves uniquely: "
            f"{record['source_path']}:{record['function']}@{source_line}; rerun preflight --refresh"
        )
    candidate = Candidate(
        sample_key=index.sample_key,
        function=matches[0],
        call_lines=tuple(int(line) for line in record.get("call_lines", [])),
        method_full_name=method_full_name,
        variant_group=(str(record["variant_group"]) if record.get("variant_group") is not None else None),
        variant_count=int(record.get("variant_count", 1)),
    )
    if candidate_source_fingerprint(candidate) != str(record.get("source_fingerprint", "")):
        raise RuntimeError(
            f"{index.sample_key}: candidate source changed after preflight; rerun preflight --refresh"
        )
    return candidate
