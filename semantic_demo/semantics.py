from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .joern import JoernMethodNotFound, JoernTimeout, JoernValidator
from .source import FunctionSource, GitRepository, normalize_expression


ALLOCATORS = {
    "malloc": (0,),
    "calloc": (0, 1),
    "realloc": (1,),
    "kmalloc": (0,),
    "kzalloc": (0,),
    "vmalloc": (0,),
}
WRITES = {
    "memcpy": (0, 2),
    "memmove": (0, 2),
    "memset": (0, 2),
    "read": (1, 2),
    "recv": (1, 2),
    "recvfrom": (1, 2),
    "fread": (0, 1),
    "ReadFile": (1, 2),
}
READS = {
    "memcpy": (1, 2),
    "memmove": (1, 2),
    "write": (1, 2),
    "send": (1, 2),
    "sendto": (1, 2),
    "fwrite": (0, 1),
    "memcmp": (0, 2),
}
UNBOUNDED_WRITES = {"sprintf", "strcpy", "strcat", "vsprintf"}
NORMALIZATION_SCHEMA_VERSION = 4
MAX_LLM_SOURCE_CHARS = 50000
NORMALIZATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summaries": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["ALLOC", "READ", "WRITE", "VALUE"],
                    },
                    "buffer": {"type": "string"},
                    "size": {"type": "string"},
                    "length": {"type": "string"},
                    "target": {"type": "string"},
                    "expression": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    "required": ["summaries"],
    "additionalProperties": False,
}

STANDARD_CALLS = (
    set(ALLOCATORS)
    | set(WRITES)
    | set(READS)
    | UNBOUNDED_WRITES
    | {
        "free",
        "strlen",
        "sizeof",
        "strcmp",
        "strchr",
        "snprintf",
        "vsnprintf",
    }
)



@dataclass(frozen=True)
class Candidate:
    sample_key: str
    function: FunctionSource
    call_lines: tuple[int, ...]


@dataclass(frozen=True)
class Validation:
    sample_key: str
    function: str
    source_path: str
    source_line: int
    summary: dict[str, str]
    passed: bool
    reason: str

    def as_json(self) -> dict[str, object]:
        return asdict(self)


def _variant_definitions(
    sample_key: str,
    name: str,
    definitions: list[FunctionSource],
) -> list[FunctionSource]:
    if len(definitions) <= 1:
        return definitions
    paths = {item.path for item in definitions}
    languages = {item.language for item in definitions}
    signatures = {item.parameter_signatures for item in definitions}
    if len(paths) == 1 and languages == {"c"} and len(signatures) == 1:
        return definitions
    raise RuntimeError(f"{sample_key}: ambiguous repository binding for {name}")


def discover_candidates(
    sample_key: str,
    repository: GitRepository,
    entry: FunctionSource,
    scopes: Iterable[str],
    max_functions: int = 128,
) -> list[Candidate]:
    """Traverse the repository-resolved project call graph without name heuristics."""
    scopes = tuple(scopes)
    discovered: dict[tuple[str, str, int], Candidate] = {}
    queue: list[FunctionSource] = [entry]
    expanded: set[tuple[str, str, int]] = set()

    while queue:
        caller = queue.pop(0)
        caller_key = (caller.path, caller.name, caller.start_line)
        if caller_key in expanded:
            continue
        expanded.add(caller_key)

        for call in caller.calls():
            if call.indirect:
                continue
            if call.name in STANDARD_CALLS or call.name == caller.name:
                continue
            callee = repository.find_function(
                call.name, preferred_path=caller.path, scopes=scopes
            )
            if callee is not None:
                definitions = [callee]
            else:
                definitions = repository.find_functions(call.name, scopes)
                if not definitions:
                    continue
                definitions = _variant_definitions(
                    sample_key, call.name, definitions
                )

            for definition in definitions:
                key = (
                    definition.path,
                    definition.name,
                    definition.start_line,
                )
                existing = discovered.get(key)
                lines = set(existing.call_lines if existing else ())
                lines.add(call.line)
                discovered[key] = Candidate(
                    sample_key=sample_key,
                    function=definition,
                    call_lines=tuple(sorted(lines)),
                )
                if len(discovered) > max_functions:
                    raise RuntimeError(
                        f"{sample_key}: candidate frontier exceeds explicit budget "
                        f"({max_functions}); analysis aborted rather than truncated"
                    )
                if key not in expanded:
                    queue.append(definition)

    return sorted(
        discovered.values(),
        key=lambda item: (
            item.function.path,
            item.function.name,
            item.function.start_line,
        ),
    )


def _arg_indices(value: str) -> list[int]:
    return [int(item) for item in re.findall(r"\barg(\d+)\b", value)]


def _buffer_root_index(value: str) -> int | None:
    match = re.match(r"^\s*arg(\d+)\b", value)
    return int(match.group(1)) if match else None


def _schema_error(summary: dict[str, object], parameter_count: int) -> str | None:
    kind = summary.get("kind")
    if kind == "ALLOC":
        required = {"kind", "buffer", "size"}
        if set(summary) != required or summary.get("buffer") != "return":
            return "ALLOC must contain exactly kind/buffer/size and buffer=return"
    elif kind in {"READ", "WRITE"}:
        required = {"kind", "buffer", "length"}
        if set(summary) != required:
            return f"{kind} must contain exactly kind/buffer/length"
    elif kind == "VALUE":
        required = {"kind", "target", "expression"}
        if set(summary) != required or summary.get("target") != "return":
            return "VALUE must contain exactly kind/target/expression and target=return"
    else:
        return "kind must be ALLOC, READ, WRITE, or VALUE"

    for value in summary.values():
        if not isinstance(value, str):
            return "all summary values must be strings"
        if any(index >= parameter_count for index in _arg_indices(value)):
            return "summary references a nonexistent parameter"
    return None


def canonicalize_summary(
    function: FunctionSource, summary: dict[str, object]
) -> dict[str, str]:
    """Normalize exact source-parameter identifiers to positional argN names."""
    clean = {str(key): str(value) for key, value in summary.items()}
    for key, value in list(clean.items()):
        if key == "kind":
            continue
        normalized = value
        for index, parameter in reversed(list(enumerate(function.parameters))):
            normalized = re.sub(
                rf"\b{re.escape(parameter)}\b",
                f"arg{index}",
                normalized,
            )
        clean[key] = normalized
    return clean


def _substitute_args(expression: str, parameters: tuple[str, ...]) -> str:
    result = expression
    for index in reversed(range(len(parameters))):
        result = re.sub(rf"\barg{index}\b", parameters[index], result)
    return result


def _known_call_indices(name: str, kind: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if kind == "ALLOC" and name in ALLOCATORS:
        return (), ALLOCATORS[name]
    if kind == "WRITE" and name in WRITES:
        buffer, length = WRITES[name]
        if name == "fread":
            return (buffer,), (1, 2)
        return (buffer,), (length,)
    if kind == "READ" and name in READS:
        buffer, length = READS[name]
        if name == "fwrite":
            return (buffer,), (1, 2)
        return (buffer,), (length,)
    return (), ()


def _joern_expr_reaches(
    facts,
    expression: str,
    call,
    argument_indices: tuple[int, ...],
) -> bool:
    params = _arg_indices(expression)
    if params:
        return all(
            any(facts.parameter_reaches(param, call, index) for index in argument_indices)
            for param in params
        )
    compact = normalize_expression(expression)
    return any(
        compact
        and compact == normalize_expression(call.arguments.get(index, ""))
        for index in argument_indices
    )


def _normalized_return_expression(value: str) -> str:
    compact = normalize_expression(value)
    if compact.startswith("return"):
        compact = compact[len("return"):]
    if compact.endswith(";"):
        compact = compact[:-1]
    return compact


def _validate_with_joern(
    candidate: Candidate,
    summary: dict[str, str],
    validator: JoernValidator,
) -> tuple[bool, str]:
    facts = validator.facts(candidate)
    kind = summary["kind"]

    if kind == "ALLOC":
        returned = {_normalized_return_expression(value) for value in facts.returns}
        source_calls = {
            (call.line, call.name): call
            for call in candidate.function.calls()
            if not call.indirect
        }
        for call in facts.call_list():
            _, size_indices = _known_call_indices(call.name, "ALLOC")
            if not size_indices:
                continue
            source_call = source_calls.get((call.line, call.name))
            if source_call is None:
                continue
            returns_allocation = source_call.returned or (
                source_call.result is not None
                and normalize_expression(source_call.result) in returned
            )
            if not returns_allocation:
                continue
            if _joern_expr_reaches(facts, summary["size"], call, size_indices):
                return True, "Joern verified returned allocation and its size flow"
        return False, "Joern found no returned specified allocator matching the declared size"

    if kind in {"READ", "WRITE"}:
        for call in facts.call_list():
            buffer_indices, length_indices = _known_call_indices(call.name, kind)
            if not buffer_indices or not length_indices:
                continue
            if _joern_expr_reaches(
                facts, summary["buffer"], call, buffer_indices
            ) and _joern_expr_reaches(
                facts, summary["length"], call, length_indices
            ):
                return (
                    True,
                    f"Joern verified {kind.lower()} flow to specified API {call.name}",
                )
        return False, f"Joern found no specified API matching the declared {kind.lower()}"

    if kind == "VALUE":
        expression = normalize_expression(
            _substitute_args(summary["expression"], candidate.function.parameters)
        )
        returned = {_normalized_return_expression(value) for value in facts.returns}
        if expression and expression in returned:
            return True, "Joern verified exact returned value expression"
        return False, "Joern found no exact return matching the declared VALUE expression"

    return False, f"unsupported semantic kind: {kind}"


def _source_call_returns_value(candidate: Candidate, source_call, facts) -> bool:
    if source_call.returned:
        return True
    if source_call.result is None:
        return False
    returned = {_normalized_return_expression(value) for value in facts.returns}
    return normalize_expression(source_call.result) in returned


def _validate_by_composition(
    candidate: Candidate,
    summary: dict[str, str],
    validator: JoernValidator,
    callee_summaries: dict[tuple[str, str], list[dict[str, str]]],
) -> tuple[bool, str]:
    """Validate a wrapper summary from already validated callee summaries."""
    facts = validator.facts(candidate)
    kind = summary["kind"]

    for call in facts.call_list():
        matches = [
            summaries
            for (path, name), summaries in callee_summaries.items()
            if name == call.name
        ]
        if len(matches) != 1:
            continue
        for child in matches[0]:
            if child.get("kind") != kind:
                continue

            if kind in {"READ", "WRITE"}:
                child_buffer_args = tuple(_arg_indices(child.get("buffer", "")))
                child_length_args = tuple(_arg_indices(child.get("length", "")))
                if not child_buffer_args or not child_length_args:
                    continue
                if _joern_expr_reaches(
                    facts, summary["buffer"], call, child_buffer_args
                ) and _joern_expr_reaches(
                    facts, summary["length"], call, child_length_args
                ):
                    return (
                        True,
                        f"composition verified {kind.lower()} through "
                        f"validated callee summary {call.name}",
                    )

            elif kind == "ALLOC":
                child_size_args = tuple(_arg_indices(child.get("size", "")))
                if not child_size_args:
                    continue
                if not _joern_expr_reaches(
                    facts, summary["size"], call, child_size_args
                ):
                    continue
                source_call = next(
                    (
                        source_call
                        for source_call in candidate.function.calls()
                        if source_call.name == call.name and source_call.line == call.line
                    ),
                    None,
                )
                if (
                    source_call is not None
                    and _source_call_returns_value(candidate, source_call, facts)
                ):
                    return (
                        True,
                        f"composition verified allocation through "
                        f"validated callee summary {call.name}",
                    )

            elif kind == "VALUE":
                child_expression = child.get("expression", "")
                child_args = _arg_indices(child_expression)
                if (
                    len(child_args) != 1
                    or child_expression != f"arg{child_args[0]}"
                ):
                    continue
                source_call = next(
                    (
                        source_call
                        for source_call in candidate.function.calls()
                        if source_call.name == call.name and source_call.line == call.line
                    ),
                    None,
                )
                if (
                    source_call is None
                    or not _source_call_returns_value(candidate, source_call, facts)
                ):
                    continue
                if _joern_expr_reaches(
                    facts, summary["expression"], call, (child_args[0],)
                ):
                    return (
                        True,
                        f"composition verified value through "
                        f"validated callee summary {call.name}",
                    )

    return False, "no validated callee summary composes to the claimed semantic role"


def candidate_validation_error(function: FunctionSource) -> str | None:
    if function.parse_has_error:
        return "candidate function contains parser error nodes"
    return None


def validate_summary(
    candidate: Candidate,
    summary: dict[str, object],
    joern: JoernValidator,
    callee_summaries: dict[tuple[str, str], list[dict[str, str]]] | None = None,
) -> Validation:
    function = candidate.function
    error = candidate_validation_error(function)
    if error is None:
        error = _schema_error(summary, len(function.parameters))
    clean_summary = canonicalize_summary(function, summary)

    if not error and clean_summary.get("kind") in {"READ", "WRITE"}:
        buffer = clean_summary.get("buffer", "")
        root_index = _buffer_root_index(buffer)
        if root_index is None:
            error = (
                f"{clean_summary.get('kind')} buffer must be rooted at a "
                "caller-supplied argN"
            )
        elif (
            root_index >= len(function.parameter_pointer_like)
            or not function.parameter_pointer_like[root_index]
        ):
            error = (
                f"{clean_summary.get('kind')} buffer root arg{root_index} "
                "is not pointer-like in the candidate signature"
            )

    if error:
        return Validation(
            candidate.sample_key,
            function.name,
            function.path,
            function.start_line,
            clean_summary,
            False,
            error,
        )

    try:
        passed, reason = _validate_with_joern(candidate, clean_summary, joern)
        if not passed and callee_summaries:
            composed, composed_reason = _validate_by_composition(
                candidate,
                clean_summary,
                joern,
                callee_summaries,
            )
            if composed:
                passed, reason = composed, composed_reason
    except JoernMethodNotFound as validation_error:
        passed = False
        reason = f"Joern candidate method unavailable: {validation_error}"
    except JoernTimeout as validation_error:
        passed = False
        reason = f"Joern candidate validation timed out: {validation_error}"
    return Validation(
        candidate.sample_key,
        function.name,
        function.path,
        function.start_line,
        clean_summary,
        passed,
        reason,
    )



def _direct_standard_summaries(function: FunctionSource) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for call in function.calls():
        if call.indirect:
            continue

        if call.name in WRITES:
            buffer_index, length_index = WRITES[call.name]
            if len(call.arguments) > max(buffer_index, length_index):
                length = call.arguments[length_index]
                if call.name == "fread" and len(call.arguments) >= 3:
                    length = f"({call.arguments[1]}) * ({call.arguments[2]})"
                summary = canonicalize_summary(
                    function,
                    {
                        "kind": "WRITE",
                        "buffer": call.arguments[buffer_index],
                        "length": length,
                    },
                )
                root = _buffer_root_index(summary["buffer"])
                if (
                    root is not None
                    and root < len(function.parameter_pointer_like)
                    and function.parameter_pointer_like[root]
                ):
                    summaries.append(summary)

        if call.name in READS:
            buffer_index, length_index = READS[call.name]
            if len(call.arguments) > max(buffer_index, length_index):
                length = call.arguments[length_index]
                if call.name == "fwrite" and len(call.arguments) >= 3:
                    length = f"({call.arguments[1]}) * ({call.arguments[2]})"
                summary = canonicalize_summary(
                    function,
                    {
                        "kind": "READ",
                        "buffer": call.arguments[buffer_index],
                        "length": length,
                    },
                )
                root = _buffer_root_index(summary["buffer"])
                if (
                    root is not None
                    and root < len(function.parameter_pointer_like)
                    and function.parameter_pointer_like[root]
                ):
                    summaries.append(summary)
                if call.name == "memcmp" and len(call.arguments) >= 3:
                    summary = canonicalize_summary(
                        function,
                        {
                            "kind": "READ",
                            "buffer": call.arguments[1],
                            "length": call.arguments[2],
                        },
                    )
                    root = _buffer_root_index(summary["buffer"])
                    if (
                        root is not None
                        and root < len(function.parameter_pointer_like)
                        and function.parameter_pointer_like[root]
                    ):
                        summaries.append(summary)

        if call.name in ALLOCATORS and call.returned:
            indices = ALLOCATORS[call.name]
            if call.name == "calloc" and len(call.arguments) >= 2:
                size = f"({call.arguments[0]}) * ({call.arguments[1]})"
            elif indices and len(call.arguments) > indices[0]:
                size = call.arguments[indices[0]]
            else:
                size = ""
            if size:
                summaries.append(
                    canonicalize_summary(
                        function,
                        {"kind": "ALLOC", "buffer": "return", "size": size},
                    )
                )

    unique: list[dict[str, str]] = []
    for summary in summaries:
        if summary not in unique:
            unique.append(summary)
    return unique


def _normalization_endpoints(
    function: FunctionSource,
) -> list[tuple[str, int | None, str]]:
    endpoints: list[tuple[str, int | None, str]] = []
    if function.has_value_return():
        endpoints.append(("return", None, "function return statements"))
    for call in function.calls():
        if call.indirect or call.name in STANDARD_CALLS:
            continue
        endpoints.append(
            (
                "call",
                call.line,
                f"direct call {call.name}({', '.join(call.arguments)}) at line {call.line}",
            )
        )
    return endpoints


def llm_normalize(
    candidate: Candidate,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 512,
    disable_proxy: bool = False,
    response_schema: dict[str, object] | None = None,
) -> list[dict[str, str]]:
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    if len(candidate.function.text) > MAX_LLM_SOURCE_CHARS:
        raise ValueError(
            f"{candidate.function.name}: function source exceeds explicit LLM budget "
            f"({len(candidate.function.text)} > {MAX_LLM_SOURCE_CHARS} chars)"
        )

    summaries = _direct_standard_summaries(candidate.function)
    indirect_calls = [
        f"{call.name}@{call.line}"
        for call in candidate.function.calls()
        if call.indirect
    ]
    opaque_text = ", ".join(indirect_calls) if indirect_calls else "none"

    for endpoint_kind, endpoint_line, endpoint_text in _normalization_endpoints(
        candidate.function
    ):
        if endpoint_kind == "return":
            allowed = "ALLOC or VALUE"
            instruction = (
                "Report only return semantics that hold for this function's return "
                "behavior. Do not report READ/WRITE effects here."
            )
        else:
            allowed = "ALLOC, READ, WRITE, or VALUE"
            instruction = (
                "Report only caller-visible summaries mediated by this one direct "
                "named call. Do not report effects from any other call site."
            )

        prompt = f"""Normalize one statically localized semantic endpoint in this C/C++ function.
Endpoint: {endpoint_text}
Allowed summary kinds: {allowed}
{instruction}

Return exactly one JSON object with key summaries. The array may contain at most four summaries.
Use only:
{{"kind":"ALLOC","buffer":"return","size":"argN expression"}}
{{"kind":"READ","buffer":"argN expression","length":"argN expression"}}
{{"kind":"WRITE","buffer":"argN expression","length":"argN expression"}}
{{"kind":"VALUE","target":"return","expression":"argN expression"}}

Use positional argN names only. READ/WRITE buffers must be rooted at caller pointer parameters.
Do not infer vulnerability labels, guards, caller behavior, or library contracts. Unresolved
indirect/function-pointer calls are opaque: never infer semantics through them, but they do not
invalidate unrelated direct evidence. Emit {{"summaries":[]}} when this endpoint exposes none.

Function: {candidate.function.name}
Parameters: {json.dumps(list(candidate.function.parameters))}
Opaque indirect calls: {opaque_text}
Source:
{candidate.function.text}
"""
        response_format: dict[str, object] = {"type": "json_object"}
        if response_schema is not None:
            response_format["schema"] = response_schema
        payload = json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You output strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": response_format,
            }
        ).encode()
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        if disable_proxy:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            response_context = opener.open(request, timeout=180)
        else:
            response_context = urllib.request.urlopen(request, timeout=180)
        with response_context as response:
            result = json.load(response)

        parsed = _extract_json_object(_response_content(result))
        endpoint_summaries = parsed.get("summaries")
        if not isinstance(endpoint_summaries, list):
            raise ValueError("LLM response summaries must be a list")
        if len(endpoint_summaries) > 4:
            raise ValueError("LLM response exceeds the endpoint summary bound")
        for raw_summary in endpoint_summaries:
            if not isinstance(raw_summary, dict):
                raise ValueError("LLM summary must be one JSON object")
            clean = canonicalize_summary(candidate.function, raw_summary)
            error = _schema_error(clean, len(candidate.function.parameters))
            if error is not None:
                raise ValueError(f"LLM summary violates schema: {error}")
            if endpoint_kind == "return" and clean["kind"] not in {"ALLOC", "VALUE"}:
                raise ValueError("return endpoint emitted non-return summary")
            if clean not in summaries:
                summaries.append(clean)

    return summaries

def _response_content(result: dict[str, object]) -> str:
    try:
        choice = result["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("LLM response has no choices[0].message") from error

    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        usage = result.get("usage")
        completion_tokens = (
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        )
        raise ValueError(
            "LLM response was truncated at max_tokens "
            f"(completion_tokens={completion_tokens!r})"
        )

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content

    reasoning = message.get("reasoning_content")
    reasoning_length = len(reasoning) if isinstance(reasoning, str) else 0
    usage = result.get("usage")
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    raise ValueError(
        "LLM returned empty content "
        f"(finish_reason={choice.get('finish_reason')!r}, "
        f"reasoning_length={reasoning_length}, "
        f"completion_tokens={completion_tokens!r}). "
        "The provider produced no final JSON answer."
    )


def _extract_json_object(content: str) -> dict[str, object]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("LLM response is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("LLM response must be one JSON object")
    return value


def load_replay(
    path: str | Path,
) -> dict[tuple[str, str, str, int], list[dict[str, str]]]:
    replay: dict[tuple[str, str, str, int], list[dict[str, str]]] = {}
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != NORMALIZATION_SCHEMA_VERSION:
                raise ValueError(
                    f"{path}: obsolete normalization schema; rerun normalize --refresh"
                )
            summaries = record.get("summaries")
            if not isinstance(summaries, list) or not all(
                isinstance(item, dict) for item in summaries
            ):
                raise ValueError(f"{path}: invalid summaries field")
            key = (
                record["sample_key"],
                record["source_path"],
                record["function"],
                int(record["source_line"]),
            )
            replay[key] = summaries
    return replay
