from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from .joern import JoernError, JoernMethodNotFound, JoernTimeout
from .source import FunctionSource, function_body_recoverable, normalize_expression
from .standard_semantics import summaries_for_function


ALLOCATORS = {
    "malloc": (0,), "calloc": (0, 1), "realloc": (1,),
    "kmalloc": (0,), "kzalloc": (0,), "vmalloc": (0,),
}
WRITES = {
    "memcpy": (0, 2), "memmove": (0, 2), "mempcpy": (0, 2),
    "memset": (0, 2), "bzero": (0, 1), "explicit_bzero": (0, 1),
    "read": (1, 2), "recv": (1, 2), "recvfrom": (1, 2),
    "fread": (0, 1), "ReadFile": (1, 2),
}
READS = {
    "memcpy": (1, 2), "memmove": (1, 2), "mempcpy": (1, 2),
    "write": (1, 2), "send": (1, 2), "sendto": (1, 2),
    "fwrite": (0, 1), "memcmp": (0, 2),
}
UNBOUNDED_WRITES = {"sprintf", "strcpy", "strcat", "vsprintf"}
NORMALIZATION_SCHEMA_VERSION = 8
NORMALIZATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summaries": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["ALLOC", "READ", "WRITE", "VALUE"]},
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


@dataclass(frozen=True)
class Candidate:
    sample_key: str
    function: FunctionSource
    call_lines: tuple[int, ...]
    method_full_name: str = ""
    variant_group: str | None = None
    variant_count: int = 1
    required_parameters: tuple[int, ...] = ()
    require_return: bool = False
    resolution: str = "joern-exact"


@dataclass(frozen=True)
class Validation:
    sample_key: str
    function: str
    source_path: str
    source_line: int
    summary: dict[str, str]
    passed: bool
    reason: str
    variant_group: str | None = None
    variant_count: int = 1
    call_lines: tuple[int, ...] = ()
    status: str = "VERIFIED"

    def as_json(self) -> dict[str, object]:
        return asdict(self)


def _type_definitely_pointer(type_text: str) -> bool:
    compact = "".join(type_text.split())
    return any(token in compact for token in ("*", "&", "["))


def _arg_indices(value: str) -> list[int]:
    return [int(item) for item in re.findall(r"\barg(\d+)\b", value)]


def _strip_leading_casts(value: str) -> str:
    text = normalize_expression(value)
    while True:
        updated = re.sub(r"^\([^()]*(?:\*|&)[^()]*\)", "", text)
        if updated == text:
            return text
        text = updated


def _buffer_root_index(value: str) -> int | None:
    text = _strip_leading_casts(value).lstrip("(")
    match = re.match(r"^arg(\d+)\b", text)
    return int(match.group(1)) if match else None


def _placeholder_value(value: str) -> bool:
    return bool(
        re.search(r"\barg(?:n|\d+)\s*expression\b", value, flags=re.I)
        or re.fullmatch(r"(?:expression|unknown|tbd|n/?a)", value.strip(), flags=re.I)
    )


def _schema_error(summary: dict[str, object], parameter_count: int) -> str | None:
    kind = summary.get("kind")
    if kind == "ALLOC":
        required = {"kind", "buffer", "size"}
        if set(summary) != required:
            return "ALLOC must contain exactly kind/buffer/size"
        buffer = str(summary.get("buffer", ""))
        if buffer != "return" and _buffer_root_index(buffer) is None:
            return "ALLOC buffer must be return or rooted at a caller-supplied argN"
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
    for key, value in summary.items():
        if not isinstance(value, str):
            return "all summary values must be strings"
        if key != "kind" and _placeholder_value(value):
            return "summary contains a prompt placeholder instead of a source expression"
        if any(index >= parameter_count for index in _arg_indices(value)):
            return "summary references a nonexistent parameter"
    return None


def canonicalize_summary(function: FunctionSource, summary: dict[str, object]) -> dict[str, str]:
    clean = {str(key): str(value) for key, value in summary.items()}
    for key, value in list(clean.items()):
        if key == "kind":
            clean[key] = value.upper()
            continue
        normalized = value
        for index, parameter in sorted(
            enumerate(function.parameters), key=lambda item: len(item[1]), reverse=True
        ):
            normalized = re.sub(rf"\b{re.escape(parameter)}\b", f"arg{index}", normalized)
        if key in {"length", "size"} and normalized.lower() == "unbounded":
            normalized = "UNBOUNDED"
        clean[key] = normalized
    return clean


def summary_matches_demand(candidate: Candidate, summary: dict[str, str]) -> bool:
    kind = summary.get("kind")
    if kind == "VALUE":
        return candidate.require_return
    if kind == "ALLOC":
        buffer = summary.get("buffer", "")
        if buffer == "return":
            return candidate.require_return
        root = _buffer_root_index(buffer)
        return root is not None and root in set(candidate.required_parameters)
    referenced = set(_arg_indices(summary.get("buffer", ""))) | set(
        _arg_indices(summary.get("length", ""))
    )
    if not candidate.required_parameters:
        return True
    return bool(referenced & set(candidate.required_parameters))


def _substitute_args(expression: str, parameters: tuple[str, ...]) -> str:
    result = expression
    for index in reversed(range(len(parameters))):
        result = re.sub(rf"\barg{index}\b", parameters[index], result)
    return result


def _substitute_call_args(expression: str, arguments: tuple[str, ...]) -> str:
    result = expression
    for index in reversed(range(len(arguments))):
        result = re.sub(rf"\barg{index}\b", f"({arguments[index]})", result)
    return result


def _known_call_indices(name: str, kind: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if kind == "ALLOC" and name in ALLOCATORS:
        return (), ALLOCATORS[name]
    if kind == "WRITE" and name in WRITES:
        buffer, length = WRITES[name]
        return (buffer,), (1, 2) if name == "fread" else (length,)
    if kind == "READ" and name in READS:
        buffer, length = READS[name]
        return (buffer,), (1, 2) if name == "fwrite" else (length,)
    return (), ()


def _expr_reaches(facts, expression: str, call, argument_indices: tuple[int, ...]) -> bool:
    params = _arg_indices(expression)
    if params:
        return all(
            any(facts.parameter_reaches(param, call, index) for index in argument_indices)
            for param in params
        )
    compact = normalize_expression(expression)
    return any(
        compact and compact == normalize_expression(call.arguments.get(index, ""))
        for index in argument_indices
    )


def _normalized_return_expression(value: str) -> str:
    compact = normalize_expression(value)
    if compact.startswith("return"):
        compact = compact[len("return"):]
    if compact.endswith(";"):
        compact = compact[:-1]
    return compact


def _source_call_for_fact(candidate: Candidate, fact_call):
    calls = [
        call for call in candidate.function.calls()
        if not call.indirect and call.line == fact_call.line and call.name == fact_call.name
    ]
    if fact_call.code:
        code = normalize_expression(fact_call.code)
        exact = [call for call in calls if normalize_expression(call.code) == code]
        if len(exact) == 1:
            return exact[0]
    return calls[0] if len(calls) == 1 else None


def _allocation_target_matches(
    candidate: Candidate,
    summary_buffer: str,
    source_call,
    returned: set[str],
) -> bool:
    if summary_buffer == "return":
        return source_call.returned or (
            source_call.result is not None
            and normalize_expression(source_call.result) in returned
        )
    if source_call.result is None:
        return False
    expected = normalize_expression(
        _substitute_args(summary_buffer, candidate.function.parameters)
    )
    return normalize_expression(source_call.result) == expected


def _validate_with_static_facts(candidate: Candidate, summary: dict[str, str], validator) -> tuple[bool, str]:
    facts = validator.facts(candidate)
    kind = summary["kind"]
    if kind == "ALLOC":
        returned = {_normalized_return_expression(value) for value in facts.returns}
        for call in facts.call_list():
            _, size_indices = _known_call_indices(call.name, "ALLOC")
            if not size_indices:
                continue
            source_call = _source_call_for_fact(candidate, call)
            if source_call is None:
                continue
            if _allocation_target_matches(
                candidate, summary["buffer"], source_call, returned
            ) and _expr_reaches(facts, summary["size"], call, size_indices):
                target = "return value" if summary["buffer"] == "return" else summary["buffer"]
                return True, f"local static flow verified allocation capacity for {target}"
        return False, "local static flow did not prove the declared allocation capacity"
    if kind in {"READ", "WRITE"}:
        for call in facts.call_list():
            buffer_indices, length_indices = _known_call_indices(call.name, kind)
            if not buffer_indices or not length_indices:
                continue
            if _expr_reaches(facts, summary["buffer"], call, buffer_indices) and _expr_reaches(
                facts, summary["length"], call, length_indices
            ):
                return True, f"local static flow verified {kind.lower()} semantics through {call.name}"
        return False, f"local static flow did not prove the declared {kind.lower()} semantics"
    if kind == "VALUE":
        expression = normalize_expression(
            _substitute_args(summary["expression"], candidate.function.parameters)
        )
        returned = {_normalized_return_expression(value) for value in facts.returns}
        if expression and expression in returned:
            return True, "local static flow verified the returned value expression"
        arg_indices = _arg_indices(summary["expression"])
        if len(arg_indices) == 1 and arg_indices[0] in facts.return_flows:
            return True, "local static flow verified parameter-to-return propagation"
        return False, "local static flow did not prove the declared returned value"
    return False, f"unsupported semantic kind: {kind}"


def _source_call_returns_value(source_call, facts) -> bool:
    if source_call.returned:
        return True
    if source_call.result is None:
        return False
    returned = {_normalized_return_expression(value) for value in facts.returns}
    return normalize_expression(source_call.result) in returned


def _validate_by_composition(
    candidate: Candidate,
    summary: dict[str, str],
    validator,
    callee_summaries: dict[tuple[str, str], list[dict[str, str]]],
) -> tuple[bool, str]:
    facts = validator.facts(candidate)
    kind = summary["kind"]
    for call in facts.call_list():
        matches = [
            summaries
            for (_path, name), summaries in callee_summaries.items()
            if name == call.name
        ]
        if len(matches) != 1:
            continue
        source_call = _source_call_for_fact(candidate, call)
        if source_call is None:
            continue
        for child in matches[0]:
            if child.get("kind") != kind:
                continue
            if kind in {"READ", "WRITE"}:
                child_buffer_args = tuple(_arg_indices(child.get("buffer", "")))
                child_length_args = tuple(_arg_indices(child.get("length", "")))
                if not child_buffer_args or not child_length_args:
                    continue
                if _expr_reaches(facts, summary["buffer"], call, child_buffer_args) and _expr_reaches(
                    facts, summary["length"], call, child_length_args
                ):
                    return True, f"validated callee composition verified {kind.lower()} through {call.name}"
            elif kind == "ALLOC":
                child_buffer = child.get("buffer", "")
                if child_buffer == "return":
                    if source_call.result is None:
                        continue
                    child_target = normalize_expression(source_call.result)
                else:
                    child_target = normalize_expression(
                        _substitute_call_args(child_buffer, source_call.arguments)
                    )
                requested_target = (
                    "return"
                    if summary["buffer"] == "return"
                    else normalize_expression(
                        _substitute_args(summary["buffer"], candidate.function.parameters)
                    )
                )
                if summary["buffer"] == "return":
                    if not _source_call_returns_value(source_call, facts):
                        continue
                elif child_target != requested_target:
                    continue
                child_size_args = tuple(_arg_indices(child.get("size", "")))
                if child_size_args and not _expr_reaches(
                    facts, summary["size"], call, child_size_args
                ):
                    continue
                return True, f"validated callee composition verified allocation through {call.name}"
            elif kind == "VALUE":
                child_expression = child.get("expression", "")
                child_args = _arg_indices(child_expression)
                if len(child_args) != 1 or child_expression != f"arg{child_args[0]}":
                    continue
                if not _source_call_returns_value(source_call, facts):
                    continue
                if _expr_reaches(facts, summary["expression"], call, (child_args[0],)):
                    return True, f"validated callee composition verified value through {call.name}"
    return False, "no validated callee summary proves the claimed semantic role"


def candidate_validation_error(function: FunctionSource) -> str | None:
    if function.parse_has_error and not function_body_recoverable(function):
        return "candidate function body cannot be structurally recovered"
    return None


def _parameter_pointer_supported(function: FunctionSource, index: int) -> bool:
    if index >= len(function.parameters):
        return False
    if index < len(function.parameter_types) and _type_definitely_pointer(
        function.parameter_types[index]
    ):
        return True
    if index < len(function.parameter_pointer_like) and function.parameter_pointer_like[index]:
        return True
    parameter = function.parameters[index]
    for access in function.direct_memory_accesses():
        if parameter in re.findall(r"\b[A-Za-z_]\w*\b", access.buffer):
            return True
    for call in function.calls():
        if call.result and parameter in re.findall(r"\b[A-Za-z_]\w*\b", call.result):
            return True
    for summary in summaries_for_function(function):
        if summary.get("kind") not in {"READ", "WRITE"}:
            continue
        if _buffer_root_index(summary.get("buffer", "")) == index:
            return True
    return False


def _validation(
    candidate: Candidate,
    summary: dict[str, str],
    passed: bool,
    reason: str,
    *,
    status: str,
) -> Validation:
    function = candidate.function
    return Validation(
        candidate.sample_key,
        function.name,
        function.path,
        function.start_line,
        summary,
        passed,
        reason,
        variant_group=candidate.variant_group,
        variant_count=candidate.variant_count,
        call_lines=candidate.call_lines,
        status=status,
    )


def _static_standard_validation(candidate: Candidate, clean: dict[str, str]) -> Validation | None:
    function = candidate.function
    if clean not in summaries_for_function(function):
        return None
    if _schema_error(clean, len(function.parameters)) is not None:
        return None
    if clean.get("kind") in {"READ", "WRITE"}:
        root = _buffer_root_index(clean.get("buffer", ""))
        if root is None or not _parameter_pointer_supported(function, root):
            return None
    return _validation(
        candidate,
        clean,
        True,
        "standard API role verified directly from source",
        status="VERIFIED",
    )


def validate_summary(
    candidate: Candidate,
    summary: dict[str, object],
    joern,
    callee_summaries: dict[tuple[str, str], list[dict[str, str]]] | None = None,
) -> Validation:
    function = candidate.function
    clean_summary = canonicalize_summary(function, summary)

    error = candidate_validation_error(function)
    if error is None:
        error = _schema_error(clean_summary, len(function.parameters))
    if error is None and not summary_matches_demand(candidate, clean_summary):
        error = "summary does not describe a caller-observable semantic role requested for this candidate"
    if not error and clean_summary.get("kind") in {"READ", "WRITE", "ALLOC"}:
        buffer = clean_summary.get("buffer", "")
        if clean_summary.get("kind") == "ALLOC" and buffer == "return":
            root_index = None
        else:
            root_index = _buffer_root_index(buffer)
            if root_index is None:
                error = f"{clean_summary.get('kind')} buffer must be rooted at a caller-supplied argN"
            elif not _parameter_pointer_supported(function, root_index):
                error = (
                    f"{clean_summary.get('kind')} buffer root arg{root_index} has no source-level pointer/object use"
                )
    if error:
        return _validation(
            candidate,
            clean_summary,
            False,
            error,
            status="REJECTED",
        )

    static = _static_standard_validation(candidate, clean_summary)
    if static is not None:
        return static

    try:
        passed, reason = _validate_with_static_facts(candidate, clean_summary, joern)
        if not passed and callee_summaries:
            composed, composed_reason = _validate_by_composition(
                candidate, clean_summary, joern, callee_summaries
            )
            if composed:
                passed, reason = True, composed_reason
    except (JoernMethodNotFound, JoernTimeout, JoernError) as validation_error:
        return _validation(
            candidate,
            clean_summary,
            False,
            f"local validation context unavailable: {validation_error}",
            status="UNRESOLVED",
        )

    return _validation(
        candidate,
        clean_summary,
        passed,
        reason,
        status="VERIFIED" if passed else "UNRESOLVED",
    )


def _response_content(result: dict[str, object]) -> str:
    try:
        choice = result["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("LLM response has no choices[0].message") from error
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        usage = result.get("usage")
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        raise ValueError(
            f"LLM response was truncated at max_tokens (completion_tokens={completion_tokens!r})"
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
        f"(finish_reason={choice.get('finish_reason')!r}, reasoning_length={reasoning_length}, "
        f"completion_tokens={completion_tokens!r}). The provider produced no final JSON answer."
    )


def _extract_json_object(content: str) -> dict[str, object]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("LLM response is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("LLM response must be one JSON object")
    return value
