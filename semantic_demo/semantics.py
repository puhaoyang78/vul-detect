from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from .joern import JoernError, JoernMethodNotFound, JoernTimeout, JoernValidator
from .source import FunctionSource, function_body_recoverable, normalize_expression
from .standard_semantics import summaries_for_function


ALLOCATORS = {
    "malloc": (0,), "calloc": (0, 1), "realloc": (1,),
    "kmalloc": (0,), "kzalloc": (0,), "vmalloc": (0,),
}
WRITES = {
    "memcpy": (0, 2), "memmove": (0, 2), "memset": (0, 2),
    "read": (1, 2), "recv": (1, 2), "recvfrom": (1, 2),
    "fread": (0, 1), "ReadFile": (1, 2),
}
READS = {
    "memcpy": (1, 2), "memmove": (1, 2), "write": (1, 2),
    "send": (1, 2), "sendto": (1, 2), "fwrite": (0, 1),
    "memcmp": (0, 2),
}
UNBOUNDED_WRITES = {"sprintf", "strcpy", "strcat", "vsprintf"}
NORMALIZATION_SCHEMA_VERSION = 6
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

    def as_json(self) -> dict[str, object]:
        return asdict(self)


def _type_definitely_pointer(type_text: str) -> bool:
    compact = "".join(type_text.split())
    return any(token in compact for token in ("*", "&", "["))


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


def canonicalize_summary(function: FunctionSource, summary: dict[str, object]) -> dict[str, str]:
    clean = {str(key): str(value) for key, value in summary.items()}
    for key, value in list(clean.items()):
        if key == "kind":
            continue
        normalized = value
        for index, parameter in reversed(list(enumerate(function.parameters))):
            normalized = re.sub(rf"\b{re.escape(parameter)}\b", f"arg{index}", normalized)
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
        return (buffer,), (1, 2) if name == "fread" else (length,)
    if kind == "READ" and name in READS:
        buffer, length = READS[name]
        return (buffer,), (1, 2) if name == "fwrite" else (length,)
    return (), ()


def _joern_expr_reaches(facts, expression: str, call, argument_indices: tuple[int, ...]) -> bool:
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


def _source_call_for_joern(candidate: Candidate, joern_call):
    candidates = [
        call for call in candidate.function.calls()
        if not call.indirect and call.line == joern_call.line and call.name == joern_call.name
    ]
    if joern_call.code:
        joern_code = normalize_expression(joern_call.code)
        exact = [call for call in candidates if normalize_expression(call.code) == joern_code]
        if len(exact) == 1:
            return exact[0]
    return candidates[0] if len(candidates) == 1 else None


def _validate_with_joern(candidate: Candidate, summary: dict[str, str], validator: JoernValidator) -> tuple[bool, str]:
    facts = validator.facts(candidate)
    kind = summary["kind"]
    if kind == "ALLOC":
        returned = {_normalized_return_expression(value) for value in facts.returns}
        for call in facts.call_list():
            _, size_indices = _known_call_indices(call.name, "ALLOC")
            if not size_indices:
                continue
            source_call = _source_call_for_joern(candidate, call)
            if source_call is None:
                continue
            returns_allocation = source_call.returned or (
                source_call.result is not None and normalize_expression(source_call.result) in returned
            )
            if returns_allocation and _joern_expr_reaches(facts, summary["size"], call, size_indices):
                return True, "Joern verified returned allocation and its size flow"
        return False, "Joern found no returned specified allocator matching the declared size"
    if kind in {"READ", "WRITE"}:
        for call in facts.call_list():
            buffer_indices, length_indices = _known_call_indices(call.name, kind)
            if not buffer_indices or not length_indices:
                continue
            if _joern_expr_reaches(facts, summary["buffer"], call, buffer_indices) and _joern_expr_reaches(
                facts, summary["length"], call, length_indices
            ):
                return True, f"Joern verified {kind.lower()} flow to specified API {call.name}"
        return False, f"Joern found no specified API matching the declared {kind.lower()}"
    if kind == "VALUE":
        expression = normalize_expression(_substitute_args(summary["expression"], candidate.function.parameters))
        returned = {_normalized_return_expression(value) for value in facts.returns}
        if expression and expression in returned:
            return True, "Joern verified exact returned value expression"
        return False, "Joern found no exact return matching the declared VALUE expression"
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
    validator: JoernValidator,
    callee_summaries: dict[tuple[str, str], list[dict[str, str]]],
) -> tuple[bool, str]:
    facts = validator.facts(candidate)
    kind = summary["kind"]
    for call in facts.call_list():
        matches = [summaries for (_path, name), summaries in callee_summaries.items() if name == call.name]
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
                if _joern_expr_reaches(facts, summary["buffer"], call, child_buffer_args) and _joern_expr_reaches(
                    facts, summary["length"], call, child_length_args
                ):
                    return True, f"composition verified {kind.lower()} through validated callee summary {call.name}"
            elif kind == "ALLOC":
                child_size_args = tuple(_arg_indices(child.get("size", "")))
                if not child_size_args or not _joern_expr_reaches(facts, summary["size"], call, child_size_args):
                    continue
                source_call = _source_call_for_joern(candidate, call)
                if source_call is not None and _source_call_returns_value(source_call, facts):
                    return True, f"composition verified allocation through validated callee summary {call.name}"
            elif kind == "VALUE":
                child_expression = child.get("expression", "")
                child_args = _arg_indices(child_expression)
                if len(child_args) != 1 or child_expression != f"arg{child_args[0]}":
                    continue
                source_call = _source_call_for_joern(candidate, call)
                if source_call is None or not _source_call_returns_value(source_call, facts):
                    continue
                if _joern_expr_reaches(facts, summary["expression"], call, (child_args[0],)):
                    return True, f"composition verified value through validated callee summary {call.name}"
    return False, "no validated callee summary composes to the claimed semantic role"


def candidate_validation_error(function: FunctionSource) -> str | None:
    if function.parse_has_error and not function_body_recoverable(function):
        return "candidate function body cannot be structurally recovered"
    return None


def _static_standard_validation(candidate: Candidate, clean: dict[str, str]) -> Validation | None:
    function = candidate.function
    if clean not in summaries_for_function(function):
        return None
    if _schema_error(clean, len(function.parameters)) is not None:
        return None
    if clean.get("kind") in {"READ", "WRITE"}:
        root = _buffer_root_index(clean.get("buffer", ""))
        if root is None or root >= len(function.parameter_types) or not _type_definitely_pointer(function.parameter_types[root]):
            return None
    return Validation(
        candidate.sample_key, function.name, function.path, function.start_line,
        clean, True, "standard API role verified from the statically resolved wrapper call",
        variant_group=candidate.variant_group, variant_count=candidate.variant_count,
    )


def validate_summary(
    candidate: Candidate,
    summary: dict[str, object],
    joern: JoernValidator,
    callee_summaries: dict[tuple[str, str], list[dict[str, str]]] | None = None,
) -> Validation:
    function = candidate.function
    clean_summary = canonicalize_summary(function, summary)
    static = _static_standard_validation(candidate, clean_summary)
    if static is not None:
        return static

    error = candidate_validation_error(function)
    if error is None:
        error = _schema_error(clean_summary, len(function.parameters))
    if not error and clean_summary.get("kind") in {"READ", "WRITE"}:
        root_index = _buffer_root_index(clean_summary.get("buffer", ""))
        if root_index is None:
            error = f"{clean_summary.get('kind')} buffer must be rooted at a caller-supplied argN"
        elif root_index >= len(function.parameter_types) or not _type_definitely_pointer(function.parameter_types[root_index]):
            error = (
                f"{clean_summary.get('kind')} buffer root arg{root_index} is not pointer-like: "
                "pointer semantics are not proven by the candidate signature"
            )
    if error:
        return Validation(
            candidate.sample_key, function.name, function.path, function.start_line,
            clean_summary, False, error,
            variant_group=candidate.variant_group, variant_count=candidate.variant_count,
        )

    try:
        passed, reason = _validate_with_joern(candidate, clean_summary, joern)
        if not passed and callee_summaries:
            composed, composed_reason = _validate_by_composition(candidate, clean_summary, joern, callee_summaries)
            if composed:
                passed, reason = composed, composed_reason
    except JoernMethodNotFound as validation_error:
        passed, reason = False, f"Joern candidate method unavailable: {validation_error}"
    except JoernTimeout as validation_error:
        passed, reason = False, f"Joern candidate validation timed out: {validation_error}"
    except JoernError as validation_error:
        passed, reason = False, f"candidate-local Joern validation unavailable: {validation_error}"
    return Validation(
        candidate.sample_key, function.name, function.path, function.start_line,
        clean_summary, passed, reason,
        variant_group=candidate.variant_group, variant_count=candidate.variant_count,
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
        raise ValueError(f"LLM response was truncated at max_tokens (completion_tokens={completion_tokens!r})")
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
