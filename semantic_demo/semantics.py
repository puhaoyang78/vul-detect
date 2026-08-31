from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .joern import JoernError, JoernValidator
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

IGNORED_CALLS = set(ALLOCATORS) | set(WRITES) | set(READS) | UNBOUNDED_WRITES | {
    "free",
    "strlen",
    "sizeof",
    "return",
    "if",
    "while",
    "for",
    "switch",
    "assert",
    "memset",
    "strcmp",
    "strchr",
}


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
    summary: dict[str, str]
    passed: bool
    reason: str

    def as_json(self) -> dict[str, object]:
        return asdict(self)


def discover_candidates(
    sample_key: str,
    repository: GitRepository,
    entry: FunctionSource,
    scopes: Iterable[str],
) -> list[Candidate]:
    by_name: dict[str, list[int]] = {}
    for call in entry.calls():
        if call.name in IGNORED_CALLS or call.name == entry.name:
            continue
        by_name.setdefault(call.name, []).append(call.line)

    candidates: list[Candidate] = []
    for name, lines in sorted(by_name.items()):
        function = repository.find_function(name, preferred_path=entry.path, scopes=scopes)
        if function is None or not _is_memory_candidate(function):
            continue
        candidates.append(
            Candidate(sample_key=sample_key, function=function, call_lines=tuple(lines))
        )
    return candidates


def _is_memory_candidate(function: FunctionSource) -> bool:
    """Keep LLM input focused while allowing value/size helpers."""
    lowered = function.name.lower()
    name_hints = (
        "alloc",
        "append",
        "bound",
        "capacity",
        "check",
        "copy",
        "ensure",
        "length",
        "memory",
        "offset",
        "read",
        "recv",
        "size",
        "valid",
        "write",
    )
    if any(hint in lowered for hint in name_hints):
        return True
    return any(
        _looks_like_alloc(call.name)
        or _looks_like_write(call.name)
        or _looks_like_read(call.name)
        or call.name in UNBOUNDED_WRITES
        for call in function.calls()
    )


def _arg_indices(value: str) -> list[int]:
    return [int(item) for item in re.findall(r"\barg(\d+)\b", value)]


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
    elif kind == "GUARD":
        required = {"kind", "relation"}
        if set(summary) != required:
            return "GUARD must contain exactly kind/relation"
    elif kind == "VALUE":
        required = {"kind", "target", "expression"}
        if set(summary) != required or summary.get("target") != "return":
            return "VALUE must contain exactly kind/target/expression and target=return"
    else:
        return "kind must be ALLOC, READ, WRITE, GUARD, or VALUE"

    for value in summary.values():
        if not isinstance(value, str):
            return "all summary values must be strings"
        if any(index >= parameter_count for index in _arg_indices(value)):
            return "summary references a nonexistent parameter"
    return None


def _substitute_args(expression: str, parameters: tuple[str, ...]) -> str:
    result = expression
    for index in reversed(range(len(parameters))):
        result = re.sub(rf"\barg{index}\b", parameters[index], result)
    return result


def _looks_like_alloc(name: str) -> bool:
    lowered = name.lower()
    return name in ALLOCATORS or any(
        token in lowered for token in ("alloc", "malloc", "realloc", "resize")
    )


def _looks_like_write(name: str) -> bool:
    lowered = name.lower()
    return name in WRITES or any(
        token in lowered for token in ("memcpy", "memmove", "read", "recv", "copy")
    )


def _looks_like_read(name: str) -> bool:
    lowered = name.lower()
    return name in READS or any(
        token in lowered for token in ("memcmp", "write", "send", "consume", "parse")
    )


def _identifier_tokens(expression: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression))


def _tainted_tokens(function: FunctionSource, source_expression: str) -> set[str]:
    """Small fallback used only when Joern is disabled in unit tests/debugging."""
    tokens = _identifier_tokens(source_expression)
    assignments = re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);", function.text, flags=re.S
    )
    changed = True
    while changed:
        changed = False
        for target, value in assignments:
            if target not in tokens and tokens & _identifier_tokens(value):
                tokens.add(target)
                changed = True
    return tokens


def _sink_arguments(
    call_name: str, arguments: tuple[str, ...], role: str
) -> tuple[str, ...]:
    if role == "alloc" and call_name in ALLOCATORS:
        indices = ALLOCATORS[call_name]
        return tuple(arguments[index] for index in indices if index < len(arguments))
    if role in {"write_buffer", "write_length"} and call_name in WRITES:
        buffer_index, length_index = WRITES[call_name]
        if call_name == "fread" and role == "write_length" and len(arguments) >= 3:
            return arguments[1:3]
        index = buffer_index if role == "write_buffer" else length_index
        return arguments[index : index + 1]
    if role in {"read_buffer", "read_length"} and call_name in READS:
        buffer_index, length_index = READS[call_name]
        if call_name == "fwrite" and role == "read_length" and len(arguments) >= 3:
            return arguments[1:3]
        index = buffer_index if role == "read_buffer" else length_index
        return arguments[index : index + 1]
    return arguments


def _flow_visible(function: FunctionSource, source_expression: str, role: str) -> bool:
    tokens = _tainted_tokens(function, source_expression)
    if not tokens:
        return False
    for call in function.calls():
        if role == "alloc":
            predicate = _looks_like_alloc
        elif role.startswith("read_"):
            predicate = _looks_like_read
        else:
            predicate = _looks_like_write
        if not predicate(call.name):
            continue
        joined = " ".join(_sink_arguments(call.name, call.arguments, role))
        if tokens & _identifier_tokens(joined):
            return True
    return False


def _expression_reaches(
    function: FunctionSource, expression: str, sink_arguments: tuple[str, ...]
) -> bool:
    joined = " ".join(sink_arguments)
    parameter_sources = _identifier_tokens(expression) & set(function.parameters)
    if parameter_sources:
        sink_tokens = _identifier_tokens(joined)
        return all(
            _tainted_tokens(function, source) & sink_tokens
            for source in parameter_sources
        )
    compact = normalize_expression(expression)
    return bool(compact) and compact in normalize_expression(joined)


def _access_flow_visible(
    function: FunctionSource,
    buffer_expression: str,
    length_expression: str,
    kind: str,
) -> bool:
    role_prefix = "read" if kind == "READ" else "write"
    predicate = _looks_like_read if kind == "READ" else _looks_like_write
    for call in function.calls():
        if not predicate(call.name):
            continue
        buffer_sink = _sink_arguments(
            call.name, call.arguments, f"{role_prefix}_buffer"
        )
        length_sink = _sink_arguments(
            call.name, call.arguments, f"{role_prefix}_length"
        )
        if _expression_reaches(
            function, buffer_expression, buffer_sink
        ) and _expression_reaches(function, length_expression, length_sink):
            return True
    return False


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
        compact and compact in normalize_expression(call.arguments.get(index, ""))
        for index in argument_indices
    )


def _validate_with_joern(
    candidate: Candidate,
    summary: dict[str, str],
    validator: JoernValidator,
) -> tuple[bool, str]:
    facts = validator.facts(candidate)
    kind = summary["kind"]

    if kind == "ALLOC":
        for call in facts.call_list():
            if not _looks_like_alloc(call.name):
                continue
            _, size_indices = _known_call_indices(call.name, "ALLOC")
            if not size_indices:
                size_indices = tuple(call.arguments)
            if _joern_expr_reaches(facts, summary["size"], call, size_indices):
                return True, "Joern verified allocation-size data flow"
        return False, "Joern found no allocation receiving the declared size"

    if kind in {"READ", "WRITE"}:
        for call in facts.call_list():
            predicate = _looks_like_read if kind == "READ" else _looks_like_write
            if not predicate(call.name):
                continue
            buffer_indices, length_indices = _known_call_indices(call.name, kind)
            if not buffer_indices or not length_indices:
                # Unknown project wrapper: require at least two explicit arguments and
                # let data-flow evidence determine which ones carry the summary.
                buffer_indices = tuple(call.arguments)
                length_indices = tuple(call.arguments)
            if _joern_expr_reaches(
                facts, summary["buffer"], call, buffer_indices
            ) and _joern_expr_reaches(
                facts, summary["length"], call, length_indices
            ):
                return True, f"Joern verified {kind.lower()} buffer/length data flow"
        return False, f"Joern found no single {kind.lower()} operation matching buffer and length"

    if kind == "GUARD":
        relation = normalize_expression(
            _substitute_args(summary["relation"], candidate.function.parameters)
        )
        if not relation:
            return False, "empty GUARD relation"
        for condition in facts.conditions:
            compact = normalize_expression(condition)
            if relation in compact or compact in relation:
                return True, "Joern verified guard expression"
        return False, "Joern found no matching control/boolean comparison"

    if kind == "VALUE":
        expression = normalize_expression(
            _substitute_args(summary["expression"], candidate.function.parameters)
        )
        for returned in facts.returns:
            compact = normalize_expression(returned)
            if expression and (expression in compact or compact in expression):
                return True, "Joern verified returned value expression"
        param_indices = _arg_indices(summary["expression"])
        if param_indices and any(index in facts.return_flows for index in param_indices):
            return True, "Joern verified parameter-to-return data flow"
        return False, "Joern found no return matching the declared VALUE expression"

    return False, f"unsupported semantic kind: {kind}"


def validate_summary(
    candidate: Candidate,
    summary: dict[str, object],
    joern: JoernValidator | None = None,
) -> Validation:
    function = candidate.function
    error = _schema_error(summary, len(function.parameters))
    clean_summary = {str(key): str(value) for key, value in summary.items()}

    referenced_names = set().union(
        *(_identifier_tokens(value) for value in clean_summary.values())
    )
    if not error and referenced_names & set(function.parameters):
        error = "source parameter names must be normalized to argN"
    if (
        not error
        and clean_summary.get("kind") in {"READ", "WRITE"}
        and not _arg_indices(clean_summary.get("buffer", ""))
    ):
        error = f"{clean_summary.get('kind')} buffer must be rooted at a caller-supplied argN"

    if error:
        return Validation(
            candidate.sample_key, function.name, function.path, clean_summary, False, error
        )

    if joern is not None:
        try:
            passed, reason = _validate_with_joern(candidate, clean_summary, joern)
        except JoernError as error:
            return Validation(
                candidate.sample_key,
                function.name,
                function.path,
                clean_summary,
                False,
                f"Joern validation failed: {error}",
            )
        return Validation(
            candidate.sample_key,
            function.name,
            function.path,
            clean_summary,
            passed,
            reason,
        )

    # Lightweight fallback for unit tests and debugging without Joern.
    kind = clean_summary["kind"]
    if kind == "ALLOC":
        size = _substitute_args(clean_summary["size"], function.parameters)
        if not _flow_visible(function, size, "alloc"):
            return Validation(
                candidate.sample_key,
                function.name,
                function.path,
                clean_summary,
                False,
                "declared size does not flow to an allocation-like operation",
            )
        if "return" not in function.text:
            return Validation(
                candidate.sample_key,
                function.name,
                function.path,
                clean_summary,
                False,
                "candidate has no returned object",
            )
    elif kind in {"READ", "WRITE"}:
        buffer_expr = _substitute_args(clean_summary["buffer"], function.parameters)
        length_expr = _substitute_args(clean_summary["length"], function.parameters)
        if not _access_flow_visible(function, buffer_expr, length_expr, kind):
            return Validation(
                candidate.sample_key,
                function.name,
                function.path,
                clean_summary,
                False,
                f"declared buffer and length do not reach the same {kind.lower()}-like operation",
            )
    elif kind == "GUARD":
        relation = _substitute_args(clean_summary["relation"], function.parameters)
        match = re.search(r"(.+?)(<=|>=|<|>|==|!=)(.+)", relation)
        if match is None:
            return Validation(
                candidate.sample_key,
                function.name,
                function.path,
                clean_summary,
                False,
                "GUARD relation has no supported comparison",
            )
        left, operator, right = match.groups()
        compact = normalize_expression(function.text)
        expected = normalize_expression(f"{left}{operator}{right}")
        reverse = {
            "<=": ">=",
            ">=": "<=",
            "<": ">",
            ">": "<",
            "==": "==",
            "!=": "!=",
        }[operator]
        reversed_expected = normalize_expression(f"{right}{reverse}{left}")
        if expected not in compact and reversed_expected not in compact:
            return Validation(
                candidate.sample_key,
                function.name,
                function.path,
                clean_summary,
                False,
                "declared comparison is absent from the candidate body",
            )
    else:
        expression = normalize_expression(
            _substitute_args(clean_summary["expression"], function.parameters)
        )
        returns = re.findall(r"\breturn\s+([^;]+);", function.text, flags=re.S)
        if not any(expression in normalize_expression(value) for value in returns):
            return Validation(
                candidate.sample_key,
                function.name,
                function.path,
                clean_summary,
                False,
                "declared VALUE expression is absent from returned expressions",
            )

    return Validation(
        candidate.sample_key,
        function.name,
        function.path,
        clean_summary,
        True,
        "validated by lightweight fallback",
    )


def rule_normalize(candidate: Candidate) -> list[dict[str, str]]:
    function = candidate.function
    summaries: list[dict[str, str]] = []
    for index, parameter in enumerate(function.parameters):
        if _flow_visible(function, parameter, "alloc") and "return" in function.text:
            summaries.append(
                {"kind": "ALLOC", "buffer": "return", "size": f"arg{index}"}
            )
            break

    for kind, predicate in (("WRITE", _looks_like_write), ("READ", _looks_like_read)):
        calls = [call for call in function.calls() if predicate(call.name)]
        for call in calls:
            parameter_hits: list[int] = []
            for index, parameter in enumerate(function.parameters):
                if any(parameter in _identifier_tokens(argument) for argument in call.arguments):
                    parameter_hits.append(index)
            if len(parameter_hits) >= 2:
                pointer_hits = [
                    index
                    for index in parameter_hits
                    if "*" in function.parameter_types[index]
                ]
                scalar_hits = [
                    index for index in parameter_hits if index not in pointer_hits
                ]
                if pointer_hits and scalar_hits:
                    summaries.append(
                        {
                            "kind": kind,
                            "buffer": f"arg{pointer_hits[-1]}",
                            "length": f"arg{scalar_hits[-1]}",
                        }
                    )
                    break

    for match in re.finditer(r"\breturn\s+([^;]+);", function.text, flags=re.S):
        expression = normalize_expression(match.group(1))
        for index, parameter in enumerate(function.parameters):
            if normalize_expression(parameter) == expression:
                summaries.append(
                    {"kind": "VALUE", "target": "return", "expression": f"arg{index}"}
                )
                return summaries
    return summaries


def llm_normalize(candidate: Candidate) -> list[dict[str, str]]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    prompt = f"""Normalize only security-relevant memory semantics implemented by this candidate C function.
Return one JSON object with key summaries, whose value is a JSON array. Use only these exact schemas:
{{"kind":"ALLOC","buffer":"return","size":"arg0"}}
{{"kind":"READ","buffer":"arg0","length":"arg2"}}
{{"kind":"WRITE","buffer":"arg0","length":"arg2"}}
{{"kind":"GUARD","relation":"arg1 <= arg0"}}
{{"kind":"VALUE","target":"return","expression":"arg0"}}

Meaning:
- ALLOC: returned object is allocated/resized with the given size expression.
- READ: function reads/consumes the given length from memory rooted at a caller-supplied argument.
- WRITE: function writes the given length into memory rooted at a caller-supplied argument.
- GUARD: a real comparison in this function constrains an argument's size, capacity, index, or offset.
- VALUE: return value is a direct value/cast/arithmetic transformation of caller arguments.

Every source parameter reference must be replaced with positional argN form. A field is arg0->field,
never the source parameter name. Do not emit local variables or internal buffers as READ/WRITE buffers.
Do not guess implied checks, library contracts, caller behavior, or vulnerability labels. Emit only
semantics directly implemented by this function. Expressions may combine argN with constants, fields,
casts, sizeof, and arithmetic. Emit {{"summaries":[]}} when none applies. Do not inspect anything
beyond this function.

Function: {candidate.function.name}
Parameters: {json.dumps(list(candidate.function.parameters))}
Source:
{candidate.function.text[:12000]}
"""
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 384,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object"},
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
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    content = result["choices"][0]["message"]["content"]
    parsed = _extract_json_object(content)
    summaries = parsed.get("summaries", [])
    if not isinstance(summaries, list):
        raise ValueError("LLM output field summaries is not a list")
    return [item for item in summaries if isinstance(item, dict)]


def _extract_json_object(content: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    try:
        whole = json.loads(content)
    except json.JSONDecodeError:
        whole = None
    if isinstance(whole, list):
        return {"summaries": whole}
    if isinstance(whole, dict):
        return whole
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"LLM response contains no JSON object: {content[:500]!r}")


def load_replay(path: str | Path) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    replay: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["sample_key"], record["source_path"], record["function"])
            replay[key] = record["summaries"]
    return replay
