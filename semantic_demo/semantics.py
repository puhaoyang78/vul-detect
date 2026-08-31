from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .source import FunctionSource, GitRepository, normalize_expression


ALLOCATORS = {
    "malloc": 0,
    "calloc": 0,
    "realloc": 1,
    "kmalloc": 0,
    "kzalloc": 0,
    "vmalloc": 0,
}
WRITES = {
    "memcpy": (0, 2),
    "memmove": (0, 2),
    "read": (1, 2),
    "recv": (1, 2),
    "fread": (0, 1),
    "ReadFile": (1, 2),
}
UNBOUNDED_WRITES = {"sprintf", "strcpy", "strcat", "vsprintf"}

IGNORED_CALLS = set(ALLOCATORS) | set(WRITES) | UNBOUNDED_WRITES | {
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
    "memcmp",
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
    """Keep the LLM input small with a source-only memory-operation filter."""
    lowered = function.name.lower()
    name_hints = (
        "alloc",
        "append",
        "bound",
        "check",
        "copy",
        "ensure",
        "length",
        "memory",
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
    elif kind == "WRITE":
        required = {"kind", "buffer", "length"}
        if set(summary) != required:
            return "WRITE must contain exactly kind/buffer/length"
    elif kind == "GUARD":
        required = {"kind", "relation"}
        if set(summary) != required:
            return "GUARD must contain exactly kind/relation"
    else:
        return "kind must be ALLOC, WRITE, or GUARD"

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


def _identifier_tokens(expression: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression))


def _tainted_tokens(function: FunctionSource, source_expression: str) -> set[str]:
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


def _sink_arguments(call_name: str, arguments: tuple[str, ...], role: str) -> tuple[str, ...]:
    if role == "alloc" and call_name in ALLOCATORS:
        if call_name == "calloc" and len(arguments) >= 2:
            return arguments[:2]
        index = ALLOCATORS[call_name]
        return arguments[index : index + 1]
    if role in {"write_buffer", "write_length"} and call_name in WRITES:
        buffer_index, length_index = WRITES[call_name]
        index = buffer_index if role == "write_buffer" else length_index
        if call_name == "fread" and role == "write_length":
            return arguments[1:3]
        return arguments[index : index + 1]
    return arguments


def _flow_visible(function: FunctionSource, source_expression: str, role: str) -> bool:
    tokens = _tainted_tokens(function, source_expression)
    if not tokens:
        return False
    for call in function.calls():
        predicate = _looks_like_alloc if role == "alloc" else _looks_like_write
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


def _write_flow_visible(
    function: FunctionSource, buffer_expression: str, length_expression: str
) -> bool:
    for call in function.calls():
        if not _looks_like_write(call.name):
            continue
        buffer_sink = _sink_arguments(call.name, call.arguments, "write_buffer")
        length_sink = _sink_arguments(call.name, call.arguments, "write_length")
        if _expression_reaches(function, buffer_expression, buffer_sink) and _expression_reaches(
            function, length_expression, length_sink
        ):
            return True
    return False


def validate_summary(candidate: Candidate, summary: dict[str, object]) -> Validation:
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
        and clean_summary.get("kind") == "WRITE"
        and not _arg_indices(clean_summary.get("buffer", ""))
    ):
        error = "WRITE buffer must be rooted at a caller-supplied argN"
    if error:
        return Validation(
            candidate.sample_key, function.name, function.path, clean_summary, False, error
        )

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

    elif kind == "WRITE":
        buffer_expr = _substitute_args(clean_summary["buffer"], function.parameters)
        length_expr = _substitute_args(clean_summary["length"], function.parameters)
        if not _write_flow_visible(function, buffer_expr, length_expr):
            return Validation(
                candidate.sample_key,
                function.name,
                function.path,
                clean_summary,
                False,
                "declared buffer and length do not reach the same write-like operation",
            )

    else:
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

    return Validation(
        candidate.sample_key,
        function.name,
        function.path,
        clean_summary,
        True,
        "validated against vulnerable-revision source",
    )


def rule_normalize(candidate: Candidate) -> list[dict[str, str]]:
    function = candidate.function
    summaries: list[dict[str, str]] = []
    for index, parameter in enumerate(function.parameters):
        if _flow_visible(function, parameter, "alloc") and "return" in function.text:
            summaries.append({"kind": "ALLOC", "buffer": "return", "size": f"arg{index}"})
            break

    write_calls = [call for call in function.calls() if _looks_like_write(call.name)]
    for call in write_calls:
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
            scalar_hits = [index for index in parameter_hits if index not in pointer_hits]
            if pointer_hits and scalar_hits:
                summaries.append(
                    {
                        "kind": "WRITE",
                        "buffer": f"arg{pointer_hits[-1]}",
                        "length": f"arg{scalar_hits[-1]}",
                    }
                )
                break
    return summaries


def llm_normalize(candidate: Candidate) -> list[dict[str, str]]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    prompt = f"""Normalize only the memory semantics implemented by this candidate C function.
Return one JSON object with key summaries, whose value is a JSON array. Use only these exact forms:
{{"kind":"ALLOC","buffer":"return","size":"arg0"}}
{{"kind":"WRITE","buffer":"arg0","length":"arg2"}}
{{"kind":"GUARD","relation":"arg1 <= arg0"}}
Every source parameter reference must be replaced with its positional argN form. A field is
written as arg0->field, never with the source parameter name. Do not emit local variables or
internal buffers. ALLOC applies only to a pointer returned to the caller. WRITE applies only when
the destination is rooted at a caller-supplied argument. GUARD applies only to a comparison that
constrains an argument's memory extent. Expressions may combine argN with constants, fields, and
arithmetic. Emit {{"summaries":[]}} when none applies. Do not discuss vulnerabilities and do not
inspect anything beyond this function.

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
            "max_tokens": 256,
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
