from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .joern import JoernValidator
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
    max_functions: int = 64,
) -> list[Candidate]:
    """Build a bounded reachable-function semantic closure.

    Traversal stops at known primitives and repeated functions. The max_functions
    limit is only a resource guard; it is not a semantic hop limit.
    """
    scopes = tuple(scopes)
    queue: list[tuple[FunctionSource, tuple[int, ...]]] = [(entry, ())]
    visited: set[tuple[str, str]] = {(entry.path, entry.name)}
    discovered: dict[tuple[str, str], Candidate] = {}
    order: list[tuple[str, str]] = []

    while queue and len(visited) < max_functions:
        caller, _ = queue.pop(0)
        calls_by_name: dict[str, list[int]] = {}
        for call in caller.calls():
            if call.name in IGNORED_CALLS or call.name == caller.name:
                continue
            calls_by_name.setdefault(call.name, []).append(call.line)

        for name, lines in sorted(calls_by_name.items()):
            function = repository.find_function(
                name, preferred_path=caller.path, scopes=scopes
            )
            if function is None:
                continue
            key = (function.path, function.name)
            if key not in visited and len(visited) < max_functions:
                visited.add(key)
                queue.append((function, tuple(lines)))

            # Include both direct memory candidates and bridge functions. Bridge
            # summaries can later be validated compositionally from child summaries.
            if key not in discovered:
                discovered[key] = Candidate(
                    sample_key=sample_key,
                    function=function,
                    call_lines=tuple(lines),
                )
                order.append(key)

    # Prefer obvious memory candidates first so fixed-point validation can seed
    # summaries from primitives before validating bridge functions.
    order.sort(
        key=lambda key: (
            0 if _is_memory_candidate(discovered[key].function) else 1,
            discovered[key].function.name,
        )
    )
    return [discovered[key] for key in order]


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


def _buffer_root_index(value: str) -> int | None:
    match = re.match(r"^\s*arg(\d+)\b", value)
    return int(match.group(1)) if match else None


def _pointer_like(declaration: str) -> bool:
    return "*" in declaration or "[" in declaration


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


def _family_role_indices(
    name: str, kind: str, argument_count: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Conservative role models for common I/O wrapper families."""
    lowered = name.lower()
    if argument_count < 3:
        return (), ()

    if kind == "WRITE" and (
        "recv" in lowered or lowered.startswith("read") or "_read" in lowered
    ):
        return (1,), (2,)

    if kind == "READ" and (
        "send" in lowered or lowered.startswith("write") or "_write" in lowered
    ):
        return (1,), (2,)

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
            buffer_indices, length_indices = _known_call_indices(call.name, kind)
            if not buffer_indices or not length_indices:
                buffer_indices, length_indices = _family_role_indices(
                    call.name, kind, len(call.arguments)
                )
            if not buffer_indices or not length_indices:
                continue
            if _joern_expr_reaches(
                facts, summary["buffer"], call, buffer_indices
            ) and _joern_expr_reaches(
                facts, summary["length"], call, length_indices
            ):
                return (
                    True,
                    f"Joern verified role-sensitive {kind.lower()} "
                    f"buffer/length flow through {call.name}",
                )
        return (
            False,
            f"Joern found no role-compatible {kind.lower()} operation "
            "matching both buffer and length",
        )

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
    if not error and clean_summary.get("kind") in {"READ", "WRITE"}:
        buffer = clean_summary.get("buffer", "")
        root_index = _buffer_root_index(buffer)
        if root_index is None:
            error = (
                f"{clean_summary.get('kind')} buffer must be rooted at a "
                "caller-supplied argN"
            )
        elif root_index >= len(function.parameter_types) or not _pointer_like(
            function.parameter_types[root_index]
        ):
            error = (
                f"{clean_summary.get('kind')} buffer root arg{root_index} "
                "is not pointer-like in the candidate signature"
            )

    if error:
        return Validation(
            candidate.sample_key, function.name, function.path, clean_summary, False, error
        )

    if joern is not None:
        passed, reason = _validate_with_joern(candidate, clean_summary, joern)
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


def llm_normalize(
    candidate: Candidate,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 2048,
    disable_proxy: bool = False,
) -> list[dict[str, str]]:
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
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
- READ: bytes are consumed FROM a caller-supplied buffer. write(fd, buf, n) or send(fd, buf, n)
  implies READ(buffer=buf,length=n).
- WRITE: bytes are written INTO a caller-supplied buffer. read(fd, buf, n) or recv(fd, buf, n)
  implies WRITE(buffer=buf,length=n).
- For memcpy(dst, src, n), emit WRITE(dst,n) and READ(src,n). Never swap these roles.
- GUARD: a real comparison in this function constrains an argument's size, capacity, index, or offset.
- VALUE: return value is a direct value/cast/arithmetic transformation of caller arguments.

Preserve the candidate signature exactly: arg0 is the first parameter, arg1 the second, etc.
A READ/WRITE buffer must be rooted at a pointer-like caller parameter, while length denotes the
extent/count used by the underlying operation. Every source parameter reference must be replaced
with positional argN form. A field is arg0->field, never the source parameter name. Do not emit
local variables or internal buffers as READ/WRITE buffers.
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
            "max_tokens": max_tokens,
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
    if disable_proxy:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        response_context = opener.open(request, timeout=180)
    else:
        response_context = urllib.request.urlopen(request, timeout=180)
    with response_context as response:
        result = json.load(response)
    content = _response_content(result)
    parsed = _extract_json_object(content)
    summaries = parsed.get("summaries", [])
    if not isinstance(summaries, list):
        raise ValueError("LLM output field summaries is not a list")
    return [item for item in summaries if isinstance(item, dict)]


def _response_content(result: dict[str, object]) -> str:
    try:
        choice = result["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("LLM response has no choices[0].message") from error

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
