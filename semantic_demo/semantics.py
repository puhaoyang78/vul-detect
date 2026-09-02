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
    max_functions: int = 128,
) -> list[Candidate]:
    """Traverse the repository-resolved project call graph without name heuristics.

    Standard-library calls are leaves. Ambiguous repository call resolution is
    skipped rather than guessed. Hitting the explicit resource budget aborts
    discovery so a truncated frontier can never silently produce a verdict.
    """
    scopes = tuple(scopes)
    discovered: dict[tuple[str, str], Candidate] = {}
    queue: list[FunctionSource] = [entry]
    expanded: set[tuple[str, str]] = set()

    while queue:
        caller = queue.pop(0)
        caller_key = (caller.path, caller.name)
        if caller_key in expanded:
            continue
        expanded.add(caller_key)

        for call in caller.calls():
            if call.name in STANDARD_CALLS or call.name == caller.name:
                continue
            callee = repository.find_function(
                call.name, preferred_path=caller.path, scopes=scopes
            )
            if callee is None:
                continue
            key = (callee.path, callee.name)
            existing = discovered.get(key)
            lines = set(existing.call_lines if existing else ())
            lines.add(call.line)
            discovered[key] = Candidate(
                sample_key=sample_key,
                function=callee,
                call_lines=tuple(sorted(lines)),
            )
            if len(discovered) > max_functions:
                raise RuntimeError(
                    f"{sample_key}: candidate frontier exceeds explicit budget "
                    f"({max_functions}); analysis aborted rather than truncated"
                )
            if key not in expanded:
                queue.append(callee)

    return sorted(
        discovered.values(),
        key=lambda item: (item.function.path, item.function.name),
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


def _comparison_variants(expression: str) -> set[str]:
    compact = normalize_expression(expression)
    match = re.match(r"^(.*?)(<=|>=|==|!=|<|>)(.*)$", compact)
    if not match:
        return {compact} if compact else set()
    left, operator, right = match.groups()
    reverse = {
        "<=": ">=", ">=": "<=", "<": ">", ">": "<", "==": "==", "!=": "!="
    }[operator]
    return {compact, f"{right}{reverse}{left}"}


def _validate_with_joern(
    candidate: Candidate,
    summary: dict[str, str],
    validator: JoernValidator,
) -> tuple[bool, str]:
    facts = validator.facts(candidate)
    kind = summary["kind"]

    if kind == "ALLOC":
        for call in facts.call_list():
            _, size_indices = _known_call_indices(call.name, "ALLOC")
            if not size_indices:
                continue
            if _joern_expr_reaches(facts, summary["size"], call, size_indices):
                return True, "Joern verified allocation-size flow to a specified allocator"
        return False, "Joern found no specified allocator receiving the declared size"

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
        param_indices = _arg_indices(summary["expression"])
        if (
            len(param_indices) == 1
            and summary["expression"] == f"arg{param_indices[0]}"
            and param_indices[0] in facts.return_flows
        ):
            return True, "Joern verified direct parameter-to-return flow"
        return False, "Joern found no exact return matching the declared VALUE expression"

    return False, f"unsupported semantic kind: {kind}"


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
                if source_call is not None and source_call.returned:
                    return (
                        True,
                        f"composition verified allocation through "
                        f"validated callee summary {call.name}",
                    )

    return False, "no validated callee summary composes to the claimed semantic role"


def validate_summary(
    candidate: Candidate,
    summary: dict[str, object],
    joern: JoernValidator,
    callee_summaries: dict[tuple[str, str], list[dict[str, str]]] | None = None,
) -> Validation:
    function = candidate.function
    error = _schema_error(summary, len(function.parameters))
    clean_summary = canonicalize_summary(function, summary)

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
            candidate.sample_key, function.name, function.path, clean_summary, False, error
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
        clean_summary,
        passed,
        reason,
    )



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

    prompt = f"""Normalize only security-relevant memory semantics implemented by this candidate C/C++ function.
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
- VALUE: return value is a direct value/cast/arithmetic transformation of caller arguments.

Preserve the candidate signature exactly: arg0 is the first parameter, arg1 the second, etc.
A READ/WRITE buffer must be rooted at a pointer-like caller parameter, while length denotes the
extent/count used by the underlying operation. Every source parameter reference must be replaced
with positional argN form. A field is arg0->field, never the source parameter name. Do not emit
local variables or internal buffers as READ/WRITE buffers.
Do not guess implied checks, library contracts, caller behavior, or vulnerability labels. Emit only
semantics directly implemented by this function. Do not emit standalone guard/check summaries:
a comparison without an explicit return contract cannot be propagated safely across a call boundary. Expressions may combine argN with constants, fields,
casts, sizeof, and arithmetic. Emit {{"summaries":[]}} when none applies. Do not inspect anything
beyond this function.

Function: {candidate.function.name}
Parameters: {json.dumps(list(candidate.function.parameters))}
Source:
{candidate.function.text}
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
