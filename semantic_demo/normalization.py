from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

from . import semantics
from .source import FunctionSource, normalize_expression
from .standard_semantics import STANDARD_LEAF_CALLS, summaries_for_function


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for name in ("normalization.py", "standard_semantics.py"):
        path = Path(__file__).with_name(name)
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


NORMALIZATION_IMPLEMENTATION_VERSION = _implementation_digest()
MAX_FULL_SOURCE_CHARS = 18000
MAX_SLICE_LINES = 140


def _ids(expression: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_]\w*\b", normalize_expression(expression)))


def _line_number(function: FunctionSource, offset: int) -> int:
    return function.start_line + function.text[:offset].count("\n")


def _assignment_lines(function: FunctionSource) -> list[tuple[int, str, str, str]]:
    result: list[tuple[int, str, str, str]] = []
    pattern = re.compile(
        r"(?m)^\s*(?:[A-Za-z_][\w\s*:&<>]*\s+)?([A-Za-z_]\w*)\s*=\s*([^;]+);"
    )
    for match in pattern.finditer(function.text):
        result.append(
            (_line_number(function, match.start()), match.group(1), match.group(2), match.group(0))
        )
    return result


def _slice_source(function: FunctionSource, endpoint_line: int, expressions: tuple[str, ...]) -> str:
    if len(function.text) <= MAX_FULL_SOURCE_CHARS:
        return function.text
    lines = function.text.splitlines()
    selected: set[int] = set()
    signature_end = next(
        (index for index, line in enumerate(lines[:24]) if "{" in line),
        min(23, len(lines) - 1),
    )
    selected.update(range(0, signature_end + 1))
    relative = endpoint_line - function.start_line
    if 0 <= relative < len(lines):
        selected.update(range(max(0, relative - 3), min(len(lines), relative + 4)))

    relevant = set().union(*(_ids(expression) for expression in expressions))
    relevant &= set(re.findall(r"\b[A-Za-z_]\w*\b", function.text))
    assignments = _assignment_lines(function)
    changed = True
    while changed:
        changed = False
        for line, left, right, _text in reversed(assignments):
            if left not in relevant or line > endpoint_line:
                continue
            before = len(relevant)
            relevant.update(_ids(right))
            if len(relevant) != before:
                changed = True
            index = line - function.start_line
            selected.update(range(max(0, index - 1), min(len(lines), index + 2)))

    for index, line in enumerate(lines):
        if function.start_line + index > endpoint_line:
            break
        stripped = line.strip()
        if not stripped.startswith(("if", "else if", "while", "for")):
            continue
        if _ids(stripped) & relevant:
            selected.update(range(max(0, index - 1), min(len(lines), index + 2)))

    ordered = sorted(selected)
    if len(ordered) > MAX_SLICE_LINES:
        head = ordered[: signature_end + 1]
        tail = ordered[-(MAX_SLICE_LINES - len(head)) :]
        ordered = sorted(set(head + tail))

    chunks: list[str] = []
    previous = None
    for index in ordered:
        if previous is not None and index > previous + 1:
            chunks.append("/* ... irrelevant source omitted by static slice ... */")
        chunks.append(lines[index])
        previous = index
    return "\n".join(chunks)


def _endpoints(function: FunctionSource):
    if function.has_value_return():
        yield "return", "function return statements", None, tuple(function.parameters)
    for call in function.calls():
        if call.indirect or call.name in STANDARD_LEAF_CALLS:
            continue
        yield (
            "call",
            f"direct call {call.name}({', '.join(call.arguments)}) at line {call.line}",
            call.line,
            tuple(call.arguments),
        )


def _request_json(
    *,
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int,
    disable_proxy: bool,
    response_schema: dict[str, object] | None,
) -> dict[str, object]:
    response_format: dict[str, object] = {"type": "json_object"}
    if response_schema is not None:
        response_format["schema"] = response_schema
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": response_format,
    }).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    if disable_proxy:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        context = opener.open(request, timeout=180)
    else:
        context = urllib.request.urlopen(request, timeout=180)
    with context as response:
        result = json.load(response)
    return semantics._extract_json_object(semantics._response_content(result))


def llm_normalize(
    candidate,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 512,
    disable_proxy: bool = False,
    response_schema: dict[str, object] | None = None,
):
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    summaries = summaries_for_function(candidate.function)
    for endpoint_kind, endpoint_text, endpoint_line, expressions in _endpoints(candidate.function):
        if endpoint_kind == "return":
            source_context = (
                candidate.function.text
                if len(candidate.function.text) <= MAX_FULL_SOURCE_CHARS
                else _slice_source(candidate.function, candidate.function.end_line, expressions)
            )
            allowed = "ALLOC or VALUE"
            instruction = "Report only a caller-visible return relation."
        else:
            assert endpoint_line is not None
            source_context = _slice_source(candidate.function, endpoint_line, expressions)
            allowed = "ALLOC, READ, WRITE, or VALUE"
            instruction = (
                "Report only caller-visible semantics mediated by this one direct custom call. "
                "Do not infer effects from unrelated calls."
            )

        prompt = f"""Normalize one statically selected semantic endpoint in this C/C++ function.
Endpoint: {endpoint_text}
Allowed summary kinds: {allowed}
{instruction}

Return exactly one JSON object with key summaries. The array may contain at most four summaries.
Use only:
{{"kind":"ALLOC","buffer":"return","size":"argN expression"}}
{{"kind":"READ","buffer":"argN expression","length":"argN expression"}}
{{"kind":"WRITE","buffer":"argN expression","length":"argN expression"}}
{{"kind":"VALUE","target":"return","expression":"argN expression"}}

Use positional argN names only. Do not infer vulnerability labels, guards, caller behavior,
or unresolved function-pointer behavior. Emit {{"summaries":[]}} when the slice is insufficient.

Function: {candidate.function.name}
Parameters: {json.dumps(list(candidate.function.parameters))}
Statically generated relevance slice:
{source_context}
"""
        parsed = _request_json(
            prompt=prompt,
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            disable_proxy=disable_proxy,
            response_schema=response_schema,
        )
        endpoint_summaries = parsed.get("summaries")
        if not isinstance(endpoint_summaries, list):
            raise ValueError("LLM response summaries must be a list")
        if len(endpoint_summaries) > 4:
            raise ValueError("LLM response exceeds the endpoint summary bound")
        for raw in endpoint_summaries:
            if not isinstance(raw, dict):
                continue
            clean = semantics.canonicalize_summary(candidate.function, raw)
            error = semantics._schema_error(clean, len(candidate.function.parameters))
            if error is not None:
                continue
            if endpoint_kind == "return" and clean.get("kind") not in {"ALLOC", "VALUE"}:
                continue
            if clean not in summaries:
                summaries.append(clean)
    return summaries
