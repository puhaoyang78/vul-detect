from __future__ import annotations

import json
import os
import re
import urllib.request

from . import semantics
from .source import FunctionSource, normalize_expression
from .standard_semantics import STANDARD_LEAF_CALLS, summaries_for_function


MAX_FULL_SOURCE_CHARS = 18000
MAX_SLICE_LINES = 140


def _ids(expression: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_]\w*\b", normalize_expression(expression)))


def _line_number(function: FunctionSource, offset: int) -> int:
    return function.start_line + function.text[:offset].count("\n")


def _assignment_lines(function: FunctionSource) -> list[tuple[int, str, str]]:
    result: list[tuple[int, str, str]] = []
    pattern = re.compile(
        r"(?m)^\s*(?:[A-Za-z_][\w\s*:&<>]*\s+)?([A-Za-z_]\w*)\s*=\s*([^;]+);"
    )
    for match in pattern.finditer(function.text):
        result.append(
            (_line_number(function, match.start()), match.group(1), match.group(2))
        )
    return result


def _slice_source(
    function: FunctionSource,
    endpoint_line: int,
    expressions: tuple[str, ...],
) -> str:
    if len(function.text) <= MAX_FULL_SOURCE_CHARS:
        return function.text

    lines = function.text.splitlines()
    selected: set[int] = set()
    signature_end = next(
        (index for index, line in enumerate(lines[:24]) if "{" in line),
        min(23, len(lines) - 1),
    )
    selected.update(range(signature_end + 1))

    relative = endpoint_line - function.start_line
    if 0 <= relative < len(lines):
        selected.update(range(max(0, relative - 3), min(len(lines), relative + 4)))

    relevant = set().union(*(_ids(expression) for expression in expressions))
    assignments = _assignment_lines(function)
    changed = True
    while changed:
        changed = False
        for line, left, right in reversed(assignments):
            if left not in relevant or line > endpoint_line:
                continue
            before = len(relevant)
            relevant.update(_ids(right))
            changed |= len(relevant) != before
            index = line - function.start_line
            selected.update(range(max(0, index - 1), min(len(lines), index + 2)))

    for index, line in enumerate(lines):
        if function.start_line + index > endpoint_line:
            break
        stripped = line.strip()
        if stripped.startswith(("if", "else if", "while", "for")) and _ids(stripped) & relevant:
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
            chunks.append("/* ... irrelevant source omitted ... */")
        chunks.append(lines[index])
        previous = index
    return "\n".join(chunks)


def _dependency_names(
    function: FunctionSource,
    expressions: tuple[str, ...],
    line: int,
) -> set[str]:
    relations = {
        normalize_expression(left): normalize_expression(right)
        for left, right in function.value_relations_before(line)
    }
    pending = [name for expression in expressions for name in _ids(expression)]
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        replacement = relations.get(name)
        if replacement:
            pending.extend(_ids(replacement) - seen)
    return seen


def _return_endpoints(candidate):
    if not candidate.require_return or not candidate.function.has_value_return():
        return
    for match in re.finditer(r"\breturn\s+([^;]+);", candidate.function.text):
        line = _line_number(candidate.function, match.start())
        yield (
            "return",
            f"return expression at line {line}: {match.group(1).strip()}",
            line,
            (match.group(1),),
        )


def _call_endpoints(candidate):
    required_names = {
        candidate.function.parameters[index]
        for index in candidate.required_parameters
        if 0 <= index < len(candidate.function.parameters)
    }
    returns = [
        (
            _line_number(candidate.function, match.start()),
            match.group(1),
        )
        for match in re.finditer(r"\breturn\s+([^;]+);", candidate.function.text)
    ]
    for call in candidate.function.calls():
        if call.indirect or call.name in STANDARD_LEAF_CALLS:
            continue
        dependencies = _dependency_names(candidate.function, tuple(call.arguments), call.line)
        return_dependency = bool(
            candidate.require_return
            and call.result
            and any(
                normalize_expression(call.result)
                in _dependency_names(candidate.function, (expression,), line)
                for line, expression in returns
            )
        )
        if not return_dependency and not dependencies & required_names:
            continue
        yield (
            "call",
            f"direct custom call {call.name}({', '.join(call.arguments)}) at line {call.line}",
            call.line,
            tuple(call.arguments),
        )


def _endpoints(candidate):
    yield from _return_endpoints(candidate) or ()
    yield from _call_endpoints(candidate) or ()


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
    opener = (
        urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if disable_proxy
        else urllib.request.build_opener()
    )
    with opener.open(request, timeout=180) as response:
        result = json.load(response)
    return semantics._extract_json_object(semantics._response_content(result))


def _demand_text(candidate) -> str:
    parameters = [
        f"arg{index} ({candidate.function.parameters[index]})"
        for index in candidate.required_parameters
        if 0 <= index < len(candidate.function.parameters)
    ]
    parts = []
    if parameters:
        parts.append("caller-observed parameters: " + ", ".join(parameters))
    if candidate.require_return:
        parts.append("caller observes the return value")
    return "; ".join(parts) or "no caller-observable role"


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

    summaries = [
        summary
        for summary in summaries_for_function(candidate.function)
        if semantics.summary_matches_demand(candidate, summary)
    ]

    for endpoint_kind, endpoint_text, endpoint_line, expressions in _endpoints(candidate):
        source_context = (
            candidate.function.text
            if len(candidate.function.text) <= MAX_FULL_SOURCE_CHARS
            else _slice_source(candidate.function, endpoint_line, expressions)
        )
        is_return = endpoint_kind == "return"
        instruction = (
            "Describe only what the caller receives from the function return. "
            "For this endpoint ALLOC must use buffer=return."
            if is_return
            else "Describe only caller-visible effects of this one custom call on the requested boundary."
        )
        examples = (
            '{"kind":"ALLOC","buffer":"return","size":"arg1"}\n'
            '{"kind":"VALUE","target":"return","expression":"arg0->len"}'
            if is_return
            else '{"kind":"ALLOC","buffer":"arg0->data","size":"arg1"}\n'
            '{"kind":"READ","buffer":"arg0","length":"arg2"}\n'
            '{"kind":"WRITE","buffer":"arg0->data","length":"arg1 + 1"}'
        )

        prompt = f"""Normalize one caller-observable semantic endpoint in this C/C++ function.
Endpoint: {endpoint_text}
Caller demand: {_demand_text(candidate)}
{instruction}

Return exactly one JSON object with key summaries and at most four summaries.
Use positional arg0, arg1, ... names. Every expression must come from the shown source.
Valid forms for this endpoint:
{examples}

Do not infer vulnerability labels, guards, caller behavior, or effects not visible here.
Never emit placeholders such as "argN expression" or "unknown".
Emit {{"summaries":[]}} when the source is insufficient.

Function: {candidate.function.name}
Parameters: {json.dumps(list(candidate.function.parameters))}
Source context:
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
        raw_summaries = parsed.get("summaries")
        if not isinstance(raw_summaries, list):
            raise ValueError("LLM response summaries must be a list")
        if len(raw_summaries) > 4:
            raise ValueError("LLM response exceeds the endpoint summary bound")

        for raw in raw_summaries:
            if not isinstance(raw, dict):
                continue
            clean = semantics.canonicalize_summary(candidate.function, raw)
            if semantics._schema_error(clean, len(candidate.function.parameters)) is not None:
                continue
            if is_return:
                if clean.get("kind") not in {"ALLOC", "VALUE"}:
                    continue
                if clean.get("kind") == "ALLOC" and clean.get("buffer") != "return":
                    continue
            if not semantics.summary_matches_demand(candidate, clean):
                continue
            if clean not in summaries:
                summaries.append(clean)
    return summaries
