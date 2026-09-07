from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .source import FunctionSource, normalize_expression


STANDARD_LEAF_CALLS = {
    "malloc", "calloc", "realloc", "kmalloc", "kzalloc", "vmalloc",
    "memcpy", "memmove", "mempcpy", "memset", "memcmp", "bcopy",
    "bzero", "explicit_bzero",
    "read", "recv", "recvfrom", "fread",
    "write", "send", "sendto", "fwrite", "ReadFile",
    "strcpy", "strcat", "strncpy", "strncat", "strlcpy", "strlcat",
    "sprintf", "vsprintf", "snprintf", "vsnprintf",
    "free", "strlen", "strnlen", "sizeof", "strcmp", "strncmp",
    "strchr", "strrchr", "strstr", "memchr",
}


@dataclass(frozen=True)
class StandardEffect:
    kind: str
    buffer: str
    extent: str


def _allocation_extent(call) -> str | None:
    args = call.arguments
    if call.name in {"malloc", "kmalloc", "kzalloc", "vmalloc"} and args:
        return args[0]
    if call.name == "calloc" and len(args) >= 2:
        return f"({args[0]}) * ({args[1]})"
    if call.name == "realloc" and len(args) >= 2:
        return args[1]
    return None


def effects_for_call(call) -> list[StandardEffect]:
    """Return source-level caller-visible standard API effects."""
    args = call.arguments
    name = call.name
    effects: list[StandardEffect] = []

    allocation_extent = _allocation_extent(call)
    if allocation_extent is not None and (call.result or call.returned):
        effects.append(
            StandardEffect(
                "ALLOC",
                call.result or "return",
                allocation_extent,
            )
        )

    if name in {"memcpy", "memmove", "mempcpy"} and len(args) >= 3:
        effects.extend([
            StandardEffect("WRITE", args[0], args[2]),
            StandardEffect("READ", args[1], args[2]),
        ])
    elif name == "bcopy" and len(args) >= 3:
        effects.extend([
            StandardEffect("READ", args[0], args[2]),
            StandardEffect("WRITE", args[1], args[2]),
        ])
    elif name == "memset" and len(args) >= 3:
        effects.append(StandardEffect("WRITE", args[0], args[2]))
    elif name in {"bzero", "explicit_bzero"} and len(args) >= 2:
        effects.append(StandardEffect("WRITE", args[0], args[1]))
    elif name in {"read", "recv", "recvfrom"} and len(args) >= 3:
        effects.append(StandardEffect("WRITE", args[1], args[2]))
    elif name == "ReadFile" and len(args) >= 3:
        effects.append(StandardEffect("WRITE", args[1], args[2]))
    elif name in {"write", "send", "sendto"} and len(args) >= 3:
        effects.append(StandardEffect("READ", args[1], args[2]))
    elif name == "fread" and len(args) >= 3:
        effects.append(StandardEffect("WRITE", args[0], f"({args[1]}) * ({args[2]})"))
    elif name == "fwrite" and len(args) >= 3:
        effects.append(StandardEffect("READ", args[0], f"({args[1]}) * ({args[2]})"))
    elif name == "memcmp" and len(args) >= 3:
        effects.extend([
            StandardEffect("READ", args[0], args[2]),
            StandardEffect("READ", args[1], args[2]),
        ])
    elif name == "strcpy" and len(args) >= 2:
        extent = f"strlen({args[1]}) + 1"
        effects.extend([
            StandardEffect("WRITE", args[0], extent),
            StandardEffect("READ", args[1], extent),
        ])
    elif name == "strcat" and len(args) >= 2:
        source_extent = f"strlen({args[1]}) + 1"
        effects.extend([
            StandardEffect("READ", args[0], f"strlen({args[0]}) + 1"),
            StandardEffect("READ", args[1], source_extent),
            StandardEffect("WRITE", f"{args[0]} + strlen({args[0]})", source_extent),
        ])
    elif name in {"strncpy", "strlcpy"} and len(args) >= 3:
        effects.extend([
            StandardEffect("WRITE", args[0], args[2]),
            StandardEffect("READ", args[1], args[2]),
        ])
    elif name in {"strncat", "strlcat"} and len(args) >= 3:
        effects.extend([
            StandardEffect("READ", args[0], f"strlen({args[0]}) + 1"),
            StandardEffect("READ", args[1], args[2]),
            StandardEffect("WRITE", f"{args[0]} + strlen({args[0]})", f"({args[2]}) + 1"),
        ])
    elif name in {"snprintf", "vsnprintf"} and len(args) >= 2:
        effects.append(StandardEffect("WRITE", args[0], args[1]))
    elif name in {"sprintf", "vsprintf"} and args:
        effects.append(StandardEffect("WRITE", args[0], "UNBOUNDED"))

    return list(dict.fromkeys(effects))


def _replace_parameters(function: FunctionSource, expression: str) -> str:
    result = expression
    for index, parameter in sorted(
        enumerate(function.parameters), key=lambda item: len(item[1]), reverse=True
    ):
        result = re.sub(rf"\b{re.escape(parameter)}\b", f"arg{index}", result)
    return result


def _result_reaches_return(function: FunctionSource, result: str | None) -> bool:
    if not result:
        return False
    name = normalize_expression(result)
    if not re.fullmatch(r"[A-Za-z_]\w*", name):
        return False
    for match in re.finditer(r"\breturn\s+([^;]+);", function.text):
        line = function.start_line + function.text[: match.start()].count("\n")
        relations = {
            normalize_expression(left): normalize_expression(right)
            for left, right in function.value_relations_before(line)
        }
        pending = set(re.findall(r"\b[A-Za-z_]\w*\b", match.group(1)))
        seen: set[str] = set()
        while pending:
            item = pending.pop()
            if item in seen:
                continue
            seen.add(item)
            if item in relations:
                pending.update(re.findall(r"\b[A-Za-z_]\w*\b", relations[item]))
        if name in seen:
            return True
    return False


def summaries_for_call(function: FunctionSource, call) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for effect in effects_for_call(call):
        if effect.kind == "ALLOC":
            if effect.buffer == "return" or _result_reaches_return(function, effect.buffer):
                summaries.append({
                    "kind": "ALLOC",
                    "buffer": "return",
                    "size": _replace_parameters(function, effect.extent),
                })
            continue
        summaries.append({
            "kind": effect.kind,
            "buffer": _replace_parameters(function, effect.buffer),
            "length": _replace_parameters(function, effect.extent),
        })

    unique: list[dict[str, str]] = []
    for summary in summaries:
        if summary not in unique:
            unique.append(summary)
    return unique


def summaries_for_function(function: FunctionSource) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for call in function.calls():
        if call.indirect:
            continue
        for summary in summaries_for_call(function, call):
            if summary not in summaries:
                summaries.append(summary)
    return summaries


def summary_is_static_standard_fact(
    function: FunctionSource, summary: dict[str, str]
) -> bool:
    return summary in summaries_for_function(function)


def standard_seed_expressions(function: FunctionSource) -> Iterable[tuple[int, str]]:
    for access in function.direct_memory_accesses():
        yield access.line, access.buffer
        yield access.line, access.extent
    for call in function.calls():
        if call.indirect:
            continue
        for effect in effects_for_call(call):
            if effect.buffer not in {"", "return"}:
                yield call.line, effect.buffer
            if effect.extent not in {"", "UNBOUNDED"}:
                yield call.line, effect.extent
