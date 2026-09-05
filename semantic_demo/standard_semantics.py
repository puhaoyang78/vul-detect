from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .source import FunctionSource


STANDARD_LEAF_CALLS = {
    "malloc", "calloc", "realloc", "kmalloc", "kzalloc", "vmalloc",
    "memcpy", "memmove", "memset", "memcmp",
    "read", "recv", "recvfrom", "fread",
    "write", "send", "sendto", "fwrite", "ReadFile",
    "strcpy", "strcat", "strncpy", "strncat", "strlcpy", "strlcat",
    "sprintf", "vsprintf", "snprintf", "vsnprintf",
    "free", "strlen", "sizeof", "strcmp", "strchr",
}


@dataclass(frozen=True)
class StandardEffect:
    kind: str
    buffer: str
    extent: str


def effects_for_call(call) -> list[StandardEffect]:
    """Return source-level effects without replacing variables by argN."""
    args = call.arguments
    name = call.name
    effects: list[StandardEffect] = []

    if name in {"memcpy", "memmove"} and len(args) >= 3:
        effects.extend([
            StandardEffect("WRITE", args[0], args[2]),
            StandardEffect("READ", args[1], args[2]),
        ])
    elif name == "memset" and len(args) >= 3:
        effects.append(StandardEffect("WRITE", args[0], args[2]))
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


def summaries_for_call(function: FunctionSource, call) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []

    if call.name in {"malloc", "kmalloc", "kzalloc", "vmalloc"} and call.returned and call.arguments:
        summaries.append({
            "kind": "ALLOC", "buffer": "return",
            "size": _replace_parameters(function, call.arguments[0]),
        })
    elif call.name == "calloc" and call.returned and len(call.arguments) >= 2:
        summaries.append({
            "kind": "ALLOC", "buffer": "return",
            "size": _replace_parameters(function, f"({call.arguments[0]}) * ({call.arguments[1]})"),
        })
    elif call.name == "realloc" and call.returned and len(call.arguments) >= 2:
        summaries.append({
            "kind": "ALLOC", "buffer": "return",
            "size": _replace_parameters(function, call.arguments[1]),
        })

    for effect in effects_for_call(call):
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
    """Source expressions whose data dependencies matter to memory-safety semantics."""
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
