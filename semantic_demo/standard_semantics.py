from __future__ import annotations

import re
from typing import Iterable

from .source import FunctionSource


# One registry is shared by candidate discovery, normalization, validation and
# target analysis. A name belongs here only when we intentionally treat it as a
# semantic leaf rather than recursively resolving its implementation.
STANDARD_LEAF_CALLS = {
    "malloc", "calloc", "realloc", "kmalloc", "kzalloc", "vmalloc",
    "memcpy", "memmove", "memset", "memcmp",
    "read", "recv", "recvfrom", "fread",
    "write", "send", "sendto", "fwrite", "ReadFile",
    "strcpy", "strcat", "strncpy", "strncat", "strlcpy", "strlcat",
    "sprintf", "vsprintf", "snprintf", "vsnprintf",
    "free", "strlen", "sizeof", "strcmp", "strchr",
}


def _replace_parameters(function: FunctionSource, expression: str) -> str:
    result = expression
    for index, parameter in sorted(
        enumerate(function.parameters), key=lambda item: len(item[1]), reverse=True
    ):
        result = re.sub(rf"\b{re.escape(parameter)}\b", f"arg{index}", result)
    return result


def _memory_summary(function: FunctionSource, kind: str, buffer: str, length: str):
    return {
        "kind": kind,
        "buffer": _replace_parameters(function, buffer),
        "length": _replace_parameters(function, length),
    }


def _allocation_summary(function: FunctionSource, size: str):
    return {
        "kind": "ALLOC",
        "buffer": "return",
        "size": _replace_parameters(function, size),
    }


def summaries_for_call(function: FunctionSource, call) -> list[dict[str, str]]:
    """Return conservative caller-visible summaries for an explicit standard API."""
    args = call.arguments
    name = call.name
    summaries: list[dict[str, str]] = []

    if name in {"malloc", "kmalloc", "kzalloc", "vmalloc"} and call.returned and args:
        summaries.append(_allocation_summary(function, args[0]))
    elif name == "calloc" and call.returned and len(args) >= 2:
        summaries.append(_allocation_summary(function, f"({args[0]}) * ({args[1]})"))
    elif name == "realloc" and call.returned and len(args) >= 2:
        summaries.append(_allocation_summary(function, args[1]))

    if name in {"memcpy", "memmove"} and len(args) >= 3:
        summaries.extend([
            _memory_summary(function, "WRITE", args[0], args[2]),
            _memory_summary(function, "READ", args[1], args[2]),
        ])
    elif name == "memset" and len(args) >= 3:
        summaries.append(_memory_summary(function, "WRITE", args[0], args[2]))
    elif name in {"read", "recv", "recvfrom"} and len(args) >= 3:
        summaries.append(_memory_summary(function, "WRITE", args[1], args[2]))
    elif name == "ReadFile" and len(args) >= 3:
        summaries.append(_memory_summary(function, "WRITE", args[1], args[2]))
    elif name in {"write", "send", "sendto"} and len(args) >= 3:
        summaries.append(_memory_summary(function, "READ", args[1], args[2]))
    elif name == "fread" and len(args) >= 3:
        summaries.append(
            _memory_summary(function, "WRITE", args[0], f"({args[1]}) * ({args[2]})")
        )
    elif name == "fwrite" and len(args) >= 3:
        summaries.append(
            _memory_summary(function, "READ", args[0], f"({args[1]}) * ({args[2]})")
        )
    elif name == "memcmp" and len(args) >= 3:
        summaries.extend([
            _memory_summary(function, "READ", args[0], args[2]),
            _memory_summary(function, "READ", args[1], args[2]),
        ])
    elif name == "strcpy" and len(args) >= 2:
        extent = f"strlen({args[1]}) + 1"
        summaries.extend([
            _memory_summary(function, "WRITE", args[0], extent),
            _memory_summary(function, "READ", args[1], extent),
        ])
    elif name == "strcat" and len(args) >= 2:
        source_extent = f"strlen({args[1]}) + 1"
        summaries.extend([
            _memory_summary(function, "READ", args[0], f"strlen({args[0]}) + 1"),
            _memory_summary(function, "READ", args[1], source_extent),
            _memory_summary(
                function,
                "WRITE",
                f"{args[0]} + strlen({args[0]})",
                source_extent,
            ),
        ])
    elif name in {"strncpy", "strlcpy"} and len(args) >= 3:
        summaries.extend([
            _memory_summary(function, "WRITE", args[0], args[2]),
            _memory_summary(function, "READ", args[1], args[2]),
        ])
    elif name in {"strncat", "strlcat"} and len(args) >= 3:
        summaries.extend([
            _memory_summary(function, "READ", args[0], f"strlen({args[0]}) + 1"),
            _memory_summary(function, "READ", args[1], args[2]),
            _memory_summary(
                function,
                "WRITE",
                f"{args[0]} + strlen({args[0]})",
                f"({args[2]}) + 1",
            ),
        ])
    elif name in {"snprintf", "vsnprintf"} and len(args) >= 2:
        # The number of bytes actually stored never exceeds the supplied output
        # capacity. Using that capacity is conservative for bounds reasoning.
        summaries.append(_memory_summary(function, "WRITE", args[0], args[1]))
    elif name in {"sprintf", "vsprintf"} and args:
        summaries.append(_memory_summary(function, "WRITE", args[0], "UNBOUNDED"))

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
    """Expressions whose data dependencies matter to memory-safety semantics."""
    for access in function.direct_memory_accesses():
        yield access.line, access.buffer
        yield access.line, access.extent
    for call in function.calls():
        if call.indirect:
            continue
        for summary in summaries_for_call(function, call):
            for field in ("buffer", "length", "size", "expression"):
                value = summary.get(field)
                if value and value not in {"return", "UNBOUNDED"}:
                    yield call.line, value
