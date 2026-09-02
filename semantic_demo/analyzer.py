from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .semantics import ALLOCATORS, READS, UNBOUNDED_WRITES, WRITES, Validation
from .source import FunctionSource
from .z3_reasoner import reason_memory_safety


@dataclass(frozen=True)
class Operation:
    kind: str
    callee: str
    buffer: str
    extent: str
    line: int
    custom: bool


@dataclass(frozen=True)
class Verdict:
    verdict: str
    reason: str
    operations: tuple[Operation, ...]
    constraint_result: dict[str, object] | None = None

    def as_json(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "operations": [asdict(item) for item in self.operations],
            "constraint_result": self.constraint_result,
        }


def _substitute(expression: str, arguments: tuple[str, ...]) -> str:
    result = expression
    for index in reversed(range(len(arguments))):
        result = re.sub(rf"\barg{index}\b", f"({arguments[index]})", result)
    return result


def _c_string_literal_length(expression: str) -> int | None:
    text = expression.strip()
    if not (len(text) >= 2 and text[0] == '"' and text[-1] == '"'):
        return None
    try:
        import ast
        value = ast.literal_eval(text)
    except Exception:
        return None
    if not isinstance(value, str):
        return None
    return len(value.encode()) + 1


def _string_copy_extent(expression: str) -> str:
    literal = _c_string_literal_length(expression)
    if literal is not None:
        return str(literal)
    return f"strlen({expression}) + 1"


def _standard_operations(entry: FunctionSource) -> list[Operation]:
    operations: list[Operation] = []
    for call in entry.calls():
        if call.name in ALLOCATORS:
            indices = ALLOCATORS[call.name]
            if call.name == "calloc" and len(call.arguments) >= 2:
                size = f"({call.arguments[0]}) * ({call.arguments[1]})"
            elif indices and len(call.arguments) > indices[0]:
                size = call.arguments[indices[0]]
            else:
                size = ""
            if size:
                operations.append(Operation("ALLOC", call.name, call.result or "return", size, call.line, False))

        if call.name in WRITES:
            buffer_index, length_index = WRITES[call.name]
            if len(call.arguments) > max(buffer_index, length_index):
                length = call.arguments[length_index]
                if call.name == "fread" and len(call.arguments) >= 3:
                    length = f"({call.arguments[1]}) * ({call.arguments[2]})"
                operations.append(Operation("WRITE", call.name, call.arguments[buffer_index], length, call.line, False))

        if call.name in READS:
            buffer_index, length_index = READS[call.name]
            if len(call.arguments) > max(buffer_index, length_index):
                length = call.arguments[length_index]
                if call.name == "fwrite" and len(call.arguments) >= 3:
                    length = f"({call.arguments[1]}) * ({call.arguments[2]})"
                operations.append(Operation("READ", call.name, call.arguments[buffer_index], length, call.line, False))
                if call.name == "memcmp" and len(call.arguments) >= 3:
                    operations.append(
                        Operation("READ", call.name, call.arguments[1], call.arguments[2], call.line, False)
                    )

        if call.name == "strcpy" and len(call.arguments) >= 2:
            extent = _string_copy_extent(call.arguments[1])
            operations.append(
                Operation("WRITE", call.name, call.arguments[0], extent, call.line, False)
            )
            operations.append(
                Operation("READ", call.name, call.arguments[1], extent, call.line, False)
            )
        elif call.name == "strcat" and len(call.arguments) >= 2:
            extent = _string_copy_extent(call.arguments[1])
            operations.append(
                Operation(
                    "WRITE",
                    call.name,
                    f"{call.arguments[0]} + strlen({call.arguments[0]})",
                    extent,
                    call.line,
                    False,
                )
            )
            operations.append(
                Operation("READ", call.name, call.arguments[1], extent, call.line, False)
            )
        elif call.name in {"sprintf", "vsprintf"} and call.arguments:
            operations.append(
                Operation("WRITE", call.name, call.arguments[0], "UNBOUNDED", call.line, False)
            )
    return operations


def _direct_ast_operations(entry: FunctionSource) -> list[Operation]:
    return [
        Operation(
            access.kind,
            access.origin,
            access.buffer,
            access.extent,
            access.line,
            False,
        )
        for access in entry.direct_memory_accesses()
    ]


def _custom_operations(
    entry: FunctionSource,
    validations: Iterable[Validation],
    variant_counts: dict[tuple[str, str], int] | None = None,
) -> list[Operation]:
    passed: dict[tuple[str, str], dict[int, list[dict[str, str]]]] = {}
    for validation in validations:
        if not validation.passed:
            continue
        key = (validation.source_path, validation.function)
        by_line = passed.setdefault(key, {})
        bucket = by_line.setdefault(validation.source_line, [])
        if validation.summary not in bucket:
            bucket.append(validation.summary)

    summaries_by_name: dict[str, list[list[dict[str, str]]]] = {}
    for (path, name), by_line in passed.items():
        expected = 1 if variant_counts is None else variant_counts.get((path, name), 1)
        if len(by_line) != expected:
            continue
        variants = list(by_line.values())
        common = [
            summary
            for summary in variants[0]
            if all(summary in summaries for summaries in variants[1:])
        ]
        if common:
            summaries_by_name.setdefault(name, []).append(common)

    unique_by_name = {
        name: groups[0]
        for name, groups in summaries_by_name.items()
        if len(groups) == 1
    }

    operations: list[Operation] = []
    for call in entry.calls():
        for summary in unique_by_name.get(call.name, []):
            kind = summary["kind"]
            if kind == "ALLOC":
                operations.append(
                    Operation(
                        "ALLOC",
                        call.name,
                        call.result or "return",
                        _substitute(summary["size"], call.arguments),
                        call.line,
                        True,
                    )
                )
            elif kind in {"READ", "WRITE"}:
                operations.append(
                    Operation(
                        kind,
                        call.name,
                        _substitute(summary["buffer"], call.arguments),
                        _substitute(summary["length"], call.arguments),
                        call.line,
                        True,
                    )
                )
            elif kind == "VALUE" and call.result:
                operations.append(
                    Operation(
                        "VALUE",
                        call.name,
                        call.result,
                        _substitute(summary["expression"], call.arguments),
                        call.line,
                        True,
                    )
                )
    return operations


def analyze(
    entry: FunctionSource,
    validations: Iterable[Validation] = (),
    variant_counts: dict[tuple[str, str], int] | None = None,
) -> Verdict:
    operations = _standard_operations(entry)
    operations.extend(_direct_ast_operations(entry))
    operations.extend(_custom_operations(entry, validations, variant_counts))

    constraint_result = reason_memory_safety(entry, operations)
    constraint_json = constraint_result.as_json()
    if constraint_result.status == "POTENTIAL_VIOLATION":
        return Verdict(
            "VULNERABLE",
            f"Z3: {constraint_result.reason}",
            tuple(operations),
            constraint_json,
        )
    return Verdict(
        "UNKNOWN",
        "verification incomplete: " + constraint_result.reason,
        tuple(operations),
        constraint_json,
    )
