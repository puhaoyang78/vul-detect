from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .semantics import ALLOCATORS, Validation
from .source import FunctionSource
from .standard_semantics import STANDARD_LEAF_CALLS, effects_for_call
from .solver import reason_memory_safety


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


def _direct_allocation_operation(call) -> Operation | None:
    if call.name not in ALLOCATORS or not call.arguments:
        return None
    if call.name == "calloc" and len(call.arguments) >= 2:
        extent = f"({call.arguments[0]}) * ({call.arguments[1]})"
    elif call.name == "realloc" and len(call.arguments) >= 2:
        extent = call.arguments[1]
    else:
        extent = call.arguments[0]
    target = call.result or ("return" if call.returned else "")
    if not target:
        return None
    return Operation("ALLOC", call.name, target, extent, call.line, False)


def _standard_operations(entry: FunctionSource) -> list[Operation]:
    operations: list[Operation] = []
    for call in entry.calls():
        if call.indirect:
            continue
        allocation = _direct_allocation_operation(call)
        if allocation is not None:
            operations.append(allocation)
        for effect in effects_for_call(call):
            operations.append(
                Operation(
                    effect.kind,
                    call.name,
                    effect.buffer,
                    effect.extent,
                    call.line,
                    False,
                )
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


def _accepted_summary_groups(validations: Iterable[Validation]):
    passed: dict[
        tuple[str, str, str],
        dict[int, list[dict[str, str]]],
    ] = {}
    expected_counts: dict[tuple[str, str, str], int] = {}
    for validation in validations:
        if not validation.passed:
            continue
        group_id = validation.variant_group or f"single:{validation.source_line}"
        key = (validation.source_path, validation.function, group_id)
        expected_counts[key] = validation.variant_count
        by_line = passed.setdefault(key, {})
        bucket = by_line.setdefault(validation.source_line, [])
        if validation.summary not in bucket:
            bucket.append(validation.summary)

    groups_by_name: dict[str, list[list[dict[str, str]]]] = {}
    for key, by_line in passed.items():
        _path, name, _group_id = key
        if len(by_line) != expected_counts[key]:
            continue
        members = list(by_line.values())
        common = [
            summary
            for summary in members[0]
            if all(summary in summaries for summaries in members[1:])
        ]
        if common:
            groups_by_name.setdefault(name, []).append(common)
    return {
        name: groups[0]
        for name, groups in groups_by_name.items()
        if len(groups) == 1
    }


def _custom_operations(
    entry: FunctionSource,
    validations: Iterable[Validation],
) -> list[Operation]:
    unique_by_name = _accepted_summary_groups(validations)
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


def _opaque_operations(
    entry: FunctionSource,
    validations: Iterable[Validation],
) -> list[Operation]:
    accepted_names = set(_accepted_summary_groups(validations))
    opaque: list[Operation] = []
    for call in entry.calls():
        if call.name in STANDARD_LEAF_CALLS:
            continue
        if not call.indirect and call.name in accepted_names:
            continue
        opaque.append(
            Operation(
                "OPAQUE",
                call.name,
                " | ".join(call.arguments),
                "",
                call.line,
                True,
            )
        )
    return opaque


def analyze(
    entry: FunctionSource,
    validations: Iterable[Validation] = (),
) -> Verdict:
    validations = tuple(validations)
    operations = _standard_operations(entry)
    operations.extend(_direct_ast_operations(entry))
    operations.extend(_custom_operations(entry, validations))
    operations.extend(_opaque_operations(entry, validations))

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
