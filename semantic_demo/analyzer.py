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


def _custom_operations(
    entry: FunctionSource,
    validations: Iterable[Validation],
) -> list[Operation]:
    accepted: dict[str, list[dict[str, str]]] = {}
    for validation in validations:
        if not validation.passed:
            continue
        accepted.setdefault(validation.function, []).append(validation.summary)

    operations: list[Operation] = []
    for call in entry.calls():
        if call.indirect:
            operations.append(
                Operation(
                    "OPAQUE",
                    call.name,
                    ",".join(call.arguments),
                    "",
                    call.line,
                    True,
                )
            )
            continue
        if call.name in STANDARD_LEAF_CALLS:
            continue
        summaries = accepted.get(call.name, [])
        if not summaries:
            operations.append(
                Operation(
                    "OPAQUE",
                    call.name,
                    ",".join(call.arguments),
                    "",
                    call.line,
                    True,
                )
            )
            continue
        for summary in summaries:
            kind = summary.get("kind")
            if kind == "ALLOC":
                if call.result:
                    operations.append(
                        Operation(
                            "ALLOC",
                            call.name,
                            call.result,
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


def _direct_operations(entry: FunctionSource) -> list[Operation]:
    operations: list[Operation] = []
    for call in entry.calls():
        if call.indirect:
            continue
        for effect in effects_for_call(call):
            if effect.kind == "ALLOC":
                if call.result:
                    operations.append(
                        Operation(
                            "ALLOC",
                            call.name,
                            call.result,
                            effect.extent,
                            call.line,
                            False,
                        )
                    )
            else:
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
    for access in entry.direct_memory_accesses():
        operations.append(
            Operation(
                access.kind,
                access.origin,
                access.buffer,
                access.extent,
                access.line,
                False,
            )
        )
    return operations


def analyze(entry: FunctionSource, validations: Iterable[Validation]) -> Verdict:
    operations = tuple(
        sorted(
            [*_direct_operations(entry), *_custom_operations(entry, validations)],
            key=lambda item: (item.line, item.kind, item.callee),
        )
    )
    result = reason_memory_safety(entry, operations)
    if result.status == "POTENTIAL_VIOLATION":
        return Verdict(
            "VULNERABLE",
            result.reason,
            operations,
            result.as_json(),
        )
    return Verdict(
        "UNKNOWN",
        result.reason,
        operations,
        result.as_json(),
    )
