from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .semantics import Validation
from .source import FunctionSource, normalize_expression
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
    return normalize_expression(result)


def _substitute_buffer(expression: str, arguments: tuple[str, ...]) -> str:
    result = expression
    for index in reversed(range(len(arguments))):
        argument = normalize_expression(arguments[index])
        if re.fullmatch(
            r"[A-Za-z_]\w*(?:(?:->|\.)[A-Za-z_]\w*|\[[^\]]+\])*",
            argument,
        ):
            replacement = argument
        else:
            replacement = f"({argument})"
        result = re.sub(rf"\barg{index}\b", replacement, result)
    result = normalize_expression(result)
    result = re.sub(
        r"\(([A-Za-z_]\w*(?:(?:->|\.)[A-Za-z_]\w*)*)\)(?=->|\.)",
        r"\1",
        result,
    )
    return result


def _accepted_by_callsite(
    validations: Iterable[Validation],
) -> tuple[
    dict[tuple[str, int], list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
]:
    by_callsite: dict[tuple[str, int], list[dict[str, str]]] = {}
    legacy_by_name: dict[str, list[dict[str, str]]] = {}
    for validation in validations:
        if not validation.passed:
            continue
        if validation.call_lines:
            for line in validation.call_lines:
                bucket = by_callsite.setdefault((validation.function, line), [])
                if validation.summary not in bucket:
                    bucket.append(validation.summary)
        else:
            bucket = legacy_by_name.setdefault(validation.function, [])
            if validation.summary not in bucket:
                bucket.append(validation.summary)
    return by_callsite, legacy_by_name


def _custom_operations(
    entry: FunctionSource,
    validations: Iterable[Validation],
) -> list[Operation]:
    accepted_by_callsite, legacy_by_name = _accepted_by_callsite(validations)

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
        summaries = accepted_by_callsite.get((call.name, call.line), [])
        if not summaries:
            summaries = legacy_by_name.get(call.name, [])
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
                if summary["buffer"] == "return":
                    if not call.result:
                        continue
                    target = normalize_expression(call.result)
                else:
                    target = _substitute_buffer(summary["buffer"], call.arguments)
                operations.append(
                    Operation(
                        "ALLOC",
                        call.name,
                        target,
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
                        _substitute_buffer(summary["buffer"], call.arguments),
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
                        normalize_expression(call.result),
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
                if effect.buffer == "return":
                    continue
                operations.append(
                    Operation(
                        "ALLOC",
                        call.name,
                        normalize_expression(effect.buffer),
                        normalize_expression(effect.extent),
                        call.line,
                        False,
                    )
                )
                continue
            operations.append(
                Operation(
                    effect.kind,
                    call.name,
                    normalize_expression(effect.buffer),
                    normalize_expression(effect.extent),
                    call.line,
                    False,
                )
            )
    for access in entry.direct_memory_accesses():
        operations.append(
            Operation(
                access.kind,
                access.origin,
                normalize_expression(access.buffer),
                normalize_expression(access.extent),
                access.line,
                False,
            )
        )
    return operations


def analyze(
    entry: FunctionSource,
    validations: Iterable[Validation] = (),
) -> Verdict:
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
