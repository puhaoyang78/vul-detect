from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .semantics import ALLOCATORS, READS, UNBOUNDED_WRITES, WRITES, Validation
from .source import FunctionSource, normalize_expression
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
                continue
            operations.append(
                Operation(
                    "ALLOC",
                    call.name,
                    call.result or "return",
                    size,
                    call.line,
                    False,
                )
            )
        elif call.name in WRITES:
            buffer_index, length_index = WRITES[call.name]
            if len(call.arguments) > max(buffer_index, length_index):
                length = call.arguments[length_index]
                if call.name == "fread" and len(call.arguments) >= 3:
                    length = f"({call.arguments[1]}) * ({call.arguments[2]})"
                operations.append(
                    Operation(
                        "WRITE",
                        call.name,
                        call.arguments[buffer_index],
                        length,
                        call.line,
                        False,
                    )
                )
        elif call.name in READS:
            buffer_index, length_index = READS[call.name]
            if len(call.arguments) > max(buffer_index, length_index):
                length = call.arguments[length_index]
                if call.name == "fwrite" and len(call.arguments) >= 3:
                    length = f"({call.arguments[1]}) * ({call.arguments[2]})"
                operations.append(
                    Operation(
                        "READ",
                        call.name,
                        call.arguments[buffer_index],
                        length,
                        call.line,
                        False,
                    )
                )
        elif call.name in UNBOUNDED_WRITES and call.arguments:
            operations.append(
                Operation("WRITE", call.name, call.arguments[0], "UNBOUNDED", call.line, False)
            )
    return operations


def _custom_operations(
    entry: FunctionSource, validations: Iterable[Validation]
) -> list[Operation]:
    by_name: dict[str, list[dict[str, str]]] = {}
    for validation in validations:
        if validation.passed:
            by_name.setdefault(validation.function, []).append(validation.summary)

    operations: list[Operation] = []
    for call in entry.calls():
        for summary in by_name.get(call.name, []):
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
            elif kind == "GUARD":
                operations.append(
                    Operation(
                        "GUARD",
                        call.name,
                        "",
                        _substitute(summary["relation"], call.arguments),
                        call.line,
                        True,
                    )
                )
            elif kind == "VALUE":
                operations.append(
                    Operation(
                        "VALUE",
                        call.name,
                        call.result or "return",
                        _substitute(summary["expression"], call.arguments),
                        call.line,
                        True,
                    )
                )
    return operations


def _assignments(text: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for name, value in re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);", text, flags=re.S
    ):
        if "==" in value or "!=" in value:
            continue
        assignments[name] = normalize_expression(value)
    return assignments


def _signed_names(entry: FunctionSource) -> set[str]:
    names: set[str] = set()
    for parameter, declaration in zip(entry.parameters, entry.parameter_types):
        if re.search(r"\b(int|short|ssize_t|long)\b", declaration) and not re.search(
            r"\b(unsigned|size_t|u\d+)\b", declaration
        ):
            names.add(parameter)
    for declaration in re.finditer(
        r"\b(?:signed\s+)?(?:int|short|ssize_t|long)\s+([^;]+);", entry.text
    ):
        for name in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", declaration.group(1)):
            if name not in {"const", "signed", "unsigned"}:
                names.add(name)
    return names


def _guards(text: str) -> list[str]:
    return [
        normalize_expression(item)
        for item in re.findall(r"\bif\s*\((.*?)\)", text, re.S)
    ]


def _local_arrays(text: str) -> dict[str, str]:
    return {
        name: normalize_expression(size)
        for name, size in re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_\s*]*\b([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([^\]]+)\s*\]",
            text,
        )
    }


def _has_nonnegative_guard(name: str, guards: Iterable[str]) -> bool:
    patterns = {
        normalize_expression(f"{name}>=0"),
        normalize_expression(f"{name}>-1"),
        normalize_expression(f"{name}<0"),
    }
    return any(any(pattern in guard for pattern in patterns) for guard in guards)


def _risky_unbounded_write(
    entry: FunctionSource, operations: list[Operation]
) -> str | None:
    arrays = _local_arrays(entry.text)
    for operation in operations:
        if operation.kind != "WRITE" or operation.extent != "UNBOUNDED":
            continue
        buffer = normalize_expression(operation.buffer)
        if buffer in arrays:
            return (
                f"{operation.callee} writes without a bound to local array {buffer}"
                f"[{arrays[buffer]}]"
            )
    return None


def _risky_offset_write(
    entry: FunctionSource, operations: list[Operation]
) -> str | None:
    guards = _guards(entry.text) + [
        normalize_expression(operation.extent)
        for operation in operations
        if operation.kind == "GUARD"
    ]
    for operation in operations:
        if operation.kind != "WRITE" or operation.custom:
            continue
        buffer = normalize_expression(operation.buffer)
        length = normalize_expression(operation.extent)
        if "+" not in buffer:
            continue
        tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", buffer + length))
        if len(tokens) < 2:
            continue
        if not any(sum(token in guard for token in tokens) >= 2 for guard in guards):
            return (
                f"direct {operation.callee} appends {length} bytes at {buffer} without a "
                "guard that relates offset and length"
            )
    return None


def _risky_allocation_arithmetic(
    entry: FunctionSource, operations: list[Operation]
) -> str | None:
    guards = _guards(entry.text)
    for operation in operations:
        if operation.kind != "ALLOC" or operation.custom:
            continue
        size = normalize_expression(operation.extent)
        if "+" not in size or "*" not in size:
            continue
        operands = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", size))
        if len(operands) < 2:
            continue
        exact_guard = any(size in guard or "SIZE_MAX-" in guard for guard in guards)
        if not exact_guard:
            return f"allocation size {size} combines unchecked addition and multiplication"
    return None


def _risky_custom_access(
    entry: FunctionSource, operations: list[Operation]
) -> str | None:
    assignments = _assignments(entry.text)
    signed = _signed_names(entry)
    guards = _guards(entry.text) + [
        normalize_expression(operation.extent)
        for operation in operations
        if operation.kind == "GUARD"
    ]

    for operation in operations:
        if operation.kind not in {"READ", "WRITE"} or not operation.custom:
            continue
        length = normalize_expression(operation.extent)
        buffer = normalize_expression(operation.buffer)
        length_tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", length))

        if buffer in {"NULL", "0", "nullptr"}:
            continue

        if operation.kind == "WRITE" and "-" in length:
            controlling = sorted(length_tokens)
            if not any(
                any(token in guard for token in controlling)
                and ("<=" in guard or re.search(r"(?<!-)>", guard))
                for guard in guards
            ):
                return (
                    f"validated {operation.kind} {operation.callee} receives {length} "
                    "with no visible upper-bound guard"
                )

        if buffer in entry.parameters:
            index = entry.parameters.index(buffer)
            capacity_candidates = entry.parameters[index + 1 : index + 3]
            capacity = next(
                (
                    item
                    for item in capacity_candidates
                    if re.search(r"(?:len|size|capacity)$", item, re.I)
                ),
                None,
            )
            if capacity:
                protecting = [guard for guard in guards if capacity in guard]
                if protecting and not any(
                    normalize_expression(length) in guard for guard in protecting
                ):
                    return (
                        f"buffer {buffer} is bounded by {capacity}, but the validated "
                        f"{operation.kind} uses {length}; the visible guard constrains a "
                        "different expression"
                    )

        if operation.kind == "WRITE":
            for token in length_tokens:
                origin = assignments.get(token, "")
                origin_tokens = set(
                    re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", origin)
                )
                signed_sources = ({token} | origin_tokens) & signed
                if signed_sources and ("+" in origin or token in signed):
                    if not any(
                        _has_nonnegative_guard(item, guards) for item in signed_sources
                    ):
                        return (
                            f"validated WRITE {operation.callee} uses signed "
                            f"length {length}; the vulnerable function has no nonnegative "
                            "constraint before the write"
                        )
    return None


def analyze(
    entry: FunctionSource,
    validations: Iterable[Validation] = (),
    proposed: bool = False,
) -> Verdict:
    operations = _standard_operations(entry)
    if proposed:
        operations.extend(_custom_operations(entry, validations))

    # Keep the baseline-compatible checks only for direct, syntactically obvious cases.
    checks = [
        _risky_unbounded_write(entry, operations),
        _risky_offset_write(entry, operations),
        _risky_allocation_arithmetic(entry, operations),
    ]
    reason = next((item for item in checks if item), None)
    if reason:
        return Verdict("VULNERABLE", reason, tuple(operations))

    if proposed:
        constraint_result = reason_memory_safety(entry, operations)
        constraint_json = constraint_result.as_json()
        if constraint_result.status == "POTENTIAL_VIOLATION":
            return Verdict(
                "VULNERABLE",
                f"Z3: {constraint_result.reason}",
                tuple(operations),
                constraint_json,
            )
        if constraint_result.status == "SAFE":
            return Verdict(
                "NOT_DETECTED",
                "Z3 proved the generated bounds conditions under the available constraints",
                tuple(operations),
                constraint_json,
            )
        return Verdict(
            "NOT_DETECTED",
            f"Z3 could not decide: {constraint_result.reason}",
            tuple(operations),
            constraint_json,
        )

    return Verdict(
        "NOT_DETECTED",
        "no direct supported sink established a capacity violation in the entry function",
        tuple(operations),
    )
