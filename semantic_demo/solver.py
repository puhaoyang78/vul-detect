from __future__ import annotations

import re

from z3 import Solver, unsat

from . import z3_reasoner as core
from .source import normalize_expression

_FIXED_WIDTH_TYPES = {
    "uint32_t": (0, 2**32 - 1),
    "int32_t": (-(2**31), 2**31 - 1),
    "uint64_t": (0, 2**64 - 1),
    "int64_t": (-(2**63), 2**63 - 1),
}


def _ids(expression: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_]\w*\b", normalize_expression(expression)))


def _parameter_ranges(entry, identifiers: set[str]):
    mapping = dict(zip(entry.parameters, entry.parameter_types))
    ranges = {}
    type_names = set()
    for name in identifiers:
        type_text = " ".join(mapping.get(name, "").replace("const", "").split())
        matched = next((token for token in _FIXED_WIDTH_TYPES if token in type_text), None)
        if matched is None:
            return None
        ranges[name] = _FIXED_WIDTH_TYPES[matched]
        type_names.add(matched)
    if len(type_names) != 1:
        return None
    result_range = _FIXED_WIDTH_TYPES[next(iter(type_names))]
    return ranges, result_range


def _arithmetic_provably_bounded(entry, line: int, expression: str) -> bool:
    text = normalize_expression(expression)
    if not re.search(r"[+*\-]", text):
        return True
    identifiers = _ids(text)
    if not identifiers:
        return True
    range_info = _parameter_ranges(entry, identifiers)
    if range_info is None:
        return False
    ranges, (minimum, maximum) = range_info
    encoder = core.ExpressionEncoder()
    solver = Solver()
    for name, (lower, upper) in ranges.items():
        symbol = encoder.encode(name)
        solver.add(symbol >= lower, symbol <= upper)
    added_path = False
    for condition in entry.continuation_constraints_before(line):
        if core._has_unresolved_compile_time_symbol(condition):
            continue
        try:
            solver.add(encoder.comparison(condition))
            added_path = True
        except Exception:
            continue
    if not added_path:
        return False
    try:
        value = encoder.encode(text)
    except Exception:
        return False
    overflow = Solver()
    overflow.add(*solver.assertions())
    overflow.add((value < minimum) | (value > maximum))
    return overflow.check() == unsat


def _access_relevant_identifiers(operation, capacities) -> set[str]:
    buffer = normalize_expression(getattr(operation, "buffer", ""))
    extent = normalize_expression(getattr(operation, "extent", ""))
    relevant = _ids(buffer) | _ids(extent)
    capacity = core._capacity_for_buffer(operation, buffer, capacities)
    if capacity is not None:
        capacity_text, offset_text = capacity
        relevant |= _ids(capacity_text) | _ids(offset_text)
    return relevant


def _opaque_dependency_error(operation, capacities, operations) -> str | None:
    relevant = _access_relevant_identifiers(operation, capacities)
    if not relevant:
        return None
    line = int(getattr(operation, "line", 0))
    for opaque in operations:
        if getattr(opaque, "kind", "") != "OPAQUE":
            continue
        if int(getattr(opaque, "line", 0)) >= line:
            continue
        if _ids(getattr(opaque, "buffer", "")) & relevant:
            return (
                f"unresolved call {getattr(opaque, 'callee', '<unknown>')}@"
                f"{getattr(opaque, 'line', 0)} shares access-dependent values"
            )
    return None


def _check_access(entry, operation, capacities, signed, unsigned, operations):
    line = int(getattr(operation, "line", 0))
    original = core._has_unmodeled_c_arithmetic
    core._has_unmodeled_c_arithmetic = lambda expression: not _arithmetic_provably_bounded(
        entry, line, expression
    )
    try:
        access = core._check_access(entry, operation, capacities, signed, unsigned, operations)
    finally:
        core._has_unmodeled_c_arithmetic = original
    dependency_error = _opaque_dependency_error(operation, capacities, operations)
    if dependency_error is not None:
        return core.AccessCheck(
            access.access_kind, access.buffer, access.extent, access.line,
            "UNKNOWN", dependency_error, access.conditions, access.path_constraints, {},
        )
    return access


def reason_memory_safety(entry, operations):
    operations = list(operations)
    capacities = core._collect_capacity_relations(entry, operations)
    signed, unsigned = entry.integer_domains()
    accesses = tuple(
        _check_access(entry, operation, capacities, signed, unsigned, operations)
        for operation in operations
        if getattr(operation, "kind", "") in {"READ", "WRITE"}
        and getattr(operation, "buffer", "") not in {"", "NULL", "0", "nullptr"}
    )
    violations = [item for item in accesses if item.status == "POTENTIAL_VIOLATION"]
    if violations:
        first = violations[0]
        return core.ConstraintResult(
            "POTENTIAL_VIOLATION",
            f"{len(violations)} memory access(es) have feasible counterexamples; first at line {first.line}: {first.reason}",
            accesses,
        )
    unknowns = [item for item in accesses if item.status == "UNKNOWN"]
    if unknowns:
        return core.ConstraintResult(
            "UNKNOWN",
            f"{len(unknowns)} memory access(es) remain unresolved; first at line {unknowns[0].line}: {unknowns[0].reason}",
            accesses,
        )
    if accesses:
        note = "; unrelated parser error nodes were ignored outside modeled access facts" if entry.parse_has_error else ""
        return core.ConstraintResult(
            "UNKNOWN",
            "all currently modeled memory accesses satisfy their generated bounds conditions, but complete function-level memory-access coverage is not established" + note,
            accesses,
        )
    return core.ConstraintResult("UNKNOWN", "no supported memory access was available for bounds analysis", tuple())
