from __future__ import annotations

import ast
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

_C_INTEGER = re.compile(
    r"(?<![A-Za-z0-9_])((?:0[xX][0-9A-Fa-f]+)|(?:0[bB][01]+)|(?:\d+))(?:[uU](?:ll|LL|l|L)?|(?:ll|LL|l|L)[uU]?)(?![A-Za-z0-9_])"
)
_COMPLEX_LVALUE = re.compile(
    r"(?<![A-Za-z0-9_])(?:\*\s*)?[A-Za-z_]\w*"
    r"(?:(?:->|\.)[A-Za-z_]\w*|\[[^\[\]]+\])+"
)
_DEREF_ATOM = re.compile(r"(?<![A-Za-z0-9_])\*\s*[A-Za-z_]\w*")
_CHAR_LITERAL = re.compile(r"'(?:\\.|[^'\\])'")


def _ids(expression: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_]\w*\b", normalize_expression(expression)))


def _char_value(match: re.Match[str]) -> str:
    try:
        value = ast.literal_eval(match.group(0))
    except Exception:
        return match.group(0)
    return str(ord(value)) if isinstance(value, str) and len(value) == 1 else match.group(0)


def _normalize_c_scalar_syntax(expression: str) -> str:
    text = normalize_expression(expression)
    text = _C_INTEGER.sub(lambda match: match.group(1), text)
    text = _CHAR_LITERAL.sub(_char_value, text)
    text = re.sub(r"\b(?:NULL|nullptr)\b", "0", text)
    text = re.sub(r"\btrue\b", "1", text, flags=re.I)
    text = re.sub(r"\bfalse\b", "0", text, flags=re.I)
    return text


class CExpressionEncoder(core.ExpressionEncoder):
    """Encode common C scalar syntax without inventing program facts.

    Member/subscript/dereference expressions are treated as stable symbolic
    lvalues. This preserves equality across repeated occurrences such as
    ``offset < iov[i].iov_len`` while leaving their values unconstrained unless
    the program supplies a relation.
    """

    def _symbolize_lvalues(self, expression: str) -> str:
        text = expression
        for pattern in (_COMPLEX_LVALUE, _DEREF_ATOM):
            while True:
                match = pattern.search(text)
                if match is None:
                    break
                raw = normalize_expression(match.group(0))
                text = (
                    text[: match.start()]
                    + str(self.symbol(raw))
                    + text[match.end() :]
                )
        return text

    def encode(self, expression: str):
        text = _normalize_c_scalar_syntax(expression)
        text = self._symbolize_lvalues(text)
        return super().encode(text)

    def comparison(self, expression: str):
        text = _normalize_c_scalar_syntax(expression)
        if not re.search(r"<=|>=|==|!=|<|>", text):
            if re.search(r"(?<![=!<>])=(?!=)", text):
                raise ValueError(f"assignment condition is not a pure constraint: {expression}")
            return self.encode(text) != 0
        return super().comparison(text)

    def equality(self, left: str, right: str):
        return self.encode(left) == self.encode(right)


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
    text = _normalize_c_scalar_syntax(expression)
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
        arguments = getattr(opaque, "buffer", "")
        if _ids(arguments) & relevant:
            return (
                f"unresolved call {getattr(opaque, 'callee', '<unknown>')}@"
                f"{getattr(opaque, 'line', 0)} shares access-dependent values"
            )
    return None


def _check_access(entry, operation, capacities, signed, unsigned, operations):
    line = int(getattr(operation, "line", 0))
    original_arithmetic = core._has_unmodeled_c_arithmetic
    original_encoder = core.ExpressionEncoder
    core.ExpressionEncoder = CExpressionEncoder
    core._has_unmodeled_c_arithmetic = lambda expression: not _arithmetic_provably_bounded(
        entry, line, expression
    )
    try:
        access = core._check_access(
            entry,
            operation,
            capacities,
            signed,
            unsigned,
            operations,
        )
    finally:
        core._has_unmodeled_c_arithmetic = original_arithmetic
        core.ExpressionEncoder = original_encoder

    dependency_error = _opaque_dependency_error(operation, capacities, operations)
    if dependency_error is not None:
        return core.AccessCheck(
            access.access_kind,
            access.buffer,
            access.extent,
            access.line,
            "UNKNOWN",
            dependency_error,
            access.conditions,
            access.path_constraints,
            {},
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
            (
                f"{len(violations)} memory access(es) have feasible counterexamples; "
                f"first at line {first.line}: {first.reason}"
            ),
            accesses,
        )

    unknowns = [item for item in accesses if item.status == "UNKNOWN"]
    if unknowns:
        return core.ConstraintResult(
            "UNKNOWN",
            (
                f"{len(unknowns)} memory access(es) remain unresolved; "
                f"first at line {unknowns[0].line}: {unknowns[0].reason}"
            ),
            accesses,
        )

    if accesses:
        parser_note = (
            "; unrelated parser error nodes were ignored outside modeled access facts"
            if entry.parse_has_error
            else ""
        )
        return core.ConstraintResult(
            "UNKNOWN",
            (
                "all currently modeled memory accesses satisfy their generated bounds "
                "conditions, but complete function-level memory-access coverage is not established"
                + parser_note
            ),
            accesses,
        )

    return core.ConstraintResult(
        "UNKNOWN",
        "no supported memory access was available for bounds analysis",
        tuple(),
    )
