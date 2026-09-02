from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass
from typing import Iterable

from z3 import If, Int, IntVal, Not, Solver, sat, unsat

from .source import FunctionSource, normalize_expression


@dataclass(frozen=True)
class VerificationCondition:
    access_kind: str
    buffer: str
    extent: str
    condition: str
    line: int


@dataclass(frozen=True)
class AccessCheck:
    access_kind: str
    buffer: str
    extent: str
    line: int
    status: str
    reason: str
    conditions: tuple[VerificationCondition, ...]
    path_constraints: tuple[str, ...]
    model: dict[str, int]

    def as_json(self) -> dict[str, object]:
        return {
            "access_kind": self.access_kind,
            "buffer": self.buffer,
            "extent": self.extent,
            "line": self.line,
            "status": self.status,
            "reason": self.reason,
            "conditions": [asdict(item) for item in self.conditions],
            "path_constraints": list(self.path_constraints),
            "model": self.model,
        }


@dataclass(frozen=True)
class ConstraintResult:
    status: str
    reason: str
    accesses: tuple[AccessCheck, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "accesses": [item.as_json() for item in self.accesses],
        }


@dataclass(frozen=True)
class CapacityInfo:
    byte_capacity: str | None = None
    element_capacity: str | None = None


class ExpressionEncoder:
    """Encode integer and bounds relations while preserving unknown terms symbolically."""

    _atom_pattern = re.compile(
        r"""
        sizeof\s*\([^()]*\)
        |[A-Za-z_][A-Za-z0-9_]*(?:->|\.)[A-Za-z_][A-Za-z0-9_.>-]*
        """,
        re.X,
    )

    def __init__(self) -> None:
        self._symbols: dict[str, object] = {}
        self._reverse: dict[str, str] = {}
        self._counter = 0

    def symbol(self, raw: str):
        key = normalize_expression(raw)
        if key not in self._symbols:
            safe = f"v_{self._counter}"
            self._counter += 1
            self._symbols[key] = Int(safe)
            self._reverse[safe] = key
        return self._symbols[key]

    def _replace_atoms(self, expression: str) -> str:
        result = expression
        while True:
            match = self._atom_pattern.search(result)
            if not match:
                break
            raw = match.group(0)
            result = (
                result[: match.start()]
                + str(self.symbol(raw))
                + result[match.end() :]
            )
        return result

    def encode(self, expression: str):
        text = normalize_expression(expression)
        if not text:
            raise ValueError("empty expression")
        text = re.sub(
            r"\(\s*(?:unsigned|signed|long|short|int|char|size_t|ssize_t)[^)]*\)",
            "",
            text,
        )
        text = self._replace_atoms(text)
        identifiers = sorted(
            set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)),
            key=len,
            reverse=True,
        )
        for name in identifiers:
            if name.startswith("v_") or name in {"min", "max"}:
                continue
            text = re.sub(
                rf"\b{re.escape(name)}\b", str(self.symbol(name)), text
            )
        tree = ast.parse(text, mode="eval")
        return self._node(tree.body)

    def comparison(self, expression: str):
        match = re.match(
            r"^(.*?)(<=|>=|==|!=|<|>)(.*)$",
            normalize_expression(expression),
        )
        if not match:
            raise ValueError(f"unsupported comparison: {expression}")
        left, operator, right = match.groups()
        lhs = self.encode(left)
        rhs = self.encode(right)
        return {
            "<=": lhs <= rhs,
            ">=": lhs >= rhs,
            "<": lhs < rhs,
            ">": lhs > rhs,
            "==": lhs == rhs,
            "!=": lhs != rhs,
        }[operator]

    def equality(self, left: str, right: str):
        return self.encode(left) == self.encode(right)

    def _node(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return IntVal(node.value)
        if isinstance(node, ast.Name):
            return Int(node.id)
        if isinstance(node, ast.UnaryOp):
            value = self._node(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in {"min", "max"}
                and len(node.args) == 2
            ):
                left = self._node(node.args[0])
                right = self._node(node.args[1])
                if node.func.id == "min":
                    return If(left <= right, left, right)
                return If(left >= right, left, right)
            raise ValueError(f"unsupported function call: {ast.dump(node)}")
        if isinstance(node, ast.BinOp):
            left = self._node(node.left)
            right = self._node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, (ast.Div, ast.FloorDiv)):
                return left / right
            if isinstance(node.op, ast.Mod):
                return left % right
        raise ValueError(f"unsupported expression node: {ast.dump(node)}")

    def model_dict(self, model) -> dict[str, int]:
        result: dict[str, int] = {}
        for declaration in model.decls():
            value = model[declaration]
            if value is None or not hasattr(value, "as_long"):
                continue
            raw = self._reverse.get(declaration.name(), declaration.name())
            result[raw] = value.as_long()
        return result



def _uppercase_symbols(expression: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\b[A-Z][A-Z0-9_]*\b", expression)
        if token not in {"NULL"}
    }


def _has_unresolved_compile_time_symbol(*expressions: str) -> bool:
    return any(_uppercase_symbols(expression) for expression in expressions)


def _has_unmodeled_c_arithmetic(expression: str) -> bool:
    """Detect arithmetic whose C wraparound semantics are not represented by Int."""
    text = normalize_expression(expression)
    if not re.search(r"[+*\-]", text):
        return False
    return bool(re.search(r"\b[A-Za-z_][A-Za-z0-9_]*(?:->\w+|\.\w+)?\b", text))



def _local_array_capacities(entry: FunctionSource) -> dict[str, CapacityInfo]:
    capacities: dict[str, CapacityInfo] = {}
    for array in entry.local_arrays():
        capacities[array.name] = CapacityInfo(
            byte_capacity=array.byte_capacity,
            element_capacity=array.elements,
        )
    return capacities


def _strip_outer_casts(expression: str) -> str:
    text = normalize_expression(expression)
    while True:
        updated = re.sub(r"^\([^()]*\*[^()]*\)", "", text)
        if updated == text:
            return text
        text = updated


def _buffer_base_and_offset(buffer: str) -> tuple[str, str]:
    text = _strip_outer_casts(buffer)
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "+" and depth == 0:
            return text[:index], text[index + 1 :]
    return text, "0"



def _collect_capacity_relations(
    entry: FunctionSource, operations: Iterable[object]
) -> dict[str, CapacityInfo]:
    """Collect byte capacities and, when justified, element counts."""
    capacities = _local_array_capacities(entry)
    for operation in operations:
        if getattr(operation, "kind", "") != "ALLOC":
            continue
        buffer = _strip_outer_casts(getattr(operation, "buffer", ""))
        extent = normalize_expression(getattr(operation, "extent", ""))
        if not buffer or buffer == "return" or not extent:
            continue
        capacities[buffer] = CapacityInfo(
            byte_capacity=extent,
            element_capacity=None,
        )
    return capacities


def _capacity_for_buffer(
    operation: object,
    buffer: str,
    capacities: dict[str, CapacityInfo],
) -> tuple[str, str] | None:
    base, offset = _buffer_base_and_offset(buffer)
    info = capacities.get(base)
    if info is None:
        return None
    if getattr(operation, "callee", "") in {"AST_SUBSCRIPT", "AST_DEREF"}:
        capacity = info.element_capacity
    else:
        capacity = info.byte_capacity
    if capacity is None:
        return None
    return capacity, offset


def _expression_identifiers(expression: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression))


def _is_represented_allocation_definition(
    entry: FunctionSource,
    target: str,
    operations: Iterable[object],
    access_line: int,
) -> bool:
    """Whether the latest direct call definition is already modeled as ALLOC."""
    direct_definitions = {
        name: (callee, line)
        for name, callee, line in entry.direct_call_definitions_before(access_line)
    }
    definition = direct_definitions.get(target)
    if definition is None:
        return False
    callee, definition_line = definition
    return any(
        getattr(operation, "kind", "") == "ALLOC"
        and int(getattr(operation, "line", 0)) == definition_line
        and _strip_outer_casts(getattr(operation, "buffer", "")) == target
        and normalize_expression(getattr(operation, "callee", ""))
        == normalize_expression(callee)
        for operation in operations
    )


def _add_program_constraints(
    solver: Solver,
    encoder: ExpressionEncoder,
    entry: FunctionSource,
    line: int,
    operations: Iterable[object],
    unsigned: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...], set[str]]:
    path_constraints = tuple(entry.continuation_constraints_before(line))
    incomplete: list[str] = []
    skipped_targets: set[str] = set()

    for relation in path_constraints:
        if _has_unresolved_compile_time_symbol(relation):
            incomplete.append(f"unresolved path symbol: {relation}")
            continue
        try:
            solver.add(encoder.comparison(relation))
        except Exception:
            incomplete.append(f"unsupported path constraint: {relation}")

    for name in unsigned:
        try:
            solver.add(encoder.encode(name) >= 0)
        except Exception:
            pass

    for left, right in entry.value_relations_before(line):
        target = normalize_expression(left)
        if _is_represented_allocation_definition(
            entry, target, operations, line
        ):
            continue
        if _has_unresolved_compile_time_symbol(left, right) or _has_unmodeled_c_arithmetic(right):
            skipped_targets.add(target)
            continue
        try:
            solver.add(encoder.equality(left, right))
        except Exception:
            skipped_targets.add(target)

    for operation in operations:
        if getattr(operation, "kind", "") != "VALUE":
            continue
        if int(getattr(operation, "line", 0)) >= line:
            continue
        target = normalize_expression(getattr(operation, "buffer", ""))
        expression = normalize_expression(getattr(operation, "extent", ""))
        if not target or target == "return" or not expression:
            continue
        if _has_unresolved_compile_time_symbol(target, expression) or _has_unmodeled_c_arithmetic(expression):
            skipped_targets.add(target)
            continue
        try:
            solver.add(encoder.equality(target, expression))
            skipped_targets.discard(target)
        except Exception:
            skipped_targets.add(target)

    return path_constraints, tuple(incomplete), skipped_targets


def _check_access(
    entry: FunctionSource,
    operation: object,
    capacities: dict[str, CapacityInfo],
    signed: set[str],
    unsigned: set[str],
    operations: Iterable[object],
) -> AccessCheck:
    kind = getattr(operation, "kind", "")
    buffer_text = _strip_outer_casts(getattr(operation, "buffer", ""))
    extent_text = normalize_expression(getattr(operation, "extent", ""))
    line = int(getattr(operation, "line", 0))

    if not extent_text or extent_text == "UNBOUNDED":
        return AccessCheck(
            kind,
            buffer_text,
            extent_text,
            line,
            "UNKNOWN",
            "access extent is not represented as a bounded integer expression",
            tuple(),
            tuple(),
            {},
        )

    encoder = ExpressionEncoder()
    solver = Solver()
    path_constraints, incomplete_paths, skipped_targets = _add_program_constraints(
        solver, encoder, entry, line, operations, unsigned
    )
    conditions: list[VerificationCondition] = []

    base, _ = _buffer_base_and_offset(buffer_text)
    local_array = next(
        (array for array in entry.local_arrays() if array.name == base),
        None,
    )
    if local_array is not None and local_array.byte_capacity is not None:
        try:
            solver.add(
                encoder.equality(
                    f"sizeof({base})",
                    local_array.byte_capacity,
                )
            )
        except Exception:
            pass

    if incomplete_paths:
        return AccessCheck(
            kind,
            buffer_text,
            extent_text,
            line,
            "UNKNOWN",
            incomplete_paths[0],
            tuple(),
            path_constraints,
            {},
        )

    if _has_unresolved_compile_time_symbol(extent_text):
        return AccessCheck(
            kind,
            buffer_text,
            extent_text,
            line,
            "UNKNOWN",
            f"unresolved compile-time symbol in access extent {extent_text}",
            tuple(),
            path_constraints,
            {},
        )
    if _has_unmodeled_c_arithmetic(extent_text):
        return AccessCheck(
            kind,
            buffer_text,
            extent_text,
            line,
            "UNKNOWN",
            f"C integer overflow semantics are not modeled for extent {extent_text}",
            tuple(),
            path_constraints,
            {},
        )

    try:
        extent = encoder.encode(extent_text)
    except Exception:
        return AccessCheck(
            kind,
            buffer_text,
            extent_text,
            line,
            "UNKNOWN",
            f"cannot encode access extent {extent_text}",
            tuple(),
            path_constraints,
            {},
        )

    # Non-negativity is a standard precondition for signed access extents.
    extent_tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", extent_text))
    signed_extent_tokens = extent_tokens & signed
    if signed_extent_tokens:
        constrained_identifiers = set().union(
            *(_expression_identifiers(relation) for relation in path_constraints)
        )
        unconstrained_parameters = sorted(
            token
            for token in signed_extent_tokens
            if token in entry.parameters and token not in constrained_identifiers
        )
        if unconstrained_parameters:
            return AccessCheck(
                kind,
                buffer_text,
                extent_text,
                line,
                "UNKNOWN",
                "signed access extent depends on unconstrained parameter domain: "
                + ", ".join(unconstrained_parameters),
                tuple(conditions),
                path_constraints,
                {},
            )
        condition_text = f"{extent_text} >= 0"
        condition = extent >= 0
        conditions.append(
            VerificationCondition(kind, buffer_text, extent_text, condition_text, line)
        )
        check = Solver()
        check.add(*solver.assertions())
        check.add(Not(condition))
        if check.check() == sat:
            return AccessCheck(
                kind,
                buffer_text,
                extent_text,
                line,
                "POTENTIAL_VIOLATION",
                f"access extent may be negative: {condition_text}",
                tuple(conditions),
                path_constraints,
                encoder.model_dict(check.model()),
            )

    capacity = _capacity_for_buffer(operation, buffer_text, capacities)
    if capacity is None:
        return AccessCheck(
            kind,
            buffer_text,
            extent_text,
            line,
            "UNKNOWN",
            f"object capacity/valid extent is unknown for {buffer_text}",
            tuple(conditions),
            path_constraints,
            {},
        )

    capacity_text, offset_text = capacity
    relevant_identifiers = (
        _expression_identifiers(extent_text)
        | _expression_identifiers(buffer_text)
        | _expression_identifiers(capacity_text)
        | _expression_identifiers(offset_text)
    )
    for uncertain_condition in entry.uncertain_control_conditions_before(line):
        if _expression_identifiers(uncertain_condition) & relevant_identifiers:
            return AccessCheck(
                kind,
                buffer_text,
                extent_text,
                line,
                "UNKNOWN",
                "control-flow effect of a call-containing guard is unresolved: "
                + uncertain_condition,
                tuple(conditions),
                path_constraints,
                {},
            )
    unresolved_definitions = sorted(skipped_targets & relevant_identifiers)
    if unresolved_definitions:
        return AccessCheck(
            kind,
            buffer_text,
            extent_text,
            line,
            "UNKNOWN",
            "reaching value definition is not safely encodable for: "
            + ", ".join(unresolved_definitions),
            tuple(conditions),
            path_constraints,
            {},
        )
    if _has_unresolved_compile_time_symbol(capacity_text, offset_text):
        return AccessCheck(
            kind,
            buffer_text,
            extent_text,
            line,
            "UNKNOWN",
            f"unresolved compile-time symbol in capacity relation for {buffer_text}",
            tuple(conditions),
            path_constraints,
            {},
        )
    if _has_unmodeled_c_arithmetic(capacity_text) or _has_unmodeled_c_arithmetic(offset_text):
        return AccessCheck(
            kind,
            buffer_text,
            extent_text,
            line,
            "UNKNOWN",
            f"C integer overflow semantics are not modeled for capacity relation {buffer_text}",
            tuple(conditions),
            path_constraints,
            {},
        )
    try:
        capacity_expr = encoder.encode(capacity_text)
        offset_expr = encoder.encode(offset_text)
    except Exception:
        return AccessCheck(
            kind,
            buffer_text,
            extent_text,
            line,
            "UNKNOWN",
            f"cannot encode capacity relation for {buffer_text}",
            tuple(conditions),
            path_constraints,
            {},
        )

    offset_tokens = set(
        re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", normalize_expression(offset_text))
    )
    if normalize_expression(offset_text) != "0" and offset_tokens & signed:
        offset_condition_text = f"{offset_text} >= 0"
        offset_condition = offset_expr >= 0
        conditions.append(
            VerificationCondition(
                kind,
                buffer_text,
                extent_text,
                offset_condition_text,
                line,
            )
        )
        check = Solver()
        check.add(*solver.assertions())
        check.add(Not(offset_condition))
        if check.check() == sat:
            return AccessCheck(
                kind,
                buffer_text,
                extent_text,
                line,
                "POTENTIAL_VIOLATION",
                f"access offset may be negative: {offset_condition_text}",
                tuple(conditions),
                path_constraints,
                encoder.model_dict(check.model()),
            )

    condition_text = (
        f"{offset_text} + {extent_text} <= {capacity_text}"
        if normalize_expression(offset_text) != "0"
        else f"{extent_text} <= {capacity_text}"
    )
    condition = offset_expr + extent <= capacity_expr
    conditions.append(
        VerificationCondition(kind, buffer_text, extent_text, condition_text, line)
    )

    check = Solver()
    check.add(*solver.assertions())
    check.add(Not(condition))
    result = check.check()
    if result == sat:
        return AccessCheck(
            kind,
            buffer_text,
            extent_text,
            line,
            "POTENTIAL_VIOLATION",
            f"bounds condition may fail: {condition_text}",
            tuple(conditions),
            path_constraints,
            encoder.model_dict(check.model()),
        )
    if result == unsat:
        return AccessCheck(
            kind,
            buffer_text,
            extent_text,
            line,
            "SAFE",
            "bounds condition is implied by the available program constraints",
            tuple(conditions),
            path_constraints,
            {},
        )
    return AccessCheck(
        kind,
        buffer_text,
        extent_text,
        line,
        "UNKNOWN",
        "solver returned unknown",
        tuple(conditions),
        path_constraints,
        {},
    )


def reason_memory_safety(
    entry: FunctionSource,
    operations: Iterable[object],
) -> ConstraintResult:
    operations = list(operations)
    capacities = _collect_capacity_relations(entry, operations)
    signed, unsigned = entry.integer_domains()

    if entry.parse_has_error:
        return ConstraintResult(
            "UNKNOWN",
            "target function contains parser error nodes; semantic verification is incomplete",
            tuple(),
        )
    if entry.has_indirect_calls():
        return ConstraintResult(
            "UNKNOWN",
            "target function contains unresolved indirect calls",
            tuple(),
        )

    accesses = tuple(
        _check_access(entry, operation, capacities, signed, unsigned, operations)
        for operation in operations
        if getattr(operation, "kind", "") in {"READ", "WRITE"}
        and getattr(operation, "buffer", "") not in {"", "NULL", "0", "nullptr"}
    )

    violations = [
        item for item in accesses if item.status == "POTENTIAL_VIOLATION"
    ]
    if violations:
        first = violations[0]
        return ConstraintResult(
            "POTENTIAL_VIOLATION",
            (
                f"{len(violations)} memory access(es) have feasible counterexamples; "
                f"first at line {first.line}: {first.reason}"
            ),
            accesses,
        )

    unknowns = [item for item in accesses if item.status == "UNKNOWN"]
    if unknowns:
        return ConstraintResult(
            "UNKNOWN",
            (
                f"{len(unknowns)} memory access(es) remain unresolved; "
                f"first at line {unknowns[0].line}: {unknowns[0].reason}"
            ),
            accesses,
        )

    if accesses:
        return ConstraintResult(
            "UNKNOWN",
            (
                "all currently modeled memory accesses satisfy their generated bounds "
                "conditions, but complete function-level memory-access coverage is not established"
            ),
            accesses,
        )

    return ConstraintResult(
        "UNKNOWN",
        "no supported memory access was available for bounds analysis",
        tuple(),
    )
