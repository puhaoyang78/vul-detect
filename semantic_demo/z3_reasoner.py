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


def _signed_names(entry: FunctionSource) -> set[str]:
    names: set[str] = set()
    for parameter, declaration in zip(entry.parameters, entry.parameter_types):
        if re.search(r"\b(?:signed|int|short|ssize_t|long)\b", declaration) and not re.search(
            r"\b(?:unsigned|size_t|u\d+)\b", declaration
        ):
            names.add(parameter)
    for match in re.finditer(
        r"\b(?:(unsigned|signed)\s+)?(?:int|short|ssize_t|long(?:\s+long)?)"
        r"\s+([A-Za-z_][A-Za-z0-9_]*)",
        entry.text,
    ):
        qualifier, name = match.groups()
        if qualifier != "unsigned" and name != entry.name:
            names.add(name)
    return names


def _local_array_capacities(entry: FunctionSource) -> dict[str, CapacityInfo]:
    capacities: dict[str, CapacityInfo] = {}
    for array in entry.local_arrays():
        capacities[array.name] = CapacityInfo(
            byte_capacity=array.byte_capacity or f"sizeof({array.name})",
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


def _allocation_element_count(extent: str) -> str | None:
    text = normalize_expression(extent)
    patterns = (
        r"^\((.+)\)\*\(sizeof\([^()]+\)\)$",
        r"^(.+)\*sizeof\([^()]+\)$",
        r"^sizeof\([^()]+\)\*(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            return normalize_expression(match.group(1))
    return None


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
            element_capacity=_allocation_element_count(extent),
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
    if getattr(operation, "callee", "") == "AST_SUBSCRIPT":
        capacity = info.element_capacity
    else:
        capacity = info.byte_capacity
    if capacity is None:
        return None
    return capacity, offset


def _identifier_terms(expression: str) -> set[str]:
    return set(
        re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*(?:->(?:[A-Za-z_][A-Za-z0-9_]*))?\b",
            normalize_expression(expression),
        )
    )


def _dependency_distances(
    entry: FunctionSource, line: int, expression: str
) -> dict[str, int]:
    """Shortest undirected def-use distance from the access extent to related values."""
    graph: dict[str, set[str]] = {}
    for left, right in entry.value_relations_before(line):
        left_terms = _identifier_terms(left)
        right_terms = _identifier_terms(right)
        for lhs in left_terms:
            graph.setdefault(lhs, set()).update(right_terms)
        for rhs in right_terms:
            graph.setdefault(rhs, set()).update(left_terms)

    distances = {term: 0 for term in _identifier_terms(expression)}
    queue = list(distances)
    while queue:
        current = queue.pop(0)
        distance = distances[current]
        for neighbor in graph.get(current, ()):
            if neighbor not in distances:
                distances[neighbor] = distance + 1
                queue.append(neighbor)
    return distances


def _direct_numeric_relation(
    entry: FunctionSource,
    line: int,
    guarded_term: str,
    extent_text: str,
) -> bool:
    """Require an encodable direct value relation between guard and access extent."""
    extent_terms = _identifier_terms(extent_text)
    if guarded_term in extent_terms:
        return True

    encoder = ExpressionEncoder()
    for left, right in entry.value_relations_before(line):
        left_terms = _identifier_terms(left)
        right_terms = _identifier_terms(right)
        if guarded_term not in left_terms:
            continue
        if not (right_terms & extent_terms):
            continue
        try:
            encoder.equality(left, right)
        except Exception:
            continue
        return True
    return False


def _guard_coverage_condition(
    entry: FunctionSource,
    line: int,
    extent_text: str,
    path_constraints: Iterable[str],
) -> str | None:
    """Construct a VC that checks whether the nearest related upper-bound guard
    also bounds the actual access extent.

    The guard remains a path constraint; its RHS is never treated as object
    capacity. Selection is based on shortest def-use distance from the guarded
    expression to the actual access extent.
    """
    distances = _dependency_distances(entry, line, extent_text)
    candidates: list[tuple[int, str]] = []

    for relation in path_constraints:
        match = re.match(r"^(.*?)(<=|<)(.*)$", normalize_expression(relation))
        if not match:
            continue
        left, operator, right = match.groups()
        left_terms = _identifier_terms(left)
        reachable_terms = [term for term in left_terms if term in distances]
        if not reachable_terms:
            continue
        supported_terms = [
            term
            for term in reachable_terms
            if _direct_numeric_relation(entry, line, term, extent_text)
        ]
        if not supported_terms:
            continue
        # Prefer the closest guard whose relationship to the access extent can
        # actually be represented in the current numeric model.
        rhs = right if operator == "<=" else f"({right})-1"
        candidates.append(
            (min(distances[term] for term in supported_terms), f"{extent_text}<={rhs}")
        )

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _add_program_constraints(
    solver: Solver,
    encoder: ExpressionEncoder,
    entry: FunctionSource,
    line: int,
) -> tuple[str, ...]:
    path_constraints = tuple(entry.continuation_constraints_before(line))
    for relation in path_constraints:
        try:
            solver.add(encoder.comparison(relation))
        except Exception:
            continue
    for left, right in entry.value_relations_before(line):
        try:
            solver.add(encoder.equality(left, right))
        except Exception:
            continue
    return path_constraints


def _check_access(
    entry: FunctionSource,
    operation: object,
    capacities: dict[str, CapacityInfo],
    signed: set[str],
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
    path_constraints = _add_program_constraints(solver, encoder, entry, line)
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
        constrained_text = " ".join(path_constraints)
        unconstrained_parameters = sorted(
            token
            for token in signed_extent_tokens
            if token in entry.parameters and token not in constrained_text
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
        _, unresolved_offset = _buffer_base_and_offset(buffer_text)
        if normalize_expression(unresolved_offset) != "0":
            return AccessCheck(
                kind,
                buffer_text,
                extent_text,
                line,
                "UNKNOWN",
                f"capacity is unknown for indexed/pointer-offset access {buffer_text}",
                tuple(conditions),
                path_constraints,
                {},
            )
        guard_vc_text = _guard_coverage_condition(
            entry, line, extent_text, path_constraints
        )
        if guard_vc_text is not None:
            try:
                guard_vc = encoder.comparison(guard_vc_text)
            except Exception:
                guard_vc = None
            if guard_vc is not None:
                conditions.append(
                    VerificationCondition(
                        kind,
                        buffer_text,
                        extent_text,
                        guard_vc_text,
                        line,
                    )
                )
                check = Solver()
                check.add(*solver.assertions())
                check.add(Not(guard_vc))
                result = check.check()
                if result == sat:
                    return AccessCheck(
                        kind,
                        buffer_text,
                        extent_text,
                        line,
                        "POTENTIAL_VIOLATION",
                        f"guard does not cover actual access extent: {guard_vc_text}",
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
                        "nearest data-dependent upper-bound guard covers the access extent",
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
            f"capacity/valid extent is unknown for {buffer_text}",
            tuple(conditions),
            path_constraints,
            {},
        )

    capacity_text, offset_text = capacity
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
    signed = _signed_names(entry)

    accesses = tuple(
        _check_access(entry, operation, capacities, signed)
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
