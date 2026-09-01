from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass
from typing import Iterable

from z3 import And, Int, IntVal, Not, Solver, sat, unsat

from .source import FunctionSource, normalize_expression


@dataclass(frozen=True)
class VerificationCondition:
    access_kind: str
    buffer: str
    extent: str
    condition: str
    line: int


@dataclass(frozen=True)
class ConstraintResult:
    status: str
    reason: str
    conditions: tuple[VerificationCondition, ...]
    model: dict[str, int]

    def as_json(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "conditions": [asdict(item) for item in self.conditions],
            "model": self.model,
        }


class ExpressionEncoder:
    """Small integer-expression encoder for bounds reasoning.

    Unknown calls, field expressions, casts and sizeof terms are preserved as opaque
    integer symbols. Arithmetic relations around them remain solvable without
    pretending to understand their internal semantics.
    """

    _atom_pattern = re.compile(
        r"""
        sizeof\s*\([^()]*\)
        |[A-Za-z_][A-Za-z0-9_]*(?:->|\.)[A-Za-z_][A-Za-z0-9_.>-]*
        |[A-Za-z_][A-Za-z0-9_]*\s*\([^()]*\)
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
            symbol = self.symbol(raw)
            result = result[: match.start()] + str(symbol) + result[match.end() :]
        return result

    def encode(self, expression: str):
        text = normalize_expression(expression)
        if not text:
            raise ValueError("empty expression")
        text = re.sub(r"\(\s*(?:unsigned|signed|long|short|int|char|size_t|ssize_t)[^)]*\)", "", text)
        text = self._replace_atoms(text)
        identifiers = sorted(
            set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)),
            key=len,
            reverse=True,
        )
        for name in identifiers:
            if name.startswith("v_"):
                continue
            symbol = self.symbol(name)
            text = re.sub(rf"\b{re.escape(name)}\b", str(symbol), text)
        tree = ast.parse(text, mode="eval")
        return self._node(tree.body)

    def comparison(self, expression: str):
        match = re.match(r"^(.*?)(<=|>=|==|!=|<|>)(.*)$", normalize_expression(expression))
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
        r"\b(?:signed\s+)?(?:int|short|ssize_t|long)\s+([A-Za-z_][A-Za-z0-9_]*)",
        entry.text,
    ):
        names.add(match.group(1))
    return names


def _local_array_capacities(entry: FunctionSource) -> dict[str, str]:
    return {
        normalize_expression(name): normalize_expression(size)
        for name, size in re.findall(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([^\]]+)\s*\]", entry.text
        )
    }


def _parameter_capacities(entry: FunctionSource) -> dict[str, str]:
    capacities: dict[str, str] = {}
    for index, (name, declaration) in enumerate(
        zip(entry.parameters, entry.parameter_types)
    ):
        if "*" not in declaration and "[" not in declaration:
            continue
        for candidate in entry.parameters[index + 1 : index + 3]:
            if re.search(r"(?:buflen|len|size|capacity|cap)$", candidate, re.I):
                capacities[normalize_expression(name)] = normalize_expression(candidate)
                break
    return capacities


def _continuation_constraints(entry: FunctionSource, access_line: int) -> list[str]:
    """Infer simple constraints established by reject-before-access checks."""
    constraints: list[str] = []
    lines = entry.text.splitlines()
    relative_access = max(1, access_line - entry.start_line + 1)
    prefix = "\n".join(lines[:relative_access])
    pattern = re.compile(
        r"if\s*\(([^()]*)\)\s*(?:\{[^{}]{0,300})?"
        r"(?:return\b[^;]*;|goto\b[^;]*;|break\s*;)",
        re.S,
    )
    for match in pattern.finditer(prefix):
        condition = normalize_expression(match.group(1))
        comparison = re.match(r"^(.*?)(<=|>=|==|!=|<|>)(.*)$", condition)
        if not comparison:
            continue
        left, operator, right = comparison.groups()
        inverse = {
            ">": "<=",
            ">=": "<",
            "<": ">=",
            "<=": ">",
            "==": "!=",
            "!=": "==",
        }[operator]
        constraints.append(f"{left}{inverse}{right}")
    return constraints


def _buffer_base_and_offset(buffer: str) -> tuple[str, str]:
    text = normalize_expression(buffer)
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "+" and depth == 0:
            return text[:index], text[index + 1 :]
    return text, "0"


def _capacity_for_buffer(
    buffer: str,
    capacities: dict[str, str],
) -> tuple[str, str] | None:
    base, offset = _buffer_base_and_offset(buffer)
    if base in capacities:
        return capacities[base], offset
    return None


def reason_memory_safety(
    entry: FunctionSource,
    operations: Iterable[object],
) -> ConstraintResult:
    operations = list(operations)
    encoder = ExpressionEncoder()
    capacities = _local_array_capacities(entry)
    capacities.update(_parameter_capacities(entry))

    # Bind allocation results to their allocation extents when the call result is known.
    for operation in operations:
        if getattr(operation, "kind", "") != "ALLOC":
            continue
        buffer = normalize_expression(getattr(operation, "buffer", ""))
        extent = normalize_expression(getattr(operation, "extent", ""))
        if buffer and buffer != "return":
            capacities[buffer] = extent

    conditions: list[VerificationCondition] = []
    potential_models: list[tuple[str, dict[str, int]]] = []
    unknown_reasons: list[str] = []
    signed = _signed_names(entry)

    for operation in operations:
        kind = getattr(operation, "kind", "")
        if kind not in {"READ", "WRITE"}:
            continue
        extent_text = normalize_expression(getattr(operation, "extent", ""))
        buffer_text = normalize_expression(getattr(operation, "buffer", ""))
        line = int(getattr(operation, "line", 0))
        if not extent_text or extent_text == "UNBOUNDED":
            continue

        path_constraints = _continuation_constraints(entry, line)
        solver = Solver()
        for relation in path_constraints:
            try:
                solver.add(encoder.comparison(relation))
            except Exception:
                continue

        try:
            extent = encoder.encode(extent_text)
        except Exception:
            unknown_reasons.append(
                f"cannot encode access extent {extent_text} at line {line}"
            )
            continue

        extent_tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", extent_text))
        if extent_tokens & signed:
            vc_text = f"{extent_text} >= 0"
            vc = extent >= 0
            conditions.append(
                VerificationCondition(kind, buffer_text, extent_text, vc_text, line)
            )
            check = Solver()
            check.add(*solver.assertions())
            check.add(Not(vc))
            if check.check() == sat:
                potential_models.append(
                    (
                        f"{kind} length may be negative: {vc_text}",
                        encoder.model_dict(check.model()),
                    )
                )

        capacity = _capacity_for_buffer(buffer_text, capacities)
        if capacity is None:
            unknown_reasons.append(
                f"capacity/valid extent is unknown for {buffer_text} at line {line}"
            )
            continue

        capacity_text, offset_text = capacity
        try:
            capacity_expr = encoder.encode(capacity_text)
            offset_expr = encoder.encode(offset_text)
        except Exception:
            unknown_reasons.append(
                f"cannot encode capacity relation for {buffer_text} at line {line}"
            )
            continue

        vc_text = (
            f"{offset_text} + {extent_text} <= {capacity_text}"
            if normalize_expression(offset_text) != "0"
            else f"{extent_text} <= {capacity_text}"
        )
        vc = offset_expr + extent <= capacity_expr
        conditions.append(
            VerificationCondition(kind, buffer_text, extent_text, vc_text, line)
        )

        check = Solver()
        check.add(*solver.assertions())
        check.add(Not(vc))
        result = check.check()
        if result == sat:
            potential_models.append(
                (
                    f"bounds check may fail: {vc_text}",
                    encoder.model_dict(check.model()),
                )
            )

    if potential_models:
        reason, model = potential_models[0]
        return ConstraintResult(
            "POTENTIAL_VIOLATION", reason, tuple(conditions), model
        )
    if conditions and not unknown_reasons:
        return ConstraintResult(
            "SAFE",
            "all generated bounds conditions are implied by the available constraints",
            tuple(conditions),
            {},
        )
    if conditions:
        return ConstraintResult(
            "UNKNOWN",
            "; ".join(dict.fromkeys(unknown_reasons)),
            tuple(conditions),
            {},
        )
    return ConstraintResult(
        "UNKNOWN",
        "; ".join(dict.fromkeys(unknown_reasons))
        or "no supported bounds condition could be generated",
        tuple(),
        {},
    )
