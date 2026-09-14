from __future__ import annotations

import re
from collections import deque
from dataclasses import asdict, dataclass

from .cpg import FunctionGraph, GraphNode


SEMANTIC_GROUPS: dict[str, tuple[str, ...]] = {
    "memory": ("MEMORY_OPERATION", "MEMORY_OBJECT"),
    "dependence": ("DATA_DEPENDENCE",),
    "constraint": ("CONTROL_CONSTRAINT",),
    "lifetime": ("LIFETIME",),
    "pattern": ("POTENTIAL_PATTERN",),
}
CATEGORY_TO_GROUP = {
    category: group
    for group, categories in SEMANTIC_GROUPS.items()
    for category in categories
}

VULNERABILITY_FEATURES = (
    "memory_write",
    "memory_read",
    "array_access",
    "pointer_dereference",
    "allocation",
    "deallocation",
    "known_capacity",
    "data_dependence",
    "parameter_dependence",
    "size_arithmetic",
    "control_constraint",
    "bounds_constraint",
    "null_constraint",
    "lifetime_relation",
)
FEATURE_TO_GROUP = {
    "memory_write": "memory",
    "memory_read": "memory",
    "array_access": "memory",
    "pointer_dereference": "memory",
    "allocation": "memory",
    "known_capacity": "memory",
    "deallocation": "lifetime",
    "data_dependence": "dependence",
    "parameter_dependence": "dependence",
    "size_arithmetic": "dependence",
    "control_constraint": "constraint",
    "bounds_constraint": "constraint",
    "null_constraint": "constraint",
    "lifetime_relation": "lifetime",
}

_WRITE_APIS = {
    "memcpy": (0, 2), "memmove": (0, 2), "mempcpy": (0, 2), "memset": (0, 2),
    "strncpy": (0, 2), "strncat": (0, 2), "strlcpy": (0, 2), "strlcat": (0, 2),
    "recv": (1, 2), "recvfrom": (1, 2), "read": (1, 2), "fread": (0, None),
    "strcpy": (0, None), "strcat": (0, None), "sprintf": (0, None), "vsprintf": (0, None),
    "snprintf": (0, 1), "vsnprintf": (0, 1), "bcopy": (1, 2), "bzero": (0, 1),
    "gets": (0, None),
}
_READ_APIS = {
    "memcpy": (1, 2), "memmove": (1, 2), "mempcpy": (1, 2), "memcmp": (0, 2),
    "bcopy": (0, 2), "write": (1, 2), "send": (1, 2), "sendto": (1, 2), "fwrite": (0, None),
}
_ALLOC_APIS = {"malloc", "calloc", "realloc", "kmalloc", "kzalloc", "vmalloc"}
_FREE_APIS = {"free", "kfree", "vfree"}
_UNBOUNDED_WRITE_APIS = {"strcpy", "strcat", "sprintf", "vsprintf", "gets"}
_CONTROL_PREFIXES = ("if ", "if(", "while ", "while(", "for ", "for(", "switch ", "switch(")

_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")
_ARRAY_ACCESS = re.compile(r"\b([A-Za-z_]\w*(?:->\w+|\.\w+)*)\s*\[\s*([^\]]+?)\s*\]")
_ARRAY_DECL = re.compile(r"\b([A-Za-z_]\w*)\s*\[\s*([A-Za-z_0-9()+\-*/<>&| ]+)\s*\]")
_ASSIGNMENT = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*(?=[A-Za-z_]\w*\s*\()")
_NEW_ASSIGNMENT = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*new\b")
_DEREF = re.compile(r"(?<![\w)])\*\s*([A-Za-z_]\w*)")
_ARROW = re.compile(r"\b([A-Za-z_]\w*)\s*->")
_DELETE = re.compile(r"\bdelete(?:\s*\[\s*\])?\s+([A-Za-z_]\w*)")
_NEW_EXTENT = re.compile(r"\bnew\b[^;\[]*\[\s*([^\]]+)\s*\]")
_INTEGER = re.compile(r"^(?:0[xX][0-9A-Fa-f]+|\d+)[uUlL]*$")
_BINARY_ARITHMETIC = re.compile(
    r"(?:[A-Za-z_0-9)\]]\s*(?:\+|-|\*|/|%|<<|>>)\s*[A-Za-z_0-9(\[])"
)
_COMPARISON_PARTS = ("<=", ">=", "==", "!=", "<", ">")
_ARITHMETIC_LABEL_PARTS = (
    "addition", "subtraction", "multiplication", "division", "modulo",
    "shiftleft", "shiftright",
)


@dataclass(frozen=True)
class SemanticItem:
    category: str
    kind: str
    detail: str

    def as_text(self) -> str:
        return f"{self.kind} {self.detail}".strip()

    def as_json(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class VulnerabilitySemantics:
    items: tuple[SemanticItem, ...]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return feature_names_from_items(item.as_json() for item in self.items)

    def render(self, excluded_groups: tuple[str, ...] = (), max_per_category: int = 32) -> str:
        return render_semantic_items(
            [item.as_json() for item in self.items],
            excluded_groups=excluded_groups,
            max_per_category=max_per_category,
        )

    def as_json(self) -> list[dict[str, str]]:
        return [item.as_json() for item in self.items]


@dataclass(frozen=True)
class _Operation:
    node_id: str
    kind: str
    object_name: str | None
    extent: str | None
    code: str


def validate_semantic_groups(groups: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for group in groups:
        value = group.strip().lower()
        if not value:
            continue
        if value not in SEMANTIC_GROUPS:
            raise ValueError(
                f"unknown semantic group {group!r}; expected one of {', '.join(SEMANTIC_GROUPS)}"
            )
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _round_robin_by_kind(rows: list[tuple[str, str]], limit: int) -> list[str]:
    groups: dict[str, list[str]] = {}
    for kind, text in rows:
        values = groups.setdefault(kind, [])
        if text not in values:
            values.append(text)
    selected: list[str] = []
    while len(selected) < limit:
        added = False
        for values in groups.values():
            if values and len(selected) < limit:
                selected.append(values.pop(0))
                added = True
        if not added:
            break
    return selected


def render_semantic_items(
    items: list[dict[str, object]] | tuple[dict[str, object], ...],
    *,
    excluded_groups: tuple[str, ...] = (),
    max_per_category: int = 32,
) -> str:
    """Render compact semantics with high-risk patterns first.

    Rows inside a category are interleaved by kind so repeated operations of one
    type cannot consume the whole fixed token budget before other pattern types.
    """
    if max_per_category <= 0:
        raise ValueError("max_per_category must be positive")
    excluded = set(validate_semantic_groups(excluded_groups))
    grouped: dict[str, list[tuple[str, str]]] = {}
    for raw in items:
        category = str(raw.get("category") or "")
        kind = str(raw.get("kind") or "")
        detail = str(raw.get("detail") or "")
        group = CATEGORY_TO_GROUP.get(category)
        if not category or not kind or group is None or group in excluded:
            continue
        grouped.setdefault(category, []).append((kind, f"{kind} {detail}".strip()))

    if not grouped:
        return "[VULNERABILITY_SEMANTICS]\nNO_VULNERABILITY_RELATED_PROGRAM_SEMANTICS"

    ordered_categories = (
        "POTENTIAL_PATTERN",
        "MEMORY_OPERATION",
        "MEMORY_OBJECT",
        "DATA_DEPENDENCE",
        "CONTROL_CONSTRAINT",
        "LIFETIME",
    )
    sections: list[str] = []
    for category in ordered_categories:
        rows = grouped.get(category)
        if not rows:
            continue
        sections.append(f"[{category}]")
        sections.extend(_round_robin_by_kind(rows, max_per_category))
    return "\n".join(sections)


def feature_names_from_items(items) -> tuple[str, ...]:
    features: set[str] = set()
    for raw in items:
        kind = str(raw.get("kind") or "")
        if kind == "MEMORY_WRITE":
            features.add("memory_write")
        elif kind == "MEMORY_READ":
            features.add("memory_read")
        elif kind == "ARRAY_ACCESS":
            features.add("array_access")
        elif kind == "POINTER_DEREFERENCE":
            features.add("pointer_dereference")
        elif kind == "ALLOCATION":
            features.add("allocation")
        elif kind == "DEALLOCATION":
            features.add("deallocation")
        elif kind in {"STATIC_CAPACITY", "DYNAMIC_CAPACITY"}:
            features.add("known_capacity")
        elif kind == "DATA_DEPENDENCE":
            features.add("data_dependence")
        elif kind == "PARAMETER_DEPENDENCE":
            features.update(("data_dependence", "parameter_dependence"))
        elif kind == "SIZE_ARITHMETIC":
            features.add("size_arithmetic")
        elif kind in {"CONTROL_CONDITION", "CONDITION_CONTROLS"}:
            features.add("control_constraint")
        elif kind == "UPPER_BOUND_RELATED_CONDITION":
            features.update(("control_constraint", "bounds_constraint"))
        elif kind in {"NULL_RELATED_CONDITION", "NONNULL_RELATED_CONDITION"}:
            features.update(("control_constraint", "null_constraint"))
        elif kind in {"FREE_THEN_FREE", "FREE_THEN_USE"}:
            features.add("lifetime_relation")
    return tuple(feature for feature in VULNERABILITY_FEATURES if feature in features)


def _compact(text: str, limit: int = 140) -> str:
    value = " ".join(text.split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _identifiers(expression: str | None) -> set[str]:
    return set(_IDENTIFIER.findall(expression)) if expression else set()


def _call_name_and_args(code: str) -> tuple[str | None, tuple[str, ...]]:
    match = re.search(r"\b([A-Za-z_]\w*)\s*\(", code)
    if not match:
        return None, ()
    name = match.group(1)
    start = match.end()
    depth = 1
    index = start
    while index < len(code) and depth:
        char = code[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        index += 1
    if depth != 0:
        return name, ()
    body = code[start : index - 1]
    args: list[str] = []
    current: list[str] = []
    depth = 0
    for char in body:
        if char in "([{":
            depth += 1
        elif char in ")]}" :
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current or body.strip():
        args.append("".join(current).strip())
    return name, tuple(args)


def _assigned_name(code: str) -> str | None:
    match = _ASSIGNMENT.search(code)
    return match.group(1) if match else None


def _argument(args: tuple[str, ...], index: int | None) -> str | None:
    if index is None or index < 0 or index >= len(args):
        return None
    value = args[index].strip()
    return value or None


def _allocation_extent(name: str, args: tuple[str, ...]) -> str | None:
    if name == "calloc" and len(args) >= 2:
        return f"({args[0]})*({args[1]})"
    if name == "realloc":
        return _argument(args, 1)
    return _argument(args, 0)


def _is_arithmetic_node(node: GraphNode) -> bool:
    label = node.label.lower()
    return any(part in label for part in _ARITHMETIC_LABEL_PARTS) or bool(
        _BINARY_ARITHMETIC.search(node.code)
    )


def _node_operations(node: GraphNode) -> tuple[_Operation, ...]:
    code = node.code
    operations: list[_Operation] = []
    name, args = _call_name_and_args(code)
    if name in _WRITE_APIS:
        object_index, extent_index = _WRITE_APIS[name]
        extent = f"({args[1]})*({args[2]})" if name == "fread" and len(args) >= 3 else _argument(args, extent_index)
        operations.append(_Operation(node.node_id, "MEMORY_WRITE", _argument(args, object_index), extent, code))
    if name in _READ_APIS:
        object_index, extent_index = _READ_APIS[name]
        extent = f"({args[1]})*({args[2]})" if name == "fwrite" and len(args) >= 3 else _argument(args, extent_index)
        operations.append(_Operation(node.node_id, "MEMORY_READ", _argument(args, object_index), extent, code))
    if name in _ALLOC_APIS:
        operations.append(_Operation(node.node_id, "ALLOCATION", _assigned_name(code), _allocation_extent(name, args), code))
    if name in _FREE_APIS:
        operations.append(_Operation(node.node_id, "DEALLOCATION", _argument(args, 0), None, code))

    label_lower = node.label.lower()
    if "<operator>.new" in label_lower or label_lower.endswith(".new"):
        extent_match = _NEW_EXTENT.search(code)
        new_assignment = _NEW_ASSIGNMENT.search(code)
        operations.append(
            _Operation(node.node_id, "ALLOCATION",
                       new_assignment.group(1) if new_assignment else None,
                       extent_match.group(1).strip() if extent_match else None, code)
        )
    if "<operator>.delete" in label_lower or label_lower.endswith(".delete"):
        delete_match = _DELETE.search(code)
        operations.append(
            _Operation(node.node_id, "DEALLOCATION",
                       delete_match.group(1) if delete_match else None, None, code)
        )

    for base, index in _ARRAY_ACCESS.findall(code):
        operations.append(_Operation(node.node_id, "ARRAY_ACCESS", base, index.strip(), code))

    pointer_names: set[str] = set()
    if "indirection" in label_lower or "fieldaccess" in label_lower:
        pointer_names.update(_DEREF.findall(code))
        pointer_names.update(_ARROW.findall(code))
    elif "->" in code:
        pointer_names.update(_ARROW.findall(code))
    for pointer in sorted(pointer_names):
        operations.append(_Operation(node.node_id, "POINTER_DEREFERENCE", pointer, None, code))

    dedup: dict[tuple[str, str | None, str | None], _Operation] = {}
    for operation in operations:
        dedup[(operation.kind, operation.object_name, operation.extent)] = operation
    return tuple(dedup.values())


def _control_expression(node: GraphNode) -> str | None:
    code = " ".join(node.code.split()).strip()
    if not code:
        return None
    lower = code.lower()
    if node.label == "CONTROL_STRUCTURE" or lower.startswith(_CONTROL_PREFIXES):
        start = code.find("(")
        if start < 0:
            return code
        depth = 0
        for index in range(start, len(code)):
            if code[index] == "(":
                depth += 1
            elif code[index] == ")":
                depth -= 1
                if depth == 0:
                    return code[start + 1 : index].strip()
    if any(part in node.label.lower() for part in ("lessthan", "greaterthan", "equals", "notequals")):
        return code
    return code if any(op in code for op in _COMPARISON_PARTS) else None


def _controlling_conditions(graph: FunctionGraph) -> dict[str, list[tuple[str, str]]]:
    conditions: dict[str, list[tuple[str, str]]] = {}
    for edge in graph.edges:
        if edge.kind != "CDG":
            continue
        source = graph.nodes.get(edge.source)
        if source is None:
            continue
        condition = _control_expression(source)
        if condition:
            conditions.setdefault(edge.target, []).append((edge.source, condition))
    return conditions


def _cfg_adjacency(graph: FunctionGraph) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind == "CFG":
            adjacency.setdefault(edge.source, []).append(edge.target)
    return adjacency


def _reachable(adjacency: dict[str, list[str]], start: str) -> set[str]:
    seen = {start}
    reachable: set[str] = set()
    queue: deque[str] = deque(adjacency.get(start, ()))
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        reachable.add(node)
        queue.extend(adjacency.get(node, ()))
    return reachable


def _static_integer(expression: str | None) -> int | None:
    if not expression:
        return None
    value = expression.strip()
    if not _INTEGER.match(value):
        return None
    value = re.sub(r"[uUlL]+$", "", value)
    try:
        return int(value, 0)
    except ValueError:
        return None


def _split_comparison(condition: str) -> tuple[str, str, str] | None:
    depth = 0
    for index, char in enumerate(condition):
        if char in "([{":
            depth += 1
        elif char in ")]}" :
            depth = max(0, depth - 1)
        if depth:
            continue
        for operator in ("<=", ">=", "<", ">"):
            if condition.startswith(operator, index):
                left = condition[:index].strip(" ()")
                right = condition[index + len(operator):].strip(" ()")
                if left and right:
                    return left, operator, right
    return None


def _upper_bound_condition(condition: str, expression: str) -> bool:
    """Return whether condition text compares expression against an upper bound.

    This only checks the comparison form. It does not claim that the condition is
    a proven safe guard because exported CDG edges do not preserve branch polarity.
    """
    comparison = _split_comparison(condition)
    if comparison is None:
        return False
    left, operator, right = comparison
    expression_names = _identifiers(expression)
    if not expression_names:
        return False
    left_names = _identifiers(left)
    right_names = _identifiers(right)
    if operator in {"<", "<="}:
        return bool(expression_names & left_names) and bool(right_names or _static_integer(right) is not None)
    if operator in {">", ">="}:
        return bool(expression_names & right_names) and bool(left_names or _static_integer(left) is not None)
    return False


def _nonnull_condition(condition: str, pointer: str) -> bool:
    escaped = re.escape(pointer)
    return bool(
        re.search(rf"\b{escaped}\b\s*!=\s*(?:NULL|nullptr|0)\b", condition)
        or re.search(rf"(?:NULL|nullptr|0)\s*!=\s*\b{escaped}\b", condition)
        or re.fullmatch(rf"\s*\b{escaped}\b\s*", condition)
        or re.fullmatch(rf"\s*!!\s*\b{escaped}\b\s*", condition)
    )


def _null_related_condition(condition: str, pointer: str) -> bool:
    escaped = re.escape(pointer)
    return bool(
        re.search(rf"\b{escaped}\b\s*(?:!=|==)\s*(?:NULL|nullptr|0)\b", condition)
        or re.search(rf"(?:NULL|nullptr|0)\s*(?:!=|==)\s*\b{escaped}\b", condition)
        or re.search(rf"!\s*\b{escaped}\b", condition)
        or re.fullmatch(rf"\s*\b{escaped}\b\s*", condition)
    )


def _matching_condition(
    conditions: dict[str, list[tuple[str, str]]],
    node_id: str,
    predicate,
) -> str | None:
    for _, condition in conditions.get(node_id, ()):
        if predicate(condition):
            return condition
    return None


def _capacity_facts(operations: list[_Operation], graph: FunctionGraph) -> dict[str, str]:
    capacities: dict[str, str] = {}
    for node in graph.nodes.values():
        if "LOCAL" not in node.label.upper():
            continue
        for name, capacity in _ARRAY_DECL.findall(node.code):
            value = " ".join(capacity.split())
            if name not in capacities and value:
                capacities[name] = value

    dynamic: dict[str, set[str]] = {}
    for operation in operations:
        if operation.kind == "ALLOCATION" and operation.object_name and operation.extent:
            dynamic.setdefault(operation.object_name, set()).add(_compact(operation.extent, 100))
    for name, extents in dynamic.items():
        if name not in capacities and len(extents) == 1:
            capacities[name] = next(iter(extents))
    return capacities


def _ddg_sources(
    graph: FunctionGraph,
    sink_id: str,
    *,
    max_depth: int = 3,
    max_sources: int = 3,
) -> tuple[GraphNode, ...]:
    reverse: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind == "DDG":
            reverse.setdefault(edge.target, []).append(edge.source)
    queue: deque[tuple[str, int]] = deque([(sink_id, 0)])
    visited = {sink_id}
    sources: list[GraphNode] = []
    while queue and len(sources) < max_sources:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for source_id in reverse.get(current, ()):
            if source_id in visited:
                continue
            visited.add(source_id)
            source = graph.nodes.get(source_id)
            if source is not None:
                sources.append(source)
                if len(sources) >= max_sources:
                    break
            queue.append((source_id, depth + 1))
    return tuple(sources)


def _operation_detail(
    operation: _Operation,
    *,
    graph: FunctionGraph,
    conditions: dict[str, list[tuple[str, str]]],
    capacities: dict[str, str],
) -> str:
    fields = [f"operation={_compact(operation.code, 100)}"]
    if operation.object_name:
        label = "pointer" if operation.kind == "POINTER_DEREFERENCE" else "object"
        fields.append(f"{label}={operation.object_name}")
    if operation.extent:
        field = "index" if operation.kind == "ARRAY_ACCESS" else "extent"
        fields.append(f"{field}={_compact(operation.extent, 60)}")
    capacity = capacities.get(operation.object_name or "")
    if capacity:
        fields.append(f"capacity={capacity}")

    if operation.extent and operation.kind in {"MEMORY_WRITE", "ARRAY_ACCESS"}:
        upper = _matching_condition(
            conditions, operation.node_id,
            lambda condition: _upper_bound_condition(condition, operation.extent or ""),
        )
        fields.append(f"upper_bound_condition={_compact(upper, 80) if upper else 'none'}")
    if operation.kind == "POINTER_DEREFERENCE" and operation.object_name:
        nonnull = _matching_condition(
            conditions, operation.node_id,
            lambda condition: _nonnull_condition(condition, operation.object_name or ""),
        )
        fields.append(f"nonnull_condition={_compact(nonnull, 80) if nonnull else 'none'}")

    sources = _ddg_sources(graph, operation.node_id)
    if sources:
        fields.append("data_from=" + "; ".join(_compact(source.code, 55) for source in sources))
    return " ".join(fields)


def _operation_items(
    operations: list[_Operation],
    *,
    graph: FunctionGraph,
    conditions: dict[str, list[tuple[str, str]]],
    capacities: dict[str, str],
) -> list[SemanticItem]:
    items: list[SemanticItem] = []
    for operation in operations:
        if operation.kind in {"MEMORY_WRITE", "MEMORY_READ", "ARRAY_ACCESS", "POINTER_DEREFERENCE"}:
            items.append(
                SemanticItem(
                    "MEMORY_OPERATION",
                    operation.kind,
                    _operation_detail(operation, graph=graph, conditions=conditions, capacities=capacities),
                )
            )
        elif operation.kind == "ALLOCATION":
            detail = f"operation={_compact(operation.code, 100)} object={operation.object_name or '?'} extent={operation.extent or '?'}"
            items.append(SemanticItem("MEMORY_OBJECT", "ALLOCATION", detail))
        elif operation.kind == "DEALLOCATION":
            items.append(
                SemanticItem(
                    "LIFETIME", "DEALLOCATION",
                    f"operation={_compact(operation.code, 100)} object={operation.object_name or '?'}",
                )
            )
    return items


def _dependence_items(graph: FunctionGraph, operations_by_node: dict[str, tuple[_Operation, ...]]) -> list[SemanticItem]:
    items: list[SemanticItem] = []
    for sink_id, operations in operations_by_node.items():
        for source in _ddg_sources(graph, sink_id):
            for operation in operations:
                detail = (
                    f"from={_compact(source.code, 90)} operation={operation.kind} "
                    f"object={operation.object_name or '?'} extent={operation.extent or '?'}"
                )
                items.append(SemanticItem("DATA_DEPENDENCE", "DATA_DEPENDENCE", detail))
                if "PARAMETER" in source.label.upper():
                    items.append(
                        SemanticItem(
                            "DATA_DEPENDENCE", "PARAMETER_DEPENDENCE",
                            f"parameter={_compact(source.code, 70)} operation={operation.kind}",
                        )
                    )
                if _is_arithmetic_node(source):
                    items.append(
                        SemanticItem(
                            "DATA_DEPENDENCE", "SIZE_ARITHMETIC",
                            f"expr={_compact(source.code, 90)} operation={operation.kind}",
                        )
                    )
    return items


def _constraint_items(
    conditions: dict[str, list[tuple[str, str]]],
    operations_by_node: dict[str, tuple[_Operation, ...]],
) -> list[SemanticItem]:
    items: list[SemanticItem] = []
    seen_conditions: set[tuple[str, str]] = set()
    for node_id, controlling in conditions.items():
        for source_id, condition in controlling:
            key = (source_id, condition)
            if key not in seen_conditions:
                seen_conditions.add(key)
                items.append(
                    SemanticItem(
                        "CONTROL_CONSTRAINT", "CONTROL_CONDITION",
                        f"node={source_id} expr={_compact(condition, 100)}",
                    )
                )
            for operation in operations_by_node.get(node_id, ()):
                detail = (
                    f"expr={_compact(condition, 90)} operation={operation.kind} "
                    f"object={operation.object_name or '?'} extent={operation.extent or '?'}"
                )
                items.append(SemanticItem("CONTROL_CONSTRAINT", "CONDITION_CONTROLS", detail))
                if (
                    operation.kind in {"MEMORY_WRITE", "ARRAY_ACCESS"}
                    and operation.extent
                    and _upper_bound_condition(condition, operation.extent)
                ):
                    items.append(
                        SemanticItem("CONTROL_CONSTRAINT", "UPPER_BOUND_RELATED_CONDITION", detail)
                    )
                if operation.kind == "POINTER_DEREFERENCE" and operation.object_name:
                    if _nonnull_condition(condition, operation.object_name):
                        items.append(
                            SemanticItem("CONTROL_CONSTRAINT", "NONNULL_RELATED_CONDITION", detail)
                        )
                    elif _null_related_condition(condition, operation.object_name):
                        items.append(
                            SemanticItem("CONTROL_CONSTRAINT", "NULL_RELATED_CONDITION", detail)
                        )
    return items


def _lifetime_items(
    graph: FunctionGraph,
    operations: list[_Operation],
    operations_by_node: dict[str, tuple[_Operation, ...]],
) -> list[SemanticItem]:
    items: list[SemanticItem] = []
    adjacency = _cfg_adjacency(graph)
    for deallocation in (
        operation for operation in operations
        if operation.kind == "DEALLOCATION" and operation.object_name
    ):
        for node_id in _reachable(adjacency, deallocation.node_id):
            for operation in operations_by_node.get(node_id, ()):
                if operation.object_name != deallocation.object_name:
                    continue
                if operation.kind == "DEALLOCATION":
                    detail = (
                        f"object={deallocation.object_name} "
                        f"first={_compact(deallocation.code, 70)} second={_compact(operation.code, 70)}"
                    )
                    items.append(SemanticItem("LIFETIME", "FREE_THEN_FREE", detail))
                    items.append(SemanticItem("POTENTIAL_PATTERN", "FREE_THEN_FREE", detail))
                elif operation.kind in {
                    "MEMORY_READ", "MEMORY_WRITE", "ARRAY_ACCESS", "POINTER_DEREFERENCE"
                }:
                    detail = (
                        f"object={deallocation.object_name} "
                        f"free={_compact(deallocation.code, 70)} later_use={_compact(operation.code, 70)}"
                    )
                    items.append(SemanticItem("LIFETIME", "FREE_THEN_USE", detail))
                    items.append(SemanticItem("POTENTIAL_PATTERN", "FREE_THEN_USE", detail))
    return items


def _potential_pattern_items(
    graph: FunctionGraph,
    operations: list[_Operation],
    conditions: dict[str, list[tuple[str, str]]],
    capacities: dict[str, str],
) -> list[SemanticItem]:
    items: list[SemanticItem] = []
    for operation in operations:
        detail = _operation_detail(
            operation, graph=graph, conditions=conditions, capacities=capacities
        )
        if operation.kind == "MEMORY_WRITE":
            api_name, _ = _call_name_and_args(operation.code)
            if api_name in _UNBOUNDED_WRITE_APIS:
                items.append(SemanticItem("POTENTIAL_PATTERN", "UNBOUNDED_WRITE", detail))
            if operation.extent and _identifiers(operation.extent):
                upper = _matching_condition(
                    conditions, operation.node_id,
                    lambda condition: _upper_bound_condition(condition, operation.extent or ""),
                )
                if upper is None:
                    items.append(
                        SemanticItem(
                            "POTENTIAL_PATTERN",
                            "WRITE_EXTENT_WITHOUT_UPPER_BOUND_CONDITION",
                            detail,
                        )
                    )
            capacity = capacities.get(operation.object_name or "")
            extent_value = _static_integer(operation.extent)
            capacity_value = _static_integer(capacity)
            if (
                extent_value is not None
                and capacity_value is not None
                and extent_value > capacity_value
            ):
                items.append(
                    SemanticItem("POTENTIAL_PATTERN", "WRITE_EXCEEDS_KNOWN_CAPACITY", detail)
                )

        elif operation.kind == "ARRAY_ACCESS" and operation.extent:
            upper = _matching_condition(
                conditions, operation.node_id,
                lambda condition: _upper_bound_condition(condition, operation.extent or ""),
            )
            if upper is None:
                items.append(
                    SemanticItem(
                        "POTENTIAL_PATTERN",
                        "ARRAY_INDEX_WITHOUT_UPPER_BOUND_CONDITION",
                        detail,
                    )
                )
            capacity = capacities.get(operation.object_name or "")
            index_value = _static_integer(operation.extent)
            capacity_value = _static_integer(capacity)
            if (
                index_value is not None
                and capacity_value is not None
                and index_value >= capacity_value
            ):
                items.append(
                    SemanticItem("POTENTIAL_PATTERN", "ARRAY_INDEX_EXCEEDS_KNOWN_CAPACITY", detail)
                )

        elif operation.kind == "POINTER_DEREFERENCE" and operation.object_name:
            nonnull = _matching_condition(
                conditions, operation.node_id,
                lambda condition: _nonnull_condition(condition, operation.object_name or ""),
            )
            if nonnull is None:
                items.append(
                    SemanticItem(
                        "POTENTIAL_PATTERN",
                        "DEREFERENCE_WITHOUT_NONNULL_CONDITION",
                        detail,
                    )
                )

    for edge in graph.edges:
        if edge.kind != "DDG":
            continue
        source = graph.nodes.get(edge.source)
        target = graph.nodes.get(edge.target)
        if source is None or target is None or not _is_arithmetic_node(source):
            continue
        for operation in _node_operations(target):
            if operation.kind not in {"MEMORY_WRITE", "ALLOCATION", "ARRAY_ACCESS"}:
                continue
            upper = _matching_condition(
                conditions, target.node_id,
                lambda condition: _upper_bound_condition(condition, source.code),
            )
            if upper is None:
                detail = (
                    f"expr={_compact(source.code, 90)} "
                    f"sink={_compact(target.code, 100)} data_from={_compact(source.code, 70)}"
                )
                items.append(
                    SemanticItem(
                        "POTENTIAL_PATTERN",
                        "SIZE_ARITHMETIC_WITHOUT_UPPER_BOUND_CONDITION",
                        detail,
                    )
                )
    return items


def extract_vulnerability_semantics(graph: FunctionGraph) -> VulnerabilitySemantics:
    operations_by_node = {
        node_id: operations
        for node_id, node in graph.nodes.items()
        if (operations := _node_operations(node))
    }
    operations = [operation for values in operations_by_node.values() for operation in values]
    conditions = _controlling_conditions(graph)
    capacities = _capacity_facts(operations, graph)

    items: list[SemanticItem] = []
    items.extend(_operation_items(
        operations, graph=graph, conditions=conditions, capacities=capacities
    ))

    static_names: set[str] = set()
    for node in graph.nodes.values():
        if "LOCAL" in node.label.upper():
            static_names.update(name for name, _ in _ARRAY_DECL.findall(node.code))
    for name, capacity in capacities.items():
        kind = "STATIC_CAPACITY" if name in static_names else "DYNAMIC_CAPACITY"
        items.append(
            SemanticItem("MEMORY_OBJECT", kind, f"object={name} capacity={_compact(capacity, 90)}")
        )

    items.extend(_dependence_items(graph, operations_by_node))
    items.extend(_constraint_items(conditions, operations_by_node))
    items.extend(_lifetime_items(graph, operations, operations_by_node))
    items.extend(_potential_pattern_items(graph, operations, conditions, capacities))

    deduplicated: list[SemanticItem] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (item.category, item.kind, item.detail)
        if key not in seen:
            seen.add(key)
            deduplicated.append(item)
    return VulnerabilitySemantics(tuple(deduplicated))