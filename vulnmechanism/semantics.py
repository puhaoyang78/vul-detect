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
    "static_capacity",
    "data_dependence",
    "parameter_dependence",
    "size_arithmetic",
    "control_guard",
    "bounds_check",
    "null_check",
    "lifetime_relation",
)
FEATURE_TO_GROUP = {
    "memory_write": "memory",
    "memory_read": "memory",
    "array_access": "memory",
    "pointer_dereference": "memory",
    "allocation": "memory",
    "static_capacity": "memory",
    "deallocation": "lifetime",
    "data_dependence": "dependence",
    "parameter_dependence": "dependence",
    "size_arithmetic": "dependence",
    "control_guard": "constraint",
    "bounds_check": "constraint",
    "null_check": "constraint",
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
_DEREF = re.compile(r"(?<![\w)])\*\s*([A-Za-z_]\w*)")
_ARROW = re.compile(r"\b([A-Za-z_]\w*)\s*->")
_INTEGER = re.compile(r"^(?:0[xX][0-9A-Fa-f]+|\d+)[uUlL]*$")
_ARITHMETIC = re.compile(r"(?<![+\-*/%&|^<>])[+\-*/%]|<<|>>")
_COMPARISON = re.compile(r"<=|>=|==|!=|<|>")


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

    def describe(self) -> str:
        fields = [f"node={self.node_id}"]
        if self.object_name:
            fields.append(f"object={self.object_name}")
        if self.extent:
            fields.append(f"extent={self.extent}")
        fields.append(f"code={_compact(self.code)}")
        return " ".join(fields)


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


def render_semantic_items(
    items: list[dict[str, object]] | tuple[dict[str, object], ...],
    *,
    excluded_groups: tuple[str, ...] = (),
    max_per_category: int = 32,
) -> str:
    if max_per_category <= 0:
        raise ValueError("max_per_category must be positive")
    excluded = set(validate_semantic_groups(excluded_groups))
    grouped: dict[str, list[str]] = {}
    for raw in items:
        category = str(raw.get("category") or "")
        kind = str(raw.get("kind") or "")
        detail = str(raw.get("detail") or "")
        group = CATEGORY_TO_GROUP.get(category)
        if not category or not kind or group is None or group in excluded:
            continue
        values = grouped.setdefault(category, [])
        text = f"{kind} {detail}".strip()
        if text not in values and len(values) < max_per_category:
            values.append(text)
    if not grouped:
        return "[VULNERABILITY_SEMANTICS]\nNO_VULNERABILITY_RELATED_PROGRAM_SEMANTICS"
    sections: list[str] = []
    ordered_categories = tuple(
        category for categories in SEMANTIC_GROUPS.values() for category in categories
    )
    for category in ordered_categories:
        values = grouped.get(category)
        if values:
            sections.append(f"[{category}]")
            sections.extend(values)
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
        elif kind == "STATIC_CAPACITY":
            features.add("static_capacity")
        elif kind == "DATA_DEPENDENCE":
            features.add("data_dependence")
        elif kind == "PARAMETER_DEPENDENCE":
            features.update(("data_dependence", "parameter_dependence"))
        elif kind == "SIZE_ARITHMETIC":
            features.add("size_arithmetic")
        elif kind in {"CONTROL_CONDITION", "GUARD_PROTECTS"}:
            features.add("control_guard")
        elif kind == "BOUNDS_CHECK":
            features.update(("control_guard", "bounds_check"))
        elif kind == "NULL_CHECK":
            features.update(("control_guard", "null_check"))
        elif kind in {"DOUBLE_FREE", "USE_AFTER_FREE"}:
            features.add("lifetime_relation")
    return tuple(feature for feature in VULNERABILITY_FEATURES if feature in features)


def _compact(text: str, limit: int = 180) -> str:
    value = " ".join(text.split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


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
        elif char in ")]}":
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

    for base, index in _ARRAY_ACCESS.findall(code):
        operations.append(_Operation(node.node_id, "ARRAY_ACCESS", base, index.strip(), code))

    pointer_names: set[str] = set()
    label_lower = node.label.lower()
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


def _guard_condition(node: GraphNode) -> str | None:
    code = " ".join(node.code.split())
    lower = code.lower()
    if node.label != "CONTROL_STRUCTURE" and not lower.startswith(_CONTROL_PREFIXES):
        return None
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
    return code


def _identifiers(expression: str | None) -> set[str]:
    return set(_IDENTIFIER.findall(expression)) if expression else set()


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


def _condition_checks_null(condition: str, pointer: str) -> bool:
    escaped = re.escape(pointer)
    patterns = (
        rf"\b{escaped}\b\s*(?:!=|==)\s*(?:NULL|nullptr|0)\b",
        rf"(?:NULL|nullptr|0)\s*(?:!=|==)\s*\b{escaped}\b",
        rf"!\s*\b{escaped}\b",
        rf"^\s*\b{escaped}\b\s*$",
    )
    return any(re.search(pattern, condition) for pattern in patterns)


def _controlling_guards(graph: FunctionGraph) -> dict[str, list[tuple[str, str]]]:
    guards: dict[str, list[tuple[str, str]]] = {}
    for edge in graph.edges:
        if edge.kind != "CDG":
            continue
        source = graph.nodes.get(edge.source)
        if source is None:
            continue
        condition = _guard_condition(source)
        if condition:
            guards.setdefault(edge.target, []).append((edge.source, condition))
    return guards


def _cfg_adjacency(graph: FunctionGraph) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind == "CFG":
            adjacency.setdefault(edge.source, []).append(edge.target)
    return adjacency


def _reachable(adjacency: dict[str, list[str]], start: str) -> set[str]:
    seen: set[str] = {start}
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


def _has_matching_guard(guards: dict[str, list[tuple[str, str]]], node_id: str, expression: str | None) -> bool:
    names = _identifiers(expression)
    return bool(names) and any(names & _identifiers(condition) for _, condition in guards.get(node_id, ()))


def _has_null_guard(guards: dict[str, list[tuple[str, str]]], node_id: str, pointer: str) -> bool:
    return any(_condition_checks_null(condition, pointer) for _, condition in guards.get(node_id, ()))


def _capacity_facts(graph: FunctionGraph) -> dict[str, str]:
    capacities: dict[str, str] = {}
    for node in graph.nodes.values():
        if "LOCAL" not in node.label.upper():
            continue
        for name, capacity in _ARRAY_DECL.findall(node.code):
            capacity = " ".join(capacity.split())
            if name not in capacities and capacity:
                capacities[name] = capacity
    return capacities


def _operation_items(operations: list[_Operation]) -> list[SemanticItem]:
    items: list[SemanticItem] = []
    for operation in operations:
        if operation.kind in {"MEMORY_WRITE", "MEMORY_READ", "ARRAY_ACCESS", "POINTER_DEREFERENCE"}:
            items.append(SemanticItem("MEMORY_OPERATION", operation.kind, operation.describe()))
        elif operation.kind == "ALLOCATION":
            items.append(SemanticItem("MEMORY_OBJECT", "ALLOCATION", operation.describe()))
        elif operation.kind == "DEALLOCATION":
            items.append(SemanticItem("LIFETIME", "DEALLOCATION", operation.describe()))
    return items


def _dependence_items(
    graph: FunctionGraph,
    operations_by_node: dict[str, tuple[_Operation, ...]],
    max_depth: int = 3,
) -> list[SemanticItem]:
    items: list[SemanticItem] = []
    reverse: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind == "DDG":
            reverse.setdefault(edge.target, []).append(edge.source)

    for sink_id, operations in operations_by_node.items():
        queue: deque[tuple[str, int]] = deque([(sink_id, 0)])
        visited = {sink_id}
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for source_id in reverse.get(current, ()):
                if source_id in visited:
                    continue
                visited.add(source_id)
                source = graph.nodes.get(source_id)
                if source is None:
                    continue
                for operation in operations:
                    detail = (
                        f"from={_compact(source.code, 100)} operation={operation.kind} "
                        f"object={operation.object_name or '?'} extent={operation.extent or '?'}"
                    )
                    items.append(SemanticItem("DATA_DEPENDENCE", "DATA_DEPENDENCE", detail))
                    if "PARAMETER" in source.label.upper():
                        items.append(
                            SemanticItem(
                                "DATA_DEPENDENCE",
                                "PARAMETER_DEPENDENCE",
                                f"parameter={_compact(source.code, 80)} operation={operation.kind} object={operation.object_name or '?'}",
                            )
                        )
                    if _ARITHMETIC.search(source.code):
                        items.append(
                            SemanticItem(
                                "DATA_DEPENDENCE",
                                "SIZE_ARITHMETIC",
                                f"expr={_compact(source.code, 100)} operation={operation.kind} object={operation.object_name or '?'}",
                            )
                        )
                queue.append((source_id, depth + 1))
    return items


def _constraint_items(
    graph: FunctionGraph,
    guards: dict[str, list[tuple[str, str]]],
    operations_by_node: dict[str, tuple[_Operation, ...]],
) -> list[SemanticItem]:
    items: list[SemanticItem] = []
    seen_conditions: set[str] = set()
    for node in graph.nodes.values():
        condition = _guard_condition(node)
        if condition and condition not in seen_conditions:
            seen_conditions.add(condition)
            items.append(
                SemanticItem("CONTROL_CONSTRAINT", "CONTROL_CONDITION", f"expr={_compact(condition)}")
            )

    for node_id, controlling in guards.items():
        for _, condition in controlling:
            for operation in operations_by_node.get(node_id, ()):
                detail = (
                    f"expr={_compact(condition, 100)} operation={operation.kind} "
                    f"object={operation.object_name or '?'} extent={operation.extent or '?'}"
                )
                items.append(SemanticItem("CONTROL_CONSTRAINT", "GUARD_PROTECTS", detail))
                if operation.kind in {"MEMORY_WRITE", "ARRAY_ACCESS"}:
                    controlled = operation.extent
                    if controlled and _COMPARISON.search(condition) and (
                        _identifiers(controlled) & _identifiers(condition)
                    ):
                        items.append(SemanticItem("CONTROL_CONSTRAINT", "BOUNDS_CHECK", detail))
                if operation.kind == "POINTER_DEREFERENCE" and operation.object_name:
                    if _condition_checks_null(condition, operation.object_name):
                        items.append(SemanticItem("CONTROL_CONSTRAINT", "NULL_CHECK", detail))
    return items


def _lifetime_items(
    graph: FunctionGraph,
    operations: list[_Operation],
    operations_by_node: dict[str, tuple[_Operation, ...]],
) -> list[SemanticItem]:
    items: list[SemanticItem] = []
    adjacency = _cfg_adjacency(graph)
    for deallocation in (
        operation
        for operation in operations
        if operation.kind == "DEALLOCATION" and operation.object_name
    ):
        for node_id in _reachable(adjacency, deallocation.node_id):
            for operation in operations_by_node.get(node_id, ()):
                if operation.object_name != deallocation.object_name:
                    continue
                if operation.kind == "DEALLOCATION":
                    items.append(
                        SemanticItem(
                            "POTENTIAL_PATTERN",
                            "DOUBLE_FREE",
                            f"object={deallocation.object_name} first_node={deallocation.node_id} second_node={node_id}",
                        )
                    )
                elif operation.kind in {
                    "MEMORY_READ", "MEMORY_WRITE", "ARRAY_ACCESS", "POINTER_DEREFERENCE"
                }:
                    items.append(
                        SemanticItem(
                            "POTENTIAL_PATTERN",
                            "USE_AFTER_FREE",
                            f"object={deallocation.object_name} free_node={deallocation.node_id} use_node={node_id} operation={operation.kind}",
                        )
                    )
    return items


def _potential_pattern_items(
    graph: FunctionGraph,
    operations: list[_Operation],
    guards: dict[str, list[tuple[str, str]]],
    capacities: dict[str, str],
) -> list[SemanticItem]:
    items: list[SemanticItem] = []
    for operation in operations:
        if operation.kind == "MEMORY_WRITE":
            api_name, _ = _call_name_and_args(operation.code)
            if api_name in _UNBOUNDED_WRITE_APIS:
                items.append(
                    SemanticItem(
                        "POTENTIAL_PATTERN",
                        "UNBOUNDED_WRITE",
                        f"api={api_name} object={operation.object_name or '?'}",
                    )
                )
            if (
                operation.extent
                and not _has_matching_guard(guards, operation.node_id, operation.extent)
                and _identifiers(operation.extent)
            ):
                items.append(
                    SemanticItem(
                        "POTENTIAL_PATTERN",
                        "UNCHECKED_WRITE_EXTENT",
                        f"object={operation.object_name or '?'} extent={_compact(operation.extent)}",
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
                    SemanticItem(
                        "POTENTIAL_PATTERN",
                        "WRITE_EXCEEDS_STATIC_CAPACITY",
                        f"object={operation.object_name} extent={extent_value} capacity={capacity_value}",
                    )
                )
        elif operation.kind == "ARRAY_ACCESS" and operation.extent:
            if not _has_matching_guard(guards, operation.node_id, operation.extent):
                items.append(
                    SemanticItem(
                        "POTENTIAL_PATTERN",
                        "UNCHECKED_ARRAY_INDEX",
                        f"object={operation.object_name or '?'} index={_compact(operation.extent)}",
                    )
                )
        elif operation.kind == "POINTER_DEREFERENCE" and operation.object_name:
            if not _has_null_guard(guards, operation.node_id, operation.object_name):
                items.append(
                    SemanticItem(
                        "POTENTIAL_PATTERN",
                        "UNCHECKED_POINTER_DEREFERENCE",
                        f"pointer={operation.object_name}",
                    )
                )

    for edge in graph.edges:
        if edge.kind != "DDG":
            continue
        source = graph.nodes.get(edge.source)
        target = graph.nodes.get(edge.target)
        if source is None or target is None or not _ARITHMETIC.search(source.code):
            continue
        target_operations = _node_operations(target)
        if any(
            operation.kind in {"MEMORY_WRITE", "ALLOCATION", "ARRAY_ACCESS"}
            for operation in target_operations
        ) and not _has_matching_guard(guards, target.node_id, source.code):
            items.append(
                SemanticItem(
                    "POTENTIAL_PATTERN",
                    "UNCHECKED_SIZE_ARITHMETIC",
                    f"expr={_compact(source.code, 100)} sink={_compact(target.code, 100)}",
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
    guards = _controlling_guards(graph)
    capacities = _capacity_facts(graph)

    items: list[SemanticItem] = []
    items.extend(_operation_items(operations))
    for name, capacity in capacities.items():
        items.append(
            SemanticItem(
                "MEMORY_OBJECT",
                "STATIC_CAPACITY",
                f"object={name} capacity={_compact(capacity)}",
            )
        )
    items.extend(_dependence_items(graph, operations_by_node))
    items.extend(_constraint_items(graph, guards, operations_by_node))
    items.extend(_lifetime_items(graph, operations, operations_by_node))
    items.extend(_potential_pattern_items(graph, operations, guards, capacities))

    deduplicated: list[SemanticItem] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (item.category, item.kind, item.detail)
        if key not in seen:
            seen.add(key)
            deduplicated.append(item)
    return VulnerabilitySemantics(tuple(deduplicated))
