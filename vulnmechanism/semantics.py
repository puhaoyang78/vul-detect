from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

from .cpg import FunctionGraph, GraphNode


MECHANISM_GROUPS: dict[str, tuple[str, ...]] = {
    "relation": ("MECHANISM_RELATION",),
    "mechanism": ("MECHANISM_CANDIDATE",),
}
CATEGORY_TO_GROUP = {
    category: group
    for group, categories in MECHANISM_GROUPS.items()
    for category in categories
}

# Descriptive metadata only. These values are not auxiliary supervision labels.
MECHANISM_FEATURES = (
    "bounds_flow",
    "null_dereference_flow",
    "use_after_free_flow",
    "double_free_flow",
    "size_arithmetic_flow",
    "known_bounds_violation",
)

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
# For these APIs the size argument is a destination-bound contract, not a
# direct byte-count claim. Without an independent capacity it is not a bounds
# mechanism by itself.
_DESTINATION_BOUND_APIS = {"snprintf", "vsnprintf", "strlcpy", "strlcat"}

_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")
_SIMPLE_IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")
_ARRAY_ACCESS = re.compile(r"\b([A-Za-z_]\w*(?:->\w+|\.\w+)*)\s*\[\s*([^\]]+?)\s*\]")
_ARRAY_DECL = re.compile(r"\b([A-Za-z_]\w*)\s*\[\s*([A-Za-z_0-9()+\-*/<>&| ]+)\s*\]")
_ASSIGN_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*(?=[A-Za-z_]\w*\s*\()")
_NEW_ASSIGNMENT = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*new\b")
_DEREF = re.compile(r"(?<![\w)])\*\s*([A-Za-z_]\w*)")
_ARROW_BASE = re.compile(
    r"\b([A-Za-z_]\w*(?:\s*->\s*[A-Za-z_]\w*)*)\s*->\s*[A-Za-z_]\w*"
)
_DELETE = re.compile(r"\bdelete(?:\s*\[\s*\])?\s+([A-Za-z_]\w*)")
_NEW_EXTENT = re.compile(r"\bnew\b[^;\[]*\[\s*([^\]]+)\s*\]")
_INTEGER = re.compile(r"^(?:0[xX][0-9A-Fa-f]+|\d+)[uUlL]*$")
_SIZEOF = re.compile(r"\bsizeof\s*(?:\([^()]*\)|[A-Za-z_]\w*)")
_SIMPLE_BINARY_ARITH = re.compile(
    r"\b([A-Za-z_]\w*(?:->\w+|\.\w+)*)\s*(<<|>>|\+|-|\*)\s*"
    r"([A-Za-z_]\w*(?:->\w+|\.\w+)*)\b"
)
_ARITHMETIC_LABEL_PARTS = (
    "<operator>.addition", "<operator>.subtraction", "<operator>.multiplication",
    "<operator>.division", "<operator>.modulo", "<operator>.shiftleft",
    "<operator>.shiftright", "<operator>.assignmentplus",
    "<operator>.assignmentminus", "<operator>.assignmentmultiplication",
)
_CONTROL_PREFIXES = ("if ", "if(", "while ", "while(", "for ", "for(", "switch ", "switch(")
_TYPE_WORDS = {
    "const", "volatile", "restrict", "signed", "unsigned", "short", "long", "void",
    "char", "int", "float", "double", "bool", "struct", "class", "enum", "union",
    "size_t", "ssize_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int8_t", "int16_t", "int32_t", "int64_t", "auto", "static",
}


@dataclass(frozen=True)
class SemanticItem:
    category: str
    kind: str
    detail: str
    key: str | None = None
    state: str | None = None

    def as_json(self) -> dict[str, str]:
        value = {"category": self.category, "kind": self.kind, "detail": self.detail}
        if self.key:
            value["key"] = self.key
        if self.state:
            value["state"] = self.state
        return value


@dataclass(frozen=True)
class MechanismSemantics:
    items: tuple[SemanticItem, ...]

    @property
    def candidate_count(self) -> int:
        return sum(item.category == "MECHANISM_CANDIDATE" for item in self.items)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return mechanism_features_from_items(item.as_json() for item in self.items)

    def render(self, excluded_groups: tuple[str, ...] = (), max_per_category: int = 24) -> str:
        return render_mechanism_items(
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
    api_name: str | None = None


def validate_mechanism_groups(groups: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for group in groups:
        value = group.strip().lower()
        if not value:
            continue
        if value not in MECHANISM_GROUPS:
            raise ValueError(
                f"unknown mechanism group {group!r}; expected one of {', '.join(MECHANISM_GROUPS)}"
            )
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _round_robin_by_kind(rows: list[tuple[str, str]], limit: int) -> list[str]:
    grouped: dict[str, list[str]] = {}
    for kind, text in rows:
        values = grouped.setdefault(kind, [])
        if text not in values:
            values.append(text)
    selected: list[str] = []
    while len(selected) < limit:
        added = False
        for kind in sorted(grouped):
            if grouped[kind] and len(selected) < limit:
                selected.append(grouped[kind].pop(0))
                added = True
        if not added:
            break
    return selected


def render_mechanism_items(
    items: list[dict[str, object]] | tuple[dict[str, object], ...],
    *,
    excluded_groups: tuple[str, ...] = (),
    max_per_category: int = 24,
) -> str:
    """Render CPG-supported mechanism candidates and high-level relations.

    SECURITY_OPERATION remains audit-only. Relations are model-facing even when
    they do not justify a candidate; this preserves useful type/data/control
    semantics without pretending that every relation is itself a vulnerability.
    """
    if max_per_category <= 0:
        raise ValueError("max_per_category must be positive")
    excluded = set(validate_mechanism_groups(excluded_groups))
    grouped: dict[str, list[tuple[str, str]]] = {}
    for item in items:
        category = str(item.get("category") or "")
        kind = str(item.get("kind") or "")
        detail = str(item.get("detail") or "")
        group = CATEGORY_TO_GROUP.get(category)
        if not category or not kind or group is None or group in excluded:
            continue
        grouped.setdefault(category, []).append((kind, f"{kind} {detail}".strip()))

    if not grouped:
        return "[VULNERABILITY_MECHANISM_CONTEXT]\nNO_CPG_DERIVED_MECHANISM_EVIDENCE"

    rows: list[str] = []
    for category in ("MECHANISM_CANDIDATE", "MECHANISM_RELATION"):
        values = grouped.get(category)
        if not values:
            continue
        rows.append(f"[{category}]")
        rows.extend(_round_robin_by_kind(sorted(set(values)), max_per_category))
    return "\n".join(rows)


def mechanism_features_from_items(items) -> tuple[str, ...]:
    features: set[str] = set()
    for item in items:
        if item.get("category") != "MECHANISM_CANDIDATE":
            continue
        kind = str(item.get("kind") or "")
        state = str(item.get("state") or "")
        if kind == "BOUNDS_FLOW":
            features.add("bounds_flow")
            if "static_violation=yes" in state:
                features.add("known_bounds_violation")
        elif kind == "NULL_DEREFERENCE_FLOW":
            features.add("null_dereference_flow")
        elif kind == "USE_AFTER_FREE_FLOW":
            features.add("use_after_free_flow")
        elif kind == "DOUBLE_FREE_FLOW":
            features.add("double_free_flow")
        elif kind == "SIZE_ARITHMETIC_FLOW":
            features.add("size_arithmetic_flow")
    return tuple(feature for feature in MECHANISM_FEATURES if feature in features)


def _compact(text: str, limit: int = 140) -> str:
    value = " ".join(text.split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _canonical(text: str | None) -> str:
    return re.sub(r"\s+", "", text or "")


def _runtime_identifiers(expression: str | None) -> set[str]:
    if not expression:
        return set()
    stripped = _SIZEOF.sub("", expression)
    return {
        identifier
        for identifier in _IDENTIFIER.findall(stripped)
        if identifier not in _TYPE_WORDS and identifier != "sizeof"
    }


def _parameter_name(code: str) -> str | None:
    identifiers = _IDENTIFIER.findall(code)
    for identifier in reversed(identifiers):
        if identifier not in _TYPE_WORDS:
            return identifier
    return identifiers[-1] if identifiers else None


def _declared_types(graph: FunctionGraph) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in graph.nodes.values():
        if node.label.upper() not in {"PARAM", "LOCAL"}:
            continue
        name = _parameter_name(node.code)
        if not name:
            continue
        match = re.search(rf"^(.*?)\b{re.escape(name)}\b(?:\s*\[.*\])?\s*$", node.code.strip())
        if match:
            type_text = " ".join(match.group(1).split()).strip()
            if type_text:
                result.setdefault(name, type_text)
    return result


def _call_name_and_args(code: str) -> tuple[str | None, tuple[str, ...]]:
    match = re.search(r"\b([A-Za-z_]\w*)\s*\(", code)
    if not match:
        return None, ()
    name = match.group(1)
    start = match.end()
    depth = 1
    index = start
    while index < len(code) and depth:
        if code[index] == "(":
            depth += 1
        elif code[index] == ")":
            depth -= 1
        index += 1
    if depth:
        return name, ()
    body = code[start:index - 1]
    args: list[str] = []
    current: list[str] = []
    nested = 0
    for char in body:
        if char in "([{":
            nested += 1
        elif char in ")]}":
            nested = max(0, nested - 1)
        if char == "," and nested == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current or body.strip():
        args.append("".join(current).strip())
    return name, tuple(args)


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


def _arrow_base(code: str) -> str | None:
    match = _ARROW_BASE.search(code)
    return re.sub(r"\s*->\s*", "->", match.group(1)) if match else None


def _node_operations(node: GraphNode) -> tuple[_Operation, ...]:
    if node.label in {
        "METHOD", "METHOD_RETURN", "BLOCK", "LOCAL", "PARAM", "IDENTIFIER",
        "FIELD_IDENTIFIER", "LITERAL", "TYPE_REF", "UNKNOWN", "CONTROL_STRUCTURE",
    }:
        return ()

    operations: list[_Operation] = []
    name, args = _call_name_and_args(node.code)
    if name in _WRITE_APIS:
        object_index, extent_index = _WRITE_APIS[name]
        extent = (
            f"({args[1]})*({args[2]})"
            if name == "fread" and len(args) >= 3
            else _argument(args, extent_index)
        )
        operations.append(_Operation(
            node.node_id, "MEMORY_WRITE", _argument(args, object_index), extent, node.code, name
        ))
    if name in _READ_APIS:
        object_index, extent_index = _READ_APIS[name]
        extent = (
            f"({args[1]})*({args[2]})"
            if name == "fwrite" and len(args) >= 3
            else _argument(args, extent_index)
        )
        operations.append(_Operation(
            node.node_id, "MEMORY_READ", _argument(args, object_index), extent, node.code, name
        ))
    if name in _ALLOC_APIS:
        assigned = _ASSIGN_CALL.search(node.code)
        operations.append(_Operation(
            node.node_id,
            "ALLOCATION",
            assigned.group(1) if assigned else None,
            _allocation_extent(name, args),
            node.code,
            name,
        ))
    if name in _FREE_APIS:
        operations.append(_Operation(
            node.node_id, "DEALLOCATION", _argument(args, 0), None, node.code, name
        ))

    lower = node.label.lower()
    if "<operator>.new" in lower or lower.endswith(".new"):
        extent = _NEW_EXTENT.search(node.code)
        assigned = _NEW_ASSIGNMENT.search(node.code)
        operations.append(_Operation(
            node.node_id,
            "ALLOCATION",
            assigned.group(1) if assigned else None,
            extent.group(1).strip() if extent else None,
            node.code,
            "new",
        ))
    if "<operator>.delete" in lower or lower.endswith(".delete"):
        deleted = _DELETE.search(node.code)
        operations.append(_Operation(
            node.node_id,
            "DEALLOCATION",
            deleted.group(1) if deleted else None,
            None,
            node.code,
            "delete",
        ))
    if "indexaccess" in lower:
        for base, index in _ARRAY_ACCESS.findall(node.code):
            operations.append(_Operation(
                node.node_id, "ARRAY_ACCESS", base, index.strip(), node.code
            ))
    if "indirection" in lower:
        for pointer in sorted(set(_DEREF.findall(node.code))):
            operations.append(_Operation(
                node.node_id, "POINTER_DEREFERENCE", pointer, None, node.code
            ))
    if "fieldaccess" in lower:
        pointer = _arrow_base(node.code)
        if pointer:
            operations.append(_Operation(
                node.node_id, "POINTER_DEREFERENCE", pointer, None, node.code
            ))

    dedup: dict[tuple[str, str | None, str | None], _Operation] = {}
    for operation in operations:
        dedup[(operation.kind, operation.object_name, operation.extent)] = operation
    return tuple(dedup.values())


def _ast_children(graph: FunctionGraph) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind == "AST":
            children.setdefault(edge.source, []).append(edge.target)
    return children


def _ast_parents(graph: FunctionGraph) -> dict[str, list[str]]:
    parents: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind == "AST":
            parents.setdefault(edge.target, []).append(edge.source)
    return parents


def _ast_descendants(graph: FunctionGraph, root_id: str, max_depth: int = 8) -> tuple[GraphNode, ...]:
    children = _ast_children(graph)
    queue: deque[tuple[str, int]] = deque([(root_id, 0)])
    seen = {root_id}
    result: list[GraphNode] = []
    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for child_id in children.get(current, ()):
            if child_id in seen:
                continue
            seen.add(child_id)
            child = graph.nodes.get(child_id)
            if child:
                result.append(child)
            queue.append((child_id, depth + 1))
    return tuple(result)


def _expression_nodes(
    graph: FunctionGraph,
    operation_node_id: str,
    expression: str | None,
) -> tuple[GraphNode, ...]:
    wanted = _canonical(expression)
    if not wanted:
        return ()
    nodes: list[GraphNode] = []
    root = graph.nodes.get(operation_node_id)
    if root:
        nodes.append(root)
    nodes.extend(_ast_descendants(graph, operation_node_id))
    exact = [node for node in nodes if _canonical(node.code) == wanted]
    if exact:
        return tuple(exact)
    if _SIMPLE_IDENTIFIER.fullmatch(expression or ""):
        return tuple(
            node for node in nodes
            if node.label.upper() == "IDENTIFIER" and node.code.strip() == expression.strip()
        )
    return ()


def _ddg_reverse(graph: FunctionGraph) -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind == "DDG":
            reverse.setdefault(edge.target, []).append(edge.source)
    return reverse


def _ddg_upstream_from_ids(
    graph: FunctionGraph,
    root_ids: tuple[str, ...] | list[str],
    *,
    max_depth: int = 6,
    max_nodes: int = 96,
) -> tuple[GraphNode, ...]:
    reverse = _ddg_reverse(graph)
    queue: deque[tuple[str, int]] = deque((root_id, 0) for root_id in root_ids)
    seen = set(root_ids)
    result: list[GraphNode] = []
    while queue and len(result) < max_nodes:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for source_id in reverse.get(current, ()):
            if source_id in seen:
                continue
            seen.add(source_id)
            source = graph.nodes.get(source_id)
            if source:
                result.append(source)
            queue.append((source_id, depth + 1))
    return tuple(result)


def _parameter_names(graph: FunctionGraph) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in graph.nodes.values():
        if node.label.upper() != "PARAM":
            continue
        name = _parameter_name(node.code)
        if name:
            result[name] = node.node_id
    return result


def _parameter_sources(
    graph: FunctionGraph,
    operation_node_id: str,
    expression: str | None,
) -> tuple[str, ...]:
    runtime_names = _runtime_identifiers(expression)
    if not runtime_names:
        return ()
    parameters = _parameter_names(graph)
    sources = set(runtime_names & parameters.keys())
    roots = _expression_nodes(graph, operation_node_id, expression)
    if not roots:
        return tuple(sorted(sources))
    upstream = _ddg_upstream_from_ids(graph, [node.node_id for node in roots])
    for node in upstream:
        if node.label.upper() == "PARAM":
            name = _parameter_name(node.code)
            if name:
                sources.add(name)
    return tuple(sorted(sources))


def _is_arithmetic_node(node: GraphNode) -> bool:
    lower = node.label.lower()
    return any(part.lower() in lower for part in _ARITHMETIC_LABEL_PARTS)


def _arithmetic_sources(
    graph: FunctionGraph,
    operation_node_id: str,
    expression: str | None,
) -> tuple[GraphNode, ...]:
    if not _runtime_identifiers(expression):
        return ()
    roots = _expression_nodes(graph, operation_node_id, expression)
    if not roots:
        return ()
    nodes: list[GraphNode] = []
    for root in roots:
        nodes.append(root)
        nodes.extend(_ast_descendants(graph, root.node_id, max_depth=5))
    nodes.extend(_ddg_upstream_from_ids(graph, [root.node_id for root in roots]))
    result: list[GraphNode] = []
    seen: set[str] = set()
    for node in nodes:
        if node.node_id in seen or not _is_arithmetic_node(node):
            continue
        seen.add(node.node_id)
        if _runtime_identifiers(node.code):
            result.append(node)
    return tuple(result[:6])


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
                    return code[start + 1:index].strip()
    if any(part in node.label.lower() for part in ("lessthan", "greaterthan", "equals", "notequals")):
        return code
    return code if any(operator in code for operator in ("<=", ">=", "==", "!=", "<", ">")) else None


def _direct_conditions(graph: FunctionGraph) -> dict[str, list[tuple[str, str]]]:
    result: dict[str, list[tuple[str, str]]] = {}
    for edge in graph.edges:
        if edge.kind != "CDG":
            continue
        source = graph.nodes.get(edge.source)
        expression = _control_expression(source) if source else None
        if expression:
            value = (edge.source, expression)
            if value not in result.setdefault(edge.target, []):
                result[edge.target].append(value)
    return result


def _related_conditions(
    node_id: str,
    parents: dict[str, list[str]],
    direct: dict[str, list[tuple[str, str]]],
) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    queue: deque[tuple[str, int]] = deque([(node_id, 0)])
    seen = {node_id}
    while queue:
        current, depth = queue.popleft()
        for condition in direct.get(current, ()):
            if condition not in result:
                result.append(condition)
        if depth >= 12:
            continue
        for parent in parents.get(current, ()):
            if parent not in seen:
                seen.add(parent)
                queue.append((parent, depth + 1))
    return tuple(result)


def _static_integer(expression: str | None) -> int | None:
    if not expression or not _INTEGER.match(expression.strip()):
        return None
    value = re.sub(r"[uUlL]+$", "", expression.strip())
    try:
        return int(value, 0)
    except ValueError:
        return None


def _split_comparison(condition: str) -> tuple[str, str, str] | None:
    depth = 0
    for index, char in enumerate(condition):
        if char in "([{":
            depth += 1
        elif char in ")]}":
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
    comparison = _split_comparison(condition)
    names = _runtime_identifiers(expression)
    if comparison is None or not names:
        return False
    left, operator, right = comparison
    left_names = _runtime_identifiers(left)
    right_names = _runtime_identifiers(right)
    if operator in {"<", "<="}:
        return bool(names & left_names) and bool(right_names or _static_integer(right) is not None)
    return bool(names & right_names) and bool(left_names or _static_integer(left) is not None)


def _condition_state(
    graph: FunctionGraph,
    conditions: tuple[tuple[str, str], ...],
    predicate,
) -> str:
    if any(predicate(expression) for _, expression in conditions):
        return "present"
    if not any(edge.kind == "CDG" for edge in graph.edges):
        return "unknown_no_cdg"
    return "not_observed"


def _all_control_conditions(graph: FunctionGraph) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for node in graph.nodes.values():
        expression = _control_expression(node)
        if expression:
            value = (node.node_id, expression)
            if value not in result:
                result.append(value)
    return tuple(result)


def _operand_guard_present(
    graph: FunctionGraph,
    operand_names: set[str],
    *,
    exclude_node_id: str,
    arithmetic_expression: str,
) -> bool:
    if len(operand_names) < 2:
        return False
    arithmetic_canonical = _canonical(arithmetic_expression)
    for node_id, condition in _all_control_conditions(graph):
        if node_id == exclude_node_id or arithmetic_canonical in _canonical(condition):
            continue
        if _split_comparison(condition) is None:
            continue
        mentioned = _runtime_identifiers(condition)
        if len(operand_names & mentioned) >= 2:
            return True
    return False


def _null_related_condition(condition: str, pointer: str) -> bool:
    escaped = re.escape(pointer)
    return bool(
        re.search(rf"\b{escaped}\b\s*(?:!=|==)\s*(?:NULL|nullptr|0)\b", condition)
        or re.search(rf"(?:NULL|nullptr|0)\s*(?:!=|==)\s*\b{escaped}\b", condition)
        or re.search(rf"!\s*\b{escaped}\b", condition)
        or re.fullmatch(rf"\s*!!?\s*\b{escaped}\b\s*", condition)
        or re.fullmatch(rf"\s*\b{escaped}\b\s*", condition)
    )


def _capacities(operations: list[_Operation], graph: FunctionGraph) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in graph.nodes.values():
        if node.label.upper() != "LOCAL":
            continue
        for name, capacity in _ARRAY_DECL.findall(node.code):
            value = _compact(capacity, 80)
            if value:
                result.setdefault(name, value)
    dynamic: dict[str, set[str]] = {}
    for operation in operations:
        if operation.kind == "ALLOCATION" and operation.object_name and operation.extent:
            dynamic.setdefault(operation.object_name, set()).add(_compact(operation.extent, 80))
    for name, values in dynamic.items():
        if name not in result and len(values) == 1:
            result[name] = next(iter(values))
    return result


def _candidate_key(kind: str, *parts: str | None) -> str:
    return "|".join([kind, *(_compact(part or "?", 80) for part in parts)])


def _redefines_name(code: str, name: str) -> bool:
    escaped = re.escape(name)
    return bool(re.search(rf"\b{escaped}\b\s*(?:=(?!=)|\+=|-=)", code))


def _null_assignment(code: str, pointer: str) -> bool:
    escaped = re.escape(pointer)
    return bool(re.search(rf"\b{escaped}\b\s*=\s*(?:NULL|nullptr|0)\b", code))


def _nullable_origins(
    graph: FunctionGraph,
    sink_id: str,
    pointer: str,
    operations: list[_Operation],
) -> tuple[str, ...]:
    if not _SIMPLE_IDENTIFIER.fullmatch(pointer):
        return ()
    roots = _expression_nodes(graph, sink_id, pointer)
    if not roots:
        return ()
    upstream = _ddg_upstream_from_ids(graph, [node.node_id for node in roots])
    upstream_ids = {node.node_id for node in upstream}
    origins: set[str] = set()
    for operation in operations:
        if (
            operation.node_id in upstream_ids
            and operation.kind == "ALLOCATION"
            and operation.object_name == pointer
            and operation.api_name in _ALLOC_APIS
        ):
            origins.add(f"nullable_allocation:{operation.api_name}")
    for node in upstream:
        if _null_assignment(node.code, pointer):
            origins.add("explicit_null_assignment")
        assigned = _ASSIGN_CALL.search(node.code)
        api, _ = _call_name_and_args(node.code)
        if assigned and assigned.group(1) == pointer and api in _ALLOC_APIS:
            origins.add(f"nullable_allocation:{api}")
    return tuple(sorted(origins))


def _security_operation_items(operations: list[_Operation]) -> list[SemanticItem]:
    result: list[SemanticItem] = []
    for operation in operations:
        fields = [f"sink={operation.kind}", f"code={_compact(operation.code, 100)}"]
        if operation.object_name:
            fields.append(f"object={_compact(operation.object_name, 60)}")
        if operation.extent:
            fields.append(
                f"{'index' if operation.kind == 'ARRAY_ACCESS' else 'extent'}="
                f"{_compact(operation.extent, 60)}"
            )
        result.append(SemanticItem("SECURITY_OPERATION", operation.kind, " ".join(fields)))
    return result


def _mechanism_items(
    graph: FunctionGraph,
    operations: list[_Operation],
    by_node: dict[str, tuple[_Operation, ...]],
) -> tuple[list[SemanticItem], list[SemanticItem]]:
    parents = _ast_parents(graph)
    direct_conditions = _direct_conditions(graph)
    capacities = _capacities(operations, graph)
    declared_types = _declared_types(graph)
    candidate_rows: dict[tuple[str, str], dict[str, object]] = {}
    relations: list[SemanticItem] = []

    def add_candidate(
        kind: str,
        key: str,
        *,
        source: str,
        sink: str,
        obj: str,
        expression: str = "",
        condition: str = "unknown",
        capacity: str = "",
        value_type: str = "",
        violation: bool = False,
    ) -> None:
        row = candidate_rows.setdefault((kind, key), {
            "sources": set(), "sinks": set(), "objects": set(), "expressions": set(),
            "conditions": set(), "capacities": set(), "types": set(),
            "violation": False, "occurrences": 0,
        })
        row["sources"].add(source or "unknown")
        row["sinks"].add(sink)
        row["objects"].add(obj or "?")
        if expression:
            row["expressions"].add(_compact(expression, 70))
        if capacity:
            row["capacities"].add(_compact(capacity, 70))
        if value_type:
            row["types"].add(_compact(value_type, 50))
        row["conditions"].add(condition)
        row["violation"] = bool(row["violation"]) or violation
        row["occurrences"] = int(row["occurrences"]) + 1

    for operation in operations:
        conditions = _related_conditions(operation.node_id, parents, direct_conditions)

        if operation.kind in {"MEMORY_WRITE", "ARRAY_ACCESS"} and operation.extent:
            capacity = capacities.get(operation.object_name or "")
            parameters = _parameter_sources(graph, operation.node_id, operation.extent)
            source = "parameter:" + ",".join(parameters) if parameters else "unknown"
            value_type = (
                declared_types.get(operation.extent.strip(), "")
                if _SIMPLE_IDENTIFIER.fullmatch(operation.extent.strip())
                else ""
            )
            relation_evidence = bool(parameters or capacity)
            if operation.api_name in _DESTINATION_BOUND_APIS and not capacity:
                relation_evidence = False

            if relation_evidence:
                condition_state = _condition_state(
                    graph,
                    conditions,
                    lambda condition: _upper_bound_condition(condition, operation.extent or ""),
                )
                value = _static_integer(operation.extent)
                cap = _static_integer(capacity)
                violation = bool(
                    value is not None
                    and cap is not None
                    and (value >= cap if operation.kind == "ARRAY_ACCESS" else value > cap)
                )
                statically_safe = bool(
                    value is not None and cap is not None and not violation
                )
                key = _candidate_key(
                    "BOUNDS_FLOW", operation.object_name, source, operation.kind
                )
                if capacity:
                    relation_kind = "BOUND_RELATION"
                elif operation.kind == "ARRAY_ACCESS":
                    relation_kind = "INDEX_FLOW_TO_MEMORY_SINK"
                else:
                    relation_kind = "EXTENT_FLOW_TO_MEMORY_SINK"
                relation_detail = (
                    f"evidence={'DDG' if parameters else 'CAPACITY'} source={source} "
                    f"sink={operation.kind} object={operation.object_name or '?'} "
                    f"expression={_compact(operation.extent, 60)} capacity={capacity or '?'} "
                    f"bound_related_condition={condition_state}"
                )
                if value_type:
                    relation_detail += f" value_type={value_type}"
                relations.append(SemanticItem(
                    "MECHANISM_RELATION", relation_kind, relation_detail,
                    key=key,
                    state=(
                        f"bound_condition={condition_state}|"
                        f"static_violation={'yes' if violation else 'no'}"
                    ),
                ))
                # A bounds candidate requires an actual object-capacity relation.
                # Missing a check alone is never enough to create the candidate.
                if capacity and not statically_safe:
                    add_candidate(
                        "BOUNDS_FLOW", key,
                        source=source, sink=operation.kind, obj=operation.object_name or "?",
                        expression=operation.extent, condition=condition_state,
                        capacity=capacity, value_type=value_type, violation=violation,
                    )

            for arithmetic_node in _arithmetic_sources(
                graph, operation.node_id, operation.extent
            ):
                arithmetic_parameters = _parameter_sources(
                    graph, arithmetic_node.node_id, arithmetic_node.code
                )
                if not arithmetic_parameters:
                    continue
                arithmetic_state = _condition_state(
                    graph,
                    conditions,
                    lambda condition: _upper_bound_condition(condition, arithmetic_node.code),
                )
                arithmetic_source = "parameter:" + ",".join(arithmetic_parameters)
                key = _candidate_key(
                    "SIZE_ARITHMETIC_FLOW", "data", arithmetic_source,
                    operation.kind, operation.object_name, arithmetic_node.code,
                )
                relations.append(SemanticItem(
                    "MECHANISM_RELATION", "ARITHMETIC_TO_MEMORY_SINK",
                    f"evidence=DDG source={arithmetic_source} "
                    f"expr={_compact(arithmetic_node.code, 80)} sink={operation.kind} "
                    f"range_related_condition={arithmetic_state}",
                    key=key,
                    state=f"range_condition={arithmetic_state}",
                ))
                # The arithmetic-to-sink relation itself defines this mechanism;
                # the observed constraint is an annotation, not the trigger.
                add_candidate(
                    "SIZE_ARITHMETIC_FLOW", key,
                    source=arithmetic_source, sink=operation.kind,
                    obj=operation.object_name or "?", expression=arithmetic_node.code,
                    condition=arithmetic_state,
                )

            if operation.kind == "ARRAY_ACCESS":
                for condition_node_id, condition in conditions:
                    for match in _SIMPLE_BINARY_ARITH.finditer(condition):
                        expression = match.group(0)
                        operands = {
                            match.group(1).split("->", 1)[0].split(".", 1)[0],
                            match.group(3).split("->", 1)[0].split(".", 1)[0],
                        }
                        control_sources = _parameter_sources(
                            graph, condition_node_id, expression
                        )
                        if not control_sources:
                            control_sources = tuple(sorted(
                                operands & _parameter_names(graph).keys()
                            ))
                        if not control_sources:
                            continue
                        guard_present = _operand_guard_present(
                            graph,
                            operands,
                            exclude_node_id=condition_node_id,
                            arithmetic_expression=expression,
                        )
                        range_state = "present" if guard_present else "not_observed"
                        control_source = "parameter:" + ",".join(control_sources)
                        key = _candidate_key(
                            "SIZE_ARITHMETIC_FLOW", "control", control_source,
                            operation.object_name, expression,
                        )
                        relations.append(SemanticItem(
                            "MECHANISM_RELATION", "ARITHMETIC_CONTROL_TO_MEMORY_SINK",
                            f"evidence=CDG source={control_source} "
                            f"expr={_compact(expression, 80)} sink=ARRAY_ACCESS "
                            f"object={operation.object_name or '?'} "
                            f"operand_range_condition={range_state}",
                            key=key,
                            state=f"range_condition={range_state}",
                        ))
                        add_candidate(
                            "SIZE_ARITHMETIC_FLOW", key,
                            source=control_source, sink="ARRAY_ACCESS",
                            obj=operation.object_name or "?", expression=expression,
                            condition=range_state,
                        )

        elif operation.kind == "ALLOCATION" and operation.extent:
            for arithmetic_node in _arithmetic_sources(
                graph, operation.node_id, operation.extent
            ):
                arithmetic_parameters = _parameter_sources(
                    graph, arithmetic_node.node_id, arithmetic_node.code
                )
                if not arithmetic_parameters:
                    continue
                state = _condition_state(
                    graph,
                    conditions,
                    lambda condition: _upper_bound_condition(condition, arithmetic_node.code),
                )
                source = "parameter:" + ",".join(arithmetic_parameters)
                key = _candidate_key(
                    "SIZE_ARITHMETIC_FLOW", "allocation", source,
                    operation.object_name, arithmetic_node.code,
                )
                relations.append(SemanticItem(
                    "MECHANISM_RELATION", "ARITHMETIC_TO_ALLOCATION",
                    f"evidence=DDG source={source} expr={_compact(arithmetic_node.code, 80)} "
                    f"object={operation.object_name or '?'} range_related_condition={state}",
                    key=key,
                    state=f"range_condition={state}",
                ))
                add_candidate(
                    "SIZE_ARITHMETIC_FLOW", key,
                    source=source, sink="ALLOCATION", obj=operation.object_name or "?",
                    expression=arithmetic_node.code, condition=state,
                )

        elif operation.kind == "POINTER_DEREFERENCE" and operation.object_name:
            origins = _nullable_origins(
                graph, operation.node_id, operation.object_name, operations
            )
            if origins:
                source = ",".join(origins)
                state = _condition_state(
                    graph,
                    conditions,
                    lambda condition: _null_related_condition(
                        condition, operation.object_name or ""
                    ),
                )
                key = _candidate_key(
                    "NULL_DEREFERENCE_FLOW", operation.object_name, source
                )
                relations.append(SemanticItem(
                    "MECHANISM_RELATION", "NULLABLE_SOURCE_TO_DEREFERENCE",
                    f"evidence=DDG source={source} pointer={operation.object_name} "
                    f"null_related_condition={state}",
                    key=key,
                    state=f"null_condition={state}",
                ))
                # Explicit nullable provenance plus a dereference is the
                # mechanism. The null-related condition only annotates it.
                add_candidate(
                    "NULL_DEREFERENCE_FLOW", key,
                    source=source, sink="POINTER_DEREFERENCE",
                    obj=operation.object_name, condition=state,
                )

    cfg: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind == "CFG":
            cfg.setdefault(edge.source, []).append(edge.target)

    for first_free in (
        operation for operation in operations
        if operation.kind == "DEALLOCATION" and operation.object_name
    ):
        obj = first_free.object_name or ""
        queue: deque[str] = deque(cfg.get(first_free.node_id, ()))
        seen = {first_free.node_id}
        while queue:
            node_id = queue.popleft()
            if node_id in seen:
                continue
            seen.add(node_id)
            node = graph.nodes.get(node_id)
            if node and _redefines_name(node.code, obj):
                continue
            for later in by_node.get(node_id, ()):
                if later.object_name != obj:
                    continue
                if later.kind == "DEALLOCATION":
                    kind, relation, sink = "DOUBLE_FREE_FLOW", "FREE_TO_FREE", "DEALLOCATION"
                elif later.kind in {
                    "MEMORY_READ", "MEMORY_WRITE", "ARRAY_ACCESS", "POINTER_DEREFERENCE"
                }:
                    kind, relation, sink = "USE_AFTER_FREE_FLOW", "FREE_TO_USE", later.kind
                else:
                    continue
                key = _candidate_key(kind, obj)
                relations.append(SemanticItem(
                    "MECHANISM_RELATION", relation,
                    f"evidence=CFG object={obj} transition=DEALLOCATION->{sink} "
                    "path_without_redefinition=present",
                    key=key,
                    state="path_without_redefinition=present",
                ))
                add_candidate(
                    kind, key, source="deallocation", sink=sink, obj=obj,
                    condition="not_applicable",
                )
            queue.extend(cfg.get(node_id, ()))

    candidates: list[SemanticItem] = []
    for (kind, key), row in sorted(candidate_rows.items()):
        conditions = set(row["conditions"])
        condition = (
            next(iter(conditions)) if len(conditions) == 1
            else ("mixed" if conditions else "unknown")
        )
        source = ",".join(sorted(row["sources"])) or "unknown"
        sink = ",".join(sorted(row["sinks"])) or "?"
        obj = ",".join(sorted(row["objects"])) or "?"
        expression = ";".join(sorted(row["expressions"])) or "?"
        capacity = ";".join(sorted(row["capacities"])) or "?"
        value_type = ";".join(sorted(row["types"])) or "?"
        violation = bool(row["violation"])

        fields = [f"source={source}", f"sink={sink}", f"object={obj}"]
        if expression != "?":
            fields.append(f"expression={expression}")
        if capacity != "?":
            fields.append(f"capacity={capacity}")
        if value_type != "?":
            fields.append(f"value_type={value_type}")
        if kind == "BOUNDS_FLOW":
            fields.append(f"bound_related_condition={condition}")
            state = (
                f"bound_condition={condition}|"
                f"static_violation={'yes' if violation else 'no'}"
            )
        elif kind == "SIZE_ARITHMETIC_FLOW":
            fields.append(f"range_related_condition={condition}")
            state = f"range_condition={condition}"
        elif kind == "NULL_DEREFERENCE_FLOW":
            fields.append(f"null_related_condition={condition}")
            state = f"null_condition={condition}"
        else:
            fields.append("redefinition_on_path=none_observed")
            state = "path_without_redefinition=present"
        fields.append(f"occurrences={row['occurrences']}")
        if violation:
            fields.append("static_violation=yes")
        candidates.append(SemanticItem(
            "MECHANISM_CANDIDATE", kind, " ".join(fields), key=key, state=state
        ))
    return candidates, relations


def extract_mechanism_semantics(graph: FunctionGraph) -> MechanismSemantics:
    by_node = {
        node_id: operations
        for node_id, node in graph.nodes.items()
        if (operations := _node_operations(node))
    }
    operations = [operation for values in by_node.values() for operation in values]
    candidates, relations = _mechanism_items(graph, operations, by_node)
    # Raw operations are retained only for audit. They are not rendered into
    # the model-facing mechanism context.
    items = [*candidates, *relations, *_security_operation_items(operations)]
    dedup: list[SemanticItem] = []
    seen: set[tuple[str, str, str, str | None, str | None]] = set()
    for item in items:
        identity = (item.category, item.kind, item.detail, item.key, item.state)
        if identity not in seen:
            seen.add(identity)
            dedup.append(item)
    return MechanismSemantics(tuple(dedup))
