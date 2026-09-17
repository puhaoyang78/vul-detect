from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass

from .cpg import FunctionGraph, GraphNode


MECHANISM_GROUPS: dict[str, tuple[str, ...]] = {
    "operation": ("SECURITY_OPERATION",),
    "relation": ("MECHANISM_RELATION",),
    "mechanism": ("MECHANISM_CANDIDATE",),
}
CATEGORY_TO_GROUP = {
    category: group
    for group, categories in MECHANISM_GROUPS.items()
    for category in categories
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
_ARITHMETIC_LABEL_PARTS = (
    "addition", "subtraction", "multiplication", "division", "modulo",
    "shiftleft", "shiftright",
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
        result = {"category": self.category, "kind": self.kind, "detail": self.detail}
        if self.key:
            result["key"] = self.key
        if self.state:
            result["state"] = self.state
        return result


@dataclass(frozen=True)
class MechanismSemantics:
    items: tuple[SemanticItem, ...]

    @property
    def candidate_count(self) -> int:
        return sum(item.category == "MECHANISM_CANDIDATE" for item in self.items)

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
    groups: dict[str, list[str]] = {}
    for kind, text in rows:
        values = groups.setdefault(kind, [])
        if text not in values:
            values.append(text)
    selected: list[str] = []
    while len(selected) < limit:
        added = False
        for kind in sorted(groups):
            values = groups[kind]
            if values and len(selected) < limit:
                selected.append(values.pop(0))
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
    if max_per_category <= 0:
        raise ValueError("max_per_category must be positive")
    excluded = set(validate_mechanism_groups(excluded_groups))
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
        return "[VULNERABILITY_MECHANISM_CONTEXT]\nNO_CPG_DERIVED_MECHANISM_EVIDENCE"

    sections: list[str] = []
    for category in ("MECHANISM_CANDIDATE", "MECHANISM_RELATION", "SECURITY_OPERATION"):
        rows = grouped.get(category)
        if not rows:
            continue
        sections.append(f"[{category}]")
        sections.extend(_round_robin_by_kind(sorted(set(rows)), max_per_category))
    return "\n".join(sections)


def _compact(text: str, limit: int = 140) -> str:
    value = " ".join(text.split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _normalize(text: str | None) -> str:
    return " ".join((text or "").split())


def _identifiers(expression: str | None) -> set[str]:
    return set(_IDENTIFIER.findall(expression)) if expression else set()


def _parameter_name(code: str) -> str | None:
    identifiers = _IDENTIFIER.findall(code)
    for identifier in reversed(identifiers):
        if identifier not in _TYPE_WORDS:
            return identifier
    return identifiers[-1] if identifiers else None


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


def _is_arithmetic_node(node: GraphNode) -> bool:
    lower = node.label.lower()
    return any(part in lower for part in _ARITHMETIC_LABEL_PARTS) or bool(
        _BINARY_ARITHMETIC.search(node.code)
    )


def _node_operations(node: GraphNode) -> tuple[_Operation, ...]:
    # Container nodes contain descendant source text and must not be interpreted
    # as separate operations. Only native operation/CALL nodes become facts.
    if node.label in {
        "METHOD", "METHOD_RETURN", "BLOCK", "LOCAL", "PARAM", "IDENTIFIER",
        "FIELD_IDENTIFIER", "LITERAL", "TYPE_REF", "UNKNOWN", "CONTROL_STRUCTURE",
    }:
        return ()

    operations: list[_Operation] = []
    code = node.code
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
        assigned = _ASSIGNMENT.search(code)
        operations.append(_Operation(
            node.node_id, "ALLOCATION", assigned.group(1) if assigned else None,
            _allocation_extent(name, args), code,
        ))
    if name in _FREE_APIS:
        operations.append(_Operation(node.node_id, "DEALLOCATION", _argument(args, 0), None, code))

    lower = node.label.lower()
    if "<operator>.new" in lower or lower.endswith(".new"):
        extent = _NEW_EXTENT.search(code)
        assigned = _NEW_ASSIGNMENT.search(code)
        operations.append(_Operation(
            node.node_id, "ALLOCATION", assigned.group(1) if assigned else None,
            extent.group(1).strip() if extent else None, code,
        ))
    if "<operator>.delete" in lower or lower.endswith(".delete"):
        deleted = _DELETE.search(code)
        operations.append(_Operation(
            node.node_id, "DEALLOCATION", deleted.group(1) if deleted else None, None, code,
        ))

    if "indexaccess" in lower:
        for base, index in _ARRAY_ACCESS.findall(code):
            operations.append(_Operation(node.node_id, "ARRAY_ACCESS", base, index.strip(), code))
    if "indirection" in lower or "fieldaccess" in lower:
        pointers = set(_DEREF.findall(code)) | set(_ARROW.findall(code))
        for pointer in sorted(pointers):
            operations.append(_Operation(node.node_id, "POINTER_DEREFERENCE", pointer, None, code))

    deduplicated: dict[tuple[str, str | None, str | None], _Operation] = {}
    for operation in operations:
        deduplicated[(operation.kind, operation.object_name, operation.extent)] = operation
    return tuple(deduplicated.values())


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


def _ast_parents(graph: FunctionGraph) -> dict[str, list[str]]:
    parents: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind == "AST":
            parents.setdefault(edge.target, []).append(edge.source)
    return parents


def _direct_conditions(graph: FunctionGraph) -> dict[str, list[str]]:
    conditions: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind != "CDG":
            continue
        source = graph.nodes.get(edge.source)
        expression = _control_expression(source) if source else None
        if expression and expression not in conditions.setdefault(edge.target, []):
            conditions[edge.target].append(expression)
    return conditions


def _related_conditions(
    node_id: str,
    parents: dict[str, list[str]],
    direct: dict[str, list[str]],
) -> tuple[str, ...]:
    # Joern commonly attaches CDG to a containing statement while an array or
    # dereference operator is an AST descendant. Inherit such conditions through
    # AST ancestry only. Branch polarity is deliberately not inferred.
    result: list[str] = []
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


def _condition_state(
    graph: FunctionGraph,
    conditions: tuple[str, ...],
    predicate,
) -> str:
    if any(predicate(condition) for condition in conditions):
        return "present"
    if not any(edge.kind == "CDG" for edge in graph.edges):
        return "unknown_no_cdg"
    return "not_observed"


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
    names = _identifiers(expression)
    if comparison is None or not names:
        return False
    left, operator, right = comparison
    left_names, right_names = _identifiers(left), _identifiers(right)
    if operator in {"<", "<="}:
        return bool(names & left_names) and bool(right_names or _static_integer(right) is not None)
    return bool(names & right_names) and bool(left_names or _static_integer(left) is not None)


def _null_related_condition(condition: str, pointer: str) -> bool:
    escaped = re.escape(pointer)
    return bool(
        re.search(rf"\b{escaped}\b\s*(?:!=|==)\s*(?:NULL|nullptr|0)\b", condition)
        or re.search(rf"(?:NULL|nullptr|0)\s*(?:!=|==)\s*\b{escaped}\b", condition)
        or re.search(rf"!\s*\b{escaped}\b", condition)
        or re.fullmatch(rf"\s*!!?\s*\b{escaped}\b\s*", condition)
        or re.fullmatch(rf"\s*\b{escaped}\b\s*", condition)
    )


def _ddg_reverse(graph: FunctionGraph) -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind == "DDG":
            reverse.setdefault(edge.target, []).append(edge.source)
    return reverse


def _ddg_upstream(
    graph: FunctionGraph,
    sink_id: str,
    *,
    max_depth: int = 5,
    max_nodes: int = 64,
) -> tuple[GraphNode, ...]:
    reverse = _ddg_reverse(graph)
    queue: deque[tuple[str, int]] = deque([(sink_id, 0)])
    seen = {sink_id}
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


def _parameter_sources(graph: FunctionGraph, sink_id: str, expression: str | None) -> tuple[str, ...]:
    names = _identifiers(expression)
    if not names:
        return ()
    upstream = _ddg_upstream(graph, sink_id)
    parameters: set[str] = set()

    for node in upstream:
        if node.label.upper() != "PARAM":
            continue
        name = _parameter_name(node.code)
        if name and name in names:
            parameters.add(name)
    if parameters:
        return tuple(sorted(parameters))

    # For a local alias/derived value, first find upstream nodes mentioning the
    # sink expression and then continue to parameter definitions.
    reverse = _ddg_reverse(graph)
    for intermediate in upstream:
        if not (names & _identifiers(intermediate.code)):
            continue
        queue: deque[tuple[str, int]] = deque([(intermediate.node_id, 0)])
        seen = {intermediate.node_id}
        while queue:
            current, depth = queue.popleft()
            if depth >= 4:
                continue
            for source_id in reverse.get(current, ()):
                if source_id in seen:
                    continue
                seen.add(source_id)
                source = graph.nodes.get(source_id)
                if not source:
                    continue
                if source.label.upper() == "PARAM":
                    name = _parameter_name(source.code)
                    if name:
                        parameters.add(name)
                queue.append((source_id, depth + 1))
    return tuple(sorted(parameters))


def _arithmetic_sources(
    graph: FunctionGraph,
    sink_id: str,
    expression: str | None,
) -> tuple[GraphNode, ...]:
    names = _identifiers(expression)
    result = []
    for node in _ddg_upstream(graph, sink_id):
        if not _is_arithmetic_node(node):
            continue
        if names and not (names & _identifiers(node.code)):
            continue
        result.append(node)
        if len(result) == 4:
            break
    return tuple(result)


def _capacities(operations: list[_Operation], graph: FunctionGraph) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in graph.nodes.values():
        if "LOCAL" not in node.label.upper():
            continue
        for name, capacity in _ARRAY_DECL.findall(node.code):
            value = _normalize(capacity)
            if value and name not in result:
                result[name] = value
    dynamic: dict[str, set[str]] = {}
    for operation in operations:
        if operation.kind == "ALLOCATION" and operation.object_name and operation.extent:
            dynamic.setdefault(operation.object_name, set()).add(_compact(operation.extent, 100))
    for name, values in dynamic.items():
        if name not in result and len(values) == 1:
            result[name] = next(iter(values))
    return result


def _candidate_key(kind: str, *parts: str | None) -> str:
    return "|".join([kind, *(_normalize(part) or "?" for part in parts)])


def _redefines_name(code: str, name: str) -> bool:
    escaped = re.escape(name)
    return bool(re.search(rf"\b{escaped}\b\s*(?:=(?!=)|\+=|-=)", code))


def _security_operation_items(operations: list[_Operation]) -> list[SemanticItem]:
    items: list[SemanticItem] = []
    for operation in operations:
        fields = [f"sink={operation.kind}", f"code={_compact(operation.code, 100)}"]
        if operation.object_name:
            fields.append(f"object={_compact(operation.object_name, 60)}")
        if operation.extent:
            label = "index" if operation.kind == "ARRAY_ACCESS" else "extent"
            fields.append(f"{label}={_compact(operation.extent, 60)}")
        items.append(SemanticItem("SECURITY_OPERATION", operation.kind, " ".join(fields)))
    return items


def _mechanism_items(
    graph: FunctionGraph,
    operations: list[_Operation],
    by_node: dict[str, tuple[_Operation, ...]],
) -> tuple[list[SemanticItem], list[SemanticItem]]:
    parents = _ast_parents(graph)
    direct_conditions = _direct_conditions(graph)
    capacities = _capacities(operations, graph)
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
        violation: bool = False,
        capacity: str = "",
        example: str,
    ) -> None:
        row = candidate_rows.setdefault((kind, key), {
            "sources": set(), "sinks": set(), "objects": set(), "expressions": set(),
            "conditions": set(), "capacities": set(), "violation": False,
            "examples": [], "occurrences": 0,
        })
        row["sources"].add(source or "unknown")
        row["sinks"].add(sink)
        row["objects"].add(obj or "?")
        if expression:
            row["expressions"].add(_compact(expression, 60))
        if capacity:
            row["capacities"].add(_compact(capacity, 60))
        row["conditions"].add(condition)
        row["violation"] = bool(row["violation"]) or violation
        example_text = _compact(example, 100)
        if example_text not in row["examples"] and len(row["examples"]) < 2:
            row["examples"].append(example_text)
        row["occurrences"] = int(row["occurrences"]) + 1

    for operation in operations:
        conditions = _related_conditions(operation.node_id, parents, direct_conditions)

        if operation.kind in {"MEMORY_WRITE", "ARRAY_ACCESS"} and operation.extent:
            capacity = capacities.get(operation.object_name or "")
            parameters = _parameter_sources(graph, operation.node_id, operation.extent)
            source = "parameter:" + ",".join(parameters) if parameters else "unknown"
            has_relation_evidence = bool(parameters or capacity)
            if has_relation_evidence:
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
                statically_safe = bool(value is not None and cap is not None and not violation)
                if not statically_safe:
                    key = _candidate_key("BOUNDS_FLOW", operation.object_name, source, operation.kind)
                    add_candidate(
                        "BOUNDS_FLOW", key,
                        source=source, sink=operation.kind, obj=operation.object_name or "?",
                        expression=operation.extent, condition=condition_state,
                        violation=violation, capacity=capacity or "", example=operation.code,
                    )
                    relations.append(SemanticItem(
                        "MECHANISM_RELATION", "BOUND_RELATION",
                        f"evidence={'DDG' if parameters else 'CAPACITY'} source={source} "
                        f"sink={operation.kind} object={operation.object_name or '?'} "
                        f"expression={_compact(operation.extent, 60)} capacity={capacity or '?'} "
                        f"bound_related_condition={condition_state}",
                        key=key,
                        state=f"bound_condition={condition_state}|static_violation={'yes' if violation else 'no'}",
                    ))

            arithmetic = _arithmetic_sources(graph, operation.node_id, operation.extent)
            if arithmetic and parameters:
                expression = arithmetic[0].code
                condition_state = _condition_state(
                    graph,
                    conditions,
                    lambda condition: _upper_bound_condition(condition, expression),
                )
                source = "parameter:" + ",".join(parameters)
                key = _candidate_key("SIZE_ARITHMETIC_FLOW", source, operation.kind, operation.object_name)
                add_candidate(
                    "SIZE_ARITHMETIC_FLOW", key,
                    source=source, sink=operation.kind, obj=operation.object_name or "?",
                    expression=expression, condition=condition_state, example=operation.code,
                )
                relations.append(SemanticItem(
                    "MECHANISM_RELATION", "ARITHMETIC_TO_MEMORY_SINK",
                    f"evidence=DDG source={source} expr={_compact(expression, 80)} sink={operation.kind}",
                    key=key,
                    state=f"range_condition={condition_state}",
                ))

        elif operation.kind == "ALLOCATION" and operation.extent:
            arithmetic = _arithmetic_sources(graph, operation.node_id, operation.extent)
            parameters = _parameter_sources(graph, operation.node_id, operation.extent)
            if arithmetic and parameters:
                expression = arithmetic[0].code
                condition_state = _condition_state(
                    graph,
                    conditions,
                    lambda condition: _upper_bound_condition(condition, expression),
                )
                source = "parameter:" + ",".join(parameters)
                key = _candidate_key("SIZE_ARITHMETIC_FLOW", source, "ALLOCATION", operation.object_name)
                add_candidate(
                    "SIZE_ARITHMETIC_FLOW", key,
                    source=source, sink="ALLOCATION", obj=operation.object_name or "?",
                    expression=expression, condition=condition_state, example=operation.code,
                )
                relations.append(SemanticItem(
                    "MECHANISM_RELATION", "ARITHMETIC_TO_ALLOCATION",
                    f"evidence=DDG source={source} expr={_compact(expression, 80)} "
                    f"object={operation.object_name or '?'}",
                    key=key,
                    state=f"range_condition={condition_state}",
                ))

        elif operation.kind == "POINTER_DEREFERENCE" and operation.object_name:
            pointer = operation.object_name
            parameters = _parameter_sources(graph, operation.node_id, pointer)
            upstream_ids = {node.node_id for node in _ddg_upstream(graph, operation.node_id)}
            allocation_result = any(
                candidate.kind == "ALLOCATION"
                and candidate.object_name == pointer
                and candidate.node_id in upstream_ids
                for candidate in operations
            )
            if parameters or allocation_result:
                source = "parameter:" + ",".join(parameters) if parameters else "allocation_result"
                condition_state = _condition_state(
                    graph,
                    conditions,
                    lambda condition: _null_related_condition(condition, pointer),
                )
                key = _candidate_key("NULL_DEREFERENCE_FLOW", pointer, source)
                add_candidate(
                    "NULL_DEREFERENCE_FLOW", key,
                    source=source, sink="POINTER_DEREFERENCE", obj=pointer,
                    condition=condition_state, example=operation.code,
                )
                relations.append(SemanticItem(
                    "MECHANISM_RELATION", "NULL_RELATION",
                    f"evidence=DDG source={source} pointer={pointer} "
                    f"null_related_condition={condition_state}",
                    key=key,
                    state=f"null_condition={condition_state}",
                ))

    cfg: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind == "CFG":
            cfg.setdefault(edge.source, []).append(edge.target)

    # Lifetime relations are path candidates: a free is followed by another free
    # or use of the same syntactic object on a CFG path with no observed pointer
    # redefinition on that path. This remains an approximation, not alias proof.
    for first_free in (op for op in operations if op.kind == "DEALLOCATION" and op.object_name):
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
            for operation in by_node.get(node_id, ()):
                if operation.object_name != obj:
                    continue
                if operation.kind == "DEALLOCATION":
                    kind, relation, sink = "DOUBLE_FREE_FLOW", "FREE_TO_FREE", "DEALLOCATION"
                elif operation.kind in {"MEMORY_READ", "MEMORY_WRITE", "ARRAY_ACCESS", "POINTER_DEREFERENCE"}:
                    kind, relation, sink = "USE_AFTER_FREE_FLOW", "FREE_TO_USE", operation.kind
                else:
                    continue
                key = _candidate_key(kind, obj)
                add_candidate(
                    kind, key,
                    source="deallocation", sink=sink, obj=obj,
                    condition="not_applicable", example=operation.code,
                )
                relations.append(SemanticItem(
                    "MECHANISM_RELATION", relation,
                    f"evidence=CFG object={obj} first={_compact(first_free.code, 70)} "
                    f"later={_compact(operation.code, 70)}",
                    key=key,
                    state="path_without_redefinition=present",
                ))
            queue.extend(cfg.get(node_id, ()))

    candidates: list[SemanticItem] = []
    for (kind, key), row in sorted(candidate_rows.items()):
        conditions = set(row["conditions"])
        condition = next(iter(conditions)) if len(conditions) == 1 else ("mixed" if conditions else "unknown")
        source = ",".join(sorted(row["sources"])) or "unknown"
        sink = ",".join(sorted(row["sinks"])) or "?"
        obj = ",".join(sorted(row["objects"])) or "?"
        expression = ";".join(sorted(row["expressions"])) or "?"
        capacity = ";".join(sorted(row["capacities"])) or "?"
        violation = bool(row["violation"])
        fields = [f"source={source}", f"sink={sink}", f"object={obj}"]
        if expression != "?":
            fields.append(f"expression={expression}")
        if capacity != "?":
            fields.append(f"capacity={capacity}")
        if kind == "BOUNDS_FLOW":
            fields.append(f"bound_related_condition={condition}")
            state = f"bound_condition={condition}|static_violation={'yes' if violation else 'no'}"
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
        if row["examples"]:
            fields.append("examples=" + "; ".join(row["examples"]))
        candidates.append(SemanticItem(
            "MECHANISM_CANDIDATE", kind, " ".join(fields), key=key, state=state
        ))
    return candidates, relations


def extract_mechanism_semantics(graph: FunctionGraph) -> MechanismSemantics:
    operations_by_node = {
        node_id: operations
        for node_id, node in graph.nodes.items()
        if (operations := _node_operations(node))
    }
    operations = [
        operation
        for values in operations_by_node.values()
        for operation in values
    ]
    candidates, relations = _mechanism_items(graph, operations, operations_by_node)
    items = [*candidates, *relations, *_security_operation_items(operations)]

    deduplicated: list[SemanticItem] = []
    seen: set[tuple[str, str, str, str | None, str | None]] = set()
    for item in items:
        identity = (item.category, item.kind, item.detail, item.key, item.state)
        if identity not in seen:
            seen.add(identity)
            deduplicated.append(item)
    return MechanismSemantics(tuple(deduplicated))
