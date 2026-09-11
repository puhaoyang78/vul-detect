from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

from .cpg import FunctionGraph, GraphNode


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


@dataclass(frozen=True)
class SemanticFact:
    category: str
    kind: str
    detail: str

    def as_text(self) -> str:
        return f"{self.kind} {self.detail}".strip()


@dataclass(frozen=True)
class VulnerabilitySemantics:
    facts: tuple[SemanticFact, ...]

    @property
    def tags(self) -> tuple[str, ...]:
        seen: list[str] = []
        for fact in self.facts:
            if fact.kind not in seen:
                seen.append(fact.kind)
        return tuple(seen)

    def render(self, max_per_category: int = 32) -> str:
        grouped: dict[str, list[str]] = {}
        for fact in self.facts:
            values = grouped.setdefault(fact.category, [])
            text = fact.as_text()
            if text not in values and len(values) < max_per_category:
                values.append(text)
        if not grouped:
            return "[SEMANTICS]\nNO_VULNERABILITY_RELEVANT_FACTS"
        sections: list[str] = []
        for category in ("MEMORY", "OBJECT", "DEPENDENCY", "GUARD", "LIFETIME", "RISK_CANDIDATE"):
            values = grouped.get(category)
            if values:
                sections.append(f"[{category}]")
                sections.extend(values)
        return "\n".join(sections)


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
        operations.append(_Operation(node.node_id, "WRITE", _argument(args, object_index), extent, code))
    if name in _READ_APIS:
        object_index, extent_index = _READ_APIS[name]
        extent = f"({args[1]})*({args[2]})" if name == "fwrite" and len(args) >= 3 else _argument(args, extent_index)
        operations.append(_Operation(node.node_id, "READ", _argument(args, object_index), extent, code))
    if name in _ALLOC_APIS:
        operations.append(_Operation(node.node_id, "ALLOC", _assigned_name(code), _allocation_extent(name, args), code))
    if name in _FREE_APIS:
        operations.append(_Operation(node.node_id, "FREE", _argument(args, 0), None, code))

    for base, index in _ARRAY_ACCESS.findall(code):
        operations.append(_Operation(node.node_id, "INDEX", base, index.strip(), code))

    pointer_names: set[str] = set()
    label_lower = node.label.lower()
    if "indirection" in label_lower or "fieldaccess" in label_lower:
        pointer_names.update(_DEREF.findall(code))
        pointer_names.update(_ARROW.findall(code))
    elif "->" in code:
        pointer_names.update(_ARROW.findall(code))
    for pointer in sorted(pointer_names):
        operations.append(_Operation(node.node_id, "DEREF", pointer, None, code))

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
    escaped = re.escape(pointer)
    patterns = (
        rf"\b{escaped}\b\s*(?:!=|==)\s*(?:NULL|nullptr|0)\b",
        rf"(?:NULL|nullptr|0)\s*(?:!=|==)\s*\b{escaped}\b",
        rf"!\s*\b{escaped}\b",
        rf"^\s*\b{escaped}\b\s*$",
    )
    return any(any(re.search(pattern, condition) for pattern in patterns) for _, condition in guards.get(node_id, ()))


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


def _operation_facts(operations: list[_Operation]) -> list[SemanticFact]:
    facts: list[SemanticFact] = []
    for operation in operations:
        if operation.kind in {"WRITE", "READ", "INDEX", "DEREF"}:
            facts.append(SemanticFact("MEMORY", operation.kind, operation.describe()))
        elif operation.kind == "ALLOC":
            facts.append(SemanticFact("OBJECT", "ALLOC", operation.describe()))
        elif operation.kind == "FREE":
            facts.append(SemanticFact("LIFETIME", "FREE", operation.describe()))
    return facts


def _dependency_facts(graph: FunctionGraph, operations_by_node: dict[str, tuple[_Operation, ...]], max_depth: int = 3) -> list[SemanticFact]:
    facts: list[SemanticFact] = []
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
                    facts.append(SemanticFact("DEPENDENCY", "DATA_DEP", f"from={_compact(source.code, 100)} sink={operation.kind} object={operation.object_name or '?'} extent={operation.extent or '?'}"))
                    if "PARAMETER" in source.label.upper():
                        facts.append(SemanticFact("DEPENDENCY", "PARAMETER_DEP", f"parameter={_compact(source.code, 80)} sink={operation.kind} object={operation.object_name or '?'}"))
                    if _ARITHMETIC.search(source.code):
                        facts.append(SemanticFact("DEPENDENCY", "SIZE_ARITHMETIC", f"expr={_compact(source.code, 100)} sink={operation.kind} object={operation.object_name or '?'}"))
                queue.append((source_id, depth + 1))
    return facts


def _guard_facts(graph: FunctionGraph, guards: dict[str, list[tuple[str, str]]], operations_by_node: dict[str, tuple[_Operation, ...]]) -> list[SemanticFact]:
    facts: list[SemanticFact] = []
    seen_conditions: set[str] = set()
    for node in graph.nodes.values():
        condition = _guard_condition(node)
        if condition and condition not in seen_conditions:
            seen_conditions.add(condition)
            facts.append(SemanticFact("GUARD", "CONDITION", f"expr={_compact(condition)}"))
    for node_id, controlling in guards.items():
        for _, condition in controlling:
            for operation in operations_by_node.get(node_id, ()):
                facts.append(SemanticFact("GUARD", "GUARD_PROTECTS", f"expr={_compact(condition, 100)} operation={operation.kind} object={operation.object_name or '?'} extent={operation.extent or '?'}"))
    return facts


def _lifetime_facts(graph: FunctionGraph, operations: list[_Operation], operations_by_node: dict[str, tuple[_Operation, ...]]) -> list[SemanticFact]:
    facts: list[SemanticFact] = []
    adjacency = _cfg_adjacency(graph)
    for free in (operation for operation in operations if operation.kind == "FREE" and operation.object_name):
        for node_id in _reachable(adjacency, free.node_id):
            for operation in operations_by_node.get(node_id, ()):
                if operation.object_name != free.object_name:
                    continue
                if operation.kind == "FREE":
                    facts.append(SemanticFact("RISK_CANDIDATE", "DOUBLE_FREE", f"object={free.object_name} first_node={free.node_id} second_node={node_id}"))
                elif operation.kind in {"READ", "WRITE", "INDEX", "DEREF"}:
                    facts.append(SemanticFact("RISK_CANDIDATE", "USE_AFTER_FREE", f"object={free.object_name} free_node={free.node_id} use_node={node_id} operation={operation.kind}"))
    return facts


def _risk_facts(graph: FunctionGraph, operations: list[_Operation], guards: dict[str, list[tuple[str, str]]], capacities: dict[str, str]) -> list[SemanticFact]:
    facts: list[SemanticFact] = []
    for operation in operations:
        if operation.kind == "WRITE":
            api_name, _ = _call_name_and_args(operation.code)
            if api_name in _UNBOUNDED_WRITE_APIS:
                facts.append(SemanticFact("RISK_CANDIDATE", "UNBOUNDED_WRITE_API", f"api={api_name} object={operation.object_name or '?'}"))
            if operation.extent and not _has_matching_guard(guards, operation.node_id, operation.extent) and _identifiers(operation.extent):
                facts.append(SemanticFact("RISK_CANDIDATE", "UNGUARDED_WRITE_EXTENT", f"object={operation.object_name or '?'} extent={_compact(operation.extent)}"))
            capacity = capacities.get(operation.object_name or "")
            extent_value = _static_integer(operation.extent)
            capacity_value = _static_integer(capacity)
            if extent_value is not None and capacity_value is not None and extent_value > capacity_value:
                facts.append(SemanticFact("RISK_CANDIDATE", "WRITE_EXCEEDS_STATIC_CAPACITY", f"object={operation.object_name} extent={extent_value} capacity={capacity_value}"))
        elif operation.kind == "INDEX" and operation.extent:
            if not _has_matching_guard(guards, operation.node_id, operation.extent):
                facts.append(SemanticFact("RISK_CANDIDATE", "UNCHECKED_INDEX", f"object={operation.object_name or '?'} index={_compact(operation.extent)}"))
        elif operation.kind == "DEREF" and operation.object_name:
            if not _has_null_guard(guards, operation.node_id, operation.object_name):
                facts.append(SemanticFact("RISK_CANDIDATE", "UNGUARDED_DEREFERENCE", f"pointer={operation.object_name}"))

    for edge in graph.edges:
        if edge.kind != "DDG":
            continue
        source = graph.nodes.get(edge.source)
        target = graph.nodes.get(edge.target)
        if source is None or target is None or not _ARITHMETIC.search(source.code):
            continue
        target_ops = _node_operations(target)
        if any(operation.kind in {"WRITE", "ALLOC", "INDEX"} for operation in target_ops):
            if not _has_matching_guard(guards, target.node_id, source.code):
                facts.append(SemanticFact("RISK_CANDIDATE", "UNGUARDED_SIZE_ARITHMETIC", f"expr={_compact(source.code, 100)} sink={_compact(target.code, 100)}"))
    return facts


def extract_vulnerability_semantics(graph: FunctionGraph) -> VulnerabilitySemantics:
    operations_by_node = {
        node_id: operations
        for node_id, node in graph.nodes.items()
        if (operations := _node_operations(node))
    }
    operations = [operation for values in operations_by_node.values() for operation in values]
    guards = _controlling_guards(graph)
    capacities = _capacity_facts(graph)

    facts: list[SemanticFact] = []
    facts.extend(_operation_facts(operations))
    for name, capacity in capacities.items():
        facts.append(SemanticFact("OBJECT", "STATIC_CAPACITY", f"object={name} capacity={_compact(capacity)}"))
    facts.extend(_dependency_facts(graph, operations_by_node))
    facts.extend(_guard_facts(graph, guards, operations_by_node))
    facts.extend(_lifetime_facts(graph, operations, operations_by_node))
    facts.extend(_risk_facts(graph, operations, guards, capacities))

    deduplicated: list[SemanticFact] = []
    seen: set[tuple[str, str, str]] = set()
    for fact in facts:
        key = (fact.category, fact.kind, fact.detail)
        if key not in seen:
            seen.add(key)
            deduplicated.append(fact)
    return VulnerabilitySemantics(tuple(deduplicated))
