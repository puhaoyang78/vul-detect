"""Intraprocedural vulnerability-pattern matching from C/C++ syntax.

Patterns are candidate program properties, not ground-truth vulnerability locations.
In particular, lexical order does not establish CFG reachability, and missing
bounds checks in a function fragment do not establish their absence in callers.
Every pattern records its source spans and a reason for audit.
"""

import re

from .syntax import byte_character_map, parser_for, walk

COPY_APIS = {
    "memcpy", "memmove", "strncpy", "wmemcpy", "wmemmove", "strcpy", "wcscpy",
    "strcat", "strncat", "memset", "wmemset", "read", "recv", "fread",
}
READ_APIS = {"read", "recv", "fread", "fgets", "fgetc", "getchar", "scanf", "gets"}
WRITE_APIS = COPY_APIS | {"sprintf", "snprintf", "vsprintf", "write", "send"}
FREE_APIS = {"free", "kfree", "delete"}
ALLOC_APIS = {"malloc", "calloc", "realloc", "alloca"}
BUFFER_COPY = {"memcpy", "memmove", "strncpy", "wmemcpy", "wmemmove", "memset", "wmemset"}
PATTERN_TYPES = (
    "memory_access", "pointer_access", "allocation", "release", "guard",
    "sensitive_read", "sensitive_write",
    "buffer_size_mismatch", "source_size_copy", "destination_size_copy",
    "off_by_one", "buffer_overread",
    "double_free", "use_after_free", "negative_index_read", "negative_index_write",
    "guarded_operation",
)
IDENT = re.compile(r"^[A-Za-z_]\w*$")
SIGNED_DECL = re.compile(r"\b(?:signed|int|short|long|ssize_t|ptrdiff_t)\s+([A-Za-z_]\w*)\b")
UNSIGNED_DECL = re.compile(r"\b(?:unsigned|size_t|uint\d+_t)\s+([A-Za-z_]\w*)\b")


def _text(source, node):
    if node is None:
        return ""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _name(source, node):
    if node is None:
        return ""
    value = _text(source, node).strip()
    return value.rsplit("::", 1)[-1].rsplit("->", 1)[-1].rsplit(".", 1)[-1]


def _field(node, name, default=0):
    child = node.child_by_field_name(name)
    if child is not None:
        return child
    named = node.named_children
    return named[default] if len(named) > default else None


def _integer(expr, constants=None, arrays=None, *, depth=0):
    """Conservative constant folding; unknown or symbolic values return None."""
    if depth > 3:
        return None
    constants, arrays = constants or {}, arrays or {}
    s = re.sub(r"\s+", "", expr)
    while s.startswith("(") and s.endswith(")") and s.count("(") == s.count(")"):
        s = s[1:-1]
    if re.fullmatch(r"0x[0-9a-fA-F]+|\d+[uUlL]*", s):
        try:
            return int(re.sub(r"[uUlL]+$", "", s), 0 if s.startswith("0x") else 10)
        except ValueError:
            return None
    if re.fullmatch(r"-[0-9]+", s):
        return -int(s[1:])
    if s in constants:
        return constants[s]
    match = re.fullmatch(r"sizeof\((\w+)\)", s)
    if match:
        return arrays.get(match[1])
    # Only simple numeric/known-symbol arithmetic. Never eval arbitrary code.
    for op in ("+", "-", "*"):
        for j in range(1, len(s)):
            if s[j] == op and s[:j].count("(") == s[:j].count(")") and s[j + 1:]:
                left = _integer(s[:j], constants, arrays, depth=depth + 1)
                right = _integer(s[j + 1:], constants, arrays, depth=depth + 1)
                if left is not None and right is not None:
                    return {"+": lambda: left + right, "-": lambda: left - right,
                            "*": lambda: left * right}[op]()
    return None


def _array_declarator(node, source):
    """Return the base identifier of a declaration, if one is present."""
    cur = node
    for _ in range(5):
        if cur is None:
            break
        if cur.type in ("identifier", "field_identifier"):
            return _text(source, cur)
        cur = cur.child_by_field_name("declarator") or (cur.named_children[0] if cur.named_children else None)
    return None


def _is_assignment_between(nodes, source, name, start, stop):
    for node in nodes:
        if start < node.start_byte < stop and node.type in ("assignment_expression", "init_declarator"):
            left = _field(node, "left")
            if node.type == "init_declarator":
                left = node.child_by_field_name("declarator") or left
            if left is not None and _name(source, left) == name:
                return True
    return False


def extract_patterns(source, *, language="c"):
    """Identify common VIPer-style families plus guard/operation context pairs.

    Uses Tree-sitter expressions and typed roles rather than mere API-presence
    labels. Conditional dominance, aliasing, and branch feasibility remain
    unverified; those cases are reported as candidates, never confirmed bugs.
    """
    data = source.encode("utf-8")
    parser = parser_for(language)
    tree = parser.parse(data)
    char = byte_character_map(source)
    nodes = list(walk(tree.root_node))
    patterns, seen = [], set()

    def span(node):
        if node is None:
            return None
        return [char[node.start_byte], char[node.end_byte]]

    def add(kind, anchor, related=(), *, status="candidate", detail=""):
        if kind not in PATTERN_TYPES or anchor is None:
            return
        a = span(anchor)
        links = [span(n) for n in related if n is not None]
        links = [s for s in links if s and s[1] > s[0] and s != a]
        key = (kind, *a, *(tuple(s) for s in links))
        if key in seen:
            return
        seen.add(key)
        patterns.append(dict(kind=kind, anchor=a, related=links, status=status, detail=detail))

    arrays, constants = {}, {}
    declarations = [n for n in nodes if n.type == "array_declarator"]
    for node in declarations:
        identifier = _array_declarator(node.child_by_field_name("declarator"), data)
        size = node.child_by_field_name("size")
        if identifier and size:
            value = _integer(_text(data, size), constants, arrays)
            if value is not None and value >= 0:
                arrays[identifier] = value
    for node in nodes:
        if node.type == "init_declarator":
            identifier = _array_declarator(node.child_by_field_name("declarator"), data)
            value = node.child_by_field_name("value")
            if identifier and value:
                number = _integer(_text(data, value), constants, arrays)
                if number is not None:
                    constants[identifier] = number

    calls = []
    for node in nodes:
        if node.type != "call_expression":
            continue
        fun = node.child_by_field_name("function")
        name = _name(data, fun)
        args_node = node.child_by_field_name("arguments")
        args = list(args_node.named_children) if args_node is not None else []
        calls.append((node, name, args))

    guards = [n for n in nodes if n.type == "if_statement"]
    subscripts = [n for n in nodes if n.type == "subscript_expression"]

    def bounds(name):
        return arrays.get(name)

    def nearest_guard(node, identifiers):
        valid = []
        for check in guards:
            cond = check.child_by_field_name("condition")
            if cond is None:
                continue
            body = check.child_by_field_name("consequence")
            if body is None or not (body.start_byte <= node.start_byte < body.end_byte):
                continue
            expr = _text(data, cond)
            if any(re.search(r"\b" + re.escape(i) + r"\b", expr) for i in identifiers if i):
                valid.append(cond)
        return min(valid, key=lambda n: n.end_byte - n.start_byte) if valid else None

    allocations = []
    for node, name, args in calls:
        if name not in ALLOC_APIS:
            continue
        add("allocation", node, args[:2], detail=name)
        parent = node.parent
        target = None
        if parent and parent.type in ("init_declarator", "assignment_expression"):
            target = _field(parent, "left")
            if parent.type == "init_declarator":
                target = parent.child_by_field_name("declarator")
        elif parent and parent.parent and parent.parent.type == "init_declarator":
            target = parent.parent.child_by_field_name("declarator")
        if target:
            identifier = _name(data, target)
            if IDENT.fullmatch(identifier):
                allocations.append((identifier, node.start_byte))
                if args:
                    number = _integer(_text(data, args[0]), constants, arrays)
                    if name == "calloc" and len(args) >= 2:
                        a = _integer(_text(data, args[0]), constants, arrays)
                        b = _integer(_text(data, args[1]), constants, arrays)
                        number = a * b if a is not None and b is not None else None
                    if number is not None:
                        arrays[identifier] = number

    for node in nodes:
        if node.type in ("pointer_expression", "field_expression"):
            expression = _text(data, node)
            if expression.startswith("*") or "->" in expression:
                add("pointer_access", node)
        if node.type in ("subscript_expression",):
            add("memory_access", node)

    for check in guards:
        condition = check.child_by_field_name("condition")
        if condition and re.search(r"<|>|==|!=", _text(data, condition)):
            add("guard", condition)

    frees = []
    for node, name, args in calls:
        if name not in FREE_APIS or not args:
            continue
        var = _text(data, args[0]).strip()
        add("release", node, args[:1], detail=name)
        if not IDENT.fullmatch(var):
            continue
        for previous, oldname, _ in frees:
            if oldname == var and not _is_assignment_between(nodes, data, var, previous.end_byte, node.start_byte):
                add("double_free", node, (previous, args[0]), detail="source-order candidate; CFG paths unchecked")
        frees.append((node, var, args[0]))

    # Collect calls before/after frees; use CFG-informed dependencies later.
    for previous, var, arg in frees:
        for use in nodes:
            if use.start_byte <= previous.end_byte or use.type not in ("subscript_expression", "pointer_expression", "field_expression"):
                continue
            raw = _text(data, use)
            if not re.search(r"\b" + re.escape(var) + r"\b", raw):
                continue
            if not (raw.startswith("*") or raw.startswith(var + "[") or raw.startswith(var + "->")):
                continue
            if not _is_assignment_between(nodes, data, var, previous.end_byte, use.start_byte):
                add("use_after_free", use, (arg,), detail="source-order candidate; path and aliasing unchecked")

    signed = set(SIGNED_DECL.findall(source)) - set(UNSIGNED_DECL.findall(source))
    for node in subscripts:
        arr = _field(node, "argument")
        idx = node.child_by_field_name("index")
        if idx is None and node.named_children:
            idx = node.named_children[-1]
        expr = _text(data, idx).strip()
        if not expr:
            continue
        index = _integer(expr, constants, arrays)
        potentially_negative = (index is not None and index < 0) or (expr in signed)
        if not potentially_negative:
            continue
        check = nearest_guard(node, [expr])
        # Condition membership alone does not prove a nonnegative bound.
        if check and re.search(r"\b" + re.escape(expr) + r"\s*(?:>=\s*0|>\s*-1)", _text(data, check)):
            add("guarded_operation", node, (check, idx), detail="local syntactic guard candidate")
            continue
        cur = node.parent
        writable = False
        while cur and cur.type in ("parenthesized_expression", "subscript_expression", "field_expression", "pointer_expression"):
            cur = cur.parent
        if cur and cur.type in ("assignment_expression", "update_expression"):
            left = _field(cur, "left")
            writable = bool(left and left.start_byte <= node.start_byte and left.end_byte >= node.end_byte)
        add("negative_index_write" if writable else "negative_index_read", node, (idx,),
            detail="index may be negative; guards outside local syntax unknown")

    for node, name, args in calls:
        if name in READ_APIS:
            add("sensitive_read", node, args[:3], detail=name)
        if name in WRITE_APIS:
            add("sensitive_write", node, args[:3], detail=name)
        if name not in COPY_APIS or len(args) < 2:
            continue
        dst = args[0]
        src = args[1] if name not in ("memset", "wmemset") else None
        size_arg = args[2] if len(args) > 2 else None
        dstname = _text(data, dst).strip()
        srcname = _text(data, src).strip()
        dcap = bounds(dstname)
        scap = bounds(srcname)
        size_text = _text(data, size_arg)
        count = _integer(size_text, constants, arrays) if size_arg else None
        if name in ("strcpy", "wcscpy") and src and src.type in ("string_literal", "concatenated_string"):
            literal = _text(data, src)
            count = len(literal[1:-1]) + 1 if literal.startswith('"') and literal.endswith('"') else None
        if size_arg and re.fullmatch(r"\s*sizeof\s*\(\s*" + re.escape(srcname) + r"\s*\)\s*", size_text) and srcname:
            add("source_size_copy", node, (src, size_arg), detail="sizeof(source) used for a write bound")
        if size_arg and re.fullmatch(r"\s*sizeof\s*\(\s*" + re.escape(dstname) + r"\s*\)\s*", size_text) and dstname:
            add("destination_size_copy", node, (dst, size_arg), detail="sizeof(destination) used for a write bound")
        if dcap is not None and count is not None:
            if count > dcap:
                add("buffer_size_mismatch", node, (dst, size_arg),
                    status="constant_comparison", detail="copy size exceeds known destination extent")
            if count == dcap + 1:
                add("off_by_one", node, (dst, size_arg),
                    status="constant_comparison", detail="copy size equals known destination extent plus one")
        if scap is not None and count is not None and src is not None and count > scap:
            add("buffer_overread", node, (src, size_arg),
                status="constant_comparison", detail="copy size exceeds known source extent")
        guard = nearest_guard(node, [dstname, srcname, size_text.strip()])
        if guard:
            add("guarded_operation", node, (guard, dst), detail="local syntactic condition; adequacy not established")

    patterns.sort(key=lambda p: (p["anchor"][0], p["kind"], p["anchor"][1]))
    return patterns
