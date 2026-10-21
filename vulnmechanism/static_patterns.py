"""Auditable intraprocedural program relations, never vulnerability labels.

Local definitions respect declaration bindings, source order and control context.
Unknown extents, ABI sizes, aliasing and path feasibility remain explicit unknowns.
"""
from collections import Counter, defaultdict, deque
from bisect import bisect_right
from dataclasses import dataclass
import re

from .syntax import (byte_character_map, declarator_identifier, function_bindings,
                     parser_for, resolve_local_binding, walk)

COPY_APIS = {
    "memcpy", "memmove", "strncpy", "wmemcpy", "wmemmove", "strcpy", "wcscpy",
    "strcat", "strncat", "memset", "wmemset", "read", "recv", "fread",
}
READ_APIS = {"read", "recv", "fread", "fgets", "fgetc", "getchar", "scanf", "gets"}
WRITE_APIS = COPY_APIS | {"sprintf", "snprintf", "vsprintf", "write", "send"}
FREE_APIS = {"free", "kfree"}
ALLOC_APIS = {"malloc", "calloc", "realloc", "alloca"}
PATTERN_TYPES = (
    "memory_access", "pointer_access", "allocation", "release", "guard",
    "sensitive_read", "sensitive_write",
    "buffer_size_mismatch", "source_size_copy", "destination_size_copy",
    "off_by_one", "buffer_overread",
    "double_free", "use_after_free", "negative_index_read", "negative_index_write",
    "guarded_operation",
)


def _text(source, node):
    return "" if node is None else source[node.start_byte:node.end_byte].decode("utf-8")


def _unwrap(node):
    while node is not None and node.type in ("parenthesized_expression", "cast_expression", "condition_clause"):
        node = node.child_by_field_name("value") if node.type in ("cast_expression", "condition_clause") else (
            node.named_children[0] if len(node.named_children) == 1 else None)
    return node


def _callee(data, node):
    value = _text(data, node.child_by_field_name("function"))
    # An unrelated object's malloc/free method is not the C library function.
    return re.sub(r"^(?:std::|::)", "", value) if re.fullmatch(
        r"(?:(?:std)?::)?[A-Za-z_]\w*", value) else ""


def _subscript(node):
    base = node.child_by_field_name("argument")
    index = node.child_by_field_name("index")
    if index is None:
        indices = node.child_by_field_name("indices")
        if indices is not None and len(indices.named_children) == 1:
            index = indices.named_children[0]
    return base, index


def _context(node):
    """Branch identity, not substring membership in a condition."""
    result, current = [], node
    while current.parent is not None:
        parent = current.parent
        if parent.type == "if_statement":
            for field, arm in (("consequence", 1), ("alternative", 0)):
                child = parent.child_by_field_name(field)
                if child is not None and child.start_byte <= node.start_byte < child.end_byte:
                    result.append((parent.start_byte, arm))
        elif parent.type in ("for_statement", "while_statement", "do_statement", "switch_statement"):
            body = parent.child_by_field_name("body")
            if body is not None and body.start_byte <= node.start_byte < body.end_byte:
                result.append((parent.start_byte, 2))
        current = parent
    return tuple(reversed(result))


def _conflict(left, right):
    other = dict(right)
    return any(key in other and value != other[key] for key, value in left)


def _return_before(first, last):
    child = first
    while child.parent is not None:
        parent = child.parent
        if parent.type == "compound_statement":
            for sibling in parent.named_children:
                if sibling.start_byte < child.end_byte:
                    continue
                if sibling.start_byte >= last.start_byte and parent.start_byte <= last.start_byte < parent.end_byte:
                    break
                if sibling.type == "return_statement":
                    return True
        if parent.type in ("function_definition", "lambda_expression"):
            break
        child = parent
    return False


def _prefix(left, right):
    return all(item in right for item in left)


def _write_mode(node):
    current = node
    while current.parent is not None and current.parent.type in (
        "parenthesized_expression", "field_expression", "subscript_expression", "pointer_expression"
    ):
        current = current.parent
    parent = current.parent
    if current.type == "pointer_expression" and any(c.type == "&" for c in current.children):
        return None
    if parent is None:
        return False
    if parent.type == "update_expression":
        return True
    if parent.type == "assignment_expression":
        left = parent.child_by_field_name("left")
        return left is not None and left.start_byte <= node.start_byte < left.end_byte
    return False


def _literal_bytes(data, node):
    """C narrow strings, including concatenation, escapes and embedded NULs."""
    if node is None:
        return None
    if node.type == "concatenated_string":
        parts = [_literal_bytes(data, n) for n in node.named_children
                 if n.type in ("string_literal", "raw_string_literal")]
        return b"".join(parts) if parts and all(p is not None for p in parts) else None
    value = _text(data, node)
    raw = re.fullmatch(r'(?:u8)?R"([^ ()\\\t\r\n]{0,16})\((.*)\)\1"', value, re.S)
    if raw:
        return raw[2].encode() if raw[2].isascii() else None
    match = re.fullmatch(r'(?:u8)?"(.*)"', value, re.S)
    if not match:
        return None
    text, out, at = match[1], bytearray(), 0
    escapes = dict(zip("abfnrtv\\'\"?", (7, 8, 12, 10, 13, 9, 11, 92, 39, 34, 63)))
    while at < len(text):
        char = text[at]
        if char != "\\":
            if ord(char) > 127:
                return None  # Execution character set is unavailable.
            out.append(ord(char)); at += 1; continue
        at += 1
        if at == len(text):
            return None
        char = text[at]
        if char in escapes:
            out.append(escapes[char]); at += 1
        elif char == "\n":
            at += 1
        elif char in "01234567":
            number = re.match(r"[0-7]{1,3}", text[at:])[0]
            if int(number, 8) > 255:
                return None
            out.append(int(number, 8)); at += len(number)
        elif char == "x":
            number = re.match(r"[0-9a-fA-F]+", text[at + 1:])
            if number is None or int(number[0], 16) > 255:
                return None
            out.append(int(number[0], 16)); at += 1 + len(number[0])
        else:
            return None
    return bytes(out)


@dataclass(frozen=True)
class _Amount:
    constant: int = 0
    terms: tuple = ()
    origins: tuple = ()
    definitions: tuple = ()

    @property
    def number(self):
        return None if self.terms else self.constant

    def describe(self):
        return dict(constant=self.constant, symbols=dict(self.terms),
                    origins=[list(x) for x in self.origins])


def _combine(a, b, scale=1):
    if a is None or b is None:
        return None
    terms = Counter(dict(a.terms))
    for name, value in b.terms:
        terms[name] += scale * value
    return _Amount(a.constant + scale * b.constant,
                   tuple(sorted((k, v) for k, v in terms.items() if v)),
                   tuple(sorted(set(a.origins + b.origins))),
                   tuple(sorted(set(a.definitions + b.definitions))))


def _scale(value, factor):
    if value is None:
        return None
    if factor == 0:
        return _Amount()
    return _Amount(value.constant * factor, tuple((k, v * factor) for k, v in value.terms),
                   value.origins, value.definitions)


def _product(a, b):
    if a is None or b is None:
        return None
    if a.number is not None:
        return _scale(b, a.number)
    if b.number is not None:
        return _scale(a, b.number)
    return None  # No speculative nonlinear range arithmetic.


def _bounds(value):
    """sizeof a complete object type is positive; its ABI byte size is unknown."""
    if value is None:
        return None, None
    lower = upper = value.constant
    for name, coefficient in value.terms:
        if not name.startswith("sizeof_type("):
            return None, None
        if coefficient > 0:
            lower = lower + coefficient if lower is not None else None
            upper = None
        else:
            lower = None
            upper = upper + coefficient if upper is not None else None
    return lower, upper


class _Facts:
    """Use-site local definitions and typed object extents; no whole-program claims."""

    def __init__(self, data, nodes, graph):
        self.data, self.nodes, self.graph = data, nodes, graph
        self.bindings, self.by_declaration = [], {}
        self.bindings_by_name = defaultdict(list)
        self.labels = sorted(n.start_byte for n in nodes if n.type == "labeled_statement")
        self.syntax_gaps = sorted((n.start_byte, n.end_byte) for n in nodes
                                  if n.type == "ERROR" or n.is_missing)
        self.memory_events = [n for n in nodes if n.type in (
            "assignment_expression", "update_expression", "call_expression")]
        self.writes, self.unknown = defaultdict(list), {}
        self.field_writes = defaultdict(list)
        self.definition_cache, self.buffer_cache, self.value_cache = {}, {}, {}
        self.definition_errors = {}
        self.binding_cache, self.context_cache = {}, {}
        for node in nodes:
            if (node.type == "function_definition" and node.child_by_field_name("body") is not None
                    and node.child_by_field_name("declarator") is not None):
                self.bindings.extend(function_bindings(node, data))
        for binding in self.bindings:
            self.by_declaration[binding["declaration"].id] = binding
            self.bindings_by_name[binding["name"]].append(binding)
        self.type_names = defaultdict(list)
        for b in self.bindings:
            self.type_names[self.type_text(b)].append((b["scope"].start_byte, b["scope"].end_byte,
                                                      b["statement"].start_byte))
        for node in nodes:
            target, value, op = None, None, "="
            if node.type == "init_declarator":
                target = declarator_identifier(node)
                value = node.child_by_field_name("value")
            elif node.type == "assignment_expression":
                target, value = node.child_by_field_name("left"), node.child_by_field_name("right")
                op = _text(data, node.child_by_field_name("operator"))
            elif node.type == "update_expression":
                target = node.child_by_field_name("argument")
                if target is None and node.named_children:
                    target = node.named_children[0]
                op = "update"
            if target is not None:
                key = self.key(target)
                if key:
                    self.writes[key].append((node, target, value, op))
                    if target.type == "field_expression":
                        self.field_writes[_text(data, target.child_by_field_name("field"))].append((key, node))
            # A call with &n may redefine n. Do not retain an old constant.
            if node.type == "call_expression":
                args = node.child_by_field_name("arguments")
                if args:
                    for arg in args.named_children:
                        if arg.type == "pointer_expression" and _text(data, arg).lstrip().startswith("&"):
                            target = arg.child_by_field_name("argument")
                            if target is not None and target.type == "identifier" and self.key(target):
                                self.writes[self.key(target)].append((node, target, None, "call_by_address"))
        for values in self.writes.values():
            values.sort(key=lambda v: v[0].start_byte)
        self.cfg = defaultdict(set)
        if graph is not None:
            for a, b, kind in graph["edges"]:
                if kind == 1:  # Existing canonical EDGE_TYPES: CFG.
                    self.cfg[a].add(b)
        self.cfg_nodes = set(self.cfg) | {v for values in self.cfg.values() for v in values}
        self.path_cache = {}

    def context(self, node):
        if node.id not in self.context_cache:
            self.context_cache[node.id] = _context(node)
        return self.context_cache[node.id]

    def fail(self, node, reason):
        if node is not None:
            self.unknown[node.id] = reason
        return None

    def binding(self, node):
        if node is None or node.type != "identifier":
            return None
        if node.id not in self.binding_cache:
            self.binding_cache[node.id] = self.by_declaration.get(node.id) or resolve_local_binding(
                node, self.bindings_by_name[_text(self.data, node)], self.data)
        binding = self.binding_cache[node.id]
        return binding if binding is not None and binding["supported"] else None

    def key(self, node):
        node = _unwrap(node)
        if node is None:
            return None
        if node.type == "identifier":
            binding = self.binding(node)
            return f"local@{binding['declaration'].start_byte}" if binding is not None else None
        if node.type == "field_expression":
            base = self.key(node.child_by_field_name("argument"))
            field = _text(self.data, node.child_by_field_name("field"))
            return base + "." + field if base and field else None
        return None

    def definition(self, node, at):
        key = self.key(node)
        if key is None:
            return self.fail(node, "unresolved_binding")
        cache = (key, at.start_byte)
        if cache in self.definition_cache:
            if cache in self.definition_errors:
                self.fail(node, self.definition_errors[cache])
            return self.definition_cache[cache]
        context = self.context(at)
        candidates = [v for v in self.writes[key] if v[0].end_byte <= at.start_byte
                      and not _conflict(self.context(v[0]), context)]
        latest = candidates[-1] if candidates else None
        if latest is not None and any(
                latest[0].end_byte <= start < at.start_byte
                for start, _ in self.syntax_gaps):
            self.definition_cache[cache] = None
            self.definition_errors[cache] = "intervening_parse_error_or_macro"
            return self.fail(node, "intervening_parse_error_or_macro")
        # Repeated loop assignments are not a single constant definition.
        for parent in self.parents(at):
            if parent.type in ("for_statement", "while_statement", "do_statement"):
                body, update = parent.child_by_field_name("body"), parent.child_by_field_name("update")
                condition = parent.child_by_field_name("condition")
                repeated = [part for part in (body, update, condition) if part is not None]
                if any(any(part.start_byte <= v[0].start_byte < part.end_byte for part in repeated)
                       for v in self.writes[key]):
                    # An unconditional assignment in this iteration can define
                    # the value; an initializer before the loop cannot.
                    local = latest is not None and any(
                        part.start_byte <= latest[0].start_byte < part.end_byte
                        for part in (body, condition) if part is not None)
                    short_circuit = latest is not None and any(
                        p.type == "binary_expression" and
                        _text(self.data, p.child_by_field_name("operator")) in ("&&", "||")
                        for p in self.parents(latest[0])
                        if parent.start_byte <= p.start_byte < parent.end_byte)
                    if local and not short_circuit and _prefix(self.context(latest[0]), context):
                        continue
                    self.definition_cache[cache] = None
                    self.definition_errors[cache] = "loop_variant_definition"
                    return self.fail(node, "loop_variant_definition")
        if latest is not None and not _prefix(self.context(latest[0]), context):
            self.definition_cache[cache] = None
            self.definition_errors[cache] = "conditional_definition"
            return self.fail(node, "conditional_definition")
        next_label = bisect_right(self.labels, latest[0].end_byte) if latest else len(self.labels)
        if latest is not None and next_label < len(self.labels) and self.labels[next_label] < at.start_byte:
            self.definition_cache[cache] = None
            self.definition_errors[cache] = "unstructured_control_flow"
            return self.fail(node, "unstructured_control_flow")
        self.definition_cache[cache] = latest
        return latest

    @staticmethod
    def parents(node):
        while node.parent is not None:
            node = node.parent
            yield node

    def type_text(self, binding):
        statement = binding["statement"]
        typ = statement.child_by_field_name("type")
        return _text(self.data, typ).strip()

    def element_size(self, binding=None, type_text=None):
        typ = (type_text if type_text is not None else self.type_text(binding))
        if typ in ("char", "signed char", "unsigned char", "int8_t", "uint8_t", "std::byte"):
            return _Amount(1)
        return _Amount(terms=((f"sizeof_type({typ})", 1),)) if typ else None

    def value(self, node, at=None, depth=0):
        if node is None or depth > 12:
            return self.fail(node, "expression_depth")
        at = node if at is None else at
        cache = (node.id, at.start_byte)
        if cache not in self.value_cache:
            self.value_cache[cache] = self._value(node, at, depth)
        return self.value_cache[cache]

    def _value(self, node, at, depth):
        data, raw = self.data, _text(self.data, node).strip()
        if node.has_error:
            return self.fail(node, "expression_parse_error")
        if node.type == "parenthesized_expression":
            return self.value(_unwrap(node), at, depth + 1)
        if node.type == "number_literal":
            text = re.sub(r"[uUlL]+$", "", raw).replace("'", "")
            try:
                base = 16 if text.lower().startswith("0x") else 2 if text.lower().startswith("0b") else (
                    8 if text.startswith("0") and len(text) > 1 else 10)
                return _Amount(int(text, base))
            except ValueError:
                return self.fail(node, "noninteger_literal")
        if node.type == "identifier":
            binding = self.binding(node)
            if binding is None:
                return self.fail(node, "external_constant_or_variable")
            definition = self.definition(node, at)
            if definition is not None:
                site, _, rhs, operator = definition
                if operator != "=" or rhs is None:
                    return self.fail(node, "unsupported_redefinition")
                value = self.value(rhs, site, depth + 1)
                if value is not None:
                    return _Amount(value.constant, value.terms, value.origins,
                                   tuple(sorted(set(value.definitions + ((site.start_byte, site.end_byte),)))))
                return self.fail(node, self.unknown.get(rhs.id, "unresolved_definition"))
            if node.id in self.unknown:
                return None
            if binding["parameter_index"] is not None:
                return _Amount(terms=((self.key(node), 1),))
            return self.fail(node, "uninitialized_or_ambiguous_definition")
        if node.type == "sizeof_expression":
            operand = node.child_by_field_name("value")
            typ = node.child_by_field_name("type")
            if typ is not None:
                return self.element_size(type_text=_text(data, typ))
            operand = _unwrap(operand)
            if operand is not None and operand.type == "identifier":
                binding = self.binding(operand)
                if binding is None and any(a <= at.start_byte < b and declared < at.start_byte
                                           for a, b, declared in self.type_names[_text(data, operand)]):
                    return self.element_size(type_text=_text(data, operand))
                if binding is not None:
                    declarators = [n for n in self.parents(binding["declaration"])
                                   if n.start_byte >= binding["statement"].start_byte
                                   and n.end_byte <= binding["statement"].end_byte]
                    array = any(n.type == "array_declarator" for n in declarators)
                    pointer = any(n.type == "pointer_declarator" for n in declarators)
                    if array and binding["parameter_index"] is None:
                        buf = self.buffer(operand, at, depth + 1)
                        if buf and buf["capacity"] is not None:
                            v = buf["capacity"]
                            return _Amount(v.constant, v.terms, (("sizeof", buf["object"]),), v.definitions)
                    if pointer or (array and binding["parameter_index"] is not None):
                        return self.fail(node, "sizeof_pointer_is_not_extent")
                    return self.element_size(binding)
            if operand is not None and operand.type in ("pointer_expression", "subscript_expression"):
                base = operand.child_by_field_name("argument")
                if operand.type == "subscript_expression":
                    base, _ = _subscript(operand)
                buf = self.buffer(base, at, depth + 1)
                return buf["element"] if buf else self.fail(node, "pointee_type_unknown")
            # sizeof a local field is a type extent, not a guessed byte count.
            key = self.key(operand)
            if key:
                return _Amount(terms=((f"sizeof_object({key})", 1),))
            return self.fail(node, "sizeof_type_unknown")
        if node.type == "unary_expression":
            operator = _text(data, node.child_by_field_name("operator"))
            arg = node.child_by_field_name("argument")
            v = self.value(arg, at, depth + 1)
            return _scale(v, -1 if operator == "-" else 1) if operator in ("+", "-") else self.fail(node, "unsupported_unary")
        if node.type == "binary_expression":
            left, right = node.child_by_field_name("left"), node.child_by_field_name("right")
            a, b = self.value(left, at, depth + 1), self.value(right, at, depth + 1)
            op = _text(data, node.child_by_field_name("operator"))
            result = _combine(a, b, -1 if op == "-" else 1) if op in ("+", "-") else (
                _product(a, b) if op == "*" else None)
            if result is not None:
                return result
            if a and b and a.number is not None and b.number is not None:
                x, y = a.number, b.number
                if op == "/" and y:
                    return _Amount((abs(x) // abs(y)) * (-1 if (x < 0) != (y < 0) else 1))
                if op in ("<<", ">>") and 0 <= y < 64 and x >= 0:
                    return _Amount(x << y if op == "<<" else x >> y)
            return self.fail(node, self.unknown.get(left.id, self.unknown.get(right.id, "unsupported_or_nonlinear_arithmetic")))
        if node.type == "call_expression" and _callee(data, node) == "strlen":
            args = node.child_by_field_name("arguments").named_children
            if len(args) == 1:
                buf = self.buffer(args[0], at, depth + 1)
                if buf:
                    length = self.string_length(buf, at)
                    if length is not None:
                        return _Amount(length.constant, length.terms, (("strlen", buf["object"]),),
                                       length.definitions)
            return self.fail(node, "string_length_unknown")
        return self.fail(node, "unsupported_expression")

    def buffer(self, node, at=None, depth=0):
        if node is None or depth > 12:
            return None
        at = node if at is None else at
        cache = (node.id, at.start_byte)
        if cache not in self.buffer_cache:
            self.buffer_cache[cache] = self._buffer(node, at, depth)
        return self.buffer_cache[cache]

    def _buffer(self, node, at, depth):
        data = self.data
        if node.has_error:
            return self.fail(node, "expression_parse_error")
        if (node.type in ("null", "nullptr") or
                node.type == "identifier" and _text(data, node) == "NULL" and self.binding(node) is None or
                node.type == "number_literal" and self.value(node) is not None and self.value(node).number == 0):
            return dict(object=None, capacity=None, offset=_Amount(), element=None, length=None,
                        definition=node, reason="null_pointer")
        if node.type == "parenthesized_expression":
            return self.buffer(_unwrap(node), at, depth + 1)
        if node.type == "cast_expression":
            base = self.buffer(node.child_by_field_name("value"), at, depth + 1)
            if base:
                typ = re.sub(r"\*|\b(?:const|volatile)\b", "", _text(data, node.child_by_field_name("type"))).strip()
                return dict(base, element=self.element_size(type_text=typ))
            return None
        literal = _literal_bytes(data, node)
        if literal is not None:
            return dict(object=f"literal@{node.start_byte}", capacity=_Amount(len(literal) + 1),
                        offset=_Amount(), element=_Amount(1), length=_Amount(len(literal.split(b"\0", 1)[0])),
                        definition=node, reason=None)
        if node.type == "identifier":
            binding = self.binding(node)
            if binding is None:
                return self.fail(node, "unresolved_buffer_binding")
            key = self.key(node)
            ancestors = [n for n in self.parents(binding["declaration"])
                         if n.start_byte >= binding["statement"].start_byte
                         and n.end_byte <= binding["statement"].end_byte]
            arrays = [n for n in ancestors if n.type == "array_declarator"]
            element = self.element_size(binding) if self.type_text(binding) != "auto" else None
            if arrays and binding["parameter_index"] is None:
                extent = element
                for arr in arrays:
                    size = arr.child_by_field_name("size")
                    extent = _product(extent, self.value(size, arr, depth + 1)) if size else None
                initializer = next((n.child_by_field_name("value") for n in ancestors
                                    if n.type == "init_declarator"), None)
                string = _literal_bytes(data, initializer)
                length = None
                if string is not None and element and element.number == 1:
                    if extent is None and any(n.child_by_field_name("size") is None for n in arrays):
                        extent = _Amount(len(string) + 1)
                    if extent and extent.number is not None and extent.number > len(string):
                        length = _Amount(len(string.split(b"\0", 1)[0]))
                return dict(object=key, capacity=extent, offset=_Amount(), element=element,
                            length=length, definition=binding["statement"],
                            reason=None if extent else "array_extent_unknown")
            definition = self.definition(node, at)
            if definition is not None:
                site, _, rhs, operator = definition
                if operator == "=" and rhs is not None:
                    value = self.value(rhs, site, depth + 1)
                    if value is not None and value.number == 0:
                        return dict(object=None, capacity=None, offset=_Amount(), element=element,
                                    length=None, definition=site, reason="null_pointer")
                    base = self.buffer(rhs, site, depth + 1)
                    if base is not None:
                        return dict(base, element=element or base["element"])
                    # An opaque return denotes an unknown object, not its size.
                    if _unwrap(rhs) is not None and _unwrap(rhs).type == "call_expression":
                        return dict(object=f"call@{rhs.start_byte}", capacity=None, offset=_Amount(),
                                    element=element, length=None, definition=site, reason="callee_extent_unknown")
                return self.fail(node, self.unknown.get(node.id, "pointer_redefinition_unknown"))
            if node.id in self.unknown:
                return None
            return dict(object=key, capacity=None, offset=_Amount(), element=element, length=None,
                        definition=binding["statement"], reason="array_parameter_decay" if arrays else "caller_extent_unknown")
        if node.type == "field_expression":
            key = self.key(node)
            if key:
                field = _text(data, node.child_by_field_name("field"))
                # Without project-level points-to information, a write via a
                # different base may alias this field. Do not assert stability.
                if any(other != key and write.end_byte <= at.start_byte and
                       not _conflict(self.context(write), self.context(at))
                       for other, write in self.field_writes[field]):
                    return self.fail(node, "field_alias_write_unknown")
                definition = self.definition(node, at)
                if definition is not None:
                    site, _, rhs, op = definition
                    base_key = self.key(node.child_by_field_name("argument"))
                    if any(site.end_byte < write.end_byte <= at.start_byte
                           for write, _, _, _ in self.writes.get(base_key, [])):
                        return self.fail(node, "field_base_redefined")
                    base = self.buffer(rhs, site, depth + 1) if rhs is not None and op == "=" else None
                    if base is not None:
                        return base
                    if rhs is not None and _unwrap(rhs) is not None and _unwrap(rhs).type == "call_expression":
                        return dict(object=f"call@{rhs.start_byte}", capacity=None, offset=_Amount(),
                                    element=None, length=None, definition=site, reason="callee_extent_unknown")
                    return self.fail(node, "field_redefinition_unknown")
                if node.id in self.unknown:
                    return None
                parent = self.buffer(node.child_by_field_name("argument"), at, depth + 1)
                if parent is None or parent["object"] is None or parent["offset"] is None:
                    return self.fail(node, "field_base_object_unknown")
                # Field identity follows the pointed-to object and offset, not
                # the spelling of a local pointer that may be reassigned.
                key = str((parent["object"], parent["offset"].constant, parent["offset"].terms,
                           _text(data, node.child_by_field_name("field"))))
                return dict(object=key, capacity=None, offset=_Amount(), element=None, length=None,
                            definition=node, reason="field_extent_unknown")
        if node.type == "pointer_expression":
            argument = node.child_by_field_name("argument")
            if _text(data, node).lstrip().startswith("&"):
                if argument is not None and argument.type == "subscript_expression":
                    base, index = _subscript(argument)
                    obj = self.buffer(base, at, depth + 1)
                    offset = _product(self.value(index, at, depth + 1), obj["element"]) if obj else None
                    return dict(obj, offset=_combine(obj["offset"], offset)) if obj and offset else None
                if argument is not None and argument.type == "identifier":
                    binding = self.binding(argument)
                    if binding is not None:
                        declarators = [n for n in self.parents(binding["declaration"])
                                       if n.start_byte >= binding["statement"].start_byte
                                       and n.end_byte <= binding["statement"].end_byte]
                        if any(n.type == "array_declarator" for n in declarators) and binding["parameter_index"] is None:
                            array = self.buffer(argument, at, depth + 1)
                            return dict(array, element=array["capacity"]) if array else None
                        if (any(n.type == "pointer_declarator" for n in declarators) or
                                any(n.type == "array_declarator" for n in declarators) and binding["parameter_index"] is not None):
                            return dict(object=f"pointer_slot@{binding['declaration'].start_byte}",
                                        capacity=None, offset=_Amount(), element=None, length=None,
                                        definition=binding["statement"], reason="pointer_slot_size_unknown")
                        return dict(object=self.key(argument), capacity=self.element_size(binding),
                                    offset=_Amount(), element=self.element_size(binding), length=None,
                                    definition=binding["statement"], reason=None)
                return self.buffer(argument, at, depth + 1)
        if node.type == "binary_expression":
            left, right = node.child_by_field_name("left"), node.child_by_field_name("right")
            op = _text(data, node.child_by_field_name("operator"))
            obj = self.buffer(left, at, depth + 1) if op in ("+", "-") else None
            delta = _product(self.value(right, at, depth + 1), obj["element"]) if obj else None
            if obj and delta is not None:
                return dict(obj, offset=_combine(obj["offset"], delta, -1 if op == "-" else 1))
        if node.type == "call_expression" and _callee(data, node) in ALLOC_APIS:
            name = _callee(data, node)
            args = node.child_by_field_name("arguments").named_children
            size = None
            if args:
                if name == "calloc" and len(args) == 2:
                    size = _product(self.value(args[0], at, depth + 1), self.value(args[1], at, depth + 1))
                else:
                    pos = 1 if name == "realloc" else 0
                    size = self.value(args[pos], at, depth + 1) if len(args) > pos else None
            return dict(object=f"allocation@{node.start_byte}", capacity=size, offset=_Amount(),
                        element=_Amount(1), length=None, definition=node,
                        reason=None if size else "allocation_extent_unknown")
        if node.type == "new_expression":
            typ = _text(data, node.child_by_field_name("type"))
            element = self.element_size(type_text=typ)
            decl = node.child_by_field_name("declarator")
            counts = list(decl.named_children) if decl is not None else []
            size = _product(element, self.value(counts[0], at, depth + 1)) if len(counts) == 1 else (
                element if not counts else None)
            return dict(object=f"allocation@{node.start_byte}", capacity=size, offset=_Amount(),
                        element=element, length=None, definition=node,
                        reason=None if size else "new_extent_unknown")
        return self.fail(node, "buffer_expression_unknown")

    def string_length(self, buf, at):
        # Track only exact writes. Unknown writes through aliases invalidate a
        # recorded length instead of silently reusing the initializer.
        definition = buf["definition"]
        current, changed = buf["length"], False
        for node in self.memory_events:
            if not definition.end_byte < node.start_byte < at.start_byte:
                continue
            if node.type in ("assignment_expression", "update_expression"):
                target = node.child_by_field_name("left") or node.child_by_field_name("argument")
                if target is not None and target.type in ("subscript_expression", "pointer_expression"):
                    base = target.child_by_field_name("argument")
                    obj = self.buffer(base, node, 1)
                    if obj and obj["object"] == buf["object"]:
                        current, changed = None, True
            if node.type == "call_expression" and _callee(self.data, node) not in (
                    "strlen", "strcmp", "strncmp", "memcmp", "write", "send"):
                args = node.child_by_field_name("arguments")
                values = list(args.named_children) if args else []
                name = _callee(self.data, node)
                if name in COPY_APIS:
                    destinations = values[1:2] if name in ("read", "recv") else values[:1]
                else:
                    destinations = values
                destinations = [self.buffer(arg, node, 1) for arg in destinations
                                if arg.type not in ("call_expression", "number_literal", "string_literal")]
                touched = [value for value in destinations if value and value["object"] == buf["object"]]
                if touched:
                    current, changed = None, True
                    if not _prefix(self.context(node), self.context(at)):
                        continue
                    # A full strcpy from a literal re-establishes a known string.
                    literal = _literal_bytes(self.data, values[1]) if len(values) > 1 else None
                    if (name == "strcpy" and literal is not None and touched[0]["offset"].number == 0
                            and buf["capacity"] is not None and buf["capacity"].number is not None
                            and len(literal.split(b"\0", 1)[0]) + 1 <= buf["capacity"].number):
                        current = _Amount(len(literal.split(b"\0", 1)[0]))
        if current is not None and buf["offset"].number == 0:
            return current
        if not changed and buf["object"] is not None and buf["offset"].number == 0:
            return _Amount(terms=((f"strlen({buf['object']})", 1),))
        return None

    def cfg_path(self, first, last, blocked=()):
        if self.graph is None or self.graph.get("status") != "ok":
            return None
        spans = self.graph["nodes"]
        def anchors(node):
            matches = [i for i in self.cfg_nodes if spans[i]["span"][0] <= node[0] < spans[i]["span"][1]]
            if not matches:
                return []
            width = min(spans[i]["span"][1] - spans[i]["span"][0] for i in matches)
            return [i for i in matches if spans[i]["span"][1] - spans[i]["span"][0] == width]
        starts, stops = anchors(first), set(anchors(last))
        if not starts or not stops:
            return None
        forbidden = {i for span in blocked for i in anchors(span)} - set(starts) - stops
        cache = (tuple(starts), tuple(sorted(stops)), tuple(sorted(forbidden)))
        if cache not in self.path_cache:
            pending, seen = deque(starts), set(starts)
            found = False
            while pending:
                current = pending.popleft()
                if current in stops:
                    found = True; break
                for nxt in self.cfg[current]:
                    if nxt not in seen and nxt not in forbidden:
                        seen.add(nxt); pending.append(nxt)
            self.path_cache[cache] = found
        return self.path_cache[cache]


def extract_patterns(source, *, language="c", graph=None, diagnostics=None):
    """One matcher for extraction and audit, with explicit per-rule outcomes."""
    data = source.encode("utf-8")
    tree = parser_for(language).parse(data)
    nodes, char = list(walk(tree.root_node)), byte_character_map(source)
    facts = _Facts(data, nodes, graph)
    patterns, seen = [], set()
    counts = {kind: Counter() for kind in PATTERN_TYPES}
    traces = []

    def span(node):
        return [char[node.start_byte], char[node.end_byte]]

    def emit(kind, anchor, related=(), *, status="candidate", detail="", evidence=None):
        a = span(anchor)
        links = sorted({tuple(span(n)) for n in related if n is not None and span(n) != a})
        key = (kind, tuple(a), tuple(links))
        if key in seen:
            return
        seen.add(key)
        patterns.append(dict(kind=kind, anchor=a, related=[list(s) for s in links],
                             status=status, detail=detail, evidence=evidence or {}))

    def outcome(kind, node, result, reason, *, related=(), evidence=None):
        missing = sum(n is not None and n.start_byte == n.end_byte for n in related)
        related = [n for n in related if n is not None and n.start_byte < n.end_byte]
        if missing:
            counts[kind]["missing_syntax_operands"] += missing
            evidence = dict(evidence or {}, missing_syntax_operands=missing)
        counts[kind]["sites"] += 1
        counts[kind][result] += 1
        counts[kind][result + ":" + reason] += 1
        if diagnostics is not None:
            traces.append(dict(kind=kind, anchor=span(node), line=node.start_point.row + 1,
                               outcome=result, reason=reason, evidence=evidence or {}))
        if result == "matched":
            emit(kind, node, related, detail=reason, evidence=evidence,
                 status="constant_comparison" if evidence and evidence.get("certainty") == "constant_relation" else "candidate")

    calls = [(n, _callee(data, n), list(n.child_by_field_name("arguments").named_children))
             for n in nodes if n.type == "call_expression" and n.child_by_field_name("arguments") is not None]
    guards = [(n, n.child_by_field_name("condition")) for n in nodes
              if n.type in ("if_statement", "for_statement", "while_statement", "do_statement")
              and n.child_by_field_name("condition") is not None]

    def guard_for(node):
        selected = []
        for check, cond in guards:
            field = "consequence" if check.type == "if_statement" else "body"
            body = check.child_by_field_name(field)
            if body is not None and body.start_byte <= node.start_byte < body.end_byte:
                selected.append(cond)
        return selected

    def nonnegative(index, at):
        """Only a required conjunction or an early-return rejection is a bound."""
        def proves(cond, truth=True):
            cond = _unwrap(cond)
            if cond is None:
                return False
            op = _text(data, cond.child_by_field_name("operator"))
            if cond.type == "binary_expression" and op == "&&" and truth:
                return any(proves(cond.child_by_field_name(f)) for f in ("left", "right"))
            if cond.type == "binary_expression" and op in (">=", ">", "<", "<="):
                left, right = cond.child_by_field_name("left"), cond.child_by_field_name("right")
                if facts.key(left) == facts.key(index) and facts.key(index):
                    bound = facts.value(right, at)
                    if bound and bound.number is not None:
                        return ((op == ">=" and bound.number >= 0) or (op == ">" and bound.number >= -1)) if truth else (
                            (op == "<" and bound.number <= 0) or (op == "<=" and bound.number <= -1))
            return False
        for cond in guard_for(at):
            if proves(cond):
                return cond
        block = at
        while block.parent is not None and block.parent.type != "compound_statement":
            block = block.parent
        if block.parent is not None:
            for sibling in block.parent.named_children:
                if sibling.end_byte > block.start_byte:
                    break
                if sibling.type == "if_statement" and sibling.child_by_field_name("alternative") is None:
                    body = sibling.child_by_field_name("consequence")
                    returns = body and (body.type == "return_statement" or (
                        body.type == "compound_statement" and body.named_children and
                        body.named_children[-1].type == "return_statement"))
                    cond = sibling.child_by_field_name("condition")
                    if returns and proves(cond, False):
                        return cond
        return None

    for node in nodes:
        if node.type == "pointer_expression" and _text(data, node).lstrip().startswith("*"):
            outcome("pointer_access", node, "matched", "pointer_dereference")
        if node.type == "field_expression" and any(c.type == "->" for c in node.children):
            outcome("pointer_access", node, "matched", "indirect_member_access")
        if node.type == "subscript_expression":
            outcome("memory_access", node, "matched", "indexed_access")
            base, index = _subscript(node)
            writable = _write_mode(node)
            if writable is None:
                counts["memory_access"]["address_only"] += 1
                continue
            kind = "negative_index_write" if writable else "negative_index_read"
            if index is None:
                outcome(kind, node, "unknown", "subscript_ast_unsupported"); continue
            val = facts.value(index)
            binding = facts.binding(_unwrap(index))
            typ = facts.type_text(binding) if binding else ""
            declaration_prefix = data[binding["statement"].start_byte:binding["declaration"].start_byte].decode() if binding else ""
            signed = bool(re.search(r"\b(?:int|short|long|ssize_t|ptrdiff_t)\b", typ)) and not (
                re.search(r"\bunsigned\b", declaration_prefix)
                or re.search(r"\b(?:size_t|uint\d+_t)\b", typ))
            bound = nonnegative(index, node)
            if bound is not None:
                outcome(kind, node, "filtered", "nonnegative_guard")
                outcome("guarded_operation", node, "matched", "required_nonnegative_condition", related=(bound, index))
            elif val is not None and val.number is not None:
                outcome(kind, node, "matched" if val.number < 0 else "filtered",
                        "negative_constant" if val.number < 0 else "nonnegative_constant", related=(index,))
            elif signed:
                outcome(kind, node, "unknown", facts.unknown.get(index.id, "signed_index_range_unknown"), evidence=dict(
                    certainty="unknown", path_feasibility="unknown"))
            else:
                outcome(kind, node, "unknown", facts.unknown.get(index.id, "index_signedness_or_range_unknown"))

    for check, cond in guards:
        if any(n.type == "binary_expression" and _text(data, n.child_by_field_name("operator")) in (
                "<", ">", "<=", ">=", "==", "!=") for n in walk(cond)):
            outcome("guard", cond, "matched", "syntactic_comparison")

    frees = [(n, args[0]) for n, name, args in calls if name in FREE_APIS and args]
    for node in nodes:
        if node.type == "new_expression":
            outcome("allocation", node, "matched", "cpp_new", related=node.named_children,
                    evidence=dict(certainty="syntax_relation"))
        elif node.type == "delete_expression" and node.named_children:
            frees.append((node, node.named_children[-1]))
    for node, name, args in calls:
        if name in ALLOC_APIS:
            outcome("allocation", node, "matched", name, related=args[:2],
                    evidence=dict(certainty="syntax_relation"))
        if name in READ_APIS:
            outcome("sensitive_read", node, "matched", name, related=args[:3])
        if name in WRITE_APIS:
            outcome("sensitive_write", node, "matched", name, related=args[:3])
    frees.sort(key=lambda v: v[0].start_byte)
    lifetime_sources = Counter()
    for free, arg in frees:
        outcome("release", free, "matched", "deallocation_call", related=(arg,),
                evidence=dict(certainty="syntax_relation"))
        obj = facts.buffer(arg, free)
        source_key = facts.key(arg)
        lifetime_sources["null" if obj and obj["object"] is None else "resolved" if obj else "unknown"] += 1
        uses = [(n, a, "double_free") for n, a in frees if n.start_byte > free.end_byte]
        for n in nodes:
            if n.start_byte <= free.end_byte:
                continue
            if n.type == "subscript_expression":
                base, _ = _subscript(n); uses.append((n, base, "use_after_free"))
            elif n.type in ("pointer_expression", "field_expression") and (
                _text(data, n).lstrip().startswith("*") or
                n.type == "field_expression" and any(c.type == "->" for c in n.children)):
                uses.append((n, n.child_by_field_name("argument"), "use_after_free"))
        for use, base, kind in uses:
            if obj is None or obj["object"] is None:
                if source_key and source_key == facts.key(base):
                    outcome(kind, use, "filtered" if obj else "unknown",
                            "null_release_noop" if obj else "released_object_identity_unknown")
                continue
            other = facts.buffer(base, use)
            if other is None or other["object"] is None:
                if source_key and source_key == facts.key(base):
                    outcome(kind, use, "unknown", "use_object_identity_unknown")
                continue
            if other["object"] != obj["object"]:
                continue
            if _conflict(facts.context(free), facts.context(use)):
                outcome(kind, use, "filtered", "mutually_exclusive_branches"); continue
            if _return_before(free, use):
                outcome(kind, use, "filtered", "unconditional_return_before_use"); continue
            keys = {facts.key(arg), facts.key(base)} - {None}
            blocked = [(n.start_byte, n.end_byte) for key in keys
                       for n, _, _, _ in facts.writes[key]]
            reachable = facts.cfg_path(span(free), span(use), [
                [char[a], char[b]] for a, b in blocked])
            evidence = dict(certainty="heuristic_candidate", object=obj["object"],
                            cfg_path=reachable, alias_scope="local bindings only", path_feasibility="unknown")
            if reachable is False:
                outcome(kind, use, "filtered", "no_redefinition_free_cfg_path", evidence=evidence)
            elif reachable is None:
                reason = "joern_parse_unavailable" if graph is None or graph.get("status") != "ok" else "cpg_site_alignment_unavailable"
                outcome(kind, use, "unknown", reason, evidence=evidence)
            else:
                outcome(kind, use, "matched", "same_object_cfg_path", related=(free, arg),
                        evidence=evidence)

    for node, name, args in calls:
        if name not in COPY_APIS | {"write", "send"} or len(args) < 2:
            continue
        src, size_node = None, args[2] if len(args) > 2 else None
        if name in ("write", "send"):
            dst, src = None, args[1]
        elif name in ("read", "recv"):
            dst = args[1]
        elif name == "fread":
            dst, size_node = args[0], None
        else:
            dst = args[0]
            src = args[1] if name not in ("memset", "wmemset") else None
        dest = facts.buffer(dst, node)
        source_buf = facts.buffer(src, node) if src is not None else None
        count = facts.value(size_node, node) if size_node is not None else None
        source_length = facts.string_length(source_buf, node) if source_buf and name in (
            "strcpy", "wcscpy", "strncpy", "strcat", "strncat") else None
        if name == "fread" and len(args) >= 3:
            count = _product(facts.value(args[1], node), facts.value(args[2], node))
        elif name in ("strcpy", "wcscpy"):
            count = _combine(source_length, _Amount(1))
        elif name in ("strcat", "strncat"):
            dest_length = facts.string_length(dest, node) if dest else None
            if name == "strncat":
                if count and source_length and count.number is not None and source_length.number is not None:
                    source_length = _Amount(min(count.number, source_length.number))
                else:
                    source_length = None
            count = _combine(_combine(dest_length, source_length), _Amount(1))
        # Wide APIs count wchar_t elements, not bytes; never assume its ABI size.
        if name in ("wmemcpy", "wmemmove", "wmemset", "wcscpy"):
            count = _product(count, dest["element"]) if dest else None
        evidence = dict(
            count=count.describe() if count else None,
            destination_capacity=dest["capacity"].describe() if dest and dest["capacity"] else None,
            source_capacity=source_buf["capacity"].describe() if source_buf and source_buf["capacity"] else None,
            destination=dest["object"] if dest else None, source=source_buf["object"] if source_buf else None,
            assumptions=["successful valid object access", "symbolic arithmetic requires no size_t wrap"],
            definitions=[[char[a], char[b]] for a, b in count.definitions] if count else [],
        )
        for kind, obj, requested in (("buffer_size_mismatch", dest, count),
                                     ("off_by_one", dest, count),
                                     ("buffer_overread", source_buf, count)):
            if kind != "buffer_overread" and dst is None:
                continue
            if kind == "buffer_overread" and src is None:
                continue
            if name in ("strcat", "strncat") and kind == "buffer_overread":
                requested = _combine(source_length, _Amount(1))
            if name == "strncpy" and kind == "buffer_overread":
                # strncpy pads the destination after the first NUL; it does not
                # read n source bytes when the source string is shorter.
                if requested is not None and requested.number == 0:
                    requested = _Amount()
                elif requested and source_length and requested.number is not None and source_length.number is not None:
                    requested = _Amount(min(requested.number, source_length.number + 1))
                else:
                    requested = None
            reason = None
            if node.has_error:
                reason = "call_expression_parse_error"
            elif obj is None:
                operand = src if kind == "buffer_overread" else dst
                reason = facts.unknown.get(operand.id, "object_identity_unknown")
            elif obj["object"] is None:
                reason = "null_object"
            elif obj["capacity"] is None:
                reason = obj["reason"] or "object_extent_unknown"
            elif requested is None:
                operand = size_node if size_node is not None else node
                reason = facts.unknown.get(operand.id, "access_length_unknown")
            elif requested.number is not None and requested.number < 0:
                reason = "negative_length_conversion_unknown"
            remaining = _combine(obj["capacity"], obj["offset"], -1) if obj else None
            difference = _combine(requested, remaining, -1)
            lower, upper = _bounds(difference)
            if reason is None:
                if difference is None:
                    decidable = False
                elif kind == "off_by_one":
                    decidable = difference.number is not None or (lower is not None and lower > 1) or (upper is not None and upper < 1)
                else:
                    decidable = (lower is not None and lower > 0) or (upper is not None and upper <= 0)
                if not decidable:
                    reason = "noncomparable_symbolic_extents"
            if reason:
                outcome(kind, node, "unknown", reason, evidence=evidence); continue
            excess = difference.number
            matched = excess == 1 if kind == "off_by_one" else lower is not None and lower > 0
            counts[kind]["resolved_extents"] += 1
            outcome(kind, node, "matched" if matched else "filtered",
                    "extent_exceeded_by_one" if matched and kind == "off_by_one" else (
                        "extent_exceeded" if matched else "not_exceeding_extent" if upper is not None and upper <= 0 else "excess_not_one"),
                    related=(dst, src, size_node, obj["definition"]), evidence=dict(
                        evidence, excess=excess, excess_lower_bound=lower, excess_upper_bound=upper,
                        certainty="constant_relation" if
                        requested.number is not None and remaining.number is not None else "symbolic_relation"))
        for kind, obj, measurement in (("source_size_copy", source_buf, "source"),
                                        ("destination_size_copy", dest, "destination")):
            if dst is None:
                continue
            if kind == "source_size_copy" and src is None:
                continue
            derived = bool(count and obj and any(key == obj["object"] for _, key in count.origins))
            if name in ("strcpy", "wcscpy", "strcat", "strncat") and kind == "source_size_copy" and src is not None:
                derived = count is not None
            outcome(kind, node, "matched" if derived else "unknown" if count is None else "filtered",
                    "length_derived_from_" + measurement if derived else
                    "length_origin_unknown" if count is None else "different_length_origin",
                    related=(src if kind == "source_size_copy" else dst, size_node), evidence=evidence)
        for guard in guard_for(node):
            names = {facts.key(n) for n in walk(guard) if n.type == "identifier"} - {None}
            operands = {facts.key(n) for arg in args for n in walk(arg) if n.type == "identifier"} - {None}
            if names & operands:
                outcome("guarded_operation", node, "matched", "shared_binding_in_controlling_condition",
                        related=(guard, dst), evidence=dict(adequacy="unknown"))

    patterns.sort(key=lambda p: (p["anchor"][0], p["kind"], p["anchor"][1]))
    if diagnostics is not None:
        diagnostics.update(tree_has_error=tree.root_node.has_error,
                           error_nodes=sum(n.type == "ERROR" or n.is_missing for n in nodes),
                           functions=sum(n.type == "function_definition" for n in nodes),
                           rule_stages={k: dict(v) for k, v in counts.items()},
                           traces=traces, calls=len(calls),
                           lifetime_sources=dict(lifetime_sources),
                           unmodeled_calls=dict(Counter(name for _, name, _ in calls if name and name not in (
                               COPY_APIS | READ_APIS | WRITE_APIS | FREE_APIS | ALLOC_APIS | {"strlen", "strcmp", "strncmp"}))),
                           allocation_calls=sum(name in ALLOC_APIS for _, name, _ in calls),
                           subscript_nodes=sum(n.type == "subscript_expression" for n in nodes))
    return patterns
