from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tree_sitter import Language, Node, Parser
import tree_sitter_c
import tree_sitter_cpp


_C_PARSER = Parser(Language(tree_sitter_c.language()))
_CPP_PARSER = Parser(Language(tree_sitter_cpp.language()))
_SOURCE_EXTENSIONS = (".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx")


def _parser_for_path(path: str) -> Parser:
    suffix = Path(path).suffix.lower()
    return _CPP_PARSER if suffix in {".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx"} else _C_PARSER


@dataclass(frozen=True)
class Call:
    name: str
    arguments: tuple[str, ...]
    line: int
    result: str | None = None
    returned: bool = False


@dataclass(frozen=True)
class MemoryAccess:
    kind: str
    buffer: str
    extent: str
    line: int


@dataclass(frozen=True)
class LocalArray:
    name: str
    elements: str
    element_type: str
    byte_capacity: str | None


@dataclass(frozen=True)
class FunctionSource:
    path: str
    name: str
    text: str
    translation_unit: str
    parameters: tuple[str, ...]
    parameter_types: tuple[str, ...]
    parameter_pointer_like: tuple[bool, ...]
    start_line: int
    parse_has_error: bool = False

    def calls(self) -> list[Call]:
        tree = _parser_for_path(self.path).parse(self.text.encode())
        return _calls(tree.root_node, self.text.encode(), self.start_line)

    def value_relations_before(self, line: int) -> list[tuple[str, str]]:
        source = self.text.encode()
        tree = _PARSER.parse(source)
        return _reaching_value_relations_before(tree.root_node, source, self.start_line, line)

    def continuation_constraints_before(self, line: int) -> list[str]:
        source = self.text.encode()
        tree = _PARSER.parse(source)
        return _continuation_constraints_before(
            tree.root_node, source, self.start_line, line
        )

    def direct_memory_accesses(self) -> list[MemoryAccess]:
        source = self.text.encode()
        tree = _PARSER.parse(source)
        return _direct_memory_accesses(tree.root_node, source, self.start_line)

    def local_arrays(self) -> list[LocalArray]:
        source = self.text.encode()
        tree = _PARSER.parse(source)
        return _local_arrays(tree.root_node, source)


class GitRepository:
    def __init__(self, git_dir: str, revision: str):
        self.git_dir = Path(git_dir)
        self.revision = revision
        if not self.git_dir.is_dir():
            raise FileNotFoundError(f"bare repository not found: {self.git_dir}")
        self._blob_cache: dict[str, str] = {}
        self._function_cache: dict[tuple[str, tuple[str, ...]], list[FunctionSource]] = {}

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", f"--git-dir={self.git_dir}", *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
        )

    def has_revision(self) -> bool:
        result = self._git("cat-file", "-e", f"{self.revision}^{{commit}}", check=False)
        return result.returncode == 0

    def read_blob(self, path: str) -> str:
        if path not in self._blob_cache:
            result = self._git("show", f"{self.revision}:{path}")
            self._blob_cache[path] = result.stdout
        return self._blob_cache[path]

    def find_functions(self, name: str, scopes: Iterable[str] = ()) -> list[FunctionSource]:
        scope_tuple = tuple(scopes)
        cache_key = (name, scope_tuple)
        if cache_key in self._function_cache:
            return self._function_cache[cache_key]

        args = ["grep", "-l", "-F", name, self.revision, "--"]
        args.extend(scope_tuple or tuple(f"*{ext}" for ext in _SOURCE_EXTENSIONS))
        result = self._git(*args, check=False)
        paths: list[str] = []
        prefix = f"{self.revision}:"
        for line in result.stdout.splitlines():
            path = line[len(prefix) :] if line.startswith(prefix) else line
            if path.lower().endswith(_SOURCE_EXTENSIONS):
                paths.append(path)

        matches: list[FunctionSource] = []
        for path in sorted(set(paths)):
            try:
                matches.extend(
                    function
                    for function in parse_functions(path, self.read_blob(path))
                    if function.name == name
                )
            except (subprocess.CalledProcessError, UnicodeError):
                continue
        self._function_cache[cache_key] = matches
        return matches

    def find_function(
        self, name: str, preferred_path: str | None = None, scopes: Iterable[str] = ()
    ) -> FunctionSource | None:
        if preferred_path:
            try:
                preferred = [
                    function
                    for function in parse_functions(
                        preferred_path, self.read_blob(preferred_path)
                    )
                    if function.name == name
                ]
                if len(preferred) == 1:
                    return preferred[0]
                if len(preferred) > 1:
                    return None
            except subprocess.CalledProcessError:
                pass

        functions = self.find_functions(name, scopes)
        if not functions:
            return None
        if preferred_path:
            same_directory = [
                item
                for item in functions
                if Path(item.path).parent == Path(preferred_path).parent
            ]
            if len(same_directory) == 1:
                return same_directory[0]
            if len(same_directory) > 1:
                return None
        return functions[0] if len(functions) == 1 else None


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode(errors="replace")


def _walk(node: Node) -> Iterable[Node]:
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _identifier(node: Node | None, source: bytes) -> str | None:
    if node is None:
        return None
    if node.type in {"identifier", "field_identifier"}:
        return _text(node, source)
    declarator = node.child_by_field_name("declarator")
    if declarator is not None:
        found = _identifier(declarator, source)
        if found:
            return found
    for child in node.children:
        found = _identifier(child, source)
        if found:
            return found
    return None


def _function_name(node: Node, source: bytes) -> str | None:
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return None
    for descendant in _walk(declarator):
        if descendant.type == "function_declarator":
            return _identifier(descendant.child_by_field_name("declarator"), source)
    return _identifier(declarator, source)


def _parameters(
    node: Node, source: bytes
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[bool, ...]]:
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return (), (), ()
    function_decl = next(
        (item for item in _walk(declarator) if item.type == "function_declarator"), None
    )
    if function_decl is None:
        return (), (), ()
    parameter_list = function_decl.child_by_field_name("parameters")
    if parameter_list is None:
        return (), (), ()
    names: list[str] = []
    types: list[str] = []
    pointer_like: list[bool] = []
    for child in parameter_list.named_children:
        if child.type not in {"parameter_declaration", "optional_parameter_declaration"}:
            continue
        parameter_decl = child.child_by_field_name("declarator")
        name = _identifier(parameter_decl, source)
        if not name:
            continue
        names.append(name)
        types.append(_text(child, source))
        pointer_like.append(
            parameter_decl is not None
            and any(
                item.type in {"pointer_declarator", "array_declarator", "reference_declarator"}
                for item in _walk(parameter_decl)
            )
        )
    return tuple(names), tuple(types), tuple(pointer_like)


def parse_functions(path: str, source_text: str) -> list[FunctionSource]:
    source = source_text.encode()
    tree = _parser_for_path(path).parse(source)
    functions: list[FunctionSource] = []
    for node in _walk(tree.root_node):
        if node.type != "function_definition":
            continue
        name = _function_name(node, source)
        if not name:
            continue
        parameters, types, pointer_like = _parameters(node, source)
        functions.append(
            FunctionSource(
                path=path,
                name=name,
                text=_text(node, source),
                translation_unit=source_text,
                parameters=parameters,
                parameter_types=types,
                parameter_pointer_like=pointer_like,
                start_line=node.start_point.row + 1,
                parse_has_error=bool(node.has_error),
            )
        )
    return functions


def _known_element_size(type_text: str, declarator: Node | None) -> int | None:
    if declarator is not None and any(
        node.type == "pointer_declarator" for node in _walk(declarator)
    ):
        return None
    text = normalize_expression(type_text)
    text = re.sub(r"\b(?:const|volatile|restrict|_Atomic)\b", "", text)
    known = {
        "char": 1,
        "signedchar": 1,
        "unsignedchar": 1,
        "int8_t": 1,
        "uint8_t": 1,
        "short": 2,
        "shortint": 2,
        "signedshort": 2,
        "unsignedshort": 2,
        "int16_t": 2,
        "uint16_t": 2,
        "int": 4,
        "signed": 4,
        "signedint": 4,
        "unsigned": 4,
        "unsignedint": 4,
        "float": 4,
        "int32_t": 4,
        "uint32_t": 4,
        "longlong": 8,
        "longlongint": 8,
        "unsignedlonglong": 8,
        "double": 8,
        "int64_t": 8,
        "uint64_t": 8,
    }
    return known.get(text)


def _local_arrays(root: Node, source: bytes) -> list[LocalArray]:
    arrays: list[LocalArray] = []
    for declaration in _walk(root):
        if declaration.type != "declaration":
            continue
        type_node = declaration.child_by_field_name("type")
        if type_node is None:
            continue
        type_text = _text(type_node, source)
        for node in _walk(declaration):
            if node.type != "array_declarator":
                continue
            # Multidimensional arrays require shape-aware indexing; do not
            # publish a partial capacity that could be interpreted as 1-D.
            if (
                (node.parent is not None and node.parent.type == "array_declarator")
                or any(
                    item is not node and item.type == "array_declarator"
                    for item in _walk(node)
                )
            ):
                continue
            size = node.child_by_field_name("size")
            declarator = node.child_by_field_name("declarator")
            name = _identifier(declarator, source)
            if not name or size is None:
                continue
            elements = normalize_expression(_text(size, source))
            if not elements:
                continue
            element_size = _known_element_size(type_text, declarator)
            byte_capacity = (
                f"({elements})*{element_size}" if element_size is not None else None
            )
            arrays.append(
                LocalArray(
                    name=normalize_expression(name),
                    elements=elements,
                    element_type=normalize_expression(type_text),
                    byte_capacity=byte_capacity,
                )
            )
    # If nested scopes reuse an array name, this lightweight representation
    # cannot resolve which declaration reaches a later textual access. Treat the
    # capacity as ambiguous instead of choosing one unsafely.
    by_name: dict[str, list[LocalArray]] = {}
    for array in arrays:
        by_name.setdefault(array.name, []).append(array)
    return [
        declarations[0]
        for declarations in by_name.values()
        if len(declarations) == 1
    ]


def _callee_name(node: Node | None, source: bytes) -> str | None:
    if node is None:
        return None
    if node.type in {"identifier", "field_identifier"}:
        return _text(node, source)
    if node.type in {"field_expression", "field_access_expression"}:
        field = node.child_by_field_name("field")
        return _text(field, source) if field is not None else None
    if node.type in {"qualified_identifier", "scoped_identifier"}:
        name = node.child_by_field_name("name")
        return _text(name, source) if name is not None else _identifier(node, source)
    return None


def _call_context(node: Node, source: bytes) -> tuple[str | None, bool]:
    current = node
    result: str | None = None
    returned = False
    while True:
        parent = current.parent
        if parent is None:
            break
        if parent.type == "return_statement":
            returned = True
            break
        if parent.type == "assignment_expression":
            right = parent.child_by_field_name("right")
            left = parent.child_by_field_name("left")
            if right is not None and left is not None and (
                right.start_byte <= current.start_byte
                and current.end_byte <= right.end_byte
            ):
                result = _text(left, source)
                break
        if parent.type == "init_declarator":
            value = parent.child_by_field_name("value")
            declarator = parent.child_by_field_name("declarator")
            if value is not None and declarator is not None and (
                value.start_byte <= current.start_byte
                and current.end_byte <= value.end_byte
            ):
                result = _identifier(declarator, source) or _text(declarator, source)
                break
        if parent.type in {"expression_statement", "argument_list"}:
            break
        current = parent
    return result, returned


def _calls(node: Node, source: bytes, line_offset: int) -> list[Call]:
    calls: list[Call] = []
    for item in _walk(node):
        if item.type != "call_expression":
            continue
        name = _callee_name(item.child_by_field_name("function"), source)
        arguments_node = item.child_by_field_name("arguments")
        if not name or arguments_node is None:
            continue
        arguments = tuple(_text(arg, source) for arg in arguments_node.named_children)
        result, returned = _call_context(item, source)
        calls.append(
            Call(
                name=name,
                arguments=arguments,
                line=line_offset + item.start_point.row,
                result=result,
                returned=returned,
            )
        )
    return calls


def _is_address_taken(node: Node, source: bytes) -> bool:
    parent = node.parent
    if parent is None or parent.type != "pointer_expression":
        return False
    text = _text(parent, source).lstrip()
    return text.startswith("&")


def _subscript_write_kind(node: Node, source: bytes) -> tuple[str, ...]:
    parent = node.parent
    if parent is None:
        return ("READ",)
    if parent.type == "assignment_expression":
        left = parent.child_by_field_name("left")
        if left is not None and left.start_byte <= node.start_byte and node.end_byte <= left.end_byte:
            between = source[left.end_byte : parent.child_by_field_name("right").start_byte].decode(
                errors="replace"
            ) if parent.child_by_field_name("right") is not None else "="
            return ("WRITE", "READ") if between.strip() != "=" else ("WRITE",)
    if parent.type == "update_expression":
        return ("READ", "WRITE")
    return ("READ",)


def _pointer_dereference_access(node: Node, source: bytes) -> MemoryAccess | None:
    if node.type != "pointer_expression":
        return None
    text = _text(node, source).lstrip()
    if not text.startswith("*"):
        return None
    operand = next((child for child in node.named_children), None)
    if operand is None:
        return None
    kind = _subscript_write_kind(node, source)
    # Compound updates require both read and write; the caller expands the tuple.
    return MemoryAccess(kind=kind[0], buffer=_text(operand, source), extent="1", line=0)


def _direct_memory_accesses(
    root: Node, source: bytes, line_offset: int
) -> list[MemoryAccess]:
    accesses: list[MemoryAccess] = []
    for node in _walk(root):
        if node.type == "subscript_expression":
            if node.parent is not None and node.parent.type == "subscript_expression":
                continue
            if _is_address_taken(node, source):
                continue
            argument = node.child_by_field_name("argument")
            index = node.child_by_field_name("index")
            if argument is None or index is None:
                continue
            base = _text(argument, source)
            offset = _text(index, source)
            for kind in _subscript_write_kind(node, source):
                accesses.append(
                    MemoryAccess(
                        kind=kind,
                        buffer=f"{base}+({offset})",
                        extent="1",
                        line=line_offset + node.start_point.row,
                    )
                )
            continue

        if node.type == "pointer_expression":
            text = _text(node, source).lstrip()
            if not text.startswith("*"):
                continue
            operand = next((child for child in node.named_children), None)
            if operand is None:
                continue
            for kind in _subscript_write_kind(node, source):
                accesses.append(
                    MemoryAccess(
                        kind=kind,
                        buffer=_text(operand, source),
                        extent="1",
                        line=line_offset + node.start_point.row,
                    )
                )
    return accesses


def _absolute_line(node: Node, line_offset: int) -> int:
    return line_offset + node.start_point.row


def _must_terminate(node: Node | None) -> bool:
    """Whether every path through this statement returns from the function."""
    if node is None:
        return False
    if node.type == "return_statement":
        return True
    if node.type == "compound_statement":
        named = node.named_children
        return bool(named) and _must_terminate(named[-1])
    if node.type == "if_statement":
        consequence = node.child_by_field_name("consequence")
        alternative = node.child_by_field_name("alternative")
        return (
            consequence is not None
            and alternative is not None
            and _must_terminate(consequence)
            and _must_terminate(alternative)
        )
    return False


def _invert_comparison(expression: str) -> str | None:
    text = normalize_expression(expression)
    match = re.match(r"^(.*?)(<=|>=|==|!=|<|>)(.*)$", text)
    if not match:
        return None
    left, operator, right = match.groups()
    inverse = {
        ">": "<=",
        ">=": "<",
        "<": ">=",
        "<=": ">",
        "==": "!=",
        "!=": "==",
    }[operator]
    return f"{left}{inverse}{right}"


def _condition_terms(node: Node | None, source: bytes, truth: bool) -> list[str]:
    if node is None:
        return []
    if node.type == "parenthesized_expression":
        child = next((item for item in node.named_children), None)
        return _condition_terms(child, source, truth)
    if node.type == "unary_expression":
        text = _text(node, source).lstrip()
        if text.startswith("!") and node.named_children:
            return _condition_terms(node.named_children[0], source, not truth)
    if node.type == "binary_expression":
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if left is not None and right is not None:
            operator = source[left.end_byte:right.start_byte].decode(errors="replace").strip()
            if operator == "&&":
                return (
                    _condition_terms(left, source, True)
                    + _condition_terms(right, source, True)
                    if truth
                    else []
                )
            if operator == "||":
                return (
                    _condition_terms(left, source, False)
                    + _condition_terms(right, source, False)
                    if not truth
                    else []
                )
    compact = normalize_expression(_text(node, source))
    if not re.match(r"^.*?(<=|>=|==|!=|<|>).*?$", compact):
        return []
    if truth:
        return [compact]
    inverted = _invert_comparison(compact)
    return [inverted] if inverted else []


def _node_contains_line(node: Node, line_offset: int, absolute_line: int) -> bool:
    start = line_offset + node.start_point.row
    end = line_offset + node.end_point.row
    return start <= absolute_line <= end


def _branch_constraint_for_access(
    node: Node, source: bytes, line_offset: int, access_line: int
) -> list[str]:
    condition = node.child_by_field_name("condition")
    if condition is None:
        return []
    consequence = node.child_by_field_name("consequence")
    alternative = node.child_by_field_name("alternative")
    if consequence is not None and _node_contains_line(consequence, line_offset, access_line):
        return _condition_terms(condition, source, True)
    if alternative is not None and _node_contains_line(alternative, line_offset, access_line):
        return _condition_terms(condition, source, False)
    return []


def _continuation_constraints_before(
    root: Node,
    source: bytes,
    line_offset: int,
    access_line: int,
) -> list[str]:
    """Extract only path facts that are structurally implied at the access."""
    constraints: list[str] = []
    for node in _walk(root):
        if node.type == "if_statement":
            if _node_contains_line(node, line_offset, access_line):
                constraints.extend(
                    _branch_constraint_for_access(node, source, line_offset, access_line)
                )
                continue
            if _absolute_line(node, line_offset) >= access_line:
                continue
            condition = node.child_by_field_name("condition")
            consequence = node.child_by_field_name("consequence")
            alternative = node.child_by_field_name("alternative")
            if condition is None:
                continue
            if _must_terminate(consequence):
                constraints.extend(_condition_terms(condition, source, False))
            elif _must_terminate(alternative):
                constraints.extend(_condition_terms(condition, source, True))
            continue

        if node.type in {"for_statement", "while_statement", "do_statement"}:
            if not _node_contains_line(node, line_offset, access_line):
                continue
            condition = node.child_by_field_name("condition")
            if condition is not None:
                constraints.extend(_condition_terms(condition, source, True))

    return list(dict.fromkeys(constraints))


def _simple_assignment(node: Node, source: bytes) -> tuple[str, str] | None:
    if node.type == "expression_statement" and node.named_children:
        node = node.named_children[0]
    if node.type == "assignment_expression":
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if left is None or right is None or left.type != "identifier":
            return None
        return _text(left, source), _text(right, source)
    if node.type == "declaration":
        init_nodes = [
            item for item in node.named_children if item.type == "init_declarator"
        ]
        if len(init_nodes) != 1:
            return None
        init = init_nodes[0]
        declarator = init.child_by_field_name("declarator")
        value = init.child_by_field_name("value")
        name = _identifier(declarator, source)
        if not name or value is None:
            return None
        return name, _text(value, source)
    return None


def _reaching_value_relations_before(
    root: Node,
    source: bytes,
    line_offset: int,
    access_line: int,
) -> list[tuple[str, str]]:
    """Conservative lexical reaching definitions for the access path.

    Only simple assignments/declarations in enclosing sequential blocks are
    considered. Assignments hidden in sibling branches/loops are intentionally
    excluded instead of being merged path-insensitively.
    """
    containers = [
        node
        for node in _walk(root)
        if node.type == "compound_statement"
        and _node_contains_line(node, line_offset, access_line)
    ]
    containers.sort(key=lambda node: (node.end_byte - node.start_byte), reverse=True)

    definitions: dict[str, str] = {}
    for compound in containers:
        for child in compound.named_children:
            if _node_contains_line(child, line_offset, access_line):
                break
            if _absolute_line(child, line_offset) >= access_line:
                break
            relation = _simple_assignment(child, source)
            if relation is not None:
                definitions[relation[0]] = relation[1]

    # Loop initializers dominate accesses in their body.
    for node in _walk(root):
        if node.type != "for_statement" or not _node_contains_line(
            node, line_offset, access_line
        ):
            continue
        initializer = node.child_by_field_name("initializer")
        if initializer is not None:
            relation = _simple_assignment(initializer, source)
            if relation is not None:
                definitions[relation[0]] = relation[1]

    return list(definitions.items())



def normalize_expression(expression: str) -> str:
    expression = re.sub(r"/\*.*?\*/", "", expression, flags=re.S)
    expression = re.sub(r"\s+", "", expression)
    while expression.startswith("(") and expression.endswith(")"):
        depth = 0
        balanced = True
        for index, char in enumerate(expression):
            depth += char == "("
            depth -= char == ")"
            if depth == 0 and index != len(expression) - 1:
                balanced = False
                break
        if not balanced:
            break
        expression = expression[1:-1]
    return expression
