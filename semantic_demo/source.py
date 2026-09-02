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


def _parser_for_language(language: str) -> Parser:
    return _CPP_PARSER if language == "cpp" else _C_PARSER


def _language_for_path(path: str, source_text: str | None = None) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx"}:
        return "cpp"
    if suffix != ".h" or source_text is None:
        return "c"
    source = source_text.encode()
    c_tree = _C_PARSER.parse(source)
    cpp_tree = _CPP_PARSER.parse(source)
    def error_count(root: Node) -> int:
        return sum(
            1 for node in _walk(root)
            if node.type == "ERROR" or node.is_missing
        )
    return "cpp" if error_count(cpp_tree.root_node) < error_count(c_tree.root_node) else "c"


@dataclass(frozen=True)
class Call:
    name: str
    arguments: tuple[str, ...]
    line: int
    result: str | None = None
    returned: bool = False
    indirect: bool = False


@dataclass(frozen=True)
class MemoryAccess:
    kind: str
    buffer: str
    extent: str
    line: int
    origin: str


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
    language: str
    parameters: tuple[str, ...]
    parameter_types: tuple[str, ...]
    parameter_pointer_like: tuple[bool, ...]
    start_line: int
    parse_has_error: bool = False

    def calls(self) -> list[Call]:
        source = self.text.encode()
        tree = _parser_for_language(self.language).parse(source)
        pointer_callables = {
            parameter
            for parameter, pointer_like in zip(
                self.parameters, self.parameter_pointer_like
            )
            if pointer_like
        }
        pointer_callables.update(_function_pointer_names(tree.root_node, source))
        return _calls(
            tree.root_node,
            source,
            self.start_line,
            pointer_callables,
        )

    def has_indirect_calls(self) -> bool:
        return any(call.indirect for call in self.calls())

    def has_value_return(self) -> bool:
        source = self.text.encode()
        tree = _parser_for_language(self.language).parse(source)
        return any(
            node.type == "return_statement" and bool(node.named_children)
            for node in _walk(tree.root_node)
        )

    def value_relations_before(self, line: int) -> list[tuple[str, str]]:
        source = self.text.encode()
        tree = _parser_for_language(self.language).parse(source)
        return _reaching_value_relations_before(tree.root_node, source, self.start_line, line)

    def direct_call_definitions_before(
        self, line: int
    ) -> list[tuple[str, str, int]]:
        source = self.text.encode()
        tree = _parser_for_language(self.language).parse(source)
        return _reaching_direct_call_definitions_before(
            tree.root_node, source, self.start_line, line
        )

    def continuation_constraints_before(self, line: int) -> list[str]:
        source = self.text.encode()
        tree = _parser_for_language(self.language).parse(source)
        return _continuation_constraints_before(
            tree.root_node, source, self.start_line, line
        )

    def uncertain_control_conditions_before(self, line: int) -> list[str]:
        source = self.text.encode()
        tree = _parser_for_language(self.language).parse(source)
        return _uncertain_control_conditions_before(
            tree.root_node, source, self.start_line, line
        )

    def direct_memory_accesses(self) -> list[MemoryAccess]:
        source = self.text.encode()
        tree = _parser_for_language(self.language).parse(source)
        return _direct_memory_accesses(tree.root_node, source, self.start_line)

    def local_arrays(self) -> list[LocalArray]:
        source = self.text.encode()
        tree = _parser_for_language(self.language).parse(source)
        return _local_arrays(tree.root_node, source)

    def integer_domains(self) -> tuple[set[str], set[str]]:
        source = self.text.encode()
        tree = _parser_for_language(self.language).parse(source)
        return _integer_domains_from_ast(
            tree.root_node,
            source,
            self.parameters,
            self.parameter_types,
        )


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
    if node.type in {"qualified_identifier", "scoped_identifier"}:
        name = node.child_by_field_name("name")
        if name is not None:
            return _identifier(name, source) or _text(name, source)
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
    language = _language_for_path(path, source_text)
    tree = _parser_for_language(language).parse(source)
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
                language=language,
                parameters=parameters,
                parameter_types=types,
                parameter_pointer_like=pointer_like,
                start_line=node.start_point.row + 1,
                parse_has_error=bool(node.has_error),
            )
        )
    return functions


def _integer_domain_from_type(type_text: str) -> str | None:
    if re.search(r"\b(?:unsigned|size_t|uint\d+_t)\b", type_text):
        return "unsigned"
    if re.search(r"\b(?:signed|ssize_t|int\d+_t|char|short|int|long)\b", type_text):
        return "signed"
    return None


def _integer_domains_from_ast(
    root: Node,
    source: bytes,
    parameters: tuple[str, ...],
    parameter_types: tuple[str, ...],
) -> tuple[set[str], set[str]]:
    signed: set[str] = set()
    unsigned: set[str] = set()

    for name, type_text in zip(parameters, parameter_types):
        domain = _integer_domain_from_type(type_text)
        if domain == "unsigned":
            unsigned.add(name)
        elif domain == "signed":
            signed.add(name)

    for declaration in _walk(root):
        if declaration.type != "declaration":
            continue
        type_node = declaration.child_by_field_name("type")
        if type_node is None:
            continue
        domain = _integer_domain_from_type(_text(type_node, source))
        if domain is None:
            continue

        for child in declaration.named_children:
            if child is type_node:
                continue
            declarator = (
                child.child_by_field_name("declarator")
                if child.type == "init_declarator"
                else child
            )
            if declarator is None or any(
                item.type in {"pointer_declarator", "array_declarator", "reference_declarator"}
                for item in _walk(declarator)
            ):
                continue
            name = _identifier(declarator, source)
            if not name:
                continue
            if domain == "unsigned":
                unsigned.add(name)
                signed.discard(name)
            else:
                signed.add(name)
                unsigned.discard(name)

    return signed, unsigned


def _known_element_size(type_text: str, declarator: Node | None) -> int | None:
    if declarator is not None and any(
        node.type == "pointer_declarator" for node in _walk(declarator)
    ):
        return None
    text = normalize_expression(type_text)
    text = re.sub(r"\b(?:const|volatile|restrict|_Atomic)\b", "", text)
    # Only sizes guaranteed by the spelling itself are encoded. ABI-dependent
    # C fundamental types intentionally remain unknown.
    known = {
        "char": 1,
        "signedchar": 1,
        "unsignedchar": 1,
        "int8_t": 1,
        "uint8_t": 1,
        "int16_t": 2,
        "uint16_t": 2,
        "int32_t": 4,
        "uint32_t": 4,
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
    # If nested scopes reuse an array name, this conservative representation
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


def _function_pointer_names(root: Node, source: bytes) -> set[str]:
    names: set[str] = set()
    for node in _walk(root):
        if node.type != "function_declarator":
            continue
        declarator = node.child_by_field_name("declarator")
        if declarator is None:
            continue
        if not any(item.type == "pointer_declarator" for item in _walk(declarator)):
            continue
        name = _identifier(declarator, source)
        if name:
            names.add(name)
    return names


def _callee_name(node: Node | None, source: bytes) -> str | None:
    if node is None:
        return None
    if node.type == "identifier":
        return _text(node, source)
    if node.type in {"qualified_identifier", "scoped_identifier"}:
        name = node.child_by_field_name("name")
        return _text(name, source) if name is not None else _identifier(node, source)
    return None


def _call_context(node: Node, source: bytes) -> tuple[str | None, bool]:
    current = node
    while (
        current.parent is not None
        and current.parent.type in {"parenthesized_expression", "cast_expression"}
    ):
        current = current.parent

    parent = current.parent
    if parent is None:
        return None, False
    if parent.type == "return_statement":
        return None, True
    if parent.type == "assignment_expression":
        right = parent.child_by_field_name("right")
        left = parent.child_by_field_name("left")
        if right is current and left is not None:
            return _text(left, source), False
    if parent.type == "init_declarator":
        value = parent.child_by_field_name("value")
        declarator = parent.child_by_field_name("declarator")
        if value is current and declarator is not None:
            return _identifier(declarator, source) or _text(declarator, source), False
    return None, False


def _calls(
    node: Node,
    source: bytes,
    line_offset: int,
    pointer_callables: set[str],
) -> list[Call]:
    calls: list[Call] = []
    for item in _walk(node):
        if item.type != "call_expression":
            continue
        function_node = item.child_by_field_name("function")
        arguments_node = item.child_by_field_name("arguments")
        if function_node is None or arguments_node is None:
            continue
        direct_name = _callee_name(function_node, source)
        function_text = _text(function_node, source)
        indirect = (
            direct_name is None
            or (
                function_node.type == "identifier"
                and direct_name in pointer_callables
            )
        )
        name = direct_name or function_text
        arguments = tuple(_text(arg, source) for arg in arguments_node.named_children)
        result, returned = _call_context(item, source)
        calls.append(
            Call(
                name=name,
                arguments=arguments,
                line=line_offset + item.start_point.row,
                result=result,
                returned=returned,
                indirect=indirect,
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
                        origin="AST_SUBSCRIPT",
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
                        origin="AST_DEREF",
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


def _contains_call_expression(node: Node | None) -> bool:
    return node is not None and any(
        item.type == "call_expression" for item in _walk(node)
    )


def _uncertain_control_conditions_before(
    root: Node,
    source: bytes,
    line_offset: int,
    access_line: int,
) -> list[str]:
    """Conditions whose preceding branch contains unresolved call effects.

    These are not treated as path facts. They only mark that feasibility may
    depend on a call whose return/termination semantics are unavailable.
    """
    conditions: list[str] = []
    for node in _walk(root):
        if node.type != "if_statement":
            continue
        if _node_contains_line(node, line_offset, access_line):
            continue
        if _absolute_line(node, line_offset) >= access_line:
            continue
        condition = node.child_by_field_name("condition")
        consequence = node.child_by_field_name("consequence")
        alternative = node.child_by_field_name("alternative")
        if condition is None:
            continue
        uncertain = (
            (_contains_call_expression(consequence) and not _must_terminate(consequence))
            or (_contains_call_expression(alternative) and not _must_terminate(alternative))
        )
        if uncertain:
            compact = normalize_expression(_text(condition, source))
            if compact:
                conditions.append(compact)
    return list(dict.fromkeys(conditions))


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


def _direct_call_assignment(
    node: Node,
    source: bytes,
    line_offset: int,
) -> tuple[str, str, int] | None:
    if node.type == "expression_statement" and node.named_children:
        node = node.named_children[0]

    left: Node | None = None
    value: Node | None = None
    if node.type == "assignment_expression":
        left = node.child_by_field_name("left")
        value = node.child_by_field_name("right")
        if left is None or left.type != "identifier":
            return None
    elif node.type == "declaration":
        init_nodes = [
            item for item in node.named_children if item.type == "init_declarator"
        ]
        if len(init_nodes) != 1:
            return None
        init = init_nodes[0]
        left = init.child_by_field_name("declarator")
        value = init.child_by_field_name("value")
    else:
        return None

    if left is None or value is None:
        return None
    target = _identifier(left, source)
    if not target:
        return None

    current = value
    while current.type in {"parenthesized_expression", "cast_expression"}:
        named = current.named_children
        if not named:
            return None
        current = named[-1]
    if current.type != "call_expression":
        return None
    function = current.child_by_field_name("function")
    callee = _callee_name(function, source)
    if callee is None:
        return None
    return target, callee, line_offset + current.start_point.row


def _reaching_direct_call_definitions_before(
    root: Node,
    source: bytes,
    line_offset: int,
    access_line: int,
) -> list[tuple[str, str, int]]:
    containers = [
        node
        for node in _walk(root)
        if node.type == "compound_statement"
        and _node_contains_line(node, line_offset, access_line)
    ]
    containers.sort(key=lambda node: (node.end_byte - node.start_byte), reverse=True)

    definitions: dict[str, tuple[str, int]] = {}
    for compound in containers:
        for child in compound.named_children:
            if _node_contains_line(child, line_offset, access_line):
                break
            if _absolute_line(child, line_offset) >= access_line:
                break
            relation = _direct_call_assignment(child, source, line_offset)
            if relation is not None:
                definitions[relation[0]] = (relation[1], relation[2])
            else:
                simple = _simple_assignment(child, source)
                if simple is not None:
                    definitions.pop(simple[0], None)

    for node in _walk(root):
        if node.type != "for_statement" or not _node_contains_line(
            node, line_offset, access_line
        ):
            continue
        initializer = node.child_by_field_name("initializer")
        if initializer is None:
            continue
        relation = _direct_call_assignment(initializer, source, line_offset)
        if relation is not None:
            definitions[relation[0]] = (relation[1], relation[2])
        else:
            simple = _simple_assignment(initializer, source)
            if simple is not None:
                definitions.pop(simple[0], None)

    return [
        (target, callee, line)
        for target, (callee, line) in definitions.items()
    ]


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
