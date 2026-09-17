from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Language, Node, Parser
import tree_sitter_c
import tree_sitter_cpp


_C = Parser(Language(tree_sitter_c.language()))
_CPP = Parser(Language(tree_sitter_cpp.language()))


@dataclass(frozen=True)
class ParsedFunction:
    name: str
    parameters: tuple[str, ...]


def parser_for(language: str) -> Parser:
    if language == "c":
        return _C
    if language == "cpp":
        return _CPP
    raise ValueError("language must be c or cpp")


def walk(node: Node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode(errors="replace")


def identifier(node: Node | None, source: bytes) -> str | None:
    if node is None:
        return None
    if node.type == "parenthesized_declarator" and node.parent is not None:
        parent = node.parent
        type_node = parent.child_by_field_name("type")
        if (
            parent.type == "function_definition"
            and type_node is not None
            and type_node.type == "type_identifier"
        ):
            return text(type_node, source)
    if node.type in {"identifier", "destructor_name", "operator_name"}:
        return text(node, source)
    if node.type == "operator_cast":
        parameters = next((n for n in walk(node) if n.type == "parameter_list"), None)
        if parameters is not None:
            return source[node.start_byte:parameters.start_byte].decode().strip()
    if node.type in {"parameter_list", "template_argument_list", "attribute_specifier"}:
        return None
    name = node.child_by_field_name("name")
    if name is not None:
        found = identifier(name, source)
        if found:
            return found
    declarator = node.child_by_field_name("declarator")
    if declarator is not None and declarator is not node:
        found = identifier(declarator, source)
        if found:
            return found
    for child in node.named_children:
        found = identifier(child, source)
        if found:
            return found
    return None


def _parameters(function: Node, source: bytes) -> tuple[str, ...]:
    declarator = function.child_by_field_name("declarator")
    parameter_list = (
        next((node for node in walk(declarator) if node.type == "parameter_list"), None)
        if declarator is not None
        else None
    )
    if parameter_list is None:
        return ()
    names: list[str] = []
    for parameter in parameter_list.named_children:
        if parameter.type not in {"parameter_declaration", "optional_parameter_declaration"}:
            continue
        name = identifier(parameter.child_by_field_name("declarator"), source)
        if name:
            names.append(name)
    return tuple(names)


def parse_function(
    source_text: str,
    language: str = "c",
    function_name: str | None = None,
) -> ParsedFunction:
    source = source_text.encode()
    tree = parser_for(language).parse(source)
    functions = [node for node in walk(tree.root_node) if node.type == "function_definition"]
    if function_name:
        functions = [
            node
            for node in functions
            if identifier(node.child_by_field_name("declarator"), source) in {function_name, function_name.rsplit("::", 1)[-1]}
        ]
    if len(functions) != 1:
        label = f" named {function_name}" if function_name else ""
        raise ValueError(f"expected exactly one function{label}, found {len(functions)}")
    node = functions[0]
    name = identifier(node.child_by_field_name("declarator"), source)
    if not name:
        raise ValueError("function name could not be parsed")
    return ParsedFunction(name=name, parameters=_parameters(node, source))


def single_function_language(source_text: str, file_name: str = "") -> str | None:
    """Infer C vs C++ only when a standalone function can be parsed unambiguously."""
    encoded = source_text.encode("utf-8")
    suffix = Path(file_name).suffix
    languages = (
        ("c",)
        if suffix == ".c"
        else (
            ("cpp",)
            if suffix in {".C", ".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hh", ".hxx"}
            else ("c", "cpp")
        )
    )
    for language in languages:
        root = parser_for(language).parse(encoded).root_node
        if root.has_error:
            continue
        nodes = [node for node in root.named_children if node.type != "comment"]
        if len(nodes) != 1:
            continue
        top = nodes[0]
        while top.type == "template_declaration":
            children = [
                node
                for node in top.named_children
                if node.type not in {"template_parameter_list", "comment"}
            ]
            if len(children) != 1:
                break
            top = children[0]
        if top.type != "function_definition":
            continue
        if top.child_by_field_name("type") is None:
            declarator = top.child_by_field_name("declarator")
            while declarator is not None and declarator.type != "qualified_identifier":
                declarator = declarator.child_by_field_name("declarator")
            if language != "cpp" or declarator is None:
                continue
            scope = declarator.child_by_field_name("scope")
            name = declarator.child_by_field_name("name")
            if scope is None or name is None:
                continue
            while scope.type == "qualified_identifier":
                scope = scope.child_by_field_name("name")
            if scope.type == "template_type":
                scope = scope.child_by_field_name("name")
            owner = encoded[scope.start_byte:scope.end_byte].decode()
            method = encoded[name.start_byte:name.end_byte].decode()
            if name.type != "operator_cast" and method not in {owner, "~" + owner}:
                continue
        if sum(node.type == "function_definition" for node in walk(root)) != 1:
            continue
        if identifier(top.child_by_field_name("declarator"), encoded):
            return language
    return None


def local_identifiers(
    source_text: str,
    language: str,
    function_name: str | None = None,
) -> tuple[str, ...]:
    function = parse_function(source_text, language, function_name)
    source = source_text.encode()
    tree = parser_for(language).parse(source)
    names = [function.name, *function.parameters]
    for node in walk(tree.root_node):
        if node.type != "declaration":
            continue
        for child in node.named_children:
            declarator = (
                child.child_by_field_name("declarator")
                if child.type == "init_declarator"
                else child
            )
            name = identifier(declarator, source)
            if name and name not in names:
                names.append(name)
    return tuple(names)


# Tokens preserve literals and punctuation; comment/whitespace removal cannot turn
# a nested method or a call into the complete source definition.
_SOURCE_TOKEN = re.compile(r'R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delimiter)"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/|[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s]', re.S)


def source_tokens(source: str) -> tuple[str, ...]:
    return tuple(m.group() for m in _SOURCE_TOKEN.finditer(source)
                 if not m.group().startswith(('//', '/*')))


def resolve_language(source: str, language: str, file_name: str = '') -> str:
    suffix = Path(file_name).suffix
    if suffix == '.c':
        return 'c'
    if suffix == '.C' or suffix.lower() in {'.cc', '.cpp', '.cxx', '.c++', '.hpp', '.hh', '.hxx', '.h++'}:
        return 'cpp'
    if language in {'c', 'cpp'}:
        return language
    encoded = source.encode()
    return min(('c', 'cpp'), key=lambda lang: (
        sum(n.is_error or n.is_missing for n in walk(parser_for(lang).parse(encoded).root_node)), lang))


def target_hint(source: str, language: str, supplied_name: str | None = None) -> ParsedFunction:
    """Advisory syntax only. Joern must independently prove the complete span."""
    try:
        parsed = parse_function(source, language, supplied_name)
    except ValueError:
        try:
            parsed = parse_function(source, language)
        except ValueError:
            return ParsedFunction(name='', parameters=())
    if parsed.name in {'if', 'for', 'while', 'switch', 'catch', 'sizeof'}:
        return ParsedFunction(name='', parameters=())
    return parsed
