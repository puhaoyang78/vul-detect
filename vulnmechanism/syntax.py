from __future__ import annotations

from dataclasses import dataclass
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
    if node.type == "identifier":
        return text(node, source)
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
    parameter_list = next(
        (node for node in walk(declarator) if node.type == "parameter_list"),
        None,
    ) if declarator is not None else None
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


def parse_function(source_text: str, language: str = "c", function_name: str | None = None) -> ParsedFunction:
    source = source_text.encode()
    tree = parser_for(language).parse(source)
    functions = [node for node in walk(tree.root_node) if node.type == "function_definition"]
    if function_name:
        matches = []
        for node in functions:
            name = identifier(node.child_by_field_name("declarator"), source)
            if name == function_name:
                matches.append(node)
        functions = matches
    if len(functions) != 1:
        label = f" named {function_name}" if function_name else ""
        raise ValueError(f"expected exactly one function{label}, found {len(functions)}")
    node = functions[0]
    name = identifier(node.child_by_field_name("declarator"), source)
    if not name:
        raise ValueError("function name could not be parsed")
    return ParsedFunction(name=name, parameters=_parameters(node, source))


def local_identifiers(source_text: str, language: str, function_name: str | None = None) -> tuple[str, ...]:
    function = parse_function(source_text, language, function_name)
    source = source_text.encode()
    tree = parser_for(language).parse(source)
    names = [function.name, *function.parameters]
    for node in walk(tree.root_node):
        if node.type != "declaration":
            continue
        for child in node.named_children:
            declarator = child.child_by_field_name("declarator") if child.type == "init_declarator" else child
            name = identifier(declarator, source)
            if name and name not in names:
                names.append(name)
    return tuple(names)
