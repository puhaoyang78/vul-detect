from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tree_sitter import Language, Node, Parser
import tree_sitter_c


_LANGUAGE = Language(tree_sitter_c.language())
_PARSER = Parser(_LANGUAGE)


@dataclass(frozen=True)
class Call:
    name: str
    arguments: tuple[str, ...]
    line: int
    result: str | None = None


@dataclass(frozen=True)
class FunctionSource:
    path: str
    name: str
    text: str
    parameters: tuple[str, ...]
    parameter_types: tuple[str, ...]
    start_line: int

    def calls(self) -> list[Call]:
        tree = _PARSER.parse(self.text.encode())
        return _calls(tree.root_node, self.text.encode(), self.start_line)

    def value_relations_before(self, line: int) -> list[tuple[str, str]]:
        source = self.text.encode()
        tree = _PARSER.parse(source)
        return _value_relations_before(tree.root_node, source, self.start_line, line)

    def continuation_constraints_before(self, line: int) -> list[str]:
        source = self.text.encode()
        tree = _PARSER.parse(source)
        return _continuation_constraints_before(
            tree.root_node, source, self.start_line, line
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

        pattern = rf"(^|[^A-Za-z0-9_]){re.escape(name)}[[:space:]]*\("
        args = ["grep", "-l", "-E", pattern, self.revision, "--"]
        args.extend(scope_tuple or ("*.c", "*.h"))
        result = self._git(*args, check=False)
        paths: list[str] = []
        prefix = f"{self.revision}:"
        for line in result.stdout.splitlines():
            path = line[len(prefix) :] if line.startswith(prefix) else line
            if path.endswith((".c", ".h")):
                paths.append(path)

        matches: list[FunctionSource] = []
        for path in sorted(set(paths))[:40]:
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
                for function in parse_functions(preferred_path, self.read_blob(preferred_path)):
                    if function.name == name:
                        return function
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
            if same_directory:
                return same_directory[0]
        return functions[0]


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode(errors="replace")


def _walk(node: Node) -> Iterable[Node]:
    yield node
    for child in node.children:
        yield from _walk(child)


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


def _parameters(node: Node, source: bytes) -> tuple[tuple[str, ...], tuple[str, ...]]:
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return (), ()
    function_decl = next(
        (item for item in _walk(declarator) if item.type == "function_declarator"), None
    )
    if function_decl is None:
        return (), ()
    parameter_list = function_decl.child_by_field_name("parameters")
    if parameter_list is None:
        return (), ()
    names: list[str] = []
    types: list[str] = []
    for child in parameter_list.named_children:
        if child.type != "parameter_declaration":
            continue
        name = _identifier(child.child_by_field_name("declarator"), source)
        if not name:
            continue
        names.append(name)
        types.append(_text(child, source))
    return tuple(names), tuple(types)


def parse_functions(path: str, source_text: str) -> list[FunctionSource]:
    source = source_text.encode()
    tree = _PARSER.parse(source)
    functions: list[FunctionSource] = []
    for node in _walk(tree.root_node):
        if node.type != "function_definition":
            continue
        name = _function_name(node, source)
        if not name:
            continue
        parameters, types = _parameters(node, source)
        functions.append(
            FunctionSource(
                path=path,
                name=name,
                text=_text(node, source),
                parameters=parameters,
                parameter_types=types,
                start_line=node.start_point.row + 1,
            )
        )
    return functions


def _callee_name(node: Node | None, source: bytes) -> str | None:
    if node is None:
        return None
    if node.type == "identifier":
        return _text(node, source)
    if node.type == "field_expression":
        field = node.child_by_field_name("field")
        return _text(field, source) if field is not None else None
    return None


def _call_result(node: Node, source: bytes) -> str | None:
    current = node
    for _ in range(4):
        parent = current.parent
        if parent is None:
            return None
        if parent.type == "assignment_expression":
            right = parent.child_by_field_name("right")
            left = parent.child_by_field_name("left")
            if right is not None and left is not None and (
                right.start_byte <= current.start_byte
                and current.end_byte <= right.end_byte
            ):
                return _text(left, source)
        if parent.type == "init_declarator":
            value = parent.child_by_field_name("value")
            declarator = parent.child_by_field_name("declarator")
            if value is not None and declarator is not None and (
                value.start_byte <= current.start_byte
                and current.end_byte <= value.end_byte
            ):
                name = _identifier(declarator, source)
                return name or _text(declarator, source)
        if parent.type in {"expression_statement", "return_statement", "argument_list"}:
            return None
        current = parent
    return None


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
        calls.append(
            Call(
                name=name,
                arguments=arguments,
                line=line_offset + item.start_point.row,
                result=_call_result(item, source),
            )
        )
    return calls


def _absolute_line(node: Node, line_offset: int) -> int:
    return line_offset + node.start_point.row


def _contains_exit(node: Node | None) -> bool:
    if node is None:
        return False
    return any(
        item.type in {"return_statement", "goto_statement", "break_statement"}
        for item in _walk(node)
    )


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


def _continuation_constraints_before(
    root: Node,
    source: bytes,
    line_offset: int,
    access_line: int,
) -> list[str]:
    constraints: list[str] = []
    for node in _walk(root):
        if node.type != "if_statement":
            continue
        if _absolute_line(node, line_offset) >= access_line:
            continue
        condition = node.child_by_field_name("condition")
        consequence = node.child_by_field_name("consequence")
        alternative = node.child_by_field_name("alternative")
        if condition is None:
            continue
        condition_text = _text(condition, source)
        if _contains_exit(consequence):
            inverted = _invert_comparison(condition_text)
            if inverted:
                constraints.append(inverted)
        elif _contains_exit(alternative):
            compact = normalize_expression(condition_text)
            if re.match(r"^.*?(<=|>=|==|!=|<|>).*?$", compact):
                constraints.append(compact)
    return constraints


def _value_relations_before(
    root: Node,
    source: bytes,
    line_offset: int,
    access_line: int,
) -> list[tuple[str, str]]:
    relations: list[tuple[str, str]] = []
    for node in _walk(root):
        if _absolute_line(node, line_offset) >= access_line:
            continue
        if node.type == "assignment_expression":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is not None and right is not None:
                relations.append((_text(left, source), _text(right, source)))
        elif node.type == "init_declarator":
            declarator = node.child_by_field_name("declarator")
            value = node.child_by_field_name("value")
            if declarator is not None and value is not None:
                name = _identifier(declarator, source)
                if name:
                    relations.append((name, _text(value, source)))
    return relations


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
