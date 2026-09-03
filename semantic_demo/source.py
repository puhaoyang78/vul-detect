from __future__ import annotations

import posixpath
import re
import subprocess
import tarfile
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


def _language_for_path(path: str, language_hint: str | None = None) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx"}:
        return "cpp"
    if suffix == ".h" and language_hint in {"c", "cpp"}:
        return language_hint
    return "c"


def source_language(path: str, inherited: str | None = None) -> str:
    return _language_for_path(path, inherited)


@dataclass(frozen=True)
class Call:
    name: str
    arguments: tuple[str, ...]
    line: int
    code: str = ""
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
    parameter_signatures: tuple[str, ...]
    start_line: int
    end_line: int
    parse_has_error: bool = False
    preprocessor_group: tuple[int, int] | None = None
    preprocessor_branch: tuple[int, int] | None = None

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

    def has_path(self, path: str) -> bool:
        result = self._git(
            "cat-file",
            "-e",
            f"{self.revision}:{path}",
            check=False,
        )
        return result.returncode == 0

    def _tree_entry(self, path: str) -> tuple[str, str, str, str]:
        normalized = self._normalize_repository_path(path)
        result = self._git("ls-tree", self.revision, "--", normalized)
        line = result.stdout.rstrip("\n")
        if not line:
            raise FileNotFoundError(
                f"path not found at {self.revision}: {normalized}"
            )
        metadata, listed_path = line.split("\t", 1)
        mode, object_type, object_id = metadata.split()
        return mode, object_type, object_id, listed_path

    def _recursive_tree_entries(
        self, path: str
    ) -> list[tuple[str, str, str, str]]:
        normalized = self._normalize_repository_path(path)
        result = subprocess.run(
            [
                "git",
                f"--git-dir={self.git_dir}",
                "ls-tree",
                "-r",
                "-z",
                self.revision,
                "--",
                normalized,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        entries: list[tuple[str, str, str, str]] = []
        for raw in result.stdout.split(b"\0"):
            if not raw:
                continue
            metadata, raw_path = raw.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode().split()
            entries.append(
                (mode, object_type, object_id, raw_path.decode(errors="replace"))
            )
        return entries

    def _symlink_target(self, path: str) -> str:
        result = self._git("show", f"{self.revision}:{path}")
        target = result.stdout.strip()
        return self._normalize_repository_path(
            posixpath.join(posixpath.dirname(path), target)
        )

    def _symlink_materialization_targets(
        self,
        path: str,
        resolving: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        if path in resolving:
            chain = " -> ".join([*resolving, path])
            raise ValueError(f"repository symlink cycle: {chain}")
        mode, object_type, _, _ = self._tree_entry(path)
        if mode != "120000":
            if mode == "160000" or object_type == "commit":
                raise ValueError(
                    f"repository submodule source is unavailable at {path}"
                )
            return ()
        target = self._symlink_target(path)
        return (
            target,
            *self._symlink_materialization_targets(
                target,
                (*resolving, path),
            ),
        )

    def materialization_paths(self, paths: Iterable[str]) -> tuple[str, ...]:
        selected = [
            self._normalize_repository_path(str(path))
            for path in paths
            if str(path)
        ]
        if not selected:
            raise ValueError("at least one repository path is required")

        resolved: list[str] = []
        queued = list(dict.fromkeys(selected))
        seen: set[str] = set()
        while queued:
            path = queued.pop(0)
            if path in seen:
                continue
            seen.add(path)
            mode, object_type, _, _ = self._tree_entry(path)
            if mode == "160000" or object_type == "commit":
                raise ValueError(
                    f"repository submodule source is unavailable at {path}"
                )
            if path not in resolved:
                resolved.append(path)
            if mode == "120000":
                for target in self._symlink_materialization_targets(path):
                    if target not in seen:
                        queued.append(target)
                continue
            if object_type != "tree":
                continue
            for child_mode, child_type, _, child_path in self._recursive_tree_entries(path):
                if child_mode == "120000":
                    for target in self._symlink_materialization_targets(
                        child_path
                    ):
                        if target not in seen:
                            queued.append(target)
                elif child_mode == "160000" or child_type == "commit":
                    # Nested submodules remain outside the source snapshot.
                    # Calls into them will stay unresolved/opaque.
                    continue
        return tuple(resolved)

    @staticmethod
    def _normalize_repository_path(path: str) -> str:
        normalized = posixpath.normpath(path)
        if (
            posixpath.isabs(normalized)
            or normalized == ".."
            or normalized.startswith("../")
        ):
            raise ValueError(f"repository path escapes root: {path}")
        return normalized

    def _tree_mode(self, path: str) -> str:
        mode, object_type, _, listed_path = self._tree_entry(path)
        if listed_path != path or object_type != "blob":
            raise ValueError(
                f"expected repository blob at {path}, got {listed_path}"
            )
        return mode

    def _read_blob(self, path: str, resolving: frozenset[str]) -> str:
        if path in self._blob_cache:
            return self._blob_cache[path]
        if path in resolving:
            chain = " -> ".join([*sorted(resolving), path])
            raise ValueError(f"repository symlink cycle: {chain}")

        mode = self._tree_mode(path)
        result = self._git("show", f"{self.revision}:{path}")
        if mode == "120000":
            target = result.stdout.strip()
            resolved = self._normalize_repository_path(
                posixpath.join(posixpath.dirname(path), target)
            )
            content = self._read_blob(
                resolved,
                resolving | frozenset({path}),
            )
        else:
            content = result.stdout

        self._blob_cache[path] = content
        return content

    def read_blob(self, path: str) -> str:
        normalized = self._normalize_repository_path(path)
        return self._read_blob(normalized, frozenset())

    def materialize(self, destination: str | Path, paths: Iterable[str]) -> Path:
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        requested = tuple(dict.fromkeys(str(path) for path in paths if str(path)))
        if not requested:
            raise ValueError("at least one repository path is required")
        missing = [path for path in requested if not self.has_path(path)]
        if missing:
            raise FileNotFoundError(
                f"paths not found at {self.revision}: {', '.join(missing)}"
            )
        selected = self.materialization_paths(requested)
        archive = subprocess.Popen(
            [
                "git",
                f"--git-dir={self.git_dir}",
                "archive",
                "--format=tar",
                self.revision,
                "--",
                *selected,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert archive.stdout is not None
        read_error: tarfile.ReadError | None = None
        try:
            with tarfile.open(fileobj=archive.stdout, mode="r|") as tar:
                tar.extractall(target)
        except tarfile.ReadError as error:
            read_error = error
        finally:
            archive.stdout.close()
        stderr = archive.stderr.read().decode(errors="replace") if archive.stderr else ""
        returncode = archive.wait()
        if returncode != 0:
            raise RuntimeError(
                f"git archive failed for {self.revision}: {stderr.strip()}"
            )
        if read_error is not None:
            raise RuntimeError(
                f"git archive produced invalid tar data for {self.revision}: "
                f"{', '.join(selected)}"
            ) from read_error
        return target

    def function_source(
        self,
        *,
        path: str,
        name: str,
        start_line: int,
        end_line: int,
        parameters: tuple[str, ...],
        parameter_types: tuple[str, ...],
        language_hint: str | None = None,
    ) -> FunctionSource:
        translation_unit = self.read_blob(path)
        lines = translation_unit.splitlines(keepends=True)
        if start_line < 1 or end_line < start_line or end_line > len(lines):
            raise ValueError(
                f"invalid Joern source range for {path}:{name}: "
                f"{start_line}-{end_line}"
            )
        text = "".join(lines[start_line - 1 : end_line])
        language = _language_for_path(path, language_hint)
        pointer_like = tuple(
            "*" in type_text or "&" in type_text or "[" in type_text
            for type_text in parameter_types
        )
        signatures = tuple(
            normalize_expression(type_text.replace(parameter, "$", 1))
            if parameter and parameter in type_text
            else normalize_expression(type_text)
            for parameter, type_text in zip(parameters, parameter_types)
        )
        parse_tree = _parser_for_language(language).parse(text.encode())
        return FunctionSource(
            path=path,
            name=name,
            text=text,
            translation_unit=translation_unit,
            language=language,
            parameters=parameters,
            parameter_types=parameter_types,
            parameter_pointer_like=pointer_like,
            parameter_signatures=signatures,
            start_line=start_line,
            end_line=end_line,
            parse_has_error=bool(parse_tree.root_node.has_error),
        )


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
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[bool, ...],
    tuple[str, ...],
]:
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return (), (), (), ()
    function_decl = next(
        (item for item in _walk(declarator) if item.type == "function_declarator"), None
    )
    if function_decl is None:
        return (), (), (), ()
    parameter_list = function_decl.child_by_field_name("parameters")
    if parameter_list is None:
        return (), (), (), ()
    names: list[str] = []
    types: list[str] = []
    pointer_like: list[bool] = []
    signatures: list[str] = []
    for child in parameter_list.named_children:
        if not _node_reliable(child):
            continue
        if child.type not in {"parameter_declaration", "optional_parameter_declaration"}:
            continue
        parameter_decl = child.child_by_field_name("declarator")
        name = _identifier(parameter_decl, source)
        if not name:
            continue
        names.append(name)
        pointer_like.append(
            parameter_decl is not None
            and any(
                item.type in {"pointer_declarator", "array_declarator", "reference_declarator"}
                for item in _walk(parameter_decl)
            )
        )
        identifier_node = next(
            (
                item
                for item in _walk(parameter_decl)
                if item.type == "identifier" and _text(item, source) == name
            ),
            None,
        ) if parameter_decl is not None else None
        if identifier_node is None:
            signature = normalize_expression(_text(child, source))
        else:
            signature_text = (
                source[child.start_byte : identifier_node.start_byte]
                + b"$"
                + source[identifier_node.end_byte : child.end_byte]
            ).decode(errors="replace")
            signature = normalize_expression(signature_text)
        signatures.append(signature)
        types.append(signature.replace("$", ""))
    return (
        tuple(names),
        tuple(types),
        tuple(pointer_like),
        tuple(signatures),
    )


def _function_like_macro_names(root: Node, source: bytes) -> set[str]:
    names: set[str] = set()
    for node in _walk(root):
        if node.type != "preproc_function_def":
            continue
        name = node.child_by_field_name("name")
        if name is not None:
            names.add(_text(name, source))
    return names


def _is_host_function_definition(
    node: Node,
    source: bytes,
    function_like_macros: set[str],
) -> bool:
    current = node.parent
    while current is not None:
        if current.type == "argument_list":
            call = current.parent
            if call is not None and call.type == "call_expression":
                function = call.child_by_field_name("function")
                macro_name = _callee_name(function, source)
                if macro_name in function_like_macros:
                    return False
        if current.type == "translation_unit":
            return True
        current = current.parent
    return False


_PREPROCESSOR_CONDITIONALS = {
    "preproc_if",
    "preproc_ifdef",
    "preproc_ifndef",
    "preproc_elif",
    "preproc_elifdef",
    "preproc_elifndef",
    "preproc_else",
}


def _preprocessor_context(
    node: Node,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    nearest: Node | None = None
    current = node.parent
    while current is not None:
        if current.type in _PREPROCESSOR_CONDITIONALS:
            nearest = current
            break
        current = current.parent
    if nearest is None:
        return None, None

    group = nearest
    while group.type in {
        "preproc_elif",
        "preproc_elifdef",
        "preproc_elifndef",
        "preproc_else",
    }:
        parent = group.parent
        if parent is None or parent.type not in _PREPROCESSOR_CONDITIONALS:
            break
        alternative = parent.child_by_field_name("alternative")
        if not _same_node(alternative, group):
            break
        group = parent

    return (
        (group.start_byte, group.end_byte),
        (nearest.start_byte, nearest.end_byte),
    )


def parse_functions(
    path: str,
    source_text: str,
    *,
    language_hint: str | None = None,
) -> list[FunctionSource]:
    source = source_text.encode()
    language = _language_for_path(path, language_hint)
    tree = _parser_for_language(language).parse(source)
    function_like_macros = _function_like_macro_names(tree.root_node, source)
    functions: list[FunctionSource] = []
    for node in _walk(tree.root_node):
        if node.type != "function_definition":
            continue
        if not _is_host_function_definition(node, source, function_like_macros):
            continue
        name = _function_name(node, source)
        if not name:
            continue
        parameters, types, pointer_like, signatures = _parameters(node, source)
        preprocessor_group, preprocessor_branch = _preprocessor_context(node)
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
                parameter_signatures=signatures,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                parse_has_error=bool(node.has_error),
                preprocessor_group=preprocessor_group,
                preprocessor_branch=preprocessor_branch,
            )
        )
    return functions


def _node_reliable(node: Node | None) -> bool:
    current = node
    while current is not None:
        if current.type == "ERROR" or current.is_missing:
            return False
        if current.type == "compound_statement":
            return True
        current = current.parent
    return True


def function_body_recoverable(function: FunctionSource) -> bool:
    source = function.text.encode()
    tree = _parser_for_language(function.language).parse(source)
    definitions = [
        node
        for node in _walk(tree.root_node)
        if node.type == "function_definition"
    ]
    if len(definitions) == 1:
        return definitions[0].child_by_field_name("body") is not None

    outer_blocks: list[Node] = []
    for node in _walk(tree.root_node):
        if node.type != "compound_statement":
            continue
        parent = node.parent
        nested = False
        while parent is not None:
            if parent.type == "compound_statement":
                nested = True
                break
            parent = parent.parent
        if not nested:
            outer_blocks.append(node)
    return len(outer_blocks) == 1


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
        if declaration.type != "declaration" or not _node_reliable(declaration):
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
        if declaration.type != "declaration" or not _node_reliable(declaration):
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


def _same_node(left: Node | None, right: Node | None) -> bool:
    return (
        left is not None
        and right is not None
        and left.type == right.type
        and left.start_byte == right.start_byte
        and left.end_byte == right.end_byte
    )


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
        if _same_node(right, current) and left is not None:
            return _text(left, source), False
    if parent.type == "init_declarator":
        value = parent.child_by_field_name("value")
        declarator = parent.child_by_field_name("declarator")
        if _same_node(value, current) and declarator is not None:
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
        if item.type != "call_expression" or not _node_reliable(item):
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
                code=_text(item, source),
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
        if not _node_reliable(node):
            continue
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
    if node is None or not _node_reliable(node):
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
    if node is None or not _node_reliable(node):
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
        item.type == "call_expression" and _node_reliable(item)
        for item in _walk(node)
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
    if not _node_reliable(node):
        return None
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
    if not _node_reliable(node):
        return None
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
