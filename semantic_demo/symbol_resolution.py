from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

from .joern import RepositoryCall, RepositoryMethod
from .source import FunctionSource, parse_functions, source_language


@dataclass(frozen=True)
class ResolvedTarget:
    method: RepositoryMethod
    source: FunctionSource | None
    resolution: str


def _normalize_path(path: str) -> str:
    return Path(path).as_posix().lstrip("./")


def _same_directory(left: str, right: str) -> bool:
    return posixpath.dirname(_normalize_path(left)) == posixpath.dirname(
        _normalize_path(right)
    )


def _source_method(source: FunctionSource, resolution: str) -> RepositoryMethod:
    calls = tuple(
        RepositoryCall(
            line=call.line,
            name=call.name,
            method_full_name="",
            dispatch_type="STATIC_DISPATCH",
        )
        for call in source.calls()
        if not call.indirect
    )
    return RepositoryMethod(
        full_name=(
            f"{resolution}:{source.path}:{source.name}:"
            f"{source.start_line}:{len(source.parameters)}"
        ),
        name=source.name,
        path=source.path,
        start_line=source.start_line,
        end_line=source.end_line,
        return_type="ANY" if source.has_value_return() else "void",
        parameters=source.parameters,
        parameter_types=source.parameter_types,
        calls=calls,
    )


def _rank_methods(
    methods: list[RepositoryMethod],
    caller_path: str,
) -> list[RepositoryMethod]:
    if not methods:
        return []
    same_file = [
        method
        for method in methods
        if _normalize_path(method.path) == _normalize_path(caller_path)
    ]
    if len(same_file) == 1:
        return same_file
    same_dir = [method for method in methods if _same_directory(method.path, caller_path)]
    if len(same_dir) == 1:
        return same_dir
    return methods if len(methods) == 1 else []


def _rank_sources(
    sources: list[FunctionSource],
    caller_path: str,
) -> list[FunctionSource]:
    if not sources:
        return []
    same_file = [
        source
        for source in sources
        if _normalize_path(source.path) == _normalize_path(caller_path)
    ]
    if len(same_file) == 1:
        return same_file
    same_dir = [source for source in sources if _same_directory(source.path, caller_path)]
    if len(same_dir) == 1:
        return same_dir
    return sources if len(sources) == 1 else []


def _git_grep_files(index, symbol: str) -> tuple[str, ...]:
    repository = index.repository
    if not hasattr(repository, "_git"):
        return ()
    roots = tuple(dict.fromkeys([*index.scopes, *getattr(index, "context_paths", ())]))
    if not roots:
        return ()
    result = repository._git(
        "grep",
        "-l",
        "-F",
        symbol,
        repository.revision,
        "--",
        *roots,
        check=False,
    )
    if result.returncode not in {0, 1}:
        return ()
    files: list[str] = []
    prefix = repository.revision + ":"
    for raw in result.stdout.splitlines():
        path = raw[len(prefix) :] if raw.startswith(prefix) else raw
        path = _normalize_path(path)
        if Path(path).suffix.lower() not in {
            ".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx",
        }:
            continue
        if path not in files:
            files.append(path)
    return tuple(files)


def _macro_blocks(text: str, name: str):
    lines = text.splitlines()
    pattern = re.compile(
        rf"^\s*#\s*define\s+{re.escape(name)}\s*\(([^)]*)\)\s*(.*)$"
    )
    index = 0
    while index < len(lines):
        match = pattern.match(lines[index])
        if match is None:
            index += 1
            continue
        start = index
        physical = [lines[index]]
        while physical[-1].rstrip().endswith("\\") and index + 1 < len(lines):
            index += 1
            physical.append(lines[index])
        joined = "\n".join(physical)
        first = pattern.match(physical[0])
        if first is None:
            index += 1
            continue
        params = [item.strip() for item in first.group(1).split(",") if item.strip()]
        if any(
            item == "..." or not re.fullmatch(r"[A-Za-z_]\w*", item)
            for item in params
        ):
            index += 1
            continue
        body_parts = [first.group(2)]
        for continuation in physical[1:]:
            body_parts.append(continuation)
        body = " ".join(
            part.rstrip().removesuffix("\\").strip()
            for part in body_parts
        ).strip()
        if body:
            yield start + 1, index + 1, tuple(params), body, joined
        index += 1


def _macro_source(
    path: str,
    translation_unit: str,
    name: str,
    start_line: int,
    end_line: int,
    params: tuple[str, ...],
    body: str,
    language: str,
) -> FunctionSource | None:
    pointer_like = tuple(
        bool(
            re.search(
                rf"(?:\*\s*{re.escape(param)}\b|\b{re.escape(param)}\s*(?:->|\[))",
                body,
            )
        )
        for param in params
    )
    expression_like = (
        ";" not in body
        and not re.match(r"^(?:do\b|\{|if\b|for\b|while\b|switch\b)", body.strip())
    )
    parameters = ", ".join(f"long {param}" for param in params) or "void"
    synthetic = (
        f"long {name}({parameters}) {{ return ({body}); }}"
        if expression_like
        else f"void {name}({parameters}) {{ {body} }}"
    )
    parsed = parse_functions(path, synthetic, language_hint=language)
    if len(parsed) != 1:
        return None
    function = parsed[0]
    return FunctionSource(
        path=path,
        name=name,
        text=function.text,
        translation_unit=translation_unit,
        language=language,
        parameters=params,
        parameter_types=tuple("ANY" for _ in params),
        parameter_pointer_like=pointer_like,
        parameter_signatures=tuple("ANY$" for _ in params),
        start_line=start_line,
        end_line=end_line,
        parse_has_error=function.parse_has_error,
    )


class SymbolResolver:
    """Resolve custom calls without making Joern a single point of failure.

    Resolution order is deliberately conservative:
      1. exact Joern static target;
      2. unique indexed method with matching name/arity and lexical proximity;
      3. unique source function with matching name/arity;
      4. function-like macro definitions.

    Ambiguous fallbacks are left unresolved instead of guessed.
    """

    def __init__(self, index) -> None:
        self.index = index
        self._source_files: dict[str, tuple[str, ...]] = {}
        self._parsed_files: dict[tuple[str, str], list[FunctionSource]] = {}
        self._resolution_cache: dict[
            tuple[str, int, str, str, str, int], tuple[ResolvedTarget, ...]
        ] = {}

    def _parsed(self, path: str, language: str) -> list[FunctionSource]:
        key = (path, language)
        if key not in self._parsed_files:
            try:
                self._parsed_files[key] = parse_functions(
                    path,
                    self.index.repository.read_blob(path),
                    language_hint=language,
                )
            except (ValueError, UnicodeError, FileNotFoundError):
                self._parsed_files[key] = []
        return self._parsed_files[key]

    def sources_for_method(
        self,
        method: RepositoryMethod,
        language: str,
    ) -> list[FunctionSource]:
        base = self.index.repository.function_source(
            path=method.path,
            name=method.name,
            start_line=method.start_line,
            end_line=method.end_line,
            parameters=method.parameters,
            parameter_types=method.parameter_types,
            language_hint=language,
        )
        if base.language != "c":
            return [base]
        parsed = [
            function
            for function in self._parsed(method.path, base.language)
            if function.name == method.name
            and len(function.parameters) == len(method.parameters)
        ]
        if len(parsed) <= 1:
            return [base]
        signatures = {function.parameter_signatures for function in parsed}
        groups = {function.preprocessor_group for function in parsed}
        branches = {function.preprocessor_branch for function in parsed}
        if (
            len(signatures) == 1
            and len(groups) == 1
            and None not in groups
            and None not in branches
            and len(branches) == len(parsed)
        ):
            return parsed
        return [base]

    def _files_for(self, name: str) -> tuple[str, ...]:
        if name not in self._source_files:
            self._source_files[name] = _git_grep_files(self.index, name)
        return self._source_files[name]

    def _source_functions(
        self,
        name: str,
        arity: int,
        caller_path: str,
        language: str,
    ) -> list[ResolvedTarget]:
        matches: list[FunctionSource] = []
        for path in self._files_for(name):
            file_language = source_language(path, language)
            for function in self._parsed(path, file_language):
                if function.name == name and len(function.parameters) == arity:
                    matches.append(function)
        ranked = _rank_sources(matches, caller_path)
        return [
            ResolvedTarget(
                _source_method(source, "source-function"),
                source,
                "source-function",
            )
            for source in ranked
        ]

    def _macros(
        self,
        name: str,
        arity: int,
        caller_path: str,
        language: str,
    ) -> list[ResolvedTarget]:
        matches: list[FunctionSource] = []
        for path in self._files_for(name):
            try:
                text = self.index.repository.read_blob(path)
            except (ValueError, UnicodeError, FileNotFoundError):
                continue
            file_language = source_language(path, language)
            for start, end, params, body, _raw in _macro_blocks(text, name):
                if len(params) != arity:
                    continue
                source = _macro_source(
                    path, text, name, start, end, params, body, file_language
                )
                if source is not None:
                    matches.append(source)
        ranked = _rank_sources(matches, caller_path)
        if not ranked and matches:
            paths = {_normalize_path(source.path) for source in matches}
            if len(paths) == 1:
                ranked = matches
        return [
            ResolvedTarget(
                _source_method(source, "source-macro"),
                source,
                "source-macro",
            )
            for source in ranked
        ]

    def resolve(
        self,
        repository_call: RepositoryCall | None,
        source_call,
        caller_source: FunctionSource,
        inherited_language: str,
    ) -> tuple[ResolvedTarget, ...]:
        name = source_call.name if source_call is not None else (
            repository_call.name if repository_call is not None else ""
        )
        arity = len(source_call.arguments) if source_call is not None else -1
        exact_identity = (
            repository_call.method_full_name if repository_call is not None else ""
        )
        call_line = int(
            source_call.line if source_call is not None else (
                repository_call.line if repository_call is not None else 0
            )
        )
        cache_key = (
            name,
            arity,
            caller_source.path,
            inherited_language,
            exact_identity,
            call_line,
        )
        if cache_key in self._resolution_cache:
            return self._resolution_cache[cache_key]

        if repository_call is not None:
            exact = self.index.callee_methods(repository_call)
            if exact:
                result = tuple(
                    ResolvedTarget(method, None, "joern-exact")
                    for method in exact
                )
                self._resolution_cache[cache_key] = result
                return result

        indexed = [
            method
            for method in self.index.methods().values()
            if method.name == name
            and (arity < 0 or len(method.parameters) == arity)
        ]
        ranked_methods = _rank_methods(indexed, caller_source.path)
        if ranked_methods:
            result = tuple(
                ResolvedTarget(method, None, "joern-name-arity")
                for method in ranked_methods
            )
            self._resolution_cache[cache_key] = result
            return result

        if arity >= 0:
            functions = self._source_functions(
                name, arity, caller_source.path, inherited_language
            )
            if functions:
                result = tuple(functions)
                self._resolution_cache[cache_key] = result
                return result
            macros = self._macros(
                name, arity, caller_source.path, inherited_language
            )
            if macros:
                result = tuple(macros)
                self._resolution_cache[cache_key] = result
                return result

        self._resolution_cache[cache_key] = ()
        return ()
