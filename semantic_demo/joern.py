from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


class JoernError(RuntimeError):
    pass


class JoernMethodNotFound(JoernError):
    """The candidate fragment parsed, but Joern produced no matching method."""


class JoernTimeout(JoernError):
    """Joern exceeded the per-function validation budget."""


@dataclass(frozen=True)
class JoernCall:
    line: int
    name: str
    arguments: dict[int, str]
    call_id: str = ""
    code: str = ""


@dataclass
class JoernFacts:
    parameters: dict[int, tuple[str, str]] = field(default_factory=dict)
    calls: dict[object, JoernCall] = field(default_factory=dict)
    flows: set[tuple[object, ...]] = field(default_factory=set)
    returns: list[str] = field(default_factory=list)
    return_flows: set[int] = field(default_factory=set)

    def call_list(self) -> list[JoernCall]:
        return list(self.calls.values())

    def parameter_reaches(
        self, parameter_index: int, call: JoernCall, argument_index: int
    ) -> bool:
        return (
            bool(call.call_id)
            and (parameter_index, call.call_id, argument_index) in self.flows
        ) or (
            parameter_index,
            call.line,
            call.name,
            argument_index,
        ) in self.flows


@dataclass(frozen=True)
class RepositoryCall:
    line: int
    name: str
    method_full_name: str
    dispatch_type: str
    call_id: str = ""


@dataclass(frozen=True)
class RepositoryMethod:
    full_name: str
    name: str
    path: str
    start_line: int
    end_line: int
    return_type: str
    parameters: tuple[str, ...]
    parameter_types: tuple[str, ...]
    calls: tuple[RepositoryCall, ...]


@functools.cache
def _tool_identity(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_file():
        return f"missing:{path}"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.name == "joern":
        try:
            result = subprocess.run(
                [str(path), "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
                check=False,
            )
            output = (result.stdout or result.stderr).strip()
            if result.returncode == 0 and output:
                return f"{output}:{digest}"
        except (OSError, subprocess.TimeoutExpired):
            pass
    return digest


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_snapshot_identity(repository) -> str:
    methods = (
        repository.materialize,
        repository.materialization_paths,
        repository._symlink_materialization_targets,
        repository._symlink_target,
        repository._tree_entry,
        repository._recursive_tree_entries,
        repository._normalize_repository_path,
    )
    try:
        source = "\n".join(inspect.getsource(method) for method in methods)
    except (OSError, TypeError):
        module_path = Path(inspect.getfile(type(repository)))
        return _file_digest(module_path)
    return hashlib.sha256(source.encode()).hexdigest()


class JoernRepositoryIndex:
    """Build and query one Joern 4 CPG per sample/revision."""

    def __init__(
        self,
        repository,
        sample_key: str,
        scopes: Iterable[str],
        entry_path: str,
        *,
        defines: Iterable[str] = (),
        include_paths: Iterable[str] = (),
        joern_dir: str | Path = "/home/phy/joern",
        java_home: str | Path | None = None,
        cache_dir: str | Path = "data/joern_cpg",
        timeout: int = 900,
    ) -> None:
        self.repository = repository
        self.sample_key = sample_key
        self.scopes = tuple(dict.fromkeys([*map(str, scopes), str(entry_path)]))
        self.entry_path = str(entry_path)
        self.defines = tuple(dict.fromkeys(map(str, defines)))
        self.explicit_include_paths = tuple(
            dict.fromkeys(map(str, include_paths))
        )
        self.context_paths = self._repository_context_paths()
        self.joern_dir = Path(os.environ.get("JOERN_HOME", str(joern_dir))).expanduser()
        self.joern = self.joern_dir / "joern"
        self.java_home = Path(
            java_home or os.environ.get("JAVA_HOME", "/home/phy/jdk21")
        ).expanduser()
        self.java = self.java_home / "bin" / "java"
        self.script = Path(__file__).with_name("joern_index.sc")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self._methods: dict[str, RepositoryMethod] | None = None

        c2cpg = self._c2cpg()
        frontend_identity = {
            "joern": _tool_identity(str(self.joern)),
            "c2cpg": _tool_identity(str(c2cpg)),
        }
        cpg_payload = json.dumps(
            {
                "revision": self.repository.revision,
                "scopes": self.scopes,
                "context_paths": self.context_paths,
                "defines": self.defines,
                "frontend": frontend_identity,
                "source_snapshot": _source_snapshot_identity(self.repository),
                "include_auto_discovery": True,
            },
            sort_keys=True,
        ).encode()
        self.cpg_fingerprint = hashlib.sha256(cpg_payload).hexdigest()[:16]
        index_payload = json.dumps(
            {
                "cpg_fingerprint": self.cpg_fingerprint,
                "index_script": _file_digest(self.script),
                "index_schema": 8,
            },
            sort_keys=True,
        ).encode()
        self.index_fingerprint = hashlib.sha256(index_payload).hexdigest()[:16]
        self.fingerprint = self.index_fingerprint
        self.cpg_path = (
            self.cache_dir / f"{sample_key}-{self.cpg_fingerprint}.bin"
        )
        self.index_path = (
            self.cache_dir / f"{sample_key}-{self.index_fingerprint}.tsv"
        )

    def _repository_context_paths(self) -> tuple[str, ...]:
        roots: list[str] = []

        def add(path: str) -> None:
            if path and path not in roots and self.repository.has_path(path):
                roots.append(path)

        # A repository-level include/ directory is the conventional header root
        # for C/C++ projects and is parse context, not an analysis scope.
        add("include")

        for path in self.explicit_include_paths:
            if not self.repository.has_path(path):
                raise FileNotFoundError(
                    f"configured include path not found at "
                    f"{self.repository.revision}: {path}"
                )
            add(path)

        # Preserve nested include roots already implied by the configured scopes,
        # e.g. src/include/foo -> src/include.
        for scope in self.scopes:
            parts = Path(scope).parts
            for index, part in enumerate(parts):
                if part == "include":
                    add(Path(*parts[: index + 1]).as_posix())
                    break

        return tuple(roots)

    def _c2cpg(self) -> Path:
        candidates = (
            self.joern_dir / "c2cpg.sh",
            self.joern_dir / "joern-cli" / "c2cpg.sh",
            self.joern_dir / "joern-cli" / "frontends" / "c2cpg" / "c2cpg.sh",
        )
        return next((path for path in candidates if path.is_file()), candidates[0])

    @property
    def diagnostics_path(self) -> Path:
        return self.cache_dir / (
            f"{self.sample_key}-{self.cpg_fingerprint}.c2cpg.log"
        )

    def ensure_available(self) -> None:
        if not self.joern.is_file():
            raise JoernError(f"Joern launcher not found at {self.joern}")
        c2cpg = self._c2cpg()
        if not c2cpg.is_file():
            raise JoernError(
                f"Joern c2cpg launcher not found under {self.joern_dir}"
            )
        if not self.java.is_file() or not os.access(self.java, os.X_OK):
            raise JoernError(f"Java executable not found at {self.java}")
        if not self.script.is_file():
            raise JoernError(f"Joern repository index script not found: {self.script}")

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["JAVA_HOME"] = str(self.java_home)
        environment["PATH"] = (
            str(self.java.parent) + os.pathsep + environment.get("PATH", "")
        )
        return environment

    def _materialize_context(
        self,
        source_root: Path,
        context_root: Path,
        source_paths: Iterable[str],
    ) -> list[str]:
        self.repository.materialize(source_root, source_paths)
        if self.context_paths:
            self.repository.materialize(context_root, self.context_paths)

        include_dirs: list[str] = []

        def add_include(path: Path) -> None:
            if not path.exists():
                return
            value = str(path.resolve())
            if value not in include_dirs:
                include_dirs.append(value)

        for scope in source_paths:
            path = source_root / str(scope)
            add_include(path if path.is_dir() else path.parent)
        for context_path in self.context_paths:
            add_include(context_root / context_path)
        return include_dirs

    def _c2cpg_command(
        self,
        input_root: Path,
        output_path: Path,
        include_dirs: Iterable[str],
    ) -> list[str]:
        command = [
            str(self._c2cpg()),
            str(input_root),
            "--output",
            str(output_path),
            "--with-include-auto-discovery",
            "--log-problems",
        ]
        for include_dir in include_dirs:
            command.extend(["--include", str(include_dir)])
        for define in self.defines:
            command.extend(["--define", define])
        return command

    def _build(self) -> None:
        self.ensure_available()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.cpg_path.is_file():
            with tempfile.TemporaryDirectory(
                prefix=f"vul-cpg-{self.sample_key}-"
            ) as directory:
                root = Path(directory)
                source_root = root / "src"
                context_root = root / "context"
                include_dirs = self._materialize_context(
                    source_root,
                    context_root,
                    self.scopes,
                )
                temporary_cpg = root / "cpg.bin"
                command = self._c2cpg_command(
                    source_root,
                    temporary_cpg,
                    include_dirs,
                )
                try:
                    result = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=self.timeout,
                        check=False,
                        env=self._environment(),
                    )
                except subprocess.TimeoutExpired as error:
                    raise JoernError(
                        f"{self.sample_key}: c2cpg timed out after "
                        f"{self.timeout}s"
                    ) from error
                diagnostics_path = (
                    self.cache_dir
                    / f"{self.sample_key}-{self.cpg_fingerprint}.c2cpg.log"
                )
                diagnostics_path.write_text(
                    (result.stdout or "")
                    + ("\n" if result.stdout and result.stderr else "")
                    + (result.stderr or "")
                )
                if result.returncode != 0 or not temporary_cpg.is_file():
                    raise JoernError(
                        f"{self.sample_key}: c2cpg failed: "
                        f"{result.stderr.strip() or result.stdout.strip()}"
                    )
                os.replace(temporary_cpg, self.cpg_path)

        if self.index_path.is_file():
            return

        # Index export needs the same source-root layout to normalize Joern
        # filenames, but reuses the already-built CPG.
        with tempfile.TemporaryDirectory(
            prefix=f"vul-index-{self.sample_key}-"
        ) as directory:
            root = Path(directory)
            source_root = root / "src"
            self.repository.materialize(source_root, self.scopes)
            temporary_index = root / "index.tsv"
            scope_file = root / "scopes.txt"
            scope_file.write_text(
                "\n".join(
                    self._normalize_repository_path(scope)
                    for scope in self.scopes
                )
                + "\n"
            )
            command = [
                str(self.joern),
                "--script",
                str(self.script),
                "--param",
                f"cpgFile={self.cpg_path.resolve()}",
                "--param",
                f"outFile={temporary_index}",
                "--param",
                f"sourceRoot={source_root.resolve()}",
                "--param",
                f"scopeFile={scope_file.resolve()}",
            ]
            try:
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                    env=self._environment(),
                )
            except subprocess.TimeoutExpired as error:
                raise JoernError(
                    f"{self.sample_key}: Joern index export timed out after "
                    f"{self.timeout}s"
                ) from error
            if result.returncode != 0 or not temporary_index.is_file():
                raise JoernError(
                    f"{self.sample_key}: Joern index export failed: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            os.replace(temporary_index, self.index_path)

    def methods(self) -> dict[str, RepositoryMethod]:
        if self._methods is not None:
            return self._methods
        self._build()

        raw_methods: dict[str, dict[str, object]] = {}
        for raw_line in self.index_path.read_text().splitlines():
            if not raw_line:
                continue
            parts = raw_line.split("\t")
            tag = parts[0]
            if tag == "ERROR":
                raise JoernError(parts[1] if len(parts) > 1 else "Joern index error")
            if tag == "METHOD" and len(parts) >= 7:
                full_name = parts[1]
                raw_methods[full_name] = {
                    "name": parts[2],
                    "path": parts[3],
                    "start_line": int(parts[4]),
                    "end_line": int(parts[5]),
                    "return_type": parts[6],
                    "parameters": {},
                    "calls": [],
                }
            elif tag == "PARAM" and len(parts) >= 5:
                method = raw_methods.get(parts[1])
                if method is not None:
                    method["parameters"][int(parts[2])] = (parts[3], parts[4])
            elif tag == "CALL" and len(parts) >= 7:
                method = raw_methods.get(parts[1])
                if method is not None:
                    method["calls"].append(
                        RepositoryCall(
                            line=int(parts[3]),
                            name=parts[4],
                            method_full_name=parts[5],
                            dispatch_type=parts[6],
                            call_id=parts[2],
                        )
                    )

        methods: dict[str, RepositoryMethod] = {}
        for full_name, raw in raw_methods.items():
            parameter_map = raw["parameters"]
            ordered = [parameter_map[index] for index in sorted(parameter_map)]
            methods[full_name] = RepositoryMethod(
                full_name=full_name,
                name=str(raw["name"]),
                path=str(raw["path"]),
                start_line=int(raw["start_line"]),
                end_line=int(raw["end_line"]),
                return_type=str(raw["return_type"]),
                parameters=tuple(item[0] for item in ordered),
                parameter_types=tuple(item[1] for item in ordered),
                calls=tuple(raw["calls"]),
            )

        self._methods = methods
        return methods

    @staticmethod
    def _normalize_repository_path(path: str) -> str:
        return Path(path).as_posix().lstrip("./")

    def find_entry(self, name: str, path: str) -> RepositoryMethod | None:
        expected_path = self._normalize_repository_path(path)
        named = [
            method
            for method in self.methods().values()
            if method.name == name
        ]
        matches = [
            method
            for method in named
            if self._normalize_repository_path(method.path) == expected_path
        ]
        if len(matches) == 1:
            return matches[0]
        if not named:
            return None
        actual = ", ".join(
            sorted(
                {
                    self._normalize_repository_path(method.path)
                    for method in named
                }
            )
        )
        raise JoernError(
            f"{self.sample_key}: Joern found {name} but repository path "
            f"did not resolve uniquely to {expected_path}; indexed paths: {actual}"
        )

    def callee_methods(self, call: RepositoryCall) -> list[RepositoryMethod]:
        if call.dispatch_type != "STATIC_DISPATCH":
            return []
        method = self.methods().get(call.method_full_name)
        return [method] if method is not None else []


class JoernValidator:
    """Extract per-function CPG/data-flow facts with a local Joern installation."""

    def __init__(
        self,
        joern_dir: str | Path = "/home/phy/joern",
        java_home: str | Path | None = None,
        timeout: int | None = None,
        repository_index: JoernRepositoryIndex | None = None,
    ) -> None:
        self.joern_dir = Path(os.environ.get("JOERN_HOME", str(joern_dir))).expanduser()
        self.joern = self.joern_dir / "joern"
        self.java_home = Path(
            java_home or os.environ.get("JAVA_HOME", "/home/phy/jdk21")
        ).expanduser()
        self.java = self.java_home / "bin" / "java"
        self.script = Path(__file__).with_name("joern_extract.sc")
        self.tu_script = Path(__file__).with_name("joern_tu_extract.sc")
        self.timeout = timeout or int(os.environ.get("JOERN_TIMEOUT", "180"))
        self.repository_index = repository_index
        self._cache: dict[str, JoernFacts] = {}
        self._errors: dict[str, str] = {}
        self._missing_methods: dict[str, str] = {}
        self._timeouts: dict[str, str] = {}

    def ensure_available(self) -> None:
        if not self.joern.is_file():
            raise JoernError(
                f"Joern launcher not found at {self.joern}. "
                "Set --joern-dir or JOERN_HOME to the Joern installation."
            )
        if not self.java.is_file() or not os.access(self.java, os.X_OK):
            raise JoernError(
                f"Java executable not found at {self.java}. "
                "Set --java-home or JAVA_HOME to the JDK installation."
            )
        if not self.script.is_file():
            raise JoernError(f"Joern extraction script not found: {self.script}")
        if not self.tu_script.is_file():
            raise JoernError(
                f"Joern TU extraction script not found: {self.tu_script}"
            )
        if self.repository_index is not None:
            self.repository_index.ensure_available()

    def _key(self, function) -> str:
        payload = (
            f"{function.path}\0{function.name}\0"
            f"{function.start_line}\0{function.end_line}\0"
            f"{function.text}\0"
            + hashlib.sha256(function.translation_unit.encode()).hexdigest()
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def _is_indexed_method(self, candidate):
        if self.repository_index is None or not candidate.method_full_name:
            return None
        method = self.repository_index.methods().get(candidate.method_full_name)
        if method is None:
            return None
        function = candidate.function
        if (
            method.path == function.path
            and method.start_line == function.start_line
            and method.end_line == function.end_line
        ):
            return method
        return None

    @staticmethod
    def _normalize_method_path(path: str) -> str:
        return Path(path).as_posix().lstrip("./")

    def _parse_tu_facts(
        self,
        text: str,
    ) -> dict[tuple[str, str, int, int], JoernFacts]:
        identities: dict[str, tuple[str, str, int, int]] = {}
        facts_by_key: dict[str, JoernFacts] = {}
        call_args: dict[
            tuple[str, str],
            tuple[int, str, str, dict[int, str]],
        ] = {}

        for raw_line in text.splitlines():
            if not raw_line:
                continue
            parts = raw_line.split("\t")
            tag = parts[0]
            if tag == "ERROR":
                raise JoernError(
                    parts[1] if len(parts) > 1 else "unknown Joern TU error"
                )
            if tag == "METHOD" and len(parts) >= 6:
                key = parts[1]
                identities[key] = (
                    parts[2],
                    self._normalize_method_path(parts[3]),
                    int(parts[4]),
                    int(parts[5]),
                )
                facts_by_key[key] = JoernFacts()
            elif tag == "PARAM" and len(parts) >= 5:
                facts = facts_by_key.get(parts[1])
                if facts is not None:
                    facts.parameters[int(parts[2])] = (parts[3], parts[4])
            elif tag == "ARG" and len(parts) >= 8:
                method_key = parts[1]
                if method_key not in facts_by_key:
                    continue
                call_id = parts[2]
                line = int(parts[3])
                name = parts[4]
                index = int(parts[5])
                code = parts[6]
                _, _, _, arguments = call_args.setdefault(
                    (method_key, call_id),
                    (line, name, code, {}),
                )
                arguments[index] = parts[7]
            elif tag == "FLOW" and len(parts) >= 5:
                facts = facts_by_key.get(parts[1])
                if facts is not None:
                    facts.flows.add(
                        (int(parts[2]), parts[3], int(parts[4]))
                    )
            elif tag == "RET" and len(parts) >= 3:
                facts = facts_by_key.get(parts[1])
                if facts is not None:
                    facts.returns.append(parts[2])
            elif tag == "RETFLOW" and len(parts) >= 3:
                facts = facts_by_key.get(parts[1])
                if facts is not None:
                    facts.return_flows.add(int(parts[2]))

        for (method_key, call_id), (
            line,
            name,
            code,
            arguments,
        ) in call_args.items():
            facts_by_key[method_key].calls[call_id] = JoernCall(
                line,
                name,
                arguments,
                call_id=call_id,
                code=code,
            )

        return {
            identities[key]: facts
            for key, facts in facts_by_key.items()
            if key in identities
        }

    def _contextual_facts(self, candidate, method) -> JoernFacts:
        assert self.repository_index is not None
        index = self.repository_index
        cache_dir = index.cache_dir / "tu"
        cache_dir.mkdir(parents=True, exist_ok=True)
        method_parent = Path(method.path).parent.as_posix()
        source_paths = (
            (method.path,)
            if method_parent in {"", "."}
            else (method_parent,)
        )
        payload = json.dumps(
            {
                "revision": index.repository.revision,
                "source_paths": source_paths,
                "frontend": index.cpg_fingerprint,
                "script": _file_digest(self.tu_script),
            },
            sort_keys=True,
        ).encode()
        fingerprint = hashlib.sha256(payload).hexdigest()[:20]
        cpg_path = cache_dir / f"{fingerprint}.bin"
        facts_path = cache_dir / f"{fingerprint}.facts.tsv"

        if not cpg_path.is_file():
            with tempfile.TemporaryDirectory(prefix="vul-tu-cpg-") as directory:
                root = Path(directory)
                source_root = root / "src"
                context_root = root / "context"
                include_dirs = index._materialize_context(
                    source_root,
                    context_root,
                    source_paths,
                )
                temporary_cpg = root / "tu.bin"
                command = index._c2cpg_command(
                    source_root,
                    temporary_cpg,
                    include_dirs,
                )
                result = self._run(command, candidate.function.name)
                if result.returncode != 0 or not temporary_cpg.is_file():
                    raise JoernError(
                        f"Joern TU c2cpg failed for {candidate.function.name}: "
                        f"{result.stderr.strip() or result.stdout.strip()}"
                    )
                os.replace(temporary_cpg, cpg_path)

        if not facts_path.is_file():
            with tempfile.TemporaryDirectory(
                prefix="vul-tu-facts-"
            ) as directory:
                temporary_facts = Path(directory) / "facts.tsv"
                command = [
                    str(self.joern),
                    "--script",
                    str(self.tu_script),
                    "--param",
                    f"cpgFile={cpg_path.resolve()}",
                    "--param",
                    f"outFile={temporary_facts}",
                ]
                result = self._run(command, candidate.function.name)
                if result.returncode != 0 or not temporary_facts.is_file():
                    raise JoernError(
                        f"Joern TU dataflow failed for "
                        f"{candidate.function.name}: "
                        f"{result.stderr.strip() or result.stdout.strip()}"
                    )
                os.replace(temporary_facts, facts_path)

        facts_map = self._parse_tu_facts(facts_path.read_text())
        target_name = method.name
        target_path = self._normalize_method_path(method.path)
        matches = [
            facts
            for (name, path, start_line, end_line), facts in facts_map.items()
            if name == target_name
            and path == target_path
            and start_line == method.start_line
            and end_line == method.end_line
        ]
        if not matches:
            raise JoernMethodNotFound(
                f"method_not_found:{method.path}:{method.name}@"
                f"{method.start_line}-{method.end_line}"
            )
        if len(matches) != 1:
            raise JoernError(
                f"ambiguous_method:{method.path}:{method.name}@"
                f"{method.start_line}-{method.end_line}"
            )
        return matches[0]

    def facts(self, candidate) -> JoernFacts:
        self.ensure_available()
        key = self._key(candidate.function)
        if key in self._cache:
            return self._cache[key]
        if key in self._missing_methods:
            raise JoernMethodNotFound(self._missing_methods[key])
        if key in self._timeouts:
            raise JoernTimeout(self._timeouts[key])
        if key in self._errors:
            raise JoernError(self._errors[key])

        indexed_method = self._is_indexed_method(candidate)
        if indexed_method is not None:
            try:
                facts = self._contextual_facts(candidate, indexed_method)
            except JoernMethodNotFound as error:
                self._missing_methods[key] = str(error)
                raise
            except JoernTimeout as error:
                self._timeouts[key] = str(error)
                raise
            except JoernError as error:
                self._errors[key] = str(error)
                raise
            self._cache[key] = facts
            return facts

        # Explicit same-file preprocessor variants may not be present in the
        # active sample CPG. Validate those variant bodies independently.
        with tempfile.TemporaryDirectory(prefix="memsem-joern-") as directory:
            root = Path(directory)
            suffix = Path(candidate.function.path).suffix or ".c"
            source_path = root / f"candidate{suffix}"
            output_path = root / "facts.tsv"
            source_path.write_text(candidate.function.translation_unit)

            command = [
                str(self.joern),
                "--script",
                str(self.script),
                "--param",
                f"sourceFile={source_path}",
                "--param",
                f"outFile={output_path}",
                "--param",
                f"functionName={candidate.function.name}",
                "--param",
                f"functionStartLine={candidate.function.start_line}",
                "--param",
                f"functionEndLine={candidate.function.end_line}",
            ]
            try:
                result = self._run(command, candidate.function.name)
                facts = self._load_facts(
                    output_path,
                    result,
                    candidate.function.name,
                )
            except JoernMethodNotFound as error:
                self._missing_methods[key] = str(error)
                raise
            except JoernTimeout as error:
                self._timeouts[key] = str(error)
                raise
            except JoernError as error:
                self._errors[key] = str(error)
                raise
            self._cache[key] = facts
            return facts

    def _run(
        self,
        command: list[str],
        function_name: str,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["JAVA_HOME"] = str(self.java_home)
        environment["PATH"] = (
            str(self.java.parent) + os.pathsep + environment.get("PATH", "")
        )
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout,
                check=False,
                env=environment,
            )
        except subprocess.TimeoutExpired as error:
            message = (
                f"Joern timed out after {self.timeout}s for {function_name}"
            )
            raise JoernTimeout(message) from error
        return result

    def _load_facts(
        self,
        output_path: Path,
        result: subprocess.CompletedProcess[str],
        function_name: str,
    ) -> JoernFacts:
        if result.returncode != 0:
            message = (
                f"Joern failed for {function_name}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
            raise JoernError(message)
        if not output_path.is_file():
            raise JoernError(
                f"Joern produced no fact file for {function_name}: "
                f"{result.stdout.strip()}"
            )
        return self._parse(output_path.read_text())

    @staticmethod
    def _parse(text: str) -> JoernFacts:
        facts = JoernFacts()
        call_args: dict[str, tuple[int, str, str, dict[int, str]]] = {}
        for raw_line in text.splitlines():
            if not raw_line:
                continue
            parts = raw_line.split("\t")
            tag = parts[0]
            if tag == "ERROR":
                message = parts[1] if len(parts) > 1 else "unknown Joern error"
                if message.startswith("method_not_found:"):
                    raise JoernMethodNotFound(message)
                raise JoernError(message)
            if tag == "PARAM" and len(parts) >= 4:
                facts.parameters[int(parts[1])] = (parts[2], parts[3])
            elif tag == "ARG" and len(parts) >= 7:
                call_id = parts[1]
                line = int(parts[2])
                name = parts[3]
                index = int(parts[4])
                code = parts[5]
                _, _, _, arguments = call_args.setdefault(
                    call_id,
                    (line, name, code, {}),
                )
                arguments[index] = parts[6]
            elif tag == "FLOW" and len(parts) >= 4:
                facts.flows.add(
                    (int(parts[1]), parts[2], int(parts[3]))
                )
            elif tag == "RET" and len(parts) >= 2:
                facts.returns.append(parts[1])
            elif tag == "RETFLOW" and len(parts) >= 2:
                facts.return_flows.add(int(parts[1]))

        for call_id, (line, name, code, arguments) in call_args.items():
            facts.calls[call_id] = JoernCall(
                line,
                name,
                arguments,
                call_id=call_id,
                code=code,
            )
        return facts

