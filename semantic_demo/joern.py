from __future__ import annotations

import hashlib
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


@dataclass
class JoernFacts:
    parameters: dict[int, tuple[str, str]] = field(default_factory=dict)
    calls: dict[tuple[int, str], JoernCall] = field(default_factory=dict)
    flows: set[tuple[int, int, str, int]] = field(default_factory=set)
    returns: list[str] = field(default_factory=list)
    return_flows: set[int] = field(default_factory=set)

    def call_list(self) -> list[JoernCall]:
        return list(self.calls.values())

    def parameter_reaches(
        self, parameter_index: int, call: JoernCall, argument_index: int
    ) -> bool:
        return (parameter_index, call.line, call.name, argument_index) in self.flows


@dataclass(frozen=True)
class RepositoryCall:
    line: int
    name: str
    method_full_name: str
    dispatch_type: str


@dataclass(frozen=True)
class RepositoryMethod:
    full_name: str
    name: str
    path: str
    start_line: int
    end_line: int
    parameters: tuple[str, ...]
    parameter_types: tuple[str, ...]
    calls: tuple[RepositoryCall, ...]


class JoernRepositoryIndex:
    """Build and query one Joern 4 CPG per sample/revision."""

    def __init__(
        self,
        repository,
        sample_key: str,
        scopes: Iterable[str],
        entry_path: str,
        *,
        joern_dir: str | Path = "/home/phy/joern",
        java_home: str | Path | None = None,
        cache_dir: str | Path = "data/joern_cpg",
        timeout: int = 900,
    ) -> None:
        self.repository = repository
        self.sample_key = sample_key
        self.scopes = tuple(dict.fromkeys([*map(str, scopes), str(entry_path)]))
        self.entry_path = str(entry_path)
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
        self._facts: dict[str, JoernFacts] | None = None

        fingerprint_payload = json.dumps(
            {
                "revision": self.repository.revision,
                "scopes": self.scopes,
                "joern_index_schema": 2,
                "joern_api": "4.0.465",
            },
            sort_keys=True,
        ).encode()
        self.fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()[:16]
        self.cpg_path = self.cache_dir / f"{sample_key}-{self.fingerprint}.bin.zip"
        self.index_path = self.cache_dir / f"{sample_key}-{self.fingerprint}.tsv"

    def _c2cpg(self) -> Path:
        candidates = (
            self.joern_dir / "c2cpg.sh",
            self.joern_dir / "joern-cli" / "frontends" / "c2cpg" / "c2cpg.sh",
        )
        return next((path for path in candidates if path.is_file()), candidates[0])

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

    def _build(self) -> None:
        self.ensure_available()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.cpg_path.is_file() and self.index_path.is_file():
            return
        self.cpg_path.unlink(missing_ok=True)
        self.index_path.unlink(missing_ok=True)

        with tempfile.TemporaryDirectory(prefix=f"vul-cpg-{self.sample_key}-") as directory:
            source_root = Path(directory) / "src"
            self.repository.materialize(source_root, self.scopes)

            command = [
                str(self._c2cpg()),
                str(source_root),
                "--output",
                str(self.cpg_path.resolve()),
                "--with-include-auto-discovery",
            ]
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout,
                check=False,
                env=self._environment(),
            )
            if result.returncode != 0 or not self.cpg_path.is_file():
                self.cpg_path.unlink(missing_ok=True)
                raise JoernError(
                    f"{self.sample_key}: c2cpg failed: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )

            command = [
                str(self.joern),
                "--script",
                str(self.script),
                "--param",
                f"cpgFile={self.cpg_path.resolve()}",
                "--param",
                f"outFile={self.index_path.resolve()}",
                "--param",
                f"sourceRoot={source_root.resolve()}",
            ]
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout,
                check=False,
                env=self._environment(),
            )
            if result.returncode != 0 or not self.index_path.is_file():
                self.index_path.unlink(missing_ok=True)
                raise JoernError(
                    f"{self.sample_key}: Joern index export failed: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )

    def methods(self) -> dict[str, RepositoryMethod]:
        if self._methods is not None:
            return self._methods
        self._build()

        raw_methods: dict[str, dict[str, object]] = {}
        facts: dict[str, JoernFacts] = {}
        fact_call_args: dict[
            str, dict[tuple[int, str], dict[int, str]]
        ] = {}

        for raw_line in self.index_path.read_text().splitlines():
            if not raw_line:
                continue
            parts = raw_line.split("\t")
            tag = parts[0]
            if tag == "ERROR":
                raise JoernError(parts[1] if len(parts) > 1 else "Joern index error")

            if tag == "METHOD" and len(parts) >= 6:
                full_name = parts[1]
                raw_methods[full_name] = {
                    "name": parts[2],
                    "path": parts[3],
                    "start_line": int(parts[4]),
                    "end_line": int(parts[5]),
                    "parameters": {},
                    "calls": [],
                }
                facts[full_name] = JoernFacts()
                fact_call_args[full_name] = {}
            elif tag == "PARAM" and len(parts) >= 5:
                full_name = parts[1]
                method = raw_methods.get(full_name)
                if method is not None:
                    index = int(parts[2])
                    method["parameters"][index] = (parts[3], parts[4])
                    facts[full_name].parameters[index] = (parts[3], parts[4])
            elif tag == "CALL" and len(parts) >= 6:
                full_name = parts[1]
                method = raw_methods.get(full_name)
                if method is not None:
                    method["calls"].append(
                        RepositoryCall(
                            line=int(parts[2]),
                            name=parts[3],
                            method_full_name=parts[4],
                            dispatch_type=parts[5],
                        )
                    )
            elif tag == "ARGFACT" and len(parts) >= 7:
                full_name = parts[1]
                if full_name in facts:
                    key = (int(parts[2]), parts[3])
                    fact_call_args[full_name].setdefault(key, {})[
                        int(parts[4])
                    ] = parts[6]
            elif tag == "FLOWFACT" and len(parts) >= 6:
                full_name = parts[1]
                if full_name in facts:
                    facts[full_name].flows.add(
                        (
                            int(parts[2]),
                            int(parts[3]),
                            parts[4],
                            int(parts[5]),
                        )
                    )
            elif tag == "RETFACT" and len(parts) >= 3:
                full_name = parts[1]
                if full_name in facts:
                    facts[full_name].returns.append(parts[2])
            elif tag == "RETFLOWFACT" and len(parts) >= 3:
                full_name = parts[1]
                if full_name in facts:
                    facts[full_name].return_flows.add(int(parts[2]))

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
                parameters=tuple(item[0] for item in ordered),
                parameter_types=tuple(item[1] for item in ordered),
                calls=tuple(raw["calls"]),
            )
            for (line, name), arguments in fact_call_args[full_name].items():
                facts[full_name].calls[(line, name)] = JoernCall(
                    line=line,
                    name=name,
                    arguments=arguments,
                )

        self._methods = methods
        self._facts = facts
        return methods

    def facts_for(self, method_full_name: str) -> JoernFacts:
        self.methods()
        assert self._facts is not None
        facts = self._facts.get(method_full_name)
        if facts is None:
            raise JoernMethodNotFound(f"method_not_found:{method_full_name}")
        return facts

    def find_entry(self, name: str, path: str) -> RepositoryMethod | None:
        matches = [
            method
            for method in self.methods().values()
            if method.name == name and method.path == path
        ]
        return matches[0] if len(matches) == 1 else None

    def callee_methods(self, call: RepositoryCall) -> list[RepositoryMethod]:
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

        indexed_method = None
        if (
            self.repository_index is not None
            and getattr(candidate, "method_full_name", "")
        ):
            indexed_method = self.repository_index.methods().get(
                candidate.method_full_name
            )

        use_sample_cpg = (
            indexed_method is not None
            and indexed_method.path == candidate.function.path
            and indexed_method.start_line == candidate.function.start_line
        )

        if use_sample_cpg:
            facts = self.repository_index.facts_for(candidate.method_full_name)
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
            result = self._run(command, candidate.function.name)
            facts = self._load_facts(output_path, result, candidate.function.name)
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
        call_args: dict[tuple[int, str], dict[int, str]] = {}
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
            elif tag == "ARG" and len(parts) >= 6:
                line = int(parts[1])
                name = parts[2]
                index = int(parts[3])
                call_args.setdefault((line, name), {})[index] = parts[5]
            elif tag == "FLOW" and len(parts) >= 5:
                facts.flows.add(
                    (int(parts[1]), int(parts[2]), parts[3], int(parts[4]))
                )
            elif tag == "RET" and len(parts) >= 2:
                facts.returns.append(parts[1])
            elif tag == "RETFLOW" and len(parts) >= 2:
                facts.return_flows.add(int(parts[1]))

        for (line, name), arguments in call_args.items():
            facts.calls[(line, name)] = JoernCall(line, name, arguments)
        return facts
