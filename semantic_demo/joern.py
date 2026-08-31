from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


class JoernError(RuntimeError):
    pass


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
    conditions: list[str] = field(default_factory=list)

    def call_list(self) -> list[JoernCall]:
        return list(self.calls.values())

    def parameter_reaches(
        self, parameter_index: int, call: JoernCall, argument_index: int
    ) -> bool:
        return (parameter_index, call.line, call.name, argument_index) in self.flows


class JoernValidator:
    """Extract per-function CPG/data-flow facts with a local Joern installation.

    The validator intentionally imports one candidate function at a time. This keeps the
    demo fast and makes Joern a verifier for LLM-proposed semantics instead of turning
    the prototype into a full repository-scale static analyzer.
    """

    def __init__(
        self,
        joern_dir: str | Path = "/home/phy/joern",
        timeout: int | None = None,
    ) -> None:
        self.joern_dir = Path(os.environ.get("JOERN_HOME", str(joern_dir))).expanduser()
        self.joern = self.joern_dir / "joern"
        self.script = Path(__file__).with_name("joern_extract.sc")
        self.timeout = timeout or int(os.environ.get("JOERN_TIMEOUT", "180"))
        self._cache: dict[str, JoernFacts] = {}
        self._errors: dict[str, str] = {}

    def ensure_available(self) -> None:
        if not self.joern.is_file():
            raise JoernError(
                f"Joern launcher not found at {self.joern}. "
                "Set --joern-dir or JOERN_HOME to the Joern installation."
            )
        if not self.script.is_file():
            raise JoernError(f"Joern extraction script not found: {self.script}")

    def _key(self, function) -> str:
        payload = f"{function.path}\0{function.name}\0{function.text}".encode()
        return hashlib.sha256(payload).hexdigest()

    def facts(self, candidate) -> JoernFacts:
        self.ensure_available()
        key = self._key(candidate.function)
        if key in self._cache:
            return self._cache[key]
        if key in self._errors:
            raise JoernError(self._errors[key])

        with tempfile.TemporaryDirectory(prefix="memsem-joern-") as directory:
            root = Path(directory)
            source_path = root / "candidate.c"
            output_path = root / "facts.tsv"
            source_path.write_text(candidate.function.text)

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
            ]
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            if result.returncode != 0:
                message = (
                    f"Joern failed for {candidate.function.name}: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
                self._errors[key] = message
                raise JoernError(message)
            if not output_path.is_file():
                message = (
                    f"Joern produced no fact file for {candidate.function.name}: "
                    f"{result.stdout.strip()}"
                )
                self._errors[key] = message
                raise JoernError(message)

            facts = self._parse(output_path.read_text())
            self._cache[key] = facts
            return facts

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
                raise JoernError(parts[1] if len(parts) > 1 else "unknown Joern error")
            if tag == "PARAM" and len(parts) >= 4:
                index = int(parts[1])
                facts.parameters[index] = (parts[2], parts[3])
            elif tag == "ARG" and len(parts) >= 6:
                line = int(parts[1])
                name = parts[2]
                index = int(parts[3])
                call_args.setdefault((line, name), {})[index] = parts[5]
            elif tag == "FLOW" and len(parts) >= 5:
                facts.flows.add(
                    (int(parts[1]), int(parts[2]), parts[3], int(parts[4]))
                )
            elif tag == "COND" and len(parts) >= 2:
                facts.conditions.append(parts[1])

        for (line, name), arguments in call_args.items():
            facts.calls[(line, name)] = JoernCall(line, name, arguments)
        return facts
