from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .joern import (
    JoernError,
    JoernMethodNotFound,
    JoernValidator as BaseJoernValidator,
    _file_digest,
    _temporary_cache_file,
)


def _implementation_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class JoernValidator(BaseJoernValidator):
    """TU validator consistent with repository-level preprocessing decisions."""

    def _contextual_facts(self, candidate, method):
        assert self.repository_index is not None
        index = self.repository_index
        cache_dir = index.cache_dir / "tu"
        cache_dir.mkdir(parents=True, exist_ok=True)
        source_paths = (method.path,)
        extra_context_paths = index.analysis_context_paths_for(method.path)
        preprocess_this_tu = bool(
            index.preprocess_entry and method.path == index.entry_path
        )
        payload = json.dumps(
            {
                "revision": index.repository.revision,
                "source_paths": source_paths,
                "context_paths": extra_context_paths,
                "frontend": index.cpg_fingerprint,
                "script": _file_digest(self.tu_script),
                "preprocess_this_tu": preprocess_this_tu,
                "validator_implementation": _implementation_digest(),
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
                    source_root, context_root, source_paths, extra_context_paths
                )
                if preprocess_this_tu:
                    index._preprocess_entry_source(source_root, include_dirs)
                temporary_cpg = _temporary_cache_file(cpg_path, ".bin")
                command = index._c2cpg_command(source_root, temporary_cpg, include_dirs)
                if not preprocess_this_tu:
                    command = [part for part in command if part != "--with-preprocessed-files"]
                result = self._run(command, candidate.function.name)
                if result.returncode != 0 or not temporary_cpg.is_file():
                    temporary_cpg.unlink(missing_ok=True)
                    raise JoernError(
                        f"Joern TU c2cpg failed for {candidate.function.name}: "
                        f"{result.stderr.strip() or result.stdout.strip()}"
                    )
                os.replace(temporary_cpg, cpg_path)

        if not facts_path.is_file():
            temporary_facts = _temporary_cache_file(facts_path, ".tsv")
            command = [
                str(self.joern),
                "--script", str(self.tu_script),
                "--param", f"cpgFile={cpg_path.resolve()}",
                "--param", f"outFile={temporary_facts}",
            ]
            result = self._run(command, candidate.function.name)
            if result.returncode != 0 or not temporary_facts.is_file():
                temporary_facts.unlink(missing_ok=True)
                raise JoernError(
                    f"Joern TU dataflow failed for {candidate.function.name}: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            os.replace(temporary_facts, facts_path)

        facts_map = self._parse_tu_facts(facts_path.read_text())
        target_name = method.name
        target_path = self._normalize_method_path(method.path)
        exact = [
            facts
            for (name, path, start_line, end_line), facts in facts_map.items()
            if name == target_name
            and path == target_path
            and start_line == method.start_line
            and end_line == method.end_line
        ]
        if len(exact) == 1:
            return exact[0]

        if preprocess_this_tu:
            compatible = [
                facts
                for (name, _path, _start, _end), facts in facts_map.items()
                if name == target_name and len(facts.parameters) == len(method.parameters)
            ]
            if len(compatible) == 1:
                return compatible[0]

        if not exact:
            raise JoernMethodNotFound(
                f"method_not_found:{method.path}:{method.name}@{method.start_line}-{method.end_line}"
            )
        raise JoernError(
            f"ambiguous_method:{method.path}:{method.name}@{method.start_line}-{method.end_line}"
        )
