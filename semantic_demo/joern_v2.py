from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .joern import (
    JoernError,
    JoernMethodNotFound,
    JoernTimeout,
    JoernValidator,
    _file_digest,
    _temporary_cache_file,
)


def _implementation_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class JoernValidatorV2(JoernValidator):
    """Validate candidate summaries from one batched Joern dataflow export per sample."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._manifest_batch_attempted = False

    def _facts_for_method(self, facts_map, method, *, preprocessed: bool):
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

        if preprocessed:
            compatible = [
                facts
                for (name, _path, _start, _end), facts in facts_map.items()
                if name == target_name
                and len(facts.parameters) == len(method.parameters)
            ]
            if len(compatible) == 1:
                return compatible[0]

        if not exact:
            raise JoernMethodNotFound(
                f"method_not_found:{method.path}:{method.name}@"
                f"{method.start_line}-{method.end_line}"
            )
        raise JoernError(
            f"ambiguous_method:{method.path}:{method.name}@"
            f"{method.start_line}-{method.end_line}"
        )

    def _batch_paths(self, methods) -> tuple[tuple[str, ...], tuple[str, ...]]:
        assert self.repository_index is not None
        index = self.repository_index
        source_paths = tuple(sorted({method.path for method in methods}))
        context_paths: list[str] = []
        for source_path in source_paths:
            for context_path in index.analysis_context_paths_for(source_path):
                if context_path not in context_paths:
                    context_paths.append(context_path)
        return source_paths, tuple(context_paths)

    def _prepare_batch_group(self, items, *, preprocessed: bool) -> None:
        assert self.repository_index is not None
        index = self.repository_index
        methods = [method for _candidate, method in items]
        source_paths, extra_context_paths = self._batch_paths(methods)
        if not source_paths:
            return

        cache_dir = index.cache_dir / "validation-batch-v1"
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "revision": index.repository.revision,
                "source_paths": source_paths,
                "context_paths": extra_context_paths,
                "frontend": index.cpg_fingerprint,
                "script": _file_digest(self.tu_script),
                "preprocessed": preprocessed,
                "validator_implementation": _implementation_digest(),
            },
            sort_keys=True,
        ).encode()
        fingerprint = hashlib.sha256(payload).hexdigest()[:20]
        cpg_path = cache_dir / f"{fingerprint}.bin"
        facts_path = cache_dir / f"{fingerprint}.facts.tsv"
        label = f"{index.sample_key}:batch:{len(source_paths)}tu"

        if not cpg_path.is_file():
            print(
                f"joern_batch_cpg_start={index.sample_key} "
                f"translation_units={len(source_paths)} candidates={len(items)}",
                flush=True,
            )
            with tempfile.TemporaryDirectory(prefix="vul-batch-cpg-") as directory:
                root = Path(directory)
                source_root = root / "src"
                context_root = root / "context"
                include_dirs = index._materialize_context(
                    source_root,
                    context_root,
                    source_paths,
                    extra_context_paths,
                )
                if preprocessed:
                    index._preprocess_entry_source(source_root, include_dirs)
                temporary_cpg = _temporary_cache_file(cpg_path, ".bin")
                command = index._c2cpg_command(
                    source_root, temporary_cpg, include_dirs
                )
                if not preprocessed:
                    command = [
                        part
                        for part in command
                        if part != "--with-preprocessed-files"
                    ]
                result = self._run(command, label)
                if result.returncode != 0 or not temporary_cpg.is_file():
                    temporary_cpg.unlink(missing_ok=True)
                    raise JoernError(
                        f"Joern batch c2cpg failed for {index.sample_key}: "
                        f"{result.stderr.strip() or result.stdout.strip()}"
                    )
                os.replace(temporary_cpg, cpg_path)
            print(
                f"joern_batch_cpg_done={index.sample_key} "
                f"translation_units={len(source_paths)}",
                flush=True,
            )

        if not facts_path.is_file():
            print(
                f"joern_batch_dataflow_start={index.sample_key} "
                f"translation_units={len(source_paths)}",
                flush=True,
            )
            temporary_facts = _temporary_cache_file(facts_path, ".tsv")
            command = [
                str(self.joern),
                "--script",
                str(self.tu_script),
                "--param",
                f"cpgFile={cpg_path.resolve()}",
                "--param",
                f"outFile={temporary_facts}",
            ]
            result = self._run(command, label)
            if result.returncode != 0 or not temporary_facts.is_file():
                temporary_facts.unlink(missing_ok=True)
                raise JoernError(
                    f"Joern batch dataflow failed for {index.sample_key}: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            os.replace(temporary_facts, facts_path)
            print(
                f"joern_batch_dataflow_done={index.sample_key}",
                flush=True,
            )

        facts_map = self._parse_tu_facts(facts_path.read_text())
        for candidate, method in items:
            key = self._key(candidate.function)
            try:
                facts = self._facts_for_method(
                    facts_map,
                    method,
                    preprocessed=preprocessed,
                )
            except JoernMethodNotFound as error:
                self._missing_methods[key] = str(error)
            except JoernError as error:
                self._errors[key] = str(error)
            else:
                self._cache[key] = facts

    def _prepare_manifest_batch(self) -> None:
        if self._manifest_batch_attempted or self.repository_index is None:
            return
        self._manifest_batch_attempted = True
        index = self.repository_index

        # Import lazily to keep candidate_graph -> joern dependencies acyclic.
        from .candidate_graph import load_manifest_candidate, read_candidate_manifest

        try:
            _header, records = read_candidate_manifest(index)
        except RuntimeError as error:
            print(
                f"joern_batch_unavailable={index.sample_key} reason={error}",
                flush=True,
            )
            return

        parse_cache = {}
        regular_items = []
        preprocessed_items = []
        for record in records:
            # static-skip candidates never have summaries to validate.
            if record.get("skip_reason") is not None:
                continue
            try:
                candidate = load_manifest_candidate(index, record, parse_cache)
            except RuntimeError:
                continue
            method = self._is_indexed_method(candidate)
            if method is None:
                continue
            item = (candidate, method)
            if index.preprocess_entry and method.path == index.entry_path:
                preprocessed_items.append(item)
            else:
                regular_items.append(item)

        prepared = 0
        for items, preprocessed in (
            (regular_items, False),
            (preprocessed_items, True),
        ):
            if not items:
                continue
            try:
                self._prepare_batch_group(items, preprocessed=preprocessed)
            except (JoernError, JoernTimeout) as error:
                # Preserve correctness: if a batch cannot be built, individual
                # contextual validation remains available for those candidates.
                print(
                    f"joern_batch_fallback={index.sample_key} "
                    f"candidates={len(items)} reason={error}",
                    flush=True,
                )
                continue
            prepared += len(items)

        print(
            f"joern_batch_ready={index.sample_key} prepared={prepared} "
            f"regular={len(regular_items)} preprocessed={len(preprocessed_items)}",
            flush=True,
        )

    def _single_contextual_facts(self, candidate, method):
        assert self.repository_index is not None
        index = self.repository_index
        cache_dir = index.cache_dir / "tu-v2"
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
            with tempfile.TemporaryDirectory(prefix="vul-tu-v2-cpg-") as directory:
                root = Path(directory)
                source_root = root / "src"
                context_root = root / "context"
                include_dirs = index._materialize_context(
                    source_root,
                    context_root,
                    source_paths,
                    extra_context_paths,
                )
                if preprocess_this_tu:
                    index._preprocess_entry_source(source_root, include_dirs)
                temporary_cpg = _temporary_cache_file(cpg_path, ".bin")
                command = index._c2cpg_command(
                    source_root, temporary_cpg, include_dirs
                )
                if not preprocess_this_tu:
                    command = [
                        part
                        for part in command
                        if part != "--with-preprocessed-files"
                    ]
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
                "--script",
                str(self.tu_script),
                "--param",
                f"cpgFile={cpg_path.resolve()}",
                "--param",
                f"outFile={temporary_facts}",
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
        return self._facts_for_method(
            facts_map,
            method,
            preprocessed=preprocess_this_tu,
        )

    def _contextual_facts(self, candidate, method):
        self._prepare_manifest_batch()
        key = self._key(candidate.function)
        if key in self._cache:
            return self._cache[key]
        if key in self._missing_methods:
            raise JoernMethodNotFound(self._missing_methods[key])
        if key in self._timeouts:
            raise JoernTimeout(self._timeouts[key])
        if key in self._errors:
            raise JoernError(self._errors[key])
        return self._single_contextual_facts(candidate, method)
