"""Independent, resumable full-graph export for the A/B/C experiment.

The original schema-9 records and membership are preserved. A failed new export
is explicit (static_graph=None); it never silently removes a function. Such a
sample gets a zero graph representation, not a learned missing-graph token.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from itertools import groupby
from pathlib import Path

from .graph_features import (
    GRAPH_SCHEMA_VERSION, canonical_json, graph_to_record, record_identity,
    records_fingerprint, validate_record_graph,
)


def read_jsonl(path: str | Path, *, recover_tail: bool = False) -> list[dict]:
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        lines = handle.readlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if recover_tail and number == len(lines) and not line.endswith("\n"):
                print(f"discard_incomplete_graph_tail={path}:{number}", flush=True)
                break
            raise ValueError(f"{path}:{number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected an object")
        key = row.get("sample_key")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError(f"{path}:{number}: missing/duplicate sample_key")
        seen.add(key)
        rows.append(row)
    return rows


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def output_lock(path: Path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"another process is writing {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def graph_audit(rows: list[dict]) -> dict:
    groups = defaultdict(Counter)
    for row in rows:
        key = f"{row['dataset']}/{row['split']}/label_{row['label']}"
        groups[key]["samples"] += 1
        graph = row["static_graph"]
        groups[key]["available" if graph is not None else "unavailable"] += 1
        if graph is not None:
            for name, count in graph["statistics"].items():
                groups[key][name] += count
    return {"samples": len(rows), "available": sum(r["static_graph"] is not None for r in rows),
            "unavailable": sum(r["static_graph"] is None for r in rows),
            "members_dropped": 0,
            "groups": {k: dict(v) for k, v in sorted(groups.items())},
            "note": "Full source membership retained; unavailable graphs yield zero graph vectors."}


def build_graph_dataset(dataset_path: str | Path, output_path: str | Path, *,
                        source_dataset: str = "primevul", joern_dir: str = "/home/phy/joern",
                        java_home: str = "/home/phy/jdk21", timeout: int = 300,
                        batch_size: int = 8, retry_failed: bool = False) -> dict:
    from .cpg import CPGError, extract_function_cpg_batch, _environment, _find_executable
    from .dataset import DATASET_SCHEMA_VERSION
    from .syntax import resolve_language

    original, target = Path(dataset_path), Path(output_path)
    if original.resolve() == target.resolve():
        raise ValueError("graph output must not overwrite the original dataset")
    if source_dataset not in {"primevul", "cleanvul", "sven"}:
        raise ValueError("unknown source dataset")
    if not 1 <= batch_size <= 32 or timeout <= 0:
        raise ValueError("batch size must be 1..32 and timeout positive")
    rows = [r for r in read_jsonl(original) if r.get("dataset") == source_dataset]
    if not rows:
        raise ValueError(f"no {source_dataset} records in {original}")
    for row in rows:
        if row.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise ValueError("input dataset schema is incompatible")
        record_identity(row)  # Require identity fields before touching output.
        if type(row["label"]) is not int or row["label"] not in {0, 1}:
            raise ValueError("invalid label")
        if not isinstance(row["raw_source"], str) or not row["raw_source"]:
            raise ValueError("raw_source is required")
        if "static_graph" in row:
            raise ValueError("build from the original dataset, not an existing graph experiment")
    input_digest = records_fingerprint(rows)
    metadata_path = target.with_suffix(".meta.json")
    error_path = target.with_suffix(".errors.jsonl")
    audit_path = target.with_suffix(".audit.json")
    source_by_key = {r["sample_key"]: r for r in rows}
    with output_lock(target):
        metadata = {"graph_schema_version": GRAPH_SCHEMA_VERSION,
                    "input_fingerprint": input_digest, "source_dataset": source_dataset,
                    "source_file": str(original), "samples": len(rows), "complete": False}
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            for name in ("graph_schema_version", "input_fingerprint", "source_dataset", "samples"):
                if previous.get(name) != metadata[name]:
                    raise ValueError(f"graph cache mismatch: {name}; use a new output path")
        elif target.exists() and target.stat().st_size:
            raise ValueError("existing graph output has no metadata; use a new output path")
        completed = {}
        if target.exists():
            for row in read_jsonl(target, recover_tail=True):
                key = row["sample_key"]
                if key not in source_by_key or record_identity(row) != record_identity(source_by_key[key]):
                    raise ValueError(f"cached graph record differs from source: {key}")
                validate_record_graph(row)
                if retry_failed and row["static_graph"] is None:
                    continue
                completed[key] = row
        pending = [r for r in rows if r["sample_key"] not in completed]
        if pending:
            # Fail early on bad installation paths instead of recording all rows as unavailable.
            root = Path(os.environ.get("JOERN_HOME", joern_dir)).expanduser()
            _find_executable(root, ("joern-parse", "joern-cli/joern-parse", "joern-cli/bin/joern-parse"))
            _find_executable(root, ("joern-export", "joern-cli/joern-export", "joern-cli/bin/joern-export"))
            _environment(java_home)
        atomic_json(metadata_path, metadata)
        atomic_jsonl(target, (completed[r["sample_key"]] for r in rows if r["sample_key"] in completed))
        print(f"graph_samples_total={len(rows)} resumed={len(completed)} pending={len(pending)}", flush=True)
        with target.open("a", encoding="utf-8") as handle:
            for _, partition in groupby(pending, key=lambda r: (r["dataset"], r["split"])):
                partition = list(partition)
                for start in range(0, len(partition), batch_size):
                    batch = partition[start:start + batch_size]
                    requests = []
                    for row in batch:
                        language = row.get("resolved_language")
                        if language not in {"c", "cpp"}:
                            language = resolve_language(row["raw_source"], row.get("language", "c_cpp"), "")
                        requests.append({"source": row["raw_source"], "language": language,
                                         "function": row.get("function_name", "")})
                    started = time.monotonic()
                    try:
                        results = extract_function_cpg_batch(requests, joern_dir=joern_dir,
                                                             java_home=java_home, timeout=timeout)
                    except (CPGError, OSError, subprocess.TimeoutExpired) as error:
                        results = [error] * len(batch)
                    if len(results) != len(batch):
                        raise RuntimeError("Joern returned an incomplete batch")
                    for row, result in zip(batch, results):
                        exported = dict(row)
                        try:
                            if isinstance(result, BaseException):
                                raise result
                            exported["static_graph"] = graph_to_record(result, row["raw_source"])
                            exported["graph_status"] = "available"
                        except (CPGError, ValueError, OSError, subprocess.TimeoutExpired) as error:
                            exported.update(static_graph=None, graph_status="unavailable",
                                            graph_error=f"{type(error).__name__}: {error}")
                        exported["graph_seconds"] = (time.monotonic() - started) / len(batch)
                        validate_record_graph(exported)
                        handle.write(canonical_json(exported) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                        completed[row["sample_key"]] = exported
                        size = len(exported["static_graph"]["cfg_nodes"]) if exported["static_graph"] else 0
                        print(f"graph_done={len(completed)}/{len(rows)} sample={row['sample_key']} "
                              f"status={exported['graph_status']} cfg_nodes={size}", flush=True)
        ordered = [completed[r["sample_key"]] for r in rows]
        atomic_jsonl(target, ordered)
        atomic_jsonl(error_path, ({"sample_key": r["sample_key"], "split": r["split"],
                                 "label": r["label"], "error": r["graph_error"]}
                                for r in ordered if r["static_graph"] is None))
        audit = graph_audit(ordered)
        atomic_json(audit_path, audit)
        metadata.update(complete=True, output_sha256=file_sha256(target), **audit)
        atomic_json(metadata_path, metadata)
        print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
        if not audit["available"]:
            raise ValueError("all graph exports failed; inspect the errors file before training")
        return audit


def load_graph_dataset(path: str | Path) -> tuple[list[dict], dict]:
    path = Path(path)
    metadata_path = path.with_suffix(".meta.json")
    if not metadata_path.exists():
        raise ValueError("graph metadata missing; finish graph_experiment build first")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("graph_schema_version") != GRAPH_SCHEMA_VERSION or metadata.get("complete") is not True:
        raise ValueError("graph export is incomplete or uses an incompatible schema")
    if metadata.get("output_sha256") != file_sha256(path):
        raise ValueError("graph dataset changed after export; refuse an unmatched comparison")
    rows = read_jsonl(path)
    if len(rows) != metadata["samples"] or records_fingerprint(rows) != metadata["input_fingerprint"]:
        raise ValueError("graph dataset membership differs from its fixed manifest")
    if {r.get("dataset") for r in rows} != {metadata["source_dataset"]}:
        raise ValueError("graph dataset must contain exactly the specified source")
    for row in rows:
        validate_record_graph(row)
    return rows, metadata
