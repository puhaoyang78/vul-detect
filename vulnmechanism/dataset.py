from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

from .cpg import CPGError, FunctionGraph, extract_function_cpg
from .semantics import extract_vulnerability_semantics
from .syntax import parse_function


# Version 5 stores unfiltered normalized CPG relations for the raw-CPG
# ablation, structured vulnerability semantics, and fixed-vocabulary
# vulnerability features used by downstream ablations.
DATASET_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class CPGRelation:
    kind: str
    source: str
    target: str

    def as_text(self) -> str:
        return f"{self.kind}|{self.source}|{self.target}"


@dataclass(frozen=True)
class FunctionSample:
    line_number: int
    sample_key: str
    source: str
    label: int
    language: str
    function_name: str | None
    split: str | None


def _node_text(node) -> str:
    code = " ".join(node.code.split())
    return f"{node.label}:{code}"


def extract_cpg_relations(graph: FunctionGraph) -> tuple[CPGRelation, ...]:
    """Normalize all exported AST/CFG/CDG/DDG relations without semantic filtering."""
    relations: list[CPGRelation] = []
    seen: set[str] = set()
    for edge in graph.edges:
        source = graph.nodes.get(edge.source)
        target = graph.nodes.get(edge.target)
        if source is None or target is None:
            continue
        relation = CPGRelation(edge.kind, _node_text(source), _node_text(target))
        text = relation.as_text()
        if text not in seen:
            seen.add(text)
            relations.append(relation)
    return tuple(relations)


def render_cpg_relations(graph: FunctionGraph, max_relations: int = 160) -> str:
    if max_relations <= 0:
        raise ValueError("max_relations must be positive")
    relations = extract_cpg_relations(graph)
    groups: dict[str, list[CPGRelation]] = {}
    for relation in relations:
        groups.setdefault(relation.kind, []).append(relation)
    selected = [
        item
        for row in zip_longest(*groups.values())
        for item in row
        if item is not None
    ] if groups else []
    return "\n".join(item.as_text() for item in selected[:max_relations]) or "NO_CPG_RELATIONS"


def _required_string(record: dict[str, object], names: tuple[str, ...], label: str) -> str:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"missing {label}; expected one of {', '.join(names)}")


def _normalize_label(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return value
    if isinstance(value, float) and value in {0.0, 1.0}:
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "benign"}:
            return 0
        if normalized in {"1", "vulnerable"}:
            return 1
    raise ValueError(f"label must be binary 0/1, got {value!r}")


def _sample_fields(record: dict[str, object], line_number: int) -> FunctionSample:
    key_value = next(
        (record[name] for name in ("sample_key", "id", "idx") if name in record and record[name] is not None),
        None,
    )
    if key_value is None or str(key_value).strip() == "":
        raise ValueError("sample record requires sample_key, id, or idx")
    key = str(key_value)
    source = _required_string(record, ("function", "func", "source", "code", "func_before"), "function source")

    if "label" in record:
        label_value = record["label"]
    elif "target" in record:
        label_value = record["target"]
    else:
        raise ValueError(f"{key}: sample record requires label or target")
    label = _normalize_label(label_value)

    language = str(record.get("language") or "c").lower()
    language = "cpp" if language in {"c++", "cpp"} else language
    if language not in {"c", "cpp"}:
        raise ValueError(f"{key}: language must be c or cpp")

    function_name = record.get("function_name")
    split_value = record.get("split")
    split = str(split_value).lower() if split_value is not None else None
    if split == "validation":
        split = "valid"
    if split is not None and split not in {"train", "valid", "test"}:
        raise ValueError(f"{key}: split must be train, valid, validation, or test")
    return FunctionSample(
        line_number=line_number,
        sample_key=key,
        source=source,
        label=label,
        language=language,
        function_name=str(function_name) if function_name else None,
        split=split,
    )


def _read_samples(samples_path: str | Path) -> list[FunctionSample]:
    samples: list[FunctionSample] = []
    seen: set[str] = set()
    with Path(samples_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {samples_path}:{line_number}: {error}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"{samples_path}:{line_number}: each JSONL row must be an object")
            sample = _sample_fields(raw, line_number)
            if sample.sample_key in seen:
                raise ValueError(f"{samples_path}:{line_number}: duplicate sample_key {sample.sample_key}")
            seen.add(sample.sample_key)
            samples.append(sample)
    return samples


def _valid_semantic_items(value: object) -> bool:
    if not isinstance(value, list):
        return False
    return all(
        isinstance(item, dict)
        and isinstance(item.get("category"), str)
        and isinstance(item.get("kind"), str)
        and isinstance(item.get("detail"), str)
        for item in value
    )


def _record_matches_sample(record: dict[str, object], sample: FunctionSample) -> bool:
    parsed = parse_function(sample.source, sample.language, sample.function_name)
    return (
        record.get("raw_source") == sample.source
        and type(record.get("label")) is int
        and record.get("label") == sample.label
        and record.get("language") == sample.language
        and record.get("function_name") == parsed.name
        and record.get("split") == sample.split
        and isinstance(record.get("cpg_relations"), str)
        and bool(str(record.get("cpg_relations")).strip())
        and isinstance(record.get("vulnerability_semantics"), str)
        and _valid_semantic_items(record.get("semantic_items"))
        and isinstance(record.get("vulnerability_features"), list)
        and all(isinstance(value, str) for value in record.get("vulnerability_features", []))
        and type(record.get("cpg_relation_count")) is int
        and type(record.get("semantic_item_count")) is int
    )


def _load_reusable_records(
    path: Path,
    samples: list[FunctionSample],
) -> tuple[dict[str, dict[str, object]], bool]:
    if not path.is_file():
        return {}, False

    sample_by_key = {sample.sample_key: sample for sample in samples}
    reusable: dict[str, dict[str, object]] = {}
    incomplete_tail = False
    with path.open("rb") as handle:
        size = path.stat().st_size
        while line := handle.readline():
            start = handle.tell() - len(line)
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                if handle.tell() == size and not line.endswith(b"\n"):
                    incomplete_tail = True
                    break
                raise ValueError(f"{path}: invalid saved JSON at byte {start}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}: saved record must be an object at byte {start}")
            if record.get("schema_version") != DATASET_SCHEMA_VERSION:
                continue
            key = record.get("sample_key")
            if not isinstance(key, str) or key not in sample_by_key:
                raise ValueError(f"{path}: unknown saved sample_key={key!r}")
            if key in reusable:
                raise ValueError(f"{path}: duplicate saved sample_key={key!r}")
            sample = sample_by_key[key]
            if not _record_matches_sample(record, sample):
                raise ValueError(f"{path}: saved record does not match input: {key}")
            reusable[key] = record
    return reusable, incomplete_tail


def _write_jsonl_line(handle, value: dict[str, object]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _rewrite_in_sample_order(
    target: Path,
    samples: list[FunctionSample],
    records_by_key: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    ordered = [
        records_by_key[sample.sample_key]
        for sample in samples
        if sample.sample_key in records_by_key
    ]
    with target.open("w", encoding="utf-8") as output:
        for record in ordered:
            _write_jsonl_line(output, record)
    return ordered


def build_function_dataset(
    samples_path: str | Path,
    output_path: str | Path,
    *,
    joern_dir: str | Path = "/home/phy/joern",
    java_home: str | Path = "/home/phy/jdk21",
    timeout: int = 300,
) -> list[dict[str, object]]:
    """Build one CPG and vulnerability-semantic record per labeled function."""
    samples_file = Path(samples_path)
    target = Path(output_path)
    errors_path = target.with_suffix(".errors.jsonl")
    resolved = {samples_file.resolve(), target.resolve(), errors_path.resolve()}
    if len(resolved) != 3:
        raise ValueError("samples, output, and error log must be different files")

    samples = _read_samples(samples_file)
    target.parent.mkdir(parents=True, exist_ok=True)

    reusable, incomplete_tail = _load_reusable_records(target, samples)
    if incomplete_tail:
        print(f"discard_incomplete_output_tail={target}", flush=True)

    records = _rewrite_in_sample_order(target, samples, reusable)
    completed = set(reusable)
    print(f"function_samples_total={len(samples)} resumed={len(completed)}", flush=True)

    failures = 0
    records_by_key = dict(reusable)
    with target.open("a", encoding="utf-8") as output, errors_path.open("w", encoding="utf-8") as errors:
        for sample in samples:
            if sample.sample_key in completed:
                continue

            stage = "syntax"
            try:
                parsed = parse_function(sample.source, sample.language, sample.function_name)
            except ValueError as error:
                failure: BaseException | None = error
            else:
                failure = None

            if failure is None:
                stage = "joern"
                try:
                    graph = extract_function_cpg(
                        sample.source,
                        parsed.name,
                        language=sample.language,
                        joern_dir=joern_dir,
                        java_home=java_home,
                        timeout=timeout,
                    )
                except (CPGError, subprocess.TimeoutExpired, OSError) as error:
                    failure = error

            if failure is not None:
                failures += 1
                _write_jsonl_line(
                    errors,
                    {
                        "sample_key": sample.sample_key,
                        "line": sample.line_number,
                        "label": sample.label,
                        "split": sample.split,
                        "language": sample.language,
                        "stage": stage,
                        "error_type": type(failure).__name__,
                        "error": str(failure),
                    },
                )
                print(
                    f"function_sample_failed={sample.sample_key} stage={stage} error={failure}",
                    flush=True,
                )
                continue

            semantics = extract_vulnerability_semantics(graph)
            relations = extract_cpg_relations(graph)
            record: dict[str, object] = {
                "schema_version": DATASET_SCHEMA_VERSION,
                "sample_key": sample.sample_key,
                "label": sample.label,
                "language": sample.language,
                "function_name": parsed.name,
                "raw_source": sample.source,
                "cpg_relations": render_cpg_relations(graph),
                "vulnerability_semantics": semantics.render(),
                "semantic_items": semantics.as_json(),
                "vulnerability_features": list(semantics.feature_names),
                "cpg_relation_count": len(relations),
                "semantic_item_count": len(semantics.items),
                "split": sample.split,
            }
            _write_jsonl_line(output, record)
            records_by_key[sample.sample_key] = record
            completed.add(sample.sample_key)
            print(
                f"function_sample_done={sample.sample_key} label={sample.label} "
                f"cpg_relations={len(relations)} semantic_items={len(semantics.items)} "
                f"features={len(semantics.feature_names)}",
                flush=True,
            )

    records = _rewrite_in_sample_order(target, samples, records_by_key)

    total = len(samples)
    success = len(records)
    rate = success / total if total else 0.0
    print(
        f"function_build_summary total={total} success={success} failed={failures} "
        f"success_rate={rate:.2%} errors={errors_path}",
        flush=True,
    )
    return records
