from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
import subprocess
import time
from dataclasses import dataclass
from itertools import groupby, zip_longest
from pathlib import Path

from .cpg import (CPGError, CPGQualityError, TargetMethodError, FunctionGraph,
                  extract_function_cpg, extract_function_cpg_batch)
from .semantics import extract_vulnerability_semantics
from .syntax import resolve_language, target_hint


DATASET_SCHEMA_VERSION = 8
_VALID_DATASETS = {"primevul", "cleanvul", "sven"}
_VALID_SPLITS = {"train", "valid", "test", "external_test"}
_PAIRED_DATASETS = {"cleanvul", "sven"}
_GRAPH_KINDS = ("AST", "CFG", "CDG", "DDG")


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
    dataset: str
    source: str
    label: int
    language: str
    function_name: str | None
    split: str
    pair_id: str | None
    file_name: str


def _node_text(node) -> str:
    code = " ".join(node.code.split())
    return f"{node.label}:{code}"


def extract_cpg_relations(graph: FunctionGraph) -> tuple[CPGRelation, ...]:
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
    selected = (
        [item for row in zip_longest(*groups.values()) for item in row if item is not None]
        if groups
        else []
    )
    return "\n".join(item.as_text() for item in selected[:max_relations]) or "NO_CPG_RELATIONS"


def _required_string(record: dict[str, object], name: str, context: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: required non-empty string field {name!r}")
    return value


def _sample_fields(record: dict[str, object], line_number: int) -> FunctionSample:
    context = f"manifest line {line_number}"
    sample_key = _required_string(record, "sample_key", context)
    dataset = _required_string(record, "dataset", context).lower()
    if dataset not in _VALID_DATASETS:
        raise ValueError(f"{sample_key}: dataset must be one of {sorted(_VALID_DATASETS)}")
    source = _required_string(record, "function", context)
    label = record.get("label")
    if type(label) is not int or label not in {0, 1}:
        raise ValueError(f"{sample_key}: label must be integer 0 or 1")
    language = _required_string(record, "language", context).lower()
    if language not in {"c", "cpp", "c_cpp"}:
        raise ValueError(f"{sample_key}: language must be c, cpp, or c_cpp")
    split = _required_string(record, "split", context).lower()
    if split not in _VALID_SPLITS:
        raise ValueError(f"{sample_key}: split must be one of {sorted(_VALID_SPLITS)}")
    if dataset == "sven" and split != "external_test":
        raise ValueError(f"{sample_key}: SVEN must be external_test")
    if dataset != "sven" and split == "external_test":
        raise ValueError(f"{sample_key}: only SVEN may use external_test")

    pair_value = record.get("pair_id")
    pair_id = str(pair_value) if isinstance(pair_value, str) and pair_value else None
    if dataset in _PAIRED_DATASETS and pair_id is None:
        raise ValueError(f"{sample_key}: {dataset} requires pair_id")
    if dataset == "primevul" and pair_id is not None:
        raise ValueError(f"{sample_key}: PrimeVul formal classification rows must not use pair_id")

    function_name = record.get("function_name")
    if function_name is not None and not isinstance(function_name, str):
        raise ValueError(f"{sample_key}: function_name must be a string when present")
    file_name = record.get("file_name")
    if file_name is not None and not isinstance(file_name, str):
        raise ValueError(f"{sample_key}: file_name must be a string when present")

    return FunctionSample(
        line_number=line_number,
        sample_key=sample_key,
        dataset=dataset,
        source=source,
        label=label,
        language=language,
        function_name=function_name or None,
        split=split,
        pair_id=pair_id,
        file_name=file_name or "",
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
    if not samples:
        raise ValueError("formal manifest is empty")
    pairs = defaultdict(list)
    for sample in samples:
        if sample.pair_id:
            pairs[(sample.dataset, sample.pair_id)].append(sample)
    for (_, pair_id), pair in pairs.items():
        if len(pair) != 2 or {s.label for s in pair} != {0, 1} or len({s.split for s in pair}) != 1:
            raise ValueError(f"pair {pair_id!r} must contain both labels in one split")
    return samples


def _valid_semantic_items(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, dict)
        and isinstance(item.get("category"), str)
        and isinstance(item.get("kind"), str)
        and isinstance(item.get("detail"), str)
        for item in value
    )


def _resolve_sample(sample: FunctionSample):
    language = resolve_language(sample.source, sample.language, sample.file_name)
    return language, target_hint(sample.source, language, sample.function_name)


def _build_graphs(samples, *, batch_size, joern_dir, java_home, timeout):
    batches = []
    for _, partition in groupby(samples, key=lambda s: (s.dataset, s.split)):
        partition = list(partition)
        batches.extend(partition[i:i + batch_size] for i in range(0, len(partition), batch_size))
    for batch in batches:
        resolved = [_resolve_sample(sample) for sample in batch]
        started = time.monotonic()
        try:
            if batch_size == 1:
                sample = batch[0]
                language, hint = resolved[0]
                results = [extract_function_cpg(sample.source, hint.name, language=language,
                           joern_dir=joern_dir, java_home=java_home, timeout=timeout)]
            else:
                requests = [dict(source=sample.source, function=hint.name, language=language)
                            for sample, (language, hint) in zip(batch, resolved)]
                results = extract_function_cpg_batch(requests, joern_dir=joern_dir,
                                                    java_home=java_home, timeout=timeout)
        except (CPGError, subprocess.TimeoutExpired, OSError) as error:
            results = [error] * len(batch)
        seconds = time.monotonic() - started
        for sample, (language, hint), result in zip(batch, resolved, results):
            yield sample, language, hint, result, seconds / len(batch)


def _cpg_quality(graph: FunctionGraph) -> dict[str, object]:
    if graph.quality is not None:
        return graph.quality
    edge_counts = Counter(edge.kind for edge in graph.edges)
    missing = [kind for kind in _GRAPH_KINDS if edge_counts[kind] == 0]
    # AST and CFG are structural requirements; CDG/DDG may legitimately be empty.
    if edge_counts["AST"] == 0 or edge_counts["CFG"] == 0:
        raise CPGError(
            f"{graph.function}: structurally incomplete CPG "
            f"(AST={edge_counts['AST']}, CFG={edge_counts['CFG']})"
        )
    return {
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "edge_counts": {kind: edge_counts[kind] for kind in _GRAPH_KINDS},
        "missing_relation_kinds": missing,
    }


def _record_matches_sample(record: dict[str, object], sample: FunctionSample) -> bool:
    resolved_language, parsed = _resolve_sample(sample)
    return (
        record.get("dataset") == sample.dataset
        and record.get("pair_id") == sample.pair_id
        and record.get("raw_source") == sample.source
        and type(record.get("label")) is int
        and record.get("label") == sample.label
        and record.get("language") == sample.language
        and record.get("resolved_language") == resolved_language
        and isinstance(record.get("function_name"), str)
        and bool(record.get("function_name"))
        and record.get("split") == sample.split
        and isinstance(record.get("cpg_relations"), str)
        and bool(str(record.get("cpg_relations")).strip())
        and isinstance(record.get("vulnerability_semantics"), str)
        and _valid_semantic_items(record.get("semantic_items"))
        and isinstance(record.get("vulnerability_features"), list)
        and all(isinstance(value, str) for value in record.get("vulnerability_features", []))
        and isinstance(record.get("cpg_quality"), dict)
        and type(record.get("cpg_relation_count")) is int
        and type(record.get("semantic_item_count")) is int
    )


def _load_reusable_records(
    path: Path, samples: list[FunctionSample]
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
            if not _record_matches_sample(record, sample_by_key[key]):
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


def _build_audit(
    samples: list[FunctionSample], records: list[dict[str, object]], errors_path: Path
) -> dict[str, object]:
    success = {str(record["sample_key"]): record for record in records}
    failures = []
    if errors_path.is_file():
        failures = [json.loads(line) for line in errors_path.read_text().splitlines() if line.strip()]

    groups: dict[str, Counter] = defaultdict(Counter)
    for sample in samples:
        key = f"{sample.dataset}/{sample.split}/label_{sample.label}"
        groups[key]["input"] += 1
        if sample.sample_key in success:
            groups[key]["success"] += 1
        else:
            groups[key]["failed"] += 1

    for failure in failures:
        group = f"{failure['dataset']}/{failure['split']}/label_{failure['label']}"
        groups[group]["failed_" + str(failure["stage"])] += 1
    quality = Counter()
    pattern_count = Counter()
    for record in records:
        q = record["cpg_quality"]
        assert isinstance(q, dict)
        edge_counts = q.get("edge_counts", {})
        if isinstance(edge_counts, dict):
            for kind in _GRAPH_KINDS:
                if int(edge_counts.get(kind, 0)) == 0:
                    quality[f"success_without_{kind.lower()}"] += 1
        items = record.get("semantic_items", [])
        if not items:
            quality["success_without_semantic_items"] += 1
        patterns = sum(
            isinstance(item, dict) and item.get("category") == "POTENTIAL_PATTERN"
            for item in items
        )
        pattern_count[str(patterns)] += 1
        if patterns == 0:
            quality["success_without_potential_pattern"] += 1

    pairs = defaultdict(list)
    for sample in samples:
        if sample.pair_id:
            pairs[(sample.dataset, sample.pair_id)].append(sample)
    pair_status = Counter()
    for (dataset, _), pair in pairs.items():
        count = sum(sample.sample_key in success for sample in pair)
        pair_status[f"{dataset}/" + ("complete" if count == 2 else "incomplete" if count else "both_failed")] += 1
    failure_stage = Counter(str(row.get("stage")) for row in failures)
    failure_type = Counter(str(row.get("error_type")) for row in failures)
    group_report = {}
    for name, counts in sorted(groups.items()):
        entry = dict(sorted(counts.items()))
        entry['success_rate'] = counts['success'] / counts['input']
        for stage in ('syntax', 'joern', 'target_method', 'low_quality_cpg'):
            entry['failed_' + stage] = counts['failed_' + stage]
            entry['failure_rate_' + stage] = counts['failed_' + stage] / counts['input']
        group_report[name] = entry
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "total": len(samples),
        "success": len(records),
        "failed": len(samples) - len(records),
        "success_rate": len(records) / len(samples) if samples else 0.0,
        "groups": group_report,
        "pair_build_status": dict(sorted(pair_status.items())),
        "failure_stage": dict(sorted(failure_stage.items())),
        "failure_type": dict(sorted(failure_type.items())),
        "quality_flags": dict(sorted(quality.items())),
        "potential_pattern_count_distribution": dict(sorted(pattern_count.items(), key=lambda x: int(x[0]))),
        "quality_note": (
            "AST and CFG presence plus target-method alignment are enforced. Empty CDG/DDG or absence "
            "of a potential pattern is reported, not treated as failure; these facts do not by themselves "
            "prove or disprove the true vulnerability mechanism."
        ),
    }


def build_function_dataset(
    samples_path: str | Path,
    output_path: str | Path,
    *,
    joern_dir: str | Path = "/home/phy/joern",
    java_home: str | Path = "/home/phy/jdk21",
    timeout: int = 300,
    batch_size: int = 1,
) -> list[dict[str, object]]:
    build_started = time.monotonic()
    if batch_size < 1 or batch_size > 32:
        raise ValueError("batch_size must be between 1 and 32")
    samples_file = Path(samples_path)
    target = Path(output_path)
    errors_path = target.with_suffix(".errors.jsonl")
    audit_path = target.with_suffix(".audit.json")
    resolved_paths = {
        samples_file.resolve(),
        target.resolve(),
        errors_path.resolve(),
        audit_path.resolve(),
    }
    if len(resolved_paths) != 4:
        raise ValueError("samples, output, error log, and audit file must be different files")

    samples = _read_samples(samples_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    reusable, incomplete_tail = _load_reusable_records(target, samples)
    if incomplete_tail:
        print(f"discard_incomplete_output_tail={target}", flush=True)

    _rewrite_in_sample_order(target, samples, reusable)
    completed = set(reusable)
    print(f"function_samples_total={len(samples)} resumed={len(completed)}", flush=True)

    records_by_key = dict(reusable)
    current_failures: list[dict[str, object]] = []
    pending = [sample for sample in samples if sample.sample_key not in completed]
    with target.open("a", encoding="utf-8") as output, errors_path.open("w", encoding="utf-8") as error_output:
        for sample, resolved_language, parsed, result, seconds in _build_graphs(
                pending, batch_size=batch_size, joern_dir=joern_dir, java_home=java_home, timeout=timeout):
            postprocessing_started = time.monotonic()
            stage = "joern"
            try:
                if isinstance(result, TargetMethodError):
                    stage = "target_method"
                elif isinstance(result, CPGQualityError):
                    stage = "low_quality_cpg"
                if isinstance(result, BaseException):
                    raise result
                graph = result
                stage = "low_quality_cpg"
                quality = _cpg_quality(graph)
            except (ValueError, CPGError, subprocess.TimeoutExpired, OSError) as error:
                failure = {
                    "sample_key": sample.sample_key,
                    "dataset": sample.dataset,
                    "pair_id": sample.pair_id,
                    "line": sample.line_number,
                    "label": sample.label,
                    "split": sample.split,
                    "language": sample.language,
                    "resolved_language": resolved_language,
                    "seconds": seconds,
                    "syntax_hint": parsed.name,
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                current_failures.append(failure)
                _write_jsonl_line(error_output, failure)
                print(
                    f"function_sample_failed={sample.sample_key} stage={stage} error={error}",
                    flush=True,
                )
                continue

            semantics = extract_vulnerability_semantics(graph)
            relations = extract_cpg_relations(graph)
            record: dict[str, object] = {
                "schema_version": DATASET_SCHEMA_VERSION,
                "sample_key": sample.sample_key,
                "dataset": sample.dataset,
                "pair_id": sample.pair_id,
                "label": sample.label,
                "language": sample.language,
                "resolved_language": resolved_language,
                "function_name": graph.function,
                "syntax_hint": parsed.name,
                "seconds": seconds + time.monotonic() - postprocessing_started,
                "raw_source": sample.source,
                "cpg_relations": render_cpg_relations(graph),
                "vulnerability_semantics": semantics.render(),
                "semantic_items": semantics.as_json(),
                "vulnerability_features": list(semantics.feature_names),
                "cpg_quality": quality,
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
    failed_keys = {sample.sample_key for sample in samples} - set(records_by_key)
    # Recreate the error log for every currently failed sample. Reusable successes never remain here.
    failure_by_key = {str(row["sample_key"]): row for row in current_failures}
    with errors_path.open("w", encoding="utf-8") as errors:
        for sample in samples:
            if sample.sample_key in failed_keys:
                row = failure_by_key.get(sample.sample_key)
                if row is None:
                    row = {
                        "sample_key": sample.sample_key,
                        "dataset": sample.dataset,
                        "pair_id": sample.pair_id,
                        "line": sample.line_number,
                        "label": sample.label,
                        "split": sample.split,
                        "language": sample.language,
                        "resolved_language": None,
                        "stage": "unknown",
                        "error_type": "UnresolvedFailure",
                        "error": "sample did not produce a reusable current-schema record",
                    }
                _write_jsonl_line(errors, row)

    audit = _build_audit(samples, records, errors_path)
    audit["build_wall_seconds"] = time.monotonic() - build_started
    audit["resumed_records"] = len(reusable)
    audit["batch_size"] = batch_size
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"function_build_summary total={audit['total']} success={audit['success']} "
        f"failed={audit['failed']} success_rate={audit['success_rate']:.2%} "
        f"errors={errors_path} audit={audit_path}",
        flush=True,
    )
    return records
