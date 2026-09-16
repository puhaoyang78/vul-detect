from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile


FORMAL_DATASETS = ("primevul", "cleanvul", "sven")
PAIRED_DATASETS = {"cleanvul", "sven"}
VALID_SPLITS = {"train", "valid", "validation", "test", "external_test"}


def _split_for_key(key: str) -> str:
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "valid"
    return "test"


def record_split(record: dict[str, object]) -> str:
    split = str(record.get("split") or "").lower()
    if split in VALID_SPLITS:
        return "valid" if split == "validation" else split
    if split:
        raise ValueError(f"unknown explicit dataset split: {split!r}")
    key = str(record.get("sample_key") or "")
    if not key:
        raise ValueError("record without an explicit split requires sample_key")
    return _split_for_key(key)


def record_dataset(record: dict[str, object]) -> str | None:
    explicit = str(record.get("dataset") or "").strip().lower()
    if explicit:
        if explicit not in FORMAL_DATASETS:
            raise ValueError(f"unknown formal dataset source: {explicit!r}")
        return explicit
    key = str(record.get("sample_key") or "")
    prefix = key.split(":", 1)[0].lower() if ":" in key else ""
    return prefix if prefix in FORMAL_DATASETS else None


def record_pair_id(record: dict[str, object]) -> str | None:
    explicit = str(record.get("pair_id") or "").strip()
    if explicit:
        return explicit
    dataset = record_dataset(record)
    if dataset not in PAIRED_DATASETS:
        return None
    key = str(record.get("sample_key") or "")
    if ":" not in key:
        return None
    pair, role = key.rsplit(":", 1)
    return pair if role in {"before", "after"} else None


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            records.append(row)
    return records


def _complete_pairs(records: list[dict[str, object]], dataset: str) -> tuple[list[dict[str, object]], int]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        pair_id = record_pair_id(record)
        if pair_id is None:
            raise ValueError(
                f"{dataset} record {record.get('sample_key')!r} has no recoverable pair identity"
            )
        groups[pair_id].append(record)

    keep: set[str] = set()
    dropped_records = 0
    for pair_id, rows in groups.items():
        if len(rows) == 1:
            dropped_records += 1
            continue
        if len(rows) != 2:
            raise ValueError(f"{dataset} pair {pair_id!r} has {len(rows)} records; expected 2")
        labels = {int(row.get("label")) for row in rows if row.get("label") in {0, 1}}
        if labels != {0, 1}:
            raise ValueError(f"{dataset} pair {pair_id!r} must contain labels 0 and 1")
        splits = {record_split(row) for row in rows}
        if len(splits) != 1:
            raise ValueError(f"{dataset} pair {pair_id!r} is split across {sorted(splits)}")
        keep.add(pair_id)

    filtered = [record for record in records if record_pair_id(record) in keep]
    return filtered, dropped_records


def select_source_records(
    records: list[dict[str, object]],
    source_dataset: str | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    detected = Counter(
        dataset for record in records if (dataset := record_dataset(record)) is not None
    )
    formal_sources = sorted(detected)

    if source_dataset is None:
        if len(formal_sources) > 1:
            raise ValueError(
                "dataset contains multiple formal benchmark sources "
                f"{formal_sources}; specify --source-dataset"
            )
        selected = list(records)
        resolved_source = formal_sources[0] if formal_sources else None
    else:
        source_dataset = source_dataset.lower()
        if source_dataset not in FORMAL_DATASETS:
            raise ValueError(
                f"source_dataset must be one of {', '.join(FORMAL_DATASETS)}, got {source_dataset!r}"
            )
        selected = [record for record in records if record_dataset(record) == source_dataset]
        resolved_source = source_dataset
        if not selected:
            raise ValueError(f"no records found for source dataset {source_dataset!r}")

    dropped_incomplete_pair_records = 0
    if resolved_source in PAIRED_DATASETS:
        selected, dropped_incomplete_pair_records = _complete_pairs(selected, resolved_source)
        if not selected:
            raise ValueError(f"no complete {resolved_source} pairs remain after build filtering")

    split_counts = Counter(record_split(record) for record in selected)
    label_counts = Counter(int(record["label"]) for record in selected if record.get("label") in {0, 1})
    summary: dict[str, object] = {
        "source_dataset": resolved_source,
        "input_records": len(records),
        "selected_records": len(selected),
        "dropped_incomplete_pair_records": dropped_incomplete_pair_records,
        "detected_sources": dict(sorted(detected.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": {str(key): value for key, value in sorted(label_counts.items())},
    }
    return selected, summary


@contextmanager
def dataset_view(path: str | Path, source_dataset: str | None):
    records = _read_jsonl(path)
    selected, summary = select_source_records(records, source_dataset)
    print("benchmark_dataset_view=" + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)

    # Legacy single-source datasets need no temporary copy.
    if len(selected) == len(records) and source_dataset is None:
        yield str(path)
        return

    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".jsonl", prefix="vulnmechanism-view-", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            for record in selected:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        yield str(temporary)
    finally:
        temporary.unlink(missing_ok=True)
