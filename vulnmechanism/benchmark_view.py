from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile


FORMAL_DATASETS = ("primevul", "cleanvul", "sven")
PAIRED_DATASETS = {"cleanvul", "sven"}
VALID_SPLITS = {"train", "valid", "test", "external_test"}


def record_split(record: dict[str, object]) -> str:
    split = record.get("split")
    if not isinstance(split, str) or split not in VALID_SPLITS:
        raise ValueError(f"record requires explicit split in {sorted(VALID_SPLITS)}, got {split!r}")
    return split


def record_dataset(record: dict[str, object]) -> str:
    dataset = record.get("dataset")
    if not isinstance(dataset, str) or dataset not in FORMAL_DATASETS:
        raise ValueError(
            f"record requires explicit dataset in {FORMAL_DATASETS}, got {dataset!r}"
        )
    return dataset


def record_pair_id(record: dict[str, object]) -> str | None:
    dataset = record_dataset(record)
    value = record.get("pair_id")
    if dataset not in PAIRED_DATASETS:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{dataset} record requires explicit pair_id")
    return value


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
            record_dataset(row)
            record_split(row)
            records.append(row)
    return records


def _complete_pairs(
    records: list[dict[str, object]], dataset: str
) -> tuple[list[dict[str, object]], int]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        pair_id = record_pair_id(record)
        assert pair_id is not None
        groups[pair_id].append(record)

    keep: set[str] = set()
    dropped_records = 0
    for pair_id, rows in groups.items():
        if len(rows) == 1:
            dropped_records += 1
            continue
        if len(rows) != 2:
            raise ValueError(f"{dataset} pair {pair_id!r} has {len(rows)} records; expected 2")
        labels = {row.get("label") for row in rows}
        if labels != {0, 1}:
            raise ValueError(f"{dataset} pair {pair_id!r} must contain labels 0 and 1")
        splits = {record_split(row) for row in rows}
        if len(splits) != 1:
            raise ValueError(f"{dataset} pair {pair_id!r} is split across {sorted(splits)}")
        keep.add(pair_id)

    return [record for record in records if record_pair_id(record) in keep], dropped_records


def _primevul_balance_key(record: dict[str, object]) -> tuple[str, str]:
    key = str(record["sample_key"])
    label = int(record["label"])
    return hashlib.sha256(
        (f"build-success-balanced-v2:{label}:" + key).encode()
    ).hexdigest(), key


def _balance_primevul(
    records: list[dict[str, object]],
) -> tuple[list[dict[str, object]], int]:
    selected: list[dict[str, object]] = []
    dropped = 0
    for split in ("train", "valid", "test"):
        rows = [record for record in records if record_split(record) == split]
        vulnerable = [record for record in rows if record.get("label") == 1]
        benign = [record for record in rows if record.get("label") == 0]
        if not vulnerable or not benign:
            raise ValueError(
                f"PrimeVul {split} requires build-success records from both labels: "
                f"benign={len(benign)}, vulnerable={len(vulnerable)}"
            )
        keep_per_label = min(len(vulnerable), len(benign))
        kept_vulnerable = sorted(vulnerable, key=_primevul_balance_key)[:keep_per_label]
        kept_benign = sorted(benign, key=_primevul_balance_key)[:keep_per_label]
        dropped += len(rows) - 2 * keep_per_label
        selected.extend(kept_vulnerable)
        selected.extend(kept_benign)
    selected.sort(
        key=lambda record: (
            ("train", "valid", "test").index(record_split(record)),
            str(record["sample_key"]),
        )
    )
    return selected, dropped


def select_source_records(
    records: list[dict[str, object]], source_dataset: str
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if source_dataset not in FORMAL_DATASETS:
        raise ValueError(
            f"source_dataset must be one of {', '.join(FORMAL_DATASETS)}, got {source_dataset!r}"
        )
    detected = Counter(record_dataset(record) for record in records)
    selected = [record for record in records if record_dataset(record) == source_dataset]
    if not selected:
        raise ValueError(f"no records found for source dataset {source_dataset!r}")

    dropped_incomplete_pair_records = 0
    dropped_primevul_balance_records = 0
    if source_dataset in PAIRED_DATASETS:
        selected, dropped_incomplete_pair_records = _complete_pairs(selected, source_dataset)
        if not selected:
            raise ValueError(f"no complete {source_dataset} pairs remain after build filtering")
    elif source_dataset == "primevul":
        selected, dropped_primevul_balance_records = _balance_primevul(selected)

    split_counts = Counter(record_split(record) for record in selected)
    label_counts = Counter(int(record["label"]) for record in selected)
    split_label_counts = Counter(
        (record_split(record), int(record["label"])) for record in selected
    )
    summary: dict[str, object] = {
        "source_dataset": source_dataset,
        "input_records": len(records),
        "selected_records": len(selected),
        "dropped_incomplete_pair_records": dropped_incomplete_pair_records,
        "dropped_primevul_balance_records": dropped_primevul_balance_records,
        "detected_sources": dict(sorted(detected.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": {str(key): value for key, value in sorted(label_counts.items())},
        "split_label_counts": {
            f"{split}/label_{label}": count
            for (split, label), count in sorted(split_label_counts.items())
        },
    }
    return selected, summary


@contextmanager
def dataset_view(path: str | Path, source_dataset: str):
    records = _read_jsonl(path)
    selected, summary = select_source_records(records, source_dataset)
    print(
        "benchmark_dataset_view=" + json.dumps(summary, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".jsonl",
        prefix="vulnmechanism-view-",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            for record in selected:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        yield str(temporary)
    finally:
        temporary.unlink(missing_ok=True)
