from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


PAIRED_DATASETS = {"cleanvul", "sven"}


def _records(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def _patterns(record: dict[str, object]) -> set[str]:
    items = record.get("semantic_items")
    if not isinstance(items, list):
        raise ValueError(f"{record.get('sample_key')}: semantic_items is missing or malformed")
    result = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"{record.get('sample_key')}: malformed semantic item")
        if item.get("category") == "POTENTIAL_PATTERN":
            kind = item.get("kind")
            if not isinstance(kind, str) or not kind:
                raise ValueError(f"{record.get('sample_key')}: malformed POTENTIAL_PATTERN")
            result.add(kind)
    return result


def audit_semantic_fidelity(
    dataset_path: str | Path,
    *,
    dataset: str,
) -> dict[str, object]:
    if dataset not in PAIRED_DATASETS:
        raise ValueError(f"dataset must be one of {sorted(PAIRED_DATASETS)}")
    rows = [row for row in _records(Path(dataset_path)) if row.get("dataset") == dataset]
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"{row.get('sample_key')}: paired record requires pair_id")
        groups[pair_id].append(row)

    complete = 0
    incomplete = 0
    before_with_pattern = 0
    after_with_pattern = 0
    any_removed = 0
    all_before_removed = 0
    unchanged_pattern_set = 0
    removed = Counter()
    persisted = Counter()
    added = Counter()
    by_split = Counter()

    for pair_id, pair_rows in sorted(groups.items()):
        if len(pair_rows) != 2 or {row.get("label") for row in pair_rows} != {0, 1}:
            incomplete += 1
            continue
        vulnerable = next(row for row in pair_rows if row["label"] == 1)
        fixed = next(row for row in pair_rows if row["label"] == 0)
        if vulnerable.get("split") != fixed.get("split"):
            raise ValueError(f"{pair_id}: pair split mismatch")
        complete += 1
        by_split[str(vulnerable.get("split"))] += 1
        before = _patterns(vulnerable)
        after = _patterns(fixed)
        before_with_pattern += bool(before)
        after_with_pattern += bool(after)
        removed_now = before - after
        persisted_now = before & after
        added_now = after - before
        any_removed += bool(removed_now)
        all_before_removed += bool(before) and not persisted_now
        unchanged_pattern_set += before == after
        removed.update(removed_now)
        persisted.update(persisted_now)
        added.update(added_now)

    def rate(value: int) -> float | None:
        return value / complete if complete else None

    report: dict[str, object] = {
        "dataset": dataset,
        "complete_build_pairs": complete,
        "incomplete_build_pairs": incomplete,
        "pairs_by_split": dict(sorted(by_split.items())),
        "before_with_potential_pattern": before_with_pattern,
        "after_with_potential_pattern": after_with_pattern,
        "before_pattern_coverage": rate(before_with_pattern),
        "after_pattern_coverage": rate(after_with_pattern),
        "pairs_with_any_pattern_removed": any_removed,
        "pair_pattern_removal_rate": rate(any_removed),
        "pairs_with_all_before_patterns_removed": all_before_removed,
        "all_before_patterns_removed_rate": rate(all_before_removed),
        "pairs_with_unchanged_pattern_set": unchanged_pattern_set,
        "unchanged_pattern_set_rate": rate(unchanged_pattern_set),
        "pattern_removed_counts": dict(sorted(removed.items())),
        "pattern_persisted_counts": dict(sorted(persisted.items())),
        "pattern_added_counts": dict(sorted(added.items())),
        "interpretation": (
            "This is a semantic-fidelity diagnostic, not ground-truth causal validation. "
            "A pattern disappearing after a real fix supports relevance; persistence does not necessarily "
            "mean the extraction is wrong because patches may address other mechanisms or the static "
            "representation may be path-insensitive."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether CPG-derived candidate patterns change across vulnerable/fixed pairs"
    )
    parser.add_argument("--dataset", default="data/function_dataset.jsonl")
    parser.add_argument("--source-dataset", choices=sorted(PAIRED_DATASETS), required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_semantic_fidelity(args.dataset, dataset=args.source_dataset)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
