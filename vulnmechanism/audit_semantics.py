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


def _candidates(record: dict[str, object]) -> dict[str, tuple[str, str]]:
    items = record.get("mechanism_items")
    if not isinstance(items, list):
        raise ValueError(f"{record.get('sample_key')}: mechanism_items is missing or malformed")
    result: dict[str, tuple[str, str]] = {}
    for item in items:
        if not isinstance(item, dict) or item.get("category") != "MECHANISM_CANDIDATE":
            continue
        kind = item.get("kind")
        key = item.get("key")
        state = item.get("state", "")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"{record.get('sample_key')}: candidate kind is malformed")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{record.get('sample_key')}: candidate key is malformed")
        if not isinstance(state, str):
            raise ValueError(f"{record.get('sample_key')}: candidate state is malformed")
        result[key] = (kind, state)
    return result


def audit_mechanism_fidelity(
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
    before_with_candidate = 0
    after_with_candidate = 0
    any_removed = 0
    any_added = 0
    any_state_changed = 0
    unchanged = 0
    removed_kinds = Counter()
    added_kinds = Counter()
    state_changes = Counter()
    persisted_kinds = Counter()
    candidate_count_before = Counter()
    candidate_count_after = Counter()
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
        before = _candidates(vulnerable)
        after = _candidates(fixed)
        before_with_candidate += bool(before)
        after_with_candidate += bool(after)
        candidate_count_before[str(len(before))] += 1
        candidate_count_after[str(len(after))] += 1

        before_keys = set(before)
        after_keys = set(after)
        removed = before_keys - after_keys
        added = after_keys - before_keys
        persisted = before_keys & after_keys
        changed = {
            key for key in persisted
            if before[key][1] != after[key][1]
        }

        any_removed += bool(removed)
        any_added += bool(added)
        any_state_changed += bool(changed)
        unchanged += not removed and not added and not changed

        for key in removed:
            removed_kinds[before[key][0]] += 1
        for key in added:
            added_kinds[after[key][0]] += 1
        for key in persisted:
            persisted_kinds[before[key][0]] += 1
        for key in changed:
            kind = before[key][0]
            state_changes[f"{kind}:{before[key][1]}->{after[key][1]}"] += 1

    def rate(value: int) -> float | None:
        return value / complete if complete else None

    return {
        "dataset": dataset,
        "complete_build_pairs": complete,
        "incomplete_build_pairs": incomplete,
        "pairs_by_split": dict(sorted(by_split.items())),
        "before_with_mechanism_candidate": before_with_candidate,
        "after_with_mechanism_candidate": after_with_candidate,
        "before_candidate_coverage": rate(before_with_candidate),
        "after_candidate_coverage": rate(after_with_candidate),
        "pairs_with_candidate_removed": any_removed,
        "candidate_removal_rate": rate(any_removed),
        "pairs_with_candidate_added": any_added,
        "candidate_addition_rate": rate(any_added),
        "pairs_with_candidate_state_changed": any_state_changed,
        "candidate_state_change_rate": rate(any_state_changed),
        "pairs_with_unchanged_mechanism_candidates": unchanged,
        "unchanged_mechanism_rate": rate(unchanged),
        "candidate_removed_by_kind": dict(sorted(removed_kinds.items())),
        "candidate_added_by_kind": dict(sorted(added_kinds.items())),
        "candidate_persisted_by_kind": dict(sorted(persisted_kinds.items())),
        "candidate_state_changes": dict(sorted(state_changes.items())),
        "candidate_count_distribution_before": dict(sorted(candidate_count_before.items(), key=lambda x: int(x[0]))),
        "candidate_count_distribution_after": dict(sorted(candidate_count_after.items(), key=lambda x: int(x[0]))),
        "interpretation": (
            "This diagnostic compares concrete mechanism keys and their states across real vulnerable/fixed pairs. "
            "Removal or a state change after a fix supports relevance but is not causal ground truth; persistence can "
            "be legitimate when the patch addresses another mechanism or when function-only CPG evidence is incomplete."
        ),
    }


# Kept only as the public function name used by existing scripts; its semantics are
# now mechanism-level rather than old pattern-kind comparison.
def audit_semantic_fidelity(dataset_path: str | Path, *, dataset: str) -> dict[str, object]:
    return audit_mechanism_fidelity(dataset_path, dataset=dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit CPG-derived vulnerability-mechanism changes across vulnerable/fixed pairs"
    )
    parser.add_argument("--dataset", default="data/function_dataset.jsonl")
    parser.add_argument("--source-dataset", choices=sorted(PAIRED_DATASETS), required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_mechanism_fidelity(args.dataset, dataset=args.source_dataset)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
