from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


PAIRED_DATASETS = {"cleanvul", "sven"}
_OCCURRENCES = re.compile(r"\s+occurrences=\d+")


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


def _mechanism_detail(detail: str) -> str:
    """Normalize a candidate for semantic comparison.

    Occurrence count is intentionally excluded: repeated uses of the same
    mechanism are useful audit metadata but do not constitute a different
    source/relation/sink mechanism.
    """
    return " ".join(_OCCURRENCES.sub("", detail).split())


def _model_context(record: dict[str, object]) -> str:
    context = record.get("mechanism_context")
    if not isinstance(context, str):
        raise ValueError(f"{record.get('sample_key')}: mechanism_context is missing or malformed")
    return "\n".join(
        _OCCURRENCES.sub("", line).strip()
        for line in context.splitlines()
        if line.strip()
    )


def _candidates(record: dict[str, object]) -> dict[str, tuple[str, str, str]]:
    items = record.get("mechanism_items")
    if not isinstance(items, list):
        raise ValueError(f"{record.get('sample_key')}: mechanism_items is missing or malformed")
    result: dict[str, tuple[str, str, str]] = {}
    for item in items:
        if not isinstance(item, dict) or item.get("category") != "MECHANISM_CANDIDATE":
            continue
        kind = item.get("kind")
        key = item.get("key")
        state = item.get("state", "")
        detail = item.get("detail", "")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"{record.get('sample_key')}: candidate kind is malformed")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{record.get('sample_key')}: candidate key is malformed")
        if not isinstance(state, str):
            raise ValueError(f"{record.get('sample_key')}: candidate state is malformed")
        if not isinstance(detail, str):
            raise ValueError(f"{record.get('sample_key')}: candidate detail is malformed")
        result[key] = (kind, state, _mechanism_detail(detail))
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
    any_detail_changed = 0
    any_model_context_changed = 0
    any_source_changed = 0
    unchanged = 0
    removed_kinds = Counter()
    added_kinds = Counter()
    state_changes = Counter()
    detail_change_kinds = Counter()
    persisted_kinds = Counter()
    candidate_count_before = Counter()
    candidate_count_after = Counter()
    by_split = Counter()
    detail_changes: list[dict[str, str]] = []

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
        state_changed = {
            key for key in persisted
            if before[key][1] != after[key][1]
        }
        detail_changed = {
            key for key in persisted
            if before[key][2] != after[key][2]
        }

        source_changed = vulnerable.get("raw_source") != fixed.get("raw_source")
        model_context_changed = _model_context(vulnerable) != _model_context(fixed)

        any_removed += bool(removed)
        any_added += bool(added)
        any_state_changed += bool(state_changed)
        any_detail_changed += bool(detail_changed)
        any_source_changed += bool(source_changed)
        any_model_context_changed += bool(model_context_changed)
        unchanged += not removed and not added and not state_changed and not detail_changed

        for key in removed:
            removed_kinds[before[key][0]] += 1
        for key in added:
            added_kinds[after[key][0]] += 1
        for key in persisted:
            persisted_kinds[before[key][0]] += 1
        for key in state_changed:
            kind = before[key][0]
            state_changes[f"{kind}:{before[key][1]}->{after[key][1]}"] += 1
        for key in detail_changed:
            kind = before[key][0]
            detail_change_kinds[kind] += 1
            if len(detail_changes) < 100:
                detail_changes.append({
                    "pair_id": pair_id,
                    "key": key,
                    "kind": kind,
                    "before": before[key][2],
                    "after": after[key][2],
                })

    def rate(value: int) -> float | None:
        return value / complete if complete else None

    return {
        "dataset": dataset,
        "complete_build_pairs": complete,
        "incomplete_build_pairs": incomplete,
        "pairs_by_split": dict(sorted(by_split.items())),
        "pairs_with_source_change": any_source_changed,
        "source_change_rate": rate(any_source_changed),
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
        "pairs_with_candidate_detail_changed": any_detail_changed,
        "candidate_detail_change_rate": rate(any_detail_changed),
        "pairs_with_model_context_changed": any_model_context_changed,
        "model_context_change_rate": rate(any_model_context_changed),
        "pairs_with_unchanged_mechanism_candidates": unchanged,
        "unchanged_mechanism_rate": rate(unchanged),
        "candidate_removed_by_kind": dict(sorted(removed_kinds.items())),
        "candidate_added_by_kind": dict(sorted(added_kinds.items())),
        "candidate_persisted_by_kind": dict(sorted(persisted_kinds.items())),
        "candidate_state_changes": dict(sorted(state_changes.items())),
        "candidate_detail_changed_by_kind": dict(sorted(detail_change_kinds.items())),
        "candidate_detail_changes": detail_changes,
        "candidate_count_distribution_before": dict(
            sorted(candidate_count_before.items(), key=lambda x: int(x[0]))
        ),
        "candidate_count_distribution_after": dict(
            sorted(candidate_count_after.items(), key=lambda x: int(x[0]))
        ),
        "interpretation": (
            "This diagnostic separates candidate addition/removal, constraint-state changes, and concrete "
            "mechanism-detail changes. The model-context metric compares the actual mechanism text given to "
            "the Code LLM while ignoring occurrence counts. These are fidelity diagnostics, not causal ground truth."
        ),
    }


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
