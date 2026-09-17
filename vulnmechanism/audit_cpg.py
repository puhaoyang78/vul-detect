"""Audit current schema-9 CPG build outputs and mechanism evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .dataset import DATASET_SCHEMA_VERSION
from .semantics import render_mechanism_items


def _rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def refresh_mechanism_context(folder: Path) -> None:
    path = folder / "function_only.jsonl"
    rows = _rows(path)
    for row in rows:
        items = row.get("mechanism_items")
        if not isinstance(items, list):
            raise ValueError(f"{row.get('sample_key')}: mechanism_items missing")
        row["mechanism_context"] = render_mechanism_items(items)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def summarize(folder: Path) -> dict[str, object]:
    built_path = folder / "function_only.jsonl"
    errors_path = folder / "function_only.errors.jsonl"
    audit_path = folder / "function_only.audit.json"
    built = _rows(built_path)
    errors = _rows(errors_path)
    build_audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}

    candidate_counts = Counter()
    candidate_kinds = Counter()
    relation_kinds = Counter()
    operation_kinds = Counter()
    states = Counter()
    warning_counts = Counter()
    no_candidate = 0

    for row in built:
        if row.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise ValueError(
                f"{row.get('sample_key')}: expected schema {DATASET_SCHEMA_VERSION}, "
                f"got {row.get('schema_version')}"
            )
        items = row.get("mechanism_items")
        if not isinstance(items, list):
            raise ValueError(f"{row.get('sample_key')}: mechanism_items missing")
        rendered = render_mechanism_items(items)
        if row.get("mechanism_context") != rendered:
            raise ValueError(f"{row.get('sample_key')}: mechanism_context is stale")

        candidates = [
            item for item in items
            if isinstance(item, dict) and item.get("category") == "MECHANISM_CANDIDATE"
        ]
        candidate_counts[str(len(candidates))] += 1
        no_candidate += not candidates
        for item in items:
            if not isinstance(item, dict):
                continue
            category = item.get("category")
            kind = str(item.get("kind") or "")
            state = str(item.get("state") or "")
            if category == "MECHANISM_CANDIDATE":
                candidate_kinds[kind] += 1
                if state:
                    states[f"{kind}:{state}"] += 1
            elif category == "MECHANISM_RELATION":
                relation_kinds[kind] += 1
            elif category == "SECURITY_OPERATION":
                operation_kinds[kind] += 1

        quality = row.get("cpg_quality")
        if isinstance(quality, dict):
            for warning in quality.get("warnings", ()):
                warning_counts[str(warning)] += 1

    failure_stage = Counter(str(row.get("stage")) for row in errors)
    report = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "built_samples": len(built),
        "failed_samples": len(errors),
        "failure_stage": dict(sorted(failure_stage.items())),
        "samples_without_mechanism_candidate": no_candidate,
        "mechanism_candidate_count_distribution": dict(
            sorted(candidate_counts.items(), key=lambda pair: int(pair[0]))
        ),
        "mechanism_candidate_kind_counts": dict(sorted(candidate_kinds.items())),
        "mechanism_candidate_states": dict(sorted(states.items())),
        "mechanism_relation_kind_counts": dict(sorted(relation_kinds.items())),
        "security_operation_kind_counts": dict(sorted(operation_kinds.items())),
        "cpg_warning_counts": dict(sorted(warning_counts.items())),
        "build_audit": build_audit,
        "interpretation": (
            "Security operations are facts, mechanism relations are CPG-supported relations, and mechanism "
            "candidates are aggregated relation-level hypotheses. Candidate absence is not a benign label, and "
            "candidate presence is not causal ground truth."
        ),
    }
    (folder / "mechanism_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path("results/cpg_debug"))
    parser.add_argument("--refresh-mechanism-context", action="store_true")
    args = parser.parse_args()
    if args.refresh_mechanism_context:
        refresh_mechanism_context(args.folder)
    else:
        summarize(args.folder)


if __name__ == "__main__":
    main()
