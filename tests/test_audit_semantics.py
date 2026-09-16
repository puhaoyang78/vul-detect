import json
from pathlib import Path
import tempfile
import unittest

from vulnmechanism.audit_semantics import audit_semantic_fidelity


def row(pair_id, suffix, label, patterns, split="test"):
    return {
        "sample_key": f"{pair_id}:{suffix}",
        "dataset": "cleanvul",
        "pair_id": pair_id,
        "split": split,
        "label": label,
        "semantic_items": [
            {"category": "POTENTIAL_PATTERN", "kind": kind, "detail": "x"}
            for kind in patterns
        ],
    }


class SemanticAuditTests(unittest.TestCase):
    def test_pair_pattern_changes_are_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            rows = [
                row("p1", "before", 1, ["FREE_THEN_USE", "DEREFERENCE_WITHOUT_NONNULL_CONDITION"]),
                row("p1", "after", 0, ["DEREFERENCE_WITHOUT_NONNULL_CONDITION"]),
                row("p2", "before", 1, ["UNBOUNDED_WRITE"]),
                row("p2", "after", 0, []),
            ]
            path.write_text("".join(json.dumps(value) + "\n" for value in rows))
            report = audit_semantic_fidelity(path, dataset="cleanvul")
        self.assertEqual(report["complete_build_pairs"], 2)
        self.assertEqual(report["pairs_with_any_pattern_removed"], 2)
        self.assertEqual(report["pattern_removed_counts"]["FREE_THEN_USE"], 1)
        self.assertEqual(report["pattern_removed_counts"]["UNBOUNDED_WRITE"], 1)
        self.assertEqual(
            report["pattern_persisted_counts"]["DEREFERENCE_WITHOUT_NONNULL_CONDITION"],
            1,
        )

    def test_incomplete_build_pair_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            path.write_text(json.dumps(row("p1", "before", 1, ["FREE_THEN_USE"])) + "\n")
            report = audit_semantic_fidelity(path, dataset="cleanvul")
        self.assertEqual(report["complete_build_pairs"], 0)
        self.assertEqual(report["incomplete_build_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
