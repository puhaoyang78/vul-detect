import json
from pathlib import Path
import tempfile
import unittest

from vulnmechanism.audit_semantics import audit_mechanism_fidelity


def row(pair_id, suffix, label, candidates, split="test"):
    return {
        "sample_key": f"{pair_id}:{suffix}",
        "dataset": "cleanvul",
        "pair_id": pair_id,
        "split": split,
        "label": label,
        "mechanism_items": [
            {
                "category": "MECHANISM_CANDIDATE",
                "kind": kind,
                "detail": f"key={key}",
                "key": key,
                "state": state,
            }
            for key, kind, state in candidates
        ],
    }


class MechanismAuditTests(unittest.TestCase):
    def test_pair_candidate_and_state_changes_are_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            rows = [
                row("p1", "before", 1, [
                    ("bounds|buf|n", "BOUNDS_FLOW", "bound_condition=not_observed|static_violation=no"),
                    ("null|p", "NULL_DEREFERENCE_FLOW", "null_condition=not_observed"),
                ]),
                row("p1", "after", 0, [
                    ("bounds|buf|n", "BOUNDS_FLOW", "bound_condition=present|static_violation=no"),
                ]),
                row("p2", "before", 1, [
                    ("uaf|p", "USE_AFTER_FREE_FLOW", "path_without_redefinition=present"),
                ]),
                row("p2", "after", 0, []),
            ]
            path.write_text("".join(json.dumps(value) + "\n" for value in rows))
            report = audit_mechanism_fidelity(path, dataset="cleanvul")

        self.assertEqual(report["complete_build_pairs"], 2)
        self.assertEqual(report["pairs_with_candidate_removed"], 2)
        self.assertEqual(report["pairs_with_candidate_state_changed"], 1)
        self.assertEqual(report["candidate_removed_by_kind"]["NULL_DEREFERENCE_FLOW"], 1)
        self.assertEqual(report["candidate_removed_by_kind"]["USE_AFTER_FREE_FLOW"], 1)
        self.assertIn(
            "BOUNDS_FLOW:bound_condition=not_observed|static_violation=no->bound_condition=present|static_violation=no",
            report["candidate_state_changes"],
        )

    def test_incomplete_build_pair_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            path.write_text(json.dumps(row(
                "p1", "before", 1,
                [("uaf|p", "USE_AFTER_FREE_FLOW", "path_without_redefinition=present")],
            )) + "\n")
            report = audit_mechanism_fidelity(path, dataset="cleanvul")
        self.assertEqual(report["complete_build_pairs"], 0)
        self.assertEqual(report["incomplete_build_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
