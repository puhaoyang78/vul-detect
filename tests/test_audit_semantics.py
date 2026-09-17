import json
from pathlib import Path
import tempfile
import unittest

from vulnmechanism.audit_semantics import audit_mechanism_fidelity


def row(pair_id, suffix, label, candidates, split="test", source="int f(void){}"):
    return {
        "sample_key": f"{pair_id}:{suffix}",
        "dataset": "cleanvul",
        "pair_id": pair_id,
        "split": split,
        "label": label,
        "raw_source": source,
        "mechanism_context": "\n".join(
            ["[MECHANISM_CANDIDATE]"]
            + [f"{kind} {detail}" for key, kind, state, detail in candidates]
        ) if candidates else "[VULNERABILITY_MECHANISM_CONTEXT]\nNO_CPG_DERIVED_MECHANISM_EVIDENCE",
        "mechanism_items": [
            {
                "category": "MECHANISM_CANDIDATE",
                "kind": kind,
                "detail": detail,
                "key": key,
                "state": state,
            }
            for key, kind, state, detail in candidates
        ],
    }


class MechanismAuditTests(unittest.TestCase):
    def test_pair_candidate_state_and_detail_changes_are_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            rows = [
                row("p1", "before", 1, [
                    (
                        "bounds|buf|n",
                        "BOUNDS_FLOW",
                        "bound_condition=not_observed|static_violation=no",
                        "source=parameter:n sink=MEMORY_WRITE object=buf expression=n capacity=64 occurrences=2",
                    ),
                    (
                        "null|p",
                        "NULL_DEREFERENCE_FLOW",
                        "null_condition=not_observed",
                        "source=nullable_allocation:malloc sink=POINTER_DEREFERENCE object=p occurrences=1",
                    ),
                ], source="int f(){ return 1; }") ,
                row("p1", "after", 0, [
                    (
                        "bounds|buf|n",
                        "BOUNDS_FLOW",
                        "bound_condition=present|static_violation=no",
                        "source=parameter:n sink=MEMORY_WRITE object=buf expression=n capacity=128 occurrences=9",
                    ),
                ], source="int f(){ return 0; }") ,
                row("p2", "before", 1, [
                    (
                        "uaf|p",
                        "USE_AFTER_FREE_FLOW",
                        "path_without_redefinition=present",
                        "source=deallocation sink=POINTER_DEREFERENCE object=p occurrences=1",
                    ),
                ]),
                row("p2", "after", 0, []),
            ]
            path.write_text("".join(json.dumps(value) + "\n" for value in rows))
            report = audit_mechanism_fidelity(path, dataset="cleanvul")

        self.assertEqual(report["complete_build_pairs"], 2)
        self.assertEqual(report["pairs_with_candidate_removed"], 2)
        self.assertEqual(report["pairs_with_candidate_state_changed"], 1)
        self.assertEqual(report["pairs_with_candidate_detail_changed"], 1)
        self.assertEqual(report["pairs_with_model_context_changed"], 2)
        self.assertEqual(report["pairs_with_source_change"], 1)
        self.assertEqual(report["candidate_removed_by_kind"]["NULL_DEREFERENCE_FLOW"], 1)
        self.assertEqual(report["candidate_removed_by_kind"]["USE_AFTER_FREE_FLOW"], 1)
        self.assertEqual(report["candidate_detail_changed_by_kind"]["BOUNDS_FLOW"], 1)
        self.assertIn(
            "BOUNDS_FLOW:bound_condition=not_observed|static_violation=no->bound_condition=present|static_violation=no",
            report["candidate_state_changes"],
        )
        # occurrence count alone must not define a mechanism-detail change
        before = report["candidate_detail_changes"][0]["before"]
        after = report["candidate_detail_changes"][0]["after"]
        self.assertNotIn("occurrences=", before)
        self.assertNotIn("occurrences=", after)

    def test_occurrence_only_change_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            rows = [
                row("p1", "before", 1, [
                    (
                        "bounds|buf|n",
                        "BOUNDS_FLOW",
                        "bound_condition=not_observed|static_violation=no",
                        "source=parameter:n sink=MEMORY_WRITE object=buf expression=n occurrences=1",
                    ),
                ]),
                row("p1", "after", 0, [
                    (
                        "bounds|buf|n",
                        "BOUNDS_FLOW",
                        "bound_condition=not_observed|static_violation=no",
                        "source=parameter:n sink=MEMORY_WRITE object=buf expression=n occurrences=8",
                    ),
                ]),
            ]
            path.write_text("".join(json.dumps(value) + "\n" for value in rows))
            report = audit_mechanism_fidelity(path, dataset="cleanvul")
        self.assertEqual(report["pairs_with_candidate_detail_changed"], 0)
        self.assertEqual(report["pairs_with_model_context_changed"], 0)

    def test_incomplete_build_pair_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            path.write_text(json.dumps(row(
                "p1", "before", 1,
                [(
                    "uaf|p",
                    "USE_AFTER_FREE_FLOW",
                    "path_without_redefinition=present",
                    "source=deallocation sink=POINTER_DEREFERENCE object=p occurrences=1",
                )],
            )) + "\n")
            report = audit_mechanism_fidelity(path, dataset="cleanvul")
        self.assertEqual(report["complete_build_pairs"], 0)
        self.assertEqual(report["incomplete_build_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
