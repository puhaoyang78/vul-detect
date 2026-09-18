import unittest

from vulnmechanism.benchmark_view import record_pair_id, record_split, select_source_records


class BenchmarkViewTests(unittest.TestCase):
    def test_primevul_build_success_is_balanced_per_split(self):
        records = []
        for split in ("train", "valid", "test"):
            records.extend(
                [
                    dict(dataset="primevul", sample_key=f"primevul:{split}:v1", split=split, label=1),
                    dict(dataset="primevul", sample_key=f"primevul:{split}:v2", split=split, label=1),
                    dict(dataset="primevul", sample_key=f"primevul:{split}:b1", split=split, label=0),
                    dict(dataset="primevul", sample_key=f"primevul:{split}:b2", split=split, label=0),
                    dict(dataset="primevul", sample_key=f"primevul:{split}:b3", split=split, label=0),
                ]
            )
        selected, summary = select_source_records(records, "primevul")
        for split in ("train", "valid", "test"):
            labels = [row["label"] for row in selected if record_split(row) == split]
            self.assertEqual(labels.count(1), 2)
            self.assertEqual(labels.count(0), 2)
        self.assertEqual(summary["dropped_primevul_balance_records"], 3)
        again, _ = select_source_records(list(reversed(records)), "primevul")
        self.assertEqual(
            {row["sample_key"] for row in selected},
            {row["sample_key"] for row in again},
        )

    def test_primevul_balance_handles_fewer_benign_than_vulnerable(self):
        records = []
        for split in ("train", "valid", "test"):
            records.extend([
                dict(dataset="primevul", sample_key=f"primevul:{split}:v1", split=split, label=1),
                dict(dataset="primevul", sample_key=f"primevul:{split}:v2", split=split, label=1),
                dict(dataset="primevul", sample_key=f"primevul:{split}:v3", split=split, label=1),
                dict(dataset="primevul", sample_key=f"primevul:{split}:b1", split=split, label=0),
                dict(dataset="primevul", sample_key=f"primevul:{split}:b2", split=split, label=0),
            ])
        selected, summary = select_source_records(records, "primevul")
        for split in ("train", "valid", "test"):
            labels = [row["label"] for row in selected if record_split(row) == split]
            self.assertEqual(labels.count(1), 2)
            self.assertEqual(labels.count(0), 2)
        self.assertEqual(summary["dropped_primevul_balance_records"], 3)

    def test_incomplete_pairs_are_removed_atomically(self):
        records = [
            dict(dataset="cleanvul", pair_id="cleanvul:4:1", sample_key="cleanvul:4:1:before", split="train", label=1),
            dict(dataset="cleanvul", pair_id="cleanvul:4:1", sample_key="cleanvul:4:1:after", split="train", label=0),
            dict(dataset="cleanvul", pair_id="cleanvul:4:2", sample_key="cleanvul:4:2:before", split="valid", label=1),
        ]
        selected, summary = select_source_records(records, "cleanvul")
        self.assertEqual(
            {row["sample_key"] for row in selected},
            {"cleanvul:4:1:before", "cleanvul:4:1:after"},
        )
        self.assertEqual(summary["dropped_incomplete_pair_records"], 1)
        self.assertEqual(record_pair_id(selected[0]), "cleanvul:4:1")

    def test_explicit_metadata_is_required(self):
        with self.assertRaisesRegex(ValueError, "explicit split"):
            record_split({"dataset": "primevul", "sample_key": "x"})
        with self.assertRaisesRegex(ValueError, "explicit dataset"):
            select_source_records([{"sample_key": "primevul:1", "split": "train", "label": 1}], "primevul")
        with self.assertRaisesRegex(ValueError, "requires explicit pair_id"):
            select_source_records(
                [
                    {"dataset": "sven", "sample_key": "sven:x:before", "split": "external_test", "label": 1},
                    {"dataset": "sven", "sample_key": "sven:x:after", "split": "external_test", "label": 0},
                ],
                "sven",
            )


if __name__ == "__main__":
    unittest.main()
