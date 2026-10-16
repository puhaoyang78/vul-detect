import csv
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from vulnmechanism.data import (
    DATASETS,
    SPLITS,
    balanced,
    counts,
    json_array,
    load_native,
    prepare,
    split_projects,
    test_rows,
)


def row(key, source, label=0, split="train", project="p"):
    return dict(sample_key=key, raw_source=source, label=label, split=split, project=project)


class DataTests(unittest.TestCase):
    def test_native_sources_and_official_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "PrimeVul_v0.1").mkdir()
            for i, split in enumerate(SPLITS):
                (root / "PrimeVul_v0.1" / f"primevul_{split}.jsonl").write_text(
                    json.dumps(
                        dict(idx=i, target=i % 2, func="unparseable fragment", project="P")
                    )
                    + "\n"
                )
            rows, projects = load_native(root, "primevul")
            self.assertIsNone(projects)
            self.assertEqual([r["split"] for r in rows], list(SPLITS))
            self.assertEqual([r["label"] for r in rows], [0, 1, 0])
            self.assertTrue(all(r["raw_source"] == "unparseable fragment" for r in rows))
            (root / "megavul").mkdir()
            raw = [
                dict(
                    is_vul=bool(i % 2),
                    repo_name=f"P{i // 2}",
                    func=f"negative {i}",
                    func_before=f"positive {i}",
                    file_path="f.c",
                )
                for i in range(40)
            ]
            (root / "megavul/megavul.json").write_text(json.dumps(raw))
            rows, projects = load_native(root, "megavul")
            self.assertEqual(projects, load_native(root, "megavul")[1])
            self.assertEqual([counts(rows)[s]["positive"] for s in SPLITS], [16, 2, 2])
            for a, b in itertools.combinations(projects.values(), 2):
                self.assertFalse(set(a) & set(b))
            for i, r in enumerate(rows):
                self.assertEqual(
                    r["raw_source"], f"positive {i}" if i % 2 else f"negative {i}"
                )
                self.assertEqual(r["label"], i % 2)

    def test_project_split_targets_positive_counts_instead_of_project_counts(self):
        rows = [
            row(f"{p}-{i}", str(i), 1, project=p)
            for p, n in (("huge", 50), ("large", 30), ("a", 10), ("b", 10))
            for i in range(n)
        ]
        assignments = split_projects(rows)
        self.assertEqual([counts(rows)[s]["positive"] for s in SPLITS], [80, 10, 10])
        self.assertEqual(assignments, split_projects(rows))
        self.assertTrue(
            all(
                len({r["split"] for r in rows if r["project"] == p}) == 1
                for p in ("huge", "large", "a", "b")
            )
        )

    def test_balance_keeps_all_positives_duplicates_and_conflicting_labels(self):
        rows = [row(f"{s}-{i}", "same code", int(i < 3), s) for s in SPLITS for i in range(10)]
        selected = balanced(rows)
        self.assertEqual(selected, balanced(rows))
        self.assertEqual(
            {r["sample_key"] for r in selected if r["label"]},
            {r["sample_key"] for r in rows if r["label"]},
        )
        self.assertTrue(
            all(v["positive"] == v["negative"] == 3 for v in counts(selected).values())
        )
        with self.assertRaisesRegex(ValueError, "at least as many negatives"):
            balanced([r for r in rows if r["label"]])

    def test_prepare_preserves_positives_and_flags_only_exact_test_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            (raw / "PrimeVul_v0.1").mkdir(parents=True)
            expected_positive = set()
            for si, split in enumerate(SPLITS):
                rows = []
                for i in range(4):
                    source = (
                        "shared"
                        if i == 0 and split in ("train", "test")
                        else f"prime {split} {i}"
                    )
                    if split == "test" and i == 1:
                        source = "shared "  # whitespace similarity is deliberately not exact leakage
                    idx = si * 4 + i
                    rows.append(dict(idx=idx, target=int(i < 2), func=source))
                    if i < 2:
                        expected_positive.add(f"primevul:{idx}")
                (raw / "PrimeVul_v0.1" / f"primevul_{split}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows)
                )
            (raw / "megavul").mkdir()
            mega = [
                dict(
                    is_vul=bool(i % 2),
                    repo_name=f"P{i // 2}",
                    func=f"mega negative {i}",
                    func_before="shared" if i == 1 else f"mega positive {i}",
                )
                for i in range(40)
            ]
            (raw / "megavul/megavul.json").write_text(json.dumps(mega))
            output = root / "data"
            (output / "primevul").mkdir(parents=True)
            (output / "primevul/members.csv").write_text("published metadata\n")
            stats = prepare(raw, output)
            from vulnmechanism.data import read_jsonl

            prepared = {
                d: [r for s in SPLITS for r in read_jsonl(output / d / f"{s}.jsonl")]
                for d in DATASETS
            }
            self.assertEqual(
                {r["sample_key"] for r in prepared["primevul"] if r["label"]},
                expected_positive,
            )
            training = {
                d: {r["raw_source"] for r in prepared[d] if r["split"] == "train"}
                for d in DATASETS
            }
            for d in DATASETS:
                self.assertEqual(stats[d]["positives_removed"], 0)
                self.assertEqual(stats[d]["native"], stats[d]["balanced"])
                for r in prepared[d]:
                    if r["split"] == "test":
                        self.assertEqual(
                            r["train_overlap"],
                            [s for s in DATASETS if r["raw_source"] in training[s]],
                        )
                evaluated = test_rows(output / d)
                self.assertTrue(
                    all(
                        r["raw_source"] not in sources
                        for r in evaluated
                        for sources in training.values()
                    )
                )
                self.assertEqual(
                    stats[d]["test_exact_overlap"]["evaluated"]["samples"], len(evaluated)
                )
                with (output / d / "members.csv").open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), len(prepared[d]))
            prime_test = read_jsonl(output / "primevul/test.jsonl")
            self.assertTrue(prime_test[0]["train_overlap"])
            self.assertFalse(prime_test[1]["train_overlap"])
            with self.assertRaises(FileExistsError):
                prepare(raw, output)

    def test_streaming_array_handles_chunk_boundaries_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "array.json"
            rows = [dict(value="x" * (1024 * 1024 + 5)), dict(value="最后")]
            path.write_text(json.dumps(rows))
            self.assertEqual(list(json_array(path)), rows)
            path.write_text('[{"value": 1}')
            with self.assertRaises(ValueError):
                list(json_array(path))


if __name__ == "__main__":
    unittest.main()
