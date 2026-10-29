"""Original binary Diff supervision with strictly bounded missing-side proxies."""

import difflib
import random
import tempfile
import unittest
from pathlib import Path
import torch

from tests.test_losver import Tokens, model
from vulnmechanism.data import (
    modified_lines, paired_encoding, paired_features, diff_neighbour_supervision,
    write_jsonl,
)
from vulnmechanism.training import LOSVER_VARIANTS, paired_collate, paired_forward
from vulnmechanism.artifacts import write_json
from vulnmechanism.evaluation import diff_neighbour_report


def pair(a, b, tokenizer=None):
    tokenizer = tokenizer or Tokens()
    rows = [dict(func=x, target=y, sample_key=str(y), pair_key="pair",
                 lines_ground_truth=modified_lines(x, other))
            for x, other, y in ((a, b, 1), (b, a, 0))]
    encodings = [paired_encoding(r["func"], tokenizer) for r in rows]
    augmented, audit = diff_neighbour_supervision(rows, encodings)
    return rows, augmented, audit, encodings


class DiffNeighbourTests(unittest.TestCase):
    def test_original_differ_labels_unchanged(self):
        rng = random.Random(42)
        for _ in range(200):
            a = "".join(rng.choices(["a();\n", "b();\n", "\n", " // x\n", "}\r\n"], k=8))
            b = "".join(rng.choices(["a();\n", "b();\n", "\n", " // x\n", "}\r\n"], k=8))
            expected, at = [], 0
            for s in difflib.Differ().compare(a.splitlines(True), b.splitlines(True)):
                if s.startswith("- "):
                    if s[2:].strip():
                        expected.append(at)
                    at += 1
                elif s.startswith("  "):
                    at += 1
            self.assertEqual(modified_lines(a, b), expected)

    def test_two_sides_and_same_edit_positions(self):
        a = "int f(){\n a();\n b();\n c();\n}\n"
        b = a.replace(" b();", " check();\n b();").replace(" c();", " check2();\n c();")
        rows, augmented, audit, _ = pair(a, b)
        self.assertEqual(augmented[0]["neighbour_lines"], [1, 2, 3])
        self.assertEqual([e["start"] for e in audit[0]["events"]], [2, 3])
        self.assertTrue(all(len(e["selected"]) == 2 for e in audit[0]["events"]))
        self.assertEqual(augmented[1]["lines_ground_truth"], rows[1]["lines_ground_truth"])
        self.assertEqual(augmented[0]["lines_original_diff"], [])

    def test_reverse_pure_deletion(self):
        p = "int f(){\n a();\n b();\n}\n"
        v = p.replace(" b();", " danger();\n b();")
        rows, augmented, audit, _ = pair(v, p)
        self.assertEqual(augmented[0]["lines_ground_truth"], rows[0]["lines_ground_truth"])
        self.assertEqual(augmented[1]["neighbour_lines"], [1, 2])
        self.assertTrue(audit[1]["eligible"])

    def test_one_comment_or_blank_but_not_two(self):
        for gap, expected in (("\n", [1, 3]), (" // note\n", [1, 3]),
                              (" /* note */\n", [1, 3]), ("\n // note\n", [4])):
            a = "int f(){\n a();\n" + gap + " b();\n}\n"
            b = a.replace(" b();", " check();\n b();")
            _, augmented, _, _ = pair(a, b)
            self.assertEqual(augmented[0]["neighbour_lines"], expected)

    def test_block_preprocessor_and_delimiter_boundaries(self):
        for boundary in ("}", "{", "if (x) {", "#endif", ");", ";"):
            a = "int f(){\n a();\n" + boundary + "\n b();\n}\n"
            b = a.replace(" b();", " check();\n b();")
            _, augmented, audit, _ = pair(a, b)
            self.assertEqual(augmented[0]["neighbour_lines"], [3])
            self.assertIn("before", audit[0]["events"][0]["rejected"])
        a = "int f(){\n#define X \\\n use(x)\n b();\n}\n"
        b = a.replace(" b();", " check();\n b();")
        self.assertEqual(pair(a, b)[1][0]["neighbour_lines"], [3])

    def test_braces_in_literals_or_comments_are_not_boundaries(self):
        a = 'int f(){\n puts("}"); // {\n b();\n}\n'
        b = a.replace(" b();", " check();\n b();")
        self.assertEqual(pair(a, b)[1][0]["neighbour_lines"], [1, 2])

    def test_replacement_and_existing_positives_unchanged(self):
        a = "int f(){\n a();\n b();\n}\n"
        b = a.replace("b()", "c()")
        rows, augmented, audit, _ = pair(a, b)
        for r, aug, detail in zip(rows, augmented, audit):
            self.assertEqual(r["lines_ground_truth"], aug["lines_ground_truth"])
            self.assertFalse(detail["eligible"])
        # Existing positives outside the input do not make a side eligible.
        a = "int f(){\n" + " x();\n" * 110 + " old();\n}\n"
        b = a.replace("old()", "new()")
        rows, augmented, audit, _ = pair(a, b)
        self.assertTrue(rows[0]["lines_ground_truth"])
        self.assertFalse(any(r["eligible"] for r in audit))

    def test_invisible_opposite_and_identical_inputs(self):
        a = "int f(){\n" + " x();\n" * 110 + " y();\n}\n"
        b = a.replace(" y();", " check();\n y();")
        _, augmented, audit, _ = pair(a, b)
        self.assertEqual(augmented[0]["neighbour_lines"], [])
        self.assertEqual(audit[0]["reason"], "identical_input")
        # Different visible leading whitespace, but the edit itself is invisible.
        b = " " + b
        _, augmented, audit, _ = pair(a, b)
        # This changes a header, so the side already has an original positive.
        self.assertFalse(audit[0]["eligible"])

    def test_only_visible_edit_is_used_when_input_is_not_identical(self):
        a = "int f(){\n" + "".join(f" a{i}();\n" for i in range(100)) + "}\n"
        b = a.replace(" a1();", " check();\n a1();").replace(" a90();", " check2();\n a90();")
        _, augmented, audit, _ = pair(a, b)
        self.assertFalse(audit[0]["identical_input"])
        self.assertEqual(augmented[0]["neighbour_lines"], [1, 2])
        self.assertEqual(audit[0]["events"][1]["reason"], "opposite_edit_not_fully_visible")

    def test_partially_visible_lines_are_not_supplemented(self):
        a = "int f(){\n a();\n" + " b" * 300 + ";\n}\n"
        b = a.replace(" a();\n", " a();\n check();\n")
        _, augmented, audit, enc = pair(a, b)
        self.assertIn(False, enc[0]["line_complete"])
        self.assertEqual(augmented[0]["neighbour_lines"], [1])
        self.assertEqual(audit[0]["events"][0]["rejected"]["after"], "neighbour_not_fully_visible")

    def test_partially_visible_opposite_edit_excludes_both_neighbours(self):
        a = "int f(){\n a();\n b();\n}\n"
        b = a.replace(" b();", " " + "check();" * 100 + "\n b();")
        _, augmented, audit, enc = pair(a, b)
        self.assertIn(False, enc[1]["line_complete"])
        self.assertEqual(augmented[0]["neighbour_lines"], [])
        self.assertEqual(audit[0]["events"][0]["reason"], "opposite_edit_not_fully_visible")

    def test_crlf_eof_and_whitespace_only_edit(self):
        a = "int f(){\r\n a();\r\n b();\r\n}"
        b = a.replace(" b();", " check();\r\n b();")
        self.assertEqual(pair(a, b)[1][0]["neighbour_lines"], [1, 2])
        b = a.replace(" b();", "\r\n b();")
        _, augmented, audit, _ = pair(a, b)
        self.assertFalse(any(r["eligible"] for r in audit))
        self.assertEqual(augmented[0]["neighbour_lines"], [])

    def test_binary_original_focal_and_encoder_backward(self):
        torch.set_num_threads(2)
        a = "int f(){\n a();\n b();\n}\n"
        b = a.replace(" b();", " check();\n b();")
        _, augmented, _, enc = pair(a, b)
        features = [paired_features(r, Tokens(), "localizer", encoding=e)
                    for r, e in zip(augmented, enc)]
        self.assertTrue(all(set(f["labels"]) <= {0, 1} for f in features))
        self.assertTrue(all("label_weights" not in f for f in features))
        locator = model(local=True)
        self.assertFalse(locator.weak_supervision)
        loss, scores = paired_forward(locator, paired_collate(features), "localizer", "cpu")
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(locator.encoder.embeddings.word_embeddings.weight.grad.norm()), 0)
        self.assertEqual(LOSVER_VARIANTS["diff_neighbour_localizer"], "localizer")
        self.assertEqual(LOSVER_VARIANTS["diff_neighbour"], "predicted")
        self.assertEqual(LOSVER_VARIANTS["repair_region_robust_fusion"], "predicted")

    def report_fixture(self, root):
        for split in ("train", "valid"):
            rows = [dict(sample_key=f"{split}:{i}", pair_key=f"{split}:{i//2}",
                         target=1-i%2, func="int f(){\n use();\n}",
                         lines_ground_truth=[] if i == 0 else [1]) for i in range(4)]
            write_jsonl(root / "data/primevul_paired" / f"{split}.lines.jsonl", rows)
            proxies = [dict(r, lines_ground_truth=[0, 1] if i == 0 else r["lines_ground_truth"])
                       for i, r in enumerate(rows)]
            write_jsonl(root / "new/supervision/diff_neighbour" / f"{split}.jsonl", proxies)
            write_jsonl(root / "new/supervision/diff_neighbour" / f"{split}.audit.jsonl", [
                dict(sample_key=r["sample_key"], eligible=i == 0, target=r["target"],
                     added_lines=[0, 1] if i == 0 else [], selection="both" if i == 0 else "none")
                for i, r in enumerate(rows)])
        write_json(root / "new/supervision/diff_neighbour/statistics.json", dict(splits={}))
        metadata = dict(training_members=[f"train:{i}" for i in range(4)],
                        validation_members=[f"valid:{i}" for i in range(4)],
                        input_mode="512", window_micro_batch=8, function_micro_batch=8,
                        gradient_accumulation=1, parameters=123,
                        training_config=dict(seed=123456), optimizer_steps=512, epochs=4,
                        pooling="hard_weight", weak_supervision=False, decision_threshold=.5,
                        selection="Valid", model_path="fixture", classifier_head="linear",
                        precision="FP32", train_samples=4, valid_samples=4,
                        selected_epochs=dict(f1=1, p_c=1))
        for output, localizer, detector in (("ref", "losver_localizer", "losver"),
                                            ("new", "diff_neighbour_localizer", "diff_neighbour")):
            for method in (localizer, detector):
                run = root / output / "primevul_paired" / method
                write_json(run / "training.metrics.json", metadata)
                write_jsonl(run / "best.training.jsonl", [dict(event="epoch", training_seconds=1,
                                                               epoch_seconds=2, optimizer_steps=512)])
            for split in ("train", "valid"):
                write_jsonl(root / output / "primevul_paired" / localizer / f"{split}.line_scores.jsonl", [
                    dict(sample_key=f"{split}:{i}", line_numbers=[0, 1], line_scores=[.2, .8]) for i in range(4)])
            write_jsonl(root / output / "primevul_paired" / detector / "valid.predictions.jsonl", [
                dict(sample_key=f"valid:{i}", pair_key=f"valid:{i//2}", target=1-i%2,
                     probability=.8 if i%2 == 0 else .2) for i in range(4)])

    def test_report_separates_original_and_proxy_labels_without_test(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.report_fixture(root)
            result = diff_neighbour_report(root / "new", root / "data", root / "ref")
            self.assertEqual([r["pairs"] for r in result["results"]], [2, 1, 2, 1])
            saved = json.loads((root / "new/comparison.json").read_text())
            loc = [r for r in saved["localizers"] if r["method"] == "diff_neighbour" and r["split"] == "valid"]
            self.assertEqual([r["lines"] for r in loc], [8, 8])
            self.assertEqual([r["positive_lines"] for r in loc], [3, 5])
            self.assertNotEqual(loc[0]["f1"], loc[1]["f1"])
            self.assertFalse((root / "data/primevul_paired/test.lines.jsonl").exists())

    def test_report_rejects_different_cohorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.report_fixture(root)
            path = root / "new/primevul_paired/diff_neighbour/valid.predictions.jsonl"
            write_jsonl(path, [dict(sample_key="other", probability=.5)])
            with self.assertRaisesRegex(ValueError, "fixed Valid cohort"):
                diff_neighbour_report(root / "new", root / "data", root / "ref")


if __name__ == "__main__":
    unittest.main()
