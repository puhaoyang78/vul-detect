"""Pair isolation, OOF provenance and the unchanged source-only detector path."""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tests.test_losver import Tokens, model
from vulnmechanism import model as core, training
from vulnmechanism.artifacts import write_json
from vulnmechanism.data import (
    paired_encoding, paired_folds, prepare_oof_predictions, read_jsonl, write_jsonl,
)


def rows(split, pairs):
    return [dict(sample_key=f"{split}:{i}", pair_key=f"{split}:{i//2}",
                 target=1-i%2, func=f"int f{i}(){{\n return {i};\n}}")
            for i in range(pairs * 2)]


class OOFTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_folds_keep_pairs_and_rng(self):
        source = rows("train", 8)
        state = random.getstate()
        folds = paired_folds(source)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(folds, paired_folds(list(reversed(source))))
        self.assertEqual(list(folds.values()).count(0), 4)
        self.assertEqual(list(folds.values()).count(1), 4)
        for fold in (0, 1):
            held = [r for r in source if folds[r["pair_key"]] == fold]
            self.assertEqual([r["target"] for r in held], [1, 0] * 4)
        with self.assertRaises(ValueError):
            paired_folds(source[:-1])
        with self.assertRaises(ValueError):
            paired_folds(source + source[:1])

    def test_fold_training_exports_only_held_out_and_never_opens_test(self):
        def factory(path, variant, **kwargs):
            return model(variant == "localizer"), Tokens()

        with tempfile.TemporaryDirectory() as tmp, patch.object(core, "losver_model", side_effect=factory):
            root = Path(tmp)
            for split, count in (("train", 4), ("valid", 2)):
                source = rows(split, count)
                write_jsonl(root / "data/primevul_paired" / f"{split}.jsonl", source)
                write_jsonl(root / "data/primevul_paired" / f"{split}.lines.jsonl",
                            [dict(r, lines_ground_truth=[1]) for r in source])
            settings = dict(training.LOSVER_CONFIG, batch_size=2, eval_batch_size=2)
            result = training.train_losver(root/"data", root/"fold", "fixture", "losver_localizer",
                config=settings, device="cpu", train_limit=8, valid_limit=4,
                oof_fold=0, epochs_override=2, save_every=100)
            run = root/"fold/primevul_paired/losver_localizer"
            predicted = read_jsonl(run/"train.predicted.jsonl")
            self.assertEqual([r["sample_key"] for r in predicted], result["oof"]["prediction_members"])
            self.assertFalse(set(result["training_members"]) & set(result["oof"]["prediction_members"]))
            self.assertFalse(any("lines_ground_truth" in r for r in predicted))
            self.assertEqual(result["optimizer_steps"], 4)
            for r in predicted:
                self.assertEqual(r["line_numbers"], paired_encoding(r["func"], Tokens())["line_numbers"])
            self.assertFalse((run/"test.predicted.jsonl").exists())
            # The detector references this same full-Valid/inference checkpoint;
            # a Train cache may contain no model checkpoint at all.
            source = rows("train", 4)
            for split in ("train", "valid"):
                rs = source if split == "train" else rows("valid", 2)
                write_jsonl(run/f"{split}.predicted.jsonl", [dict(r, line_numbers=[0,1,2],
                            lines_predicted=[1]) for r in rs])
            write_jsonl(root/"cache/train.predicted.jsonl", [dict(r, line_numbers=[0,1,2],
                        lines_predicted=[0]) for r in source])
            results = []
            for name, override in (("in_sample", None), ("oof", root/"cache")):
                result = training.train_losver(root/"data", root/name, "fixture", "losver",
                    config=settings, device="cpu", epochs_override=1, train_limit=8, valid_limit=4,
                    localizer_dir=run, train_localizer_dir=override)
                cp = torch.load(root/name/"primevul_paired/losver/best-p_c.pt", weights_only=False)
                self.assertEqual(cp["localizer_path"], str(run/"best-f1.pt"))
                # Inference is single-function and does not open annotations or caches.
                with patch("vulnmechanism.data.read_jsonl", side_effect=AssertionError("no Diff in inference")):
                    p = core.predict_paired(cp, [dict(raw_source=source[0]["func"])], device="cpu")
                self.assertEqual(len(p), 1)
                results.append(result)
            for field in ("training_members", "validation_members", "training_config", "optimizer_steps"):
                self.assertEqual(results[0][field], results[1][field])

    def test_merge_rejects_leakage_and_mapping_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = rows("train", 512)
            write_jsonl(root/"data/primevul_paired/train.jsonl", source)
            folds = paired_folds(source)
            full = dict(input_mode="512", supervision="original_diff", train_samples=1024,
                valid_samples=256, training_members=[r["sample_key"] for r in source],
                validation_members=[r["sample_key"] for r in rows("valid", 128)],
                training_config=training.LOSVER_CONFIG, optimizer_steps=512,
                model_path="fixture", parameters=1, selection="Valid F1")
            reference = root/"reference/primevul_paired/losver_localizer"
            write_json(reference/"training.metrics.json", full)
            scores = [dict(sample_key=r["sample_key"], line_numbers=[0,1,2], line_scores=[.1,.8,.1]) for r in source]
            write_jsonl(reference/"train.line_scores.jsonl", scores)
            for fold in (0, 1):
                held = [r["sample_key"] for r in source if folds[r["pair_key"]] == fold]
                trained = [r["sample_key"] for r in source if folds[r["pair_key"]] != fold]
                config = dict(full, training_members=trained,
                    oof=dict(held_out_fold=fold, prediction_members=held), epochs=8, selected_epochs=dict(f1=7))
                run = root/f"output/fold_{fold}/primevul_paired/losver_localizer"
                write_json(run/"training.metrics.json", config)
                write_jsonl(run/"train.line_scores.jsonl", [s for s in scores if s["sample_key"] in set(held)])
            result = prepare_oof_predictions(root/"data", root/"output", root/"reference")
            self.assertEqual(result["cross_fold_pair_overlap"], 0)
            merged = read_jsonl(root/"output/oof_predictions/train.line_scores.jsonl")
            self.assertEqual(merged, scores)
            run = root/"output/fold_0/primevul_paired/losver_localizer"
            cached = read_jsonl(run/"train.line_scores.jsonl")
            cached[0]["line_numbers"][0] = 42
            write_jsonl(run/"train.line_scores.jsonl", cached)
            with self.assertRaisesRegex(ValueError, "physical lines"):
                prepare_oof_predictions(root/"data", root/"output", root/"reference")
            config = json.loads((run/"training.metrics.json").read_text())
            config["training_members"].append(config["oof"]["prediction_members"][0])
            write_json(run/"training.metrics.json", config)
            with self.assertRaisesRegex(ValueError, "provenance"):
                prepare_oof_predictions(root/"data", root/"output", root/"reference")
