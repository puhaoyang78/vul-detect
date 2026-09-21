import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from vulnmechanism.diagnostic import (
    Probe, check_membership, choose_threshold, fold_jobs, group_folds, load_cohort,
    make_folds, metrics, relation_features, run_oof, run_probes, shuffled_content,
    validate_scores, write_json, write_rows,
)


def synthetic_records(n=240):
    return [dict(sample_key=f"s:{i:04}", label=i % 2, split="train", dataset="primevul",
                 raw_source=f"int f{i}() {{ return {i}; }}", mechanism_items=[dict(
                     category="MECHANISM_RELATION", kind="BOUND_RELATION",
                     detail="bound=not_observed" if i % 2 else "bound=present", state="")])
            for i in range(n)]


def features_and_folds():
    records = synthetic_records()
    groups = {r["sample_key"]: f"commit:{i//4}" for i, r in enumerate(records)}
    features = [relation_features(r, groups[r["sample_key"]]) for r in records]
    folds = make_folds(records, groups, outer_folds=3, inner_folds=2, calibration_folds=3, seed=42)
    return records, groups, features, folds


class DiagnosticTests(unittest.TestCase):
    def test_nested_partitions_are_disjoint_and_cover_training(self):
        rows, groups, _, folds = features_and_folds()
        all_keys = {r["sample_key"] for r in rows}
        outer_seen = []
        for f in folds:
            partitions = [set(f[k]) for k in ("develop", "calibration", "evaluation")]
            self.assertEqual(set.union(*partitions), all_keys)
            for i in range(3):
                for j in range(i):
                    self.assertFalse({groups[k] for k in partitions[i]} & {groups[k] for k in partitions[j]})
            inner_seen = []
            for inner in f["inner"]:
                self.assertEqual(set(inner["fit"]) | set(inner["predict"]), partitions[0])
                self.assertFalse(set(inner["fit"]) & set(inner["predict"]))
                self.assertFalse({groups[k] for k in inner["fit"]} & {groups[k] for k in inner["predict"]})
                inner_seen.extend(inner["predict"])
            self.assertEqual(len(inner_seen), len(set(inner_seen)))
            self.assertEqual(set(inner_seen), partitions[0])
            outer_seen.extend(f["evaluation"])
        self.assertEqual(len(outer_seen), len(all_keys))
        self.assertEqual(set(outer_seen), all_keys)

    def test_shuffle_preserves_controls_avoids_related_donors_and_labels(self):
        _, _, features, _ = features_and_folds()
        features.append(dict(sample_key="rare", label=1, group="rare", counts={"RARE": 1},
                             length=2, content="xx", observed_relation=True))
        original = json.dumps(features, sort_keys=True)
        shuffled, report = shuffled_content(features, seed=7, length_bin=64)
        by_key = {r["sample_key"]: r for r in features}
        self.assertNotIn("rare", report["eligible"])
        self.assertEqual(report["eligible_count"], 240)
        self.assertGreater(report["changed_count"], 0)
        self.assertEqual(len(set(report["donors"].values())), 240)
        for receiver, donor in report["donors"].items():
            a, b = by_key[receiver], by_key[donor]
            self.assertNotEqual(a["group"], b["group"])
            self.assertEqual(a["counts"], b["counts"])
            self.assertEqual(a["length"]//64, b["length"]//64)
        for a, b in zip(features, shuffled):
            self.assertEqual(a["sample_key"], b["sample_key"])
            self.assertEqual(a["label"], b["label"])
            self.assertEqual(a["counts"], b["counts"])
        changed_labels = [dict(r, label=1-r["label"]) for r in features]
        _, other = shuffled_content(changed_labels, seed=7, length_bin=64)
        self.assertEqual(report, other)
        self.assertEqual(original, json.dumps(features, sort_keys=True))

    def test_majority_provenance_bucket_and_empty_evidence_are_not_shuffled(self):
        rows = [dict(sample_key=str(i), group="a" if i < 3 else "b", counts={"R": 1},
                     content=str(i), length=1, observed_relation=True) for i in range(4)]
        shuffled, report = shuffled_content(rows, seed=1, length_bin=64)
        self.assertEqual(report["eligible_count"], 0)
        self.assertEqual(shuffled, rows)
        rows.append(dict(sample_key="empty", group="c", counts={}, content="", length=0, observed_relation=False))
        self.assertEqual(shuffled_content(rows, seed=2, length_bin=64)[1]["observed_count"], 4)

    def test_features_exclude_candidates_and_provenance(self):
        row = synthetic_records(1)[0]
        row["mechanism_items"].append(dict(category="MECHANISM_CANDIDATE", kind="SECRET", detail="LEAK"))
        feature = relation_features(row, "private_commit")
        self.assertNotIn("LEAK", feature["content"])
        self.assertNotIn("SECRET", feature["counts"])
        train = [dict(feature, logit=float(i), label=i%2) for i in range(10)]
        probe = Probe("S+U+R", c=1, max_features=100).fit(train)
        self.assertNotIn("private_commit", probe.text.vocabulary_)
        width = probe.features(train, fit=False).shape[1]
        held = [dict(train[0], content="UNSEEN_TOKEN", counts={"NEW": 999})]
        probe.predict(held)
        self.assertNotIn("UNSEEN_TOKEN", probe.text.vocabulary_)
        self.assertEqual(probe.features(held, fit=False).shape[1], width)

    def test_content_probe_handles_no_evidence_without_fabricating_tokens(self):
        rows = [dict(logit=i-2, label=i%2, observed_relation=False, counts={}, content="") for i in range(6)]
        probe = Probe("S+U+R", c=1, max_features=100).fit(rows)
        self.assertIsNone(probe.text)
        self.assertTrue(np.isfinite(probe.predict(rows)).all())

    def test_score_coverage_and_finite_values_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "p.jsonl"
            rows = {"a": {"label": 1}, "b": {"label": 0}}
            write_rows(p, [dict(sample_key="a", label=1, logit=40), dict(sample_key="b", label=0, logit=-40)])
            self.assertEqual(validate_scores(p, ["a", "b"], rows)["a"], 40)
            with self.assertRaisesRegex(ValueError, "coverage"):
                validate_scores(p, ["a"], rows)
            write_rows(p, [dict(sample_key="a", label=0, logit=1)])
            with self.assertRaisesRegex(ValueError, "invalid prediction"):
                validate_scores(p, ["a"], rows)

    def test_oof_jobs_fit_only_owned_records_and_resume_completed_predictions(self):
        import torch
        from vulnmechanism.cli import build_parser
        records, _, _, folds = features_and_folds()
        with tempfile.TemporaryDirectory() as temp:
            args = build_parser().parse_args(["diagnose", "--output", temp, "--device", "cpu"])
            def fit(dataset, path, **kwargs):
                self.assertIsNone(dataset)
                self.assertTrue(kwargs["fixed_epochs"])
                self.assertTrue(all(r["split"] == "train" for r in kwargs["records"]))
                ownership = json.loads((path.parent/"membership.json").read_text())
                self.assertEqual([r["sample_key"] for r in kwargs["records"]], ownership["fit"])
                self.assertFalse(set(ownership["fit"]) & set(ownership["predict"]))
                torch.save(dict(selection="fixed_epochs", variant="baseline", seed=kwargs["seed"],
                                selected_epoch=args.epochs, model_path=args.model), path)
            def predict(path, rows, **kwargs):
                ownership = json.loads((path.parent/"membership.json").read_text())
                self.assertEqual([r["sample_key"] for r in rows], ownership["predict"])
                return torch.linspace(-1, 1, len(rows))
            with patch("vulnmechanism.model.train_model", side_effect=fit) as trainer, \
                 patch("vulnmechanism.model.predict_checkpoint", side_effect=predict) as predictor:
                run_oof(args, records, folds[:1])
                self.assertEqual(trainer.call_count, 3)
                self.assertEqual(predictor.call_count, 3)
                run_oof(args, records, folds[:1])
                self.assertEqual(trainer.call_count, 3)
                self.assertEqual(predictor.call_count, 3)
                path = Path(temp)/"outer_00"/"inner_00"/"predictions.jsonl"
                path.unlink()
                run_oof(args, records, folds[:1])
                self.assertEqual(trainer.call_count, 3)
                self.assertEqual(predictor.call_count, 4)

    def test_threshold_and_empty_subgroup(self):
        rows = [{"label": y} for y in (0, 0, 1, 1)]
        threshold = choose_threshold(rows, [0.6, 0.7, 0.8, 0.9])
        self.assertGreater(threshold, 0.7)
        self.assertLessEqual(threshold, 0.8)
        self.assertEqual(metrics([0, 0, 1, 1], [0.6, 0.7, 0.8, 0.9], threshold)["mcc"], 1)
        self.assertIsNone(metrics([], [], threshold)["auc"])
        self.assertIsNone(metrics([1], [0.8], threshold)["auc"])

    def test_manifest_join_transitive_commit_groups_and_source_verification(self):
        rows = []
        for split in ("train", "valid", "test"):
            for i in range(4):
                rows.append(dict(sample_key=f"{split}:{i}", dataset="primevul", split=split,
                                 label=i%2, raw_source=f"int {split}{i}() {{ return {i}; }}"))
        metadata = [dict(sample_key=r["sample_key"], dataset="primevul", split=r["split"],
                         label=r["label"], function=r["raw_source"], repo="repo", commit=f"c{i//2}",
                         counterpart_groups=["linked"] if i in (1, 2) else []) for i, r in enumerate(rows)]
        with tempfile.TemporaryDirectory() as temp, patch("vulnmechanism.model._read_jsonl", return_value=rows):
            path = Path(temp) / "manifest.jsonl"
            write_rows(path, metadata)
            selected, groups = load_cohort("unused", str(path))
            self.assertEqual(len(selected), 4)
            self.assertEqual(len(set(groups.values())), 1)
            metadata[0]["function"] = "wrong"
            write_rows(path, metadata)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_cohort("unused", str(path))

    def test_probe_pipeline_on_synthetic_predictions_and_evaluation_label_isolation(self):
        _, _, features, folds = features_and_folds()
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(output=temp, seed=42, shuffle_repeats=2, length_bin=64, probe_c=1, max_features=100)
            by_key = {r["sample_key"]: r for r in features}
            rng = np.random.default_rng(42)
            logits = {k: float(rng.normal()) for k in by_key}
            for f in folds:
                for directory, fit, predict, seed in fold_jobs(args, f):
                    write_json(directory / "membership.json", dict(fit=fit, predict=predict, seed=seed))
                    write_rows(directory / "predictions.jsonl", [dict(sample_key=k, label=by_key[k]["label"], logit=logits[k]) for k in predict])
            report = run_probes(args, folds[:1], features)
            result = report["folds"][0]["results"]
            self.assertGreater(result["S+U+R"]["all"]["auc"], result["S+U"]["all"]["auc"] + 0.2)
            before = read_predictions(Path(temp) / "probe_predictions.jsonl")
            # Only outer evaluation labels change; fitted predictions and thresholds must not.
            evaluation = set(folds[0]["evaluation"])
            changed = [dict(r, label=1-r["label"]) if r["sample_key"] in evaluation else r for r in features]
            path = Path(temp) / "outer_00" / "outer_model" / "predictions.jsonl"
            values = [json.loads(l) for l in path.read_text().splitlines()]
            for r in values:
                if r["sample_key"] in evaluation:
                    r["label"] = 1-r["label"]
            write_rows(path, values)
            run_probes(args, folds[:1], changed)
            after = read_predictions(Path(temp) / "probe_predictions.jsonl")
            self.assertEqual(before, after)
            self.assertFalse((Path(temp) / "index.html").exists())
            path = Path(temp) / "outer_00" / "inner_00" / "membership.json"
            path.unlink()
            with self.assertRaisesRegex(ValueError, "provenance"):
                run_probes(args, folds[:1], changed)


def read_predictions(path):
    return [(r["sample_key"], r["variant"], r["probability"], r["threshold"])
            for r in map(json.loads, path.read_text().splitlines())]


if __name__ == "__main__":
    unittest.main()
