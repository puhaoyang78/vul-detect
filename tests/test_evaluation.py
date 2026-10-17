import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_codebert_source import CodeTokenizer
from vulnmechanism import model as core
from vulnmechanism.cli import main
from vulnmechanism.data import read_jsonl, write_jsonl
from vulnmechanism.evaluation import evaluate, evaluate_transfer, report
from vulnmechanism.metrics import metrics, select_threshold


class EvaluationTests(unittest.TestCase):
    def test_metrics_threshold_and_single_class(self):
        result = metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["mcc"], 0.0)
        self.assertAlmostEqual(result["auc"], 0.75)
        self.assertAlmostEqual(result["auprc"], 5 / 6)
        self.assertIsNone(metrics([0, 0], [0.1, 0.9])["auc"])
        threshold, selected = select_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(threshold, 0.5)
        self.assertEqual(selected["mcc"], 1.0)

    def test_four_dataset_evaluation_reads_only_own_valid_and_test(self):
        from vulnmechanism.data import DATASETS
        for dataset in DATASETS:
            target = "unused_dataset"
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for d in (dataset, target):
                    rows = [
                        dict(
                            sample_key=f"{d}:{i}",
                            raw_source=f"int f{i}(){{return {i};}}",
                            label=i % 2,
                            dataset=d,
                            split="test",
                            train_overlap=[dataset] if i == 3 else [],
                        )
                        for i in range(4)
                    ]
                    write_jsonl(root / "data" / d / "test.jsonl", rows)
                valid = [
                    dict(
                        sample_key=f"v{i}",
                        raw_source=f"valid {i}",
                        label=i,
                        dataset=dataset,
                        split="valid",
                    )
                    for i in (0, 1)
                ]
                write_jsonl(root / "data" / dataset / "valid.jsonl", valid)
                (root / "data" / target / "valid.jsonl").write_text(
                    "TARGET VALID MUST NEVER BE READ"
                )
                run = root / "runs" / dataset / "bce"
                run.mkdir(parents=True)
                (run / "training.metrics.json").write_text("{}")
                torch.save(
                    dict(
                        dataset=dataset,
                        model_path="fixture",
                        source_max_length=2048,
                        decision_threshold=0.73,
                        selected_epoch=2,
                        training_config=dict(batch_size=2),
                    ),
                    run / "best.pt",
                )

                def predict(model, rows, builder, **kwargs):
                    self.assertEqual(kwargs["batch_size"], 2)
                    return torch.logit(
                        torch.tensor([0.1 if r["label"] == 0 else 0.7 for r in rows])
                    )

                with patch.object(
                    core.AutoTokenizer, "from_pretrained", return_value=CodeTokenizer()
                ), patch.object(core, "load_model", return_value=object()), patch.object(
                    core, "predict_logits", side_effect=predict
                ) as predictor:
                    evaluate(
                        root / "data",
                        root / "runs",
                        dataset,
                        ("bce",),
                        device="cpu",
                        robust=False,
                    )
                self.assertEqual(predictor.call_count, 2)
                self.assertFalse((run / f"{target}.test.metrics.json").exists())
                result = json.loads((run / "test.metrics.json").read_text())
                self.assertEqual(result["dataset"], dataset)
                self.assertEqual(result["split"], "test")
                self.assertEqual(result["selected_epoch"], 2)
                self.assertEqual(result["metrics"]["threshold"], 0.73)
                self.assertEqual(result["metrics"]["samples"], 3)
                self.assertEqual(result["metrics"]["confusion_matrix"], [[2, 0], [1, 0]])
                saved = read_jsonl(run / "test.predictions.jsonl")
                self.assertEqual(
                    [r["sample_key"] for r in saved], [f"{dataset}:{i}" for i in range(3)]
                )
                self.assertTrue(all(r["threshold"] == 0.73 for r in saved))

    def test_report_contains_twelve_models_and_both_test_transfer_deltas(self):
        from vulnmechanism.data import DATASETS
        from vulnmechanism.training import METHODS
        from vulnmechanism.artifacts import write_json
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in DATASETS:
                write_json(root / "data" / dataset / "statistics.json", dict(dataset=dataset))
                target = next(d for d in DATASETS if d != dataset)
                for method in METHODS:
                    run = root / "runs" / dataset / method
                    threshold = 0.73 if dataset == "primevul" else 0.61
                    own = metrics([0, 1], [0.1, 0.9], threshold)
                    transferred = metrics([0, 1], [0.9, 0.1], threshold)
                    write_json(
                        run / "training.metrics.json", dict(dataset=dataset, selected_epoch=2)
                    )
                    for split in ("valid", "test"):
                        write_json(run / f"{split}.metrics.json", dict(metrics=own))
                    write_json(run / f"{target}.test.metrics.json", dict(metrics=transferred))
                    result = dict(
                        worst_complete=transferred,
                        asr=dict(count=2, denominator=2, rate=1.0),
                        eligible=2,
                    )
                    bank = {
                        kind: result
                        for kind in ("all", "strict_semantic", "input_only_stress")
                    }
                    write_json(
                        run / "robustness.metrics.json",
                        dict(fixed_overall=bank, heldout_overall=bank, adaptive=bank),
                    )
            summary = report(root / "runs", root / "data")
            self.assertEqual((summary["models"], summary["rows"]), (12, 144))
            result = json.loads((root / "runs/comparison.json").read_text())
            transferred = [r for r in result["results"] if r["view"] == "transfer"]
            self.assertEqual(len(transferred), 12)
            self.assertTrue(
                all(
                    r["delta_accuracy"] == -1.0 and r["delta_mcc"] == -2.0 for r in transferred
                )
            )
            self.assertTrue(
                all(
                    r["dataset"] != r["training_dataset"] and r["split"] == "test"
                    for r in transferred
                )
            )
            with (root / "runs/comparison.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 144)

    def test_transfer_reuses_own_predictions_and_shared_overlap_free_targets(self):
        import csv
        from vulnmechanism.artifacts import write_json

        datasets = ("primevul", "secvuleval")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in datasets:
                other = next(d for d in datasets if d != dataset)
                for split in ("train", "valid"):
                    write_jsonl(root / "data" / dataset / f"{split}.jsonl", [dict(
                        raw_source=f"{dataset}-{split}", label=1,
                    )])
                rows = [dict(
                    sample_key=f"{dataset}:{i}", label=i % 2, dataset=dataset, split="test",
                    raw_source=source, train_overlap=[], project="fixture",
                    source_file="fixture.jsonl", source_row=i+1,
                ) for i, source in enumerate((
                    f"{other}-train", f"{other}-valid", f"{dataset}-clean0", f"{dataset}-clean1"
                ))]
                write_jsonl(root / "data" / dataset / "test.jsonl", rows)
                threshold = .73 if dataset == "primevul" else .61
                run = root / "runs" / dataset / "bce"
                epoch = 2 if dataset == "primevul" else 1
                write_json(run / "training.metrics.json", dict(selected_epoch=epoch))
                write_jsonl(run / "test.predictions.jsonl", [dict(
                    sample_key=r["sample_key"], label=r["label"],
                    score=.9 if r["label"] else .1, threshold=threshold,
                ) for r in rows])
                write_json(run / "test.metrics.json", dict(
                    selected_epoch=epoch, metrics=metrics([0, 1, 0, 1], [.1, .9, .1, .9], threshold),
                ))
                torch.save(dict(
                    dataset=dataset, method="bce", model_path="fixture", source_max_length=2048,
                    selected_epoch=epoch, decision_threshold=threshold, training_config=dict(batch_size=2),
                ), run / "best.pt")

            def predict(model, rows, builder, **kwargs):
                self.assertNotEqual(model, rows[0]["dataset"])
                self.assertEqual(kwargs["batch_size"], 2)
                return torch.logit(torch.tensor([.7 if r["label"] else .8 for r in rows]))

            with patch.object(core.AutoTokenizer, "from_pretrained", return_value=CodeTokenizer()), patch.object(
                core, "load_model", side_effect=lambda cp, *a, **kw: cp["dataset"]
            ), patch.object(core, "predict_logits", side_effect=predict) as predictor:
                summary = evaluate_transfer(root / "data", root / "runs", datasets, device="cpu")
            self.assertEqual(predictor.call_count, 2)
            self.assertEqual(summary["rows"], 8)
            output = root / "runs/transfer_primevul_secvuleval"
            result = json.loads((output / "results.json").read_text())
            for dataset in datasets:
                self.assertEqual(result["cohorts"][dataset]["independent"], dict(samples=2, positive=1, negative=1))
                with (output / f"{dataset}.members.csv").open() as handle:
                    members = list(csv.DictReader(handle))
                self.assertEqual([m["sample_key"] for m in members if m["independent"] == "True"],
                    [f"{dataset}:2", f"{dataset}:3"])
            cross = [r for r in result["results"] if r["training_dataset"] != r["test_dataset"]]
            self.assertTrue(all(r["auc"] == 0 and r["delta_source_auc"] == -1 and r["delta_target_auc"] == -1 for r in cross))
            self.assertTrue(all(r["threshold"] == (.73 if r["training_dataset"] == "primevul" else .61) for r in cross))
            with patch.object(core, "predict_logits", side_effect=AssertionError("cached predictions must be reused")):
                evaluate_transfer(root / "data", root / "runs", datasets, device="cpu")

    def test_single_function_inference_requires_only_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.c"
            path.write_text("int f(){return 0;}")
            output = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "vulnmechanism",
                    "infer",
                    "--checkpoint",
                    "fixture.pt",
                    "--source-file",
                    str(path),
                    "--device",
                    "cpu",
                ],
            ), patch("torch.load", return_value=dict(decision_threshold=0.7)), patch.object(
                core, "predict_checkpoint", return_value=torch.tensor([2.0])
            ) as predictor, redirect_stdout(
                output
            ):
                main()
            result = json.loads(output.getvalue())
            self.assertEqual(result["prediction"], 1)
            self.assertEqual(result["threshold"], 0.7)
            self.assertEqual(predictor.call_args.args[1], [{"raw_source": path.read_text()}])


if __name__ == "__main__":
    unittest.main()
