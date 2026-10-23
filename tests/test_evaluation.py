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
from vulnmechanism.cli import build_parser, main
from vulnmechanism.data import read_jsonl, write_jsonl
from vulnmechanism.evaluation import evaluate, report
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

    def test_clean_evaluation_reads_only_own_valid_and_test(self):
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
                ), patch.object(
                    core, "load_model", return_value=object()
                ), patch.object(
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
                self.assertEqual(
                    result["metrics"]["confusion_matrix"], [[2, 0], [1, 0]]
                )
                saved = read_jsonl(run / "test.predictions.jsonl")
                self.assertEqual(
                    [r["sample_key"] for r in saved],
                    [f"{dataset}:{i}" for i in range(3)],
                )
                self.assertTrue(all(r["threshold"] == 0.73 for r in saved))

    def test_report_contains_twelve_clean_and_robust_models(self):
        from vulnmechanism.data import DATASETS
        from vulnmechanism.training import METHODS
        from vulnmechanism.artifacts import write_json
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in DATASETS:
                write_json(
                    root / "data" / dataset / "statistics.json", dict(dataset=dataset)
                )
                target = next(d for d in DATASETS if d != dataset)
                for method in METHODS:
                    run = root / "runs" / dataset / method
                    threshold = 0.73 if dataset == "primevul" else 0.61
                    own = metrics([0, 1], [0.1, 0.9], threshold)
                    transferred = metrics([0, 1], [0.9, 0.1], threshold)
                    write_json(
                        run / "training.metrics.json",
                        dict(dataset=dataset, selected_epoch=2),
                    )
                    for split in ("valid", "test"):
                        write_json(run / f"{split}.metrics.json", dict(metrics=own))
                    write_json(
                        run / f"{target}.test.metrics.json", dict(metrics=transferred)
                    )
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
            self.assertEqual((summary["models"], summary["rows"]), (12, 132))
            result = json.loads((root / "runs/comparison.json").read_text())
            self.assertFalse(any(r["view"] == "transfer" for r in result["results"]))
            with (root / "runs/comparison.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 132)

    def test_clean_cache_predicts_only_restored_members_and_recomputes_metrics(self):
        from argparse import Namespace
        from vulnmechanism.artifacts import write_json
        from vulnmechanism.evaluation import _evaluate_clean

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                dict(
                    sample_key=str(i),
                    label=i % 2,
                    dataset="primevul",
                    split="test",
                    raw_source=str(i),
                )
                for i in range(4)
            ]
            cached = [
                dict(
                    sample_key=str(i),
                    label=i % 2,
                    logit=float(i * 2 - 1),
                    score=float(torch.sigmoid(torch.tensor(i * 2 - 1.0))),
                    threshold=0.4,
                    prediction=i,
                )
                for i in (0, 1)
            ]
            write_jsonl(root / "test.predictions.jsonl", cached)
            write_json(
                root / "test.metrics.json",
                dict(selected_epoch=3, metrics={"samples": 2}),
            )
            cp = dict(selected_epoch=3, decision_threshold=0.4, source_max_length=2048)
            args = Namespace(model="fixture", device="cpu", batch_size=4)
            with patch.object(
                core.AutoTokenizer, "from_pretrained", return_value=CodeTokenizer()
            ), patch.object(core, "load_model", return_value=object()), patch.object(
                core, "predict_logits", return_value=torch.tensor([2.0, -2.0])
            ) as predictor:
                result = _evaluate_clean(rows, root, "test", cp, args)
            self.assertEqual(
                [r["sample_key"] for r in predictor.call_args.args[1]], ["2", "3"]
            )
            merged = read_jsonl(root / "test.predictions.jsonl")
            self.assertEqual(merged[:2], cached)
            self.assertEqual([r["sample_key"] for r in merged], ["0", "1", "2", "3"])
            self.assertEqual(
                result["metrics"],
                metrics([0, 1, 0, 1], [r["score"] for r in merged], 0.4),
            )
            with patch.object(
                core,
                "predict_logits",
                side_effect=AssertionError("no repeat inference"),
            ), patch(
                "vulnmechanism.evaluation.write_json",
                side_effect=AssertionError("no rewriting complete results"),
            ), patch(
                "vulnmechanism.evaluation.write_jsonl",
                side_effect=AssertionError("no rewriting complete predictions"),
            ):
                self.assertEqual(_evaluate_clean(rows, root, "test", cp, args), result)

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
            ), patch(
                "torch.load", return_value=dict(decision_threshold=0.7)
            ), patch.object(
                core, "predict_checkpoint", return_value=torch.tensor([2.0])
            ) as predictor, redirect_stdout(
                output
            ):
                main()
            result = json.loads(output.getvalue())
            self.assertEqual(result["prediction"], 1)
            self.assertEqual(result["threshold"], 0.7)
            self.assertEqual(
                predictor.call_args.args[1], [{"raw_source": path.read_text()}]
            )


if __name__ == "__main__":
    unittest.main()
