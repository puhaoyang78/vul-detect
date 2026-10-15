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
from vulnmechanism.evaluation import evaluate_megavul
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

    def test_megavul_balanced_reuses_natural_predictions_and_never_reads_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                dict(
                    sample_key=str(i),
                    raw_source="int f(){return 0;}",
                    label=i % 2,
                    dataset="megavul",
                    split="valid",
                )
                for i in range(3)
            ]
            write_jsonl(root / "data/valid.natural.jsonl", rows)
            write_jsonl(root / "data/valid.jsonl", rows[:2])
            (root / "data/test.jsonl").write_text("TEST MUST NEVER BE PARSED")
            run = root / "runs/bce"
            run.mkdir(parents=True)
            (run / "training.metrics.json").write_text("{}")
            torch.save(
                dict(
                    model_path="fixture",
                    source_max_length=2048,
                    decision_threshold=0.73,
                    selected_epoch=2,
                    training_config=dict(batch_size=2),
                ),
                run / "best.pt",
            )
            with patch.object(
                core.AutoTokenizer, "from_pretrained", return_value=CodeTokenizer()
            ), patch.object(core, "load_model", return_value=object()), patch.object(
                core, "predict_logits", return_value=torch.logit(torch.tensor([0.1, 0.7, 0.9]))
            ) as predictor:
                evaluate_megavul(root / "data", root / "runs", ("bce",), device="cpu")
            self.assertEqual(predictor.call_count, 1)
            self.assertEqual(predictor.call_args.kwargs["batch_size"], 2)
            result = json.loads((run / "megavul.valid.metrics.json").read_text())
            self.assertEqual(result["selected_epoch"], 2)
            self.assertEqual(result["threshold"], 0.73)
            self.assertEqual(result["balanced"]["confusion_matrix"], [[1, 0], [1, 0]])
            self.assertEqual(result["natural"]["samples"], 3)
            saved = read_jsonl(run / "megavul.valid.predictions.jsonl")
            self.assertEqual([r["prediction"] for r in saved], [0, 0, 1])
            self.assertTrue(all(r["threshold"] == 0.73 for r in saved))

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
