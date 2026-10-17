import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_codebert_source import CodeTokenizer, tiny_base
from vulnmechanism import csd_diagnostics as diagnostics, model as core
from vulnmechanism.data import read_jsonl, write_jsonl
from vulnmechanism.training import TRAINING


class CSDDiagnosticTests(unittest.TestCase):
    def test_reduction_factor_paired_metrics_and_single_class_projects(self):
        head = core.CSD(3, domains=3)
        with torch.no_grad():
            head.common.weight.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
            head.specific_weight.copy_(torch.tensor([[[0., 1., 0.], [1., 0., 0.]]]))
            head.coefficients.weight.copy_(torch.tensor([[0.], [.01], [.02]]))
            head.specific_bias.zero_()
        zero = diagnostics.head_statistics(head)
        self.assertEqual(zero["sum_to_mean_factor"], 8)
        self.assertEqual(zero["orthonormality_sum"], 0.)
        self.assertIsNone(zero["diagonal_fraction"])
        with torch.no_grad():
            head.common.weight[0, 0] = 2.
        stats = diagnostics.head_statistics(head)
        self.assertEqual(stats["orthonormality_sum"], 9.)
        self.assertEqual(stats["orthonormality_mean"], 9. / 8)
        projects = ["negative", "positive", "mixed"]
        labels = [0, 0, 1, 1, 0, 1]
        common = [-1., .1, -.1, 1., -1., 1.]
        specific = [-1., -.1, .1, 1., -1., 1.]
        predictions = [dict(sample_key=str(i), label=y, common_logit=a, specific_logit=b)
                       for i, (y, a, b) in enumerate(zip(labels, common, specific))]
        value = diagnostics.paired_metrics(predictions, .5)
        self.assertEqual((value["flips"], value["wrong_to_correct"], value["correct_to_wrong"]), (2, 2, 0))
        self.assertEqual(value["denominator"], 6)
        self.assertGreater(value["bce_reduction"], 0.)
        self.assertEqual(value["specific"]["auc"], 1.)
        rows = [dict(sample_key=str(i), label=y, project=projects[i // 2]) for i, y in enumerate(labels)]
        records, composition, correlations, groups = diagnostics.project_statistics(
            rows, predictions, head, projects, .5)
        self.assertEqual((composition["single_class_projects"], composition["single_class_samples"]), (2, 4))
        self.assertIsNone(records[0]["common_auc"])
        self.assertIsNone(records[1]["specific_auc"])
        self.assertEqual(records[2]["common_auc"], 1.)
        self.assertEqual(sum(r["samples"] for r in groups), 6)

    def test_diagnosis_reads_only_source_train_and_never_updates_weights(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiny_base(root / "base")
            csd = dict(rank=1, projects=["p0", "p1"])
            model = core.CodeBERT(str(root / "base"), "cpu", csd=csd)
            run = root / "results/primevul/csd"
            run.mkdir(parents=True)
            checkpoint = dict(dataset="primevul", method="csd", csd=csd,
                              model_path=str(root / "base"), training_config=dict(TRAINING, batch_size=2),
                              source_max_length=2048, selected_epoch=3, decision_threshold=.5,
                              environment=dict(torch=torch.__version__), state_dict=model.state_dict())
            path = run / "best.pt"
            torch.save(checkpoint, path)
            before = path.read_bytes()
            rows = [dict(sample_key=str(i), dataset="primevul", split="train", project=f"p{i // 2}",
                         label=i % 2, raw_source=f"int f{i}(){{return {i};}}") for i in range(4)]
            write_jsonl(root / "data/primevul/train.jsonl", rows)
            (root / "data/primevul/test.jsonl").write_text("FORBIDDEN TARGET TEST")
            (root / "data/primevul/valid.jsonl").write_text("FORBIDDEN VALID")
            write_jsonl(run / "best.training.jsonl", [dict(event="epoch", epoch=e,
                csd_losses=dict(common_loss=.7, specific_loss=.69, orthonormality=8.)) for e in (1, 2, 3)])

            def source_only(path):
                self.assertIn(Path(path).name, ("train.jsonl", "best.training.jsonl"))
                return read_jsonl(path)

            with patch.object(core.AutoTokenizer, "from_pretrained", return_value=CodeTokenizer()), patch.object(
                diagnostics, "read_jsonl", side_effect=source_only
            ), patch("torch.optim.AdamW", side_effect=AssertionError("no optimizer allowed")), redirect_stdout(io.StringIO()):
                diagnostics.diagnose(root / "data", root / "results", "primevul", device="cpu")
            self.assertEqual(path.read_bytes(), before)
            saved = json.loads((root / "results/csd_diagnostics/primevul.json").read_text())
            self.assertEqual(saved["protocol"]["optimizer_updates"], 0)
            self.assertFalse(saved["protocol"]["target_data_read"])
            self.assertEqual(saved["train"]["denominator"], 4)
            self.assertEqual(saved["project_composition"]["projects"], 2)
            self.assertEqual(saved["final_gradient_probe"]["head_regularizer_grad_norms"]["coefficients.weight"], 0.)
            self.assertEqual(saved["final_gradient_probe"]["head_regularizer_grad_norms"]["common.bias"], 0.)
            self.assertEqual(len(saved["epochs"]), 3)
