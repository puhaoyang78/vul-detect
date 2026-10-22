"""Official-pair membership, stable gradients and shared single-function trainer."""

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_codebert_source import CodeTokenizer, tiny_base
from vulnmechanism import model as core
from vulnmechanism.data import prepare_paired, paired_subset, read_jsonl, write_jsonl
from vulnmechanism.evaluation import evaluate, paired_report
from vulnmechanism.metrics import metrics, paired_metrics
from vulnmechanism.training import PAIRED_METHODS, TRAINING, pair_regularizer, train


class PairedTests(unittest.TestCase):
    def test_joint_value_gradient_direction_and_extreme_logits(self):
        z = torch.tensor([[-.1, .2], [10., -10.]], dtype=torch.float64, requires_grad=True)
        expected = torch.logsumexp(torch.stack((torch.zeros(2), .2-z[:, 0],
                                               .2+z[:, 1], .6-z[:, 0]+z[:, 1]), dim=1), dim=1)
        actual = pair_regularizer(z, "pair_joint")
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.autograd.gradcheck(lambda x: pair_regularizer(x, "pair_joint"), (z,)))
        for method in PAIRED_METHODS[1:]:
            values = torch.tensor([[-10000., 10000.], [10000., -10000.], [0., 0.]], requires_grad=True)
            loss = pair_regularizer(values, method).mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss) and torch.isfinite(values.grad).all())
            self.assertTrue((values.grad[:, 0] <= 0).all())
            self.assertTrue((values.grad[:, 1] >= 0).all())
            self.assertLess(float(values.grad[-1, 0]), 0)
            self.assertGreater(float(values.grad[-1, 1]), 0)
        for margin, gap in ((.2, .4), (-.1, .6), (.2, float("nan"))):
            with self.assertRaises(ValueError):
                pair_regularizer(z, "pair_joint", margin=margin, gap=gap)
        logits = torch.zeros(2, 2, requires_grad=True)
        (pair_regularizer(logits, "pair_joint") * torch.tensor([1., 0.])).sum().backward()
        torch.testing.assert_close(logits.grad[1], torch.zeros(2), rtol=0, atol=0)

    def test_pair_metrics_distinguish_ranking_and_scaling_from_classification(self):
        rows = [dict(pair_key="a", label=1, pair_distinguishable=True),
                dict(pair_key="a", label=0, pair_distinguishable=True),
                dict(pair_key="b", label=1, pair_distinguishable=False),
                dict(pair_key="b", label=0, pair_distinguishable=False)]
        z = [1.1, 1., -.1, -.1]
        original = paired_metrics(rows, z)
        scaled = paired_metrics(rows, [10*v for v in z])
        self.assertEqual(original["strict_pairwise_accuracy"], 0)
        self.assertEqual(original["pairwise_ranking_accuracy"], .5)
        self.assertEqual(original["indistinguishable_pairs"], 1)
        self.assertEqual(original["ranking_ties"], 1)
        self.assertEqual(original["strict_pairwise_accuracy"], scaled["strict_pairwise_accuracy"])
        self.assertEqual(original["pairwise_ranking_accuracy"], scaled["pairwise_ranking_accuracy"])
        self.assertGreater(scaled["logit_gap"], original["logit_gap"])
        noisy_tie = paired_metrics(rows, [1.1, 1., -.1, -.10000001])
        self.assertEqual(noisy_tie["pairwise_ranking_accuracy"], .5)
        self.assertEqual(noisy_tie["ranking_ties"], 1)

    def test_official_rows_duplicates_input_collisions_and_heldout_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw/PrimeVul_v0.1"
            shared = "int v(){return 5;}"
            long = "/*" + "a" * 2100 + "*/"
            def native(idx, source, label, commit="c"):
                return dict(idx=idx, func=source, target=label, project="p", commit_id=commit, cve="cve")
            train_rows = [native(1, shared, 1), native(2, "int p(){return 2;}", 0),
                          native(3, long+"int f(){return 1;}", 1), native(4, long+"int f(){return 0;}", 0),
                          native(3, "int a(){return 3;}", 1), native(5, "int b(){return 4;}", 0, "other")]
            valid = [native(7, shared, 1), native(8, "int q(){return 6;}", 0)]
            test = [native(9, "int s(){return 7;}", 1), native(10, "int t(){return 8;}", 0)]
            for split, rows in (("train", train_rows), ("valid", valid), ("test", test)):
                write_jsonl(raw / f"primevul_{split}_paired.jsonl", rows)
            with patch.object(core.AutoTokenizer, "from_pretrained", return_value=CodeTokenizer()):
                prepare_paired(root / "raw", root / "data", "fixture")
            dest = root / "data/primevul_paired"
            prepared = read_jsonl(dest / "train.jsonl")
            self.assertEqual(len(prepared), 4)
            self.assertEqual(len({r["sample_key"] for r in prepared}), 4)
            self.assertEqual(sum(not r["pair_distinguishable"] for r in prepared), 2)
            for split, original in (("valid", valid), ("test", test)):
                rows = read_jsonl(dest / f"{split}.jsonl")
                self.assertEqual([r["raw_source"] for r in rows], [r["func"] for r in original])
                self.assertEqual([r["label"] for r in rows], [r["target"] for r in original])
            statistics = json.loads((dest / "statistics.json").read_text())
            self.assertEqual(len(statistics["removed_training_pairs"]), 1)
            self.assertEqual(statistics["within_split_duplicates"]["train"]["repeated_upstream_idx_rows"], 1)
            self.assertEqual(len(statistics["metadata_anomalies"]), 1)
            subset = paired_subset(prepared, 1)
            self.assertEqual(len(subset), 2)
            self.assertEqual(len({r["pair_key"] for r in subset}), 1)

    def test_four_ablations_share_training_budget_and_never_read_test(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiny_base(root / "base")
            rows = [dict(sample_key=str(i), pair_key=f"p{i//2}", label=1-i%2,
                         pair_distinguishable=i//2 == 0, dataset="primevul_paired", split="train",
                         raw_source=f"int f{i}(){{return {i};}}") for i in range(6)]
            data = root / "data/primevul_paired"
            write_jsonl(data / "train.jsonl", rows)
            valid = [dict(r, sample_key="v"+r["sample_key"], split="valid") for r in rows]
            write_jsonl(data / "valid.jsonl", valid)
            (data / "test.jsonl").write_text("FORBIDDEN_HELDOUT")
            initial = []
            constructor = core.CodeBERT
            def capture(*args, **kwargs):
                model = constructor(*args, **kwargs)
                initial.append(model.task_modules["classifier"].weight.detach().clone())
                return model
            for method in PAIRED_METHODS:
                with self.subTest(method=method), patch.object(core.AutoTokenizer, "from_pretrained", return_value=CodeTokenizer()), \
                        patch("vulnmechanism.training.CodeBERT", side_effect=capture), redirect_stdout(io.StringIO()), \
                        patch("vulnmechanism.training.select_threshold", side_effect=AssertionError("fixed 0.5; never search")):
                    config = dict(TRAINING, batch_size=2, gradient_accumulation=2)
                    result = train(root / "data", root / "runs", str(root / "base"), "primevul_paired", method,
                                   device="cpu", config=config, train_pair_limit=3)
                    self.assertEqual(result["optimizer_steps"], 6)
                    self.assertEqual(result["paired"]["eligible_train_pairs"], 1)
                    evaluate(root / "data", root / "runs", "primevul_paired", (method,),
                             device="cpu", robust=False, splits=("valid",))
                run = root / "runs/primevul_paired" / method
                cp = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
                self.assertEqual(cp["decision_threshold"], .5)
                self.assertFalse((run / "test.metrics.json").exists())
                evaluated = json.loads((run / "valid.metrics.json").read_text())
                self.assertEqual(evaluated["metrics"]["pairs"], 3)
                for k in ("strict_pairwise_accuracy", "mcc", "auc", "f1"):
                    self.assertEqual(evaluated["metrics"][k], cp["validation"][k])
                with patch.object(core.AutoTokenizer, "from_pretrained", return_value=CodeTokenizer()):
                    single = core.predict_checkpoint(run / "best.pt", [valid[0]], device="cpu")
                self.assertEqual(single.numel(), 1)
                self.assertTrue(torch.isfinite(single).all())
            for weight in initial[1:]:
                torch.testing.assert_close(weight, initial[0], rtol=0, atol=0)
            statistics = dict(native_pairs=dict(train=3, valid=3, test=1),
                              final={s: dict(pairs=n, indistinguishable_pairs=0) for s, n in
                                     (("train", 3), ("valid", 3), ("test", 1))},
                              removed_training_pairs=[], metadata_anomalies=[])
            (data / "statistics.json").write_text(json.dumps(statistics))
            report = paired_report(root / "runs", root / "data")
            self.assertEqual(report["rows"], 4)
            self.assertIsInstance(report["valid_gate"]["passed"], bool)
            results = json.loads((root / "runs/results.json").read_text())
            self.assertEqual(set(results["valid_pair_changes"]), set(PAIRED_METHODS))
            saved = (root / "runs/results.json").read_bytes()
            paired_report(root / "runs", root / "data")
            self.assertEqual((root / "runs/results.json").read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
