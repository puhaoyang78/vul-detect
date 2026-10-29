"""One encoder, unchanged inference, function-balanced original-Diff Focal."""

import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import RobertaModel

from tests.test_losver import Tokens, config
from vulnmechanism.data import paired_features
from vulnmechanism.model import LOSVERDetector, LOSVERLocalizer, predict_checkpoint
from vulnmechanism.training import paired_collate, paired_forward, shuffle_line_labels
from vulnmechanism.evaluation import validate_paired
from vulnmechanism.cli import build_parser


def detector(weight=None):
    c = config()
    return LOSVERDetector(RobertaModel(c, add_pooling_layer=False), c,
                          SimpleNamespace(class_weight=1., line_auxiliary_weight=weight))


def rows():
    return [dict(func=source, target=target, lines_ground_truth=gt,
                 sample_key=f"valid:{i}", pair_key="valid:0")
            for i, (source, target, gt) in enumerate([
                ("a();\r\n\r\n b();", 1, [2]),
                ("safe();\n check();\n use();\n return;", 0, []),
            ])]


def batch(records=None):
    return paired_collate([paired_features(r, Tokens(), "baseline") for r in (rows() if records is None else records)])


class LineAuxiliaryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_zero_weight_preserves_initialization_rng_and_baseline(self):
        torch.manual_seed(123456)
        original = detector().eval()
        rng = torch.get_rng_state()
        torch.manual_seed(123456)
        zero = detector(0.).eval()
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, zero.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(zero.line_classifier.weight.shape, (1, 16))
        data = batch()
        loss, probabilities = paired_forward(original, data, "baseline", "cpu")
        with patch.object(zero.encoder, "forward", wraps=zero.encoder.forward) as forward:
            total, actual, details = paired_forward(zero, data, "baseline", "cpu", return_auxiliary=True)
            self.assertEqual(forward.call_count, 1)
        torch.testing.assert_close(actual, probabilities, rtol=0, atol=0)
        torch.testing.assert_close(total, loss, rtol=0, atol=0)
        total.backward()
        loss.backward()
        for name, parameter in original.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(zero.named_parameters())[name].grad, rtol=0, atol=0)
        self.assertEqual(float(zero.line_classifier.weight.grad.abs().max()), 0.)

    def test_focal_is_function_mean_and_updates_shared_encoder(self):
        model = detector(.3).eval()
        data = batch()
        loss, _, details = paired_forward(model, data, "baseline", "cpu", return_auxiliary=True)
        losses = [LOSVERLocalizer.focal_loss(torch.logit(p), labels[:len(p)].float())
                  for p, labels in zip(details["line_probabilities"], data["line_labels"])]
        expected = torch.stack([x.mean() for x in losses]).mean()
        torch.testing.assert_close(details["line_focal"], expected)
        self.assertGreater(float((torch.cat(losses).mean() - expected).abs()), 1e-4)
        torch.testing.assert_close(loss, details["bce"] + .3 * expected)
        details["line_focal"].backward()
        for parameter in (model.encoder.embeddings.word_embeddings.weight, model.line_classifier.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertIsNone(model.classifier.weight.grad)
        # The all-zero patched-side Diff is retained and has nonzero loss.
        self.assertEqual(data["line_labels"][1].sum(), 0.)
        self.assertGreater(float(losses[1].mean()), 0.)

    def test_padding_and_diff_do_not_change_function_probabilities(self):
        model = detector(.1).eval()
        data = batch()
        first, probability, details = paired_forward(model, data, "baseline", "cpu", return_auxiliary=True)
        padded = copy.deepcopy(data)
        padded["line_labels"][padded["line_numbers"] < 0] = float("nan")
        second, actual, _ = paired_forward(model, padded, "baseline", "cpu", return_auxiliary=True)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(probability, actual, rtol=0, atol=0)
        changed = copy.deepcopy(data)
        changed["line_labels"] = 1 - changed["line_labels"]
        _, changed_prob, _ = paired_forward(model, changed, "baseline", "cpu", return_auxiliary=True)
        torch.testing.assert_close(probability, changed_prob, rtol=0, atol=0)
        with patch.object(model.line_classifier, "forward", side_effect=AssertionError("inference used line head")):
            source_only = paired_forward(model, data, "baseline", "cpu", with_labels=False)
        torch.testing.assert_close(source_only, probability, rtol=0, atol=0)

    def test_random_labels_are_fixed_within_function_and_rng_isolated(self):
        fs = [dict(line_labels=[1, 0, 0, 0, 0], line_numbers=[0, 2, 3, 5, 9]),
              dict(line_labels=[0, 0, 0], line_numbers=[0, 1, 2])]
        a, b = copy.deepcopy(fs), copy.deepcopy(fs)
        py_rng, tensor_rng = random.getstate(), torch.get_rng_state()
        shuffle_line_labels(a, 123456)
        shuffle_line_labels(b, 123456)
        self.assertEqual(a, b)
        self.assertEqual(py_rng, random.getstate())
        torch.testing.assert_close(tensor_rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertNotEqual(a, fs)
        for original, shuffled in zip(fs, a):
            self.assertEqual(sum(original["line_labels"]), sum(shuffled["line_labels"]))
            self.assertEqual(original["line_numbers"], shuffled["line_numbers"])

    def test_validation_true_diff_ap_and_source_only_save_reload(self):
        model = detector(.1).eval()
        metrics, predictions = validate_paired(model, rows(), Tokens(), "baseline", "cpu", loader=[batch()])
        from sklearn.metrics import average_precision_score
        y = [0, 1, 0, 0, 0, 0]
        p = [p for r in predictions for p in r["line_scores"]]
        self.assertEqual(metrics["line_samples"], 6)
        self.assertEqual(metrics["line_ap"], average_precision_score(y, p))
        cp = dict(protocol="losver", line_mapping="source_offsets", model_path="fixture", variant="baseline",
                  input_mode="512", line_auxiliary=dict(weight=.1), state_dict=model.state_dict())
        def factory(path, variant, **kwargs):
            return detector(kwargs.get("line_auxiliary_weight")), Tokens()
        with tempfile.TemporaryDirectory() as tmp, patch("vulnmechanism.model.losver_model", side_effect=factory):
            checkpoint = Path(tmp) / "model.pt"
            torch.save(cp, checkpoint)
            logit = predict_checkpoint(checkpoint, [dict(raw_source=rows()[0]["func"])], device="cpu")
        torch.testing.assert_close(logit.sigmoid(), torch.tensor([predictions[0]["probability"]]), rtol=1e-6, atol=1e-7)
        args = build_parser().parse_args(["train", "--line-auxiliary", "random", "--line-auxiliary-weight", ".3"])
        self.assertEqual((args.line_auxiliary, args.line_auxiliary_weight), ("random", .3))


if __name__ == "__main__":
    unittest.main()
