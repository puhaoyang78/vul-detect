"""Exact upstream heads, truncation, Focal loss, optimizer and oracle isolation."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from transformers import RobertaConfig, RobertaModel, RobertaForSequenceClassification
from vulnmechanism.data import modified_lines, paired_features, write_jsonl
from vulnmechanism.model import LOSVERLocalizer, LOSVERBaseline, LOSVERDetector
from vulnmechanism.metrics import paired_metrics
from vulnmechanism.training import LOSVERAdamW, paired_rows


class Tokens:
    cls_token = "<s>"
    sep_token = "</s>"
    pad_token = "<pad>"
    pad_token_id = 1
    unk_token_id = 3

    def tokenize(self, text):
        return list(text)

    def convert_tokens_to_ids(self, tokens):
        mapping = {"<s>": 0, "</s>": 2, "<pad>": 1, "<unk>": 3, "\n": 4, "Ċ": 4}
        one = lambda t: mapping[t] if t in mapping else 5 + ord(t) % 120
        return one(tokens) if isinstance(tokens, str) else [one(t) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        return [
            (
                "Ċ"
                if int(i) == 4
                else "</s>" if int(i) == 2 else "<pad>" if int(i) == 1 else "x"
            )
            for i in ids
        ]

    def encode(self, text, **kwargs):
        return self.convert_tokens_to_ids(self.tokenize(text))


def config():
    return RobertaConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=514,
        pad_token_id=1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_labels=1,
    )


def models():
    c = config()
    t = Tokens()
    args = SimpleNamespace(class_weight=1.0, dropout_probability=0.0)
    return [
        LOSVERBaseline(RobertaForSequenceClassification(c), c, t, args),
        LOSVERLocalizer(RobertaModel(c), c, t, args),
        LOSVERDetector(RobertaModel(c), c, t, args),
    ]


class LOSVERTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_diff_insert_delete_and_nonblank_rule(self):
        a = "int f(){\n return 1;\n}\n"
        b = "int f(){\n check();\n return 1;\n}\n"
        self.assertEqual(modified_lines(a, b), [])
        self.assertEqual(modified_lines(b, a), [1])
        self.assertEqual(modified_lines(a, a.replace("1", "2")), [1])
        self.assertEqual(modified_lines(a, a.replace(" return", "\n return")), [])

    def test_distinct_official_truncation_no_task_prefix(self):
        t = Tokens()
        row = dict(func="x" * 700, lines_ground_truth=[0], target=1)
        baseline = paired_features(row, t, "baseline")
        weighted = paired_features(row, t, "oracle")
        local = paired_features(row, t, "localizer")
        self.assertEqual(len(baseline["input_ids"]), 512)
        self.assertEqual(baseline["input_ids"][-1], 2)
        self.assertEqual(weighted["input_ids"][-1], t.convert_tokens_to_ids("x"))
        self.assertEqual(local["input_ids"][:3], [0, 2, 1])
        self.assertTrue(all(y == -1 for y in local["labels"]))
        row["func"] = "a\nb\nc"
        row["lines_ground_truth"] = [1]
        f = paired_features(row, t, "localizer")
        self.assertEqual(f["labels"][:5], [0, 1, 0, -1, -1])
        self.assertEqual(
            f["input_ids"][:4],
            [0, t.convert_tokens_to_ids("a"), 4, t.convert_tokens_to_ids("b")],
        )

    def test_custom_focal_formula_and_gradients(self):
        model = models()[1]
        z = torch.tensor([-0.7, 0.4], requires_grad=True)
        y = torch.tensor([0.0, 1.0])
        loss = model.focal_loss(z, y)
        bce = nn.functional.binary_cross_entropy_with_logits(
            z, y, pos_weight=torch.tensor(5.0), reduction="none"
        )
        p = z.sigmoid()
        pt = p * y + (1 - p) * (1 - y)
        expected = (0.8 * y + 0.2 * (1 - y)) * (1 - pt) ** 1.5 * bce
        torch.testing.assert_close(loss, expected)
        loss.mean().backward()
        self.assertGreater(float(z.grad[0]), 0)
        self.assertLess(float(z.grad[1]), 0)

    def test_weighted_coordinate_zero_and_padding_softmax_are_upstream(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.h = nn.Parameter(torch.arange(64.0).reshape(1, 4, 16) / 100)

            def forward(self, *args, **kwargs):
                return (self.h,)

        t = Tokens()
        c = config()
        model = LOSVERDetector(Encoder(), c, t, SimpleNamespace(class_weight=1.0))
        ids = torch.tensor([[0, 5, 4, 1]])
        mask = ids.ne(1)
        loss, p = model(
            input_ids=ids, labels=torch.ones(1), attention_mask=mask, line_indices=[[0]]
        )
        weights = torch.tensor([5.0, 5.0, 5.0, 0.0]).softmax(0)
        expected = (model.encoder.h * weights[None, :, None]).sum(1).sigmoid()
        torch.testing.assert_close(p, expected)
        self.assertGreater(float(weights[-1]), 0)
        loss.backward()
        self.assertGreater(float(model.encoder.h.grad[0, :, 0].abs().sum()), 0)
        self.assertEqual(float(model.encoder.h.grad[0, :, 1:].abs().sum()), 0.0)

    def test_real_encoder_gradient_and_strict_save_reload(self):
        row = dict(
            func="a\nb\nc\n", target=1, lines_ground_truth=[1], lines_predicted=[1]
        )
        t = Tokens()
        for m, v in zip(models(), ["baseline", "localizer", "predicted"]):
            f = paired_features(row, t, v)
            kw = dict(input_ids=torch.tensor([f["input_ids"]]))
            if v == "predicted":
                kw.update(
                    attention_mask=torch.tensor([f["attention_mask"]]),
                    line_indices=[f["line_indices"]],
                )
            m.train()
            loss, _ = m(**kw, labels=torch.tensor([f["labels"]]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(
                all(p.grad is None or p.grad.isfinite().all() for p in m.parameters())
            )
            self.assertGreater(
                float(m.encoder.get_input_embeddings().weight.grad.abs().sum()), 0
            )
            m.eval()
            expected = m(**kw)
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "weights.pt"
                torch.save(m.state_dict(), p)
                restored = copy.deepcopy(m)
                restored.load_state_dict(torch.load(p, weights_only=True), strict=True)
                actual = restored(**kw)
                if isinstance(expected, list):
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b, atol=0, rtol=0)
                else:
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_batched_localizer_matches_independent_official_line_queries(self):
        model = models()[1].eval()
        t = Tokens()
        f = paired_features(
            dict(func="a\n\nb\n", target=1, lines_ground_truth=[2]), t, "localizer"
        )
        ids = torch.tensor([f["input_ids"]])
        hidden = model.encoder(ids, attention_mask=ids.ne(1))[0][0]
        boundaries = torch.cat(
            [torch.tensor([0]), (ids[0] == 4).nonzero().flatten() + 1]
        )
        context = hidden[: boundaries[-1]]
        logits = []
        for start, end in zip(boundaries[:-1], boundaries[1:] - 1):
            if end <= start:
                logits.append(torch.tensor(-10.0))
                continue
            line = hidden[start:end]
            attended, _ = model.line_attention(line[None], context[None], context[None])
            logits.append(
                model.line_classifier(
                    torch.cat([line.mean(0), attended[0].mean(0)])
                ).squeeze(-1)
            )
        expected = torch.stack(logits).sigmoid()
        actual = model(input_ids=ids)[0]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        self.assertAlmostEqual(
            float(actual[1]), float(torch.tensor(-10.0).sigmoid()), places=8
        )

    def test_metrics_four_pair_categories_and_strict_tie(self):
        rows = [dict(target=y) for y in [1, 0] * 4]
        r = paired_metrics(rows, [0.8, 0.2, 0.8, 0.8, 0.2, 0.2, 0.2, 0.8])
        self.assertEqual(r["pair_counts"], dict(p_c=1, p_v=1, p_b=1, p_r=1))
        self.assertEqual(r["accuracy"], 0.5)
        self.assertEqual(paired_metrics(rows[:2], [0.5, 0.5])["p_b"], 1.0)

    def test_predicted_test_cannot_read_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(func="a", target=1, lines_predicted=[0])]
            write_jsonl(
                root / "runs/primevul_paired/losver_localizer/test.predicted.jsonl",
                rows,
            )
            write_jsonl(
                root / "data/primevul_paired/test.oracle.jsonl",
                [dict(func="FORBIDDEN", lines_ground_truth=[999])],
            )
            self.assertEqual(
                paired_rows(root / "data", root / "runs", "predicted", "test"), rows
            )
            a = dict(func="a", target=0, lines_predicted=[0], lines_ground_truth=[999])
            b = dict(a, lines_ground_truth=[2])
            self.assertEqual(
                paired_features(a, Tokens(), "predicted"),
                paired_features(b, Tokens(), "predicted"),
            )

    def test_missing_weight_signals_raise_instead_of_becoming_unweighted(self):
        row = dict(func="a", target=0)
        for variant in ("predicted", "oracle"):
            with self.assertRaises(KeyError):
                paired_features(row, Tokens(), variant)

    def test_transformers_419_adamw_update(self):
        p = nn.Parameter(torch.tensor([0.1, -0.3]))
        q = p.detach().clone()
        avg = torch.zeros_like(q)
        sq = torch.zeros_like(q)
        opt = LOSVERAdamW([p], lr=0.003, eps=1e-8)
        for step in range(1, 4):
            g = torch.tensor([0.2 / step, -0.4])
            p.grad = g.clone()
            opt.step()
            avg.mul_(0.9).add_(g, alpha=0.1)
            sq.mul_(0.999).addcmul_(g, g, value=0.001)
            size = 0.003 * (1 - 0.999**step) ** 0.5 / (1 - 0.9**step)
            q.addcdiv_(avg, sq.sqrt() + 1e-8, value=-size)
            torch.testing.assert_close(p, q, atol=1e-7, rtol=0)
