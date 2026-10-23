"""Source-offset alignment, code-only pooling, common heads and Oracle isolation."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast
from vulnmechanism.data import (
    modified_lines,
    paired_encoding,
    paired_features,
    write_jsonl,
)
from vulnmechanism.model import LOSVERLocalizer, LOSVERDetector, predict_checkpoint
from vulnmechanism.metrics import paired_metrics
from vulnmechanism.training import (
    LOSVERAdamW,
    paired_rows,
    paired_collate,
    paired_forward,
)
from vulnmechanism.evaluation import validate_paired, load_paired


class Tokens:
    """Deterministic offsets; whitespace runs deliberately merge line endings."""

    is_fast = True
    cls_token_id, pad_token_id, sep_token_id, unk_token_id = range(4)
    all_special_ids = [0, 1, 2, 3]

    def __init__(self):
        self.calls = 0

    def __call__(self, text, **kwargs):
        self.calls += 1
        ids, offsets, at = [], [], 0
        while at < len(text):
            end = at + 1
            if text[at].isspace():
                while end < len(text) and text[end].isspace():
                    end += 1
                token = 4
            else:
                token = 5 + ord(text[at]) % 120
            ids.append(token)
            offsets.append((at, end))
            at = end
        return dict(input_ids=ids, offset_mapping=offsets)


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
    )


def model(local=False):
    c = config()
    encoder = RobertaModel(c, add_pooling_layer=False)
    cls = LOSVERLocalizer if local else LOSVERDetector
    return cls(encoder, c, SimpleNamespace(class_weight=1.0))


def feature(row, variant, tokenizer=None):
    t = tokenizer or Tokens()
    e = paired_encoding(row["func"], t)
    row = dict(row, line_numbers=e["line_numbers"])
    return paired_features(row, t, variant, encoding=e)


class LOSVERTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_diff_rule_unchanged(self):
        a = "int f(){\n return 1;\n}\n"
        b = "int f(){\n check();\n return 1;\n}\n"
        self.assertEqual(modified_lines(a, b), [])
        self.assertEqual(modified_lines(b, a), [1])
        self.assertEqual(modified_lines(a, a.replace("1", "2")), [1])
        self.assertEqual(modified_lines(a, a.replace(" return", "\n return")), [])

    def test_physical_lines_merged_whitespace_crlf_and_eof(self):
        for code, expected in [
            ("a\n\n  b\n", [0, 2]),
            ("a\r\n \r\n b", [0, 2]),
            ("\n  a\n\n\t b", [1, 3]),
            ("a", [0]),
        ]:
            with self.subTest(code=repr(code)):
                t = Tokens()
                e = paired_encoding(code, t)
                self.assertEqual(e["line_numbers"], expected)
                self.assertEqual(t.calls, 1)
                for n, ts in zip(e["line_numbers"], e["line_token_indices"]):
                    for i in ts:
                        a, b = e["offset_mapping"][i]
                        self.assertTrue(code[a:b].strip())
                        parts = code.splitlines(keepends=True)
                        start = sum(len(part) for part in parts[:n])
                        end = start + len(parts[n])
                        self.assertTrue(code[max(a, start) : min(b, end)].strip())
                self.assertFalse(e["code_mask"][0])
                self.assertFalse(e["code_mask"][-1])
        with self.assertRaisesRegex(ValueError, "no non-whitespace"):
            paired_encoding(" \r\n\t\n", Tokens())

    def test_actual_byte_bpe_offsets(self):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

        backend = Tokenizer(models.BPE(unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        corpus = ["int f(){\n  a();  \n\n  b();\n}", "a \r\n \r\nb", "中😀\n x"] * 10
        backend.train_from_iterator(
            corpus,
            trainers.BpeTrainer(
                vocab_size=300,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
            ),
        )
        t = RobertaTokenizerFast(
            tokenizer_object=backend,
            trim_offsets=False,
            bos_token="<s>",
            eos_token="</s>",
            cls_token="<s>",
            sep_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
            mask_token="<mask>",
        )
        for source, expected in [
            (corpus[0], [0, 1, 3, 4]),
            (corpus[1], [0, 2]),
            (corpus[2], [0, 1]),
        ]:
            self.assertEqual(paired_encoding(source, t)["line_numbers"], expected)

    def test_cross_line_token_has_shared_memberships(self):
        class Span(Tokens):
            def __call__(self, text, **kwargs):
                self.calls += 1
                return dict(input_ids=[5], offset_mapping=[(0, len(text))])

        t = Span()
        r = dict(func="a\nb", target=1, lines_ground_truth=[1])
        e = paired_encoding(r["func"], t)
        self.assertEqual(e["line_numbers"], [0, 1])
        self.assertEqual(e["line_token_indices"], [[1], [1]])
        a = paired_features(r, t, "localizer", encoding=e)
        b = paired_features(
            dict(r, line_numbers=[0, 1], lines_predicted=[1]),
            t,
            "predicted",
            encoding=e,
        )
        self.assertEqual(a["labels"], [0, 1])
        self.assertEqual(b["token_weights"][1], 5)
        self.assertEqual(t.calls, 1)

    def test_overlapping_byte_offsets_still_mark_incomplete_line(self):
        class Bytes(Tokens):
            def __call__(self, text, **kwargs):
                return dict(
                    input_ids=[5] * 512,
                    offset_mapping=[(i, i + 1) for i in range(509)] + [(509, 510)] * 3,
                )

        f = paired_encoding("a" * 509 + "😀", Bytes())
        self.assertEqual(f["source_tokens"], 512)
        self.assertEqual(f["line_numbers"], [0])
        self.assertEqual(f["line_complete"], [False])

    def test_ragged_batch_preserves_sparse_physical_ids(self):
        rows = [
            dict(
                func="\n a\n\n b",
                target=1,
                lines_ground_truth=[3],
                sample_key="train:0",
            ),
            dict(func="c", target=0, lines_ground_truth=[], sample_key="train:1"),
        ]
        batch = paired_collate([feature(r, "localizer") for r in rows])
        self.assertEqual(batch["line_numbers"].tolist(), [[1, 3], [0, -1]])
        self.assertEqual(batch["labels"].tolist(), [[0, 1], [0, -1]])
        m = model(True).eval()
        _, outputs = validate_paired(
            m, rows, Tokens(), "localizer", "cpu", loader=[batch]
        )
        self.assertEqual([r["line_numbers"] for r in outputs], [[1, 3], [0]])

    def test_all_variants_share_truncation_and_partial_final_line(self):
        r = dict(
            func="a\n" + "x" * 700,
            target=1,
            lines_ground_truth=[1],
            lines_predicted=[1],
        )
        t = Tokens()
        e = paired_encoding(r["func"], t)
        r["line_numbers"] = e["line_numbers"]
        fs = [
            paired_features(r, t, v, encoding=e)
            for v in ["baseline", "localizer", "oracle", "predicted"]
        ]
        self.assertEqual(t.calls, 1)
        for f in fs:
            self.assertEqual(len(f["input_ids"]), 512)
            self.assertEqual(f["input_ids"][-1], 2)
            self.assertEqual(f["line_numbers"], [0, 1])
            self.assertEqual(f["line_complete"], [True, False])
            self.assertEqual(f["input_ids"], fs[0]["input_ids"])
            self.assertEqual(f["code_mask"], fs[0]["code_mask"])
        self.assertEqual(fs[1]["labels"], [0, 1])
        self.assertTrue(
            all(fs[3]["token_weights"][i] == 5 for i in e["line_token_indices"][1])
        )
        single = feature(
            dict(func="x" * 700, target=0, lines_ground_truth=[]), "localizer"
        )
        self.assertEqual(single["line_numbers"], [0])
        self.assertEqual(single["line_complete"], [False])
        self.assertEqual(sum(single["code_mask"]), 510)

    def test_softmax_excludes_padding_and_specials_and_trains_all_dimensions(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.h = nn.Parameter(torch.arange(80.0).reshape(1, 5, 16) / 100)

            def forward(self, *args, **kwargs):
                return (self.h,)

        m = LOSVERDetector(Encoder(), config(), SimpleNamespace(class_weight=1.0))
        with torch.no_grad():
            m.classifier.weight.fill_(1 / 16)
            m.classifier.bias.zero_()
        code = torch.tensor([[False, True, True, False, False]])
        weights = torch.tensor([[999.0, 5.0, 1.0, 999.0, 999.0]])
        pooled = m.pool(m.encoder.h, code, weights)
        expected = (
            m.encoder.h[:, 1:3] * torch.tensor([5.0, 1.0]).softmax(0)[None, :, None]
        ).sum(1)
        torch.testing.assert_close(pooled, expected)
        extra = torch.cat((m.encoder.h, torch.full((1, 30, 16), 1e6)), 1)
        torch.testing.assert_close(
            m.pool(
                extra,
                torch.cat((code, torch.zeros(1, 30, dtype=torch.bool)), 1),
                torch.cat((weights, torch.full((1, 30), 999.0)), 1),
            ),
            expected,
        )
        loss, p = m(
            input_ids=torch.tensor([[0, 5, 6, 2, 1]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
            code_mask=code,
            token_weights=weights,
            window_owners=torch.tensor([0]),
            labels=torch.ones(1),
        )
        loss.backward()
        self.assertTrue((m.encoder.h.grad[:, 1:3].abs() > 0).all())
        self.assertEqual(float(m.encoder.h.grad[:, [0, 3, 4]].abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, "no valid"):
            m.pool(m.encoder.h, torch.zeros_like(code), weights)

    def test_encoder_padding_invariance_for_both_heads(self):
        row = dict(func="a\n\n b", target=1, lines_ground_truth=[2])
        for local, v in [(False, "oracle"), (True, "localizer")]:
            m = model(local).eval()
            b = paired_collate([feature(row, v)])
            full = paired_forward(m, b, v, "cpu", with_labels=False)
            end = int(b["attention_mask"].sum())
            short = dict(b)
            for k in [
                "input_ids",
                "attention_mask",
                "code_mask",
                "context_mask",
                "token_weights",
            ]:
                short[k] = b[k][:, :end]
            short["line_token_mask"] = b["line_token_mask"][:, :, :end]
            actual = paired_forward(m, short, v, "cpu", with_labels=False)
            if local:
                full, actual = full[0], actual[0]
            torch.testing.assert_close(full, actual, atol=1e-6, rtol=1e-5)

    def test_same_factory_and_linear_head_for_three_detectors(self):
        from vulnmechanism import model as core

        t = Tokens()
        c = config()
        row = dict(func="a\n b", target=0, lines_ground_truth=[], lines_predicted=[])
        outputs = []
        states = []
        with patch.object(
            core.RobertaConfig, "from_pretrained", return_value=c
        ), patch.object(
            core.RobertaTokenizerFast, "from_pretrained", return_value=t
        ), patch.object(
            core.RobertaModel,
            "from_pretrained",
            side_effect=lambda *a, **kw: RobertaModel(c, add_pooling_layer=False),
        ):
            for v in ["baseline", "predicted", "oracle"]:
                torch.manual_seed(42)
                m, _ = core.losver_model("fixture", v)
                m.eval()
                self.assertIs(type(m), LOSVERDetector)
                self.assertEqual(tuple(m.classifier.weight.shape), (1, 16))
                states.append(m.state_dict())
                outputs.append(
                    paired_forward(
                        m,
                        paired_collate([feature(row, v)]),
                        v,
                        "cpu",
                        with_labels=False,
                    )
                )
        for z in outputs[1:]:
            torch.testing.assert_close(z, outputs[0], atol=0, rtol=0)
        for state in states[1:]:
            for k in state:
                torch.testing.assert_close(state[k], states[0][k], atol=0, rtol=0)

    def test_original_focal_and_adamw_are_preserved(self):
        m = model(True)
        z = torch.tensor([-0.7, 0.4], requires_grad=True)
        y = torch.tensor([0.0, 1.0])
        bce = nn.functional.binary_cross_entropy_with_logits(
            z, y, pos_weight=torch.tensor(5.0), reduction="none"
        )
        pt = z.sigmoid() * y + (1 - z.sigmoid()) * (1 - y)
        expected = (0.8 * y + 0.2 * (1 - y)) * (1 - pt) ** 1.5 * bce
        torch.testing.assert_close(m.focal_loss(z, y), expected)
        expected.mean().backward()
        self.assertGreater(z.grad[0], 0)
        self.assertLess(z.grad[1], 0)
        p = nn.Parameter(torch.tensor([0.1, -0.3]))
        q = p.detach().clone()
        avg = torch.zeros_like(q)
        sq = avg.clone()
        opt = LOSVERAdamW([p], lr=0.003)
        for step in range(1, 4):
            g = torch.tensor([0.2 / step, -0.4])
            p.grad = g.clone()
            opt.step()
            avg.mul_(0.9).add_(g, alpha=0.1)
            sq.mul_(0.999).addcmul_(g, g, value=0.001)
            q.addcdiv_(
                avg,
                sq.sqrt() + 1e-8,
                value=-0.003 * (1 - 0.999**step) ** 0.5 / (1 - 0.9**step),
            )
            torch.testing.assert_close(p, q, atol=1e-7, rtol=0)

    def test_real_encoder_backward_and_strict_reload(self):
        row = dict(
            func="a\n\n b", target=1, lines_ground_truth=[2], lines_predicted=[2]
        )
        for v in ["baseline", "localizer", "predicted", "oracle"]:
            m = model(v == "localizer")
            b = paired_collate([feature(row, v)])
            loss, _ = paired_forward(m, b, v, "cpu")
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(
                all(p.grad is None or p.grad.isfinite().all() for p in m.parameters())
            )
            self.assertGreater(
                float(m.encoder.get_input_embeddings().weight.grad.abs().sum()), 0
            )
            m.eval()
            expected = paired_forward(m, b, v, "cpu", with_labels=False)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.pt"
                torch.save(m.state_dict(), path)
                restored = copy.deepcopy(m)
                restored.load_state_dict(
                    torch.load(path, weights_only=True), strict=True
                )
                actual = paired_forward(restored, b, v, "cpu", with_labels=False)
                if v == "localizer":
                    actual, expected = actual[0], expected[0]
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_localizer_positive_f1_and_physical_output_ids(self):
        row = dict(
            func="a\n" * 10, target=1, lines_ground_truth=[0], sample_key="train:0"
        )
        batch = paired_collate([feature(row, "localizer")])
        with patch(
            "vulnmechanism.training.paired_forward",
            return_value=(torch.tensor(0.0), [torch.full((10,), 0.1)]),
        ):
            metrics, outputs = validate_paired(
                nn.Identity(), [row], Tokens(), "localizer", "cpu", loader=[batch]
            )
        self.assertEqual(metrics["f1"], 0)
        self.assertGreater(metrics["weighted_f1"], 0.8)
        self.assertEqual(metrics["precision"], 0)
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(outputs[0]["line_numbers"], list(range(10)))

    def test_pair_metrics_and_oracle_isolation(self):
        rows = [dict(target=y) for y in [1, 0] * 4]
        self.assertEqual(
            paired_metrics(rows, [0.8, 0.2, 0.8, 0.8, 0.2, 0.2, 0.2, 0.8])[
                "pair_counts"
            ],
            dict(p_c=1, p_v=1, p_b=1, p_r=1),
        )
        self.assertEqual(paired_metrics(rows[:2], [0.5, 0.5])["p_b"], 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r = dict(func="a", target=1, lines_predicted=[0], line_numbers=[0])
            write_jsonl(
                root / "runs/primevul_paired/losver_localizer/test.predicted.jsonl", [r]
            )
            write_jsonl(
                root / "data/primevul_paired/test.oracle.jsonl",
                [dict(func="FORBIDDEN", lines_ground_truth=[999])],
            )
            self.assertEqual(
                paired_rows(root / "data", root / "runs", "predicted", "test"), [r]
            )
            self.assertEqual(
                paired_features(
                    dict(r, lines_ground_truth=[999]), Tokens(), "predicted"
                ),
                paired_features(dict(r, lines_ground_truth=[]), Tokens(), "predicted"),
            )
        with self.assertRaises(KeyError):
            paired_features(dict(func="a", target=0), Tokens(), "oracle")
        with self.assertRaises(KeyError):
            paired_features(dict(func="a", target=0), Tokens(), "predicted")
        with self.assertRaisesRegex(ValueError, "historical"):
            load_paired(dict(model_path="fixture", variant="localizer"), "cpu")

    def test_source_only_inference_uses_one_tokenization(self):
        from vulnmechanism import model as core

        t = Tokens()
        det = model().eval()
        loc = model(True).eval()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lp = root / "localizer.pt"
            dp = root / "detector.pt"
            common = dict(
                model_path="fixture",
                line_mapping="source_offsets",
                input_mode="512",
                decision_threshold=0.5,
                protocol="losver",
            )
            torch.save(
                dict(common, variant="localizer", state_dict=loc.state_dict()), lp
            )
            torch.save(
                dict(
                    common,
                    variant="predicted",
                    state_dict=det.state_dict(),
                    localizer_path=str(lp),
                ),
                dp,
            )

            def factory(path, variant, **kwargs):
                return copy.deepcopy(loc if variant == "localizer" else det), t

            with patch.object(core, "losver_model", side_effect=factory), patch(
                "vulnmechanism.data.read_jsonl",
                side_effect=AssertionError("inference must not open annotations"),
            ):
                actual = predict_checkpoint(
                    dp, [dict(raw_source="a\n\n b")], device="cpu"
                )
            self.assertTrue(torch.isfinite(actual).all())
            self.assertEqual(t.calls, 1)

    def test_l1_boundaries_and_unique_source_cores(self):
        for n in (1, 384, 509, 510, 511, 512, 768, 5000):
            with self.subTest(tokens=n):
                e = paired_encoding("a" * n, Tokens(), "l1")
                coverage = [0] * n
                for left, right, a, b in e["windows"]:
                    self.assertLessEqual(right - left + 2, 512)
                    self.assertLessEqual(a - left, 63)
                    self.assertLessEqual(right - b, 63)
                    if n > 510:
                        self.assertLessEqual(b - a, 384)
                    for p in range(a, b):
                        coverage[p] += 1
                self.assertEqual(coverage, [1] * n)
                self.assertEqual(e["line_complete"], [True])
                f = paired_features(
                    dict(func="a" * n, target=1, lines_ground_truth=[0]),
                    Tokens(),
                    "oracle",
                    encoding=e,
                )
                batch = paired_collate([f])
                self.assertEqual(batch["input_ids"].shape, (len(e["windows"]), 512))
                core_positions = batch["source_positions"][batch["code_mask"]].tolist()
                self.assertEqual(sorted(core_positions), list(range(n)))
                self.assertEqual(
                    int((batch["token_weights"][batch["code_mask"]] == 5).sum()), n
                )

    def test_short_l1_is_identical_to_512_for_all_variants(self):
        for source in ("a\r\n\r\n b", "a" * 510):
            r = dict(func=source, target=1, lines_ground_truth=[0], lines_predicted=[0])
            for variant in ("baseline", "predicted", "oracle", "localizer"):
                batches = []
                for mode in ("512", "l1"):
                    e = paired_encoding(source, Tokens(), mode)
                    f = paired_features(
                        dict(r, line_numbers=e["line_numbers"]),
                        Tokens(),
                        variant,
                        encoding=e,
                    )
                    batches.append(paired_collate([f]))
                for key in batches[0]:
                    torch.testing.assert_close(
                        batches[0][key], batches[1][key], atol=0, rtol=0
                    )
                m = model(variant == "localizer").eval()
                p = paired_forward(m, batches[0], variant, "cpu", with_labels=False)
                q = paired_forward(m, batches[1], variant, "cpu", with_labels=False)
                if variant == "localizer":
                    p, q = p[0], q[0]
                torch.testing.assert_close(p, q, atol=0, rtol=0)

    def test_l1_global_softmax_does_not_recount_context(self):
        class IdentityEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(16))

            def forward(self, input_ids, **kwargs):
                return (F.one_hot(input_ids % 16, 16).float() * self.scale,)

        r = dict(func="a" * 700 + "\n b" * 30, target=1, lines_ground_truth=[0])
        e = paired_encoding(r["func"], Tokens(), "l1")
        f = paired_features(r, Tokens(), "oracle", encoding=e)
        b = paired_collate([f])
        m = LOSVERDetector(
            IdentityEncoder(), config(), SimpleNamespace(class_weight=1.0)
        )
        m.window_micro_batch = 1
        m.classifier.weight.data.copy_(torch.arange(16)[None] / 16)
        m.classifier.bias.data.zero_()
        loss, p = paired_forward(m, b, "oracle", "cpu")
        ids = torch.tensor(f["input_ids"])
        mask = torch.tensor(f["code_mask"])
        weights = torch.tensor(f["token_weights"])[mask].softmax(0)
        expected = (F.one_hot(ids[mask] % 16, 16).float() * weights[:, None]).sum(0)
        torch.testing.assert_close(
            p[0, 0], m.classifier(expected).sigmoid()[0], atol=1e-7, rtol=0
        )
        uniform = dict(b, token_weights=torch.ones_like(b["token_weights"]))
        _, plain = paired_forward(m, uniform, "baseline", "cpu")
        mean = F.one_hot(ids[mask] % 16, 16).float().mean(0)
        torch.testing.assert_close(
            plain[0, 0], m.classifier(mean).sigmoid()[0], atol=1e-7, rtol=0
        )
        poisoned = dict(b)
        poisoned["token_weights"] = b["token_weights"].masked_fill(~b["code_mask"], 999)
        _, same = paired_forward(m, poisoned, "oracle", "cpu")
        torch.testing.assert_close(p, same, atol=0, rtol=0)
        loss.backward()
        self.assertTrue(torch.isfinite(m.encoder.scale.grad).all())
        self.assertGreater(float(m.encoder.scale.grad.norm()), 0)

    def test_long_physical_line_is_classified_once_after_core_mean(self):
        class IdentityEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(16))

            def forward(self, input_ids, **kwargs):
                return (F.one_hot(input_ids % 16, 16).float() * self.scale,)

        class EchoAttention(nn.Module):
            def forward(self, q, k, v):
                return q, None

        r = dict(
            func="a" * 900 + "b" * 500 + "\r\n\r\n c", target=1, lines_ground_truth=[0]
        )
        e = paired_encoding(r["func"], Tokens(), "l1")
        b = paired_collate([paired_features(r, Tokens(), "localizer", encoding=e)])
        m = LOSVERLocalizer(
            IdentityEncoder(), config(), SimpleNamespace(class_weight=1.0)
        )
        m.line_attention = EchoAttention()
        m.line_classifier = nn.Sequential(nn.Linear(32, 1, bias=False))
        m.line_classifier[0].weight.data.copy_(torch.arange(32)[None] / 32)
        loss, p = paired_forward(m, b, "localizer", "cpu")
        self.assertEqual(e["line_numbers"], [0, 2])
        self.assertEqual(len(p[0]), 2)
        for i, tokens in enumerate(e["line_token_indices"]):
            mean = (
                F.one_hot(torch.tensor(e["input_ids"])[tokens] % 16, 16).float().mean(0)
            )
            expected = m.line_classifier(torch.cat((mean, mean))).sigmoid()[0]
            torch.testing.assert_close(p[0][i], expected)
        loss.backward()
        self.assertGreater(float(m.encoder.scale.grad.norm()), 0)

    def test_window_checkpoint_preserves_dropout_and_parameter_gradients(self):
        r = dict(func="ab" * 270 + "\n c", target=1, lines_ground_truth=[0])
        e = paired_encoding(r["func"], Tokens(), "l1")
        for local, variant in ((False, "oracle"), (True, "localizer")):
            b = paired_collate([paired_features(r, Tokens(), variant, encoding=e)])
            plain = model(local).train()
            plain.window_micro_batch = 1
            checked = copy.deepcopy(plain)
            checked.gradient_checkpointing = True
            torch.manual_seed(42)
            lp, _ = paired_forward(plain, b, variant, "cpu")
            lp.backward()
            torch.manual_seed(42)
            lc, _ = paired_forward(checked, b, variant, "cpu")
            lc.backward()
            torch.testing.assert_close(lp, lc)
            for (name, p), (other, q) in zip(
                plain.named_parameters(), checked.named_parameters()
            ):
                self.assertEqual(name, other)
                if p.grad is None:
                    self.assertIsNone(q.grad)
                else:
                    torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)

    def test_paired_report_preserves_input_budget_and_rejects_mixing(self):
        import json
        from vulnmechanism.artifacts import write_json
        from vulnmechanism.evaluation import paired_report

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "data/primevul_paired/statistics.json", dict(release="fixture")
            )
            records = [
                dict(sample_key="v", pair_key="p", target=1, probability=0.9),
                dict(sample_key="p", pair_key="p", target=0, probability=0.1),
            ]
            value = paired_metrics(records, [0.9, 0.1])
            out = root / "results/primevul_paired"
            for method in ("unixcoder", "losver", "losver_oracle"):
                run = out / method
                write_json(
                    run / "training.metrics.json",
                    dict(
                        line_mapping="source_offsets",
                        input_mode="l1",
                        selected_epochs=dict(p_c=1),
                    ),
                )
                for split in ("valid", "test"):
                    write_json(
                        run / f"{split}.metrics.json",
                        dict(selected_epoch=1, metrics=value),
                    )
                    write_jsonl(run / f"{split}.predictions.jsonl", records)
            write_json(
                out / "losver_localizer/best-f1.json",
                dict(epoch=1, validation=dict(f1=0.5, precision=0.4, recall=0.6)),
            )
            result = paired_report(root / "results", root / "data")
            self.assertEqual(len(result), 6)
            comparison = json.loads((out / "comparison.json").read_text())
            self.assertEqual(comparison["input_mode"], "l1")
            self.assertIn("auc", (out / "comparison.csv").read_text().splitlines()[0])
            self.assertIn("384核心Token", (out / "report.md").read_text())
            path = out / "losver/training.metrics.json"
            changed = json.loads(path.read_text())
            changed["input_mode"] = "512"
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, "mix input budgets"):
                paired_report(root / "results", root / "data")
