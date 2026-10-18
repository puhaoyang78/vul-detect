"""Binding-safe dynamic search, VAT copy metrics and complete attack denominators."""

import json
import random
import tempfile
import unittest
from argparse import Namespace
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import RobertaConfig, RobertaModel
from vulnmechanism import model as core, syntax, augmentation as data
from vulnmechanism import vat
from vulnmechanism.training import function_view_batches
from tests.test_codebert_source import CodeTokenizer


class OffsetTokenizer(CodeTokenizer):
    is_fast = True

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        if text == "FORBIDDEN_HELDOUT":
            raise AssertionError("Test reached tokenizer")
        kwargs.pop("verbose", None)
        result = super().__call__(text, **kwargs)
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(result["input_ids"]))]
        return result


def tiny_model(root):
    path = root / "encoder"
    RobertaModel(
        RobertaConfig(
            vocab_size=16,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=514,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=1,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        ),
        add_pooling_layer=False,
    ).save_pretrained(path)
    core.seed_everything(42)
    model = core.CodeBERT(str(path), torch.device("cpu"))
    return path, model


MATERIALS = dict(
    source="unit fixture",
    train=dict(
        names=["count", "item", "array", "buffer", "total", "entry"],
        words=["ordinary", "section", "value", "later", "routine", "return"],
    ),
    heldout=dict(names=["orchard", "passage"], words=["linen", "cedar"]),
    rules=syntax.SURFACE_RULES,
)


def proposals(source, kind=None, seed=42, **kwargs):
    return syntax.surface_preserving_candidates(
        source,
        "c",
        rng=random.Random(seed),
        materials=MATERIALS,
        tokenizer=OffsetTokenizer(),
        kinds=[kind] if kind else None,
        **kwargs,
    )


class SurfaceVATTests(unittest.TestCase):
    def test_boundary_finishes_layer_skips_wrong_original_and_keeps_hardest_default(self):
        source = "int f(int x){/* ordinary section */int y=api(x);return x+y;}"
        rows = [dict(sample_key=k, raw_source=source, language="c", label=1)
                for k in ("cross", "wrong", "safe")]
        calls = Counter()
        full_layers = []

        def score(batch):
            result = []
            for row in batch:
                key = row["sample_key"]
                original = row["raw_source"] == source
                calls[(key, original)] += 1
                if original:
                    result.append(-1.0 if key == "wrong" else 2.0)
                else:
                    self.assertNotEqual(key, "wrong")
                    if key == "cross":
                        # All four candidates must be scored, even after a crossing.
                        value = [0.4, -2.0, -0.1, 0.3][calls[(key, False)] - 1]
                        full_layers.append(value)
                        result.append(value)
                    else:
                        result.append(0.5)
            return result

        tokenizer = OffsetTokenizer()
        found = data.search_surface_views(rows, score, tokenizer, 2025, MATERIALS,
                                          selection="boundary")
        cross, wrong, safe = found
        self.assertEqual(cross["stop_depth"], 1)
        self.assertEqual(cross["selection_reason"], "first_boundary_min_bce")
        self.assertEqual(cross["selected"]["logit"], -0.1)
        self.assertEqual(cross["candidate_queries"], len(full_layers))
        self.assertEqual(len(full_layers), 4)
        self.assertEqual(wrong["selection_reason"], "original_wrong_random")
        self.assertEqual(wrong["candidate_queries"], 0)
        self.assertEqual(calls[("wrong", False)], 0)
        self.assertIsNone(wrong["loss_delta"])
        self.assertTrue(all("logit" not in c and "loss" not in c for c in wrong["visited"]))
        self.assertEqual(safe["selection_reason"], "no_boundary_max_bce")
        self.assertTrue(any(len(c["units"]) > 1 for c in safe["visited"]))
        solo = data.search_surface_views([rows[1]], score, tokenizer, 2025, MATERIALS,
                                         selection="boundary")[0]
        self.assertEqual(wrong, solo)  # RNG is independent of batch neighbours.
        full = data.search_surface_views(
            [rows[0]], lambda batch: [2.0 if r["raw_source"] == source else -2.0 for r in batch],
            tokenizer, 2025, MATERIALS,
        )[0]
        self.assertGreater(full["candidate_queries"], cross["candidate_queries"])
        self.assertEqual(full["selected"]["loss"], max(c["loss"] for c in full["visited"]))
        with self.assertRaisesRegex(ValueError, "training"):
            data.search_surface_views(rows, score, tokenizer, 2025, MATERIALS,
                                      selection="boundary", purpose="adaptive")
        with self.assertRaisesRegex(ValueError, "no logit"):
            data.search_surface_views(
                [rows[0]], lambda batch: [2.0 if r["raw_source"] == source else None for r in batch],
                tokenizer, 2025, MATERIALS,
            )

    def test_search_reuses_offsets_and_passes_the_same_source_ids_to_scoring(self):
        class CountingTokenizer(OffsetTokenizer):
            def __init__(self):
                self.sources = Counter()

            def __call__(self, text, **kwargs):
                if kwargs.get("return_offsets_mapping"):
                    self.sources[text] += 1
                return super().__call__(text, **kwargs)

        tokenizer = CountingTokenizer()
        row = dict(sample_key="cache", language="c", label=1,
                   raw_source="int f(int x){/* ordinary section */return api(x);}")

        def score(batch):
            for r in batch:
                self.assertEqual(r["_source_token_ids"],
                                 OffsetTokenizer()(r["raw_source"])["input_ids"][:2025])
            return [1.0] * len(batch)

        result = data.search_surface_views([row], score, tokenizer, 2025, MATERIALS)[0]
        self.assertEqual(result["visibility_encodings"], len(tokenizer.sources))
        self.assertTrue(all(count == 1 for count in tokenizer.sources.values()))

    def test_logit_vat_uses_raw_sign_before_copy_correction_and_keeps_final_gradients(self):
        ids = torch.tensor([[0, 0, 1, -1]])
        owners, windows = torch.tensor([0, 0]), torch.tensor([0])
        initial = torch.randn((2, 1), generator=torch.Generator().manual_seed(42))
        initial, _ = vat.normalize_vat_groups(initial, ids, owners, windows, 1, 1.0)
        raw = torch.tensor([[3.0 / initial[0, 0]], [-2.0 / initial[1, 0]]])
        counts = torch.tensor([[2.0], [1.0]])
        self.assertGreater(float((raw * initial).sum()), 0)
        self.assertLess(float((raw / counts * initial).sum()), 0)

        class Builder:
            def vat_batch(self, records, *, device):
                batch = (torch.ones_like(ids), torch.ones_like(ids), windows,
                         torch.ones_like(ids), 1)
                return batch, ids, owners, [True]

        class LinearLogit(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = SimpleNamespace(config=SimpleNamespace(hidden_size=1))
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, *batch, embedding_delta=None, force_fp32=False):
                delta = torch.zeros((1, 4, 1)) if embedding_delta is None else embedding_delta
                weights = torch.tensor([raw[0, 0] / 2, raw[0, 0] / 2, raw[1, 0], 0.0])
                return (0.7 + (delta.double()[:, :, 0] * weights.double()).sum(1)) * self.scale

        model = LinearLogit()
        fd, old, a = vat.vat_loss(model, [{}], Builder(), device="cpu", noise_seed=42, return_delta=True)
        direct, new, b = vat.vat_loss(
            model, [{}], Builder(), device="cpu", noise_seed=42,
            direction_method="logit", return_delta=True,
        )
        expected, _ = vat.normalize_vat_groups(raw / counts, ids, owners, windows, 1, 1.0)
        torch.testing.assert_close(b, vat.expand_vat_groups(expected, ids), rtol=1e-6, atol=1e-7)
        self.assertGreater(float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)), 0.9999)
        self.assertEqual((old["function_forwards"], new["function_forwards"]), (3, 2))
        self.assertFalse(b.requires_grad)
        self.assertIsNone(model.scale.grad)
        direct.sum().backward()
        self.assertTrue(torch.isfinite(model.scale.grad))
        self.assertGreater(float(model.scale.grad.abs()), 0)
        torch.testing.assert_close(fd, direct, rtol=1e-5, atol=1e-6)
        with torch.no_grad():
            model.scale.zero_()
        zero, info, delta = vat.vat_loss(
            model, [{}], Builder(), device="cpu", noise_seed=42,
            direction_method="logit", return_delta=True,
        )
        self.assertEqual(info["zero_direction"], 1)
        self.assertEqual(info["zero_initial_projection"], 1)
        self.assertTrue(torch.all(delta == 0))
        self.assertEqual(float(zero.detach().sum()), 0)

    def test_four_training_switches_use_the_same_budget_and_save_their_config(self):
        from vulnmechanism.training import TRAINING, train
        from vulnmechanism.data import write_jsonl

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            encoder, _ = tiny_model(root)
            rows = [dict(sample_key=str(i), dataset="primevul", split="train", label=i % 2,
                         language="c", raw_source=f"int f{i}(int x){{/* ordinary section */return api(x)+{i};}}")
                    for i in range(8)]
            write_jsonl(root / "data/primevul/train.jsonl", rows)
            write_jsonl(root / "data/primevul/valid.jsonl",
                        [dict(r, split="valid", sample_key="v" + r["sample_key"]) for r in rows])
            config = dict(TRAINING, epochs=1, batch_size=2, gradient_accumulation=2)
            for selection in ("hardest", "boundary"):
                for direction in vat.VAT_DIRECTIONS:
                    run = root / f"{selection}_{direction}"
                    with patch.object(core.AutoTokenizer, "from_pretrained", return_value=OffsetTokenizer()):
                        summary = train(root / "data", run, str(encoder), "primevul", "hard_aug_vat",
                                        config=config, materials=MATERIALS, hard_selection=selection,
                                        vat_direction=direction, train_limit=4, valid_limit=2)
                    self.assertEqual(summary["optimizer_steps"], 1)
                    self.assertEqual((summary["train_samples"], summary["valid_samples"]), (4, 2))
                    self.assertEqual(summary["vat"]["direction"], direction)
                    self.assertEqual(summary["augmentation"]["selection"], selection)
                    logs = [json.loads(line) for line in
                            (run / "primevul/hard_aug_vat/best.training.jsonl").read_text().splitlines()]
                    epoch = next(r for r in logs if r["event"] == "epoch")
                    self.assertEqual(epoch["search_costs"]["function_forwards"],
                                     epoch["candidate_queries"] + epoch["original_search_queries"])
                    self.assertEqual(epoch["vat_costs"]["function_forwards"],
                                     4 * (3 if direction == "finite_difference" else 2))

    def test_binding_shadowing_utf8_members_macros_and_self_references(self):
        source = (
            "int f(int 变量){int y=fetch(变量);{int 变量=2;use(变量);}return y+变量+GLOBAL;}"
        )
        layout = syntax.surface_binding_layout(source, "c")
        self.assertEqual([b["name"] for b in layout["bindings"]], ["变量", "y", "变量"])
        selected = set()
        for seed in range(15):
            for candidate in proposals(source, "local_name", seed)[0]:
                selected.add(candidate["binding_token_index"])
                changed = candidate["raw_source"]
                binding = next(
                    b
                    for b in layout["bindings"]
                    if b["declaration"] == candidate["edits"][0][0]
                )
                self.assertEqual(
                    [(a, b) for a, b, v in candidate["edits"]],
                    list(map(tuple, binding["spans"])),
                )
                self.assertIn("fetch(", changed)
                self.assertIn("GLOBAL", changed)
        self.assertEqual(len(selected), 3)
        for c in proposals("int f(int x){return obj.x+x;}", "local_name")[0]:
            self.assertIn("obj.x+", c["raw_source"])
        layout = syntax.surface_binding_layout(
            "int f(int x,int y){CHECK(x);use(y);return y;}", "c"
        )
        self.assertEqual([b["name"] for b in layout["bindings"]], ["y"])
        self.assertGreater(layout["reasons"]["macro_ambiguous_binding"], 0)
        layout = syntax.surface_binding_layout(
            "int f(int x){{int (*x)(int);x(2);}return x;}", "c"
        )
        self.assertEqual(len(layout["bindings"][0]["spans"]), 2)
        for c in proposals("int f(int x){return x ? f(x-1) : 0;}", "function_name")[0]:
            self.assertEqual(len(c["edits"]), 2)
            self.assertFalse(c["strict_semantic"])
            self.assertTrue(c["function_name_input_only"])
            self.assertNotIn("f(", c["raw_source"])
        self.assertFalse(
            proposals("int f(int x){log(__func__);return x;}", "function_name")[0]
        )

    def test_gradient_copy_metric_and_expanded_radius(self):
        ids = torch.tensor([[0, 0, 1, -1], [2, 2, 2, -1]])
        owners = torch.tensor([0, 0, 1])
        windows = torch.tensor([0, 1])
        values = torch.ones(3, 5, requires_grad=True)
        vat.expand_vat_groups(values, ids).sum().backward()
        counts = torch.bincount(ids[ids >= 0], minlength=3)
        torch.testing.assert_close(values.grad / counts[:, None], torch.ones_like(values))
        for magnitude in (1.0, 1e-22, 1e-23, 1e-40):
            normalized, norms = vat.normalize_vat_groups(
                torch.full((3, 5), magnitude), ids, owners, windows, 2, 1.0
            )
            actual = vat.expand_vat_groups(normalized, ids)
            self.assertTrue(torch.all(norms > 0))
            self.assertTrue(torch.all(actual[ids < 0] == 0))
            torch.testing.assert_close(
                actual.double().square().sum((1, 2)).sqrt(), torch.ones(2, dtype=torch.float64)
            )
        normalized, norms = vat.normalize_vat_groups(
            torch.zeros(3, 5), ids, owners, windows, 2, 1.0
        )
        self.assertTrue(torch.all(norms == 0))
        self.assertTrue(torch.all(normalized == 0))

    def test_comment_edits_and_visibility(self):
        source = "int f(int x){/* ordinary\n section */return api(x);}"
        for kind in ("comment_insert", "comment_delete", "comment_replace"):
            candidates, reasons = proposals(source, kind)
            self.assertTrue(candidates, (kind, reasons))
            for c in candidates:
                self.assertEqual(
                    [t for t, _, _ in syntax.source_token_spans(source)],
                    [t for t, _, _ in syntax.source_token_spans(c["raw_source"])],
                )
                self.assertEqual(source.count("\n"), c["raw_source"].count("\n"))
                self.assertLessEqual(c["added_comment_tokens"], 64)
        for c in proposals("int/**/f(int x){return x;}", "comment_delete")[0]:
            self.assertIn("int f(", c["raw_source"])
        tokenizer = OffsetTokenizer()
        source = "int f(int x){/* ordinary section */return x+2;}"
        kept, reasons = data.visible_candidates(
            source, tokenizer, len(source), proposals=proposals(source, "comment_insert")[0]
        )
        self.assertFalse(kept)
        self.assertGreater(reasons["visible_execution_removed"], 0)
        deleted = proposals(source, "comment_delete")[0]
        kept, reasons = data.visible_candidates(
            source, tokenizer, source.index("return") + 3, proposals=deleted
        )
        self.assertFalse(kept)
        self.assertGreater(reasons["extra_execution_exposed"], 0)
        self.assertTrue(data.visible_candidates(source, tokenizer, 2048, proposals=deleted)[0])

    def test_dynamic_reproducibility_holdout_and_hardest_search(self):
        source = "int f(int x){/* ordinary section */int y=api(x);return x+y;}"
        row = dict(sample_key="stable", raw_source=source, language="c", label=1)
        tokenizer = OffsetTokenizer()
        parameter = torch.tensor([0.7], requires_grad=True)
        parameter.grad = torch.tensor([2.0])
        before = parameter.detach().clone()

        def score(rows):
            return [float(parameter.detach()) - len(r["raw_source"]) / 100 for r in rows]

        a = data.search_surface_views([row], score, tokenizer, 2025, MATERIALS, epoch=1)[0]
        self.assertEqual(
            a, data.search_surface_views([row], score, tokenizer, 2025, MATERIALS, epoch=1)[0]
        )
        self.assertNotEqual(
            a["visited"],
            data.search_surface_views([row], score, tokenizer, 2025, MATERIALS, epoch=2)[0][
                "visited"
            ],
        )
        self.assertLessEqual(a["candidate_queries"], 12)
        self.assertEqual(a["original_queries"], 1)
        self.assertEqual(a["selected"]["loss"], max(v["loss"] for v in a["visited"]))
        self.assertTrue(any(len(v["units"]) > 1 for v in a["visited"]))
        for v in a["visited"]:
            self.assertLessEqual(len(v["units"]), 3)
            self.assertLessEqual(v["added_comment_tokens"], 64)
            self.assertEqual(v["raw_source"], data.replay_surface_units(source, v["units"]))
            for combo in syntax.SURFACE_RULES["heldout_combinations"]:
                self.assertTrue(Counter(combo) - Counter(u["kind"] for u in v["units"]))
            for u in v["units"]:
                self.assertNotIn(
                    u["rule"],
                    syntax.SURFACE_RULES["heldout_names"]
                    + syntax.SURFACE_RULES["heldout_comments"],
                )
                for aa, bb, value in u["edits"]:
                    self.assertNotIn("orchard", value)
                    self.assertNotIn("cedar", value)
        torch.testing.assert_close(before, parameter)
        torch.testing.assert_close(parameter.grad, torch.tensor([2.0]))
        improved = data.search_surface_views(
            [row],
            lambda batch: [10.0 if r["raw_source"] != source else -10.0 for r in batch],
            tokenizer,
            2025,
            MATERIALS,
        )[0]
        self.assertIsNotNone(improved["selected"])
        self.assertLess(improved["loss_delta"], 0)
        random_view = data.search_surface_views(
            [{k: v for k, v in row.items() if k != "label"}],
            lambda batch: (_ for _ in ()).throw(
                AssertionError("random ablation scored candidates")
            ),
            tokenizer,
            2025,
            MATERIALS,
            random_selection=True,
        )[0]
        self.assertIsNotNone(random_view["selected"])
        self.assertEqual(random_view["candidate_queries"], 0)
        attack = data.search_surface_views(
            [row], score, tokenizer, 2025, MATERIALS, query_limit=48, purpose="adaptive"
        )[0]
        self.assertLessEqual(attack["candidate_queries"], 48)
        self.assertEqual(
            len(set(v["raw_source"] for v in attack["visited"])), len(attack["visited"])
        )

    def test_function_weights_and_complete_robustness_denominators(self):
        rows = [dict(sample_key=str(i), raw_source="original", label=i % 2) for i in range(3)]
        counts = {str(i): 0.0 for i in range(3)}
        for batch, weights in function_view_batches(
            rows, {"0": ["changed"], "1": [], "2": ["changed"]}, 2
        ):
            for r, w in zip(batch, weights):
                counts[r["sample_key"]] += w
        self.assertEqual(counts, {"0": 1.0, "1": 1.0, "2": 1.0})
        from vulnmechanism.evaluation import _robustness_summary

        originals = [dict(r, logit=-1.0, prediction=0) for r in rows]
        report = _robustness_summary(
            originals,
            {"0": [dict(logit=1.0, prediction=1)], "1": [dict(logit=1.0, prediction=1)]},
            0.5,
        )
        self.assertEqual(report["total"], 3)
        self.assertEqual(report["eligible"], 2)
        self.assertEqual(report["asr"]["denominator"], 1)
        self.assertEqual(report["asr"]["count"], 1)
        self.assertEqual(report["wrong_to_correct"]["count"], 1)
        self.assertEqual(report["worst_complete"]["accuracy"], 1 / 3)

    def test_evaluation_uses_the_saved_float32_threshold_boundary(self):
        from vulnmechanism.evaluation import _robustness_evaluation
        from vulnmechanism.metrics import decision_boundary

        z = torch.logit(torch.tensor(0.29))
        self.assertEqual(torch.sigmoid(z).item(), decision_boundary(0.29))

        class Constant(torch.nn.Module):
            def forward(self, *batch):
                return z.expand(batch[-1])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "bce"
            run.mkdir()
            target = run / "best.pt"
            source = "int f(int x){return x;}"
            rows = [
                dict(sample_key=str(i), raw_source=source, label=i, language="c", split="test")
                for i in (0, 1)
            ]
            originals = [
                dict(
                    sample_key=r["sample_key"], label=r["label"], logit=z.item(), prediction=1
                )
                for r in rows
            ]
            (run / "test.predictions.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in originals)
            )
            entries = [
                dict(
                    sample_key=r["sample_key"],
                    groups={
                        "comment_insert": [
                            dict(
                                strict_semantic=True,
                                units=[
                                    dict(
                                        edits=[
                                            [
                                                source.index("{") + 1,
                                                source.index("{") + 1,
                                                " /* note */ ",
                                            ]
                                        ]
                                    )
                                ],
                            )
                        ]
                    },
                    reasons={},
                )
                for r in rows
            ]
            for bank in ("fixed", "heldout"):
                (root / f"{bank}.candidates.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in entries)
                )
            tokenizer = OffsetTokenizer()
            builder = core.InputBuilder(tokenizer, source_max_length=2048)
            cp = dict(
                dataset="fixture",
                decision_threshold=0.29,
                selected_epoch=3,
                source_max_length=2048,
            )
            args = Namespace(device="cpu", batch_size=4, model="fixture")
            with patch.object(
                core.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ), patch.object(core, "load_model", return_value=Constant()), patch(
                "vulnmechanism.evaluation._evaluate_clean", return_value={}
            ):
                result = _robustness_evaluation(rows, run, target, cp, root, MATERIALS, args)
            self.assertEqual(
                result["fixed"]["comment_insert"]["all"]["perturbed"]["recall"], 1.0
            )
            self.assertEqual(
                result["fixed"]["comment_insert"]["all"]["all_candidates_correct"], 1
            )
            with (run / "fixed.predictions.jsonl").open() as handle:
                cached = [json.loads(l) for l in handle]
            self.assertTrue(
                all(
                    v["prediction"] == 1 for r in cached for v in r["groups"]["comment_insert"]
                )
            )
