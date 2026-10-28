"""Weak repair relations, source-only scores and bounded continuous fusion."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch

from tests.test_losver import Tokens, model
from vulnmechanism.data import paired_features, repair_region_supervision
from vulnmechanism.model import LOSVERDetector
from vulnmechanism.training import paired_collate, paired_forward
from vulnmechanism.evaluation import load_paired


class RepairRegionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_insert_maps_bound_operation_and_leaves_unknown_unlabeled(self):
        old = "int f(char *p,int n){\n log_event();\n return p[n];\n}\n"
        new = old.replace(" return", " if(n<0)return -1;\n return")
        v, p = repair_region_supervision(old, new)
        self.assertEqual(v["region_levels"][2], "bound_relation")
        self.assertEqual(v["region_labels"][2], 1.0)
        self.assertEqual(v["region_labels"][1], -1.0)
        self.assertEqual(v["region_weights"][1], 0.0)
        self.assertEqual(v["region_labels"][3], 0.0)
        self.assertEqual(p["region_levels"][2], "direct")
        for r, code in ((v, old), (p, new)):
            for span in r["region_spans"]:
                self.assertEqual(
                    code[span["start"] : span["end"]],
                    code.splitlines(keepends=True)[span["line"]],
                )

    def test_unknown_parse_and_rename_never_get_arbitrary_expansion(self):
        a, b = repair_region_supervision("int f(", "int f(int x){return x;}")
        self.assertIn("parse_failure_direct_only", a["region_reasons"])
        a, b = repair_region_supervision("int f(){return 1;}", "int g(){return 2;}")
        self.assertEqual(a["region_weights"], [0.0])
        self.assertEqual(b["region_weights"], [0.0])

    def test_array_initializer_is_not_whole_region(self):
        old = 'int f(){\n const char *a[]={\n "first",\n "second",\n "third"\n };\n return 0;\n}\n'
        new = old.replace(' "second",\n', "")
        r, _ = repair_region_supervision(old, new)
        self.assertEqual(r["region_levels"][3], "direct")
        self.assertEqual(r["region_levels"][2], "unknown")
        self.assertEqual(r["region_levels"][4], "unknown")

    def test_shadowing_and_field_paths_are_not_broad_same_name_matches(self):
        old = "int f(S *p,int n){\n consume(p->unrelated);\n return p->buffer[n];\n}\n"
        new = old.replace(" return", " if(p->size<0)return -1;\n return")
        r, _ = repair_region_supervision(old, new)
        self.assertEqual(r["region_weights"][1], 0.0)
        self.assertEqual(r["region_weights"][2], 0.0)

    def test_fusion_mean_fallback_padding_and_attenuation(self):
        torch.manual_seed(42)
        states = torch.randn(2, 7, 16, requires_grad=True)
        mask = torch.tensor(
            [[0, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0]], dtype=torch.bool
        )
        mean = (states * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        low = torch.full((2, 7), 0.49)
        low[~mask] = 1.0
        output = LOSVERDetector.robust_pool(states, mask, low, 0.25)
        torch.testing.assert_close(output, mean, rtol=0, atol=0)
        scores = torch.ones(2, 7)
        scores[:, 2] = 0.501
        scores[:, 1] = 0.3
        scores[:, 3:] = 0.3
        output = LOSVERDetector.robust_pool(states, mask, scores, 0.25)
        bound = 0.25 * 0.002 * (states[:, 2] - mean).norm(dim=-1)
        self.assertTrue(((output - mean).norm(dim=-1) <= bound + 1e-6).all())
        output.sum().backward()
        self.assertTrue(torch.isfinite(states.grad).all())
        self.assertEqual(float(states.grad[~mask].abs().sum()), 0.0)
        padded = LOSVERDetector.robust_pool(
            torch.cat((states.detach(), torch.randn(2, 3, 16)), 1),
            torch.cat((mask, torch.zeros(2, 3, dtype=torch.bool)), 1),
            torch.cat((scores, torch.ones(2, 3)), 1),
            0.25,
        )
        torch.testing.assert_close(padded, output.detach())

    def test_nested_fields_and_generic_call_context_are_unknown(self):
        old = "int f(S *p,C *ctx){\n alloc(ctx,&p);\n return p->state.value;\n}\n"
        new = old.replace(" alloc", " if(p->state.limit<0)return -1;\n alloc")
        a, _ = repair_region_supervision(old, new)
        self.assertEqual(a["region_weights"][2], 0.0)
        new = old.replace(" alloc", " CHECK(ctx,p->size>0);\n alloc")
        a, _ = repair_region_supervision(old, new)
        self.assertEqual(a["region_weights"][1], 0.0)
        self.assertNotIn(0.5, a["region_labels"])

    def test_no_visible_positive_disables_line_loss_but_keeps_all_tokens(self):
        row = dict(
            func="int f(){\n return 1;\n}",
            target=1,
            region_labels=[-1, -1, 0],
            region_weights=[0.0, 0.0, 0.2],
        )
        f = paired_features(row, Tokens(), "localizer")
        self.assertEqual(sum(f["label_weights"]), 0.0)
        self.assertTrue(any(f["code_mask"]))
        b = paired_features(row, Tokens(), "baseline")
        self.assertEqual(f["input_ids"], b["input_ids"])
        self.assertEqual(b["labels"], 1)

    def test_prediction_features_do_not_read_diff_and_scores_share_offsets(self):
        row = dict(func="int f(){\r\n return 1;\r\n}", target=1)
        f = paired_features(row, Tokens(), "localizer")
        row.update(
            line_numbers=f["line_numbers"],
            line_scores=[0.8] * len(f["line_numbers"]),
            lines_predicted=f["line_numbers"],
        )
        a = paired_features(dict(row, lines_ground_truth=[]), Tokens(), "predicted")
        b = paired_features(dict(row, lines_ground_truth=[1, 2]), Tokens(), "predicted")
        self.assertEqual(a, b)
        for i, valid in enumerate(a["code_mask"]):
            if valid:
                self.assertEqual(a["region_scores"][i], 0.8)
        bad = dict(row, line_numbers=[99])
        with self.assertRaises(ValueError):
            paired_features(bad, Tokens(), "predicted")

    def test_score_cache_join_checks_physical_lines_and_threshold(self):
        from vulnmechanism.data import write_jsonl
        from vulnmechanism.training import paired_rows

        row = dict(sample_key="a", line_numbers=[0, 2], lines_predicted=[2])
        score = dict(sample_key="a", line_numbers=[0, 2], line_scores=[0.4, 0.8])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "primevul_paired/diff_localizer"
            root.mkdir(parents=True)
            write_jsonl(root / "valid.predicted.jsonl", [row])
            for ns, ps in (
                ([0, 2], [0.4, 0.8]),
                ([0, 1], [0.4, 0.8]),
                ([0, 2], [0.8, 0.4]),
            ):
                write_jsonl(
                    root / "valid.line_scores.jsonl",
                    [dict(score, line_numbers=ns, line_scores=ps)],
                )
                if ns == [0, 2] and ps == [0.4, 0.8]:
                    actual = paired_rows(
                        tmp, tmp, "predicted", "valid", method="diff_hard_weight"
                    )
                    self.assertEqual(actual[0]["line_scores"], ps)
                else:
                    with self.assertRaisesRegex(ValueError, "physical lines"):
                        paired_rows(
                            tmp, tmp, "predicted", "valid", method="diff_hard_weight"
                        )

    def test_ignored_line_gradients_and_real_encoder_backward(self):
        m = model(True)
        m.weak_supervision = True
        row = dict(
            func="int f(){\n return 1;\n}",
            target=1,
            region_labels=[1.0, -1.0, 0.0],
            region_weights=[1.0, 0.0, 0.2],
        )
        batch = paired_collate([paired_features(row, Tokens(), "localizer")])
        captured = []

        def hook(_, args, out):
            out.retain_grad()
            captured.append(out)

        handle = m.line_classifier.register_forward_hook(hook)
        loss, _ = paired_forward(m, batch, "localizer", "cpu")
        loss.backward()
        handle.remove()
        self.assertEqual(float(captured[0].grad[1]), 0.0)
        self.assertGreater(
            float(m.encoder.embeddings.word_embeddings.weight.grad.abs().sum()), 0.0
        )
        self.assertTrue(torch.isfinite(loss))

    def test_save_reload_robust_settings_and_forward(self):
        m = model()
        m.pooling = "robust_fusion"
        m.fusion_beta = 0.1
        m.eval()
        row = dict(func="int f(){\nreturn 1;\n}", target=1)
        e = paired_features(row, Tokens(), "baseline")
        row.update(
            line_numbers=e["line_numbers"],
            line_scores=[0.7] * len(e["line_numbers"]),
            lines_predicted=e["line_numbers"],
        )
        batch = paired_collate([paired_features(row, Tokens(), "predicted")])
        p = paired_forward(m, batch, "predicted", "cpu", with_labels=False)
        cp = dict(
            model_path="unused",
            variant="predicted",
            line_mapping="source_offsets",
            input_mode="512",
            pooling="robust_fusion",
            selected_fusion_beta=0.1,
            state_dict=m.state_dict(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.pt"
            torch.save(cp, path)
            with patch(
                "vulnmechanism.model.losver_model", return_value=(model(), Tokens())
            ):
                loaded, _ = load_paired(torch.load(path, weights_only=False), "cpu")
            q = paired_forward(loaded, batch, "predicted", "cpu", with_labels=False)
        torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_unknown_excluded_from_weighted_denominator_and_binary_bce_retained(self):
        m = model(True)
        m.weak_supervision = True
        m.eval()
        row = dict(
            func="int f(){\n return 1;\n}",
            target=1,
            region_labels=[-1, 1, -1],
            region_weights=[0.0, 0.6, 0.0],
        )
        batch = paired_collate([paired_features(row, Tokens(), "localizer")])
        loss, _ = paired_forward(m, batch, "localizer", "cpu")
        batch["label_weights"][0, 0] = 100.0
        equal, _ = paired_forward(m, batch, "localizer", "cpu")
        torch.testing.assert_close(loss, equal, rtol=0, atol=0)
        batch["label_weights"][0, 1] = 0.3
        scaled, _ = paired_forward(m, batch, "localizer", "cpu")
        torch.testing.assert_close(loss, scaled, rtol=0, atol=0)
        d = model()
        d.pooling = "robust_fusion"
        row.update(region_labels=[-1, -1, -1], region_weights=[0.0, 0.0, 0.0])
        f = paired_features(row, Tokens(), "baseline")
        f["region_scores"] = [0.0] * len(f["input_ids"])
        cls, _ = paired_forward(d, paired_collate([f]), "baseline", "cpu")
        cls.backward()
        self.assertGreater(float(d.classifier.weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(d.encoder.embeddings.word_embeddings.weight.grad.abs().sum()), 0.0
        )

    def test_cached_official_diff_produces_identical_supervision(self):
        from vulnmechanism.data import modified_lines

        a = "int f(char *p,int n){\n return p[n];\n}\n"
        b = a.replace(" return", " if(n<0)return -1;\n return")
        expected = repair_region_supervision(a, b)
        actual = repair_region_supervision(
            a, b, diffs=(modified_lines(a, b), modified_lines(b, a))
        )
        self.assertEqual(expected, actual)

    def test_new_methods_reject_l1_without_loading_data(self):
        from vulnmechanism.training import train_losver

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "512"):
                train_losver(
                    tmp, tmp, "unused", "repair_region_robust_fusion", input_mode="l1"
                )

    def test_paired_epoch_override_is_not_silently_ignored_by_codebert(self):
        from vulnmechanism.training import train

        with self.assertRaisesRegex(ValueError, "paired"):
            train("unused", "unused", "unused", "primevul", "bce", paired_epochs=4)

    def test_pilot_checkpoint_rejects_formal_test_before_model_loading(self):
        from vulnmechanism.evaluation import evaluate_paired
        from vulnmechanism.artifacts import write_json

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "primevul_paired/diff_hard_weight"
            run.mkdir(parents=True)
            torch.save(dict(train_samples=1024), run / "best-p_c.pt")
            write_json(run / "training.metrics.json", {})
            with patch("vulnmechanism.evaluation.load_paired") as loader:
                with self.assertRaisesRegex(ValueError, "pilot"):
                    evaluate_paired(
                        tmp, tmp, ["diff_hard_weight"], device="cpu", splits=("test",)
                    )
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
