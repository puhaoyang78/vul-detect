"""Native labels, leakage groups, original MIL and a genuine CLS classifier."""

import copy
import csv
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from tests.test_losver import Tokens, config
from vulnmechanism.data import load_rgcl_native, split_rgcl_groups, balanced, prepare_rgcl, paired_features, read_jsonl, write_jsonl
from vulnmechanism.model import LOSVERDetector, predict_checkpoint
from vulnmechanism.training import (paired_collate, paired_forward, train_losver, LOSVER_CONFIG,
    save_training_state, shuffle_line_labels, RGCL_REGION_TARGETS,
    restrict_region_supervision, region_supervision_statistics)
from vulnmechanism.evaluation import load_paired, validate_paired
from vulnmechanism.metrics import metrics


def factory(path, variant, **kwargs):
    args = dict(class_weight=1., window_micro_batch=8)
    args.update(kwargs)
    c = config()
    return LOSVERDetector(RobertaModel(c, add_pooling_layer=False), c, SimpleNamespace(**args)), Tokens()


def record(i, target, *, code=None):
    return dict(sample_key=f"native:{i}", pair_key=f"native:{i}", target=target,
        func=code or f"int f{i}() {{\n int x = {i};\n return x;\n}}",
        project="p", commit_id=f"commit:{i}", cves=[f"CVE-{i}"],
        repair_key=None, normal_kind="vulnerable" if target else "ordinary",
        lines_ground_truth=[1], diff_reliable=bool(target))


class RGCLTests(unittest.TestCase):
    def test_no_context_removes_only_explicit_interaction_with_same_initial_loss_and_rng(self):
        from tests.test_bounded_attention import batch
        torch.manual_seed(42)
        plain,_=factory("tiny","baseline",bounded_attention=True,attention_region="diff",attention_context=False)
        rng=torch.get_rng_state()
        torch.manual_seed(42)
        context,_=factory("tiny","baseline",bounded_attention=True,attention_region="diff",attention_context=True)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        self.assertEqual(set(context.state_dict())-set(plain.state_dict()),{"context_u.weight","context_v.weight"})
        for name,value in plain.state_dict().items():
            torch.testing.assert_close(value,context.state_dict()[name],rtol=0,atol=0)
        plain.eval();context.eval()
        a,pa,da=paired_forward(plain,batch(),"baseline","cpu",return_attention=True)
        b,pb,db=paired_forward(context,batch(),"baseline","cpu",return_attention=True)
        for left,right in ((a,b),(pa,pb),(da["region_loss"],db["region_loss"]),(da["line_rates"],db["line_rates"])):
            torch.testing.assert_close(left,right,rtol=0,atol=0)
        torch.testing.assert_close(a,da["bce"]+.1*da["region_loss"],rtol=0,atol=0)
        a.backward();b.backward()
        contextual=dict(context.named_parameters())
        for name,p in plain.named_parameters():
            if p.grad is not None:
                torch.testing.assert_close(p.grad,contextual[name].grad,rtol=1e-6,atol=1e-7)
        self.assertGreater(float(plain.line_attention.weight.grad.abs().sum()),0.)
        self.assertGreater(float(context.context_v.weight.grad.abs().sum()),0.)

    def test_side_supervision_preserves_members_inputs_labels_rng_and_token_mass(self):
        template = dict(line_labels=[1, 0, 0], line_token_indices=[[1, 2], [2, 3], [4]],
                        code_mask=[False, True, True, True, True])
        features = [dict(copy.deepcopy(template), labels=y) for y in (1, 0)]
        features += [dict(copy.deepcopy(template), labels=1, line_labels=[0, 0, 0]),
                     dict(copy.deepcopy(template), labels=0, line_labels=[1, 1, 1])]
        before = copy.deepcopy(features)
        py, th = random.getstate(), torch.get_rng_state()
        for target in (1, 0):
            actual = copy.deepcopy(features)
            stats = region_supervision_statistics(actual, target)
            restrict_region_supervision(actual, target)
            self.assertEqual(stats["functions"], 4)
            self.assertEqual(stats["eligible_functions"], 2)
            self.assertEqual(stats["supervised_functions"], 1)
            self.assertEqual(stats["supervised_region_tokens"], 1.5)
            self.assertEqual(stats["supervised_background_tokens"], 2.5)
            self.assertEqual(stats["code_tokens"], 16)
            self.assertEqual(stats["empty_region_functions"], 1)
            self.assertEqual(stats["all_region_functions"], 1)
            self.assertEqual(stats["by_target"][str(1-target)]["supervised_functions"], 0)
            for original, selected in zip(before, actual):
                for key in original:
                    expected = [0, 0, 0] if key == "line_labels" and original["labels"] != target else original[key]
                    self.assertEqual(selected[key], expected)
        self.assertEqual(features, before)
        self.assertEqual(random.getstate(), py)
        torch.testing.assert_close(torch.get_rng_state(), th, rtol=0, atol=0)

    def test_side_region_loss_is_original_eligible_function_mean_without_compensation(self):
        rates = torch.tensor([[3., 3., 3.], [3., 3., 3.], [3., 3., 3.], [3., 3., 3.]], requires_grad=True)
        counts = torch.tensor([[1., 9., 0.], [2., 3., 1.], [1., 0., 0.], [0., 0., 0.]])
        truth = torch.tensor([[1., 0., -1.], [0., 1., 0.], [1., -1., -1.], [-1., -1., -1.]])
        function_targets = torch.tensor([1, 0, 1, 0])
        for target in (0, 1):
            labels = truth.clone()
            labels[function_targets != target] = 0
            loss, number = LOSVERDetector.attention_region_loss(rates, counts, labels)
            selected = function_targets == target
            expected, _ = LOSVERDetector.attention_region_loss(rates[selected], counts[selected], truth[selected])
            torch.testing.assert_close(loss, expected, rtol=0, atol=0)
            self.assertEqual(int(number), 1)
            gradient = torch.autograd.grad(loss, rates, retain_graph=True)[0]
            self.assertEqual(float(gradient[~selected].abs().sum()), 0)
            self.assertGreater(float(gradient[selected].abs().sum()), 0)
            self.assertEqual(float(gradient[counts == 0].abs().sum()), 0)

    def test_random_region_exact_code_token_mass_cross_line_and_rng(self):
        features = [dict(line_labels=[1, 0, 0, 0, 1],
            line_token_indices=[[1, 2], [2, 3], [4, 5], [6], [0, 7]],
            code_mask=[False, True, True, True, True, True, True, False]),
            dict(line_labels=[0, 0], line_token_indices=[[1], [2]], code_mask=[False, True, True]),
            dict(line_labels=[1, 1], line_token_indices=[[1], [2]], code_mask=[False, True, True]),
            dict(line_labels=[1, 0, 0, 0], line_token_indices=[[1], [2], [3], [4]],
                 code_mask=[False, True, True, True, True])]
        before = copy.deepcopy(features)
        py, th = random.getstate(), torch.get_rng_state()
        a, b = copy.deepcopy(features), copy.deepcopy(features)
        stats = shuffle_line_labels(a, 42, match_tokens=True)
        self.assertEqual(stats, shuffle_line_labels(b, 42, match_tokens=True))
        self.assertEqual(a, b)
        self.assertEqual(features, before)
        self.assertEqual(random.getstate(), py)
        torch.testing.assert_close(torch.get_rng_state(), th, rtol=0, atol=0)
        self.assertTrue(all(r["diff_region_tokens"] == r["random_region_tokens"] for r in stats))
        self.assertEqual(stats[0]["diff_region_tokens"], 1.5)
        self.assertEqual(a[0]["line_labels"][-1], 0)
        self.assertEqual([r["eligible"] for r in stats], [True, False, False, True])
        self.assertGreater(len({tuple(self.random_labels(before[-1], seed)) for seed in range(8)}), 1)

    @staticmethod
    def random_labels(feature, seed):
        copied = copy.deepcopy(feature)
        shuffle_line_labels([copied], seed, match_tokens=True)
        return copied["line_labels"]

    def test_megavul_uses_native_members_without_synthesizing_fixed_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "megavul"
            root.mkdir()
            base = dict(repo_name="p", commit_hash="fix", file_path="a.c", func_name="f", cve_id="CVE-1")
            raw = [dict(base, is_vul=True, func_before="int f(){return 1;}", func="int f(){return 0;}"),
                   dict(base, is_vul=False, func_before=None, func="int g(){return 0;}")]
            (root / "megavul.json").write_text(json.dumps(raw))
            rows, other, stats = load_rgcl_native(tmp, "megavul")
            self.assertEqual([r["target"] for r in rows], [1, 0])
            self.assertEqual(rows[0]["func"], raw[0]["func_before"])
            self.assertEqual(rows[1]["func"], raw[1]["func"])
            self.assertEqual(rows[1]["normal_kind"], "ordinary")
            self.assertEqual(other, {"megavul:1": raw[0]["func"]})
            self.assertEqual(stats["synthesized_fixed_negatives"], 0)

    def test_native_labels_and_ordinary_negatives_survive_missing_repairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "SecVulEval/data"
            root.mkdir(parents=True)
            base = dict(project_url="p", filepath="a.c", commit_id="fix", func_name="f", cve_list=["CVE-1"])
            raw = [dict(base, idx=0, is_vulnerable=True, fixed_func_idx=1., func_body="int f(){return 1;}"),
                   dict(base, idx=1, is_vulnerable=False, fixed_func_idx=1., func_body="int f(){return 0;}"),
                   dict(base, idx=2, is_vulnerable=False, fixed_func_idx=None, func_body="int g(){return 2;}"),
                   dict(base, idx=3, is_vulnerable=True, fixed_func_idx=None, func_body="int h(){return 3;}")]
            pq.write_table(pa.Table.from_pylist(raw), root / "train-00000-of-00001.parquet")
            rows, others, stats = load_rgcl_native(tmp, "secvuleval")
            self.assertEqual([r["target"] for r in rows], [1, 0, 0, 1])
            self.assertEqual([r["normal_kind"] for r in rows], ["vulnerable", "repaired", "ordinary", "vulnerable"])
            self.assertEqual(len(others), 2)
            self.assertEqual(stats["issues"]["missing_fixed_id"], 1)
            self.assertEqual(stats["synthesized_fixed_negatives"], 0)

    def test_transitive_groups_and_whitespace_duplicates_never_cross(self):
        rows = [record(i, i % 2) for i in range(80)]
        rows[1]["cves"] = rows[0]["cves"]
        rows[2]["commit_id"] = rows[1]["commit_id"]
        rows[3]["repair_key"] = rows[2]["sample_key"]
        rows[4]["func"] = rows[3]["func"].replace("\n", "\n  ")
        original = copy.deepcopy(rows)
        summary = split_rgcl_groups(rows, {}, seed=42)
        again = copy.deepcopy(original)
        self.assertEqual(split_rgcl_groups(again, {}, seed=42), summary)
        self.assertEqual([r["split"] for r in rows], [r["split"] for r in again])
        self.assertEqual(len({r["split"] for r in rows[:5]}), 1)
        self.assertEqual([r["target"] for r in rows], [r["target"] for r in original])
        self.assertEqual(summary["members_removed"], 0)
        for field in ("commit_id", "group_key"):
            by = {}
            for r in rows:
                by.setdefault(r[field], set()).add(r["split"])
            self.assertTrue(all(len(s) == 1 for s in by.values()))

    def test_counterpart_duplicate_is_grouped_without_becoming_a_negative(self):
        rows = [record(i, i % 2) for i in range(40)]
        other = {rows[0]["sample_key"]: rows[-1]["func"]}
        split_rgcl_groups(rows, other)
        self.assertEqual(rows[0]["group_key"], rows[-1]["group_key"])
        self.assertEqual(len(rows), 40)

    def test_balanced_native_subset_is_sampled_after_full_group_split(self):
        rows = [record(i, int(i % 5 == 0)) for i in range(200)]
        # An excluded negative can connect two positives before downsampling.
        rows[1]["cves"] = rows[0]["cves"]
        rows[5]["commit_id"] = rows[1]["commit_id"]
        split_rgcl_groups(rows, {})
        before = copy.deepcopy(rows)
        rng = random.getstate()
        selected = balanced(rows, seed=42, label_key="target")
        self.assertEqual(random.getstate(), rng)
        self.assertEqual(selected, balanced(rows, seed=42, label_key="target"))
        self.assertEqual(rows, before)
        self.assertEqual({r["sample_key"] for r in selected if r["target"]},
                         {r["sample_key"] for r in rows if r["target"]})
        self.assertEqual(rows[0]["group_key"], rows[5]["group_key"])
        for split in ("train", "valid", "test"):
            targets = [r["target"] for r in selected if r["split"] == split]
            self.assertGreater(sum(targets), 0)
            self.assertEqual(sum(targets), len(targets) - sum(targets))
        by_group = {}
        for r in selected:
            by_group.setdefault(r["group_key"], set()).add(r["split"])
        self.assertTrue(all(len(s) == 1 for s in by_group.values()))
        self.assertEqual(len({r["sample_key"] for r in selected}), len(selected))

    def test_megavul_preparation_persists_full_groups_and_selected_members(self):
        native = [record(i, int(i % 5 == 0)) for i in range(100)]
        for r in native:
            del r["lines_ground_truth"], r["diff_reliable"]
        expected = copy.deepcopy(native)
        split_rgcl_groups(expected, {})
        selected = balanced(expected, seed=42, label_key="target")
        with tempfile.TemporaryDirectory() as tmp:
            with patch("vulnmechanism.data.load_rgcl_native", return_value=(native, {}, {"raw_functions": 100})), \
                 patch("transformers.RobertaTokenizerFast.from_pretrained", return_value=Tokens()):
                stats = prepare_rgcl("unused", tmp, "megavul", "tiny")
            dest = Path(tmp) / "megavul"
            actual = [r for s in ("train", "valid", "test") for r in read_jsonl(dest / f"{s}.jsonl")]
            self.assertEqual({r["sample_key"] for r in actual}, {r["sample_key"] for r in selected})
            self.assertTrue(all("diff_reliable" not in r and "lines_ground_truth" not in r for r in actual))
            with (dest / "grouped_members.csv").open() as f:
                grouped = list(csv.DictReader(f))
            self.assertEqual(len(grouped), 100)
            self.assertEqual(sum(int(r["selected"]) for r in grouped), 40)
            self.assertTrue(all(r["selected"] == "1" for r in grouped if r["target"] == "1"))
            self.assertEqual(stats["negative_sampling"]["positive_members_removed"], 0)
            self.assertEqual(stats["classification_positive_weight"], 1.)
            for split, d in stats["splits"].items():
                self.assertEqual(d["class_counts"]["1"], d["class_counts"]["0"])
                self.assertEqual([r["sample_key"] for r in read_jsonl(dest / f"{split}.jsonl")],
                                 [r["sample_key"] for r in selected if r["split"] == split])

    def test_unreliable_and_truncated_regions_keep_original_classification(self):
        t = Tokens()
        missing = record(0, 1)
        missing["diff_reliable"] = False
        f = paired_features(missing, t, "baseline")
        self.assertEqual(f["labels"], 1)
        self.assertFalse(any(f["line_labels"]))
        partial = record(1, 1, code="int f(){\n" + "x" * 600 + ";\nreturn 0;\n}")
        f = paired_features(partial, t, "baseline")
        self.assertIn(1, f["line_numbers"])
        self.assertFalse(any(f["line_labels"]))
        self.assertEqual(f["labels"], 1)

    def test_cls_head_and_context_initialization_rng_gradient_restore(self):
        rows = [record(0, 1), record(1, 0), record(2, 0)]
        batch = paired_collate([paired_features(r, Tokens(), "baseline") for r in rows])
        initial, sample_rng = None, None
        for method in ("plm_baseline", "context_only", "rgcl"):
            torch.manual_seed(123456)
            model, _ = factory(None, "baseline", cls_baseline=method == "plm_baseline", class_weight=2.,
                bounded_attention=method != "plm_baseline", attention_context=method != "plm_baseline",
                attention_region="diff" if method == "rgcl" else None)
            model.native_binary = True
            model.eval()
            rng = torch.get_rng_state()
            if sample_rng is None:
                sample_rng = rng
            else:
                torch.testing.assert_close(rng, sample_rng, rtol=0, atol=0)
            loss, p = paired_forward(model, batch, "baseline", "cpu")
            if method == "plm_baseline":
                self.assertIsInstance(model.classifier, RobertaClassificationHead)
                self.assertEqual(model.classifier.out_proj.out_features, 1)
            elif method == "context_only":
                initial = p.detach()
            else:
                torch.testing.assert_close(p, initial, rtol=0, atol=0)
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.classifier.parameters()), 0)
            cp = dict(protocol="rgcl_native_binary", model_path="tiny", variant="baseline", line_mapping="source_offsets", input_mode="512",
                native_binary=True, cls_baseline=method == "plm_baseline", pooling=model.pooling,
                training_config=dict(class_weight=2.), state_dict=model.state_dict())
            if method != "plm_baseline":
                cp["bounded_attention"] = dict(k=5, context={"dimension": 32}, region_supervision=model.attention_region)
            with patch("vulnmechanism.model.losver_model", side_effect=factory):
                restored, _ = load_paired(cp, torch.device("cpu"))
                with tempfile.TemporaryDirectory() as tmp:
                    filename = Path(tmp) / "model.pt"
                    torch.save(cp, filename)
                    single = predict_checkpoint(filename, [dict(raw_source=rows[0]["func"])], device="cpu")
            torch.testing.assert_close(single.sigmoid(), p[:1, 0], rtol=1e-5, atol=1e-6)
            q = paired_forward(restored, batch, "baseline", "cpu", with_labels=False)
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            value, pred = validate_paired(restored, rows, Tokens(), "baseline", "cpu", loader=[batch])
            self.assertEqual(value["samples"], 3)
            self.assertEqual(len(pred), 3)
            self.assertNotIn("pairs", value)
            self.assertEqual(value["comparison"], ">")

    def test_threshold_is_fixed_and_strict(self):
        result = metrics([0, 1], [.5, .5], strict=True)
        self.assertEqual(result["tp"], 0)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(metrics([0, 1], [.5, .5])["tp"], 1)

    def test_primevul_uses_same_cls_head_and_selects_only_full_valid_pc(self):
        from tests.test_insertion_localization import records
        from vulnmechanism.training import train
        from vulnmechanism.evaluation import evaluate
        from vulnmechanism.cli import build_parser
        with tempfile.TemporaryDirectory() as tmp, patch("vulnmechanism.model.losver_model", side_effect=factory):
            root = Path(tmp)
            for split in ("train", "valid"):
                rows = records(pairs=[("int f(){\n use(x);\n return x;\n}",
                                       "int f(){\n checked_use(x);\n return x;\n}")], split=split)
                write_jsonl(root/"data/primevul_paired"/f"{split}.jsonl", rows)
                write_jsonl(root/"data/primevul_paired"/f"{split}.lines.jsonl", rows)
            cfg = dict(LOSVER_CONFIG, epochs=dict(LOSVER_CONFIG["epochs"], baseline=2), batch_size=2, eval_batch_size=2)
            for method in ("plm_baseline", "context_only", "rgcl", "random_region", *RGCL_REGION_TARGETS, "rgcl_no_context"):
                result = train_losver(root/"data", root/"out", "tiny", method, dataset="primevul_paired",
                                      config=cfg, seed=42, device="cpu")
                self.assertEqual(result["selected_epochs"].keys(), {"p_c"})
                self.assertFalse(result.get("native_binary", False))
                run = root/"out/primevul_paired"/method
                cp = torch.load(run/"best-p_c.pt", weights_only=False)
                restored, _ = load_paired(cp, "cpu")
                self.assertEqual(restored.cls_baseline, method == "plm_baseline")
                self.assertEqual(restored.attention_region, "random" if method == "random_region" else "diff" if method in ("rgcl", *RGCL_REGION_TARGETS, "rgcl_no_context") else None)
                self.assertEqual(restored.attention_context,method not in ("plm_baseline","rgcl_no_context"))
                if method in RGCL_REGION_TARGETS:
                    self.assertEqual(cp["bounded_attention"]["region_target"], RGCL_REGION_TARGETS[method])
                    stats = result["bounded_attention"]["side_supervision"]
                    self.assertEqual(stats["supervised_functions"], 1)
                    self.assertEqual(result["region_function_visits"], 2)
                    masks = read_jsonl(run/"train.attention_regions.jsonl")
                    self.assertEqual([any(r["labels"]) for r in masks],
                                     [r["target"] == RGCL_REGION_TARGETS[method] for r in rows])
                if method == "random_region":
                    self.assertEqual(result["bounded_attention"]["randomization"]["exact_token_mass_matches"], len(rows))
                evaluate(root/"data", root/"out", "primevul_paired", [method], device="cpu", splits=("valid",))
                value = json.loads((run/"valid.metrics.json").read_text())["metrics"]
                self.assertEqual(value["p_c"], cp["validation"]["p_c"])
            with patch("vulnmechanism.training.train_losver", return_value={}) as shared:
                train(root/"data", root/"out", "tiny", "primevul_paired", "plm_baseline", device="cpu")
                self.assertEqual(shared.call_args.args[3], "plm_baseline")
            args = build_parser().parse_args(["rgcl-study", "run", "--seed", "1234", "--methods", "rgcl"])
            self.assertEqual(args.seed, 1234)
            self.assertEqual(args.methods, ["rgcl"])

    def test_study_reuses_original_paths_without_retraining_or_rewriting(self):
        from vulnmechanism import rgcl_study as study
        self.assertEqual(study.SEEDS, (42, 1234, 123456))
        self.assertNotIn(123457, study.SEEDS)
        self.assertEqual(study.run_path("unused", "primevul_paired", 123456, "rgcl"),
                         Path("results/attention_context_512/context_diff/primevul_paired/unixcoder"))
        self.assertEqual(study.run_path("out", "primevul_paired", 123456, "random_region"),
                         Path("out/seed123456/primevul_paired/random_region"))
        self.assertEqual(study.run_path("out","primevul_paired",123456,"rgcl_no_context"),
                         Path("results/bounded_attention_512/diff_region/primevul_paired/unixcoder"))
        self.assertEqual(study.run_path("out","primevul_paired",42,"rgcl_no_context",reference_dir="history"),
                         Path("out/seed42/primevul_paired/rgcl_no_context"))
        self.assertEqual(study.run_path("out", "primevul_paired", 42, "rgcl", reference_dir="history"),
                         Path("history/seed42/primevul_paired/rgcl"))
        for method in RGCL_REGION_TARGETS:
            self.assertEqual(study.run_path("out", "primevul_paired", 123456, method, reference_dir="history"),
                             Path(f"out/seed123456/primevul_paired/{method}"))
            with self.assertRaises(ValueError):
                study.study_methods("codebert", [method])
        self.assertEqual(study.study_methods("unixcoder"), ("plm_baseline", "context_only", "rgcl"))
        with self.assertRaises(ValueError):
            study.study_methods("codebert", ["random_region"])
        with self.assertRaises(ValueError):
            study.study_methods("codebert", ["rgcl_no_context"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root/"out"; historical = root/"historical"
            historical.mkdir(); out.mkdir()
            for name in ("test.metrics.json", "test.predictions.jsonl"):
                (historical/name).write_text("retained")
            (out/"gpu-smoke.json").write_text('{"passed":true}')
            before = {p.name:p.read_bytes() for p in historical.iterdir()}
            with patch.object(study, "run_path", return_value=historical), patch.object(study, "inspect_run"), \
                 patch.object(study, "train_losver") as fit:
                study.run(root, out, "tiny", "primevul_paired", 123456, ["rgcl"], device="cpu")
                fit.assert_not_called()
            self.assertEqual({p.name:p.read_bytes() for p in historical.iterdir()}, before)

    def test_no_context_route_matches_legacy_and_resumes_exactly(self):
        from tests.test_insertion_localization import records
        with tempfile.TemporaryDirectory() as tmp,patch("vulnmechanism.model.losver_model",side_effect=factory):
            root=Path(tmp)
            for split in ("train","valid"):
                rows=records(pairs=[("int f(){\n use(x);\n return x;\n}",
                                     "int f(){\n checked_use(x);\n return x;\n}")],split=split)
                write_jsonl(root/"data/primevul_paired"/f"{split}.lines.jsonl",rows)
            cfg=dict(LOSVER_CONFIG,epochs=dict(LOSVER_CONFIG["epochs"],baseline=2),batch_size=1,eval_batch_size=2)
            options=dict(config=cfg,seed=42,device="cpu",save_every=1)
            train_losver(root/"data",root/"legacy","tiny","unixcoder",bounded_attention=True,attention_region="diff",**options)
            train_losver(root/"data",root/"direct","tiny","rgcl_no_context",**options)
            def interrupt(*args,**kwargs):
                save_training_state(*args,**kwargs)
                if args[4]["global_step"]==1:raise RuntimeError("no-context interruption")
            with patch("vulnmechanism.training.save_training_state",side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,"no-context interruption"):
                    train_losver(root/"data",root/"resumed","tiny","rgcl_no_context",**options)
            train_losver(root/"data",root/"resumed","tiny","rgcl_no_context",resume=True,**options)
            reference=torch.load(root/"legacy/primevul_paired/unixcoder/last.pt",weights_only=False)
            for path in (root/"direct",root/"resumed"):
                actual=torch.load(path/"primevul_paired/rgcl_no_context/last.pt",weights_only=False)
                self.assertFalse(any(k.startswith(("context_u.","context_v.")) for k in actual["model"]))
                for key,value in reference["model"].items():
                    torch.testing.assert_close(value,actual["model"][key],rtol=0,atol=0)
                torch.testing.assert_close(reference["rng"]["torch"],actual["rng"]["torch"],rtol=0,atol=0)
                for key,state in reference["optimizer"]["state"].items():
                    for field,value in state.items():
                        other=actual["optimizer"]["state"][key][field]
                        if isinstance(value,torch.Tensor):torch.testing.assert_close(value,other,rtol=0,atol=0)
                        else:self.assertEqual(value,other)

    def test_context_ablation_report_retains_gains_and_losses_and_shared_supervision(self):
        from vulnmechanism import rgcl_study as study
        from vulnmechanism.metrics import paired_metrics
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);entries=[]
            rows=[dict(sample_key=f"s:{i}",pair_key="complete:pair:key",target=1-i) for i in range(2)]
            for split in ("valid","test"):write_jsonl(root/"data/primevul_paired"/f"{split}.jsonl",rows)
            for seed in study.SEEDS:
                for method in ("rgcl_no_context","rgcl"):
                    scores=([.8,.7] if (seed==42 and method=="rgcl_no_context") or
                            (seed==1234 and method=="rgcl") else [.8,.5])
                    value=paired_metrics(rows,scores)
                    path=root/f"{seed}-{method}";path.mkdir()
                    entries.append(dict(dataset="primevul_paired",seed=seed,method=method,run=str(path),
                                        validation=value,selected_epoch=1,reused=True))
                    pred=[dict(r,probability=p,prediction=int(p>.5)) for r,p in zip(rows,scores)]
                    write_jsonl(path/"valid.best-p_c.predictions.jsonl",pred)
                    write_jsonl(path/"test.predictions.jsonl",pred)
                    (path/"test.metrics.json").write_text(json.dumps(dict(selected_epoch=1,metrics=value)))
                    cfg=dict(training_config={"seed":seed},training_members=[r["sample_key"] for r in rows],
                        validation_members=[r["sample_key"] for r in rows],input_mode="512",line_mapping="source_offsets",
                        model_path="tiny",function_micro_batch=8,gradient_accumulation=1,input_coverage={},
                        parameters=1000+(49152 if method=="rgcl" else 0))
                    (path/"training.metrics.json").write_text(json.dumps(cfg))
                    write_jsonl(path/"train.attention_regions.jsonl",[dict(sample_key=r["sample_key"],labels=[1,0]) for r in rows])
                    torch.save(dict(rng=dict(torch=torch.arange(5,dtype=torch.uint8),cuda=[])),path/"last.pt")
            with patch.object(study,"inventory",return_value=dict(missing=0,experiments=entries)):
                result=study.report(root/"data",root,"tiny",datasets=("primevul_paired",),methods=("rgcl_no_context","rgcl"))
            changes=[r for r in result["context_pair_changes"] if r["split"]=="test"]
            self.assertEqual([(r["gains"],r["losses"],r["net"]) for r in changes],[(1,0,1),(0,1,-1),(0,0,0)])
            self.assertTrue(all(r["cpu_sampler_rng_equal"] and r["train_diff_masks_equal"] for r in result["context_interaction_checks"]))
            self.assertIn("h_i 本身仍是 UniXcoder",(root/"report.md").read_text())
            corrupted=root/"42-rgcl/test.predictions.jsonl"
            bad=read_jsonl(corrupted);bad[0]["pair_key"]="different:pair:key";write_jsonl(corrupted,bad)
            with patch.object(study,"inventory",return_value=dict(missing=0,experiments=entries)):
                with self.assertRaisesRegex(ValueError,"members/labels differ"):
                    study.report(root/"data",root,"tiny",datasets=("primevul_paired",),methods=("rgcl_no_context","rgcl"))

    def test_side_training_exact_resume_preserves_supervision_visits(self):
        from tests.test_insertion_localization import records
        with tempfile.TemporaryDirectory() as tmp, patch("vulnmechanism.model.losver_model", side_effect=factory):
            root = Path(tmp)
            for split in ("train", "valid"):
                rows = records(pairs=[("int f(){\n use(x);\n return x;\n}",
                                       "int f(){\n checked_use(x);\n return x;\n}")], split=split)
                write_jsonl(root/"data/primevul_paired"/f"{split}.lines.jsonl", rows)
            cfg = dict(LOSVER_CONFIG, epochs=dict(LOSVER_CONFIG["epochs"], baseline=2), batch_size=1, eval_batch_size=2)
            for method in RGCL_REGION_TARGETS:
                clean = train_losver(root/"data", root/"clean", "tiny", method, config=cfg, seed=42, device="cpu")
                def interrupt(*args, **kwargs):
                    save_training_state(*args, **kwargs)
                    if args[4]["global_step"] == 1:
                        raise RuntimeError("side interruption probe")
                with patch("vulnmechanism.training.save_training_state", side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, "side interruption probe"):
                        train_losver(root/"data", root/"resumed", "tiny", method, config=cfg, seed=42, device="cpu", save_every=1)
                recovered = train_losver(root/"data", root/"resumed", "tiny", method, config=cfg, seed=42,
                                          device="cpu", save_every=1, resume=True)
                self.assertEqual(clean["selected_epochs"], recovered["selected_epochs"])
                self.assertEqual(clean["region_function_visits"], recovered["region_function_visits"])
                a = torch.load(root/f"clean/primevul_paired/{method}/last.pt", weights_only=False)
                b = torch.load(root/f"resumed/primevul_paired/{method}/last.pt", weights_only=False)
                for key, value in a["model"].items():
                    torch.testing.assert_close(value, b["model"][key], rtol=0, atol=0)
                torch.testing.assert_close(a["rng"]["torch"], b["rng"]["torch"], rtol=0, atol=0)

    def test_study_report_three_seed_mean_sample_std_and_fixed_threshold(self):
        import statistics
        from vulnmechanism import rgcl_study as study
        from vulnmechanism.metrics import paired_metrics
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); entries=[]
            for dataset in study.DATASETS:
                rows=[dict(sample_key=f"{dataset}:{i}",pair_key="pair",target=1-i) for i in range(2)]
                for split in ("valid","test"):
                    write_jsonl(study.data_root(root/"data",dataset)/dataset/f"{split}.jsonl",rows)
                for seed in study.SEEDS:
                    scores={42:[.8,.2],1234:[.5,.2],123456:[.8,.7]}[seed]
                    value=paired_metrics(rows,scores) if dataset=="primevul_paired" else metrics([1,0],scores,strict=True)
                    for method in ("plm_baseline","context_only","rgcl"):
                        run=root/f"{dataset}-{seed}-{method}";run.mkdir()
                        entry=dict(dataset=dataset,seed=seed,method=method,run=str(run),validation=value,selected_epoch=1,reused=False)
                        entries.append(entry)
                        pred=[dict(r,probability=p) for r,p in zip(rows,scores)]
                        write_jsonl(run/("valid.best-p_c.predictions.jsonl" if dataset=="primevul_paired" else "valid.predictions.jsonl"),pred)
                        write_jsonl(run/"test.predictions.jsonl",pred)
                        (run/"test.metrics.json").write_text(json.dumps(dict(selected_epoch=1,metrics=value)))
            with patch.object(study,"inventory",return_value=dict(missing=0,experiments=entries)):
                result=study.report(root/"data",root,"tiny")
            row=next(r for r in result["aggregate"] if r["dataset"]=="primevul_paired" and r["split"]=="valid" and r["method"]=="rgcl")
            self.assertAlmostEqual(row["metrics"]["p_c"]["mean"],1/3)
            self.assertAlmostEqual(row["metrics"]["p_c"]["std"],statistics.stdev([1.,0.,0.]))
            self.assertEqual(len(result["records"]),36)

    def test_random_region_report_reuses_controls_and_summarizes_test(self):
        from vulnmechanism import rgcl_study as study
        from vulnmechanism.metrics import paired_metrics

        methods = ("context_only", "random_region", "rgcl")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(sample_key=f"s{i}", pair_key="p", target=1-i) for i in range(2)]
            for split in ("valid", "test"):
                write_jsonl(root/"data/primevul_paired"/f"{split}.jsonl", rows)
            entries = []
            for seed in study.SEEDS:
                for method in methods:
                    run = root/f"{seed}-{method}"
                    run.mkdir()
                    scores = [.8, .7] if method == "random_region" else [.8, .2]
                    value = paired_metrics(rows, scores)
                    pred = [dict(r, probability=p) for r, p in zip(rows, scores)]
                    write_jsonl(run/"valid.best-p_c.predictions.jsonl", pred)
                    write_jsonl(run/"test.predictions.jsonl", pred)
                    (run/"test.metrics.json").write_text(json.dumps(dict(selected_epoch=1, metrics=value)))
                    entries.append(dict(dataset="primevul_paired", seed=seed, method=method, run=str(run),
                        validation=value, selected_epoch=1, reused=method != "random_region"))
                    if method == "random_region":
                        stats = dict(seed=seed, functions=2, eligible_functions=2, exact_token_mass_matches=2,
                            unchanged_eligible_functions=0, positive_tokens=4., overlap_tokens=1.)
                        (run/"training.metrics.json").write_text(json.dumps(dict(training_members=["s0", "s1"],
                            bounded_attention=dict(randomization=stats))))
                        write_jsonl(run/"train.random_regions.jsonl", [dict(sample_key=r["sample_key"],
                            diff_region_tokens=2., random_region_tokens=2., eligible=True) for r in rows])
            with patch.object(study, "inventory", return_value=dict(missing=0, experiments=entries)) as inspected:
                result = study.report(root/"data", root/"out", "tiny", datasets=("primevul_paired",),
                                      methods=methods, reference_dir="history")
            self.assertEqual(inspected.call_args.kwargs["reference_dir"], "history")
            self.assertEqual(len(result["records"]), 18)
            self.assertEqual(len(result["aggregate"]), 6)
            self.assertEqual(len(result["random_supervision"]), 3)
            self.assertEqual(result["random_supervision"][0]["token_overlap_fraction"], .25)
            delta = next(r for r in result["deltas"] if r["split"] == "test" and r["before"] == "random_region" and r["after"] == "rgcl")
            self.assertEqual(delta["metrics"]["p_c"], 1.)

    def test_side_report_checks_original_masks_counts_and_four_method_metrics(self):
        from tests.test_insertion_localization import records
        from vulnmechanism import rgcl_study as study
        from vulnmechanism.metrics import paired_metrics

        methods = ("context_only", "vulnerable_only", "repaired_only", "rgcl")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = records(pairs=[("int f(){\n use(x);\n return x;\n}",
                                   "int f(){\n checked_use(x);\n return x;\n}")])
            for name in ("train.lines", "valid", "test"):
                write_jsonl(root/"data/primevul_paired"/f"{name}.jsonl", rows)
            features = [paired_features(row, Tokens(), "baseline") for row in rows]
            entries = []
            for seed in study.SEEDS:
                for method in methods:
                    run = root/f"{seed}-{method}"
                    run.mkdir()
                    scores = [.8, .2] if method == "rgcl" else [.8, .7]
                    value = paired_metrics(rows, scores)
                    pred = [dict(sample_key=row["sample_key"], pair_key=row["pair_key"], target=row["target"], probability=p)
                            for row, p in zip(rows, scores)]
                    write_jsonl(run/"valid.best-p_c.predictions.jsonl", pred)
                    write_jsonl(run/"test.predictions.jsonl", pred)
                    (run/"test.metrics.json").write_text(json.dumps(dict(selected_epoch=1, metrics=value)))
                    config = {}
                    if method in RGCL_REGION_TARGETS:
                        target = RGCL_REGION_TARGETS[method]
                        stats = region_supervision_statistics(features, target)
                        config = dict(bounded_attention=dict(side_supervision=stats),
                                      region_function_visits=16*stats["supervised_functions"])
                        masks = copy.deepcopy(features)
                        restrict_region_supervision(masks, target)
                        write_jsonl(run/"train.attention_regions.jsonl", [dict(sample_key=row["sample_key"],
                            line_numbers=f["line_numbers"], labels=f["line_labels"]) for row, f in zip(rows, masks)])
                    (run/"training.metrics.json").write_text(json.dumps(config))
                    entries.append(dict(dataset="primevul_paired", seed=seed, method=method, run=str(run),
                        validation=value, selected_epoch=1, reused=method in ("context_only", "rgcl")))
            with patch.object(study, "inventory", return_value=dict(missing=0, experiments=entries)), \
                 patch("vulnmechanism.model.RobertaTokenizerFast.from_pretrained", return_value=Tokens()):
                result = study.report(root/"data", root/"out", "tiny", datasets=("primevul_paired",), methods=methods)
            self.assertEqual(len(result["records"]), 24)
            self.assertEqual(len(result["aggregate"]), 8)
            self.assertEqual([r["supervised_functions"] for r in result["side_supervision"][:4]], [0, 1, 1, 2])
            self.assertEqual([r["actual_function_visits"] for r in result["side_supervision"][:4]], [None, 16, 16, None])
            self.assertEqual(len(result["deltas"]), 10)
            self.assertEqual(result["original_train_coverage"]["supervised_functions"], 2)

    def test_native_training_uses_mcc_and_shared_original_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data/secvuleval"
            train = [record(i, i % 2) for i in range(6)]
            valid = [record(20 + i, i % 2) for i in range(3)]
            write_jsonl(data / "train.lines.jsonl", train)
            write_jsonl(data / "valid.jsonl", [{k: v for k, v in r.items() if k not in ("lines_ground_truth", "diff_reliable")} for r in valid])
            (data / "statistics.json").write_text(json.dumps(dict(classification_positive_weight=1.)))
            cfg = dict(LOSVER_CONFIG, epochs=dict(LOSVER_CONFIG["epochs"], baseline=2), batch_size=2, eval_batch_size=2)
            with patch("vulnmechanism.model.losver_model", side_effect=factory):
                result = train_losver(root / "data", root / "out", "tiny", "rgcl", dataset="secvuleval",
                                      config=cfg, device="cpu", save_every=100)
            self.assertEqual(result["optimizer_steps"], 6)
            self.assertEqual(set(result["selected_epochs"]), {"mcc"})
            run = root / "out/secvuleval/rgcl"
            chosen = json.loads((run / "best-mcc.json").read_text())
            self.assertIn(chosen["epoch"], (1, 2))
            self.assertEqual(result["training_members"], [r["sample_key"] for r in train])
            self.assertEqual(result["validation_members"], [r["sample_key"] for r in valid])
            def interrupt(*args, **kwargs):
                save_training_state(*args, **kwargs)
                if args[4]["global_step"] == 2:
                    raise RuntimeError("native interruption probe")
            with patch("vulnmechanism.model.losver_model", side_effect=factory):
                with patch("vulnmechanism.training.save_training_state", side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, "native interruption probe"):
                        train_losver(root / "data", root / "resumed", "tiny", "rgcl", dataset="secvuleval",
                                     config=cfg, device="cpu", save_every=2)
                recovered = train_losver(root / "data", root / "resumed", "tiny", "rgcl", dataset="secvuleval",
                                          config=cfg, device="cpu", save_every=2, resume=True)
            self.assertEqual(result["selected_epochs"], recovered["selected_epochs"])
            clean = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            resumed = torch.load(root / "resumed/secvuleval/rgcl/last.pt", map_location="cpu", weights_only=False)
            for key, value in clean["model"].items():
                torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)
            torch.testing.assert_close(clean["rng"]["torch"], resumed["rng"]["torch"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
