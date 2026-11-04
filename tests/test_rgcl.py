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
from vulnmechanism.training import paired_collate, paired_forward, train_losver, LOSVER_CONFIG, save_training_state
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
                rows = records(split=split)
                write_jsonl(root/"data/primevul_paired"/f"{split}.jsonl", rows)
                write_jsonl(root/"data/primevul_paired"/f"{split}.lines.jsonl", rows)
            cfg = dict(LOSVER_CONFIG, epochs=dict(LOSVER_CONFIG["epochs"], baseline=2), batch_size=2, eval_batch_size=2)
            for method in ("plm_baseline", "context_only", "rgcl"):
                result = train_losver(root/"data", root/"out", "tiny", method, dataset="primevul_paired",
                                      config=cfg, seed=42, device="cpu")
                self.assertEqual(result["selected_epochs"].keys(), {"p_c"})
                self.assertFalse(result.get("native_binary", False))
                run = root/"out/primevul_paired"/method
                cp = torch.load(run/"best-p_c.pt", weights_only=False)
                restored, _ = load_paired(cp, "cpu")
                self.assertEqual(restored.cls_baseline, method == "plm_baseline")
                self.assertEqual(restored.attention_region, "diff" if method == "rgcl" else None)
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
