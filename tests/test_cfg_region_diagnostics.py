"""Fixed H/P diagnostics: arithmetic, provenance, and no-update regressions."""
import copy
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from vulnmechanism import cfg_data as data
from vulnmechanism import cfg_region_diagnostics as diag
from vulnmechanism.cfg_alignment import region_candidates, region_targets
from vulnmechanism.cfg_dependency import RegionReconstructionHead, region_function_loss
from vulnmechanism.model import InputBuilder
from tests.test_cfg_dependency import CharTokenizer, TinyBase, TinyEncoder, record


def region(index=0, tokens=(2,), targets=None):
    return {"region": index, "nodes": list(range(len(tokens))), "tokens": list(tokens),
            "spans": [[i, i+1] for i in range(len(tokens))],
            "targets": targets if targets is not None else [[[2, .25], [3, .75]], [], [[2, 1.]], []]}


class DiagnosticTests(unittest.TestCase):
    def test_ce_entropy_kl_empty_family(self):
        p = torch.tensor([.4, .6], dtype=torch.float64)
        metrics = diag.target_metrics([[2, .25], [3, .75]], p.log())
        entropy = -.25*math.log(.25)-.75*math.log(.75)
        ce = -.25*math.log(.4)-.75*math.log(.6)
        self.assertAlmostEqual(metrics["ce"], ce)
        self.assertAlmostEqual(metrics["entropy"], entropy)
        self.assertAlmostEqual(metrics["kl"], ce-entropy)
        self.assertIsNone(diag.target_metrics([], p.log()))
        self.assertAlmostEqual(diag.target_metrics([[2, .4], [3, .6]], p.log())["kl"], 0)

    def test_hierarchical_means_and_groups(self):
        def row(key, values, operations=1):
            return {"sample_key": key, "operations": operations, "families": [
                {"family": data.FAMILIES[i], "target": [[2, 1.]],
                 "fixed_model": {"ce": v, "entropy": 0., "kl": v},
                 "constant": {"ce": v+1, "entropy": 0., "kl": v+1}}
                for i, v in enumerate(values)]}
        rows = [row("a", [2, 4]), row("a", [9], 2), row("b", [1])]
        out = diag.summarize_predictions(rows)
        self.assertEqual(out["all"]["fixed_model"]["ce"], ((3+9)/2+1)/2)
        self.assertEqual(out["api"]["fixed_model"]["ce"], ((2+9)/2+1)/2)
        self.assertEqual(out["multiple_operations"]["functions"], 1)
        self.assertEqual(out["datatype"]["fixed_model"]["ce"], 4)
        self.assertIsNone(out["multiple_categories"]["fixed_model"]["ce"])
        self.assertEqual(out, diag.summarize_predictions(list(reversed(rows))))
        self.assertAlmostEqual(sum(out[f]["objective_contribution"]["fixed_model"]["ce"]
                                   for f in data.FAMILIES), out["all"]["fixed_model"]["ce"])

    def test_train_reference_weights_and_valid_rejection(self):
        rows = [record("abcd", "a"), record("abcd", "b")]
        supervision = {"a": [region(targets=[[[2, 1.]], [[2, 1.]], [], []]),
                              region(1, targets=[[[3, 1.]], [], [], []])],
                       "b": [region(targets=[[[3, 1.]], [], [], []])]}
        ref = diag.constant_reference(rows, supervision, [4]*4)
        expected = (torch.tensor([.25, 1.5], dtype=torch.float64)/1.75 + diag.SMOOTHING)/(1+2*diag.SMOOTHING)
        torch.testing.assert_close(ref[0], expected)
        self.assertIsNone(ref[2])
        with self.assertRaisesRegex(ValueError, "train-only"):
            diag.constant_reference([record("abcd", "c", "valid")], {}, [4]*4)
        with self.assertRaisesRegex(ValueError, "non-train"):
            diag.constant_reference(rows, {**supervision, "valid": []}, [4]*4)

    def test_fixed_eval_batch_invariance_single_forward_no_update_target_exclusion(self):
        torch.manual_seed(8)
        encoder, head = TinyEncoder(), RegionReconstructionHead(4, [4]*4)
        rows = [record("abcd", str(i)) for i in range(3)]
        supervision = {"0": [region(), region(1, (2, 3))], "1": [region()], "2": []}
        ref = diag.constant_reference(rows, supervision, [4]*4)
        builder = InputBuilder(CharTokenizer(), source_max_length=2048, context_max_length=384)
        state = [p.detach().clone() for m in (encoder, head) for p in m.parameters()]
        with redirect_stdout(io.StringIO()):
            a = diag.fixed_predictions(encoder, head, builder, rows, supervision, ref, 1, torch.device("cpu"))
            self.assertEqual(encoder.calls, 2)
            b = diag.fixed_predictions(encoder, head, builder, rows, supervision, ref, 2, torch.device("cpu"))
        self.assertEqual(encoder.calls, 3)
        self.assertEqual(a, b)
        for previous, p in zip(state, (p for m in (encoder, head) for p in m.parameters())):
            self.assertTrue(torch.equal(previous, p))
            self.assertIsNone(p.grad)
        self.assertFalse(encoder.training)
        self.assertFalse(head.training)
        ids, mask = builder.sequence_batch(rows[:2], variant="baseline", excluded_groups=(), device=torch.device("cpu"))
        with torch.no_grad():
            hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            expected = region_function_loss(hidden, [supervision["0"], supervision["1"]], head)
        self.assertAlmostEqual(diag.summarize_predictions(a)["all"]["fixed_model"]["ce"], expected.item(), places=6)
        changed = copy.deepcopy(supervision)
        changed["0"][0]["targets"][0] = [[2, 1.]]
        with redirect_stdout(io.StringIO()):
            c = diag.fixed_predictions(encoder, head, builder, rows, changed, ref, 1, torch.device("cpu"))
        self.assertEqual(a[0]["families"][0]["prediction_top"], c[0]["families"][0]["prediction_top"])
        self.assertNotEqual(a[0]["families"][0]["fixed_model"]["ce"], c[0]["families"][0]["fixed_model"]["ce"])

    def test_candidates_do_not_sample(self):
        source = " ".join(["x"]*20)
        view = {"node_ids": list(map(str, range(20))), "edges": [],
                "locations": [{"code": "x", "OFFSET": i*2, "OFFSET_END": i*2+1} for i in range(20)]}
        partition = data.cfg_regions(view)
        builder = InputBuilder(CharTokenizer(), source_max_length=2048, context_max_length=384)
        with patch("random.Random.sample", side_effect=AssertionError("resampling")):
            candidates, _ = region_candidates(record(source), view, [[2]*4]*20, partition, builder)
        selected, _ = region_targets(record(source), view, [[2]*4]*20, partition, builder)
        self.assertEqual(len(candidates), 20)
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(item in candidates for item in selected))
        scope = diag._scope(candidates, 25)
        self.assertEqual(scope["zero_operation"], 5)
        self.assertEqual(scope["single_operation"], 20)

    def test_real_joern_cache_audit_checkpoint_and_test_isolation(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/joern_operand_read.json").read_text())
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            rows = [record(fixture["source"], name, split) for name, split in
                    (("train", "train"), ("valid", "valid"), ("test", "test"))]
            dataset, graphs, reference, stage = root/"data.jsonl", root/"graphs.jsonl", root/"c", root/"p1"
            reference.mkdir(); stage.mkdir()
            dataset.write_text("".join(json.dumps(r)+"\n" for r in rows))
            from vulnmechanism.cpg import JOERN_SOURCE_PREPROCESSING_VERSION
            graphs.write_text("".join(json.dumps({**data.identity(r), "graph_schema_version": data.GRAPH_SCHEMA,
                "preprocessing_version": JOERN_SOURCE_PREPROCESSING_VERSION, "preprocessing_applied": False,
                "original_source_sha256": data.source_hash(r["raw_source"]),
                "parsed_source_sha256": data.source_hash(r["raw_source"]), "graph": fixture["graph"]})+"\n" for r in rows))
            view = data.abstract_cfg(fixture["graph"])
            vocab = data.AttributeVocabulary.fit(rows[:1], {"train": view})
            c = {"dataset": str(dataset), "graphs": str(graphs), "source_dataset": "primevul", "source_max_length": 2048,
                 "seed": 42, "graph_steps": 5, "model_path": "tiny", "lora_r": 16, "lora_alpha": 32,
                 "lora_dropout": .05, "vocabulary_sha256": data.digest(vocab.values),
                 "graph_file_sha256": data.file_sha256(graphs), "cohort_sha256": data.cohort_hash(rows),
                 "train_cohort_sha256": data.cohort_hash(rows[:1]), "valid_cohort_sha256": data.cohort_hash(rows[1:2])}
            data.atomic_json(reference/"config.json", c)
            data.atomic_json(reference/"vocabulary.json", vocab.values)
            cache = root/"regions"
            audit = data.prepare_regions(reference, cache, CharTokenizer())
            config = {"c_config": c, "reference_run_dir": str(reference), "region_dir": str(cache),
                      "objective": "source_clm_plus_dependency_plus_region", "region_schema": data.REGION_SCHEMA,
                      "regions_sha256": audit["regions_sha256"], "region_targets_sha256": audit["train_targets_sha256"]}
            data.atomic_json(stage/"config.json", config)
            checkpoint_dir = stage/"region_pretrain"; checkpoint_dir.mkdir()
            encoder, head = TinyEncoder(), RegionReconstructionHead(4, vocab.sizes())
            with torch.no_grad():
                for parameter in head.parameters():
                    parameter.zero_()
            torch.save({"mode": "region_pretrain", "pretrain_config": config,
                        "adapter_state": {"adapter.weight": encoder.adapter.weight.detach().clone()},
                        "region_head_state": head.state_dict()}, checkpoint_dir/"last.pt")
            data.atomic_json(checkpoint_dir/"complete.json", {"mode": "region_pretrain",
                "checkpoint_sha256": data.file_sha256(checkpoint_dir/"last.pt")})
            # Invalid test-only source/label must never enter diagnostic validation.
            rows[2].pop("label"); rows[2].pop("raw_source")
            dataset.write_text("".join(json.dumps(r)+"\n" for r in rows))
            base = TinyBase()
            with patch.object(data, "cfg_regions", side_effect=AssertionError("repartition")), \
                 patch.object(data.AttributeVocabulary, "fit", side_effect=AssertionError("refit")), \
                 patch("random.Random.sample", side_effect=AssertionError("resample")), \
                 patch("torch.optim.AdamW", side_effect=AssertionError("optimizer created")):
                report = diag.audit_regions(stage, root/"audit", base)
                result = diag.evaluate_regions(stage, root/"evaluation", base=base)
            self.assertEqual(set(report["splits"]), {"train", "valid"})
            self.assertTrue(result["complete"])
            self.assertEqual(base.encoders[0].calls, 2)
            for family, size in zip(data.FAMILIES, vocab.sizes()):
                if result["splits"]["train"][family]["functions"]:
                    self.assertAlmostEqual(result["splits"]["train"][family]["fixed_model"]["ce"], math.log(size-2))
            torch.testing.assert_close(base.encoders[0].adapter.weight, encoder.adapter.weight)
            self.assertTrue(all(p.grad is None for p in base.encoders[0].parameters()))
            with self.assertRaises(FileExistsError):
                diag.evaluate_regions(stage, root/"evaluation", base=base)
            (checkpoint_dir/"last.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                diag.evaluate_regions(stage, root/"invalid", base=base)


if __name__ == "__main__":
    unittest.main()
