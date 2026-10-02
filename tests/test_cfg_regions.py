"""H/P CPU regressions: shared C math, source-only targets, and existing run flow."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from vulnmechanism import cfg_data as data, cfg_experiment as exp
from vulnmechanism.cfg_alignment import region_targets
from vulnmechanism.cfg_dependency import (DirectedRelationHead, RegionReconstructionHead,
    region_function_loss, relation_function_loss, accumulation_window_loss,
    accumulation_window_counts, pretrain_causal_dependency, _checked_region_rows)
from vulnmechanism.cfg_network import AttributeCFGEncoder, collate_graphs
from vulnmechanism.model import InputBuilder
from tests.test_cfg_dependency import CharTokenizer, TinyBase, TinyEncoder, record


def view(n, edges):
    return {"node_ids": [str(i) for i in range(n)], "edges": edges}


def target(tokens=(2,), targets=None):
    return {"tokens": list(tokens), "targets": targets or [[[2, .25], [3, .75]], [], [[2, 1.]], []]}


class RegionTests(unittest.TestCase):
    def test_partition_edges_branches_cycles_and_permutation(self):
        v = view(12, [(0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5),
                      (5, 6), (6, 7), (7, 6), (7, 8), (9, 10), (10, 9), (11, 11)])
        r = data.cfg_regions(v)
        self.assertEqual(r["members"], [[0, 1, 2], [3], [4], [5], [6], [7], [8], [9], [10], [11]])
        self.assertEqual(sorted(n for m in r["members"] for n in m), list(range(12)))
        for (a, b), mapping in zip(v["edges"], r["edge_map"]):
            ra, rb = r["node_to_region"][a], r["node_to_region"][b]
            if ra == rb:
                self.assertEqual(mapping, {"internal_region": ra})
            else:
                self.assertEqual(r["edges"][mapping["region_edge"]], [ra, rb])
        permutation = [6, 2, 0, 11, 5, 9, 3, 8, 1, 7, 10, 4]
        inverse = {old: new for new, old in enumerate(permutation)}
        changed = view(12, [(inverse[a], inverse[b]) for a, b in reversed(v["edges"])])
        rr = data.cfg_regions(changed)
        self.assertEqual({frozenset(m) for m in r["members"]},
                         {frozenset(permutation[n] for n in m) for m in rr["members"]})
        self.assertEqual(data.cfg_regions(view(3, [(0, 1), (1, 2), (2, 0)]))["members"], [[0], [1], [2]])
        self.assertEqual(data.cfg_regions(view(1, []))["members"], [[0]])

    def test_singletons_equal_five_step_c_outputs_gradients_and_initialization(self):
        v = view(4, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 3)])
        partition = data.cfg_regions(v)
        self.assertTrue(all(len(m) == 1 for m in partition["members"]))
        batch = collate_graphs([v, v], [[[2, 3, 2, 0]]*4]*2, regions=[partition]*2)
        torch.manual_seed(42)
        c = AttributeCFGEncoder([4]*4, hidden_size=8, steps=5).double()
        after_c = torch.rand(4)
        torch.manual_seed(42)
        h = AttributeCFGEncoder([4]*4, hidden_size=8, steps=5, mode="cfg_hierarchical").double()
        self.assertTrue(torch.equal(after_c, torch.rand(4)))
        self.assertEqual(c.state_dict().keys(), h.state_dict().keys())
        self.assertEqual(sum(p.numel() for p in c.parameters()), sum(p.numel() for p in h.parameters()))
        for k in c.state_dict():
            torch.testing.assert_close(c.state_dict()[k], h.state_dict()[k], rtol=0, atol=0)
        torch.testing.assert_close(c(batch), h(batch), atol=1e-12, rtol=1e-12)
        c(batch).square().sum().backward()
        h(batch).square().sum().backward()
        for a, b in zip(c.parameters(), h.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-12, rtol=1e-12)
        with self.assertRaises(ValueError):
            AttributeCFGEncoder([4]*4, steps=4, mode="cfg_hierarchical")

    def test_multinode_math_and_numbering_invariance(self):
        v = view(5, [(0, 1), (1, 2), (2, 3), (2, 4)])
        values = [[i % 3]*4 for i in range(5)]
        r = data.cfg_regions(v)
        batch = collate_graphs([v], [values], regions=[r])
        torch.manual_seed(42)
        h = AttributeCFGEncoder([3]*4, hidden_size=8, mode="cfg_hierarchical")
        output = h(batch)
        permutation = [4, 2, 0, 1, 3]
        inv = {old: new for new, old in enumerate(permutation)}
        vv = view(5, [(inv[a], inv[b]) for a, b in v["edges"]])
        bb = collate_graphs([vv], [[values[i] for i in permutation]], regions=[data.cfg_regions(vv)])
        torch.testing.assert_close(output, h(bb))
        output.square().sum().backward()
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in h.parameters()))

    def test_targets_visibility_duplicates_empty_unk_and_hand_counts(self):
        source = "aa bb cc dd ee"
        spans = [(0, 2), (0, 2), (3, 5), (6, 8), (9, 11), (12, 14)]
        v = view(6, [(i, i+1) for i in range(5)])
        v["locations"] = [{"code": source[a:b], "OFFSET": a, "OFFSET_END": b} for a, b in spans]
        encoded = [[2, 0, 1, 0], [2, 0, 1, 0], [3, 2, 0, 0],
                   [3, 0, 2, 0], [0, 0, 0, 0], [2, 2, 3, 2]]
        builder = InputBuilder(CharTokenizer(), source_max_length=10, context_max_length=384)
        result, counts = region_targets(record(source), v, encoded, data.cfg_regions(v), builder)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["nodes"], [0, 2, 3])
        self.assertEqual(result[0]["targets"], [[[2, 1/3], [3, 2/3]], [[2, 1.]], [[2, 1.]], []])
        self.assertEqual(counts["duplicate_operations"], 1)
        self.assertEqual(counts["target_excluded_outside_visible_source"], 1)
        self.assertEqual(counts["api_nonconstant_regions"], 1)
        self.assertEqual(len(result[0]["tokens"]), 3)
        # Nested operations and two operations ending in the same token cannot count twice.
        v["locations"][2] = {"code": source[0:5], "OFFSET": 0, "OFFSET_END": 5}
        nested, counts = region_targets(record(source), v, encoded, data.cfg_regions(v), builder)
        self.assertEqual(nested[0]["nodes"], [2, 3])
        self.assertEqual(counts["overlapping_operations"], 1)
        with self.assertRaises(ValueError):
            region_targets(record(source, split="test"), v, encoded, data.cfg_regions(v), builder)

    def test_shared_token_and_cross_region_duplicate_exclusion(self):
        class PairTokenizer(CharTokenizer):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                if kwargs.get("return_offsets_mapping"):
                    result["offset_mapping"] = [(i, min(i+2, len(text))) for i in range(0, len(text), 2)]
                    result["input_ids"] = result["input_ids"][::2]
                return result
        builder = InputBuilder(PairTokenizer(), source_max_length=2048, context_max_length=384)
        v = view(2, [(0, 1)])
        v["locations"] = [{"code": "a", "OFFSET": 0, "OFFSET_END": 1},
                          {"code": "b", "OFFSET": 1, "OFFSET_END": 2}]
        targets, counts = region_targets(record("ab"), v, [[2]*4, [3]*4], data.cfg_regions(v), builder)
        self.assertEqual(targets[0]["nodes"], [0])
        self.assertEqual(counts["overlapping_operations"], 1)
        v["edges"] = []
        v["locations"][1] = dict(v["locations"][0])
        targets, counts = region_targets(record("ab"), v, [[2]*4]*2, data.cfg_regions(v), builder)
        self.assertEqual(targets, [])
        self.assertEqual(counts["ambiguous_duplicate_operations"], 2)

    def test_uniform_region_sampling_is_label_blind_and_rng_isolated(self):
        source = " ".join(["xx"]*24)
        v = view(24, [])
        v["locations"] = [{"code": "xx", "OFFSET": i*3, "OFFSET_END": i*3+2} for i in range(24)]
        builder = InputBuilder(CharTokenizer(), source_max_length=2048, context_max_length=384)
        random.seed(4)
        before = random.getstate()
        a, _ = region_targets(record(source, label=0), v, [[2]*4]*24, data.cfg_regions(v), builder)
        b, _ = region_targets(record(source, label=1), v, [[2]*4]*24, data.cfg_regions(v), builder)
        self.assertEqual(a, b)
        self.assertEqual(before, random.getstate())
        self.assertEqual(len(a), 8)
        self.assertTrue(any(r["region"] > 16 for r in a))

    def test_soft_ce_and_source_gradient_no_target_input(self):
        torch.manual_seed(42)
        head = RegionReconstructionHead(4, [4]*4)
        hidden = torch.randn(1, 6, 4, requires_grad=True)
        r = target((1, 3))
        prediction = head(hidden[0], r["tokens"])
        manual = -(prediction[0].log_softmax(-1)*torch.tensor([.25, .75])).sum()/2
        manual -= prediction[2].log_softmax(-1)[0]/2
        loss = region_function_loss(hidden, [[r]], head)
        torch.testing.assert_close(loss, manual)
        loss.backward()
        self.assertGreater(hidden.grad[0, [1, 3]].abs().sum(), 0)
        self.assertEqual(hidden.grad[0, [0, 2, 4, 5]].abs().sum(), 0)
        self.assertGreater(head.projection.weight.grad.abs().sum(), 0)
        altered = copy.deepcopy(r)
        altered["targets"] = [[[3, 1.]], [], [], []]
        for a, b in zip(prediction, head(hidden[0], altered["tokens"])):
            torch.testing.assert_close(a, b)

    def test_independent_window_denominators_and_unclipped_gradients(self):
        torch.manual_seed(42)
        source = torch.randn(5, 6, 4)
        dependency = [[{"definition_token": 0, "use_token": 2, "label": i%2}] if i in (0, 1, 4) else []
                      for i in range(5)]
        regions = [[target((1, 3))] if i in (1, 3) else [] for i in range(5)]
        full = nn.ModuleList([nn.Linear(4, 4), DirectedRelationHead(4, 3), RegionReconstructionHead(4, [4]*4)])
        def backward(model, partitions):
            result = 0
            for indices in partitions:
                hidden = model[0](source[indices])
                dep_batch, region_batch = [dependency[i] for i in indices], [regions[i] for i in indices]
                dep, _, _ = relation_function_loss(hidden, dep_batch, model[1])
                reg = region_function_loss(hidden, region_batch, model[2])
                loss = accumulation_window_loss(hidden.square().mean(), dep, len(indices), 5,
                                                sum(map(bool, dep_batch)), 3)
                loss += reg * (sum(map(bool, region_batch))/2)
                loss.backward()
                result += loss.detach()
            return result
        expected = backward(full, [list(range(5))])
        for partitions in ([[0], [1], [2], [3], [4]], [[0, 1], [2, 3], [4]], [[0, 1, 2], [3, 4]]):
            model = copy.deepcopy(full)
            model.zero_grad(set_to_none=True)
            torch.testing.assert_close(backward(model, partitions), expected)
            for a, b in zip(model.parameters(), full.parameters()):
                if a.grad is None or b.grad is None:
                    self.assertIs(a.grad, b.grad)
                else:
                    torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=2e-5)

    def test_real_joern_graph_partition_and_positions(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/joern_operand_read.json").read_text())
        v = data.abstract_cfg(fixture["graph"])
        r = data.cfg_regions(v)
        self.assertEqual(sum(map(len, r["members"])), len(v["node_ids"]))
        rows = [record(fixture["source"])]
        vocabulary = data.AttributeVocabulary.fit(rows, {rows[0]["sample_key"]: v})
        targets, counts = region_targets(rows[0], v, vocabulary.encode(v), r,
            InputBuilder(CharTokenizer(), source_max_length=2048, context_max_length=384))
        self.assertTrue(targets)
        self.assertGreater(counts["selected_operations"], 0)
        for region in targets:
            for span in region["spans"]:
                self.assertIn(fixture["source"][slice(*span)], [loc["code"] for loc in v["locations"]])


class HPSource(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = TinyEncoder()
        self.task_modules = nn.ModuleDict({"classifier": nn.Linear(4, 1)})


class HPBase(TinyBase):
    SequenceVulnerabilityClassifier = HPSource

    def __init__(self):
        super().__init__()
        self.loaded = []

    def set_peft_model_state_dict(self, encoder, state):
        self.loaded.append({k: v.clone() for k, v in state.items()})
        return super().set_peft_model_state_dict(encoder, state)


class PipelineTests(unittest.TestCase):
    def test_prepare_p0_p1_single_forward_and_bcd_reuse_only_lora(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/joern_operand_read.json").read_text())
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            rows = [record(fixture["source"]+f"\n/* {i} */", f"p{i}", split, i%2)
                    for i, split in enumerate(("train", "train", "valid", "valid", "test", "test"))]
            dataset, graphs, reference = root/"data.jsonl", root/"graphs.jsonl", root/"c"
            dataset.write_text("".join(json.dumps(r)+"\n" for r in rows))
            from vulnmechanism.cpg import JOERN_SOURCE_PREPROCESSING_VERSION
            graphs.write_text("".join(json.dumps({"sample_key": r["sample_key"], "dataset": r["dataset"],
                "split": r["split"], "label": r["label"], "source_sha256": data.source_hash(r["raw_source"]),
                "graph_schema_version": data.GRAPH_SCHEMA, "preprocessing_version": JOERN_SOURCE_PREPROCESSING_VERSION,
                "preprocessing_applied": False, "original_source_sha256": data.source_hash(r["raw_source"]),
                "parsed_source_sha256": data.source_hash(r["raw_source"]), "graph": fixture["graph"]})+"\n" for r in rows))
            common = ["--dataset", str(dataset), "--graphs", str(graphs), "--model-path", "tiny-test-only",
                      "--device", "cpu", "--epochs", "1", "--batch-size", "1", "--gradient-accumulation", "2",
                      "--graph-hidden-size", "8", "--graph-steps", "5"]
            base = HPBase()
            exp.run_experiment(exp.parser().parse_args(["run", *common, "--output-dir", str(reference),
                                                       "--variants", "cfg"]), base=base)
            cache = root/"regions"
            audit = data.prepare_regions(reference, cache, CharTokenizer())
            self.assertFalse((cache/"test.regions.jsonl").exists())
            from vulnmechanism.cfg_behavior import prepare_behavior
            from vulnmechanism.cfg_behavior import supplement_behavior
            from vulnmechanism.cpg import FunctionGraph, GraphNode, GraphEdge
            def fresh_graph():
                return FunctionGraph("fixture", {
                    "fresh-"+n["id"]:GraphNode("fresh-"+n["id"],n["label"],n["code"],n["properties"])
                    for n in fixture["graph"]["nodes"]}, tuple(
                    GraphEdge(e["kind"],"fresh-"+e["source"],"fresh-"+e["target"],
                              {"test_attribute":"preserved"} if e["kind"]=="CFG" else {})
                    for e in fixture["graph"]["edges"]))
            supplement=root/"supplement.jsonl"
            with patch("vulnmechanism.cpg.extract_function_cpg_batch", side_effect=lambda *a,**k:[fresh_graph()]):
                supplement_behavior(reference,supplement)
            with patch("vulnmechanism.cpg.extract_function_cpg_batch", side_effect=AssertionError("duplicate export")):
                supplement_behavior(reference,supplement)
            supplemented=data.read_jsonl(supplement)
            self.assertEqual(len(supplemented),len(rows))
            self.assertEqual({n["id"] for n in supplemented[0]["graph"]["nodes"]},
                             {n["id"] for n in fixture["graph"]["nodes"]})
            self.assertTrue(all(e["properties"]["test_attribute"]=="preserved"
                                for e in supplemented[0]["graph"]["edges"] if e["kind"]=="CFG"))
            changed=fresh_graph()
            first_cfg=next(e for e in changed.edges if e.kind=="CFG")
            changed=FunctionGraph(changed.function,changed.nodes,tuple(e for e in changed.edges if e is not first_cfg))
            with patch("vulnmechanism.cpg.extract_function_cpg_batch", return_value=[changed]), \
                    self.assertRaisesRegex(ValueError,"CFG changed"):
                supplement_behavior(reference,root/"rejected.jsonl")
            behavior_cache=root/"behavior"
            prepare_behavior(reference,behavior_cache,supplement)
            partitions, _ = data.load_regions(cache, rows, reference)
            builder = InputBuilder(CharTokenizer(), source_max_length=2048, context_max_length=384)
            regions, _ = _checked_region_rows(cache, rows[:2], builder, partitions)
            self.assertEqual(set(regions), {"p0", "p1"})
            with self.assertRaises(ValueError):
                _checked_region_rows(cache, rows[4:], builder, partitions, split="test")
            supervision = root/"dep"
            supervision.mkdir()
            offset = len(builder.source_prefix)
            rels = [dict(sample_key=r["sample_key"], split="train", source_sha256=data.source_hash(r["raw_source"]),
                definition_node_id=f"d{i}", use_node_id=f"u{i}", definition_span=[i, i+1], use_span=[i+1, i+2],
                definition_token=offset+i, use_token=offset+i+1, definition_coordinates={"OFFSET": i, "OFFSET_END": i+1},
                use_coordinates={"OFFSET": i+1, "OFFSET_END": i+2}, label=i) for r in rows[:2] for i in (0, 1)]
            (supervision/"relations.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rels))
            data.atomic_json(supervision/"audit.json", {"reference_run_dir": str(reference),
                "train_cohort_sha256": data.cohort_hash(rows[:2]),
                "relations_sha256": data.file_sha256(supervision/"relations.jsonl"),
                "counts": {"functions_with_relations": 2, "selected_positive": 2, "selected_negative": 2}})
            p0, p1, a = root/"p0", root/"p1", root/"a"
            with patch("vulnmechanism.cfg_dependency._load_qwen_lm_weight", return_value=torch.ones(32, 4)):
                out0 = pretrain_causal_dependency(reference, supervision, p0, modes=("dep_pretrain",), base=base)
                rng0 = torch.get_rng_state().clone()
                out1 = pretrain_causal_dependency(reference, supervision, p1, modes=("region_pretrain",), base=base,
                    region_dir=cache, reference_pretrain_dir=p0)
                rng1 = torch.get_rng_state().clone()
            self.assertTrue(torch.equal(rng0, rng1))
            self.assertTrue(torch.equal(base.initial[0], base.initial[1]))
            self.assertEqual([e.calls for e in base.encoders], [2, 2])
            self.assertEqual(out0["dep_pretrain"]["optimizer_steps"], out1["region_pretrain"]["optimizer_steps"])
            self.assertEqual(out1["region_pretrain"]["effective_region_functions"], 2)
            self.assertGreater(out1["region_pretrain"]["region_loss"], 0)
            exp.run_experiment(exp.parser().parse_args(["run", *common, "--output-dir", str(a), "--variants",
                "dep_pretrain_cfg", "--pretrain-dir", str(p0), "--reference-run-dir", str(reference)]), base=base)
            for stage1, folder, variants in ((p0, root/"b", ["dep_pretrain_hierarchical"]),
                    (p1, root/"cd", ["region_pretrain_cfg", "region_pretrain_hierarchical"]),
                    (p0, root/"context", ["region_local", "region_context"]),
                    (p0, root/"behavior_run", ["behavior_nodes", "behavior_edges", "behavior_joint", "behavior_masked"])):
                loaded_before = len(base.loaded)
                expected_adapter = torch.load(stage1/("dep_pretrain" if stage1 == p0 else "region_pretrain")/"last.pt",
                                               weights_only=False)["adapter_state"]
                result = exp.run_experiment(exp.parser().parse_args(["run", *common, "--output-dir", str(folder),
                    "--variants", *variants, "--pretrain-dir", str(stage1), "--reference-run-dir", str(reference),
                    "--region-dir", str(cache), "--behavior-dir", str(behavior_cache), "--comparison-run-dir", str(a)]), base=base)
                self.assertEqual(len(base.loaded)-loaded_before, len(variants))
                for loaded in base.loaded[loaded_before:]:
                    self.assertEqual(set(loaded), {"adapter.weight"})
                    torch.testing.assert_close(loaded["adapter.weight"], expected_adapter["adapter.weight"])
                self.assertEqual(set(result["changes_vs_A"]), set(variants))
                hashes = []
                for variant in variants:
                    ckpt = torch.load(folder/variant/"best.pt", weights_only=False)
                    self.assertFalse(any("region_head" in k or "relation_head" in k for k in ckpt["task_state"]))
                    hashes.append(ckpt["model_config"]["pretrain_checkpoint_sha256"])
                self.assertEqual(len(set(hashes)), 1)
            context_comparison = exp.compare_run(root/"context", "valid", reference_root=a,
                                                   reference_variant="dep_pretrain_cfg")
            self.assertIn("changes_region_context_vs_local", context_comparison)
            old_h = exp.compare_run(root/"context", "valid", reference_root=root/"b",
                                     reference_variant="dep_pretrain_hierarchical")
            self.assertEqual(set(old_h["changes_vs_old_H"]), {"region_local", "region_context"})
            # Test evaluation consumes a frozen valid-selected threshold, never selects it.
            exp.evaluate_run(exp.parser().parse_args(["eval", "--run-dir", str(reference), "--variants",
                "cfg", "--split", "test", "--device", "cpu"]), base=base)
            exp.evaluate_run(exp.parser().parse_args(["eval", "--run-dir", str(a), "--variants",
                "dep_pretrain_cfg", "--split", "test", "--device", "cpu"]), base=base)
            with patch.object(exp, "select_threshold", side_effect=AssertionError("test tuning")):
                result = exp.evaluate_run(exp.parser().parse_args(["eval", "--run-dir", str(root/"cd"),
                    "--variants", "region_pretrain_cfg", "region_pretrain_hierarchical", "--split", "test",
                    "--device", "cpu"]), base=base)
            self.assertEqual(set(result["changes_vs_A"]), {"region_pretrain_cfg", "region_pretrain_hierarchical"})
            with patch.object(exp, "select_threshold", side_effect=AssertionError("test tuning")):
                exp.evaluate_run(exp.parser().parse_args(["eval", "--run-dir", str(root/"context"),
                    "--variants", "region_local", "region_context", "--split", "test", "--device", "cpu"]), base=base)
            with patch.object(exp, "select_threshold", side_effect=AssertionError("test tuning")):
                exp.evaluate_run(exp.parser().parse_args(["eval", "--run-dir", str(root/"behavior_run"),
                    "--variants", "behavior_nodes", "behavior_edges", "behavior_joint", "behavior_masked",
                    "--split", "test", "--device", "cpu"]), base=base)
            # An old H classifier cannot be substituted for a new variant.
            (root/"context"/"region_context"/"best.pt").write_bytes(
                (root/"b"/"dep_pretrain_hierarchical"/"best.pt").read_bytes())
            with self.assertRaisesRegex(ValueError, "checkpoint/config"):
                exp.evaluate_run(exp.parser().parse_args(["eval", "--run-dir", str(root/"context"),
                    "--variants", "region_context", "--split", "test", "--device", "cpu",
                    "--replace-predictions"]), base=base)
            invalid_path = cache/"train.regions.jsonl"
            invalid_path.write_text((cache/"valid.regions.jsonl").read_text())
            audit["train_targets_sha256"] = data.file_sha256(invalid_path)
            data.atomic_json(cache/"audit.json", audit)
            with self.assertRaisesRegex(ValueError, "source/split mismatch"):
                _checked_region_rows(cache, rows[:2], builder, partitions)



if __name__ == "__main__":
    unittest.main()
