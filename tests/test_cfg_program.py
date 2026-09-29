"""CPU checks for conservative check/update/use facts and shared graph readout."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
import unittest

import torch

from vulnmechanism.cfg_data import (GRAPH_SCHEMA, cohort_hash, file_sha256,
                                   source_hash)
from vulnmechanism.cpg import JOERN_SOURCE_PREPROCESSING_VERSION
from vulnmechanism.cfg_dependency import (ComposedRelationHead, _checked_program_queries,
                                         evaluate_fixed_relations, pretrain_causal_dependency,
                                         relation_function_loss)
from vulnmechanism.cfg_network import AttributeCFGEncoder, ProgramCFGEncoder, collate_graphs
from vulnmechanism.cfg_program import (JOERN_OPS, build_program, load_programs,
                                       prepare_program, program_cost)
from vulnmechanism.model import InputBuilder
from vulnmechanism.syntax import parser_for, walk
from tests.test_cfg_dependency import CharTokenizer, TinyBase, record


def cached_graph(source, *, renumber=False):
    """Test-only Joern-shaped positions and AST/CFG edges at exact syntax spans."""
    root = parser_for("c").parse(source.encode()).root_node
    selected = []
    for node in walk(root):
        kind, name = None, None
        if node.type == "identifier":
            kind = "IDENTIFIER"
        elif node.type == "if_statement":
            kind = "CONTROL_STRUCTURE"
        elif node.type == "return_statement":
            kind = "RETURN"
        elif node.type == "subscript_expression":
            kind, name = "CALL", "<operator>.indirectIndexAccess"
        elif node.type == "assignment_expression":
            kind, name = "CALL", "<operator>.assignment"
        elif node.type == "binary_expression":
            operator = node.child_by_field_name("operator")
            if operator is not None and operator.text.decode() in JOERN_OPS:
                kind, name = "CALL", "<operator>." + JOERN_OPS[operator.text.decode()]
        if kind:
            selected.append((node, kind, name))
    by_id = {node.id: f"n{len(selected)-index if renumber else index}"
             for index, (node, _, _) in enumerate(selected)}
    nodes = [{"id": "m", "label": "METHOD", "code": "f", "properties": {"kind": "METHOD"}}]
    edges, cfg = [], []
    for node, kind, name in selected:
        node_id = by_id[node.id]
        start = len(source.encode()[:node.start_byte].decode().encode("utf-16-le")) // 2
        end = len(source.encode()[:node.end_byte].decode().encode("utf-16-le")) // 2
        properties = {"kind": kind, "OFFSET": start, "OFFSET_END": end}
        if name:
            properties["NAME"] = name
        nodes.append({"id": node_id, "label": kind, "code": node.text.decode(),
                      "properties": properties})
        parent = node.parent
        while parent is not None and parent.id not in by_id:
            parent = parent.parent
        edges.append({"kind": "AST", "source": by_id[parent.id] if parent else "m",
                      "target": node_id})
        if kind != "IDENTIFIER":
            cfg.append((start, end, node_id))
    cfg.sort()
    edges.extend({"kind": "CFG", "source": before[2], "target": after[2]}
                 for before, after in zip(cfg, cfg[1:]))
    return {"nodes": nodes, "edges": edges}


def facts(source, *, graph=None, key="t0"):
    builder = InputBuilder(CharTokenizer(), source_max_length=2048, context_max_length=384)
    row = record(source, key)
    return build_program(row, graph or cached_graph(source), CharTokenizer(),
                         prefix_tokens=len(builder.source_prefix))


class ProgramFactTests(unittest.TestCase):
    def test_check_overwrite_copy_and_constant_proofs(self):
        cases = (
            ("int f(int i){if(i<10){return a[i];}return 0;}", 1, "array_index"),
            ("int f(int i){if(i<10){i=20;return a[i];}return 0;}", 0, "array_index"),
            ("int f(int i){if(i<10){i=5;return a[i];}return 0;}", 1, "array_index"),
            ("int f(int i){int j; if(i<10){j=i;return a[j];}return 0;}", 1, "array_index"),
            ("int f(int i){int j;int k;if(i<10){j=i;k=j;return a[k];}return 0;}",
             1, "array_index"),
        )
        for source, label, operation in cases:
            with self.subTest(source=source):
                program, counts = facts(source)
                selected = [query for query in program["queries"] if query["label"] is not None
                            and query["operation"] == operation]
                self.assertTrue(selected, counts)
                self.assertIn(label, {query["label"] for query in selected})
                self.assertTrue(all(query["check_token"] <= query["update_token"] <=
                                    query["use_token"] for query in selected))

    def test_early_return_branch_and_opposite_query(self):
        source = "int f(int i){if(i>=10)return 0;return a[i];}"
        program, _ = facts(source)
        selected = [query for query in program["queries"] if query["operation"] == "array_index"]
        self.assertEqual({(query["branch"], query["label"]) for query in selected},
                         {(1, 0), (2, 1)})
        self.assertNotEqual(selected[0]["update_token"], selected[1]["update_token"])

    def test_shadowing_multi_path_and_unknown_update(self):
        shadow = "int f(int i){if(i<10){int i=20;return a[i];}return 0;}"
        program, _ = facts(shadow)
        self.assertFalse(any(q["label"] is not None for q in program["queries"]))
        merged = "int f(int i,int j){if(i<10){i=20;}else{i=j;}return a[i];}"
        program, _ = facts(merged)
        self.assertFalse(any(q["label"] is not None for q in program["queries"]))
        unsupported = "int f(int i){if(i<10){i+=1;return a[i];}return 0;}"
        program, _ = facts(unsupported)
        self.assertFalse(any(q["label"] is not None for q in program["queries"]))

    def test_renaming_and_node_ids_keep_program_graph_equal(self):
        first = "int f(int i){if(i<10){return a[i];}return 0;}"
        renamed = "int f(int j){if(j<10){return b[j];}return 0;}"
        left, _ = facts(first)
        right, _ = facts(renamed, graph=cached_graph(renamed, renumber=True))
        view = {"node_ids": ["a", "b"], "edges": [(0, 1)]}
        encoded = [[0, 0, 0, 0], [0, 0, 0, 0]]
        batch_left = collate_graphs([view], [encoded], programs=[left])
        batch_right = collate_graphs([view], [encoded], programs=[right])
        self.assertTrue(torch.equal(batch_left.program.features, batch_right.program.features))
        self.assertTrue(torch.equal(batch_left.program.edges, batch_right.program.edges))
        self.assertTrue(torch.equal(batch_left.program.plain_edges, batch_right.program.plain_edges))
        torch.manual_seed(42)
        encoder = ProgramCFGEncoder([2] * 4, hidden_size=8, steps=2, mode="program_state")
        encoder.eval()
        torch.testing.assert_close(encoder(batch_left), encoder(batch_right))

    def test_changed_check_update_branch_and_unknown_cases(self):
        target = "int f(int i,int j){if(i<10){return a[i];}return 0;}"
        other = "int f(int i,int j){if(j<10){return a[i];}return 0;}"
        first, _ = facts(target)
        second, _ = facts(other)
        self.assertTrue(any(q["label"] == 1 for q in first["queries"]))
        self.assertFalse(any(q["label"] is not None for q in second["queries"]))
        left = "int f(int i){if(i<10){i=20;return a[i];}return 0;}"
        right = "int f(int i){if(i>10){i=20;return a[i];}return 0;}"
        left_program, _ = facts(left)
        right_program, _ = facts(right)
        self.assertEqual({q["label"] for q in left_program["queries"] if q["label"] is not None}, {0})
        self.assertEqual({q["label"] for q in right_program["queries"] if q["label"] is not None}, {1})
        exchange = "int f(int i,int j){if(i<10){j=i;i=20;return a[j];}return 0;}"
        exchanged, _ = facts(exchange)
        self.assertTrue(any(q["label"] == 1 for q in exchanged["queries"]))
        repeated = "int f(int i){if(i>=10)return 0;if(i<5)return a[i];return 0;}"
        repeated_program, _ = facts(repeated)
        next_check = repeated.index("i<5")
        self.assertTrue(any(q["label"] == 1 and q["use_span"] == [next_check, next_check + 1]
                            for q in repeated_program["queries"]))
        for source in ("int f(volatile int i){if(i<10)return a[i];return 0;}",
                       "int f(int i){if(i<10){foo();return a[i];}return 0;}",
                       "int f(int i){if(i<10){for(;;){}return a[i];}return 0;}"):
            with self.subTest(source=source):
                program, _ = facts(source)
                self.assertFalse(any(q["label"] is not None for q in program["queries"]))
        truncated = "int f(int i){/*" + "x" * 2100 + "*/if(i<10)return a[i];return 0;}"
        program, counts = facts(truncated)
        self.assertFalse(any(q["label"] is not None for q in program["queries"]))
        self.assertGreater(counts["query_outside_visible_or_unaligned"], 0)

    def test_unrelated_binding_check_is_excluded_from_use_state(self):
        source = "int f(int i,int j){if(i<10){if(j<5){return a[i];}}return 0;}"
        program, _ = facts(source)
        use = next(item for item in program["slices"] if item["operation"] == "array_index")
        check_texts = [source[slice(*event["span"])] for path in use["paths"]
                       for event in path["events"] if event["kind"] == "check"]
        self.assertIn("i<10", check_texts)
        self.assertNotIn("j<5", check_texts)

    def test_composition_head_grads_and_train_only_preparation(self):
        hidden = torch.randn(2, 8, 4, requires_grad=True)
        head = ComposedRelationHead(4, rank=2)
        loss, labels, _ = relation_function_loss(hidden, [
            [{"check_token": 1, "update_token": 2, "use_token": 5, "label": 1}],
            [{"check_token": 0, "update_token": 3, "use_token": 6, "label": 0}]], head)
        self.assertEqual(labels, [1, 0])
        loss.backward()
        self.assertGreater(hidden.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in head.parameters()))

        sources = ["int f(int i){if(i<10){return a[i];}return 0;}",
                   "int f(int i){if(i<10){i=20;return a[i];}return 0;}",
                   "int f(int i){if(i>=10)return 0;return a[i];}",
                   "int f(int i){return a[i];}"]
        rows = [record(source, f"p{index}", split="train" if index < 2 else
                       "valid" if index == 2 else "test", label=index % 2)
                for index, source in enumerate(sources)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, graphs, reference, prepared = (root / "data.jsonl", root / "graphs.jsonl",
                                                     root / "c", root / "program")
            dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
            graphs.write_text("".join(json.dumps({
                "sample_key": row["sample_key"], "dataset": row["dataset"],
                "split": row["split"], "label": row["label"],
                "source_sha256": source_hash(row["raw_source"]),
                "graph_schema_version": GRAPH_SCHEMA,
                "preprocessing_version": JOERN_SOURCE_PREPROCESSING_VERSION,
                "preprocessing_applied": False,
                "original_source_sha256": source_hash(row["raw_source"]),
                "parsed_source_sha256": source_hash(row["raw_source"]),
                "graph": cached_graph(row["raw_source"])}) + "\n" for row in rows))
            reference.mkdir()
            config = {"dataset": str(dataset), "graphs": str(graphs), "source_dataset": "primevul",
                      "source_max_length": 2048, "seed": 42, "model_path": "tiny-test-only",
                      "cohort_sha256": cohort_hash(rows),
                      "train_cohort_sha256": cohort_hash(rows[:2]),
                      "valid_cohort_sha256": cohort_hash(rows[2:3]),
                      "graph_file_sha256": file_sha256(graphs), "lora_r": 16,
                      "lora_alpha": 32, "lora_dropout": .05, "learning_rate": .0002,
                      "graph_learning_rate": .001, "batch_size": 2,
                      "gradient_accumulation": 2, "weight_decay": .01, "log_every": 1}
            (reference / "config.json").write_text(json.dumps(config))
            with redirect_stdout(io.StringIO()):
                audit = prepare_program(str(dataset), str(graphs), str(reference),
                                        str(prepared), CharTokenizer())
            self.assertGreater(audit["splits"]["train"]["query_positive"], 0)
            self.assertGreater(audit["splits"]["train"]["query_negative"], 0)
            self.assertFalse((prepared / "test.queries.jsonl").exists())
            programs, _ = load_programs(prepared, rows, reference)
            self.assertEqual(set(programs), {row["sample_key"] for row in rows})
            self.assertGreater(program_cost(programs, rows, 2)["train"]["event_states"], 0)
            builder = InputBuilder(CharTokenizer(), source_max_length=2048, context_max_length=384)
            train_queries, _ = _checked_program_queries(prepared, rows[:2], builder)
            self.assertEqual(set(train_queries), {"p0", "p1"})
            base = TinyBase()
            with (patch("vulnmechanism.cfg_dependency._load_qwen_lm_weight",
                        return_value=torch.randn(32, 4)), redirect_stdout(io.StringIO())):
                result = pretrain_causal_dependency(str(reference), str(root / "unused"),
                    str(root / "stage1"), modes=("composition_pretrain",),
                    program_dir=str(prepared), base=base)["composition_pretrain"]
            self.assertEqual(base.encoders[0].calls, 1)
            self.assertEqual(result["effective_relation_functions"], 2)
            self.assertEqual((result["relation_positive"], result["relation_negative"]), (1, 1))
            with redirect_stdout(io.StringIO()):
                report = evaluate_fixed_relations(str(root / "stage1"),
                    str(root / "evaluation"), mode="composition_pretrain", base=TinyBase())
            self.assertEqual(report["splits"]["valid"]["relation_pairs"], 2)
            self.assertFalse((root / "evaluation" / "test.predictions.jsonl").exists())
            self.assertEqual(set(report["splits"]["valid"]["examples"]),
                             {"correct", "incorrect"})
            valid_query = json.loads((prepared / "valid.queries.jsonl").read_text().splitlines()[0])
            train_path = prepared / "train.queries.jsonl"
            train_path.write_text(json.dumps(valid_query) + "\n")
            changed = json.loads((prepared / "audit.json").read_text())
            changed["train_queries_sha256"] = file_sha256(train_path)
            (prepared / "audit.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "outside the requested original C split"):
                _checked_program_queries(prepared, rows[:2], builder)

    def test_five_official_controls_share_cached_inputs_and_valid_only_selection(self):
        from tests.test_cfg_ablation import TinyEncoder, TinyInputBuilder, fake_base
        from vulnmechanism import cfg_experiment as exp

        sources = [
            "int f(int i){if(i<10){return a[i];}return 0;}",
            "int f(int i){if(i<10){i=20;return a[i];}return 0;}",
            "int f(int i){if(i>=5)return 0;return a[i];}",
            "int f(int i){if(i<9){i=4;return a[i];}return 0;}",
            "int f(int i){if(i<7){return a[i];}return 0;}",
            "int f(int i){if(i<8){i=3;return a[i];}return 0;}",
        ]
        splits = ["train", "train", "valid", "valid", "test", "test"]
        rows = [record(source, f"s{index}", split=split, label=index % 2)
                for index, (source, split) in enumerate(zip(sources, splits))]
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            dataset, graphs = root / "data.jsonl", root / "graphs.jsonl"
            reference, prepared = root / "c", root / "program"
            dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
            graphs.write_text("".join(json.dumps({
                "sample_key": row["sample_key"], "dataset": row["dataset"],
                "split": row["split"], "label": row["label"],
                "source_sha256": source_hash(row["raw_source"]),
                "graph_schema_version": GRAPH_SCHEMA,
                "preprocessing_version": JOERN_SOURCE_PREPROCESSING_VERSION,
                "preprocessing_applied": False,
                "original_source_sha256": source_hash(row["raw_source"]),
                "parsed_source_sha256": source_hash(row["raw_source"]),
                "graph": cached_graph(row["raw_source"])}) + "\n" for row in rows))
            api = fake_base()
            common = ["--dataset", str(dataset), "--graphs", str(graphs),
                      "--model-path", "tiny-test-only", "--device", "cpu", "--epochs", "1",
                      "--batch-size", "2", "--gradient-accumulation", "2",
                      "--graph-hidden-size", "8", "--graph-steps", "2"]
            exp.run_experiment(exp.parser().parse_args([
                "run", *common, "--output-dir", str(reference), "--variants", "cfg"]), base=api)
            audit = prepare_program(dataset, graphs, reference, prepared, CharTokenizer())
            self.assertEqual(audit["splits"]["train"]["functions"], 2)
            c_config = json.loads((reference / "config.json").read_text())
            adapter = TinyEncoder().state_dict()
            old, composed = root / "old_pretrain", root / "composed_pretrain"
            for stage1, mode in ((old, "dep_pretrain"), (composed, "composition_pretrain")):
                stage1.mkdir()
                pretrain_config = {"c_config": c_config,
                    "reference_run_dir": str(reference.resolve()), "pretrain_epochs": 1,
                    "relation_alpha": 1.0, "relations_sha256": "synthetic-only"}
                if mode == "composition_pretrain":
                    pretrain_config.update(program_dir=str(prepared.resolve()),
                                           program_sha256=audit["program_sha256"])
                (stage1 / "config.json").write_text(json.dumps(pretrain_config))
                folder = stage1 / mode
                folder.mkdir()
                torch.save({"mode": mode, "pretrain_config": pretrain_config,
                            "adapter_state": adapter}, folder / "last.pt")
                (folder / "complete.json").write_text(json.dumps({
                    "mode": mode, "checkpoint_sha256": file_sha256(folder / "last.pt")}))
            controls = ((old, root / "old_stage2", ["dep_pretrain_cfg",
                         "dep_pretrain_program_plain", "dep_pretrain_program_state"]),
                        (composed, root / "composed_stage2", ["composition_pretrain_cfg",
                         "composition_pretrain_program_state"]))
            for stage1, stage2, variants in controls:
                trained = exp.run_experiment(exp.parser().parse_args([
                    "run", *common, "--output-dir", str(stage2), "--variants", *variants,
                    "--pretrain-dir", str(stage1), "--program-dir", str(prepared),
                    "--reference-run-dir", str(reference)]), base=api)
                self.assertEqual(set(trained["changes_vs_cfg"]), set(variants))
                for variant in variants:
                    self.assertTrue((stage2 / variant / "valid.predictions.jsonl").exists())
            TinyInputBuilder.allow_test = True
            try:
                exp.evaluate_run(exp.parser().parse_args([
                    "eval", "--run-dir", str(reference), "--split", "test",
                    "--variants", "cfg", "--device", "cpu"]), base=api)
                for _, stage2, variants in controls:
                    with patch.object(exp, "select_threshold", side_effect=AssertionError("test tuning")):
                        evaluated = exp.evaluate_run(exp.parser().parse_args([
                            "eval", "--run-dir", str(stage2), "--split", "test",
                            "--variants", *variants, "--device", "cpu"]), base=api)
                    self.assertEqual(set(evaluated["changes_vs_cfg"]), set(variants))
            finally:
                TinyInputBuilder.allow_test = False

    def test_plain_and_versioned_have_matched_weights_and_gradients(self):
        source = "int f(int i){if(i<10){i=20;return a[i];}return 0;}"
        program, _ = facts(source)
        view = {"node_ids": ["a", "b"], "edges": [(0, 1)]}
        batch = collate_graphs([view], [[[0]*4, [0]*4]], programs=[program])
        torch.manual_seed(42)
        plain = ProgramCFGEncoder([2]*4, hidden_size=8, steps=2, mode="program_plain")
        torch.manual_seed(42)
        versioned = ProgramCFGEncoder([2]*4, hidden_size=8, steps=2, mode="program_state")
        self.assertEqual(sum(p.numel() for p in plain.parameters()),
                         sum(p.numel() for p in versioned.parameters()))
        for key, value in plain.state_dict().items():
            self.assertTrue(torch.equal(value, versioned.state_dict()[key]), key)
        output = versioned(batch)
        self.assertEqual(tuple(output.shape), (1, 16))
        output.square().sum().backward()
        self.assertGreater(versioned.program_message.weight.grad.abs().sum().item(), 0)
        self.assertGreater(versioned.program_embedding[0].weight.grad[1].abs().sum().item(), 0)
        self.assertFalse(torch.allclose(plain(batch), versioned(batch)))
        fallback = collate_graphs([view], [[[0]*4, [0]*4]], programs=[{"slices": []}])
        torch.manual_seed(42)
        original = AttributeCFGEncoder([2]*4, hidden_size=8, steps=2, mode="cfg")
        self.assertTrue(torch.allclose(versioned(fallback), original(fallback), atol=1e-6))
        self.assertFalse(torch.equal(batch.program.edges, batch.program.plain_edges))
        events = program["slices"][0]["paths"][0]["events"]
        check = next(i for i, e in enumerate(events) if e["kind"] == "check")
        write = next(i for i, e in enumerate(events) if e["kind"] == "write")
        self.assertNotIn([check, write], program["slices"][0]["paths"][0]["edges"])


if __name__ == "__main__":
    unittest.main()
