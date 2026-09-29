"""CPU checks for conservative check/update/use facts and shared graph readout."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import copy
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
import unittest

import torch

from vulnmechanism.cfg_data import (GRAPH_SCHEMA, cohort_hash, file_sha256,
                                   source_hash)
from vulnmechanism.cpg import JOERN_SOURCE_PREPROCESSING_VERSION
from vulnmechanism.cfg_dependency import (COMPOSED_HEAD_VERSION, ComposedRelationHead, accumulation_window_loss,
                                         _checked_program_queries,
                                         evaluate_fixed_relations, pretrain_causal_dependency,
                                         relation_function_loss)
from vulnmechanism.cfg_network import AttributeCFGEncoder, ProgramCFGEncoder, collate_graphs
from vulnmechanism.cfg_program import (_Builder, JOERN_OPS, PROGRAM_SCHEMA, build_program, load_programs,
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
        if kind == "CONTROL_STRUCTURE":
            properties["CONTROL_STRUCTURE_TYPE"] = "IF"
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
                self.assertTrue(all(query["check_token"] <= query["use_token"] and
                                    all(query["check_token"] <= token <= query["use_token"]
                                        for path in query["paths"] if not path["missing"]
                                        for token in path["update_tokens"])
                                    for query in selected))

    def test_early_return_branch_and_opposite_query(self):
        source = "int f(int i){if(i>=10)return 0;return a[i];}"
        program, _ = facts(source)
        selected = [query for query in program["queries"] if query["operation"] == "array_index"]
        self.assertEqual({(query["branch"], query["label"]) for query in selected},
                         {(1, 0), (2, 1)})
        self.assertEqual(selected[0]["paths"][0]["update_tokens"], [])
        self.assertEqual(selected[1]["paths"][0]["update_tokens"], [])
        self.assertNotEqual(selected[0]["branch"], selected[1]["branch"])
        right, _ = facts("int f(int i){if(10>i){return a[i];}return 0;}")
        self.assertEqual({(q["target_side"], q["branch"], q["label"])
                          for q in right["queries"]}, {(2, 1, 1), (2, 2, 0)})

    def test_shadowing_multi_path_and_unknown_update(self):
        shadow = "int f(int i){if(i<10){int i=20;return a[i];}return 0;}"
        program, _ = facts(shadow)
        self.assertFalse(any(q["label"] is not None for q in program["queries"]))
        merged = "int f(int i,int j){if(i<10){i=20;}else{i=j;}return a[i];}"
        program, _ = facts(merged)
        self.assertEqual({q["branch"]: q["label"] for q in program["queries"]
                          if q["operation"] == "array_index"}, {1: 0, 2: None})
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
        self.assertEqual({q["branch"]: q["label"] for q in left_program["queries"]
                          if q["operation"] == "array_index"}, {1: 0, 2: 1})
        self.assertEqual({q["branch"]: q["label"] for q in right_program["queries"]
                          if q["operation"] == "array_index"}, {1: 1, 2: 0})
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
        first, _ = facts("int f(int i){if(i<10){i=5;i=20;return a[i];}return 0;}")
        second, _ = facts("int f(int i){if(i<10){return a[i];}return 0;}")
        hidden = torch.randn(2, 512, 4, requires_grad=True)
        head = ComposedRelationHead(4, rank=2)
        loss, labels, _ = relation_function_loss(hidden, [first["queries"], second["queries"]], head)
        self.assertEqual(labels, [0, 1, 1, 0])
        loss.backward()
        self.assertGreater(hidden.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in head.parameters()))

        sources = ["int f(int i){if(i<10){return a[i];}return 0;}",
                   "int f(int i){if(i<10){i=20;return a[i];}return 0;}",
                   "int f(int i){if(i>=10)return 0;return a[i];}",
                   "int f(int i){if(i<7){i=4;return a[i];}return 0;}",
                   "int f(int i){return a[i];}"]
        rows = [record(source, f"p{index}", split="train" if index < 2 else
                       "valid" if index < 4 else "test", label=index % 2)
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
                      "valid_cohort_sha256": cohort_hash(rows[2:4]),
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
            with redirect_stdout(io.StringIO()):
                comparison = prepare_program(str(dataset), str(graphs), str(reference),
                    str(root / "compared"), CharTokenizer(), comparison_program_dir=prepared)
            for split in ("train", "valid"):
                compared = comparison["comparison"]["splits"][split]
                self.assertEqual(compared["added_functions"], [])
                self.assertEqual(compared["lost_functions"], [])
                self.assertEqual(compared["old_effective_combinations"], compared["new_effective_combinations"])

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
            self.assertEqual((result["relation_positive"], result["relation_negative"]), (2, 2))
            with redirect_stdout(io.StringIO()):
                report = evaluate_fixed_relations(str(root / "stage1"),
                    str(root / "evaluation"), mode="composition_pretrain", base=TinyBase())
            self.assertEqual(report["splits"]["valid"]["relation_pairs"], 4)
            self.assertFalse((root / "evaluation" / "test.predictions.jsonl").exists())
            self.assertEqual(set(report["splits"]["valid"]["examples"]),
                             {"correct", "incorrect"})
            readiness_audit = json.loads((prepared / "audit.json").read_text())
            readiness_audit["readiness"]["valid"]["eligible"] = False
            (prepared / "audit.json").write_text(json.dumps(readiness_audit))
            blocked_base = TinyBase()
            with self.assertRaisesRegex(ValueError, "valid independent-function coverage is too small"):
                pretrain_causal_dependency(str(reference), str(root / "unused"),
                    str(root / "blocked"), modes=("composition_pretrain",),
                    program_dir=str(prepared), base=blocked_base)
            self.assertEqual(blocked_base.encoders, [])
            self.assertFalse((root / "blocked").exists())
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
                                           program_sha256=audit["program_sha256"],
                                           program_schema=PROGRAM_SCHEMA,
                                           composed_head_version=COMPOSED_HEAD_VERSION)
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
        fallback = collate_graphs([view], [[[0]*4, [0]*4]], programs=[{"schema": PROGRAM_SCHEMA, "slices": []}])
        torch.manual_seed(42)
        original = AttributeCFGEncoder([2]*4, hidden_size=8, steps=2, mode="cfg")
        self.assertTrue(torch.allclose(versioned(fallback), original(fallback), atol=1e-6))
        self.assertFalse(torch.equal(batch.program.edges, batch.program.plain_edges))
        events = program["slices"][0]["paths"][0]["events"]
        check = next(i for i, e in enumerate(events) if e["kind"] == "check")
        write = next(i for i, e in enumerate(events) if e["kind"] == "write")
        reference = next(i for i, e in enumerate(events) if e["kind"] == "reference")
        self.assertNotIn([check, write], program["slices"][0]["paths"][0]["edges"])
        self.assertNotIn([check, reference], program["slices"][0]["paths"][0]["edges"])

    def test_complete_query_input_and_conflict_detection(self):
        source = "int f(int i){if(i<10){return a[i];}return 0;}"
        program, _ = facts(source)
        positive, negative = program["queries"]
        self.assertEqual((positive["check_token"], positive["use_token"]),
                         (negative["check_token"], negative["use_token"]))
        self.assertEqual((positive["branch"], negative["branch"]), (1, 2))
        self.assertEqual(positive["paths"][0]["update_tokens"], [])
        self.assertEqual(negative["paths"][0]["update_tokens"], [])
        hidden = torch.randn(512, 4)
        head = ComposedRelationHead(4, 2)
        inputs = []
        hook = head.path_query.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach()))
        head(hidden, positive)
        head(hidden, negative)
        repeated = copy.deepcopy(positive)
        repeated["paths"].append(copy.deepcopy(repeated["paths"][0]))
        head(hidden, repeated)
        hook.remove()
        self.assertEqual(inputs[0].shape, (1, 5 * head.rank + 17))
        torch.testing.assert_close(inputs[0][:, :5 * head.rank], inputs[1][:, :5 * head.rank])
        self.assertEqual(inputs[0][0, 5 * head.rank + 6:5 * head.rank + 8].tolist(), [1, 0])
        self.assertEqual(inputs[1][0, 5 * head.rank + 6:5 * head.rank + 8].tolist(), [0, 1])
        self.assertEqual(inputs[2][:, -1].tolist(), [0.25, 0.25])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = record(source, "t0")
            builder = InputBuilder(CharTokenizer(), source_max_length=2048, context_max_length=384)
            query_path = root / "train.queries.jsonl"
            query_path.write_text("".join(json.dumps(query) + "\n" for query in program["queries"]))
            (root / "audit.json").write_text(json.dumps({"program_schema": PROGRAM_SCHEMA,
                "train_queries_sha256": file_sha256(query_path)}))
            loaded, _ = _checked_program_queries(root, [row], builder)
            self.assertEqual(len(loaded["t0"]), 2)
            conflicting = copy.deepcopy(positive)
            conflicting["label"] = 0
            query_path.write_text(query_path.read_text() + json.dumps(conflicting) + "\n")
            (root / "audit.json").write_text(json.dumps({"program_schema": PROGRAM_SCHEMA,
                "train_queries_sha256": file_sha256(query_path)}))
            with self.assertRaisesRegex(ValueError, "input collision/conflict.*t0"):
                _checked_program_queries(root, [row], builder)

    def test_all_paths_counterexample_unknown_and_incomplete(self):
        cases = (
            ("int f(int i){if(i<10){i=5;}else{i=5;}return a[i];}", {1: 1, 2: 0}),
            ("int f(int i){if(i<10){i=5;}else{i=20;}return a[i];}", {1: 0, 2: 0}),
            ("int f(int i,int j){if(i<10){i=5;}else{i=j;}return a[i];}", {1: None, 2: 0}),
            ("int f(int i){if(i<10){i=20;}else{foo();}return a[i];}", {1: 0, 2: None}),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                program, _ = facts(source)
                actual = {q["branch"]: q["label"] for q in program["queries"]
                          if q["operation"] == "array_index"}
                self.assertEqual(actual, expected)
        incomplete, _ = facts(cases[-1][0])
        paths = incomplete["slices"][0]["paths"]
        self.assertEqual(len(paths), 2)
        self.assertTrue(any(path["incomplete"] for path in paths))

    def test_reference_version_is_frozen_and_target_updates_are_ordered(self):
        source = "int f(int i,int j){j=10;if(i<j){j=20;return a[i];}return 0;}"
        program, _ = facts(source)
        query = next(q for q in program["queries"] if q["operation"] == "array_index" and
                     q["branch"] == 1)
        self.assertEqual(query["label"], 1)
        self.assertEqual(query["reference_version"], 1)
        self.assertEqual(query["paths"][0]["reference_version"], 1)
        self.assertEqual(query["paths"][0]["update_tokens"], [])
        self.assertLess(query["paths"][0]["reference_definition_token"], query["check_token"])
        source = "int f(int i){int j;if(i<10){j=i;return a[j];}return 0;}"
        program, _ = facts(source)
        query = next(q for q in program["queries"] if q["operation"] == "array_index" and
                     q["branch"] == 1)
        self.assertEqual(query["label"], 1)
        self.assertEqual(len(query["paths"][0]["update_tokens"]), 1)
        self.assertEqual(query["paths"][0]["current_definition_token"],
                         query["paths"][0]["update_tokens"][0])
        source = "int f(int i){if(i<10){i=5;i=20;return a[i];}return 0;}"
        program, _ = facts(source)
        query = next(q for q in program["queries"] if q["operation"] == "array_index" and
                     q["branch"] == 1)
        updates = query["paths"][0]["update_tokens"]
        self.assertEqual(len(updates), 2)
        self.assertLess(updates[0], updates[1])

    def test_infeasible_branch_is_not_a_negative_witness(self):
        source = "int f(int i){if(i<5){if(i>10){return a[i];}}return 0;}"
        program, counts = facts(source)
        self.assertEqual(counts["infeasible_branch_skipped"], 1)
        self.assertFalse(any(query["operation"] == "array_index"
                             for query in program["queries"]))
        source = "int f(int i){if(i==i){return a[i];}else{return a[i];}}"
        program, counts = facts(source)
        self.assertEqual(counts["infeasible_branch_skipped"], 1)
        self.assertEqual(len([p for use in program["slices"] for p in use["paths"]]), 1)

    def test_composed_effective_function_accumulation_gradient(self):
        a, _ = facts("int f(int i){if(i<10){i=5;i=20;return a[i];}return 0;}")
        b, _ = facts("int f(int i){if(i>=10)return 0;return a[i];}")
        torch.manual_seed(42)
        hidden = torch.randn(3, 512, 4)
        all_relations = [a["queries"], [], b["queries"]]
        full = ComposedRelationHead(4, 2)
        micro = ComposedRelationHead(4, 2)
        micro.load_state_dict(full.state_dict())
        full_loss, _, _ = relation_function_loss(hidden, all_relations, full)
        full_loss.backward()
        first, _, _ = relation_function_loss(hidden[:2], all_relations[:2], micro)
        second, _, _ = relation_function_loss(hidden[2:], all_relations[2:], micro)
        (accumulation_window_loss(torch.zeros(()), first, 2, 3, 1, 2) +
         accumulation_window_loss(torch.zeros(()), second, 1, 3, 1, 2)).backward()
        for name, parameter in full.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(micro.named_parameters())[name].grad)

    def test_query_interaction_cpu_fit_and_noncomplementary_paths(self):
        torch.manual_seed(42)
        sources = ["int f(int i){if(i<10)return a[i];return 0;}",
                   "int f(int i){if(i<10){i=20;return a[i];}return 0;}",
                   "int f(int i){if(i<10){i=5;}else{i=20;}return a[i];}"]
        queries = [[q for q in facts(source)[0]["queries"] if q["operation"] == "array_index"]
                   for source in sources]
        self.assertEqual([[q["label"] for q in group] for group in queries],
                         [[1, 0], [0, 1], [0, 0]])
        hidden = torch.randn(3, 512, 4)  # Fixed CPU states; no source model is loaded.
        head = ComposedRelationHead(4, 8)
        # Reproduce the schema-7 affine readout on its actual pre-MLP path features.
        captured = []
        hook = head.path_query.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        for x in hidden[:2]:
            for query in queries[0]:
                head(x, query)
        hook.remove()
        old_linear = torch.nn.Linear(10 * head.rank + 17, 1)
        old_logits = torch.stack([old_linear(torch.cat((v[:, :40].amin(0),
            v[:, :40].amax(0), v[0, 40:]))) for v in captured]).reshape(2, 2)
        torch.testing.assert_close(old_logits[0, 0] - old_logits[0, 1],
                                   old_logits[1, 0] - old_logits[1, 1])
        optimizer = torch.optim.Adam(head.parameters(), lr=0.03)
        targets = torch.tensor([1., 0., 0., 1., 0., 0.])
        for _ in range(350):
            optimizer.zero_grad()
            logits = torch.stack([head(x, q) for x, group in zip(hidden, queries) for q in group])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            probabilities = torch.stack([head(x, q).sigmoid() for x, group in zip(hidden, queries)
                                         for q in group])
        self.assertTrue(torch.all(probabilities[targets == 1] > .95), probabilities)
        self.assertTrue(torch.all(probabilities[targets == 0] < .05), probabilities)
        for query in queries[2]:
            reversed_query = copy.deepcopy(query)
            reversed_query["paths"].reverse()
            torch.testing.assert_close(head(hidden[2], query), head(hidden[2], reversed_query))
        changed = copy.deepcopy(queries[0][0])
        changed.update(label=0, source="arbitrary analyzer verdict", operand_read=True)
        torch.testing.assert_close(head(hidden[0], changed), head(hidden[0], queries[0][0]))
        x = hidden[1].clone().requires_grad_()
        head.zero_grad()
        head(x, queries[1][0]).backward()
        q = queries[1][0]
        for token in {q["check_token"], q["use_token"], q["target_token"], q["reference_token"],
                      *q["paths"][0]["update_tokens"]}:
            self.assertGreater(float(x.grad[token].abs().sum()), 0)
        self.assertGreater(float(head.path_query[0].weight.grad[:, 5 * head.rank:].abs().sum()), 0)

    def test_pure_operand_reads_and_excluded_contexts(self):
        for expression in ("i+1", "(i)+1", "a[i+1]", "i+i", "~i", "(i<3)+1"):
            source = "int f(int i){if(i<10){return " + expression + ";}return 0;}"
            program, counts = facts(source)
            selected = [q for q in program["queries"] if q["label"] is not None]
            self.assertTrue(selected, (source, counts))
            self.assertTrue(all(q["operation"] == "scalar_read" and q["operand_read"] for q in selected))
            self.assertTrue(all(source[slice(*q["use_span"])] == "i" for q in selected))
            self.assertEqual(len(selected), 4 if expression == "i+i" else 2)
            self.assertEqual(len({(tuple(q["use_span"]), q["branch"]) for q in selected}), len(selected))
        for expression in ("sizeof(i+1)", "sizeof(a[i])", "0 && a[i]", "1 || i",
                           "i ? a[i] : i", "foo(i)", "MACRO(i)", "i++ + i", "&i", "*p+i",
                           "(short)i", "i+UNKNOWN_MACRO"):
            source = "int f(int i,int *p){if(i<10){return " + expression + ";}return 0;}"
            program, _ = facts(source)
            self.assertFalse(program["queries"], source)
        for assignment in ("a[i++]=i+1", "a[foo()]=i+1", "a[i=20]=i+1"):
            p, _ = facts("int f(int i){if(i<10){" + assignment + ";}return 0;}")
            self.assertFalse(p["queries"], assignment)
        source = "int f(int i){if(i<10){i=20;return i+1;}return 0;}"
        p, _ = facts(source)
        self.assertEqual({q["branch"]: q["label"] for q in p["queries"]}, {1: 0, 2: 1})
        self.assertTrue(all(q["use_span"][0] > source.index("i=20") for q in p["queries"]))
        copied, _ = facts("int f(int i){int j;if(i<10){j=i;return j+1;}return 0;}")
        self.assertTrue(any(q["operand_read"] and q["label"] == 1 for q in copied["queries"]))
        shadow, _ = facts("int f(int i){if(i<10){int i=20;return i+1;}return 0;}")
        self.assertFalse(shadow["queries"])
        truncated, _ = facts("int f(int i){/*" + "x"*2100 + "*/if(i<10){return i+1;}return 0;}")
        self.assertTrue(truncated["queries"])
        self.assertTrue(all(q["label"] is None for q in truncated["queries"]))
        # No relaxed suffix, sign or conversion semantics in the check analyzer.
        for literal in ("10U", "-1", "40000"):
            p, _ = facts("int f(int i){if(i<" + literal + ")return i+1;return 0;}")
            self.assertFalse(p["queries"])

    def test_event_limit_keeps_queries_unknown(self):
        source = "int f(int i){if(i<10){i=5;i=20;return i+1;}return 0;}"
        with patch("vulnmechanism.cfg_program.MAX_EVENTS", 3):
            program, counts = facts(source)
        self.assertGreater(counts["too_many_events_for_use"], 0)
        self.assertTrue(program["queries"])
        self.assertTrue(all(q["label"] is None for q in program["queries"]))
        self.assertTrue(all(p["incomplete"] for q in program["queries"] for p in q["paths"]))

    def test_old_query_head_weights_rejected(self):
        head = ComposedRelationHead(4, 8)
        old = {name: value for name, value in head.state_dict().items()
               if not name.startswith("path_query.")}
        old["output.weight"] = torch.zeros(1, 10 * head.rank + 17)
        with self.assertRaises(RuntimeError):
            head.load_state_dict(old)

    def test_old_program_schema_rejected(self):
        source = "int f(int i){if(i<10)return a[i];return 0;}"
        program, _ = facts(source)
        row = record(source, "t0")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "train.queries.jsonl"
            path.write_text("".join(json.dumps(q) + "\n" for q in program["queries"]))
            (root / "audit.json").write_text(json.dumps({
                "program_schema": PROGRAM_SCHEMA - 1,
                "train_queries_sha256": file_sha256(path)}))
            builder = InputBuilder(CharTokenizer(), source_max_length=2048, context_max_length=384)
            with self.assertRaisesRegex(ValueError, "old program query schema"):
                _checked_program_queries(root, [row], builder)
            with self.assertRaisesRegex(ValueError, "current prepared structure schema"):
                collate_graphs([{"node_ids": ["a"], "edges": []}], [[[0]*4]],
                               programs=[{"schema": PROGRAM_SCHEMA - 1, "slices": []}])

    def test_unknown_effect_and_control_flow_scope(self):
        before, _ = facts("int f(int i){foo();if(i<10)return a[i];return 0;}")
        self.assertEqual({q["label"] for q in before["queries"] if
                          q["operation"] == "array_index"}, {0, 1})
        after, _ = facts("int f(int i){if(i<10)return foo(a[i]);return 0;}")
        self.assertFalse(any(q["operation"] == "array_index" for q in after["queries"]))
        loop, counts = facts("int f(int i){for(;;){}if(i<10)return a[i];return 0;}")
        self.assertGreater(counts["unsupported_control_flow"], 0)
        self.assertTrue(all(q["label"] is None for q in loop["queries"] if
                            q["operation"] == "array_index"))

    def test_real_joern_branch_not_source_order(self):
        fragment = json.loads((Path(__file__).parent / "fixtures" /
                               "joern_if_early_return.json").read_text())
        source, graph = fragment["source"], fragment["graph"]
        program, _ = facts(source, graph=graph)
        self.assertEqual({(q["branch"], q["label"]) for q in program["queries"]},
                         {(1, 1), (2, 0)})
        comparison = program["queries"][0]["check_node_id"]
        successors = {e["target"] for e in graph["edges"]
                      if e["kind"] == "CFG" and e["source"] == comparison}
        self.assertEqual(len(successors), 2)
        cut = copy.deepcopy(graph)
        cut["edges"] = [e for e in cut["edges"] if not (
            e["kind"] == "CFG" and e["source"] == comparison)]
        cut_program, _ = facts(source, graph=cut)
        self.assertFalse(any(q["label"] is not None for q in cut_program["queries"]))

    def test_real_joern_compound_index_is_only_an_operand_read(self):
        fixture = json.loads((Path(__file__).parent / "fixtures" / "joern_operand_read.json").read_text())
        self.assertEqual(fixture["split"], "train")
        source, graph = fixture["source"], fixture["graph"]
        program, _ = facts(source, graph=graph)
        self.assertEqual({q["branch"]: q["label"] for q in program["queries"]}, {1: 0, 2: None})
        self.assertIn("args[nargs - 1]", source)
        for query in program["queries"]:
            self.assertEqual(query["operation"], "scalar_read")
            self.assertTrue(query["operand_read"])
            self.assertEqual(source[slice(*query["use_span"])], "nargs")
            self.assertEqual(query["use_span"], [328, 333])
        altered = copy.deepcopy(graph)
        use_id = program["queries"][0]["use_node_id"]
        node = next(n for n in altered["nodes"] if n["id"] == use_id)
        node["code"] = "wrong_source_position"
        invalid, _ = facts(source, graph=altered)
        self.assertFalse(invalid["queries"])

    def test_real_joern_long_control_uses_exact_ast_start(self):
        fragment = json.loads((Path(__file__).parent / "fixtures" /
                               "joern_long_control_span.json").read_text())
        source, graph = fragment["source"], fragment["graph"]
        builder = _Builder(record(source, "long-if"), graph, None, 2048, 0)
        node = next(node for node in walk(builder.function)
                    if node.type == "if_statement" and builder.span(node)[0] == 352)
        self.assertNotIn(("CONTROL_STRUCTURE", builder.span(node)), builder.cache)
        self.assertIsNone(builder.compare({"env": {}}, node))
        self.assertEqual(builder.stats["control_ast_start_recovered"], 1)
        self.assertEqual(builder.stats["missing_check_cfg_site"], 0)
        self.assertEqual(builder.stats["unsupported_check_operand"], 1)

    def test_relation_path_readout_preserves_attribution_and_order(self):
        source = "int f(int i){if(i<10){i=5;}else{i=20;}return a[i];}"
        program, _ = facts(source)
        view = {"node_ids": ["a", "b"], "edges": [(0, 1)]}
        attributes = [[[0]*4, [0]*4]]
        batch = collate_graphs([view], attributes, programs=[program])
        self.assertEqual(len(batch.program.relation_groups[0]), 1)
        self.assertEqual(len(batch.program.relation_groups[0][0][1]), 2)
        reversed_paths = copy.deepcopy(program)
        reversed_paths["slices"][0]["paths"].reverse()
        reversed_batch = collate_graphs([view], attributes, programs=[reversed_paths])
        identical_paths = copy.deepcopy(program)
        identical_paths["slices"][0]["paths"][1] = copy.deepcopy(
            identical_paths["slices"][0]["paths"][0])
        identical_batch = collate_graphs([view], attributes, programs=[identical_paths])
        single_path = copy.deepcopy(identical_paths)
        single_path["slices"][0]["paths"].pop()
        single_batch = collate_graphs([view], attributes, programs=[single_path])
        torch.manual_seed(42)
        model = ProgramCFGEncoder([2]*4, hidden_size=8, steps=2, mode="program_state").eval()
        torch.testing.assert_close(model(batch), model(reversed_batch))
        self.assertFalse(torch.allclose(model(batch), model(identical_batch)))
        self.assertFalse(torch.allclose(model(identical_batch), model(single_batch)))
        # Labels and proof explanations exist only in the separate query file.
        changed = copy.deepcopy(program)
        for query in changed["queries"]:
            query["label"] = None
            query["source"] = "redacted"
        redacted = collate_graphs([view], attributes, programs=[changed])
        torch.testing.assert_close(model(batch), model(redacted))
        for path in program["slices"][0]["paths"]:
            self.assertNotIn("label", path)
            self.assertNotIn("source", path)


if __name__ == "__main__":
    unittest.main()
