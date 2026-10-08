"""Focused CPU tests for scoped relations and the two-stage source pretraining path."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from vulnmechanism.cfg_data import cohort_hash, file_sha256, source_hash
from vulnmechanism.cfg_dependency import (DirectedRelationHead, _checked_relation_rows,
                                         _fixed_relation_predictions, accumulation_window_counts,
                                         accumulation_window_loss, clm_shifted_loss,
                                         evaluate_fixed_relations, function_relations,
                                         lexical_bindings, prepare_supervision,
                                         pretrain_causal_dependency, relation_function_loss)
from vulnmechanism.cfg_network import build_model
from vulnmechanism.model import InputBuilder


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    is_fast = True

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, text, *, truncation=False, max_length=None,
                 return_offsets_mapping=False, **kwargs):
        width = min(len(text), max_length) if truncation and max_length is not None else len(text)
        result = {"input_ids": [(ord(char) % 30) + 2 for char in text[:width]]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(width)]
        return result


def record(source, key="primevul:dep", split="train", label=0):
    return dict(schema_version=9, sample_key=key, dataset="primevul", split=split,
                label=label, raw_source=source, resolved_language="c")


def _u16(source, position):
    return len(source[:position].encode("utf-16-le")) // 2


def scalar_graph(source, definitions, use_at, cfg_edges, ddg_edges):
    """Build a minimal Joern-shaped cached graph from test-known exact spans."""
    nodes = [dict(id="m", label="METHOD", code="f", properties={"kind": "METHOD"}),
             dict(id="b", label="BLOCK", code="{}", properties={"kind": "BLOCK"})]
    edges = [dict(kind="AST", source="m", target="b")]
    for name, position in definitions.items():
        stmt = f"s_{name}"
        nodes.append(dict(id=stmt, label="assignment", code="x=1",
                          properties={"kind": "CALL", "NAME": "<operator>.assignment"}))
        nodes.append(dict(id=name, label="IDENTIFIER", code="x",
                          properties={"kind": "IDENTIFIER", "NAME": "x", "ARGUMENT_INDEX": 1,
                                      "OFFSET": _u16(source, position),
                                      "OFFSET_END": _u16(source, position + 1)}))
        edges.extend((dict(kind="AST", source="b", target=stmt),
                      dict(kind="AST", source=stmt, target=name)))
    nodes.append(dict(id="ret", label="RETURN", code="return x;", properties={"kind": "RETURN"}))
    nodes.append(dict(id="use", label="IDENTIFIER", code="x",
                      properties={"kind": "IDENTIFIER", "NAME": "x", "ARGUMENT_INDEX": 1,
                                  "OFFSET": _u16(source, use_at),
                                  "OFFSET_END": _u16(source, use_at + 1)}))
    edges.extend((dict(kind="AST", source="b", target="ret"),
                  dict(kind="AST", source="ret", target="use")))
    edges.extend(dict(kind="CFG", source=a, target=b) for a, b in cfg_edges)
    edges.extend(dict(kind="DDG", source=a, target=b) for a, b in ddg_edges)
    return {"nodes": nodes, "edges": edges}


def relation_rows(source, graph, *, budget=2048):
    builder = InputBuilder(CharTokenizer(), source_max_length=budget, context_max_length=384)
    return function_relations(record(source), graph, builder.tokenizer,
                              source_max_length=budget, prefix_tokens=len(builder.source_prefix))


class BindingTests(unittest.TestCase):
    def test_assignment_kill_is_the_only_negative(self):
        source = "int f(){int x=0; x=1; x=2; return x;}"
        first, second, use = source.index("x=1"), source.index("x=2"), source.rindex("x")
        graph = scalar_graph(source, {"d1": first, "d2": second}, use,
                             [("m", "s_d1"), ("s_d1", "s_d2"), ("s_d2", "ret")],
                             [("d2", "use")])
        rows, counts = relation_rows(source, graph)
        self.assertEqual([(row["definition_node_id"], row["label"]) for row in rows],
                         [("d1", 0), ("d2", 1)])
        self.assertEqual(rows[0]["source"], "cfg_killed_same_binding")
        self.assertEqual(counts["proved_negative_candidates"], 1)

    def test_branch_keeps_two_reaching_definitions(self):
        source = "int f(int c){int x=0; if(c){x=1;}else{x=2;} return x;}"
        first, second, use = source.index("x=1"), source.index("x=2"), source.rindex("x")
        graph = scalar_graph(source, {"d1": first, "d2": second}, use,
                             [("m", "s_d1"), ("m", "s_d2"),
                              ("s_d1", "ret"), ("s_d2", "ret")],
                             [("d1", "use"), ("d2", "use")])
        rows, counts = relation_rows(source, graph)
        self.assertEqual({row["definition_node_id"] for row in rows}, {"d1", "d2"})
        self.assertEqual({row["label"] for row in rows}, {1})
        self.assertEqual(counts["proved_negative_candidates"], 0)

    def test_missing_ddg_edge_alone_is_not_a_negative(self):
        source = "int f(int c){int x=0; if(c){x=1;}else{x=2;} return x;}"
        first, second, use = source.index("x=1"), source.index("x=2"), source.rindex("x")
        graph = scalar_graph(source, {"d1": first, "d2": second}, use,
                             [("m", "s_d1"), ("m", "s_d2"),
                              ("s_d1", "ret"), ("s_d2", "ret")], [("d2", "use")])
        rows, counts = relation_rows(source, graph)
        self.assertEqual([(row["definition_node_id"], row["label"]) for row in rows],
                         [("d2", 1)])
        self.assertEqual(counts["proved_negative_candidates"], 0)
        self.assertGreater(counts["unproved_non_edges"], 0)

    def test_shadowed_local_is_a_distinct_binding(self):
        source = "int f(){int x=0; x=1; {int x=0; x=2; return x;}}"
        first, second, use = source.index("x=1"), source.index("x=2"), source.rindex("x")
        graph = scalar_graph(source, {"outer": first, "inner": second}, use,
                             [("m", "s_outer"), ("s_outer", "s_inner"), ("s_inner", "ret")],
                             [("outer", "use"), ("inner", "use")])
        rows, counts = relation_rows(source, graph)
        self.assertEqual([row["definition_node_id"] for row in rows], ["inner"])
        self.assertEqual(counts["ddg_binding_conflict"], 1)

    def test_arguments_have_separate_bindings_and_macro_calls_are_skipped(self):
        source = "int f(int a,int b){int x=0; x=a; MACRO(a,b); return b;}"
        bindings, error = lexical_bindings(source, "c")
        self.assertIsNone(error)
        a_call = source.index("a,b", source.index("MACRO"))
        b_call = a_call + 2
        self.assertNotEqual(bindings[(a_call, a_call + 1)]["binding"],
                            bindings[(b_call, b_call + 1)]["binding"])
        self.assertEqual(bindings[(a_call, a_call + 1)]["role"], "call_or_macro_ambiguous")
        self.assertEqual(bindings[(b_call, b_call + 1)]["role"], "call_or_macro_ambiguous")

    def test_utf16_offsets_truncation_and_unreliable_positions(self):
        source = "int f(){/*😀*/int x=0; x=1; return x;}"
        definition, use = source.index("x=1"), source.rindex("x")
        graph = scalar_graph(source, {"d": definition}, use,
                             [("m", "s_d"), ("s_d", "ret")], [("d", "use")])
        rows, _ = relation_rows(source, graph)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["definition_span"], [definition, definition + 1])
        self.assertEqual(rows[0]["use_span"], [use, use + 1])
        truncated, counts = relation_rows(source, graph, budget=definition + 1)
        self.assertEqual(truncated, [])
        self.assertGreater(counts["alignment_outside_visible_source"], 0)
        graph["nodes"][-1]["properties"]["OFFSET"] += 1
        invalid, counts = relation_rows(source, graph)
        self.assertEqual(invalid, [])
        self.assertGreater(counts["alignment_position_mismatch"], 0)

    def test_prepare_reads_original_cache_and_exports_train_only(self):
        from vulnmechanism.cfg_data import GRAPH_SCHEMA
        from vulnmechanism.cpg import JOERN_SOURCE_PREPROCESSING_VERSION

        source = "int f(){int x=0; x=1; x=2; return x;}"
        first, second, use = source.index("x=1"), source.index("x=2"), source.rindex("x")
        graph = scalar_graph(source, {"d1": first, "d2": second}, use,
                             [("m", "s_d1"), ("s_d1", "s_d2"), ("s_d2", "ret")],
                             [("d2", "use")])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, graphs, reference, output = (root / "dataset.jsonl", root / "graphs.jsonl",
                                                   root / "c", root / "supervision")
            rows = [record(source, "train:1"), record(source, "valid:1", split="valid"),
                    record(source, "test:1", split="test")]
            dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
            graphs.write_text("".join(json.dumps({
                "sample_key": row["sample_key"], "dataset": row["dataset"],
                "split": row["split"], "label": row["label"],
                "source_sha256": source_hash(source),
                "graph_schema_version": GRAPH_SCHEMA,
                "preprocessing_version": JOERN_SOURCE_PREPROCESSING_VERSION,
                "preprocessing_applied": False,
                "original_source_sha256": source_hash(source),
                "parsed_source_sha256": source_hash(source), "graph": graph}) + "\n"
                for row in rows))
            original = dataset.read_bytes(), graphs.read_bytes()
            reference.mkdir()
            (reference / "config.json").write_text(json.dumps({
                "source_dataset": "primevul", "source_max_length": 2048,
                "model_path": "tiny-test-only", "dataset": str(dataset),
                "graph_file_sha256": file_sha256(graphs),
                "cohort_sha256": cohort_hash(rows),
                "train_cohort_sha256": cohort_hash(rows[:1])}))
            audit = prepare_supervision(str(dataset), str(graphs), str(reference),
                                        str(output), CharTokenizer())
            relations = [json.loads(line) for line in (output / "relations.jsonl").read_text().splitlines()]
            self.assertEqual({row["sample_key"] for row in relations}, {"train:1"})
            self.assertEqual({row["label"] for row in relations}, {0, 1})
            self.assertEqual(len((output / "function_coverage.jsonl").read_text().splitlines()), 1)
            self.assertEqual(audit["counts"]["functions"], 1)
            valid_relations, _ = function_relations(rows[1], graph, CharTokenizer(),
                                                   prefix_tokens=len(InputBuilder(
                                                       CharTokenizer(), source_max_length=2048,
                                                       context_max_length=384).source_prefix),
                                                   for_valid=True)
            self.assertEqual({relation["split"] for relation in valid_relations}, {"valid"})
            with self.assertRaisesRegex(ValueError, "train split"):
                function_relations(rows[1], graph, CharTokenizer())
            with self.assertRaisesRegex(ValueError, "valid split"):
                function_relations(rows[2], graph, CharTokenizer(), for_valid=True)
            relfile = output / "relations.jsonl"
            relfile.write_text(json.dumps({**relations[0], "sample_key": rows[1]["sample_key"],
                                           "split": "valid"}) + "\n")
            changed_audit = json.loads((output / "audit.json").read_text())
            changed_audit["relations_sha256"] = file_sha256(relfile)
            (output / "audit.json").write_text(json.dumps(changed_audit))
            with self.assertRaisesRegex(ValueError, "outside the original C train cohort"):
                _checked_relation_rows(str(output), rows[:1], InputBuilder(
                    CharTokenizer(), source_max_length=2048, context_max_length=384))
            self.assertEqual((dataset.read_bytes(), graphs.read_bytes()), original)

    def test_alias_field_and_compound_update_are_not_accepted(self):
        source = "int f(){int x=0; x=1; int *p=&x; x++; return x;}"
        first, use = source.index("x=1"), source.rindex("x")
        graph = scalar_graph(source, {"d": first}, use,
                             [("m", "s_d"), ("s_d", "ret")], [("d", "use")])
        rows, counts = relation_rows(source, graph)
        self.assertEqual(rows, [])
        self.assertGreater(counts["skip_alias_macro_or_unmodelled_binding"], 0)
        source2 = "int f(){struct S s; return s.field;}"
        bindings, _ = lexical_bindings(source2, "c")
        field_base = source2.index("s.field")
        self.assertEqual(bindings[(field_base, field_base + 1)]["role"], "field_access")
        reference_source = "int f(){int x=0; x=1; int &r=x; return x;}"
        reference_bindings, error = lexical_bindings(reference_source, "cpp")
        self.assertIsNone(error)
        captured = reference_source.index("=x") + 1
        self.assertEqual(reference_bindings[(captured, captured + 1)]["role"],
                         "alias_or_address")


class LossTests(unittest.TestCase):
    def test_clm_shift_and_prefix_padding_mask(self):
        hidden = torch.tensor([[[1., 0.], [0., 1.], [1., 1.], [0., 1.], [1., 0.]]],
                              requires_grad=True)
        ids = torch.tensor([[2, 3, 1, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        weight = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        loss, count = clm_shifted_loss(hidden, ids, mask, weight, prefix_tokens=1, chunk_size=1)
        expected = torch.nn.functional.cross_entropy(torch.stack((hidden[0, 0], hidden[0, 1])) @ weight.T,
                                                      torch.tensor([3, 1]))
        self.assertEqual(count, 2)
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))
        loss.backward()
        self.assertGreater(hidden.grad[0, :2].abs().sum().item(), 0)
        self.assertEqual(hidden.grad[0, 2:].abs().sum().item(), 0)

    def test_relation_function_means_and_lora_gradient(self):
        torch.manual_seed(4)
        lora = nn.Linear(3, 3, bias=False)
        head = DirectedRelationHead(3, rank=2)
        hidden = lora(torch.randn(2, 5, 3))
        pairs = [[{"definition_token": 1, "use_token": 2, "label": 1},
                  {"definition_token": 1, "use_token": 3, "label": 0}],
                 [{"definition_token": 0, "use_token": 4, "label": 1}]]
        loss, labels, logits = relation_function_loss(hidden, pairs, head)
        first = torch.nn.functional.binary_cross_entropy_with_logits(
            head(hidden[0, [1, 1]], hidden[0, [2, 3]]), torch.tensor([1., 0.]))
        second = torch.nn.functional.binary_cross_entropy_with_logits(
            head(hidden[1, [0]], hidden[1, [4]]), torch.tensor([1.]))
        self.assertTrue(torch.allclose(loss, (first + second) / 2))
        self.assertEqual(labels, [1, 0, 1])
        self.assertEqual(len(logits), 3)
        loss.backward()
        self.assertGreater(lora.weight.grad.abs().sum().item(), 0)
        self.assertGreater(head.definition.weight.grad.abs().sum().item(), 0)
        self.assertGreater(head.use.weight.grad.abs().sum().item(), 0)


    def test_window_relation_mean_and_unclipped_gradients_across_partitions(self):
        inputs = torch.arange(75, dtype=torch.float32).reshape(5, 5, 3) / 25
        cases = (([0, 2, 0, 1, 3], [[0], [1, 2], [3, 4]]),
                 ([1, 2, 3, 1, 2], [[0, 1], [2, 3, 4]]),
                 ([0, 0, 0, 0, 0], [[0, 1, 2], [3, 4]]),
                 ([2, 0, 1, 0, 0], [[0, 1], [2, 3], [4]]))
        for relation_counts, partitions in cases:
            with self.subTest(relation_counts=relation_counts, partitions=partitions):
                pairs = [[{"definition_token": 0, "use_token": 2,
                           "label": index % 2} for _ in range(count)]
                         for index, count in enumerate(relation_counts)]
                def modules():
                    torch.manual_seed(19)
                    return nn.Linear(3, 3, bias=False), DirectedRelationHead(3, rank=2)
                adapter, head = modules()
                hidden = adapter(inputs)
                lm_by_function = hidden.square().mean(dim=(1, 2))
                relation, _, _ = relation_function_loss(hidden, pairs, head)
                expected = lm_by_function.mean() + relation
                expected.backward()
                expected_gradients = [p.grad.clone() if p.grad is not None else None
                                      for p in (*adapter.parameters(), *head.parameters())]
                adapter, head = modules()
                actual = 0.0
                total = len(pairs)
                effective = sum(bool(item) for item in pairs)
                for indices in partitions:
                    hidden = adapter(inputs[indices])
                    lm = hidden.square().mean()
                    relation, _, _ = relation_function_loss(hidden, [pairs[i] for i in indices], head)
                    micro = accumulation_window_loss(lm, relation, len(indices), total,
                                                     sum(bool(pairs[i]) for i in indices), effective)
                    actual += float(micro.detach())
                    micro.backward()
                self.assertAlmostEqual(actual, float(expected.detach()), places=6)
                for expected_grad, parameter in zip(expected_gradients,
                                                    (*adapter.parameters(), *head.parameters())):
                    if expected_grad is None:
                        self.assertIsNone(parameter.grad)
                    else:
                        self.assertTrue(torch.allclose(parameter.grad, expected_grad, atol=1e-6))
                # The CLM term keeps the existing m/M scaling, even when K differs.
                self.assertAlmostEqual(sum(float(lm_by_function[indices].mean().detach()) *
                                           len(indices) / total for indices in partitions),
                                       float(lm_by_function.mean().detach()), places=6)

    def test_window_counts_include_batch_size_two_and_partial_final_window(self):
        train = [{"sample_key": str(index)} for index in range(5)]
        order = list(range(5))
        supervision = {"0": [{"label": 1}], "3": [{"label": 0}]}
        self.assertEqual(accumulation_window_counts(order, train, supervision, 2, 2),
                         [(4, 2), (4, 2), (1, 0)])
        self.assertEqual(accumulation_window_counts(order, train, {}, 2, 2),
                         [(4, 0), (4, 0), (1, 0)])
        self.assertEqual(accumulation_window_counts(order, train,
                         {str(index): [{}] for index in order}, 2, 2),
                         [(4, 4), (4, 4), (1, 1)])
        self.assertEqual(float(accumulation_window_loss(torch.tensor(2.),
                                                     torch.tensor(9.), 1, 1, 0, 0)), 2.0)

    def test_fixed_relation_scoring_does_not_update_encoder_or_head(self):
        source = "int f(){int x=0; x=1; return x;}"
        rows = [record(source, "t0"), record(source, "v0", split="valid")]
        builder = InputBuilder(CharTokenizer(), source_max_length=2048, context_max_length=384)
        offset = len(builder.source_prefix)
        relations = {"t0": [{"sample_key": "t0", "split": "train", "label": 0,
                              "definition_token": offset, "use_token": offset + 1}],
                     "v0": [{"sample_key": "v0", "split": "valid", "label": 1,
                              "definition_token": offset, "use_token": offset + 1}]}
        encoder, head = TinyEncoder(), DirectedRelationHead(4, rank=2)
        before = [parameter.detach().clone() for parameter in
                  (*encoder.parameters(), *head.parameters())]
        predictions, summary = _fixed_relation_predictions(
            encoder, head, builder, rows, relations, 2, torch.device("cpu"))
        self.assertEqual(encoder.calls, 1)
        self.assertFalse(encoder.training)
        self.assertFalse(head.training)
        self.assertEqual([row["sample_key"] for row in predictions], ["t0", "v0"])
        self.assertEqual(summary["fixed_model"]["threshold"], 0.5)
        self.assertEqual(summary["all_positive_reference"]["recall"], 1.0)
        self.assertTrue(all(torch.equal(old, parameter) for old, parameter in zip(
            before, (*encoder.parameters(), *head.parameters()))))
        self.assertTrue(all(parameter.grad is None for parameter in
                            (*encoder.parameters(), *head.parameters())))


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.embedding.weight.requires_grad_(False)
        self.adapter = nn.Linear(4, 4, bias=False)
        self.calls = 0

    def forward(self, input_ids, attention_mask, use_cache=False):
        self.calls += 1
        raw = self.embedding(input_ids)
        return types.SimpleNamespace(last_hidden_state=raw + self.adapter(raw))


class TinyBase:
    InputBuilder = InputBuilder
    AutoTokenizer = CharTokenizer

    def __init__(self):
        self.initial = []
        self.encoders = []

    def _resolve_device(self, device):
        return torch.device("cpu")

    def _seed_everything(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def _build_lora_encoder(self, *args, **kwargs):
        encoder = TinyEncoder()
        self.initial.append(encoder.adapter.weight.detach().clone())
        self.encoders.append(encoder)
        return encoder, 4

    def get_peft_model_state_dict(self, encoder):
        return {"adapter.weight": encoder.adapter.weight.detach().clone()}

    def set_peft_model_state_dict(self, encoder, state):
        with torch.no_grad():
            encoder.adapter.weight.copy_(state["adapter.weight"])

    def _cpu_state(self, state):
        return {key: value.detach().clone().cpu() for key, value in state.items()}


class StageTests(unittest.TestCase):
    def test_mocked_one_epoch_common_start_and_single_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = record("int f(){return 1;}", "primevul:1")
            valid = record("int g(){return 0;}", "primevul:2", split="valid")
            dataset = root / "data.jsonl"
            dataset.write_text("".join(json.dumps(row) + "\n" for row in (train, valid)))
            graphs = root / "graphs.jsonl"
            graphs.write_text("cached graph identity\n")
            reference = root / "c"
            reference.mkdir()
            config = {"dataset": str(dataset), "graphs": str(graphs), "source_dataset": "primevul",
                      "source_max_length": 2048, "seed": 42, "model_path": str(root / "qwen"),
                      "cohort_sha256": cohort_hash([train, valid]),
                      "train_cohort_sha256": cohort_hash([train]),
                      "graph_file_sha256": file_sha256(graphs), "lora_r": 16,
                      "lora_alpha": 32, "lora_dropout": .05, "learning_rate": .0002,
                      "graph_learning_rate": .001, "batch_size": 1,
                      "gradient_accumulation": 1, "weight_decay": .01, "log_every": 1}
            (reference / "config.json").write_text(json.dumps(config))
            supervision = root / "supervision"
            supervision.mkdir()
            builder = InputBuilder(CharTokenizer(), source_max_length=2048, context_max_length=384)
            offset = len(builder.source_prefix)
            relations = [dict(sample_key=train["sample_key"], split="train",
                              source_sha256=source_hash(train["raw_source"]),
                              definition_node_id=f"d{index}", use_node_id=f"u{index}",
                              definition_token=offset + index,
                              use_token=offset + index + 1,
                              definition_span=[index, index + 1],
                              use_span=[index + 1, index + 2],
                              definition_coordinates={"OFFSET": index, "OFFSET_END": index + 1},
                              use_coordinates={"OFFSET": index + 1, "OFFSET_END": index + 2},
                              label=label)
                         for index, label in enumerate((1, 0))]
            relfile = supervision / "relations.jsonl"
            relfile.write_text("".join(json.dumps(row) + "\n" for row in relations))
            (supervision / "audit.json").write_text(json.dumps({
                "reference_run_dir": str(reference.resolve()),
                "train_cohort_sha256": config["train_cohort_sha256"],
                "relations_sha256": file_sha256(relfile),
                "counts": {"functions_with_relations": 1,
                           "selected_positive": 1, "selected_negative": 1}}))
            base = TinyBase()
            lm_weight = torch.randn(32, 4)
            with patch("vulnmechanism.cfg_dependency._load_qwen_lm_weight", return_value=lm_weight):
                result = pretrain_causal_dependency(str(reference), str(supervision),
                                                    str(root / "stage1"), base=base)
            self.assertEqual(result["lm_pretrain"]["optimizer_steps"], 1)
            self.assertEqual(result["dep_pretrain"]["optimizer_steps"], 1)
            self.assertEqual(base.encoders[0].calls, 1)
            self.assertEqual(base.encoders[1].calls, 1)
            self.assertTrue(torch.equal(base.initial[0], base.initial[1]))
            self.assertEqual(result["dep_pretrain"]["relation_pairs"], 2)
            self.assertEqual(result["dep_pretrain"]["total_functions"], 1)
            self.assertEqual(result["dep_pretrain"]["effective_relation_functions"], 1)
            self.assertEqual(result["dep_pretrain"]["relation_coverage"], 1.0)
            self.assertEqual(result["dep_pretrain"]["relation_positive"], 1)
            self.assertEqual(result["dep_pretrain"]["relation_negative"], 1)
            self.assertEqual(result["lm_pretrain"]["relation_pairs"], 0)
            self.assertEqual(result["lm_pretrain"]["effective_relation_functions"], 0)
            self.assertTrue((root / "stage1" / "lm_pretrain" / "last.pt").exists())
            self.assertTrue((root / "stage1" / "dep_pretrain" / "last.pt").exists())
            control={row['sample_key']:[dict(condition_token=offset,use_token=offset+2,branch=b,
                        case_token=None,label=b) for b in (0,1)] for row in (train,valid)}
            with patch('vulnmechanism.cfg_control.load_control',return_value=(control,{'schema':1,'queries_sha256':'tiny'})), \
                 patch('vulnmechanism.cfg_dependency._load_qwen_lm_weight',return_value=lm_weight):
                result=pretrain_causal_dependency(str(reference),str(supervision),str(root/'project'),
                    modes=('control_pretrain',),base=base,control_dir=str(root/'control'),
                    reference_pretrain_dir=str(root/'stage1'),control_gradient_policy='project')
            self.assertEqual(base.encoders[-1].calls,1)
            self.assertEqual(result['control_pretrain']['optimizer_steps'],1)
            saved=torch.load(root/'project/control_pretrain/last.pt',weights_only=False)
            fresh=TinyEncoder();base.set_peft_model_state_dict(fresh,saved['adapter_state'])
            self.assertFalse(torch.equal(base.initial[-1],fresh.adapter.weight))
            head=DirectedRelationHead(4,32);head.load_state_dict(saved['relation_head_state'])
            predictions,summary=_fixed_relation_predictions(fresh,head,builder,[train],
                {train['sample_key']:relations},1,torch.device('cpu'))
            self.assertEqual(len(predictions),2)
            self.assertTrue(0 <= summary['fixed_model']['accuracy'] <= 1)


    def test_stage1_window_logging_uses_effective_functions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = [record("int f(){return 1;}", f"t{index}") for index in range(5)]
            dataset, graphs, reference = root / "data.jsonl", root / "graphs.jsonl", root / "c"
            dataset.write_text("".join(json.dumps(row) + "\n" for row in train))
            graphs.write_text("graph identity\n")
            reference.mkdir()
            config = {"dataset": str(dataset), "graphs": str(graphs), "source_dataset": "primevul",
                      "source_max_length": 2048, "seed": 42, "model_path": str(root / "qwen"),
                      "cohort_sha256": cohort_hash(train),
                      "train_cohort_sha256": cohort_hash(train),
                      "graph_file_sha256": file_sha256(graphs), "lora_r": 16,
                      "lora_alpha": 32, "lora_dropout": .05, "learning_rate": .0002,
                      "graph_learning_rate": .001, "batch_size": 2,
                      "gradient_accumulation": 2, "weight_decay": .01, "log_every": 1}
            (reference / "config.json").write_text(json.dumps(config))
            supervision = {"t3": [{"label": 1}], "t2": [{"label": 0}]}
            audit = {"reference_run_dir": str(reference.resolve()),
                     "train_cohort_sha256": config["train_cohort_sha256"],
                     "relations_sha256": "test", "counts": {"selected_positive": 1,
                                                           "selected_negative": 1}}
            def fake_relation(hidden, batch, head):
                present = [relations[0]["label"] for relations in batch if relations]
                dep = hidden.float().sum() * 0
                if present:
                    dep = dep + (2.0 if present[0] else 4.0)
                return dep, present, [0.0] * len(present)
            base = TinyBase()
            with (patch("vulnmechanism.cfg_dependency._checked_relation_rows",
                        return_value=(supervision, audit)),
                  patch("vulnmechanism.cfg_dependency._load_qwen_lm_weight",
                        return_value=torch.randn(32, 4)),
                  patch("vulnmechanism.cfg_dependency.relation_function_loss",
                        side_effect=fake_relation), redirect_stdout(io.StringIO())):
                summary = pretrain_causal_dependency(str(reference), str(root / "unused"),
                                                     str(root / "stage1"),
                                                     modes=("dep_pretrain",), base=base)["dep_pretrain"]
            self.assertEqual(base.encoders[0].calls, 3)
            self.assertEqual(summary["optimizer_steps"], 2)
            self.assertEqual(summary["total_functions"], 5)
            self.assertEqual(summary["effective_relation_functions"], 2)
            self.assertEqual(summary["relation_coverage"], .4)
            self.assertEqual(summary["relation_loss"], 3.0)
            self.assertEqual((summary["relation_positive"], summary["relation_negative"]), (1, 1))
            events = [json.loads(line) for line in (root / "stage1" / "dep_pretrain" /
                      "history.jsonl").read_text().splitlines()]
            self.assertEqual([event["effective_relation_functions_so_far"] for event in
                              events if event["event"] == "step"], [2, 2])
            self.assertEqual([event["relation_loss_mean_so_far"] for event in
                              events if event["event"] == "step"], [3.0, 3.0])

    def test_fixed_model_evaluation_uses_train_and_cached_valid_only(self):
        from vulnmechanism.cfg_data import GRAPH_SCHEMA
        from vulnmechanism.cpg import JOERN_SOURCE_PREPROCESSING_VERSION

        source = "int f(){int x=0; x=1; x=2; return x;}"
        first, second, use = source.index("x=1"), source.index("x=2"), source.rindex("x")
        graph = scalar_graph(source, {"d1": first, "d2": second}, use,
                             [("m", "s_d1"), ("s_d1", "s_d2"), ("s_d2", "ret")],
                             [("d2", "use")])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [record(source, "t0"), record(source, "v0", split="valid"),
                    record(source, "x0", split="test")]
            dataset, graphs, reference = root / "data.jsonl", root / "graphs.jsonl", root / "c"
            dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
            graphs.write_text("".join(json.dumps({
                "sample_key": row["sample_key"], "dataset": row["dataset"],
                "split": row["split"], "label": row["label"],
                "source_sha256": source_hash(source), "graph_schema_version": GRAPH_SCHEMA,
                "preprocessing_version": JOERN_SOURCE_PREPROCESSING_VERSION,
                "preprocessing_applied": False, "original_source_sha256": source_hash(source),
                "parsed_source_sha256": source_hash(source), "graph": graph}) + "\n"
                for row in rows))
            reference.mkdir()
            c_config = {"source_dataset": "primevul", "source_max_length": 2048,
                        "seed": 42, "model_path": "tiny-test-only", "dataset": str(dataset),
                        "graphs": str(graphs), "graph_file_sha256": file_sha256(graphs),
                        "cohort_sha256": cohort_hash(rows),
                        "train_cohort_sha256": cohort_hash(rows[:1]),
                        "valid_cohort_sha256": cohort_hash(rows[1:2]),
                        "lora_r": 16, "lora_alpha": 32, "lora_dropout": .05}
            (reference / "config.json").write_text(json.dumps(c_config))
            supervision_dir = root / "supervision"
            audit = prepare_supervision(str(dataset), str(graphs), str(reference),
                                        str(supervision_dir), CharTokenizer())
            stage1 = root / "stage1"
            (stage1 / "dep_pretrain").mkdir(parents=True)
            config = {"c_config": c_config, "reference_run_dir": str(reference.resolve()),
                      "supervision_dir": str(supervision_dir.resolve()),
                      "relations_sha256": audit["relations_sha256"], "relation_rank": 2}
            (stage1 / "config.json").write_text(json.dumps(config))
            checkpoint = {"mode": "dep_pretrain", "pretrain_config": config,
                          "adapter_state": TinyBase().get_peft_model_state_dict(TinyEncoder()),
                          "relation_head_state": DirectedRelationHead(4, 2).state_dict()}
            path = stage1 / "dep_pretrain" / "last.pt"
            torch.save(checkpoint, path)
            (path.parent / "complete.json").write_text(json.dumps({
                "mode": "dep_pretrain", "checkpoint_sha256": file_sha256(path)}))
            base = TinyBase()
            with redirect_stdout(io.StringIO()):
                report = evaluate_fixed_relations(str(stage1), str(root / "evaluation"),
                                                  batch_size=2, device="cpu", base=base)
            self.assertEqual(set(report["splits"]), {"train", "valid"})
            self.assertEqual([report["splits"][split]["relation_pairs"]
                              for split in ("train", "valid")], [2, 2])
            self.assertEqual(base.encoders[0].calls, 2)
            self.assertTrue(all(parameter.grad is None for parameter in
                                base.encoders[0].parameters()))
            self.assertTrue(torch.equal(base.encoders[0].adapter.weight,
                                        checkpoint["adapter_state"]["adapter.weight"]))
            self.assertFalse((root / "evaluation" / "test.predictions.jsonl").exists())
            for split, key in (("train", "t0"), ("valid", "v0")):
                predictions = [json.loads(line) for line in (root / "evaluation" /
                               f"{split}.predictions.jsonl").read_text().splitlines()]
                self.assertEqual({row["sample_key"] for row in predictions}, {key})
                self.assertEqual({row["split"] for row in predictions}, {split})
                self.assertEqual({row["threshold"] for row in predictions}, {0.5})
                self.assertIsNotNone(report["splits"][split]["fixed_model"])
                self.assertIsNotNone(report["splits"][split]["all_positive_reference"])
            with self.assertRaises(FileExistsError):
                evaluate_fixed_relations(str(stage1), str(root / "evaluation"), base=base)

    def test_stage2_uses_original_c_architecture_and_loads_only_lora(self):
        class Source(nn.Module):
            def __init__(self, model_path, *, device, **kwargs):
                super().__init__()
                self.encoder = TinyEncoder()
                self.task_modules = nn.ModuleDict({"classifier": nn.Linear(4, 1)})
                self.to(device)

        base = types.SimpleNamespace(SequenceVulnerabilityClassifier=Source)
        size = [3, 3, 3, 3]
        states = []
        for variant in ("cfg", "lm_pretrain_cfg", "dep_pretrain_cfg"):
            torch.manual_seed(42)
            model = build_model(base, dict(model_path="unused", lora_r=16, lora_alpha=32,
                                           lora_dropout=.05, variant=variant,
                                           graph_hidden_size=8, graph_steps=5),
                                size, torch.device("cpu"), training=True)
            states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
            if variant == "dep_pretrain_cfg":
                before = {key: value.detach().clone() for key, value in
                          model.task_modules.state_dict().items()}
                model.encoder.adapter.weight.data.fill_(0.75)
                self.assertTrue(all(torch.equal(before[key], value) for key, value in
                                    model.task_modules.state_dict().items()))
        self.assertEqual(states[0].keys(), states[1].keys())
        self.assertEqual(states[1].keys(), states[2].keys())
        for key in states[0]:
            self.assertTrue(torch.equal(states[0][key], states[1][key]), key)
            self.assertTrue(torch.equal(states[1][key], states[2][key]), key)


    def test_official_c_run_entry_and_valid_test_comparison(self):
        from tests.test_cfg_ablation import (TinyEncoder, TinyInputBuilder, fake_base,
                                             fixture_graph, fixture_rows, write_jsonl)
        from vulnmechanism import cfg_data as data
        from vulnmechanism import cfg_experiment as exp

        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            dataset, graphs = root / "dataset.jsonl", root / "graphs.jsonl"
            c_run, stage1, stage2 = root / "c", root / "stage1", root / "stage2"
            rows = fixture_rows()
            write_jsonl(dataset, rows)
            write_jsonl(graphs, [dict(data.identity(row), graph_schema_version=data.GRAPH_SCHEMA,
                                      preprocessing_version=data.JOERN_SOURCE_PREPROCESSING_VERSION,
                                      preprocessing_applied=False,
                                      original_source_sha256=data.source_hash(row["raw_source"]),
                                      parsed_source_sha256=data.source_hash(row["raw_source"]),
                                      graph=fixture_graph()) for row in rows])
            api = fake_base()
            common = ["--dataset", str(dataset), "--graphs", str(graphs),
                      "--model-path", "tiny-test-only", "--device", "cpu", "--epochs", "1",
                      "--batch-size", "2", "--gradient-accumulation", "2",
                      "--graph-hidden-size", "8", "--graph-steps", "2"]
            exp.run_experiment(exp.parser().parse_args([
                "run", *common, "--output-dir", str(c_run), "--variants", "cfg"]), base=api)
            c_config = json.loads((c_run / "config.json").read_text())
            stage1.mkdir()
            pretrain_config = {"c_config": c_config, "reference_run_dir": str(c_run.resolve()),
                               "pretrain_epochs": 1, "relation_alpha": 1.0,
                               "relations_sha256": "test-relation-identity"}
            (stage1 / "config.json").write_text(json.dumps(pretrain_config))
            for mode in ("lm_pretrain", "dep_pretrain"):
                folder = stage1 / mode
                folder.mkdir()
                torch.manual_seed(42)
                adapter = TinyEncoder().state_dict()
                torch.save({"mode": mode, "pretrain_config": pretrain_config,
                            "adapter_state": adapter}, folder / "last.pt")
                (folder / "complete.json").write_text(json.dumps({
                    "mode": mode, "checkpoint_sha256": file_sha256(folder / "last.pt")}))
            trained = exp.run_experiment(exp.parser().parse_args([
                "run", *common, "--output-dir", str(stage2), "--variants",
                "lm_pretrain_cfg", "dep_pretrain_cfg", "--pretrain-dir", str(stage1),
                "--reference-run-dir", str(c_run)]), base=api)
            self.assertEqual(set(trained["changes_vs_cfg"]),
                             {"lm_pretrain_cfg", "dep_pretrain_cfg"})
            self.assertEqual(trained["initialization_policy"], "source_dependency_pretraining")
            self.assertEqual(json.loads((stage2 / "vocabulary.json").read_text()),
                             json.loads((c_run / "vocabulary.json").read_text()))
            TinyInputBuilder.allow_test = True
            try:
                exp.evaluate_run(exp.parser().parse_args([
                    "eval", "--run-dir", str(c_run), "--split", "test",
                    "--variants", "cfg", "--device", "cpu"]), base=api)
                evaluation = exp.parser().parse_args([
                    "eval", "--run-dir", str(stage2), "--split", "test", "--variants",
                    "lm_pretrain_cfg", "dep_pretrain_cfg", "--device", "cpu"])
                with patch.object(exp, "select_threshold", side_effect=AssertionError("test tuning")):
                    tested = exp.evaluate_run(evaluation, base=api)
                self.assertEqual(set(tested["changes_vs_cfg"]),
                                 {"lm_pretrain_cfg", "dep_pretrain_cfg"})
                self.assertEqual({row["split"] for row in data.read_jsonl(
                    stage2 / "dep_pretrain_cfg" / "test.predictions.jsonl")}, {"test"})
            finally:
                TinyInputBuilder.allow_test = False

    def test_stage2_training_applies_last_lora_before_fresh_optimizer(self):
        from vulnmechanism import cfg_experiment as exp

        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(1, 1, bias=False)
                self.task_modules = nn.ModuleDict({
                    "cfg_encoder": nn.Linear(1, 1),
                    "cfg_classifier": nn.Linear(1, 1, bias=False)})

            def forward(self, ids, mask, graph):
                source = self.encoder(ids.float())
                graph_logit = self.task_modules["cfg_classifier"](
                    self.task_modules["cfg_encoder"](ids.float()))
                return (source + graph_logit).squeeze(-1)

        class SmallBase:
            def __init__(self):
                self.loaded = []

            def _seed_everything(self, seed):
                torch.manual_seed(seed)

            def set_peft_model_state_dict(self, encoder, state):
                with torch.no_grad():
                    encoder.weight.copy_(state["weight"])
                self.loaded.append(encoder.weight.detach().clone())

            def get_peft_model_state_dict(self, encoder):
                return {"weight": encoder.weight.detach().clone()}

            def _cpu_state(self, state):
                return {key: value.detach().cpu().clone() for key, value in state.items()}

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = [record("int f(){return 0;}", "t0", label=0),
                    record("int g(){return 1;}", "t1", label=1),
                    record("int h(){return 0;}", "v0", split="valid", label=0),
                    record("int i(){return 1;}", "v1", split="valid", label=1)]
            config = {"seed": 42, "epochs": 1, "batch_size": 1,
                      "gradient_accumulation": 2, "learning_rate": .001,
                      "graph_learning_rate": .001, "weight_decay": 0.,
                      "log_every": 1, "variant": "lm_pretrain_cfg"}
            vocab = types.SimpleNamespace(encode=lambda view: [[0, 0, 0, 0]],
                                          sizes=lambda: [2] * 4, values={})
            base = SmallBase()
            input_side = lambda model, batch, builder, views, encoded, device, **kwargs: (
                torch.tensor([[float(row["label"])] for row in batch]), None, None)
            with (patch("vulnmechanism.cfg_network.build_model", return_value=SmallModel()),
                  patch.object(exp, "_tokenizer", return_value=object()),
                  patch.object(exp, "_graph_inputs", side_effect=input_side),
                  patch.object(exp, "_graph_scores", return_value=[.1, .9])):
                scores, threshold, coverage, source = exp._train_graph(
                    base, config, rows, {row["sample_key"]: {} for row in rows}, vocab,
                    folder, torch.device("cpu"),
                    initial_adapter_state={"weight": torch.tensor([[.75]])})
            self.assertEqual(len(base.loaded), 1)
            self.assertTrue(torch.equal(base.loaded[0], torch.tensor([[.75]])))
            self.assertEqual(scores, [.1, .9])
            self.assertIsNone(coverage)
            self.assertIsNone(source)
            self.assertTrue((folder / "best.pt").exists())


if __name__ == "__main__":
    unittest.main()
