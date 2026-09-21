"""Tiny, offline encoder tests. No model download, GPU, or Joern process needed."""
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from tests.test_graph_features import sample_graph, sample_row
from vulnmechanism import model as core
from vulnmechanism.cpg import CPGError, GraphNode, resolve_target_graph
from vulnmechanism.graph_dataset import atomic_json, atomic_jsonl, build_graph_dataset, file_sha256, load_graph_dataset
from vulnmechanism.graph_experiment import compare, evaluate_one, train_one
from vulnmechanism.graph_features import fit_graph_config, records_fingerprint


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [2 + ord(char) % 14 for char in text]
        return {"input_ids": ids[:max_length] if truncation else ids}


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 12)
    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def mock_models():
    stack = ExitStack()
    stack.enter_context(patch.object(core, "_build_lora_encoder", side_effect=lambda *a, **k: (TinyEncoder(), 12)))
    stack.enter_context(patch.object(core.AutoTokenizer, "from_pretrained", return_value=TinyTokenizer()))
    stack.enter_context(patch.object(core, "get_peft_model_state_dict", side_effect=lambda m: m.state_dict()))
    stack.enter_context(patch.object(core, "set_peft_model_state_dict", side_effect=lambda m, s: m.load_state_dict(s)))
    return stack


def build_model(variant, config):
    return core._build_model(variant, "tiny", device=torch.device("cpu"), lora_r=2,
                             lora_alpha=4, lora_dropout=0.0, target_modules=(),
                             gradient_checkpointing=False, fusion_dim=8, fusion_heads=2,
                             graph_config=config if variant != "baseline" else None)


class GraphTrainingTests(unittest.TestCase):
    def test_source_initialization_rng_input_and_logits_match_baseline(self):
        row = sample_row()
        config = fit_graph_config([row], embedding_dim=4, steps=2)
        builder = core.InputBuilder(TinyTokenizer(), source_max_length=16, context_max_length=8)
        models, states, logits = [], [], []
        with mock_models():
            for variant in ("baseline", "graph_attributes", "graph_cfg"):
                torch.manual_seed(42)
                model = build_model(variant, config)
                states.append(torch.get_rng_state().clone())
                models.append(model)
                logits.append(core._forward_batch(model, [row], builder, variant=variant,
                                                  excluded_groups=(), device=torch.device("cpu")))
        for i in (1, 2):
            torch.testing.assert_close(states[0], states[i])
            torch.testing.assert_close(models[0].encoder.embedding.weight, models[i].encoder.embedding.weight)
            torch.testing.assert_close(models[0].task_modules["classifier"].weight,
                                       models[i].task_modules["classifier"].weight)
            torch.testing.assert_close(logits[0], logits[i])
            self.assertEqual(models[i].task_modules["classifier"].in_features, 12)
        for key, value in models[1].task_modules.state_dict().items():
            torch.testing.assert_close(value, models[2].task_modules.state_dict()[key])

    def test_graph_and_source_gradients_after_zero_head_initialization(self):
        row = sample_row()
        config = fit_graph_config([row], embedding_dim=4, steps=2)
        with mock_models():
            for variant in ("graph_attributes", "graph_cfg"):
                model = build_model(variant, config)
                builder = core.InputBuilder(TinyTokenizer(), source_max_length=16, context_max_length=8)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    logits = core._forward_batch(model, [row], builder, variant=variant,
                                                 excluded_groups=(), device=torch.device("cpu"))
                    nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
                    self.assertGreater(model.encoder.embedding.weight.grad.abs().sum().item(), 0)
                    self.assertGreater(model.task_modules["graph_classifier"].weight.grad.abs().sum().item(), 0)
                    if step:
                        self.assertGreater(model.task_modules["graph_encoder"].message.weight.grad.abs().sum().item(), 0)
                    optimizer.step()
                ids, mask = builder.sequence_batch([row], variant="baseline", excluded_groups=(), device=torch.device("cpu"))
                torch.testing.assert_close(model(ids, mask, [None]),
                                           core.SequenceVulnerabilityClassifier.forward(model, ids, mask))

    def test_train_vocab_checkpoint_roundtrip_and_validation_is_excluded(self):
        rows = [sample_row(str(i), label=i % 2, literal=str(4+i)) for i in range(4)]
        rows += [sample_row("v0", "valid", 0, "999"), sample_row("v1", "valid", 1, "999")]
        with tempfile.TemporaryDirectory() as folder, mock_models():
            for variant in ("baseline", "graph_attributes", "graph_cfg"):
                path = Path(folder) / f"{variant}.pt"
                saved = core.train_model(None, path, records=copy.deepcopy(rows), variant=variant,
                                         model_path="tiny", device="cpu", epochs=1, batch_size=2,
                                         gradient_accumulation=2, source_max_length=16,
                                         graph_options=None if variant == "baseline" else {"embedding_dim": 4, "steps": 2})
                if variant != "baseline":
                    self.assertNotIn('["999"]', saved["graph_config"]["vocabulary"]["literal"])
                else:
                    self.assertNotIn("graph_config", saved)
                reloaded = torch.load(path, weights_only=False)
                a, b = core._load_model(saved, device=torch.device("cpu")), core._load_model(reloaded, device=torch.device("cpu"))
                builder = core.InputBuilder(TinyTokenizer(), source_max_length=16, context_max_length=384)
                kwargs = dict(variant=variant, excluded_groups=(), device=torch.device("cpu"))
                torch.testing.assert_close(core._forward_batch(a, rows, builder, **kwargs),
                                           core._forward_batch(b, rows, builder, **kwargs))

    def test_experiment_train_eval_compare_resume_end_to_end(self):
        rows = [sample_row(str(i), label=i % 2) for i in range(4)]
        rows += [sample_row("v0", "valid", 0), sample_row("v1", "valid", 1)]
        with tempfile.TemporaryDirectory() as folder, mock_models():
            directory = Path(folder)
            dataset = directory / "data.jsonl"
            atomic_jsonl(dataset, rows)
            atomic_json(dataset.with_suffix(".meta.json"), dict(graph_schema_version=1, complete=True,
                        source_dataset="primevul", samples=len(rows), available=len(rows),
                        input_fingerprint=records_fingerprint(rows), output_sha256=file_sha256(dataset)))
            for variant in ("baseline", "graph_attributes", "graph_cfg"):
                args = SimpleNamespace(dataset=str(dataset), output=str(directory / f"{variant}.pt"),
                    variant=variant, resume=False, model="tiny", source_max_length=16, batch_size=2,
                    gradient_accumulation=1, epochs=1, learning_rate=0.001, weight_decay=0.01,
                    lora_r=2, lora_alpha=4, lora_dropout=0.0, seed=42, device="cpu", log_every=10,
                    graph_embedding_dim=4, graph_steps=2, graph_vocab_size=20)
                train_one(args)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    train_one(args)
                args.resume = True
                before = file_sha256(args.output)
                train_one(args)
                self.assertEqual(file_sha256(args.output), before)
                result = evaluate_one(SimpleNamespace(dataset=str(dataset), checkpoint=args.output,
                                                      split="valid", batch_size=2, device="cpu"))
                self.assertEqual(result["samples"], 2)
                args.epochs = 2
                with self.assertRaisesRegex(ValueError, "config differs"):
                    train_one(args)
            result = compare(SimpleNamespace(output_dir=folder, split="valid"))
            self.assertEqual(len(result["rows"]), 3)
            self.assertIn("graph_cfg", result["flips"])


class GraphExportTests(unittest.TestCase):
    def test_native_node_properties_keep_legacy_equality(self):
        self.assertEqual(GraphNode("1", "CALL", "x"), GraphNode("1", "CALL", "x", {"NAME": "f"}))

    def test_resolved_graph_retains_joern_type_and_positions(self):
        source = "int f() { return 0; }"
        nodes = {
            "file": {"kind": "FILE", "NAME": "sample.c", "CONTENT": source},
            "method": {"kind": "METHOD", "NAME": "f", "CODE": source, "FILENAME": "sample.c",
                       "LINE_NUMBER": 1, "LINE_NUMBER_END": 1},
            "body": {"kind": "BLOCK", "CODE": "{ return 0; }"},
            "ret": {"kind": "RETURN", "CODE": "return 0;", "LINE_NUMBER": 1, "COLUMN_NUMBER": 11},
            "lit": {"kind": "LITERAL", "CODE": "0", "TYPE_FULL_NAME": "int", "ARGUMENT_INDEX": 1},
        }
        edges = [("AST", "method", "body"), ("AST", "body", "ret"),
                 ("AST", "ret", "lit"), ("CFG", "method", "ret")]
        graph = resolve_target_graph(nodes, edges, filename="sample.c", source=source)
        self.assertEqual(graph.nodes["lit"].properties["TYPE_FULL_NAME"], "int")
        self.assertEqual(graph.nodes["ret"].properties["COLUMN_NUMBER"], 11)

    def test_export_resume_failure_retry_and_original_unchanged(self):
        rows = [sample_row("a"), sample_row("b", label=1)]
        for row in rows:
            del row["static_graph"], row["graph_status"]
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            original, output = Path(folder) / "original.jsonl", Path(folder) / "graphs.jsonl"
            atomic_jsonl(original, rows)
            old_hash = file_sha256(original)
            stack.enter_context(patch("vulnmechanism.cpg._find_executable", return_value=Path("mock")))
            stack.enter_context(patch("vulnmechanism.cpg._environment", return_value={}))
            extract = stack.enter_context(patch("vulnmechanism.cpg.extract_function_cpg_batch",
                                              return_value=[sample_graph(), CPGError("bad graph")]))
            report = build_graph_dataset(original, output)
            self.assertEqual(report["unavailable"], 1)
            self.assertEqual(report["members_dropped"], 0)
            loaded, _ = load_graph_dataset(output)
            self.assertEqual([r["sample_key"] for r in loaded], ["a", "b"])
            extract.reset_mock()
            build_graph_dataset(original, output)
            extract.assert_not_called()
            extract.return_value = [sample_graph()]
            report = build_graph_dataset(original, output, retry_failed=True)
            self.assertEqual(report["available"], 2)
            self.assertEqual(file_sha256(original), old_hash)
            self.assertEqual(len(extract.call_args.args[0]), 1)


if __name__ == "__main__":
    unittest.main()
