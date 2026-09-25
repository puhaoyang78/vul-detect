"""Frozen C readout tests with a tiny source model; no Qwen or Joern run."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from tests.test_cfg_ablation import fake_base, fixture_graph, fixture_rows, write_jsonl
from vulnmechanism import cfg_data as data
from vulnmechanism import cfg_experiment as exp
from vulnmechanism import cfg_readout as readout


class ReadoutMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initial_c_logit_parameter_counts_and_gradients(self):
        torch.manual_seed(42)
        source = torch.randn(4, 8)
        graph = torch.randn(4, 16)
        c_logit = torch.randn(4)
        labels = torch.tensor([0., 1., 0., 1.])
        counts = {}
        for mode in exp.READOUT_VARIANTS:
            model = readout.ResidualReadout(mode, 8, 16)
            counts[mode] = sum(parameter.numel() for parameter in model.parameters())
            self.assertTrue(torch.equal(model(source, graph, c_logit), c_logit))
            if mode == "mlp":
                self.assertGreater(model.source_graph.weight.abs().sum().item(), 0)
            if mode == "interaction":
                self.assertGreater(model.source.weight.abs().sum().item(), 0)
                self.assertGreater(model.graph.weight.abs().sum().item(), 0)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            F.binary_cross_entropy_with_logits(model(source, graph, c_logit), labels).backward()
            self.assertGreater(model.output.weight.grad.abs().sum().item(), 0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(model(source, graph, c_logit), labels).backward()
            if mode == "mlp":
                self.assertGreater(model.source_graph.weight.grad.abs().sum().item(), 0)
            if mode == "interaction":
                self.assertGreater(model.source.weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.graph.weight.grad.abs().sum().item(), 0)
        self.assertEqual(counts, {"linear": 25, "mlp": 800, "interaction": 800})


class ReadoutPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_frozen_extraction_alignment_training_and_valid_comparison(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            dataset, graphs = root / "dataset.jsonl", root / "graphs.jsonl"
            c_run, screen = root / "c", root / "screen"
            cache_path = screen / "representations.pt"
            rows = fixture_rows()
            write_jsonl(dataset, rows)
            write_jsonl(graphs, [dict(data.identity(row), graph_schema_version=data.GRAPH_SCHEMA,
                                      preprocessing_version=data.JOERN_SOURCE_PREPROCESSING_VERSION,
                                      preprocessing_applied=False,
                                      original_source_sha256=data.source_hash(row["raw_source"]),
                                      parsed_source_sha256=data.source_hash(row["raw_source"]),
                                      graph=fixture_graph()) for row in rows])
            api = fake_base()
            args = exp.parser().parse_args([
                "run", "--dataset", str(dataset), "--graphs", str(graphs),
                "--output-dir", str(c_run), "--model-path", "tiny-test-only",
                "--variants", "cfg", "--epochs", "1", "--batch-size", "2",
                "--gradient-accumulation", "2", "--graph-hidden-size", "8",
                "--graph-steps", "2", "--device", "cpu"])
            exp.run_experiment(args, base=api)
            c_hash = exp.file_sha256(c_run / "cfg" / "best.pt")
            constructed = []
            original_build = readout.build_model
            def capture_model(*args, **kwargs):
                model = original_build(*args, **kwargs)
                constructed.append(model)
                return model
            with patch.object(readout, "build_model", side_effect=capture_model):
                report = readout.extract_representations(c_run, cache_path, device="cpu", base=api)
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            expected = [data.identity(row) for row in rows if row["split"] in {"train", "valid"}]
            self.assertEqual(cache["samples"], expected)
            self.assertEqual(report["counts"], {"train": 5, "valid": 2})
            self.assertEqual(cache["source"].shape, (7, 8))
            self.assertEqual(cache["graph"].shape, (7, 16))
            self.assertEqual(cache["c_logit"].shape, (7,))
            self.assertEqual(exp.file_sha256(c_run / "cfg" / "best.pt"), c_hash)
            self.assertLessEqual(report["valid_reference_max_score_difference"], 1e-4)
            frozen = constructed[0]
            self.assertFalse(frozen.training)
            self.assertTrue(all(not p.requires_grad for p in frozen.parameters()))
            self.assertEqual(frozen.encoder.calls, 4)  # Seven train/valid rows, batch size two.
            c_config = json.loads((c_run / "config.json").read_text())
            prepared_rows, views = exp._prepare(str(dataset), str(graphs), "primevul")
            checkpoint = torch.load(c_run / "cfg" / "best.pt", map_location="cpu", weights_only=False)
            vocabulary = data.AttributeVocabulary(checkpoint["vocabulary"])
            encoded = {rows[0]["sample_key"]: vocabulary.encode(views[rows[0]["sample_key"]])}
            inputs = exp._graph_inputs(frozen, [prepared_rows[0]], exp._tokenizer(api, c_config),
                                       views, encoded, torch.device("cpu"))
            with torch.no_grad():
                source, graph, c_logit = frozen.representations(*inputs)
                original_logit = frozen(*inputs)
            torch.testing.assert_close(cache["source"][0], source[0], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(cache["graph"][0], graph[0], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(cache["c_logit"][0], original_logit[0], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(c_logit, original_logit, rtol=0, atol=0)

            tampered = copy.deepcopy(cache)
            tampered["samples"][0]["label"] = 1 - tampered["samples"][0]["label"]
            tampered_path = root / "tampered.pt"
            torch.save(tampered, tampered_path)
            with self.assertRaisesRegex(ValueError, "sample keys, labels, splits"):
                readout._load_cache(tampered_path, c_run)

            trained = readout.train_readouts(cache_path, c_run, screen, device="cpu")
            self.assertEqual(set(trained["metrics"]), set(exp.READOUT_VARIANTS))
            self.assertEqual(trained["trainable_parameters"],
                             {"linear": 25, "mlp": 800, "interaction": 800})
            for mode in exp.READOUT_VARIANTS:
                folder = screen / mode
                checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
                predictions = data.read_jsonl(folder / "valid.predictions.jsonl")
                self.assertEqual([row["sample_key"] for row in predictions],
                                 [row["sample_key"] for row in expected if row["split"] == "valid"])
                self.assertEqual({row["threshold"] for row in predictions},
                                 {checkpoint["decision_threshold"]})
                self.assertFalse((folder / "test.predictions.jsonl").exists())
                self.assertEqual(checkpoint["trainable_parameters"],
                                 trained["trainable_parameters"][mode])
            comparison = exp.compare_run(screen, "valid", reference_root=c_run)
            self.assertEqual(set(comparison["metrics"]), {"cfg", *exp.READOUT_VARIANTS})
            self.assertEqual(set(comparison["changes_vs_cfg"]), set(exp.READOUT_VARIANTS))
            self.assertTrue((screen / "comparison.valid.csv").exists())
            self.assertFalse((screen / "comparison.test.json").exists())
            with self.assertRaisesRegex(ValueError, "valid-only"):
                exp.compare_run(screen, "test", reference_root=c_run)

    def test_cli_help(self):
        for command in ("extract-readout", "train-readout"):
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
                exp.parser().parse_args([command, "--help"])
            self.assertEqual(result.exception.code, 0)
