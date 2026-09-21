import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from vulnmechanism.graph_dataset import (
    atomic_json, atomic_jsonl, file_sha256, graph_audit, load_graph_dataset, read_jsonl,
)
from vulnmechanism.graph_experiment import flip_counts
from vulnmechanism.graph_features import (
    FEATURE_NAMES, feature_signature, fit_graph_config, graph_to_record,
    records_fingerprint, validate_record_graph,
)
from vulnmechanism.graph_nn import StaticGraphEncoder


def sample_graph(literal="4"):
    def node(key, kind, code, **properties):
        properties.update(kind=kind, CODE=code)
        return SimpleNamespace(node_id=key, label=kind, code=code, properties=properties)
    nodes = {
        "1": node("1", "CALL", "p = malloc(n * 4)", NAME="<operator>.assignment"),
        "2": node("2", "IDENTIFIER", "p", TYPE_FULL_NAME="char *", ARGUMENT_INDEX=1, ORDER=1),
        "3": node("3", "CALL", "malloc(n * 4)", NAME="malloc", ARGUMENT_INDEX=2, ORDER=2),
        "4": node("4", "CALL", "n * 4", NAME="<operator>.multiplication", ARGUMENT_INDEX=1),
        "5": node("5", "IDENTIFIER", "n", TYPE_FULL_NAME="size_t", ARGUMENT_INDEX=1),
        "6": node("6", "LITERAL", literal, TYPE_FULL_NAME="int", ARGUMENT_INDEX=2),
        "7": node("7", "CALL", "n = 4", NAME="<operator>.assignment"),
        "8": node("8", "IDENTIFIER", "n", TYPE_FULL_NAME="size_t", ARGUMENT_INDEX=1),
        "9": node("9", "LITERAL", "4", TYPE_FULL_NAME="int", ARGUMENT_INDEX=2),
        "10": node("10", "METHOD_RETURN", "RET", TYPE_FULL_NAME="void"),
    }
    edges = [SimpleNamespace(kind="AST", source=u, target=v) for u, v in
             [("1", "2"), ("1", "3"), ("3", "4"), ("4", "5"), ("4", "6"), ("7", "8"), ("7", "9")]]
    edges += [SimpleNamespace(kind="CFG", source=u, target=v) for u, v in [("1", "7"), ("7", "10")]]
    return SimpleNamespace(function="f", nodes=nodes, edges=edges, quality={"status": "accepted"})


def sample_row(key="a", split="train", label=0, literal="4"):
    source = f"void f() {{ int n = {literal}; }}"
    return dict(schema_version=9, sample_key=key, dataset="primevul", split=split,
                label=label, raw_source=source, language="c", resolved_language="c",
                function_name="f", cpg_relations="AST|a|b", mechanism_items=[],
                mechanism_context="", cpg_quality={}, graph_status="available",
                static_graph=graph_to_record(sample_graph(literal), source))


class GraphFeatureTests(unittest.TestCase):
    def test_native_attributes_and_identical_text_nodes_are_preserved(self):
        row = sample_row()
        graph = row["static_graph"]
        self.assertEqual(len(graph["nodes"]), 10)
        self.assertEqual(len(graph["edges"]), 9)
        self.assertEqual(sum(n["code"] == "4" for n in graph["nodes"]), 2)
        by_id = {n["id"]: n for n in graph["cfg_nodes"]}
        self.assertEqual(by_id["1"]["features"], {
            "api": ["malloc"], "datatype": ["char *"], "literal": ["4"], "operator": ["multiplication"]})
        self.assertTrue(all(not values for values in by_id["10"]["features"].values()))
        self.assertEqual(graph["statistics"]["definitions"], 2)
        validate_record_graph(row)

    def test_literals_remain_distinct(self):
        a, b = sample_row(literal="4"), sample_row(literal="16")
        a = next(n for n in a["static_graph"]["cfg_nodes"] if n["id"] == "1")
        b = next(n for n in b["static_graph"]["cfg_nodes"] if n["id"] == "1")
        self.assertNotEqual(a["features"]["literal"], b["features"]["literal"])

    def test_missing_type_is_not_guessed_from_rhs(self):
        graph = sample_graph()
        graph.nodes["2"].properties["TYPE_FULL_NAME"] = "ANY"
        packed = graph_to_record(graph, "source")
        lhs = next(n for n in packed["cfg_nodes"] if n["id"] == "1")
        self.assertEqual(lhs["features"]["datatype"], ["<TYPE_UNKNOWN>"])

    def test_operator_alias_and_argument_index(self):
        graph = sample_graph()
        graph.nodes["1"].properties["NAME"] = "<operators>.assignment"
        graph.nodes["2"].properties["ORDER"] = 99
        graph.nodes["3"].properties["ORDER"] = 1
        node = next(n for n in graph_to_record(graph, "x")["cfg_nodes"] if n["id"] == "1")
        self.assertEqual(node["features"]["datatype"], ["char *"])

    def test_missing_native_attributes_fail_loudly(self):
        graph = sample_graph()
        graph.nodes["1"].properties = None
        with self.assertRaisesRegex(ValueError, "native Joern"):
            graph_to_record(graph, "x")

    def test_source_mismatch_and_dangling_cfg_are_rejected(self):
        row = sample_row()
        row["raw_source"] += " /* different */"
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            validate_record_graph(row)
        row = sample_row()
        row["static_graph"]["cfg_edges"][0][1] = 999
        with self.assertRaisesRegex(ValueError, "endpoints"):
            validate_record_graph(row)

    def test_no_labels_or_identifiers_in_attribute_channels(self):
        row = sample_row()
        encoded = json.dumps([n["features"] for n in row["static_graph"]["cfg_nodes"]])
        self.assertNotIn('"p"', encoded)
        self.assertNotIn('"n"', encoded)
        self.assertNotIn('"label"', encoded)
        self.assertEqual(set(row["static_graph"]["cfg_nodes"][0]["features"]), set(FEATURE_NAMES))

    def test_vocabulary_is_training_only_and_stable(self):
        train = sample_row("train", literal="4")
        valid = sample_row("valid", split="valid", literal="999")
        config = fit_graph_config([train])
        self.assertIn(feature_signature(["4"]), config["vocabulary"]["literal"])
        self.assertNotIn(feature_signature(["999"]), config["vocabulary"]["literal"])
        with self.assertRaisesRegex(ValueError, "training records only"):
            fit_graph_config([train, valid])
        relabeled = copy.deepcopy(train)
        relabeled["label"] = 1
        self.assertEqual(config["vocabulary"], fit_graph_config([relabeled])["vocabulary"])

    def test_missing_graph_is_explicit_not_a_silent_empty_experiment(self):
        row = sample_row()
        row.update(static_graph=None, graph_status="unavailable")
        validate_record_graph(row)
        with self.assertRaisesRegex(ValueError, "no usable"):
            fit_graph_config([row])
        row["graph_status"] = "available"
        with self.assertRaisesRegex(ValueError, "unavailable"):
            validate_record_graph(row)


class GraphEncoderTests(unittest.TestCase):
    def setUp(self):
        self.row = sample_row()
        self.graph = self.row["static_graph"]
        self.config = fit_graph_config([self.row], embedding_dim=4, steps=2, vocab_size=20)

    def models(self):
        torch.manual_seed(42)
        b = StaticGraphEncoder(self.config, propagate=False)
        torch.manual_seed(42)
        c = StaticGraphEncoder(self.config, propagate=True)
        return b, c

    def test_matched_parameter_counts_and_initialization(self):
        b, c = self.models()
        self.assertEqual(sum(p.numel() for p in b.parameters()), sum(p.numel() for p in c.parameters()))
        for key, value in b.state_dict().items():
            torch.testing.assert_close(value, c.state_dict()[key])

    def test_b_ignores_edges_but_c_uses_direction(self):
        b, c = self.models()
        reverse = copy.deepcopy(self.graph)
        reverse["cfg_edges"] = [[v, u] for u, v in reverse["cfg_edges"]]
        torch.testing.assert_close(b([self.graph]), b([reverse]))
        self.assertFalse(torch.allclose(c([self.graph]), c([reverse]), atol=1e-7))

    def test_node_permutation_invariance(self):
        _, c = self.models()
        permuted = copy.deepcopy(self.graph)
        order = [2, 0, 1]
        inverse = {old: new for new, old in enumerate(order)}
        permuted["cfg_nodes"] = [permuted["cfg_nodes"][i] for i in order]
        permuted["cfg_edges"] = [[inverse[u], inverse[v]] for u, v in permuted["cfg_edges"]]
        torch.testing.assert_close(c([self.graph]), c([permuted]), rtol=1e-5, atol=1e-6)

    def test_batch_isolation_and_exact_missing_graph_zero(self):
        _, c = self.models()
        other = sample_row(literal="16")["static_graph"]
        together = c([self.graph, None, other])
        torch.testing.assert_close(together[0], c([self.graph])[0])
        torch.testing.assert_close(together[2], c([other])[0])
        self.assertEqual(together[1].abs().sum().item(), 0)
        self.assertEqual(c([None, None]).abs().sum().item(), 0)

    def test_gradients_reach_embeddings_message_and_gru(self):
        for model in self.models():
            model([self.graph]).square().sum().backward()
            for parameter in (model.embeddings["literal"].weight, model.message.weight, model.update.weight_hh):
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_no_truncation_of_large_graph(self):
        _, c = self.models()
        large = copy.deepcopy(self.graph)
        large["cfg_nodes"] = [copy.deepcopy(large["cfg_nodes"][0]) for _ in range(180)]
        large["cfg_edges"] = [[i, i+1] for i in range(179)]
        self.assertEqual(c([large]).shape, (1, c.out_dim))


class GraphDatasetAndCompareTests(unittest.TestCase):
    def test_dataset_integrity_and_member_preservation(self):
        rows = [sample_row("a"), sample_row("b", split="valid", label=1)]
        rows[1].update(static_graph=None, graph_status="unavailable")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "graph.jsonl"
            atomic_jsonl(path, rows)
            metadata = {"graph_schema_version": 1, "complete": True, "source_dataset": "primevul",
                        "samples": 2, "input_fingerprint": records_fingerprint(rows),
                        "output_sha256": file_sha256(path)}
            atomic_json(path.with_suffix(".meta.json"), metadata)
            restored, _ = load_graph_dataset(path)
            self.assertEqual([r["sample_key"] for r in restored], ["a", "b"])
            self.assertEqual(graph_audit(restored)["members_dropped"], 0)
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changed after export"):
                load_graph_dataset(path)

    def test_torn_tail_only_is_recoverable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            path.write_text('{"sample_key":"a"}\n{"sam')
            self.assertEqual(len(read_jsonl(path, recover_tail=True)), 1)
            with self.assertRaises(ValueError):
                read_jsonl(path)
            path.write_text('{"bad\n{"sample_key":"a"}\n')
            with self.assertRaises(ValueError):
                read_jsonl(path, recover_tail=True)

    def test_incomplete_export_cannot_train(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "graph.jsonl"
            atomic_json(path.with_suffix(".meta.json"), {"graph_schema_version": 1, "complete": False})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_graph_dataset(path)

    def test_all_four_flip_directions(self):
        old, new = [], []
        for i, (truth, a, b) in enumerate([(1, 0, 1), (0, 1, 0), (1, 1, 0), (0, 0, 1)]):
            row = dict(sample_key=str(i), dataset="primevul", split="valid", label=truth,
                       source_tokens=10, source_truncated=False, graph_available=True)
            old.append(dict(row, prediction=a, probability=0.8 if a else 0.2))
            new.append(dict(row, prediction=b, probability=0.8 if b else 0.2))
        counts = flip_counts(old, new)
        for key in ("fn_to_tp", "fp_to_tn", "tp_to_fn", "tn_to_fp"):
            self.assertEqual(counts[key], 1)
        self.assertEqual(counts["net_corrected"], 0)
        self.assertEqual(counts, flip_counts(old, new, shared_threshold=0.5))
        with self.assertRaisesRegex(ValueError, "identical"):
            flip_counts(old, new[:-1])
        new[0]["label"] = 0
        with self.assertRaisesRegex(ValueError, "mismatch"):
            flip_counts(old, new)


if __name__ == "__main__":
    unittest.main()
