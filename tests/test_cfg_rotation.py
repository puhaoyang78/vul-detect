"""CPU tests for official-train negative rotation; no Qwen or Joern runs."""
from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_cfg_ablation import (TinyInputBuilder, TinySource, fake_base, fixture_graph,
                                     fixture_rows, raw_graph, write_jsonl)
from vulnmechanism import cfg_data as data
from vulnmechanism import cfg_experiment as exp
from vulnmechanism import cfg_rotation as rotation
from vulnmechanism import prepare_benchmark as benchmark
from vulnmechanism.cfg_network import build_model


def benchmark_row(key, source, label=0, split="train", dataset="primevul", groups=()):
    return dict(dataset=dataset, sample_key=key, function=source, label=label, split=split,
                official_split=split, source_file="PrimeVul_v0.1/primevul_train.jsonl",
                source_row=int(key.rsplit(":", 1)[-1]) + 1 if key.rsplit(":", 1)[-1].isdigit() else 1,
                language="c", file_name="x.c", repo="", commit="",
                counterpart_groups=list(groups))


class CandidateTests(unittest.TestCase):
    def test_screen_rechecks_conflicts_all_existing_splits_and_internal_duplicates(self):
        positive = benchmark_row("primevul:1", "int vulnerable(int x) { return x + 100; }", 1)
        train_negative = benchmark_row("primevul:2", "int original(int x) { return x * 50; }")
        valid = benchmark_row("primevul:3", "int held_valid(int x) { return x - 20; }", split="valid")
        test = benchmark_row("primevul:4", "int held_test(int x) { return x / 10; }", split="test")
        external = benchmark_row("sven:ext", "int external(int x) { return x | 8; }",
                                 split="external_test", dataset="sven")
        same_valid = benchmark_row("primevul:10", valid["function"])
        normalized_test = benchmark_row("primevul:11", "  int held_test ( int x ) { /*c*/ return x / 10 ; } ")
        near_source = "int long_func(int x) { " + " ".join(f"x += {i};" for i in range(55)) + " return x; }"
        near = benchmark_row("primevul:12", near_source.replace("x += 40;", "x += 999;"))
        external_copy = benchmark_row("primevul:13", external["function"])
        paired = benchmark_row("primevul:14", "int patched(int x) { return x + 101; }")
        conflict_positive = benchmark_row("primevul:15", "int conflict(int x) { return x * 5; }", 1)
        conflict_negative = benchmark_row("primevul:16", "int conflict ( int x ) { return x * 5 ; }")
        safe1 = benchmark_row("primevul:17", "int safe_alpha(int a) { return a & 131; }")
        safe2 = benchmark_row("primevul:18", "long safe_beta(long b) { return b ^ 739; }")
        duplicate_safe = benchmark_row("primevul:19", safe1["function"])
        raw = [positive, train_negative, valid, test, external, same_valid,
               normalized_test, near, external_copy, paired, conflict_positive,
               conflict_negative, safe1, safe2, duplicate_safe]
        protected = [positive, train_negative, valid, test, external,
                     benchmark_row("primevul:20", near_source)]
        counterpart = (benchmark.digest(benchmark.normalized_source(positive["function"])),
                       benchmark.digest(benchmark.normalized_source(paired["function"])))
        group = min(counterpart)
        protected[0]["counterpart_groups"] = [group]
        base = [dict(sample_key=r["sample_key"], dataset=r["dataset"],
                     split=r["split"], label=r["label"], raw_source=r["function"])
                for r in protected[:5]]
        units = [benchmark.unit([r]) for r in raw]
        first, audit = rotation.screen_candidate_units(units, [counterpart], protected, base, 42)
        second, _ = rotation.screen_candidate_units(units, [counterpart], protected, base, 42)
        self.assertEqual([r["sample_key"] for r in first], [r["sample_key"] for r in second])
        selected = {r["sample_key"] for r in first}
        self.assertEqual(len(selected), 2)
        self.assertIn("primevul:18", selected)
        self.assertEqual(len(selected & {"primevul:17", "primevul:19"}), 1)
        self.assertGreaterEqual(audit["label_conflict_units"], 2)
        self.assertGreaterEqual(audit["overlap_near_duplicate"], 1)
        self.assertGreaterEqual(audit["overlap_pair_counterpart"], 1)
        self.assertGreaterEqual(audit["overlap_normalized_source"], 1)

    def test_epoch_schedule_preserves_slots_counts_and_repeats_only_fixed_negatives(self):
        train = fixture_rows()[:5]
        selected = [dict(train[0], sample_key=f"primevul:{100+i}",
                         raw_source=f"int extra{i}(){{return {i};}}") for i in range(6)]
        fixed = rotation.epoch_train_rows(train, selected, "cfg_rotation_fixed", 3)
        rotating = rotation.epoch_train_rows(train, selected, "cfg_rotation_rotating", 3)
        for epoch in range(3):
            self.assertEqual([r["label"] for r in fixed[epoch]], [r["label"] for r in rotating[epoch]])
            self.assertEqual([r["sample_key"] for r in fixed[epoch] if r["label"] == 1],
                             [r["sample_key"] for r in rotating[epoch] if r["label"] == 1])
            self.assertEqual(Counter(r["label"] for r in rotating[epoch]), Counter({0: 3, 1: 2}))
            order = list(range(len(train)))
            random.Random(42 + epoch).shuffle(order)
            fixed_batch = [fixed[epoch][i] for i in order]
            rotating_batch = [rotating[epoch][i] for i in order]
            self.assertEqual([r["label"] for r in fixed_batch], [r["label"] for r in rotating_batch])
            self.assertEqual([r["sample_key"] for r in fixed_batch if r["label"] == 1],
                             [r["sample_key"] for r in rotating_batch if r["label"] == 1])
        self.assertEqual(rotation.schedule_report(fixed)[2]["previous_negative_overlap"], 3)
        self.assertEqual(rotation.schedule_report(rotating)[2]["previous_negative_overlap"], 0)
        self.assertEqual(rotation.schedule_report(rotating)[2]["pairwise_total_overlap"], [2, 2])
        with self.assertRaises(ValueError):
            rotation.epoch_train_rows(train, selected[:-1], "cfg_rotation_rotating", 3)

    def test_build_refills_failed_graphs_in_fixed_order_without_modifying_raw(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            reference = root / "reference"
            (reference / "cfg").mkdir(parents=True)
            (reference / "cfg" / "complete.json").write_text("{}\n")
            config = dict(source_dataset="primevul", seed=42, epochs=3, source_max_length=2048)
            (reference / "config.json").write_text(json.dumps(config))
            raw = root / "primevul_train.jsonl"
            source_rows = [dict(idx=i, target=0, func=f"int f{i}() {{ return {i}; }}") for i in range(3)]
            write_jsonl(raw, source_rows)
            before = raw.read_bytes()
            candidates = []
            with raw.open("rb") as stream:
                for i, original in enumerate(source_rows):
                    offset = stream.tell()
                    stream.readline()
                    candidates.append(dict(rank=i, sample_key=f"primevul:{i}", source_row=i+1,
                                           byte_offset=offset, source_sha256=data.source_hash(original["func"]),
                                           language="c", file_name="x.c", counterpart_groups=[]))
            prepared = root / "rotation"
            prepared.mkdir()
            rotation.write_jsonl(prepared / "candidates.jsonl", candidates)
            rotation.atomic_json(prepared / "preparation.json", dict(
                reference_run_dir=str(reference), reference_config_sha256=data.digest(config),
                raw_train=str(raw), raw_train_sha256=rotation.file_sha256(raw),
                candidates_sha256=rotation.file_sha256(prepared / "candidates.jsonl"),
                candidate_count=3, required_new_negatives=2, negative_slots=1))
            def extractor(requests, **_kwargs):
                return [RuntimeError("synthetic graph failure") if "f1" in item["source"]
                        else raw_graph(str(4+i)) for i, item in enumerate(requests)]
            result = rotation.build_rotation(prepared, extractor=extractor, batch_size=2)
            self.assertEqual((result["attempted"], result["failed"], result["selected_count"]), (3, 1, 2))
            self.assertEqual(result["selected_sample_keys"], ["primevul:0", "primevul:2"])
            self.assertEqual(len(list((prepared / "parts").glob("*.dataset.jsonl"))), 2)
            failures = data.read_jsonl(prepared / "parts" / "000000.graphs.jsonl.errors.jsonl")
            self.assertEqual([r["sample_key"] for r in failures], ["primevul:1"])
            self.assertEqual(rotation.build_rotation(prepared, extractor=extractor), result)
            self.assertEqual(raw.read_bytes(), before)
            self.assertEqual(len(data.read_jsonl(prepared / "selected.graphs.jsonl")), 2)
            shortage = root / "shortage"
            shortage.mkdir()
            shutil.copy2(prepared / "candidates.jsonl", shortage / "candidates.jsonl")
            insufficient = json.loads((prepared / "preparation.json").read_text())
            insufficient["required_new_negatives"] = 3
            rotation.atomic_json(shortage / "preparation.json", insufficient)
            with self.assertRaisesRegex(ValueError, "only 2 usable new graphs; need 3"):
                rotation.build_rotation(shortage, extractor=extractor, batch_size=2)
            self.assertFalse((shortage / "selected.jsonl").exists())


class TrainingEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_original_cfg_network_initialization_and_rotation_entry_smoke(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source, graph_path, c_run, rotation_dir, output = (
                root / "source.jsonl", root / "graphs.jsonl", root / "c",
                root / "rotation", root / "run")
            rows = fixture_rows()
            write_jsonl(source, rows)
            exports = [dict(data.identity(row), graph_schema_version=data.GRAPH_SCHEMA,
                            preprocessing_version=data.JOERN_SOURCE_PREPROCESSING_VERSION,
                            preprocessing_applied=False,
                            original_source_sha256=data.source_hash(row["raw_source"]),
                            parsed_source_sha256=data.source_hash(row["raw_source"]),
                            graph=fixture_graph(str(i+4))) for i, row in enumerate(rows)]
            write_jsonl(graph_path, exports)
            original_bytes = source.read_bytes(), graph_path.read_bytes()
            api = fake_base()
            class TrackedTinySource(TinySource):
                created = []
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.created.append(self)
            api.SequenceVulnerabilityClassifier = TrackedTinySource
            common = ["--dataset", str(source), "--graphs", str(graph_path),
                      "--model-path", "tiny-test-only", "--device", "cpu", "--epochs", "3",
                      "--batch-size", "2", "--gradient-accumulation", "2", "--graph-hidden-size", "8",
                      "--graph-steps", "5", "--source-max-length", "2048", "--seed", "42"]
            TinyInputBuilder.allow_test = False
            c_args = exp.parser().parse_args(["run", *common, "--output-dir", str(c_run), "--variants", "cfg"])
            exp.run_experiment(c_args, base=api)
            TinyInputBuilder.allow_test = True
            exp.evaluate_run(exp.parser().parse_args(["eval", "--run-dir", str(c_run),
                                                     "--variants", "cfg", "--split", "test",
                                                     "--device", "cpu"]), base=api)
            TinyInputBuilder.allow_test = False
            new_rows = [dict(rows[0], sample_key=f"primevul:{100+i}",
                             raw_source=f"int extra{i}() {{ return {i}; }}", function_name=f"extra{i}")
                        for i in range(6)]
            rotation_dir.mkdir()
            rotation.write_jsonl(rotation_dir / "selected.jsonl", new_rows)
            rotation.write_jsonl(rotation_dir / "selected.graphs.jsonl", [
                dict(data.identity(row), graph_schema_version=data.GRAPH_SCHEMA,
                     preprocessing_version=data.JOERN_SOURCE_PREPROCESSING_VERSION,
                     preprocessing_applied=False,
                     original_source_sha256=data.source_hash(row["raw_source"]),
                     parsed_source_sha256=data.source_hash(row["raw_source"]),
                     graph=fixture_graph(str(i+14))) for i, row in enumerate(new_rows)])
            candidates = [dict(rank=i, sample_key=row["sample_key"],
                               source_sha256=data.source_hash(row["raw_source"]))
                          for i, row in enumerate(new_rows)]
            rotation.write_jsonl(rotation_dir / "candidates.jsonl", candidates)
            c_config = json.loads((c_run / "config.json").read_text())
            rotation.atomic_json(rotation_dir / "preparation.json", dict(
                reference_run_dir=str(c_run), reference_config_sha256=data.digest(c_config),
                candidates_sha256=rotation.file_sha256(rotation_dir / "candidates.jsonl"),
                required_new_negatives=6))
            rotation.atomic_json(rotation_dir / "selection.json", dict(
                selected_count=6, selected_sample_keys=[r["sample_key"] for r in new_rows],
                selected_sha256=rotation.file_sha256(rotation_dir / "selected.jsonl"),
                selected_graphs_sha256=rotation.file_sha256(rotation_dir / "selected.graphs.jsonl")))
            run_args = exp.parser().parse_args(["run", *common, "--output-dir", str(output),
                "--variants", *rotation.ROTATION_VARIANTS, "--rotation-dir", str(rotation_dir),
                "--reference-run-dir", str(c_run)])
            result = exp.run_experiment(run_args, base=api)
            self.assertEqual(set(result["changes_vs_cfg"]), set(rotation.ROTATION_VARIANTS))
            self.assertEqual(result["training_member_policy"], "primevul_negative_rotation")
            self.assertEqual((source.read_bytes(), graph_path.read_bytes()), original_bytes)
            self.assertEqual(sum(model.encoder.calls == 12 for model in TrackedTinySource.created), 3)
            original_vocab = json.loads((c_run / "vocabulary.json").read_text())
            self.assertEqual(json.loads((output / "vocabulary.json").read_text()), original_vocab)
            new_view = data.abstract_cfg(fixture_graph("14"))
            literal_node = new_view["node_ids"].index("1")
            self.assertEqual(data.AttributeVocabulary(original_vocab).encode(new_view)[literal_node][2], 1)
            for variant in rotation.ROTATION_VARIANTS:
                cp = torch.load(output / variant / "best.pt", map_location="cpu", weights_only=False)
                self.assertEqual(cp["vocabulary"], original_vocab)
                history = data.read_jsonl(output / variant / "history.jsonl")
                epochs = [r for r in history if r["event"] == "epoch"]
                self.assertEqual(len(epochs), 3)
                self.assertEqual([r["optimizer_steps"] for r in epochs], [2, 2, 2])
                plan = json.loads((output / variant / "train_epochs.json").read_text())
                self.assertEqual([(r["positive_count"], r["negative_count"]) for r in plan], [(2, 3)] * 3)
                self.assertEqual([r["sample_key"] for r in data.read_jsonl(
                    output / variant / "valid.predictions.jsonl")],
                    [r["sample_key"] for r in data.read_jsonl(c_run / "cfg" / "valid.predictions.jsonl")])
            self.assertEqual(json.loads((output / "cfg_rotation_fixed" / "train_epochs.json").read_text())[2]
                             ["previous_negative_overlap"], 3)
            self.assertEqual(json.loads((output / "cfg_rotation_rotating" / "train_epochs.json").read_text())[2]
                             ["previous_negative_overlap"], 0)
            vocab = data.AttributeVocabulary(original_vocab)
            c_config["variant"] = "cfg"
            states = {}
            for variant in ("cfg", *rotation.ROTATION_VARIANTS):
                api._seed_everything(42)
                model = build_model(api, dict(c_config, variant=variant), vocab.sizes(),
                                    torch.device("cpu"), training=True)
                states[variant] = {k: v.detach().clone() for k, v in model.state_dict().items()}
                self.assertEqual(model.task_modules["cfg_encoder"].mode, "cfg")
            for variant in rotation.ROTATION_VARIANTS:
                self.assertEqual(states["cfg"].keys(), states[variant].keys())
                for key in states["cfg"]:
                    self.assertTrue(torch.equal(states["cfg"][key], states[variant][key]))
            TinyInputBuilder.allow_test = True
            try:
                with patch.object(exp, "select_threshold", side_effect=AssertionError("test threshold tuning")):
                    evaluated = exp.evaluate_run(exp.parser().parse_args([
                        "eval", "--run-dir", str(output), "--variants", *rotation.ROTATION_VARIANTS,
                        "--split", "test", "--device", "cpu"]), base=api)
                self.assertEqual(set(evaluated["changes_vs_cfg"]), set(rotation.ROTATION_VARIANTS))
                prediction_path = output / "cfg_rotation_fixed" / "test.predictions.jsonl"
                original_predictions = data.read_jsonl(prediction_path)
                for field, value in (("source_sha256", "0" * 64), ("label", 1 - original_predictions[0]["label"]),
                                     ("sample_key", "primevul:other")):
                    bad = [dict(row) for row in original_predictions]
                    bad[0][field] = value
                    write_jsonl(prediction_path, bad)
                    with self.assertRaises(ValueError):
                        exp.compare_run(output, "test", reference_root=c_run)
                write_jsonl(prediction_path, original_predictions)
            finally:
                TinyInputBuilder.allow_test = False
