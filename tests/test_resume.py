"""Interrupted and uninterrupted training must reach the same optimizer state."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tests.test_losver import Tokens, model as paired_model
from tests.test_surface_vat import MATERIALS, OffsetTokenizer, tiny_model
from vulnmechanism import model as core, training
from vulnmechanism.cli import build_parser
from vulnmechanism.artifacts import write_json
from vulnmechanism.data import paired_encoding, read_jsonl, write_jsonl


class Interrupted(Exception):
    pass


class ResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def assert_state_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_state_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_state_equal(x, y)
        else:
            self.assertEqual(a, b)

    def compare_runs(self, control, resumed, best_names):
        a = torch.load(control / "last.pt", weights_only=False)
        b = torch.load(resumed / "last.pt", weights_only=False)
        for name in ("model", "optimizer", "scheduler", "rng"):
            self.assert_state_equal(a[name], b[name])
        for name in best_names:
            ac = torch.load(control / name, weights_only=False)
            bc = torch.load(resumed / name, weights_only=False)
            for key in ("state_dict", "selected_epoch", "validation"):
                self.assert_state_equal(ac[key], bc[key])
            if "decision_threshold" in ac:
                self.assertEqual(ac["decision_threshold"], bc["decision_threshold"])
        # Timings differ after an interruption; all learned/selection statistics
        # and the number/order of canonical log events must remain identical.
        aa = read_jsonl(control / "best.training.jsonl")
        bb = read_jsonl(resumed / "best.training.jsonl")
        self.assertEqual(len(aa), len(bb))
        for x, y in zip(aa, bb):
            keys = {k for k in x if not k.endswith("seconds")}
            self.assert_state_equal({k: x[k] for k in keys}, {k: y[k] for k in keys})
        for name in ("training.candidates.jsonl",):
            if (control / name).exists():
                self.assertEqual((control / name).read_bytes(), (resumed / name).read_bytes())

    def interruption(self, *, epoch=1, step=2, after_save=False):
        real_save = training.save_training_state

        def save(run, model, optimizer, metadata, loop, **kwargs):
            stop = loop["epoch"] == epoch and loop["global_step"] == step
            if stop and not after_save:
                raise Interrupted()
            real_save(run, model, optimizer, metadata, loop, **kwargs)
            if stop and after_save:
                raise Interrupted()

        return patch.object(training, "save_training_state", side_effect=save)

    def test_all_codebert_methods_resume_with_identical_weights_candidates_and_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            encoder, _ = tiny_model(root)
            rows = [dict(sample_key=str(i), dataset="primevul", split="train",
                         label=i % 2, language="c",
                         raw_source=f"int f{i}(int x){{/* ordinary section */return api(x)+{i};}}")
                    for i in range(10)]
            write_jsonl(root / "data/primevul/train.jsonl", rows)
            write_jsonl(root / "data/primevul/valid.jsonl",
                        [dict(r, split="valid", sample_key="v" + r["sample_key"]) for r in rows[:4]])
            config = dict(training.TRAINING, epochs=2, batch_size=2, gradient_accumulation=2)
            for method in training.METHODS:
                with self.subTest(method=method), patch.object(
                    core.AutoTokenizer, "from_pretrained", return_value=OffsetTokenizer()
                ):
                    options = dict(device="cpu", config=config, materials=copy.deepcopy(MATERIALS), save_every=1)
                    control = root / method / "control"
                    resumed = root / method / "resumed"
                    np.random.seed(42)
                    training.train(root / "data", control, encoder, "primevul", method, **options)
                    np.random.seed(42)
                    with self.interruption(), self.assertRaises(Interrupted):
                        training.train(root / "data", resumed, encoder, "primevul", method, **options)
                    run = resumed / "primevul" / method
                    self.assertFalse((run / "training.metrics.json").exists())
                    # Configuration mismatch must fail before modifying the log.
                    before = (run / "best.training.jsonl").read_bytes()
                    with self.assertRaisesRegex(ValueError, "configuration"):
                        training.train(root / "data", resumed, encoder, "primevul", method,
                                       **dict(options, config=dict(config, seed=7)), resume=True)
                    self.assertEqual(before, (run / "best.training.jsonl").read_bytes())
                    training.train(root / "data", resumed, encoder, "primevul", method, resume=True, **options)
                    self.compare_runs(control / "primevul" / method, run, ["best.pt"])
                    before = (run / "last.pt").stat().st_mtime_ns
                    training.train(root / "data", resumed, encoder, "primevul", method, resume=True, **options)
                    self.assertEqual(before, (run / "last.pt").stat().st_mtime_ns)

    def paired_fixture(self, root, mode):
        for split, count in (("train", 10), ("valid", 4), ("test", 4)):
            rows = [dict(sample_key=f"{split}:{i}", pair_key=f"{split}:{i//2}", target=1-i%2,
                         func=f"int f{i}(){{\n return {i};\n}}", lines_ground_truth=[1])
                    for i in range(count)]
            # Exercise a two-window function during resumed L1 training.
            rows[0]["func"] = "int f(){\n" + "a" * 550 + ";\n return 1;\n}"
            write_jsonl(root / "data/primevul_paired" / f"{split}.jsonl", rows)
            write_jsonl(root / "data/primevul_paired" / (
                "test.oracle.jsonl" if split == "test" else f"{split}.lines.jsonl"
            ), rows)
            for output in ("control", "resumed"):
                write_jsonl(root / output / "primevul_paired/losver_localizer" / f"{split}.predicted.jsonl",
                            [dict(r, lines_predicted=[1],
                                  line_numbers=paired_encoding(r["func"], Tokens(), mode)["line_numbers"])
                             for r in rows])

    def test_all_losver_variants_resume_random_sampler_dropout_and_scheduler(self):
        def factory(path, variant, *, window_micro_batch=8, gradient_checkpointing=False):
            model = paired_model(variant == "localizer")
            model.window_micro_batch = window_micro_batch
            model.gradient_checkpointing = gradient_checkpointing
            return model, Tokens()

        for mode in ("512", "l1"):
            for method in training.LOSVER_METHODS:
                with self.subTest(mode=mode, method=method), tempfile.TemporaryDirectory() as tmp, patch.object(
                    core, "losver_model", side_effect=factory
                ):
                    root = Path(tmp)
                    self.paired_fixture(root, mode)
                    settings = dict(training.LOSVER_CONFIG, batch_size=4, eval_batch_size=2,
                                    epochs=dict(baseline=2, localizer=2, predicted=2, oracle=2))
                    options = dict(device="cpu", config=settings, input_mode=mode,
                                   function_micro_batch=1, window_micro_batch=1, save_every=1)
                    training.train_losver(root / "data", root / "control", "fixture", method, **options)
                    # Pause after epoch 1 has selected a best model, in epoch 2.
                    with self.interruption(epoch=2, step=4, after_save=True), self.assertRaises(Interrupted):
                        training.train_losver(root / "data", root / "resumed", "fixture", method, **options)
                    run = root / "resumed/primevul_paired" / method
                    # Leave a torn, uncommitted log tail to verify rollback.
                    with (run / "best.training.jsonl").open("a") as f:
                        f.write('{"unfinished":')
                    training.train_losver(root / "data", root / "resumed", "fixture", method, resume=True, **options)
                    names = ["best-f1.pt"] if method == "losver_localizer" else ["best-p_c.pt"]
                    self.compare_runs(root / "control/primevul_paired" / method, run, names)
                    if method == "losver_localizer":
                        for split in ("train", "valid", "test"):
                            self.assertEqual(
                                (root / "control/primevul_paired/losver_localizer" / f"{split}.predicted.jsonl").read_bytes(),
                                (run / f"{split}.predicted.jsonl").read_bytes(),
                            )

    def test_atomic_save_preserves_previous_state_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            training.save_training_state(run, model, optimizer, {}, {"global_step": 0})
            before = (run / "last.pt").read_bytes()

            def broken_save(cp, path):
                Path(path).write_bytes(b"partial write")
                raise OSError("simulated disk error")

            with patch.object(torch, "save", side_effect=broken_save), self.assertRaises(OSError):
                training.save_training_state(run, model, optimizer, {}, {"global_step": 1})
            self.assertEqual(before, (run / "last.pt").read_bytes())
            target = run / "training.metrics.json"
            write_json(target, {"completed_epochs": 1})
            with patch("vulnmechanism.artifacts.json.dump", side_effect=OSError("disk error")), self.assertRaises(OSError):
                write_json(target, {"completed_epochs": 2})
            self.assertEqual(json.loads(target.read_text()), {"completed_epochs": 1})
        args = build_parser().parse_args(["train", "--resume", "--save-every", "25"])
        self.assertTrue(args.resume)
        self.assertEqual(args.save_every, 25)

    def test_interrupted_initial_save_can_start_again(self):
        def fail_initial(run, *args, **kwargs):
            (Path(run) / "last.pt.tmp").write_bytes(b"incomplete initial checkpoint")
            raise Interrupted()

        def factory(path, variant, *, window_micro_batch=8, gradient_checkpointing=False):
            m = paired_model(variant == "localizer")
            m.window_micro_batch = window_micro_batch
            m.gradient_checkpointing = gradient_checkpointing
            return m, Tokens()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.paired_fixture(root, "l1")
            with patch.object(core, "losver_model", side_effect=factory):
                options = dict(device="cpu", input_mode="l1", config=dict(
                    training.LOSVER_CONFIG, batch_size=4,
                    epochs=dict(baseline=1, localizer=1, predicted=1, oracle=1)))
                with patch.object(training, "save_training_state", side_effect=fail_initial), self.assertRaises(Interrupted):
                    training.train_losver(root / "data", root / "initial", "fixture", "unixcoder", **options)
                run = root / "initial/primevul_paired/unixcoder"
                self.assertFalse((run / "best.training.jsonl").exists())
                result = training.train_losver(root / "data", root / "initial", "fixture", "unixcoder", resume=True, **options)
                self.assertEqual(result["optimizer_steps"], 3)
            encoder, _ = tiny_model(root)
            rows = [dict(sample_key=str(i), raw_source=f"int f{i}(){{return {i};}}",
                         label=i % 2, dataset="primevul", split="train", language="c") for i in range(4)]
            write_jsonl(root / "data/primevul/train.jsonl", rows)
            write_jsonl(root / "data/primevul/valid.jsonl", [dict(r, split="valid") for r in rows])
            with patch.object(core.AutoTokenizer, "from_pretrained", return_value=OffsetTokenizer()):
                options = dict(device="cpu", config=dict(training.TRAINING, epochs=1))
                with patch.object(training, "save_training_state", side_effect=fail_initial), self.assertRaises(Interrupted):
                    training.train(root / "data", root / "initial", encoder, "primevul", "bce", **options)
                self.assertFalse((root / "initial/primevul/bce/best.training.jsonl").exists())
                result = training.train(root / "data", root / "initial", encoder, "primevul", "bce", resume=True, **options)
                self.assertEqual(result["optimizer_steps"], 1)

    def test_validation_and_localizer_export_can_resume_without_more_updates(self):
        from vulnmechanism import evaluation

        def factory(path, variant, *, window_micro_batch=8, gradient_checkpointing=False):
            m = paired_model(variant == "localizer")
            m.window_micro_batch = window_micro_batch
            m.gradient_checkpointing = gradient_checkpointing
            return m, Tokens()

        for method in ("unixcoder", "losver_localizer"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp, patch.object(
                core, "losver_model", side_effect=factory
            ):
                root = Path(tmp)
                self.paired_fixture(root, "l1")
                settings = dict(training.LOSVER_CONFIG, batch_size=4, eval_batch_size=2,
                                epochs=dict(baseline=1, localizer=1, predicted=1, oracle=1))
                options = dict(device="cpu", config=settings, input_mode="l1", save_every=1)
                training.train_losver(root / "data", root / "control", "fixture", method, **options)
                if method == "unixcoder":
                    # The last optimizer update is saved, but validation has not run.
                    with self.interruption(step=3, after_save=True), self.assertRaises(Interrupted):
                        training.train_losver(root / "data", root / "resumed", "fixture", method, **options)
                else:
                    # Validation/best selection is committed, but source-only
                    # localization export fails. This must not mark training done.
                    with patch.object(evaluation, "localize_paired", side_effect=Interrupted), self.assertRaises(Interrupted):
                        training.train_losver(root / "data", root / "resumed", "fixture", method, **options)
                run = root / "resumed/primevul_paired" / method
                self.assertFalse((run / "training.metrics.json").exists())
                with patch.object(training.LOSVERAdamW, "step", side_effect=AssertionError("updates must not replay")):
                    training.train_losver(root / "data", root / "resumed", "fixture", method, resume=True, **options)
                names = ["best-f1.pt"] if method == "losver_localizer" else ["best-p_c.pt"]
                self.compare_runs(root / "control/primevul_paired" / method, run, names)


if __name__ == "__main__":
    unittest.main()
