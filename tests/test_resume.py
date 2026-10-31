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
        if "attention_ema" in a:
            self.assert_state_equal(a["attention_ema"], b["attention_ema"])
        for name in ("ema_updates", "suppression_rng", "attention_previous"):
            if name in a["loop"]:
                self.assert_state_equal(a["loop"][name], b["loop"][name])
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
        for name in ("training.candidates.jsonl", "attention.diagnostics.jsonl"):
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
                for supervision in ("diff", "repair_region", "neighbour_region", "diff_neighbour"):
                    weak = []
                    predicted = []
                    scores = []
                    for row in rows:
                        count = len(row["func"].splitlines())
                        weak.append(
                            dict(row, lines_original_diff=row["lines_ground_truth"], neighbour_lines=[])
                            if supervision == "diff_neighbour"
                            else dict(
                                row,
                                region_labels=[float(i == 1) for i in range(count)],
                                region_weights=[
                                    0.2 if i != 1 else 1.0 for i in range(count)
                                ],
                            )
                        )
                        ns = paired_encoding(row["func"], Tokens(), mode)[
                            "line_numbers"
                        ]
                        predicted.append(
                            dict(row, line_numbers=ns, lines_predicted=[1])
                        )
                        scores.append(
                            dict(
                                sample_key=row["sample_key"],
                                line_numbers=ns,
                                line_scores=[0.8 if i == 1 else 0.2 for i in ns],
                            )
                        )
                    write_jsonl(
                        root / output / "supervision" / supervision / f"{split}.jsonl",
                        weak,
                    )
                    location = (
                        root / output / "primevul_paired" / (supervision + "_localizer")
                    )
                    write_jsonl(location / f"{split}.predicted.jsonl", predicted)
                    write_jsonl(location / f"{split}.line_scores.jsonl", scores)

    def test_all_losver_variants_resume_random_sampler_dropout_and_scheduler(self):
        def factory(path, variant, *, window_micro_batch=8, gradient_checkpointing=False):
            model = paired_model(variant == "localizer")
            model.window_micro_batch = window_micro_batch
            model.gradient_checkpointing = gradient_checkpointing
            return model, Tokens()

        for mode in ("512", "l1"):
            for method in (
                training.LOSVER_METHODS
                if mode == "512"
                else ("unixcoder", "losver_localizer", "losver", "losver_oracle")
            ):
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
                    names = (
                        ["best-f1.pt"]
                        if training.LOSVER_VARIANTS[method] == "localizer"
                        else ["best-p_c.pt"]
                    )
                    self.compare_runs(root / "control/primevul_paired" / method, run, names)
                    if training.LOSVER_VARIANTS[method] == "localizer":
                        for split in (
                            ("train", "valid", "test")
                            if method == "losver_localizer"
                            else ("train", "valid")
                        ):
                            self.assertEqual(
                                (
                                    root
                                    / "control/primevul_paired"
                                    / method
                                    / f"{split}.predicted.jsonl"
                                ).read_bytes(),
                                (run / f"{split}.predicted.jsonl").read_bytes(),
                            )

    def test_beta_validation_grid_keeps_hard_weight_sampling_rng(self):
        def factory(path, variant, **kwargs):
            return paired_model(variant == "localizer"), Tokens()

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            core, "losver_model", side_effect=factory
        ):
            root = Path(tmp)
            self.paired_fixture(root, "512")
            config = dict(
                training.LOSVER_CONFIG,
                batch_size=4,
                eval_batch_size=2,
                epochs=dict(baseline=2, localizer=2, predicted=2, oracle=2),
            )
            for output, method in (
                ("control", "diff_hard_weight"),
                ("resumed", "diff_robust_fusion"),
            ):
                training.train_losver(
                    root / "data",
                    root / output,
                    "fixture",
                    method,
                    config=config,
                    device="cpu",
                    save_every=100,
                )
            a = torch.load(
                root / "control/primevul_paired/diff_hard_weight/last.pt",
                weights_only=False,
            )
            b = torch.load(
                root / "resumed/primevul_paired/diff_robust_fusion/last.pt",
                weights_only=False,
            )
            self.assert_state_equal(a["rng"], b["rng"])
            self.assertEqual(a["loop"]["global_step"], b["loop"]["global_step"])

    def test_localizer_rdrop_and_decay_resume_exactly(self):
        from tests.test_localizer_regularization import localizer

        def factory(path, variant, **kwargs):
            return localizer(kwargs.get("dropout_probability", 0.)), Tokens()

        for coefficient in (0., .1):
            with self.subTest(coefficient=coefficient), tempfile.TemporaryDirectory() as tmp, patch.object(core,"losver_model",side_effect=factory):
                root=Path(tmp)
                self.paired_fixture(root,"512")
                settings=dict(training.LOSVER_CONFIG,batch_size=4,eval_batch_size=2,
                              epochs=dict(baseline=2,localizer=2,predicted=2,oracle=2))
                options=dict(device="cpu",config=settings,input_mode="512",save_every=1,
                             localizer_dropout=.1,localizer_weight_decay=.01,localizer_rdrop=coefficient)
                training.train_losver(root/"data",root/"control","fixture","losver_localizer",**options)
                with self.interruption(epoch=2,step=4,after_save=True), self.assertRaises(Interrupted):
                    training.train_losver(root/"data",root/"resumed","fixture","losver_localizer",**options)
                training.train_losver(root/"data",root/"resumed","fixture","losver_localizer",resume=True,**options)
                self.compare_runs(root/"control/primevul_paired/losver_localizer",root/"resumed/primevul_paired/losver_localizer",["best-f1.pt"])

    def test_line_auxiliary_resume_preserves_labels_components_and_rng(self):
        from tests.test_line_auxiliary import detector

        def factory(path, variant, **kwargs):
            return detector(kwargs.get("line_auxiliary_weight")), Tokens()

        for supervision in ("diff", "random"):
            with self.subTest(supervision=supervision), tempfile.TemporaryDirectory() as tmp, patch.object(core, "losver_model", side_effect=factory):
                root = Path(tmp)
                self.paired_fixture(root, "512")
                settings = dict(training.LOSVER_CONFIG, batch_size=4, eval_batch_size=2,
                                epochs=dict(baseline=2, localizer=2, predicted=2, oracle=2))
                options = dict(device="cpu", config=settings, input_mode="512", save_every=1,
                               line_auxiliary=supervision, line_auxiliary_weight=.3)
                training.train_losver(root/"data", root/"control", "fixture", "unixcoder", **options)
                with self.interruption(epoch=2, step=4, after_save=True), self.assertRaises(Interrupted):
                    training.train_losver(root/"data", root/"resumed", "fixture", "unixcoder", **options)
                training.train_losver(root/"data", root/"resumed", "fixture", "unixcoder", resume=True, **options)
                a, b = [root/name/"primevul_paired/unixcoder" for name in ("control", "resumed")]
                self.compare_runs(a, b, ["best-p_c.pt"])
                self.assertEqual((a/"train.line_labels.jsonl").read_bytes(), (b/"train.line_labels.jsonl").read_bytes())

    def test_bounded_attention_resume_with_eligible_function_accumulation(self):
        from tests.test_bounded_attention import detector

        def factory(path,variant,**kwargs):
            return detector(kwargs.get("bounded_attention",False),kwargs.get("attention_region")),Tokens()

        for supervision in (None,"diff","random"):
            with self.subTest(supervision=supervision),tempfile.TemporaryDirectory() as tmp,patch.object(core,"losver_model",side_effect=factory):
                root=Path(tmp);self.paired_fixture(root,"512")
                path=root/"data/primevul_paired/train.lines.jsonl"
                rows=read_jsonl(path)
                for i,r in enumerate(rows):
                    if i%3==0: r["lines_ground_truth"]=[]
                    if i%5==0: r["lines_ground_truth"]=list(range(len(r["func"].splitlines())))
                write_jsonl(path,rows)
                settings=dict(training.LOSVER_CONFIG,batch_size=4,eval_batch_size=2,
                              epochs=dict(baseline=2,localizer=2,predicted=2,oracle=2))
                options=dict(device="cpu",config=settings,input_mode="512",save_every=1,
                             bounded_attention=True,attention_region=supervision,function_micro_batch=2)
                training.train_losver(root/"data",root/"control","fixture","unixcoder",**options)
                with self.interruption(epoch=2,step=4,after_save=True),self.assertRaises(Interrupted):
                    training.train_losver(root/"data",root/"resumed","fixture","unixcoder",**options)
                training.train_losver(root/"data",root/"resumed","fixture","unixcoder",resume=True,**options)
                self.compare_runs(root/"control/primevul_paired/unixcoder",root/"resumed/primevul_paired/unixcoder",["best-p_c.pt"])

    def test_attention_regularization_resume_including_epoch_diagnostics(self):
        from tests.test_bounded_attention import detector

        def factory(path,variant,**kwargs):
            return detector(True,"diff",kwargs.get("attention_suppression")),Tokens()

        for mode in ("none","random","high","ema"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp,patch.object(core,"losver_model",side_effect=factory):
                root=Path(tmp);self.paired_fixture(root,"512")
                for split in ("train","valid"):
                    path=root/"data/primevul_paired"/f"{split}.lines.jsonl"
                    rows=read_jsonl(path)
                    for i,r in enumerate(rows):
                        r["func"]="\n".join(f"x += {j+i};" for j in range(20))
                        r["lines_ground_truth"]=[] if i%3==0 else list(range(20)) if i%5==0 else [2,3]
                    write_jsonl(path,rows)
                settings=dict(training.LOSVER_CONFIG,batch_size=4,eval_batch_size=2,
                              epochs=dict(baseline=2,localizer=2,predicted=2,oracle=2))
                options=dict(device="cpu",config=settings,save_every=1,bounded_attention=True,
                    attention_region="diff",function_micro_batch=2,attention_diagnostics=True,
                    attention_suppression=mode if mode in ("random","high") else None,attention_ema=mode=="ema")
                training.train_losver(root/"data",root/"control","fixture","unixcoder",**options)
                with self.interruption(epoch=2,step=4,after_save=True),self.assertRaises(Interrupted):
                    training.train_losver(root/"data",root/"resumed","fixture","unixcoder",**options)
                path=root/"resumed/primevul_paired/unixcoder/attention.diagnostics.jsonl"
                with path.open("a") as handle: handle.write('uncommitted diagnostic\n')
                training.train_losver(root/"data",root/"resumed","fixture","unixcoder",resume=True,**options)
                self.compare_runs(root/"control/primevul_paired/unixcoder",root/"resumed/primevul_paired/unixcoder",["best-p_c.pt"])
                cp=torch.load(root/"control/primevul_paired/unixcoder/last.pt",weights_only=False)
                if mode=="ema": self.assertEqual(cp["loop"]["ema_updates"],6)
                if mode=="random": self.assertIn("suppression_rng",cp["loop"])
                changed=dict(options,attention_suppression=None,attention_ema=mode!="ema")
                with self.assertRaisesRegex(ValueError,"completed training configuration"):
                    training.train_losver(root/"data",root/"control","fixture","unixcoder",resume=True,**changed)

    def test_attention_regularization_accumulation_and_actual_ema_objective(self):
        from tests.test_bounded_attention import detector

        def factory(path,variant,**kwargs):
            m=detector(True,"diff",kwargs.get("attention_suppression"))
            for module in m.modules():
                if isinstance(module,torch.nn.Dropout): module.p=0.
            # FP64 isolates reduction/normalization from FP32 kernel roundoff.
            return m.double(),Tokens()

        for mode in ("random","high","ema"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp,patch.object(core,"losver_model",side_effect=factory):
                root=Path(tmp);self.paired_fixture(root,"512")
                path=root/"data/primevul_paired/train.lines.jsonl"
                rows=read_jsonl(path)
                for i,r in enumerate(rows):
                    r["func"]="\n".join(f"x += {j+i};" for j in range(12))
                    r["lines_ground_truth"]=[] if i%3==0 else [2,3]
                write_jsonl(path,rows)
                settings=dict(training.LOSVER_CONFIG,batch_size=4,eval_batch_size=2,learning_rate=.005,
                              epochs=dict(baseline=2,localizer=2,predicted=2,oracle=2))
                options=dict(device="cpu",config=settings,save_every=3,bounded_attention=True,attention_region="diff",
                    attention_suppression=mode if mode in ("random","high") else None,attention_ema=mode=="ema")
                for micro in (4,2):
                    training.train_losver(root/"data",root/f"micro{micro}","fixture","unixcoder",function_micro_batch=micro,**options)
                a,b=(torch.load(root/f"micro{m}/primevul_paired/unixcoder/last.pt",weights_only=False) for m in (4,2))
                for k in a["model"]: torch.testing.assert_close(a["model"][k],b["model"][k],rtol=1e-8,atol=1e-9)
                if mode=="ema":
                    self.assertEqual(a["loop"]["ema_updates"],6)
                    self.assertEqual(b["loop"]["ema_updates"],6)
                    events=[r for r in read_jsonl(root/"micro2/primevul_paired/unixcoder/best.training.jsonl") if r["event"]=="epoch"]
                    self.assertGreater(events[-1]["train_components"]["attention_kl"],0.)
                    for r in events:
                        c=r["train_components"]
                        self.assertAlmostEqual(r["train_loss"],c["bce"]+.1*c["region_loss"]+.1*c["attention_kl"],places=8)

    def test_attention_diagnostics_preserve_training_rng_weights_and_selection(self):
        from tests.test_bounded_attention import detector

        def factory(path,variant,**kwargs):
            return detector(True,"diff"),Tokens()

        with tempfile.TemporaryDirectory() as tmp,patch.object(core,"losver_model",side_effect=factory):
            root=Path(tmp);self.paired_fixture(root,"512")
            settings=dict(training.LOSVER_CONFIG,batch_size=4,eval_batch_size=2,
                          epochs=dict(baseline=2,localizer=2,predicted=2,oracle=2))
            options=dict(device="cpu",config=settings,bounded_attention=True,attention_region="diff",function_micro_batch=2)
            for flag in (False,True):
                training.train_losver(root/"data",root/str(flag),"fixture","unixcoder",attention_diagnostics=flag,**options)
            paths=[root/str(f)/"primevul_paired/unixcoder" for f in (False,True)]
            a,b=(torch.load(p/"last.pt",weights_only=False) for p in paths)
            for k in ("model","optimizer","scheduler","rng"):
                self.assert_state_equal(a[k],b[k])
            for k in ("best","chosen"):
                self.assert_state_equal(a["loop"][k],b["loop"][k])

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
