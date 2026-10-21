import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_codebert_source import CodeTokenizer
from vulnmechanism import model as core
from vulnmechanism.cli import build_parser, main
from vulnmechanism.data import read_jsonl, write_jsonl
from vulnmechanism.evaluation import evaluate, evaluate_transfer, report
from vulnmechanism.metrics import metrics, select_threshold


class EvaluationTests(unittest.TestCase):
    def test_reveal_reuses_train_evaluate_and_transfer_cli_without_changing_defaults(self):
        from vulnmechanism.data import DATASETS

        parser = build_parser()
        train = parser.parse_args(["train", "--dataset", "reveal", "--methods", "bce"])
        self.assertEqual((train.dataset, train.methods, train.train_limit, train.valid_limit),
                         ("reveal", ["bce"], None, None))
        evaluate_args = parser.parse_args(["evaluate", "--dataset", "reveal", "--clean-only"])
        self.assertEqual((evaluate_args.dataset, evaluate_args.clean_only), ("reveal", True))
        transfer = parser.parse_args(["transfer", "--datasets", "primevul", "reveal"])
        self.assertEqual((transfer.datasets, transfer.methods), (["primevul", "reveal"], ["bce"]))
        self.assertEqual(parser.parse_args(["train"]).dataset, "all")
        self.assertNotIn("reveal", DATASETS)

    def test_metrics_threshold_and_single_class(self):
        result = metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["mcc"], 0.0)
        self.assertAlmostEqual(result["auc"], 0.75)
        self.assertAlmostEqual(result["auprc"], 5 / 6)
        self.assertIsNone(metrics([0, 0], [0.1, 0.9])["auc"])
        threshold, selected = select_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(threshold, 0.5)
        self.assertEqual(selected["mcc"], 1.0)

    def test_clean_evaluation_reads_only_own_valid_and_test(self):
        from vulnmechanism.data import DATASETS
        for dataset in (*DATASETS, "reveal"):
            target = "unused_dataset"
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for d in (dataset, target):
                    rows = [
                        dict(
                            sample_key=f"{d}:{i}",
                            raw_source=f"int f{i}(){{return {i};}}",
                            label=i % 2,
                            dataset=d,
                            split="test",
                            train_overlap=[dataset] if i == 3 else [],
                        )
                        for i in range(4)
                    ]
                    write_jsonl(root / "data" / d / "test.jsonl", rows)
                valid = [
                    dict(
                        sample_key=f"v{i}",
                        raw_source=f"valid {i}",
                        label=i,
                        dataset=dataset,
                        split="valid",
                    )
                    for i in (0, 1)
                ]
                write_jsonl(root / "data" / dataset / "valid.jsonl", valid)
                (root / "data" / target / "valid.jsonl").write_text(
                    "TARGET VALID MUST NEVER BE READ"
                )
                run = root / "runs" / dataset / "bce"
                run.mkdir(parents=True)
                (run / "training.metrics.json").write_text("{}")
                torch.save(
                    dict(
                        dataset=dataset,
                        model_path="fixture",
                        source_max_length=2048,
                        decision_threshold=0.73,
                        selected_epoch=2,
                        training_config=dict(batch_size=2),
                    ),
                    run / "best.pt",
                )

                def predict(model, rows, builder, **kwargs):
                    self.assertEqual(kwargs["batch_size"], 2)
                    return torch.logit(
                        torch.tensor([0.1 if r["label"] == 0 else 0.7 for r in rows])
                    )

                with patch.object(
                    core.AutoTokenizer, "from_pretrained", return_value=CodeTokenizer()
                ), patch.object(core, "load_model", return_value=object()), patch.object(
                    core, "predict_logits", side_effect=predict
                ) as predictor:
                    evaluate(
                        root / "data",
                        root / "runs",
                        dataset,
                        ("bce",),
                        device="cpu",
                        robust=False,
                    )
                self.assertEqual(predictor.call_count, 2)
                self.assertFalse((run / f"{target}.test.metrics.json").exists())
                result = json.loads((run / "test.metrics.json").read_text())
                self.assertEqual(result["dataset"], dataset)
                self.assertEqual(result["split"], "test")
                self.assertEqual(result["selected_epoch"], 2)
                self.assertEqual(result["metrics"]["threshold"], 0.73)
                self.assertEqual(result["metrics"]["samples"], 3)
                self.assertEqual(result["metrics"]["confusion_matrix"], [[2, 0], [1, 0]])
                saved = read_jsonl(run / "test.predictions.jsonl")
                self.assertEqual(
                    [r["sample_key"] for r in saved], [f"{dataset}:{i}" for i in range(3)]
                )
                self.assertTrue(all(r["threshold"] == 0.73 for r in saved))

    def test_report_contains_twelve_models_and_both_test_transfer_deltas(self):
        from vulnmechanism.data import DATASETS
        from vulnmechanism.training import METHODS
        from vulnmechanism.artifacts import write_json
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in DATASETS:
                write_json(root / "data" / dataset / "statistics.json", dict(dataset=dataset))
                target = next(d for d in DATASETS if d != dataset)
                for method in METHODS:
                    run = root / "runs" / dataset / method
                    threshold = 0.73 if dataset == "primevul" else 0.61
                    own = metrics([0, 1], [0.1, 0.9], threshold)
                    transferred = metrics([0, 1], [0.9, 0.1], threshold)
                    write_json(
                        run / "training.metrics.json", dict(dataset=dataset, selected_epoch=2)
                    )
                    for split in ("valid", "test"):
                        write_json(run / f"{split}.metrics.json", dict(metrics=own))
                    write_json(run / f"{target}.test.metrics.json", dict(metrics=transferred))
                    result = dict(
                        worst_complete=transferred,
                        asr=dict(count=2, denominator=2, rate=1.0),
                        eligible=2,
                    )
                    bank = {
                        kind: result
                        for kind in ("all", "strict_semantic", "input_only_stress")
                    }
                    write_json(
                        run / "robustness.metrics.json",
                        dict(fixed_overall=bank, heldout_overall=bank, adaptive=bank),
                    )
            summary = report(root / "runs", root / "data")
            self.assertEqual((summary["models"], summary["rows"]), (12, 144))
            result = json.loads((root / "runs/comparison.json").read_text())
            transferred = [r for r in result["results"] if r["view"] == "transfer"]
            self.assertEqual(len(transferred), 12)
            self.assertTrue(
                all(
                    r["delta_accuracy"] == -1.0 and r["delta_mcc"] == -2.0 for r in transferred
                )
            )
            self.assertTrue(
                all(
                    r["dataset"] != r["training_dataset"] and r["split"] == "test"
                    for r in transferred
                )
            )
            with (root / "runs/comparison.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 144)

    def test_transfer_reuses_own_predictions_and_shared_overlap_free_targets(self):
        import csv
        from vulnmechanism.artifacts import write_json

        datasets = ("primevul", "secvuleval")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in datasets:
                other = next(d for d in datasets if d != dataset)
                for split in ("train", "valid"):
                    write_jsonl(root / "data" / dataset / f"{split}.jsonl", [dict(
                        raw_source=f"{dataset}-{split}", label=1,
                    )])
                rows = [dict(
                    sample_key=f"{dataset}:{i}", label=i % 2, dataset=dataset, split="test",
                    raw_source=source, train_overlap=[], project="fixture",
                    source_file="fixture.jsonl", source_row=i+1,
                ) for i, source in enumerate((
                    f"{other}-train", f"{other}-valid", f"{dataset}-clean0", f"{dataset}-clean1"
                ))]
                write_jsonl(root / "data" / dataset / "test.jsonl", rows)
                threshold = .73 if dataset == "primevul" else .61
                run = root / "runs" / dataset / "bce"
                epoch = 2 if dataset == "primevul" else 1
                write_json(run / "training.metrics.json", dict(selected_epoch=epoch))
                write_jsonl(run / "test.predictions.jsonl", [dict(
                    sample_key=r["sample_key"], label=r["label"],
                    logit=float(torch.logit(torch.tensor(.9 if r["label"] else .1))),
                    score=.9 if r["label"] else .1, threshold=threshold,
                ) for r in rows])
                write_json(run / "test.metrics.json", dict(
                    selected_epoch=epoch, metrics=metrics([0, 1, 0, 1], [.1, .9, .1, .9], threshold),
                ))
                torch.save(dict(
                    dataset=dataset, method="bce", model_path="fixture", source_max_length=2048,
                    selected_epoch=epoch, decision_threshold=threshold, training_config=dict(batch_size=2),
                ), run / "best.pt")

            def predict(model, rows, builder, **kwargs):
                self.assertNotEqual(model, rows[0]["dataset"])
                self.assertEqual(kwargs["batch_size"], 2)
                return torch.logit(torch.tensor([.7 if r["label"] else .8 for r in rows]))

            with patch.object(core.AutoTokenizer, "from_pretrained", return_value=CodeTokenizer()), patch.object(
                core, "load_model", side_effect=lambda cp, *a, **kw: cp["dataset"]
            ), patch.object(core, "predict_logits", side_effect=predict) as predictor:
                summary = evaluate_transfer(root / "data", root / "runs", datasets, device="cpu")
            self.assertEqual(predictor.call_count, 2)
            self.assertEqual(summary["rows"], 8)
            output = root / "runs/transfer_primevul_secvuleval"
            result = json.loads((output / "results.json").read_text())
            for dataset in datasets:
                self.assertEqual(result["cohorts"][dataset]["independent"], dict(samples=2, positive=1, negative=1))
                with (output / f"{dataset}.members.csv").open() as handle:
                    members = list(csv.DictReader(handle))
                self.assertEqual([m["sample_key"] for m in members if m["independent"] == "True"],
                    [f"{dataset}:2", f"{dataset}:3"])
            cross = [r for r in result["results"] if r["training_dataset"] != r["test_dataset"]]
            self.assertTrue(all(r["auc"] == 0 and r["delta_source_auc"] == -1 and r["delta_target_auc"] == -1 for r in cross))
            self.assertTrue(all(r["threshold"] == (.73 if r["training_dataset"] == "primevul" else .61) for r in cross))
            with patch.object(core, "predict_logits", side_effect=AssertionError("cached predictions must be reused")):
                evaluate_transfer(root / "data", root / "runs", datasets, device="cpu")
            for dataset in datasets:
                for method in ("lp_ft", "swa"):
                    run = root / "runs" / dataset / method
                    write_json(run / "training.metrics.json", dict(selected_epoch=3))
                    torch.save(dict(
                        dataset=dataset, method=method, model_path="fixture", source_max_length=2048,
                        selected_epoch=3, decision_threshold=.55, training_config=dict(batch_size=2),
                    ), run / "best.pt")

            def predict_new_method(model, rows, builder, **kwargs):
                self.assertIn(model[1], ("lp_ft", "swa"))  # BCE must reuse its saved predictions.
                return torch.logit(torch.tensor([.9 if r["label"] else .1 for r in rows]))

            with patch.object(core.AutoTokenizer, "from_pretrained", return_value=CodeTokenizer()), patch.object(
                core, "load_model", side_effect=lambda cp, *a, **kw: (cp["dataset"], cp["method"])
            ), patch.object(core, "predict_logits", side_effect=predict_new_method) as predictor:
                summary = evaluate_transfer(root / "data", root / "runs", datasets,
                                            methods=("bce", "lp_ft", "swa"), device="cpu")
            self.assertEqual((summary["models"], summary["rows"], predictor.call_count), (6, 24, 8))
            updated = json.loads((output / "results.json").read_text())
            self.assertEqual(updated["cohorts"], result["cohorts"])
            new_cross = [r for r in updated["results"] if r["method"] in ("lp_ft", "swa") and
                         r["training_dataset"] != r["test_dataset"]]
            self.assertTrue(all(r["auc"] == 1. and r["delta_bce_auc"] == 1. for r in new_cross))
            self.assertEqual([r for r in updated["results"] if r["method"] == "bce"], result["results"])

    def test_three_dataset_transfer_uses_actual_pilot_references_and_fixed_thresholds(self):
        from vulnmechanism.artifacts import write_json

        datasets = ("bigvul", "devign", "reveal")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, dataset in enumerate(datasets):
                for split in ("train", "valid"):
                    write_jsonl(root / "data/pilot" / dataset / f"{split}.jsonl",
                                [dict(raw_source=f"{dataset}-{split}", label=1)])
                    # This unused full-Train member was never used by the pilot.
                    write_jsonl(root / "data" / dataset / f"{split}.jsonl",
                                [dict(raw_source=f"{dataset}-unused", label=0)])
                other, third = datasets[(index + 1) % 3], datasets[(index + 2) % 3]
                write_jsonl(root / "data" / dataset / "test.jsonl", [dict(
                    sample_key=f"{dataset}:{i}", raw_source=source, label=i % 2,
                    dataset=dataset, split="test", train_overlap=False,
                    project="fixture", source_row=i + 1,
                ) for i, source in enumerate((f"{other}-train", f"{third}-valid",
                                             f"{dataset}-unused", f"{dataset}-clean",
                                             f"{dataset}-extra0", f"{dataset}-extra1"))])
                run = root / "runs" / dataset / "bce"
                write_json(run / "training.metrics.json", dict(selected_epoch=index + 1))
                torch.save(dict(dataset=dataset, method="bce", model_path="fixture",
                                source_max_length=2048, selected_epoch=index + 1,
                                decision_threshold=.6 + index * .1,
                                training_config=dict(batch_size=2)), run / "best.pt")
            with patch.object(core.AutoTokenizer, "from_pretrained", return_value=CodeTokenizer()), \
                 patch.object(core, "load_model", return_value=object()), \
                 patch.object(core, "predict_logits", side_effect=lambda model, rows, *a, **kw:
                              torch.tensor([2. if r["label"] else -2. for r in rows])) as predictor:
                summary = evaluate_transfer(root / "data", root / "runs", datasets,
                                            train_data_dir=root / "data/pilot", device="cpu")
            self.assertEqual((summary["models"], summary["rows"], predictor.call_count), (3, 18, 9))
            result = json.loads((root / "runs/transfer_bigvul_devign_reveal/results.json").read_text())
            for dataset in datasets:
                self.assertEqual(result["cohorts"][dataset]["excluded"]["samples"], 2)
                self.assertEqual(result["cohorts"][dataset]["independent"],
                                 dict(samples=4, positive=2, negative=2))
                import csv
                with (root / "runs/transfer_bigvul_devign_reveal" / f"{dataset}.members.csv").open() as handle:
                    saved_members = list(csv.DictReader(handle))
                self.assertTrue(all(r["source_file"] == "" and
                                    r["prepared_file"] == str(root / "data" / dataset / "test.jsonl")
                                    for r in saved_members))
            independent = [r for r in result["results"] if r["view"] == "independent"]
            for target in datasets:
                values = [r for r in independent if r["test_dataset"] == target]
                self.assertEqual(len(values), 3)
                self.assertTrue(all(r["samples"] == 4 and r["auc"] == 1 for r in values))
            for r in independent:
                self.assertAlmostEqual(r["threshold"], .6 + datasets.index(r["training_dataset"]) * .1)
            with patch.object(core, "predict_logits", side_effect=AssertionError("reuse cached predictions")):
                evaluate_transfer(root / "data", root / "runs", datasets,
                                  train_data_dir=root / "data/pilot", device="cpu")
            from vulnmechanism.static_experiment import transfer_report
            output = root / "runs/transfer_bigvul_devign_reveal"
            write_json(output / "overlap.json", dict(
                near_pairs=[dict(target_key=f"{d}:{i}", source_label=i % 2, target_label=i % 2)
                            for d in datasets for i in (2, 3)],
                shared_commit_pairs=[],
            ))
            with patch.object(core, "predict_logits", side_effect=AssertionError("report only saved predictions")):
                transfer_report(output, root / "runs")
                transfer_report(output, root / "runs")
            result = json.loads((output / "results.json").read_text())
            near = [r for r in result["results"] if r["view"] == "near_free"]
            self.assertEqual((len(result["results"]), len(near)), (27, 9))
            self.assertTrue(all(r["samples"] == 2 and r["positive"] == r["negative"] == 1
                                and r["auc"] == 1 for r in near))

    def test_clean_cache_predicts_only_restored_members_and_recomputes_metrics(self):
        from argparse import Namespace
        from vulnmechanism.artifacts import write_json
        from vulnmechanism.evaluation import _evaluate_clean

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(sample_key=str(i), label=i % 2, dataset="primevul", split="test", raw_source=str(i))
                    for i in range(4)]
            cached = [dict(sample_key=str(i), label=i % 2, logit=float(i * 2 - 1),
                           score=float(torch.sigmoid(torch.tensor(i * 2 - 1.))), threshold=.4,
                           prediction=i) for i in (0, 1)]
            write_jsonl(root / "test.predictions.jsonl", cached)
            write_json(root / "test.metrics.json", dict(selected_epoch=3, metrics={"samples": 2}))
            cp = dict(selected_epoch=3, decision_threshold=.4, source_max_length=2048)
            args = Namespace(model="fixture", device="cpu", batch_size=4)
            with patch.object(core.AutoTokenizer, "from_pretrained", return_value=CodeTokenizer()), patch.object(
                core, "load_model", return_value=object()
            ), patch.object(core, "predict_logits", return_value=torch.tensor([2., -2.])) as predictor:
                result = _evaluate_clean(rows, root, "test", cp, args)
            self.assertEqual([r["sample_key"] for r in predictor.call_args.args[1]], ["2", "3"])
            merged = read_jsonl(root / "test.predictions.jsonl")
            self.assertEqual(merged[:2], cached)
            self.assertEqual([r["sample_key"] for r in merged], ["0", "1", "2", "3"])
            self.assertEqual(result["metrics"], metrics([0, 1, 0, 1], [r["score"] for r in merged], .4))
            with patch.object(core, "predict_logits", side_effect=AssertionError("no repeat inference")), \
                    patch("vulnmechanism.evaluation.write_json", side_effect=AssertionError("no rewriting complete results")), \
                    patch("vulnmechanism.evaluation.write_jsonl", side_effect=AssertionError("no rewriting complete predictions")):
                self.assertEqual(_evaluate_clean(rows, root, "test", cp, args), result)

    def test_single_function_inference_requires_only_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.c"
            path.write_text("int f(){return 0;}")
            output = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "vulnmechanism",
                    "infer",
                    "--checkpoint",
                    "fixture.pt",
                    "--source-file",
                    str(path),
                    "--device",
                    "cpu",
                ],
            ), patch("torch.load", return_value=dict(decision_threshold=0.7)), patch.object(
                core, "predict_checkpoint", return_value=torch.tensor([2.0])
            ) as predictor, redirect_stdout(
                output
            ):
                main()
            result = json.loads(output.getvalue())
            self.assertEqual(result["prediction"], 1)
            self.assertEqual(result["threshold"], 0.7)
            self.assertEqual(predictor.call_args.args[1], [{"raw_source": path.read_text()}])


if __name__ == "__main__":
    unittest.main()
