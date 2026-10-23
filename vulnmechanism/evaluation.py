"""Clean and robustness evaluation, and saved experiment reports."""

from __future__ import annotations
from argparse import Namespace
import csv
import json
from pathlib import Path
import torch
from .artifacts import write_json
from .data import (
    DATASETS,
    balanced_subset,
    class_counts,
    read_jsonl,
    test_rows,
    validation_rows,
    write_jsonl,
)
from .metrics import decision_boundary, metrics
from . import model as core
from .progress import training_bar
from .training import METHODS, prepare_materials


def _evaluate_clean(rows, run, name, checkpoint, args, *, logits=None):
    output = run / f"{name}.metrics.json"
    predictions = run / f"{name}.predictions.jsonl"
    threshold = checkpoint["decision_threshold"]
    by_key = {}
    previous, cached = None, []
    if output.exists() and logits is None:
        previous = json.loads(output.read_text())
        if previous["selected_epoch"] != checkpoint["selected_epoch"]:
            raise ValueError("cached evaluation checkpoint selection differs")
        cached = read_jsonl(predictions)
        labels = {r["sample_key"]: r["label"] for r in rows}
        by_key = {r["sample_key"]: r for r in cached}
        if len(by_key) != len(cached) or any(
            r["sample_key"] not in labels
            or labels[r["sample_key"]] != r["label"]
            or r["threshold"] != threshold
            for r in cached
        ):
            raise ValueError("cached evaluation members, labels or threshold differ")
    missing = [r for r in rows if r["sample_key"] not in by_key]
    if missing:
        if logits is None:
            print(
                f"{run.name}/{name}: reuse {len(by_key)}, predict {len(missing)} members",
                flush=True,
            )
            device = core.resolve_device(args.device)
            builder = core.InputBuilder(
                core.AutoTokenizer.from_pretrained(args.model),
                source_max_length=checkpoint["source_max_length"],
            )
            model = core.load_model(checkpoint, device, model_path=args.model)
            logits = core.predict_logits(
                model, missing, builder, device=device, batch_size=args.batch_size
            )
        elif len(logits) != len(rows):
            raise ValueError("provided evaluation logits must cover the entire cohort")
        for row, logit, score in zip(missing, logits, logits.sigmoid().tolist()):
            by_key[row["sample_key"]] = dict(
                sample_key=row["sample_key"],
                label=row["label"],
                logit=float(logit),
                score=score,
                prediction=int(score >= decision_boundary(threshold)),
                threshold=threshold,
            )
    ordered = [by_key[r["sample_key"]] for r in rows]
    scores = [r["score"] for r in ordered]
    logits = torch.tensor([r["logit"] for r in ordered], dtype=torch.float32)
    result = dict(
        selected_epoch=checkpoint["selected_epoch"],
        dataset=rows[0]["dataset"],
        split=rows[0]["split"],
        metrics=metrics([r["label"] for r in rows], scores, threshold),
        bce=float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.tensor([r["label"] for r in rows], dtype=torch.float32)
            )
        ),
    )
    if result == previous and ordered == cached:
        return result
    write_jsonl(predictions, ordered)
    write_json(output, result)
    return result


def _robustness_fixed_candidates(rows, materials, root, tokenizer, source_budget):
    from . import augmentation as data
    from .syntax import SURFACE_OPERATIONS
    from .artifacts import write_json
    import random

    if not rows or any(r["split"] != "test" for r in rows):
        raise ValueError("robustness evaluation requires nonempty Test rows")
    evaluation_rows = rows
    paths = {}
    for bank in ("train", "heldout"):
        name = "fixed" if bank == "train" else "heldout"
        path = root / f"{name}.candidates.jsonl"
        paths[name] = path
        if path.exists():
            with path.open() as handle:
                cached = [json.loads(l) for l in handle]
            if [r["sample_key"] for r in cached] != [
                r["sample_key"] for r in evaluation_rows
            ]:
                raise ValueError("robustness candidate cache members/order differ")
            continue
        entries = []
        for index, row in enumerate(evaluation_rows):
            groups = {}
            reasons = {}
            for kind in (*SURFACE_OPERATIONS, "combined"):
                combos = (
                    ([kind],)
                    if kind != "combined"
                    else (
                        materials["rules"]["heldout_combinations"]
                        if bank == "heldout"
                        else (
                            ["local_name", "comment_insert"],
                            ["local_name", "local_name", "comment_replace"],
                        )
                    )
                )
                candidates = []
                for combo in combos:
                    result = data.search_surface_views(
                        [row],
                        None,
                        tokenizer,
                        source_budget,
                        materials,
                        seed=42,
                        bank=bank,
                        purpose=name + "/" + kind + "/" + ",".join(combo),
                        required_combination=combo,
                    )[0]
                    for key, value in result["reasons"].items():
                        reasons[key] = reasons.get(key, 0) + value
                    candidates.extend(
                        c for c in result["visited"] if len(c["units"]) == len(combo)
                    )
                rng = random.Random(
                    data.surface_seed(42, 0, row["sample_key"], name + "/" + kind)
                )
                rng.shuffle(candidates)
                groups[kind] = [
                    {k: v for k, v in c.items() if k != "raw_source"}
                    for c in candidates[:2]
                ]
            entries.append(
                dict(sample_key=row["sample_key"], groups=groups, reasons=reasons)
            )
            if (index + 1) % 100 == 0:
                print(
                    f"Test {name} candidates {index+1}/{len(evaluation_rows)}",
                    flush=True,
                )
        with path.open("x") as handle:
            for entry in entries:
                handle.write(json.dumps(entry) + "\n")
        write_json(
            root / f"{name}.coverage.metrics.json",
            {
                kind: dict(
                    eligible=sum(bool(e["groups"][kind]) for e in entries),
                    total=len(entries),
                    candidates=sum(len(e["groups"][kind]) for e in entries),
                )
                for kind in (*SURFACE_OPERATIONS, "combined")
            },
        )
    return paths


def _robustness_summary(originals, views, threshold):
    import numpy as np
    import torch
    from .metrics import metrics

    probability = lambda z: torch.sigmoid(torch.tensor(z, dtype=torch.float32)).item()
    selected = []
    worst = []
    eligible = []
    correct_to_wrong = 0
    wrong_to_correct = 0
    robust = 0
    all_correct = 0
    attacked = 0
    escape = 0
    false_alarm = 0
    for row in originals:
        candidates = views.get(row["sample_key"], [])
        y = row["label"]
        original_correct = row["prediction"] == y
        pool = [dict(logit=row["logit"], prediction=row["prediction"])] + candidates
        worst_view = max(
            pool, key=lambda v: float(np.logaddexp(0, v["logit"]) - y * v["logit"])
        )
        worst.append(dict(label=y, score=probability(worst_view["logit"])))
        if not candidates:
            continue
        first = candidates[0]
        prediction = first["prediction"]
        selected.append(dict(label=y, score=probability(first["logit"])))
        eligible.append(row)
        correct = prediction == y
        robust += original_correct and correct
        correct_to_wrong += original_correct and not correct
        wrong_to_correct += not original_correct and correct
        protected = original_correct and all(v["prediction"] == y for v in candidates)
        all_correct += protected
        failed = original_correct and not protected
        attacked += failed
        escape += failed and y == 1
        false_alarm += failed and y == 0
    n = len(eligible)
    original_correct = sum(r["prediction"] == r["label"] for r in eligible)
    positive = sum(r["label"] for r in eligible)
    negative = n - positive
    rate = lambda value, denominator: value / denominator if denominator else None
    return dict(
        eligible=n,
        total=len(originals),
        positive=positive,
        negative=negative,
        coverage=rate(n, len(originals)),
        candidates=sum(len(c) for c in views.values()),
        perturbed=(
            metrics(
                [r["label"] for r in selected],
                [r["score"] for r in selected],
                threshold,
            )
            if selected
            else None
        ),
        worst_complete=metrics(
            [r["label"] for r in worst], [r["score"] for r in worst], threshold
        ),
        paired_robust_correct=robust,
        paired_robust_accuracy=rate(robust, n),
        all_candidates_correct=all_correct,
        all_candidates_correct_rate=rate(all_correct, n),
        correct_to_wrong=dict(
            count=correct_to_wrong,
            denominator=original_correct,
            rate=rate(correct_to_wrong, original_correct),
        ),
        wrong_to_correct=dict(
            count=wrong_to_correct,
            denominator=n - original_correct,
            rate=rate(wrong_to_correct, n - original_correct),
        ),
        asr=dict(
            count=attacked,
            denominator=original_correct,
            rate=rate(attacked, original_correct),
        ),
        vulnerability_escape=dict(
            count=escape,
            denominator=sum(r["label"] == 1 and r["prediction"] == 1 for r in eligible),
            rate=rate(
                escape, sum(r["label"] == 1 and r["prediction"] == 1 for r in eligible)
            ),
        ),
        benign_false_alarm=dict(
            count=false_alarm,
            denominator=sum(r["label"] == 0 and r["prediction"] == 0 for r in eligible),
            rate=rate(
                false_alarm,
                sum(r["label"] == 0 and r["prediction"] == 0 for r in eligible),
            ),
        ),
    )


def _robustness_evaluation(rows, run, target, cp, root, materials, args):
    import torch, time
    import numpy as np
    from . import model as core, augmentation as data
    from .artifacts import write_json
    from .metrics import decision_boundary

    boundary = decision_boundary(cp["decision_threshold"])
    evaluation_rows = rows
    by = {r["sample_key"]: r for r in rows}
    if not rows or any(r["split"] != "test" for r in rows):
        raise ValueError("robustness evaluation requires nonempty Test rows")
    clean = _evaluate_clean(rows, run, "test", cp, args)
    with (run / "test.predictions.jsonl").open() as handle:
        originals = [json.loads(l) for l in handle]
    output = run / "robustness.metrics.json"
    if output.exists():
        cached = json.loads(output.read_text())
        if (
            cached.get("threshold") != cp["decision_threshold"]
            or cached.get("selected_epoch") != cp["selected_epoch"]
            or cached.get("checkpoint") != str(target)
            or cached.get("clean") != clean
        ):
            raise ValueError("cached robustness checkpoint/evaluation differs")
        return cached
    model = core.load_model(cp, device=core.resolve_device(args.device))
    builder = core.InputBuilder(
        core.AutoTokenizer.from_pretrained(args.model),
        source_max_length=cp["source_max_length"],
    )
    costs = dict(
        calls=0,
        function_forwards=0,
        encoder_sequences=0,
        encoder_token_slots=0,
        seconds=0.0,
    )

    def score(batch):
        started = time.monotonic()
        values = []
        with torch.no_grad(), core.auxiliary_mode(model):
            for at in range(0, len(batch), args.batch_size):
                grouped = builder.codebert_batch(
                    batch[at : at + args.batch_size],
                    device=core.resolve_device(args.device),
                )
                values.extend(model(*grouped).detach().cpu().tolist())
                costs["calls"] += 1
                costs["function_forwards"] += grouped[-1]
                costs["encoder_sequences"] += len(grouped[0])
                costs["encoder_token_slots"] += grouped[0].numel()
        costs["seconds"] += time.monotonic() - started
        return values

    report = dict(
        clean=clean,
        threshold=cp["decision_threshold"],
        selected_epoch=cp["selected_epoch"],
        checkpoint=str(target),
    )
    report["split"] = "test"
    for name in ("fixed", "heldout"):
        costs.update({k: 0 for k in costs})
        started = time.monotonic()
        with (root / f"{name}.candidates.jsonl").open() as handle:
            entries = [json.loads(l) for l in handle]
        views = {}
        flat = []
        index = []
        for entry in entries:
            row = by[entry["sample_key"]]
            for kind, candidates in entry["groups"].items():
                for ci, candidate in enumerate(candidates):
                    flat.append(
                        dict(
                            row,
                            raw_source=data.replay_surface_units(
                                row["raw_source"], candidate["units"]
                            ),
                        )
                    )
                    index.append(
                        (row["sample_key"], kind, ci, candidate["strict_semantic"])
                    )
        logits = score(flat) if flat else []
        for (key, kind, ci, strict), z in zip(index, logits):
            views.setdefault(kind, {}).setdefault(key, []).append(
                dict(
                    logit=z,
                    prediction=int(torch.sigmoid(torch.tensor(z)).item() >= boundary),
                    strict_semantic=strict,
                )
            )
        report[name] = {}
        from .syntax import SURFACE_OPERATIONS

        for kind in (*SURFACE_OPERATIONS, "combined"):
            group = views.get(kind, {})
            strict = {
                key: [v for v in candidates if v["strict_semantic"]]
                for key, candidates in group.items()
            }
            stress = {
                key: [v for v in candidates if not v["strict_semantic"]]
                for key, candidates in group.items()
            }
            report[name][kind] = dict(
                all=_robustness_summary(originals, group, cp["decision_threshold"]),
                strict_semantic=_robustness_summary(
                    originals, strict, cp["decision_threshold"]
                ),
                input_only_stress=_robustness_summary(
                    originals, stress, cp["decision_threshold"]
                ),
            )
        combined = {
            r["sample_key"]: [
                v for group in views.values() for v in group.get(r["sample_key"], [])
            ]
            for r in originals
        }
        report[name + "_overall"] = dict(
            all=_robustness_summary(originals, combined, cp["decision_threshold"]),
            strict_semantic=_robustness_summary(
                originals,
                {
                    k: [v for v in vs if v["strict_semantic"]]
                    for k, vs in combined.items()
                },
                cp["decision_threshold"],
            ),
            input_only_stress=_robustness_summary(
                originals,
                {
                    k: [v for v in vs if not v["strict_semantic"]]
                    for k, vs in combined.items()
                },
                cp["decision_threshold"],
            ),
        )
        with (run / f"{name}.predictions.jsonl").open("x") as handle:
            for row in originals:
                handle.write(
                    json.dumps(
                        dict(
                            sample_key=row["sample_key"],
                            label=row["label"],
                            groups={
                                k: g.get(row["sample_key"], [])
                                for k, g in views.items()
                            },
                        )
                    )
                    + "\n"
                )
        report.setdefault("evaluation_costs", {})[name] = dict(
            costs, total_seconds=time.monotonic() - started
        )
    # Each model receives the same initial generator states, then its own beams.
    started = time.monotonic()
    attacks = []
    groups = {}
    strict = {}
    stress = {}
    queries = 0
    original_queries = 0
    original_logits = {r["sample_key"]: r["logit"] for r in originals}
    attack_path = run / "adaptive.predictions.jsonl"
    if attack_path.exists():
        with attack_path.open() as handle:
            attacks = [json.loads(l) for l in handle]
    else:
        with attack_path.open("x") as handle:
            for at in range(0, len(evaluation_rows), args.batch_size):
                costs.update({k: 0 for k in costs})
                batch_started = time.monotonic()
                result = data.search_surface_views(
                    evaluation_rows[at : at + args.batch_size],
                    score,
                    builder.tokenizer,
                    cp["source_max_length"] - len(builder.source_prefix) - 2,
                    materials,
                    seed=42,
                    query_limit=48,
                    purpose="adaptive",
                    original_logits=original_logits,
                )
                result[0]["batch_cost"] = dict(
                    costs, total_seconds=time.monotonic() - batch_started
                )
                for entry in result:
                    entry["visited"] = [
                        {k: v for k, v in c.items() if k != "raw_source"}
                        for c in entry["visited"]
                    ]
                    entry["selected"] = (
                        None
                        if entry["selected"] is None
                        else {
                            k: v
                            for k, v in entry["selected"].items()
                            if k != "raw_source"
                        }
                    )
                    handle.write(json.dumps(entry) + "\n")
                    attacks.append(entry)
                if (at // args.batch_size + 1) % 25 == 0:
                    print(
                        f"{cp['dataset']}/{run.name} Test adaptive: {min(at+args.batch_size,len(evaluation_rows))}/{len(evaluation_rows)}",
                        flush=True,
                    )
    if [e["sample_key"] for e in attacks] != [r["sample_key"] for r in evaluation_rows]:
        raise ValueError("incomplete adaptive evaluation; no silent retries")
    for entry in attacks:
        queries += entry["candidate_queries"]
        original_queries += entry["original_queries"]
        key = entry["sample_key"]
        group = [
            dict(
                logit=c["logit"],
                prediction=int(
                    torch.sigmoid(torch.tensor(c["logit"])).item() >= boundary
                ),
                strict_semantic=c["strict_semantic"],
            )
            for c in entry["visited"]
        ]
        group.sort(
            key=lambda c: float(
                np.logaddexp(0, c["logit"]) - by[key]["label"] * c["logit"]
            ),
            reverse=True,
        )
        groups[key] = group
        strict[key] = [c for c in group if c["strict_semantic"]]
        stress[key] = [c for c in group if not c["strict_semantic"]]
    report["adaptive"] = dict(
        all=_robustness_summary(originals, groups, cp["decision_threshold"]),
        strict_semantic=_robustness_summary(
            originals, strict, cp["decision_threshold"]
        ),
        input_only_stress=_robustness_summary(
            originals, stress, cp["decision_threshold"]
        ),
        candidate_queries=queries,
        original_queries=original_queries,
        query_limit=48,
        costs={
            k: sum(e.get("batch_cost", {}).get(k, 0) for e in attacks)
            for k in (*costs, "total_seconds")
        },
        per_function_queries={e["sample_key"]: e["candidate_queries"] for e in attacks},
        definition="model-specific width-two depth-three BCE search; original included in worst-case results",
    )
    del model
    write_json(output, report)
    return report


def prepare_candidates(data_dir, output_dir, model_path, dataset, *, test_limit=None):
    root = Path(output_dir) / dataset
    materials = prepare_materials(data_dir, output_dir, dataset)
    tokenizer = core.AutoTokenizer.from_pretrained(model_path)
    builder = core.InputBuilder(tokenizer)
    budget = builder.source_max_length - len(builder.source_prefix) - 2
    rows = balanced_subset(test_rows(Path(data_dir) / dataset), test_limit)
    _robustness_fixed_candidates(rows, materials, root, tokenizer, budget)


def evaluate(
    data_dir,
    output_dir,
    dataset,
    methods=METHODS,
    *,
    device="auto",
    robust=True,
    valid_limit=None,
    test_limit=None,
    splits=("valid", "test"),
):
    if dataset == "primevul_paired":
        if valid_limit is not None or test_limit is not None:
            raise ValueError("LOSVER evaluation keeps all official paired members")
        return evaluate_paired(
            data_dir, output_dir, methods, device=device, splits=splits
        )
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    if not splits or any(s not in ("valid", "test") for s in splits):
        raise ValueError("evaluation splits must be valid/test")
    if robust and "test" not in splits:
        raise ValueError("robustness evaluation requires Test")
    root = Path(output_dir) / dataset
    valid = (
        balanced_subset(validation_rows(Path(data_dir) / dataset), valid_limit)
        if "valid" in splits
        else None
    )
    own_test = (
        balanced_subset(test_rows(Path(data_dir) / dataset), test_limit)
        if "test" in splits
        else None
    )
    materials = prepare_materials(data_dir, output_dir, dataset) if robust else None
    for method in methods:
        run = root / method
        if not (run / "training.metrics.json").exists():
            raise ValueError(
                f"{dataset}/{method}: training must finish before evaluation"
            )
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        if checkpoint["dataset"] != dataset:
            raise ValueError("checkpoint training dataset differs")
        if checkpoint.get("diagnostic_limits", {}).get("valid") != valid_limit:
            raise ValueError("evaluation Valid limit differs from training selection")
        args = Namespace(
            model=checkpoint["model_path"],
            device=device,
            batch_size=checkpoint["training_config"]["batch_size"],
        )
        if valid is not None:
            _evaluate_clean(valid, run, "valid", checkpoint, args)
        if own_test is not None:
            _evaluate_clean(own_test, run, "test", checkpoint, args)
        if robust:
            if not all(
                (root / f"{name}.candidates.jsonl").exists()
                for name in ("fixed", "heldout")
            ):
                prepare_candidates(
                    data_dir,
                    output_dir,
                    checkpoint["model_path"],
                    dataset,
                    test_limit=test_limit,
                )
            _robustness_evaluation(
                own_test, run, run / "best.pt", checkpoint, root, materials, args
            )
        del checkpoint


def report(output_dir, data_dir):
    root = Path(output_dir)
    rows, training_runs = [], {}
    metric_names = ("accuracy", "precision", "recall", "f1", "mcc", "auc", "auprc")
    for dataset in ("primevul", "megavul"):
        training_runs[dataset] = {}
        for method in METHODS:
            run = root / dataset / method
            training = json.loads((run / "training.metrics.json").read_text())
            training_runs[dataset][method] = training
            common = dict(
                training_dataset=dataset,
                method=method,
                selected_epoch=training["selected_epoch"],
            )
            clean = {}
            for split in ("valid", "test"):
                clean[split] = json.loads((run / f"{split}.metrics.json").read_text())[
                    "metrics"
                ]
                rows.append(
                    dict(
                        common,
                        dataset=dataset,
                        split=split,
                        view="clean",
                        **clean[split],
                    )
                )
            robustness = json.loads((run / "robustness.metrics.json").read_text())
            for setting in ("fixed", "heldout", "adaptive"):
                value = robustness[
                    setting if setting == "adaptive" else setting + "_overall"
                ]
                for kind in ("all", "strict_semantic", "input_only_stress"):
                    result = value[kind]
                    rows.append(
                        dict(
                            common,
                            dataset=dataset,
                            split="test",
                            view=f"{setting}/{kind}",
                            **result["worst_complete"],
                            asr=result["asr"]["rate"],
                            asr_count=result["asr"]["count"],
                            asr_denominator=result["asr"]["denominator"],
                            eligible=result["eligible"],
                        )
                    )
    data = {
        d: json.loads((Path(data_dir) / d / "statistics.json").read_text())
        for d in ("primevul", "megavul")
    }
    write_json(
        root / "comparison.json", dict(data=data, training=training_runs, results=rows)
    )
    fields = [
        "training_dataset",
        "method",
        "dataset",
        "split",
        "view",
        "selected_epoch",
        "threshold",
        "samples",
        *metric_names,
        "tp",
        "tn",
        "fp",
        "fn",
        "asr",
        "asr_count",
        "asr_denominator",
        "eligible",
    ]
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return dict(
        datasets=2,
        models=2 * len(METHODS),
        rows=len(rows),
        comparison=str(root / "comparison.csv"),
    )


@torch.no_grad()
def validate_paired(model, rows, tokenizer, variant, device, *, loader=None):
    """Fixed >.5 decisions; localization uses modified-line positive-class F1."""
    import numpy as np
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )
    from .training import paired_loader, paired_forward
    from .metrics import paired_metrics

    loader = (
        loader
        if loader is not None
        else paired_loader(rows, tokenizer, variant, input_mode=model.input_mode)
    )
    model.eval()
    predictions = []
    scores = []
    line_y = []
    line_p = []
    loss = 0.0
    at = 0
    for batch in training_bar(loader, desc=f"Valid {variant}"):
        value, prob = paired_forward(model, batch, variant, device)
        loss += float(value)
        if variant == "localizer":
            for y, numbers, p in zip(batch["labels"], batch["line_numbers"], prob):
                p = p.cpu().tolist()
                valid = numbers >= 0
                ns = numbers[valid].tolist()
                ys = y[valid].tolist()
                if len(p) != len(ns) or any(v < 0 for v in ys):
                    raise ValueError("localizer scores and physical line labels differ")
                line_y.extend(ys)
                line_p.extend(p)
                predictions.append(
                    dict(
                        sample_key=rows[at]["sample_key"],
                        line_numbers=ns,
                        line_scores=p,
                        lines_predicted=[n for n, v in zip(ns, p) if v > 0.5],
                    )
                )
                at += 1
        else:
            scores.extend(prob[:, 0].cpu().tolist())
    if variant == "localizer":
        pred = np.asarray(line_p) > 0.5
        result = dict(
            f1=f1_score(line_y, pred, average="binary", zero_division=0),
            weighted_f1=f1_score(line_y, pred, average="weighted", zero_division=0),
            precision=precision_score(line_y, pred, zero_division=0),
            recall=recall_score(line_y, pred, zero_division=0),
            lines=len(line_y),
            modified_lines=sum(line_y),
            confusion_matrix=confusion_matrix(line_y, pred, labels=[0, 1]).tolist(),
            loss=loss / len(loader),
        )
    else:
        result = paired_metrics(rows, scores)
        result["loss"] = loss / len(loader)
        predictions = [
            dict(
                sample_key=r["sample_key"],
                pair_key=r["pair_key"],
                target=r["target"],
                probability=p,
                prediction=int(p > 0.5),
            )
            for r, p in zip(rows, scores)
        ]
    return result, predictions


def load_paired(checkpoint, device):
    if checkpoint.get("line_mapping") != "source_offsets":
        raise ValueError(
            "historical LOSVER checkpoint predates source-offset mapping; use a fresh corrected run"
        )
    model, tokenizer = core.losver_model(
        checkpoint["model_path"],
        checkpoint["variant"],
        window_micro_batch=checkpoint.get("window_micro_batch", 8),
    )
    model.input_mode = checkpoint["input_mode"]
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), tokenizer


@torch.no_grad()
def localize_paired(data_dir, output_dir, split, *, device="auto"):
    """Single-function prediction: this function never opens Diff annotations."""
    from .training import paired_loader, paired_forward

    device = core.resolve_device(device)
    root = Path(output_dir) / "primevul_paired/losver_localizer"
    cp = torch.load(root / "best-f1.pt", map_location="cpu", weights_only=False)
    model, tokenizer = load_paired(cp, device)
    rows = read_jsonl(Path(data_dir) / "primevul_paired" / f"{split}.jsonl")
    # No training/Valid/Test counterpart or modification label enters this loader.
    probabilities, line_numbers = [], []
    for batch in training_bar(
        paired_loader(rows, tokenizer, "localizer", input_mode=cp["input_mode"]),
        desc=f"Localize {split}",
    ):
        probabilities.extend(
            p.cpu().tolist()
            for p in paired_forward(
                model, batch, "localizer", device, with_labels=False
            )
        )
        line_numbers.extend(ns[ns >= 0].tolist() for ns in batch["line_numbers"])
    if len(probabilities) != len(rows):
        raise ValueError("localizer prediction count differs")
    predicted = [
        dict(r, line_numbers=ns, lines_predicted=[n for n, p in zip(ns, v) if p > 0.5])
        for r, ns, v in zip(rows, line_numbers, probabilities)
    ]
    write_jsonl(root / f"{split}.predicted.jsonl", predicted)
    write_jsonl(
        root / f"{split}.line_scores.jsonl",
        [
            dict(sample_key=r["sample_key"], line_numbers=ns, line_scores=v)
            for r, ns, v in zip(rows, line_numbers, probabilities)
        ],
    )
    del model, cp
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predicted


def evaluate_paired(
    data_dir, output_dir, methods, *, device="auto", splits=("valid", "test")
):
    from .training import LOSVER_METHODS, LOSVER_VARIANTS, paired_rows

    if any(m not in LOSVER_METHODS for m in methods):
        raise ValueError("official paired evaluation requires LOSVER methods")
    device = core.resolve_device(device)
    for method in methods:
        variant = LOSVER_VARIANTS[method]
        run = Path(output_dir) / "primevul_paired" / method
        cp = torch.load(
            run / ("best-f1.pt" if variant == "localizer" else "best-p_c.pt"),
            map_location="cpu",
            weights_only=False,
        )
        if not (run / "training.metrics.json").exists():
            raise ValueError("training must complete before formal evaluation")
        model, tokenizer = load_paired(cp, device)
        for split in splits:
            if variant == "localizer" and split == "test":
                # Normal detector localization is source-only. Oracle line-quality metrics are separate.
                localize_paired(data_dir, output_dir, split, device=device)
                continue
            rows = paired_rows(data_dir, output_dir, variant, split)
            value, predictions = validate_paired(
                model, rows, tokenizer, variant, device
            )
            write_json(
                run / f"{split}.metrics.json",
                dict(
                    selected_epoch=cp["selected_epoch"],
                    oracle=variant == "oracle",
                    metrics=value,
                ),
            )
            write_jsonl(run / f"{split}.predictions.jsonl", predictions)
            print(method, split, json.dumps(value), flush=True)
        del cp, model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def paired_report(output_dir, data_dir):
    from .training import LOSVER_CONFIG

    root = Path(output_dir) / "primevul_paired"
    data = json.loads((Path(data_dir) / "primevul_paired/statistics.json").read_text())
    results = []
    paper = dict(unixcoder=0.1011, losver=0.1401, losver_oracle=0.3316)
    input_modes = set()
    input_coverage = {}
    for method in ("unixcoder", "losver", "losver_oracle"):
        run = root / method
        training = json.loads((run / "training.metrics.json").read_text())
        if training.get("line_mapping") != "source_offsets":
            raise ValueError(
                "historical results are preserved; corrected runs require an empty output directory"
            )
        input_modes.add(training["input_mode"])
        input_coverage[method] = training.get("input_coverage")
        for split in ("valid", "test"):
            record = json.loads((run / f"{split}.metrics.json").read_text())
            if record["selected_epoch"] != training["selected_epochs"]["p_c"]:
                raise ValueError("saved evaluation uses a different selected epoch")
            v = record["metrics"]
            rows = read_jsonl(run / f"{split}.predictions.jsonl")
            from .metrics import paired_metrics

            checked = paired_metrics(rows, [r["probability"] for r in rows])
            if any(checked[k] != v[k] for k in checked):
                raise ValueError("saved metrics differ from predictions")
            results.append(
                dict(
                    method=method,
                    input_mode=training["input_mode"],
                    split=split,
                    oracle=method == "losver_oracle",
                    epoch=record["selected_epoch"],
                    **v,
                    paper_p_c=paper[method] if split == "test" else None,
                    delta_paper_p_c=(
                        v["p_c"] - paper[method] if split == "test" else None
                    ),
                )
            )
    if len(input_modes) != 1:
        raise ValueError("report cannot mix input budgets")
    input_mode = input_modes.pop()
    write_json(
        root / "comparison.json",
        dict(
            config=LOSVER_CONFIG,
            input_mode=input_mode,
            data_preparation=data,
            input_coverage=input_coverage,
            results=results,
        ),
    )
    fields = [
        "method",
        "input_mode",
        "split",
        "oracle",
        "epoch",
        "pairs",
        "p_c",
        "p_v",
        "p_b",
        "p_r",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "auc",
        "auprc",
        "paper_p_c",
        "delta_paper_p_c",
    ]
    with (root / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    lines = [
        f"# LOSVER：PrimeVul Paired，输入模式{input_mode}",
        "",
        "官方原始 Train/Valid/Test：4354/562/564 对；不重划分、不重采样。Seed 123456，函数Batch 8，每窗口最多512 Token。定位器40 Epoch按Valid修改行正类F1选模；三个分类器16 Epoch按Valid P-C选模。固定概率>0.5。",
        "",
        "| 模型 | Epoch | Test P-C | P-V | P-B | P-R | Accuracy | Precision | Recall | F1 | MCC | AUC | AUPRC | 论文 P-C |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if r["split"] == "test":
            lines.append(
                "| "
                + r["method"]
                + f" | {r['epoch']}"
                + " | "
                + " | ".join(
                    (
                        f"{r[k]:.4f}"
                        if k in ("mcc", "auc", "auprc")
                        else f"{r[k]*100:.2f}%"
                    )
                    for k in (
                        "p_c",
                        "p_v",
                        "p_b",
                        "p_r",
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "mcc",
                        "auc",
                        "auprc",
                        "paper_p_c",
                    )
                )
                + " |"
            )
    test = {r["method"]: r for r in results if r["split"] == "test"}
    lines += [
        "",
        "论文P-C仅为历史参考；当前输入与分类头协议已修正，不作为公平对照。L1应与同一修正版512训练结果比较。",
        "",
        "预测修改行LOSVER相对baseline："
        + "、".join(
            f"{key} {(test['losver'][key] - test['unixcoder'][key])*100:+.2f}个百分点"
            for key in ("p_c", "accuracy", "f1")
        )
        + "。这些是本轮固定协议的实测差值。",
        "",
        "源Valid选模："
        + "；".join(
            f"{r['method']} Epoch {r['epoch']}，P-C {r['p_c']*100:.2f}%"
            for r in results
            if r["split"] == "valid"
        )
        + "。",
    ]
    locator = json.loads((root / "losver_localizer/best-f1.json").read_text())
    lv = locator["validation"]
    lines += [
        "",
        f"定位器选定Epoch {locator['epoch']}：Valid修改行正类F1 {lv['f1']*100:.2f}%，"
        f"Precision {lv['precision']*100:.2f}%、Recall {lv['recall']*100:.2f}%。",
        "",
        "三个检测器共享UniXcoder、核心代码Token聚合及线性分类头。统一原始源码offset物理行映射；仅改变修改行5/1权重，Padding、特殊符号与纯空白Token不参与归一化。"
        + (
            "输入为510源码Token+BOS/EOS，截断末行保留可见片段。"
            if input_mode == "512"
            else "L1输入使用384核心Token及左右最多63上下文，所有源码Token恰属一个核心区域，无函数截断上限；每个物理行只预测一次。"
        ),
        "",
        "Oracle单独读取真实修改行；正常单函数推理只使用源码与定位器预测。未使用Test调参。data_preparation仅保留原准备记录；当前输入覆盖审计见check-paired及训练配置中的input_coverage，不将历史截断统计当作L1覆盖率。",
        "",
        "Seed123456、学习率2e-5、Batch8、旧Transformers AdamW、10% warmup和原Focal参数不变；定位器40 Epoch、三个检测器各16 Epoch。输入/聚合/分类头及定位选模已按本次要求修正，不能称为原始源码的数值一比一复现。",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    return results


def check_paired(data_dir, output_dir, model_path, *, device="auto", input_mode="512"):
    """Audit coverage, one-batch GPU gradients, and the shared trainer on a pilot."""
    import random
    import tempfile
    import time
    from transformers import RobertaTokenizerFast
    from .data import paired_encoding, paired_features, paired_statistics
    from .training import LOSVER_CONFIG, LOSVERAdamW, paired_collate, paired_forward

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "smoke.json").exists():
        raise FileExistsError(f"{root}: check results already exist")
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, trim_offsets=False)
    special_ids = set(tokenizer.all_special_ids)
    coverage_path = root / "coverage.json"
    coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
    after, reference = {}, {}
    train_rows, train_encodings = None, None
    train_prefix = None
    valid_rows = valid_encodings = None
    for split, expected_pairs in (("train", 4354), ("valid", 562), ("test", 564)):
        started = time.monotonic()
        name = f"{split}.lines.jsonl" if split != "test" else "test.oracle.jsonl"
        rows = read_jsonl(Path(data_dir) / "primevul_paired" / name)
        if (
            len(rows) != 2 * expected_pairs
            or [r["target"] for r in rows] != [1, 0] * expected_pairs
        ):
            raise ValueError(f"{split}: official paired members required")
        features, prefix = [], []
        for row in training_bar(rows, desc=f"Input audit {split} {input_mode}"):
            encoded = tokenizer(
                row["func"],
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
                verbose=False,
            )
            f = paired_encoding(
                row["func"], tokenizer, input_mode, tokenization=encoded
            )
            features.append(f)
            prefix.append(
                f
                if input_mode == "512"
                else paired_encoding(
                    row["func"], tokenizer, "512", tokenization=encoded
                )
            )
        counts = paired_statistics(rows, features)
        reference[split] = dict(counts=paired_statistics(rows, prefix))
        errors = invalid = 0
        for row, f in zip(rows, features):
            code = row["func"]
            physical = [
                n for n, part in enumerate(code.splitlines(keepends=True)) for _ in part
            ]
            actual = [set() for _ in f["input_ids"]]
            for n, tokens in zip(f["line_numbers"], f["line_token_indices"]):
                for token in tokens:
                    actual[token].add(n)
            for token, (token_id, (a, b)) in enumerate(
                zip(f["input_ids"], f["offset_mapping"])
            ):
                expected = (
                    {physical[p] for p in range(a, b) if not code[p].isspace()}
                    if token_id not in special_ids
                    else set()
                )
                errors += actual[token] != expected
                invalid += bool(f["code_mask"][token]) and (
                    not expected or not f["attention_mask"][token]
                )
                if bool(expected) != bool(f["code_mask"][token]):
                    errors += 1
            ownership = [0] * f["source_tokens"]
            for left, right, a, b in f["windows"]:
                if (
                    not (0 <= left <= a <= b <= right <= f["source_tokens"])
                    or right - left > 510
                ):
                    raise ValueError("invalid native window bounds")
                for pos in range(a, b):
                    ownership[pos] += 1
            if any(v > 1 for v in ownership) or (
                input_mode == "l1" and any(v != 1 for v in ownership)
            ):
                raise ValueError("core source ownership is not exactly once")
        counts.update(mapping_errors=errors, invalid_aggregation_memberships=invalid)
        if errors or invalid:
            raise ValueError(f"{split}: source mapping validation failed: {counts}")
        if (
            "before" in coverage
            and counts["functions"] != coverage["before"][split]["counts"]["functions"]
        ):
            raise ValueError("before/after cohorts differ")
        after[split] = dict(counts=counts, seconds=time.monotonic() - started)
        print(split, json.dumps(after[split]), flush=True)
        if split == "train":
            train_rows, train_encodings = rows, features
            train_prefix = prefix
        if split == "valid":
            selected_valid = random.Random(LOSVER_CONFIG["seed"]).sample(
                range(expected_pairs), 4
            )
            idx = [j for i in selected_valid for j in (2 * i, 2 * i + 1)]
            valid_rows = [rows[i] for i in idx]
            valid_encodings = [features[i] for i in idx]
    coverage.update(
        after=after,
        reference_512=reference,
        input_mode=input_mode,
        policy=dict(
            tokens=(
                "BOS/EOS per window; single window up to 510 source tokens; L1 uses 384 core + at most 63 left/right context, without a function limit"
                if input_mode == "l1"
                else "BOS + first 510 original-source tokens + EOS; right-pad to 512"
            ),
            lines="zero-based physical IDs; empty/whitespace-only lines have no predictions",
            crossing_tokens="membership in every physical line with non-whitespace overlap; weight 5 if any selected line overlaps",
            partial_line="retain visible tokens and unchanged Diff label; count partial/full coverage separately",
            selection="modified-line positive F1; detectors Valid P-C; strict probability >0.5",
            test_usage="input/annotation coverage only; no Test model inference or selection",
        ),
    )
    write_json(coverage_path, coverage)
    # Choose geometry-visible pairs only for this diagnostic, never alter Train members.
    eligible = [
        i
        for i in range(0, len(train_rows), 2)
        if all(
            set(train_rows[j]["lines_ground_truth"])
            & set(train_prefix[j]["line_numbers"])
            for j in (i, i + 1)
        )
    ]
    selected = random.Random(LOSVER_CONFIG["seed"]).sample(eligible, 4)
    indices = [j for i in selected for j in (i, i + 1)]
    rows = [train_rows[i] for i in indices]
    encodings = [train_encodings[i] for i in indices]
    short = min(
        range(len(train_rows)), key=lambda i: sum(train_encodings[i]["attention_mask"])
    )
    device = core.resolve_device(device)
    result = dict(
        smoke_only=True,
        input_mode=input_mode,
        gpu_split="train",
        device=str(device),
        device_name=(
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        torch_version=torch.__version__,
        model_path=str(model_path),
        optimizer_steps_per_variant=1,
        learning_rate=LOSVER_CONFIG["learning_rate"],
        optimizer="LOSVERAdamW (Transformers 4.19 rule)",
        seed=LOSVER_CONFIG["seed"],
        functions=8,
        sample_keys=[r["sample_key"] for r in rows],
        valid_sample_keys=[r["sample_key"] for r in valid_rows],
        variants={},
    )
    head_reference = None
    with tempfile.TemporaryDirectory(prefix="losver-smoke-") as tmp:
        localizer_path = Path(tmp) / "localizer.pt"
        predicted = valid_predicted = None
        for variant in ("localizer", "baseline", "predicted", "oracle"):
            started = time.monotonic()
            core.seed_everything(LOSVER_CONFIG["seed"])
            model, _ = core.losver_model(
                model_path,
                variant,
                window_micro_batch=1 if input_mode == "l1" else 8,
                gradient_checkpointing=input_mode == "l1",
            )
            model.input_mode = input_mode
            model.to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                memory_start = torch.cuda.memory_allocated(device)
            if variant != "localizer":
                current = model.classifier.weight.detach().cpu().clone()
                if head_reference is None:
                    head_reference = current
                else:
                    torch.testing.assert_close(current, head_reference, atol=0, rtol=0)
            inputs = predicted if variant == "predicted" else rows
            features = [
                paired_features(r, tokenizer, variant, encoding=e)
                for r, e in zip(inputs, encodings)
            ]
            batch = paired_collate(features)
            optimizer = LOSVERAdamW(
                model.parameters(),
                lr=LOSVER_CONFIG["learning_rate"],
                eps=LOSVER_CONFIG["adam_epsilon"],
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_started = time.monotonic()
            model.train()
            loss, _ = paired_forward(model, batch, variant, device)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                LOSVER_CONFIG["gradient_clip"],
                error_if_nonfinite=True,
            )
            encoder_norm = float(
                model.encoder.get_input_embeddings().weight.grad.norm()
            )
            head = model.line_classifier if variant == "localizer" else model.classifier
            head_norm = sum(
                float(p.grad.norm()) for p in head.parameters() if p.grad is not None
            )
            if not torch.isfinite(loss) or encoder_norm <= 0 or head_norm <= 0:
                raise ValueError("nonfinite/zero smoke gradient")
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_seconds = time.monotonic() - train_started
            train_peak = (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            )
            model.eval()
            inference_started = time.monotonic()
            with torch.no_grad():
                probabilities = paired_forward(
                    model, batch, variant, device, with_labels=False
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds = time.monotonic() - inference_started
            cp = dict(
                protocol="losver",
                line_mapping="source_offsets",
                input_mode=input_mode,
                window_micro_batch=model.window_micro_batch,
                variant=variant,
                model_path=str(model_path),
                decision_threshold=0.5,
                smoke_only=True,
                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                localizer_path=str(localizer_path) if variant == "predicted" else None,
            )
            path = (
                localizer_path
                if variant == "localizer"
                else Path(tmp) / f"{variant}.pt"
            )
            torch.save(cp, path)
            restored, _ = load_paired(
                torch.load(path, map_location="cpu", weights_only=False), device
            )
            with torch.no_grad():
                reloaded = paired_forward(
                    restored, batch, variant, device, with_labels=False
                )
                a = (
                    torch.cat(probabilities)
                    if variant == "localizer"
                    else probabilities
                )
                b = torch.cat(reloaded) if variant == "localizer" else reloaded
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            record = dict(
                loss=float(loss.detach()),
                training_seconds=train_seconds,
                inference_seconds=inference_seconds,
                train_peak_allocated_bytes=train_peak,
                gradient_norm=float(norm),
                encoder_gradient_norm=encoder_norm,
                head_gradient_norm=head_norm,
                parameters=sum(p.numel() for p in model.parameters()),
                train_windows=len(batch["input_ids"]),
                train_encoder_tokens=batch["input_ids"].numel(),
                reload_max_probability_difference=float((a - b).abs().max()),
            )
            if variant == "localizer":
                predicted = [
                    dict(
                        {
                            k: value
                            for k, value in r.items()
                            if k != "lines_ground_truth"
                        },
                        line_numbers=e["line_numbers"],
                        lines_predicted=[
                            n for n, p in zip(e["line_numbers"], ps) if p > 0.5
                        ],
                    )
                    for r, e, ps in zip(rows, encodings, probabilities)
                ]
            vr = valid_predicted if variant == "predicted" else valid_rows
            vf = [
                paired_features(r, tokenizer, variant, encoding=e)
                for r, e in zip(vr, valid_encodings)
            ]
            validation, vp = validate_paired(
                model, vr, tokenizer, variant, device, loader=[paired_collate(vf)]
            )
            record["valid_after_one_update"] = validation
            if variant == "localizer":
                valid_predicted = [
                    dict(
                        {k: v for k, v in r.items() if k != "lines_ground_truth"},
                        line_numbers=p["line_numbers"],
                        lines_predicted=p["lines_predicted"],
                    )
                    for r, p in zip(valid_rows, vp)
                ]
            # A real short Train function, with and without its padding; same code boundaries.
            sr = train_rows[short]
            sf = paired_features(
                sr,
                tokenizer,
                "localizer" if variant == "localizer" else "baseline",
                encoding=train_encodings[short],
            )
            sb = paired_collate([sf])
            end = int(sb["attention_mask"].sum())
            stripped = dict(sb)
            for k in (
                "input_ids",
                "attention_mask",
                "code_mask",
                "context_mask",
                "token_weights",
            ):
                stripped[k] = sb[k][:, :end]
            stripped["line_token_mask"] = sb["line_token_mask"][:, :, :end]
            with torch.no_grad():
                p = paired_forward(model, sb, variant, device, with_labels=False)
                q = paired_forward(model, stripped, variant, device, with_labels=False)
                if variant == "localizer":
                    p, q = p[0], q[0]
                torch.testing.assert_close(p, q, atol=2e-6, rtol=1e-5)
                record["padding_max_probability_difference"] = float(
                    (p - q).abs().max()
                )
            if variant in ("baseline", "predicted"):
                end_to_end = core.predict_paired(
                    cp, [dict(raw_source=r["func"]) for r in rows[:2]], device=device
                ).sigmoid()
                expected = probabilities[:2, 0].cpu()
                torch.testing.assert_close(end_to_end, expected, atol=2e-5, rtol=1e-5)
                record["source_only_inference_max_difference"] = float(
                    (end_to_end - expected).abs().max()
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                record["incremental_peak_bytes"] = (
                    record["peak_allocated_bytes"] - memory_start
                )
            record["seconds"] = time.monotonic() - started
            result["variants"][variant] = record
            print(variant, json.dumps(record), flush=True)
            del restored, optimizer, model, cp, batch, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
        # Exercise the actual trainer, including function micro-batches and the
        # smaller final group. Both input modes use these same fixed members.
        from .training import train_losver

        pilot_pairs = random.Random(LOSVER_CONFIG["seed"]).sample(eligible, 9)
        pilot_rows = [train_rows[j] for i in pilot_pairs for j in (i, i + 1)]
        pilot_data = Path(tmp) / "data/primevul_paired"
        for split, members in (("train", pilot_rows), ("valid", valid_rows)):
            write_jsonl(pilot_data / f"{split}.lines.jsonl", members)
            write_jsonl(
                pilot_data / f"{split}.jsonl",
                [
                    {k: v for k, v in r.items() if k != "lines_ground_truth"}
                    for r in members
                ],
            )
        pilot_settings = dict(
            LOSVER_CONFIG, epochs=dict(LOSVER_CONFIG["epochs"], baseline=1)
        )
        pilot_output = Path(tmp) / "pilot"
        pilot = train_losver(
            Path(tmp) / "data",
            pilot_output,
            model_path,
            "unixcoder",
            device=str(device),
            config=pilot_settings,
            input_mode=input_mode,
            window_micro_batch=1 if input_mode == "l1" else 8,
            function_micro_batch=1 if input_mode == "l1" else 8,
        )
        selected = json.loads(
            (pilot_output / "primevul_paired/unixcoder/best-p_c.json").read_text()
        )
        result["training_loop_pilot"] = dict(
            train_functions=len(pilot_rows),
            valid_functions=len(valid_rows),
            train_sample_keys=[r["sample_key"] for r in pilot_rows],
            optimizer_steps=pilot["optimizer_steps"],
            epochs=1,
            validation=selected["validation"],
            function_micro_batch=pilot["function_micro_batch"],
            effective_batch=LOSVER_CONFIG["batch_size"],
        )
        if pilot["optimizer_steps"] != 3:
            raise ValueError(
                "pilot optimizer budget differs from effective function batches"
            )
    write_json(root / "smoke.json", result)
    lines = [
        f"# LOSVER {input_mode} 输入与GPU检查",
        "",
        "官方4354/562/564配对成员不变。全量统计仅检查源码/标注覆盖；GPU仅对固定4个Train配对更新一步，另4个Valid配对检查指标。Test不推理、不选模。这不是正式训练性能。",
        "",
        "| Split | 函数 | 窗口数 512→本次 | 截断率 512→本次 | 修改行可见 512→本次 | 本次完整可见 | 编码相同对 512→本次 | 映射错误 |",
        "|---|---:|---:|---|---|---:|---|---:|",
    ]
    for split, current in after.items():
        n = current["counts"]
        r = reference[split]["counts"]
        lines.append(
            f"| {split} | {n['functions']} | {r['windows']}→{n['windows']} | {r['truncated_functions']/r['functions']:.2%}→{n['truncated_functions']/n['functions']:.2%} | {r['visible_modified_lines']}→{n['visible_modified_lines']} / {n['modified_lines']} | {n['fully_visible_modified_lines']} | {r['identical_pairs']}→{n['identical_pairs']} | {n['mapping_errors']} |"
        )
    lines += [
        "",
        "窗口最多512 Token。短函数保留单窗口；L1长函数每384个核心Token配左右至多63个上下文Token，无函数总长度上限。每个源码Token恰属一个核心区域，Padding/特殊符号/纯空白Token不参与最终聚合。修改行可见仅指该物理行包含可见代码Token，完整可见另计。",
        "",
        "Localizer只在窗口内计算原MHA上下文，按核心Token数量将同一物理行的表示求均值后分类一次。检测器沿用5/1二值权重，跨全部核心代码Token执行一个全局Softmax，复用原生窗口函数聚合。三个检测器共用同一编码器结构和线性头。",
        "",
        f"GPU：{result['device_name']}，Torch {result['torch_version']}；同一Train/Valid成员和初始化、有效函数Batch 8、Seed123456、固定阈值>0.5。下表显存为本进程训练峰值allocated，不包括其他训练进程和CUDA上下文。时间为一步训练/该批推理，完整save/reload验证耗时另存JSON。",
        "",
        "| Smoke | Loss | 梯度范数 | 训练窗口 | Encoder Token | 训练/推理秒 | 训练峰值GiB | 重载误差 | Padding误差 |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for v, r in result["variants"].items():
        memory = (
            r["train_peak_allocated_bytes"] / (1024**3)
            if r["train_peak_allocated_bytes"] is not None
            else 0
        )
        lines.append(
            f"| {v} | {r['loss']:.6f} | {r['gradient_norm']:.6g} | {r['train_windows']} | {r['train_encoder_tokens']} | {r['training_seconds']:.3f}/{r['inference_seconds']:.3f} | {memory:.3f} | {r['reload_max_probability_difference']:.3g} | {r['padding_max_probability_difference']:.3g} |"
        )
    lines += [
        "",
        "| 一步更新后的8函数Valid Smoke | P-C | Accuracy | AUC | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for v, r in result["variants"].items():
        if v == "localizer":
            continue
        m = r["valid_after_one_update"]
        lines.append(
            f"| {v} | {m['p_c']:.4f} | {m['accuracy']:.4f} | {m['auc']:.4f} | {m['f1']:.4f} |"
        )
    lines += [
        "",
        "Smoke仅验证可训练性与输入/推理正确性。一次更新和8个Valid函数不能支持分类收益结论；正式比较须待两种输入预算的完整Valid选模和Test评价完成。历史结果和权重不覆盖。定位器按Valid修改行正类F1，检测器按Valid P-C，均不使用Test选模。",
        "",
    ]
    pilot = result["training_loop_pilot"]
    m = pilot["validation"]
    lines += [
        f"正式训练循环Pilot：18个Train函数、8个Valid函数、1 Epoch、{pilot['optimizer_steps']}个优化步，有效Batch8，末组2个函数；函数Micro-batch为{pilot['function_micro_batch']}。源Valid按P-C选模：P-C {m['p_c']:.4f}、Accuracy {m['accuracy']:.4f}、AUC {m['auc']:.4f}、MCC {m['mcc']:.4f}、F1 {m['f1']:.4f}。该小样本结果只验证完整训练调用链，不能代替正式实验。",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    return dict(
        coverage=str(coverage_path),
        smoke=str(root / "smoke.json"),
        report=str(root / "report.md"),
    )
