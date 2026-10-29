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
        average_precision_score,
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
    auxiliary_predictions = []
    loss = 0.0
    loss_mass = 0.0
    at = 0
    for batch in training_bar(loader, desc=f"Valid {variant}"):
        joint = getattr(model, "line_auxiliary_weight", None) is not None and "line_labels" in batch
        if joint:
            value, prob, details = paired_forward(model, batch, variant, device, return_auxiliary=True)
            for labels, numbers, p in zip(batch["line_labels"], batch["line_numbers"], details["line_probabilities"]):
                valid = numbers >= 0
                ns, ys, ps = numbers[valid].tolist(), labels[valid].tolist(), p.cpu().tolist()
                if len(ps) != len(ns):
                    raise ValueError("auxiliary scores and visible physical lines differ")
                line_y.extend(ys)
                line_p.extend(ps)
                auxiliary_predictions.append(dict(line_numbers=ns, line_scores=ps))
        else:
            value, prob = paired_forward(model, batch, variant, device)
        if variant == "localizer" and getattr(model, "weak_supervision", False):
            mass = float((batch["label_weights"] * (batch["labels"] >= 0)).sum())
            loss += float(value) * mass
            loss_mass += mass
        else:
            loss += float(value)
        if variant == "localizer":
            for row_index, (y, numbers, p) in enumerate(
                zip(batch["labels"], batch["line_numbers"], prob)
            ):
                p = p.cpu().tolist()
                valid = numbers >= 0
                ns = numbers[valid].tolist()
                ys = y[valid].tolist()
                if len(p) != len(ns):
                    raise ValueError("localizer scores and physical line labels differ")
                trusted = [
                    i
                    for i, yv in enumerate(ys)
                    if yv in (0, 1)
                    and (
                        not getattr(model, "weak_supervision", False)
                        or batch["label_weights"][row_index, i] > 0
                    )
                ]
                line_y.extend(ys[i] for i in trusted)
                line_p.extend(p[i] for i in trusted)
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
            auprc=float(average_precision_score(line_y, line_p)) if sum(line_y) else None,
            predicted_positive_fraction=float(pred.mean()) if len(pred) else None,
            lines=len(line_y),
            modified_lines=sum(line_y),
            confusion_matrix=confusion_matrix(line_y, pred, labels=[0, 1]).tolist(),
            loss=(
                loss / max(1e-12, loss_mass)
                if getattr(model, "weak_supervision", False)
                else loss / len(loader)
            ),
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
        if auxiliary_predictions:
            if len(auxiliary_predictions) != len(predictions):
                raise ValueError("auxiliary and function prediction members differ")
            for prediction, auxiliary in zip(predictions, auxiliary_predictions):
                prediction.update(auxiliary)
            result["line_ap"] = float(average_precision_score(line_y, line_p)) if sum(line_y) else None
            result["line_samples"] = len(line_y)
            result["line_positives"] = int(sum(line_y))
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
        **(
            dict(dropout_probability=checkpoint["training_config"]["dropout_probability"])
            if checkpoint.get("training_config", {}).get("dropout_probability", 0.0)
            else {}
        ),
        **(dict(line_auxiliary_weight=checkpoint["line_auxiliary"]["weight"])
           if "line_auxiliary" in checkpoint else {}),
    )
    model.input_mode = checkpoint["input_mode"]
    model.weak_supervision = checkpoint.get("weak_supervision", False)
    if checkpoint["variant"] != "localizer":
        model.pooling = checkpoint.get("pooling", "hard_weight")
        if model.pooling == "robust_fusion":
            model.fusion_beta = float(checkpoint["selected_fusion_beta"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), tokenizer


@torch.no_grad()
def localize_paired(
    data_dir,
    output_dir,
    split,
    *,
    device="auto",
    method="losver_localizer",
    sample_keys=None,
):
    """Single-function prediction: this function never opens Diff annotations."""
    from .training import paired_loader, paired_forward

    device = core.resolve_device(device)
    root = Path(output_dir) / "primevul_paired" / method
    cp = torch.load(root / "best-f1.pt", map_location="cpu", weights_only=False)
    model, tokenizer = load_paired(cp, device)
    rows = read_jsonl(Path(data_dir) / "primevul_paired" / f"{split}.jsonl")
    if sample_keys is not None:
        rows = [r for r in rows if r["sample_key"] in sample_keys]
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
    target = root / f"{split}.predicted.jsonl"
    temporary = target.with_suffix(".jsonl.tmp")
    write_jsonl(temporary, predicted)
    temporary.replace(target)
    target = root / f"{split}.line_scores.jsonl"
    temporary = target.with_suffix(".jsonl.tmp")
    write_jsonl(
        temporary,
        [
            dict(sample_key=r["sample_key"], line_numbers=ns, line_scores=v)
            for r, ns, v in zip(rows, line_numbers, probabilities)
        ],
    )
    temporary.replace(target)
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
        if "test" in splits and cp.get("train_samples", 8708) < 8708:
            raise ValueError("paired pilot checkpoints must not evaluate Test")
        model, tokenizer = load_paired(cp, device)
        for split in splits:
            if variant == "localizer" and split == "test":
                # Normal detector localization is source-only. Oracle line-quality metrics are separate.
                localize_paired(
                    data_dir, output_dir, split, device=device, method=method
                )
                continue
            if variant == "localizer" and cp.get("weak_supervision"):
                rows = read_jsonl(
                    Path(output_dir)
                    / "supervision"
                    / cp["supervision"]
                    / f"{split}.jsonl"
                )
            else:
                rows = paired_rows(data_dir, output_dir, variant, split, method=method)
            if split == "valid":
                selected = set(
                    cp.get("validation_members", [r["sample_key"] for r in rows])
                )
                rows = [r for r in rows if r["sample_key"] in selected]
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


def repair_region_report(output_dir, data_dir):
    """Report aligned source-Valid/final-Test results, with no model selection."""
    import csv
    import numpy as np
    from collections import Counter
    from sklearn.metrics import precision_score, recall_score, f1_score
    from .metrics import paired_metrics
    from .syntax import edit_hunks

    root = Path(output_dir)
    runs = root / "primevul_paired"
    pilot = (runs / "unixcoder/training.metrics.json").exists() and json.loads(
        (runs / "unixcoder/training.metrics.json").read_text()
    )["train_samples"] < 8708
    specs = [
        (
            "global_mean",
            (
                runs / "unixcoder"
                if pilot
                else Path("results/losver_512/primevul_paired/unixcoder")
            ),
        ),
        (
            "original_losver",
            (
                runs / "losver"
                if pilot
                else Path("results/losver_512/primevul_paired/losver")
            ),
        ),
    ]
    specs += [
        (m, runs / m)
        for m in (
            "diff_hard_weight",
            "repair_region_hard_weight",
            "diff_robust_fusion",
            "repair_region_robust_fusion",
            "neighbour_region_robust_fusion",
        )
    ]
    table = []
    groups = []
    cohorts = {}
    for name, run in specs:
        if not (run / "training.metrics.json").exists():
            continue
        training = json.loads((run / "training.metrics.json").read_text())
        for split in ("valid", "test"):
            predpath = run / f"{split}.predictions.jsonl"
            if not predpath.exists():
                continue
            if pilot and split == "test":
                raise ValueError("pilot must not evaluate Test")
            predictions = read_jsonl(predpath)
            source = read_jsonl(Path(data_dir) / "primevul_paired" / f"{split}.jsonl")
            bykey = {p["sample_key"]: p for p in predictions}
            cohort = [r for r in source if r["sample_key"] in bykey]
            if len(cohort) != len(predictions) or len(bykey) != len(predictions):
                raise ValueError("prediction members differ")
            if split in cohorts and cohorts[split] != set(bykey):
                raise ValueError("methods have different evaluation cohorts")
            cohorts[split] = set(bykey)
            metric = paired_metrics(
                cohort, [bykey[r["sample_key"]]["probability"] for r in cohort]
            )
            selected = json.loads((run / "best-p_c.json").read_text())
            table.append(
                dict(
                    method=name,
                    split=split,
                    epoch=selected["epoch"],
                    training_functions=training["train_samples"],
                    selected_beta=selected.get("selected_fusion_beta"),
                    **{
                        k: metric[k]
                        for k in (
                            "samples",
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
                            "pair_ranking_accuracy",
                            "pair_ranking_ties",
                            "patched_fpr",
                        )
                    },
                )
            )
            labels = read_jsonl(
                Path(data_dir)
                / "primevul_paired"
                / (f"{split}.lines.jsonl" if split != "test" else "test.oracle.jsonl")
            )
            ann = {r["sample_key"]: r for r in labels}
            strata = {
                s: []
                for s in ("addition_only", "deletion_only", "replacement_only", "mixed")
            }
            for i in range(0, len(cohort), 2):
                a, b = cohort[i : i + 2]
                dv, dp = (
                    ann[a["sample_key"]]["lines_ground_truth"],
                    ann[b["sample_key"]]["lines_ground_truth"],
                )
                kind = (
                    "addition_only"
                    if not dv
                    else (
                        "deletion_only"
                        if not dp
                        else (
                            "replacement_only"
                            if all(
                                h["tag"] == "replace"
                                for h in edit_hunks(a["func"], b["func"])
                            )
                            else "mixed"
                        )
                    )
                )
                strata[kind].extend((a, b))
            for kind, rr in strata.items():
                if rr:
                    groups.append(
                        dict(
                            method=name,
                            split=split,
                            repair_type=kind,
                            **paired_metrics(
                                rr, [bykey[r["sample_key"]]["probability"] for r in rr]
                            ),
                        )
                    )
    # Use only the normal source-only line-score caches, aligned by member key.
    quality = []
    for supervision in ("diff", "repair_region", "neighbour_region"):
        run = runs / (supervision + "_localizer")
        if not (run / "training.metrics.json").exists():
            continue
        for split in ("train", "valid"):
            scores = read_jsonl(run / f"{split}.line_scores.jsonl")
            annotations = {
                r["sample_key"]: r
                for r in read_jsonl(
                    root / "supervision" / supervision / f"{split}.jsonl"
                )
            }
            ys = []
            ps = []
            diff_y = []
            diff_p = []
            counts = Counter()
            predicted_by_label = Counter()
            no_positive = 0
            predicted = 0
            line_total = 0
            for s in scores:
                a = annotations[s["sample_key"]]
                ns = s["line_numbers"]
                probs = s["line_scores"]
                if len(ns) != len(probs):
                    raise ValueError("line score alignment differs")
                active = any(a["region_labels"][n] == 1 for n in ns)
                no_positive += not active
                gt = set(a["lines_ground_truth"])
                for n, p in zip(ns, probs):
                    y = a["region_labels"][n]
                    counts[int(y)] += 1
                    predicted_by_label[int(y)] += p > 0.5
                    line_total += 1
                    predicted += p > 0.5
                    if active and y in (0, 1) and a["region_weights"][n] > 0:
                        ys.append(y)
                        ps.append(p > 0.5)
                    diff_y.append(n in gt)
                    diff_p.append(p > 0.5)
            quality.append(
                dict(
                    supervision=supervision,
                    split=split,
                    functions=len(scores),
                    visible_lines=line_total,
                    line_label_counts=dict(counts),
                    unknown_fraction=counts[-1] / line_total,
                    positive_fraction=counts[1] / line_total,
                    background_fraction=counts[0] / line_total,
                    no_positive_functions=no_positive,
                    no_positive_fraction=no_positive / len(scores),
                    evaluated_supervised_lines=len(ys),
                    predicted_positive_lines=predicted,
                    predicted_fraction=predicted / line_total,
                    unknown_predicted_positive_fraction=predicted_by_label[-1]
                    / max(1, counts[-1]),
                    supervised_precision=float(
                        precision_score(ys, ps, zero_division=0)
                    ),
                    supervised_recall=float(recall_score(ys, ps, zero_division=0)),
                    supervised_f1=float(f1_score(ys, ps, zero_division=0)),
                    diff_reference_f1=float(f1_score(diff_y, diff_p, zero_division=0)),
                    warning="own weak-label F1 is not vulnerability-location accuracy; original Diff F1 is an auxiliary common reference",
                )
            )

    def save_csv(path, rows):
        if not rows:
            return
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    for row in table:
        baseline = next(
            r
            for r in table
            if r["split"] == row["split"] and r["method"] == "global_mean"
        )
        for metric in ("p_c", "auc", "mcc", "f1"):
            row["delta_" + metric + "_vs_global"] = row[metric] - baseline[metric]
    fusion_usage = []
    for method, run in specs:
        if not method.endswith("robust_fusion") or not (run / "best-p_c.json").exists():
            continue
        selection = json.loads((run / "best-p_c.json").read_text())
        beta = selection["selected_fusion_beta"]
        localizer = runs / (method.removesuffix("_robust_fusion") + "_localizer")
        for split in ("train", "valid", "test"):
            path = localizer / f"{split}.line_scores.jsonl"
            if not path.exists():
                continue
            scores = read_jsonl(path)
            maxima = np.asarray([max(r["line_scores"]) for r in scores])
            strength = np.clip((maxima - 0.5) / 0.5, 0, 1)
            effective = beta * strength
            fusion_usage.append(
                dict(
                    method=method,
                    split=split,
                    functions=len(scores),
                    selected_beta=beta,
                    no_above_gate_fraction=float((strength == 0).mean()),
                    local_branch_disabled_by_valid=beta == 0,
                    nonzero_coefficient_fraction=float((effective > 0).mean()),
                    effective_beta_quantiles=np.quantile(
                        effective, [0, 0.5, 0.9, 1]
                    ).tolist(),
                )
            )
    save_csv(root / "comparison.csv", table)
    required = (
        "diff_hard_weight",
        "repair_region_hard_weight",
        "diff_robust_fusion",
        "repair_region_robust_fusion",
    )
    pending = [m for m in required if not (runs / m / "training.metrics.json").exists()]
    untested = [
        m for m in required if not (runs / m / "test.predictions.jsonl").exists()
    ]
    write_json(
        root / "comparison.json",
        dict(
            pilot=pilot,
            results=table,
            pending_training=pending,
            pending_formal_test=untested if not pilot else [],
            test_used=any(r["split"] == "test" for r in table),
        ),
    )
    write_json(root / "repair_types.json", groups)
    write_json(root / "localizer_quality.json", quality)
    write_json(root / "fusion_usage.json", fusion_usage)
    lines = [
        "# Repair-region supervision and robust global-local fusion (512)",
        "",
        f"Status: {'fixed Train/Valid pilot; Test not used' if pilot else 'full-data runs; formal Test is reported only where completed'}. Pending training: {pending}; pending formal Test: {untested if not pilot else 'not part of pilot'}.",
        "",
        "All functions retain binary labels 0=safe / 1=vulnerable. Line labels are 1=repair-related, 0=weak structural background, -1=Unknown. Unknown has zero focal weight and stays in encoding/BCE. Functions with no visible positive region have no line loss, but remain in detector training and inference.",
        "",
        "Both weak localizers use the same masked focal loss (alpha=.5, gamma=1.5, pos_weight=1). Direct statement weight=1, bounded bound/control relation=.6, background=.2, Unknown=0. Heuristic/nearest definitions and unproven neighbours are Unknown. Original LOSVER uses its historical focal definition and is an additional reference, not the matched weak-loss control.",
        "",
        "Robust fusion: s=clamp((p-.5)/.5,0,1) on valid code tokens; h_region=sum(s*h)/sum(s); h=h_global+beta*max(s)*(h_region-h_global). With no positive mass, h=h_global exactly. Focal sigmoid outputs are not assumed calibrated. Train beta=.25; each epoch selects inference beta from {0,.1,.25} on source Valid P-C, ties prefer smaller beta, then earlier epoch. No Test choice.",
        "",
        "Same UniXcoder pretrained initialization, 512 token offsets/code mask, Linear(768,1), BCE and optimizer budget across detector controls. No GT regions are used in detector training. No OOF, pair loss, extra encoder or adaptive network.",
        "",
        "The localizer stage optimizes only its line objective; the detector stage retains all original functions and optimizes function BCE. The reported syntactic relations are repair-related candidates, not proven vulnerability locations or a complete data-flow analysis. The existing corrected 512 global/LOSVER results are full-data references; the pilot trains its references on the same subset and budget.",
        "",
        "Sampling/RNG checks are in [protocol-checks.json](protocol-checks.json). The matched pilot compares saved RNG states against its freshly trained global baseline. Historical full-data references retain selected weights and recorded configuration, but no last optimizer/RNG checkpoint; an exact historical RNG-state comparison is unavailable. Full new detectors are compared against one another without retraining the historical baseline.",
        "",
        "| Model | Split | Epoch | Beta | P-C | AUC | MCC | F1 | Recall | Patched FPR | Pair ranking |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in table:
        lines.append(
            f"| {r['method']} | {r['split']} | {r['epoch']} | {r['selected_beta']} | {r['p_c']:.4f} | {r['auc']:.4f} | {r['mcc']:.4f} | {r['f1']:.4f} | {r['recall']:.4f} | {r['patched_fpr']:.4f} | {r['pair_ranking_accuracy']:.4f} |"
        )
    lines += ["", "## Classification evidence", ""]
    for split in ("valid", "test"):
        full = next(
            (
                r
                for r in table
                if r["split"] == split and r["method"] == "repair_region_robust_fusion"
            ),
            None,
        )
        if full is None:
            continue
        for reference in (
            "global_mean",
            "original_losver",
            "diff_robust_fusion",
            "repair_region_hard_weight",
        ):
            base = next(
                (r for r in table if r["split"] == split and r["method"] == reference),
                None,
            )
            if base is not None:
                lines.append(
                    f"- {split}: full method versus {reference}: delta P-C={full['p_c']-base['p_c']:+.4f}, AUC={full['auc']-base['auc']:+.4f}, MCC={full['mcc']-base['mcc']:+.4f}, F1={full['f1']-base['f1']:+.4f}."
                )
        if full["delta_p_c_vs_global"] <= 0 and full["delta_auc_vs_global"] <= 0:
            lines += [
                f"- {split}: no improvement over the global baseline on either P-C or AUC."
            ]
        else:
            lines += [
                f"- {split}: the table shows metric-specific differences from one seed; it does not establish stable or causal classification gains."
            ]
        baseline = next(
            r for r in table if r["split"] == split and r["method"] == "global_mean"
        )
        lines.append(
            f"- {split}: strict correct pairs={round(full['p_c']*full['pairs'])}/{full['pairs']} versus global {round(baseline['p_c']*baseline['pairs'])}/{baseline['pairs']}; delta vulnerable Recall={full['recall']-baseline['recall']:+.4f}, patched FPR={full['patched_fpr']-baseline['patched_fpr']:+.4f}. Pair ranking correct={round(full['pair_ranking_accuracy']*full['pairs'])}, ties={full['pair_ranking_ties']}; global ranking correct={round(baseline['pair_ranking_accuracy']*baseline['pairs'])}, ties={baseline['pair_ranking_ties']}. These are observed decision/ranking changes, not causal attribution."
        )
        if full["selected_beta"]:
            lines.append(
                f"- {split}: source Valid selected beta={full['selected_beta']}; the full-data result is not explained by beta=0 disabling the local branch. A positive coefficient does not establish a large or useful representation residual."
            )
    lines += [
        "",
        "Beta=0 disables the local branch at inference; this is the global path of a model trained with beta=.25, not the independently trained global baseline. [fusion_usage.json](fusion_usage.json) records score-gate fallback and effective beta. No module is added in response to these results.",
        "",
        "The neighbour-region pilot control matches added label counts and confidence weights per full-source function. Truncation can change its visible positive count and eligible line-loss denominator, so it is not an exact visible-area control and small differences cannot be attributed solely to program relations.",
    ]
    lines += [
        "",
        "## Supervision and train–Valid mismatch",
        "",
        "| Localizer | Split | Functions | Unknown | Positive | Background | No positive function | Evaluated lines | Predicted lines | Weak-label F1 | Diff reference F1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in quality:
        lines.append(
            f"| {r['supervision']} | {r['split']} | {r['functions']} | {r['unknown_fraction']:.2%} | {r['positive_fraction']:.2%} | {r['background_fraction']:.2%} | {r['no_positive_fraction']:.2%} | {r['evaluated_supervised_lines']} | {r['predicted_fraction']:.2%} | {r['supervised_f1']:.4f} | {r['diff_reference_f1']:.4f} |"
        )
    lines += [
        "",
        "Train caches are in-sample predictions, Valid uses unseen functions. This mismatch remains; the experiment does not establish a causal effect from any association between localization and detection. Own weak-label metrics use different positive regions across supervision types and must not be compared as localization ground truth.",
        "",
        "The negative labels here are weak structural background, not a representative set of unrelated executable statements. High supervised F1 can therefore reflect easy code-versus-structure separation; it is insufficient evidence of precise repair localization. A high score on an Unknown line cannot by itself be called a false positive. The broad predicted-region coverage below is evidence of weak selectivity, with semantic accuracy still unmeasured on most executable lines.",
        "",
        "This supports a supervision-selectivity explanation, but the experiment does not isolate its causal effect. The small Train/Valid gap on these weak labels also does not establish that in-sample localization is the primary cause of failure. No OOF or additional loss/background variant was run.",
        "",
        "Repair-type metrics and denominators: [repair_types.json](repair_types.json). Aligned detector predictions, selected beta, epoch histories, optimizer steps and input/Unknown coverage are in each method directory. Historical checkpoints/results are referenced without modification.",
        "",
    ]
    for r in quality:
        if r["split"] == "valid":
            lines.append(
                f"- {r['supervision']}: {r['unknown_predicted_positive_fraction']:.2%} of Unknown-supervision visible lines receive score >.5. Unknown is a training-label mask, not an uncertain model output; this fraction does not measure an error rate without independent labels."
            )
    supervision_statistics = json.loads(
        (root / "supervision/statistics.json").read_text()
    )
    lines += ["", "## Supervision extraction limits", ""]
    for split in ("train", "valid"):
        stats = supervision_statistics["splits"]["repair_region:" + split]
        reasons = Counter()
        for reason, count in stats["reasons"].items():
            reasons[reason.split(":")[0]] += count
        levels = stats["levels"]
        lines.append(
            f"- Full-source {split}, {stats['functions']} functions: direct positive lines={levels.get('direct', 0)}, bound-operation positives={levels.get('bound_relation', 0)}, enclosing-control positives={levels.get('control_relation', 0)}, no positive function={stats['no_positive_functions']}; own parse-failure/direct-only functions={reasons['parse_failure_direct_only']}, correspondence-Unknown functions={reasons['pair_correspondence_unknown']}. These counts precede 512 truncation and are not pilot-subset counts. All functions remain in function BCE."
        )
    lines += [
        "",
        "Most added positives are enclosing-control lines; the current conservative mapping does not establish full def-use or resource-state semantics. Unproved lexical definition/use paths stay Unknown. Strict parse failures and the 512 visibility boundary limit useful supervision.",
        "",
        "The pre-existing Train development review in ../train_region_review.csv (or train_region_review.csv in the full root) contains 86 additional-line judgments: 13/55 reviewed related lines become positive, 40/55 stay Unknown and 2/55 weak background; 0/21 reviewed unrelated and 0/10 uncertain lines become positive. This reused development review is not independent precision/recall validation or vulnerability-location ground truth.",
    ]
    coverage_path = runs / "diff_localizer/config.json"
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text())["input_coverage"]
        lines += [
            "",
            "| 512 input | Functions | Truncated functions | Visible original Diff lines / all | Identical encoded pairs |",
            "| --- | --- | --- | --- | --- |",
        ]
        for split, c in coverage.items():
            lines.append(
                f"| {split} | {c['functions']} | {c['truncated_functions']} | {c['visible_modified_lines']}/{c['modified_lines']} | {c['identical_pairs']} |"
            )
        lines += [
            "",
            "Input coverage is common to all compared methods; identical encoded pairs and invisible repair changes remain in function BCE and evaluation. Region labels cannot recover code outside the fixed 512 input. This is a shared limitation and cannot alone explain differences between the methods. No member or input rule was changed to improve these counts.",
        ]
    lines += [
        "",
        "## Addition-only repair pairs",
        "",
        "| Model | Split | Pairs | P-C | AUC | Recall | Patched FPR |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in groups:
        if r["repair_type"] == "addition_only":
            lines.append(
                f"| {r['method']} | {r['split']} | {r['pairs']} | {r['p_c']:.4f} | {r['auc']:.4f} | {r['recall']:.4f} | {r['patched_fpr']:.4f} |"
            )
    lines += [
        "",
        "Pure addition is defined from the original nonempty Diff deletion labels, before any new supervision. It includes checks, initialization, state/cleanup and comments; these are not all new checks. The group is descriptive and does not select hyperparameters.",
        "",
    ]
    lines += [
        "## Logged computational cost",
        "",
        "| Stage | Completed epochs | Optimizer steps | Train seconds | Logged epoch seconds |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, run in specs + [
        (s + "_localizer", runs / (s + "_localizer"))
        for s in ("diff", "repair_region", "neighbour_region")
    ]:
        history_path = run / "best.training.jsonl"
        if not (run / "training.metrics.json").exists() or not history_path.exists():
            continue
        epochs = [e for e in read_jsonl(history_path) if e["event"] == "epoch"]
        lines.append(
            f"| {name} | {len(epochs)} | {sum(e['optimizer_steps'] for e in epochs)} | {sum(e['training_seconds'] for e in epochs):.1f} | {sum(e['epoch_seconds'] for e in epochs):.1f} |"
        )
    lines += [
        "",
        "Times are logged wall times, including loader/CPU/IO effects and concurrent GPU jobs, not isolated kernel benchmarks. Logged epoch time includes validation before end-of-epoch checkpoint serialization; startup, localization export and final evaluation are excluded. Robust methods use three Valid beta passes per epoch versus one in other detectors; training optimizer steps remain equal. Localizer cost is a separate stage shared by detectors using its scores.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    return dict(
        pilot=pilot,
        results=table,
        pending_training=pending,
        pending_formal_test=untested if not pilot else [],
    )


def check_repair_regions(
    data_dir, output_dir, model_path, *, device="auto", input_mode="512"
):
    """Real Train-only GPU forward/backward and single-function reload checks."""
    import tempfile, time
    from .data import paired_features, repair_region_supervision
    from .training import paired_collate, paired_forward, LOSVERAdamW, LOSVER_CONFIG

    if input_mode != "512":
        raise ValueError("repair region checks require 512")
    device = core.resolve_device(device)
    if device.type != "cuda":
        raise ValueError(
            "the requested repair-region smoke check requires a real CUDA device"
        )
    root = Path(output_dir)
    source = read_jsonl(Path(data_dir) / "primevul_paired/train.jsonl")
    selected = [
        r
        for r in source
        if r["pair_key"] in {"primevul_paired:train:641", "primevul_paired:train:111"}
    ]
    if len(selected) != 4:
        raise ValueError("expected four fixed Train smoke members")
    # Use an existing, actual localization checkpoint. Neither Diff nor its labels
    # supplies detector scores in this check or in normal detector training.
    cp = torch.load(
        "results/losver_512/primevul_paired/losver_localizer/best-f1.pt",
        map_location="cpu",
        weights_only=False,
    )
    locator, tokenizer = load_paired(cp, device)
    scores = []
    for row in selected:
        f = paired_features(row, tokenizer, "localizer")
        with torch.no_grad():
            p = paired_forward(
                locator, paired_collate([f]), "localizer", device, with_labels=False
            )[0]
        scores.append(
            dict(
                row,
                line_numbers=f["line_numbers"],
                line_scores=p.cpu().tolist(),
                lines_predicted=[n for n, v in zip(f["line_numbers"], p) if v > 0.5],
            )
        )
    del locator, cp
    torch.cuda.empty_cache()
    results = []
    for name, variant, pooling in (
        ("diff_localizer", "localizer", "hard_weight"),
        ("repair_region_localizer", "localizer", "hard_weight"),
        ("global_mean", "baseline", "hard_weight"),
        ("hard_weight", "predicted", "hard_weight"),
        ("robust_fusion", "predicted", "robust_fusion"),
    ):
        core.seed_everything(LOSVER_CONFIG["seed"])
        m, _ = core.losver_model(model_path, variant)
        m.input_mode = "512"
        m.weak_supervision = variant == "localizer"
        if variant != "localizer":
            m.pooling = pooling
        m.to(device)
        m.train()
        torch.cuda.reset_peak_memory_stats(device)
        if variant == "localizer":
            mode = name.removesuffix("_localizer")
            rows = []
            for i in range(0, len(selected), 2):
                a, b = selected[i : i + 2]
                labels = repair_region_supervision(
                    a["func"], b["func"], relation=mode == "repair_region"
                )
                rows.extend(dict(r, **lab) for r, lab in zip((a, b), labels))
        else:
            rows = scores
        features = [paired_features(r, tokenizer, variant) for r in rows]
        reference = [paired_features(r, tokenizer, "baseline") for r in rows]
        for f, ref in zip(features, reference):
            for key in ("input_ids", "code_mask", "offset_mapping", "line_numbers"):
                if f[key] != ref[key]:
                    raise ValueError("experimental inputs differ")
        batch = paired_collate(features)
        optimizer = LOSVERAdamW(m.parameters(), lr=2e-5)
        torch.cuda.synchronize()
        started = time.monotonic()
        loss, _ = paired_forward(m, batch, variant, device)
        loss.backward()
        encoder_grad = float(
            sum(
                p.grad.detach().float().square().sum()
                for p in m.encoder.parameters()
                if p.grad is not None
            ).sqrt()
        )
        if not torch.isfinite(loss) or not encoder_grad > 0:
            raise ValueError("invalid encoder backward")
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        m.eval()
        with torch.no_grad():
            p = paired_forward(m, batch, variant, device, with_labels=False)
        checkpoint = dict(
            protocol="losver",
            line_mapping="source_offsets",
            input_mode="512",
            model_path=str(model_path),
            variant=variant,
            pooling=pooling,
            weak_supervision=m.weak_supervision,
            selected_fusion_beta=0.25,
            state_dict={k: v.detach().cpu() for k, v in m.state_dict().items()},
        )
        del optimizer
        torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory(prefix="repair-smoke-", dir=root) as tmp:
            path = Path(tmp) / "check.pt"
            torch.save(checkpoint, path)
            other, _ = load_paired(
                torch.load(path, map_location="cpu", weights_only=False), device
            )
            with torch.no_grad():
                q = paired_forward(other, batch, variant, device, with_labels=False)
            x = torch.cat(p) if variant == "localizer" else p
            y = torch.cat(q) if variant == "localizer" else q
            error = float((x - y).abs().max())
            if error > 1e-6:
                raise ValueError("save/reload predictions differ")
            del other
        torch.cuda.synchronize()
        results.append(
            dict(
                method=name,
                loss=float(loss),
                encoder_gradient=encoder_grad,
                reload_max_error=error,
                seconds=time.monotonic() - started,
                peak_memory_bytes=torch.cuda.max_memory_allocated(device),
                functions=len(rows),
                encoder_windows=len(batch["input_ids"]),
                encoder_tokens=batch["input_ids"].numel(),
                supervised_line_mass=(
                    float(batch["label_weights"].sum())
                    if variant == "localizer"
                    else None
                ),
            )
        )
        del m, checkpoint, p, q, loss
        torch.cuda.empty_cache()
    result = dict(
        device=str(device),
        members=[r["sample_key"] for r in selected],
        test_used=False,
        checks=results,
    )
    write_json(root / "gpu-smoke.json", result)
    return result


def diff_neighbour_report(output_dir, data_dir, reference_dir):
    """Compare matched pilots using saved scores, original Diff and proxy labels."""
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    from .metrics import paired_metrics

    if reference_dir is None:
        raise ValueError("diff_neighbour report requires --reference-dir")
    root = Path(output_dir)
    supervision = root / "supervision/diff_neighbour"
    reference = Path(reference_dir) / "primevul_paired"
    runs = root / "primevul_paired"
    paths = dict(losver=(reference / "losver_localizer", reference / "losver"),
                 diff_neighbour=(runs / "diff_neighbour_localizer", runs / "diff_neighbour"))
    configs = {name: [json.loads((p / "training.metrics.json").read_text()) for p in ps]
               for name, ps in paths.items()}
    checks = []
    for stage in (0, 1):
        a, b = configs["losver"][stage], configs["diff_neighbour"][stage]
        fields = ("training_members", "validation_members", "input_mode", "window_micro_batch",
                  "function_micro_batch", "gradient_accumulation", "parameters", "training_config",
                  "optimizer_steps", "epochs", "pooling", "weak_supervision", "decision_threshold",
                  "selection", "model_path", "classifier_head", "precision")
        mismatches = [k for k in fields if a[k] != b[k]]
        if mismatches or a["input_mode"] != "512" or a["weak_supervision"]:
            raise ValueError(f"unmatched diff_neighbour pilot: {mismatches}")
        if a["train_samples"] >= 8708:
            raise ValueError("diff_neighbour report is for the requested Train/Valid pilot")
        checks.append(dict(stage=("localizer", "detector")[stage], compared_fields=list(fields),
                           optimizer_steps=a["optimizer_steps"], train_functions=a["train_samples"],
                           valid_functions=a["valid_samples"], matched=True))
    members = configs["losver"][1]["validation_members"]
    if any(c["validation_members"] != members for cs in configs.values() for c in cs):
        raise ValueError("localizer/detector evaluation cohorts differ")
    annotations = {s: read_jsonl(Path(data_dir) / "primevul_paired" / f"{s}.lines.jsonl")
                   for s in ("train", "valid")}
    proxies = {s: {r["sample_key"]: r for r in read_jsonl(supervision / f"{s}.jsonl")}
               for s in ("train", "valid")}
    valid = [r for r in annotations["valid"] if r["sample_key"] in set(members)]
    if [r["sample_key"] for r in valid] != members:
        raise ValueError("Valid members are not in the original source order")
    addition = {valid[i]["pair_key"] for i in range(0, len(valid), 2)
                if not valid[i]["lines_ground_truth"] and valid[i+1]["lines_ground_truth"]}
    table, localizers, transitions = [], [], []
    predictions_by_method = {}
    for name, (locator, detector) in paths.items():
        predictions = read_jsonl(detector / "valid.predictions.jsonl")
        bykey = {r["sample_key"]: r for r in predictions}
        if len(bykey) != len(predictions) or list(bykey) != members:
            raise ValueError("saved detector predictions do not match the fixed Valid cohort")
        predictions_by_method[name] = bykey
        for group in ("all", "pure_addition"):
            rows = valid if group == "all" else [r for r in valid if r["pair_key"] in addition]
            table.append(dict(method=name, group=group,
                              epoch=configs[name][1]["selected_epochs"]["p_c"],
                              **paired_metrics(rows, [bykey[r["sample_key"]]["probability"] for r in rows])))
        for split in ("train", "valid"):
            keys = configs[name][0]["training_members" if split == "train" else "validation_members"]
            original = {r["sample_key"]: r for r in annotations[split]}
            scores = read_jsonl(locator / f"{split}.line_scores.jsonl")
            if len(scores) != len(keys) or [r["sample_key"] for r in scores] != keys:
                raise ValueError("saved localization scores do not match the source cohort")
            for label_type in ("original_diff", "proxy_diff_neighbour"):
                y, p = [], []
                for r in scores:
                    row = (original if label_type == "original_diff" else proxies[split])[r["sample_key"]]
                    gt = set(row["lines_ground_truth"])
                    if len(r["line_numbers"]) != len(r["line_scores"]):
                        raise ValueError("localization scores and physical lines differ")
                    y.extend(int(n in gt) for n in r["line_numbers"])
                    p.extend(v > 0.5 for v in r["line_scores"])
                localizers.append(dict(method=name, split=split, labels=label_type,
                    functions=len(scores), lines=len(y), positive_lines=sum(y), predicted_lines=sum(p),
                    precision=float(precision_score(y, p, zero_division=0)),
                    recall=float(recall_score(y, p, zero_division=0)),
                    f1=float(f1_score(y, p, zero_division=0)),
                    confusion_matrix=confusion_matrix(y, p, labels=[0, 1]).tolist()))
    for i in range(0, len(valid), 2):
        v, p = valid[i:i+2]
        outcomes = {}
        for method, pred in predictions_by_method.items():
            outcomes[method] = (pred[v["sample_key"]]["probability"] > .5
                                and pred[p["sample_key"]]["probability"] <= .5)
        transitions.append(dict(pair_key=v["pair_key"], pure_addition=v["pair_key"] in addition,
                                **outcomes))
    statistics = json.loads((supervision / "statistics.json").read_text())
    pilot_coverage = {}
    for split in ("train", "valid"):
        keys = set(configs["losver"][0]["training_members" if split == "train" else "validation_members"])
        audit = [r for r in read_jsonl(supervision / f"{split}.audit.jsonl") if r["sample_key"] in keys]
        eligible = [r for r in audit if r["eligible"] and r["target"] == 1]
        pilot_coverage[split] = dict(functions=len(keys), pure_addition_pairs=len(eligible),
            supplemented=sum(bool(r["added_lines"]) for r in eligible),
            added_positive_lines=sum(len(r["added_lines"]) for r in eligible),
            single=sum(r["selection"] == "single" for r in eligible),
            both=sum(r["selection"] == "both" for r in eligible),
            none=sum(r["selection"] == "none" for r in eligible))
    proxy_recovery = []
    for name in paths:
        for split in ("train", "valid"):
            original, proxy = [r for r in localizers if r["method"] == name and r["split"] == split]
            count = proxy["positive_lines"] - original["positive_lines"]
            hits = proxy["confusion_matrix"][1][1] - original["confusion_matrix"][1][1]
            proxy_recovery.append(dict(method=name, split=split, added_lines=count,
                                       hits=hits, recall=hits/count if count else None))
    write_json(root / "comparison.json", dict(detectors=table, localizers=localizers,
               proxy_recovery=proxy_recovery, pilot_coverage=pilot_coverage,
               protocol_checks=checks, reference_dir=str(reference_dir)))
    for filename, rows in (("comparison.csv", table), ("localizer_metrics.csv", localizers),
                           ("pair_transitions.csv", transitions)):
        with (root / filename).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    loc_config, det_config = configs["diff_neighbour"]
    lines = ["# 512 diff_neighbour pilot", "",
        "Only line supervision changes. Original LOSVER localizer/Focal loss (alpha=.8, gamma=1.5, pos_weight=5), 5/1 code-token Softmax, UniXcoder, linear detector, source input and >.5 threshold are unchanged. All members remain in training and evaluation. No Unknown/weak-region loss or robust fusion is used.", "",
        f"Reference: `{reference_dir}`. The existing completed original LOSVER pilot is reused without retraining or overwriting it. Both methods use {loc_config['train_samples']} Train/{loc_config['valid_samples']} Valid functions, localizer/detector epochs={loc_config['epochs']}/{det_config['epochs']}, optimizer steps={loc_config['optimizer_steps']}/{det_config['optimizer_steps']}, training seed {loc_config['training_config']['seed']} and member-sampling seed 42. The localizer selects by ORIGINAL Valid Diff positive F1; detector selects by Valid P-C. Test is not opened or evaluated.", "",
        "A proxy is added only to a side whose FULL-source original Diff positives are empty. One code line is selected before/after each edit in the same Differ alignment. Each direction crosses at most one blank/comment line; brace-bearing, delimiter-only and preprocessor lines stop selection. The entire opposite nonblank edit and selected neighbours must be completely visible inside 510 source tokens. Identical encoded pairs are unchanged. These nearby lines are proxies, not verified vulnerability locations.", "",
        "## Supervision coverage", "",
        "| Scope | Split | Pure-add pairs | Supplemented | Coverage | Added positives | Single / both / none |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for split, c in statistics["splits"].items():
        coverage = c.get('pure_addition_covered',0) / max(1, c.get('pure_addition_pairs',0))
        lines.append(f"| Full official | {split} | {c.get('pure_addition_pairs',0)} | {c.get('pure_addition_covered',0)} | {coverage:.2%} | {c.get('pure_addition_added_positives',0)} | {c.get('pure_addition_single',0)} / {c.get('pure_addition_both',0)} / {c.get('pure_addition_none',0)} |")
    for split, c in pilot_coverage.items():
        coverage = c['supplemented'] / max(1, c['pure_addition_pairs'])
        lines.append(f"| Fixed pilot | {split} | {c['pure_addition_pairs']} | {c['supplemented']} | {coverage:.2%} | {c['added_positive_lines']} | {c['single']} / {c['both']} / {c['none']} |")
    lines += ["", "Single/both denotes whether a supplemented function has at least one edit selecting both directions; per-edit counts and explicit rejection reasons are in `supervision/diff_neighbour/statistics.json` and the source-aligned audit tables. Multiple edits are deduplicated.", "",
        "| Full official 512 input | Visible lines | Original positives | Added positives (both repair types) | New positives |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for split, c in statistics["splits"].items():
        lines.append(f"| {split} | {c['visible_lines']} | {c['original_visible_positives']} | {c['added_positives']} | {c['augmented_visible_positives']} |")
    lines += ["",
        "## Valid function classification", "",
        "| Model | Subset | Pairs | Epoch | P-C | AUC | MCC | F1 | Recall | Patched FPR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in table:
        lines.append(f"| {r['method']} | {r['group']} | {r['pairs']} | {r['epoch']} | {r['p_c']:.4f} | {r['auc']:.4f} | {r['mcc']:.4f} | {r['f1']:.4f} | {r['recall']:.4f} | {r['patched_fpr']:.4f} |")
    for group in ("all", "pure_addition"):
        before = next(r for r in table if r["method"] == "losver" and r["group"] == group)
        after = next(r for r in table if r["method"] == "diff_neighbour" and r["group"] == group)
        chosen = [r for r in transitions if group == "all" or r["pure_addition"]]
        gains = sum(not r["losver"] and r["diff_neighbour"] for r in chosen)
        losses = sum(r["losver"] and not r["diff_neighbour"] for r in chosen)
        lines += ["", f"{group}, new minus original: P-C {after['p_c']-before['p_c']:+.4f}, AUC {after['auc']-before['auc']:+.4f}, MCC {after['mcc']-before['mcc']:+.4f}, F1 {after['f1']-before['f1']:+.4f}; newly both-correct pairs={gains}, regressed pairs={losses}, denominator={len(chosen)}."]
    lines += ["", "Pure addition is defined by the original vulnerable-side Diff labels being empty and patched-side positives being nonempty, before augmentation or model results.", "",
        "## Line localization: original versus proxy labels", "",
        "| Model | Split | Labels | Lines | Positive | Predicted | Precision | Recall | F1 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in localizers:
        lines.append(f"| {r['method']} | {r['split']} | {r['labels']} | {r['lines']} | {r['positive_lines']} | {r['predicted_lines']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")
    lines += ["", "Original-Diff and proxy metrics use the SAME visible lines and predictions, with different label sets. Better proxy F1 alone cannot establish better localization or function classification. Full per-function code and labels stay in the local supervision cache; source-free audit records retain zero-based physical line IDs.", "",
        "| Model | Split | Added proxy lines | Predicted hits | Proxy-only recall |",
        "| --- | --- | ---: | ---: | ---: |"]
    for r in proxy_recovery:
        rate = f"{r['recall']:.4f}" if r["recall"] is not None else "NA"
        lines.append(f"| {r['method']} | {r['split']} | {r['added_lines']} | {r['hits']} | {rate} |")
    lines += ["", "This measures recovery of the added adjacency proxies, not semantic vulnerability localization. Train is in-sample for the localizer; Valid is unseen. The original/proxy supervision gap and the classifier changes are observational, not a causal attribution.", "",
        "## Runtime", "", "| Stage | Train seconds | Logged epoch seconds | Steps |",
        "| --- | ---: | ---: | ---: |"]
    for name, ps in paths.items():
        for stage, path in zip(("localizer", "detector"), ps):
            events = [r for r in read_jsonl(path / "best.training.jsonl") if r["event"] == "epoch"]
            lines.append(f"| {name} {stage} | {sum(e['training_seconds'] for e in events):.1f} | {sum(e['epoch_seconds'] for e in events):.1f} | {sum(e['optimizer_steps'] for e in events)} |")
    lines += ["", "Times include loader/IO and validation effects; they are not isolated GPU benchmarks. These are one-seed pilot results; Test and full-data training are not performed.", "",
        "## Implementation and verification", "",
        "- `data.py`: preserve original Differ labels; supplement only fully visible edit-adjacent proxies and record each rejection. `training.py`/CLI: two new method choices in the existing stages; original Focal and selection retained. `evaluation.py`: aligned comparison, original/proxy line metrics and the same fixed source Valid cohort. `model.py` is unchanged.",
        "- [Data checks](data-checks.json) verify original source/targets/members/order, all original positives and all non-target function labels. [Protocol checks](protocol-checks.json) compare actual saved optimizer/scheduler steps, parameter shapes, input coverage, members and final Python/NumPy/CPU/CUDA RNG states against the reused pilot.",
        "- [Train review](train_review.md) / [records](train_review.jsonl): Seed42 stratified inspection of 40 Train pair sides, covering supplemented, boundary/visibility rejection, identical-input and pure-deletion cases. These syntactic checks do not certify semantic relevance. `train:5179` contains an added test invocation; `train:1831` selects a signature line; `train:2698` has a signature-only side. Such cases remain, as required by the unchanged data protocol.",
        "- [GPU smoke](smoke.json): real Train forward/backward, encoder/attention/head gradients, exact save/reload and source-only single-function inference. [Full tests](tests.log), [targeted tests](neighbour-tests.log) and [resume tests](resume-tests.log) retain the actual execution logs. The smoke log preserves an initial diagnostic API-call error and its corrected successful check.",
        "",
        "## Interpretation", "",
        "AUC and P-C measure different behavior. Ranking can improve while two sides remain on the same side of the fixed .5 boundary. Read the pure-add Recall and patched FPR together: fewer false alarms can accompany fewer detected vulnerabilities. Proxy coverage or proxy F1 cannot substitute for both-correct classification.", "",
        "The Train review confirms that bounded neighbours need not identify the vulnerability operation. Original-Diff localization precision remains low, and hard weighting is intentionally unchanged. These observations are possible limitations, not a demonstrated cause of a classifier difference. The reported metrics have mixed directions and do not establish a stable overall classification gain; no further module, full-data training or Test tuning is launched.", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return dict(results=table, pilot_coverage=pilot_coverage)


def localizer_regularization_report(output_dir, data_dir, reference_dir):
    """Original Diff line ranking and threshold metrics on matched Train/Valid."""
    import numpy as np
    from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

    if reference_dir is None:
        raise ValueError("localizer regularization report requires --reference-dir")
    root = Path(output_dir)
    paths = dict(
        original_losver=Path(reference_dir) / "primevul_paired/losver_localizer",
        dropout_weight_decay=root / "dropout_weight_decay/primevul_paired/losver_localizer",
        line_rdrop=root / "line_rdrop/primevul_paired/losver_localizer",
    )
    configs = {name: json.loads((p / "training.metrics.json").read_text()) for name, p in paths.items()}
    reference = configs["original_losver"]
    fields = ("training_members", "validation_members", "input_mode", "parameters", "epochs",
              "optimizer_steps", "input_coverage", "selection", "selection_ties", "model_path",
              "function_micro_batch", "window_micro_batch", "gradient_accumulation")
    for name, config in configs.items():
        if any(config[k] != reference[k] for k in fields):
            raise ValueError(f"unmatched localizer data/budget/selection: {name}")
        if config["weak_supervision"] or config["supervision"] != "original_diff" or config["input_mode"] != "512":
            raise ValueError("regularization comparison requires original binary Diff and 512 input")
        settings = {k:v for k,v in config["training_config"].items()
                    if k not in ("weight_decay", "dropout_probability")}
        expected = {k:v for k,v in reference["training_config"].items()
                    if k not in ("weight_decay", "dropout_probability")}
        if settings != expected:
            raise ValueError("learning rate, schedule, seed or other settings differ")
    for name, coefficient in (("dropout_weight_decay", 0.0), ("line_rdrop", 0.1)):
        reg = configs[name]["localizer_regularization"]
        if (reg["dropout"], reg["weight_decay"], reg["rdrop_lambda"], reg["training_forwards"]) != (.1, .01, coefficient, 2):
            raise ValueError("expected matched dropout/decay/two-forward R-Drop controls")
    annotations = {s: {r["sample_key"]:r for r in read_jsonl(Path(data_dir) / "primevul_paired" / f"{s}.lines.jsonl")}
                   for s in ("train", "valid")}
    table, coordinates = [], {}
    for name, run in paths.items():
        epoch = configs[name]["selected_epochs"]["f1"]
        for split in ("train", "valid"):
            scores = read_jsonl(run / f"{split}.line_scores.jsonl")
            keys = reference["training_members" if split == "train" else "validation_members"]
            if [r["sample_key"] for r in scores] != keys:
                raise ValueError("line score members/order differ")
            ns = [(r["sample_key"], r["line_numbers"]) for r in scores]
            if split in coordinates and ns != coordinates[split]:
                raise ValueError("methods have different visible physical lines")
            coordinates[split] = ns
            for side in ("all", "vulnerable", "patched"):
                selected = [r for r in scores if side == "all" or annotations[split][r["sample_key"]]["target"] == int(side == "vulnerable")]
                y, p = [], []
                for row in selected:
                    gt = set(annotations[split][row["sample_key"]]["lines_ground_truth"])
                    if len(row["line_numbers"]) != len(row["line_scores"]):
                        raise ValueError("physical lines and probabilities differ")
                    y.extend(int(n in gt) for n in row["line_numbers"])
                    p.extend(row["line_scores"])
                predicted = np.asarray(p) > .5
                table.append(dict(method=name, split=split, side=side, epoch=epoch,
                    functions=len(selected), lines=len(y), positive_lines=sum(y),
                    predicted_positive_lines=int(predicted.sum()),
                    predicted_positive_fraction=float(predicted.mean()),
                    auprc=float(average_precision_score(y, p)) if sum(y) else None,
                    f1=float(f1_score(y, predicted, zero_division=0)),
                    precision=float(precision_score(y, predicted, zero_division=0)),
                    recall=float(recall_score(y, predicted, zero_division=0))))
        valid = next(r for r in table if r["method"] == name and r["split"] == "valid" and r["side"] == "all")
        selected = json.loads((run / "best-f1.json").read_text())
        if selected["epoch"] != epoch or abs(selected["validation"]["f1"] - valid["f1"]) > 1e-12:
            raise ValueError("selected checkpoint and exported line scores differ")
    gaps = []
    for name in paths:
        for side in ("all", "vulnerable", "patched"):
            train, valid = [r for r in table if r["method"] == name and r["side"] == side]
            gaps.append(dict(method=name, side=side, auprc_train_minus_valid=train["auprc"]-valid["auprc"],
                             f1_train_minus_valid=train["f1"]-valid["f1"]))
    write_json(root / "comparison.json", dict(results=table, generalization_gaps=gaps,
               reference_dir=str(reference_dir), test_used=False, matched_fields=list(fields)))
    for filename, records in (("comparison.csv", table), ("generalization_gaps.csv", gaps)):
        with (root / filename).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    lines = ["# Original LOSVER localizer: regularization and line R-Drop", "",
        f"512 token Pilot, {reference['train_samples']} Train/{reference['valid_samples']} Valid functions, {reference['epochs']} epochs and {reference['optimizer_steps']} optimizer steps per method. Member seed42; training seed{reference['training_config']['seed']}. A reuses `{reference_dir}`. B/C start from the same pretrained UniXcoder with identical initialization, members, Focal, LR/schedule and two-forward budget; C alone adds lambda=.1.", "",
        "Original binary Diff labels and visible physical lines are identical. Focal alpha=.8, gamma=1.5, pos_weight=5. Best checkpoint remains original Valid positive F1 rounded to4 decimals, earlier tie; threshold remains >.5. AUPRC is reported at that checkpoint, never used to select a different epoch. Test and detectors are not evaluated or trained.", "",
        "New attention/head dropout: A=0, B=C=.1. Encoder hidden/attention dropout stays .1. Weight decay: A=0, B=C=.01, decoupled AFTER the unchanged adaptive AdamW update; all bias (including in_proj_bias) and LayerNorm parameters are excluded. The two Focal losses are averaged in B/C; independent Bernoulli symmetric KL is averaged over real lines only, with gradients on both logits. Validation/inference uses one dropout-disabled forward.", "",
        "## Selected checkpoint metrics", "",
        "Metrics pool valid visible lines; vulnerable/patched refer to the original function target, not predicted classes. Positive-line fraction uses >.5; AUPRC uses continuous probabilities. All count denominators are saved in comparison.csv.", "",
        "| Method | Split | Side | Epoch | Lines | AUPRC | F1 | Precision | Recall | Predicted positive fraction |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in table:
        lines.append(f"| {r['method']} | {r['split']} | {r['side']} | {r['epoch']} | {r['lines']} | {r['auprc']:.4f} | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['predicted_positive_fraction']:.4f} |")
    lines += ["", "## Train minus Valid gaps", "", "| Method | Side | AUPRC gap | F1 gap |",
              "| --- | --- | ---: | ---: |"]
    for r in gaps:
        lines.append(f"| {r['method']} | {r['side']} | {r['auprc_train_minus_valid']:.4f} | {r['f1_train_minus_valid']:.4f} |")
    lines += ["", "## Valid differences at the selected checkpoints", "",
              "Differences below are absolute metric differences, not relative percentages.", "",
              "| Comparison | Side | AUPRC difference | F1 difference |",
              "| --- | --- | ---: | ---: |"]
    for newer, older in (("dropout_weight_decay", "original_losver"),
                         ("line_rdrop", "dropout_weight_decay")):
        for side in ("all", "vulnerable", "patched"):
            new, old = [next(r for r in table if r["method"] == name
                            and r["split"] == "valid" and r["side"] == side)
                        for name in (newer, older)]
            lines.append(f"| {newer} minus {older} | {side} | {new['auprc']-old['auprc']:+.6f} | {new['f1']-old['f1']:+.6f} |")
    lines += ["", "A smaller gap can arise from worse Train ranking; read it together with Valid absolute AUPRC. Compare C against B to isolate KL from the base regularization and the two supervised forwards. B/C double the training forwards, not optimizer steps. These are one-seed, four-epoch pilot results, not full-data or Test claims.", "",
        "## Checks and reproduction", "",
        "[GPU smoke](gpu-smoke.json) records actual Train forward/backward, new-layer stochastic dropout, zero-gradient decoupled weight decay, padding-invariant KL, both logit gradients and exact save/reload. [Protocol checks](protocol-checks.json) inspect the persisted optimizer/scheduler steps and RNG states. The [full test log](tests.log) and [resume test log](resume-tests.log) retain the executed correctness checks.", "",
        "The exact B/C train and report commands are in the [README](../../README.md). Configs, epoch/component losses and source-only line probabilities are retained under `dropout_weight_decay/` and `line_rdrop/`. Checkpoints stay locally in those same run directories; historical A remains at its original path.", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return dict(results=table, generalization_gaps=gaps, test_used=False)


def line_auxiliary_report(output_dir, data_dir):
    """Five complete, matched runs; select checkpoints/lambda using Valid P-C only."""
    from sklearn.metrics import average_precision_score
    from .metrics import paired_metrics

    root = Path(output_dir)
    experiments = (("baseline", "diff", 0.), ("diff_0.1", "diff", .1),
                   ("diff_0.3", "diff", .3), ("random_0.1", "random", .1),
                   ("random_0.3", "random", .3))
    paths = {name: root / name / "primevul_paired/unixcoder" for name, _, _ in experiments}
    configs = {name: json.loads((run / "training.metrics.json").read_text()) for name, run in paths.items()}
    reference = configs["baseline"]
    fields = ("training_members", "validation_members", "input_coverage", "training_config",
              "input_mode", "parameters", "epochs", "optimizer_steps", "model_path",
              "function_micro_batch", "window_micro_batch", "gradient_accumulation", "selection")
    annotations = read_jsonl(Path(data_dir) / "primevul_paired/valid.lines.jsonl")
    if (reference["train_samples"], reference["valid_samples"], reference["epochs"],
        reference["optimizer_steps"], reference["input_mode"]) != (8708, 1124, 16, 17424, "512"):
        raise ValueError("line auxiliary report requires the complete official 16-epoch/512 experiment")
    if [r["sample_key"] for r in annotations] != reference["validation_members"]:
        raise ValueError("full official Valid members do not match the experiment")
    labels = {name: read_jsonl(run / "train.line_labels.jsonl") for name, run in paths.items()}
    if [r["sample_key"] for r in labels["baseline"]] != reference["training_members"]:
        raise ValueError("Train line labels do not match the training member order")
    if labels["diff_0.1"] != labels["baseline"] or labels["diff_0.3"] != labels["baseline"]:
        raise ValueError("real-Diff configurations have different Train labels")
    if labels["random_0.1"] != labels["random_0.3"]:
        raise ValueError("random controls must reuse the same precomputed permutations")
    changed = 0
    for actual, shuffled in zip(labels["baseline"], labels["random_0.1"]):
        if (actual["sample_key"] != shuffled["sample_key"] or actual["line_numbers"] != shuffled["line_numbers"]
                or sum(actual["labels"]) != sum(shuffled["labels"])):
            raise ValueError("random line control changed members, visible coordinates or positive counts")
        changed += actual["labels"] != shuffled["labels"]
    table, epochs, coordinates = [], [], None
    for name, supervision, weight in experiments:
        config, run = configs[name], paths[name]
        if any(config[k] != reference[k] for k in fields) or config["completed_epochs"] != 16:
            raise ValueError(f"unmatched data, initialization settings or budget: {name}")
        setting = config["line_auxiliary"]
        if (setting["supervision"], setting["weight"], setting["validation_supervision"]) != (supervision, weight, "original_diff"):
            raise ValueError("auxiliary supervision/weight is inconsistent")
        predictions = read_jsonl(run / "valid.best-p_c.predictions.jsonl")
        if [(r["sample_key"], r["pair_key"], r["target"]) for r in predictions] != [
            (r["sample_key"], r["pair_key"], r["target"]) for r in annotations
        ]:
            raise ValueError("selected predictions and original Valid annotations are misaligned")
        ns = [(r["sample_key"], r["line_numbers"]) for r in predictions]
        if coordinates is not None and ns != coordinates:
            raise ValueError("visible physical lines differ across configurations")
        coordinates = ns
        y, p = [], []
        for actual, prediction in zip(annotations, predictions):
            gt = set(actual["lines_ground_truth"])
            if len(prediction["line_numbers"]) != len(prediction["line_scores"]):
                raise ValueError("line probabilities and coordinates differ")
            y.extend(int(n in gt) for n in prediction["line_numbers"])
            p.extend(prediction["line_scores"])
        metrics = paired_metrics(annotations, [r["probability"] for r in predictions])
        line_ap = float(average_precision_score(y, p))
        selected = json.loads((run / "best-p_c.json").read_text())
        for key in ("p_c", "auc", "mcc", "f1"):
            if abs(metrics[key] - selected["validation"][key]) > 1e-12:
                raise ValueError("selected function metrics and saved predictions differ")
        if abs(line_ap - selected["validation"]["line_ap"]) > 1e-12:
            raise ValueError("line AP is not computed on the original Diff")
        history = [r for r in read_jsonl(run / "best.training.jsonl") if r["event"] == "epoch"]
        if len(history) != 16 or max(history, key=lambda r: round(r["validation"]["p_c"], 4))["epoch"] != selected["epoch"]:
            raise ValueError("checkpoint selection does not follow original Valid P-C")
        for row in history:
            parts = row["train_components"]
            if abs(row["train_loss"] - parts["bce"] - weight * parts["line_focal"]) > 1e-7:
                raise ValueError("training losses do not reproduce the joint objective")
            epochs.append(dict(configuration=name, epoch=row["epoch"], step=row["global_step"],
                bce=parts["bce"], line_focal=parts["line_focal"], total=row["train_loss"],
                valid_p_c=row["validation"]["p_c"], valid_auc=row["validation"]["auc"],
                valid_line_ap=row["validation"]["line_ap"], seconds=row["epoch_seconds"]))
        table.append(dict(configuration=name, supervision=supervision, weight=weight,
            epoch=selected["epoch"], correct_pairs=metrics["pair_counts"]["p_c"], pairs=metrics["pairs"],
            p_c=metrics["p_c"], auc=metrics["auc"], mcc=metrics["mcc"], f1=metrics["f1"],
            accuracy=metrics["accuracy"], recall=metrics["recall"], patched_fpr=metrics["patched_fpr"],
            line_ap=line_ap, lines=len(y), positive_lines=sum(y)))
    selected = {family: max((r for r in table if r["supervision"] == family and r["weight"] > 0),
                           key=lambda r: (round(r["p_c"], 4), -r["weight"]))
                for family in ("diff", "random")}
    result = dict(results=table, selected_by_valid_p_c=selected, test_used=False,
                  random_changed_functions=changed, matched_fields=list(fields))
    write_json(root / "comparison.json", result)
    for filename, records in (("comparison.csv", table), ("training.csv", epochs)):
        with (root / filename).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    lines = ["# Function classification with original-Diff line auxiliary supervision", "",
        "Full official PrimeVul Original Train=4,354 pairs/8,708 functions and Valid=562 pairs/1,124 functions. All five runs use 512 tokens, seed123456, batch8, 16 epochs/17,424 optimizer steps, LR2e-5, 10% warmup and unchanged LOSVERAdamW (weight decay0). Threshold is strictly >.5. Test is never read.", "",
        "All runs share the original UniXcoder and function linear-head initialization. A single additional Linear(hidden_size,1), initialized without advancing the sampler RNG, is present in all five controls. Each function's visible code-line states are means from the SAME encoder output. Original Focal (alpha=.8, gamma=1.5, pos_weight=5) is averaged over visible lines WITHIN each function, then over functions. Empty Diff is all-zero line supervision and never drops function BCE.", "",
        "Independent `random.Random(123456)` permutes each Train function's visible line labels once, before training. Counts are preserved and the same permutations are used at both weights. Valid line AP pools all visible lines against real Diff, whereas Focal is function-balanced. Only function-level Valid P-C selects the epoch and lambda; epoch ties keep the earlier checkpoint, lambda ties prefer .1. Line AP and AUC do not select anything.", "",
        "Inference runs only the original global code-token mean and original function linear head. Diff and line logits do not affect pooling, token weights or the function prediction. The lambda0 line head is untrained: its AP is a random-projection diagnostic, not a learned localizer baseline.", "",
        "## Valid results", "",
        "| Configuration | Selected epoch | Correct pairs /562 | P-C | AUC | MCC | F1 | True-Diff line AP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in table:
        lines.append(f"| {row['configuration']} | {row['epoch']} | {row['correct_pairs']}/562 | {row['p_c']:.4f} | {row['auc']:.4f} | {row['mcc']:.4f} | {row['f1']:.4f} | {row['line_ap']:.4f} |")
    lines += ["", "## Selection and interpretation", ""]
    baseline = table[0]
    for family, row in selected.items():
        lines.append(f"{family}: Valid P-C selects lambda={row['weight']}; P-C difference from baseline={row['p_c']-baseline['p_c']:+.6f}, AUC difference={row['auc']-baseline['auc']:+.6f}, MCC difference={row['mcc']-baseline['mcc']:+.6f}.")
        lines.append("")
    for weight in (.1, .3):
        real, shuffled = [next(r for r in table if r["supervision"] == family and r["weight"] == weight)
                          for family in ("diff", "random")]
        lines += [f"At matched lambda={weight}, real Diff minus random: P-C={real['p_c']-shuffled['p_c']:+.6f}, AUC={real['auc']-shuffled['auc']:+.6f}, MCC={real['mcc']-shuffled['mcc']:+.6f}, F1={real['f1']-shuffled['f1']:+.6f}.", ""]
    if selected["diff"]["p_c"] <= baseline["p_c"]:
        lines += ["The predefined real-Diff auxiliary grid does not improve the primary Valid P-C over baseline. The other metrics must be read separately; no further module or Test experiment is added.", ""]
    for row in table:
        if row["supervision"] == "diff" and row["weight"] > 0:
            lines += [f"{row['configuration']}: F1={row['f1']:.6f} versus baseline {baseline['f1']:.6f}; vulnerable Recall={row['recall']:.6f} versus {baseline['recall']:.6f}; patched-side FPR={row['patched_fpr']:.6f} versus {baseline['patched_fpr']:.6f}. A higher F1 does not by itself imply better pair discrimination or a lower false-positive rate.", ""]
    lines += [f"Random permutations changed {changed}/8708 functions' visible labels. Unchanged all-zero/all-one or coincidentally identical permutations are retained. Line denominators are identical across the five runs and are saved in comparison.csv. Increasing line AP alone does not establish a classification gain; compare real Diff against the count-matched random control at BOTH preset weights. This is one training seed and one predefined lambda grid, with no Test selection and no additional method.", "",
        "[Training losses](training.csv) contain both unweighted BCE and function-balanced Focal, total objective and Valid metrics for all 80 epochs. Each run retains configuration, best-P-C predictions (function probabilities plus physical-line scores), logs and resumable local checkpoints. [GPU smoke](gpu-smoke.json), [tests](tests.log) and [resume checks](resume-tests.log) record the actual correctness checks; [protocol checks](protocol-checks.json) inspect the completed runs. Reproduction commands are in the [README](../../README.md).", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def paired_report(output_dir, data_dir, *, reference_dir=None, localizer_regularization=False):
    if (Path(output_dir) / "baseline/primevul_paired/unixcoder/config.json").exists():
        return line_auxiliary_report(output_dir, data_dir)
    if localizer_regularization:
        return localizer_regularization_report(output_dir, data_dir, reference_dir)
    if (Path(output_dir) / "supervision/diff_neighbour/statistics.json").exists():
        return diff_neighbour_report(output_dir, data_dir, reference_dir)
    if (Path(output_dir) / "supervision/statistics.json").exists():
        return repair_region_report(output_dir, data_dir)
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
