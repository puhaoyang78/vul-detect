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
    attention_predictions = []
    loss = 0.0
    loss_mass = 0.0
    at = 0
    for batch in training_bar(loader, desc=f"Valid {variant}"):
        joint = getattr(model, "line_auxiliary_weight", None) is not None and "line_labels" in batch
        if getattr(model,"pooling",None)=="bounded_attention":
            # Predict without any function or Diff label. Diagnostics are
            # attached only after these single-source probabilities exist.
            prob, details = paired_forward(model,batch,variant,device,with_labels=False,return_attention=True)
            value = torch.nn.functional.binary_cross_entropy_with_logits(
                details["logits"],batch["labels"].float().to(device)
            )
            for numbers,rates,counts in zip(batch["line_numbers"],details["line_rates"],details["line_counts"]):
                n = int((numbers>=0).sum())
                attention_predictions.append(dict(line_numbers=numbers[:n].tolist(),
                    line_rates=rates[:n].cpu().tolist(),line_token_counts=counts[:n].cpu().tolist()))
        elif joint:
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
        if attention_predictions:
            if len(attention_predictions)!=len(predictions):
                raise ValueError("attention and function prediction members differ")
            for prediction,attention in zip(predictions,attention_predictions):
                prediction.update(attention)
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
        **(dict(bounded_attention=True,
                attention_region=checkpoint["bounded_attention"]["region_supervision"])
           if "bounded_attention" in checkpoint else {}),
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
        if {r["sample_key"] for r in rows} != set(sample_keys):
            raise ValueError("requested localization members are missing from the source split")
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
                rows = paired_rows(data_dir, output_dir, variant, split, method=method,
                                   localizer_dir=cp.get("localizer_predictions", {}).get(split))
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


def oof_report(output_dir, data_dir):
    """Pilot-only localization shift and aligned detector comparison; no Test reads."""
    import numpy as np
    from collections import Counter
    from scipy.stats import wasserstein_distance
    from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
    from .data import paired_encoding
    from .metrics import paired_metrics

    root = Path(output_dir)
    setup = json.loads((root / "oof.json").read_text())
    full = Path(setup["full_localizer_dir"])
    reference = json.loads((full / "training.metrics.json").read_text())
    tokenizer = core.RobertaTokenizerFast.from_pretrained(reference["model_path"], trim_offsets=False)
    annotations = {s: {r["sample_key"]:r for r in read_jsonl(
        Path(data_dir) / "primevul_paired" / f"{s}.lines.jsonl")} for s in ("train", "valid")}
    encodings = {}
    conditions = (("in_sample", "train", full),
                  ("oof", "train", root/"oof_predictions"), ("full_valid", "valid", full),
                  ("fold_0_valid", "valid", root/"fold_0/primevul_paired/losver_localizer"),
                  ("fold_1_valid", "valid", root/"fold_1/primevul_paired/losver_localizer"))
    details, line_table = [], []
    for name, split, path in conditions:
        predictions = read_jsonl(path/f"{split}.line_scores.jsonl")
        keys = reference["training_members" if split == "train" else "validation_members"]
        if [r["sample_key"] for r in predictions] != keys:
            raise ValueError("OOF report line prediction members/order differ")
        side_lines = {side: ([], []) for side in ("all", "vulnerable", "patched")}
        for pred in predictions:
            key = pred["sample_key"]
            row = annotations[split][key]
            if key not in encodings:
                encodings[key] = paired_encoding(row["func"], tokenizer, "512")
            f = encodings[key]
            ns, ps = pred["line_numbers"], pred["line_scores"]
            if (ns != f["line_numbers"] or len(ns) != len(ps)
                    or any(not np.isfinite(p) or not 0 <= p <= 1 for p in ps)):
                raise ValueError("OOF line scores do not match actual token/physical-line mapping")
            gt = set(row["lines_ground_truth"])
            ys = [int(n in gt) for n in ns]
            for side in ("all", "vulnerable" if row["target"] else "patched"):
                side_lines[side][0].extend(ys)
                side_lines[side][1].extend(ps)
            code = np.asarray(f["code_mask"], dtype=bool)
            true_tokens, selected_tokens = np.zeros(len(code), dtype=bool), np.zeros(len(code), dtype=bool)
            for n, p, indices in zip(ns, ps, f["line_token_indices"]):
                if n in gt:
                    true_tokens[indices] = True
                if p > .5:
                    selected_tokens[indices] = True
            true_tokens &= code
            selected_tokens &= code
            weights = np.exp(4 * selected_tokens.astype(float)) * code
            weights /= weights.sum() if weights.sum() else 1
            details.append(dict(condition=name, split=split, sample_key=key, pair_key=row["pair_key"],
                target=row["target"], visible_lines=len(ns), true_lines=sum(ys),
                selected_lines=sum(p > .5 for p in ps), code_tokens=int(code.sum()),
                true_tokens=int(true_tokens.sum()), selected_tokens=int(selected_tokens.sum()),
                selected_true_tokens=int((selected_tokens & true_tokens).sum()),
                true_diff_mass=float(weights[true_tokens].sum()),
                background_mass=float(weights[code & ~true_tokens].sum()),
                selected_background_mass=float(weights[selected_tokens & ~true_tokens].sum()),
                selected_mass=float(weights[selected_tokens].sum()),
                equal_weight_diff_mass=float(true_tokens.sum()/code.sum()) if code.any() else 0.))
        for side, (ys, ps) in side_lines.items():
            selected = np.asarray(ps) > .5
            fs = [r for r in details if r["condition"] == name
                  and (side == "all" or r["target"] == int(side == "vulnerable"))]
            nt = sum(r["true_tokens"] for r in fs)
            npred = sum(r["selected_tokens"] for r in fs)
            hits = sum(r["selected_true_tokens"] for r in fs)
            with_diff = [r for r in fs if r["true_tokens"]]
            line_table.append(dict(condition=name, split=split, side=side, functions=len(fs),
                lines=len(ys), positive_lines=sum(ys), predicted_positive_lines=int(selected.sum()),
                precision=float(precision_score(ys, selected, zero_division=0)),
                recall=float(recall_score(ys, selected, zero_division=0)),
                f1=float(f1_score(ys, selected, zero_division=0)),
                auprc=float(average_precision_score(ys, ps)) if sum(ys) else None,
                selected_line_fraction=float(selected.mean()),
                no_prediction_functions=sum(r["selected_lines"] == 0 for r in fs),
                token_precision=hits/npred if npred else 0., token_recall=hits/nt if nt else 0.,
                true_tokens=nt, selected_tokens=npred, selected_true_tokens=hits,
                mean_true_diff_mass=float(np.mean([r["true_diff_mass"] for r in fs])),
                mean_background_mass=float(np.mean([r["background_mass"] for r in fs])),
                mean_selected_background_mass=float(np.mean([r["selected_background_mass"] for r in fs])),
                mean_selected_mass=float(np.mean([r["selected_mass"] for r in fs])),
                mean_equal_weight_diff_mass=float(np.mean([r["equal_weight_diff_mass"] for r in fs])),
                visible_diff_functions=len(with_diff),
                mean_diff_mass_when_visible=float(np.mean([r["true_diff_mass"] for r in with_diff])) if with_diff else None))
    distances = []
    for name in ("in_sample", "oof"):
        for field in ("selected_line_fraction", "true_diff_mass", "selected_background_mass"):
            def values(condition):
                return [r["selected_lines"]/max(1,r["visible_lines"]) if field == "selected_line_fraction"
                        else r[field] for r in details if r["condition"] == condition]
            distances.append(dict(condition=name, reference="full_valid", quantity=field,
                                  wasserstein_distance=float(wasserstein_distance(values(name), values("full_valid")))))
    runs = {name: root/name/"primevul_paired/losver" for name in ("in_sample", "oof")}
    configs = {name: json.loads((run/"training.metrics.json").read_text()) for name,run in runs.items()}
    matched = ("training_members", "validation_members", "training_config", "epochs", "optimizer_steps",
               "input_mode", "input_coverage", "model_path", "parameters", "selection", "selection_ties",
               "function_micro_batch", "window_micro_batch", "localizer_path")
    if any(configs["in_sample"][k] != configs["oof"][k] for k in matched):
        raise ValueError("detector comparison differs beyond Train line prediction provenance")
    for name, config in configs.items():
        if (config["training_members"] != reference["training_members"]
                or config["validation_members"] != reference["validation_members"]
                or config["epochs"] != 4 or config["optimizer_steps"] != 512
                or config["localizer_predictions"]["valid"] != str(full)
                or config["localizer_predictions"]["train"] != str(full if name == "in_sample" else root/"oof_predictions")):
            raise ValueError("detectors must use exact original Pilot members, budget and common Valid localizer")
    comparisons, predictions, costs = [], {}, []
    for name, run in runs.items():
        ps = read_jsonl(run/"valid.best-p_c.predictions.jsonl")
        keys = reference["validation_members"]
        if [p["sample_key"] for p in ps] != keys:
            raise ValueError("detector Valid prediction alignment differs")
        rows = [annotations["valid"][k] for k in keys]
        if any(p["target"] != r["target"] or p["pair_key"] != r["pair_key"] for p,r in zip(ps,rows)):
            raise ValueError("detector prediction labels/pairs differ from source")
        scores = [p["probability"] for p in ps]
        value = paired_metrics(rows, scores)
        selected = json.loads((run/"best-p_c.json").read_text())
        if any(abs(value[k]-selected["validation"][k]) > 1e-12 for k in ("p_c", "auc", "mcc", "f1", "patched_fpr")):
            raise ValueError("selected detector metrics and saved predictions differ")
        comparisons.append(dict(method=name, epoch=selected["epoch"], **value))
        predictions[name] = scores
    for name, run in dict(runs, **{f"fold_{i}":root/f"fold_{i}/primevul_paired/losver_localizer" for i in (0,1)}).items():
        events = [e for e in read_jsonl(run/"best.training.jsonl") if e.get("event") == "epoch"]
        costs.append(dict(run=name, epochs=len(events),
            training_seconds=sum(e["training_seconds"] for e in events),
            training_and_validation_seconds=sum(e["epoch_seconds"] for e in events)))
    transitions, changes = Counter(), []
    def category(v, p):
        return "p_c" if v > .5 and p <= .5 else "p_v" if v > .5 else "p_r" if p > .5 else "p_b"
    for i in range(0, len(rows), 2):
        a, b = predictions["in_sample"][i:i+2], predictions["oof"][i:i+2]
        before, after = category(*a), category(*b)
        transitions[before,after] += 1
        changes.append(dict(pair_key=rows[i]["pair_key"], vulnerable_key=rows[i]["sample_key"],
            patched_key=rows[i+1]["sample_key"], in_sample_state=before, oof_state=after,
            in_sample_vulnerable=a[0], in_sample_patched=a[1],
            oof_vulnerable=b[0], oof_patched=b[1], delta_vulnerable=b[0]-a[0], delta_patched=b[1]-a[1]))
    probability_changes = dict(
        mean_vulnerable_delta=float(np.mean([r["delta_vulnerable"] for r in changes])),
        mean_patched_delta=float(np.mean([r["delta_patched"] for r in changes])))
    for side in ("vulnerable", "patched"):
        for before, after in ((0,1), (1,0)):
            probability_changes[f"{side}_{before}_to_{after}"] = sum(
                (r[f"in_sample_{side}"] > .5) == before and (r[f"oof_{side}"] > .5) == after for r in changes)
    rngs = [torch.load(run/"last.pt", map_location="cpu", mmap=True, weights_only=False)["rng"] for run in runs.values()]
    same_sampler_rng = torch.equal(rngs[0]["torch"], rngs[1]["torch"])
    if not same_sampler_rng:
        raise ValueError("detector CPU sampler/loader RNG advancement differs")
    historical = read_jsonl(Path(setup["reference_dir"])/"primevul_paired/losver/valid.best-p_c.predictions.jsonl")
    if [r["sample_key"] for r in historical] != reference["validation_members"]:
        raise ValueError("historical detector comparison members differ")
    historical_error = max(abs(r["probability"]-p) for r,p in zip(historical,predictions["in_sample"]))
    checks = dict(matched_fields=list(matched), sampler_rng_equal=same_sampler_rng,
        same_ordered_members=True, shared_valid_predictions=True, test_used=False,
        pair_overlap=setup["cross_fold_pair_overlap"],
        baseline_historical_max_probability_error=historical_error,
        initialization="same pretrained factory and seed; no localizer weights initialize either detector")
    comparison = dict(setup=setup, localization=line_table, detectors=comparisons,
        transitions=[dict(before=a,after=b,pairs=n) for (a,b),n in sorted(transitions.items())],
        costs=costs, checks=checks, distribution_distances=distances, probability_changes=probability_changes)
    write_json(root/"comparison.json", comparison)
    for filename, values in (("localization.csv", line_table), ("token_mass.samples.csv", details),
                             ("detectors.csv", comparisons), ("valid.pair_changes.csv", changes),
                             ("distribution_shift.csv", distances), ("costs.csv", costs)):
        with (root/filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    lines = ["# LOSVER two-fold OOF 512 Pilot", "",
        "Train 1,024 functions / 512 pairs; Valid 256 / 128. No Test was read. "
        "Member sampling seed 42, fold/training seed 123456. Original Diff, Focal and binary 5/1 Softmax unchanged. "
        "Both detectors use the same completed full Pilot localizer on Valid and for single-function inference.", "",
        "## Localization", "",
        "| Prediction source | Lines | Precision | Recall | F1 | AP | Selected lines | No prediction functions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    all_lines = [r for r in line_table if r["side"] == "all"]
    for r in all_lines:
        lines.append(f"| {r['condition']} | {r['lines']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['auprc']:.4f} | {r['selected_line_fraction']:.2%} | {r['no_prediction_functions']}/{r['functions']} |")
    lines += ["", "Line metrics use original visible Diff labels. Vulnerable/patched breakdowns are in localization.csv.", "",
        "| Prediction source | Selected-token precision | Selected-token recall | Diff mass | Background mass | Selected background mass |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in all_lines:
        lines.append(f"| {r['condition']} | {r['token_precision']:.4f} | {r['token_recall']:.4f} | {r['mean_true_diff_mass']:.4f} | {r['mean_background_mass']:.4f} | {r['mean_selected_background_mass']:.4f} |")
    lines += ["", "Token P/R are micro counts over valid code tokens; mass is averaged per function using the actual exp(5)/exp(1) weights. "
        "Functions with no visible Diff have Diff mass zero; visible-positive-only means and equal-pooling reference are also saved. "
        "Diff agreement is a supervision diagnostic, not proof of vulnerability-semantic correctness.", "", "## Detector Valid", "",
        "| Train predictions | Epoch | P-C | AUC | MCC | F1 | Vulnerable recall | Patched FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in comparisons:
        lines.append(f"| {r['method']} | {r['epoch']} | {r['pair_counts']['p_c']}/{r['pairs']} ({r['p_c']:.2%}) | {r['auc']:.4f} | {r['mcc']:.4f} | {r['f1']:.4f} | {r['recall']:.4f} | {r['patched_fpr']:.4f} |")
    baseline, oof = comparisons
    lines += ["", f"OOF minus in-sample: P-C {oof['p_c']-baseline['p_c']:+.4f}, AUC {oof['auc']-baseline['auc']:+.4f}, MCC {oof['mcc']-baseline['mcc']:+.4f}, F1 {oof['f1']-baseline['f1']:+.4f}.", "",
        f"Strict-pair gains: {sum(n for (a,b),n in transitions.items() if a != 'p_c' and b == 'p_c')}; losses: {sum(n for (a,b),n in transitions.items() if a == 'p_c' and b != 'p_c')}. "
        "All 128 pairs and their probability changes are saved in valid.pair_changes.csv; the full state transition table is in comparison.json.", "",
        f"Mean probability change: vulnerable {probability_changes['mean_vulnerable_delta']:+.4f}, patched {probability_changes['mean_patched_delta']:+.4f}.", "",
        "| In-sample → OOF | P-C | P-V | P-B | P-R |", "|---|---:|---:|---:|---:|"]
    for before in ("p_c", "p_v", "p_b", "p_r"):
        lines.append(f"| {before} | " + " | ".join(str(transitions[before,after]) for after in ("p_c", "p_v", "p_b", "p_r")) + " |")
    ins, out, valid = all_lines[:3]
    lines += ["", "## Observations", "",
        f"OOF vs in-sample Train AP: {out['auprc']:.4f} vs {ins['auprc']:.4f}; the common Valid AP is {valid['auprc']:.4f}. "
        f"Absolute pooled selected-line-rate gap to Valid is {abs(ins['selected_line_fraction']-valid['selected_line_fraction']):.4f} in-sample and {abs(out['selected_line_fraction']-valid['selected_line_fraction']):.4f} OOF. "
        "Ranking/label agreement and selection rate therefore need separate interpretation.", "",
        "Function-level Wasserstein distances to the same full-Valid predictions:", "",
        "| Quantity | In-sample → Valid | OOF → Valid |", "|---|---:|---:|"]
    for field in ("selected_line_fraction", "true_diff_mass", "selected_background_mass"):
        ds = [r["wasserstein_distance"] for r in distances if r["quantity"] == field]
        lines.append(f"| {field} | {ds[0]:.4f} | {ds[1]:.4f} |")
    lines += ["", "The distributions are closer on these three per-function statistics, especially selected-background mass; "
        "the pooled selected-line-rate gap still grows. Reduced distribution distances do not establish correct localization or classification benefit.", "",
        ("OOF does not improve either selected Valid P-C or AUC in this Pilot. "
         "Any higher recall/F1 must be interpreted alongside patched-side FPR; there is no classification benefit supporting a full run."
         if oof["p_c"] <= baseline["p_c"] and oof["auc"] <= baseline["auc"] else
         "The selected Valid changes are shown above; a single-seed Pilot does not establish stable classification benefit."), "",
        "## Budget and interpretation", "",
        "The full localizer trained 4 epochs on 1,024 functions; each fold localizer trains 8 epochs on 512 functions. "
        "Each runs 512 optimizer steps with the same LR/warmup schedule, but the fold models see half as many unique functions twice as often. "
        "Fold checkpoint selection uses only the same 256-function source Valid, never held-out Train. "
        "Eight versus four validation opportunities and potentially different selected steps remain comparison limitations.", ""]
    for f in setup["folds"]:
        lines.append(f"- Fold {f['fold']}: selected epoch {f['selected_epoch']}, step {f['selected_optimizer_steps']}/{f['optimizer_steps']}; held-out functions {f['held_out_functions']}.")
    lines += ["", f"Cross-fold pair overlap: 0. Exact-source overlap across different pairs: {setup['cross_fold_exact_source_groups']} groups / {setup['cross_fold_exact_source_functions']} functions; no member was removed or resampled.", "",
        "Fold-specific Valid rows above diagnose half-data model quality only; neither detector uses those Valid predictions. "
        "Both use full_valid. Function-level distribution distances are in distribution_shift.csv; "
        "closer values on one statistic do not establish that every aspect of the shift improved.", "",
        "On this common Valid, the fold models are not consistently worse than the full model in F1/AP. "
        "The much lower OOF Train quality than in-sample quality alone cannot distinguish missing in-sample exposure from half-data training effects.", "",
        "Training/validation loop times are in costs.csv. Jobs ran concurrently on two RTX 4090 GPUs; "
        "these times exclude initial loading/feature extraction and final checkpoint/export I/O, and are not a controlled speed comparison. "
        "Real GPU cache and source-only single-function checks are saved in gpu_checks.json.", "",
        "OOF changes both training membership exposure and the effective quality of localization. "
        "A weaker OOF localizer cannot isolate the effect of removing in-sample exposure alone. "
        "This is one fixed-seed Pilot, not evidence of stable population-level improvement. "
        "No extra method, full-data training or Test evaluation was launched.", ""]
    (root/"report.md").write_text("\n".join(lines), encoding="utf-8")
    return comparison


def check_bounded_attention(data_dir,output_dir,model_path,*,device="auto",input_mode="512",regularization=False):
    """Actual pretrained encoder, Train examples, two updates; no experiment claim."""
    import tempfile
    import time
    from .training import paired_collate,paired_forward,LOSVERAdamW
    from .data import paired_features
    if input_mode!="512":
        raise ValueError("bounded attention checks require 512")
    device=core.resolve_device(device)
    started=time.perf_counter()
    core.seed_everything(123456)
    baseline,tokenizer=core.losver_model(model_path,"baseline")
    initial_rng=torch.get_rng_state()
    core.seed_everything(123456)
    model,_=core.losver_model(model_path,"baseline",bounded_attention=True,attention_region="diff")
    assert torch.equal(torch.get_rng_state(),initial_rng)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in baseline.state_dict().items())
    records=read_jsonl(Path(data_dir)/"primevul_paired/train.lines.jsonl")
    features=[];selected=[]
    for r in records:
        f=paired_features(r,tokenizer,"baseline")
        if (0<sum(f["line_labels"])<len(f["line_labels"])
                and (not regularization or len(f["line_numbers"])>=10)):
            features.append(f);selected.append(r["sample_key"])
        if len(features)==4:
            break
    if len(features)!=4:
        raise ValueError("need four real Train functions with nonempty proper visible edits")
    batch=paired_collate(features)
    baseline=baseline.to(device).eval()
    with torch.no_grad():
        expected=paired_forward(baseline,batch,"baseline",device,with_labels=False)
    baseline=baseline.cpu()
    model=model.to(device).eval()
    with torch.no_grad():
        actual,details=paired_forward(model,batch,"baseline",device,with_labels=False,return_attention=True)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    delta=float((actual-expected).abs().max())
    weights=details["token_attention"]
    mask=batch["code_mask"].to(device)
    assert bool((weights[mask]>0).all()) and not bool(weights[~mask].any())
    torch.testing.assert_close(weights.sum(-1),torch.ones(len(features),device=device))
    del baseline,details
    optimizer=LOSVERAdamW(model.parameters(),lr=2e-5)
    _,_,parts=paired_forward(model,batch,"baseline",device,return_attention=True)
    parts["region_loss"].backward()
    encoder=model.encoder.embeddings.word_embeddings.weight
    initial_encoder_gradient=float(encoder.grad.abs().sum())
    assert initial_encoder_gradient==0 and float(model.line_attention.weight.grad.abs().sum())>0
    optimizer.step();optimizer.zero_grad()
    _,_,parts=paired_forward(model,batch,"baseline",device,return_attention=True)
    parts["region_loss"].backward()
    encoder_gradient=float(encoder.grad.abs().sum())
    assert encoder_gradient>0 and bool(torch.isfinite(encoder.grad).all())
    optimizer.zero_grad()
    model.train()
    loss,_,parts=paired_forward(model,batch,"baseline",device,return_attention=True)
    loss.backward()
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    optimizer.step();optimizer.zero_grad();model.eval()
    with torch.no_grad():
        probability=paired_forward(model,batch,"baseline",device,with_labels=False)
        changed=dict(batch,line_labels=1-batch["line_labels"])
        torch.testing.assert_close(probability,paired_forward(model,changed,"baseline",device,with_labels=False),rtol=0,atol=0)
    cp=dict(protocol="losver",line_mapping="source_offsets",input_mode="512",variant="baseline",
            model_path=str(model_path),pooling="bounded_attention",bounded_attention=dict(region_supervision="diff"),
            state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()})
    with tempfile.TemporaryDirectory(prefix="bounded-mil-check-") as tmp:
        path=Path(tmp)/"check.pt";torch.save(cp,path)
        del model,optimizer,parts
        if device.type=="cuda": torch.cuda.empty_cache()
        restored,_=load_paired(torch.load(path,map_location="cpu",weights_only=False),device)
        with torch.no_grad():
            reloaded=paired_forward(restored,batch,"baseline",device,with_labels=False)
        torch.testing.assert_close(probability,reloaded,rtol=0,atol=0)
    regularization_checks={}
    if regularization:
        import random
        from .training import random_attention_suppression,make_attention_ema,attention_ema_loss,update_attention_ema

        for mode in ("random","high"):
            restored.attention_suppression=mode;restored.train()
            mask=random_attention_suppression(batch,random.Random(123456)) if mode=="random" else None
            loss,_,d=paired_forward(restored,batch,"baseline",device,return_attention=True,suppression_mask=mask)
            valid=batch["code_mask"].to(device)
            token=d["suppressed_token_attention"]
            ratios=[float(x.max()/x.min()) for x in (t[v] for t,v in zip(token,valid))]
            assert bool((token[valid]>0).all()) and not bool(token[~valid].any()) and max(ratios)<=5+1e-6
            torch.testing.assert_close(d["bce"],(d["bce_normal"]+d["bce_suppressed"])/2)
            loss.backward()
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in restored.parameters())
            restored.zero_grad();restored.eval()
            with torch.no_grad():
                p=paired_forward(restored,batch,"baseline",device,with_labels=False)
                torch.testing.assert_close(p,probability,rtol=0,atol=0)
            regularization_checks[mode]=dict(selected_lines=d["suppression_mask"].sum(-1).tolist(),
                all_valid_tokens_positive=True,max_token_weight_ratio=max(ratios),inference_unchanged=True)
            del d,loss
        restored.attention_suppression=None
        ema=make_attention_ema(restored)
        # Controlled nonzero discrepancy checks gradients; not a learned result.
        with torch.no_grad():
            restored.line_attention.weight.add_(torch.linspace(-.01,.01,restored.line_attention.weight.numel(),
                device=device).reshape_as(restored.line_attention.weight))
        _,d=paired_forward(restored,batch,"baseline",device,with_labels=False,return_attention=True)
        kl=attention_ema_loss(restored,ema,batch,d["line_attention"],device)
        kl.backward()
        eg=float(restored.encoder.embeddings.word_embeddings.weight.grad.abs().sum())
        hg=float(restored.line_attention.weight.grad.abs().sum())
        assert eg>0 and hg>0 and all(p.grad is None for p in ema.parameters())
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in restored.parameters())
        before=ema["line_attention"].weight.detach().clone()
        LOSVERAdamW(restored.parameters(),lr=2e-5).step()
        update_attention_ema(ema,restored)
        torch.testing.assert_close(ema["line_attention"].weight,.99*before+.01*restored.line_attention.weight.detach())
        regularization_checks["ema"]=dict(kl=float(kl.detach()),encoder_gradient=eg,attention_gradient=hg,
            reference_has_no_gradient=True,healthy_parameter_update=True,controlled_head_perturbation=True)
    result=dict(device=str(device),train_members=selected,updates=2,initialization_and_rng_equal=True,
                initial_probability_max_delta=delta,initial_region_encoder_gradient=initial_encoder_gradient,
                region_encoder_gradient_after_head_update=encoder_gradient,all_gradients_finite=True,
                diff_independent_inference=True,save_reload_equal=True,seconds=time.perf_counter()-started,
                peak_memory_mib=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=="cuda" else None,
                test_used=False,full_experiment=False)
    if regularization:
        result["attention_regularization"]=regularization_checks
    write_json(Path(output_dir)/"gpu-smoke.json",result)
    return result


def bounded_region_record(annotation,prediction,labels=None):
    """Post-prediction diagnostics; edit membership never changes the weights."""
    import math
    numbers=prediction["line_numbers"]
    counts=prediction["line_token_counts"]
    rates=prediction["line_rates"]
    gt=set(annotation["lines_ground_truth"])
    selected=([int(n in gt) for n in numbers]
              if labels is None else labels)
    if (annotation["sample_key"]!=prediction["sample_key"] or
        not len(numbers)==len(counts)==len(rates)==len(selected) or
        any(n<=0 for n in counts) or any(not 1<=r<=5 for r in rates) or any(y not in (0,1) for y in selected)):
        raise ValueError("invalid aligned bounded-attention diagnostics")
    total=sum(counts);nr=sum(n*y for n,y in zip(counts,selected));nb=total-nr
    weighted=sum(n*r for n,r in zip(counts,rates))
    region_mass=sum(n*r*y for n,r,y in zip(counts,rates,selected))/weighted
    ratio=((sum(n*r*y for n,r,y in zip(counts,rates,selected))/nr) /
           (sum(n*r*(1-y) for n,r,y in zip(counts,rates,selected))/nb)) if nr>0 and nb>0 else None
    return dict(sample_key=annotation["sample_key"],pair_key=annotation["pair_key"],target=annotation["target"],
                visible_lines=len(numbers),region_lines=sum(selected),code_tokens=total,region_tokens=nr,
                token_coverage=nr/total,region_attention_mass=region_mass,
                token_weight_ratio=max(rates)/min(rates),
                attention_enrichment=region_mass/(nr/total) if nr>0 else None,
                mean_rate_ratio=ratio,eligible=nr>0 and nb>0,
                region_margin=max(0.,math.log(2)-math.log(ratio)) if ratio is not None else None)


def attention_distribution_record(annotation, prediction, previous=None):
    """Post-prediction JSD; uniform means equal per-token weights."""
    import math

    def distribution(r):
        mass = [n*v for n,v in zip(r["line_token_counts"],r["line_rates"])]
        return [x/sum(mass) for x in mass]

    def jsd(p, q):
        middle = [(a+b)/2 for a,b in zip(p,q)]
        return sum(a*math.log(a/m)+b*math.log(b/m) for a,b,m in zip(p,q,middle))/2

    record = bounded_region_record(annotation, prediction)
    a = distribution(prediction)
    counts, rates = prediction["line_token_counts"], prediction["line_rates"]
    mean_rate = sum(rates)/len(rates)
    consecutive = None
    if previous is not None:
        if any(previous[k]!=prediction[k] for k in ("sample_key","line_numbers","line_token_counts")):
            raise ValueError("consecutive attention diagnostics changed input coordinates")
        consecutive = jsd(a,distribution(previous))
    record.update(
        consecutive_jsd=consecutive,
        token_mean_jsd=jsd(a,[n/sum(counts) for n in counts]),
        normalized_line_entropy=-sum(x*math.log(x) for x in a)/math.log(len(a)) if len(a)>1 else 1.,
        rate_cv=(sum((x-mean_rate)**2 for x in rates)/len(rates))**.5/mean_rate,
    )
    return record


def attention_epoch_diagnostics(model, rows, loaders, tokenizer, device, epoch, previous, output):
    """Fixed source-only eval cohorts. No effect on training RNG or model mode."""
    from statistics import mean
    from .training import rng_state,restore_rng

    before, mode = rng_state(), model.training
    summaries, current, records = {}, {}, []
    try:
        model.eval()
        for split in ("train","valid"):
            _, predictions = validate_paired(model,rows[split],tokenizer,"baseline",device,loader=loaders[split])
            old = {r["sample_key"]:r for r in previous.get(split,[])}
            if old and set(old)!={r["sample_key"] for r in predictions}:
                raise ValueError("fixed attention diagnostic members changed")
            stats = [attention_distribution_record(ann,pred,old.get(pred["sample_key"]))
                     for ann,pred in zip(rows[split],predictions)]
            current[split] = predictions
            summary = dict(functions=len(stats),eligible=sum(r["eligible"] for r in stats))
            for key in ("mean_rate_ratio","attention_enrichment","consecutive_jsd","token_mean_jsd",
                        "normalized_line_entropy","rate_cv"):
                values = [r[key] for r in stats if r[key] is not None]
                summary[key] = mean(values) if values else None
            summary["max_token_weight_ratio"] = max(r["token_weight_ratio"] for r in stats)
            summaries[split] = summary
            records.extend({**prediction,**stat,"epoch":epoch,"split":split}
                           for prediction,stat in zip(predictions,stats))
    finally:
        restore_rng(before)
        model.train(mode)
    with Path(output).open("a",encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record,ensure_ascii=False)+"\n")
    return summaries,current


def bounded_attention_report(output_dir,data_dir,*,device="auto"):
    """Complete fixed-budget comparison, Train diagnostics and fixed-weight C intervention."""
    from statistics import mean,median
    from .training import paired_loader
    from .metrics import paired_metrics
    from .losver_diagnosis import write_csv

    root=Path(output_dir)
    names=("global_mean","attention_bce","diff_region","random_region")
    paths={name:root/name/"primevul_paired/unixcoder" for name in names}
    configs={name:json.loads((p/"training.metrics.json").read_text()) for name,p in paths.items()}
    base=configs["global_mean"]
    common=("training_members","validation_members","training_config","input_coverage","input_mode",
            "epochs","optimizer_steps","model_path","function_micro_batch","window_micro_batch","gradient_accumulation")
    if (base["train_samples"],base["valid_samples"],base["epochs"],base["optimizer_steps"])!=(8708,1124,16,17424):
        raise ValueError("bounded-attention report requires all four complete official 16-epoch runs")
    annotations={s:read_jsonl(Path(data_dir)/"primevul_paired"/f"{s}.lines.jsonl") for s in ("train","valid")}
    for name,c in configs.items():
        if any(c[k]!=base[k] for k in common) or c["completed_epochs"]!=16:
            raise ValueError(f"unmatched input, optimizer or full budget: {name}")
        if c["decision_threshold"]!=.5 or c["comparison"]!=">":
            raise ValueError("classification must use the unchanged strict >.5 rule")
        if name!="global_mean":
            cfg=c["bounded_attention"]
            expected={"attention_bce":None,"diff_region":"diff","random_region":"random"}[name]
            if cfg["k"]!=5 or cfg["region_supervision"]!=expected or cfg["region_weight"]!=(0 if expected is None else .1):
                raise ValueError("unexpected K, region loss or supervision")
            if c["parameters"]!=base["parameters"]+769:
                raise ValueError("only one Linear(768,1) attention head may be added")
    for split in annotations:
        field="training_members" if split=="train" else "validation_members"
        if [r["sample_key"] for r in annotations[split]]!=base[field]:
            raise ValueError("official members or source order changed")
    # A common loader is reused for post-selection inference, never training.
    device=core.resolve_device(device)
    loaders={}
    valid={};train={};table=[];uniform=None
    for name in names:
        p=paths[name];config=configs[name]
        valid[name]=read_jsonl(p/"valid.best-p_c.predictions.jsonl")
        actual=[(r["sample_key"],r["pair_key"],r["target"]) for r in valid[name]]
        expected=[(r["sample_key"],r["pair_key"],r["target"]) for r in annotations["valid"]]
        if actual!=expected:
            raise ValueError(f"Valid predictions are not aligned: {name}")
        history=read_jsonl(p/"best.training.jsonl")
        epochs=[r for r in history if r.get("event")=="epoch"]
        chosen=max(epochs,key=lambda r:(round(r["validation"]["p_c"],4),-r["epoch"]))
        if len(epochs)!=16 or chosen["epoch"]!=config["selected_epochs"]["p_c"]:
            raise ValueError("epoch selection did not use only Valid P-C and earlier-epoch ties")
        if name=="global_mean":
            continue
        cache=p/"train.selected.metrics.json"
        if cache.exists():
            record=json.loads(cache.read_text())
            if record["selected_epoch"]!=chosen["epoch"]:
                raise ValueError("cached Train diagnostics use another checkpoint")
            train[name]=read_jsonl(p/"train.selected.predictions.jsonl")
        else:
            cp=torch.load(p/"best-p_c.pt",map_location="cpu",weights_only=False)
            model,tokenizer=load_paired(cp,device)
            if not loaders:
                loaders={s:paired_loader(annotations[s],tokenizer,"baseline",batch_size=8) for s in ("train","valid")}
            value,pred=validate_paired(model,annotations["train"],tokenizer,"baseline",device,loader=loaders["train"])
            train[name]=pred
            write_jsonl(p/"train.selected.predictions.jsonl",pred)
            write_json(cache,dict(selected_epoch=chosen["epoch"],metrics=value))
            del model,cp
            if device.type=="cuda":torch.cuda.empty_cache()
    # Global baseline line multipliers are analytically one; do not copy another
    # model's classification probabilities into the baseline Train diagnostics.
    for split,storage in (("train",train),("valid",valid)):
        coordinates=storage["attention_bce"]
        for name in ("diff_region","random_region"):
            if [(r["sample_key"],r["line_numbers"],r["line_token_counts"]) for r in storage[name]] != [
                (r["sample_key"],r["line_numbers"],r["line_token_counts"]) for r in coordinates]:
                raise ValueError("models use different per-function physical lines or token budgets")
        if split=="train":
            storage["global_mean"]=[dict(sample_key=r["sample_key"],line_numbers=r["line_numbers"],
                line_token_counts=r["line_token_counts"],line_rates=[1.]*len(r["line_numbers"])) for r in coordinates]
        else:
            for r,c in zip(storage["global_mean"],coordinates):
                r.update(line_numbers=c["line_numbers"],line_token_counts=c["line_token_counts"],
                         line_rates=[1.]*len(c["line_numbers"]))
    cp_path=paths["diff_region"]/"best-p_c.pt"
    uniform_path=paths["diff_region"]/"valid.uniform.metrics.json"
    if uniform_path.exists():
        uniform=json.loads(uniform_path.read_text())
        if uniform["selected_epoch"]!=configs["diff_region"]["selected_epochs"]["p_c"]:
            raise ValueError("uniform intervention checkpoint differs")
        uniform_pred=read_jsonl(paths["diff_region"]/"valid.uniform.predictions.jsonl")
    else:
        model,tokenizer=load_paired(torch.load(cp_path,map_location="cpu",weights_only=False),device)
        if "valid" not in loaders:
            loaders["valid"]=paired_loader(annotations["valid"],tokenizer,"baseline",batch_size=8)
        model.force_uniform_attention=True
        value,uniform_pred=validate_paired(model,annotations["valid"],tokenizer,"baseline",device,loader=loaders["valid"])
        uniform=dict(selected_epoch=configs["diff_region"]["selected_epochs"]["p_c"],metrics=value,
                     intervention="all r_i=1; unchanged encoder/classifier; no retraining or reselection")
        write_json(uniform_path,uniform)
        write_jsonl(paths["diff_region"]/"valid.uniform.predictions.jsonl",uniform_pred)
        del model
    pure={annotations["valid"][i]["pair_key"] for i in range(0,len(annotations["valid"]),2)
          if not annotations["valid"][i]["lines_ground_truth"] and annotations["valid"][i+1]["lines_ground_truth"]}
    for name,preds in (*valid.items(),("diff_region_uniform",uniform_pred)):
        for group,selected in (("all",annotations["valid"]),("addition_only",[r for r in annotations["valid"] if r["pair_key"] in pure])):
            by_key={r["sample_key"]:r for r in preds}
            value=paired_metrics(selected,[by_key[r["sample_key"]]["probability"] for r in selected])
            table.append(dict(model=name,group=group,selected_epoch=configs["diff_region" if name=="diff_region_uniform" else name]["selected_epochs"]["p_c"],**value))
    diagnostics=[];random_comparison=[]
    masks={name:read_jsonl(paths[name]/"train.attention_regions.jsonl") for name in ("diff_region","random_region")}
    if any(len(v)!=len(annotations["train"]) for v in masks.values()):
        raise ValueError("region masks must retain every official Train function")
    for i,(a,b) in enumerate(zip(masks["diff_region"],masks["random_region"])):
        gt=set(annotations["train"][i]["lines_ground_truth"])
        if (a["sample_key"]!=b["sample_key"] or a["line_numbers"]!=b["line_numbers"] or sum(a["labels"])!=sum(b["labels"])):
            raise ValueError("random control changed visible-line counts, coordinates or availability")
        if a["labels"]!=[int(n in gt) for n in a["line_numbers"]]:
            raise ValueError("the real region masks are not the original physical Diff lines")
        random_comparison.append(dict(sample_key=a["sample_key"],positive_lines=sum(a["labels"]),
            visible_lines=len(a["labels"]),eligible=0<sum(a["labels"])<len(a["labels"]),
            identical=a["labels"]==b["labels"],overlap_lines=sum(x*y for x,y in zip(a["labels"],b["labels"]))))
    if len(random_comparison)!=8708:
        raise ValueError("random region control omitted training functions")
    for split,storage in (("train",train),("valid",valid)):
        for name in names:
            if len(storage[name])!=len(annotations[split]):
                raise ValueError("attention diagnostic members differ")
            for i,(ann,pred) in enumerate(zip(annotations[split],storage[name])):
                record=bounded_region_record(ann,pred)
                diagnostics.append(dict(model=name,split=split,region_source="original_diff",**record))
                if split=="train" and name=="random_region":
                    random_record=bounded_region_record(ann,pred,masks[name][i]["labels"])
                    if record["eligible"]!=random_record["eligible"]:
                        raise ValueError("random regions changed effective supervision availability")
                    diagnostics.append(dict(model=name,split=split,region_source="fixed_random",**random_record))
    summaries=[]
    for name,split,source in sorted({(r["model"],r["split"],r["region_source"]) for r in diagnostics}):
        for target in (None,0,1):
            rs=[r for r in diagnostics if (r["model"],r["split"],r["region_source"])==(name,split,source) and (target is None or r["target"]==target)]
            eligible=[r for r in rs if r["eligible"]]
            ratios=[r["mean_rate_ratio"] for r in eligible]
            summaries.append(dict(model=name,split=split,region_source=source,target="all" if target is None else target,
                functions=len(rs),eligible=len(eligible),empty=sum(r["region_tokens"]==0 for r in rs),
                full=sum(r["region_tokens"]==r["code_tokens"] for r in rs),
                max_token_weight_ratio=max(r["token_weight_ratio"] for r in rs),
                token_coverage_macro=mean(r["token_coverage"] for r in rs),
                token_coverage_micro=sum(r["region_tokens"] for r in rs)/sum(r["code_tokens"] for r in rs),
                mean_rate_ratio=mean(ratios) if ratios else None,median_rate_ratio=median(ratios) if ratios else None,
                ratio_at_least_two=sum(v>=2 for v in ratios)/len(ratios) if ratios else None,
                attention_enrichment=mean(r["attention_enrichment"] for r in eligible) if eligible else None))
    write_csv(root/"comparison.csv",table)
    write_csv(root/"region_statistics.csv",diagnostics)
    write_csv(root/"region_summary.csv",summaries)
    write_csv(root/"random_overlap.csv",random_comparison)
    epochs=[]
    for name,p in paths.items():
        for r in read_jsonl(p/"best.training.jsonl"):
            if r.get("event")=="epoch":
                epochs.append(dict(model=name,epoch=r["epoch"],total_loss=r["train_loss"],
                    bce=r.get("train_components",{}).get("bce",r["train_loss"]),
                    region_loss=r.get("train_components",{}).get("region_loss",0.),
                    training_seconds=r["training_seconds"],epoch_seconds=r["epoch_seconds"],
                    valid_p_c=r["validation"]["p_c"],valid_auc=r["validation"]["auc"]))
    write_csv(root/"training.csv",epochs)
    # Equal CPU RNG state and unchanged sampling code support matched shuffling;
    # this is not a claim of bit-identical CUDA floating-point training.
    states=[torch.load(p/"last.pt",map_location="cpu",weights_only=False,mmap=True)["rng"]["torch"] for p in paths.values()]
    sampling_rng_equal=all(torch.equal(s,states[0]) for s in states[1:])
    if not sampling_rng_equal:
        raise ValueError("end-of-training CPU RNG states differ; investigate sample-order matching")
    all_rows={r["model"]:r for r in table if r["group"]=="all"}
    evidence=dict(config=dict(seed=123456,epochs=16,batch_size=8,k=5,region_weight=.1,threshold=.5,comparison=">"),
        sampling_rng_equal=sampling_rng_equal,all_epochs_completed=64,optimizer_steps_per_run=17424,test_used=False,
        results=table,region_summaries=summaries,
        random_overlap=dict(functions=len(random_comparison),identical=sum(r["identical"] for r in random_comparison),
            eligible=sum(r["eligible"] for r in random_comparison),
            eligible_identical=sum(r["eligible"] and r["identical"] for r in random_comparison)),
        deltas={other:{m:all_rows["diff_region"][m]-all_rows[other][m] for m in ("p_c","auc","mcc","f1","recall","patched_fpr")}
                for other in ("global_mean","attention_bce","random_region","diff_region_uniform")})
    write_json(root/"comparison.json",evidence)
    lines=["# 补丁引导的有界注意力MIL：完整512实验","",
        "官方Train4354对/8708函数，Valid562对/1124函数；四组均从相同预训练UniXcoder初始化，Seed123456、16 Epoch、每组17424步。Test未读取。",
        "A=global_mean，B=attention_bce，C=diff_region，D=random_region。",
        "保持原512输入、有效代码Token掩码、Linear(768,1)函数头和优化器。B/C/D仅增加零初始化关注头；K=5。C/D使用固定0.1区域损失，不搜索超参数。",
        "各组仅按完整Valid P-C选epoch，四位小数严格改善；平局保留较早epoch。固定概率>.5。", "",
        "|模型|组|Epoch|Pairs|P-C|AUC|MCC|F1|漏洞Recall|修复侧FPR|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f"|{r['model']}|{r['group']}|{r['selected_epoch']}|{r['pairs']}|{r['p_c']:.4f}|{r['auc']:.4f}|{r['mcc']:.4f}|{r['f1']:.4f}|{r['recall']:.4f}|{r['patched_fpr']:.4f}|")
    lines += ["","`addition_only`按完整原始双侧Diff定义：旧侧无删除行、新侧有新增行，不把截断导致的不可见误作纯新增。没有丢弃这些函数的BCE。",
        "","## 关注量诊断（所选checkpoint、eval模式）","",
        "B/C/D的Train/Valid统计来自所选checkpoint的单函数eval预测，再对照Diff计算；Valid Diff从不传入模型。A的倍率恒为1，关注量按定义计算，没有伪造A的Train分类预测。比例只表示编辑区域关注，不代表漏洞定位准确率。",
        "|模型|Split|区域|有效函数|Token覆盖micro|区域/背景平均倍率|关注富集|",
        "|---|---|---|---:|---:|---:|---:|"]
    for r in summaries:
        if r["target"]=="all":
            lines.append(f"|{r['model']}|{r['split']}|{r['region_source']}|{r['eligible']}|{r['token_coverage_micro']:.4f}|{r['mean_rate_ratio']:.4f}|{r['attention_enrichment']:.4f}|")
    overlap=evidence["random_overlap"]
    region_by_split={r["split"]:r for r in summaries
                     if r["model"]=="diff_region" and r["target"]=="all"}
    addition_rows={r["model"]:r for r in table if r["group"]=="addition_only"}
    c,a,b,u=(all_rows[n] for n in ("diff_region","global_mean","attention_bce","diff_region_uniform"))
    lines += ["",f"随机对照保持可见正行数量及监督可用性；{overlap['identical']}/{overlap['functions']}个函数集合完全相同，其中有效区域中重合{overlap['eligible_identical']}/{overlap['eligible']}。没有强制重抽消除巧合，也没有声称Token数严格匹配。", "",
        *[f"{split}：{r['eligible']}/{r['functions']}个函数具有非平凡的可见编辑区域；空区域{r['empty']}，全Token区域{r['full']}。"
          for split,r in region_by_split.items()],
        "Train的全部函数保留BCE，只有非平凡区域参与区域损失；Valid仅在预测后统计区域，不用于训练。", "",
        "## 分类收益与解释","",
        *[f"C减{other}：P-C {d['p_c']:+.6f}，AUC {d['auc']:+.6f}，MCC {d['mcc']:+.6f}，F1 {d['f1']:+.6f}。" for other,d in evidence["deltas"].items()],"",
        ("C在此固定Seed下的Valid P-C超过A、B和D；这是该协议下的位置关联证据，不作多Seed稳定性结论，也不忽略随机对照的Token覆盖差异。"
         if all(all_rows["diff_region"]["p_c"]>all_rows[n]["p_c"] for n in ("global_mean","attention_bce","random_region"))
         else "C虽在Valid P-C超过A和B，但未超过随机D，不能将分类变化归因于真实编辑位置；本轮未验证位置监督有效，不叠加模块或搜索参数。"
         if all(all_rows["diff_region"]["p_c"]>all_rows[n]["p_c"] for n in ("global_mean","attention_bce"))
         else "C没有同时超过A和B的Valid P-C；按预定标准，本轮未验证净分类收益，按无效报告，不叠加模块或搜索参数。"),"",
        f"C相对A：漏洞Recall {a['recall']:.4f}→{c['recall']:.4f}，修复侧FPR {a['patched_fpr']:.4f}→{c['patched_fpr']:.4f}。F1提升不能代替成对判别收益。C减A的AUC为{c['auc']-a['auc']:+.4f}、P-C正确对数变化为{c['pair_counts']['p_c']-a['pair_counts']['p_c']:+}；C减B的MCC为{c['mcc']-b['mcc']:+.4f}。",
        f"纯新增组共{addition_rows['diff_region']['pairs']}对：C P-C为{addition_rows['diff_region']['p_c']:.4f}，A为{addition_rows['global_mean']['p_c']:.4f}，D为{addition_rows['random_region']['p_c']:.4f}。真实位置相对随机位置的P-C差为{addition_rows['diff_region']['p_c']-addition_rows['random_region']['p_c']:+.4f}。",
        f"区域/背景平均倍率：Train {region_by_split['train']['mean_rate_ratio']:.4f}、Valid {region_by_split['valid']['mean_rate_ratio']:.4f}；达到2倍的有效函数比例分别为{region_by_split['train']['ratio_at_least_two']:.2%}与{region_by_split['valid']['ratio_at_least_two']:.2%}。这些统计用于检查区域监督的泛化差距，不能直接作为分类失败的因果解释。", "",
        "`diff_region_uniform`是同一个C最佳checkpoint的仅推理干预：倍率全部置1，编码器与分类头不变；不重新训练、不重新选epoch。它不是独立训练的Baseline。关注量提高本身不能代替分类收益。",
        f"保留注意力较等权干预多判对{c['pair_counts']['p_c']-u['pair_counts']['p_c']}对，P-C {u['p_c']:.4f}→{c['p_c']:.4f}，AUC {u['auc']:.4f}→{c['auc']:.4f}。这说明注意力在C自身checkpoint中实际参与分类，不代表C训练方案优于独立训练的A。", "",
        "## 正确性与复现","",
        "[映射检查](mapping-check.json)、[真实GPU检查](gpu-smoke.json)、[相关测试](tests.log)、[新增恢复测试](resume-tests.log)及[命令](commands.json)保留实际记录。全量Train/Valid没有有效Token重复行成员；实现对跨行Token保留总质量，等倍率等价于原Token均值。",
        "注意力头零初始化时区域损失对encoder的梯度为0，这是链式法则的结果；头更新后真实GPU区域梯度进入encoder，见gpu-smoke.json。函数BCE从第一步正常训练。",
        "[训练损失](training.csv)、[逐函数关注统计](region_statistics.csv)、[随机重合](random_overlap.csv)与各组逐样本预测支持复核。R为空或覆盖全部有效Token只跳过区域项，全部函数仍参与BCE。",
        "四个完整任务在两张4090上并发；训练时间是并发墙钟记录，不用它推断方法单独运行的加速。权重留本地，不上传；没有读取Test、追加参数搜索或模块。",""]
    (root/"report.md").write_text("\n".join(lines),encoding="utf-8")
    return evidence


def attention_regularization_report(output_dir, data_dir, *, device="auto"):
    """Four fixed-budget runs, source-only best-checkpoint inference and epoch diagnostics."""
    from statistics import mean
    from .training import paired_loader
    from .metrics import paired_metrics
    from .losver_diagnosis import write_csv

    root=Path(output_dir)
    names=("diff_region","random_suppression","high_suppression","ema_attention")
    paths={n:root/n/"primevul_paired/unixcoder" for n in names}
    configs={n:json.loads((p/"training.metrics.json").read_text()) for n,p in paths.items()}
    base=configs[names[0]]
    if (base["train_samples"],base["valid_samples"],base["epochs"],base["optimizer_steps"])!=(8708,1124,16,17424):
        raise ValueError("regularization comparison requires four complete official 16-epoch runs")
    common=("training_config","training_members","validation_members","environment","input_coverage",
            "input_mode","model_path","parameters","function_micro_batch","window_micro_batch",
            "gradient_accumulation","bounded_attention","attention_diagnostics")
    annotations={s:read_jsonl(Path(data_dir)/"primevul_paired"/f"{s}.lines.jsonl") for s in ("train","valid")}
    if base["training_config"]["seed"]!=123456 or base["bounded_attention"]["region_supervision"]!="diff":
        raise ValueError("unchanged seed and original Diff are required")
    for split,field in (("train","training_members"),("valid","validation_members")):
        if [r["sample_key"] for r in annotations[split]]!=base[field]:
            raise ValueError("official source order or members changed")
        if len(base["attention_diagnostics"]["members"][split])!=256:
            raise ValueError("the four fixed diagnostic cohorts must each contain 256 functions")
    valid={};train={};loaders={};epochs=[];states=[];cuda_states=[]
    reference_masks=read_jsonl(paths[names[0]]/"train.attention_regions.jsonl")
    if len(reference_masks)!=8708:
        raise ValueError("every Train function must retain its original Diff mask")
    for ann,mask in zip(annotations["train"],reference_masks):
        if ann["sample_key"]!=mask["sample_key"] or mask["labels"]!=[
            int(n in ann["lines_ground_truth"]) for n in mask["line_numbers"]
        ]:
            raise ValueError("region supervision differs from original physical Diff lines")
    device=core.resolve_device(device)
    for name,p in paths.items():
        c=configs[name]
        if any(c[k]!=base[k] for k in common) or (c["completed_epochs"],c["optimizer_steps"])!=(16,17424):
            raise ValueError(f"unmatched data, initialization protocol, environment or full budget: {name}")
        if c["decision_threshold"]!=.5 or c["comparison"]!=">" or c["input_mode"]!="512":
            raise ValueError("keep original 512 input and strict probability >.5")
        reg=c.get("attention_regularization",{})
        suppression={"random_suppression":"random","high_suppression":"high"}.get(name)
        if reg.get("suppression")!=suppression or bool(reg.get("ema_components"))!=(name=="ema_attention"):
            raise ValueError("run suppression and EMA independently")
        if suppression and (reg["suppression_fraction"]!=.1 or reg["suppression_rate"]!=1):
            raise ValueError("unexpected suppression settings")
        if name=="ema_attention" and (reg["ema_decay"],reg["ema_kl_weight"],reg["ema_components"])!=(.99,.1,["encoder","line_attention"]):
            raise ValueError("EMA must include the encoder and attention head")
        if read_jsonl(p/"train.attention_regions.jsonl")!=reference_masks:
            raise ValueError("regularization changed original Diff masks")
        history=[r for r in read_jsonl(p/"best.training.jsonl") if r.get("event")=="epoch"]
        chosen=max(history,key=lambda r:(round(r["validation"]["p_c"],4),-r["epoch"]))
        if len(history)!=16 or chosen["epoch"]!=c["selected_epochs"]["p_c"]:
            raise ValueError("selection must use only Valid P-C with earlier-epoch ties")
        for h in history:
            for split,summary in h["attention_diagnostics"].items():
                epochs.append(dict(model=name,epoch=h["epoch"],split=split,**summary,
                    diagnostic_seconds=h["diagnostic_seconds"],train_loss=h["train_loss"],
                    **{k:h["train_components"].get(k) for k in (
                        "bce","region_loss","bce_normal","bce_suppressed","attention_kl")}))
        records=read_jsonl(p/"attention.diagnostics.jsonl")
        for epoch in range(1,17):
            for split in ("train","valid"):
                selected=[r for r in records if r["epoch"]==epoch and r["split"]==split]
                if [r["sample_key"] for r in selected]!=c["attention_diagnostics"]["members"][split]:
                    raise ValueError("epoch diagnostics changed fixed members")
                if any((r["consecutive_jsd"] is None)!=(epoch==1) for r in selected):
                    raise ValueError("consecutive-epoch attention state was lost")
        valid[name]=read_jsonl(p/"valid.best-p_c.predictions.jsonl")
        cache=p/"train.selected.metrics.json"
        if cache.exists():
            if json.loads(cache.read_text())["selected_epoch"]!=chosen["epoch"]:
                raise ValueError("cached Train diagnostics use a different checkpoint")
            train[name]=read_jsonl(p/"train.selected.predictions.jsonl")
        else:
            model,tokenizer=load_paired(torch.load(p/"best-p_c.pt",map_location="cpu",weights_only=False),device)
            if not loaders:
                loaders["train"]=paired_loader(annotations["train"],tokenizer,"baseline",batch_size=8)
            value,pred=validate_paired(model,annotations["train"],tokenizer,"baseline",device,loader=loaders["train"])
            train[name]=pred
            write_jsonl(p/"train.selected.predictions.jsonl",pred)
            write_json(cache,dict(selected_epoch=chosen["epoch"],metrics=value))
            del model
            if device.type=="cuda":torch.cuda.empty_cache()
        last=torch.load(p/"last.pt",map_location="cpu",weights_only=False,mmap=True)
        states.append(last["rng"]["torch"])
        cuda_states.append(last["rng"]["cuda"])
        if name=="ema_attention":
            if last["loop"]["ema_updates"]!=17424 or "attention_ema" not in last:
                raise ValueError("EMA state or per-optimizer-step updates missing")
            ema_updates=last["loop"]["ema_updates"]
            ema_parameter_deltas={k:float((last["attention_ema"][k]-last["model"][k]).norm())
                for k in ("line_attention.weight","encoder.encoder.layer.0.attention.self.query.weight")}
        if name=="random_suppression" and "suppression_rng" not in last["loop"]:
            raise ValueError("independent suppression RNG was not checkpointed")
        del last
    sampling_rng_equal=all(torch.equal(s,states[0]) for s in states[1:])
    if not sampling_rng_equal:
        raise ValueError("end-of-training sampler RNG differs; investigate diagnostic RNG preservation")
    cuda_rng_equal=all(len(s)==len(cuda_states[0]) and all(torch.equal(a,b) for a,b in zip(s,cuda_states[0]))
                       for s in cuda_states[1:])
    if not cuda_rng_equal:
        raise ValueError("online dropout RNG states differ; investigate extra stochastic forwards")
    pure={annotations["valid"][i]["pair_key"] for i in range(0,len(annotations["valid"]),2)
          if not annotations["valid"][i]["lines_ground_truth"] and annotations["valid"][i+1]["lines_ground_truth"]}
    table=[];diagnostics=[];summaries=[]
    for name in names:
        for split,storage in (("train",train),("valid",valid)):
            expected=[(r["sample_key"],r["pair_key"],r["target"]) for r in annotations[split]]
            if [(r["sample_key"],r["pair_key"],r["target"]) for r in storage[name]]!=expected:
                raise ValueError("best-checkpoint diagnostic predictions have mismatched members")
            coordinates=lambda n:[(r["line_numbers"],r["line_token_counts"]) for r in storage[n]]
            if coordinates(name)!=coordinates(names[0]):
                raise ValueError("models use different code tokens or physical-line coordinates")
            rs=[bounded_region_record(ann,pred) for ann,pred in zip(annotations[split],storage[name])]
            diagnostics.extend(dict(model=name,split=split,**r) for r in rs)
            for target in (None,0,1):
                subset=[r for r in rs if target is None or r["target"]==target]
                eligible=[r for r in subset if r["eligible"]]
                summaries.append(dict(model=name,split=split,target="all" if target is None else target,
                    functions=len(subset),eligible=len(eligible),empty=sum(r["region_tokens"]==0 for r in subset),
                    full=sum(r["region_tokens"]==r["code_tokens"] for r in subset),
                    mean_rate_ratio=mean(r["mean_rate_ratio"] for r in eligible),
                    attention_enrichment=mean(r["attention_enrichment"] for r in eligible),
                    max_token_weight_ratio=max(r["token_weight_ratio"] for r in subset),
                    token_coverage_micro=sum(r["region_tokens"] for r in subset)/sum(r["code_tokens"] for r in subset)))
        for group,rows in (("all",annotations["valid"]),("addition_only",[r for r in annotations["valid"] if r["pair_key"] in pure])):
            by_key={r["sample_key"]:r for r in valid[name]}
            metrics=paired_metrics(rows,[by_key[r["sample_key"]]["probability"] for r in rows])
            table.append(dict(model=name,group=group,epoch=configs[name]["selected_epochs"]["p_c"],**metrics))
    all_results={r["model"]:r for r in table if r["group"]=="all"}
    deltas={n:{k:all_results[n][k]-all_results[names[0]][k] for k in ("p_c","auc","mcc","f1","recall","patched_fpr")}
            for n in names[1:]}
    write_csv(root/"comparison.csv",table);write_csv(root/"region_statistics.csv",diagnostics)
    write_csv(root/"region_summary.csv",summaries);write_csv(root/"attention_epochs.csv",epochs)
    suppression_coverage={split:dict(functions=len(storage[names[0]]),
        suppressible_functions=sum(len(r["line_numbers"])>=10 for r in storage[names[0]]),
        selected_lines_per_pass=sum(len(r["line_numbers"])//10 for r in storage[names[0]]))
        for split,storage in (("train",train),("valid",valid))}
    matched=[]
    for name in names:
        for split in ("train","valid"):
            transitions=[r for r in epochs if r["model"]==name and r["split"]==split and r["epoch"]>=2]
            if len(transitions)!=15:
                raise ValueError("matched diagnostics require all 15 consecutive-epoch transitions")
            matched.append(dict(model=name,split=split,transitions=15,
                mean_consecutive_jsd=mean(r["consecutive_jsd"] for r in transitions),
                mean_token_mean_jsd=mean(r["token_mean_jsd"] for r in transitions),
                mean_rate_cv=mean(r["rate_cv"] for r in transitions)))
    write_csv(root/"attention_stability.csv",matched)
    result=dict(test_used=False,full_runs_completed=4,epochs_per_run=16,optimizer_steps_per_run=17424,
                sampling_rng_equal=sampling_rng_equal,cuda_rng_equal=cuda_rng_equal,
                ema_updates=ema_updates,ema_parameter_deltas=ema_parameter_deltas,
                suppression_coverage=suppression_coverage,results=table,
                deltas=deltas,region_summaries=summaries,epoch_diagnostics=epochs,
                matched_epoch_diagnostics=matched)
    write_json(root/"comparison.json",result)
    lines=["# 有界注意力泛化：独立正则化对照","",
        "Train4354对/8708函数，Valid562对/1124函数，512 Token，Seed123456；每组完整16轮/17424步。Test未读取。",
        "四组均从同一预训练UniXcoder初始化，K=5、原始双侧Diff、区域损失权重.1、函数分类头与优化器保持一致。",
        "只按完整Valid P-C选模（四位小数严格改善，平局保留较早epoch），概率>.5。没有搜索参数或组合正则。", "",
        "|方法|子集|Epoch|Pairs|P-C|AUC|MCC|F1|漏洞Recall|修复侧FPR|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines.extend(f"|{r['model']}|{r['group']}|{r['epoch']}|{r['pairs']}|{r['p_c']:.4f}|{r['auc']:.4f}|{r['mcc']:.4f}|{r['f1']:.4f}|{r['recall']:.4f}|{r['patched_fpr']:.4f}|" for r in table)
    lines += ["", "纯新增按完整原始Diff旧侧无删除、新侧有新增定义，保留旧侧无区域监督；不补邻接或切片标签。", "",
        "## 实现与公平性","",
        "random/high只把floor(.1*N)条有效行倍率改为1，同一次编码的正常与抑制聚合BCE取平均；原区域损失始终取正常倍率。high按detach的r_i排序，平局按物理行；random使用独立、可恢复的Python RNG。不删除Token，不缩放其余行。",
        "EMA复制encoder+attention head，eval/no_grad；每个实际optimizer.step后无条件更新.99一次。KL(reference||online)按有效行求和、函数平均，.1*KL明确加入实际累积objective。best checkpoint及正常推理仅包含在线模型。",
        "四组都重跑；旧对照只存最佳/最后权重，无法恢复逐epoch诊断。输入、环境、初始化规则、训练预算和诊断成员均已核对。EMA更新17424次，独立随机状态与EMA随last.pt保存，结束时四组CPU采样RNG一致。", "",
        "## 最佳checkpoint完整Train/Valid关注量","",
        "|方法|Split|有效区域函数|区域/背景倍率|关注富集|编辑Token比例|",
        "|---|---|---:|---:|---:|---:|"]
    lines.extend(f"|{r['model']}|{r['split']}|{r['eligible']}/{r['functions']}|{r['mean_rate_ratio']:.4f}|{r['attention_enrichment']:.4f}|{r['token_coverage_micro']:.4f}|" for r in summaries if r["target"]=="all")
    lines += ["", "这些统计来自所选在线checkpoint的eval单函数预测，再对照Diff计算；完整Train/Valid关注量不代替分类结果。",
        f"按输入结构，Train中{suppression_coverage['train']['suppressible_functions']}/{suppression_coverage['train']['functions']}函数满足N≥10、可执行非空抑制。Valid对应{suppression_coverage['valid']['suppressible_functions']}/{suppression_coverage['valid']['functions']}，仅用于长度诊断，推理不启用抑制。", "",
        "## 固定成员的逐epoch诊断","",
        "每组相同Train256/Valid256函数（各128对），只用于eval诊断，不用于选择epoch。成员见各config.json；逐函数概率、行倍率、坐标、JSD保存在attention.diagnostics.jsonl，汇总见[attention_epochs.csv](attention_epochs.csv)。",
        "JSD使用自然对数，第一轮相邻epoch JSD为null。均匀参照为n_i/Σn_j，表示每Token等权；同时报告行注意力熵和倍率CV，不能把行等权误当Token均值。诊断保存/恢复Python、NumPy、Torch与CUDA随机状态，恢复保留前一epoch分布及日志边界。", "",
        "以下汇总全部15个相邻过渡（第2–16轮），固定成员和学习率阶段对齐。最佳checkpoint的epoch不同，其单轮JSD不能直接归因于正则差异。", "",
        "|方法|Split|平均相邻JSD|平均到Token均值JSD|平均倍率CV|",
        "|---|---|---:|---:|---:|"]
    lines.extend(f"|{r['model']}|{r['split']}|{r['mean_consecutive_jsd']:.6f}|{r['mean_token_mean_jsd']:.6f}|{r['mean_rate_cv']:.6f}|" for r in matched)
    stable={r["model"]:r for r in matched if r["split"]=="valid"}
    lines += ["", f"同轮对齐的Valid平均相邻JSD相对对照变化：random {stable['random_suppression']['mean_consecutive_jsd']/stable['diff_region']['mean_consecutive_jsd']-1:+.2%}，high {stable['high_suppression']['mean_consecutive_jsd']/stable['diff_region']['mean_consecutive_jsd']-1:+.2%}，EMA {stable['ema_attention']['mean_consecutive_jsd']/stable['diff_region']['mean_consecutive_jsd']-1:+.2%}。不能以较晚最佳epoch的较小JSD直接声称抑制提高稳定性。",
        f"EMA的平均到Token均值JSD为{stable['ema_attention']['mean_token_mean_jsd']:.6f}、倍率CV为{stable['ema_attention']['mean_rate_cv']:.6f}；这些值描述注意力与等权的距离，不能代替分类指标。"]
    lines += ["", "下表列出各自最佳epoch及最终epoch的固定成员统计（相同epoch只列一次）：", "",
        "|方法|Epoch|Split|区域/背景倍率|相邻epoch JSD|到Token均值JSD|倍率CV|",
        "|---|---:|---|---:|---:|---:|---:|"]
    fmt=lambda v:"—" if v is None else f"{v:.6f}"
    lines.extend(f"|{r['model']}|{r['epoch']}|{r['split']}|{fmt(r['mean_rate_ratio'])}|{fmt(r['consecutive_jsd'])}|{fmt(r['token_mean_jsd'])}|{fmt(r['rate_cv'])}|"
                 for r in epochs if r["epoch"] in (configs[r["model"]]["selected_epochs"]["p_c"],16))
    lines += ["", "## 分类差值与结论", ""]
    lines.extend(f"{n}减当前方法：P-C {d['p_c']:+.6f}，AUC {d['auc']:+.6f}，MCC {d['mcc']:+.6f}，Recall {d['recall']:+.6f}，修复侧FPR {d['patched_fpr']:+.6f}。" for n,d in deltas.items())
    additions={r["model"]:r for r in table if r["group"]=="addition_only"}
    lines += ["", f"纯新增{additions['diff_region']['pairs']}对的P-C正确对数：对照{additions['diff_region']['pair_counts']['p_c']}，random {additions['random_suppression']['pair_counts']['p_c']}，high {additions['high_suppression']['pair_counts']['p_c']}，EMA {additions['ema_attention']['pair_counts']['p_c']}。该子集不能由整体P-C改善代替解释。"]
    lines += ["", ("三种新正则的AUC、MCC均低于对照。部分P-C上升伴随漏洞Recall与修复侧FPR下降，不能将其等同于一致的分类泛化收益。本轮停止，不追加模块或参数搜索。"
                  if all(d["auc"]<0 and d["mcc"]<0 for d in deltas.values()) else
                  "存在Valid P-C改善的配置，需结合AUC、MCC及误报变化解释；这仍是单Seed Valid对照，不以注意力稳定性宣称分类成功。"
                  if any(d["p_c"]>0 for d in deltas.values()) else
                  "三种正则均未超过当前方法的Valid P-C，本轮未验证净分类收益，按负结果停止，不追加模块或搜索参数。"),"",
        "Train倍率下降、Train/Valid差距缩小、相邻epoch JSD下降或趋于均匀，均不能单独证明分类泛化改善。新增正则只改变训练；推理不读取Diff、不依赖EMA或抑制。", "",
        "[相关测试](tests.log)、[诊断隔离](diagnostic-isolation-test.log)、[恢复配置复验](resume-config-test.log)、[真实GPU检查](gpu-smoke.json)、[正式入口smoke](training-smoke.json)、[旧对照核验](reference-check.json)及[四组命令](commands.json)保留实际记录。smoke明确不是完整实验结果。权重保留本地，不上传。训练时间包含两卡并发影响，不能据此直接宣称单独运行加速。", ""]
    (root/"report.md").write_text("\n".join(lines),encoding="utf-8")
    return result


def paired_report(output_dir, data_dir, *, reference_dir=None, localizer_regularization=False,
                  bounded_attention=False, attention_regularization=False, device="auto"):
    if attention_regularization:
        if localizer_regularization:
            raise ValueError("choose one experiment report")
        return attention_regularization_report(output_dir,data_dir,device=device)
    if bounded_attention:
        return bounded_attention_report(output_dir,data_dir,device=device)
    if (Path(output_dir) / "oof.json").exists():
        return oof_report(output_dir, data_dir)
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
