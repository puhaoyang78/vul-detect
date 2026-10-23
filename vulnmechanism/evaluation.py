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
    """Official probability threshold and line-weighted F1 checkpoint criterion."""
    import numpy as np
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )
    from .training import paired_loader, paired_forward
    from .metrics import paired_metrics

    loader = loader if loader is not None else paired_loader(rows, tokenizer, variant)
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
            for y, p in zip(batch[1], prob):
                p = p.cpu().tolist()
                ys = y[: len(p)].tolist()
                if any(v < 0 for v in ys):
                    raise ValueError(
                        "upstream localizer emitted a padded/undefined line; inspect truncation before training"
                    )
                line_y.extend(ys)
                line_p.extend(p)
                predictions.append(
                    dict(
                        sample_key=rows[at]["sample_key"],
                        line_scores=p,
                        lines_predicted=[i for i, v in enumerate(p) if v > 0.5],
                    )
                )
                at += 1
        else:
            scores.extend(prob[:, 0].cpu().tolist())
    if variant == "localizer":
        pred = np.asarray(line_p) > 0.5
        result = dict(
            f1=f1_score(line_y, pred, average="weighted", zero_division=0),
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
    model, tokenizer = core.losver_model(
        checkpoint["model_path"], checkpoint["variant"]
    )
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
    probabilities = []
    for batch in training_bar(
        paired_loader(rows, tokenizer, "localizer"), desc=f"Localize {split}"
    ):
        probabilities.extend(
            p.cpu().tolist()
            for p in paired_forward(
                model, batch, "localizer", device, with_labels=False
            )
        )
    if len(probabilities) != len(rows):
        raise ValueError("localizer prediction count differs")
    predicted = [
        dict(r, lines_predicted=[i for i, p in enumerate(v) if p > 0.5])
        for r, v in zip(rows, probabilities)
    ]
    write_jsonl(root / f"{split}.predicted.jsonl", predicted)
    write_jsonl(
        root / f"{split}.line_scores.jsonl",
        [
            dict(sample_key=r["sample_key"], line_scores=v)
            for r, v in zip(rows, probabilities)
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
    for method in ("unixcoder", "losver", "losver_oracle"):
        run = root / method
        training = json.loads((run / "training.metrics.json").read_text())
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
    write_json(
        root / "comparison.json", dict(config=LOSVER_CONFIG, data=data, results=results)
    )
    fields = [
        "method",
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
        "paper_p_c",
        "delta_paper_p_c",
    ]
    with (root / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    lines = [
        "# LOSVER：PrimeVul 原始 Paired 复现",
        "",
        "官方原始 Train/Valid/Test：4354/562/564 对；不重划分、不重采样。Seed 123456，Batch 8，512 Token。定位器40 Epoch按Valid加权行F1选模；三个分类器16 Epoch按Valid P-C选模。固定概率>0.5。",
        "",
        "| 模型 | Epoch | Test P-C | P-V | P-B | P-R | Accuracy | Precision | Recall | F1 | 论文 P-C |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if r["split"] == "test":
            lines.append(
                "| "
                + r["method"]
                + f" | {r['epoch']}"
                + " | "
                + " | ".join(
                    f"{r[k]*100:.2f}%"
                    for k in (
                        "p_c",
                        "p_v",
                        "p_b",
                        "p_r",
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "paper_p_c",
                    )
                )
                + " |"
            )
    test = {r["method"]: r for r in results if r["split"] == "test"}
    lines += [
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
        "",
        "| Split | 原始对数 | 函数数 | baseline编码相同对 | 加权检测器Token ID相同对 | 可见修改行/全部修改行 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, info in data["splits"].items():
        lines.append(
            f"| {split} | {info['pairs']} | {info['functions']} | "
            f"{info['identical_inputs']['baseline']} | {info['identical_inputs']['oracle']} | "
            f"{info['visible_modified_lines']}/{info['modified_lines']} |"
        )
    locator = json.loads((root / "losver_localizer/best-f1.json").read_text())
    lv = locator["validation"]
    lines += [
        "",
        "编码相同指Token ID相同，Oracle的真实修改行权重仍可能不同；不删除这些官方成员。跨Split交集："
        + "；".join(
            f"{name} 源码={info['exact_sources']}、baseline编码={info['encoded_inputs']['baseline']}"
            for name, info in data["overlap"].items()
        )
        + "。",
        "",
        f"定位器选定Epoch {locator['epoch']}：Valid加权行F1 {lv['f1']*100:.2f}%，"
        f"修改行Precision {lv['precision']*100:.2f}%、Recall {lv['recall']*100:.2f}%。"
        "加权行F1与修改行正类F1不同。",
        "",
        f"官方换行匹配保留：Valid物理可见行{data['splits']['valid']['visible_lines']}，"
        f"实际定位器输出行{lv['lines']}；可见正标签{data['splits']['valid']['visible_modified_lines']}，"
        f"实际消费正标签{lv['modified_lines']}。Tokenizer会将空白和换行合并，官方精确换行ID匹配可能遗漏这些边界；"
        "以上数量差不代表已量化全部错位。没有据此修改规则或使用Test调参。",
        "",
        "Oracle单独读取真实修改行；正常检测只读取单函数及定位器预测，不读测试Diff。",
        "",
        "模型和预处理忠实对照官方源码 a158905cb45b4400f481f18eeb1cb4473ccfccdc；加权检测器保留原实现的第0维风险、Softmax前padding分数0及512截断细节。基线使用官方RobertaForSequenceClassification头。",
        "",
        "环境差异：当前Torch 2.9.1/Transformers 4.57.3，官方Torch 2.4.1/Transformers 4.19.0；使用eager FP32注意力和Transformers4.19的AdamW更新。完整参数、逐Epoch日志、预测和权重分别保存在各方法目录。",
        "",
        "数据与编码重复、元数据异常及不可区分配对见data/primevul_paired/statistics.json；原始成员全部保留，不通过修改Test口径追逐论文指标。",
        "",
        "论文与源码差异：正式论文IV-E的通用检测设置为8 Epoch、按Valid Accuracy选模；官方PrimeVul专用脚本默认16 Epoch，Test加载最佳Valid P-C权重。本轮采用后者，未根据Test切换权重或选择协议。",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    return results
