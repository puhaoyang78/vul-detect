"""Test robustness, fixed-checkpoint transfer and saved experiment reports."""

from __future__ import annotations
from argparse import Namespace
import csv
import json
from pathlib import Path
import torch
from .artifacts import write_json
from .data import DATASETS, balanced_subset, class_counts, read_jsonl, test_rows, validation_rows, write_jsonl
from .metrics import decision_boundary, metrics
from . import model as core
from .training import METHODS, GENERALIZATION_METHODS, prepare_materials


def _evaluate_clean(rows, run, name, checkpoint, args):
    output = run / f"{name}.metrics.json"
    predictions = run / f"{name}.predictions.jsonl"
    threshold = checkpoint["decision_threshold"]
    by_key = {}
    previous, cached = None, []
    if output.exists():
        previous = json.loads(output.read_text())
        if previous["selected_epoch"] != checkpoint["selected_epoch"]:
            raise ValueError("cached evaluation checkpoint selection differs")
        cached = read_jsonl(predictions)
        labels = {r["sample_key"]: r["label"] for r in rows}
        by_key = {r["sample_key"]: r for r in cached}
        if len(by_key) != len(cached) or any(
            r["sample_key"] not in labels or labels[r["sample_key"]] != r["label"]
            or r["threshold"] != threshold for r in cached
        ):
            raise ValueError("cached evaluation members, labels or threshold differ")
    missing = [r for r in rows if r["sample_key"] not in by_key]
    if missing:
        print(f"{run.name}/{name}: reuse {len(by_key)}, predict {len(missing)} members", flush=True)
        device = core.resolve_device(args.device)
        builder = core.InputBuilder(
            core.AutoTokenizer.from_pretrained(args.model),
            source_max_length=checkpoint["source_max_length"],
        )
        model = core.load_model(checkpoint, device, model_path=args.model)
        logits = core.predict_logits(model, missing, builder, device=device, batch_size=args.batch_size)
        for row, logit, score in zip(missing, logits, logits.sigmoid().tolist()):
            by_key[row["sample_key"]] = dict(
                sample_key=row["sample_key"], label=row["label"], logit=float(logit), score=score,
                prediction=int(score >= decision_boundary(threshold)), threshold=threshold,
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
            if [r["sample_key"] for r in cached] != [r["sample_key"] for r in evaluation_rows]:
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
                    {k: v for k, v in c.items() if k != "raw_source"} for c in candidates[:2]
                ]
            entries.append(dict(sample_key=row["sample_key"], groups=groups, reasons=reasons))
            if (index + 1) % 100 == 0:
                print(f"Test {name} candidates {index+1}/{len(evaluation_rows)}", flush=True)
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
            metrics([r["label"] for r in selected], [r["score"] for r in selected], threshold)
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
            count=attacked, denominator=original_correct, rate=rate(attacked, original_correct)
        ),
        vulnerability_escape=dict(
            count=escape,
            denominator=sum(r["label"] == 1 and r["prediction"] == 1 for r in eligible),
            rate=rate(escape, sum(r["label"] == 1 and r["prediction"] == 1 for r in eligible)),
        ),
        benign_false_alarm=dict(
            count=false_alarm,
            denominator=sum(r["label"] == 0 and r["prediction"] == 0 for r in eligible),
            rate=rate(
                false_alarm, sum(r["label"] == 0 and r["prediction"] == 0 for r in eligible)
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
        calls=0, function_forwards=0, encoder_sequences=0, encoder_token_slots=0, seconds=0.0
    )

    def score(batch):
        started = time.monotonic()
        values = []
        with torch.no_grad(), core.auxiliary_mode(model):
            for at in range(0, len(batch), args.batch_size):
                grouped = builder.codebert_batch(
                    batch[at : at + args.batch_size], device=core.resolve_device(args.device)
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
                    index.append((row["sample_key"], kind, ci, candidate["strict_semantic"]))
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
                {k: [v for v in vs if v["strict_semantic"]] for k, vs in combined.items()},
                cp["decision_threshold"],
            ),
            input_only_stress=_robustness_summary(
                originals,
                {k: [v for v in vs if not v["strict_semantic"]] for k, vs in combined.items()},
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
                            groups={k: g.get(row["sample_key"], []) for k, g in views.items()},
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
                        else {k: v for k, v in entry["selected"].items() if k != "raw_source"}
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
                prediction=int(torch.sigmoid(torch.tensor(c["logit"])).item() >= boundary),
                strict_semantic=c["strict_semantic"],
            )
            for c in entry["visited"]
        ]
        group.sort(
            key=lambda c: float(np.logaddexp(0, c["logit"]) - by[key]["label"] * c["logit"]),
            reverse=True,
        )
        groups[key] = group
        strict[key] = [c for c in group if c["strict_semantic"]]
        stress[key] = [c for c in group if not c["strict_semantic"]]
    report["adaptive"] = dict(
        all=_robustness_summary(originals, groups, cp["decision_threshold"]),
        strict_semantic=_robustness_summary(originals, strict, cp["decision_threshold"]),
        input_only_stress=_robustness_summary(originals, stress, cp["decision_threshold"]),
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
    data_dir, output_dir, dataset, methods=METHODS, *, device="auto", robust=True,
    valid_limit=None, test_limit=None,
):
    if dataset not in (*DATASETS, "reveal"):
        raise ValueError(f"unknown dataset: {dataset}")
    root = Path(output_dir) / dataset
    valid = balanced_subset(validation_rows(Path(data_dir) / dataset), valid_limit)
    own_test = balanced_subset(test_rows(Path(data_dir) / dataset), test_limit)
    materials = prepare_materials(data_dir, output_dir, dataset) if robust else None
    for method in methods:
        run = root / method
        if not (run / "training.metrics.json").exists():
            raise ValueError(f"{dataset}/{method}: training must finish before evaluation")
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
        _evaluate_clean(valid, run, "valid", checkpoint, args)
        _evaluate_clean(own_test, run, "test", checkpoint, args)
        if robust:
            if not all(
                (root / f"{name}.candidates.jsonl").exists() for name in ("fixed", "heldout")
            ):
                prepare_candidates(
                    data_dir, output_dir, checkpoint["model_path"], dataset, test_limit=test_limit
                )
            _robustness_evaluation(
                own_test, run, run / "best.pt", checkpoint, root, materials, args
            )
        del checkpoint


def evaluate_transfer(data_dir, output_dir, datasets, *, methods=("bce",), device="auto",
                      train_data_dir=None):
    """Compare fixed source models on identical full and exact-overlap-free Test cohorts."""
    from .static_data import DATASETS as STATIC_DATASETS
    if (len(datasets) < 2 or len(set(datasets)) != len(datasets)
            or any(d not in (*DATASETS, *STATIC_DATASETS) for d in datasets)):
        raise ValueError("at least two distinct supported datasets required")
    if not methods or len(set(methods)) != len(methods) or any(m not in GENERALIZATION_METHODS for m in methods):
        raise ValueError("distinct supported generalization methods required")
    datasets = tuple(sorted(datasets))
    root = Path(output_dir) / ("transfer_" + "_".join(datasets))
    root.mkdir(parents=True, exist_ok=True)
    tests = {d: test_rows(Path(data_dir) / d) for d in datasets}
    train_data_dir = Path(data_dir if train_data_dir is None else train_data_dir)
    seen = {}
    training_members = {}
    for dataset in datasets:
        for split in ("train", "valid"):
            values = (
                read_jsonl(train_data_dir / dataset / "train.jsonl")
                if split == "train" else validation_rows(train_data_dir / dataset)
            )
            seen[f"{dataset}:{split}"] = {r["raw_source"] for r in values}
            training_members[f"{dataset}:{split}"] = class_counts(values)
    independent, cohorts = {}, {}
    for dataset, rows in tests.items():
        overlap = {
            r["sample_key"]: [name for name, sources in seen.items() if r["raw_source"] in sources]
            for r in rows
        }
        independent[dataset] = [r for r in rows if not overlap[r["sample_key"]]]
        if {r["label"] for r in independent[dataset]} != {0, 1}:
            raise ValueError(f"{dataset}: independent Test needs both classes")
        cohorts[dataset] = dict(
            established=class_counts(rows), independent=class_counts(independent[dataset]),
            excluded=class_counts([r for r in rows if overlap[r["sample_key"]]]),
            exact_overlap_by_split={
                name: class_counts([r for r in rows if name in overlap[r["sample_key"]]])
                for name in seen
            },
        )
        members = [dict(
            sample_key=r["sample_key"], label=r["label"], project=r["project"],
            source_file=r.get("source_file", ""),
            source_row=r["source_row"],
            prepared_file=str(Path(data_dir) / dataset / "test.jsonl"),
            independent=not overlap[r["sample_key"]],
            exact_overlap=";".join(overlap[r["sample_key"]]),
        ) for r in rows]
        with (root / f"{dataset}.members.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(members[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(members)
    core.seed_everything(42)
    scores, training = {}, {}
    for source, method in ((source, method) for source in datasets for method in methods):
        run = Path(output_dir) / source / method
        training.setdefault(source, {})[method] = json.loads((run / "training.metrics.json").read_text())
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        if checkpoint["dataset"] != source or checkpoint["method"] != method:
            raise ValueError("checkpoint source dataset/method differs")
        args = Namespace(
            model=checkpoint["model_path"], device=device,
            batch_size=checkpoint["training_config"]["batch_size"],
        )
        for target in datasets:
            name = "test" if source == target else f"{target}.test"
            print(f"Fixed {method}: {source} -> {target}, {len(tests[target])} Test samples", flush=True)
            result = _evaluate_clean(tests[target], run, name, checkpoint, args)
            predictions = read_jsonl(run / f"{name}.predictions.jsonl")
            threshold = checkpoint["decision_threshold"]
            if result["selected_epoch"] != checkpoint["selected_epoch"] or any(
                r["threshold"] != threshold for r in predictions
            ):
                raise ValueError("cached transfer checkpoint selection differs")
            recomputed = metrics(
                [r["label"] for r in predictions], [r["score"] for r in predictions], threshold
            )
            if recomputed != result["metrics"]:
                raise ValueError("cached transfer metrics differ")
            scores[method, source, target] = (
                {r["sample_key"]: r["score"] for r in predictions}, threshold,
            )
        del checkpoint
    results = []
    metric_names = ("accuracy", "f1", "mcc", "auc", "auprc")
    for view, members in (("established", tests), ("independent", independent)):
        values = {}
        for method, source in ((method, source) for method in methods for source in datasets):
            for target in datasets:
                by_key, threshold = scores[method, source, target]
                rows = members[target]
                values[method, source, target] = metrics(
                    [r["label"] for r in rows], [by_key[r["sample_key"]] for r in rows], threshold
                )
        for (method, source, target), value in values.items():
            results.append(dict(
                view=view, training_dataset=source, test_dataset=target, method=method,
                selected_epoch=training[source][method]["selected_epoch"], threshold=value["threshold"],
                **class_counts(members[target]), **{k: value[k] for k in metric_names},
                **{k: value[k] for k in ("tp", "tn", "fp", "fn")},
                **{f"delta_source_{k}": value[k] - values[method, source, source][k] for k in metric_names},
                **{f"delta_target_{k}": value[k] - values[method, target, target][k] for k in metric_names},
                **{f"delta_bce_{k}": value[k] - values["bce", source, target][k]
                   if "bce" in methods else None for k in metric_names},
            ))
    write_json(root / "results.json", dict(
        cohorts=cohorts, training=training, training_members=training_members, results=results,
        protocol=dict(
            seed=42, methods=list(methods), selection="source-Valid checkpoint and threshold fixed",
            established="full original balanced Test members, without resampling; exclude own Train/Valid exact overlap only",
            independent="exclude exact raw-source overlap with any participating dataset's actual Train or model-selection Valid; same target members for all models; no other datasets, project/CVE or near-duplicate filtering; no resampling",
            delta_source="target Test metric minus the same model's own Test metric",
            delta_target="transfer metric minus the same method trained on the target dataset, on exactly the same target Test",
            delta_bce="method metric minus BCE with the same source checkpoint dataset and target Test members",
        ),
    ))
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    return dict(datasets=list(datasets), models=len(datasets) * len(methods), rows=len(results), output=str(root))


def report(output_dir, data_dir):
    root = Path(output_dir)
    rows, training_runs = [], {}
    metric_names = ("accuracy", "precision", "recall", "f1", "mcc", "auc", "auprc")
    for dataset in ("primevul", "megavul"):
        target_dataset = next(d for d in ("primevul", "megavul") if d != dataset)
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
                    dict(common, dataset=dataset, split=split, view="clean", **clean[split])
                )
            robustness = json.loads((run / "robustness.metrics.json").read_text())
            for setting in ("fixed", "heldout", "adaptive"):
                value = robustness[setting if setting == "adaptive" else setting + "_overall"]
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
            target = json.loads((run / f"{target_dataset}.test.metrics.json").read_text())[
                "metrics"
            ]
            differences = {
                f"delta_{key}": (
                    target[key] - clean["test"][key]
                    if target[key] is not None and clean["test"][key] is not None
                    else None
                )
                for key in metric_names
            }
            rows.append(
                dict(
                    common,
                    dataset=target_dataset,
                    split="test",
                    view="transfer",
                    **target,
                    **differences,
                )
            )
    data = {
        d: json.loads((Path(data_dir) / d / "statistics.json").read_text()) for d in ("primevul", "megavul")
    }
    write_json(root / "comparison.json", dict(data=data, training=training_runs, results=rows))
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
        *(f"delta_{key}" for key in metric_names),
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
