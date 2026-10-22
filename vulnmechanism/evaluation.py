"""Test robustness, fixed-checkpoint transfer and saved experiment reports."""

from __future__ import annotations
from argparse import Namespace
import csv
import json
from pathlib import Path
import torch
from .artifacts import write_json
from .data import DATASETS, balanced_subset, class_counts, read_jsonl, test_rows, validation_rows, write_jsonl
from .metrics import decision_boundary, metrics, paired_metrics
from . import model as core
from .training import METHODS, GENERALIZATION_METHODS, prepare_materials


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
            r["sample_key"] not in labels or labels[r["sample_key"]] != r["label"]
            or r["threshold"] != threshold for r in cached
        ):
            raise ValueError("cached evaluation members, labels or threshold differ")
    missing = [r for r in rows if r["sample_key"] not in by_key]
    if missing:
        if logits is None:
            print(f"{run.name}/{name}: reuse {len(by_key)}, predict {len(missing)} members", flush=True)
            device = core.resolve_device(args.device)
            builder = core.InputBuilder(
                core.AutoTokenizer.from_pretrained(args.model, trim_offsets=not checkpoint.get("input_protocol", {}).get("full_source", False)),
                source_max_length=checkpoint["source_max_length"],
                full_source=checkpoint.get("input_protocol", {}).get("full_source", False),
            )
            model = core.load_model(checkpoint, device, model_path=args.model)
            logits = core.predict_logits(model, missing, builder, device=device, batch_size=args.batch_size)
        elif len(logits) != len(rows):
            raise ValueError("provided evaluation logits must cover the entire cohort")
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
    if checkpoint.get("paired") and all("pair_key" in r for r in rows):
        if threshold != 0.5:
            raise ValueError("paired evaluation requires fixed threshold 0.5")
        pair_args = ({} if checkpoint.get("line_mil") else
                     dict(margin=checkpoint["paired"]["margin"], gap=checkpoint["paired"]["gap"]))
        result["metrics"].update(paired_metrics(rows, logits.tolist(), **pair_args))
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
    splits=("valid", "test"), single_function_test=False,
):
    if dataset not in (*DATASETS, "reveal", "primevul_paired"):
        raise ValueError(f"unknown dataset: {dataset}")
    if not splits or any(s not in ("valid", "test") for s in splits):
        raise ValueError("evaluation splits must be valid/test")
    if dataset == "primevul_paired" and (robust or valid_limit is not None or test_limit is not None):
        raise ValueError("paired evaluation preserves whole official pairs and uses Clean only")
    if single_function_test and (dataset != "primevul_paired" or "test" not in splits):
        raise ValueError("single-function Test is only available with final paired Test evaluation")
    if robust and "test" not in splits:
        raise ValueError("robustness evaluation requires Test")
    root = Path(output_dir) / dataset
    valid = (balanced_subset(validation_rows(Path(data_dir) / dataset), valid_limit)
             if "valid" in splits and dataset != "primevul_paired" else None)
    own_test = (balanced_subset(test_rows(Path(data_dir) / dataset), test_limit)
                if "test" in splits and dataset != "primevul_paired" else None)
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
        if dataset == "primevul_paired" and not checkpoint.get("line_mil"):
            valid = validation_rows(Path(data_dir) / dataset) if "valid" in splits else None
            own_test = test_rows(Path(data_dir) / dataset) if "test" in splits else None
        if checkpoint.get("line_mil"):
            from .pair_representation import representation_data
            if "test" in splits:
                raise ValueError("repair representation pilot is Valid-only; Test is not used for development")
            builder = core.InputBuilder(core.AutoTokenizer.from_pretrained(args.model, trim_offsets=False), full_source=True)
            _, valid, auxiliary, _, _ = representation_data(
                checkpoint["pair_raw_root"], builder,
                limit=checkpoint["paired"]["train_pair_limit"],
                seed=42,
                context=checkpoint["line_mil"]["context_lines"])
            _evaluate_clean(auxiliary, run, "paired.valid", checkpoint, args)
        if valid is not None:
            _evaluate_clean(valid, run, "valid", checkpoint, args)
        if own_test is not None:
            _evaluate_clean(own_test, run, "test", checkpoint, args)
        if single_function_test:
            _evaluate_clean(test_rows(Path(data_dir) / "primevul"), run, "primevul.test", checkpoint, args)
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


def paired_report(output_dir, data_dir):
    if (Path(output_dir) / "primevul_paired/mil_patch_rank/training.metrics.json").exists():
        from .pair_representation import report as representation_report
        return representation_report(output_dir)
    """Summarize saved Valid evidence before deciding whether full training is warranted."""
    from .training import PAIRED_METHODS
    from .data import paired_groups
    root = Path(output_dir)
    data = json.loads((Path(data_dir) / "primevul_paired/statistics.json").read_text())
    names = ("accuracy", "strict_pairwise_accuracy", "pairwise_ranking_accuracy", "auc", "mcc", "f1",
             "logit_gap", "mean_abs_logit", "boundary_violation_rate", "gap_violation_rate",
             "indistinguishable_pairs", "bce")
    rows, configs, reloaded = [], {}, {}
    for method in PAIRED_METHODS:
        run = root / "primevul_paired" / method
        if not (run / "training.metrics.json").exists():
            continue
        configs[method] = json.loads((run / "training.metrics.json").read_text())
        history = read_jsonl(run / "best.training.jsonl")
        selected = next(r["validation"] for r in history if r["event"] == "epoch" and
                        r["epoch"] == configs[method]["selected_epoch"])
        for split in ("valid", "test", "primevul.test"):
            path = run / f"{split}.metrics.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            values = result["metrics"]
            if split == "valid":
                reloaded[method] = {k: dict(training=selected[k], reloaded=values[k],
                                            difference=values[k] - selected[k])
                                     for k in ("strict_pairwise_accuracy", "pairwise_ranking_accuracy", "auc", "mcc", "f1")}
            rows.append(dict(method=method, split=split, selected_epoch=result["selected_epoch"],
                             threshold=values["threshold"], samples=values["samples"],
                             pairs=values.get("pairs"), **{k: values.get(k, result.get(k)) for k in names}))
    valid = {r["method"]: r for r in rows if r["split"] == "valid"}
    if not {"bce", "pair_joint"}.issubset(valid):
        raise ValueError("BCE and Joint require completed Valid predictions")
    budgets = {(c["train_samples"], c["valid_samples"], c["optimizer_steps"],
                json.dumps(c["training_config"], sort_keys=True)) for c in configs.values()}
    if len(budgets) != 1 or any(r["threshold"] != 0.5 for r in rows):
        raise ValueError("paired controls must share data budgets and threshold 0.5")
    baseline, joint = valid["bce"], valid["pair_joint"]
    members = read_jsonl(Path(data_dir) / "primevul_paired/valid.jsonl")
    pairs = paired_groups(members)
    statuses = {}
    for method in valid:
        predictions = read_jsonl(root / "primevul_paired" / method / "valid.predictions.jsonl")
        if [(r["sample_key"], r["label"]) for r in predictions] != [(r["sample_key"], r["label"]) for r in members]:
            raise ValueError("all pair methods must evaluate the same ordered official Valid members")
        by_key = {r["sample_key"]: r for r in predictions}
        statuses[method] = [all(by_key[r["sample_key"]]["prediction"] == r["label"] for r in pair) for pair in pairs]
    changes = {method: dict(
        fixed_pairs=sum(not a and b for a, b in zip(statuses["bce"], own)),
        regressed_pairs=sum(a and not b for a, b in zip(statuses["bce"], own)),
        common_wrong_pairs=sum(not a and not b for a, b in zip(statuses["bce"], own)))
        for method, own in statuses.items()}
    gains = {k: joint[k] - baseline[k] for k in ("strict_pairwise_accuracy", "auc", "mcc", "f1", "logit_gap", "mean_abs_logit")}
    passed = gains["strict_pairwise_accuracy"] >= 0.02 and gains["auc"] >= 0 and gains["mcc"] >= 0
    gate = dict(passed=passed, rule="Joint minus BCE: Strict Pairwise Accuracy >= +0.02, AUC >= 0, MCC >= 0; fixed before pilot training; Test never used", differences=gains)
    write_json(root / "results.json", dict(data=data, training=configs, results=rows, valid_gate=gate,
                                         valid_pair_changes=changes, training_validation_comparison=reloaded))
    with (root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    lines = ["# PrimeVul 官方漏洞—修复对判别学习", "",
             "CodeBERT、2048逻辑输入、原生滑动窗口及共享线性分类头不变。单函数推理；固定Sigmoid阈值0.5，不搜索阈值。", "",
             "BCE为两函数BCE的平均；Ranking=logsumexp(0, Δ−z_v+z_p)，Boundary=logsumexp(0,m−z_v,m+z_p)，Joint=logsumexp(0,m−z_v,m+z_p,Δ−z_v+z_p)。三组约束统一λ=0.2、m=0.2、Δ=0.6。按配对平均。", "",
             "官方配对按相邻互补标签读取，不用项目/提交元数据异常删对。重复的上游idx用原始行位置区分并保留；源idx另存。Train与任意Valid/Test编码输入相同的整对不进入训练。对内不可区分的Train配对保留相同BCE成员，但不施加任何成对约束。Valid/Test原始成员全部保留。", "",
             "选模：仅Clean Valid，Strict Pairwise Accuracy/MCC/AUC/F1顺序，完全相同保留早轮。严格配对正确要求漏洞预测为1、修复预测为0；Ranking要求z_v>z_p，平局不计正确；已核实编码相同的配对视为排序平局，保留在分母中，不把浮点归约噪声当成功。训练日志的原始浮点Ranking值原样保留；最终指标从已保存预测统一重算，此修正不影响不使用Ranking的选模。边界违反率表示z_v<m或z_p>−m，不等同于分类错误。", "",
             "| Split | 原始配对 | 使用配对 | 对内不可区分 |", "|---|---:|---:|---:|"]
    for split in ("train", "valid", "test"):
        item = data["final"][split]
        lines.append(f"| {split} | {data['native_pairs'][split]} | {item['pairs']} | {item['indistinguishable_pairs']} |")
    lines += ["", f"Train泄漏排除{len(data['removed_training_pairs'])}对；元数据异常{len(data['metadata_anomalies'])}对。原始跨Split重复及上游idx重复详见results.json。Test这里只作数据核验，是否模型评价由后面的结果行明确区分。", "",
              "| 方法 | Split | Epoch | N对 | Strict Pair Acc | Ranking Acc | AUC | MCC | F1 | Logit Gap | 平均绝对Logit | 边界违反率 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        values = [r[k] for k in ("strict_pairwise_accuracy", "pairwise_ranking_accuracy", "auc", "mcc", "f1", "logit_gap", "mean_abs_logit", "boundary_violation_rate")]
        lines.append(f"| {r['method']} | {r['split']} | {r['selected_epoch']} | {r['pairs'] or '—'} | " + " | ".join("—" if v is None else f"{v:.4f}" for v in values) + " |")
    lines += ["", "完整指标、逐样本Logit、混淆计数和成对可区分子集指标保存在各方法目录。AUC和MCC/F1同时用于检查分类收益，不能仅依据Logit Gap或绝对Logit扩大判定有效。", "",
              "## 预设Valid门槛", "", gate["rule"], "",
              f"Joint−BCE：Strict {gains['strict_pairwise_accuracy']:+.4f}，AUC {gains['auc']:+.4f}，MCC {gains['mcc']:+.4f}，F1 {gains['f1']:+.4f}。", "",
              "门槛通过，可按预设协议进入完整Train的BCE/Joint对照及一次最终Test评价。" if passed else "门槛未通过，停止全量训练；不根据Test调整模型或损失。", "",
              "单Seed、固定512对试验用于检验最小假设，不证明跨种子稳定性。若目录中没有test指标，说明没有进行Test推理。", ""]
    lines += ["| 方法 | BCE错误→正确对 | BCE正确→错误对 | 共同错误对 |", "|---|---:|---:|---:|"]
    lines += [f"| {method} | {c['fixed_pairs']} | {c['regressed_pairs']} | {c['common_wrong_pairs']} |" for method, c in changes.items()]
    lines += ["", f"Joint的Logit Gap相对BCE改变{gains['logit_gap']:+.6f}，平均绝对Logit改变{gains['mean_abs_logit']:+.6f}。"
              f"但严格配对正确数为{round(joint['strict_pairwise_accuracy'] * len(pairs))}/{len(pairs)}，"
              f"BCE为{round(baseline['strict_pairwise_accuracy'] * len(pairs))}/{len(pairs)}。"
              "分数差或尺度扩大不能替代固定阈值下的分类改善；方法间也可能同时修正和损伤不同配对，不能将其直接解释成单一标量缩放。", ""]
    auc_drift = max(abs(v["auc"]["difference"]) for v in reloaded.values())
    lines += [f"训练记录与真实重载的AUC最大差为{auc_drift:.8f}；具体差异保存在training_validation_comparison。"
              "最终表使用已保存的重载预测。编码相同的函数仍可能因浮点归约产生微小分数差，排序指标按已核实的输入等价关系处理平局。", ""]
    (root / "report.md").write_text("\n".join(lines))
    return dict(output=str(root), rows=len(rows), valid_gate=gate)


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
