"""Small static-relation experiment using the existing BCE trainer and metrics."""

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .artifacts import write_json
from .data import read_jsonl, write_jsonl
from .metrics import metrics
from .model import AutoTokenizer, InputBuilder, load_model, predict_logits, resolve_device
from .static_data import DATASETS
from .training import train

MODES = ("source", "sequence", "shuffled", "real")


def run(args):
    torch.set_num_threads(4)
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    if args.stage == "prepare":
        from .static_data import prepare
        prepare(args.raw_root, args.data_dir, args.data_dir / "indices", datasets)
    elif args.stage == "benchmark":
        if args.dataset not in ("all", "devign"):
            raise ValueError("the fixed parser benchmark uses upstream Devign Train")
        from .static_analysis import benchmark
        benchmark(args.raw_root, args.data_dir / "indices", args.output_dir)
    elif args.stage == "extract":
        from .static_analysis import prepare_relations
        prepare_relations(args.data_dir, args.output_dir, datasets, cache_root=args.cache_root)
    elif args.stage == "audit":
        from .static_analysis import audit_patterns
        audit_patterns(args.data_dir, args.output_dir, datasets, model_path=args.model)
    elif args.stage == "transfer":
        from .evaluation import evaluate_transfer
        for dataset in datasets:
            config = json.loads((args.output_dir / "source" / dataset / "bce/training.metrics.json").read_text())
            if config["static_relations"] is not None:
                raise ValueError("this transfer experiment uses Source-only BCE checkpoints")
        result = evaluate_transfer(args.data_dir, args.output_dir / "source", datasets,
                                   train_data_dir=args.data_dir / "pilot", device=args.device)
        from .static_data import transfer_overlap
        transfer_overlap(args.data_dir, result["output"], datasets)
        transfer_report(result["output"], args.output_dir / "source")
    elif args.stage == "train":
        for dataset in datasets:
            for mode in args.modes:
                train(args.data_dir / "pilot", args.output_dir / mode, args.model, dataset, "bce",
                      device=args.device, relation_mode="none" if mode == "source" else mode)
    elif args.stage == "report":
        if args.dataset != "all":
            raise ValueError("the unified report requires all three datasets")
        report(args.data_dir, args.output_dir, device=args.device)


def transfer_report(output_dir, runs_dir):
    """Summarize fixed BCE transfer and the predeclared near-free sensitivity view."""
    root = Path(output_dir)
    result = json.loads((root / "results.json").read_text())
    overlap = json.loads((root / "overlap.json").read_text())
    datasets = tuple(result["cohorts"])
    excluded = {p["target_key"] for p in overlap["near_pairs"]}
    members = {}
    for target in datasets:
        with (root / f"{target}.members.csv").open() as handle:
            members[target] = [r for r in csv.DictReader(handle)
                               if r["independent"] == "True" and r["sample_key"] not in excluded]
    names = ("accuracy", "f1", "mcc", "auc", "auprc")
    thresholds = {r["training_dataset"]: r["threshold"] for r in result["results"]}
    values = {}
    missing = {d: len(rows) for d, rows in members.items()
               if {int(r["label"]) for r in rows} != {0, 1}}
    if not missing:
        for source in datasets:
            for target in datasets:
                name = "test" if source == target else f"{target}.test"
                records = read_jsonl(Path(runs_dir) / source / "bce" / f"{name}.predictions.jsonl")
                scores = {r["sample_key"]: r for r in records}
                rows = members[target]
                if any(scores[r["sample_key"]]["label"] != int(r["label"]) for r in rows):
                    raise ValueError("near-free labels differ from saved predictions")
                values[source, target] = metrics([int(r["label"]) for r in rows],
                                                 [scores[r["sample_key"]]["score"] for r in rows],
                                                 thresholds[source])
    rows = [r for r in result["results"] if r["view"] != "near_free"]
    for (source, target), value in values.items():
        row = next(r.copy() for r in rows if r["view"] == "independent" and
                   r["training_dataset"] == source and r["test_dataset"] == target)
        row.update(view="near_free", samples=len(members[target]),
                   positive=sum(int(r["label"]) for r in members[target]))
        row["negative"] = row["samples"] - row["positive"]
        row.update({k: value[k] for k in (*names, "tp", "tn", "fp", "fn")})
        row.update({f"delta_source_{k}": value[k] - values[source, source][k] for k in names})
        row.update({f"delta_target_{k}": value[k] - values[target, target][k] for k in names})
        rows.append(row)
    result.update(results=rows, near_free_skipped=missing, overlap_audit="overlap.json")
    result["protocol"]["near_free"] = (
        "Independent cohort minus any participating actual Train/Valid token-5-gram Jaccard >= .90; "
        "fixed before inspecting model results, no resampling or threshold adjustment; "
        "all models share each target cohort; shared commits recorded but not excluded")
    write_json(root / "results.json", result)
    with (root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    lines = ["# " + " / ".join(datasets) + "：Source-only BCE 跨域评价", "",
             "使用已完成训练的源库 CodeBERT BCE checkpoint，Seed42；实际训练规模见下表。评价只使用源 Valid 已选权重和阈值，没有目标校准、源码增强或静态关系输入。", "",
             "主结果 independent 排除目标 Test 与参与实验的两侧实际 Train/Valid 的原始源码精确重叠；同一目标队列供所有模型共用，不重新平衡。established 保存原始 Test 参考。near_free 是预先定义的近重复排除敏感性检查。", "",
             "| 目标库 | 原始 Test | 精确排除后：正/负 | 再排除近重复：正/负 |", "|---|---:|---:|---:|"]
    for target in datasets:
        cohort = result["cohorts"][target]
        valid = cohort["independent"]
        near = members[target]
        positive = sum(int(r["label"]) for r in near)
        lines.append(f"| {target} | {cohort['established']['samples']} | {valid['positive']}/{valid['negative']} | {positive}/{len(near)-positive} |")
    lines += ["", "| 源模型 | 实际 Train / Valid | 选定 Epoch | 固定阈值 |", "|---|---:|---:|---:|"]
    for source in datasets:
        chosen = next(r for r in rows if r["training_dataset"] == source)
        train = result["training_members"][f"{source}:train"]["samples"]
        valid = result["training_members"][f"{source}:valid"]["samples"]
        lines.append(f"| {source} | {train}/{valid} | {chosen['selected_epoch']} | {chosen['threshold']:.2f} |")
    for view in ("independent", "near_free"):
        lines += ["", f"## {view}", "", "| 训练→Test | N | Accuracy | F1 | MCC | AUC | AUPRC | Δ自身Test AUC | Δ目标库BCE AUC |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            if row["view"] == view:
                lines.append(f"| {row['training_dataset']}→{row['test_dataset']} | {row['samples']} | " +
                             " | ".join(f"{row[k]:.4f}" for k in names) +
                             f" | {row['delta_source_auc']:+.4f} | {row['delta_target_auc']:+.4f} |")
    conflicts = sum(p["source_label"] != p["target_label"] for p in overlap["near_pairs"])
    lines += ["", "## 独立性与解释", "",
              f"Jaccard≥0.90 共 {len(overlap['near_pairs'])} 对；可用项目+提交元数据共 {len(overlap['shared_commit_pairs'])} 对。具体 sample_key、标签分歧及源码是否精确一致见 overlap.json。ReVeal 没有可靠提交/CVE，不能据此证明不存在共享漏洞记录。", "",
              f"近重复对中有 {conflicts} 对标签不一致；这只表示上游标注分歧，不判定哪方正确，也不把共享提交自动当成同一漏洞位置。", "",
              "Δ自身Test跨越了不同评价人群；Δ目标库BCE在同一目标 Test 上比较。跨域 AUC 高于源库自身 AUC，并不等于迁移没有损失。MCC/F1 还受固定阈值及域间分数尺度变化影响。", "",
              "这些是单 Seed 的普通 BCE Test 结果，训练规模见上表，不代表静态引导方法的泛化结论。", "",
              "评价和报告复用 evaluate_transfer、transfer_overlap、transfer_report；已有预测直接复用，不重新训练或选择阈值。", ""]
    cross = [r for r in rows if r["view"] == "independent" and r["training_dataset"] != r["test_dataset"]]
    worse = sum(r["delta_target_auc"] < 0 for r in cross)
    lines += [f"本轮 {worse}/{len(cross)} 个跨域方向的 AUC 点估计低于同目标库训练的 BCE。单 Seed 点估计不作为统计显著性或普遍性证明。", ""]
    (root / "report.md").write_text("\n".join(lines))
    print("Source-only BCE transfer report:", root, len(rows), "rows", flush=True)


def paired_auc(labels, baseline, comparison, groups):
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels)
    baseline, comparison = np.asarray(baseline), np.asarray(comparison)
    label_groups = [np.flatnonzero(labels == y) for y in (0, 1)]
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(1000):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in label_groups])
        differences.append(roc_auc_score(labels[idx], comparison[idx]) - roc_auc_score(labels[idx], baseline[idx]))
    membership = {}
    for i, key in enumerate(groups):
        membership.setdefault(key, []).append(i)
    units = list(membership.values())
    clustered = []
    rng = np.random.default_rng(42)
    if len(units) > 1:
        for _ in range(1000):
            idx = np.concatenate([units[i] for i in rng.integers(len(units), size=len(units))])
            if len(set(labels[idx])) == 2:
                clustered.append(roc_auc_score(labels[idx], comparison[idx]) - roc_auc_score(labels[idx], baseline[idx]))
    return dict(delta=float(roc_auc_score(labels, comparison) - roc_auc_score(labels, baseline)),
                paired_function_bootstrap_95ci=np.quantile(differences, [.025, .975]).tolist(),
                paired_group_bootstrap_95ci=np.quantile(clustered, [.025, .975]).tolist() if len(clustered) >= 10 else None,
                groups=len(units), valid_group_replicates=len(clustered), replicates=1000, seed=42)


@torch.no_grad()
def relation_ablation(model, rows, builder, device, threshold):
    """Re-run actual CodeBERT layers with and without selected static relations.

    Unlike the removed post-encoding pool, relations now modify intermediate
    Transformer hidden states. Final hidden-state masking cannot evaluate their
    contribution, so every counterfactual must run the encoder.
    """
    from .static_analysis import EDGE_TYPES, RELATIONS
    variants = ("all", "none", "rewired", "no_dataflow", "no_control", "no_patterns")
    values = {name: [] for name in variants}
    logits = {name: [] for name in variants}
    rewired = InputBuilder(builder.tokenizer, relation_mode="shuffled")
    module = model.task_modules["relations"]
    model.eval()
    for at in range(0, len(rows), 4):
        batch = rows[at:at + 4]
        inputs = builder.codebert_batch(batch, device=device)
        edges = inputs[-1]
        shuffled = rewired.codebert_batch(batch, device=device)[-1]
        for kind in variants:
            if kind == "all":
                selected = edges
            elif kind == "none":
                selected = edges[:0]
            elif kind == "rewired":
                selected = shuffled
            elif kind == "no_dataflow":
                ignored = {RELATIONS.index("REACHING_DEF"), RELATIONS.index("REF")}
                selected = edges[~torch.isin(edges[:, 3], edges.new_tensor(sorted(ignored)))]
            elif kind == "no_control":
                selected = edges[edges[:, 3] != RELATIONS.index("CDG")]
            else:
                selected = edges[edges[:, 3] < len(EDGE_TYPES)]
            scores = model(*(*inputs[:-1], selected)).float().cpu()
            logits[kind].extend(scores.tolist())
            values[kind].extend(scores.sigmoid().tolist())
    labels = [r["label"] for r in rows]
    original = np.array(logits["all"])
    no_structure = np.array(logits["none"])
    rewired_scores = np.array(logits["rewired"])
    strength = dict(
        abs_logit_change_without_relations=float(np.mean(np.abs(original - no_structure))),
        abs_logit_change_rewired=float(np.mean(np.abs(original - rewired_scores))),
        changed_logit_fraction_without_relations=float(np.mean(np.abs(original - no_structure) > 1e-4)),
        changed_logit_fraction_rewired=float(np.mean(np.abs(original - rewired_scores) > 1e-4)),
    )
    result = dict(
        metrics={kind: metrics(labels, v, threshold) for kind, v in values.items()},
        sensitivity=strength,
        adapter_layer_indices=list(module.layer_indices),
        weight_norms={name: float(p.norm()) for name, p in module.named_parameters()},
        interpretation="actual full-encoder inference with fixed real-model parameters, removed/reconnected relations; unchanged Valid threshold and membership",
    )
    return result, values["all"]

def report(data_dir, output_dir, *, device="auto"):
    device = resolve_device(device)
    tables, predictions, ablations, snapshot_comparisons = [], {}, {}, {}
    for dataset in DATASETS:
        rows = read_jsonl(Path(data_dir) / "pilot" / dataset / "valid.jsonl")
        labels = [r["label"] for r in rows]
        predictions[dataset] = {}
        snapshot_comparisons[dataset] = {}
        for mode in MODES:
            run = Path(output_dir) / mode / dataset / "bce"
            path = run / "best.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = load_model(checkpoint, device)
            builder = InputBuilder(AutoTokenizer.from_pretrained(checkpoint["model_path"]),
                                   relation_mode="none" if mode == "source" else mode)
            started = time.monotonic()
            scores = predict_logits(model, rows, builder, device=device).sigmoid().tolist()
            metric = metrics(labels, scores, checkpoint["decision_threshold"])
            stored = checkpoint["validation"]
            if metric["samples"] != stored["samples"]:
                raise ValueError(f"reloaded Valid sample count changed: {dataset}/{mode}")
            # Old checkpoints used nondeterministic CUDA atomics. Near-tied
            # rankings and near-threshold predictions can change under
            # repeatable reduction. Keep both measurements and mark differences;
            # never substitute the training snapshot for current predictions.
            counts_match = all(metric[k] == stored[k] for k in ("tp", "tn", "fp", "fn"))
            snapshot_comparisons[dataset][mode] = dict(
                training=stored, reloaded=metric, confusion_counts_equal=counts_match,
                deltas={k: metric[k] - stored[k] for k in
                        ("accuracy", "f1", "mcc", "auc", "auprc", "tp", "tn", "fp", "fn")},
            )
            ranking = dict(training_auc=stored["auc"], training_auprc=stored["auprc"],
                           auc_delta_from_training=metric["auc"] - stored["auc"],
                           auprc_delta_from_training=metric["auprc"] - stored["auprc"],
                           training_confusion_counts_equal=counts_match)
            if ranking["auc_delta_from_training"] or ranking["auprc_delta_from_training"] or not counts_match:
                print(f"{dataset}/{mode}: stable reload AUC {metric['auc']:.9f} "
                      f"(training snapshot {stored['auc']:.9f}); confusion counts "
                      f"{stored['confusion_matrix']} -> {metric['confusion_matrix']}", flush=True)
            elapsed = time.monotonic() - started
            predictions[dataset][mode] = scores
            write_jsonl(run / "valid.predictions.jsonl", [dict(sample_key=r["sample_key"], label=r["label"], score=s) for r, s in zip(rows, scores)])
            history = read_jsonl(run / "best.training.jsonl")
            epochs = [r for r in history if r["event"] == "epoch"]
            config = json.loads((run / "training.metrics.json").read_text())
            if config["optimizer_steps"] != 384 or config["completed_epochs"] != 3:
                raise ValueError("pilot budget must be exactly 384 optimizer steps")
            tables.append(dict(dataset=dataset, mode=mode, epoch=checkpoint["selected_epoch"],
                               **{k: metric[k] for k in ("samples", "accuracy", "f1", "mcc", "auc", "auprc", "threshold")},
                               parameters=checkpoint["parameter_count"], steps=config["optimizer_steps"],
                               epoch_seconds=sum(e["epoch_seconds"] for e in epochs),
                               train_encoder_sequences=sum(e["bce_costs"]["encoder_sequences"] for e in epochs),
                               train_token_slots=sum(e["bce_costs"]["encoder_token_slots"] for e in epochs),
                               prediction_seconds=elapsed, **ranking))
            if mode == "real":
                ablations[dataset], all_scores = relation_ablation(model, rows, builder, device, metric["threshold"])
                difference = float(np.max(np.abs(np.asarray(all_scores) - scores)))
                if not np.allclose(all_scores, scores, atol=1e-7, rtol=1e-6):
                    raise ValueError(f"full-encoder relation ablation does not reproduce normal predictions: "
                                     f"{dataset}/real, max probability difference={difference:.8g}")
                ablations[dataset]["clean_prediction_max_abs_difference"] = difference
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(dataset, [(r["mode"], round(r["auc"], 4), round(r["mcc"], 4)) for r in tables if r["dataset"] == dataset], flush=True)
        costs = {(r["train_encoder_sequences"], r["train_token_slots"]) for r in tables if r["dataset"] == dataset}
        if len(costs) != 1:
            raise ValueError("encoder input budgets differ between controls")
    contrasts = {}
    for dataset in DATASETS:
        rows = read_jsonl(Path(data_dir) / "pilot" / dataset / "valid.jsonl")
        labels = [r["label"] for r in rows]
        groups = [r["project"] if dataset == "bigvul" else
                  (r["project"], r["commit_id"] if dataset == "devign" else r["upstream_group"]) for r in rows]
        contrasts[dataset] = {mode: paired_auc(labels, predictions[dataset][mode], predictions[dataset]["real"], groups)
                              for mode in ("source", "sequence", "shuffled")}
    out = Path(output_dir)
    coverage = {}
    for dataset in DATASETS:
        info = json.loads((out / "real" / dataset / "bce/relation.coverage.json").read_text())
        shuffled = json.loads((out / "shuffled" / dataset / "bce/relation.coverage.json").read_text())
        edges = sum(v["window_edge_instances"] for v in shuffled.values())
        coverage[dataset] = dict(
            extraction=json.loads((out / f"{dataset}.coverage.json").read_text()),
            model_functions_with_relations=sum(v["selected_edges"] > 0 for v in info.values()),
            model_functions_with_window_relations=sum(
                v.get("pairs_in_shared_windows", 0) > 0 for v in info.values()
            ),
            model_functions=len(info),
            selected_cpg_pairs=sum(v.get("selected_cpg", 0) for v in info.values()),
            selected_pattern_pairs=sum(v.get("selected_patterns", 0) for v in info.values()),
            pairs_in_shared_windows=sum(
                v.get("pairs_in_shared_windows", 0) for v in info.values()
            ),
            pairs_outside_shared_windows=sum(
                v.get("pairs_outside_shared_windows", 0) for v in info.values()
            ),
            visible_located_nodes=sum(v["visible_nodes"] for v in info.values()),
            located_nodes=sum(v.get("located_nodes", 0) for v in info.values()),
            shuffled_changed_endpoints=sum(v.get("changed_endpoints", 0) for v in shuffled.values()),
            shuffled_total_endpoints=edges,
            shuffled_changed_fraction=sum(v.get("changed_endpoints", 0) for v in shuffled.values()) / edges if edges else None,
        )
    with (out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(tables[0]));writer.writeheader();writer.writerows(tables)
    write_json(out / "results.json", dict(validation=tables, real_minus_control=contrasts,
                                          relation_readout_ablation=ablations,
                                          training_validation_comparison=snapshot_comparisons,
                                          coverage=coverage,
                                          protocol="Seed42 Train1024/Valid256, 3 epochs/effective batch8; same initialization, optimizer and 384 steps; source Valid epoch+threshold selection; Test unused",
                                          interpretation="Validation selection and one seed: pilot evidence, not independent Test confirmation; ablations re-run the full encoder without retraining; deterministic query reductions preserve attention semantics; metrics come from stable reload predictions at the frozen epoch/threshold; original training snapshots and all metric/confusion differences are explicitly retained, not declared equal"))
    print(json.dumps(contrasts), flush=True)
