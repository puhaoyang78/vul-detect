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
        prepare_relations(args.data_dir, args.output_dir, datasets)
    elif args.stage == "train":
        for dataset in datasets:
            for mode in args.modes:
                train(args.data_dir / "pilot", args.output_dir / mode, args.model, dataset, "bce",
                      device=args.device, relation_mode="none" if mode == "source" else mode)
    elif args.stage == "report":
        if args.dataset != "all":
            raise ValueError("the unified report requires all three datasets")
        report(args.data_dir, args.output_dir, device=args.device)


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
    """Attention-readout Valid ablation, reusing encoder windows across edge controls."""
    from .model import _window_mean
    from .static_analysis import RELATIONS
    values = {kind: [] for kind in ("all", "none", "rewired", *RELATIONS)}
    rewired = InputBuilder(builder.tokenizer, relation_mode="shuffled")
    strength = dict(base_norm=[], residual_norm=[], logit_delta=[])
    model.eval()
    for at in range(0, len(rows), 4):
        inputs = builder.codebert_batch(rows[at:at + 4], device=device)
        ids, mask, owners, pooling, n, positions, edges = inputs
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model.encoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        base = _window_mean(hidden, mask, owners, pooling, n)
        head, module = model.task_modules["classifier"], model.task_modules["relations"]
        shuffled_edges = rewired.codebert_batch(rows[at:at + 4], device=device)[-1]
        for kind in values:
            selected = (edges if kind == "all" else edges[:0] if kind == "none"
                        else shuffled_edges if kind == "rewired" else edges[edges[:, 3] != RELATIONS.index(kind)])
            focused = module(hidden, positions, owners, pooling, selected, n, base)
            values[kind].extend(head(focused).squeeze(-1).sigmoid().tolist())
            if kind == "all":
                difference = focused - base
                strength["base_norm"].extend(base.norm(dim=1).tolist())
                strength["residual_norm"].extend(difference.norm(dim=1).tolist())
                strength["logit_delta"].extend(torch.nn.functional.linear(difference, head.weight).squeeze(-1).tolist())
    labels = [r["label"] for r in rows]
    result = dict(metrics={k: metrics(labels, v, threshold) for k, v in values.items()},
                  strength={k: dict(mean=float(np.mean(v)), median=float(np.median(v)),
                                    rms=float(np.sqrt(np.mean(np.square(v)))), p95=float(np.quantile(np.abs(v), .95)))
                            for k, v in strength.items()},
                  weight_norms={k: float(p.norm()) for k, p in module.named_parameters()},
                  interpretation="same trained encoder; attention readout under real/removed/rewired relations, no threshold reselection")
    return result, values["all"]


def report(data_dir, output_dir, *, device="auto"):
    device = resolve_device(device)
    tables, predictions, ablations = [], {}, {}
    for dataset in DATASETS:
        rows = read_jsonl(Path(data_dir) / "pilot" / dataset / "valid.jsonl")
        labels = [r["label"] for r in rows]
        predictions[dataset] = {}
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
            if any(abs(metric[k] - checkpoint["validation"][k]) > 1e-6 for k in ("auc", "mcc", "f1")):
                raise ValueError(f"reloaded Valid metrics changed: {dataset}/{mode}")
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
                               prediction_seconds=elapsed))
            if mode == "real":
                ablations[dataset], all_scores = relation_ablation(model, rows, builder, device, metric["threshold"])
                if not np.allclose(all_scores, scores, atol=1e-7, rtol=1e-6):
                    raise ValueError("cached-window readout ablation does not reproduce normal predictions")
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
        edges = sum(v["selected_edges"] for v in shuffled.values())
        coverage[dataset] = dict(
            extraction=json.loads((out / f"{dataset}.coverage.json").read_text()),
            model_functions_with_relations=sum(v["selected_edges"] > 0 for v in info.values()),
            model_functions=len(info),
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
                                          coverage=coverage,
                                          protocol="Seed42 Train1024/Valid256, 3 epochs/effective batch8; same initialization, optimizer and 384 steps; source Valid epoch+threshold selection; Test unused",
                                          interpretation="Validation selection and one seed: pilot evidence, not independent Test confirmation; readout ablation does not retrain the encoder"))
    print(json.dumps(contrasts), flush=True)
