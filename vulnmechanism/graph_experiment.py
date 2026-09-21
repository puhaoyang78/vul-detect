"""Run the fixed Source / Source+attributes / Source+CFG experiment.

Build is CPU/Joern-only. Run launches each fit in a fresh process, reuses the
existing training loop, and evaluates VALIDATION only. Test evaluation is an
explicit command and never chooses a threshold or checkpoint.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .graph_dataset import (
    atomic_json, atomic_jsonl, build_graph_dataset, file_sha256,
    load_graph_dataset, output_lock, read_jsonl,
)
from .graph_features import GRAPH_SCHEMA_VERSION, fit_graph_config, records_fingerprint

VARIANTS = ("baseline", "graph_attributes", "graph_cfg")
FIT_FLAGS = ("model", "source_max_length", "batch_size", "gradient_accumulation", "epochs",
             "learning_rate", "weight_decay", "lora_r", "lora_alpha", "lora_dropout", "seed",
             "device", "log_every", "graph_embedding_dim", "graph_steps", "graph_vocab_size")


def _paths(checkpoint: str | Path, split: str = "valid") -> tuple[Path, Path]:
    checkpoint = Path(checkpoint)
    return (checkpoint.with_suffix(f".{split}.predictions.jsonl"),
            checkpoint.with_suffix(f".{split}.metrics.json"))


def _protocol(args, rows: list[dict], metadata: dict) -> dict:
    common = {name: getattr(args, name) for name in FIT_FLAGS if name not in {"device", "log_every"}}
    return {"version": 1, "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "source_dataset": metadata["source_dataset"], "variant": args.variant,
            "data_sha256": metadata["output_sha256"],
            "membership_fingerprint": records_fingerprint(rows),
            "split_fingerprints": {split: records_fingerprint([r for r in rows if r["split"] == split])
                                   for split in ("train", "valid", "test", "external_test")},
            "config": common}


def train_one(args) -> dict:
    import torch
    from .model import train_model

    rows, metadata = load_graph_dataset(args.dataset)
    if metadata["source_dataset"] == "sven":
        raise ValueError("SVEN is external-test-only")
    if not metadata.get("available"):
        raise ValueError("no usable graphs; inspect graph export errors")
    expected = _protocol(args, rows, metadata)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_path = output.with_suffix(".run.json")
    with output_lock(output):
        if output.exists():
            if not args.resume:
                raise ValueError(f"{output} already exists; use --resume to verify and reuse it")
            checkpoint = torch.load(output, map_location="cpu", weights_only=False)
            if checkpoint.get("graph_experiment") != expected:
                raise ValueError(f"{output}: dataset/config differs; use a new output directory")
            print(f"reuse_completed_checkpoint={output}", flush=True)
            return checkpoint
        atomic_json(run_path, {"status": "training", "protocol": expected})
        graph_options = None if args.variant == "baseline" else dict(
            embedding_dim=args.graph_embedding_dim, steps=args.graph_steps,
            vocab_size=args.graph_vocab_size)
        # Use the exact same train_model loop as every existing source baseline.
        # Passing the path also retains its standard schema/label validation.
        del rows
        checkpoint = train_model(
            args.dataset, output, variant=args.variant, model_path=args.model,
            source_max_length=args.source_max_length, context_max_length=384,
            batch_size=args.batch_size, gradient_accumulation=args.gradient_accumulation,
            epochs=args.epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            seed=args.seed, device=args.device, log_every=args.log_every,
            graph_options=graph_options,
        )
        checkpoint["graph_experiment"] = expected
        temporary = output.with_name(output.name + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(output)
        atomic_json(run_path, {"status": "complete", "protocol": expected,
                               "selected_epoch": checkpoint["selected_epoch"],
                               "validation": checkpoint["validation"],
                               "decision_threshold": checkpoint["decision_threshold"],
                               "checkpoint_sha256": file_sha256(output)})
        return checkpoint


def _confusion(rows: list[dict], probabilities, threshold: float) -> dict:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for row, value in zip(rows, probabilities):
        prediction = float(value) >= threshold
        counts[("tp" if prediction else "fn") if row["label"] else ("fp" if prediction else "tn")] += 1
    return counts


def evaluate_one(args) -> dict:
    import torch
    from transformers import AutoTokenizer
    from .model import (
        InputBuilder, _classification_metrics, _load_model, _predict, _resolve_device,
    )

    all_rows, metadata = load_graph_dataset(args.dataset)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    protocol = checkpoint.get("graph_experiment")
    if protocol is None or protocol.get("data_sha256") != metadata["output_sha256"]:
        raise ValueError("checkpoint and graph dataset are not from the same A/B/C experiment")
    rows = [r for r in all_rows if r["split"] == args.split]
    if not rows:
        raise ValueError(f"{args.split} split is empty")
    if args.split not in {"valid", "test"}:
        raise ValueError("this comparison accepts valid/test only")
    threshold = float(checkpoint["decision_threshold"])
    device = _resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["model_path"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    builder = InputBuilder(tokenizer, source_max_length=checkpoint["source_max_length"], context_max_length=384)
    model = _load_model(checkpoint, device=device)
    probabilities = _predict(model, rows, builder, variant=checkpoint["variant"], excluded_groups=(),
                             batch_size=args.batch_size, device=device)
    token_lengths = [len(tokenizer(r["raw_source"], add_special_tokens=False, truncation=False)["input_ids"])
                     for r in rows]
    prediction_rows = []
    for row, probability, tokens in zip(rows, probabilities.tolist(), token_lengths):
        prediction_rows.append({"sample_key": row["sample_key"], "dataset": row["dataset"],
                                "split": row["split"], "label": row["label"],
                                "probability": probability, "prediction": int(probability >= threshold),
                                "decision_threshold": threshold,
                                "graph_available": row["static_graph"] is not None,
                                "source_tokens": tokens,
                                "source_truncated": tokens > checkpoint["source_max_length"]})
    subsets = {}
    for name, indices in {
        "source_not_truncated": [i for i, n in enumerate(token_lengths) if n <= checkpoint["source_max_length"]],
        "source_truncated": [i for i, n in enumerate(token_lengths) if n > checkpoint["source_max_length"]],
        "graph_available": [i for i, r in enumerate(rows) if r["static_graph"] is not None],
        "graph_unavailable": [i for i, r in enumerate(rows) if r["static_graph"] is None],
    }.items():
        if indices:
            selected = [rows[i] for i in indices]
            subsets[name] = {"samples": len(indices),
                             "metrics": _classification_metrics(selected, probabilities[indices], threshold=threshold).as_json()}
    result = {"variant": checkpoint["variant"], "split": args.split, "samples": len(rows),
              "selected_epoch": checkpoint["selected_epoch"], "decision_threshold": threshold,
              "metrics": _classification_metrics(rows, probabilities, threshold=threshold).as_json(),
              "metrics_at_0_5": _classification_metrics(rows, probabilities, threshold=0.5).as_json(),
              "confusion": _confusion(rows, probabilities, threshold), "subgroups": subsets,
              "protocol": protocol, "checkpoint_sha256": file_sha256(args.checkpoint),
              "note": "Uses saved validation threshold; no threshold selection during evaluation."}
    predictions_path, metrics_path = _paths(args.checkpoint, args.split)
    atomic_jsonl(predictions_path, prediction_rows)
    result["predictions_sha256"] = file_sha256(predictions_path)
    atomic_json(metrics_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def flip_counts(baseline_rows: list[dict], other_rows: list[dict], *, shared_threshold: float | None = None) -> dict:
    baseline = {r["sample_key"]: r for r in baseline_rows}
    other = {r["sample_key"]: r for r in other_rows}
    if len(baseline) != len(baseline_rows) or len(other) != len(other_rows) or set(baseline) != set(other):
        raise ValueError("prediction members must be unique and identical")
    result = dict(fn_to_tp=0, fp_to_tn=0, tp_to_fn=0, tn_to_fp=0, unchanged_correct=0, unchanged_wrong=0)
    for key, old in baseline.items():
        new = other[key]
        for name in ("dataset", "split", "label", "source_tokens", "source_truncated", "graph_available"):
            if old.get(name) != new.get(name):
                raise ValueError(f"prediction identity/coverage mismatch: {key}/{name}")
        truth = int(old["label"])
        a = int(old["prediction"]) if shared_threshold is None else int(old["probability"] >= shared_threshold)
        b = int(new["prediction"]) if shared_threshold is None else int(new["probability"] >= shared_threshold)
        if a == b:
            result["unchanged_correct" if a == truth else "unchanged_wrong"] += 1
        elif a != truth:
            result["fn_to_tp" if truth else "fp_to_tn"] += 1
        else:
            result["tp_to_fn" if truth else "tn_to_fp"] += 1
    result["corrected"] = result["fn_to_tp"] + result["fp_to_tn"]
    result["damaged"] = result["tp_to_fn"] + result["tn_to_fp"]
    result["net_corrected"] = result["corrected"] - result["damaged"]
    result["net_fn_reduction"] = result["fn_to_tp"] - result["tp_to_fn"]
    result["net_fp_reduction"] = result["fp_to_tn"] - result["tn_to_fp"]
    return result


def compare(args) -> dict:
    directory = Path(args.output_dir)
    results, predictions = {}, {}
    reference = None
    for variant in VARIANTS:
        pred_path, metric_path = _paths(directory / f"{variant}.pt", args.split)
        result = json.loads(metric_path.read_text(encoding="utf-8"))
        if result["variant"] != variant or result["split"] != args.split:
            raise ValueError("wrong prediction variant/split")
        if result["predictions_sha256"] != file_sha256(pred_path):
            raise ValueError("prediction file changed after evaluation")
        if result["checkpoint_sha256"] != file_sha256(directory / f"{variant}.pt"):
            raise ValueError("checkpoint changed after evaluation")
        protocol = {k: v for k, v in result["protocol"].items() if k != "variant"}
        if reference is not None and protocol != reference:
            raise ValueError("A/B/C do not share the same data and training configuration")
        reference = protocol
        results[variant] = result
        predictions[variant] = read_jsonl(pred_path)
    output = {"split": args.split, "rows": [], "flips": {}}
    base = results["baseline"]
    for variant in VARIANTS:
        result = results[variant]
        output["rows"].append({"variant": variant, **result["metrics"],
                               "threshold": result["decision_threshold"], **result["confusion"]})
        if variant != "baseline":
            output["flips"][variant] = {
                "saved_validation_thresholds": flip_counts(predictions["baseline"], predictions[variant]),
                "shared_baseline_threshold": flip_counts(predictions["baseline"], predictions[variant],
                                                          shared_threshold=base["decision_threshold"]),
            }
    atomic_json(directory / f"{args.split}_comparison.json", output)
    print("\nvariant                   Accuracy Precision Recall   F1       MCC      AUC", flush=True)
    for row in output["rows"]:
        values = [f"{row[name]:.4f}" if row[name] is not None else "n/a" for name in
                  ("accuracy", "precision", "recall", "f1", "mcc", "auc")]
        print(f"{row['variant']:<26}" + " ".join(f"{v:>8}" for v in values), flush=True)
    for variant, counts in output["flips"].items():
        print(variant, json.dumps(counts["saved_validation_thresholds"]), flush=True)
    return output


def run_all(args) -> dict:
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # Validate integrity and coverage before launching any costly model loading.
    rows, metadata = load_graph_dataset(args.dataset)
    if metadata["source_dataset"] == "sven" or not metadata.get("available"):
        raise ValueError("run requires a training dataset with usable graphs")
    # Validate the train-only graph vocabulary before investing in baseline A.
    fit_graph_config([r for r in rows if r["split"] == "train"],
                     embedding_dim=args.graph_embedding_dim, steps=args.graph_steps,
                     vocab_size=args.graph_vocab_size)
    del rows
    for variant in VARIANTS:
        checkpoint = directory / f"{variant}.pt"
        command = [sys.executable, "-m", "vulnmechanism.graph_experiment", "train",
                   "--dataset", args.dataset, "--variant", variant, "--output", str(checkpoint)]
        for name in FIT_FLAGS:
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        if args.resume:
            command.append("--resume")
        print("\nTRAIN", variant, flush=True)
        subprocess.run(command, check=True)
        print("\nVALIDATE", variant, flush=True)
        subprocess.run([sys.executable, "-m", "vulnmechanism.graph_experiment", "eval",
                        "--dataset", args.dataset, "--checkpoint", str(checkpoint),
                        "--split", "valid", "--batch-size", str(args.batch_size), "--device", args.device],
                       check=True)
    return compare(argparse.Namespace(output_dir=str(directory), split="valid"))


def _fit_args(parser):
    parser.add_argument("--model", default="/home/phy/models/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--source-max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--graph-embedding-dim", type=int, default=32)
    parser.add_argument("--graph-steps", type=int, default=5)
    parser.add_argument("--graph-vocab-size", type=int, default=2048)
    parser.add_argument("--resume", action="store_true", help="reuse only completed, exactly matching checkpoints; incomplete fits restart")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="export full node attributes/edges without changing sample membership")
    build.add_argument("--dataset", default="data/function_dataset.jsonl")
    build.add_argument("--output", default="data/graph_abc/primevul.jsonl")
    build.add_argument("--source-dataset", choices=("primevul", "cleanvul", "sven"), default="primevul")
    build.add_argument("--joern-dir", default="/home/phy/joern")
    build.add_argument("--java-home", default="/home/phy/jdk21")
    build.add_argument("--timeout", type=int, default=300)
    build.add_argument("--batch-size", type=int, default=8)
    build.add_argument("--retry-failed", action="store_true")
    train = sub.add_parser("train", help="fit one variant with the existing training loop")
    train.add_argument("--dataset", required=True)
    train.add_argument("--variant", choices=VARIANTS, required=True)
    train.add_argument("--output", required=True)
    _fit_args(train)
    run = sub.add_parser("run", help="fit A/B/C sequentially; compare VALIDATION only")
    run.add_argument("--dataset", default="data/graph_abc/primevul.jsonl")
    run.add_argument("--output-dir", default="results/graph_abc")
    _fit_args(run)
    evaluate = sub.add_parser("eval", help="evaluate with the saved validation threshold; export per-sample predictions")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=("valid", "test"), default="valid")
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--device", default="auto")
    summary = sub.add_parser("compare")
    summary.add_argument("--output-dir", default="results/graph_abc")
    summary.add_argument("--split", choices=("valid", "test"), default="valid")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            return build_graph_dataset(args.dataset, args.output, source_dataset=args.source_dataset,
                                       joern_dir=args.joern_dir, java_home=args.java_home,
                                       timeout=args.timeout, batch_size=args.batch_size,
                                       retry_failed=args.retry_failed)
        return {"train": train_one, "run": run_all, "eval": evaluate_one, "compare": compare}[args.command](args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
