from __future__ import annotations

import argparse

from .benchmark_view import FORMAL_DATASETS, dataset_view
from .dataset import build_function_dataset
from .model import MODEL_VARIANTS, evaluate_model, train_model
from .semantics import MECHANISM_GROUPS, validate_mechanism_groups


def _mechanism_groups(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    return validate_mechanism_groups(tuple(part.strip() for part in value.split(",")))


def _fit_arguments(parser):
    parser.add_argument("--model", default="/home/phy/models/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--source-max-length", type=int, default=1536)
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
    parser.add_argument("--log-every", type=int, default=10, help="print and save loss every N optimizer steps; batch progress is continuous")


def _train(args):
    if args.source_dataset == "sven":
        raise ValueError("SVEN is external-test-only and cannot be used for training")
    with dataset_view(args.dataset, args.source_dataset) as dataset:
        return train_model(
            dataset,
            args.output or f"results/{args.source_dataset}_{args.variant}.pt",
            variant=args.variant,
            model_path=args.model,
            source_max_length=args.source_max_length,
            context_max_length=args.context_max_length,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            fusion_dim=args.fusion_dim,
            fusion_heads=args.fusion_heads,
            excluded_groups=args.exclude_groups,
            seed=args.seed,
            device=args.device,
            log_every=args.log_every,
        )


def _evaluate(args):
    expected_split = "external_test" if args.source_dataset == "sven" else args.split
    if args.source_dataset == "sven" and args.split != "external_test":
        raise ValueError("SVEN evaluation must use --split external_test")
    if args.source_dataset != "sven" and args.split == "external_test":
        raise ValueError("external_test is reserved for SVEN")
    with dataset_view(args.dataset, args.source_dataset) as dataset:
        return evaluate_model(
            dataset,
            args.checkpoint,
            split=expected_split,
            batch_size=args.batch_size,
            device=args.device,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Function-level C/C++ CPG-guided vulnerability-mechanism learning"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--samples", required=True)
    build.add_argument("--output", default="data/function_dataset.jsonl")
    build.add_argument("--joern-dir", default="/home/phy/joern")
    build.add_argument("--java-home", default="/home/phy/jdk21")
    build.add_argument("--timeout", type=int, default=300)
    build.add_argument("--batch-size", type=int, default=8)
    build.set_defaults(
        func=lambda args: build_function_dataset(
            args.samples,
            args.output,
            joern_dir=args.joern_dir,
            java_home=args.java_home,
            timeout=args.timeout,
            batch_size=args.batch_size,
        )
    )

    train = sub.add_parser("train")
    train.add_argument("--dataset", default="data/function_dataset.jsonl")
    train.add_argument(
        "--source-dataset",
        choices=("primevul", "cleanvul"),
        required=True,
    )
    train.add_argument("--variant", choices=MODEL_VARIANTS, default="mechanism_fusion")
    train.add_argument("--output")
    _fit_arguments(train)
    train.add_argument("--context-max-length", type=int, default=384)
    train.add_argument("--fusion-dim", type=int, default=256)
    train.add_argument("--fusion-heads", type=int, default=8)
    train.add_argument(
        "--exclude-groups",
        type=_mechanism_groups,
        default=(),
        metavar="GROUPS",
        help="comma-separated mechanism ablation groups: " + ",".join(MECHANISM_GROUPS),
    )
    train.set_defaults(func=_train)

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--dataset", default="data/function_dataset.jsonl")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--source-dataset", choices=FORMAL_DATASETS, required=True)
    evaluate.add_argument(
        "--split",
        choices=("train", "valid", "test", "external_test"),
        default="test",
    )
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--device", default="auto")
    evaluate.set_defaults(func=_evaluate)

    diagnostic = sub.add_parser("diagnose", help="train-only nested OOF and incremental-information probes")
    diagnostic.add_argument("--dataset", default="data/function_dataset.jsonl")
    diagnostic.add_argument("--manifest", default="data/benchmark/manifest.jsonl")
    diagnostic.add_argument("--output", default="results/primevul_incremental")
    diagnostic.add_argument("--stage", choices=("prepare", "oof", "probe", "all"), default="prepare")
    diagnostic.add_argument("--outer-folds", type=int, default=3)
    diagnostic.add_argument("--inner-folds", type=int, default=3)
    diagnostic.add_argument("--calibration-folds", type=int, default=5)
    diagnostic.add_argument("--shuffle-repeats", type=int, default=5)
    diagnostic.add_argument("--length-bin", type=int, default=64, help="exact character-length bins for shuffling")
    diagnostic.add_argument("--probe-c", type=float, default=1.0, help="fixed logistic regularization; no held-out tuning")
    diagnostic.add_argument("--max-features", type=int, default=2000)
    _fit_arguments(diagnostic)
    diagnostic.set_defaults(func=_diagnose)

    pilot = sub.add_parser("source-pilot", help="bounded reviewed-error transfer experiment")
    pilot.add_argument("--dataset", default="data/function_dataset.jsonl")
    pilot.add_argument("--manifest", default="data/benchmark/manifest.jsonl")
    pilot.add_argument("--diagnostic", default="results/primevul_incremental")
    pilot.add_argument("--review", default="results/primevul_incremental/source_error_review.jsonl")
    pilot.add_argument("--output", default="results/primevul_source_pilot")
    pilot.add_argument("--stage", choices=("prepare", "run"), default="prepare")
    pilot.add_argument("--device", default="cuda")
    pilot.set_defaults(func=_source_pilot)
    return parser


def _diagnose(args):
    from .diagnostic import run_diagnostic
    return run_diagnostic(args)


def _source_pilot(args):
    from . import source_pilot
    return source_pilot.prepare(args) if args.stage == "prepare" else source_pilot.run(args)


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
