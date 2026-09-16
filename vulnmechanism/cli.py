from __future__ import annotations

import argparse

from .dataset import build_function_dataset
from .model import MODEL_VARIANTS, evaluate_model, train_model
from .semantics import SEMANTIC_GROUPS, validate_semantic_groups


def _semantic_groups(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    return validate_semantic_groups(tuple(part.strip() for part in value.split(",")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Function-level C/C++ vulnerability-semantic learning"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--samples", required=True)
    build.add_argument("--output", default="data/function_dataset.jsonl")
    build.add_argument("--joern-dir", default="/home/phy/joern")
    build.add_argument("--java-home", default="/home/phy/jdk21")
    build.add_argument("--timeout", type=int, default=300)
    build.set_defaults(func=lambda args: build_function_dataset(
        args.samples,
        args.output,
        joern_dir=args.joern_dir,
        java_home=args.java_home,
        timeout=args.timeout,
    ))

    train = sub.add_parser("train")
    train.add_argument("--dataset", default="data/function_dataset.jsonl")
    train.add_argument("--variant", choices=MODEL_VARIANTS, default="full")
    train.add_argument("--output")
    train.add_argument("--model", default="/home/phy/models/Qwen2.5-Coder-7B-Instruct")
    train.add_argument("--source-max-length", type=int, default=1536)
    train.add_argument("--context-max-length", type=int, default=384)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--gradient-accumulation", type=int, default=8)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--lora-r", type=int, default=16)
    train.add_argument("--lora-alpha", type=int, default=32)
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.add_argument("--fusion-dim", type=int, default=256)
    train.add_argument("--fusion-heads", type=int, default=8)
    train.add_argument("--feature-loss-weight", type=float, default=0.2)
    train.add_argument(
        "--exclude-groups",
        type=_semantic_groups,
        default=(),
        metavar="GROUPS",
        help=(
            "comma-separated semantic ablation groups: "
            + ",".join(SEMANTIC_GROUPS)
        ),
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto")
    train.set_defaults(func=lambda args: train_model(
        args.dataset,
        args.output or f"results/{args.variant}.pt",
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
        feature_loss_weight=args.feature_loss_weight,
        excluded_groups=args.exclude_groups,
        seed=args.seed,
        device=args.device,
    ))

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--dataset", default="data/function_dataset.jsonl")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=("train", "valid", "test", "external_test"), default="test")
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--device", default="auto")
    evaluate.set_defaults(func=lambda args: evaluate_model(
        args.dataset,
        args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
    ))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
