from __future__ import annotations

import argparse

from .mechanism import build_mechanism_dataset
from .model import evaluate_mechanism_models, train_mechanism_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Function-only vulnerability mechanism learning'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    build = sub.add_parser('build')
    build.add_argument('--pairs', required=True)
    build.add_argument('--output', default='data/mechanism_dataset.jsonl')
    build.add_argument('--joern-dir', default='/home/phy/joern')
    build.add_argument('--java-home', default='/home/phy/jdk21')
    build.add_argument('--timeout', type=int, default=300)
    build.set_defaults(func=lambda args: build_mechanism_dataset(
        args.pairs, args.output,
        joern_dir=args.joern_dir,
        java_home=args.java_home,
        timeout=args.timeout,
    ))

    train = sub.add_parser('train')
    train.add_argument('--dataset', default='data/mechanism_dataset.jsonl')
    train.add_argument('--output', default='results/mechanism_model.pt')
    train.add_argument('--model', default='/home/phy/models/Qwen2.5-Coder-7B-Instruct')
    train.add_argument('--max-length', type=int, default=2048)
    train.add_argument('--encoder-batch-size', type=int, default=2)
    train.add_argument('--head-batch-size', type=int, default=64)
    train.add_argument('--epochs', type=int, default=20)
    train.add_argument('--learning-rate', type=float, default=1e-3)
    train.add_argument('--mechanism-weight', type=float, default=1.0)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--device', default='auto')
    train.set_defaults(func=lambda args: train_mechanism_models(
        args.dataset,
        args.output,
        model_path=args.model,
        max_length=args.max_length,
        encoder_batch_size=args.encoder_batch_size,
        head_batch_size=args.head_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        mechanism_weight=args.mechanism_weight,
        seed=args.seed,
        device=args.device,
    ))

    evaluate = sub.add_parser('eval')
    evaluate.add_argument('--dataset', default='data/mechanism_dataset.jsonl')
    evaluate.add_argument('--checkpoint', default='results/mechanism_model.pt')
    evaluate.add_argument('--split', choices=('train', 'valid', 'test'), default='test')
    evaluate.add_argument('--encoder-batch-size', type=int, default=2)
    evaluate.add_argument('--device', default='auto')
    evaluate.set_defaults(func=lambda args: evaluate_mechanism_models(
        args.dataset,
        args.checkpoint,
        split=args.split,
        encoder_batch_size=args.encoder_batch_size,
        device=args.device,
    ))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
