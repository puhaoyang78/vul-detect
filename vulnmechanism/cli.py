"""Shared data preparation, training, evaluation and Motivation experiments."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from .data import DATASETS


def build_parser():
    from .training import METHODS

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare", help="retain positives, balance negatives and check exact leakage"
    )
    prepare.add_argument("--raw-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, default=Path("data"))
    prepare.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    for command in ("train", "candidates", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--data-dir", type=Path, default=Path("data"))
        p.add_argument("--output-dir", type=Path, default=Path("results"))
        p.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
        if command in ("train", "candidates"):
            p.add_argument("--model", default="microsoft/codebert-base")
        if command in ("train", "evaluate"):
            p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
            p.add_argument("--device", default="auto")
        if command == "evaluate":
            p.add_argument("--clean-only", action="store_true")
    p = sub.add_parser("report", help="summarize the saved PrimeVul/MegaVul six-method experiment")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p = sub.add_parser("motivation", help="four-dataset ordinary-BCE and Shortcut Features diagnostics")
    p.add_argument("stage", choices=("features", "report"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    p.add_argument("--model", default="microsoft/codebert-base")
    p = sub.add_parser("infer")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source-file", type=Path, required=True)
    p.add_argument("--model", help="pretrained CodeBERT location if the checkpoint has moved")
    p.add_argument("--device", default="auto")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "prepare":
        from .data import prepare

        prepare(args.raw_root, args.output_dir, datasets=args.datasets)
        return
    import torch

    torch.set_num_threads(4)
    datasets = (
        DATASETS
        if getattr(args, "dataset", None) == "all"
        else (getattr(args, "dataset", None),)
    )
    if args.command == "train":
        from .training import train

        for dataset in datasets:
            for method in args.methods:
                train(
                    args.data_dir,
                    args.output_dir,
                    args.model,
                    dataset,
                    method,
                    device=args.device,
                )
    elif args.command == "candidates":
        from .evaluation import prepare_candidates

        for dataset in datasets:
            prepare_candidates(args.data_dir, args.output_dir, args.model, dataset)
    elif args.command == "evaluate":
        from .evaluation import evaluate

        for dataset in datasets:
            evaluate(
                args.data_dir,
                args.output_dir,
                dataset,
                args.methods,
                device=args.device,
                robust=not args.clean_only,
            )
    elif args.command == "report":
        from .evaluation import report

        print(json.dumps(report(args.output_dir, args.data_dir), indent=2))
    elif args.command == "motivation":
        from .motivation import feature_experiment, motivation_report

        if args.stage == "features":
            for dataset in datasets:
                feature_experiment(args.data_dir, args.output_dir, dataset, args.model)
        else:
            print(json.dumps(motivation_report(args.data_dir, args.output_dir, datasets), indent=2))
    elif args.command == "infer":
        from .metrics import decision_boundary
        from .model import predict_checkpoint

        cp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        logits = predict_checkpoint(
            args.checkpoint,
            [{"raw_source": args.source_file.read_text(encoding="utf-8")}],
            device=args.device,
            model_path=args.model,
        )
        score = float(logits.sigmoid()[0])
        threshold = cp["decision_threshold"]
        print(
            json.dumps(
                dict(
                    logit=float(logits[0]),
                    probability=score,
                    threshold=threshold,
                    prediction=int(score >= decision_boundary(threshold)),
                )
            )
        )


if __name__ == "__main__":
    main()
