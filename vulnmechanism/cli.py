"""Prepare native datasets, train six methods, evaluate, report and infer."""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def build_parser():
    from .training import METHODS

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="independently clean and balance raw datasets")
    prepare.add_argument("--raw-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, default=Path("data"))
    prepare.add_argument("--dataset", choices=("primevul", "megavul", "all"), default="all")
    for command in ("train", "candidates", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--data-dir", type=Path, default=Path("data"))
        p.add_argument("--output-dir", type=Path, default=Path("results"))
        if command in ("train", "candidates"):
            p.add_argument("--model", default="microsoft/codebert-base")
        if command in ("train", "evaluate"):
            p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
            p.add_argument("--device", default="auto")
        if command in ("candidates", "evaluate"):
            p.add_argument("--split", choices=("valid", "test", "all"), default="all")
        if command == "evaluate":
            p.add_argument("--scope", choices=("primevul", "megavul", "all"), default="all")
            p.add_argument("--clean-only", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
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

        for dataset in ("primevul", "megavul") if args.dataset == "all" else (args.dataset,):
            prepare(args.raw_root, args.output_dir, dataset)
        return
    import torch

    torch.set_num_threads(4)
    if args.command == "train":
        from .training import prepare_materials, train

        materials = prepare_materials(args.data_dir / "primevul", args.output_dir)
        for method in args.methods:
            train(
                args.data_dir / "primevul",
                args.output_dir,
                args.model,
                method,
                device=args.device,
                materials=materials,
            )
    elif args.command == "candidates":
        from .evaluation import prepare_candidates

        splits = ("valid", "test") if args.split == "all" else (args.split,)
        prepare_candidates(args.data_dir / "primevul", args.output_dir, args.model, splits=splits)
    elif args.command == "evaluate":
        from .evaluation import evaluate_primevul, evaluate_megavul

        if args.scope in ("primevul", "all"):
            splits = ("valid", "test") if args.split == "all" else (args.split,)
            evaluate_primevul(
                args.data_dir / "primevul",
                args.output_dir,
                args.methods,
                device=args.device,
                splits=splits,
                robust=not args.clean_only,
            )
        if args.scope in ("megavul", "all"):
            if args.scope == "megavul" and args.split == "test":
                raise ValueError("MegaVul Test is reserved; only Valid evaluation is supported")
            evaluate_megavul(
                args.data_dir / "megavul",
                args.output_dir,
                args.methods,
                device=args.device,
            )
    elif args.command == "report":
        from .evaluation import report

        print(json.dumps(report(args.output_dir), indent=2))
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
