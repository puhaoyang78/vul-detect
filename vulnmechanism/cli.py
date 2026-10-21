"""Shared data preparation, training, evaluation and Motivation experiments."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from .data import DATASETS


def build_parser():
    from .training import METHODS, GENERALIZATION_METHODS

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
            p.add_argument("--methods", nargs="+", choices=(*METHODS, "lp_ft", "swa"), default=list(METHODS))
            p.add_argument("--device", default="auto")
        if command == "evaluate":
            p.add_argument("--clean-only", action="store_true")
        if command == "train":
            from .prsc import MASK_METHODS
            p.add_argument("--hard-selection", choices=("hardest", "boundary"), default="hardest")
            p.add_argument("--vat-direction", choices=("finite_difference", "logit"), default="finite_difference")
            p.add_argument("--representation-mask", choices=MASK_METHODS, default="none")
            p.add_argument("--mask-fraction", type=float, default=0.05)
            p.add_argument("--mask-weight", type=float, default=0.2)
            p.add_argument("--train-limit", type=int, help="balanced diagnostic subset; default: full Train")
        if command in ("train", "evaluate"):
            p.add_argument("--valid-limit", type=int, help="balanced diagnostic subset; default: full Valid")
        if command in ("candidates", "evaluate"):
            p.add_argument("--test-limit", type=int, help="balanced diagnostic subset; default: full Test")
    p = sub.add_parser("transfer", help="bidirectional Clean transfer with source-selected checkpoints and thresholds")
    p.add_argument("--datasets", nargs=2, choices=DATASETS, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--device", default="auto")
    p.add_argument("--methods", nargs="+", choices=GENERALIZATION_METHODS, default=["bce"])
    p = sub.add_parser("report", help="summarize the saved PrimeVul/MegaVul six-method experiment")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p = sub.add_parser("motivation", help="four-dataset ordinary-BCE and Shortcut Features diagnostics")
    p.add_argument("stage", choices=("features", "report", "tfidf", "tfidf-report", "transfer-diagnosis", "transfer-train-probe"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    p.add_argument("--model", default="microsoft/codebert-base")
    p.add_argument("--device", default="auto")
    p = sub.add_parser("infer")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source-file", type=Path, required=True)
    p.add_argument("--model", help="pretrained CodeBERT location if the checkpoint has moved")
    p.add_argument("--device", default="auto")
    p = sub.add_parser("ablation", help="real-checkpoint search/VAT diagnostics and four-combination report")
    p.add_argument("stage", choices=("probe", "report"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("results/search_vat_ablation"))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--dataset", choices=DATASETS, default="primevul")
    p.add_argument("--sample-limit", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--train-limit", type=int, default=1024)
    p.add_argument("--valid-limit", type=int, default=256)
    p.add_argument("--test-limit", type=int, default=256)
    p = sub.add_parser("prsc", help="source-Train representation diagnostic and matched Clean pilot report")
    p.add_argument("stage", choices=("probe", "report"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("results/prsc"))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--sample-limit", type=int, choices=(128, 256), default=256)
    p.add_argument("--device", default="auto")
    p = sub.add_parser("static", help="BigVul/Devign/ReVeal static-relation pilot; Test remains held out")
    p.add_argument("stage", choices=("prepare", "benchmark", "extract", "train", "report"))
    p.add_argument("--cache-root", type=Path, help="optional existing Joern cache root for incremental parsing")
    p.add_argument("--raw-root", type=Path, default=Path("/home/PublicData/PHY-data/vul_detect/data"))
    p.add_argument("--data-dir", type=Path, default=Path("data/static_guided"))
    p.add_argument("--output-dir", type=Path, default=Path("results/static_guided"))
    p.add_argument("--dataset", choices=("bigvul", "devign", "reveal", "all"), default="all")
    p.add_argument("--modes", nargs="+", choices=("source", "sequence", "shuffled", "real"), default=["source", "sequence", "shuffled", "real"])
    p.add_argument("--model", default="/home/PublicData/PHY-data/resource/codebert-base")
    p.add_argument("--device", default="auto")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "static":
        from .static_experiment import run
        run(args)
        return
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
                    hard_selection=args.hard_selection,
                    vat_direction=args.vat_direction,
                    train_limit=args.train_limit,
                    valid_limit=args.valid_limit,
                    representation_mask=args.representation_mask,
                    mask_fraction=args.mask_fraction,
                    mask_weight=args.mask_weight,
                )
    elif args.command == "candidates":
        from .evaluation import prepare_candidates

        for dataset in datasets:
            prepare_candidates(
                args.data_dir, args.output_dir, args.model, dataset, test_limit=args.test_limit
            )
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
                valid_limit=args.valid_limit,
                test_limit=args.test_limit,
            )
    elif args.command == "transfer":
        from .evaluation import evaluate_transfer

        print(json.dumps(evaluate_transfer(
            args.data_dir, args.output_dir, args.datasets, methods=args.methods, device=args.device
        ), indent=2))
    elif args.command == "report":
        from .evaluation import report

        print(json.dumps(report(args.output_dir, args.data_dir), indent=2))
    elif args.command == "motivation":
        from .motivation import feature_experiment, motivation_report, tfidf_experiment, tfidf_report, transfer_diagnostics, transfer_train_probe

        if args.stage == "transfer-train-probe":
            print(json.dumps(transfer_train_probe(args.data_dir, args.output_dir, args.model, args.device), indent=2))
        elif args.stage == "transfer-diagnosis":
            print(json.dumps(transfer_diagnostics(args.data_dir, args.output_dir, args.model), indent=2))
        elif args.stage == "features":
            for dataset in datasets:
                feature_experiment(args.data_dir, args.output_dir, dataset, args.model)
        elif args.stage == "report":
            print(json.dumps(motivation_report(args.data_dir, args.output_dir, datasets), indent=2))
        else:
            if args.stage == "tfidf":
                for dataset in datasets:
                    tfidf_experiment(args.data_dir, args.output_dir, dataset)
            print(json.dumps(tfidf_report(args.output_dir, datasets), indent=2))
    elif args.command == "ablation":
        from .ablation import comparison, probe

        if args.stage == "probe":
            if args.checkpoint is None:
                raise ValueError("ablation probe requires --checkpoint")
            probe(args.data_dir, args.output_dir, args.checkpoint, sample_limit=args.sample_limit,
                  repeats=args.repeats, device=args.device)
        else:
            comparison(args.data_dir, args.output_dir, args.dataset, train_limit=args.train_limit,
                       valid_limit=args.valid_limit, test_limit=args.test_limit)
    elif args.command == "prsc":
        from . import prsc

        if args.stage == "probe":
            if args.checkpoint is None:
                raise ValueError("prsc probe requires --checkpoint")
            prsc.probe(args.data_dir, args.output_dir, args.checkpoint,
                       sample_limit=args.sample_limit, device=args.device)
        else:
            prsc.comparison(args.data_dir, args.output_dir)
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
