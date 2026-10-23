"""Prepare, train and evaluate dynamic augmentation and official LOSVER."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from .data import DATASETS


def build_parser():
    from .training import METHODS, LOSVER_METHODS

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument(
        "--paired",
        action="store_true",
        help="official PrimeVul original pairs for LOSVER",
    )
    p.add_argument(
        "--model", default="/home/PublicData/PHY-data/resource/unixcoder-base-nine"
    )
    for command in ("train", "candidates", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--data-dir", type=Path, default=Path("data"))
        p.add_argument("--output-dir", type=Path, default=Path("results"))
        p.add_argument(
            "--dataset", choices=(*DATASETS, "primevul_paired", "all"), default="all"
        )
        if command in ("train", "candidates"):
            p.add_argument(
                "--model",
                help="defaults to the local pretrained model for the selected method",
            )
        if command in ("train", "evaluate"):
            p.add_argument(
                "--methods",
                nargs="+",
                choices=(*METHODS, *LOSVER_METHODS),
                default=list(METHODS),
            )
            p.add_argument("--device", default="auto")
        if command == "train":
            p.add_argument(
                "--hard-selection", choices=("hardest", "boundary"), default="hardest"
            )
            p.add_argument(
                "--vat-direction",
                choices=("finite_difference", "logit"),
                default="finite_difference",
            )
            p.add_argument("--train-limit", type=int)
        if command in ("train", "evaluate"):
            p.add_argument("--valid-limit", type=int)
        if command in ("candidates", "evaluate"):
            p.add_argument("--test-limit", type=int)
        if command == "evaluate":
            p.add_argument("--clean-only", action="store_true")
            p.add_argument("--split", choices=("valid", "test", "all"), default="all")
    p = sub.add_parser("report")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--dataset", choices=(*DATASETS, "primevul_paired", "all"), default="all"
    )
    p = sub.add_parser("infer")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source-file", type=Path, required=True)
    p.add_argument("--model")
    p.add_argument("--device", default="auto")
    p = sub.add_parser(
        "ablation", help="existing Hardest/Boundary and VAT direction diagnostics"
    )
    p.add_argument("stage", choices=("probe", "report"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--output-dir", type=Path, default=Path("results/search_vat_ablation")
    )
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--dataset", choices=DATASETS, default="primevul")
    p.add_argument("--sample-limit", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--train-limit", type=int, default=1024)
    p.add_argument("--valid-limit", type=int, default=256)
    p.add_argument("--test-limit", type=int, default=256)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "prepare":
        from .data import prepare, prepare_paired

        if args.paired:
            if args.datasets != ["primevul"]:
                raise ValueError("--paired requires --datasets primevul")
            prepare_paired(args.raw_root, args.output_dir, args.model)
        else:
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
        from .training import train, LOSVER_METHODS

        for dataset in datasets:
            for method in args.methods:
                path = args.model or (
                    "/home/PublicData/PHY-data/resource/unixcoder-base-nine"
                    if method in LOSVER_METHODS
                    else "/home/PublicData/PHY-data/resource/codebert-base"
                )
                train(
                    args.data_dir,
                    args.output_dir,
                    path,
                    dataset,
                    method,
                    device=args.device,
                    hard_selection=args.hard_selection,
                    vat_direction=args.vat_direction,
                    train_limit=args.train_limit,
                    valid_limit=args.valid_limit,
                )
    elif args.command == "candidates":
        from .evaluation import prepare_candidates

        if "primevul_paired" in datasets:
            raise ValueError("LOSVER uses the official clean paired evaluation")
        for dataset in datasets:
            prepare_candidates(
                args.data_dir,
                args.output_dir,
                args.model or "/home/PublicData/PHY-data/resource/codebert-base",
                dataset,
                test_limit=args.test_limit,
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
                splits=("valid", "test") if args.split == "all" else (args.split,),
            )
    elif args.command == "report":
        from .evaluation import report, paired_report

        print(
            json.dumps(
                (paired_report if args.dataset == "primevul_paired" else report)(
                    args.output_dir, args.data_dir
                ),
                indent=2,
            )
        )
    elif args.command == "infer":
        from .metrics import decision_boundary
        from .model import predict_checkpoint

        cp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        with args.source_file.open(encoding="utf-8", newline="") as f:
            source = f.read()
        logits = predict_checkpoint(
            args.checkpoint,
            [{"raw_source": source}],
            device=args.device,
            model_path=args.model,
        )
        score = float(logits.sigmoid()[0])
        threshold = cp["decision_threshold"]
        prediction = (
            score > threshold
            if cp.get("protocol") == "losver"
            else score >= decision_boundary(threshold)
        )
        print(
            json.dumps(
                dict(
                    logit=float(logits[0]),
                    probability=score,
                    threshold=threshold,
                    prediction=int(prediction),
                )
            )
        )
    elif args.command == "ablation":
        from .ablation import probe, comparison

        if args.stage == "probe":
            if args.checkpoint is None:
                raise ValueError("probe requires --checkpoint")
            probe(
                args.data_dir,
                args.output_dir,
                args.checkpoint,
                sample_limit=args.sample_limit,
                repeats=args.repeats,
                device=args.device,
            )
        else:
            comparison(
                args.data_dir,
                args.output_dir,
                args.dataset,
                train_limit=args.train_limit,
                valid_limit=args.valid_limit,
                test_limit=args.test_limit,
            )


if __name__ == "__main__":
    main()
