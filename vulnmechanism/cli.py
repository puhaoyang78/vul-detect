"""Prepare, train and evaluate dynamic augmentation and LOSVER."""

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
    supervision = p.add_mutually_exclusive_group()
    supervision.add_argument(
        "--paired",
        action="store_true",
        help="official PrimeVul original pairs for LOSVER",
    )
    supervision.add_argument(
        "--repair-regions",
        action="store_true",
        help="generate Train/Valid weak supervision without touching original data",
    )
    supervision.add_argument(
        "--diff-neighbour", action="store_true",
        help="supplement empty-side Diff labels with visible edit-adjacent lines",
    )
    supervision.add_argument(
        "--oof-predictions", action="store_true",
        help="check and merge the two completed Pilot localizers' held-out predictions",
    )
    p.add_argument("--reference-dir", type=Path)
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
                "--resume", action="store_true",
                help="resume last.pt, skip completed methods, start unstarted methods",
            )
            p.add_argument(
                "--save-every", type=int, default=100,
                help="save training state every N optimizer steps (also at epoch boundaries)",
            )
            p.add_argument("--paired-input", choices=("512", "l1"), default="512")
            p.add_argument("--window-micro-batch", type=int)
            p.add_argument("--function-micro-batch", type=int)
            p.add_argument("--line-auxiliary", choices=("diff", "random"))
            p.add_argument("--line-auxiliary-weight", type=float)
            p.add_argument("--bounded-attention", action="store_true",
                           help="512 UniXcoder line MIL with token-count weights and fixed K=5")
            p.add_argument("--attention-region", choices=("diff", "random"),
                           help="fixed 0.1 log(2) region constraint; requires bounded attention")
            p.add_argument("--oof-fold", type=int, choices=(0, 1),
                           help="held-out Train pair fold; train the original localizer on the other fold")
            p.add_argument("--localizer-dir", type=Path,
                           help="completed localizer directory for Valid, inference and default Train predictions")
            p.add_argument("--train-localizer-dir", type=Path,
                           help="override only the detector's Train prediction directory")
            p.add_argument("--localizer-dropout", type=float)
            p.add_argument("--localizer-weight-decay", type=float)
            p.add_argument(
                "--localizer-rdrop", type=float,
                help="two stochastic Focal forwards plus lambda * line KL; 0 keeps the two-forward control; omit for one forward",
            )
            p.add_argument(
                "--hard-selection", choices=("hardest", "boundary"), default="hardest"
            )
            p.add_argument(
                "--vat-direction",
                choices=("finite_difference", "logit"),
                default="finite_difference",
            )
            p.add_argument("--train-limit", type=int)
            p.add_argument(
                "--paired-epochs",
                type=int,
                help="fixed pilot budget; omit for unchanged 40/16 epochs",
            )
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
        "--reference-dir", type=Path,
        help="completed original LOSVER pilot used as the comparison reference",
    )
    p.add_argument("--localizer-regularization", action="store_true")
    p.add_argument("--bounded-attention",action="store_true")
    p.add_argument("--device",default="auto")
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
    p = sub.add_parser(
        "check-paired", help="input coverage and tiny GPU/training-loop checks"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("results/losver_checks"))
    p.add_argument(
        "--model", default="/home/PublicData/PHY-data/resource/unixcoder-base-nine"
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--paired-input", choices=("512", "l1"), default="512")
    p.add_argument(
        "--repair-regions",
        action="store_true",
        help="check weak supervision and bounded fusion on source Train",
    )
    p.add_argument("--bounded-attention",action="store_true",
                   help="real UniXcoder bounded-pooling/gradient/save-reload checks on Train only")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "prepare":
        from .data import prepare, prepare_paired, prepare_repair_regions, prepare_diff_neighbour

        if args.oof_predictions:
            from .data import prepare_oof_predictions
            prepare_oof_predictions(args.raw_root, args.output_dir, args.reference_dir)
        elif args.diff_neighbour:
            prepare_diff_neighbour(args.raw_root, args.output_dir, args.model)
        elif args.repair_regions:
            prepare_repair_regions(args.raw_root, args.output_dir)
        elif args.paired:
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
                    paired_input=args.paired_input,
                    window_micro_batch=args.window_micro_batch,
                    function_micro_batch=args.function_micro_batch,
                    resume=args.resume,
                    save_every=args.save_every,
                    paired_epochs=args.paired_epochs,
                    localizer_dropout=args.localizer_dropout,
                    localizer_weight_decay=args.localizer_weight_decay,
                    localizer_rdrop=args.localizer_rdrop,
                    line_auxiliary=args.line_auxiliary,
                    line_auxiliary_weight=args.line_auxiliary_weight,
                    oof_fold=args.oof_fold,
                    localizer_dir=args.localizer_dir,
                    train_localizer_dir=args.train_localizer_dir,
                    bounded_attention=args.bounded_attention,
                    attention_region=args.attention_region,
                )
    elif args.command == "check-paired":
        from .evaluation import check_paired, check_repair_regions, check_bounded_attention

        if args.repair_regions and args.bounded_attention:
            raise ValueError("choose one paired check")

        print(
            json.dumps(
                (check_bounded_attention if args.bounded_attention else
                 check_repair_regions if args.repair_regions else check_paired)(
                    args.data_dir,
                    args.output_dir,
                    args.model,
                    device=args.device,
                    input_mode=args.paired_input,
                ),
                indent=2,
            )
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
                paired_report(args.output_dir, args.data_dir, reference_dir=args.reference_dir,
                              localizer_regularization=args.localizer_regularization,
                              bounded_attention=args.bounded_attention,device=args.device)
                if args.dataset == "primevul_paired"
                else report(args.output_dir, args.data_dir),
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
