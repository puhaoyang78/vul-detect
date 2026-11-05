"""Prepare, train and evaluate dynamic augmentation and LOSVER."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from .data import DATASETS, RGCL_DATASETS


def build_parser():
    from .training import METHODS, LOSVER_METHODS, RGCL_METHODS

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    p.add_argument("--datasets", nargs="+", choices=(*DATASETS, "secvuleval"), default=list(DATASETS))
    supervision = p.add_mutually_exclusive_group()
    supervision.add_argument(
        "--rgcl", action="store_true", help="native binary labels and leakage-grouped SecVulEval/MegaVul splits"
    )
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
            "--dataset", choices=(*DATASETS, "secvuleval", "cleanvul", "primevul_paired", "all"), default="all"
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
                choices=(*METHODS, *LOSVER_METHODS, *RGCL_METHODS),
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
                           help="512 UniXcoder line MIL with token-count weights (default K=5)")
            p.add_argument("--attention-max-rate", type=int, choices=(5, 10), default=5,
                           help="bounded attention upper multiplier; 10 for the line-region experiment")
            p.add_argument("--attention-region-weighting", choices=("token", "line"), default="token",
                           help="region-loss averaging only; function pooling keeps token-count weights")
            p.add_argument("--attention-context", action="store_true",
                           help="condition bounded line scores on the valid code-token mean, rank 32")
            p.add_argument("--attention-local", action="store_true",
                           help="add learned physical-line neighbors within +/-3; requires attention context")
            p.add_argument("--attention-context-scale", action="store_true",
                           help="function-adaptive context_scale: 2*sigmoid(Linear(non-affine LayerNorm(g)))")
            p.add_argument("--local-mil", choices=("mean", "lse"),
                           help="zero-head additive two-class line MIL on Context + Diff; fixed temperature 1")
            p.add_argument("--insertion-supervision", choices=("diff", "random"),
                           help="fixed .1 function-normalized gap ranking loss on original Context + Diff")
            p.add_argument("--rccl", choices=("baseline", "diff", "random"),
                           help="paired-batch Context + Diff control or fixed .05 repair-context contrastive loss")
            p.add_argument("--attention-region", choices=("diff", "random"),
                           help="fixed 0.1 log(2) region constraint; requires bounded attention")
            p.add_argument("--attention-suppression", choices=("random", "high"),
                           help="training only: set floor(.1*N) line multipliers to 1; mean two BCE")
            p.add_argument("--attention-ema", action="store_true",
                           help="training only: encoder+attention EMA(.99), valid-line KL weight .1")
            p.add_argument("--attention-diagnostics", action="store_true",
                           help="fixed Train/Valid eval cohorts each epoch; preserve training RNG")
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
            p.add_argument("--paired-seed",type=int,help="paired protocol seed; default 123456")
            p.add_argument("--marginalize-inputs",action="store_true",
                           help="Train-only conditional BCE targets for exactly identical 512 detector inputs")
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
    p.add_argument("--rgcl", action="store_true", help="compare the three native-binary RGCL methods")
    p.add_argument("--bounded-attention",action="store_true")
    p.add_argument("--attention-context",action="store_true")
    p.add_argument("--attention-local",action="store_true")
    p.add_argument("--attention-context-scale",action="store_true")
    p.add_argument("--attention-region-weighting",choices=("token","line"),default="token",
                   help="report equal-line region loss against historical Context + Diff")
    p.add_argument("--attention-max-rate",type=int,choices=(5,10),default=10,
                   help="primary equal-line report: K=5 loss-only control or historical K=10 comparison")
    p.add_argument("--insertion-localization", action="store_true")
    p.add_argument("--rccl", action="store_true", help="report the three matched RCCL experiments")
    p.add_argument("--attention-regularization",action="store_true")
    p.add_argument("--device",default="auto")
    p.add_argument(
        "--dataset", choices=(*DATASETS, "secvuleval", "cleanvul", "primevul_paired", "all"), default="all"
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
    p = sub.add_parser("repair-transfer", help="frozen cross-sample repair-region retrieval; Train/Valid only")
    p.add_argument("stage", choices=("extract", "retrieve", "report"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--audit-dir", type=Path, default=Path("results/primevul_targeted_slicing"))
    p.add_argument("--checkpoint", type=Path, default=Path("results/attention_context_512/context_diff/primevul_paired/unixcoder/best-p_c.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("results/repair_transfer_512"))
    p.add_argument("--device", default="auto")
    p = sub.add_parser("rgcl-study", help="fixed RGCL comparisons and repair-pair data inspection")
    p.add_argument("stage", choices=("inventory", "check", "run", "report", "inspect", "prepare"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("results/rgcl_multiseed_512"))
    p.add_argument("--model", default="/home/PublicData/PHY-data/resource/unixcoder-base-nine")
    p.add_argument("--dataset", choices=("primevul_paired", "secvuleval", "reposvul", "cleanvul"), default="primevul_paired")
    p.add_argument("--raw-root", type=Path, help="raw dataset directory for inspection/preparation")
    p.add_argument("--official-dir", type=Path, help="original official Train/Valid/Test JSONL directory")
    p.add_argument("--seed", type=int, choices=(42, 1234, 123456), default=123456)
    p.add_argument("--methods", nargs="+", choices=RGCL_METHODS, default=list(RGCL_METHODS))
    p.add_argument("--device", default="auto")
    p.add_argument("--save-every", type=int, default=500)
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
    p.add_argument("--dataset", choices=(*RGCL_DATASETS, "primevul_paired"), default="primevul_paired")
    p.add_argument(
        "--repair-regions",
        action="store_true",
        help="check weak supervision and bounded fusion on source Train",
    )
    p.add_argument("--bounded-attention",action="store_true",
                   help="real UniXcoder bounded-pooling/gradient/save-reload checks on Train only")
    p.add_argument("--attention-max-rate",type=int,choices=(5,10),default=5)
    p.add_argument("--attention-region-weighting",choices=("token","line"),default="token")
    p.add_argument("--attention-context",action="store_true",
                   help="also check rank-32 context interaction and its initialization")
    p.add_argument("--attention-local",action="store_true",
                   help="also check local/global attention and physical-line neighborhoods")
    p.add_argument("--attention-context-scale",action="store_true",
                   help="also check context_scale initialization, gradients and fixed-alpha inference")
    p.add_argument("--insertion-localization", action="store_true",
                   help="check pure-addition gaps, shared-encoder ranking and single-source inference")
    p.add_argument("--rccl", action="store_true", help="RCCL Train/Valid coverage and real GPU checks")
    p.add_argument("--attention-regularization",action="store_true",
                   help="also check real suppression and full-encoder EMA gradients")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "prepare":
        from .data import prepare, prepare_paired, prepare_repair_regions, prepare_diff_neighbour

        if args.rgcl:
            from .data import prepare_rgcl
            for dataset in args.datasets:
                if dataset not in RGCL_DATASETS:
                    raise ValueError("--rgcl preparation requires secvuleval or megavul")
                prepare_rgcl(args.raw_root, args.output_dir, dataset, args.model)
        elif args.oof_predictions:
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
        from .training import train, LOSVER_METHODS, RGCL_METHODS

        for dataset in datasets:
            for method in args.methods:
                path = args.model or (
                    "/home/PublicData/PHY-data/resource/unixcoder-base-nine"
                    if method in (*LOSVER_METHODS, *RGCL_METHODS)
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
                    paired_seed=args.paired_seed,
                    marginalize_inputs=args.marginalize_inputs,
                    localizer_dropout=args.localizer_dropout,
                    localizer_weight_decay=args.localizer_weight_decay,
                    localizer_rdrop=args.localizer_rdrop,
                    line_auxiliary=args.line_auxiliary,
                    line_auxiliary_weight=args.line_auxiliary_weight,
                    oof_fold=args.oof_fold,
                    localizer_dir=args.localizer_dir,
                    train_localizer_dir=args.train_localizer_dir,
                    bounded_attention=args.bounded_attention,
                    attention_context=args.attention_context,
                    attention_local=args.attention_local,
                    attention_context_scale=args.attention_context_scale,
                    insertion_supervision=args.insertion_supervision,
                    rccl=args.rccl,
                    local_mil=args.local_mil,
                    attention_region=args.attention_region,
                    attention_max_rate=args.attention_max_rate,
                    attention_region_weighting=args.attention_region_weighting,
                    attention_suppression=args.attention_suppression,
                    attention_ema=args.attention_ema,
                    attention_diagnostics=args.attention_diagnostics,
                )
    elif args.command == "repair-transfer":
        from . import repair_transfer
        if args.stage == "report":
            print(json.dumps(repair_transfer.report(args.output_dir), indent=2))
        else:
            getattr(repair_transfer, args.stage)(args.data_dir, args.audit_dir, args.checkpoint, args.output_dir, args.device)
    elif args.command == "rgcl-study":
        if args.dataset == "cleanvul" and args.stage in {"inspect", "prepare"}:
            if args.raw_root is None or args.seed != 42:
                raise ValueError("CleanVul inspect/prepare requires --raw-root and fixed split --seed 42")
            from . import cleanvul
            if args.stage == "inspect":
                result = cleanvul.inspect(args.raw_root, args.data_dir, args.output_dir, args.model)
            else:
                result = cleanvul.prepare(args.raw_root, args.data_dir, args.model)
            print(json.dumps(result, indent=2))
            return
        if args.stage == "prepare":
            raise ValueError("rgcl-study prepare is only for CleanVul repair pairs")
        if args.stage == "inspect":
            from .reposvul import inspect
            if args.dataset != "reposvul" or args.raw_root is None:
                raise ValueError("inspection requires --dataset reposvul and --raw-root")
            print(json.dumps(inspect(args.raw_root, args.output_dir, args.model,
                                     official_dir=args.official_dir), indent=2))
            return
        if args.dataset == "reposvul":
            if args.stage == "check" and args.official_dir is not None:
                from .reposvul import check
                print(json.dumps(check(args.official_dir, args.output_dir, device=args.device), indent=2))
                return
            raise ValueError("ReposVul is inspection-only until label and repair quality are resolved")
        from . import rgcl_study
        common = dict(data_dir=args.data_dir, output_dir=args.output_dir, model_path=args.model)
        selected = dict(datasets=("cleanvul",)) if args.dataset == "cleanvul" else {}
        if args.stage == "check":
            result = rgcl_study.check(**common, **selected, device=args.device)
        elif args.stage == "run":
            result = rgcl_study.run(**common, dataset=args.dataset, seed=args.seed, methods=args.methods,
                                     device=args.device, save_every=args.save_every)
        else:
            result = getattr(rgcl_study, args.stage)(**common, **selected)
        print(json.dumps(result, indent=2))
    elif args.command == "check-paired":
        from .evaluation import check_paired, check_repair_regions, check_bounded_attention, check_insertion_localization

        if args.dataset in RGCL_DATASETS:
            from .evaluation import check_rgcl
            print(json.dumps(check_rgcl(args.data_dir, args.output_dir, args.model,
                                       args.dataset, device=args.device), indent=2))
            return

        if args.rccl:
            from .rccl import check
            print(json.dumps(check(args.data_dir,args.output_dir,args.model,device=args.device),indent=2))
            return
        if args.insertion_localization:
            if args.paired_input != "512" or args.repair_regions or args.attention_regularization or args.attention_local or args.attention_context_scale:
                raise ValueError("insertion checks use original 512 Context + Diff only")
            print(json.dumps(check_insertion_localization(args.data_dir, args.output_dir, args.model, device=args.device), indent=2))
            return

        if args.repair_regions and args.bounded_attention:
            raise ValueError("choose one paired check")
        if (args.attention_max_rate != 5 or args.attention_region_weighting != "token") and not args.bounded_attention:
            raise ValueError("attention rate/region weighting checks require bounded attention")
        if args.attention_regularization and not args.bounded_attention:
            raise ValueError("attention regularization checks require bounded attention")
        if args.attention_context and (not args.bounded_attention or args.attention_regularization):
            raise ValueError("context checks require bounded attention without other regularization")
        if args.attention_local and not args.attention_context:
            raise ValueError("local checks require attention context")
        if args.attention_context_scale and (not args.attention_context or args.attention_local):
            raise ValueError("context_scale checks require context attention without local attention")

        print(
            json.dumps(
                (check_bounded_attention if args.bounded_attention else
                 check_repair_regions if args.repair_regions else check_paired)(
                    args.data_dir,
                    args.output_dir,
                    args.model,
                    device=args.device,
                    input_mode=args.paired_input,
                    **(dict(regularization=args.attention_regularization,
                            context=args.attention_context, local=args.attention_local,
                            context_scale=args.attention_context_scale,
                            max_rate=args.attention_max_rate,
                            region_weighting=args.attention_region_weighting) if args.bounded_attention else {}),
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

        if args.rgcl:
            from .evaluation import rgcl_report
            print(json.dumps(rgcl_report(args.output_dir, args.data_dir, args.dataset), indent=2))
            return

        if args.rccl:
            from .rccl import report as rccl_report
            print(json.dumps(rccl_report(args.output_dir,args.data_dir,device=args.device),indent=2))
            return

        print(
            json.dumps(
                paired_report(args.output_dir, args.data_dir, reference_dir=args.reference_dir,
                              localizer_regularization=args.localizer_regularization,
                              bounded_attention=args.bounded_attention,
                              attention_context=args.attention_context,
                              attention_local=args.attention_local,
                              attention_context_scale=args.attention_context_scale,
                              attention_region_weighting=args.attention_region_weighting,
                              attention_max_rate=args.attention_max_rate,
                              insertion_localization=args.insertion_localization,
                              attention_regularization=args.attention_regularization,device=args.device)
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
            if cp.get("protocol") in ("losver", "rgcl_native_binary")
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
