from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    if args.variant == 'codebert_source' and not args.preserve_members:
        raise ValueError('CodeBERT comparison requires --preserve-members to retain the original cohort')
    with dataset_view(args.dataset, args.source_dataset, preserve_members=args.preserve_members) as dataset:
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
            classifier_head=args.classifier_head,
            finetuning_mode=args.finetuning_mode,
            excluded_groups=args.exclude_groups,
            seed=args.seed,
            device=args.device,
            log_every=args.log_every,
        )


def _evaluate(args):
    if args.output and Path(args.output).exists():
        raise FileExistsError(args.output)
    expected_split = "external_test" if args.source_dataset == "sven" else args.split
    if args.source_dataset == "sven" and args.split != "external_test":
        raise ValueError("SVEN evaluation must use --split external_test")
    if args.source_dataset != "sven" and args.split == "external_test":
        raise ValueError("external_test is reserved for SVEN")
    with dataset_view(args.dataset, args.source_dataset, preserve_members=args.preserve_members) as dataset:
        result = evaluate_model(
            dataset,
            args.checkpoint,
            split=expected_split,
            batch_size=args.batch_size,
            device=args.device,
            prediction_path=args.predictions,
        )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


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
    train.add_argument("--preserve-members", action="store_true", help="keep original PrimeVul members and order without post-build rebalancing")
    _fit_arguments(train)
    train.add_argument("--context-max-length", type=int, default=384)
    train.add_argument("--fusion-dim", type=int, default=256)
    train.add_argument("--fusion-heads", type=int, default=8)
    train.add_argument("--classifier-head", choices=("linear", "mlp4", "prototype4"), default="linear")
    train.add_argument('--finetuning-mode', choices=('lora', 'full'),
                       help='matched CodeBERT BCE modes; omitted retains historical LoRA behavior')
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
    evaluate.add_argument("--predictions", help="save per-function probabilities and raw logits; refuses overwrite")
    evaluate.add_argument("--output", help="save evaluation metrics JSON; refuses overwrite")
    evaluate.add_argument("--preserve-members", action="store_true", help="evaluate original PrimeVul members without post-build rebalancing")
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

    change = sub.add_parser("change-learning", help="matched source-only invariance/real-repair ranking ablations")
    change.add_argument("--stage", choices=("prepare", "train", "eval", "report"), required=True)
    change.add_argument("--manifest", default="data/benchmark/manifest_sven_commit_disjoint.jsonl")
    change.add_argument("--raw-dir", default="/home/PublicData/PHY-data/vul_detect/data")
    change.add_argument("--data-dir", default="data/change_learning_seed42")
    change.add_argument("--output-dir", default="results/change_learning_seed42")
    change.add_argument("--source-dataset", choices=("primevul", "cleanvul"), default="primevul")
    change.add_argument("--encoder", choices=("codebert", "qwen"), default="codebert")
    from .change_learning import OBJECTIVES
    change.add_argument("--objective", choices=OBJECTIVES, help="default: all five matched ablations")
    change.add_argument("--split", choices=("valid", "test", "external_test"), default="valid")
    _fit_arguments(change)
    change.set_defaults(model=None, source_max_length=512, device="cuda:0", batch_size=1,
                        gradient_accumulation=8)
    change.add_argument("--invariance-weight", type=float, default=1.0)
    change.add_argument("--ranking-weight", type=float, default=1.0)
    change.add_argument("--smoke-samples", type=int, default=0, help="explicit diagnostic subset; forbids Test/SVEN evaluation")
    change.set_defaults(func=_change_learning)

    local = sub.add_parser('local-risk', help='repair-guided line risk learning, matched A/B/C/D controls')
    local.add_argument('--stage', choices=('prepare', 'train', 'eval', 'report'), required=True)
    local.add_argument('--data-dir', default='data/change_learning_seed42')
    local.add_argument('--output-dir', default='results/local_risk_seed42')
    local.add_argument('--source-dataset', choices=('primevul', 'cleanvul'), default='primevul')
    local.add_argument('--encoder', choices=('codebert', 'qwen'), default='codebert')
    from .change_learning import LOCAL_OBJECTIVES
    local.add_argument('--objective', choices=LOCAL_OBJECTIVES, help='default: A/B/C/D; C_random is opt-in')
    local.add_argument('--split', choices=('valid', 'test', 'external_test'), default='valid')
    _fit_arguments(local)
    local.set_defaults(model=None, source_max_length=2048, device='cuda:0', local_risk=True)
    local.add_argument('--smoke-samples', type=int, default=0)
    local.set_defaults(func=_change_learning)

    finetuning = sub.add_parser('codebert-finetuning', help='fixed six-run PrimeVul LoRA/full comparison, Train/Valid only')
    finetuning.add_argument('--manifest', default='data/benchmark/manifest_sven_commit_disjoint.jsonl')
    finetuning.add_argument('--model', default='/home/PublicData/PHY-data/resource/codebert-base')
    finetuning.add_argument('--output-dir', default='results/codebert_finetuning_seed42')
    finetuning.add_argument('--stage', choices=('smoke', 'train', 'report', 'all'), default='all')
    finetuning.add_argument('--finetuning-mode', choices=('lora', 'full'), help='train one fixed three-LR family')
    finetuning.add_argument('--device', default='cuda:0')
    finetuning.add_argument('--batch-size', type=int, default=4)
    finetuning.add_argument('--gradient-accumulation', type=int, default=2)
    finetuning.set_defaults(func=_codebert_finetuning)
    return parser


def _codebert_finetuning_smoke(rows, root, common):
    import torch
    from transformers import AutoTokenizer
    from . import model as core
    from .change_learning import write_json
    train = [r for r in rows if r['split'] == 'train']
    chosen = [r for y in (0, 1) for r in [x for x in train if x['label'] == y][:4]]
    chosen += [next(r for r in rows if r['split'] == 'valid' and r['label'] == y) for y in (0, 1)]
    builder = core.InputBuilder(AutoTokenizer.from_pretrained(common['model_path']),
                                source_max_length=2048, context_max_length=384)
    report, heads = {}, []
    for mode in ('lora', 'full'):
        target = root/'smoke'/mode/'best.pt'
        if target.exists():
            raise FileExistsError(f'smoke already exists: {target}; reuse its completed smoke.metrics.json')
        core._seed_everything(42)
        fresh = core._build_model('codebert_source', common['model_path'], device=torch.device(common['device']),
            lora_r=16, lora_alpha=32, lora_dropout=.05,
            target_modules=('attention.self.query', 'attention.self.key', 'attention.self.value', 'attention.output.dense'),
            gradient_checkpointing=False, fusion_dim=256, fusion_heads=8, finetuning_mode=mode)
        heads.append(core._cpu_state(fresh.task_modules.state_dict()))
        before = {name: p.detach().cpu().clone() for name, p in fresh.encoder.named_parameters()}
        trainable = {name for name, p in fresh.encoder.named_parameters() if p.requires_grad}
        if mode == 'lora':
            assert trainable and all('lora_' in n for n in trainable)
        else:
            assert len(trainable) == len(before) and not any('lora_' in n for n in before)
        del fresh
        options = dict(common, records=chosen, epochs=1, batch_size=2, gradient_accumulation=4,
                       learning_rate=2e-5, finetuning_mode=mode)
        cp = core.train_model(None, target, **options)
        restored = core._load_model(cp, device=torch.device(common['device']))
        after = dict(restored.encoder.named_parameters())
        changed = {name for name in before if not torch.equal(before[name], after[name].detach().cpu())}
        assert changed and changed <= trainable
        assert any(not torch.equal(heads[-1][k], cp['task_state'][k]) for k in heads[-1])
        if mode == 'full':
            assert any('word_embeddings.weight' in n for n in changed)
            assert any('attention.self.query.weight' in n for n in changed)
        expected = core._predict_logits(restored, chosen[-2:], builder, variant='codebert_source',
            excluded_groups=(), batch_size=2, device=torch.device(common['device']))
        del restored, after, before
        actual = core.predict_checkpoint(target, chosen[-2:], batch_size=2, device=common['device'])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        report[mode] = dict(train_functions=8, valid_functions=2, trainable_encoder_tensors=len(trainable),
            changed_encoder_tensors=len(changed), frozen_encoder_unchanged=True, classifier_updated=True,
            save_reload_exact=True, trainable_parameters=cp['trainable_parameters'])
    assert all(torch.equal(heads[0][k], heads[1][k]) for k in heads[0])
    report['initial_heads_identical'] = True
    write_json(root/'smoke.metrics.json', report)
    return report


def _finetuning_evaluate(rows, run, split, checkpoint, args):
    from . import model as core
    from .cfg_data import identity
    from .change_learning import write_json
    output, predictions = run/f'{split}.metrics.json', run/f'{split}.predictions.jsonl'
    if output.exists() != predictions.exists():
        raise FileExistsError(f'incomplete evaluation artifacts: {run}/{split}; inspect before rerunning')
    if output.exists():
        cached = [json.loads(l) for l in predictions.open()]
        expected = [identity(r) for r in rows if r['split'] == split]
        if expected != [{k:r[k] for k in expected[0]} for r in cached]:
            raise ValueError(f'cached prediction members/source differ: {predictions}')
        if any(r['threshold'] != checkpoint['decision_threshold'] for r in cached):
            raise ValueError('cached evaluation thresholds differ')
        result = json.loads(output.read_text())
        if result.get('selected_epoch') != checkpoint['selected_epoch']:
            raise ValueError('cached selected epoch differs')
    else:
        result = core.evaluate_model(None, run/'best.pt', records=rows, split=split, device=args.device,
                                    batch_size=args.batch_size, prediction_path=predictions)
        result.update(finetuning_mode=checkpoint['finetuning_mode'],
                      learning_rate=checkpoint['training_config']['learning_rate'],
                      selected_epoch=checkpoint['selected_epoch'])
        write_json(output, result)
    return result


def _codebert_finetuning(args):
    import csv
    import math
    import torch
    import transformers
    from collections import Counter
    from . import model as core
    from .change_learning import write_json
    if args.batch_size * args.gradient_accumulation != 8:
        raise ValueError('CodeBERT comparison requires effective batch=8')
    rows = [r for r in core._read_jsonl(args.manifest, source_only=True)
            if r['dataset'] == 'primevul' and r['split'] in ('train', 'valid')]
    if not rows or any(r.get('repair_supplement') for r in rows):
        raise ValueError('requires the original pre-recovery PrimeVul classification manifest')
    counts = dict(Counter(f"{r['split']}/label_{r['label']}" for r in rows))
    members = {s: [(r['sample_key'], r['label']) for r in rows if r['split'] == s] for s in ('train', 'valid')}
    rates = {'lora': (2e-5, 1e-4, 2e-4), 'full': (1e-5, 2e-5, 5e-5)}
    root = Path(args.output_dir)
    config = dict(manifest=args.manifest, model=args.model, seed=42, source_max_length=2048,
        epochs=3, batch_size=args.batch_size, gradient_accumulation=args.gradient_accumulation,
        weight_decay=.01, gradient_clip=1., lora_r=16, lora_alpha=32, lora_dropout=.05,
        learning_rates={m: list(rs) for m, rs in rates.items()}, counts=counts,
        input='unchanged pre-recovery PrimeVul functions; no CPG success filtering; Train/Valid only',
        initialization='fresh common CodeBERT base; identical linear head; isolated LoRA initialization RNG',
        precision='FP32 weights/optimizer; BF16 CUDA autocast; FP32 head/pooling/loss',
        selection='Valid MCC then F1/accuracy/AUC; original threshold grid and tie-break',
        environment=dict(torch=torch.__version__, transformers=transformers.__version__))
    config_path = root/'config.metrics.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError('existing finetuning comparison configuration differs')
    else:
        write_json(config_path, config)
    common = dict(variant='codebert_source', model_path=args.model, source_max_length=2048,
        context_max_length=384, batch_size=args.batch_size, gradient_accumulation=args.gradient_accumulation,
        epochs=3, weight_decay=.01, lora_r=16, lora_alpha=32, lora_dropout=.05,
        seed=42, device=args.device, log_every=25)
    smoke_path = root/'smoke.metrics.json'
    if args.stage in ('smoke', 'all'):
        if smoke_path.exists():
            smoke = json.loads(smoke_path.read_text())
            if not smoke.get('initial_heads_identical') or not all(smoke[m]['save_reload_exact'] for m in rates):
                raise ValueError('existing smoke did not verify both modes')
        else:
            smoke = _codebert_finetuning_smoke(rows, root, common)
        if args.stage == 'smoke':
            return smoke
    if args.stage in ('train', 'all'):
        if not smoke_path.exists():
            raise ValueError('run --stage smoke successfully before full training')
        for mode in ([args.finetuning_mode] if args.finetuning_mode else rates):
            for lr in rates[mode]:
                run = root/mode/f'lr_{lr:g}'
                target = run/'best.pt'
                if target.exists():
                    cp = torch.load(target, map_location='cpu', weights_only=False)
                    expected = dict(finetuning_mode=mode, seed=42, source_max_length=2048, model_path=args.model,
                        cohort_members=members, classifier_head='linear', initial_checkpoint=None,
                        checkpoint_version=core._CHECKPOINT_VERSION, dataset_schema_version=core.DATASET_SCHEMA_VERSION,
                        variant='codebert_source', context_max_length=384, trained_on='primevul',
                        lora_r=16, lora_alpha=32, lora_dropout=.05, sample_weights=None, selection='validation_mcc',
                        encoder_precision=config['precision'], classifier_initialization=core._CLASSIFIER_INITIALIZATION,
                        training_config=dict(epochs=3, batch_size=args.batch_size,
                            gradient_accumulation=args.gradient_accumulation, learning_rate=lr, weight_decay=.01))
                    if any(cp.get(k) != v for k, v in expected.items()) or any(
                            k in cp for k in ('change_supervision', 'local_supervision', 'synthetic_supervision')):
                        raise ValueError(f'existing checkpoint does not match: {target}')
                    history = [json.loads(l) for l in target.with_suffix('.training.jsonl').read_text().splitlines()]
                    epochs = [r for r in history if r.get('event') == 'epoch']
                    if [r['epoch'] for r in epochs] != [1, 2, 3] or any(
                            r['optimizer_steps'] != math.ceil(len(members['train'])/8) for r in epochs):
                        raise ValueError(f'incomplete or different training budget: {target}')
                    print(f'Reusing completed matching run: {target}', flush=True)
                else:
                    cp = core.train_model(None, target, records=rows, **common,
                                          learning_rate=lr, finetuning_mode=mode)
                _finetuning_evaluate(rows, run, 'valid', cp, args)
    if args.stage not in ('report', 'all'):
        return dict(counts=counts, completed_family=args.finetuning_mode)
    results, selected = [], {}
    for mode, candidates in rates.items():
        family = []
        for lr in candidates:
            run = root/mode/f'lr_{lr:g}'
            cp = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
            if cp.get('cohort_members') != members or cp.get('finetuning_mode') != mode:
                raise ValueError('report checkpoint cohort/mode differs')
            result = _finetuning_evaluate(rows, run, 'valid', cp, args)
            family.append((run, result))
            results.append(dict(mode=mode, learning_rate=lr, selected_epoch=result['selected_epoch'],
                threshold=result['decision_threshold'], bce=result['bce'], **result['metrics']))
        run, valid = max(family, key=lambda item: tuple(item[1]['metrics'][k] for k in ('mcc','f1','accuracy','auc')))
        cp = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
        train = _finetuning_evaluate(rows, run, 'train', cp, args)
        selected[mode] = dict(checkpoint=str(run/'best.pt'), learning_rate=valid['learning_rate'],
            selected_epoch=valid['selected_epoch'], threshold=valid['decision_threshold'],
            train=train, valid=valid)
    summary = dict(config=config, candidates=results, selected=selected)
    write_json(root/'comparison.metrics.json', summary)
    with (root/'comparison.metrics.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    return summary


def _diagnose(args):
    from .diagnostic import run_diagnostic
    return run_diagnostic(args)


def _source_pilot(args):
    from . import source_pilot
    return source_pilot.prepare(args) if args.stage == "prepare" else source_pilot.run(args)


def _change_learning(args):
    from . import change_learning
    if args.stage == "prepare":
        result = (change_learning.prepare_local(args.data_dir) if getattr(args, 'local_risk', False)
                  else change_learning.prepare(args.manifest, args.raw_dir, args.data_dir))
    elif args.stage == "train":
        result = change_learning.train_local(args) if getattr(args, 'local_risk', False) else change_learning.train(args)
    elif args.stage == "eval":
        result = change_learning.evaluate(args)
    else:
        result = change_learning.report(args.output_dir, local_risk=getattr(args, 'local_risk', False))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return result


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
