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
    finetuning.add_argument('--output-dir', help='experiment-specific result directory')
    finetuning.add_argument('--stage', choices=('prepare', 'smoke', 'train', 'report', 'all'), default='all')
    finetuning.add_argument('--finetuning-mode', choices=('lora', 'full'), help='train one fixed three-LR family')
    shallow_method = finetuning.add_mutually_exclusive_group()
    shallow_method.add_argument('--shallow-reweighting', action='store_true', help='one full fine-tuning run with fixed Train-only shallow OOF weights')
    shallow_method.add_argument('--shallow-augmentation', nargs='?', const='both', choices=('both','random','high_low'),
        help='same-pool random/high-low semantic views; default: both new full fine-tuning runs')
    finetuning.add_argument('--baseline-checkpoint', default='results/codebert_finetuning_seed42/full/lr_1e-05/best.pt')
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


def _finetuning_evaluate(rows, run, split, checkpoint, args, *, output_name=None, checkpoint_path=None):
    from . import model as core
    from .cfg_data import identity
    from .change_learning import write_json
    name = output_name or split
    output, predictions = run/f'{name}.metrics.json', run/f'{name}.predictions.jsonl'
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
        result = core.evaluate_model(None, checkpoint_path or run/'best.pt', records=rows, split=split, device=args.device,
                                    batch_size=args.batch_size, prediction_path=predictions)
        result.update(finetuning_mode=checkpoint['finetuning_mode'],
                      learning_rate=checkpoint['training_config']['learning_rate'],
                      selected_epoch=checkpoint['selected_epoch'])
        write_json(output, result)
    return result


def _codebert_finetuning(args):
    if getattr(args, 'shallow_augmentation', None):
        return _codebert_shallow_augmentation(args)
    if getattr(args, 'shallow_reweighting', False):
        return _codebert_shallow_reweighting(args)
    if args.stage == 'prepare':
        raise ValueError('prepare requires a shallow experiment')
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
    root = Path(args.output_dir or 'results/codebert_finetuning_seed42')
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


def _reweighting_strata(items, source_budget, threshold=.5, *, oof=False):
    from .audit_statistics import summary
    groups = {}
    for name, lo, hi in [('<=128', 0, 128), ('129-512', 128, 512),
                          (f'513-{source_budget}', 512, source_budget),
                          (f'>{source_budget}', source_budget, float('inf'))]:
        subset = [r for r in items if lo < r['source_token_count'] <= hi]
        positive = sum(r['label'] for r in subset)
        negative = len(subset)-positive
        result = summary([r['label'] for r in subset],
                         [r['oof_probability' if oof else 'score'] for r in subset], threshold)
        result.update(positive=positive, negative=negative,
                      false_positive_rate=result.get('fp', 0)/negative if negative else None)
        if oof:
            result.update(hard_positive=sum(r['hard'] and r['label'] == 1 for r in subset),
                          hard_negative=sum(r['hard'] and r['label'] == 0 for r in subset))
        groups[name] = result
    return groups


def _matching_full_baseline(args):
    import math
    import torch
    from . import model as core
    if args.finetuning_mode not in (None, 'full'):
        raise ValueError('shallow experiments require the full fine-tuning baseline')
    rows = [r for r in core._read_jsonl(args.manifest, source_only=True)
            if r['dataset'] == 'primevul' and r['split'] in ('train', 'valid')]
    train = [r for r in rows if r['split'] == 'train']
    members = {s: [(r['sample_key'], r['label']) for r in rows if r['split'] == s] for s in ('train', 'valid')}
    baseline_path = Path(args.baseline_checkpoint)
    base = torch.load(baseline_path, map_location='cpu', weights_only=False)
    if (base.get('cohort_members') != members or any(r.get('repair_supplement') for r in rows)
            or base.get('finetuning_mode') != 'full' or base.get('variant') != 'codebert_source'
            or base.get('classifier_head') != 'linear' or base.get('sample_weights') is not None
            or base.get('initial_checkpoint') is not None or base.get('seed') != 42
            or base.get('source_max_length') != 2048 or base.get('model_path') != args.model
            or base.get('classifier_initialization') != core._CLASSIFIER_INITIALIZATION
            or any(k in base for k in ('change_supervision', 'local_supervision', 'synthetic_supervision', 'augmentation_config'))):
        raise ValueError('requires the matching fresh full-CodeBERT BCE baseline and original manifest')
    training = base['training_config']
    if (training['epochs'] != 3 or training['weight_decay'] != .01
            or training['batch_size'] != args.batch_size
            or training['gradient_accumulation'] != args.gradient_accumulation
            or args.batch_size*args.gradient_accumulation != 8):
        raise ValueError('training budget must match the existing full baseline exactly')
    history = [json.loads(l) for l in baseline_path.with_suffix('.training.jsonl').open()]
    budget = [(r['epoch'], r['optimizer_steps']) for r in history if r['event'] == 'epoch']
    if budget != [(i, math.ceil(len(train)/8)) for i in (1, 2, 3)]:
        raise ValueError('baseline did not complete the matching three-epoch budget')
    # Validate both saved cohorts/source identities; these artifacts already exist.
    for split in ('train', 'valid'):
        if not all((baseline_path.parent/f'{split}.{suffix}').exists()
                   for suffix in ('metrics.json', 'predictions.jsonl')):
            raise FileNotFoundError(f'missing saved baseline {split} evaluation')
        _finetuning_evaluate(rows, baseline_path.parent, split, base, args)
    return rows, base, budget


def _codebert_shallow_reweighting(args):
    import numpy as np
    import torch
    import sklearn
    import transformers
    from . import model as core, audit_statistics as shallow
    from .change_learning import write_json
    if args.stage == 'smoke':
        raise ValueError('shallow reweighting uses prepare/train/report/all stages')
    rows, base, budget = _matching_full_baseline(args)
    train = [r for r in rows if r['split'] == 'train']
    members = base['cohort_members']
    training, input_protocol = base['training_config'], base['input_protocol']
    baseline_path = Path(args.baseline_checkpoint)
    source_budget = input_protocol['source_budget']
    root = Path(args.output_dir or 'results/codebert_shallow_reweight_seed42')
    config = dict(manifest=args.manifest, baseline_checkpoint=str(baseline_path), model=base['model_path'],
        seed=42, finetuning_mode='full', training=training, input_protocol=input_protocol,
        encoder_precision=base['encoder_precision'], classifier_initialization=base['classifier_initialization'],
        weighting=dict(folds=3, threshold=.5, hard_weight=3., easy_weight=1.,
            normalization='divide by raw mean within each Train label',
            features='log1p visible tokens/chars/lines + lexical TF-IDF; fold-local scaler/vocabulary',
            grouping='transitive counterpart/pair identities and full/visible normalized source duplicates',
            tfidf=shallow.TFIDF_OPTIONS, logistic_regression=shallow.LR_OPTIONS),
        selection='original Valid MCC threshold/epoch selection and tie-break',
        loss='mean(weight * BCEWithLogits); fixed weights; no oversampling',
        environment=dict(torch=torch.__version__, transformers=transformers.__version__, sklearn=sklearn.__version__),
        train_functions=len(train), valid_functions=len(members['valid']))
    config = json.loads(json.dumps(config))
    config_path = root/'config.metrics.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('existing shallow reweighting configuration differs')
    if not config_path.exists():
        write_json(config_path, config)
    weight_path, oof_path = root/'oof.weights.jsonl', root/'oof.metrics.json'
    if weight_path.exists() != oof_path.exists():
        raise FileExistsError('incomplete OOF artifacts; inspect before rerunning')
    if weight_path.exists():
        cache = [json.loads(l) for l in weight_path.open()]
        if [(r['sample_key'], r['label']) for r in cache] != members['train']:
            raise ValueError('cached OOF training members differ')
        hard, raw, normalized = shallow.normalize_hard_weights(
            [r['label'] for r in cache], [r['oof_probability'] for r in cache])
        if (not np.array_equal(hard, [r['hard'] for r in cache])
                or not np.array_equal(raw, [r['raw_weight'] for r in cache])
                or not np.allclose(normalized, [r['weight'] for r in cache], rtol=0, atol=1e-12)):
            raise ValueError('cached weights disagree with fixed OOF rule')
        oof = json.loads(oof_path.read_text())
    else:
        from transformers import AutoTokenizer
        cache, oof = shallow.shallow_oof_weights(train, AutoTokenizer.from_pretrained(args.model), source_budget)
        usage = {r['sample_key']: r for r in (json.loads(l) for l in
                 (baseline_path.parent/'train.predictions.jsonl').open())}
        for r in cache:
            r.update(source_token_count=usage[r['sample_key']]['source_token_count'],
                     source_truncated=usage[r['sample_key']]['source_truncated'])
            if r['visible_tokens'] != usage[r['sample_key']]['visible_source_tokens']:
                raise ValueError('shallow visible budget differs from the encoder input')
        oof['length_groups'] = _reweighting_strata(cache, source_budget, oof=True)
        with weight_path.open('x') as f:
            for r in cache:
                f.write(json.dumps(r, ensure_ascii=False)+'\n')
        write_json(oof_path, oof)
    if args.stage == 'prepare':
        return dict(config=config, oof=oof)
    weights = {r['sample_key']: r['weight'] for r in cache}
    expected = {k: base[k] for k in ('checkpoint_version', 'dataset_schema_version', 'variant', 'model_path',
        'source_max_length', 'context_max_length', 'lora_r', 'lora_alpha', 'lora_dropout', 'seed',
        'finetuning_mode', 'classifier_head', 'classifier_initialization', 'encoder_precision',
        'cohort_members', 'initial_checkpoint', 'training_config', 'input_protocol', 'target_modules')}
    expected['sample_weights'] = weights
    del base
    target = root/'best.pt'
    if target.exists():
        cp = torch.load(target, map_location='cpu', weights_only=False)
        if any(cp.get(k) != v for k, v in expected.items()):
            raise ValueError('existing weighted checkpoint differs from the baseline protocol/weights')
        history = [json.loads(l) for l in target.with_suffix('.training.jsonl').open()]
        if [(r['epoch'], r['optimizer_steps']) for r in history if r['event'] == 'epoch'] != budget:
            raise ValueError('weighted training budget is incomplete')
    elif args.stage in ('train', 'all'):
        cp = core.train_model(None, target, records=rows, variant='codebert_source', model_path=args.model,
            source_max_length=2048, context_max_length=expected['context_max_length'],
            lora_r=expected['lora_r'], lora_alpha=expected['lora_alpha'], lora_dropout=expected['lora_dropout'],
            finetuning_mode='full', seed=42, device=args.device, log_every=25, sample_weights=weights, **training)
    else:
        raise FileNotFoundError(target)
    result = _finetuning_evaluate(rows, root, 'valid', cp, args)
    if cp['input_protocol'] != input_protocol:
        raise ValueError('weighted checkpoint input protocol differs')
    comparisons = {}
    for mode, folder, metrics in [('bce', baseline_path.parent, json.loads((baseline_path.parent/'valid.metrics.json').read_text())),
                                   ('weighted', root, result)]:
        predictions = [json.loads(l) for l in (folder/'valid.predictions.jsonl').open()]
        comparisons[mode] = dict(checkpoint=str(folder/'best.pt'), selected_epoch=metrics['selected_epoch'],
            threshold=metrics['decision_threshold'], bce=metrics['bce'], **metrics['metrics'],
            length_groups=_reweighting_strata(predictions, source_budget, metrics['decision_threshold']))
        short = [r for r in predictions if r['label'] == 1 and r['source_token_count'] <= 128]
        long = [r for r in predictions if r['label'] == 0 and r['source_token_count'] > 512]
        comparisons[mode]['short_positive'] = dict(samples=len(short), detected=sum(r['prediction'] for r in short),
            recall=sum(r['prediction'] for r in short)/len(short) if short else None, definition='source tokens <=128')
        comparisons[mode]['long_negative'] = dict(samples=len(long), false_positives=sum(r['prediction'] for r in long),
            false_positive_rate=sum(r['prediction'] for r in long)/len(long) if long else None,
            definition='source tokens >512, including truncated functions')
    report = dict(config=config, oof=oof, valid=comparisons)
    write_json(root/'comparison.metrics.json', report)
    return report


def _augmentation_sources(rows, entries, mode):
    from .syntax import apply_source_edits
    if [e['sample_key'] for e in entries] != [r['sample_key'] for r in rows]:
        raise ValueError('candidate cache members/order differ from the original Train cohort')
    sources, choices = {}, {}
    for row, entry in zip(rows, entries):
        pool, selected = entry['candidates'], entry['selections'][mode]
        if (len(pool) > 8 or len({c['visible_tokens'] for c in pool}) > 1
                or len(selected) not in (0, 2) or len(set(selected)) != len(selected)
                or any(not 0 <= i < len(pool) for i in selected)):
            raise ValueError('invalid candidate pool or selection budget')
        if selected and len(pool) < 2:
            raise ValueError('two distinct candidates are required')
        sources[row['sample_key']] = [apply_source_edits(row['raw_source'], pool[i]['edits']) for i in selected]
        choices[row['sample_key']] = selected
    return sources, choices


def _codebert_augmentation_smoke(rows, entries, root, base, args):
    import tempfile
    import torch
    from . import model as core
    from .change_learning import write_json
    train = [r for r in rows if r['split'] == 'train']
    valid = [r for r in rows if r['split'] == 'valid']
    covered = {e['sample_key'] for e in entries if e['selections']['random']}
    chosen = [r for label in (0,1) for r in [x for x in train if x['label'] == label and x['sample_key'] in covered][:4]]
    if len(chosen) != 8:
        raise ValueError('smoke needs four transformable real Train functions of each label')
    chosen += [next(r for r in valid if r['label'] == label) for label in (0,1)]
    common = dict(variant='codebert_source', model_path=args.model, source_max_length=2048,
        context_max_length=base['context_max_length'], seed=42, epochs=1, batch_size=4,
        gradient_accumulation=2, learning_rate=base['training_config']['learning_rate'],
        weight_decay=.01, finetuning_mode='full', device=args.device, log_every=1)
    core._seed_everything(42)
    initial = core._build_model('codebert_source', args.model, device=core._resolve_device(args.device),
        lora_r=16, lora_alpha=32, lora_dropout=.05, target_modules=tuple(base['target_modules']),
        gradient_checkpointing=False, fusion_dim=256, fusion_heads=8, finetuning_mode='full')
    embedding = initial.encoder.state_dict()['embeddings.word_embeddings.weight'].detach().cpu().clone()
    head = core._cpu_state(initial.task_modules.state_dict())
    del initial
    report = dict(train_members=[r['sample_key'] for r in chosen[:8]], valid_members=[r['sample_key'] for r in chosen[8:]])
    budgets = []
    with tempfile.TemporaryDirectory(prefix='vul_detect_augmentation_smoke_') as tmp:
        for mode in ('random', 'high_low'):
            sources, choices = _augmentation_sources(train, entries, mode)
            sources = {r['sample_key']: sources[r['sample_key']] for r in chosen[:8]}
            target = Path(tmp)/mode/'best.pt'
            cp = core.train_model(None, target, records=chosen, function_views=sources,
                augmentation_config=dict(mode=mode, choices={k:choices[k] for k in sources}), **common)
            if torch.equal(cp['encoder_state']['embeddings.word_embeddings.weight'], embedding):
                raise ValueError('smoke did not update the full encoder')
            if all(torch.equal(cp['task_state'][k], v) for k,v in head.items()):
                raise ValueError('smoke did not update the classifier')
            restored = core._load_model(cp, device=core._resolve_device(args.device))
            tokenizer = core.AutoTokenizer.from_pretrained(args.model)
            builder = core.InputBuilder(tokenizer, source_max_length=2048, context_max_length=base['context_max_length'])
            before = core._predict_logits(restored, chosen[8:], builder, variant='codebert_source', excluded_groups=(),
                batch_size=4, device=core._resolve_device(args.device))
            del restored
            after = core.predict_checkpoint(target, chosen[8:], batch_size=4, device=args.device)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            epoch = [json.loads(l) for l in target.with_suffix('.training.jsonl').open() if json.loads(l)['event']=='epoch'][0]
            budget = tuple(epoch[k] for k in ('optimizer_steps','bce_function_forwards','bce_encoder_sequences','encoder_token_slots'))
            if budget[0:2] != (1,24):
                raise ValueError('smoke did not execute the expected per-function view BCE budget')
            budgets.append(budget)
            report[mode] = dict(encoder_updated=True, classifier_updated=True, save_reload_exact=True,
                optimizer_steps=budget[0], function_forwards=budget[1], encoder_sequences=budget[2], encoder_token_slots=budget[3])
            del cp
    if budgets[0] != budgets[1]:
        raise ValueError('random and guided smoke forward budgets differ')
    report['budgets_identical'] = True
    write_json(root/'smoke.metrics.json', report)
    return report


def _codebert_shallow_augmentation(args):
    import pickle
    import torch
    import sklearn
    import transformers
    from . import model as core, audit_statistics as shallow
    from .syntax import AUGMENT_NAMES, AUGMENT_COMMENTS, UNSEEN_NAMES, UNSEEN_COMMENTS
    from .change_learning import write_json
    rows, base, budget = _matching_full_baseline(args)
    train = [r for r in rows if r['split']=='train']
    root = Path(args.output_dir or 'results/codebert_shallow_augmentation_seed42')
    config = dict(version=1, manifest=args.manifest, baseline_checkpoint=args.baseline_checkpoint,
        model=args.model, seed=42, training=base['training_config'], input_protocol=base['input_protocol'],
        encoder_precision=base['encoder_precision'], classifier_initialization=base['classifier_initialization'],
        initialization='fresh matching pretrained CodeBERT; identical seed42 linear head',
        augmentation=dict(max_candidates=8, names=AUGMENT_NAMES, comments=AUGMENT_COMMENTS,
            source_visibility='preserve the entire original visible character prefix under the native 2048 protocol',
            pool='largest equal-visible-token-count bucket; ties prefer first proposal bucket; label/score independent',
            selection='fixed two distinct views per function; random seed42 or shallow minimum/maximum; no label access',
            loss='mean BCE over original plus two views, then mean over original functions; insufficient pools use original only'),
        shallow=dict(folds=3, grouping='existing transitive repair/duplicate groups',
            features='log1p visible tokens/chars/lines + lexical TF-IDF',
            tfidf=shallow.TFIDF_OPTIONS, logistic_regression=shallow.LR_OPTIONS,
            scoring='frozen fold model scores its heldout function only'),
        robustness=dict(split='valid', names=UNSEEN_NAMES, comments=UNSEEN_COMMENTS,
            selection='one eligible unseen view per kind, deterministic and independent of labels/model scores'),
        selection='original Valid MCC threshold/epoch and tie-break',
        train_functions=len(train), valid_functions=len(rows)-len(train),
        environment=dict(torch=torch.__version__, transformers=transformers.__version__, sklearn=sklearn.__version__))
    config = json.loads(json.dumps(config))
    config_path = root/'config.metrics.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('existing augmentation configuration differs')
    if not config_path.exists(): write_json(config_path, config)
    paths = [root/name for name in ('candidates.jsonl','candidates.metrics.json','oof.models.pkl')]
    if any(p.exists() for p in paths) and not all(p.exists() for p in paths):
        raise FileExistsError('incomplete candidate/frozen-OOF cache; inspect before rerunning')
    if all(p.exists() for p in paths):
        entries = [json.loads(l) for l in paths[0].open()]
        preparation = json.loads(paths[1].read_text())
    else:
        tokenizer = core.AutoTokenizer.from_pretrained(args.model)
        entries, preparation, fitted = shallow.shallow_augmentation_views(train, tokenizer, base['input_protocol']['source_budget'])
        with paths[0].open('x') as handle:
            for entry in entries: handle.write(json.dumps(entry)+'\n')
        write_json(paths[1], preparation)
        with paths[2].open('xb') as handle: pickle.dump(fitted, handle)
        del fitted
    selections = {mode: _augmentation_sources(train, entries, mode) for mode in ('random','high_low')}
    if args.stage == 'prepare': return dict(config=config, candidates=preparation)
    smoke_path = root/'smoke.metrics.json'
    if args.stage in ('smoke','all') and not smoke_path.exists():
        _codebert_augmentation_smoke(rows, entries, root, base, args)
    if args.stage == 'smoke': return json.loads(smoke_path.read_text())
    if not smoke_path.exists() or not json.loads(smoke_path.read_text()).get('budgets_identical'):
        raise ValueError('complete the real full-CodeBERT augmentation smoke before formal training')
    modes = ('random','high_low') if args.shallow_augmentation=='both' else (args.shallow_augmentation,)
    expected = {k:base[k] for k in ('checkpoint_version','dataset_schema_version','variant','model_path',
        'source_max_length','context_max_length','lora_r','lora_alpha','lora_dropout','seed','finetuning_mode',
        'classifier_head','classifier_initialization','encoder_precision','cohort_members','initial_checkpoint','training_config','target_modules')}
    for mode in modes:
        run, (sources, choices) = root/mode, selections[mode]
        augmentation = dict(mode=mode, candidates_cache=str(paths[0]), choices=choices,
            loss=config['augmentation']['loss'], protocol_version=config['version'])
        target = run/'best.pt'
        if target.exists():
            cp = torch.load(target, map_location='cpu', weights_only=False)
            if (any(cp.get(k) != v for k,v in expected.items()) or cp.get('sample_weights') is not None
                    or cp.get('augmentation_config') != augmentation):
                raise ValueError('existing augmentation checkpoint differs from the frozen experiment')
            history = [json.loads(l) for l in target.with_suffix('.training.jsonl').open()]
            if [(r['epoch'],r['optimizer_steps']) for r in history if r['event']=='epoch'] != budget:
                raise ValueError('augmentation run did not complete the matching three epochs')
        elif args.stage in ('train','all'):
            cp = core.train_model(None, target, records=rows, variant='codebert_source', model_path=args.model,
                source_max_length=2048, context_max_length=base['context_max_length'], seed=42, device=args.device,
                finetuning_mode='full', function_views=sources, augmentation_config=augmentation,
                log_every=25, **base['training_config'])
        else:
            raise FileNotFoundError(target)
        _finetuning_evaluate(rows, run, 'valid', cp, args)
        del cp
    if args.stage in ('report','all'):
        return _augmentation_report(rows, entries, root, config, preparation, base, args)
    return dict(config=config, candidates=preparation, completed_modes=modes)


def _augmentation_report(rows, entries, root, config, preparation, base, args):
    import torch
    from . import audit_statistics as shallow
    from .syntax import apply_source_edits
    from .change_learning import write_json
    valid = [r for r in rows if r['split']=='valid']
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    views = {'comment':[], 'rename':[]}
    for index, row in enumerate(valid):
        candidates, _ = shallow.visible_semantic_candidates(row['raw_source'], row.get('language','c'),
            tokenizer, base['input_protocol']['source_budget'], unseen=True, file_name=row.get('file_name',''))
        for kind in views:
            pool = [c for c in candidates if c['kind']==kind]
            if pool:
                chosen = pool[index % len(pool)]
                views[kind].append(dict(row,raw_source=apply_source_edits(row['raw_source'],chosen['edits'])))
    result, histories = {}, []
    for mode in ('bce','random','high_low'):
        checkpoint_path = Path(args.baseline_checkpoint) if mode=='bce' else root/mode/'best.pt'
        cp = base if mode=='bce' else torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        run = Path(args.baseline_checkpoint).parent if mode=='bce' else root/mode
        clean = _finetuning_evaluate(rows, run, 'valid', cp, args)
        original = {r['sample_key']:r for r in map(json.loads,(run/'valid.predictions.jsonl').open())}
        robust = {}
        robust_run = root/'baseline' if mode=='bce' else run
        for kind, changed in views.items():
            if not changed:
                robust[kind] = dict(eligible=0,total=len(valid),coverage=0.,robust_accuracy=None,flip_rate=None)
                continue
            _finetuning_evaluate(changed,robust_run,'valid',cp,args,output_name=f'unseen_{kind}',checkpoint_path=checkpoint_path)
            predictions = list(map(json.loads,(robust_run/f'unseen_{kind}.predictions.jsonl').open()))
            both_correct = sum(r['prediction']==original[r['sample_key']]['prediction']==r['label'] for r in predictions)
            flipped = sum(r['prediction']!=original[r['sample_key']]['prediction'] for r in predictions)
            robust[kind] = dict(eligible=len(predictions),total=len(valid),coverage=len(predictions)/len(valid),
                robust_correct=both_correct,robust_accuracy=both_correct/len(predictions),flipped=flipped,flip_rate=flipped/len(predictions),
                clean_accuracy_same_subset=sum(original[r['sample_key']]['prediction']==r['label'] for r in predictions)/len(predictions),
                definition='both original and unseen view correct; flip rate over eligible functions; ineligible count reported')
        result[mode] = dict(checkpoint=str(checkpoint_path),selected_epoch=clean['selected_epoch'],threshold=clean['decision_threshold'],
            bce=clean['bce'],**clean['metrics'],length_groups=_reweighting_strata(list(original.values()),
                base['input_protocol']['source_budget'],clean['decision_threshold']),robustness=robust)
        if mode!='bce':
            history = [json.loads(l) for l in checkpoint_path.with_suffix('.training.jsonl').open()]
            histories.append([tuple(r[k] for k in ('epoch','optimizer_steps','bce_function_forwards','bce_encoder_sequences','encoder_token_slots'))
                              for r in history if r['event']=='epoch'])
        if mode!='bce': del cp
    if histories[0]!=histories[1]: raise ValueError('random/guided formal training forward budgets differ')
    report = dict(config=config,candidates=preparation,valid=result,matched_training_budgets=histories[0])
    write_json(root/'comparison.metrics.json',report)
    return report


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
