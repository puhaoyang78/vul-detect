"""Bounded source-only continuation experiment, using reviewed training errors."""
from __future__ import annotations

import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

from .benchmark_view import select_source_records
from .diagnostic import load_cohort, metrics, read_rows, write_json, write_rows
from .model import _read_jsonl, _select_validation_threshold, predict_checkpoint, train_model
from .progress import print_table


def class_normalized_weights(records, hard_keys, factor=3.0):
    if not np.isfinite(factor) or factor < 1:
        raise ValueError("factor must be finite and >= 1")
    keys = {r['sample_key'] for r in records}
    if len(keys) != len(records) or not set(hard_keys) <= keys:
        raise ValueError("invalid or duplicate training keys")
    result = {}
    for label in (0, 1):
        subset = [r for r in records if r['label'] == label]
        if not subset:
            raise ValueError("both classes required")
        raw = {r['sample_key']: factor if r['sample_key'] in hard_keys else 1.0 for r in subset}
        mean = sum(raw.values()) / len(raw)
        result.update({key: value / mean for key, value in raw.items()})
    return result


def project(repo):
    # Explicit Linux upstream/mirror aliases present in this cohort.
    repo = repo.lower().rstrip('/')
    if 'kernel.org' in repo or repo.endswith('/torvalds/linux') or repo.endswith('/gregkh/linux'):
        return 'linux'
    return repo


def check_isolation(train_keys, check_keys, fit_keys, groups, repos):
    training = set(train_keys)
    check = set(check_keys)
    fit = set(fit_keys)
    if check & (training | fit):
        raise ValueError("check samples overlap training")
    if {groups[k] for k in check} & {groups[k] for k in training | fit}:
        raise ValueError("check provenance groups overlap training")
    if {repos[k] for k in check} & {repos[k] for k in training}:
        raise ValueError("adaptation/check projects overlap")


def prepare(args):
    output = Path(args.output)
    if (output / 'experiment.json').exists():
        raise ValueError("experiment already prepared; use --stage run")
    diagnostic = Path(args.diagnostic)
    records, groups = load_cohort(args.dataset, args.manifest)
    rows = {r['sample_key']: r for r in records}
    meta = {r['sample_key']: r for r in read_rows(Path(args.manifest)) if r['sample_key'] in rows}
    repos = {k: project(m['repo']) for k, m in meta.items()}
    # Fixed before training: fold 1 has the largest reviewed-error coverage.
    fold = json.loads((diagnostic / 'folds.json').read_text())[1]
    base_dir = diagnostic / 'outer_01' / 'outer_model'
    base_path = base_dir / 'baseline.pt'
    base = torch.load(base_path, map_location='cpu', weights_only=False)
    membership = json.loads((base_dir / 'membership.json').read_text())
    if set(membership['fit']) != set(fold['develop']):
        raise ValueError("fold/checkpoint membership mismatch")
    fit = set(membership['fit'])
    reviewed = read_rows(Path(args.review))
    hard = sorted(r['sample_key'] for r in reviewed if r['review_status'] == 'local_evidence')
    if len(hard) != len(set(hard)) or not hard:
        raise ValueError("invalid reviewed keys")
    for r in reviewed:
        if r['sample_key'] not in hard:
            continue
        k = r['sample_key']
        if k not in rows or r['label'] != rows[k]['label'] or r['split'] != 'train':
            raise ValueError("review/cohort mismatch")
        if r['source_truncated'] or not r['upstream_diff_inspected'] or not r['evidence']:
            raise ValueError("incomplete review evidence")
        for evidence in r['evidence']:
            for loc in evidence['occurrences']:
                if rows[k]['raw_source'][loc['start_char']:loc['end_char']] != evidence['source_text']:
                    raise ValueError("stale review evidence")
    tok = AutoTokenizer.from_pretrained(base['model_path'], local_files_only=True)
    limit = base['source_max_length']
    length_cache = {}
    def complete(k):
        if k not in length_cache:
            length_cache[k] = len(tok.encode(rows[k]['raw_source'], add_special_tokens=False))
        return length_cache[k] <= limit
    if not all(complete(k) for k in hard):
        raise ValueError("reviewed input exceeds initial checkpoint limit")
    hard_repos = {repos[k] for k in hard}
    hard_groups = {groups[k] for k in hard}
    background = sorted(k for k in fit if repos[k] in hard_repos and groups[k] not in hard_groups
                        and len(rows[k]['raw_source']) <= 6000)
    random.Random(42).shuffle(background)
    train = list(hard)
    for label in (0, 1):
        need = 64 - sum(rows[k]['label'] == label for k in hard)
        if need < 0:
            raise ValueError("too many reviewed samples for bounded pilot")
        candidates = [k for k in background if rows[k]['label'] == label and complete(k)]
        if len(candidates) < need:
            raise ValueError("not enough complete background samples")
        train.extend(candidates[:need])
    train = sorted(train)
    excluded_groups = {groups[k] for k in set(train) | fit}
    pool = sorted(k for k in fold['evaluation'] if groups[k] not in excluded_groups
                  and repos[k] not in hard_repos and len(rows[k]['raw_source']) <= 6000)
    # Operational near-duplicate screen, fitted only on initial/adaptation training sources.
    reference = sorted(fit | set(train))
    vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=30000, min_df=2)
    ref = vec.fit_transform([rows[k]['raw_source'] for k in reference])
    sim = (vec.transform([rows[k]['raw_source'] for k in pool]) @ ref.T).max(axis=1).toarray().ravel()
    similarities = dict(zip(pool, map(float, sim)))
    pool = [k for k in pool if similarities[k] < 0.90]
    random.Random(43).shuffle(pool)
    check = []
    for label in (0, 1):
        candidates = [k for k in pool if rows[k]['label'] == label and complete(k)]
        if len(candidates) < 64:
            raise ValueError("not enough isolated check samples")
        check.extend(candidates[:64])
    check = sorted(check)
    check_isolation(train, check, fit, groups, repos)
    calibration = [k for k in fold['calibration'] if groups[k] not in {groups[t] for t in train}]
    saved = {r['sample_key']: r for r in read_rows(base_dir / 'predictions.jsonl')}
    for k in calibration + check:
        if saved[k]['label'] != rows[k]['label']:
            raise ValueError("saved predictions mismatch")
    threshold, _ = _select_validation_threshold([rows[k] for k in calibration],
                                                torch.tensor([expit(saved[k]['logit']) for k in calibration]))
    plan = dict(seed=42, base_checkpoint=str(base_path.resolve()), train=train, hard=hard, check=check,
                calibration=calibration, threshold=threshold, epochs=1, learning_rate=2e-5,
                gradient_accumulation=8, hard_weight_factor=3.0,
                source_max_length=limit, dataset=str(Path(args.dataset).resolve()),
                manifest=str(Path(args.manifest).resolve()), review=str(Path(args.review).resolve()),
                fit=sorted(fit), check_max_training_cosine=max(similarities[k] for k in check),
                hard_previously_seen_by_base=sorted(set(hard) & fit),
                project_isolation_scope='continuation training vs check; initial model may know other functions from check projects',
                evaluation_scope='exploratory function-label transfer; check labels are original PrimeVul labels, not independent mechanism adjudication',
                initial_check_logits={k: saved[k]['logit'] for k in check},
                versions=dict(torch=torch.__version__))
    write_json(output / 'experiment.json', plan)
    print_table('Source-only pilot prepared', ['Train', 'Reviewed', 'Unseen check', 'Threshold', 'Max cosine'],
                [[len(train), len(hard), len(check), f'{threshold:.2f}', f'{plan["check_max_training_cosine"]:.3f}']])


def run(args):
    output = Path(args.output)
    plan = json.loads((output / 'experiment.json').read_text())
    if (output / 'results.json').exists():
        print((output / 'results.json').read_text())
        return
    cohort, groups = load_cohort(plan['dataset'], plan['manifest'])
    all_rows, _ = select_source_records(_read_jsonl(plan['dataset']), 'primevul')
    rows = {r['sample_key']: r for r in all_rows}
    meta = {r['sample_key']: r for r in read_rows(Path(plan['manifest'])) if r['sample_key'] in groups}
    check_isolation(plan['train'], plan['check'], plan['fit'], groups,
                    {k: project(r['repo']) for k, r in meta.items()})
    train = [rows[k] for k in plan['train']]
    valid = [r for r in all_rows if r['split'] == 'valid']
    check = [rows[k] for k in plan['check']]
    hard = [rows[k] for k in plan['hard']]
    evaluation = check + valid + hard
    base = torch.load(plan['base_checkpoint'], map_location='cpu', weights_only=False)
    results = {}
    for arm in ('initial', 'uniform', 'weighted'):
        checkpoint = Path(plan['base_checkpoint']) if arm == 'initial' else output / f'{arm}.pt'
        if arm != 'initial' and not checkpoint.exists():
            weights = class_normalized_weights(train, set(plan['hard']), plan['hard_weight_factor']) if arm == 'weighted' else None
            fitted = train_model(None, checkpoint, records=train, variant='baseline',
                model_path=base['model_path'], source_max_length=base['source_max_length'],
                context_max_length=base['context_max_length'], lora_r=base['lora_r'],
                lora_alpha=base['lora_alpha'], lora_dropout=base['lora_dropout'],
                fusion_dim=base['fusion_dim'], fusion_heads=base['fusion_heads'],
                epochs=plan['epochs'], learning_rate=plan['learning_rate'], batch_size=1,
                gradient_accumulation=plan['gradient_accumulation'], seed=plan['seed'],
                fixed_epochs=True, initial_checkpoint=plan['base_checkpoint'], sample_weights=weights,
                device=args.device, log_every=1)
            del fitted
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        pred_path = output / f'{arm}.predictions.jsonl'
        if not pred_path.exists():
            pending = valid + hard if arm == 'initial' else evaluation
            logits = predict_checkpoint(checkpoint, pending, device=args.device).tolist()
            values = {r['sample_key']: z for r, z in zip(pending, logits)}
            if arm == 'initial':
                values.update(plan['initial_check_logits'])
            write_rows(pred_path, [dict(sample_key=r['sample_key'], label=r['label'], logit=values[r['sample_key']]) for r in evaluation])
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        saved = read_rows(pred_path)
        if [r['sample_key'] for r in saved] != [r['sample_key'] for r in evaluation]:
            raise ValueError('prediction membership mismatch')
        values = {r['sample_key']: expit(r['logit']) for r in saved}
        results[arm] = {name: metrics([r['label'] for r in subset], [values[r['sample_key']] for r in subset], plan['threshold'])
                        for name, subset in [('check', check), ('valid', valid), ('trained_hard', hard)]}
        print_table(f'{arm}: fixed baseline threshold {plan["threshold"]:.2f}',
            ['Split', 'AUC', 'MCC', 'TP', 'FP', 'F1'], [[name, f'{m["auc"]:.4f}', f'{m["mcc"]:.4f}', m['tp'], m['fp'], f'{m["f1"]:.4f}'] for name, m in results[arm].items()])
        write_json(output / 'results.partial.json', results)
    write_json(output / 'results.json', results)
