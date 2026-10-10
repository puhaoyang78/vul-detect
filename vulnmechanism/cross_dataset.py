"""Source-only cross-dataset experiments, with an immutable project assignment."""
import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
import random
import re
from pathlib import Path
import time

import ijson
import numpy as np
from sklearn.metrics import average_precision_score

from .cascade import read_rows, write_json
from .cfg_metrics import metrics, paired_changes

RAW = Path('/home/PublicData/PHY-data/vul_detect/data')
PRIME = {
    'A': 'results/codebert_prototype_seed42/A/best.pt',
    'B': 'results/codebert_prototype_seed42/B/best.pt',
    'C': 'results/cfg_abc_seed42/baseline/best.pt',
}


def project_split(project, previous):
    if project in previous:
        return previous[project]
    value = int(hashlib.sha256(('cross-dataset-42:' + project).encode()).hexdigest(), 16) % 100
    return 'train' if value < 80 else 'valid' if value < 90 else 'test'


def mega_source(row):
    if type(row.get('is_vul')) is not bool:
        raise ValueError('MegaVul requires a native boolean label')
    # Native vulnerable functions are BEFORE repair; unrelated negatives use func.
    source = row.get('func_before') if row['is_vul'] else row.get('func')
    if not isinstance(source, str) or not source.strip():
        raise ValueError('missing native labeled function')
    return source, int(row['is_vul'])


def summarize_predictions(rows, threshold):
    labels = [r['label'] for r in rows]
    scores = [r['score'] for r in rows]
    result = metrics(labels, scores, threshold)
    result['positive'] = sum(labels)
    result['fpr'] = result['fp'] / (result['fp'] + result['tn']) if result['fp'] + result['tn'] else None
    result['fnr'] = result['fn'] / (result['fn'] + result['tp']) if result['fn'] + result['tp'] else None
    result['auprc_average_precision'] = float(average_precision_score(labels, scores)) if len(set(labels)) == 2 else None
    if all('logit' in r for r in rows):
        logits = np.array([r['logit'] for r in rows], dtype=np.float64)
        result['bce'] = float(np.mean(np.logaddexp(0, logits) - np.array(labels) * logits))
    else:
        p = np.array(scores, dtype=np.float64)
        y = np.array(labels)
        result['bce'] = float(-(np.log(p[y == 1]).sum() + np.log1p(-p[y == 0]).sum()) / len(p))
    return result


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    previous = {}
    for line in Path('data/pairs_all.jsonl').open():
        r = json.loads(line)
        if r['project'] in previous and previous[r['project']] != r['split']:
            raise ValueError('historical project split conflict')
        previous[r['project']] = r['split']
    projects = {}; counts = Counter(); project_counts = defaultdict(Counter)
    invalid = Counter(); raw_labels = Counter(); fields = Counter()
    with (out / 'megavul.jsonl').open('x') as dest, (RAW / 'megavul/megavul.json').open('rb') as src:
        for i, r in enumerate(ijson.items(src, 'item', use_float=True)):
            raw_labels[str(r.get('is_vul'))] += 1
            fields.update(k for k in r if k in ('split', 'partition', 'subset'))
            p = r['repo_name']; split = project_split(p, previous); projects[p] = split
            extension = Path(r['file_path']).suffix
            language = 'c' if extension == '.c' else 'cpp' if extension in ('.cc', '.cpp', '.cxx', '.C', '.h', '.hpp', '.hxx', '.hh') else None
            if language is None:
                invalid['unsupported_extension:' + extension] += 1
                continue
            counts[split + '/rows'] += 1
            counts[split + '/positive'] += int(r['is_vul'])
            project_counts[p][str(int(r['is_vul']))] += 1
            # No held-out source is exported, tokenized or evaluated.
            if split == 'test':
                continue
            source, label = mega_source(r)
            row = dict(sample_key=f'megavul:{i}', dataset='megavul', split=split,
                       label=label, raw_source=source, language=language, project=p,
                       commit=r['commit_hash'], commit_url=r['git_url'], cwes=r.get('cwe_ids', []),
                       source_row=i, source_field='func_before' if label else 'func')
            dest.write(json.dumps(row) + '\n')
    write_json(out / 'project_splits.json', projects)
    write_json(out / 'inventory.json', dict(raw_rows=sum(raw_labels.values()), raw_labels=raw_labels,
        native_split_fields=fields, c_cpp_counts=counts, exclusions=invalid,
        projects=len(projects), historical_projects=len(previous), project_counts=project_counts,
        split_policy='Historical pair-project assignments retained; new projects SHA256(cross-dataset-42:project) mod 100, 80/10/10. User authorized. No class balancing, CPG or parsing admission.',
        source_policy='One native labeled function per raw row. Vulnerable=func_before; negative=func. No additional repaired negatives.',
        test_source_used=False))
    prime = {name: read_rows(Path(path).parent / 'valid.predictions.jsonl') for name, path in PRIME.items()}
    result = {}
    for name, rows in prime.items():
        if len(rows) != 741:
            raise ValueError('PrimeVul cohort changed')
        result[name] = dict(checkpoint=PRIME[name], selected=summarize_predictions(rows, rows[0]['threshold']),
                            fixed_0_5=summarize_predictions(rows, .5))
    result['B_vs_A'] = paired_changes(prime['A'], prime['B'])
    result['C_vs_A'] = paired_changes(prime['A'], prime['C'])
    write_json(out / 'primevul.metrics.json', result)
    print(json.dumps(dict(counts=counts, projects=len(projects), raw_labels=raw_labels), indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'audit', 'views', 'provenance', 'train', 'evaluate', 'compare'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', choices=['A', 'B', 'C'])
    p.add_argument('--origin', choices=['primevul', 'megavul'], default='megavul')
    p.add_argument('--target', choices=['primevul', 'megavul'], default='megavul')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args.output)
    elif args.action == 'audit':
        audit(args.output)
    elif args.action == 'views':
        prepare_views(args.output)
    elif args.action == 'compare':
        compare(args.output)
    elif args.action == 'provenance':
        provenance(args.output)
    elif not args.model:
        p.error('--model required')
    elif args.action == 'train':
        train(args.output, args.model, args.device)
    else:
        evaluate(args.output, args.model, args.origin, args.target, args.device)


def dataset(out, name):
    path = out / 'megavul.jsonl' if name == 'megavul' else Path('results/codebert_prototype_seed42/records.jsonl')
    rows = read_rows(path)
    if name == 'primevul':
        metadata = {r['sample_key']: r for r in read_rows('results/codebert_prototype_seed42/members.jsonl')}
        for row in rows:
            m = metadata[row['sample_key']]
            row.update(project=m['project'], commit=m['commit'], cwes=m['cwes'])
    return rows


def audit(out):
    from .prepare_benchmark import shingles, normalized_source
    datasets = {name: dataset(out, name) for name in ('primevul', 'megavul')}
    valid = [r for rows in datasets.values() for r in rows if r['split'] == 'valid']
    signatures = [shingles(r['raw_source']) for r in valid]
    frequencies = Counter(t for sig in signatures for t in sig)
    # A single global rare-first order is valid for the exact prefix join and
    # avoids quadratic candidate lists for ubiquitous punctuation shingles.
    def prefix(sig):
        return sorted(sig, key=lambda t: (frequencies[t], t))[:len(sig) - (9 * len(sig) + 9) // 10 + 1]
    postings = defaultdict(list)
    for i, sig in enumerate(signatures):
        for t in prefix(sig): postings[t].append(i)
    exact = defaultdict(list)
    for row in valid:
        exact[hashlib.sha256(normalized_source(row['raw_source']).encode()).hexdigest()].append(row['sample_key'])
    excluded = {a + '_to_' + b: defaultdict(set) for a in datasets for b in datasets}
    by_key = {r['sample_key']: r for r in valid}
    projects = {}; commits = {}
    with (out / 'overlaps.jsonl').open('x') as dest:
        for name, rows in datasets.items():
            train = [r for r in rows if r['split'] == 'train']
            projects[name] = {str(r.get('project')) for r in train if r.get('project')}
            commits[name] = {str(r.get('commit', r.get('commit_id'))) for r in train if r.get('commit', r.get('commit_id'))}
            for n, row in enumerate(train):
                signature = shingles(row['raw_source'])
                candidates = {i for token in prefix(signature) for i in postings.get(token, ())}
                hits = {}
                for i in candidates:
                    other, key = signatures[i], valid[i]['sample_key']
                    if min(len(signature), len(other)) * 10 < max(len(signature), len(other)) * 9:
                        continue
                    common = len(signature.intersection(other)); union = len(signature) + len(other) - common
                    if common * 10 >= union * 9:
                        hits[key] = 'token_5gram_jaccard_ge_0.90'
                norm = hashlib.sha256(normalized_source(row['raw_source']).encode()).hexdigest()
                for key in exact.get(norm, []): hits[key] = 'normalized_exact'
                for key, reason in hits.items():
                    target = by_key[key]['dataset']
                    excluded[name + '_to_' + target][key].add(reason)
                    dest.write(json.dumps(dict(source_train=row['sample_key'], target_valid=key, reason=reason,
                                               labels=[row['label'], by_key[key]['label']])) + '\n')
                if n % 10000 == 0: print(f'audit {name} {n}/{len(train)}', flush=True)
    report = {}
    for origin in datasets:
        for target in datasets:
            key = origin + '_to_' + target
            rows = [r for r in valid if r['dataset'] == target]
            for r in rows:
                commit = str(r.get('commit', r.get('commit_id', '')))
                if commit and commit in commits[origin]: excluded[key][r['sample_key']].add('repair_commit')
            kept = [r for r in rows if r['sample_key'] not in excluded[key]]
            report[key] = dict(before=len(rows), before_positive=sum(r['label'] for r in rows), after=len(kept),
                after_positive=sum(r['label'] for r in kept),
                shared_project_valid=sum(str(r.get('project')) in projects[origin] for r in rows),
                excluded={k: sorted(v) for k, v in excluded[key].items()})
    write_json(out / 'leakage.json', report)
    print(json.dumps({k: {n:v for n,v in r.items() if n!='excluded'} for k,r in report.items()}, indent=2))


def train(out, name, device):
    from . import model as core
    folder = out / 'megavul' / name
    if not (out / 'leakage.json').exists():
        raise ValueError('run leakage audit before training')
    folder.mkdir(parents=True, exist_ok=False)
    config = dict(variant='baseline' if name == 'C' else 'codebert_source',
        model_path='/home/phy/models/Qwen2.5-Coder-7B-Instruct' if name == 'C' else '/home/PublicData/PHY-data/resource/codebert-base',
        classifier_head='mlp4' if name == 'B' else 'linear', source_max_length=2048 if name == 'C' else 512,
        batch_size=1, gradient_accumulation=8, epochs=3, learning_rate=2e-4, weight_decay=.01,
        lora_r=16, lora_alpha=32, lora_dropout=.05, seed=42, device=device, log_every=100)
    write_json(folder / 'config.json', config)
    core.train_model(None, folder / 'best.pt', records=dataset(out, 'megavul'), **config)


def evaluate(out, name, origin, target, device):
    import torch
    from transformers import AutoTokenizer
    from . import model as core
    from .cfg_data import identity
    from .cfg_metrics import decision_boundary
    folder = out / (origin + '_to_' + target) / name
    folder.mkdir(parents=True, exist_ok=False)
    path = PRIME[name] if origin == 'primevul' else out / 'megavul' / name / 'best.pt'
    cp = torch.load(path, map_location='cpu', weights_only=False)
    rows = [r for r in dataset(out, target) if r['split'] == 'valid']
    tok = AutoTokenizer.from_pretrained(cp['model_path'])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    builder = core.InputBuilder(tok, source_max_length=cp['source_max_length'], context_max_length=cp['context_max_length'])
    dev = torch.device(device); model = core._load_model(cp, device=dev)
    if origin != 'primevul' or target != 'megavul':
        raise ValueError('Current fixed-view experiment is PrimeVul -> MegaVul')
    if not (out / 'views.json').exists():
        raise ValueError('freeze evaluation members before inference')
    batch_size = 4 if name == 'C' else 32
    rows.sort(key=lambda r: (len(r['raw_source']), r['sample_key']))
    predictions = []
    # Fixed label-blind sample for single-query latency and batching agreement.
    probe = sorted(rows, key=lambda r: hashlib.sha256(('latency42:' + r['sample_key']).encode()).digest())[:128]
    single = {}; latency = []
    with torch.no_grad():
        core._forward_batch(model, rows[:1], builder, variant=cp['variant'], excluded_groups=(), device=dev)
        for row in probe:
            if dev.type == 'cuda': torch.cuda.synchronize(dev)
            start = time.perf_counter()
            logit = float(core._forward_batch(model, [row], builder, variant=cp['variant'], excluded_groups=(), device=dev).float().cpu()[0])
            elapsed = time.perf_counter() - start
            single[row['sample_key']] = logit
            latency.append(dict(sample_key=row['sample_key'], seconds=elapsed, logit=logit))
        write_json(folder / 'latency.json', dict(batch_size=1, warmup=1, samples=latency,
            mean=float(np.mean([r['seconds'] for r in latency])), p95=float(np.quantile([r['seconds'] for r in latency], .95)),
            device=torch.cuda.get_device_name(dev) if dev.type=='cuda' else 'cpu',
            scope='tokenization, H2D, forward and synchronized CPU logit; excludes model load'))
        with (folder / 'valid.predictions.jsonl').open('x') as stream:
            for start_index in range(0, len(rows), batch_size):
                batch = rows[start_index:start_index + batch_size]
                if dev.type == 'cuda': torch.cuda.synchronize(dev)
                start = time.perf_counter()
                logits = core._forward_batch(model, batch, builder, variant=cp['variant'], excluded_groups=(), device=dev).float().cpu()
                elapsed = time.perf_counter() - start
                for row, value in zip(batch, logits):
                    logit = float(value); score = float(torch.sigmoid(value))
                    count = len(builder._encode(row['raw_source']))
                    budget = cp.get('input_protocol', {}).get('source_budget', cp['source_max_length'])
                    pred = dict(identity(row), logit=logit, score=score, threshold=cp['decision_threshold'],
                        prediction=int(score >= decision_boundary(cp['decision_threshold'])), batch_amortized_seconds=elapsed / len(batch),
                        source_token_count=count, visible_source_tokens=min(count, budget), source_truncated=count > budget)
                    predictions.append(pred); stream.write(json.dumps(pred) + '\n')
                if start_index % (batch_size * 100) == 0:
                    stream.flush(); print(f'{name}: {start_index + len(batch)}/{len(rows)}', flush=True)
    differences = [abs(r['logit'] - single[r['sample_key']]) for r in predictions if r['sample_key'] in single]
    write_json(folder / 'metrics.json', dict(checkpoint=str(path), source_threshold=cp['decision_threshold'],
        selected=summarize_predictions(predictions, cp['decision_threshold']), fixed_0_5=summarize_predictions(predictions, .5),
        inference_batch_size=batch_size, batching_probe_max_logit_difference=max(differences),
        note='Length-sorted batched evaluation; single-query latency separately measured. Source threshold unchanged; development valid; no target calibration.'))


def balanced_keys(rows):
    groups = [[r['sample_key'] for r in sorted(rows, key=lambda r:r['sample_key']) if r['label']==y] for y in (0, 1)]
    n = min(map(len, groups))
    rng = random.Random(42)
    return sorted(k for group in groups for k in rng.sample(group, n))


def prepare_views(out):
    from .prepare_benchmark import NearIndex, normalized_source, prime_provenance, repository
    rows = [r for r in dataset(out, 'megavul') if r['split']=='valid']
    prime = dataset(out, 'primevul')
    wanted = {r['sample_key'] for r in prime}; source_meta = {}
    for split in ('train', 'valid'):
        with (RAW / 'PrimeVul_v0.1' / f'primevul_{split}.jsonl').open() as f:
            for line in f:
                r=json.loads(line); key=f"primevul:{r['idx']}"
                if key in wanted:
                    repo, commit, _ = prime_provenance(r)
                    source_meta[key]=dict(repo=repo, commit=commit, split=split)
    if set(source_meta)!=wanted: raise ValueError('missing source provenance')
    commits={r['commit'] for r in source_meta.values()}; repos={r['repo'] for r in source_meta.values() if r['repo']}
    index=NearIndex()
    for r in prime:
        if r['split']=='valid':
            index.add(dict(id=r['sample_key'], dataset='primevul', split='valid', rows=[dict(function=r['raw_source'],label=r['label'])]))
    leak=json.loads((out/'leakage.json').read_text())['primevul_to_megavul']['excluded']
    excluded={k:set(v) for k,v in leak.items()}; normalized=defaultdict(list)
    metadata=[]
    for i,r in enumerate(rows):
        key=r['sample_key']; reasons=excluded.setdefault(key,set())
        if r['commit'] in commits: reasons.add('source_train_or_valid_repair_commit')
        match=index.matches(dict(dataset='megavul', split='valid', rows=[dict(function=r['raw_source'],label=r['label'])]))
        if match: reasons.add('source_valid_near_duplicate')
        norm=normalized_source(r['raw_source']); normalized[norm].append(r)
        url=r['commit_url']; repo=repository(url.split('/commit/')[0]) if '/commit/' in url else ''
        metadata.append(dict(sample_key=key,label=r['label'],project=r['project'],repo=repo,
            seen_source_repo=bool(repo and repo in repos),repo_known=bool(repo),characters=len(r['raw_source']),cwes=r['cwes']))
        if i%20000==0: print(f'views {i}/{len(rows)}',flush=True)
    for group in normalized.values():
        group=sorted(group,key=lambda r:r['sample_key'])
        if len({r['label'] for r in group})>1:
            for r in group: excluded[r['sample_key']].add('identical_tokens_conflicting_labels')
        else:
            # If a group has a source-overlap exclusion, all its copies are excluded.
            reasons=set().union(*(excluded[r['sample_key']] for r in group))
            for r in group: excluded[r['sample_key']].update(reasons)
            for r in group[1:]: excluded[r['sample_key']].add('within_valid_normalized_duplicate')
    clean=[r for r in rows if not excluded[r['sample_key']]]
    views=dict(raw_natural=[r['sample_key'] for r in rows],raw_balanced=balanced_keys(rows),
               clean_natural=[r['sample_key'] for r in clean],clean_balanced=balanced_keys(clean))
    lookup={r['sample_key']:r for r in rows}
    write_json(out/'views.json',dict(seed=42,views=views,
        counts={k:dict(samples=len(v),positive=sum(lookup[x]['label'] for x in v)) for k,v in views.items()},
        excluded={k:sorted(v) for k,v in excluded.items() if v},
        policy='Frozen before inference; uniform label-wise sample without model scores. Clean removes source train/selection-valid overlap and repair commits, token-identical conflicts and repeated target functions. Near-identical target functions with distinct token sequences are retained, not relabeled.',
        project_split_counts=dict(Counter(json.loads((out/'project_splits.json').read_text()).values()))))
    with (out/'valid.metadata.jsonl').open('x') as f:
        for r in metadata:f.write(json.dumps(r)+'\n')
    print(json.dumps({k:dict(samples=len(v),positive=sum(lookup[x]['label'] for x in v)) for k,v in views.items()},indent=2))


def provenance(out):
    """Audit CVE sources without reading held-out test or model scores."""
    wanted={r['sample_key'] for r in read_rows('results/codebert_prototype_seed42/members.jsonl')}
    counts=Counter(); source_cves=set(); found=set()
    for split in ('train','valid'):
        with (RAW/'PrimeVul_v0.1'/f'primevul_{split}.jsonl').open() as f:
            for line in f:
                r=json.loads(line); key=f"primevul:{r['idx']}"
                if key not in wanted: continue
                found.add(key); cves=set(re.findall(r'CVE-\d{4}-\d+',str(r.get('cve',''))))
                counts[split+'_rows']+=1; counts[split+'_with_cve']+=bool(cves)
                source_cves.update(cves)
    if found!=wanted: raise ValueError('missing source CVE provenance members')
    views=json.loads((out/'views.json').read_text())['views']
    target=set(views['raw_natural']); shared={}; labels={}; present=set()
    with (RAW/'megavul/megavul.json').open('rb') as f:
        for i,r in enumerate(ijson.items(f,'item',use_float=True)):
            key=f'megavul:{i}'
            if key not in target: continue
            cves=set(re.findall(r'CVE-\d{4}-\d+',str(r.get('cve_id',''))))
            labels[key]=int(r['is_vul'])
            if cves: present.add(key)
            if cves & source_cves: shared[key]=sorted(cves & source_cves)
    write_json(out/'cve_overlap.json',dict(source_counts=counts,source_unique_cves=len(source_cves),
        views={v:dict(samples=len(keys),with_cve=sum(k in present for k in keys),
            shared_cve_rows=sum(k in shared for k in keys),
            shared_cve_positive=sum(labels[k] for k in keys if k in shared)) for v,keys in views.items()},
        shared_members=shared,note='Provenance only; shared CVE is not proof of identical source or function mechanism.'))


def compare(out):
    from .cfg_metrics import decision_boundary
    config=json.loads((out/'views.json').read_text())
    shared=json.loads((out/'cve_overlap.json').read_text())['shared_members']
    for suffix in ('natural','balanced'):
        config['views']['cve_disjoint_'+suffix]=[k for k in config['views']['clean_'+suffix] if k not in shared]
    predictions={name:read_rows(out/'primevul_to_megavul'/name/'valid.predictions.jsonl') for name in PRIME}
    lookup={name:{r['sample_key']:r for r in rows} for name,rows in predictions.items()}
    expected=set(config['views']['raw_natural'])
    for name in PRIME:
        if len(predictions[name])!=len(expected) or set(lookup[name])!=expected:
            raise ValueError('incomplete or duplicate predictions: '+name)
        if not (out/'primevul_to_megavul'/name/'metrics.json').exists():
            raise ValueError('evaluation did not finish: '+name)
    result={}
    table=[]
    meta={r['sample_key']:r for r in read_rows(out/'valid.metadata.jsonl')}
    source=dataset(out,'primevul')
    short,long=np.quantile([len(r['raw_source']) for r in source if r['split']=='train'],[.25,.75])
    for view,keys in config['views'].items():
        parts={name:[lookup[name][k] for k in keys] for name in PRIME}
        result[view]={name:dict(selected=summarize_predictions(rows,rows[0]['threshold']),fixed_0_5=summarize_predictions(rows,.5)) for name,rows in parts.items()}
        result[view]['B_vs_A']=paired_changes(parts['A'],parts['B'])
        result[view]['C_vs_A']=paired_changes(parts['A'],parts['C'])
        result[view]['C_vs_B']=paired_changes(parts['B'],parts['C'])
        common={kind:[] for kind in ('common_fp','common_fn','A_only_correct','B_only_correct','C_only_correct')}
        for key in keys:
            rows=[lookup[n][key] for n in PRIME]; y=rows[0]['label']
            correct=[r['prediction']==y for r in rows]
            if not any(correct): common['common_fn' if y else 'common_fp'].append(key)
            if sum(correct)==1: common[list(PRIME)[correct.index(True)]+'_only_correct'].append(key)
        result[view]['common_errors']={k:dict(count=len(v),sample_keys=v) for k,v in common.items()}
        result[view]['length_cutoffs_primevul_train_characters']=[float(short),float(long)]
        result[view]['subgroups']={}
        selectors={
            'short_positive':lambda k:meta[k]['label']==1 and meta[k]['characters']<=short,
            'long_negative':lambda k:meta[k]['label']==0 and meta[k]['characters']>=long,
            'source_repository_url_matched':lambda k:meta[k]['seen_source_repo'],
            'source_repository_url_unmatched':lambda k:meta[k]['repo_known'] and not meta[k]['seen_source_repo'],
            'repository_unknown':lambda k:not meta[k]['repo_known'],
            'linux':lambda k:meta[k]['project']=='torvalds/linux',
            'other_projects':lambda k:meta[k]['project']!='torvalds/linux',
            'both_inputs_complete':lambda k:not lookup['A'][k]['source_truncated'] and not lookup['C'][k]['source_truncated'],
            'only_qwen_complete':lambda k:lookup['A'][k]['source_truncated'] and not lookup['C'][k]['source_truncated'],
            'both_inputs_truncated':lambda k:lookup['A'][k]['source_truncated'] and lookup['C'][k]['source_truncated'],
            'only_codebert_complete':lambda k:not lookup['A'][k]['source_truncated'] and lookup['C'][k]['source_truncated'],
        }
        for group,select in selectors.items():
            selected=[k for k in keys if select(k)]
            if selected:
                result[view]['subgroups'][group]={n:summarize_predictions([lookup[n][k] for k in selected],parts[n][0]['threshold']) for n in PRIME}
        for name in PRIME:
            for point in ('selected','fixed_0_5'):
                table.append(dict(view=view,model=name,point=point,**result[view][name][point]))
            rows=parts[name]
            result[view][name]['truncation']=dict(samples=sum(r['source_truncated'] for r in rows),
                mean_visible_fraction=float(np.mean([r['visible_source_tokens']/max(1,r['source_token_count']) for r in rows])))
            m=result[view][name]['selected']; tpr=m['recall']; fpr=m['fpr']
            if fpr is not None:
                # Analytic prevalence-only change: same TPR/FPR, hypothetical pi=.5.
                tp,fn,fp,tn=tpr/2,(1-tpr)/2,fpr/2,(1-fpr)/2
                den=((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
                result[view][name]['prevalence_standardized_50_percent']=dict(
                    precision=tp/(tp+fp) if tp+fp else 0,recall=tpr,
                    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                    mcc=(tp*tn-fp*fn)/den if den else 0,
                    note='Analytic diagnostic, not another fitted threshold or empirical cohort.')
    costs={}
    for name in PRIME:
        latency=json.loads((out/'primevul_to_megavul'/name/'latency.json').read_text())
        threshold=predictions[name][0]['threshold']
        logit_boundary=float(np.log(decision_boundary(threshold)/(1-decision_boundary(threshold))))
        costs[name]=dict(batch1_mean_seconds=latency['mean'],batch1_p95_seconds=latency['p95'],
            probe_samples=len(latency['samples']),
            batch_probe_decision_disagreements=sum(int(r['logit']>=logit_boundary)!=lookup[name][r['sample_key']]['prediction'] for r in latency['samples']),
            bulk_forward_seconds=sum(r['batch_amortized_seconds'] for r in predictions[name]),
            scope=latency['scope'],device=latency['device'])
    write_json(out/'costs.json',costs)
    write_json(out/'comparison.json',result)
    with (out/'metrics.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,metric in zip(axes,('auc','mcc','precision')):
        for i,view in enumerate(('clean_balanced','clean_natural')):
            ax.bar(np.arange(3)+(i-.5)*.36,[result[view][n]['selected'][metric] for n in PRIME],width=.36,label=view)
        ax.set_xticks(range(3),['CodeBERT linear','CodeBERT nonlinear','Qwen linear'],rotation=15)
        ax.set_title(metric.upper());ax.axhline(0,color='gray',linewidth=.5)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'transfer.png',dpi=160);plt.close(fig)
    print(json.dumps({v:{n:r['selected'] for n,r in rs.items() if n in PRIME} for v,rs in result.items()},indent=2))


if __name__ == '__main__':
    main()
