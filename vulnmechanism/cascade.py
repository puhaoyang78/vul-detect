"""Fixed-budget, label-blind CodeBERT -> Qwen diagnostic; no router fitting."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import random
import time

from .cfg_metrics import metrics, paired_changes, decision_boundary

BUDGETS = (0, 10, 25, 50, 75, 100)
STRATEGIES = ('margin', 'probability_uncertainty', 'positive_first', 'negative_first')
PATHS = {
    'codebert512': Path('results/codebert_source_seed42/native512'),
    'codebert2048': Path('results/codebert_source_seed42/window2048'),
    'qwen': Path('results/cfg_abc_seed42/baseline'),
}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_json(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def route(keys, scores, threshold, strategy, budget_percent):
    """Accept ONLY upstream information. Return batch budget selections."""
    if strategy not in STRATEGIES or budget_percent not in BUDGETS:
        raise ValueError('unsupported fixed protocol')
    if len(keys) != len(scores) or len(set(keys)) != len(keys):
        raise ValueError('keys/scores must be aligned and unique')
    if not 0 < threshold < 1 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError('invalid upstream probabilities')
    t = decision_boundary(threshold)
    def order(i):
        p = scores[i]
        if strategy == 'probability_uncertainty':
            return (abs(p - .5), keys[i])
        margin = abs(p - t)
        if strategy in ('positive_first', 'negative_first'):
            preferred = 1 if strategy == 'positive_first' else 0
            return (int(int(p >= t) != preferred), margin, keys[i])
        return (margin, keys[i])
    count = len(keys) * budget_percent // 100
    return set(sorted(range(len(keys)), key=order)[:count])


def decision_metrics(rows):
    if not rows:
        return {'samples': 0}
    result = metrics([r['label'] for r in rows], [float(r['prediction']) for r in rows])
    # Hard decisions are not calibrated risks. Do not report their pseudo-AUC.
    result.pop('auc'); result.pop('threshold')
    return result


def validate(pred):
    reference = pred['qwen']
    if len(reference) != 741:
        raise ValueError('expected complete 741-function valid')
    keys = [r['sample_key'] for r in reference]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate identity')
    for name, rows in pred.items():
        if [r['sample_key'] for r in rows] != keys:
            raise ValueError(f'{name}: cohort/order mismatch')
        thresholds = {r['threshold'] for r in rows}
        if len(thresholds) != 1:
            raise ValueError('inconsistent threshold')
        for a, b in zip(reference, rows):
            if any(a[f] != b[f] for f in ('dataset', 'split', 'label', 'source_sha256')):
                raise ValueError('identity mismatch')
            if b['split'] != 'valid' or b['dataset'] != 'primevul':
                raise ValueError('only PrimeVul valid allowed')
            if b['prediction'] != int(b['score'] >= decision_boundary(b['threshold'])):
                raise ValueError('cached decision mismatch')
        actual = metrics([r['label'] for r in rows], [r['score'] for r in rows], rows[0]['threshold'])
        saved = json.loads((PATHS[name]/'valid.metrics.json').read_text())
        expected = saved['selected'] if name == 'qwen' else saved['metrics']
        for field in ('auc', 'mcc', 'f1', 'accuracy', 'precision', 'recall'):
            if abs(actual[field] - expected[field]) > 1e-10:
                raise ValueError(f'{name}: metric mismatch {field}')
    return keys


def changes(before, after):
    result = paired_changes(before, after)
    c = result['counts']
    result['corrected'] = c['fn_to_tp'] + c['fp_to_tn']
    result['damaged'] = c['tp_to_fn'] + c['tn_to_fp']
    return result


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(budgets=BUDGETS, strategies=STRATEGIES, seed=42, test_used=False,
        routing='batch sorting; floor(N * budget / 100); sample_key tie break',
        margin='abs(CodeBERT probability - its existing float32 decision threshold)',
        probability_uncertainty='abs(CodeBERT probability - 0.5), not threshold margin',
        class_aware='all preferred predicted class first, then other class; ascending margin within class',
        calibration='none; mixed probabilities have no reported BCE/AUC',
        caveat='Existing checkpoints and thresholds selected on this valid; exploratory, not independent confirmation',
        subgroup='train Qwen-token quartiles; unseen project relative to train; no subgroup retuning',
        learned_router='no valid fitting; no new base-model cross-training',
        latency='same CUDA device, batch1, original dtype/loaders, 8 warmups, 2 shuffled passes, synchronized wall time including tokenization/H2D/forward/readout; excludes loading, disk IO, queueing and routing')
    write_json(output/'config.json', protocol)
    pred = {n: read_rows(p/'valid.predictions.jsonl') for n, p in PATHS.items()}
    keys = validate(pred)
    feat = read_rows('results/function_statistics_seed42/features.jsonl')
    train = [r for r in feat if r['split'] == 'train']
    valid = {r['sample_key']: r for r in feat if r['split'] == 'valid'}
    if set(keys) != set(valid) or len(train) != 5886:
        raise ValueError('feature cohort mismatch')
    for r in pred['qwen']:
        if any(r[f] != valid[r['sample_key']][f] for f in ('source_sha256', 'label', 'split')):
            raise ValueError('feature identity mismatch')
    import numpy as np
    cuts = np.quantile([r['features']['tokens'] for r in train], [.25, .5, .75]).tolist()
    projects = {r['project'] for r in train}
    groups = {'all': set(keys), 'unseen_project': {k for k in keys if valid[k]['project'] not in projects}}
    for q in range(4):
        groups[f'length_q{q+1}'] = {k for k in keys if int(np.searchsorted(cuts, valid[k]['features']['tokens'], side='left')) == q}
    groups['short_positive'] = {k for k in groups['length_q1'] if valid[k]['label'] == 1}
    groups['long_negative'] = {k for k in groups['length_q4'] if valid[k]['label'] == 0}
    errors = {n: {r['sample_key'] for r in rs if r['prediction'] != r['label']} for n, rs in pred.items()}
    overlap = {f'{a}__{b}': len(errors[a] & errors[b]) for a,b in [('qwen','codebert512'),('qwen','codebert2048'),('codebert512','codebert2048')]}
    assert overlap['qwen__codebert512'] == 114 and overlap['qwen__codebert2048'] == 133
    assert len(set.intersection(*errors.values())) == 107
    results = []
    def append(name, cheap, selected, budget, strategy, rows):
        groups_report = {g: decision_metrics([r for r in rows if r['sample_key'] in ids]) for g,ids in groups.items()}
        results.append(dict(name=name, cheap=cheap, strategy=strategy, budget_percent=budget,
            qwen_calls=len(selected), qwen_fraction=len(selected)/len(keys),
            cheap_calls=0 if name=='qwen_only' else len(keys),
            selected_keys=[keys[i] for i in sorted(selected)], metrics=decision_metrics(rows),
            versus_codebert512=changes(pred['codebert512'], rows), versus_qwen=changes(pred['qwen'], rows),
            subgroups=groups_report,
            predicted_positive_upgrades=sum(pred[cheap][i]['prediction']==1 for i in selected),
            predicted_negative_upgrades=sum(pred[cheap][i]['prediction']==0 for i in selected)))
    for n in pred:
        append(n+'_only', n, set(range(len(keys))) if n=='qwen' else set(), 100 if n=='qwen' else 0, 'only', pred[n])
    for cheap in ('codebert512','codebert2048'):
        source = pred[cheap];t = source[0]['threshold'];scores = [r['score'] for r in source]
        for strategy in STRATEGIES:
            for budget in BUDGETS:
                selected = route(keys, scores, t, strategy, budget)
                rows = [pred['qwen'][i] if i in selected else r for i,r in enumerate(source)]
                append(f'{cheap}_{strategy}_{budget}', cheap, selected, budget, strategy, rows)
    # Oracle is explicitly label-informed and never enters practical Pareto sets.
    oracle = {}
    for cheap in ('codebert512','codebert2048'):
        source = pred[cheap]
        useful = [i for i in range(len(keys)) if source[i]['prediction']!=source[i]['label'] and pred['qwen'][i]['prediction']==source[i]['label']]
        rows = [pred['qwen'][i] if i in useful else source[i] for i in range(len(keys))]
        oracle[cheap] = {'label_informed_not_deployable':True,'useful_calls':len(useful),'fraction':len(useful)/len(keys),'metrics':decision_metrics(rows),'max_corrected_at_budget':{str(b):min(len(useful),len(keys)*b//100) for b in BUDGETS}}
    # Post-evaluation gain analysis; no resulting bins feed routing.
    gain = {}
    for cheap in ('codebert512','codebert2048'):
        src = pred[cheap];t=decision_boundary(src[0]['threshold'])
        bins = {}
        margin_order = sorted(range(len(keys)), key=lambda i:(abs(src[i]['score']-t),keys[i]))
        for j in range(5):bins[f'margin_rank_quintile{j+1}'] = set(margin_order[j*len(keys)//5:(j+1)*len(keys)//5])
        for j,(lo,hi) in enumerate(zip([0,.05,.15,.3,.5,.75],[.05,.15,.3,.5,.75,1.0000001])):
            bins[f'probability_{lo}_{min(hi,1)}'] = {i for i,r in enumerate(src) if lo<=r['score']<hi}
        for y in (0,1):bins[f'predicted_{y}'] = {i for i,r in enumerate(src) if r['prediction']==y}
        for g,ids in groups.items():bins[g] = {i for i,k in enumerate(keys) if k in ids}
        gain[cheap] = {}
        for g,ix in bins.items():
            if not ix:continue
            a=[src[i] for i in sorted(ix)];b=[pred['qwen'][i] for i in sorted(ix)]
            d=changes(a,b);d.pop('sample_keys')
            gain[cheap][g] = {'samples':len(ix), **d, 'net_per_call':d['net_corrected']/len(ix)}
    result=dict(identity_verified=True, samples=len(keys), train_length_cuts=cuts,
        overlap=overlap, three_model_common_errors=107, results=results, oracle=oracle, gain_bins=gain,
        no_oof='Existing supplied checkpoints saw all 5886 train members; no verified OOF/independent routing cohort supplied. Do not fit router on in-sample train or this valid.')
    write_json(output/'comparison.json', result)
    fields=['name','qwen_calls','qwen_fraction','accuracy','precision','recall','f1','mcc','tp','fp','tn','fn','corrected_vs_cb512','damaged_vs_cb512','corrected_vs_qwen','damaged_vs_qwen']
    with (output/'metrics.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in results:
            w.writerow({**{k:r[k] for k in fields[:3]},**{k:r['metrics'][k] for k in fields[3:12]},'corrected_vs_cb512':r['versus_codebert512']['corrected'],'damaged_vs_cb512':r['versus_codebert512']['damaged'],'corrected_vs_qwen':r['versus_qwen']['corrected'],'damaged_vs_qwen':r['versus_qwen']['damaged']})
    print(json.dumps({'output':str(output),'configurations':len(results),'overlap':overlap,'oracle':oracle},indent=2))


def measure(output, device):
    """Profile the actual checkpoints; all calls here are timing, not routing."""
    import gc
    import torch
    from transformers import AutoTokenizer
    from . import model as core
    from .cfg_data import identity
    if (output/'latency.json').exists() or (output/'latency.rows.jsonl').exists():
        raise FileExistsError('timing artifacts already exist')
    pred={n:read_rows(p/'valid.predictions.jsonl') for n,p in PATHS.items()}
    keys=validate(pred);wanted=set(keys);records={}
    for line in Path('data/function_dataset.jsonl').open():
        r=json.loads(line)
        if r.get('split')=='valid' and r.get('sample_key') in wanted:records[r['sample_key']]=r
    if set(records)!=wanted:raise ValueError('missing raw function')
    for ref in pred['qwen']:
        if identity(records[ref['sample_key']]) != {f:ref[f] for f in ('sample_key','dataset','split','label','source_sha256')}:raise ValueError('raw source mismatch')
    dev=torch.device(device);torch.set_num_threads(2)
    summary={'device':str(dev),'gpu':torch.cuda.get_device_name(dev),'torch':torch.__version__,'passes':2,'warmup_calls':8,'batch_size':1,'scope':'synchronized wall ms including tokenization/H2D/forward/readout; excludes checkpoint load, IO, routing and queueing','models':{}}
    with (output/'latency.rows.jsonl').open('x') as handle:
        for name,folder in PATHS.items():
            core._seed_everything(42)
            checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
            tok=AutoTokenizer.from_pretrained(checkpoint['model_path'],trust_remote_code=True)
            if tok.pad_token_id is None:tok.pad_token=tok.eos_token
            builder=core.InputBuilder(tok,source_max_length=checkpoint['source_max_length'],context_max_length=checkpoint['context_max_length'])
            model=core._load_model(checkpoint,device=dev)
            assert checkpoint['decision_threshold']==pred[name][0]['threshold']
            def infer(i):
                logits=core._forward_batch(model,[records[keys[i]]],builder,variant=checkpoint['variant'],excluded_groups=tuple(checkpoint['excluded_groups']),device=dev)
                return float(torch.sigmoid(logits.float()).cpu()[0])
            maximum=0.;mismatch=0;times=[]
            with torch.no_grad():
                warm=sorted(range(len(keys)),key=lambda i:pred[name][i]['source_token_count'])
                for j in range(8):infer(warm[j*(len(keys)-1)//7])
                for repeat in range(2):
                    order=list(range(len(keys)));random.Random(42+repeat).shuffle(order)
                    for i in order:
                        torch.cuda.synchronize(dev);start=time.perf_counter();p=infer(i);torch.cuda.synchronize(dev)
                        ms=(time.perf_counter()-start)*1000;times.append(ms)
                        maximum=max(maximum,abs(p-pred[name][i]['score']))
                        mismatch+=int(int(p>=decision_boundary(checkpoint['decision_threshold']))!=pred[name][i]['prediction'])
                        handle.write(json.dumps({'model':name,'sample_key':keys[i],'pass':repeat,'ms':ms,'score':p})+'\n')
                    handle.flush()
            import numpy as np
            summary['models'][name]={'model_path':checkpoint['model_path'],'variant':checkpoint['variant'],'source_max_length':checkpoint['source_max_length'],'selected_epoch':checkpoint.get('selected_epoch'),'training_config':checkpoint.get('training_config'),'dtype':str(next(model.encoder.parameters()).dtype),'total_parameters':sum(p.numel() for p in model.parameters()),'mean_ms':float(np.mean(times)),'p50_ms':float(np.percentile(times,50)),'p95_ms':float(np.percentile(times,95)),'p99_ms':float(np.percentile(times,99)),'max_probability_delta':maximum,'decision_mismatches':mismatch}
            print(name,json.dumps(summary['models'][name]),flush=True)
            if mismatch or maximum>1e-4:raise ValueError('checkpoint replay mismatch; timing retained, do not treat as verified')
            del model,checkpoint;gc.collect();torch.cuda.empty_cache()
    write_json(output/'latency.json',summary)


def plot(output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads((output/'comparison.json').read_text())
    latency_path=output/'latency.json'
    timing=None
    if latency_path.exists():
        timing={n:{} for n in PATHS}
        for r in read_rows(output/'latency.rows.jsonl'):
            timing[r['model']].setdefault(r['sample_key'],[]).append(r['ms'])
        timing={n:{k:float(np.mean(v)) for k,v in d.items()} for n,d in timing.items()}
        if any(len(v)!=741 for v in timing.values()):raise ValueError('incomplete timings')
    cost=[]
    for r in data['results']:
        row={'name':r['name'],'qwen_fraction':r['qwen_fraction'],'accuracy':r['metrics']['accuracy'],'mcc':r['metrics']['mcc'],'f1':r['metrics']['f1']}
        # Sensitivity proxies explicitly assume per-call cost ratio, not GPU time.
        row['call_cost_proxy']={str(ratio): (1. if r['name']=='qwen_only' else ratio+r['qwen_fraction']) for ratio in (.01,.05,.1)}
        if timing:
            if r['name']=='qwen_only':ms=list(timing['qwen'].values())
            else:
                selected=set(r['selected_keys'])
                ms=[t+(timing['qwen'][k] if k in selected else 0) for k,t in timing[r['cheap']].items()]
            row.update(component_estimated_mean_ms=float(np.mean(ms)),component_estimated_p95_ms=float(np.percentile(ms,95)),relative_measured_component_cost=float(sum(ms)/sum(timing['qwen'].values())))
        cost.append(row)
    xfield='relative_measured_component_cost' if timing else 'qwen_fraction'
    # Evaluate each detector/strategy's frontier and the full fixed-rule set.
    for row in cost:
        row['pareto_accuracy']=not any(o[xfield]<=row[xfield] and o['accuracy']>=row['accuracy'] and (o[xfield]<row[xfield] or o['accuracy']>row['accuracy']) for o in cost)
        row['pareto_mcc']=not any(o[xfield]<=row[xfield] and o['mcc']>=row['mcc'] and (o[xfield]<row[xfield] or o['mcc']>row['mcc']) for o in cost)
    write_json(output/'cost_comparison.json',{'cost_scope':'sum of separately measured per-function components; NOT deployed end-to-end latency or energy','rows':cost})
    fig,axes=plt.subplots(1,3,figsize=(15,4.3))
    costs={r['name']:r for r in cost}
    for ax,metric in zip(axes,['accuracy','mcc','f1']):
        for cheap,style in [('codebert512','-'),('codebert2048','--')]:
            for strategy in STRATEGIES:
                rs=[r for r in data['results'] if r['cheap']==cheap and r['strategy']==strategy]
                ax.plot([costs[r['name']][xfield] for r in rs],[r['metrics'][metric] for r in rs],style,marker='o',markersize=3,label=cheap+' '+strategy)
        q=costs['qwen_only'];ax.scatter([q[xfield]],[q[metric]],marker='*',s=110,color='black',label='Qwen only')
        front=sorted([r for r in cost if r.get('pareto_'+metric,False)],key=lambda r:r[xfield])
        if front:ax.plot([r[xfield] for r in front],[r[metric] for r in front],':',color='black',alpha=.6)
        ax.set_xlabel('Relative measured component cost (Qwen only=1)' if timing else 'Qwen call fraction');ax.set_ylabel(metric);ax.grid(alpha=.2)
    axes[-1].legend(fontsize=6);fig.tight_layout();fig.savefig(output/'performance_cost.png',dpi=180);fig.savefig(output/'performance_cost.svg');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','measure','plot']);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.action=='run':run(a.output)
    elif a.action=='measure':measure(a.output,a.device)
    else:plot(a.output)


if __name__=='__main__':main()
