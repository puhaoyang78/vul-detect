"""Train-side, out-of-base-sample post-hoc deferral experiments."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import time
import numpy as np
from .cascade import PATHS, read_rows, write_json, validate, decision_metrics, changes, route
from .cfg_metrics import decision_boundary

OLD = Path('results/codebert_qwen_cascade_seed42')
FN_COSTS = (1, 3)
LAMBDA = .1  # one mean Qwen call costs 0.1 false-positive error units


def jsonl(path, rows):
    with Path(path).open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')


def threshold_rows(rows):
    """All attainable deterministic score thresholds, preserving score ties."""
    y=[r['label'] for r in rows];p=[r['score'] for r in rows]
    # Existing scores are float32. max+epsilon excludes all without rounding ambiguity.
    thresholds=[1.000001]+sorted(set(p),reverse=True)
    result=[]
    for t in thresholds:
        decisions=[dict(r,prediction=int(r['score']>=t)) for r in rows]
        m=decision_metrics(decisions);m.update(threshold=t,tpr=m['recall'],fpr=m['fp']/sum(v==0 for v in y))
        result.append(m)
    return result


def matched_points(curve, target):
    eligible=[r for r in curve if r['fp']<=target['fp']]
    at_fp=max(eligible,key=lambda r:(r['tp'],-r['fp']))
    at_recall=min((r for r in curve if r['tp']>=target['tp']),key=lambda r:(r['fp'],r['tp']))
    return dict(at_most_same_fp=at_fp,at_least_same_recall=at_recall)


def fair(output):
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',dict(seed=42,test_used_for_design=False,
        budgets=[25,50],fn_costs=FN_COSTS,fp_cost=1,lambda_cost=LAMBDA,
        router='StandardScaler + LogisticRegression C=1 lbfgs max_iter=2000, no class balancing or hyperparameter search',
        features=['CodeBERT probability','abs(probability - .15)','predicted class','log1p(CodeBERT source token count)'],
        confidence_target='CodeBERT error probability; no Qwen reliability labels',
        utility_target='five joint outcome categories: no change, fix FN, fix FP, introduce FN, introduce FP',
        cost_model='Ridge alpha=1 on log Qwen latency from log1p(CodeBERT token count); mean train Qwen time normalization',
        partitions='original train only; SHA256(seed42:repo) mod 5 == 0 internal check, rest router fit; no refit on check',
        fixed_calls='rank predicted net utility; forced top-k even if negative utility, report negative fraction',
        fixed_cost='same predicted incremental cost budget as margin at 25/50%; rank-prefix until predicted budget exhausted, report realized cost mismatch',
        calibration='internal-check utility reliability bins fixed at [<-.1,-.1..0,0...1,.1...3,>=.3]; no post-hoc calibration fit',
        admission='minimum 100 fit and 50 check functions; both labels >=20 fit and >=10 check; grouped provenance exclusions; no test source/labels/predictions read'))
    pred={n:read_rows(p/'valid.predictions.jsonl') for n,p in PATHS.items()};validate(pred)
    curves={n:threshold_rows(pred[n]) for n in ('codebert512','qwen')}
    result={'diagnostic_only':True,'no_threshold_adopted':True,'curves':curves,'comparisons':{}}
    old=json.loads((OLD/'comparison.json').read_text())
    for budget in (25,50):
        c=next(r for r in old['results'] if r['name']==f'codebert512_margin_{budget}')
        ix=set(c['selected_keys']);cascade=[pred['qwen'][i] if r['sample_key'] in ix else r for i,r in enumerate(pred['codebert512'])]
        entry={'cascade':c['metrics'],'single_models':{}}
        for n,curve in curves.items():
            pts=matched_points(curve,c['metrics'])
            for point in pts.values():
                decisions=[dict(r,prediction=int(r['score']>=point['threshold'])) for r in pred[n]]
                point['versus_cascade']=changes(cascade,decisions)
                point['versus_original']=changes(pred[n],decisions)
            entry['single_models'][n]=pts
        result['comparisons'][str(budget)]=entry
    from sklearn.metrics import average_precision_score,roc_auc_score
    result['standalone']={n:{'auc':float(roc_auc_score([r['label'] for r in pred[n]],[r['score'] for r in pred[n]])), 'average_precision':float(average_precision_score([r['label'] for r in pred[n]],[r['score'] for r in pred[n]]))} for n in curves}
    write_json(output/'threshold_diagnostic.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for n,curve in curves.items():
        axes[0].plot([r['fpr'] for r in curve],[r['recall'] for r in curve],label=n)
        axes[1].step([r['recall'] for r in curve],[r['precision'] if r['tp']+r['fp'] else 1 for r in curve],where='post',label=n)
    for budget,entry in result['comparisons'].items():
        r=entry['cascade'];axes[0].scatter(r['fp']/368,r['recall'],label='margin '+budget+'%');axes[1].scatter(r['recall'],r['precision'],label='margin '+budget+'%')
    axes[0].set(xlabel='FPR',ylabel='Recall');axes[1].set(xlabel='Recall',ylabel='Precision')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'roc_pr.png',dpi=180);plt.close(fig)
    print(json.dumps({'single':result['standalone'],'matched':{b:{n:{p:{k:v[k] for k in ('threshold','fp','fn','recall','f1','mcc')} for p,v in pts.items()} for n,pts in e['single_models'].items()} for b,e in result['comparisons'].items()}},indent=2))


def partition(repo):
    return 'check' if int(hashlib.sha256(('seed42:'+repo).encode()).hexdigest(),16)%5==0 else 'fit'


def audit(output):
    from .prepare_benchmark import NearIndex, unit
    from .syntax import parser_for
    features=read_rows('results/function_statistics_seed42/features.jsonl')
    used={r['sample_key'] for r in features if r['split']=='train'}
    if len(used)!=5886:raise ValueError('unexpected base training cohort')
    train=[];protected=[];valid=[]
    # Held-out portions of this PREEXISTING manifest contribute ONLY exclusion metadata.
    # No raw official test file, test source, label, score or test-driven statistic is inspected.
    for line in Path('data/benchmark/manifest.jsonl').open():
        r=json.loads(line)
        if r['dataset']=='primevul' and r['split']=='train':train.append(r)
        else:
            protected.append({k:r.get(k) for k in ('sample_key','repo','commit','counterpart_groups')})
            if r['dataset']=='primevul' and r['split']=='valid':valid.append(r)
    base=[r for r in train if r['sample_key'] in used]
    if {r['sample_key'] for r in base}!=used:raise ValueError('base cohort not in audited manifest')
    for r in base:
        f=next(x for x in features if x['sample_key']==r['sample_key'])
        if hashlib.sha256(r['function'].encode()).hexdigest()!=f['source_sha256'] or r['label']!=f['label']:raise ValueError('base provenance mismatch')
    protected += [{k:r.get(k) for k in ('sample_key','repo','commit','counterpart_groups')} for r in base]
    revisions=defaultdict(set)
    for r in protected:
        if r['repo']:revisions[r['repo']].add(r['commit'])
    groups={g for r in protected for g in (r['counterpart_groups'] or [])}
    near=NearIndex()
    for r in base+valid:near.add(unit([r]))
    errors={r['sample_key']:r for r in read_rows('data/function_dataset.errors.jsonl') if r.get('dataset')=='primevul' and r.get('split')=='train'}
    candidates=[r for r in train if r['sample_key'] not in used]
    rejected=[];accepted=[]
    for r in candidates:
        reason=None
        if not r['repo'] or not re.fullmatch('[0-9a-f]{40}',r['commit']):reason='unresolved_provenance'
        elif any(r['commit'].startswith(c) or c.startswith(r['commit']) for c in revisions[r['repo']] if c):reason='shared_repair_commit'
        elif groups.intersection(r['counterpart_groups']):reason='shared_repair_counterpart'
        elif near.matches(unit([r])):reason='near_duplicate_base_or_valid'
        if reason:rejected.append({'sample_key':r['sample_key'],'reason':reason});continue
        source=r['function'];lang=r['language'] if r['language'] in ('c','cpp') else 'cpp'
        tree=parser_for(lang).parse(source.encode());err=errors.get(r['sample_key'])
        accepted.append(dict(sample_key=r['sample_key'],dataset='primevul',split='train',router_split=partition(r['repo']),label=r['label'],raw_source=source,source_sha256=hashlib.sha256(source.encode()).hexdigest(),repo=r['repo'],commit=r['commit'],counterpart_groups=r['counterpart_groups'],source_file=r['source_file'],source_row=r['source_row'],language=r['language'],source_chars=len(source),syntax_error=tree.root_node.has_error,not_in_base_reason=err['stage'] if err else 'no_saved_build_failure',build_error_type=err.get('error_type') if err else None))
    # Cross-project clones between router partitions must not leak into internal checking.
    check_index=NearIndex()
    for r in accepted:
        if r['router_split']=='check':check_index.add(unit([dict(r,function=r['raw_source'])]))
    clean=[]
    for r in accepted:
        if r['router_split']=='fit' and check_index.matches(unit([dict(r,function=r['raw_source'])])):
            rejected.append({'sample_key':r['sample_key'],'reason':'near_duplicate_router_check'})
        else:clean.append(r)
    accepted=clean
    # Verify each admitted source/label against ORIGINAL official train only.
    index={r['sample_key']:r for r in accepted};official=Counter();found=set()
    rawpath=Path('/home/PublicData/PHY-data/vul_detect/data/PrimeVul_v0.1/primevul_train.jsonl')
    for line in rawpath.open():
        r=json.loads(line);official[r['target']]+=1;k=f"primevul:{r['idx']}"
        if k in index:
            a=index[k]
            if a['label']!=r['target'] or a['raw_source']!=r['func']:raise ValueError('official train identity mismatch')
            found.add(k)
    if found!=set(index):raise ValueError('not in official train')
    exclusions=Counter()
    for line in Path('data/benchmark/score_4/exclusions.jsonl').open():
        r=json.loads(line)
        if r['dataset']=='primevul' and r['split']=='train':exclusions[r['reason']]+=1
    def summary(rows):
        return {'samples':len(rows),'labels':dict(Counter(r['label'] for r in rows)),'projects':len({r['repo'] for r in rows}),'commits':len({(r['repo'],r['commit']) for r in rows}),'chars_quartiles':np.quantile([r['source_chars'] for r in rows],[.25,.5,.75]).tolist() if rows else [],'syntax_errors':sum(r['syntax_error'] for r in rows),'build_failure_reasons':dict(Counter(r['not_in_base_reason'] for r in rows)),'project_distribution':dict(Counter(r['repo'] for r in rows))}
    splits={s:summary([r for r in accepted if r['router_split']==s]) for s in ('fit','check')}
    ready=all(splits[s]['samples']>=minimum and all(splits[s]['labels'].get(y,0)>=perclass for y in (0,1)) for s,minimum,perclass in [('fit',100,20),('check',50,10)])
    report=dict(official_train_labels=dict(official),base_used=len(used),manifest_train=len(train),unseen_manifest_candidates=len(candidates),candidate_labels=dict(Counter(r['label'] for r in candidates)),admitted=summary(accepted),partitions=splits,rejected=dict(Counter(r['reason'] for r in rejected)),original_exclusion_reasons=dict(exclusions),ready=ready,
        provenance='official train exact code/label match; no current base train members; project+commit prefix and counterpart disjoint against base and preexisting held-out metadata',
        near_check='existing benchmark exact/normalized/counterpart/0.90 token-shingle screening reused; extra 0.90 near check against base/valid and between fit/check',
        test_boundary='no raw test source/label/prediction read; existing manifest held-out metadata used ONLY to exclude shared provenance. Prior near-screen covers retained benchmark held-out members, not proof against every excluded official test function.',
        selection_bias='ONLY already-screened manifest train functions absent from CPG-success base cohort; largely graph failures, not representative of all unused official train. Unscreened downsampled negatives are not silently admitted.',
        source_quality='parse errors logged, not repaired; no CPG success admission; raw function labels unchanged')
    jsonl(output/'members.jsonl',accepted);jsonl(output/'excluded_members.jsonl',rejected);write_json(output/'audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('admitted','partitions')},indent=2));print({s:{k:v for k,v in x.items() if k!='project_distribution'} for s,x in splits.items()})


def predict(output, device):
    """Reuse existing encoders and source-only forwards on admitted original train."""
    import gc
    import torch
    from transformers import AutoTokenizer
    from . import model as core
    audit_report=json.loads((output/'audit.json').read_text())
    if not audit_report['ready']:raise ValueError('routing membership gate failed')
    rows=read_rows(output/'members.jsonl')
    dev=torch.device(device);torch.set_num_threads(2)
    for name in ('codebert512','qwen'):
        path=output/f'{name}.predictions.jsonl'
        if path.exists():raise FileExistsError(path)
        cp=torch.load(PATHS[name]/'best.pt',map_location='cpu',weights_only=False)
        tok=AutoTokenizer.from_pretrained(cp['model_path'],trust_remote_code=True)
        if tok.pad_token_id is None:tok.pad_token=tok.eos_token
        builder=core.InputBuilder(tok,source_max_length=cp['source_max_length'],context_max_length=cp['context_max_length'])
        core._seed_everything(42);model=core._load_model(cp,device=dev)
        count=[len(builder._encode(r['raw_source'])) for r in rows]
        order=list(range(len(rows)));random.Random(42).shuffle(order)
        result=[]
        def infer(i):
            z=core._forward_batch(model,[rows[i]],builder,variant=cp['variant'],excluded_groups=tuple(cp['excluded_groups']),device=dev)
            return float(z.float().cpu()[0])
        with torch.no_grad():
            warm=sorted(order,key=lambda i:count[i])
            for j in range(8):infer(warm[j*(len(warm)-1)//7])
            for i in order:
                torch.cuda.synchronize(dev);start=time.perf_counter();z=infer(i);torch.cuda.synchronize(dev)
                ms=(time.perf_counter()-start)*1000
                # Match legacy float32 sigmoid decision semantics.
                probability=float(torch.sigmoid(torch.tensor(z,dtype=torch.float32)))
                result.append({k:rows[i][k] for k in ('sample_key','dataset','split','router_split','label','source_sha256','repo','commit')})
                result[-1].update(score=probability,logit=z,threshold=cp['decision_threshold'],prediction=int(probability>=decision_boundary(cp['decision_threshold'])),source_token_count=count[i],ms=ms)
        ix={r['sample_key']:r for r in result};jsonl(path,[ix[r['sample_key']] for r in rows])
        write_json(output/f'{name}.inference.json',dict(checkpoint=str(PATHS[name]/'best.pt'),model_path=cp['model_path'],variant=cp['variant'],source_max_length=cp['source_max_length'],selected_epoch=cp.get('selected_epoch'),initial_checkpoint=cp.get('initial_checkpoint'),training_config=cp.get('training_config'),device=str(dev),gpu=torch.cuda.get_device_name(dev),dtype=str(next(model.encoder.parameters()).dtype),torch=torch.__version__,functions=len(rows),batch_size=1,warmups=8,passes=1,mean_ms=float(np.mean([r['ms'] for r in result])),p95_ms=float(np.percentile([r['ms'] for r in result],95)),cost_scope='synchronized tokenization/H2D/forward/output; excludes load and source token counting'))
        print(name,'completed',len(rows),'functions',flush=True)
        del model,cp;gc.collect();torch.cuda.empty_cache()


def router_inputs(scores, token_counts, threshold=.15):
    """Only upstream quantities; no label or downstream outputs accepted."""
    p=np.asarray(scores,dtype=float);n=np.asarray(token_counts,dtype=float)
    if p.shape!=n.shape or p.ndim!=1 or np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)) or np.any(~np.isfinite(n)) or np.any(n<0):raise ValueError('invalid upstream inputs')
    return np.column_stack((p,np.abs(p-decision_boundary(threshold)),p>=decision_boundary(threshold),np.log1p(n)))


def joint_outcomes(y, cb, q):
    y=np.asarray(y);cb=np.asarray(cb);q=np.asarray(q)
    if y.shape!=cb.shape or y.shape!=q.shape or any(np.any(~np.isin(v,[0,1])) for v in (y,cb,q)):raise ValueError('invalid outcome labels')
    z=np.zeros(len(y),dtype=int)
    z[(y==1)&(cb==0)&(q==1)]=1
    z[(y==0)&(cb==1)&(q==0)]=2
    z[(y==1)&(cb==1)&(q==0)]=3
    z[(y==0)&(cb==0)&(q==1)]=4
    return z


def gain_from_probabilities(p, classes, fn_cost):
    table=np.array([0,fn_cost,1,-fn_cost,-1],dtype=float)
    return np.asarray(p)@table[np.asarray(classes,dtype=int)]


def allocate(keys, scores, budget, costs=None, cap=None):
    ix=sorted(range(len(keys)),key=lambda i:(-float(scores[i]),keys[i]))
    if costs is None:return set(ix[:len(keys)*budget//100])
    chosen=set();spent=0.
    for i in ix:
        if spent+costs[i]>cap+1e-10:break
        chosen.add(i);spent+=costs[i]
    return chosen


def learn(output):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression,Ridge
    from sklearn.metrics import brier_score_loss,mean_absolute_error
    import joblib
    if (output/'router.joblib').exists() or (output/'results.json').exists():raise FileExistsError('router outputs exist')
    cb=read_rows(output/'codebert512.predictions.jsonl');q=read_rows(output/'qwen.predictions.jsonl')
    if len(cb)!=len(q):raise ValueError('joint cohort mismatch')
    for a,b in zip(cb,q):
        if any(a[k]!=b[k] for k in ('sample_key','label','router_split','source_sha256','repo','commit')):raise ValueError('joint identity mismatch')
    x=router_inputs([r['score'] for r in cb],[r['source_token_count'] for r in cb]);y=np.array([r['label'] for r in cb]);pc=np.array([r['prediction'] for r in cb]);pq=np.array([r['prediction'] for r in q]);z=joint_outcomes(y,pc,pq)
    tr=np.array([r['router_split']=='fit' for r in cb]);cal=~tr
    outcomes={s:dict(Counter(map(int,z[ix]))) for s,ix in [('fit',tr),('check',cal)]}
    if sum(np.isin(z[tr],[1,2]))<10 or sum(np.isin(z[tr],[3,4]))<10:raise ValueError('fewer than 10 useful or harmful upgrades in router-fit; stop')
    def logistic():return make_pipeline(StandardScaler(),LogisticRegression(C=1,solver='lbfgs',max_iter=2000,random_state=42))
    confidence=logistic().fit(x[tr],(pc[tr]!=y[tr]).astype(int))
    utility=logistic().fit(x[tr],z[tr])
    cost=make_pipeline(StandardScaler(),Ridge(alpha=1)).fit(x[tr,3:4],np.log([r['ms'] for i,r in enumerate(q) if tr[i]]))
    mean_cost=float(np.mean([r['ms'] for i,r in enumerate(q) if tr[i]]))
    for model in (confidence,utility):
        if max(model[-1].n_iter_)>=2000:raise ValueError('router optimizer did not converge')
    joblib.dump(dict(confidence=confidence,utility=utility,cost=cost,cost_normalization_ms=mean_cost,features=['cb_probability','margin','cb_class','log1p_cb_tokens'],fn_costs=FN_COSTS,lambda_cost=LAMBDA),output/'router.joblib')
    predvalid={n:read_rows(p/'valid.predictions.jsonl') for n,p in PATHS.items()};validate(predvalid)
    cv=predvalid['codebert512'];qv=predvalid['qwen']
    xv=router_inputs([r['score'] for r in cv],[r['source_token_count'] for r in cv])
    features=read_rows('results/function_statistics_seed42/features.jsonl');projects={r['project'] for r in features if r['split']=='train'}
    vf={r['sample_key']:r for r in features if r['split']=='valid'}
    cuts=json.loads((OLD/'comparison.json').read_text())['train_length_cuts']
    times=defaultdict(lambda:defaultdict(list))
    for r in read_rows(OLD/'latency.rows.jsonl'):times[r['model']][r['sample_key']].append(r['ms'])
    times={n:{k:float(np.mean(v)) for k,v in d.items()} for n,d in times.items()}
    result={'test_used':False,'outcomes':outcomes,'models':{'confidence_parameters':int(confidence[-1].coef_.size+confidence[-1].intercept_.size),'utility_parameters':int(utility[-1].coef_.size+utility[-1].intercept_.size),'fit_only':True},'cohorts':{}}
    scores_saved=[]
    curves={n:threshold_rows(predvalid[n]) for n in ('codebert512','qwen')}
    for cohort,a,b,xx in [('internal_check',[r for i,r in enumerate(cb) if cal[i]],[r for i,r in enumerate(q) if cal[i]],x[cal]),('valid',cv,qv,xv)]:
        keys=[r['sample_key'] for r in a];truth=np.array([r['label'] for r in a]);pa=np.array([r['prediction'] for r in a]);pb=np.array([r['prediction'] for r in b])
        actual_cost=np.array([r['ms'] for r in b]) if cohort=='internal_check' else np.array([times['qwen'][k] for k in keys])
        cheap_cost=np.array([r['ms'] for r in a]) if cohort=='internal_check' else np.array([times['codebert512'][k] for k in keys])
        estimated_cost=np.exp(cost.predict(xx[:,3:4]))
        pr=utility.predict_proba(xx);pe=confidence.predict_proba(xx)[:,list(confidence.classes_).index(1)]
        report={'samples':len(a),'base':{'codebert512':decision_metrics(a),'qwen':decision_metrics(b)},'cost_prediction_mae_ms':float(mean_absolute_error(actual_cost,estimated_cost)),'confidence_brier':float(brier_score_loss(pa!=truth,pe)),'settings':[],'utility_calibration':{}}
        for w in FN_COSTS:
            g=gain_from_probabilities(pr,utility.classes_,w);u=g-LAMBDA*estimated_cost/mean_cost
            confidence_score=pe*np.where(pa==0,w,1)-LAMBDA*estimated_cost/mean_cost
            true_gain=(pa!=truth)*np.where(truth==1,w,1)-(pb!=truth)*np.where(truth==1,w,1)
            bins=[]
            for lo,hi in [(-1e9,-.1),(-.1,0),(0,.1),(.1,.3),(.3,1e9)]:
                ix=(g>=lo)&(g<hi)
                if np.any(ix):bins.append({'range':[lo,hi],'n':int(sum(ix)),'predicted_gain':float(np.mean(g[ix])),'observed_gain':float(np.mean(true_gain[ix]))})
            worthwhile=u>0
            report['utility_calibration'][str(w)]={'bins':bins,'gain_mae':float(np.mean(np.abs(g-true_gain))),'positive_net_utility_n':int(sum(worthwhile)),'positive_utility_observed_gain':float(np.mean(true_gain[worthwhile])) if np.any(worthwhile) else None,'positive_utility_observed_net':float(np.mean(true_gain[worthwhile]-LAMBDA*actual_cost[worthwhile]/mean_cost)) if np.any(worthwhile) else None}
            for i,k in enumerate(keys):scores_saved.append({'sample_key':k,'cohort':cohort,'fn_cost':w,'estimated_gain':float(g[i]),'estimated_qwen_ms':float(estimated_cost[i]),'net_utility_score':float(u[i]),'confidence_score':float(confidence_score[i]),'label_used_in_route':False})
            for budget in (25,50):
                margin_selected=route(keys,[r['score'] for r in a],a[0]['threshold'],'margin',budget)
                cap=float(sum(estimated_cost[i] for i in margin_selected))
                for method,score in [('margin',-xx[:,1]),('confidence',confidence_score),('utility',u)]:
                    for mode in ('calls','predicted_cost'):
                        selected=allocate(keys,score,budget) if mode=='calls' else allocate(keys,score,budget,costs=estimated_cost,cap=cap)
                        out=[b[i] if i in selected else r for i,r in enumerate(a)]
                        m=decision_metrics(out);rel=float((sum(cheap_cost)+sum(actual_cost[i] for i in selected))/sum(actual_cost))
                        baseline=[b[i] if i in margin_selected else r for i,r in enumerate(a)]
                        entry={'method':method,'fn_cost':w,'budget_percent':budget,'budget_mode':mode,'qwen_calls':len(selected),'qwen_fraction':len(selected)/len(keys),'relative_component_cost':rel,'mean_component_ms':float((sum(cheap_cost)+sum(actual_cost[i] for i in selected))/len(keys)),'estimated_qwen_cost_cap_ms':cap,'selected_estimated_qwen_ms':float(sum(estimated_cost[i] for i in selected)),'actual_qwen_ms':float(sum(actual_cost[i] for i in selected)),'metrics':m,'classification_risk':float((m['fp']+w*m['fn'])/len(keys)),'net_objective':float((m['fp']+w*m['fn'])/len(keys)+LAMBDA*sum(actual_cost[i] for i in selected)/(mean_cost*len(keys))),'versus_codebert':changes(a,out),'versus_qwen':changes(b,out),'versus_margin_calls':changes(baseline,out),'selected_keys':[keys[i] for i in sorted(selected)],'selected_negative_estimated_utility':int(sum(u[i]<0 for i in selected))}
                        if cohort=='valid':
                            masks={'short_positive':{k for k in keys if vf[k]['features']['tokens']<=cuts[0] and vf[k]['label']==1},'long_negative':{k for k in keys if vf[k]['features']['tokens']>cuts[2] and vf[k]['label']==0},'unseen_project':{k for k in keys if vf[k]['project'] not in projects}}
                            entry['subgroups']={n:decision_metrics([r for r in out if r['sample_key'] in ids]) for n,ids in masks.items()}
                            entry['single_model_matched']={n:matched_points(curve,m) for n,curve in curves.items()}
                        report['settings'].append(entry)
        result['cohorts'][cohort]=report
    write_json(output/'results.json',result);jsonl(output/'routing_scores.jsonl',scores_saved)
    import csv
    fields=['cohort','method','fn_cost','budget_percent','budget_mode','qwen_calls','relative_component_cost','accuracy','precision','recall','f1','mcc','tp','fp','tn','fn','classification_risk','net_objective']
    with (output/'metrics.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for cohort,report in result['cohorts'].items():
            for r in report['settings']:writer.writerow({'cohort':cohort,**{k:r[k] for k in fields[1:7]},**{k:r['metrics'][k] for k in fields[7:16]},**{k:r[k] for k in fields[16:]}})
    print(json.dumps({'outcomes':outcomes,'models':result['models'],'internal_calibration':result['cohorts']['internal_check']['utility_calibration']},indent=2))


def plot(output):
    """Plot saved fixed-policy results; does not reselect or retrain anything."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=json.loads((output/'results.json').read_text())['cohorts']['valid']
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for col,w in enumerate(FN_COSTS):
        for method in ('margin','confidence','utility'):
            rows=[r for r in result['settings'] if r['fn_cost']==w and r['method']==method and r['budget_mode']=='calls']
            xs=[r['relative_component_cost'] for r in rows]
            axes[0,col].plot(xs,[r['metrics']['mcc'] for r in rows],'o-',label=method)
            axes[1,col].plot(xs,[r['classification_risk'] for r in rows],'o-',label=method)
            for x,r in zip(xs,rows):axes[0,col].annotate(str(r['budget_percent'])+'%',(x,r['metrics']['mcc']),fontsize=7)
        for name,m in result['base'].items():
            axes[0,col].axhline(m['mcc'],linestyle=':',label=name)
            axes[1,col].axhline((m['fp']+w*m['fn'])/741,linestyle=':',label=name)
        axes[0,col].set(title=f'FN cost={w}, FP cost=1',ylabel='MCC')
        axes[1,col].set(ylabel='Weighted classification error per function')
        for row in range(2):
            axes[row,col].set_xlabel('Measured component cost / Qwen-only cost')
            axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=8)
    fig.suptitle('Development valid: fixed 25% / 50% call budgets, no policy selection')
    fig.tight_layout();fig.savefig(output/'learned_cost.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['fair','audit','predict','learn','plot']);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.action=='predict':predict(a.output,a.device)
    else:globals()[a.action](a.output)


if __name__=='__main__':main()
