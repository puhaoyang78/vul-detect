"""Bounded synthetic memory semantics and matched frozen CLM experiments."""
from __future__ import annotations
import copy
import itertools
import json
import random
import subprocess
from collections import Counter
from pathlib import Path
import torch
from .cfg_data import atomic_json
from .cgvl import local_rank_loss


def outcomes(spec):
    """Enumerate x=0..3 for allocated objects; malloc failure returns before the target.

    The target is an initialized int read/store, a null dereference, or a
    dereference of an owned allocation after free. Reassignment can repair the
    target pointer without resurrecting a freed allocation.
    """
    result=[]
    for x in range(4):
        s=(x&1)==spec['s']; r=((x>>1)&1)==spec['r']
        t=(x&1)==spec['t']
        if spec['composition']: t=t and ((x>>1)&1)==spec['u']
        if spec['predicate']=='xor':
            s=(((x&1)^((x>>1)&1))==spec['s'])
            t=(((x&1)^((x>>1)&1))==spec['t']) and ((not spec['composition']) or ((x>>1)&1)==spec['u'])
        # Independent object choices at initialization and both updates.
        # Choice 0 is a local object / in-range element; choice 1 is the
        # alternate object / out-of-range element. Labels are computed AFTER
        # executing the path, never used to select generated combinations.
        choice=spec['initial']
        steps=[(s,spec['first']),(r,spec['second'])]
        if spec['reverse']: steps.reverse()
        for enabled, value in steps:
            if enabled: choice=value
        if spec['family']=='Bounds':
            state=spec['capacity']+spec['offset'] if choice else spec['capacity']-1
            violation=t and not 0<=state<spec['capacity']
        else:
            state=('null' if spec['family']=='Pointer' else 'freed') if choice else 'live_local'
            violation=t and state in ('null','freed')
        result.append(dict(input=x,target_executed=t,state=state,violation=bool(violation)))
    return result


def render(spec, *, renamed=False, neutral=False):
    """Compose state/action, repair, target and control nodes into plain C."""
    bit='(x & 1)' if spec['predicate']=='bit' else '((x & 1) ^ ((x >> 1) & 1))'
    s=f'{bit} == {spec["s"]}';r=f'((x >> 1) & 1) == {spec["r"]}'
    t=f'{bit} == {spec["t"]}'
    if spec['composition']:t=f'({t}) && (((x >> 1) & 1) == {spec["u"]})'
    def branch(cond,body):
        return 'if ('+cond+') { '+body+' }'
    n=spec['capacity'];family=spec['family']
    init=['int value = 7;', 'int result = 0;']
    if family=='Bounds':
        values=[str(n-1),f'{n} + {spec["offset"]}']
        init += [f'int a[{n}] = {{0}};',f'int index = {values[spec["initial"]]};']
        updates=[f'index = {values[spec[k]]};' for k in ('first','second')]
        target='result = a[index];' if spec['access']=='read' else 'a[index] = value; result = value;'
        cleanup=''
    elif family=='Pointer':
        values=['&value','0']
        init += [f'int *p = {values[spec["initial"]]};']
        updates=[f'p = {values[spec[k]]};' for k in ('first','second')]
        target='result = *p;' if spec['access']=='read' else '*p = 8; result = value;'
        cleanup=''
    else:
        # Exactly one free, after alias assignment and before the target.
        # No pointer comparison or read of q after free; no double free.
        values=['&value','q']
        init += ['int *q = malloc(sizeof(int));','if (!q) return 0;', '*q = 7;',f'int *p = {values[spec["initial"]]};']
        updates=[f'p = {values[spec[k]]};' for k in ('first','second')]
        target='result = *p;' if spec['access']=='read' else '*p = 8; result = value;'
        cleanup=''
    steps=[branch(s,updates[0]),branch(r,updates[1])]
    if spec['reverse']:steps.reverse()
    body=init+steps
    if family=='Lifetime':body.append('free(q);')
    if spec['control']=='if':body += [branch(t,target),cleanup,'return result;']
    elif spec['control']=='early':body += ['if (!('+t+')) { '+cleanup+' return result; }',target,cleanup,'return result;']
    elif spec['control']=='switch':body += ['switch (!!('+t+')) { case 1: '+target+' break; default: break; }',cleanup,'return result;']
    else:raise ValueError(spec['control'])
    if neutral:body.insert(0,'int spare = x ^ 3; (void)spare;')
    source='int f(int x) {\n  '+'\n  '.join(x for x in body if x)+'\n}\n'
    if renamed:
        import re
        for a,b in [('x','input_value'),('value','cell'),('result','answer'),('index','position'),('p','cursor'),('q','allocation'),('a','buffer'),('dead','released')]:
            source=re.sub(r'\b'+a+r'\b',b,source)
    return source


def generate():
    rows=[];pairs=[];controls=[];seen={}
    def add(spec,split,scenario,variant='original'):
        source=render(spec,renamed=variant=='rename',neutral=variant=='neutral')
        if source in seen:
            i=seen[source]
            if rows[i]['split']!=split:raise ValueError('source leakage across structure split')
            return i
        state=outcomes(spec);w=[o['input'] for o in state if o['violation']]
        if not any(o['target_executed'] for o in state):raise ValueError('vacuous target')
        row=dict(id=len(rows),split=split,scenario=scenario,family=spec['family'],
            structure_family=[spec['control'],spec['predicate'],spec['composition']],spec=spec,
            source=source,label=int(bool(w)),witness=w,outcomes=state,variant=variant,
            scope='x in 0..3; standard int and malloc/free; target property only; allocation failure returns before target')
        seen[source]=row['id'];rows.append(row);return row['id']
    # Fixed grammar split, then COMPLETE Cartesian products, before labels.
    # All t-flips survive: SS, SU, US and UU. Cosmetic descendants stay grouped.
    scenarios=[('train','train_if','if','bit',False,[4,6]),
               ('valid','unseen_control_combination','switch','xor',True,[4,6]),
               ('valid','unseen_constants_combination','switch','xor',True,[9])]
    for split,scenario,control,predicate,composition,capacities in scenarios:
        for family in ('Bounds','Pointer','Lifetime'):
            if scenario=='unseen_constants_combination' and family!='Bounds':continue
            for n,offset,access,s,r,u,initial,first,second,reverse in itertools.product(
                    capacities if family=='Bounds' else (4,),
                    (0,1) if family=='Bounds' else (0,),('read','write'),(0,1),(0,1),
                    (0,1) if composition else (0,), (0,1),(0,1),(0,1),(False,True)):
                spec=dict(family=family,capacity=n,offset=offset,access=access,s=s,r=r,u=u,
                          t=0,control=control,predicate=predicate,composition=composition,
                          initial=initial,first=first,second=second,reverse=reverse)
                other=dict(spec,t=1)
                for variant in ('original','rename','neutral'):
                    ids=[add(z,split,scenario,variant) for z in (spec,other)]
                    states=[rows[i]['label'] for i in ids]
                    info=dict(split=split,scenario=scenario,family=family)
                    if states[0]!=states[1]:
                        pairs.append(dict(safe=ids[states.index(0)],unsafe=ids[states.index(1)],**info))
                    else:
                        controls.append(dict(original=ids[0],changed=ids[1],
                            kind='safe_safe' if states[0]==0 else 'unsafe_unsafe',**info))
                    if variant=='original': originals=ids
                    else:
                        for i,j in zip(originals,ids):
                            controls.append(dict(original=i,changed=j,kind=variant,**info))
    pairs=[dict(x) for x in {tuple(sorted(p.items())) for p in pairs}]
    pairs.sort(key=lambda p:(p['safe'],p['unsafe']))
    # Dedup specifications that have no effect for a particular memory family.
    controls=[dict(x) for x in {tuple(sorted(c.items())) for c in controls}]
    controls.sort(key=lambda x:(x['original'],x['kind'],x['changed']))
    return rows,pairs,controls


def verify_execution(root, rows):
    """Every admitted source, every domain input; Safe is established by semantics."""
    import os
    root=Path(root);source='#include <stdlib.h>\n#include <stdio.h>\n'
    for row in rows:source+=row['source'].replace('int f(',f'int case_{row["id"]}(',1)
    source+='int main(int argc,char **argv) { if(argc!=3)return 3; int i=atoi(argv[1]), x=atoi(argv[2]); switch(i) {\n'
    source+=''.join(f'case {r["id"]}: (void)case_{r["id"]}(x); break;\n' for r in rows)
    source+='default:return 4;}return 0;}\n'
    path=root/'execution.c';path.write_text(source);exe=root/'execution'
    cp=subprocess.run(['cc','-std=c11','-O0','-g','-no-pie','-fsanitize=address,undefined','-fno-sanitize-recover=all',str(path),'-o',str(exe)],capture_output=True,text=True)
    if cp.returncode:raise RuntimeError(cp.stderr)
    counts=Counter();failures=[]
    env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0:halt_on_error=1')
    with (root/'execution.jsonl').open('x') as out:
        for row in rows:
            for o in row['outcomes']:
                p=subprocess.run([str(exe.resolve()),str(row['id']),str(o['input'])],capture_output=True,text=True,env=env,timeout=10)
                diagnostic=('out of bounds' if row['family']=='Bounds' else 'null pointer' if row['family']=='Pointer' else 'heap-use-after-free')
                agrees=(p.returncode!=0 and diagnostic in p.stderr) if o['violation'] else p.returncode==0
                counts['checked']+=1;counts['expected_violations']+=int(o['violation']);counts['matched']+=int(agrees)
                record=dict(id=row['id'],input=o['input'],expected_violation=o['violation'],exit=p.returncode,agrees=agrees,diagnostic=p.stderr[:1200] if p.returncode else '')
                out.write(json.dumps(record)+'\n')
                if not agrees:failures.append(record)
    atomic_json(root/'execution_summary.json',dict(counts=counts,failures=failures,compiler=subprocess.check_output(['cc','--version'],text=True).splitlines()[0]))
    if failures:raise ValueError('sanitizer disagreed; no training admission')


def prepare(output_dir):
    root=Path(output_dir)
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    rows,pairs,controls=generate()
    training_groups(rows,pairs,controls)
    for name,data in [('samples',rows),('pairs',pairs),('controls',controls)]:
        (root/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in data))
    atomic_json(root/'coverage.json',dict(samples=Counter(r['split']+':'+r['family']+':'+str(r['label']) for r in rows),
        pairs=Counter(p['split']+':'+p['family'] for p in pairs),scenarios=Counter(r['scenario'] for r in rows),
        structure_families={s:sorted({tuple(r['structure_family']) for r in rows if r['split']==s}) for s in ('train','valid')},
        controls=Counter(c['split']+':'+c['kind'] for c in controls),unknown=0,
        scope='Finite semantics; not full C coverage; not independent real functions',seed=42))
    atomic_json(root/'structural_shortcut.json',structural_shortcut_audit(rows))
    verify_execution(root,rows)
    return dict(json.loads((root/'coverage.json').read_text()),
                admission=json.loads((root/'structural_shortcut.json').read_text())['admission'])


def training_groups(rows,pairs,controls):
    groups={(p['safe'],p['unsafe']) for p in pairs if p['split']=='train'}
    groups.update((c['original'],c['changed']) for c in controls
                  if c['split']=='train' and c['kind'] in ('safe_safe','unsafe_unsafe'))
    members=Counter(i for group in groups for i in group)
    expected={r['id'] for r in rows if r['split']=='train'}
    if set(members)!=expected or any(n!=1 for n in members.values()):
        raise ValueError('A/B must visit every train member exactly once per epoch')
    return sorted(groups)


def fit_heads(features, rows, pairs, controls, initial, output_dir):
    """Identical frozen source linear heads; identical paired batches and budgets."""
    from .cfg_metrics import metrics, select_threshold
    root=Path(output_dir)
    x=features.float();labels=torch.tensor([r['label'] for r in rows],dtype=torch.float32)
    train=torch.tensor([r['id'] for r in rows if r['split']=='train'])
    valid=torch.tensor([r['id'] for r in rows if r['split']=='valid'])
    # Reuse the train-only normalization of existing frozen_origin_probes.
    mean=x[train].mean(0);std=x[train].std(0,unbiased=False).clamp_min(1e-4);x=(x-mean)/std
    matched=training_groups(rows,pairs,controls)
    pair_tensor=torch.tensor(matched)
    reports={}
    for mode in ('bce','bce_rank'):
        torch.manual_seed(42);head=torch.nn.Linear(x.shape[1],1);head.load_state_dict(initial)
        optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
        history=[];best=None;best_key=None
        for epoch in range(20):
            order=list(range(len(matched)));random.Random(42+epoch).shuffle(order)
            losses=[]
            for start in range(0,len(order),16):
                ids=pair_tensor[order[start:start+16]];logits=head(x[ids]).squeeze(-1)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels[ids])
                different=labels[ids[:,0]]!=labels[ids[:,1]]
                if mode=='bce_rank' and different.any():
                    z=logits[different]; y=labels[ids[different]]
                    loss=loss+local_rank_loss(z[y==0],z[y==1])
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step();losses.append(float(loss))
            with torch.no_grad():logits=head(x).squeeze(-1);scores=logits.sigmoid().tolist()
            threshold,m=select_threshold(labels[valid].int().tolist(),[scores[i] for i in valid.tolist()])
            row=dict(epoch=epoch+1,train_loss=sum(losses)/len(losses),metrics=m,
                bce=float(torch.nn.functional.binary_cross_entropy_with_logits(logits[valid],labels[valid])))
            history.append(row);key=(m['mcc'],m['f1'],m['accuracy'],m['auc'])
            if best_key is None or key>best_key:
                best_key=key;best=dict(**row,threshold=threshold,scores=scores,logits=logits.tolist(),head_state=copy.deepcopy(head.state_dict()))
        scores=best['scores'];threshold=best['threshold']
        def evaluate(ids):
            y=[rows[i]['label'] for i in ids];p=[scores[i] for i in ids]
            logits=torch.tensor([best['logits'][i] for i in ids])
            members=set(ids)
            ps=[q for q in pairs if q['safe'] in members and q['unsafe'] in members]
            return dict(metrics=metrics(y,p,threshold),bce=float(torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor(y).float())),
                pair_accuracy=sum(scores[q['unsafe']]>scores[q['safe']] for q in ps)/len(ps) if ps else None,pairs=len(ps),samples=len(ids))
        report=dict(selected_epoch=best['epoch'],threshold=threshold,history=history,
            valid=evaluate(valid.tolist()),train=evaluate(train.tolist()),
            scenarios={s:evaluate([r['id'] for r in rows if r['scenario']==s]) for s in sorted({r['scenario'] for r in rows if r['split']=='valid'})},
            families={f:evaluate([r['id'] for r in rows if r['split']=='valid' and r['family']==f]) for f in ('Bounds','Pointer','Lifetime')})
        report['stability']={}
        for kind in ('rename','neutral','safe_safe','unsafe_unsafe'):
            cs=[c for c in controls if c['split']=='valid' and c['kind']==kind]
            report['stability'][kind]=dict(pairs=len(cs),mean_absolute_score_change=sum(abs(scores[c['original']]-scores[c['changed']]) for c in cs)/len(cs),
                prediction_flip_rate=sum((scores[c['original']]>=threshold)!=(scores[c['changed']]>=threshold) for c in cs)/len(cs))
        atomic_json(root/(mode+'.json'),report)
        torch.save(dict(head_state=best['head_state'],mean=mean,std=std,threshold=threshold),root/(mode+'.pt'))
        (root/(mode+'.predictions.jsonl')).write_text(''.join(json.dumps(dict(id=r['id'],split=r['split'],label=r['label'],score=scores[r['id']],logit=best['logits'][r['id']]))+'\n' for r in rows))
        reports[mode]=report
    return reports


def frozen_experiment(data_dir, output_dir, device='cuda:0'):
    from .cfg_experiment import _base_module, _tokenizer
    from .cfg_network import build_model
    root=Path(output_dir)
    if root.exists():raise FileExistsError(root)
    data=Path(data_dir)
    require_admission(data)
    audit=json.loads((data/'execution_summary.json').read_text())
    if audit['failures'] or audit['counts']['matched']!=audit['counts']['checked']:raise ValueError('unverified data')
    rows=[json.loads(l) for l in (data/'samples.jsonl').read_text().splitlines()]
    pairs=[json.loads(l) for l in (data/'pairs.jsonl').read_text().splitlines()]
    controls=[json.loads(l) for l in (data/'controls.jsonl').read_text().splitlines()]
    config=json.loads(Path('results/cfg_clm_source_seed42/config.json').read_text())
    cp_path=Path(config['pretrain_run_dir'])/'lm_pretrain'/'last.pt'
    base=_base_module();base._seed_everything(42);resolved=base._resolve_device(device)
    builder=_tokenizer(base,config);model=build_model(base,dict(config,variant='baseline'),None,resolved,training=False)
    cp=torch.load(cp_path,map_location='cpu',weights_only=False)
    if cp['mode']!='lm_pretrain':raise ValueError('CLM-only required')
    base.set_peft_model_state_dict(model.encoder,cp['adapter_state']);del cp
    initial={k:v.detach().cpu().clone() for k,v in model.task_modules['classifier'].state_dict().items()}
    model.requires_grad_(False);model.eval();pools=[];lengths=[]
    root.mkdir(parents=True)
    atomic_json(root/'config.json',dict(synthetic_data=str(data.resolve()),clm_checkpoint=str(cp_path),seed=42,epochs=20,batch_pairs=16,learning_rate=.001,
        weight_decay=.01,encoder_frozen=True,source_max_length=2048,pooling='original masked mean including prefix/EOS',
        normalization='existing train-only frozen probe mean/std',selection='valid MCC/F1/accuracy/AUC',
        transfer_gate='BCE+Ranking must improve valid AUC by >=.02 and MCC by >=.02, not worsen BCE/F1 or rename/neutral stability; positive pair accuracy gain; all family AUC >=.60. No test.'))
    with torch.no_grad():
        for i,row in enumerate(rows):
            record=dict(raw_source=row['source'])
            length=len(builder.tokenizer(row['source'],add_special_tokens=False)['input_ids'])
            if length>2048:raise ValueError('synthetic evidence truncated')
            lengths.append(length)
            ids,mask=builder.sequence_batch([record],variant='baseline',excluded_groups=(),device=resolved)
            h=model.encoder(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
            w=mask.unsqueeze(-1).to(h.dtype);pools.append(((h*w).sum(1)/w.sum(1)).cpu().float())
            if (i+1)%200==0:print(f'frozen sources {i+1}/{len(rows)}',flush=True)
    features=torch.cat(pools);torch.save(dict(pool=features,initial_head=initial,lengths=lengths),root/'features.pt')
    del model
    if resolved.type=='cuda':torch.cuda.empty_cache()
    reports=fit_heads(features,rows,pairs,controls,initial,root)
    atomic_json(root/'complete.json',dict(samples=len(rows),max_source_tokens=max(lengths),encoder_updates=0,
        bce=reports['bce']['valid'],bce_rank=reports['bce_rank']['valid'],test_used=False))
    return json.loads((root/'complete.json').read_text())


def transfer_experiment(data_dir, output_dir, objective, device='cuda:0', frozen_run_dir=None):
    """Original PrimeVul trainer + one matched synthetic pair per real update."""
    import gc
    from .cfg_experiment import _base_module, _tokenizer, _prediction_outputs
    from .cfg_data import read_records, cohort_hash, file_sha256
    root=Path(output_dir)
    if root.exists():raise FileExistsError(root)
    if objective not in ('bce','bce_rank'):raise ValueError(objective)
    data=Path(data_dir);require_admission(data)
    if frozen_run_dir is None:raise ValueError('a completed matching frozen run is required')
    frozen=Path(frozen_run_dir)
    config=json.loads((frozen/'config.json').read_text())
    if Path(config['synthetic_data']).resolve()!=data.resolve():raise ValueError('frozen data mismatch')
    a=json.loads((frozen/'bce.json').read_text());b=json.loads((frozen/'bce_rank.json').read_text())
    av=a['valid'];bv=b['valid']
    passed=(bv['metrics']['auc']>=av['metrics']['auc']+.02 and
        bv['metrics']['mcc']>=av['metrics']['mcc']+.02 and
        bv['metrics']['f1']>=av['metrics']['f1'] and bv['bce']<=av['bce'] and
        bv['pair_accuracy']>av['pair_accuracy'] and
        all(v['metrics']['auc']>=.6 for v in b['families'].values()) and
        all(b['stability'][k]['prediction_flip_rate']<=a['stability'][k]['prediction_flip_rate'] and
            b['stability'][k]['mean_absolute_score_change']<=a['stability'][k]['mean_absolute_score_change']
            for k in ('rename','neutral')))
    if not passed:raise ValueError('absolute classification and stability transfer gate not met')
    sources=[json.loads(l) for l in (data/'samples.jsonl').read_text().splitlines()]
    pairs=[json.loads(l) for l in (data/'pairs.jsonl').read_text().splitlines()]
    controls=[json.loads(l) for l in (data/'controls.jsonl').read_text().splitlines()]
    synthetic=[[dict(raw_source=sources[i]['source'],label=sources[i]['label'],split='train') for i in group]
               for group in training_groups(sources,pairs,controls)]
    reference=Path('results/cfg_clm_source_seed42');config=json.loads((reference/'config.json').read_text())
    rows=read_records(config['dataset'],config['source_dataset'])
    if cohort_hash(rows)!=config['cohort_sha256']:raise ValueError('PrimeVul cohort changed')
    valid=[r for r in rows if r['split']=='valid']
    cp_path=Path(config['pretrain_run_dir'])/'lm_pretrain'/'last.pt'
    cp=torch.load(cp_path,map_location='cpu',weights_only=False)
    if cp['mode']!='lm_pretrain':raise ValueError('CLM-only initialization required')
    root.mkdir(parents=True)
    atomic_json(root/'config.json',dict(config,synthetic_objective=objective,synthetic_weight=1.,
        synthetic_data=str(data),synthetic_pairs_per_update=1,unique_synthetic_pairs=len(synthetic),
        reference_baseline=str(reference/'lm_pretrain_source'),
        selection='unchanged PrimeVul valid MCC/F1/accuracy/AUC; no test',
        gate_note='Data audit and absolute classification/stability gates passed; transfer remains an independent hypothesis.'))
    base=_base_module()
    checkpoint=base.train_model(None,root/'best.pt',variant='baseline',model_path=config['model_path'],records=rows,
        source_max_length=2048,context_max_length=384,device=device,initial_adapter_state=cp['adapter_state'],
        synthetic_pairs=synthetic,synthetic_objective=objective,
        **{k:config[k] for k in ('batch_size','gradient_accumulation','epochs','learning_rate','weight_decay',
                              'lora_r','lora_alpha','lora_dropout','seed','log_every')})
    threshold=float(checkpoint['decision_threshold']);del checkpoint,cp;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    logits=base.predict_checkpoint(root/'best.pt',valid,batch_size=config['batch_size'],device=device)
    scores=logits.sigmoid().tolist()
    report=_prediction_outputs(root,'valid',valid,scores,threshold,_tokenizer(base,config),checkpoint_hash=file_sha256(root/'best.pt'))
    bce=float(torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),torch.tensor([r['label'] for r in valid]).float()))
    (root/'valid.logits.jsonl').write_text(''.join(json.dumps(dict(sample_key=r['sample_key'],label=r['label'],logit=float(z)))+'\n' for r,z in zip(valid,logits)))
    atomic_json(root/'complete.json',dict(validation=report,unscaled_bce=bce,test_used=False))
    return report


def structural_shortcut_audit(rows):
    """CPU-only, fail-closed screen. Passing is necessary, never proof of semantics.

    Fixed screening limits: nuisance/lexical valid AUC >= .80 or cross-split
    normalized token similarity >= .95 requires review; never filter members.
    A semantic state interpreter is deliberately NOT one of these proxies.
    """
    import re
    from difflib import SequenceMatcher
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    if {r['split'] for r in rows}!={'train','valid'}:raise ValueError('train/valid required')
    y=np.array([r['label'] for r in rows]);tr=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    va=np.array([i for i,r in enumerate(rows) if r['split']=='valid'])
    features=[];skeletons={'train':{},'valid':{}};counts={}
    keywords={'int','if','else','return','switch','case','default','break','void','sizeof','malloc','free'}
    tokens_pattern=r'[A-Za-z_]\w*|\d+|==|!=|&&|\|\||>>|<<|[^\s]'
    for row in rows:
        src=row['source'];tokens=re.findall(tokens_pattern,src)
        f=dict(characters=len(src),lines=src.count('\n'),tokens=len(tokens))
        for token in tokens:
            if re.fullmatch(r'\d+|[A-Za-z_]\w*|==|!=|&&|\|\||[<>&*+^]',token):
                f['count:'+token]=f.get('count:'+token,0)+1
        literals=[];condition_offsets=[];offset=0
        for line in src.splitlines(keepends=True):
            if re.search(r'\b(?:x|input_value) & 1',line):
                m=re.search(r'== ([01])',line)
                if m:
                    literals.append(int(m.group(1)))
                    condition_offsets.append(offset+m.start(1))
            offset+=len(line)
        f['equal_conditions']=int(len(literals)==2 and literals[0]==literals[1])
        f['target_condition_edit_offset']=condition_offsets[-1] if len(condition_offsets)==2 else -1
        f['target_line']=next((i for i,l in enumerate(src.splitlines()) if any(t in l for t in
            ('result = *p','a[index]','answer = *cursor','buffer[position]','*p = 8','*cursor = 8'))),-1)
        features.append(f)
        # Consistent alpha normalization preserves binding, unlike replacing all
        # identifiers with one symbol. Numeric abstraction checks near clones.
        names={};normalized=[]
        for tok in tokens:
            if tok.isdigit():normalized.append('NUM')
            elif re.fullmatch(r'[A-Za-z_]\w*',tok) and tok not in keywords:
                if tok not in names:names[tok]='ID'+str(len(names))
                normalized.append(names[tok])
            else:normalized.append(tok)
        key=tuple(normalized)
        skeletons[row['split']].setdefault(key,[]).append(row['id'])
    for split,idx in [('train',tr),('valid',va)]:
        pred=np.array([features[i]['equal_conditions'] for i in idx])
        counts[split]=dict(samples=len(idx),correct=int((pred==y[idx]).sum()),
            accuracy=float((pred==y[idx]).mean()),balanced_accuracy=float(balanced_accuracy_score(y[idx],pred)),
            unsafe=int(y[idx].sum()))
    vectorizer=DictVectorizer(sparse=False);x=vectorizer.fit_transform(features)
    family_masks={f:np.array([j for j,i in enumerate(va) if rows[i]['family']==f])
                  for f in ('Bounds','Pointer','Lifetime')}
    def score_report(scores):
        report=dict(auc=float(roc_auc_score(y[va],scores)),
                    balanced_accuracy=float(balanced_accuracy_score(y[va],scores>=.5)))
        report['families']={f:dict(auc=float(roc_auc_score(y[va][ids],scores[ids])),
            balanced_accuracy=float(balanced_accuracy_score(y[va][ids],scores[ids]>=.5)))
            for f,ids in family_masks.items() if len(ids) and len(set(y[va][ids]))==2}
        return report
    def evaluate(model, xx):
        model.fit(xx[tr],y[tr])
        return score_report(model.predict_proba(xx[va])[:,1])
    single={}
    for j,name in enumerate(vectorizer.get_feature_names_out()):
        single[name]=evaluate(DecisionTreeClassifier(max_depth=1,random_state=42),x[:,j:j+1])
    statistics=evaluate(DecisionTreeClassifier(max_depth=3,min_samples_leaf=16,random_state=42),x)
    lexical=CountVectorizer(token_pattern=tokens_pattern,lowercase=False,ngram_range=(1,2))
    xx=lexical.fit_transform([rows[i]['source'] for i in tr])
    model=LogisticRegression(C=1.,max_iter=1000,solver='liblinear',random_state=42)
    model.fit(xx,y[tr]);scores=model.predict_proba(lexical.transform([rows[i]['source'] for i in va]))[:,1]
    lexical_report=score_report(scores)
    lexical_report['top_weights']=sorted(zip(lexical.get_feature_names_out().tolist(),model.coef_[0].tolist()),key=lambda z:-abs(z[1]))[:20]
    forest=RandomForestClassifier(n_estimators=100,max_depth=6,min_samples_leaf=16,n_jobs=2,random_state=42)
    forest.fit(xx,y[tr]);forest_report=score_report(forest.predict_proba(lexical.transform([rows[i]['source'] for i in va]))[:,1])
    # This is a conservative provenance family, not an assertion that distinct
    # Boolean predicates compute identical functions. Their shared core must
    # not be mistaken for independent program instances.
    def core_instance(row):
        spec=row['spec']
        return tuple(spec[k] for k in ('family','capacity','offset','access','initial','first','second','reverse','s','r'))
    train_instances={core_instance(rows[i]) for i in tr}
    overlap=sum(core_instance(rows[i]) in train_instances for i in va)
    state_combinations={split:sorted({tuple(r['spec'][k] for k in ('initial','first','second'))
        for r in rows if r['split']==split}) for split in ('train','valid')}
    unseen_states=set(state_combinations['valid'])-set(state_combinations['train'])
    collisions=set(skeletons['train'])&set(skeletons['valid'])
    near=[];maximum=0.
    for vk,ids in skeletons['valid'].items():
        best=0.;best_ids=[]
        for tk,tids in skeletons['train'].items():
            # Cheap length bound; exact token sequence similarity after that.
            if 2*min(len(tk),len(vk))/(len(tk)+len(vk))<.95:continue
            ratio=SequenceMatcher(None,tk,vk,autojunk=False).ratio()
            if ratio>best:best=ratio;best_ids=tids
        maximum=max(maximum,best)
        if best>=.95:near.append(dict(valid=ids[0],train=best_ids[0],similarity=best,valid_members=len(ids)))
    groups={s:{tuple(r['structure_family']) for r in rows if r['split']==s} for s in ('train','valid')}
    failures=[]
    if groups['train']&groups['valid']:failures.append('structure_family_overlap')
    if collisions:failures.append('normalized_source_overlap')
    if near:failures.append('normalized_near_duplicates')
    def exceeds(report):
        return report['auc']>=.8 or any(v['auc']>=.8 for v in report['families'].values())
    aggregate_passed=(not failures and not any(v['auc']>=.8 for v in single.values())
                      and statistics['auc']<.8 and lexical_report['auc']<.8)
    if any(exceeds(v) for v in single.values()):failures.append('single_feature_auc_at_least_0.80')
    if exceeds(statistics):failures.append('shallow_statistics_auc_at_least_0.80')
    if exceeds(lexical_report):failures.append('lexical_auc_at_least_0.80_requires_semantic_review')
    if exceeds(forest_report):failures.append('lexical_forest_auc_at_least_0.80_requires_semantic_review')
    if any(v['correct']==v['samples'] for v in counts.values()):failures.append('old_equality_rule_perfect')
    if overlap:failures.append('shared_core_program_instances')
    if not unseen_states:failures.append('no_heldout_object_state_combinations')
    return dict(admission=not failures,failures=failures,counts=counts,single_features=single,
        shallow_statistics=statistics,lexical=lexical_report,lexical_forest=forest_report,
        aggregate_only_screen_passed=aggregate_passed,
        core_instance_overlap=dict(valid_rows=overlap,valid_total=len(va),
            definition='Same operation/object choices/order/condition literals; ignores control spelling, predicate and added conjunction.'),
        object_state_combinations=state_combinations,unseen_object_state_combinations=sorted(unseen_states),
        limits=dict(proxy_auc=.8,near_duplicate_similarity=.95),
        near_duplicates=near,maximum_screened_similarity=maximum,
        normalized_source_overlap=len(collisions),
        unique_normalized_structures={s:len(v) for s,v in skeletons.items()},
        note='A high lexical score is a gate for review, not proof of a nonsemantic shortcut. No label-dependent filtering is permitted.')


def require_admission(data):
    """Must run before device resolution or encoder construction."""
    data=Path(data)
    admission=json.loads((data/'structural_shortcut.json').read_text())
    if admission.get('admission') is not True or 'lexical_forest' not in admission or 'unseen_object_state_combinations' not in admission:
        raise ValueError('synthetic data admission failed: '+str(admission.get('failures',admission.get('conclusion'))))
    execution=json.loads((data/'execution_summary.json').read_text())
    if execution['failures'] or execution['counts']['matched']!=execution['counts']['checked']:
        raise ValueError('unverified execution labels')
