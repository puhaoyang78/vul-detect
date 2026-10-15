from __future__ import annotations
import argparse
import json
from pathlib import Path
from .model import ROBUSTNESS_MODES, ROBUSTNESS_HISTORY_MODES

def _finetuning_evaluate(rows, run, split, checkpoint, args, *, output_name=None, checkpoint_path=None):
    from . import model as core
    from .artifacts import identity
    from .artifacts import write_json
    name = output_name or split
    output, predictions = run/f'{name}.metrics.json', run/f'{name}.predictions.jsonl'
    if output.exists() != predictions.exists():
        raise FileExistsError(f'incomplete evaluation artifacts: {run}/{split}; inspect before rerunning')
    if output.exists():
        with predictions.open() as handle:cached = [json.loads(l) for l in handle]
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
            or any(k in base for k in ('change_supervision', 'local_supervision', 'synthetic_supervision',
                                       'augmentation_config', 'conditional_length_config','surface_vat_config','robustness_options',
                                       'repair_supervision','repair_training_members'))):
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


def _check_robustness_checkpoint(cp, rows, opts, materials, training, args, source_budget):
    from . import model as core
    expected_members = {split: [(r['sample_key'], r['label']) for r in rows if r['split'] == split]
                        for split in ('train', 'valid')}
    if (cp.get('robustness_options') != opts or cp.get('training_config') != training
            or cp.get('cohort_members') != expected_members
            or cp.get('classifier_initialization') != core._CLASSIFIER_INITIALIZATION or cp.get('model_path') != args.model
            or cp.get('source_max_length') != 2048 or cp.get('classifier_head') != 'linear' or cp.get('finetuning_mode') != 'full'
            or cp.get('initial_checkpoint') is not None or cp.get('sample_weights') is not None
            or cp.get('seed') != 42 or cp.get('selection') != 'validation_mcc' or cp.get('variant') != 'codebert_source'
            or cp.get('context_max_length') != 384 or cp.get('input_protocol', {}).get('source_budget') != source_budget
            or cp.get('encoder_precision') != 'FP32 weights/optimizer; BF16 CUDA autocast; FP32 head/pooling/loss'
            or any(k in cp for k in ('change_supervision', 'local_supervision', 'synthetic_supervision',
                                     'augmentation_config', 'conditional_length_config'))
            or cp.get('surface_materials') != (materials if opts is not None and opts['source_views'] else None)):
        raise ValueError('completed robustness checkpoint differs from this experiment')


def _robustness_fixed_candidates(rows,materials,root,tokenizer,source_budget,*,split='valid'):
    from . import augmentation as data
    from .syntax import SURFACE_OPERATIONS
    from .artifacts import write_json
    import random
    if split not in ('valid','test'):raise ValueError('robustness evaluation requires Valid or Test')
    evaluation_rows=[r for r in rows if r['split']==split]
    if not evaluation_rows:raise ValueError(f'{split} split is empty')
    prefix='' if split=='valid' else 'test.'
    paths={}
    for bank in ('train','heldout'):
        name='fixed' if bank=='train' else 'heldout'
        path=root/f'{prefix}{name}.candidates.jsonl';paths[name]=path
        if path.exists():
            with path.open() as handle:cached=[json.loads(l) for l in handle]
            if [r['sample_key'] for r in cached]!=[r['sample_key'] for r in evaluation_rows]:
                raise ValueError('robustness candidate cache members/order differ')
            continue
        entries=[]
        for index,row in enumerate(evaluation_rows):
            groups={};reasons={}
            for kind in (*SURFACE_OPERATIONS,'combined'):
                combos=([kind],) if kind!='combined' else (
                    materials['rules']['heldout_combinations'] if bank=='heldout' else
                    (['local_name','comment_insert'],['local_name','local_name','comment_replace']))
                candidates=[]
                for combo in combos:
                    result=data.search_surface_views([row],None,tokenizer,source_budget,materials,
                        seed=42,bank=bank,purpose=name+'/'+kind+'/'+','.join(combo),required_combination=combo)[0]
                    for key,value in result['reasons'].items():reasons[key]=reasons.get(key,0)+value
                    candidates.extend(c for c in result['visited'] if len(c['units'])==len(combo))
                rng=random.Random(data.surface_seed(42,0,row['sample_key'],name+'/'+kind))
                rng.shuffle(candidates)
                groups[kind]=[{k:v for k,v in c.items() if k!='raw_source'} for c in candidates[:2]]
            entries.append(dict(sample_key=row['sample_key'],groups=groups,reasons=reasons))
            if (index+1)%100==0:print(f'{split} {name} candidates {index+1}/{len(evaluation_rows)}',flush=True)
        with path.open('x') as handle:
            for entry in entries:handle.write(json.dumps(entry)+'\n')
        write_json(root/f'{prefix}{name}.coverage.metrics.json',{
            kind:dict(eligible=sum(bool(e['groups'][kind]) for e in entries),total=len(entries),
                candidates=sum(len(e['groups'][kind]) for e in entries)) for kind in (*SURFACE_OPERATIONS,'combined')})
    return paths


def _robustness_smoke(rows,root,config,materials,args,baseline,*,modes=('bce','vat','ours'),output_name='smoke.metrics.json'):
    import tempfile
    import torch
    from . import model as core,augmentation as data
    from .artifacts import write_json
    from .syntax import surface_binding_layout,resolve_language
    train=[r for r in rows if r['split']=='train']
    unsupported=[r for r in train if any(k in surface_binding_layout(r['raw_source'],resolve_language(r['raw_source'],r.get('language','c'),r.get('file_name','')))['reasons']
                                     for k in ('parse_error','ambiguous_function','missing_body'))]
    if not unsupported:raise ValueError('real smoke requires a Train function without legal syntax transformations')
    chosen=[r for y in (0,1) for r in [r for r in train if r['label']==y and r['sample_key']!=unsupported[0]['sample_key']][:4]]
    chosen[0]=unsupported[0]
    if {r['label'] for r in chosen}!={0,1}:raise ValueError('smoke needs both labels')
    chosen+=[next(r for r in rows if r['split']=='valid' and r['label']==y) for y in (0,1)]
    device=core._resolve_device(args.device);tokenizer=core.AutoTokenizer.from_pretrained(args.model)
    builder=core.InputBuilder(tokenizer,source_max_length=2048,context_max_length=384)
    core._seed_everything(42)
    fresh=core._build_model('codebert_source',args.model,device=device,lora_r=16,lora_alpha=32,lora_dropout=.05,
        target_modules=tuple(baseline['target_modules']),gradient_checkpointing=False,fusion_dim=256,fusion_heads=8,finetuning_mode='full')
    initial_embedding=fresh.encoder.state_dict()['embeddings.word_embeddings.weight'].detach().cpu().clone()
    initial_head=core._cpu_state(fresh.task_modules.state_dict())
    # Train-only numerical diagnosis at both fresh and already fitted parameters.
    lengths=[len(builder._encode(r['raw_source'])) for r in train]
    probe=[]
    for low,high in ((0,128),(129,512),(513,2025),(2026,10**9)):
        for y in (0,1):probe.extend([r for r,n in zip(train,lengths) if r['label']==y and low<=n<=high][:2])
    numeric={};xi=.001
    fitted=core._load_model(baseline,device=device)
    for scale in (.001,.01):
        checks={};lost=0
        for label,model in (('fresh',fresh),('fitted',fitted)):
            for alpha in (0.,):
                infos=[]
                for i in range(0,len(probe),2):
                    values,info=core.vat_loss(model,probe[i:i+2],builder,dict(core.VAT_CONFIG,xi=scale),
                        alpha=alpha,device=device,noise_seed=42+i,diagnose=True)
                    infos.append(info);lost+=info['signal_loss'];del values
                adversarial=[v for info in infos for v in info['adversarial_kl']]
                noise=[v for info in infos for v in info['random_kl']]
                if sum(adversarial)<=sum(noise):
                    raise ValueError(f'{label} alpha={alpha}: VAT direction does not beat same-radius random noise on Train probes')
                checks[label+'/'+str(alpha)]=infos
        numeric[str(scale)]=checks
        if lost==0:xi=scale;break
        if scale==.01:raise ValueError('VAT probe still loses nonzero analytical direction signal')
    del fitted
    cfg=dict(core.VAT_CONFIG,xi=xi)
    signals={}
    for alpha in (0.,):
        values,info,delta=core.vat_loss(fresh,[chosen[1]],builder,cfg,alpha=alpha,device=device,noise_seed=42,
                                      return_delta=True,diagnose=True)
        gradient=torch.autograd.grad(values.sum(),fresh.task_modules['classifier'].weight)[0]
        if float(values.detach().sum())<=0 or float(gradient.norm())<=0:raise ValueError('real VAT has no model gradient')
        signals[str(alpha)]=dict(info,classifier_gradient_norm=float(gradient.norm()))
        del values,gradient,delta
    report=dict(xi=xi,numerical=numeric,signals=signals,untransformable=unsupported[0]['sample_key'],checks={})
    del fresh
    with tempfile.TemporaryDirectory(prefix='vul_detect_robustness_smoke_') as tmp:
        for mode in modes:
            target=Path(tmp)/mode/'best.pt'
            opts=None if mode=='bce' else core.robustness_config(mode,xi=xi)
            cp=core.train_model(None,target,records=chosen,variant='codebert_source',model_path=args.model,
                source_max_length=2048,context_max_length=384,seed=42,device=args.device,finetuning_mode='full',
                robustness_options=opts,surface_materials=materials if opts is not None and opts['source_views'] else None,log_every=1,
                **dict(config['training'],epochs=1))
            checks=dict(encoder_updated=not torch.equal(cp['encoder_state']['embeddings.word_embeddings.weight'],initial_embedding),
                classifier_updated=any(not torch.equal(cp['task_state'][k],v) for k,v in initial_head.items()))
            restored=core._load_model(cp,device=device)
            expected=core._predict_logits(restored,chosen[8:],builder,variant='codebert_source',excluded_groups=(),batch_size=4,device=device)
            del restored
            torch.testing.assert_close(expected,core.predict_checkpoint(target,chosen[8:],batch_size=4,device=args.device),rtol=0,atol=0)
            checks['save_reload_exact']=True
            if not all(checks.values()):raise ValueError(f'{mode} real smoke failed')
            with target.with_suffix('.training.jsonl').open() as handle:epochs=[json.loads(l) for l in handle]
            report['checks'][mode]=dict(checks,epoch=next(r for r in epochs if r['event']=='epoch'))
            if opts is not None and opts['source_views']:
                with target.with_suffix('.candidates.jsonl').open() as handle:selections=[json.loads(l) for l in handle]
                missing=next(s for s in selections if s['sample_key']==unsupported[0]['sample_key'])
                if missing['selected'] is not None:raise ValueError('untransformable real function was falsely augmented')
            del cp
    write_json(root/output_name,report)
    return report


def _robustness_summary(originals,views,threshold):
    import numpy as np
    import torch
    from .metrics import metrics
    probability=lambda z:torch.sigmoid(torch.tensor(z,dtype=torch.float32)).item()
    selected=[];worst=[];eligible=[];correct_to_wrong=0;wrong_to_correct=0;robust=0;all_correct=0;attacked=0
    escape=0;false_alarm=0
    for row in originals:
        candidates=views.get(row['sample_key'],[]);y=row['label'];original_correct=row['prediction']==y
        pool=[dict(logit=row['logit'],prediction=row['prediction'])]+candidates
        worst_view=max(pool,key=lambda v:float(np.logaddexp(0,v['logit'])-y*v['logit']))
        worst.append(dict(label=y,score=probability(worst_view['logit'])))
        if not candidates:continue
        first=candidates[0];prediction=first['prediction']
        selected.append(dict(label=y,score=probability(first['logit'])))
        eligible.append(row);correct=prediction==y
        robust+=original_correct and correct
        correct_to_wrong+=original_correct and not correct;wrong_to_correct+=not original_correct and correct
        protected=original_correct and all(v['prediction']==y for v in candidates);all_correct+=protected
        failed=original_correct and not protected;attacked+=failed
        escape+=failed and y==1;false_alarm+=failed and y==0
    n=len(eligible);original_correct=sum(r['prediction']==r['label'] for r in eligible)
    positive=sum(r['label'] for r in eligible);negative=n-positive
    rate=lambda value,denominator:value/denominator if denominator else None
    return dict(eligible=n,total=len(originals),positive=positive,negative=negative,coverage=rate(n,len(originals)),
        candidates=sum(len(c) for c in views.values()),
        perturbed=metrics([r['label'] for r in selected],[r['score'] for r in selected],threshold) if selected else None,
        worst_complete=metrics([r['label'] for r in worst],[r['score'] for r in worst],threshold),
        paired_robust_correct=robust,paired_robust_accuracy=rate(robust,n),
        all_candidates_correct=all_correct,all_candidates_correct_rate=rate(all_correct,n),
        correct_to_wrong=dict(count=correct_to_wrong,denominator=original_correct,rate=rate(correct_to_wrong,original_correct)),
        wrong_to_correct=dict(count=wrong_to_correct,denominator=n-original_correct,rate=rate(wrong_to_correct,n-original_correct)),
        asr=dict(count=attacked,denominator=original_correct,rate=rate(attacked,original_correct)),
        vulnerability_escape=dict(count=escape,denominator=sum(r['label']==1 and r['prediction']==1 for r in eligible),
            rate=rate(escape,sum(r['label']==1 and r['prediction']==1 for r in eligible))),
        benign_false_alarm=dict(count=false_alarm,denominator=sum(r['label']==0 and r['prediction']==0 for r in eligible),
            rate=rate(false_alarm,sum(r['label']==0 and r['prediction']==0 for r in eligible))))


def _robustness_evaluation(rows,run,target,cp,root,materials,args,*,split='valid'):
    import torch,time
    import numpy as np
    from . import model as core,augmentation as data
    from .artifacts import write_json
    from .metrics import decision_boundary
    boundary=decision_boundary(cp['decision_threshold'])
    if split not in ('valid','test'):raise ValueError('robustness evaluation requires Valid or Test')
    evaluation_rows=[r for r in rows if r['split']==split];by={r['sample_key']:r for r in evaluation_rows}
    if not evaluation_rows:raise ValueError(f'{split} split is empty')
    prefix='' if split=='valid' else 'test.'
    clean_run=target.parent if split=='valid' else run
    clean=_finetuning_evaluate(rows,clean_run,split,cp,args,checkpoint_path=target)
    with (clean_run/f'{split}.predictions.jsonl').open() as handle:originals=[json.loads(l) for l in handle]
    output=run/f'{prefix}robustness.metrics.json'
    if output.exists():
        cached=json.loads(output.read_text())
        if (cached.get('threshold')!=cp['decision_threshold'] or cached.get('selected_epoch')!=cp['selected_epoch']
                or cached.get('checkpoint')!=str(target) or cached.get('clean')!=clean):
            raise ValueError('cached robustness checkpoint/evaluation differs')
        return cached
    model=core._load_model(cp,device=core._resolve_device(args.device))
    builder=core.InputBuilder(core.AutoTokenizer.from_pretrained(args.model),source_max_length=2048,context_max_length=384)
    costs=dict(calls=0,function_forwards=0,encoder_sequences=0,encoder_token_slots=0,seconds=0.)
    def score(batch):
        started=time.monotonic();values=[]
        with torch.no_grad(),core._auxiliary_mode(model):
            for at in range(0,len(batch),args.batch_size):
                grouped=builder.codebert_batch(batch[at:at+args.batch_size],device=core._resolve_device(args.device))
                values.extend(model(*grouped).detach().cpu().tolist())
                costs['calls']+=1;costs['function_forwards']+=grouped[-1]
                costs['encoder_sequences']+=len(grouped[0]);costs['encoder_token_slots']+=grouped[0].numel()
        costs['seconds']+=time.monotonic()-started
        return values
    report=dict(clean=clean,threshold=cp['decision_threshold'],selected_epoch=cp['selected_epoch'],checkpoint=str(target))
    if split=='test':report['split']=split
    for name in ('fixed','heldout'):
        costs.update({k:0 for k in costs});started=time.monotonic()
        with (root/f'{prefix}{name}.candidates.jsonl').open() as handle:entries=[json.loads(l) for l in handle]
        views={};flat=[];index=[]
        for entry in entries:
            row=by[entry['sample_key']]
            for kind,candidates in entry['groups'].items():
                for ci,candidate in enumerate(candidates):
                    flat.append(dict(row,raw_source=data.replay_surface_units(row['raw_source'],candidate['units'])))
                    index.append((row['sample_key'],kind,ci,candidate['strict_semantic']))
        logits=score(flat) if flat else []
        for (key,kind,ci,strict),z in zip(index,logits):
            views.setdefault(kind,{}).setdefault(key,[]).append(dict(logit=z,prediction=int(torch.sigmoid(torch.tensor(z)).item()>=boundary),strict_semantic=strict))
        report[name]={}
        from .syntax import SURFACE_OPERATIONS
        for kind in (*SURFACE_OPERATIONS,'combined'):
            group=views.get(kind,{})
            strict={key:[v for v in candidates if v['strict_semantic']] for key,candidates in group.items()}
            stress={key:[v for v in candidates if not v['strict_semantic']] for key,candidates in group.items()}
            report[name][kind]=dict(all=_robustness_summary(originals,group,cp['decision_threshold']),
                strict_semantic=_robustness_summary(originals,strict,cp['decision_threshold']),
                input_only_stress=_robustness_summary(originals,stress,cp['decision_threshold']))
        combined={r['sample_key']:[v for group in views.values() for v in group.get(r['sample_key'],[])] for r in originals}
        report[name+'_overall']=dict(all=_robustness_summary(originals,combined,cp['decision_threshold']),
            strict_semantic=_robustness_summary(originals,{k:[v for v in vs if v['strict_semantic']] for k,vs in combined.items()},cp['decision_threshold']),
            input_only_stress=_robustness_summary(originals,{k:[v for v in vs if not v['strict_semantic']] for k,vs in combined.items()},cp['decision_threshold']))
        with (run/f'{prefix}{name}.predictions.jsonl').open('x') as handle:
            for row in originals:handle.write(json.dumps(dict(sample_key=row['sample_key'],label=row['label'],
                groups={k:g.get(row['sample_key'],[]) for k,g in views.items()}))+'\n')
        report.setdefault('evaluation_costs',{})[name]=dict(costs,total_seconds=time.monotonic()-started)
    # Each model receives the same initial generator states, then its own beams.
    started=time.monotonic();attacks=[];groups={};strict={};stress={};queries=0;original_queries=0
    original_logits={r['sample_key']:r['logit'] for r in originals}
    attack_path=run/f'{prefix}adaptive.predictions.jsonl'
    if attack_path.exists():
        with attack_path.open() as handle:attacks=[json.loads(l) for l in handle]
    else:
        with attack_path.open('x') as handle:
            for at in range(0,len(evaluation_rows),args.batch_size):
                costs.update({k:0 for k in costs});batch_started=time.monotonic()
                result=data.search_surface_views(evaluation_rows[at:at+args.batch_size],score,builder.tokenizer,
                    cp['input_protocol']['source_budget'],materials,seed=42,query_limit=48,
                    purpose='adaptive',original_logits=original_logits)
                result[0]['batch_cost']=dict(costs,total_seconds=time.monotonic()-batch_started)
                for entry in result:
                    entry['visited']=[{k:v for k,v in c.items() if k!='raw_source'} for c in entry['visited']]
                    entry['selected']=None if entry['selected'] is None else {k:v for k,v in entry['selected'].items() if k!='raw_source'}
                    handle.write(json.dumps(entry)+'\n');attacks.append(entry)
                if (at//args.batch_size+1)%25==0:print(f'{split} adaptive {run.name}: {min(at+args.batch_size,len(evaluation_rows))}/{len(evaluation_rows)}',flush=True)
    if [e['sample_key'] for e in attacks]!=[r['sample_key'] for r in evaluation_rows]:raise ValueError('incomplete adaptive evaluation; no silent retries')
    for entry in attacks:
        queries+=entry['candidate_queries'];original_queries+=entry['original_queries'];key=entry['sample_key']
        group=[dict(logit=c['logit'],prediction=int(torch.sigmoid(torch.tensor(c['logit'])).item()>=boundary),
                    strict_semantic=c['strict_semantic']) for c in entry['visited']]
        group.sort(key=lambda c:float(np.logaddexp(0,c['logit'])-by[key]['label']*c['logit']),reverse=True)
        groups[key]=group;strict[key]=[c for c in group if c['strict_semantic']]
        stress[key]=[c for c in group if not c['strict_semantic']]
    report['adaptive']=dict(all=_robustness_summary(originals,groups,cp['decision_threshold']),
        strict_semantic=_robustness_summary(originals,strict,cp['decision_threshold']),
        input_only_stress=_robustness_summary(originals,stress,cp['decision_threshold']),
        candidate_queries=queries,original_queries=original_queries,query_limit=48,
        costs={k:sum(e.get('batch_cost',{}).get(k,0) for e in attacks) for k in (*costs,'total_seconds')},
        per_function_queries={e['sample_key']:e['candidate_queries'] for e in attacks},
        definition='model-specific width-two depth-three BCE search; original included in worst-case results')
    del model
    write_json(output,report)
    return report


def _codebert_robustness(args):
    import torch,transformers,math
    from collections import Counter
    from . import model as core
    from .syntax import surface_materials,SURFACE_RULES
    from .artifacts import write_json
    all_rows=[r for r in core._read_jsonl(args.manifest,source_only=True) if r['dataset']=='primevul' and r['split'] in ('train','valid','test')]
    rows=[r for r in all_rows if r['split'] in ('train','valid')]
    if not rows or any(r.get('repair_supplement') for r in rows):raise ValueError('requires original PrimeVul classification cohort')
    include_incremental=getattr(args,'include_incremental',False)
    include_ablations=False
    if include_incremental and (args.robustness!='all' or args.stage not in ('report','test')):
        raise ValueError('--include-history requires --robustness all and report/test; retired methods cannot train')
    modes=(ROBUSTNESS_MODES+ROBUSTNESS_HISTORY_MODES if include_incremental else ROBUSTNESS_MODES) if args.robustness=='all' else (args.robustness,)
    if include_incremental:
        cache_root=Path(args.output_dir or 'results/codebert_robustness_seed42')
        prefix='test.' if args.stage=='test' else ''
        missing=[mode for mode in modes if not (cache_root/mode/(prefix+'robustness.metrics.json')).exists()]
        if missing:raise FileNotFoundError(f'history requires saved evaluation caches: {missing}')
    ablations=();incremental=();needs_ablation=False;needs_incremental=False
    if args.stage=='test' and (not any(r['split']=='test' for r in all_rows)
                              or any(r.get('repair_supplement') for r in all_rows if r['split']=='test')):
        raise ValueError('requires the original nonempty PrimeVul Test cohort')
    root=Path(args.output_dir or 'results/codebert_robustness_seed42');root.mkdir(parents=True,exist_ok=True)
    training=dict(epochs=3,batch_size=args.batch_size,gradient_accumulation=args.gradient_accumulation,learning_rate=1e-5,weight_decay=.01)
    if args.batch_size*args.gradient_accumulation!=8:raise ValueError('robustness requires effective batch eight')
    try:
        _,base,budget=_matching_full_baseline(args)
        if base['training_config']!=training or base['selection']!='validation_mcc':raise ValueError('baseline learning rate/selection differs')
        bce_target=Path(args.baseline_checkpoint);reuse_reason='exact data/source/input/initialization/training/selection match'
    except (ValueError,FileNotFoundError) as error:
        base=None;bce_target=root/'bce/best.pt';reuse_reason='retrain: '+str(error)
        budget=[(i,math.ceil(sum(r['split']=='train' for r in rows)/8)) for i in (1,2,3)]
    config_path=root/'config.metrics.json';material_path=root/'materials.metrics.json'
    if material_path.exists():materials=json.loads(material_path.read_text())
    else:
        materials=surface_materials([r for r in rows if r['split']=='train']);write_json(material_path,materials)
    tokenizer=core.AutoTokenizer.from_pretrained(args.model)
    builder=core.InputBuilder(tokenizer,source_max_length=2048,context_max_length=384)
    if base is not None:
        capacity=512-len(builder.source_prefix)-2
        protocol=dict(encoder='roberta',native_window=512,logical_budget=2048,
            source_budget=2048-len(builder.source_prefix)-2,window_source_capacity=capacity,stride=capacity//2,
            pooling='mean over unique source positions; repeated prefix/BOS/EOS averaged across windows',
            function_loss='one BCE per original function')
        if base.get('input_protocol')!=protocol or base.get('context_max_length')!=384:
            base=None;bce_target=root/'bce/best.pt';reuse_reason='retrain: baseline input protocol differs'
    config=dict(version=1,manifest=args.manifest,model=args.model,seed=42,training=training,
        source_max_length=2048,source_budget=2048-len(builder.source_prefix)-2,native_window=512,
        stride=(512-len(builder.source_prefix)-2)//2,initialization=core._CLASSIFIER_INITIALIZATION,
        head='mean pooling + linear; fresh pretrained full CodeBERT',selection='original clean Valid MCC and tie-break',
        main_groups=['bce','vat','ours'],vat=core.VAT_CONFIG,
        surface_rules=SURFACE_RULES,materials=str(material_path),function_name='input-only stress; resolved self-calls updated; separate strict metrics',
        bce_checkpoint=str(bce_target),bce_reuse_reason=reuse_reason,
        counts=dict(Counter(f"{r['split']}/label_{r['label']}" for r in rows)),
        environment=dict(torch=torch.__version__,transformers=transformers.__version__))
    if config_path.exists():
        saved=json.loads(config_path.read_text())
        expected=dict(config,vat=dict(core.VAT_CONFIG,xi=saved['vat']['xi']))
        if saved!=expected:raise ValueError('existing robustness preparation configuration differs')
        config=saved
    else:write_json(config_path,config)
    if args.stage in ('prepare','report','all'):
        _robustness_fixed_candidates(rows,materials,root,tokenizer,config['source_budget'])
    if args.stage=='prepare':return config
    smoke=root/'mainline.smoke.metrics.json'
    if args.stage in ('all','smoke') and not smoke.exists():
        if base is None:base=torch.load(bce_target,map_location='cpu',weights_only=False)
        result=_robustness_smoke(rows,root,config,materials,args,base,modes=ROBUSTNESS_MODES,output_name=smoke.name)
        if result['xi']!=config['vat']['xi']:raise ValueError('regression smoke differs from the frozen VAT setting')
    if args.stage=='smoke':return json.loads(smoke.read_text())
    xi=config['vat']['xi']
    if args.stage in ('train','all') and not smoke.exists():raise ValueError('complete the mainline real smoke before training')
    if args.stage=='test':
        missing=[mode for mode in modes if not (bce_target if mode=='bce' else root/mode/'best.pt').exists()]
        if missing:raise FileNotFoundError(f'Test requires completed Valid-selected checkpoints: {missing}')
        _robustness_fixed_candidates(all_rows,materials,root,tokenizer,config['source_budget'],split='test')
    results={}
    for mode in modes:
        run=root/mode;run.mkdir(exist_ok=True);target=bce_target if mode=='bce' else run/'best.pt'
        opts=None if mode=='bce' else core.robustness_config(mode,xi=xi)
        expected_members={split:[(r['sample_key'],r['label']) for r in rows if r['split']==split] for split in ('train','valid')}
        if target.exists():
            cp=torch.load(target,map_location='cpu',weights_only=False)
            _check_robustness_checkpoint(cp,rows,opts,materials,training,args,config['source_budget'])
            if cp.get('repair_supervision') is not None:
                raise ValueError('repair-supervised checkpoint cannot masquerade as a robustness-only result')
        elif args.stage in ('train','all'):
            if target.with_suffix('.training.jsonl').exists():raise FileExistsError('running/interrupted robustness job; inspect before retrying')
            cp=core.train_model(None,target,records=rows,variant='codebert_source',model_path=args.model,
                source_max_length=2048,context_max_length=384,seed=42,device=args.device,finetuning_mode='full',
                robustness_options=opts,surface_materials=materials if opts is not None and opts['source_views'] else None,
                log_every=25,**training)
        else:raise FileNotFoundError(target)
        with target.with_suffix('.training.jsonl').open() as handle:history=[json.loads(l) for l in handle]
        if [(r['epoch'],r['optimizer_steps']) for r in history if r['event']=='epoch']!=budget:
            raise ValueError('robustness optimizer/epoch budget differs')
        for split in ('train','valid'):_finetuning_evaluate(rows,target.parent,split,cp,args,checkpoint_path=target)
        if args.stage in ('all','report','test'):
            results[mode]=_robustness_evaluation(all_rows if args.stage=='test' else rows,run,target,cp,root,materials,args,
                                              split='test' if args.stage=='test' else 'valid')
            results[mode]['training']=[r for r in history if r['event']=='epoch']
            epochs=results[mode]['training']
            results[mode]['training_cost']=dict(optimizer_steps=sum(r['optimizer_steps'] for r in epochs),
                original_functions=sum(r['split']=='train' for r in rows)*len(epochs),
                seconds=sum(r['epoch_seconds'] for r in epochs) if all('epoch_seconds' in r for r in epochs) else None,
                time_scope='epoch wall time including clean Valid selection; historical BCE wall time was not logged',
                candidate_queries=sum(r.get('candidate_queries',0) for r in epochs),
                original_search_queries=sum(r.get('original_search_queries',0) for r in epochs),
                search_seconds=sum(r.get('search_seconds',0.) for r in epochs),
                encoder_sequences={stage:sum(r.get(field,0) for r in epochs) for stage,field in (
                    ('bce','bce_encoder_sequences'),('search','search_encoder_sequences'),
                    ('real_js','real_encoder_sequences'),('vat','vat_encoder_sequences'))})
            if mode=='bce':results[mode]['training_cost']['encoder_sequences']['bce']=None
        del cp
    if results:
        name=('history.comparison.metrics.json' if include_incremental else
              'mainline.comparison.metrics.json') if args.robustness=='all' else f'mainline.{args.robustness}.comparison.metrics.json'
        report_config=dict(config,active_groups=list(ROBUSTNESS_MODES),historical_groups=list(ROBUSTNESS_HISTORY_MODES) if include_incremental else [])
        if include_incremental:
            report_config=dict(report_config,incremental_groups=list(incremental),
                               incremental_options={mode:core.robustness_config(mode,xi=xi) for mode in incremental})
        if args.stage=='test':
            name='test.'+name
            report_config=dict(report_config,evaluation_split='test',evaluation_counts=dict(Counter(
                f"label_{r['label']}" for r in all_rows if r['split']=='test')))
        write_json(root/name,dict(config=report_config,results=results))
        import csv
        with (root/name.replace('.json','.csv')).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['method','setting','accuracy','precision','recall','f1','mcc','auc']);writer.writeheader()
            for method,result in results.items():
                writer.writerow(dict(method=method,setting='clean',**{k:result['clean']['metrics'][k] for k in writer.fieldnames[2:]}))
                for setting in ('fixed','heldout'):
                    m=result[setting+'_overall']['all']['worst_complete']
                    writer.writerow(dict(method=method,setting=setting+'/all_families/worst_complete',**{k:m[k] for k in writer.fieldnames[2:]}))
                    for kind,metric in result[setting].items():
                        m=metric['all']['perturbed']
                        if m:writer.writerow(dict(method=method,setting=setting+'/'+kind,**{k:m[k] for k in writer.fieldnames[2:]}))
                        m=metric['all']['worst_complete']
                        writer.writerow(dict(method=method,setting=setting+'/'+kind+'/worst_complete',**{k:m[k] for k in writer.fieldnames[2:]}))
                m=result['adaptive']['all']['worst_complete']
                writer.writerow(dict(method=method,setting='adaptive/worst_complete',**{k:m[k] for k in writer.fieldnames[2:]}))
        return results
    return dict(completed_modes=modes,config=config)



def _repair_smoke(rows, pairs, root, materials, args):
    import tempfile
    import torch
    from . import model as core, repair_supervision as audit
    from .artifacts import write_json
    train = [r for r in rows if r['split'] == 'train']
    chosen = [r for label in (0, 1) for r in [r for r in train if r['label'] == label][:4]]
    chosen += [next(r for r in rows if r['split'] == 'valid' and r['label'] == label) for label in (0, 1)]
    selected = None
    short = [pair for pair in pairs if max(r['actual_tokens'] for r in pair['regions']) < 256 and pair['group'] is not None]
    for pair in short:
        partner = next((other for other in short if other['group'] == pair['group']
                        and other['event_id'] != pair['event_id']), None)
        if partner:
            selected = [pair, partner]
            break
    if not selected:
        raise ValueError('smoke requires two short independent audited repairs of one mechanism')
    device = core._resolve_device(args.device)
    builder = core.InputBuilder(core.AutoTokenizer.from_pretrained(args.model), source_max_length=2048, context_max_length=384)
    core._seed_everything(42)
    fresh = core._build_model('codebert_source', args.model, device=device, lora_r=16, lora_alpha=32,
            lora_dropout=.05, target_modules=(), gradient_checkpointing=device.type == 'cuda',
            fusion_dim=256, fusion_heads=8, finetuning_mode='full')
    if device.type == 'cuda':
        fresh.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    embedding = fresh.encoder.get_input_embeddings().weight
    initial_embedding = embedding.detach().cpu().clone()
    initial_head = core._cpu_state(fresh.task_modules.state_dict())
    with core._auxiliary_mode(fresh):
        vectors, logits, _, _ = audit.region_readouts(fresh, builder, selected, device=device)
        state, contrast, info = audit.local_losses(vectors, logits, selected, beta=.1, tau=.1)
        state_gradient = torch.autograd.grad(state.mean(), (embedding, fresh.task_modules['classifier'].weight), retain_graph=True)
        contrast_gradient = torch.autograd.grad(contrast.mean(), embedding)[0]
    signals = dict(state_encoder_gradient=float(state_gradient[0].norm()), state_head_gradient=float(state_gradient[1].norm()),
                   contrast_encoder_gradient=float(contrast_gradient.norm()), contrast_anchors=info['contrast_anchors'])
    if min(signals[k] for k in ('state_encoder_gradient', 'state_head_gradient', 'contrast_encoder_gradient')) <= 0:
        raise ValueError('real auxiliary gradients are zero')
    # A small local-only direction check uses the same public encoder/readout.
    optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-5, weight_decay=.01)
    values = []
    for step in range(5):
        with core._auxiliary_mode(fresh):
            vectors, logits, _, _ = audit.region_readouts(fresh, builder, selected[:1], device=device)
            state, _, _ = audit.local_losses(vectors, logits, selected[:1], beta=0., tau=.1)
            values.append(dict(loss=float(state.detach().mean()), margin=float((logits[0] - logits[1]).detach())))
            if step < 4:
                state.mean().backward()
        if step < 4:
            torch.nn.utils.clip_grad_norm_(fresh.parameters(), 1.)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
    if values[-1]['loss'] >= values[0]['loss'] or values[-1]['margin'] <= values[0]['margin']:
        raise ValueError('local-only real smoke did not move in the supervised direction')
    del fresh, optimizer, state_gradient, contrast_gradient, vectors, logits, state, contrast, embedding
    if device.type == 'cuda':torch.cuda.empty_cache()
    checks = {}
    with tempfile.TemporaryDirectory(prefix='repair-smoke-') as temporary:
        for mode in ('state', 'state_contrast'):
            target = Path(temporary) / mode / 'best.pt'
            cp = core.train_model(None, target, records=chosen, variant='codebert_source', model_path=args.model,
                    source_max_length=2048, context_max_length=384, seed=42, device=args.device, finetuning_mode='full',
                    epochs=1, batch_size=4, gradient_accumulation=2, learning_rate=1e-5, weight_decay=.01,
                    robustness_options=core.robustness_config('hard_aug_bce'), surface_materials=materials,
                    repair_supervision=audit.supervision_config(mode), repair_pairs=selected, repair_pairs_path=args.repair_pairs)
            encoder_updated = not torch.equal(initial_embedding, cp['encoder_state']['embeddings.word_embeddings.weight'])
            head_updated = any(not torch.equal(value, cp['task_state'][key]) for key, value in initial_head.items())
            memory_model = core._load_model(cp, device=device)
            first = core._predict_logits(memory_model, chosen[-2:], builder, variant='codebert_source',
                                         excluded_groups=(), batch_size=2, device=device)
            del memory_model
            second = core.predict_checkpoint(target, chosen[-2:], device=args.device, batch_size=2)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            history = [json.loads(line) for line in target.with_suffix('.training.jsonl').open() if json.loads(line)['event'] == 'epoch']
            if not encoder_updated or not head_updated or history[0]['audit_pairs'] != 2:
                raise ValueError('real repair smoke update/member check failed')
            if mode == 'state_contrast' and history[0]['audit_contrast_anchors'] != 4:
                raise ValueError('real smoke contrast positives missing')
            checks[mode] = dict(encoder_updated=encoder_updated, shared_head_updated=head_updated,
                               save_reload_exact=True, memory_vs_serialized_checkpoint_exact=True, epoch=history[0])
            del cp
    result = dict(kind='real CodeBERT smoke, not formal results', pair_members=[p['sample_id'] for p in selected],
                  signals=signals, local_direction=values, checks=checks)
    write_json(root / 'smoke.metrics.json', result)
    return result


def _codebert_repair_supervision(args):
    import math
    import torch
    from collections import Counter
    from . import model as core, repair_supervision as audit
    from .artifacts import write_json
    if args.robustness != 'hard_aug_bce' or args.include_incremental or args.stage == 'test':
        raise ValueError('repair supervision supports only hard_aug_bce and Train/Valid stages; no VAT/Test/history')
    if args.batch_size != 4 or args.gradient_accumulation != 2:
        raise ValueError('repair experiment requires batch=4, accumulation=2')
    root = Path(args.output_dir or 'results/codebert_repair_supervision_seed42')
    reference = Path('results/codebert_robustness_seed42')
    if root.resolve() == reference.resolve():
        raise ValueError('new repair results must not overwrite existing robustness results')
    all_records = core._read_jsonl(args.manifest, source_only=True)
    rows = [r for r in all_records if r['dataset'] == 'primevul' and r['split'] in ('train', 'valid')]
    _, base, budget = _matching_full_baseline(args)
    training = dict(epochs=3, batch_size=4, gradient_accumulation=2, learning_rate=1e-5, weight_decay=.01)
    if base['training_config'] != training:
        raise ValueError('existing full baseline training configuration differs')
    del base
    materials = json.loads((reference / 'materials.metrics.json').read_text())
    opts = core.robustness_config('hard_aug_bce')
    tokenizer = core.AutoTokenizer.from_pretrained(args.model)
    builder = core.InputBuilder(tokenizer, source_max_length=2048, context_max_length=384)
    source_budget = 2048 - len(builder.source_prefix) - 2
    a_target = reference / 'hard_aug_bce/best.pt'
    a = torch.load(a_target, map_location='cpu', weights_only=False)
    _check_robustness_checkpoint(a, rows, opts, materials, training, args, source_budget)
    if a.get('repair_supervision') is not None:
        raise ValueError('A must be the unmodified matching hard_aug_bce checkpoint')
    del a
    neighbor = Path(__file__).resolve().parents[2] / 'knowledge-guided-vulnerability-detection'
    samples, valid_samples, preparation = audit.copy_training_source(neighbor, args.repair_pairs, all_records)
    pairs, region_stats = audit.prepare_regions(samples, builder)
    valid_pairs, valid_stats = audit.prepare_regions(valid_samples, builder)
    if not pairs:
        raise ValueError('no eligible audited training regions')
    positive_coverage = []
    for epoch in (1, 2, 3):
        schedule = audit.auxiliary_schedule(pairs, budget[0][1], seed=42, epoch=epoch)
        batches = [batch for grouped in schedule.values() for batch in grouped]
        positive_coverage.append(dict(epoch=epoch, pairs=len(pairs), batches=len(batches),
                                      positive_pairs=sum(len(batch) for batch in batches if len(batch) == 2),
                                      state_only_pairs=sum(len(batch) for batch in batches if len(batch) == 1)))
    root.mkdir(parents=True, exist_ok=True)
    config = dict(version=1, manifest=args.manifest, model=args.model, seed=42, training=training,
                  counts=dict(Counter(f"{r['split']}/label_{r['label']}" for r in rows)),
                  source_max_length=2048, source_budget=source_budget, native_window=512, stride=(512 - len(builder.source_prefix) - 2) // 2,
                  initialization=core._CLASSIFIER_INITIALIZATION, a_checkpoint=str(a_target), repair_pairs=args.repair_pairs,
                  supervision={mode: audit.supervision_config(mode) for mode in ('state', 'state_contrast')},
                  selection='original clean Valid MCC and tie-break; local diagnosis never selects models')
    stats = dict(copy=preparation, train_regions=region_stats, local_valid_regions=valid_stats, positive_coverage=positive_coverage)
    for name, value in [('config.metrics.json', config), ('preparation.metrics.json', stats)]:
        path = root / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f'existing repair preparation differs: {path}')
        if not path.exists():write_json(path, value)
    if args.stage == 'prepare':return stats
    smoke = root / 'smoke.metrics.json'
    if args.stage in ('smoke', 'all') and not smoke.exists():
        _repair_smoke(rows, pairs, root, materials, args)
    if args.stage == 'smoke':return json.loads(smoke.read_text())
    if args.stage in ('train', 'all') and not smoke.exists():
        raise ValueError('complete real repair smoke before training')
    _robustness_fixed_candidates(rows, materials, reference, tokenizer, source_budget)
    modes = audit.MODES if args.repair_supervision == 'all' else (args.repair_supervision,)
    results = {}
    for mode in modes:
        run = reference / 'hard_aug_bce' if mode == 'none' else root / mode
        run.mkdir(exist_ok=True)
        target = a_target if mode == 'none' else run / 'best.pt'
        repair_config = None if mode == 'none' else audit.supervision_config(mode)
        members = [dict(sample_id=p['sample_id'], dataset=p['dataset'], event_id=p['event_id'], group=p['group']) for p in pairs]
        if target.exists():
            cp = torch.load(target, map_location='cpu', weights_only=False)
            _check_robustness_checkpoint(cp, rows, opts, materials, training, args, source_budget)
            if cp.get('repair_supervision') != repair_config or mode != 'none' and (cp.get('repair_training_members') != members or cp.get('repair_pairs_path') != args.repair_pairs):
                raise ValueError('existing checkpoint does not match this auxiliary training')
        elif args.stage in ('train', 'all'):
            if target.with_suffix('.training.jsonl').exists():
                raise FileExistsError('running/interrupted repair job; inspect before retrying')
            cp = core.train_model(None, target, records=rows, variant='codebert_source', model_path=args.model,
                                  source_max_length=2048, context_max_length=384, seed=42, device=args.device,
                                  finetuning_mode='full', robustness_options=opts, surface_materials=materials,
                                  repair_supervision=repair_config, repair_pairs=pairs, repair_pairs_path=args.repair_pairs,
                                  log_every=25, **training)
        else:
            raise FileNotFoundError(target)
        epochs = [json.loads(line) for line in target.with_suffix('.training.jsonl').open() if json.loads(line)['event'] == 'epoch']
        if [(e['epoch'], e['optimizer_steps']) for e in epochs] != budget:
            raise ValueError('repair experiment optimizer/epoch budget differs')
        train_result = _finetuning_evaluate(rows, run, 'train', cp, args, checkpoint_path=target)
        _finetuning_evaluate(rows, run, 'valid', cp, args, checkpoint_path=target)
        if args.stage in ('all', 'report'):
            formal = _robustness_evaluation(rows, run, target, cp, reference, materials, args)
            local_output = root / 'none' if mode == 'none' else run
            local_output.mkdir(exist_ok=True)
            local = audit.evaluate_local(target, valid_pairs, local_output / 'local_valid.metrics.json', device=args.device)
            results[mode] = dict(formal, train=train_result, local_valid=local, training=epochs,
                                 training_cost=dict(optimizer_steps=sum(e['optimizer_steps'] for e in epochs),
                                     seconds=sum(e['epoch_seconds'] for e in epochs),
                                     candidate_queries=sum(e['candidate_queries'] for e in epochs),
                                     auxiliary_seconds=sum(e.get('audit_seconds', 0.) for e in epochs),
                                     auxiliary_function_forwards=sum(e.get('audit_function_forwards', 0) for e in epochs),
                                     auxiliary_encoder_sequences=sum(e.get('audit_encoder_sequences', 0) for e in epochs)))
        del cp
    if results:
        name = 'comparison.metrics.json' if args.repair_supervision == 'all' else f'{args.repair_supervision}.comparison.metrics.json'
        write_json(root / name, dict(config=config, results=results))
        import csv
        with (root / name.replace('.json', '.csv')).open('w', newline='') as handle:
            fields = ['method', 'setting', 'accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc']
            writer = csv.DictWriter(handle, fieldnames=fields);writer.writeheader()
            for mode, result in results.items():
                settings = {'train': result['train']['metrics'], 'clean_valid': result['clean']['metrics'],
                            **{name: result[name]['all']['worst_complete'] for name in ('fixed_overall', 'heldout_overall', 'adaptive')}}
                for setting, metrics in settings.items():
                    writer.writerow(dict(method=mode, setting=setting, **{key: metrics[key] for key in fields[2:]}))
        return results
    return dict(completed_modes=modes, config=config)


def _codebert_finetuning(args):
    if getattr(args, 'repair_supervision', 'none') != 'none':
        return _codebert_repair_supervision(args)
    if getattr(args,'include_incremental',False) and (args.robustness!='all' or args.stage not in ('report','test')):
        raise ValueError('historical methods are report/test only; they cannot start training')
    return _codebert_robustness(args)

def _infer(args):
    import torch
    from .model import predict_checkpoint
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    source=Path(args.source_file).read_text(encoding='utf-8')
    if not source.strip():raise ValueError('single-function source must be nonempty')
    row=dict(raw_source=source,language=args.language,file_name=Path(args.source_file).name)
    logit=predict_checkpoint(args.checkpoint,[row],device=args.device)[0]
    probability=torch.sigmoid(logit)
    return dict(logit=float(logit),probability=float(probability),threshold=checkpoint['decision_threshold'],
                prediction=int(probability>=checkpoint['decision_threshold']))

def build_parser():
    parser=argparse.ArgumentParser(description='Single-function CodeBERT vulnerability detection and dynamic hard augmentation')
    sub=parser.add_subparsers(dest='command',required=True)
    fit=sub.add_parser('codebert-finetuning')
    fit.add_argument('--manifest',default='data/benchmark/manifest_sven_commit_disjoint.jsonl')
    fit.add_argument('--model',default='/home/PublicData/PHY-data/resource/codebert-base')
    fit.add_argument('--output-dir')
    fit.add_argument('--stage',choices=('prepare','smoke','train','report','test','all'),default='all')
    fit.add_argument('--robustness',nargs='?',const='all',default='hard_aug_bce',choices=('all',)+ROBUSTNESS_MODES)
    fit.add_argument('--repair-supervision', default='none', choices=('none', 'state', 'state_contrast', 'all'))
    fit.add_argument('--repair-pairs', default='data/high_quality_repair_pairs.json')
    fit.add_argument('--include-history','--include-incremental',dest='include_incremental',action='store_true',
                     help='report/test all eleven saved methods; historical methods cannot train')
    fit.add_argument('--baseline-checkpoint',default='results/codebert_finetuning_seed42/full/lr_1e-05/best.pt')
    fit.add_argument('--device',default='cuda:0')
    fit.add_argument('--batch-size',type=int,default=4)
    fit.add_argument('--gradient-accumulation',type=int,default=2)
    fit.set_defaults(func=_codebert_finetuning,finetuning_mode='full')
    infer=sub.add_parser('infer')
    infer.add_argument('--checkpoint',required=True)
    infer.add_argument('--source-file',required=True)
    infer.add_argument('--language',choices=('c','cpp'),default='c')
    infer.add_argument('--device',default='auto')
    infer.set_defaults(func=_infer)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    result=args.func(args)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
