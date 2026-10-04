"""Bounded fixed-checkpoint transfer diagnosis; never trains or assigns patch labels."""
from collections import Counter, defaultdict
import difflib
import gc
import json
import math
from pathlib import Path
import random

from .cfg_data import (read_records, read_jsonl, iter_jsonl, atomic_json, source_hash,
                       cohort_hash, file_sha256, digest, abstract_cfg, AttributeVocabulary)
from .syntax import parser_for

MODELS = {
    'baseline': ('results/cfg_abc_seed42', 'baseline'),
    'C': ('results/cfg_abc_seed42', 'cfg'),
    'P0_C': ('results/cfg_dep_cfg_windowfix_seed42', 'dep_pretrain_cfg'),
    'control_C': ('results/cfg_control_cfg_seed42', 'control_cfg'),
    'structure_pool': ('results/cfg_pool_seed42', 'joint_structure_pool'),
}
COMMENTS = {'neutral':'\n/* diagnostic note is available */\n',
            'risk_cue':'\n/* buffer overflow is possible */\n',
            'safe_cue':'\n/* buffer overflow is impossible */\n'}


def executable_tree(source, language):
    raw=source.encode();tree=parser_for(language).parse(raw)
    if tree.root_node.has_error:raise ValueError('syntax_error')
    if sum(n.type=='function_definition' for n in tree.root_node.named_children)!=1:
        raise ValueError('not_one_top_level_function')
    def visit(n):
        if n.type=='comment':return None
        children=tuple(v for c in n.children if (v:=visit(c)) is not None)
        return (n.type,children if n.children else raw[n.start_byte:n.end_byte].decode())
    return visit(tree.root_node)


def comment_variants(row, builder):
    source=row['raw_source'];language=row.get('resolved_language') or row['language']
    if '#' in source or '\\\n' in source or '??/' in source:raise ValueError('preprocessor_or_splice')
    original=executable_tree(source,language)
    result={}
    for name,suffix in COMMENTS.items():
        changed=source+suffix
        if executable_tree(changed,language)!=original:raise ValueError('executable_ast_changed')
        ids=builder._encode(changed)
        if len(ids)>builder.source_max_length:raise ValueError('truncation')
        result[name]=dict(raw_source=changed,source_tokens=len(ids))
    if len({v['source_tokens'] for v in result.values()})!=1:raise ValueError('unequal_comment_token_counts')
    return result


def natural_pairs(rows):
    groups=defaultdict(list)
    for row in rows:
        if row['split']=='valid':groups[row['function_name']].append(row)
    pairs=[]
    for name,group in sorted(groups.items()):
        if len(group)!=2 or {r['label'] for r in group}!={0,1}:continue
        a,b=sorted(group,key=lambda r:-r['label'])
        ratio=difflib.SequenceMatcher(None,a['raw_source'],b['raw_source'],autojunk=False).ratio()
        if ratio>=.70:
            pairs.append(dict(function=name,positive=a['sample_key'],negative=b['sample_key'],similarity=ratio,
                provenance='same-name near-neighbours, not verified repair pairs',
                expectation='dataset-label discrimination only; local safety direction requires manual review',
                diff=''.join(difflib.unified_diff(a['raw_source'].splitlines(True),b['raw_source'].splitlines(True)))))
    return pairs


def prepare(output, base=None):
    from .cfg_experiment import _base_module,_tokenizer
    root=Path(output)
    if root.exists():raise FileExistsError(f'diagnostic output exists: {root}')
    c=json.loads(Path('results/cfg_abc_seed42/config.json').read_text())
    all_rows=read_records(c['dataset'],c['source_dataset']);rows=[r for r in all_rows if r['split'] in ('train','valid')]
    if cohort_hash(all_rows)!=c['cohort_sha256']:raise ValueError('original cohort changed')
    base=_base_module() if base is None else base;builder=_tokenizer(base,c)
    eligible={0:[],1:[]};excluded=Counter();variants={}
    for row in sorted(rows,key=lambda r:r['sample_key']):
        if row['split']!='valid':continue
        try:changed=comment_variants(row,builder)
        except ValueError as exc:excluded[str(exc)]+=1;continue
        eligible[row['label']].append(row);variants[row['sample_key']]=changed
    rng=random.Random(42);selected=[]
    for label in (0,1):selected+=rng.sample(eligible[label],min(8,len(eligible[label])))
    pairs=natural_pairs(rows);keys={r['sample_key'] for r in selected}|{p[k] for p in pairs for k in ('positive','negative')}
    originals=[r for r in rows if r['sample_key'] in keys]
    cases=[]
    for row in originals:
        cases.append(dict(sample_key=row['sample_key'],case='original',raw_source=row['raw_source'],label=row['label'],
                          split='valid',function_name=row['function_name'],language=row.get('resolved_language') or row['language'],
                          source_tokens=len(builder._encode(row['raw_source']))))
        if row['sample_key'] in {r['sample_key'] for r in selected}:
            for name,changed in variants[row['sample_key']].items():cases.append(dict(cases[-1],case=name,**changed))
    for pair in pairs:
        left=next(r for r in originals if r['sample_key']==pair['positive'])
        right=next(r for r in originals if r['sample_key']==pair['negative'])
        pair['visible_source_ids_equal']=builder.source_ids(left)==builder.source_ids(right)
        pair['source_tokens']=[len(builder._encode(r['raw_source'])) for r in (left,right)]
    models={}
    for name,(directory,variant) in MODELS.items():
        run=Path(directory);config=json.loads((run/'config.json').read_text());complete=json.loads((run/variant/'complete.json').read_text())
        shared=('cohort_sha256','source_max_length','model_path','graph_file_sha256','vocabulary_sha256',
                'seed','epochs','learning_rate','graph_learning_rate','batch_size','gradient_accumulation')
        if any(config[k]!=c[k] for k in shared):raise ValueError('model cohort/graph/training configuration mismatch')
        checkpoint=run/variant/'best.pt'
        if file_sha256(checkpoint)!=complete['checkpoint_sha256']:raise ValueError('checkpoint changed')
        models[name]=dict(directory=directory,variant=variant,checkpoint_sha256=complete['checkpoint_sha256'])
    root.mkdir(parents=True)
    protocol=dict(seed=42,c_config=c,models=models,comment_selection='8 per dataset label; sorted keys then fixed seed; before score access',
        eligible={k:len(v) for k,v in eligible.items()},excluded=dict(excluded),
        comments=COMMENTS,selected_comment_functions=[r['sample_key'] for r in selected],natural_pairs=pairs,
        graph_policy='append comments after function; executable AST and original byte positions invariant; reuse original graph and regenerate formal view/attributes',
        no_new_labels=True,threshold_policy='saved checkpoint thresholds only')
    atomic_json(root/'protocol.json',protocol)
    with (root/'cases.jsonl').open('x') as f:
        for case in cases:f.write(json.dumps(case)+'\n')
    # Selection is persisted before inspecting any model response.
    saved_analysis(root)
    position_reference(root)
    return protocol


def saved_analysis(root):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text())
    rows=read_records(protocol['c_config']['dataset'],'primevul',splits=('train','valid'))
    valid={r['sample_key']:r for r in rows if r['split']=='valid'}
    predictions={}
    for name,model in protocol['models'].items():
        path=Path(model['directory'])/model['variant']/'valid.predictions.jsonl'
        complete=json.loads((path.parent/'complete.json').read_text())
        if file_sha256(path)!=complete['validation_predictions_sha256']:raise ValueError('saved prediction identity mismatch')
        p={r['sample_key']:r for r in read_jsonl(path)}
        if set(p)!=set(valid) or any(p[k]['label']!=r['label'] for k,r in valid.items()):raise ValueError('prediction cohort mismatch')
        predictions[name]=p
    result={'natural_pairs':[],'comment_original_correctness':{},'auxiliary_associations':{}}
    for pair in protocol['natural_pairs']:
        result['natural_pairs'].append(dict(pair,models={name:dict(positive_score=p[pair['positive']]['score'],
            negative_score=p[pair['negative']]['score'],threshold=p[pair['positive']]['threshold'],
            correct_order=p[pair['positive']]['score']>p[pair['negative']]['score'],
            both_correct=p[pair['positive']]['score']>=p[pair['positive']]['threshold'] and p[pair['negative']]['score']<p[pair['negative']]['threshold']) for name,p in predictions.items()}))
    for name,p in predictions.items():
        result['comment_original_correctness'][name]=dict(Counter('correct' if (p[k]['score']>=p[k]['threshold'])==bool(p[k]['label']) else 'wrong'
            for k in protocol['selected_comment_functions']))
    for task,path in [('dependency','results/cfg_dep_relation_eval_windowfix_seed42/valid.predictions.jsonl'),
                      ('control','results/cfg_control_relations_seed42/valid.control.jsonl')]:
        grouped=defaultdict(list)
        for q in read_jsonl(path):
            if q['split']!='valid' or q['sample_key'] not in valid:raise ValueError('auxiliary valid cohort mismatch')
            if 'source_sha256' in q and q['source_sha256']!=source_hash(valid[q['sample_key']]['raw_source']):raise ValueError('auxiliary source changed')
            grouped[q['sample_key']].append(q)
        association=[]
        for key,queries in sorted(grouped.items()):
            accuracy=sum((q['score']>=.5)==bool(q['label']) for q in queries)/len(queries)
            association.append(dict(sample_key=key,query_count=len(queries),auxiliary_accuracy=accuracy,
                mixed_labels={q['label'] for q in queries}=={0,1},
                classification_correct={name:(p[key]['score']>=p[key]['threshold'])==bool(p[key]['label']) for name,p in predictions.items()}))
        result['auxiliary_associations'][task]={'functions':len(association),'all_auxiliary_correct':{
            name:dict(total=sum(r['auxiliary_accuracy']==1 for r in association),
                      classification_errors=sum(r['auxiliary_accuracy']==1 and not r['classification_correct'][name] for r in association)) for name in predictions}}
        result['auxiliary_associations'][task]['mixed_at_least_8_all_correct']={name:dict(
            total=sum(r['auxiliary_accuracy']==1 and r['mixed_labels'] and r['query_count']>=8 for r in association),
            classification_errors=sum(r['auxiliary_accuracy']==1 and r['mixed_labels'] and r['query_count']>=8 and not r['classification_correct'][name] for r in association)) for name in predictions}
        with (root/f'{task}.associations.jsonl').open('w') as f:
            for r in association:f.write(json.dumps(r)+'\n')
    atomic_json(root/'saved_analysis.json',result)
    return result


def fixed_logits(model, inputs, *, baseline=False):
    import torch
    model.eval()
    with torch.no_grad():
        if baseline:
            source=model(*inputs).reshape(-1)
            return source,torch.zeros_like(source)
        return model.branch_logits(*inputs)


def evaluate(output, *, device='cuda:0',base=None):
    import torch
    from .cfg_experiment import _base_module,_tokenizer,_graph_inputs,CHECKPOINT_SCHEMA
    from .cfg_network import build_model
    from .cfg_behavior import extract_bound_behavior,BoundVocabulary
    from .progress import training_bar
    root=Path(output);protocol=json.loads((root/'protocol.json').read_text());c=protocol['c_config']
    if (root/'responses.jsonl').exists():raise FileExistsError('responses already exist; preserve fixed evaluation')
    rows=read_records(c['dataset'],c['source_dataset']);by_key={r['sample_key']:r for r in rows}
    if cohort_hash(rows)!=c['cohort_sha256'] or file_sha256(c['graphs'])!=c['graph_file_sha256']:raise ValueError('source/graph identity mismatch')
    cases=read_jsonl(root/'cases.jsonl');keys={r['sample_key'] for r in cases};graphs={}
    for row in iter_jsonl(c['graphs'],description='Read diagnostic graphs'):
        if row['sample_key'] in keys:
            if row['source_sha256']!=source_hash(by_key[row['sample_key']]['raw_source']):raise ValueError('graph source mismatch')
            graphs[row['sample_key']]=row['graph']
    if set(graphs)!=keys:raise ValueError('diagnostic graph missing')
    base=_base_module() if base is None else base;resolved=base._resolve_device(device);builder=_tokenizer(base,c)
    for case in cases:
        original=by_key[case['sample_key']]
        if original['split']!='valid' or original['label']!=case['label']:raise ValueError('diagnostic membership changed')
        if case['case']=='original':
            if case['raw_source']!=original['raw_source']:raise ValueError('original source changed')
        else:
            variants=comment_variants(original,builder)
            if case['raw_source']!=variants[case['case']]['raw_source']:raise ValueError('unverified transformation')
    responses=[]
    with (root/'responses.jsonl').open('x') as out:
        for name,entry in protocol['models'].items():
            path=Path(entry['directory'])/entry['variant']/'best.pt'
            if file_sha256(path)!=entry['checkpoint_sha256']:raise ValueError('fixed model changed')
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            if entry['variant']=='baseline':model=base._load_model(checkpoint,device=resolved);views={};encoded={}
            else:
                if checkpoint['cfg_ablation_version']!=CHECKPOINT_SCHEMA:raise ValueError('checkpoint schema mismatch')
                config=checkpoint['model_config'];vocab=AttributeVocabulary(checkpoint['vocabulary'])
                views={k:abstract_cfg(g) for k,g in graphs.items()}
                if config.get('behavior_dir'):
                    directory=Path(config['behavior_dir']);values=json.loads((directory/'vocabulary.json').read_text())
                    if digest(values)!=config['behavior_vocabulary_sha256']:raise ValueError('attribute vocabulary mismatch')
                    bv=BoundVocabulary(values)
                    for key in views:views[key]['behavior']=bv.encode(extract_bound_behavior(graphs[key]))
                encoded={k:vocab.encode(v) for k,v in views.items()}
                model=build_model(base,config,vocab.sizes(),resolved,training=False)
                base.set_peft_model_state_dict(model.encoder,checkpoint['adapter_state'])
                model.task_modules.load_state_dict(checkpoint['task_state'])
            model.eval();threshold=float(checkpoint['decision_threshold'])
            with torch.no_grad():
                for case in training_bar(cases,desc=f'Fixed diagnosis · {name}'):
                    row=dict(by_key[case['sample_key']],raw_source=case['raw_source'])
                    if entry['variant']=='baseline':
                        ids,mask=builder.sequence_batch([row],variant='baseline',excluded_groups=(),device=resolved)
                        source,graph=fixed_logits(model,(ids,mask),baseline=True)
                    else:source,graph=fixed_logits(model,_graph_inputs(model,[row],builder,views,encoded,resolved))
                    z=float((source+graph).item());score=float(torch.sigmoid(source+graph).item())
                    record=dict(sample_key=case['sample_key'],case=case['case'],model=name,label=case['label'],
                        logit=z,score=score,source_logit=float(source.item()),graph_logit=float(graph.item()),threshold=threshold,
                        prediction=int(score>=threshold),source_tokens=case['source_tokens'])
                    out.write(json.dumps(record)+'\n');out.flush();responses.append(record)
            del model,checkpoint;gc.collect()
            if resolved.type=='cuda':torch.cuda.empty_cache()
    return summarize(root,responses)


def summarize(root,responses=None):
    root=Path(root);responses=read_jsonl(root/'responses.jsonl') if responses is None else responses
    protocol=json.loads((root/'protocol.json').read_text()) if (root/'protocol.json').exists() else None
    if protocol:
        cases=read_jsonl(root/'cases.jsonl')
        expected={(m,c['sample_key'],c['case']) for m in protocol['models'] for c in cases}
        actual={(r['model'],r['sample_key'],r['case']) for r in responses}
        if actual!=expected or len(responses)!=len(expected):raise ValueError('incomplete or duplicate fixed responses')
    report={}
    for model in sorted({r['model'] for r in responses}):
        by_key=defaultdict(dict)
        for r in responses:
            if r['model']==model:by_key[r['sample_key']][r['case']]=r
        changes=[]
        for key,cases in by_key.items():
            if not all(k in cases for k in COMMENTS):continue
            original=cases['original'];risk=cases['risk_cue'];safe=cases['safe_cue']
            changes.append(dict(sample_key=key,original_correct=original['prediction']==original['label'],
                cue_logit_gap=risk['logit']-safe['logit'],cue_prediction_flip=risk['prediction']!=safe['prediction'],
                variants={k:dict(delta_logit=cases[k]['logit']-original['logit'],
                    prediction_flip=cases[k]['prediction']!=original['prediction'],
                    delta_graph_logit=cases[k]['graph_logit']-original['graph_logit']) for k in COMMENTS}))
        report[model]=dict(functions=len(changes),cue_flips=sum(x['cue_prediction_flip'] for x in changes),
            mean_cue_logit_gap=sum(x['cue_logit_gap'] for x in changes)/max(1,len(changes)),
            variants={k:dict(flips=sum(x['variants'][k]['prediction_flip'] for x in changes),
                corrected_errors=sum(x['variants'][k]['prediction_flip'] and not x['original_correct'] for x in changes),
                new_errors=sum(x['variants'][k]['prediction_flip'] and x['original_correct'] for x in changes),
                mean_absolute_logit_change=sum(abs(x['variants'][k]['delta_logit']) for x in changes)/max(1,len(changes))) for k in COMMENTS},
            per_function=changes)
        if protocol:
            entry=protocol['models'][model]
            saved={r['sample_key']:r for r in read_jsonl(Path(entry['directory'])/entry['variant']/'valid.predictions.jsonl')}
            error=max(abs(c['original']['score']-saved[k]['score']) for k,c in by_key.items())
            if error>1e-4 or any(c['original']['threshold']!=saved[k]['threshold'] for k,c in by_key.items()):
                raise ValueError('original checkpoint responses or thresholds not reproduced')
            if any(abs(v['delta_graph_logit'])>1e-6 for x in changes for v in x['variants'].values()):
                raise ValueError('graph branch changed under inert trailing comment')
            report[model]['original_reproduction_max_score_error']=error
    atomic_json(root/'summary.json',report)
    return report


def position_reference(output):
    """Train-only tabulated positional reference, not a new learned task or classifier."""
    from .cfg_metrics import metrics
    root=Path(output);counts=defaultdict(lambda:[0,0])
    def cell(row):
        distance=row['use_token']-row['condition_token']
        magnitude=abs(distance)
        bucket=0 if magnitude==0 else 1 if magnitude<=32 else 2 if magnitude<=128 else 3
        return (row['branch'],(distance>0)-(distance<0),bucket)
    for row in iter_jsonl('results/cfg_control_relations_seed42/train.control.jsonl'):
        if row['split']!='train':raise ValueError('position reference must be train-only')
        counts[cell(row)][0]+=row['label'];counts[cell(row)][1]+=1
    labels=[];scores=[];losses=defaultdict(list)
    for row in iter_jsonl('results/cfg_control_relations_seed42/valid.control.jsonl'):
        if row['split']!='valid':raise ValueError('expected fixed valid relations')
        positive,total=counts.get(cell(row),(0,0));score=(positive+1)/(total+2)
        labels.append(row['label']);scores.append(score)
        losses[row['sample_key']].append(-row['label']*math.log(score)-(1-row['label'])*math.log1p(-score))
    report=dict(features='branch, sign(use-condition), absolute token-distance bins 0 / 1..32 / 33..128 / >128',
                laplace_smoothing=1,threshold=.5,training='train query counts only; no Qwen or parameter updates',
                metrics=metrics(labels,scores,.5),function_mean_bce=sum(sum(v)/len(v) for v in losses.values())/len(losses),
                functions=len(losses),cells=[dict(key=list(k),positive=v[0],total=v[1]) for k,v in sorted(counts.items())])
    atomic_json(root/'position_reference.json',report)
    return report


def readout_intervention(output, *, device='cuda:0'):
    """Same fixed model, same common-prefix token set; only diagnostic readout mask changes."""
    import torch
    from .cfg_experiment import _base_module,_tokenizer,_graph_inputs
    from .cfg_network import build_model
    from .cfg_behavior import extract_bound_behavior,BoundVocabulary
    root=Path(output);protocol=json.loads((root/'protocol.json').read_text());entry=protocol['models']['structure_pool']
    path=Path(entry['directory'])/entry['variant']/'best.pt'
    if (root/'readout_intervention.json').exists():raise FileExistsError('readout intervention exists')
    if file_sha256(path)!=entry['checkpoint_sha256']:raise ValueError('fixed checkpoint changed')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False);config=checkpoint['model_config']
    base=_base_module();resolved=base._resolve_device(device);builder=_tokenizer(base,config)
    rows=read_records(config['dataset'],'primevul');originals={r['sample_key']:r for r in rows}
    if cohort_hash(rows)!=config['cohort_sha256'] or file_sha256(config['graphs'])!=config['graph_file_sha256']:raise ValueError('source/graph changed')
    keys=set(protocol['selected_comment_functions']);views={}
    if any(k not in originals or originals[k]['split']!='valid' for k in keys):raise ValueError('readout diagnosis is valid-only')
    bv=BoundVocabulary(json.loads((Path(config['behavior_dir'])/'vocabulary.json').read_text()))
    if digest(bv.values)!=config['behavior_vocabulary_sha256']:raise ValueError('vocabulary changed')
    for r in iter_jsonl(config['graphs'],description='Read intervention graphs'):
        if r['sample_key'] in keys:
            views[r['sample_key']]=abstract_cfg(r['graph'])
            views[r['sample_key']]['behavior']=bv.encode(extract_bound_behavior(r['graph']))
    vocab=AttributeVocabulary(checkpoint['vocabulary']);encoded={k:vocab.encode(v) for k,v in views.items()}
    model=build_model(base,config,vocab.sizes(),resolved,training=False)
    base.set_peft_model_state_dict(model.encoder,checkpoint['adapter_state']);model.task_modules.load_state_dict(checkpoint['task_state']);model.eval()
    captured={}
    h1=model.encoder.register_forward_hook(lambda module,inputs,out:captured.update(hidden=out.last_hidden_state))
    h2=model.task_modules['cfg_encoder'].register_forward_hook(lambda module,inputs,out:captured.update(graph=out))
    result=[]
    try:
        with torch.no_grad():
            for key in sorted(keys):
                original=originals[key];changed=dict(original,**{'raw_source':comment_variants(original,builder)['neutral']['raw_source']})
                before=builder.source_ids(original);after=builder.source_ids(changed)
                common=0
                for a,b in zip(before,after):
                    if a!=b:break
                    common+=1
                if common<len(before)-3:raise ValueError('tail intervention changes more than EOS/boundary tokens')
                trials=[];states=[]
                for row in (original,changed):
                    inputs=_graph_inputs(model,[row],builder,views,encoded,resolved)
                    source,graph_logit=model.branch_logits(*inputs)
                    hidden=captured['hidden'];g=captured['graph'];mask=inputs[1]
                    pool=model.task_modules['source_pool'];weights=pool.weights(hidden,mask,g)
                    tail_mass=float(weights[:,common:].sum())
                    pool_mask=torch.zeros_like(mask);pool_mask[:,:common]=1
                    typed=pool_mask.unsqueeze(-1).to(hidden.dtype)
                    mean=((hidden*typed).sum(1)/typed.sum(1)).float()
                    trimmed=pool(hidden,pool_mask,g,mean)
                    trim_logit=float((model.task_modules['classifier'](trimmed).squeeze(-1)+graph_logit).item())
                    trials.append(dict(full_logit=float((source+graph_logit).item()),trimmed_logit=trim_logit,tail_attention_mass=tail_mass))
                    states.append(hidden[:,:common].float().cpu())
                result.append(dict(sample_key=key,common_tokens=common,removed_original_tokens=len(before)-common,
                    prefix_hidden_max_difference=float((states[0]-states[1]).abs().max()),original=trials[0],neutral=trials[1],
                    full_delta=trials[1]['full_logit']-trials[0]['full_logit'],trimmed_delta=trials[1]['trimmed_logit']-trials[0]['trimmed_logit']))
    finally:h1.remove();h2.remove()
    report=dict(checkpoint_sha256=entry['checkpoint_sha256'],diagnostic_only=True,parameter_updates=0,
        common_prefix_policy='same token IDs, excluding changed boundary and EOS on both inputs',functions=len(result),
        mean_absolute_full_delta=sum(abs(r['full_delta']) for r in result)/len(result),
        mean_absolute_trimmed_delta=sum(abs(r['trimmed_delta']) for r in result)/len(result),per_function=result)
    atomic_json(root/'readout_intervention.json',report)
    return report
