"""Branch-conditioned, termination-insensitive control dependence on the saved CFG.

A node w depends on edge (c,s) iff w postdominates s but does not strictly
postdominate c. Only entry-reachable CFGs whose every reachable node can reach
METHOD_RETURN are admitted. Missing edges are never sampled as negative labels.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from .cfg_behavior import LocalGraph, extract_behavior
from .cfg_data import (abstract_cfg, atomic_json, cohort_hash, file_sha256, load_graphs,
                       read_records, read_jsonl, source_hash)

CONTROL_SCHEMA = 1
BRANCHES = ('true','false','case','default')


def postdominators(view, kinds):
    n=len(kinds); succ=[set() for _ in kinds]; pred=[set() for _ in kinds]
    for u,v in view['edges']:succ[u].add(v);pred[v].add(u)
    entries=[i for i,k in enumerate(kinds) if k=='METHOD']
    exits=[i for i,k in enumerate(kinds) if k=='METHOD_RETURN']
    if len(entries)!=1 or len(exits)!=1:return None,'entry_exit_ambiguous'
    def reachable(start,edges):
        seen={start};todo=[start]
        while todo:
            for v in edges[todo.pop()]:
                if v not in seen:seen.add(v);todo.append(v)
        return seen
    active=reachable(entries[0],succ);terminating=reachable(exits[0],pred)
    if not active<=terminating:return None,'reachable_nonterminating_or_incomplete'
    if any(not succ[v] and v!=exits[0] for v in active):return None,'unexpected_sink'
    if any(kinds[v]=='UNKNOWN' for v in active):return None,'unknown_cfg_node'
    all_bits=sum(1<<v for v in active);pd={v:all_bits for v in active};pd[exits[0]]=1<<exits[0]
    changed=True
    while changed:
        changed=False
        for v in active-{exits[0]}:
            mask=all_bits
            for w in succ[v]:mask &= pd[w]
            mask |= 1<<v
            if mask!=pd[v]:pd[v]=mask;changed=True
    return pd,None


def control_queries(record, graph, builder, *, limit=64):
    from .cfg_alignment import align_nodes
    if record['split'] not in ('train','valid'):
        raise ValueError('control development is train/valid only')
    view=abstract_cfg(graph);g=LocalGraph(graph);ids=view['node_ids']
    stats=Counter(functions=1,cfg_nodes=len(ids),cfg_edges=len(view['edges']))
    pd,reason=postdominators(view,[g.kind(n) for n in ids])
    if reason:
        stats['unknown_function_'+reason]+=1
        return [],stats
    stats['complete_cfg_functions']=1
    stats['unreachable_cfg_nodes']=len(ids)-len(pd)
    item=extract_behavior(graph)
    tokens=builder.tokenizer(record['raw_source'],add_special_tokens=False,truncation=True,
                             max_length=builder.source_max_length,return_offsets_mapping=True)
    pairs,counts=align_nodes(record['raw_source'],view['locations'],tokens['offset_mapping'],len(builder.source_prefix))
    stats.update({'alignment_'+k:v for k,v in counts.items()})
    ends={}
    for node,token in pairs:ends[node]=max(token,ends.get(node,token))
    # Ambiguous overlapping spans that collapse to one predictor input must not
    # silently create contradictory supervision.
    operations=[i for i,n in enumerate(ids) if i in pd and i in ends and g.kind(n) in ('CALL','RETURN')]
    groups=defaultdict(list)
    for (c,s),alts in zip(view['edges'],item['edges']):
        for alt in alts:
            branch=alt[0][0]
            if branch not in BRANCHES:continue
            case=[t for t in alt[1] if t.startswith('case:')]
            groups[(c,branch,tuple(case))].append(s)
    outgoing=defaultdict(set)
    for (c,_),alts in zip(view['edges'],item['edges']):
        outgoing[c].update(a[0][0] for a in alts)
    complete_conditions={c for c,branches in outgoing.items() if
        branches=={'true','false'} or ('default' in branches and 'case' in branches and branches<={'case','default'})}
    stats['unknown_incomplete_decisions']=sum(c not in complete_conditions for c,_,_ in groups)
    candidates={}; conflicts=set()
    for (condition,branch,case), successors in groups.items():
        if condition not in complete_conditions:continue
        stats['branch_queries_before_alignment']+=1
        if condition not in ends or condition not in pd:
            stats['condition_excluded_alignment']+=1;continue
        case_token=None
        if branch=='case':
            # The query names the case, not the target operation's real branch.
            labels=[s for s in successors if g.kind(ids[s])=='JUMP_TARGET']
            if len(set(labels))!=1 or labels[0] not in ends:
                stats['case_excluded_alignment']+=1;continue
            case_token=ends[labels[0]]
        labels_for_branch=[]
        for successor in successors:
            if successor not in pd:continue
            labels_for_branch.append(pd[successor] & ~(pd[condition] & ~(1<<condition)))
        if not labels_for_branch:continue
        # For a branch descriptor with more than one possible actual transition,
        # require agreement; neither conjunction nor a guessed edge is a label.
        for operation in operations:
            outcomes={int(bool(mask & (1<<operation))) for mask in labels_for_branch}
            if len(outcomes)!=1:
                stats['unknown_ambiguous_transition']+=1;continue
            label=outcomes.pop()
            query={'sample_key':record['sample_key'],'split':record['split'],'condition_token':ends[condition],'use_token':ends[operation],
                   'branch':BRANCHES.index(branch),'case_token':case_token,'label':label,
                   'condition_node':ids[condition],'use_node':ids[operation]}
            key=(query['condition_token'],query['use_token'],query['branch'],case_token)
            if key in candidates and candidates[key]['label']!=label:conflicts.add(key)
            candidates[key]=query
    for key in conflicts:candidates.pop(key,None)
    stats['unknown_input_conflicts']=len(conflicts)
    canonical=lambda q:(q['condition_token'],q['use_token'],q['branch'],q['case_token'] if q['case_token'] is not None else -1)
    positives=sorted((q for q in candidates.values() if q['label']==1),key=canonical)
    negatives=sorted((q for q in candidates.values() if q['label']==0),key=canonical)
    stats.update(candidate_positive=len(positives),candidate_negative=len(negatives))
    # Per-condition matched controls first, with fixed random tie-breaking; no
    # replication to fill the budget and no use of vulnerability labels.
    rng=random.Random('42/control/'+record['sample_key'])
    rng.shuffle(positives);rng.shuffle(negatives)
    selected=[];used=set()
    def key(q):return (q['condition_token'],q['use_token'],q['branch'],q['case_token'])
    by_condition=defaultdict(list)
    for q in negatives:by_condition[(q['condition_token'],q['branch'],q['case_token'])].append(q)
    for q in positives:
        if len(selected)>=limit:break
        selected.append(q);used.add(key(q))
        controls=by_condition[(q['condition_token'],q['branch'],q['case_token'])]
        if controls and len(selected)<limit:
            # Nearby source operations are harder controls than random nonedges.
            other=min(controls,key=lambda n:abs(n['use_token']-q['use_token']))
            controls.remove(other);selected.append(other);used.add(key(other))
    for q in positives+negatives:
        if len(selected)>=limit:break
        if key(q) not in used:selected.append(q);used.add(key(q))
    stats.update(selected_positive=sum(q['label'] for q in selected),
                 selected_negative=sum(1-q['label'] for q in selected),effective_functions=int(bool(selected)),
                 independent_condition_operations=len({(q['condition_node'],q['use_node']) for q in selected}))
    return selected,stats


def prepare_control(reference_run_dir, output_dir, tokenizer):
    from .model import InputBuilder
    reference=Path(reference_run_dir).resolve();root=Path(output_dir)
    if root.exists():raise FileExistsError(f'control output exists: {root}')
    config=json.loads((reference/'config.json').read_text())
    rows=read_records(config['dataset'],config['source_dataset'])
    if cohort_hash(rows)!=config['cohort_sha256'] or file_sha256(config['graphs'])!=config['graph_file_sha256']:
        raise ValueError('control inputs differ from C')
    if not getattr(tokenizer,'is_fast',False):raise ValueError('control targets require fast tokenizer offsets')
    selected=[r for r in rows if r['split'] in ('train','valid')]
    graphs=load_graphs(config['graphs'],rows)
    builder=InputBuilder(tokenizer,source_max_length=config['source_max_length'],context_max_length=384)
    root.mkdir(parents=True);stats={s:Counter() for s in ('train','valid')};examples={s:[] for s in stats}
    dependency_counts=Counter()
    with (root/'queries.jsonl').open('x') as out, (root/'dependency.valid.jsonl').open('x') as dep_out:
        for i,row in enumerate(selected):
            graph=graphs.pop(row['sample_key'])
            queries,counts=control_queries(row,graph,builder)
            if row['split']=='valid':
                from .cfg_dependency import function_relations
                relations,dep_counts=function_relations(row,graph,builder.tokenizer,
                    source_max_length=builder.source_max_length,prefix_tokens=len(builder.source_prefix),for_valid=True)
                dependency_counts.update(dep_counts)
                dep_out.write(json.dumps({'sample_key':row['sample_key'],'split':'valid',
                    'source_sha256':source_hash(row['raw_source']),'relations':relations})+'\n')
            stats[row['split']].update(counts)
            record={'sample_key':row['sample_key'],'split':row['split'],'source_sha256':source_hash(row['raw_source']),
                    'queries':queries,'counts':dict(counts)}
            out.write(json.dumps(record)+'\n')
            if queries and len(examples[row['split']])<2:
                examples[row['split']].append(dict(record,source=row['raw_source']))
            if (i+1)%200==0:print(f'control_prepare={i+1}/{len(selected)}',flush=True)
    audit={'schema':CONTROL_SCHEMA,'c_config':config,'reference_run_dir':str(reference),
           'queries_sha256':file_sha256(root/'queries.jsonl'),'seed':42,'limit':64,
           'definition':'edge_postdominance_termination_insensitive','splits':stats,'examples':examples,
           'dependency_valid_sha256':file_sha256(root/'dependency.valid.jsonl'),
           'dependency_valid_counts':dict(dependency_counts)}
    atomic_json(root/'audit.json',audit)
    return audit


def load_control(directory, rows, c_config):
    root=Path(directory);audit=json.loads((root/'audit.json').read_text())
    if audit['schema']!=CONTROL_SCHEMA or audit['c_config']!=c_config or audit['queries_sha256']!=file_sha256(root/'queries.jsonl'):
        raise ValueError('control cache identity/schema mismatch')
    selected={r['sample_key']:r for r in rows if r['split'] in ('train','valid')};seen=set();result={}
    for item in read_jsonl(root/'queries.jsonl'):
        key=item['sample_key']
        if key not in selected:continue
        row=selected[key]
        if key in seen or item['split']!=row['split'] or item['source_sha256']!=source_hash(row['raw_source']):
            raise ValueError('control source/split mismatch')
        seen.add(key);queries=item['queries'];inputs=set()
        for q in queries:
            inp=(q['condition_token'],q['use_token'],q['branch'],q['case_token'])
            if q['label'] not in (0,1) or q['branch'] not in range(4) or inp in inputs:
                raise ValueError('invalid/duplicate control query')
            inputs.add(inp)
        if queries:result[key]=queries
    if seen!=set(selected):raise ValueError('incomplete control cache')
    return result,audit


def evaluate_control(pretrain_dir, output_dir, *, device='auto', batch_size=1, base=None):
    """Fixed stage-1 evaluation of both tasks, sharing one source forward per batch."""
    if batch_size < 1:raise ValueError('evaluation batch_size must be positive')
    import math
    import torch
    from .cfg_experiment import _base_module, _tokenizer
    from .cfg_dependency import (DirectedRelationHead, ControlRelationHead, relation_function_loss,
                                 _checked_relation_rows)
    from .cfg_metrics import metrics
    root=Path(output_dir);stage=Path(pretrain_dir).resolve()
    if root.exists():raise FileExistsError(f'fixed relation output exists: {root}')
    config=json.loads((stage/'config.json').read_text());c=config['c_config']
    if config.get('control_schema')!=CONTROL_SCHEMA:raise ValueError('control checkpoint schema mismatch')
    reference=Path(config['reference_run_dir'])
    if json.loads((reference/'config.json').read_text())!=c:raise ValueError('C reference changed')
    rows=read_records(c['dataset'],c['source_dataset']);rows_tv=[r for r in rows if r['split'] in ('train','valid')]
    if cohort_hash(rows)!=c['cohort_sha256'] or file_sha256(c['graphs'])!=c['graph_file_sha256']:
        raise ValueError('fixed evaluation cohort/graph changed')
    control,audit=load_control(config['control_dir'],rows_tv,c)
    if audit['queries_sha256']!=config['control_queries_sha256']:raise ValueError('control targets changed')
    path=stage/'control_pretrain/last.pt'
    complete=json.loads((path.parent/'complete.json').read_text())
    if complete['mode']!='control_pretrain' or complete['checkpoint_sha256']!=file_sha256(path):
        raise ValueError('fixed checkpoint identity mismatch')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint['pretrain_config']!=config or checkpoint['mode']!='control_pretrain':
        raise ValueError('fixed model/config mismatch')
    base=_base_module() if base is None else base;builder=_tokenizer(base,c)
    train=[r for r in rows_tv if r['split']=='train']
    dependency,dep_audit=_checked_relation_rows(config['supervision_dir'],train,builder)
    if dep_audit['relations_sha256']!=config['relations_sha256']:raise ValueError('dependency targets changed')
    valid_path=Path(config['control_dir'])/'dependency.valid.jsonl'
    if file_sha256(valid_path)!=audit['dependency_valid_sha256']:
        raise ValueError('cached valid dependency targets changed')
    expected={r['sample_key']:r for r in rows_tv if r['split']=='valid'};seen=set()
    for item in read_jsonl(valid_path):
        key=item['sample_key'];row=expected.get(key)
        if row is None or key in seen or item['split']!='valid' or item['source_sha256']!=source_hash(row['raw_source']):
            raise ValueError('valid dependency source/split mismatch')
        seen.add(key)
        if item['relations']:dependency[key]=item['relations']
    if seen!=set(expected):raise ValueError('incomplete cached valid dependencies')
    valid_dep_counts=audit['dependency_valid_counts']
    resolved=base._resolve_device(device)
    encoder,hidden_size=base._build_lora_encoder(c['model_path'],device=resolved,lora_r=c['lora_r'],
        lora_alpha=c['lora_alpha'],lora_dropout=c['lora_dropout'],target_modules=('q_proj','k_proj','v_proj','o_proj'),
        gradient_checkpointing=False)
    encoder.to(resolved);base.set_peft_model_state_dict(encoder,checkpoint['adapter_state']);encoder.eval()
    heads={'dependency':DirectedRelationHead(hidden_size,config['relation_rank']).to(resolved),
           'control':ControlRelationHead(hidden_size,config['relation_rank']).to(resolved)}
    heads['dependency'].load_state_dict(checkpoint['relation_head_state'])
    heads['control'].load_state_dict(checkpoint['control_head_state'])
    for head in heads.values():head.eval()
    targets={'dependency':dependency,'control':control}
    epsilon=1e-6;priors={}
    for task,groups in targets.items():
        values=[sum(q['label'] for q in groups[r['sample_key']])/len(groups[r['sample_key']])
                for r in train if groups.get(r['sample_key'])]
        if not values:raise ValueError('train-only reference has no effective functions')
        priors[task]=max(epsilon,min(1-epsilon,sum(values)/len(values)))
    root.mkdir(parents=True);report={'fixed_model':True,'checkpoint_sha256':file_sha256(path),
        'threshold':0.5,'constant_epsilon':epsilon,'train_only_priors':priors,
        'valid_dependency_audit':dict(valid_dep_counts),'splits':{}}
    for split in ('train','valid'):
        records=[r for r in rows_tv if r['split']==split]
        losses={t:[] for t in heads};constant={t:[] for t in heads};predictions={t:[] for t in heads}
        with torch.no_grad():
            for start in range(0,len(records),batch_size):
                batch=records[start:start+batch_size]
                ids,mask=builder.sequence_batch(batch,variant='baseline',excluded_groups=(),device=resolved)
                hidden=encoder(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
                for i,row in enumerate(batch):
                    for task,head in heads.items():
                        queries=targets[task].get(row['sample_key'],[])
                        if not queries:continue
                        loss,labels,logits=relation_function_loss(hidden[i:i+1],[queries],head)
                        losses[task].append(float(loss))
                        p=priors[task]
                        constant[task].append(sum(-y*math.log(p)-(1-y)*math.log1p(-p) for y in labels)/len(labels))
                        scores=torch.sigmoid(torch.tensor(logits)).tolist()
                        predictions[task].extend(dict(q,sample_key=row['sample_key'],split=split,logit=z,score=p)
                                                 for q,z,p in zip(queries,logits,scores))
        report['splits'][split]={}
        for task,items in predictions.items():
            labels=[q['label'] for q in items];scores=[q['score'] for q in items]
            with (root/f'{split}.{task}.jsonl').open('x') as out:
                for item in items:out.write(json.dumps(item)+'\n')
            report['splits'][split][task]={'total_functions':len(records),'effective_functions':len(losses[task]),
                'queries':len(items),'positive':sum(labels),'negative':len(labels)-sum(labels),
                'function_mean_bce':sum(losses[task])/len(losses[task]) if losses[task] else None,
                'constant_function_mean_bce':sum(constant[task])/len(constant[task]) if constant[task] else None,
                'pooled_metrics':metrics(labels,scores,.5) if labels else None,
                'constant_pooled_metrics':metrics(labels,[priors[task]]*len(labels),.5) if labels else None}
    atomic_json(root/'metrics.json',report)
    return report
