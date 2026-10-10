"""CPU-only train/valid descriptive audit and pre-specified linear baselines.

No safety labels, model training on valid, Joern execution, or test evaluation.
Graph statistics describe saved Joern edges, not feasible paths or correctness.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sys
import warnings

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .cfg_alignment import _node_span, _source_coordinates
from .cfg_data import iter_jsonl, source_hash
from .cfg_metrics import metrics, select_threshold, paired_changes
from .syntax import parser_for

MODELS = {
    'Source': 'cfg_abc_seed42/baseline',
    'CLM_Source': 'cfg_clm_source_seed42/lm_pretrain_source',
    'P0_Source': 'cfg_origin_source_seed42/dep_pretrain_source',
    'C': 'cfg_abc_seed42/cfg',
    'P0_C': 'cfg_dep_cfg_windowfix_seed42/dep_pretrain_cfg',
    'Attributes': 'cfg_abc_seed42/attributes',
    'CLM_C': 'cfg_dep_cfg_seed42/lm_pretrain_cfg',
}
SIZE = ['visible_tokens', 'visible_chars', 'visible_lines']
SYNTAX = ['ast_nodes','statements','identifiers','unique_identifiers','parameters',
          'declarations','calls','assignments','updates','binary_ops','branches',
          'loops','returns','subscripts','field_access','pointer_expr','ast_depth']
GRAPH = ['nodes','AST_edges','CFG_edges','DDG_edges','CDG_edges','CFG_nodes',
         'CFG_components','CFG_cycle_rank','CFG_branch_nodes','CFG_max_out',
         'DDG_nodes','DDG_max_out','CDG_nodes','CDG_max_out',
         'CFG_edges_per_node','DDG_edges_per_node','CDG_edges_per_node']
# Fixed before examining new valid probe results. No parameter search.
SPECS = ['size', 'syntax', 'structure', 'lexical', 'lexical_structure',
         'project_size', 'project_structure', 'frozen_clm', 'frozen_clm_structure']


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def table(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def syntax_counts(source, language, end):
    raw=source.encode(); root=parser_for(language).parse(raw).root_node
    cutoff=len(source[:end].encode()); out=Counter(); names=set(); stack=[(root,0)]
    while stack:
        n,depth=stack.pop()
        # Count only complete visible syntax units, but traverse ancestors crossing cutoff.
        if n.start_byte>=cutoff: continue
        if n.end_byte<=cutoff:
            out['ast_depth']=max(out['ast_depth'],depth)
            out['ast_nodes']+=int(n.is_named)
            t=n.type
            out['statements']+=int(t.endswith('_statement'))
            out['identifiers']+=int(t=='identifier')
            if t=='identifier': names.add(raw[n.start_byte:n.end_byte])
            for k,types in {
                'parameters':('parameter_declaration','optional_parameter_declaration'),
                'declarations':('declaration',), 'calls':('call_expression',),
                'assignments':('assignment_expression','init_declarator'),
                'updates':('update_expression',), 'binary_ops':('binary_expression',),
                'branches':('if_statement','switch_statement','conditional_expression'),
                'loops':('for_statement','while_statement','do_statement','for_range_loop'),
                'returns':('return_statement',), 'subscripts':('subscript_expression',),
                'field_access':('field_expression',), 'pointer_expr':('pointer_expression',),
            }.items(): out[k]+=int(t in types)
        stack.extend((c,depth+1) for c in n.children)
    out['unique_identifiers']=len(names)
    return {k:int(out[k]) for k in SYNTAX}, bool(root.has_error)


def graph_counts(graph, accepted=None):
    ids={n['id'] for n in graph['nodes']} if accepted is None else set(accepted)
    edges=defaultdict(set)
    for e in graph['edges']:
        if e['source'] in ids and e['target'] in ids: edges[e['kind']].add((e['source'],e['target']))
    out={'nodes':len(ids)}
    for kind in ('AST','CFG','DDG','CDG'):
        es=edges[kind]; out[kind+'_edges']=len(es)
        if kind=='AST': continue
        used={v for e in es for v in e}; counts=Counter(a for a,b in es)
        out[kind+'_nodes']=len(used);out[kind+'_max_out']=max(counts.values(),default=0)
        out[kind+'_edges_per_node']=len(es)/max(1,len(used))
        if kind=='CFG':
            adj=defaultdict(set)
            for a,b in es: adj[a].add(b);adj[b].add(a)
            todo=set(used);components=0
            while todo:
                components+=1; stack=[todo.pop()]
                while stack:
                    for v in adj[stack.pop()]:
                        if v in todo:todo.remove(v);stack.append(v)
            out['CFG_components']=components
            # Undirected cycle-space rank of directed edge multigraph, NOT path count.
            out['CFG_cycle_rank']=len(es)-len(used)+components
            out['CFG_branch_nodes']=sum(c>1 for c in counts.values())
    return {k:out[k] for k in GRAPH}


def summary(y, p, threshold=.5):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    if not len(y): return {'samples':0}
    q=np.clip(p,1e-12,1-1e-12)
    return {**metrics(y.tolist(),p.tolist(),float(threshold)),
            'bce':float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q))),
            'probability_clip_count':int(np.sum(q!=p))}


def prepare(args, out):
    from transformers import AutoTokenizer
    rows=[]
    # Mixed physical files are streamed for identity filtering; non-train/valid bodies
    # are never retained, inspected, featurized, or passed to any analysis.
    for r in iter_jsonl(args.dataset):
        if r.get('dataset')=='primevul' and r.get('split') in ('train','valid'):
            assert r['schema_version']==9 and type(r['label']) is int and r['label'] in (0,1)
            rows.append(r)
    rows.sort(key=lambda r:r['sample_key']);by={r['sample_key']:r for r in rows}
    assert len(by)==len(rows)
    assert Counter(r['split'] for r in rows)=={'train':5886,'valid':741}
    cached={r['sample_key']:r for r in iter_jsonl(Path(args.operation_cache)/'operations.jsonl')}
    assert set(cached)==set(by)
    tok=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True)
    features={};texts={};quality={}
    for i,r in enumerate(rows):
        k=r['sample_key'];c=cached[k];src=r['raw_source']
        assert all(c[f]==r[f] for f in ('raw_source','label','split'))
        enc=tok(src,add_special_tokens=False,truncation=True,max_length=2048,return_offsets_mapping=True)
        end=max((b for a,b in enc['offset_mapping']),default=0)
        texts[k]=src[:end]
        syn,error=syntax_counts(src,r['resolved_language'],end)
        fullsyn,_=syntax_counts(src,r['resolved_language'],len(src)) if end<len(src) else (syn,error)
        features[k]={'tokens':c['tokens'],'chars':len(src),'lines':src.count('\n')+1,
            'truncated':int(c['tokens']>2048),'visible_tokens':len(enc['input_ids']),
            'visible_chars':end,'visible_lines':src[:end].count('\n')+1,
            'visible_fraction':min(1,2048/max(1,c['tokens'])), 'syntax_error':int(error),
            **syn,**{'full_'+a:v for a,v in fullsyn.items()}}
        for family in ('Bounds','Pointer','Lifetime'):
            ops=[o for o in c['operations'] if o['family']==family]
            features[k]['op_'+family]=len(ops)
            features[k]['visible_op_'+family]=sum(o['span'][1]<=end for o in ops)
        if (i+1)%1000==0:print('syntax',i+1,flush=True)
    seen=set()
    for g in iter_jsonl(args.graphs):
        k=g['sample_key']
        if k not in by:continue
        assert k not in seen;seen.add(k);r=by[k]
        assert g['source_sha256']==source_hash(r['raw_source'])
        assert g['label']==r['label'] and g['split']==r['split']
        coords=_source_coordinates(r['raw_source']); accepted=set();reasons=Counter()
        for n in g['graph']['nodes']:
            span,reason=_node_span(r['raw_source'],{**n['properties'],'code':n['code']},*coords)
            if reason:reasons[reason]+=1
            elif span[1]<=features[k]['visible_chars']:accepted.add(n['id'])
            else:reasons['outside_visible']+=1
        features[k].update({'g_'+a:v for a,v in graph_counts(g['graph'],accepted).items()})
        features[k].update({'full_g_'+a:v for a,v in graph_counts(g['graph']).items()})
        features[k]['graph_visible_node_fraction']=len(accepted)/max(1,len(g['graph']['nodes']))
        quality[k]={'alignment':dict(reasons),'cpg':g['cpg_quality']}
    assert seen==set(by),'missing CPG members; do not silently impute coverage'
    with (out/'features.jsonl').open('x') as f:
        for r in rows:
            k=r['sample_key'];c=cached[k]
            f.write(json.dumps({'sample_key':k,'split':r['split'],'label':r['label'],
                'project':c['project'],'commit_id':c['commit_id'],'file_name':c['file_name'],
                'source_sha256':source_hash(r['raw_source']),'features':features[k],
                'quality':quality[k]},ensure_ascii=False)+'\n')
    with (out/'visible_sources.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps({'sample_key':r['sample_key'],'source':texts[r['sample_key']]})+'\n')
    return rows


def feature_blocks(rows):
    def block(names): return np.log1p([[r['features'][f] for f in names] for r in rows])
    sizes=block(SIZE);syn=block(SYNTAX);graph=block(['g_'+f for f in GRAPH])
    # Rates supplement counts; no test/valid feature selection.
    den=np.maximum(1,np.array([r['features']['visible_tokens'] for r in rows]))[:,None]
    synrate=np.array([[r['features'][f] for f in SYNTAX] for r in rows])/den*100
    return {'size':sizes,'syntax':np.c_[sizes,syn,synrate],
            'structure':np.c_[sizes,syn,synrate,graph]}


def fit_predict(spec, ix, jx, blocks, texts, projects, frozen):
    matrices=[]; targets=[]
    stat = ('structure' if spec in ('structure','lexical_structure','project_structure','frozen_clm_structure')
            else 'syntax' if spec=='syntax' else 'size' if spec in ('size','project_size') else None)
    if stat:
        scale=StandardScaler().fit(blocks[stat][ix]);matrices.append(scale.transform(blocks[stat][ix]));targets.append(scale.transform(blocks[stat][jx]))
    if spec.startswith('lexical'):
        vec=TfidfVectorizer(token_pattern=r'(?u)\b[A-Za-z_]\w*\b|\d+|[^\w\s]',
                            lowercase=False,ngram_range=(1,2),min_df=3,max_features=30000,sublinear_tf=True)
        matrices.append(vec.fit_transform(texts[ix]));targets.append(vec.transform(texts[jx]))
    if spec.startswith('project'):
        enc=OneHotEncoder(handle_unknown='ignore')
        matrices.append(enc.fit_transform(projects[ix,None]));targets.append(enc.transform(projects[jx,None]))
    if spec.startswith('frozen'):
        scale=StandardScaler().fit(frozen[ix]);matrices.append(scale.transform(frozen[ix]));targets.append(scale.transform(frozen[jx]))
    x=sparse.hstack([sparse.csr_matrix(m) for m in matrices],format='csr')
    z=sparse.hstack([sparse.csr_matrix(m) for m in targets],format='csr')
    model=LogisticRegression(C=1.,solver='liblinear',max_iter=2000,random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(x,blocks['labels'][ix])
    return model.predict_proba(z)[:,1],int(x.shape[1]),int(model.n_iter_[0])


def distributions(rows, tr, va, out):
    y=np.array([r['label'] for r in rows]);projects=np.array([r['project'] for r in rows])
    names=list(rows[0]['features']);result=[]
    # Train-only residual association after log visible length + project adjustment.
    enc=OneHotEncoder(handle_unknown='ignore');proj=enc.fit_transform(projects[tr,None])
    size=np.log1p([[rows[i]['features'][k] for k in SIZE] for i in tr]);size=StandardScaler().fit_transform(size)
    controls=sparse.hstack([proj,size],format='csr')
    yl=Ridge(alpha=1.,solver='lsqr').fit(controls,y[tr]);yr=y[tr]-yl.predict(controls)
    for name in names:
        x=np.array([r['features'][name] for r in rows]);a=x[tr][y[tr]==0];b=x[tr][y[tr]==1]
        auc=roc_auc_score(y[tr],x[tr])
        v=np.log1p(x[tr]);reg=Ridge(alpha=1.,solver='lsqr').fit(controls,v);res=v-reg.predict(controls)
        corr=float(np.corrcoef(res,yr)[0,1]) if np.std(res)>1e-12 else 0.
        row={'feature':name,'train_rank_auc':float(auc),'rank_biserial':float(2*auc-1),
             'train_partial_pearson_log_size_project':corr}
        for split,ix in [('train',tr),('valid',va)]:
            for label in (0,1):
                vals=x[ix][y[ix]==label]
                for labelq,q in [('q25',25),('median',50),('q75',75)]: row[f'{split}_{label}_{labelq}']=float(np.percentile(vals,q))
                row[f'{split}_{label}_mean']=float(np.mean(vals))
                row[f'{split}_{label}_present']=float(np.mean(vals>0))
        result.append(row)
    table(out/'distributions.csv',result)
    projectrows=[]
    for p in sorted(set(projects)):
        d={'project':p}
        for split,ix in [('train',tr),('valid',va)]:
            ids=ix[projects[ix]==p];d[split+'_n']=len(ids);d[split+'_positive']=int(y[ids].sum());d[split+'_positive_rate']=float(y[ids].mean()) if len(ids) else None
        projectrows.append(d)
    table(out/'projects.csv',projectrows)
    return result,projectrows


def paired_auc_ci(y,a,b):
    rng=np.random.default_rng(42);diff=[]
    for _ in range(1000):
        ix=rng.integers(0,len(y),len(y))
        if len(set(y[ix]))==2:diff.append(roc_auc_score(y[ix],b[ix])-roc_auc_score(y[ix],a[ix]))
    return {'delta':float(roc_auc_score(y,b)-roc_auc_score(y,a)),
            'paired_iid_bootstrap_95_interval':np.percentile(diff,[2.5,97.5]).tolist(),
            'note':'conditional on selected models; not independent confirmation or project-cluster inference'}


def analyze(args,out):
    rows=list(iter_jsonl(out/'features.jsonl'));keys=[r['sample_key'] for r in rows];keyindex={k:i for i,k in enumerate(keys)}
    y=np.array([r['label'] for r in rows]);tr=np.array([i for i,r in enumerate(rows) if r['split']=='train']);va=np.array([i for i,r in enumerate(rows) if r['split']=='valid'])
    textmap={r['sample_key']:r['source'] for r in iter_jsonl(out/'visible_sources.jsonl')};texts=np.array([textmap[k] for k in keys]);projects=np.array([r['project'] for r in rows])
    dist,projectrows=distributions(rows,tr,va,out)
    leakage=json.loads((Path(args.operation_cache)/'coverage.json').read_text())['leakage']
    groups=defaultdict(list)
    for k,t in zip(keys,texts):groups[t].append(k)
    visible_dupes=[ks for ks in groups.values() if len(ks)>1]
    fullgroups=defaultdict(list)
    for r in rows:fullgroups[r['source_sha256']].append(r['sample_key'])
    full_dupes=[ks for ks in fullgroups.values() if len(ks)>1]
    def dupdetails(gs):
        return [{'keys':ks,'cross_split':len({rows[keyindex[k]]['split'] for k in ks})>1,
                 'conflicting_labels':len({rows[keyindex[k]]['label'] for k in ks})>1} for ks in gs]
    audit={'test_used':False,'counts':dict(Counter(f"{r['split']}:{r['label']}" for r in rows)),
           'project_counts':{s:len({r['project'] for r in rows if r['split']==s}) for s in ('train','valid')},
           'valid_unseen_project_functions':int(sum(p not in set(projects[tr]) for p in projects[va])),
           'full_exact_duplicates':dupdetails(full_dupes),'visible_exact_duplicates':dupdetails(visible_dupes),
           'previous_clone_screen':leakage,'quality':{}}
    for split,ix in [('train',tr),('valid',va)]:
        audit['quality'][split]={'cpg_status':dict(Counter(rows[i]['quality']['cpg'].get('status') for i in ix)),
          'preprocessing_applied':sum(bool(rows[i]['quality']['cpg'].get('preprocessing_applied')) for i in ix),
          'alignment_reasons':dict(sum((Counter(rows[i]['quality']['alignment']) for i in ix),Counter())),
          'empty_visible_cfg':sum(rows[i]['features']['g_CFG_edges']==0 for i in ix),
          'empty_full_cfg':sum(rows[i]['features']['full_g_CFG_edges']==0 for i in ix),
          'truncated':sum(rows[i]['features']['truncated'] for i in ix),
          'syntax_error':sum(rows[i]['features']['syntax_error'] for i in ix)}
    save(out/'data_quality.json',audit)
    blocks=feature_blocks(rows);blocks['labels']=y
    import torch
    cache=torch.load(args.frozen_cache,map_location='cpu',weights_only=False)
    fi={k:i for i,k in enumerate(cache['keys'])};assert set(fi)==set(keys)
    assert all(int(cache['labels'][fi[k]])==rows[i]['label'] and cache['splits'][fi[k]]==rows[i]['split'] for i,k in enumerate(keys))
    frozen=cache['pool'][[fi[k] for k in keys]].float().numpy();del cache
    # Same fixed regularization and CPU optimization for every family; no tuning.
    folds=list(StratifiedKFold(3,shuffle=True,random_state=42).split(tr,y[tr]))
    gfolds=list(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(tr,y[tr],projects[tr]))
    new={};probe={};predrows=[]
    for spec in SPECS:
        oof=np.zeros(len(tr));goof=np.zeros(len(tr))
        for a,b in folds:oof[b]=fit_predict(spec,tr[a],tr[b],blocks,texts,projects,frozen)[0]
        for a,b in gfolds:goof[b]=fit_predict(spec,tr[a],tr[b],blocks,texts,projects,frozen)[0]
        th,_=select_threshold(y[tr].tolist(),oof.tolist())
        p,dim,niter=fit_predict(spec,tr,va,blocks,texts,projects,frozen);new[spec]=p
        probe[spec]={'train_oof':summary(y[tr],oof,th),'train_project_disjoint_oof':summary(y[tr],goof,th),
                     'valid':summary(y[va],p,th),'valid_fixed_05':summary(y[va],p),
                     'features':dim,'iterations':niter,'threshold_source':'train stratified OOF'}
        for i,q in enumerate(p):predrows.append({'model':spec,'sample_key':keys[va[i]],'label':int(y[va[i]]),'score':float(q),'threshold':th})
        save(out/'probes.json',probe)
        print('finished probe',spec,flush=True)
    table(out/'probe_predictions.csv',predrows)
    # Historical predictions are only opened for valid, and exact identities checked.
    existing={};hist={};original={}
    for name,folder in MODELS.items():
        path=Path('results')/folder;pred=list(iter_jsonl(path/'valid.predictions.jsonl'));b={r['sample_key']:r for r in pred}
        assert len(b)==len(pred)==len(va) and set(b)=={keys[i] for i in va}
        ordered=[b[keys[i]] for i in va]
        for i,p in zip(va,ordered):
            assert p['label']==rows[i]['label'] and p['split']=='valid' and p['source_sha256']==rows[i]['source_sha256']
        thresholds={p['threshold'] for p in pred};assert len(thresholds)==1;th=thresholds.pop()
        p=np.array([r['score'] for r in ordered]);existing[name]=p;original[name]=ordered
        config=json.loads((path/'config.json').read_text()) if (path/'config.json').exists() else json.loads((path.parent/'config.json').read_text())
        hist[name]={'path':str(path),'valid':summary(y[va],p,th),'valid_fixed_05':summary(y[va],p),
                    'protocol':{k:config.get(k) for k in ('seed','epochs','source_max_length','learning_rate','gradient_accumulation','model_path','pretrain_run_dir')}}
    save(out/'historical_models.json',hist)
    comparisons={}
    for a,b in [('Source','CLM_Source'),('Source','P0_Source'),('Source','C'),('Source','P0_C'),('CLM_Source','P0_C'),('P0_Source','P0_C'),('C','P0_C')]:
        comparisons[a+'__'+b]={'selected':paired_changes(original[a],original[b]),
            'fixed_05':paired_changes(original[a],original[b],base_threshold=.5,candidate_threshold=.5),
            'auc':paired_auc_ci(y[va],existing[a],existing[b])}
    for a,b in [('size','syntax'),('syntax','structure'),('lexical','lexical_structure'),('project_size','project_structure'),('frozen_clm','frozen_clm_structure')]:
        pa,pb=new[a],new[b];ta=probe[a]['valid']['threshold'];tb=probe[b]['valid']['threshold']
        ca=(pa>=ta)==y[va];cb=(pb>=tb)==y[va]
        comparisons[a+'__'+b]={'corrected':int(sum(~ca&cb)),'introduced':int(sum(ca&~cb)),
                              'auc':paired_auc_ci(y[va],pa,pb)}
    save(out/'comparisons.json',comparisons)
    # Predeclared train-defined quartiles. No favorable subgroup selection.
    masks={'all':np.ones(len(va),dtype=bool)};cuts={}
    for f in ('tokens','full_g_CFG_cycle_rank','full_g_DDG_edges','full_g_CDG_edges','g_DDG_edges_per_node','g_CFG_branch_nodes'):
        vals=np.array([r['features'][f] for r in rows]);edges=np.unique(np.quantile(vals[tr],[.25,.5,.75]));cuts[f]=edges.tolist()
        bins=np.searchsorted(edges,vals[va],side='left')
        for k in range(len(edges)+1):masks[f+f':bin{k}']=bins==k
    for f in ('truncated','syntax_error'):
        vals=np.array([rows[i]['features'][f] for i in va]);masks[f+':0']=vals==0;masks[f+':1']=vals==1
    for fam in ('Bounds','Pointer','Lifetime'):
        vals=np.array([rows[i]['features']['op_'+fam] for i in va]);masks[fam+':absent']=vals==0;masks[fam+':present']=vals>0
    for p in sorted(set(projects[tr])):
        if sum(projects[tr]==p)>=100:masks['project:'+p]=projects[va]==p
    leakvalid=set()
    for g in leakage['alpha_normalized_cross_split_groups']+leakage['shared_commit_groups']:leakvalid.update(g)
    for pair in leakage['near_clone_candidates']:leakvalid.add(pair['valid'])
    masks['no_known_cross_split_clone_or_commit']=np.array([keys[i] not in leakvalid for i in va])
    allp={**existing,**new};ths={**{n:v['valid']['threshold'] for n,v in hist.items()},**{n:v['valid']['threshold'] for n,v in probe.items()}}
    sub=[]
    for g,m in masks.items():
        if not m.any():continue
        for n,p in allp.items():sub.append({'group':g,'model':n,'positives':int(y[va][m].sum()),**summary(y[va][m],p[m],ths[n])})
    table(out/'subgroups.csv',sub);save(out/'bin_definitions.json',cuts)
    records=[]
    for j,i in enumerate(va):
        r=rows[i];records.append({'sample_key':keys[i],'label':int(y[i]),'project':r['project'],
           **{n+'_score':float(p[j]) for n,p in allp.items()},
           **{n+'_correct':int((p[j]>=ths[n])==y[i]) for n,p in allp.items()},
           **r['features']})
    table(out/'valid_sample_analysis.csv',records)
    # Representative examples by fixed error patterns; choose closest-to-train-median size.
    sourcecache={r['sample_key']:r for r in iter_jsonl(Path(args.operation_cache)/'operations.jsonl')}
    patterns={
      'all_five_wrong':lambda r:all(not r[n+'_correct'] for n in ('Source','CLM_Source','P0_Source','C','P0_C')),
      'CLM_corrects_Source':lambda r:not r['Source_correct'] and r['CLM_Source_correct'],
      'P0C_corrects_CLM':lambda r:not r['CLM_Source_correct'] and r['P0_C_correct'],
      'P0C_breaks_CLM':lambda r:r['CLM_Source_correct'] and not r['P0_C_correct'],
    }
    examples=[];median=np.median([rows[i]['features']['tokens'] for i in tr])
    for pattern,fn in patterns.items():
        for label in (0,1):
            candidates=sorted([r for r in records if fn(r) and r['label']==label],key=lambda r:(abs(np.log1p(r['tokens'])-np.log1p(median)),r['sample_key']))
            if candidates:
                r=candidates[0];src=sourcecache[r['sample_key']]
                examples.append({'selection':pattern,'analysis':r,'file_name':src['file_name'],'commit_id':src['commit_id'],'raw_source':src['raw_source']})
    save(out/'examples.json',examples)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,f,label in zip(axes,('tokens','full_g_DDG_edges','full_g_CFG_cycle_rank'),('Qwen tokens','DDG edges','CFG cycle rank (not paths)')):
        for cls in (0,1):
            vals=np.log1p([rows[i]['features'][f] for i in tr if y[i]==cls]);ax.hist(vals,bins=30,density=True,alpha=.5,label=f'label {cls}')
        ax.set_xlabel('log(1 + '+label+')');ax.set_ylabel('train density');ax.legend()
    fig.tight_layout();fig.savefig(out/'train_distributions.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));names=list(hist);x=np.arange(len(names))
    for j,metric in enumerate(('auc','mcc','bce')):ax.bar(x+(j-1)*.25,[hist[n]['valid'][metric] for n in names],width=.25,label=metric)
    ax.set_xticks(x,names,rotation=15);ax.legend();ax.set_title('Saved valid predictions; original selected thresholds; BCE lower is better')
    fig.tight_layout();fig.savefig(out/'historical_valid.png',dpi=170);plt.close(fig)
    save(out/'complete.json',{'test_used':False,'functions':len(rows),'probes':list(probe),'models':list(hist),'seed':42})


def conditional_auc(labels, scores, groups):
    """Pair-weighted AUC restricted to positive/negative pairs in one stratum.

    This is descriptive conditioning, not a causal adjustment or a new metric
    for checkpoint selection. Single-class strata contribute no comparable pairs.
    """
    indices=defaultdict(list)
    for i,g in enumerate(groups):indices[g].append(i)
    numerator=0.;pairs=0;functions=0;eligible=0
    labels=np.asarray(labels);scores=np.asarray(scores)
    for ix in indices.values():
        n1=int(labels[ix].sum());n0=len(ix)-n1
        if not n0 or not n1:continue
        weight=n0*n1;pairs+=weight;functions+=len(ix);eligible+=1
        numerator+=weight*roc_auc_score(labels[ix],scores[ix])
    return {'auc':numerator/pairs if pairs else None,'comparable_pairs':pairs,
            'eligible_functions':functions,'eligible_strata':eligible}


def describe(args,out):
    """Reproducible descriptive follow-up; consumes saved scores, never refits."""
    from importlib.metadata import version
    rows=list(iter_jsonl(out/'features.jsonl'));by={r['sample_key']:r for r in rows}
    valid=list(csv.DictReader((out/'valid_sample_analysis.csv').open()))
    keys=[r['sample_key'] for r in valid];y=np.array([int(r['label']) for r in valid])
    cuts=json.loads((out/'bin_definitions.json').read_text())['tokens']
    bins=np.searchsorted(cuts,[float(r['tokens']) for r in valid],side='left')
    projects=[r['project'] for r in valid]
    hist=json.loads((out/'historical_models.json').read_text());probes=json.loads((out/'probes.json').read_text())
    models={**hist,**probes};scores={n:np.array([float(r[n+'_score']) for r in valid]) for n in models}
    conditions={'length_quartile':bins,'project_field':projects,
                'project_and_length':list(zip(projects,bins))}
    conditional=[];associations=[];calibration=[]
    for n,p in scores.items():
        for name,groups in conditions.items():conditional.append({'model':n,'condition':name,**conditional_auc(y,p,groups)})
        for f in ('tokens','full_g_CFG_cycle_rank','full_g_DDG_edges','full_g_CDG_edges'):
            x=np.array([float(r[f]) for r in valid])
            for label in (0,1):
                ix=y==label;associations.append({'model':n,'feature':f,'label':label,
                    'n':int(sum(ix)),'score_spearman':float(spearmanr(x[ix],p[ix]).statistic)})
        b=np.minimum(9,(p*10).astype(int));ece=0.
        for j in range(10):
            ix=b==j
            if ix.any():ece+=float(ix.mean()*abs(p[ix].mean()-y[ix].mean()))
        calibration.append({'model':n,'ece_10_equal_width_bins':ece,
           'confident_false_positive_score_ge_09':int(sum((y==0)&(p>=.9))),
           'confident_false_negative_score_le_01':int(sum((y==1)&(p<=.1))),
           'bce_from_probabilities':models[n]['valid']['bce'],
           'probability_clip_count':models[n]['valid']['probability_clip_count']})
    table(out/'conditional_metrics.csv',conditional);table(out/'score_associations.csv',associations)
    table(out/'calibration.csv',calibration)
    main=('Source','CLM_Source','P0_Source','C','P0_C')
    wrong=np.all([[not int(r[n+'_correct']) for r in valid] for n in main],axis=0)
    errors={'all_five_wrong_keys':[k for k,b in zip(keys,wrong) if b],
            'all_five_wrong_count':int(sum(wrong)),'all_five_wrong_positive':int(sum(y[wrong])),
            'all_five_wrong_truncated':sum(int(r['truncated']) for r,b in zip(valid,wrong) if b),
            'by_length':[], 'paired_by_length':[]}
    for j in range(4):
        ix=bins==j
        for n in main:
            p=scores[n];th=hist[n]['valid']['threshold'];m=summary(y[ix],p[ix],th)
            errors['by_length'].append({'bin':j,'model':n,'positives':int(sum(y[ix])),
                'negatives':int(sum(1-y[ix])),**m})
        for a,b in [('Source','CLM_Source'),('Source','P0_C'),('CLM_Source','P0_C')]:
            ca=np.array([bool(int(r[a+'_correct'])) for r in valid]);cb=np.array([bool(int(r[b+'_correct'])) for r in valid])
            errors['paired_by_length'].append({'bin':j,'comparison':a+'__'+b,
                'corrected':int(sum(ix&~ca&cb)),'introduced':int(sum(ix&ca&~cb))})
    save(out/'error_patterns.json',errors)

    # Inspect selection before CPG on the existing manifest; never rebuild/change it.
    selected={};missing=set();counts=Counter();lengths=defaultdict(list)
    for r in iter_jsonl('data/benchmark/manifest.jsonl'):
        if r['dataset']!='primevul' or r['split'] not in ('train','valid'):continue
        k=r['sample_key'];retained=k in by
        if retained:
            assert r['label']==by[k]['label'] and r['split']==by[k]['split']
            assert source_hash(r['function'])==by[k]['source_sha256']
        else:missing.add(k)
        selected[k]=r
        prefix=f"{r['split']}:{r['label']}"
        counts[prefix+':manifest']+=1;counts[prefix+':retained']+=int(retained)
        lengths[prefix+(':retained' if retained else ':excluded')].append(len(r['function']))
    assert set(by)<=set(selected)
    rejected=defaultdict(set)
    for r in iter_jsonl('data/function_dataset.errors.jsonl'):
        if r['sample_key'] in missing:
            rejected[(r['split'],r['label'],r['stage'],r['error_type'])].add(r['sample_key'])
    covered=set().union(*rejected.values()) if rejected else set()
    save(out/'selection_coverage.json',{'counts':dict(counts),'excluded_functions':len(missing),
         'excluded_with_error_record':len(covered),'excluded_without_error_record':sorted(missing-covered),
         'errors':[{'split':s,'label':y,'stage':stage,'error_type':err,'functions':len(ks)} for (s,y,stage,err),ks in sorted(rejected.items())],
         'source_chars':{k:{'n':len(v),'median':float(np.median(v))} for k,v in lengths.items()}})

    # Check the one visible-source collision against actual tokenizer IDs.
    from transformers import AutoTokenizer
    quality=json.loads((out/'data_quality.json').read_text());collisions=[]
    tokenizer=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True)
    for group in quality['visible_exact_duplicates']:
        if not group['conflicting_labels']:continue
        rr=[selected[k] for k in group['keys']]
        ids=[tokenizer(r['function'],add_special_tokens=False,truncation=True,max_length=2048)['input_ids'] for r in rr]
        collisions.append({'keys':group['keys'],'source_token_ids_equal':all(v==ids[0] for v in ids),
            'labels':[r['label'] for r in rr],'splits':[r['split'] for r in rr],
            'function_prefix':rr[0]['function'][:220]})
    tr=[r for r in rows if r['split']=='train'];ty=np.array([r['label'] for r in tr]);tp=np.array([r['project'] for r in tr])
    folds=[]
    for a,b in StratifiedGroupKFold(3,shuffle=True,random_state=42).split(np.arange(len(tr)),ty,tp):
        assert not set(tp[a])&set(tp[b])
        folds.append({'fit_n':len(a),'evaluation_n':len(b),'evaluation_positive':int(ty[b].sum()),
                      'fit_projects':len(set(tp[a])),'evaluation_projects':len(set(tp[b]))})
    save(out/'diagnostics.json',{'token_collisions':collisions,'project_folds':folds,
        'versions':{n:version(n) for n in ('numpy','scipy','scikit-learn','torch','transformers','tree-sitter')},
        'python':sys.version,'test_used':False,
        'interpretation':'Conditional AUC and score correlations are descriptive; project names need not isolate forks. Frozen L2 heads are not historical AdamW probes.'})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for n in main:
        group=[r for r in errors['by_length'] if r['model']==n]
        axes[0].plot(range(4),[r['fn']/r['positives'] for r in group],marker='o',label=n)
        axes[1].plot(range(4),[r['fp']/r['negatives'] for r in group],marker='o',label=n)
    for ax,label in zip(axes,('False negative rate','False positive rate')):
        ax.set_ylabel(label);ax.set_xlabel('Length bin (cutpoints from train)');ax.set_xticks(range(4));ax.set_ylim(0,1);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'length_errors.png',dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='data/function_dataset.jsonl');p.add_argument('--graphs',default='data/graphs/primevul_cfg.jsonl')
    p.add_argument('--operation-cache',default='results/operation_context_checked_seed42')
    p.add_argument('--model-path',default='/home/phy/models/Qwen2.5-Coder-7B-Instruct')
    p.add_argument('--frozen-cache',default='results/cfg_clm_probe_seed42/lm.features.pt')
    p.add_argument('--output-dir',required=True);p.add_argument('--phase',choices=('prepare','analyze','describe'),required=True)
    args=p.parse_args();out=Path(args.output_dir)
    if args.phase=='prepare':
        out.mkdir(parents=True,exist_ok=False)
        save(out/'protocol.json',{'command':sys.argv,'seed':42,'test_used':False,'specs':SPECS,'size':SIZE,'syntax':SYNTAX,'graph':GRAPH,
           'probe':'L2 logistic regression C=1 liblinear max_iter=2000; same unweighted BCE, no search',
           'validation':'3-fold train OOF thresholds, full-train fit, valid once; additional project-disjoint train OOF',
           'visibility':'Only first 2048 source tokens in probes; graph endpoints must have exact fully-visible source spans. Full CPG descriptive only.',
           'frozen':'Existing CLM phase-1 mean source pool; no new encoder computation',
           'units':'original function labels; no operation/safety labels'})
        prepare(args,out)
    elif args.phase=='analyze':
        if (out/'probes.json').exists():raise FileExistsError('refuse repeating/overwriting probe run')
        analyze(args,out)
    else:
        if not (out/'complete.json').exists():raise ValueError('complete the fixed probes before descriptive follow-up')
        describe(args,out)

if __name__=='__main__':main()
