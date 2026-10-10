"""Fixed K=4 CodeBERT head comparison; train/valid only, shared formal trainer."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from . import model as core
from .cascade import read_rows, write_json, changes
from .cfg_data import identity, cohort_hash
from .cfg_metrics import metrics
from .defer import jsonl

BASE=Path('results/codebert_source_seed42/native512')
RAW=Path('/home/PublicData/PHY-data/vul_detect/data/PrimeVul_v0.1')
HEADS={'A':'linear','B':'mlp4','C':'prototype4'}


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    # Historical Qwen-token counts are length covariates, never CodeBERT inputs.
    features=read_rows('results/function_statistics_seed42/features.jsonl')
    fi={r['sample_key']:r for r in features}
    rows=[]
    for line in Path('data/function_dataset.jsonl').open():
        r=json.loads(line)
        if r.get('dataset')=='primevul' and r.get('split') in ('train','valid'):
            if r['sample_key'] not in fi:raise ValueError('unexpected cohort member')
            if identity(r)['source_sha256']!=fi[r['sample_key']]['source_sha256']:raise ValueError('source mismatch')
            rows.append(r)
    conf=json.loads(Path('results/codebert_source_seed42/config.json').read_text())
    for split,n in [('train',5886),('valid',741)]:
        part=[r for r in rows if r['split']==split]
        assert len(part)==n and cohort_hash(part)==conf[split+'_cohort_sha256']
    original={}
    for split in ('train','valid'):
        for line in (RAW/f'primevul_{split}.jsonl').open():
            r=json.loads(line);k=f"primevul:{r['idx']}"
            if k in fi:
                if r['target']!=fi[k]['label'] or hashlib.sha256(r['func'].encode()).hexdigest()!=fi[k]['source_sha256']:raise ValueError('official mismatch')
                cwes=sorted(set(c for c in (r.get('cwe') or []) if isinstance(c,str) and c.startswith('CWE-') and c[4:].isdigit()))
                original[k]=dict(cwes=cwes,raw_cwe=r.get('cwe'),project=r['project'],commit=r['commit_id'])
    assert set(original)==set(fi)
    metadata=[dict(identity(r),**original[r['sample_key']],tokens=fi[r['sample_key']]['features']['tokens']) for r in rows]
    jsonl(out/'records.jsonl',rows);jsonl(out/'members.jsonl',metadata)
    cwe={}
    for split in ('train','valid'):
        pos=[r for r in metadata if r['split']==split and r['label']==1]
        counts={c:{'functions':sum(c in r['cwes'] for r in pos),'projects':len({r['project'] for r in pos if c in r['cwes']})} for c in sorted({c for r in pos for c in r['cwes']})}
        cwe[split]=dict(positive=len(pos),covered=sum(bool(r['cwes']) for r in pos),multiple=sum(len(r['cwes'])>1 for r in pos),missing=sum(not r['cwes'] for r in pos),categories=counts)
    write_json(out/'config.json',dict(seed=42,heads=HEADS,k=4,temperature=1,aggregation='logsumexp - log(K)',epochs=3,training=conf,metadata='CWE associated with source repair/CVE record, not verified function mechanism; no inferred hierarchy or negative CWE labels',test_used=False,clustering='K=4 train-positive KMeans on L2-normalized selected-baseline representations; seeds42/43 stability only, no K selection',capacity='linear d+1; MLP d*4+4+4+1; prototype d*4+4',initialization='original encoder and training RNG preserved; prototype centered default random weights plus common baseline weight, common bias; MLP standard init',bootstrap='1000 paired function resamples seed42, selected thresholds held fixed; descriptive development valid intervals'))
    write_json(out/'cwe.json',cwe)
    print(json.dumps({s:{k:v for k,v in cwe[s].items() if k!='categories'} for s in cwe},indent=2))


def extract(out,name,device):
    path=BASE/'best.pt' if name=='baseline' else out/name/'best.pt'
    target=out/(name+'.representations.npz')
    if target.exists():raise FileExistsError(target)
    cp=torch.load(path,map_location='cpu',weights_only=False)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(cp['model_path']);builder=core.InputBuilder(tok,source_max_length=512,context_max_length=cp['context_max_length'])
    model=core._load_model(cp,device=torch.device(device));rows=read_rows(out/'records.jsonl')
    vectors=[];logits=[];parts=[]
    with torch.no_grad():
        for r in rows:
            h=model.encode(*builder.codebert_batch([r],device=device))
            vectors.append(h.cpu().numpy()[0]);head=model.task_modules['classifier'];logits.append(float(head(h)[0,0]))
            if isinstance(head,core.MultiPrototypeHead):parts.append(head.prototypes(h).cpu().numpy()[0])
    np.savez_compressed(target,vectors=np.asarray(vectors),logits=np.asarray(logits),prototype_scores=np.asarray(parts))
    print(name,'extracted',len(rows),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()


def audit(out):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
    from sklearn.metrics import silhouette_score,adjusted_rand_score,adjusted_mutual_info_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score,StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    rows=read_rows(out/'members.jsonl');saved=np.load(out/'baseline.representations.npz');x=normalize(saved['vectors']);ix=np.array([r['split']=='train' and r['label']==1 for r in rows]);pos=[r for r in rows if r['split']=='train' and r['label']==1]
    first=KMeans(n_clusters=4,n_init=10,random_state=42).fit(x[ix]);second=KMeans(n_clusters=4,n_init=10,random_state=43).fit(x[ix]);z=first.labels_;allz=first.predict(x)
    lens=np.log1p([r['tokens'] for r in pos]);cuts=np.quantile(lens,[.25,.5,.75]);lengthbins=np.digitize(lens,cuts)
    p=1/(1+np.exp(-saved['logits']));threshold=.15
    report=dict(silhouette=float(silhouette_score(x[ix],z,sample_size=min(1500,sum(ix)),random_state=42)),seed_stability_ari=float(adjusted_rand_score(z,second.labels_)),project_ami=float(adjusted_mutual_info_score([r['project'] for r in pos],z)),length_quartile_ami=float(adjusted_mutual_info_score(lengthbins,z)),cwe_signature_ami=float(adjusted_mutual_info_score(['+'.join(r['cwes']) or 'Unknown' for r in pos],z)),length_only_cluster_cv_accuracy=float(cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),lens[:,None],z,cv=StratifiedKFold(5,shuffle=True,random_state=42)).mean()),majority_cluster_accuracy=float(np.bincount(z).max()/len(z)),clusters=[])
    for split in ('train','valid'):
        for c in range(4):
            ids=[i for i,r in enumerate(rows) if r['split']==split and r['label']==1 and allz[i]==c]
            report['clusters'].append(dict(split=split,cluster=c,n=len(ids),recall=float(np.mean(p[ids]>=threshold)) if ids else None,median_tokens=float(np.median([rows[i]['tokens'] for i in ids])) if ids else None,projects=dict(Counter(rows[i]['project'] for i in ids).most_common(5))))
    write_json(out/'heterogeneity.json',report)
    jsonl(out/'cluster_assignments.jsonl',[dict(sample_key=r['sample_key'],split=r['split'],cluster=int(allz[i])) for i,r in enumerate(rows)])
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    coords=PCA(n_components=2,random_state=42).fit_transform(x[ix]);fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].scatter(*coords.T,c=z,s=5,cmap='tab10');sc=axes[1].scatter(*coords.T,c=lens,s=5);fig.colorbar(sc,ax=axes[1],label='log(1+tokens)')
    axes[0].set_title('Train positives: fixed K=4');axes[1].set_title('Same representations, source length');fig.tight_layout();fig.savefig(out/'heterogeneity.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))


def train(out,name,device):
    target=out/name;target.mkdir(exist_ok=False)
    cp=torch.load(BASE/'best.pt',map_location='cpu',weights_only=False)
    rows=read_rows(out/'records.jsonl')
    core.train_model(None,target/'best.pt',records=rows,variant='codebert_source',model_path=cp['model_path'],source_max_length=512,context_max_length=cp['context_max_length'],lora_r=cp['lora_r'],lora_alpha=cp['lora_alpha'],lora_dropout=cp['lora_dropout'],seed=42,device=device,log_every=100,classifier_head=HEADS[name],**cp['training_config'])
    gc.collect();torch.cuda.empty_cache()
    result=core.evaluate_model(out/'records.jsonl',target/'best.pt',split='valid',device=device,prediction_path=target/'valid.predictions.jsonl')
    write_json(target/'valid.metrics.json',result)


def heterogeneity_controls(out):
    """Training-only controls: do clusters add more than the existing score?"""
    from sklearn.preprocessing import normalize, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.decomposition import PCA
    rows=read_rows(out/'members.jsonl');cl=read_rows(out/'cluster_assignments.jsonl')
    data=np.load(out/'baseline.representations.npz')
    ix=np.array([r['split']=='train' and r['label']==1 for r in rows])
    x=normalize(data['vectors'][ix]);z=np.array([r['cluster'] for r in cl])[ix]
    score=data['logits'][ix,None]
    accuracy=float(cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)),score,z,cv=StratifiedKFold(5,shuffle=True,random_state=42)).mean())
    rng=np.random.default_rng(42);stability=[]
    for _ in range(5):
        indices=rng.choice(len(x),len(x),replace=True)
        fitted=KMeans(n_clusters=4,n_init=10,random_state=42).fit(x[indices])
        stability.append(float(adjusted_rand_score(z,fitted.predict(x))))
    return dict(logit_only_cluster_cv_accuracy=accuracy,bootstrap_cluster_ari=stability,
        normalized_pca_variance=PCA(3,random_state=42).fit(x).explained_variance_ratio_.tolist(),
        independent_positive_source_hashes={split:len({r['source_sha256'] for r in rows if r['split']==split and r['label']==1}) for split in ('train','valid')},
        project_definition='official project field; not a newly reconstructed repository identity',
        interpretation='Already binary-supervised representations; score recoverability and stable clustering do not establish distinct vulnerability types.')


def analyze(out):
    from sklearn.metrics import roc_auc_score,matthews_corrcoef,adjusted_mutual_info_score
    rows=read_rows(out/'members.jsonl');valid=[r for r in rows if r['split']=='valid'];y=np.array([r['label'] for r in valid]);projects={r['project'] for r in rows if r['split']=='train'}
    cuts=np.quantile([r['tokens'] for r in rows if r['split']=='train'],[.25,.5,.75])
    pred={name:read_rows(out/name/'valid.predictions.jsonl') for name in HEADS};pred['original']=read_rows(BASE/'valid.predictions.jsonl')
    result={'models':{},'paired_bootstrap':{},'prototype_usage':{},'heterogeneity_controls':heterogeneity_controls(out)}
    for name,rs in pred.items():
        for a,b in zip(rs,valid):
            assert all(a[k]==b[k] for k in ('sample_key','source_sha256','label'))
        assert len(rs)==741
        p=np.array([r['score'] for r in rs]);t=rs[0]['threshold'];m=metrics(y.tolist(),p.tolist(),threshold=t)
        bce=float(np.mean(np.logaddexp(0,[r['logit'] for r in rs])-y*np.array([r['logit'] for r in rs])))
        groups={'short_positive':[i for i,r in enumerate(valid) if r['label']==1 and r['tokens']<=cuts[0]],'long_negative':[i for i,r in enumerate(valid) if r['label']==0 and r['tokens']>cuts[2]],'seen_project':[i for i,r in enumerate(valid) if r['project'] in projects],'unseen_project':[i for i,r in enumerate(valid) if r['project'] not in projects]}
        for j in range(4):groups[f'length_quartile_{j}']=[i for i,r in enumerate(valid) if np.digitize(r['tokens'],cuts)==j]
        for c in sorted({c for r in valid if r['label']==1 for c in r['cwes']}):
            ids=[i for i,r in enumerate(valid) if r['label']==1 and c in r['cwes']]
            if len(ids)>=10:groups[c]=ids
        checkpoint=torch.load((BASE if name=='original' else out/name)/'best.pt',map_location='cpu',weights_only=False)
        result['models'][name]=dict(selected_epoch=checkpoint['selected_epoch'],head_parameters=sum(v.numel() for k,v in checkpoint['task_state'].items() if k.startswith('classifier.')),trainable_parameters=checkpoint['trainable_parameters'],selected=m,bce=bce,fixed_05=metrics(y.tolist(),p.tolist()),fixed_original_015=metrics(y.tolist(),p.tolist(),threshold=.15),versus_original=changes(pred['original'],rs),subgroups={k:metrics(y[ix].tolist(),p[ix].tolist(),threshold=t) for k,ix in groups.items() if ix})
    for base in ('A','B'):
        a=pred[base];c=pred['C'];pa=np.array([r['score'] for r in a]);pc=np.array([r['score'] for r in c]);da=np.array([r['prediction'] for r in a]);dc=np.array([r['prediction'] for r in c]);rng=np.random.default_rng(42);diff=[]
        for _ in range(1000):
            ix=rng.integers(0,len(y),len(y));diff.append([roc_auc_score(y[ix],pc[ix])-roc_auc_score(y[ix],pa[ix]),matthews_corrcoef(y[ix],dc[ix])-matthews_corrcoef(y[ix],da[ix])])
        result['paired_bootstrap']['C_vs_'+base]=dict(changes=changes(a,c),auc_delta=float(roc_auc_score(y,pc)-roc_auc_score(y,pa)),mcc_delta=float(matthews_corrcoef(y,dc)-matthews_corrcoef(y,da)),ci95=np.quantile(diff,[.025,.975],axis=0).tolist(),columns=['auc','mcc'],limitation='function bootstrap, no project dependence correction; repeated-development valid and selection uncertainty not covered')
    saved=np.load(out/'C.representations.npz');s=saved['prototype_scores'];w=np.exp(s-s.max(1,keepdims=True));w/=w.sum(1,keepdims=True);z=w.argmax(1)
    cp=torch.load(out/'C'/'best.pt',map_location='cpu',weights_only=False);weights=cp['task_state']['classifier.prototypes.weight'].numpy();norm=weights/np.linalg.norm(weights,axis=1,keepdims=True)
    result['prototype_usage']['weight_cosine']= (norm@norm.T).tolist()
    result['prototype_usage']['score_correlation']=np.corrcoef(s.T).tolist()
    vi=np.array([i for i,r in enumerate(rows) if r['split']=='valid'])
    np.testing.assert_allclose(saved['logits'][vi],[r['logit'] for r in pred['C']],atol=1e-6,rtol=0)
    mean_logits=s[vi].mean(1);mean_p=torch.sigmoid(torch.from_numpy(mean_logits)).tolist()
    gap=saved['logits'][vi]-mean_logits
    result['prototype_usage']['mean_head_intervention']=dict(
        description='Fixed C encoder and threshold; replace logmeanexp by arithmetic mean, no training or threshold selection',
        metrics=metrics(y.tolist(),mean_p,threshold=pred['C'][0]['threshold']),
        bce=float(np.mean(np.logaddexp(0,mean_logits)-y*mean_logits)),
        logit_gap_mean=float(gap.mean()),logit_gap_std=float(gap.std()),logit_gap_max=float(gap.max()))
    for split in ('train','valid'):
        for label in (0,1):
            ids=np.array([i for i,r in enumerate(rows) if r['split']==split and r['label']==label]);subset=[rows[i] for i in ids]
            result['prototype_usage'][f'{split}_{label}']=dict(n=len(ids),winner_counts=np.bincount(z[ids],minlength=4).tolist(),mean_responsibilities=w[ids].mean(0).tolist(),mean_entropy=float(-(w[ids]*np.log(w[ids].clip(1e-15))).sum(1).mean()),project_ami=float(adjusted_mutual_info_score([r['project'] for r in subset],z[ids])),length_ami=float(adjusted_mutual_info_score(np.digitize([r['tokens'] for r in subset],cuts),z[ids])),cwe_ami=float(adjusted_mutual_info_score(['+'.join(r['cwes']) or 'Unknown' for r in subset],z[ids])))
    jsonl(out/'prototype_assignments.jsonl',[dict(sample_key=r['sample_key'],split=r['split'],label=r['label'],winner=int(z[i]),responsibilities=w[i].tolist()) for i,r in enumerate(rows)])
    write_json(out/'comparison.json',result)
    print(json.dumps({k:{a:v[a] for a in ['selected','bce']} for k,v in result['models'].items()},indent=2));print(json.dumps(result['paired_bootstrap'],indent=2));print(json.dumps(result['prototype_usage'],indent=2))


def plot(out):
    import csv
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads((out/'comparison.json').read_text())
    fields=['model','selected_epoch','head_parameters','threshold','accuracy','precision','recall','f1','mcc','auc','bce','tp','fp','tn','fn']
    f=io.StringIO()
    writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for name,v in r['models'].items():
        writer.writerow(dict(model=name,selected_epoch=v['selected_epoch'],head_parameters=v['head_parameters'],bce=v['bce'],**{k:v['selected'][k] for k in fields[3:] if k!='bce'}))
    path=out/'metrics.csv'
    if path.exists():
        if path.read_text()!=f.getvalue():raise ValueError('saved metric summary differs from results')
    else:
        with path.open('x') as handle:handle.write(f.getvalue())
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    names=['A','B','C'];positions=np.arange(3)
    for j,metric in enumerate(('mcc','auc')):
        axes[0].bar(positions+j*.35,[r['models'][n]['selected'][metric] for n in names],.35,label=metric.upper())
    axes[0].set_xticks(positions+.175,names);axes[0].set_title('Development valid, selected checkpoints');axes[0].legend()
    for i,group in enumerate(('train_1','valid_1')):
        values=r['prototype_usage'][group]['mean_responsibilities'];left=0
        for j,value in enumerate(values):
            axes[1].barh(i,value,left=left,color=f'C{j}',label=f'Prototype {j+1}' if i==0 else None)
            axes[1].text(left+value/2,i,f'{value:.1%}',ha='center',va='center',fontsize=8);left+=value
    axes[1].set_yticks([0,1],['Train positive','Valid positive']);axes[1].set(xlim=(0,1),title='C: soft responsibilities (not hard assignments)');axes[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.08),ncol=4)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','extract','audit','train','analyze','plot']);p.add_argument('--output',type=Path,required=True);p.add_argument('--name',choices=['baseline','A','B','C'],default='baseline');p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(2)
    if a.action in ('extract','train'):globals()[a.action](a.output,a.name,a.device)
    else:globals()[a.action](a.output)

if __name__=='__main__':main()
