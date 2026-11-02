"""Train/Valid-only frozen MIL diagnosis through the formal source-only forward."""

from pathlib import Path
import argparse
import json

import numpy as np
import torch

from .artifacts import write_json
from .data import read_jsonl, write_jsonl
from .metrics import paired_metrics


def bounded_logit_extrema(logits, counts):
    """Exact box-constrained fractional-linear extrema for independent r in [1,5].

    At an extremum r=5 below/above the achieved mean, r=1 elsewhere. Sorting
    therefore reduces all 2**L vertices to L+1 threshold candidates. These are
    label-informed capacity bounds, not deployable attention weights.
    """
    z,n=logits.double(),counts.double()
    total=(z*n).sum();mass=n.sum();extrema=[]
    for descending in (False,True):
        ix=z.argsort(descending=descending,stable=True)
        prefix_n=torch.cat((n.new_zeros(1),n[ix].cumsum(0)))
        prefix_z=torch.cat((n.new_zeros(1),(n[ix]*z[ix]).cumsum(0)))
        candidates=(total+4*prefix_z)/(mass+4*prefix_n)
        extrema.append(candidates.max() if descending else candidates.min())
    return extrema


def pooling_interventions(logits, rates, counts, attention_scores=None):
    """Exact linear-readout decomposition; no re-encoding or annotation input.

    Max/min are convex-pooling bounds, not proposed inference methods. All other
    interventions are fixed before seeing their validation performance.
    """
    valid = counts > 0
    if not valid.any():
        raise ValueError("a function must contain valid code lines")
    z, r, n = logits[valid], rates[valid], counts[valid]
    mass = n * r
    a = mass / mass.sum()
    baseline = (a * z).sum()
    uniform = (n * z).sum() / n.sum()
    top = a.argsort(descending=True, stable=True)[:3]
    rate_top = r.argsort(descending=True, stable=True)[:3]
    rest = torch.ones(len(z), dtype=torch.bool, device=z.device)
    rest[top] = False
    top_risk = z.argsort(descending=True, stable=True)[:3]
    scores = dict(
        original=baseline, token_mean=uniform, line_mean=z.mean(),
        attention_top3=(n[top] * z[top]).sum() / n[top].sum(),
        rate_top3=(n[rate_top] * z[rate_top]).sum() / n[rate_top].sum(),
        risk_top3=(n[top_risk] * z[top_risk]).sum() / n[top_risk].sum(),
        remove_attention_top3=((mass[rest] * z[rest]).sum() / mass[rest].sum()
                               if rest.any() else baseline),
    )
    if attention_scores is not None:
        log_mass=n.log()+torch.nn.functional.logsigmoid(attention_scores[valid])
        scores['zero_floor']=(log_mass.softmax(0)*z).sum()
    lower,upper=bounded_logit_extrema(z,n)
    return scores, dict(
        min_line_logit=float(z.min()), max_line_logit=float(z.max()),
        line_logit_std=float(z.std(unbiased=False)),
        attention_logit_shift=float(baseline - uniform),
        attention_effective_fraction=float(1 / a.square().sum() / len(a)),
        token_effective_fraction=float(n.sum().square() / n.square().sum() / len(n)),
        rate_saturation_fraction=float(((r < 1.05) | (r > 4.95)).float().mean()),
        attention_top3_mass=float(a[top].sum()),
        rate_top3_mass=float(a[rate_top].sum()),
        bounded_min_logit=float(lower),bounded_max_logit=float(upper),
    )


def summarize(rows, records):
    methods = list(records[0]["probabilities"])
    groups = dict(all=list(range(0, len(rows), 2)), visible_difference=[], identical_input=[],
                  pure_addition=[], truncated=[], untruncated=[])
    for j in range(0, len(rows), 2):
        a, b = records[j:j+2]
        different = a["visible_inputs_differ"]
        groups["visible_difference" if different else "identical_input"].append(j)
        if not rows[j]["lines_ground_truth"] and rows[j+1]["lines_ground_truth"]:
            groups["pure_addition"].append(j)
        truncated = any(f["source_tokens"] > 510 for f in (a, b))
        groups["truncated" if truncated else "untruncated"].append(j)
    result = dict(test_used=False, groups={}, convex_pooling_bounds={})
    for group, indices in groups.items():
        ix = [i for j in indices for i in (j, j+1)]
        if not ix:
            continue
        result["groups"][group] = {
            method: paired_metrics([rows[i] for i in ix],
                                   [records[i]["probabilities"][method] for i in ix])
            for method in methods
        }
        # Label-informed upper bound under this checkpoint's SAME linear readout.
        possible = sum(records[j]["visible_inputs_differ"]
                       and records[j]["max_line_logit"] > 0 and records[j+1]["min_line_logit"] <= 0
                       for j in indices)
        wrong = [records[i] for i in ix
                 if (records[i]["probabilities"]["original"] > .5) != bool(rows[i]["target"])]
        impossible = sum((r["max_line_logit"] <= 0 if r["target"] else r["min_line_logit"] > 0)
                         for r in wrong)
        bounded_impossible=sum(r['bounded_max_logit']<=0 if r['target'] else r['bounded_min_logit']>0 for r in wrong)
        bounded_possible=sum(records[j]['visible_inputs_differ'] and records[j]['bounded_max_logit']>0
                             and records[j+1]['bounded_min_logit']<=0 for j in indices)
        result["convex_pooling_bounds"][group] = dict(
            pairs=len(indices), possible_p_c_pairs=possible,
            wrong_functions=len(wrong), wrong_functions_no_pooling_can_fix=impossible,
            wrong_functions_bounded_cannot_fix=int(bounded_impossible),possible_bounded_p_c_pairs=int(bounded_possible),
        )
    result["line_statistics"] = {
        key: dict(mean=float(np.mean([r[key] for r in records])),
                  median=float(np.median([r[key] for r in records])))
        for key in ["line_logit_std", "attention_logit_shift", "attention_effective_fraction",
                    "token_effective_fraction", "rate_saturation_fraction", "attention_top3_mass"]
    }
    return result


@torch.no_grad()
def frozen(checkpoint, data_dir, output_dir, device, splits, layers=()):
    from .evaluation import load_paired
    from .training import paired_loader, paired_forward, paired_collate
    from .data import paired_features

    torch.set_num_threads(4)
    cp = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "bounded_attention" not in cp or cp["input_mode"] != "512":
        raise ValueError("frozen MIL diagnosis requires a bounded 512 checkpoint")
    model, tokenizer = load_paired(cp, device)
    root = Path(output_dir)
    strict_failures=[]
    for split in splits:
        rows = read_jsonl(Path(data_dir) / "primevul_paired" / f"{split}.lines.jsonl")
        expected = 8708 if split == "train" else 1124
        if len(rows) != expected:
            raise ValueError("complete official Train/Valid required")
        loader = paired_loader(rows, tokenizer, "baseline")
        if split == 'train' and cp.get('input_marginalization'):
            from .training import marginalize_input_targets
            target_summary=marginalize_input_targets(loader.dataset)
            if any(cp['input_marginalization'][k]!=v for k,v in target_summary.items()):
                raise ValueError('reconstructed Train conditional targets differ from training')
            write_jsonl(root/'train.classification_targets.jsonl',
                        [dict(sample_key=r['sample_key'],target=r['target'],training_target=f['classification_target'])
                         for r,f in zip(rows,loader.dataset)])
        records, pooled, at = [], [], 0
        single_delta = None
        single_check = None
        layer_pooled = {layer:[] for layer in layers}
        for batch in loader:
            prob, d = paired_forward(model, batch, "baseline", device, with_labels=False,
                                      return_attention=True, return_line_states=True, diagnostic_layers=layers)
            if at == 0:
                source_only=paired_features(dict(func=rows[0]['func'],target=0),tokenizer,'baseline')
                single_batch=paired_collate([source_only])
                for key in ('input_ids','attention_mask','code_mask','token_weights'):
                    torch.testing.assert_close(batch[key][0],single_batch[key][0],atol=0,rtol=0)
                ns=len(source_only['line_numbers'])
                torch.testing.assert_close(batch['line_token_mask'][0,:ns],single_batch['line_token_mask'][0,:ns],atol=0,rtol=0)
                one=paired_forward(model,single_batch,'baseline',device,with_labels=False)
                assert torch.equal(prob[:1]>.5,one>.5), 'single-function classification differs'
                single_delta=float((prob[:1]-one).abs().max())
                single_check=dict(tolerance=2e-6,probability_delta=single_delta,
                                  passed=bool(torch.allclose(prob[:1],one,atol=2e-6,rtol=0)),
                                  inputs_equal=True,decision_equal=True)
                if not single_check['passed']:
                    strict_failures.append(dict(split=split,sample_key=rows[0]['sample_key'],**single_check))
            h = d["line_states"]
            line_logits = model.classifier(h).squeeze(-1)
            for i in range(len(prob)):
                r, f = rows[at+i], loader.dataset[at+i]
                n = int((batch["line_numbers"][i] >= 0).sum())
                z, rates, counts = (x[i, :n].cpu() for x in
                                    (line_logits, d["line_rates"], d["line_counts"]))
                raw_scores=d['attention_scores'][i,:n].cpu()
                scores, stats = pooling_interventions(z, rates, counts,
                                                      None if cp.get('input_marginalization') else raw_scores)
                if abs(float(scores["original"].sigmoid()) - float(prob[i, 0])) > 2e-6:
                    raise ValueError("physical-line decomposition differs from formal forward")
                ns = batch["line_numbers"][i, :n].tolist()
                labels = [int(x in r["lines_ground_truth"]) for x in ns]
                a = counts * rates; a = a / a.sum()
                edit = torch.tensor(labels, dtype=torch.bool)
                stats["edit_attention_mass"] = float(a[edit].sum())
                stats["edit_token_fraction"] = float(counts[edit].sum() / counts.sum())
                stats["edit_mean_logit"] = float(z[edit].mean()) if edit.any() else None
                stats["background_mean_logit"] = float(z[~edit].mean()) if (~edit).any() else None
                probabilities={k:float(v.sigmoid()) for k,v in scores.items()}
                # Classification metrics use the actual formal forward. The
                # line decomposition is a checked diagnostic, not a replacement
                # implementation of the classifier's floating-point reduction.
                probabilities['original']=float(prob[i,0])
                records.append(dict(sample_key=r["sample_key"], pair_key=r["pair_key"], target=r["target"],
                                    project=r["project"], source_tokens=f["source_tokens"],
                                    visible_inputs_differ=(loader.dataset[(at+i)//2*2]["input_ids"]
                                                          != loader.dataset[(at+i)//2*2+1]["input_ids"]),
                                    line_numbers=ns, line_logits=z.tolist(), line_rates=rates.tolist(),
                                    attention_scores=raw_scores.tolist(),
                                    line_token_counts=counts.tolist(), line_labels=labels,
                                    probabilities=probabilities, **stats))
                weights = d["line_attention"][i, :n]
                pooled.append(torch.stack(((h[i,:n] * weights[:,None]).sum(0),
                                          (h[i,:n] * counts.to(device)[:,None]).sum(0) / counts.sum().to(device))).cpu())
                for layer, values in layer_pooled.items():
                    hh = d["diagnostic_line_states"][layer][i,:n]
                    values.append(torch.stack(((hh * weights[:,None]).sum(0),
                                               (hh * counts.to(device)[:,None]).sum(0) / counts.sum().to(device))).cpu())
            at += len(prob)
        result = summarize(rows, records)
        result.update(checkpoint=str(checkpoint), selected_epoch=cp.get("selected_epoch",cp.get("epoch")), split=split,
                      source_only_single_function_max_delta=single_delta,checkpoint_restored=True,
                      single_function_check=single_check)
        write_jsonl(root / f"{split}.lines.jsonl", records)
        write_json(root / f"{split}.summary.json", result)
        # Local cache only; no source text or annotation enters the pooled vectors.
        root.mkdir(parents=True, exist_ok=True)
        torch.save(dict(features=torch.stack(pooled), sample_keys=[r["sample_key"] for r in rows]),
                   root / f"{split}.features.pt")
        for layer,values in layer_pooled.items():
            layer_dir = root / f"layer{layer}"
            layer_dir.mkdir(parents=True,exist_ok=True)
            torch.save(dict(features=torch.stack(values),sample_keys=[r["sample_key"] for r in rows]),
                       layer_dir/f"{split}.features.pt")
        print(json.dumps(dict(split=split, groups={k:{m:round(v["p_c"],4) for m,v in d.items()}
                                                  for k,d in result["groups"].items()},
                              bounds=result["convex_pooling_bounds"], statistics=result["line_statistics"])), flush=True)
    if strict_failures:
        # Preserve the original criterion and a failing exit status, while
        # retaining the other completed diagnostics for independent inspection.
        raise AssertionError('strict single-function probability check failed: '+json.dumps(strict_failures))


def existing(output_dir, data_dir=Path('data')):
    """Compare source-only Valid predictions; never load the aggregate Test reports."""
    paths = {
        "losver": "losver_512/primevul_paired/losver",
        "losver_mean": "losver_512/primevul_paired/unixcoder",
        "global_mean": "bounded_attention_512/global_mean/primevul_paired/unixcoder",
        "attention_bce": "bounded_attention_512/attention_bce/primevul_paired/unixcoder",
        "diff_region": "bounded_attention_512/diff_region/primevul_paired/unixcoder",
        "context_diff": "attention_context_512/context_diff/primevul_paired/unixcoder",
        "context_bce": "attention_context_512/context_bce/primevul_paired/unixcoder",
        "context_random": "attention_context_512/context_random/primevul_paired/unixcoder",
        "ema": "attention_regularization_512/ema_attention/primevul_paired/unixcoder",
        "local": "attention_local_512/dual_diff/primevul_paired/unixcoder",
        "context_scale": "context_scale_512/context_scale_diff/primevul_paired/unixcoder",
        "insertion": "insertion_localization_512/insertion_diff/primevul_paired/unixcoder",
        "paired_context_diff": "rccl_512/matched_context_diff/primevul_paired/unixcoder",
        "rccl": "rccl_512/rccl_diff/primevul_paired/unixcoder",
        "rccl_random": "rccl_512/rccl_random/primevul_paired/unixcoder",
    }
    rows = read_jsonl(Path(data_dir)/"primevul_paired/valid.lines.jsonl")
    train_rows = read_jsonl(Path(data_dir)/"primevul_paired/train.lines.jsonl")
    result = dict(test_used=False, models={})
    reference = None
    successes = {}
    for name, directory in paths.items():
        path = Path("results") / directory / "valid.best-p_c.predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        preds = read_jsonl(path)
        if [p["sample_key"] for p in preds] != [r["sample_key"] for r in rows]:
            raise ValueError(f"{name}: Valid membership/order differs")
        scores = np.asarray([r["probability"] for r in preds])
        pc = (scores[::2] > .5) & (scores[1::2] <= .5)
        successes[name] = pc
        if name == "losver":
            reference = pc
        c = json.loads(path.with_name("config.json").read_text())
        result["models"][name] = dict(
            path=str(path), metrics=paired_metrics(rows, scores),
            gains_vs_losver=int((pc & ~reference).sum()), losses_vs_losver=int((~pc & reference).sum()),
            training_config=c["training_config"], parameters=c["parameters"],
        )
        train_path = path.with_name("train.selected.predictions.jsonl")
        if train_path.exists():
            train_preds = read_jsonl(train_path)
            if [x["sample_key"] for x in train_preds] != [x["sample_key"] for x in train_rows]:
                raise ValueError("Train predictions must contain the complete official split")
            result["models"][name]["train_metrics"] = paired_metrics(
                train_rows, [x["probability"] for x in train_preds])
        history = read_jsonl(path.with_name("best.training.jsonl"))
        result["models"][name]["epochs"] = [
            dict(epoch=e["epoch"],train_loss=e["train_loss"],valid=e["validation"])
            for e in history if e["event"] == "epoch"
        ]
    delta = successes["context_diff"].astype(float) - reference
    rng = np.random.default_rng(123456)
    resampled = delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
    result["paired_bootstrap_context_minus_losver"] = dict(
        seed=123456, resamples=10000, difference=float(delta.mean()),
        interval_95=np.quantile(resampled,[.025,.975]).tolist(),
        scope="same Valid pairs; does not measure training-seed variability")
    train_sources = {x["func"] for x in train_rows}
    result["source_overlap_train_valid"] = sum(r["func"] in train_sources for r in rows)
    write_json(Path(output_dir) / "existing.valid.json", result)
    print(json.dumps({k:dict(p_c=v["metrics"]["p_c"],auc=v["metrics"]["auc"],mcc=v["metrics"]["mcc"],
                            gains=v["gains_vs_losver"], losses=v["losses_vs_losver"])
                      for k,v in result["models"].items()}, indent=2))


def probe(feature_dir, data_dir, output_dir):
    """Fixed C=1 linear probes: BCE vs BCE+paired logistic; Train fit only.

    This is a cheap hypothesis check, not a replacement training framework. The
    frozen encoder and pooling are identical between the two readout objectives.
    """
    from scipy.optimize import minimize
    from scipy.special import expit

    data = {}
    rows = {}
    for split in ("train", "valid"):
        cache = torch.load(Path(feature_dir) / f"{split}.features.pt", map_location="cpu", weights_only=True)
        rows[split] = read_jsonl(Path(data_dir) / "primevul_paired" / f"{split}.lines.jsonl")
        if cache["sample_keys"] != [r["sample_key"] for r in rows[split]]:
            raise ValueError("probe source members differ")
        data[split] = cache["features"].double().numpy()
    n, _, dim = data["train"].shape
    y = np.asarray([r["target"] for r in rows["train"]])
    results = dict(test_used=False, feature_dir=str(feature_dir), c=1., pair_weight=1., results=[])
    for pooling in (0, 1):
        x = np.column_stack((data["train"][:,pooling],np.ones(n)))
        dx = x[::2] - x[1::2]
        for pair_weight in (0., 1.):
            def objective(w):
                z = x @ w
                value = np.mean(np.logaddexp(0.,z)-y*z) + .5*np.square(w[:-1]).sum()/n
                gradient = x.T @ (expit(z)-y)/n
                gradient[:-1] += w[:-1]/n
                if pair_weight:
                    gap = dx @ w
                    value += pair_weight * np.logaddexp(0.,-gap).mean()
                    gradient += pair_weight * dx.T @ (expit(gap)-1)/len(dx)
                return value, gradient
            fit = minimize(objective,np.zeros(dim+1),jac=True,method="L-BFGS-B",
                           options=dict(maxiter=2000,gtol=1e-6,ftol=1e-10))
            if not fit.success:
                raise RuntimeError(f"linear probe failed: {fit.message}")
            scores = {s:expit(data[s][:,pooling] @ fit.x[:-1] + fit.x[-1]) for s in data}
            record = dict(pooling="attention" if pooling==0 else "token_mean",pair_weight=pair_weight,
                          iterations=int(fit.nit),train_objective=float(fit.fun),
                          metrics={s:paired_metrics(rows[s],p) for s,p in scores.items()})
            results["results"].append(record)
            print(json.dumps(record),flush=True)
    write_json(Path(output_dir)/"frozen_probes.json",results)


def target_smoke(checkpoint, data_dir, output_dir, device):
    from . import model as core
    from .evaluation import load_paired
    from .training import paired_loader,paired_forward,paired_collate,marginalize_input_targets

    core.seed_everything(123456)
    torch.set_num_threads(4)
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    m,tok=load_paired(cp,device)
    # Fixed Train pairs from the previously measured source-only coverage.
    coverage=read_jsonl('results/rccl_512/coverage.train.jsonl')
    selected=[]
    for different in (False,True):
        selected.extend([r['pair_key'] for r in coverage if r['visible_inputs_differ']==different][:2])
    rs=[r for r in read_jsonl(Path(data_dir)/'primevul_paired/train.lines.jsonl') if r['pair_key'] in set(selected)]
    loader=paired_loader(rs,tok,'baseline',train=True,marginalize_inputs=True)
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state(device)
    summary=marginalize_input_targets(loader.dataset)
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state(device))
    b=paired_collate(loader.dataset)
    with torch.no_grad():
        p=paired_forward(m,b,'baseline',device,with_labels=False)
        one=paired_forward(m,paired_collate(loader.dataset[:1]),'baseline',device,with_labels=False)
    torch.testing.assert_close(p[:1],one,rtol=0,atol=2e-6)
    original=m.classifier.weight.detach().clone()
    m.train()
    loss,_,d=paired_forward(m,b,'baseline',device,return_attention=True)
    torch.testing.assert_close(d['bce'],torch.nn.functional.binary_cross_entropy_with_logits(
        d['logits'],b['classification_targets'].to(device)))
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    encoder_grad=float(m.encoder.embeddings.word_embeddings.weight.grad.abs().sum())
    assert encoder_grad>0
    torch.optim.SGD(m.parameters(),lr=1e-4).step()
    update=float((m.classifier.weight-original).abs().sum())
    assert update>0
    m.eval()
    with torch.no_grad():
        after=paired_forward(m,b,'baseline',device,with_labels=False)
        b['classification_targets'].fill_(float('nan'));b['line_labels'].fill_(float('nan'))
        poisoned=paired_forward(m,b,'baseline',device,with_labels=False)
    torch.testing.assert_close(after,poisoned,rtol=0,atol=0)
    restored,_=load_paired(dict(cp,state_dict=m.state_dict(),input_marginalization=summary),device)
    with torch.no_grad():
        actual=paired_forward(restored,b,'baseline',device,with_labels=False)
    torch.testing.assert_close(actual,after,rtol=0,atol=0)
    result=dict(device=torch.cuda.get_device_name(device),seed=123456,test_used=False,
                target_summary=summary,bce=float(d['bce']),region_loss=float(d['region_loss']),
                encoder_gradient_l1=encoder_grad,classifier_update_l1=update,
                rng_preserved=True,checkpoint_restore=True,single_function_consistent=True,
                inference_ignores_targets_and_diff=True,parameters_added=0)
    write_json(Path(output_dir)/'gpu-smoke.json',result)
    print(json.dumps(result),flush=True)


def report(output_dir, data_dir=Path('data')):
    """Finalize the predeclared pilot decision; all evaluation remains on Valid."""
    import csv
    root=Path(output_dir)
    experiment=json.loads((root/'experiment.json').read_text())
    rows=read_jsonl(Path(data_dir)/'primevul_paired/valid.lines.jsonl')
    runs={}
    for name in ('pilot_baseline','pilot_marginalized'):
        run=root/name/'primevul_paired/unixcoder'
        config=json.loads((run/'config.json').read_text())
        complete=json.loads((run/'training.metrics.json').read_text())
        events=[r for r in read_jsonl(run/'best.training.jsonl') if r['event']=='epoch']
        if complete['completed_epochs']!=16 or complete['optimizer_steps']!=4096 or len(events)!=16:
            raise ValueError('the declared pilot must finish before making a decision')
        p=read_jsonl(run/'valid.best-p_c.predictions.jsonl')
        if [r['sample_key'] for r in p]!=[r['sample_key'] for r in rows]:
            raise ValueError('pilot must evaluate complete official Valid')
        selected=json.loads((run/'best-p_c.json').read_text())
        runs[name]=dict(config=config,selected=selected,metrics=paired_metrics(rows,[r['probability'] for r in p]),
                        last_four_mean_p_c=float(np.mean([e['validation']['p_c'] for e in events[-4:]])),
                        training_seconds=sum(e['training_seconds'] for e in events),epochs=events)
        ix=[i for j in range(0,len(rows),2)
            if not rows[j]['lines_ground_truth'] and rows[j+1]['lines_ground_truth'] for i in (j,j+1)]
        runs[name]['pure_addition_metrics']=paired_metrics([rows[i] for i in ix],[p[i]['probability'] for i in ix])
    a,b=(runs[n] for n in ('pilot_baseline','pilot_marginalized'))
    for field in ('training_members','validation_members','training_config','parameters','epochs','selection'):
        if a['config'][field]!=b['config'][field]:
            raise ValueError(f'pilot control differs: {field}')
    rule=experiment['promotion_rule']
    deltas=dict(p_c=b['metrics']['p_c']-a['metrics']['p_c'],auc=b['metrics']['auc']-a['metrics']['auc'],
                last_four_mean_p_c=b['last_four_mean_p_c']-a['last_four_mean_p_c'])
    promote=(deltas['p_c']>=rule['selected_p_c_gain_min'] and
             deltas['auc']>=rule['selected_auc_change_min'] and
             deltas['last_four_mean_p_c']>=rule['last_four_epochs_mean_p_c_change_min'])
    decisions=dict(test_used=False,pilot={n:{k:v for k,v in d.items() if k not in ('config','epochs')}
                                        for n,d in runs.items()},deltas=deltas,promotion_rule=rule,
                   promote_to_full_training=bool(promote),parameters_equal=True,members_equal=True)
    states=[torch.load(root/name/'primevul_paired/unixcoder/last.pt',map_location='cpu',weights_only=False)
            for name in ('pilot_baseline','pilot_marginalized')]
    decisions['final_cpu_rng_equal']=torch.equal(states[0]['rng']['torch'],states[1]['rng']['torch'])
    devices=[int(r['argv'][r['argv'].index('--device')+1].split(':')[1])
             for r in experiment['pilot']['runs']]
    decisions['final_cuda_rng_equal']=torch.equal(states[0]['rng']['cuda'][devices[0]],
                                                 states[1]['rng']['cuda'][devices[1]])
    if not decisions['final_cpu_rng_equal'] or not decisions['final_cuda_rng_equal']:
        raise ValueError('matched pilot consumed different sampler/Dropout RNG')
    write_json(root/'pilot.comparison.json',decisions)
    with (root/'pilot.epochs.csv').open('w') as h:
        writer=csv.DictWriter(h,fieldnames=['method','epoch','train_loss','bce','p_c','auc','mcc','f1','recall','patched_fpr'])
        writer.writeheader()
        for name,d in runs.items():
            for e in d['epochs']:
                writer.writerow(dict(method=name,epoch=e['epoch'],train_loss=e['train_loss'],bce=e['train_components']['bce'],
                                     **{k:e['validation'][k] for k in ['p_c','auc','mcc','f1','recall','patched_fpr']}))
    print(json.dumps(decisions,indent=2))
    return decisions


@torch.no_grad()
def losver_frozen(data_dir, output_dir, device):
    """Reproduce the released Valid readout, then remove its predicted mask."""
    from .evaluation import load_paired
    from .training import paired_rows,paired_loader,paired_forward
    torch.set_num_threads(4)
    directory=Path('results/losver_512/primevul_paired/losver')
    cp=torch.load(directory/'best-p_c.pt',map_location='cpu',weights_only=False)
    model,tokenizer=load_paired(cp,device)
    rows=paired_rows(data_dir,Path('results/losver_512'),'predicted','valid',method='losver')
    reference=read_jsonl(directory/'valid.best-p_c.predictions.jsonl')
    source=read_jsonl(Path(data_dir)/'primevul_paired/valid.jsonl')
    if [r['sample_key'] for r in rows]!=[r['sample_key'] for r in source]:
        raise ValueError('LOSVER Valid members differ')
    predictions=dict(original=[],uniform=[])
    for batch in paired_loader(rows,tokenizer,'predicted'):
        predictions['original'].extend(paired_forward(model,batch,'predicted',device,with_labels=False)[:,0].cpu().tolist())
        batch['token_weights'].fill_(1.)
        predictions['uniform'].extend(paired_forward(model,batch,'predicted',device,with_labels=False)[:,0].cpu().tolist())
    before=np.array([r['probability'] for r in reference]);after=np.array(predictions['original'])
    if np.max(np.abs(before-after))>2e-6 or not np.array_equal(before>.5,after>.5):
        raise ValueError('formal LOSVER forward no longer reproduces its saved Valid predictions')
    result=dict(test_used=False,checkpoint=str(directory/'best-p_c.pt'),
                reproduction_max_probability_delta=float(np.max(np.abs(before-after))),
                metrics={k:paired_metrics(rows,p) for k,p in predictions.items()})
    write_json(Path(output_dir)/'losver_frozen.valid.json',result)
    write_jsonl(Path(output_dir)/'losver_frozen.valid.predictions.jsonl',
                [dict(sample_key=r['sample_key'],pair_key=r['pair_key'],target=r['target'],
                      **{k:p[i] for k,p in predictions.items()}) for i,r in enumerate(rows)])
    print(json.dumps(result),flush=True)


def full_report(output_dir,data_dir=Path('data')):
    import csv
    root=Path(output_dir)
    experiment=json.loads((root/'experiment.json').read_text())
    spec=experiment['full_experiment']
    directories={'context_diff_seed123456':Path(spec['reference'])}
    for family in ('full_marginalized','full_context'):
        for seed in (123456,123457,123458):
            run=root/family/f'seed{seed}'/'primevul_paired/unixcoder'
            if (run/'training.metrics.json').exists():
                directories[f'{family}_seed{seed}']=run
    if 'full_marginalized_seed123456' not in directories:
        raise ValueError('complete the first full experiment before applying its promotion rule')
    rows=read_jsonl(Path(data_dir)/'primevul_paired/valid.lines.jsonl')
    results={};configs={}
    for name,run in directories.items():
        c=json.loads((run/'config.json').read_text());configs[name]=c
        completed=json.loads((run/'training.metrics.json').read_text())
        if completed['completed_epochs']!=16 or completed['optimizer_steps']!=17424:
            raise ValueError('full runs require 16 epochs and 17424 steps')
        preds=read_jsonl(run/'valid.best-p_c.predictions.jsonl')
        if [(r['sample_key'],r['target']) for r in preds]!=[(r['sample_key'],r['target']) for r in rows]:
            raise ValueError('full Valid membership or original hard labels differ')
        events=[r for r in read_jsonl(run/'best.training.jsonl') if r['event']=='epoch']
        chosen=max(events,key=lambda r:round(r['validation']['p_c'],4))['epoch']
        if chosen!=completed['selected_epochs']['p_c']:
            raise ValueError('full run did not follow the established P-C selection rule')
        results[name]=dict(seed=c['training_config']['seed'],epoch=chosen,
                           metrics=paired_metrics(rows,[r['probability'] for r in preds]),
                           last_four_mean_p_c=float(np.mean([r['validation']['p_c'] for r in events[-4:]])),
                           training_seconds=sum(r['training_seconds'] for r in events),path=str(run))
    baseline=results['context_diff_seed123456'];candidate=results['full_marginalized_seed123456']
    for k in ('training_members','validation_members','parameters','training_config','input_coverage'):
        if configs['context_diff_seed123456'][k]!=configs['full_marginalized_seed123456'][k]:
            raise ValueError(f'full comparison differs in {k}')
    delta={k:candidate['metrics'][k]-baseline['metrics'][k] for k in
           ('p_c','auc','mcc','f1','recall','patched_fpr','pair_ranking_accuracy')}
    rule=spec['multiseed_promotion_rule']
    promote=(delta['p_c']>=rule['selected_p_c_gain_min'] and
             delta['auc']>=rule['selected_auc_change_min'] and
             delta['mcc']>=rule['selected_mcc_change_min'])
    summary=dict(test_used=False,results=results,matched_seed123456_deltas=delta,
                 multiseed_promotion_rule=rule,promote_candidate_to_multiple_seeds=bool(promote),
                 note='LOSVER historical reference has one fixed localizer/detector seed; no claim of full LOSVER pipeline seed stability')
    old=torch.load(directories['context_diff_seed123456']/'last.pt',map_location='cpu',weights_only=False,mmap=True)
    new=torch.load(directories['full_marginalized_seed123456']/'last.pt',map_location='cpu',weights_only=False,mmap=True)
    indices=spec['rng_indices']
    summary['rng_control']=dict(cpu_equal=torch.equal(old['rng']['torch'],new['rng']['torch']),
                                active_cuda_equal=torch.equal(old['rng']['cuda'][indices['reference']],
                                                               new['rng']['cuda'][indices['candidate']]))
    if not all(summary['rng_control'].values()):
        raise ValueError('the full comparison did not preserve sampler/Dropout RNG consumption')
    write_json(root/'full.comparison.json',summary)
    fields=['model','seed','epoch','p_c','auc','mcc','f1','recall','patched_fpr','pair_ranking_accuracy','training_seconds']
    with (root/'full.classification.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
        for name,d in results.items():
            w.writerow(dict(model=name,seed=d['seed'],epoch=d['epoch'],training_seconds=d['training_seconds'],
                            **{k:d['metrics'][k] for k in fields[3:-1]}))
    print(json.dumps(summary,indent=2))
    return summary


@torch.no_grad()
def numerics(checkpoint,data_dir,output_dir,device):
    """Keep the 2e-6 criterion and check B=8 versus B=1 on complete Valid."""
    from .evaluation import load_paired
    from .training import paired_loader,paired_forward,paired_collate
    torch.set_num_threads(4)
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model,tokenizer=load_paired(cp,device)
    rows=read_jsonl(Path(data_dir)/'primevul_paired/valid.lines.jsonl')
    if len(rows)!=1124:
        raise ValueError('complete official Valid required')
    loader=paired_loader(rows,tokenizer,'baseline')
    records=[];at=0
    for batch in loader:
        probabilities=paired_forward(model,batch,'baseline',device,with_labels=False)[:,0].cpu().tolist()
        for i,p in enumerate(probabilities):
            f=dict(loader.dataset[at+i]);f.pop('line_labels',None);f.pop('classification_target',None);f['labels']=0
            single=paired_collate([f])
            for key in ('input_ids','attention_mask','code_mask','token_weights'):
                torch.testing.assert_close(batch[key][i],single[key][0],atol=0,rtol=0)
            q=float(paired_forward(model,single,'baseline',device,with_labels=False)[0,0])
            records.append(dict(sample_key=rows[at+i]['sample_key'],target=rows[at+i]['target'],
                                batch_probability=p,single_probability=q,absolute_delta=abs(p-q),
                                strict_probability_passed=abs(p-q)<=2e-6,decision_equal=(p>.5)==(q>.5)))
        at+=len(probabilities)
    result=dict(test_used=False,checkpoint=str(checkpoint),functions=len(rows),tolerance=2e-6,
                strict_probability_failures=sum(not r['strict_probability_passed'] for r in records),
                decision_mismatches=sum(not r['decision_equal'] for r in records),
                max_probability_delta=max(r['absolute_delta'] for r in records),
                metrics={name:paired_metrics(rows,[r[name+'_probability'] for r in records])
                         for name in ('batch','single')})
    root=Path(output_dir)
    write_jsonl(root/'valid.inference_check.predictions.jsonl',records)
    write_json(root/'valid.inference_check.json',result)
    print(json.dumps(result),flush=True)
    if result['strict_probability_failures'] or result['decision_mismatches']:
        raise AssertionError('strict probability/decision consistency failed; full results retained')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("existing", "frozen", "probe", "target-smoke", "report", "losver-frozen", "full-report", "numerics"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--feature-dir", type=Path)
    p.add_argument("--layers",nargs="+",type=int,default=[])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--splits", nargs="+", choices=("train", "valid"), default=["train", "valid"])
    a = p.parse_args()
    if a.command == "existing":
        existing(a.output_dir,a.data_dir)
    elif a.command == "report":
        report(a.output_dir,a.data_dir)
    elif a.command == "losver-frozen":
        losver_frozen(a.data_dir,a.output_dir,a.device)
    elif a.command == "full-report":
        full_report(a.output_dir,a.data_dir)
    elif a.command == 'numerics':
        if a.checkpoint is None:
            p.error('numerics requires --checkpoint')
        numerics(a.checkpoint,a.data_dir,a.output_dir,a.device)
    elif a.command == "probe":
        if a.feature_dir is None:
            p.error("probe requires --feature-dir")
        probe(a.feature_dir, a.data_dir, a.output_dir)
    elif a.command == "target-smoke":
        if a.checkpoint is None:
            p.error("target-smoke requires --checkpoint")
        target_smoke(a.checkpoint,a.data_dir,a.output_dir,a.device)
    else:
        if a.checkpoint is None:
            p.error("frozen requires --checkpoint")
        frozen(a.checkpoint, a.data_dir, a.output_dir, a.device, a.splits, a.layers)


if __name__ == "__main__":
    main()
