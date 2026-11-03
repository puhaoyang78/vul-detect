"""Global/local MIL smoke and matched-result analysis; training stays in the CLI."""
from pathlib import Path
import argparse
import csv
import json
import tempfile
from itertools import permutations

import numpy as np
import torch

from .artifacts import write_json
from .data import read_jsonl, write_jsonl, paired_features
from .metrics import paired_metrics


def distribution(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std()),
                quantiles=dict(zip(('min', 'p10', 'p50', 'p90', 'max'),
                                   np.quantile(a, [0, .1, .5, .9, 1]).tolist())))


def pair_changes(rows, candidate, reference):
    if len(rows) != len(candidate) or len(rows) != len(reference) or len(rows) % 2:
        raise ValueError('paired comparison requires the same complete members')
    a, b = np.asarray(candidate), np.asarray(reference)
    pa = (a[::2] > .5) & (a[1::2] <= .5)
    pb = (b[::2] > .5) & (b[1::2] <= .5)
    return dict(gains=int((pa & ~pb).sum()), losses=int((~pa & pb).sum()),
                gained_pairs=[rows[2*i]['pair_key'] for i in np.flatnonzero(pa & ~pb)],
                lost_pairs=[rows[2*i]['pair_key'] for i in np.flatnonzero(~pa & pb)])


def align_predictions(rows, predictions):
    """Join on the COMPLETE pair key and side; never an index suffix or order."""
    expected = {(r['pair_key'], int(r['target'])): r for r in rows}
    actual = {}
    if len(expected) != len(rows):
        raise ValueError('duplicate official pair side')
    for p in predictions:
        key = (p['pair_key'], int(p['target']))
        if key in actual:
            raise ValueError('duplicate predicted pair side')
        actual[key] = p
    if set(actual) != set(expected):
        raise ValueError('prediction members differ from the complete official split')
    ordered = [actual[(r['pair_key'], int(r['target']))] for r in rows]
    if any(p['sample_key'] != r['sample_key'] for r, p in zip(rows, ordered)):
        raise ValueError('pair side and sample key disagree')
    return ordered


def pair_category(v, p):
    return 'p_c' if v > .5 and p <= .5 else 'p_v' if v > .5 else 'p_r' if p > .5 else 'p_b'


def saved_logit(prediction):
    """Exact if exported; otherwise implied by saved probability, never clipped."""
    if 'logit' in prediction:
        return float(prediction['logit']), 'exported_logit'
    if 'global_logit' in prediction and 'local_residual' in prediction:
        return prediction['global_logit'] + prediction['local_residual'], 'exported_components'
    p = prediction['probability']
    if 0 < p < 1:
        return float(np.log(p) - np.log1p(-p)), 'inverse_saved_probability'
    return None, 'saturated_probability_unknown_logit'


def historical_errors(data_dir, output_dir, model_path):
    """Retrospective saved-result audit, separate from new Train/Valid validation."""
    from transformers import RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, trim_offsets=False)
    paths = {
        'unixcoder': Path('results/losver_512/primevul_paired/unixcoder'),
        'losver': Path('results/losver_512/primevul_paired/losver'),
        'context_diff': Path('results/attention_context_512/context_diff/primevul_paired/unixcoder'),
        'rccl': Path('results/rccl_512/rccl_diff/primevul_paired/unixcoder'),
        'insertion': Path('results/insertion_localization_512/insertion_diff/primevul_paired/unixcoder'),
        **{'dual_'+m: Path(f'results/global_local_mil_512/seed123456/{m}/primevul_paired/unixcoder')
           for m in ('baseline', 'mean', 'lse')},
    }
    root = Path(output_dir)
    result = dict(role='historical Test error diagnosis only; no structure/parameter/threshold/epoch selection',
                  prediction_sources={k: str(v) for k, v in paths.items()}, splits={})
    all_records = []; comparisons = {}; case_keys = set()
    for split, expected in (('train', 8708), ('valid', 1124), ('test', 1128)):
        rows = read_jsonl(Path(data_dir)/'primevul_paired'/('test.oracle.jsonl' if split == 'test' else f'{split}.lines.jsonl'))
        if len(rows) != expected:
            raise ValueError('complete official members required')
        features = [paired_features(r, tokenizer, 'baseline') for r in rows]
        groups = cohort_indices(rows, features)
        stored = {}; unavailable = []
        for name, path in paths.items():
            filename = ('train.selected.predictions.jsonl' if split == 'train' else
                        'valid.best-p_c.predictions.jsonl' if split == 'valid' else 'test.predictions.jsonl')
            if not (path/filename).exists():
                if split != 'train':
                    raise FileNotFoundError(path/filename)
                unavailable.append(name); continue
            stored[name] = align_predictions(rows, read_jsonl(path/filename))
        pair_records = []
        for i in range(0, len(rows), 2):
            same_ids = features[i]['input_ids'] == features[i+1]['input_ids']
            same_membership = features[i]['line_token_indices'] == features[i+1]['line_token_indices']
            rec = dict(split=split, pair_key=rows[i]['pair_key'], project=rows[i]['project'],
                       same_encoder_input=same_ids, same_line_membership=same_membership,
                       visible_difference=not same_ids,
                       modified_lines=[r['lines_ground_truth'] for r in rows[i:i+2]],
                       visible_lines=[f['line_numbers'] for f in features[i:i+2]], methods={})
            for name, pred in stored.items():
                a, b = pred[i:i+2]; va, sa = saved_logit(a); vb, sb = saved_logit(b)
                rec['methods'][name] = dict(category=pair_category(a['probability'], b['probability']),
                    probabilities=[a['probability'], b['probability']], logits=[va, vb], logit_sources=[sa, sb],
                    logit_gap=va-vb if va is not None and vb is not None else None,
                    ranking='correct' if a['probability'] > b['probability'] else
                            'reversed' if a['probability'] < b['probability'] else 'tie')
            pair_records.append(rec)
        summary = dict(functions=len(rows), pairs=len(rows)//2, missing_saved_train_predictions=unavailable,
                       identical_ids_different_line_membership=sum(r['same_encoder_input'] and not r['same_line_membership'] for r in pair_records),
                       groups={})
        for group in ('all', 'visible_difference', 'identical_input', 'pure_addition'):
            ix = groups[group]
            selected_records = [pair_records[i//2] for i in ix[::2]]
            if not ix: continue
            methods = {}
            for name, pred in stored.items():
                data = [r['methods'][name] for r in selected_records]
                gaps = [r['logit_gap'] for r in data if r['logit_gap'] is not None]
                methods[name] = dict(metrics=paired_metrics([rows[i] for i in ix], [pred[i]['probability'] for i in ix]),
                    logit_gap=distribution(gaps) if gaps else None,
                    gap_unknown_due_to_saturation=sum(r['logit_gap'] is None for r in data),
                    ranking_counts={k: sum(r['ranking']==k for r in data) for k in ('correct','reversed','tie')},
                    error_ranking={c: {k:sum(r['category']==c and r['ranking']==k for r in data)
                                       for k in ('correct','reversed','tie')} for c in ('p_v','p_b','p_r')})
            summary['groups'][group] = methods
            comparisons[split+':'+group] = {}
            for a, b in permutations(stored, 2):
                ca = [stored[a][i]['probability'] for i in ix]; cb = [stored[b][i]['probability'] for i in ix]
                transitions = {x: {y:0 for y in ('p_c','p_v','p_b','p_r')} for x in ('p_c','p_v','p_b','p_r')}
                for r in selected_records:
                    transitions[r['methods'][b]['category']][r['methods'][a]['category']] += 1
                comparisons[split+':'+group][a+'_minus_'+b] = dict(
                    **pair_changes([rows[i] for i in ix], ca, cb), transitions=transitions)
        # Deterministic descriptive examples, not rules or training members.
        if split == 'test':
            for group in ('visible_difference', 'identical_input'):
                for category in ('p_v','p_b','p_r','p_c'):
                    selected = [r for r in pair_records if r['visible_difference']==(group=='visible_difference')
                                and r['methods']['context_diff']['category']==category]
                    case_keys.update(r['pair_key'] for r in selected[:2])
        result['splits'][split] = summary
        all_records.extend(pair_records)
        print(json.dumps(dict(historical_split=split, pairs=summary['pairs'], methods=list(stored))), flush=True)
    write_json(root/'history.summary.json', result)
    write_json(root/'history.comparisons.json', comparisons)
    write_jsonl(root/'history.pairs.jsonl', all_records)
    write_json(root/'history.cases.json', dict(pair_keys=sorted(case_keys),
        selection='first two official pairs in each Context+Diff category and visibility cohort; retrospective only'))
    render_historical_errors(root)
    return result


def render_historical_errors(output_dir):
    """Render saved historical statistics without tokenizing or predicting again."""
    root = Path(output_dir)
    result = json.loads((root/'history.summary.json').read_text())
    comparisons = json.loads((root/'history.comparisons.json').read_text())
    paths = result['prediction_sources']
    fields = ['split','cohort','method','pairs','p_c','p_v','p_b','p_r','auc','mcc','patched_fpr',
              'pair_ranking_accuracy','finite_logit_gaps','mean_logit_gap','unknown_logit_gaps']
    with (root/'history.classification.csv').open('w') as h:
        writer = csv.DictWriter(h, fieldnames=fields, lineterminator='\n'); writer.writeheader()
        for split, d in result['splits'].items():
            for group, models in d['groups'].items():
                for name, data in models.items():
                    m = data['metrics']; gap = data['logit_gap']
                    writer.writerow(dict(split=split, cohort=group, method=name,
                        **{k:m[k] for k in fields[3:12]},
                        finite_logit_gaps=gap['n'] if gap else 0, mean_logit_gap=gap['mean'] if gap else None,
                        unknown_logit_gaps=data['gap_unknown_due_to_saturation']))
    context_test = result['splits']['test']['groups']['all']['context_diff']['metrics']
    visible_test = result['splits']['test']['groups']['visible_difference']['context_diff']['metrics']
    overlap = comparisons['test:all']['context_diff_minus_losver']
    lines = ['# 历史 Test 错误诊断', '',
        '仅复用既有预测，不训练，不改变历史 checkpoint、阈值或轮次；完整 pair_key 与 target 连接，并核对 sample_key。',
        '本节不能作为本轮结构或参数选择依据。本轮零局部头/直接加法设计由先前 Train/Valid 融合近零和首步梯度阻断支持。',
        f"完整 Test 为 {context_test['pairs']} 对，Context + Diff 正确 {context_test['pair_counts']['p_c']} 对（{context_test['p_c']:.2%}）；相对 LOSVER 新增/丢失正确对 {overlap['gains']}/{overlap['losses']}，历史 Valid 的优势未保持。",
        f"其中 {visible_test['pairs']} 对具有可见差异：P-C/P-V/P-B/P-R 分别为 "+'/'.join(str(visible_test['pair_counts'][k]) for k in ('p_c','p_v','p_b','p_r'))+'；输入不可辨识不能解释其余这些错误。',
        'P-C：漏洞侧判正、修复侧判负；P-V：两侧均正；P-B：两侧均负；P-R：方向完全相反。',
        '行号从零开始。可见性按正式 512 Token ID 比较，另核对物理行 Token 成员；不删除不可辨识配对。',
        '可见差异只表示编码输入不同，不保证关键修复语义完整可见；截断与源码抽取异常在案例中另行核实。',
        'Test 缺少原始 Logit 的模型由冻结原函数头补齐；已保存的双分支分量按 FP32 相加恢复。Train/Valid 未保存 Logit 的非饱和项用概率逆变换，饱和项冻结补查。原预测和指标保持不变，不裁剪概率。',
        '成对排序仍按保存概率计算；补查的 Logit 差距只解释饱和，不改变历史评价口径。',
        '不同模型 Logit 尺度不同，平均差距不直接作为性能排序。',
        '历史模型训练顺序与辅助任务不同，跨模型得失仅描述错误重合；本轮新的三组 Train/Valid 才是匹配训练对照。', '',
        '| Split | 可见组 | 方法 | 对数 | P-C | P-V | P-B | P-R | 排序正确率 | 平均 Logit 差 | 差距 Unknown |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split, d in result['splits'].items():
        for group in ('all','visible_difference','identical_input'):
            for name, data in d['groups'][group].items():
                m = data['metrics']; gap = data['logit_gap']
                gap_text = f"{gap['mean']:.4f}" if gap else 'Unknown'
                lines.append(f"| {split} | {group} | {name} | {m['pairs']} | {m['p_c']:.2%} | {m['p_v']:.2%} | {m['p_b']:.2%} | {m['p_r']:.2%} | {m['pair_ranking_accuracy']:.2%} | {gap_text} | {data['gap_unknown_due_to_saturation']} |")
    names = list(paths)
    lines += ['', '## Test 正确配对的相互得失', '',
        '每格为行方法相对列方法的新增/丢失正确对；保留完整 564 对分母。', '',
        '| 方法 | '+' | '.join(names)+' |', '|---|'+'---:|'*len(names)]
    for a in names:
        cells = []
        for b in names:
            c = comparisons['test:all'].get(a+'_minus_'+b)
            cells.append(f"{c['gains']}/{c['losses']}" if c else '—')
        lines.append('| '+a+' | '+' | '.join(cells)+' |')
    lines += ['', 'Train 中无保存预测的 UniXcoder/LOSVER 不重训、不补造；Train 机制核对使用已有 Context/RCCL/插入/双分支结果。',
        '完整分组、错误类型内排序方向、Logit 分布及配对转移见 history.summary.json、history.pairs.jsonl 和 history.comparisons.json。',
        '局部行高分和 Diff 重叠仅为诊断证据，不等价于行漏洞真值；编码信息缺失与当前线性读出失效必须区分。']
    capacity_path = root/'history.line_capacity.json'
    if capacity_path.exists():
        capacity = json.loads(capacity_path.read_text())
        lines += ['', '## 冻结行表示与当前线性分类器的容量', '',
            '逐函数允许标签知情的最优权重，只用于区分冻结读出限制和权重选择问题。它不是可部署模型；相同输入的配对仍不能分别选择不同权重。',
            '| Split | 可见组 | 函数错误 | 1–5倍也无法纠正 | 任意凸聚合也无法纠正 | 1–5倍存在可纠正权重 |',
            '|---|---|---:|---:|---:|---:|']
        for split, groups in capacity['splits'].items():
            for name, v in groups.items():
                lines.append(f"| {split} | {name} | {v['wrong_functions']} | {v['bounded_cannot_fix']} | {v['any_convex_cannot_fix']} | {v['bounded_can_fix']} |")
        lines += ['', '任意凸聚合也无法纠正，只能说明该冻结线性读出缺少正确方向，不能证明编码器完全没有信息；有可行权重也不证明正常推理能学会选择。']
    plan = json.loads((root/'history.cases.json').read_text())
    lines += ['', '## 源码与真实 Diff 复核', '']
    for key, review in plan.get('source_reviews', {}).items():
        evidence = review.get('frozen_evidence', '')
        lines.append(f"- {key}: {review['observation']} {evidence} 机理判断：{review['security_mechanism']}。")
    if plan.get('cases'):
        lines += ['', '每个预选样本的全局行投影范围、1–5倍极值、修改行分数、注意力前三行和历史局部头两类前三行见 history.cases.json。',
            '这些样本的逐行资料是历史诊断，不作为新局部头的标签、特征或特判。']
    (root/'history.report.md').write_text('\n'.join(lines)+'\n')
    return result


@torch.no_grad()
def historical_line_diagnosis(data_dir, output_dir, device):
    """Frozen historical Context+Diff Test lines, no model or threshold changes.

    Train/Valid reuse the previous frozen caches. Bounds are label-informed
    diagnostics of the SAME linear readout, not inference alternatives.
    """
    from .evaluation import load_paired
    from .training import paired_loader, paired_forward
    from .mil_diagnosis import pooling_interventions
    from .model import resolve_device
    torch.set_num_threads(4)
    root = Path(output_dir)
    if not (root/'history.summary.json').exists():
        raise ValueError('run the saved-prediction history diagnosis first')
    run = Path('results/attention_context_512/context_diff/primevul_paired/unixcoder')
    cp = torch.load(run/'best-p_c.pt', map_location='cpu', weights_only=False)
    device = resolve_device(device)
    model, tokenizer = load_paired(cp, device)
    rows = read_jsonl(Path(data_dir)/'primevul_paired/test.oracle.jsonl')
    if len(rows) != 1128:
        raise ValueError('complete official historical Test required')
    saved = align_predictions(rows, read_jsonl(run/'test.predictions.jsonl'))
    loader = paired_loader(rows, tokenizer, 'baseline')
    records = []; at = 0
    for batch in loader:
        probabilities, details = paired_forward(model, batch, 'baseline', device, with_labels=False,
                                                return_attention=True, return_line_states=True)
        z = model.classifier(details['line_states']).squeeze(-1)
        for i in range(len(probabilities)):
            row = rows[at+i]; f = loader.dataset[at+i]
            n = len(f['line_numbers'])
            line_z, rates, counts = [v[i,:n].cpu() for v in (z, details['line_rates'], details['line_counts'])]
            scores, stats = pooling_interventions(line_z, rates, counts)
            probability = float(probabilities[i,0])
            if abs(probability-saved[at+i]['probability']) > 2e-6:
                raise AssertionError('restored historical prediction differs from saved Test')
            if (probability>.5) != (saved[at+i]['probability']>.5):
                raise AssertionError('historical classification differs')
            if abs(float(scores['original'].sigmoid())-probability) > 2e-6:
                raise AssertionError('line readout decomposition differs')
            records.append(dict(sample_key=row['sample_key'], pair_key=row['pair_key'], target=row['target'],
                probability=saved[at+i]['probability'], logit=float(details['logits'][i]),
                line_numbers=f['line_numbers'], line_logits=line_z.tolist(), line_rates=rates.tolist(),
                line_token_counts=counts.tolist(), modified_lines=row['lines_ground_truth'],
                visible_inputs_differ=(loader.dataset[(at+i)//2*2]['input_ids'] != loader.dataset[(at+i)//2*2+1]['input_ids']),
                **stats))
        at += len(probabilities)
    write_jsonl(root/'history.context_test_lines.jsonl', records)
    summary = dict(role='retrospective historical diagnosis only', checkpoint=str(run/'best-p_c.pt'),
                   selected_epoch=cp['selected_epoch'], splits={})
    for split in ('train','valid','test'):
        data = (records if split == 'test' else
                read_jsonl(Path('results/mil_research/context_diff')/f'{split}.lines.jsonl'))
        summary['splits'][split] = {}
        for group in ('all', 'visible_difference', 'identical_input'):
            selected = [r for r in data if group=='all' or r['visible_inputs_differ']==(group=='visible_difference')]
            def probability(r):
                return r['probability'] if 'probability' in r else r['probabilities']['original']
            wrong = [r for r in selected if (probability(r)>.5) != bool(r['target'])]
            summary['splits'][split][group] = dict(
                functions=len(selected), wrong_functions=len(wrong),
                bounded_cannot_fix=sum(r['bounded_max_logit']<=0 if r['target'] else r['bounded_min_logit']>0 for r in wrong),
                any_convex_cannot_fix=sum(r['max_line_logit']<=0 if r['target'] else r['min_line_logit']>0 for r in wrong),
                bounded_can_fix=sum(r['bounded_max_logit']>0 if r['target'] else r['bounded_min_logit']<=0 for r in wrong))
    summary['interpretation'] = ('A bound that cannot cross zero excludes reweighting under this frozen classifier. '
        'It does not prove the encoder contains no useful information. A feasible bound is not evidence that a source-only learner can choose it.')
    write_json(root/'history.line_capacity.json', summary)
    del model, details, z
    torch.cuda.empty_cache()
    recover_historical_logits(data_dir, root, device)
    historical_cases(data_dir, root)
    print(json.dumps(summary), flush=True)
    return summary


@torch.no_grad()
def recover_historical_logits(data_dir, output_dir, device):
    """Recover raw Test logits and saturated Train/Valid margins, no retraining.

    Saved probabilities and all classification statistics are retained. Only
    logit-gap statistics change; no choice is made from these historical scores.
    """
    from .evaluation import load_paired
    from .training import paired_rows, paired_loader, paired_forward
    root = Path(output_dir)
    pairs = read_jsonl(root/'history.pairs.jsonl')
    summary = json.loads((root/'history.summary.json').read_text())
    context = {(r['pair_key'],r['target']):r['logit'] for r in read_jsonl(root/'history.context_test_lines.jsonl')}
    for row in pairs:
        for name, data in row['methods'].items():
            if row['split']=='test' and name=='context_diff':
                data['logits'] = [context[(row['pair_key'],y)] for y in (1,0)]
                data['logit_sources'] = ['frozen_function_head']*2
            elif data['logit_sources']==['exported_components']*2:
                data['logits'] = [float(np.float32(z)) for z in data['logits']]
            if all(z is not None for z in data['logits']):
                data['logit_gap'] = data['logits'][0]-data['logits'][1]
    recovered = []
    for name, directory in summary['prediction_sources'].items():
        needed = {r['pair_key'] for r in pairs if name in r['methods'] and (
            r['methods'][name]['logit_gap'] is None or (r['split']=='test' and
            'inverse_saved_probability' in r['methods'][name]['logit_sources']))}
        if not needed: continue
        run = Path(directory)
        cp = torch.load(run/'best-p_c.pt', map_location='cpu', weights_only=False)
        model, tokenizer = load_paired(cp, device)
        logits = []
        handle = model.classifier.register_forward_hook(lambda _m, _i, output: logits.append(output.detach().cpu()))
        for split in ('train','valid','test'):
            if not any(r['split']==split and r['pair_key'] in needed for r in pairs):
                continue
            official = paired_rows(data_dir, run.parents[1], cp['variant'], split, method=run.name)
            batch_size = cp['training_config']['eval_batch_size']
            selected_batches = {i//batch_size for i,r in enumerate(official) if r['pair_key'] in needed}
            # Retain original evaluation companions and shapes, including the
            # final partial batch, rather than repacking selected examples.
            rows = [r for i,r in enumerate(official) if i//batch_size in selected_batches]
            if not rows: continue
            loader = paired_loader(rows, tokenizer, cp['variant'], batch_size=batch_size,
                                   insertion_supervision=model.insertion_supervision)
            at = 0; actual = {}
            for batch in loader:
                logits.clear()
                probability = paired_forward(model, batch, cp['variant'], device, with_labels=False).cpu().flatten().tolist()
                if len(logits) != 1 or logits[0].numel() != len(probability):
                    raise AssertionError('historical function head must run exactly once')
                for row, z, p in zip(rows[at:at+len(probability)], logits[0].flatten().tolist(), probability):
                    actual[(row['pair_key'], row['target'])] = (z, p)
                    recovered.append(dict(method=name, split=split, pair_key=row['pair_key'], sample_key=row['sample_key'],
                                          target=row['target'], logit=z, restored_probability=p))
                at += len(probability)
            for row in pairs:
                if row['split'] != split or row['pair_key'] not in needed: continue
                data = row['methods'][name]
                outputs = [actual[(row['pair_key'], y)] for y in (1,0)]
                if any(abs(p-old)>2e-6 or (p>.5)!=(old>.5) for (_,p),old in zip(outputs,data['probabilities'])):
                    raise AssertionError('frozen recovery differs from historical saved prediction')
                data.update(logits=[a[0] for a in outputs], logit_sources=['frozen_function_head']*2,
                            logit_gap=outputs[0][0]-outputs[1][0])
        handle.remove(); del model, cp
        torch.cuda.empty_cache()
    for split, item in summary['splits'].items():
        for group, methods in item['groups'].items():
            selected = [r for r in pairs if r['split']==split and (group=='all' or
                (group=='visible_difference' and r['visible_difference']) or
                (group=='identical_input' and not r['visible_difference']) or
                (group=='pure_addition' and not r['modified_lines'][0] and r['modified_lines'][1]))]
            for name, data in methods.items():
                gaps = [r['methods'][name]['logit_gap'] for r in selected if r['methods'][name]['logit_gap'] is not None]
                data['logit_gap'] = distribution(gaps) if gaps else None
                data['gap_unknown_due_to_saturation'] = len(selected)-len(gaps)
                data['logit_gap_sign_counts'] = dict(positive=sum(z>0 for z in gaps),
                    negative=sum(z<0 for z in gaps), zero=sum(z==0 for z in gaps))
    summary['logit_recovery'] = dict(functions=len(recovered), role='frozen heads for missing raw Test logits and saturated Train/Valid gaps; original batch companions and classification retained')
    write_json(root/'history.summary.json', summary)
    write_jsonl(root/'history.pairs.jsonl', pairs)
    write_jsonl(root/'history.recovered_logits.jsonl', recovered)
    render_historical_errors(root)


def historical_cases(data_dir, output_dir):
    """Persist line evidence for the preselected historical examples, no source dump."""
    from .data import _line_diff
    root = Path(output_dir)
    plan = json.loads((root/'history.cases.json').read_text())
    keys = set(plan['pair_keys'])
    rows = read_jsonl(Path(data_dir)/'primevul_paired/test.oracle.jsonl')
    all_pairs = {r['pair_key']:r for r in read_jsonl(root/'history.pairs.jsonl') if r['split']=='test'}
    frozen = {(r['pair_key'],r['target']):r for r in read_jsonl(root/'history.context_test_lines.jsonl')}
    local = {name: {(p['pair_key'],p['target']):p for p in read_jsonl(
        Path(f'results/global_local_mil_512/seed123456/{name}/primevul_paired/unixcoder/test.predictions.jsonl'))}
        for name in ('mean','lse')}
    cases = []
    for i in range(0,len(rows),2):
        old, new = rows[i:i+2]
        if old['pair_key'] not in keys: continue
        deleted, _, alignment = _line_diff(old['func'],new['func'],return_alignment=True)
        if deleted != old['lines_ground_truth'] or _line_diff(new['func'],old['func'])[0] != new['lines_ground_truth']:
            raise AssertionError('stored Diff and actual source alignment differ')
        r = dict(all_pairs[old['pair_key']], source_line_counts=[len(x['func'].splitlines()) for x in (old,new)],
                 edit_blocks=alignment['blocks'], sides=[])
        for row in (old,new):
            f = frozen[(row['pair_key'],row['target'])]
            mass = np.asarray(f['line_token_counts'])*np.asarray(f['line_rates']);mass/=mass.sum()
            side = dict(target=row['target'], line_bounds={k:f[k] for k in
                ('min_line_logit','max_line_logit','bounded_min_logit','bounded_max_logit')},
                attention_top3=[dict(line=f['line_numbers'][j], mass=float(mass[j]),rate=f['line_rates'][j],
                                    logit=f['line_logits'][j]) for j in np.argsort(-mass,kind='stable')[:3]],
                diff_lines=[dict(line=n, logit=z, rate=v, mass=float(a))
                            for n,z,v,a in zip(f['line_numbers'],f['line_logits'],f['line_rates'],mass)
                            if n in row['lines_ground_truth']], local={})
            for name, pred in local.items():
                p=pred[(row['pair_key'],row['target'])]; scores=np.asarray(p['local_line_scores'])
                side['local'][name] = {k:p[k] for k in ('global_logit','local_logit','local_residual','local_fusion')}
                for c, title in enumerate(('safe_top3','vulnerable_top3')):
                    side['local'][name][title] = [dict(line=p['line_numbers'][j], score=float(scores[j,c]))
                                                 for j in np.argsort(-scores[:,c],kind='stable')[:3]]
            r['sides'].append(side)
        cases.append(r)
    plan.update(cases=cases, interpretation='frozen linear readout bounds do not prove absence of encoder information; local scores are latent evidence')
    write_json(root/'history.cases.json',plan)
    render_historical_errors(root)


def cohort_indices(rows, features):
    groups = {k: [] for k in ('all', 'pure_addition', 'replacement', 'pure_deletion',
                              'no_changed_lines', 'visible_difference', 'identical_input',
                              'lines_1_20', 'lines_21_40', 'lines_41_plus')}
    for i in range(0, len(rows), 2):
        a, b = rows[i:i+2]
        if a['target'] != 1 or b['target'] != 0 or a['pair_key'] != b['pair_key']:
            raise ValueError('official vulnerable/patched adjacency required')
        kind = ('replacement' if a['lines_ground_truth'] and b['lines_ground_truth'] else
                'pure_deletion' if a['lines_ground_truth'] else
                'pure_addition' if b['lines_ground_truth'] else 'no_changed_lines')
        different = features[i]['input_ids'] != features[i+1]['input_ids']
        length = max(len(f['line_numbers']) for f in features[i:i+2])
        size = 'lines_1_20' if length <= 20 else 'lines_21_40' if length <= 40 else 'lines_41_plus'
        for name in ('all', kind, 'visible_difference' if different else 'identical_input', size):
            groups[name].extend((i, i+1))
    return groups


def branch_summary(rows, predictions):
    if not predictions or 'local_logit' not in predictions[0]:
        return {}
    global_p = [p['global_probability'] for p in predictions]
    local_p = [p['scaled_local_probability'] for p in predictions]
    combined = [p['probability'] for p in predictions]
    patched = [i for i, r in enumerate(rows) if r['target'] == 0]
    vulnerable = [i for i, r in enumerate(rows) if r['target'] == 1]
    out = dict(fusion=predictions[0]['local_fusion'],
               global_only_metrics=paired_metrics(rows, global_p),
               scaled_local_only_metrics=paired_metrics(rows, local_p),
               fused_vs_own_global=pair_changes(rows, combined, global_p),
               introduced_patched_false_positives=sum(global_p[i] <= .5 < combined[i] for i in patched),
               removed_patched_false_positives=sum(combined[i] <= .5 < global_p[i] for i in patched),
               recovered_vulnerable_true_positives=sum(global_p[i] <= .5 < combined[i] for i in vulnerable),
               introduced_vulnerable_false_negatives=sum(combined[i] <= .5 < global_p[i] for i in vulnerable),
               function_decision_changes=sum((a > .5) != (b > .5) for a, b in zip(combined, global_p)),
               absolute_residual=distribution([abs(p['local_residual']) for p in predictions]),
               maximum_probability_change=max(abs(a-b) for a, b in zip(combined, global_p)),
               by_target={})
    for y in (0, 1):
        ix = [i for i, r in enumerate(rows) if r['target'] == y]
        out['by_target'][str(y)] = {
            key: distribution([predictions[i][key] for i in ix])
            for key in ('global_logit', 'local_logit', 'local_residual')}
        for c, name in enumerate(('safe_score', 'vulnerable_score')):
            out['by_target'][str(y)][name] = distribution([predictions[i]['local_class_scores'][c] for i in ix])
    out['interpretation'] = 'latent bag-level class evidence, not calibrated per-line risk; version 2 uses direct additive residual, version 1 a learned scalar'
    # Measure how much the actual reduction departs from a line mean on the
    # SAME head scores. This is a score statistic, not another inference model.
    within_line_std = np.asarray([np.std(p['local_line_scores'], axis=0) for p in predictions])
    margins = [np.mean(np.asarray(p['local_line_scores'])[:, 1] - np.asarray(p['local_line_scores'])[:, 0])
               for p in predictions]
    out['within_function_line_score_std'] = {
        name: distribution(within_line_std[:, c]) for c, name in enumerate(('safe', 'vulnerable'))}
    out['reduction_minus_line_mean_margin'] = distribution([p['local_logit']-z for p, z in zip(predictions, margins)])
    global_z = np.asarray([p['global_logit'] for p in predictions])
    residual = np.asarray([p['local_residual'] for p in predictions])
    final_z = np.asarray([p.get('logit', g+r) for p,g,r in zip(predictions,global_z,residual)])
    targets = np.asarray([r['target'] for r in rows])
    out['function_bce'] = {name: float(np.mean(np.logaddexp(0,z)-targets*z))
                           for name,z in (('global_only',global_z),('fused',final_z))}
    out['global_residual_correlation'] = (float(np.corrcoef(global_z,residual)[0,1])
        if global_z.std()>0 and residual.std()>0 else None)
    global_gap = global_z[::2]-global_z[1::2]
    local_gap = residual[::2]-residual[1::2]
    gp = np.asarray(global_p); fused = np.asarray(combined)
    out['pair_logit_gaps'] = dict(global_branch=distribution(global_gap), local_branch=distribution(local_gap),
        fused=distribution(global_gap+local_gap),
        local_correct_order_pairs=int((local_gap>0).sum()), local_tied_pairs=int((local_gap==0).sum()),
        corrected_global_ranking=int(((gp[::2]<=gp[1::2]) & (fused[::2]>fused[1::2])).sum()),
        broken_global_ranking=int(((gp[::2]>gp[1::2]) & (fused[::2]<=fused[1::2])).sum()))
    return out


def check(data_dir, output_dir, model_path, device):
    """Real pretrained encoder, original forward, two optimization steps, disk restore."""
    from . import model as core
    from .evaluation import load_paired
    from .training import paired_loader, paired_collate, paired_forward, LOSVERAdamW

    device = core.resolve_device(device)
    if device.type != 'cuda':
        raise ValueError('dual-stream smoke requires a real GPU')
    torch.set_num_threads(4)
    rows = read_jsonl(Path(data_dir)/'primevul_paired/train.lines.jsonl')[:8]
    options = dict(bounded_attention=True, attention_context=True, attention_region='diff')
    core.seed_everything(123456)
    baseline, tokenizer = core.losver_model(model_path, 'baseline', **options)
    rng = torch.get_rng_state().clone()
    order = torch.randperm(100)
    loader = paired_loader(rows, tokenizer, 'baseline')
    b = paired_collate(loader.dataset)
    baseline.to(device).eval()
    with torch.no_grad():
        expected = paired_forward(baseline, b, 'baseline', device, with_labels=False).cpu()
    baseline.cpu(); torch.cuda.empty_cache()
    heads = []; results = {}
    for mode in ('mean', 'lse'):
        core.seed_everything(123456)
        m, _ = core.losver_model(model_path, 'baseline', **options, local_mil=mode)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        torch.testing.assert_close(torch.randperm(100), order, atol=0, rtol=0)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(m.state_dict()[name], value, atol=0, rtol=0)
        heads.append({k: v.clone() for k, v in m.local_classifier.state_dict().items()})
        m.to(device).eval()
        calls = []
        handle = m.encoder.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            p, d = paired_forward(m, b, 'baseline', device, with_labels=False, return_attention=True)
        handle.remove()
        assert calls == [1]
        torch.testing.assert_close(p.cpu(), expected, atol=0, rtol=0)
        torch.testing.assert_close(d['line_counts'].sum(-1), b['code_mask'].sum(-1).to(device).float())
        assert not d['local_residual'].any()
        added = sum(p.numel() for p in m.local_classifier.parameters())
        before = m.local_classifier.weight.detach().clone()
        opt = LOSVERAdamW(m.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.)
        steps = []
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(2):
            m.train()
            loss, _, d = paired_forward(m, b, 'baseline', device, return_attention=True)
            torch.testing.assert_close(loss, d['bce']+.1*d['region_loss'])
            loss.backward()
            head_grad = float(m.local_classifier.weight.grad.abs().sum())
            encoder_grad = float(m.encoder.embeddings.word_embeddings.weight.grad.abs().sum())
            assert head_grad > 0 and encoder_grad > 0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
            opt.step(); opt.zero_grad()
            m.eval()
            with torch.no_grad():
                _, after = paired_forward(m, b, 'baseline', device, with_labels=False, return_attention=True)
            steps.append(dict(loss=float(loss.detach()), head_gradient_l1=head_grad,
                              encoder_gradient_l1=encoder_grad,
                              head_weight_norm=float(m.local_classifier.weight.detach().norm()),
                              residual_abs_mean=float(after['local_residual'].abs().mean()),
                              maximum_probability_change=float((after['logits'].sigmoid()-after['global_logits'].sigmoid()).abs().max()),
                              function_decision_changes=int(((after['logits']>0)!=(after['global_logits']>0)).sum())))
            assert after['local_residual'].abs().sum() > 0
        assert not torch.equal(before, m.local_classifier.weight)
        m.eval()
        source = dict(func=rows[0]['func'], target=0)
        one = paired_collate([paired_features(source, tokenizer, 'baseline')])
        for key in ('input_ids', 'attention_mask', 'code_mask', 'token_weights'):
            torch.testing.assert_close(b[key][0], one[key][0], atol=0, rtol=0)
        with torch.no_grad():
            p = paired_forward(m, b, 'baseline', device, with_labels=False)
            q = paired_forward(m, one, 'baseline', device, with_labels=False)
            torch.testing.assert_close(p[:1], q, atol=2e-6, rtol=0)
            poisoned = dict(b, line_labels=torch.full_like(b['line_labels'], -999), labels=1-b['labels'])
            torch.testing.assert_close(p, paired_forward(m, poisoned, 'baseline', device, with_labels=False), atol=0, rtol=0)
        cp = dict(protocol='losver', model_path=str(model_path), variant='baseline', input_mode='512',
                  line_mapping='source_offsets', pooling='bounded_attention',
                  bounded_attention=dict(context=True, region_supervision='diff'), local_mil=dict(pooling=mode, version=2))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pt'
            torch.save(dict(cp, state_dict={k: v.cpu() for k, v in m.state_dict().items()}), path)
            del opt, d, before, m
            torch.cuda.empty_cache()
            restored, _ = load_paired(torch.load(path, map_location='cpu', weights_only=False), device)
            with torch.no_grad():
                actual = paired_forward(restored, one, 'baseline', device, with_labels=False)
            torch.testing.assert_close(actual, q, atol=0, rtol=0)
            del restored
        results[mode] = dict(steps=steps, added_parameters=added, initialization_equal=True,
                             rng_and_order_equal=True, one_encoding=True, checkpoint_restored=True,
                             inference_ignores_diff=True, single_function_delta=float((p[:1]-q).abs().max()),
                             peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))
        torch.cuda.empty_cache()
    for k in heads[0]:
        torch.testing.assert_close(heads[0][k], heads[1][k], atol=0, rtol=0)
    result = dict(test_used=False, seed=123456, optimizer='LOSVERAdamW lr=2e-5 eps=1e-8 clip=1; two smoke steps without scheduler',
                  device=torch.cuda.get_device_name(device),
                  local_initialization_equal=True, results=results)
    write_json(Path(output_dir)/'gpu-smoke.json', result)
    print(json.dumps(result), flush=True)
    return result


def analyze(run_dir, data_dir, device):
    """Selected checkpoint, complete Train/Valid; labels used only after inference."""
    from .evaluation import load_paired, validate_paired, bounded_region_record
    from .training import paired_loader, paired_collate, paired_forward
    from . import model as core

    torch.set_num_threads(4)
    run = Path(run_dir)
    complete = json.loads((run/'training.metrics.json').read_text())
    if complete['completed_epochs'] != 16 or complete['optimizer_steps'] != 17424:
        raise ValueError('finish all 16 epochs before selected-checkpoint analysis')
    cp = torch.load(run/'best-p_c.pt', map_location='cpu', weights_only=False)
    model, tokenizer = load_paired(cp, core.resolve_device(device))
    result = dict(test_used=False, selected_epoch=cp['selected_epoch'], splits={})
    if model.local_mil is not None:
        result['selected_local_head'] = dict(
            fusion=float(model.local_fusion.detach()),
            weight_norm=float(model.local_classifier.weight.detach().norm()),
            class_weight_norms=model.local_classifier.weight.detach().norm(dim=1).cpu().tolist(),
            bias=model.local_classifier.bias.detach().cpu().tolist(),
        )
    for split, count in (('train', 8708), ('valid', 1124)):
        rows = read_jsonl(Path(data_dir)/f'primevul_paired/{split}.lines.jsonl')
        if len(rows) != count:
            raise ValueError('complete official members required')
        loader = paired_loader(rows, tokenizer, 'baseline')
        metrics, predictions = validate_paired(model, rows, tokenizer, 'baseline', core.resolve_device(device), loader=loader)
        if split == 'valid':
            saved = read_jsonl(run/'valid.best-p_c.predictions.jsonl')
            assert [(p['sample_key'], p['probability']) for p in predictions] == [(p['sample_key'], p['probability']) for p in saved]
        indices = cohort_indices(rows, loader.dataset)
        regions = [bounded_region_record(r, p) for r, p in zip(rows, predictions)]
        # Follow the original visible-input definition used in mil_research.
        summary = dict(metrics=metrics, cohorts={}, branches=branch_summary(rows, predictions), region_attention={})
        for name, ix in indices.items():
            if ix:
                summary['cohorts'][name] = paired_metrics([rows[i] for i in ix], [predictions[i]['probability'] for i in ix])
        for y in (0, 1):
            values = [r['mean_rate_ratio'] for r in regions if r['target'] == y and r['eligible']]
            if values:
                summary['region_attention'][str(y)] = distribution(values)
        with torch.no_grad():
            b = paired_collate(loader.dataset[:8])
            probs = paired_forward(model, b, 'baseline', core.resolve_device(device), with_labels=False)
            one = paired_collate([paired_features(dict(func=rows[0]['func'], target=0), tokenizer, 'baseline')])
            one_p = paired_forward(model, one, 'baseline', core.resolve_device(device), with_labels=False)
        summary['single_function_check'] = dict(tolerance=2e-6, probability_delta=float((probs[:1]-one_p).abs().max()),
                                                passed=bool(torch.allclose(probs[:1], one_p, atol=2e-6, rtol=0)),
                                                decision_equal=bool(torch.equal(probs[:1]>.5, one_p>.5)))
        if not summary['single_function_check']['decision_equal']:
            raise AssertionError('single-source classification decision differs')
        # Valid was already exported by the formal checkpoint selection path;
        # the restored predictions above are checked against that canonical file.
        if split == 'train':
            write_jsonl(run/'train.selected.predictions.jsonl', predictions)
        write_jsonl(run/f'{split}.region_attention.jsonl', regions)
        result['splits'][split] = summary
    write_json(run/'branch_diagnostics.json', result)
    print(json.dumps(result), flush=True)
    return result


def report(output_dir, data_dir, include_test=False):
    """All three fixed controls, exact cohort comparison and predeclared seed rule."""
    root = Path(output_dir)
    experiment = json.loads((root/'experiment.json').read_text())
    if include_test:
        decision = json.loads((root/'comparison.json').read_text())
        if decision['promote_to_second_seed'] and not all(
            (root/'seed123457'/m/'primevul_paired/unixcoder/training.metrics.json').exists()
            for m in ('baseline', 'mean', 'lse')
        ):
            raise ValueError('complete the three second-seed controls before Test reporting')
    rows = read_jsonl(Path(data_dir)/'primevul_paired/valid.lines.jsonl')
    test_rows = read_jsonl(Path(data_dir)/'primevul_paired/test.jsonl') if include_test else None
    results = dict(test_used=include_test, seeds={})
    cases = []
    for seed in (123456, 123457):
        seed_root = root/f'seed{seed}'
        if not seed_root.exists():
            continue
        configs = {}; all_predictions = {}; test_predictions = {}; item = dict(models={}, comparisons={})
        for method in ('baseline', 'mean', 'lse'):
            run = seed_root/method/'primevul_paired/unixcoder'
            complete = json.loads((run/'training.metrics.json').read_text())
            c = json.loads((run/'config.json').read_text()); configs[method] = c
            if c.get('local_mil', {}).get('pooling') != (None if method == 'baseline' else method):
                raise ValueError('control path and actual local pooling differ')
            if complete['completed_epochs'] != 16 or complete['optimizer_steps'] != 17424:
                raise ValueError('all three runs must finish before a decision')
            events = read_jsonl(run/'best.training.jsonl')
            epochs = [e for e in events if e['event'] == 'epoch']
            selected = max(epochs, key=lambda e: round(e['validation']['p_c'], 4))['epoch']
            if selected != complete['selected_epochs']['p_c']:
                raise ValueError('Valid P-C selection differs')
            pred = read_jsonl(run/'valid.best-p_c.predictions.jsonl')
            if [(p['sample_key'], p['target']) for p in pred] != [(r['sample_key'], r['target']) for r in rows]:
                raise ValueError('Valid members/hard labels differ')
            all_predictions[method] = [p['probability'] for p in pred]
            item['models'][method] = dict(epoch=selected, metrics=paired_metrics(rows, all_predictions[method]),
                training_seconds=sum(e['training_seconds'] for e in epochs), parameters=c['parameters'],
                diagnostics=json.loads((run/'branch_diagnostics.json').read_text()),
                local_training=[dict(epoch=e['epoch'], step=e['step'], **e['local_mil'])
                                for e in events if e['event'] == 'step' and 'local_mil' in e],
                local_validation=[dict(epoch=e['epoch'], **e['validation']['local_mil'])
                                  for e in epochs if 'local_mil' in e['validation']],
                first_step_loss=next(e['loss'] for e in events if e['event'] == 'step'))
            if include_test:
                test = json.loads((run/'test.metrics.json').read_text())
                ps = read_jsonl(run/'test.predictions.jsonl')
                if [(p['sample_key'], p['target']) for p in ps] != [(r['sample_key'], r['target']) for r in test_rows]:
                    raise ValueError('Test members/hard labels differ from the complete official split')
                recomputed = paired_metrics(test_rows, [p['probability'] for p in ps])
                if any(test['metrics'][k] != v for k, v in recomputed.items()) or test['selected_epoch'] != selected:
                    raise ValueError('Test scores/epoch differ from the fixed selected checkpoint')
                item['models'][method]['test'] = test
                item['models'][method]['test_branches'] = branch_summary(test_rows, ps)
                test_predictions[method] = [p['probability'] for p in ps]
        for m in ('mean', 'lse'):
            for k in ('training_members', 'validation_members', 'training_config', 'input_coverage',
                      'bounded_attention', 'model_path', 'line_mapping', 'input_mode',
                      'supervision', 'pooling', 'variant', 'precision',
                      'decision_threshold', 'comparison'):
                if configs[m][k] != configs['baseline'][k]:
                    raise ValueError(f'control differs: {m}.{k}')
        if configs['mean']['parameters'] != configs['lse']['parameters']:
            raise ValueError('Mean/LSE parameter counts differ')
        if len({d['first_step_loss'] for d in item['models'].values()}) != 1:
            raise ValueError('first-step loss differs despite zero residual initialization')
        for a, b in (('mean', 'baseline'), ('lse', 'baseline'), ('lse', 'mean')):
            item['comparisons'][a+'_minus_'+b] = dict(
                **pair_changes(rows, all_predictions[a], all_predictions[b]),
                deltas={k: item['models'][a]['metrics'][k]-item['models'][b]['metrics'][k]
                        for k in ('p_c', 'p_r', 'auc', 'mcc', 'f1', 'recall', 'patched_fpr', 'pair_ranking_accuracy')})
        if include_test:
            item['test_comparisons'] = {a+'_minus_'+b: pair_changes(test_rows, test_predictions[a], test_predictions[b])
                                        for a, b in (('mean', 'baseline'), ('lse', 'baseline'), ('lse', 'mean'))}
        states = [torch.load(seed_root/m/'primevul_paired/unixcoder/last.pt', map_location='cpu',
                             weights_only=False, mmap=True) for m in ('baseline', 'mean', 'lse')]
        item['rng_control'] = dict(cpu_equal=all(torch.equal(s['rng']['torch'], states[0]['rng']['torch']) for s in states),
                                   active_cuda_equal=all(torch.equal(s['rng']['cuda'][0], states[0]['rng']['cuda'][0]) for s in states))
        if not all(item['rng_control'].values()):
            raise ValueError('initialization/forward altered sampler or Dropout RNG consumption')
        results['seeds'][str(seed)] = item
        # Fixed prior cases plus first gains/losses; no claim that line scores are ground truth.
        selected_pairs = {'primevul_paired:valid:2', 'primevul_paired:valid:18', 'primevul_paired:valid:11'}
        for comp in item['comparisons'].values():
            selected_pairs.update(comp['gained_pairs'][:2]); selected_pairs.update(comp['lost_pairs'][:2])
        for method in ('mean', 'lse'):
            comp = item['models'][method]['diagnostics']['splits']['valid']['branches']['fused_vs_own_global']
            selected_pairs.update(comp['gained_pairs'][:2]); selected_pairs.update(comp['lost_pairs'][:2])
        for method in ('baseline', 'mean', 'lse'):
            pred = read_jsonl(seed_root/method/'primevul_paired/unixcoder/valid.best-p_c.predictions.jsonl')
            for r, p in zip(rows, pred):
                if r['pair_key'] not in selected_pairs:
                    continue
                record = dict(seed=seed, method=method, pair_key=r['pair_key'], sample_key=r['sample_key'],
                              project=r['project'], target=r['target'], probability=p['probability'],
                              modified_lines=r['lines_ground_truth'])
                if 'local_line_scores' in p:
                    record.update({k: p[k] for k in ('global_logit', 'local_logit', 'local_residual', 'local_fusion', 'local_class_scores')})
                    values = np.asarray(p['local_line_scores'])
                    for c, name in enumerate(('safe', 'vulnerable')):
                        ix = np.argsort(-values[:, c], kind='stable')[:3]
                        record[name+'_top3'] = [dict(line=p['line_numbers'][int(j)], score=float(values[j, c]),
                                                    modified=p['line_numbers'][int(j)] in r['lines_ground_truth']) for j in ix]
                cases.append(record)
    rule = experiment['second_seed_rule']
    comps = results['seeds']['123456']['comparisons']
    advance = all(comps['lse_minus_'+m]['deltas']['p_c'] >= rule['p_c_gain_min']
                  and comps['lse_minus_'+m]['deltas']['auc'] >= 0
                  and comps['lse_minus_'+m]['deltas']['mcc'] >= 0 for m in ('baseline', 'mean'))
    results.update(second_seed_rule=rule, promote_to_second_seed=bool(advance),
                   test_role='only after all Valid decisions; never structural/parameter/checkpoint selection')
    if include_test:
        previous = json.loads((root/'comparison.json').read_text())
        if previous['promote_to_second_seed'] != advance:
            raise ValueError('Valid-only decision changed after Test')
    write_json(root/'comparison.json', results)
    write_json(root/'cases.json', dict(cases=cases, interpretation='latent class scores are not line vulnerability labels'))
    fields = ['seed', 'method', 'epoch', 'p_c', 'p_v', 'p_b', 'p_r', 'auc', 'mcc', 'f1', 'recall', 'patched_fpr', 'pair_ranking_accuracy']
    with (root/'classification.csv').open('w') as h:
        w = csv.DictWriter(h, fieldnames=['split']+fields, lineterminator='\n'); w.writeheader()
        for seed, item in results['seeds'].items():
            for method, d in item['models'].items():
                w.writerow(dict(split='valid', seed=seed, method=method, epoch=d['epoch'], **{k: d['metrics'][k] for k in fields[3:]}))
                if include_test:
                    w.writerow(dict(split='test', seed=seed, method=method, epoch=d['epoch'], **{k: d['test']['metrics'][k] for k in fields[3:]}))
    additive = experiment.get('architecture', {}).get('version') == 2
    architecture_text = (
        '最终 logit = global_logit + local_class[1]-local_class[0]；局部 Linear 权重与偏置均为零，新增 1,538 个参数，没有可学习融合系数。'
        if additive else '最终 logit = global_logit + c*(local_class[1]-local_class[0])，c 无约束且零初始化。随机局部头在 CPU fork_rng 内初始化；新增 1,539 个参数。')
    lines = ['# Global–Local Dual-Stream MIL', '',
        '三组使用同一正式模型、数据读取、优化器与评价入口；训练均从同一预训练 UniXcoder 初始化。',
        '官方 paired Train 8,708 个函数，Valid 1,124 个函数；Seed123456、Batch8、512 Token、16 Epoch、17,424 步。',
        '按完整 Valid P-C 四位小数选模，同分保留较早 Epoch；固定 probability > 0.5。历史权重不覆盖。', '',
        '全局分支保留原 Context + Diff。局部分支为独立 Linear(768,2)，两个通道分别记作安全/漏洞证据；复用同一次编码的物理行均值。',
        'Mean 对有效行等权平均；LSE 使用 logsumexp(scores)-log(N)，温度固定 1。共同的 log(N) 在类别差值中抵消，保留它以便比较类别分数。',
        architecture_text,
        '仅使用最终函数 BCE + 0.1 原 Diff 区域损失。没有新区域标签、额外辅助任务或编码；推理只输入单函数。',
        '两个局部通道是包级潜在证据，不是经过校准的行漏洞概率。', '',
        '## 完整分类结果', '', '| Split | Seed | 方法 | Epoch | P-C | P-R | AUC | MCC | F1 | Recall | 修复侧 FPR | 配对排序正确率 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seed, item in results['seeds'].items():
        for method, d in item['models'].items():
            for split, metrics in [('Valid', d['metrics'])]+([('Test', d['test']['metrics'])] if include_test else []):
                lines.append(f"| {split} | {seed} | {method} | {d['epoch']} | {metrics['p_c']:.2%} | {metrics['p_r']:.2%} | {metrics['auc']:.4f} | {metrics['mcc']:.4f} | {metrics['f1']:.4f} | {metrics['recall']:.4f} | {metrics['patched_fpr']:.4f} | {metrics['pair_ranking_accuracy']:.4f} |")
    lines += ['', '## 匹配比较与残差贡献', '']
    for seed, item in results['seeds'].items():
        for name, c in item['comparisons'].items():
            lines.append(f"- Valid Seed{seed} {name}: 新增正确 {c['gains']} 对，丢失 {c['losses']} 对；P-C {c['deltas']['p_c']*100:+.2f} pp，AUC {c['deltas']['auc']:+.5f}，MCC {c['deltas']['mcc']:+.5f}。")
        if include_test:
            for name, c in item['test_comparisons'].items():
                lines.append(f"- Test Seed{seed} {name}: 新增正确 {c['gains']} 对，丢失 {c['losses']} 对。")
        for method in ('mean', 'lse'):
            d = item['models'][method]; b = d['diagnostics']['splits']['valid']['branches']
            delta = b['fused_vs_own_global']
            coefficient = '直接加法（系数固定为 1）' if additive else f"所选 c={b['fusion']:.8g}"
            lines.append(f"- Seed{seed} {method}: {coefficient}；同一权重、同一编码下 global-only P-C={b['global_only_metrics']['p_c']:.2%}，融合后={d['metrics']['p_c']:.2%}，新增/丢失={delta['gains']}/{delta['losses']}；局部残差新增修复侧误报 {b['introduced_patched_false_positives']} 个、消除 {b['removed_patched_false_positives']} 个。")
            lines.append(f"  漏洞侧恢复正确阳性 {b['recovered_vulnerable_true_positives']} 个、新增漏报 {b['introduced_vulnerable_false_negatives']} 个；平均绝对 logit 残差 {b['absolute_residual']['mean']:.8g}，最大概率变化 {b['maximum_probability_change']:.8g}。")
            if 'function_bce' in b:
                corr=b['global_residual_correlation']; corr_text=f'{corr:.4f}' if corr is not None else 'NA'
                lines.append(f"  Valid 局部项改变 {b['function_decision_changes']} 个函数类别；局部单支 AUC={b['scaled_local_only_metrics']['auc']:.4f}，全局 logit 与残差相关系数={corr_text}。按函数平均的 BCE：global-only={b['function_bce']['global_only']:.5f}，融合={b['function_bce']['fused']:.5f}。")
                gaps=b['pair_logit_gaps']
                lines.append(f"  Valid 平均成对 Logit 差：global={gaps['global_branch']['mean']:.5f}，local={gaps['local_branch']['mean']:.5f}，融合={gaps['fused']['mean']:.5f}；局部项纠正/破坏排序 {gaps['corrected_global_ranking']}/{gaps['broken_global_ranking']} 对。")
            train_branch=d['diagnostics']['splits']['train']['branches']
            lines.append(f"  Train 局部项改变 {train_branch['function_decision_changes']} 个函数类别，新增/丢失正确对 {train_branch['fused_vs_own_global']['gains']}/{train_branch['fused_vs_own_global']['losses']}。")
            if additive:
                first = d['local_training'][0]
                head = d['diagnostics']['selected_local_head']
                lines.append(f"  首步局部头梯度 L1={first['head_gradient_l1']:.6f}，所选局部头权重范数={head['weight_norm']:.6f}。原 warmup 首次更新的学习率为零，因此首步记录的权重范数为零；后续正常更新。")
            spread = b['within_function_line_score_std']
            reduction = b['reduction_minus_line_mean_margin']
            lines.append(f"  Valid 函数内安全/漏洞行分数标准差的均值={spread['safe']['mean']:.6f}/{spread['vulnerable']['mean']:.6f}；实际聚合 margin 减同头行均值 margin 的均值/标准差={reduction['mean']:.8g}/{reduction['std']:.8g}。这是同一组分数的统计分解，没有评价新的推理方案。")
            if include_test:
                tb = d['test_branches']
                lines.append(f"  固定 Test：局部项改变 {tb['function_decision_changes']} 个函数类别，相对同权重 global-only 新增/丢失正确对 {tb['fused_vs_own_global']['gains']}/{tb['fused_vs_own_global']['losses']}。")
        lines.append(f"- Seed{seed}: 三组首批损失相同，末态 CPU 与活动 CUDA RNG 一致；Mean/LSE 参数量一致。训练循环耗时（分钟）："+', '.join(f"{m}={d['training_seconds']/60:.2f}" for m, d in item['models'].items())+'。不含初始化、Tokenization 和离线诊断，不据此声称速度提升。')
    lines += ['', '## 修复类型、可见性与行数', '',
        '行数分组依据每对函数中较大的有效物理行数，分界固定为 20/40。完整成员始终参与总指标；相同可见输入不可区分。', '',
        '| Seed | 方法 | Valid 子集 | 对数 | P-C | AUC | MCC | F1 | Recall | FPR | 排序正确率 |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seed, item in results['seeds'].items():
        for method, d in item['models'].items():
            for group, v in d['diagnostics']['splits']['valid']['cohorts'].items():
                if group == 'all': continue
                lines.append(f"| {seed} | {method} | {group} | {v['pairs']} | {v['p_c']:.2%} | {v['auc']:.4f} | {v['mcc']:.4f} | {v['f1']:.4f} | {v['recall']:.4f} | {v['patched_fpr']:.4f} | {v['pair_ranking_accuracy']:.4f} |")
    lines += ['', '## 类别分数、融合系数与区域注意力', '',
              '每个 Epoch 的完整 Valid 类别分数均值/标准差和融合系数保存在 best.training.jsonl 及 comparison JSON 的 local_validation。',
              '训练步记录中，批次分数来自更新前，融合系数和头权重范数来自更新后；只在同一含义下比较。', '',
              '| Seed | 方法 | Split | 标签 | 安全类别分数均值 | 漏洞类别分数均值 | 局部 logit 均值 | 残差均值 | 修改区域倍率均值 |',
              '|---:|---|---|---:|---:|---:|---:|---:|---:|']
    trajectories = []
    for seed, item in results['seeds'].items():
        for method, d in item['models'].items():
            for split, summary in d['diagnostics']['splits'].items():
                for y in ('0', '1'):
                    stats = summary['branches'].get('by_target', {}).get(y)
                    ratio = summary['region_attention'].get(y, {}).get('mean')
                    ratio_text = f'{ratio:.4f}' if ratio is not None else 'NA'
                    values = (' | '.join(f"{stats[k]['mean']:.6f}" for k in ('safe_score', 'vulnerable_score', 'local_logit', 'local_residual'))
                              if stats else 'NA | NA | NA | NA')
                    lines.append(f"| {seed} | {method} | {split} | {y} | {values} | {ratio_text} |")
            if method == 'baseline': continue
            start, end = d['local_training'][0], d['local_training'][-1]
            trajectories.append(f"Seed{seed} {method} 的训练记录：c 从 {start['fusion']:.8g} 到 {end['fusion']:.8g}，局部头权重范数从 {start['head_weight_norm']:.6f} 到 {end['head_weight_norm']:.6f}。末步指最后一个每百步记录；所选 checkpoint 的系数见上文。")
    lines += ['', *trajectories]
    lines += ['', '## 判定及复核材料', '',
              '第二种子扩展标准预先固定为：LSE 相对 baseline 和 Mean 的 P-C 均至少增加 1 pp，且 AUC/MCC 均不下降。',
              f"Seed123456 是否通过：{bool(advance)}。没有进行温度、Top-K、融合权重或结构搜索。",
              'cases.json 给出此前固定错误，以及匹配模型比较和同权重残差消融各自前两个新增/丢失配对的两类局部最高分行、全局 logit、局部 logit 和实际残差。高行分数不作为漏洞标签。',
              '单测和真实 GPU smoke 见 tests.json、gpu-smoke.json；所选权重的 Train/Valid 首函数推理差异见各 branch_diagnostics.json。',
              '权重和张量缓存只保留本地。命令、环境、初始化和固定规则见 experiment.json。']
    if (root/'history.report.md').exists():
        lines += ['', '历史 Test 八种方法的对齐、错误类型、成对 Logit 差、冻结聚合容量及源码/Diff 案例另见 [history.report.md](history.report.md)。该历史诊断与本轮匹配 Train/Valid 方法验证分开报告，不作为新方法选择依据。']
    if include_test:
        lines += ['', 'Test 在完整 Valid 决策和所有 checkpoint 固定后统一评价，不用于结构、参数或选模。它已在历史研究中多次被查看，本轮数值为固定协议的描述性评价。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(promote_to_second_seed=advance, seeds={s: {m: d['metrics'] for m, d in i['models'].items()}
                                                               for s, i in results['seeds'].items()})), flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('check', 'analyze', 'report', 'history', 'history-lines'))
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--output-dir', type=Path, default=Path('results/global_local_mil_512'))
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--model', default='/home/PublicData/PHY-data/resource/unixcoder-base-nine')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--include-test', action='store_true')
    a = p.parse_args()
    if a.command == 'check':
        check(a.data_dir, a.output_dir, a.model, a.device)
    elif a.command == 'history':
        historical_errors(a.data_dir, a.output_dir, a.model)
    elif a.command == 'history-lines':
        historical_line_diagnosis(a.data_dir, a.output_dir, a.device)
    elif a.command == 'analyze':
        if a.run_dir is None: p.error('analyze requires --run-dir')
        analyze(a.run_dir, a.data_dir, a.device)
    else:
        report(a.output_dir, a.data_dir, a.include_test)


if __name__ == '__main__':
    main()
