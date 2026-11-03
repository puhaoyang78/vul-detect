"""Global/local MIL smoke and matched-result analysis; training stays in the CLI."""
from pathlib import Path
import argparse
import csv
import json
import tempfile

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
    out['interpretation'] = 'latent bag-level class evidence, not calibrated per-line risk; scalar fusion is unconstrained'
    # Measure how much the actual reduction departs from a line mean on the
    # SAME head scores. This is a score statistic, not another inference model.
    within_line_std = np.asarray([np.std(p['local_line_scores'], axis=0) for p in predictions])
    margins = [np.mean(np.asarray(p['local_line_scores'])[:, 1] - np.asarray(p['local_line_scores'])[:, 0])
               for p in predictions]
    out['within_function_line_score_std'] = {
        name: distribution(within_line_std[:, c]) for c, name in enumerate(('safe', 'vulnerable'))}
    out['reduction_minus_line_mean_margin'] = distribution([p['local_logit']-z for p, z in zip(predictions, margins)])
    return out


def check(data_dir, output_dir, model_path, device):
    """Real pretrained encoder, original forward, two optimization steps, disk restore."""
    from . import model as core
    from .evaluation import load_paired
    from .training import paired_loader, paired_collate, paired_forward

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
        added = sum(p.numel() for p in m.local_classifier.parameters()) + m.local_fusion.numel()
        before = m.local_classifier.weight.detach().clone()
        opt = torch.optim.SGD(m.parameters(), lr=1e-3)
        steps = []
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(2):
            m.train()
            loss, _, d = paired_forward(m, b, 'baseline', device, return_attention=True)
            torch.testing.assert_close(loss, d['bce']+.1*d['region_loss'])
            loss.backward()
            head_grad = float(m.local_classifier.weight.grad.abs().sum())
            fusion_grad = float(m.local_fusion.grad.abs())
            encoder_grad = float(m.encoder.embeddings.word_embeddings.weight.grad.abs().sum())
            assert fusion_grad > 0 and encoder_grad > 0
            assert (head_grad == 0) == (step == 0)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
            opt.step(); opt.zero_grad()
            steps.append(dict(loss=float(loss.detach()), fusion=float(m.local_fusion.detach()),
                              fusion_gradient=fusion_grad, head_gradient_l1=head_grad,
                              encoder_gradient_l1=encoder_grad))
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
                  bounded_attention=dict(context=True, region_supervision='diff'), local_mil=dict(pooling=mode))
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
    result = dict(test_used=False, seed=123456, device=torch.cuda.get_device_name(device),
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
                test_predictions[method] = [p['probability'] for p in ps]
        for m in ('mean', 'lse'):
            for k in ('training_members', 'validation_members', 'training_config', 'input_coverage'):
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
                        for k in ('p_c', 'auc', 'mcc', 'f1', 'recall', 'patched_fpr', 'pair_ranking_accuracy')})
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
    fields = ['seed', 'method', 'epoch', 'p_c', 'auc', 'mcc', 'f1', 'recall', 'patched_fpr', 'pair_ranking_accuracy']
    with (root/'classification.csv').open('w') as h:
        w = csv.DictWriter(h, fieldnames=['split']+fields, lineterminator='\n'); w.writeheader()
        for seed, item in results['seeds'].items():
            for method, d in item['models'].items():
                w.writerow(dict(split='valid', seed=seed, method=method, epoch=d['epoch'], **{k: d['metrics'][k] for k in fields[3:]}))
                if include_test:
                    w.writerow(dict(split='test', seed=seed, method=method, epoch=d['epoch'], **{k: d['test']['metrics'][k] for k in fields[3:]}))
    lines = ['# Global–Local Dual-Stream MIL', '',
        '三组使用同一正式模型、数据读取、优化器与评价入口；训练均从同一预训练 UniXcoder 初始化。',
        '官方 paired Train 8,708 个函数，Valid 1,124 个函数；Seed123456、Batch8、512 Token、16 Epoch、17,424 步。',
        '按完整 Valid P-C 四位小数选模，同分保留较早 Epoch；固定 probability > 0.5。历史权重不覆盖。', '',
        '全局分支保留原 Context + Diff。局部分支为独立 Linear(768,2)，两个通道分别记作安全/漏洞证据；复用同一次编码的物理行均值。',
        'Mean 对有效行等权平均；LSE 使用 logsumexp(scores)-log(N)，温度固定 1。共同的 log(N) 在类别差值中抵消，保留它以便比较类别分数。',
        '最终 logit = global_logit + c*(local_class[1]-local_class[0])，c 无约束且零初始化。随机局部头在 CPU fork_rng 内初始化；新增 1,539 个参数。',
        '仅使用最终函数 BCE + 0.1 原 Diff 区域损失。没有新区域标签、额外辅助任务或编码；推理只输入单函数。',
        '两个局部通道是包级潜在证据，不是经过校准的行漏洞概率；c 的符号也应结合解释。', '',
        '## 完整分类结果', '', '| Split | Seed | 方法 | Epoch | P-C | AUC | MCC | F1 | Recall | 修复侧 FPR | 配对排序正确率 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seed, item in results['seeds'].items():
        for method, d in item['models'].items():
            for split, metrics in [('Valid', d['metrics'])]+([('Test', d['test']['metrics'])] if include_test else []):
                lines.append(f"| {split} | {seed} | {method} | {d['epoch']} | {metrics['p_c']:.2%} | {metrics['auc']:.4f} | {metrics['mcc']:.4f} | {metrics['f1']:.4f} | {metrics['recall']:.4f} | {metrics['patched_fpr']:.4f} | {metrics['pair_ranking_accuracy']:.4f} |")
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
            lines.append(f"- Seed{seed} {method}: 所选 c={b['fusion']:.8g}；同一权重、同一编码下 global-only P-C={b['global_only_metrics']['p_c']:.2%}，融合后={d['metrics']['p_c']:.2%}，新增/丢失={delta['gains']}/{delta['losses']}；局部残差新增修复侧误报 {b['introduced_patched_false_positives']} 个、消除 {b['removed_patched_false_positives']} 个。")
            lines.append(f"  漏洞侧恢复正确阳性 {b['recovered_vulnerable_true_positives']} 个、新增漏报 {b['introduced_vulnerable_false_negatives']} 个；平均绝对 logit 残差 {b['absolute_residual']['mean']:.8g}，最大概率变化 {b['maximum_probability_change']:.8g}。")
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
              'cases.json 给出此前固定错误配对及本轮新增/丢失配对的两类局部最高分行、全局 logit、局部 logit 和实际残差。高行分数不作为漏洞标签。',
              '单测和真实 GPU smoke 见 tests.json、gpu-smoke.json；所选权重的 Train/Valid 首函数推理差异见各 branch_diagnostics.json。',
              '权重和张量缓存只保留本地。命令、环境、初始化和固定规则见 experiment.json。']
    if include_test:
        lines += ['', 'Test 在完整 Valid 决策和所有 checkpoint 固定后统一评价，不用于结构、参数或选模。它已在历史研究中多次被查看，本轮数值为固定协议的描述性评价。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(promote_to_second_seed=advance, seeds={s: {m: d['metrics'] for m, d in i['models'].items()}
                                                               for s, i in results['seeds'].items()})), flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('check', 'analyze', 'report'))
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--output-dir', type=Path, default=Path('results/global_local_mil_512'))
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--model', default='/home/PublicData/PHY-data/resource/unixcoder-base-nine')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--include-test', action='store_true')
    a = p.parse_args()
    if a.command == 'check':
        check(a.data_dir, a.output_dir, a.model, a.device)
    elif a.command == 'analyze':
        if a.run_dir is None: p.error('analyze requires --run-dir')
        analyze(a.run_dir, a.data_dir, a.device)
    else:
        report(a.output_dir, a.data_dir, a.include_test)


if __name__ == '__main__':
    main()
