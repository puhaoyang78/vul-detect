"""Bounded CPG diagnostics on fixed, provenance-preserving sample rows."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import time

from .cpg import CPGError, CPGQualityError, TargetMethodError, extract_function_cpg_batch
from .dataset import _read_samples, _resolve_sample, _load_reusable_records, extract_cpg_relations
from .benchmark_view import select_source_records
from .semantics import extract_vulnerability_semantics, render_semantic_items


def _unpack(value):
    if isinstance(value, dict) and '@value' in value:
        return _unpack(value['@value'])
    if isinstance(value, list):
        values = [_unpack(item) for item in value]
        return values[0] if len(values) == 1 else values
    return value


def read_graphson(path: Path):
    data = json.loads(path.read_text())['@value']
    nodes = {str(_unpack(n['id'])): dict(kind=n['label'], **{
        key: _unpack(value) for key, value in n['properties'].items()}) for n in data['vertices']}
    edges = [(e['label'], str(_unpack(e['outV'])), str(_unpack(e['inV']))) for e in data['edges']]
    return nodes, edges


def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def full_file_comparison(samples_path: Path, contexts_path: Path, output: Path, batch_size=2):
    samples = _read_samples(samples_path)
    if not 100 <= len(samples) <= 500:
        raise ValueError('diagnostic input must contain 100–500 samples')
    contexts = json.loads(contexts_path.read_text())
    targets = [sample for sample in samples if sample.sample_key in contexts]
    if not 100 <= len(targets) <= 200:
        raise ValueError('full-file comparison requires 100–200 verified contexts')
    with output.open('w') as handle:
        for offset in range(0, len(targets), batch_size):
            batch = targets[offset:offset + batch_size]
            requests = []
            for sample in batch:
                context = contexts[sample.sample_key]
                language, hint = _resolve_sample(sample)
                requests.append(dict(source=sample.source, function=hint.name, language=language,
                                     full_source=Path(context['path']).read_text(errors='replace'),
                                     start_line=context['start_line']))
            started = time.monotonic()
            try:
                graphs = extract_function_cpg_batch(requests, timeout=180)
            except (CPGError, OSError, subprocess.TimeoutExpired) as error:
                graphs = [error] * len(batch)
            seconds = (time.monotonic() - started) / len(batch)
            for sample, graph in zip(batch, graphs):
                row = dict(sample_key=sample.sample_key, dataset=sample.dataset, label=sample.label,
                           split=sample.split, seconds=seconds)
                if isinstance(graph, BaseException):
                    stage = ('low_quality_cpg' if isinstance(graph, CPGQualityError) else
                             'target_method' if isinstance(graph, TargetMethodError) else 'joern')
                    row.update(status='failed', stage=stage, error=str(graph))
                else:
                    row.update(status='done', quality=graph.quality,
                               relations=len(extract_cpg_relations(graph)),
                               semantic_items=len(extract_vulnerability_semantics(graph).items))
                handle.write(json.dumps(row, sort_keys=True) + '\n')
                handle.flush()
                print(sample.sample_key, row['status'], seconds, flush=True)


def refresh_semantic_text(folder: Path):
    """Replay only the public renderer on persisted diagnostic semantic evidence."""
    for name in ('function_only.jsonl', 'macro.jsonl'):
        path = folder / name
        rows = read_rows(path)
        for row in rows:
            row['vulnerability_semantics'] = render_semantic_items(row['semantic_items'])
        path.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows))


def summarize(folder: Path):
    samples = read_rows(folder / 'samples.jsonl')
    manifest = {r['sample_key']: r for r in read_rows(Path('data/benchmark/manifest.jsonl'))}
    assert len(samples) == len({r['sample_key'] for r in samples})
    assert all(r == manifest[r['sample_key']] for r in samples)
    baseline = read_rows(folder / 'baseline.jsonl')
    built = read_rows(folder / 'function_only.jsonl')
    errors = read_rows(folder / 'function_only.errors.jsonl')
    reusable, incomplete_tail = _load_reusable_records(folder / 'function_only.jsonl', _read_samples(folder / 'samples.jsonl'))
    assert len(reusable) == len(built) and not incomplete_tail
    assert all(r['vulnerability_semantics'] == render_semantic_items(r['semantic_items']) for r in built)
    integrity = dict(reusable_success_records=len(reusable),
                     source_label_split_pair_exactly_preserved=True,
                     semantic_text_replayed_and_verified=True,
                     pair_views={d: select_source_records(built, d)[1] for d in ('cleanvul', 'sven')})
    (folder / 'integrity_checks.json').write_text(json.dumps(integrity, indent=2) + '\n')
    equivalence = []
    errors_by_key = {r['sample_key']: r for r in errors}
    for single in read_rows(folder / 'single_sample_check.jsonl'):
        current = reusable.get(single['sample_key'])
        equivalent = (current is not None and single['status'] == 'done'
                      and single['quality']['edge_counts'] == current['cpg_quality']['edge_counts']
                      and single['relations'] == current['cpg_relation_count']
                      and render_semantic_items(single['semantic_items']) == current['vulnerability_semantics'])
        if single['status'] == 'failed':
            failure = errors_by_key.get(single['sample_key'])
            equivalent = current is None and failure is not None and failure['error_type'] == single['error_type']
        assert equivalent, single['sample_key']
        equivalence.append(dict(sample_key=single['sample_key'], equivalent=equivalent))
    (folder / 'batch_equivalence.json').write_text(json.dumps(equivalence, indent=2) + '\n')
    full = read_rows(folder / 'full_file.jsonl')
    new = [dict(sample_key=r['sample_key'], dataset=r['dataset'], label=r['label'], split=r['split'],
                status='done', seconds=r['seconds'], quality=r['cpg_quality'],
                relations=r['cpg_relation_count'], semantic_items=r['semantic_item_count']) for r in built]
    new.extend(dict(r, status='failed') for r in errors)
    keys = {r['sample_key'] for r in samples}
    assert {r['sample_key'] for r in baseline} == keys and len(baseline) == len(keys)
    assert {r['sample_key'] for r in new} == keys and len(new) == len(keys)
    assert len(full) == 100 and len({r['sample_key'] for r in full}) == 100

    def stats(rows):
        groups = defaultdict(Counter)
        for r in rows:
            g = groups[f"{r['dataset']}/label_{r['label']}"]
            g['total'] += 1
            g[r['status']] += 1
            if r['status'] == 'failed':
                stage = r.get('stage')
                if not stage:
                    stage = 'syntax' if 'stage=syntax' in r.get('detail', '') else 'joern'
                g['failed_' + stage] += 1
        done = [r for r in rows if r['status'] == 'done']
        group_report = {}
        for key, counts in sorted(groups.items()):
            group_report[key] = dict(counts)
            for stage in ('syntax', 'joern', 'target_method', 'low_quality_cpg'):
                group_report[key]['failure_rate_' + stage] = counts['failed_' + stage] / counts['total']
        return dict(total=len(rows), success=len(done), success_rate=len(done) / len(rows),
                    groups=group_report,
                    success_without_semantic_items=sum(r.get('semantic_items') == 0 for r in done),
                    success_at_most_10_relations=sum(r.get('relations', 0) <= 10 for r in done))
    full_keys = {r['sample_key'] for r in full}
    new_by_key = {r['sample_key']: r for r in new}
    measured = [r for r in baseline if r.get('seconds') is not None]
    report = dict(baseline=stats(baseline), function_only=stats(new),
                  full_file=stats(full), function_only_matched_full_file=stats([new_by_key[k] for k in sorted(full_keys)]),
                  timing=dict(baseline_measured_samples=len(measured),
                              baseline_measured_seconds=sum(r['seconds'] for r in measured),
                              new_same_samples_amortized_seconds=sum(new_by_key[r['sample_key']]['seconds'] for r in measured),
                              new_total_seconds=sum(r['seconds'] for r in new),
                              full_file_total_seconds=sum(r['seconds'] for r in full),
                              note='Batch wall time amortized across samples; baseline reuses logged outcomes where available. Concurrent local diagnostic jobs; not an isolated throughput benchmark.'),
                  full_file_transitions=dict(Counter(f"{new_by_key[r['sample_key']]['status']}->{r['status']}" for r in full)),
                  provenance=dict(sample_rows_match_manifest=True, labels_and_splits_unchanged=True,
                                  input_pairs='Both sides retained; output pair completeness in function_only.audit.json',
                                  sampling='Targeted failures, low-relation successes and ordinary successes; not a population success-rate estimate'))
    rechecked = read_rows(folder / 'full_file_csv_recheck.jsonl')
    replacements = {r['sample_key']: r for r in rechecked}
    assert set(replacements) == {r['sample_key'] for r in full if r.get('stage') == 'joern'}
    corrected_full = [replacements.get(r['sample_key'], r) for r in full]
    report['full_file_after_csv_recheck'] = stats(corrected_full)
    report['full_file_after_csv_transitions'] = dict(Counter(
        f"{new_by_key[r['sample_key']]['status']}->{r['status']}" for r in corrected_full))
    report['export_memory_recheck'] = dict(samples=len(rechecked),
        seconds=sum(r['seconds'] for r in rechecked),
        remaining_backend_failures=sum(r.get('stage') == 'joern' for r in rechecked),
        accepted=sum(r['status'] == 'done' for r in rechecked),
        note='Only the four original GraphSON heap failures were rerun with CSV at the same 2 GB heap; original 100-run results retained.')
    report['export_format_equivalence'] = json.loads((folder / 'export_format_equivalence.json').read_text())
    report['old_to_new_transitions'] = dict(Counter(f"{r['status']}->{new_by_key[r['sample_key']]['status']}" for r in baseline))
    report['batch_equivalence'] = json.loads((folder / 'batch_equivalence.json').read_text())
    report['new_build_audit'] = json.loads((folder / 'function_only.audit.json').read_text())
    report['macro_probe_audit'] = json.loads((folder / 'macro.audit.json').read_text())
    report['fname_full_file_probe'] = json.loads((folder / 'fname_full_file.json').read_text())
    report.update(json.loads((folder / 'run_metadata.json').read_text()))
    (folder / 'comparison.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=Path('results/cpg_debug'))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--full-file', action='store_true')
    modes.add_argument('--refresh-semantic-text', action='store_true')
    args = parser.parse_args()
    if args.refresh_semantic_text:
        refresh_semantic_text(args.folder)
    elif args.full_file:
        full_file_comparison(args.folder / 'samples.jsonl', args.folder / 'full_file_contexts.json',
                             args.folder / 'full_file.jsonl')
    else:
        summarize(args.folder)


if __name__ == '__main__':
    main()
