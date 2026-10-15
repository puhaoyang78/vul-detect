"""Audited local states: data admission, original-source regions and pair losses.

No function labels are inferred from repair sides. Only the public trainer uses
these regions; the single-function inference path never reads this dataset.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from bisect import bisect_right
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import random
import re

import torch
from torch.nn import functional as F

from .prepare_benchmark import NearIndex, normalized_source, unit
from .syntax import byte_character_map, source_token_spans

MODES = ('none', 'state', 'state_contrast')
RAW_ROOT = Path('/home/PublicData/PHY-data/vul_detect/data')


def supervision_config(mode):
    if mode not in MODES[1:]:
        raise ValueError('repair supervision requires state or state_contrast')
    return dict(version=1, mode=mode, lambda_audit=.1,
                beta=.1 if mode == 'state_contrast' else 0., tau=.1,
                region='union of validated original coordinates; lexical overlap / BPE span; coverage-weighted mean',
                sampling='independent seed/epoch RNG; same mechanism, two independent events when available; once per pair per epoch',
                gradient='lambda_audit * optimizer_steps / pairs * sum of per-pair terms; dropout disabled')


def validate_config(config):
    if not isinstance(config, dict) or config != supervision_config(config.get('mode')):
        raise ValueError('incompatible repair supervision configuration')


def find_source(neighbor):
    neighbor = Path(neighbor)
    direct = neighbor / 'high_quality_repair_pairs.json'
    paths = [direct] if direct.is_file() else sorted(neighbor.rglob('high_quality_repair_pairs.json'))
    if not paths:
        raise FileNotFoundError(neighbor / 'high_quality_repair_pairs.json')
    documents = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    if any(document != documents[0] for document in documents[1:]):
        raise ValueError(f'different repair sources: {[str(path) for path in paths]}')
    return paths[0], documents[0]


def source_partition(sample):
    """Only a declared paired source basename establishes an official split."""
    path = sample.get('pair_provenance', {}).get('source_path', '')
    if not isinstance(path, str):
        raise ValueError(f'{sample["sample_id"]}: source_path must be a string')
    basename = path.replace('\\', '/').rsplit('/', 1)[-1]
    declared = {f'primevul_{split}_paired.jsonl': split for split in ('train', 'valid', 'test')}
    split = declared.get(basename)
    if split is not None:
        match = re.fullmatch(r'primevul_(train|valid|test)_\d+', sample['sample_id'])
        if sample.get('dataset', '').lower() != 'primevul' or not match or match[1] != split:
            raise ValueError(f'{sample["sample_id"]}: dataset/sample_id conflicts with {path}')
    return split


def cves(value):
    return set(re.findall(r'CVE-\d{4}-\d{4,}', str(value).upper()))


def commit(value):
    value = str(value or '').lower()
    return value if re.fullmatch(r'[0-9a-f]{7,40}', value) else ''


def shared_commit(value, others):
    return bool(value) and any(value.startswith(other) or other.startswith(value) for other in others)


def protected_records(records, *, raw_root=RAW_ROOT):
    """Enrich only original identity metadata, never predictions or model outputs."""
    protected = [dict(row) for row in records if
                 row['dataset'] == 'primevul' and row['split'] in ('valid', 'test')
                 or row['dataset'] == 'sven']
    native = defaultdict(dict)
    for row in protected:
        if row['dataset'] == 'primevul' and 'cve' not in row:
            if not row.get('source_file') or not row.get('source_row') or 'idx' not in row:
                raise ValueError(f'{row["sample_key"]}: missing reliable native evaluation identity')
            native[row['source_file']][row['source_row']] = row
    for relative, wanted in native.items():
        path = Path(raw_root) / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError(f'native identity outside raw data root: {relative}')
        found = set()
        with path.open(encoding='utf-8') as source:
            for line, value in enumerate(source, 1):
                if line not in wanted:
                    continue
                original, row = json.loads(value), wanted[line]
                if original['idx'] != row['idx'] or original['func'] != row['raw_source'] or original['target'] != row['label']:
                    raise ValueError(f'{row["sample_key"]}: native source identity mismatch at {path}:{line}')
                row['cve'] = original.get('cve', '')
                found.add(line)
        if found != set(wanted):
            raise ValueError(f'incomplete native evaluation identities: {path}')
    return protected


def verify_prime_identities(samples, raw_root):
    requested = defaultdict(dict)
    differences = []
    for sample in samples:
        split = source_partition(sample)
        if split is None:
            continue
        indices = sample['pair_provenance'].get('source_row_indices')
        if not isinstance(indices, list) or len(indices) != 2 or any(type(index) is not int for index in indices):
            raise ValueError(f'{sample["sample_id"]}: missing official function idx identities')
        for side, index in enumerate(indices):
            requested[split].setdefault(index, []).append((sample, side))
    for split, wanted in requested.items():
        path = Path(raw_root) / 'PrimeVul_v0.1' / f'primevul_{split}_paired.jsonl'
        found = set()
        with path.open(encoding='utf-8') as handle:
            for value in handle:
                native = json.loads(value)
                if native['idx'] not in wanted:
                    continue
                for sample, side in wanted[native['idx']]:
                    source = sample['code_' + ('before' if side == 0 else 'after')]
                    if native['target'] != 1 - side:
                        raise ValueError(f'{sample["sample_id"]}: official idx/source/side conflicts at {path}')
                    if normalized_source(native['func']) != normalized_source(source):
                        # Audited source can contain an explicitly supplied
                        # completed function. Do not rewrite it or assert exact
                        # native equality. Independently require the original
                        # idx, side, CVE and full fixing revision to agree.
                        native_cves = cves(native.get('cve', ''))
                        revision = commit(sample.get('fix_commit'))
                        if (not native_cves.intersection(cves(sample.get('cve', '')))
                                or len(revision) != 40 or revision != commit(native.get('commit_id'))):
                            raise ValueError(f'{sample["sample_id"]}: native provenance conflicts for differing audited source at {path}')
                        differences.append(dict(sample_id=sample['sample_id'], side='before' if side == 0 else 'after',
                                                native_idx=native['idx'], native_label=native['target'],
                                                cve=sample['cve'], fix_commit=revision,
                                                reason='audited/native lexical source differs; original idx/side/CVE/full fix revision agree; audited source retained verbatim'))
                found.add(native['idx'])
        if found != set(wanted):
            raise ValueError(f'missing official repair endpoint identities: {path}: {sorted(set(wanted)-found)}')
    return differences


def isolate_samples(samples, protected, removed):
    """Reject a whole pair on either endpoint/event; same project is not a gate."""
    sources, events = {}, set()
    near = NearIndex()
    for row in protected:
        code = row['raw_source']
        sources.setdefault(normalized_source(code), []).append(row['sample_key'])
        events.update(cves(row.get('cve', '')))
        near.add(unit([dict(row, function=code)]))
    blocked_commits = {commit(row.get('commit')) for row in protected} - {''}
    protected_native_ids = {row['idx'] for row in protected if row['dataset'] == 'primevul' and 'idx' in row}
    for sample in removed:
        events.update(cves(sample.get('cve', '')))
        revision = commit(sample.get('fix_commit'))
        if revision:
            blocked_commits.add(revision)
        for side in ('before', 'after'):
            code = sample['code_' + side]
            key = sample['sample_id'] + ':' + side
            sources.setdefault(normalized_source(code), []).append(key)
            near.add(unit([dict(dataset='official_repair_holdout', split='valid_or_test', sample_key=key,
                                label=int(side == 'before'), function=code, repo='', commit='')]))
    retained, excluded = [], []
    for sample in samples:
        evidence = []
        if source_partition(sample) is not None:
            shared_ids = sorted(set(sample['pair_provenance'].get('source_row_indices', [])).intersection(protected_native_ids))
            if shared_ids:
                evidence.append(dict(reason='protected_native_sample', identities=shared_ids))
        shared = sorted(cves(sample.get('cve', '')).intersection(events))
        if shared:
            evidence.append(dict(reason='protected_cve', identities=shared))
        revision = commit(sample.get('fix_commit'))
        if shared_commit(revision, blocked_commits):
            evidence.append(dict(reason='protected_fix_commit', identities=[revision]))
        for side in ('before', 'after'):
            key = normalized_source(sample['code_' + side])
            if key in sources:
                evidence.append(dict(reason='protected_normalized_source', side=side, identities=sources[key]))
            candidate = unit([dict(dataset='audit_auxiliary', split='train', sample_key=sample['sample_id'],
                                   label=int(side == 'before'), function=sample['code_' + side], repo='', commit='')])
            matches = near.matches(candidate)
            if matches:
                evidence.append(dict(reason='protected_near_duplicate', side=side, identities=matches))
        if evidence:
            excluded.append(dict(sample_id=sample['sample_id'], dataset=sample['dataset'], evidence=evidence))
        else:
            retained.append(sample)
    return retained, excluded


def copy_training_source(neighbor, destination, records, *, raw_root=RAW_ROOT):
    source, document = find_source(neighbor)
    if document.get('schema_version') != 2 or not isinstance(document.get('samples'), list):
        raise ValueError('requires the admitted compact schema-2 repair source')
    samples = document['samples']
    if len({sample['sample_id'] for sample in samples}) != len(samples):
        raise ValueError('duplicate repair sample_id')
    native_differences = verify_prime_identities(samples, raw_root)
    removed = [sample for sample in samples if source_partition(sample) in ('valid', 'test')]
    candidates = [sample for sample in samples if source_partition(sample) not in ('valid', 'test')]
    retained, excluded = isolate_samples(candidates, protected_records(records, raw_root=raw_root), removed)
    base_document = {key: value for key, value in document.items() if key != 'heldout_exclusion'}
    expected = dict(base_document, samples=retained)
    destination = Path(destination)
    if destination.resolve() == source.resolve():
        raise ValueError('destination must not be the neighboring source')
    if destination.exists():
        existing = json.loads(destination.read_text(encoding='utf-8'))
        if existing != expected:
            # Only the exact source-filtered copy may receive additional proven
            # whole-pair exclusions. Any edits to content/provenance stop here.
            if existing != dict(base_document, samples=candidates):
                raise ValueError(f'{destination}: existing content differs from this verified source; preserve user edits')
        else:
            existing = expected
    else:
        existing = None
    if existing != expected:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        if temporary.exists():
            raise FileExistsError(temporary)
        try:
            with temporary.open('x', encoding='utf-8', newline='\n') as handle:
                json.dump(expected, handle, ensure_ascii=False, indent=2)
                handle.write('\n')
            if json.loads(temporary.read_text(encoding='utf-8')) != expected:
                raise ValueError('repair source write verification failed')
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    stats = dict(source=str(source), destination=str(destination), input_pairs=len(samples),
                 native_source_differences=native_differences,
                 official_removed=dict(Counter(source_partition(sample) for sample in removed)),
                 source_filtered_pairs=len(candidates), source_filtered_datasets=dict(Counter(s['dataset'] for s in candidates)),
                 final_pairs=len(retained), final_datasets=dict(Counter(s['dataset'] for s in retained)),
                 extra_excluded=excluded, expected_source_filter=dict(input=1279, valid=61, test=45, retained=1173),
                 protected_scope='current PrimeVul Valid/Test, SVEN and all 106 explicitly removed official repair events')
    return retained, [sample for sample in removed if source_partition(sample) == 'valid'], stats


def merge_spans(spans):
    result = []
    for left, right in sorted(spans):
        if left >= right:
            continue
        if result and left <= result[-1][1]:
            result[-1] = (result[-1][0], max(right, result[-1][1]))
        else:
            result.append((left, right))
    return result


def location_span(source, location):
    """One-based lines/columns, exclusive end; validate character/UTF-8 units.

    The real source's 4,001 annotations use exclusive ends. Byte and character
    interpretations agree at those boundaries. source_text resolves Unicode
    boundaries when they differ; ambiguous locations are rejected explicitly.
    CRLF is normalized only for comparison, never in the encoder input.
    """
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    try:
        start_line, end_line = location['start_line'] - 1, location['end_line'] - 1
        left, right = location['start_column'] - 1, location['end_column'] - 1
        if not 0 <= start_line <= end_line < len(lines) or left < 0 or right < 0:
            raise ValueError('coordinate bounds')
        expected = location['source_text'].replace('\r\n', '\n')
        candidates = set()
        for byte_units in (False, True):
            a, b = left, right
            if byte_units:
                first = byte_character_map(lines[start_line].rstrip('\r\n'))
                last = byte_character_map(lines[end_line].rstrip('\r\n'))
                if a not in first or b not in last:
                    continue
                a, b = first[a], last[b]
            if a > len(lines[start_line].rstrip('\r\n')) or b > len(lines[end_line].rstrip('\r\n')):
                continue
            span = starts[start_line] + a, starts[end_line] + b
            if span[0] < span[1] and source[span[0]:span[1]].replace('\r\n', '\n') == expected:
                candidates.add(span)
        if len(candidates) != 1:
            raise ValueError('source_text mismatch or ambiguous coordinate units')
        return candidates.pop()
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('malformed audited location') from error


def intersect_spans(first, second):
    first, second = merge_spans(first), merge_spans(second)
    result, i, j = [], 0, 0
    while i < len(first) and j < len(second):
        a, b = max(first[i][0], second[j][0]), min(first[i][1], second[j][1])
        if a < b:
            result.append((a, b))
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1
    return merge_spans(result)


def lexical_character_spans(source, tokens):
    # RoBERTa trims whitespace-only BPE offsets. Whitespace remains in the
    # encoder input; only non-whitespace code characters establish a readout.
    return merge_spans((left + part.start(), left + part.end()) for _, left, right in tokens
                       for part in re.finditer(r'\S+', source[left:right]))


def source_region(source, locations, builder):
    if not builder.tokenizer.is_fast:
        raise ValueError('audited regions require a fast tokenizer')
    encoded = builder.tokenizer(source, add_special_tokens=False, return_offsets_mapping=True,
                                truncation=False, verbose=False)
    offsets, ids = encoded['offset_mapping'], encoded['input_ids']
    if len(offsets) != len(ids) or any(not 0 <= a <= b <= len(source) for a, b in offsets):
        raise ValueError('invalid original-source tokenizer offsets')
    budget = builder.source_max_length - len(builder.source_prefix) - 2
    visible_ids, visible_offsets = list(ids[:budget]), offsets[:budget]
    if visible_ids != builder._encode(source, budget):
        raise ValueError('offset tokenizer IDs differ from actual encoder source input')
    if not locations:
        raise ValueError('missing_region')
    regions = merge_spans(location_span(source, loc) for loc in locations)
    lexical = lexical_character_spans(source, source_token_spans(source))
    effective = intersect_spans(regions, lexical)
    if not effective:
        raise ValueError('empty_code_region')
    covered = intersect_spans(effective, visible_offsets)
    if covered != effective:
        reason = 'region_truncated' if max(b for _, b in effective) > max((b for _, b in visible_offsets), default=0) else 'region_offset_uncovered'
        raise ValueError(reason)
    ends = [end for _, end in effective]
    weights = []
    for left, right in visible_offsets:
        overlap, index = 0, bisect_right(ends, left)
        while index < len(effective) and effective[index][0] < right:
            a, b = effective[index]
            overlap += max(0, min(b, right) - max(a, left))
            index += 1
        weights.append(overlap / (right - left) if right > left else 0.)
    if not sum(weights) > 0:
        raise ValueError('empty_region_token_weight')
    return dict(token_weights=weights, spans=regions, visible_ids=visible_ids,
                visible_spans=merge_spans(visible_offsets), actual_tokens=len(ids), source_chars=len(source))


def prepare_regions(samples, builder):
    prepared, skipped = [], []
    for sample in samples:
        try:
            if any(not isinstance(sample.get(key), str) or not sample[key].strip()
                   for key in ('unsafe_behavior', 'security_change', 'keep_reason')):
                raise ValueError('missing_existing_audit_scope')
            review = sample['review_annotations']
            regions = [source_region(sample['code_' + side], review['locations'][side], builder)
                       for side in ('before', 'after')]
            if regions[0]['visible_ids'] == regions[1]['visible_ids']:
                raise ValueError('identical_actual_encoder_input')
            tokens = [source_token_spans(sample['code_' + side]) for side in ('before', 'after')]
            edits = [opcode for opcode in SequenceMatcher(None, [t[0] for t in tokens[0]],
                     [t[0] for t in tokens[1]], autojunk=False).get_opcodes() if opcode[0] != 'equal']
            if not edits:
                raise ValueError('no_lexical_security_difference')
            for _, a, b, c, d in edits:
                for side, left, right in [(0, a, b), (1, c, d)]:
                    changed = lexical_character_spans(sample['code_' + ('before' if side == 0 else 'after')], tokens[side][left:right])
                    if intersect_spans(changed, regions[side]['visible_spans']) != changed:
                        raise ValueError('changed_code_truncated_or_offset_uncovered')
            group = (sample.get('vulnerability_type'), review.get('repair_pattern'))
            if any(not isinstance(value, str) or value.lower() in ('', 'other', 'unknown', 'none', 'unspecified') for value in group):
                group = None
            prepared.append(dict(sample_id=sample['sample_id'], dataset=sample['dataset'],
                                 project=sample.get('project', ''), cve=sample.get('cve', ''),
                                 fix_commit=sample.get('fix_commit', ''), group=group,
                                 sources=[sample['code_before'], sample['code_after']], regions=regions))
        except (ValueError, KeyError) as error:
            skipped.append(dict(sample_id=sample['sample_id'], reason=str(error)))
    assign_events(prepared)
    stats = dict(total_pairs=len(samples), usable_pairs=len(prepared), skipped=skipped,
                 skip_counts=dict(Counter(row['reason'] for row in skipped)),
                 mechanism_groups=dict(Counter('/'.join(pair['group']) if pair['group'] else 'unclassified' for pair in prepared)),
                 coordinate_convention='one-based line/column, exclusive end; source_text-validated character or UTF-8 boundaries',
                 difference_visibility='all non-equal lexical token spans must be visible; conservative gate, no patch-centred input',
                 membership=[pair['sample_id'] for pair in prepared])
    return prepared, stats


def assign_events(pairs):
    """CVE, fix revision or equal endpoint links prohibit duplicate positives."""
    parents = list(range(len(pairs)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen, revisions = {}, []
    for i, pair in enumerate(pairs):
        keys = [('cve', value) for value in cves(pair['cve'])]
        keys += [('source', normalized_source(source)) for source in pair['sources']]
        revision = commit(pair['fix_commit'])
        for old, owner in revisions:
            if revision and (old.startswith(revision) or revision.startswith(old)):
                parents[find(i)] = find(owner)
        if revision:
            revisions.append((revision, i))
        for key in keys:
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    for i, pair in enumerate(pairs):
        pair['event_id'] = pairs[find(i)]['sample_id']


def auxiliary_schedule(pairs, steps, *, seed, epoch):
    if not pairs or steps <= 0:
        raise ValueError('positive auxiliary pair and optimizer-step counts required')
    rng = random.Random(f'repair-supervision:{seed}:{epoch}')
    groups = defaultdict(list)
    for i, pair in enumerate(pairs):
        groups[pair['group']].append(i)
    batches = []
    for group in groups.values():
        rng.shuffle(group)
        while group:
            first = group.pop()
            partners = [i for i in group if pairs[i]['event_id'] != pairs[first]['event_id']]
            cross_project = [i for i in partners if pairs[i]['project'].rsplit('/', 1)[-1].lower() != pairs[first]['project'].rsplit('/', 1)[-1].lower()]
            partner = (cross_project or partners)[0] if partners else None
            if partner is not None and pairs[first]['group'] is not None:
                group.remove(partner)
                batches.append([first, partner])
            else:
                batches.append([first])
    rng.shuffle(batches)
    result = defaultdict(list)
    for index, batch in enumerate(batches):
        result[index * steps // len(batches)].append(batch)
    return dict(result)


def local_losses(vectors, logits, pairs, *, beta, tau):
    if len(vectors) != 2 * len(pairs) or logits.shape != (len(vectors),) or tau <= 0 or beta < 0:
        raise ValueError('invalid local loss inputs')
    targets = logits.new_tensor([1., 0.] * len(pairs))
    state = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction='none').reshape(-1, 2).mean(1)
    contrast = vectors.new_zeros(len(pairs), dtype=torch.float32)
    eligible = 0
    if beta:
        normalized = F.normalize(vectors.float(), p=2, dim=-1)
        similarities = normalized @ normalized.T / tau
        anchors = []
        for i, pair in enumerate(pairs):
            for side in range(2):
                anchor = 2 * i + side
                positive = [2 * j + side for j, other in enumerate(pairs)
                            if pair['group'] is not None and other['group'] == pair['group']
                            and i != j and pair['event_id'] != other['event_id']]
                negative = [2 * i + (1 - side)]
                negative += [2 * j + (1 - side) for j, other in enumerate(pairs)
                             if i != j and pair['group'] is not None and other['group'] == pair['group']]
                if positive:
                    neg_sum = torch.logsumexp(similarities[anchor, negative], dim=0)
                    pos = similarities[anchor, positive]
                    anchors.append((torch.logaddexp(pos, neg_sum) - pos).mean())
                    eligible += 1
                else:
                    anchors.append(similarities[anchor, anchor] * 0.)
        contrast = torch.stack(anchors).reshape(-1, 2).mean(1)
    return state, contrast, dict(contrast_anchors=eligible, no_positive_anchors=len(vectors) - eligible,
                                contrast_disabled=not bool(beta))


def auxiliary_scale(steps, pair_count, weight):
    if steps <= 0 or pair_count <= 0 or not 0 < weight <= 1:
        raise ValueError('invalid auxiliary epoch normalization')
    return weight * steps / pair_count


def region_readouts(model, builder, pairs, *, device):
    """Keep all vectors' graphs alive; repeated windows are still one function."""
    vectors, logits, global_logits = [], [], []
    cost = dict(function_forwards=0, encoder_sequences=0, encoder_token_slots=0)
    for pair in pairs:
        for side in range(2):
            inputs = builder.codebert_batch([dict(raw_source=pair['sources'][side])],
                                           device=device, return_source_positions=True)
            z, u, g = model.region_readout(*inputs, [pair['regions'][side]['token_weights']])
            vectors.append(u)
            logits.append(z)
            global_logits.append(g)
            cost['function_forwards'] += 1
            cost['encoder_sequences'] += len(inputs[0])
            cost['encoder_token_slots'] += inputs[0].numel()
    if not pairs:
        raise ValueError('empty audit readout batch')
    return torch.cat(vectors), torch.cat(logits), torch.cat(global_logits), cost


def local_summary(predictions):
    if not predictions:
        return dict(pairs=0, local_state_accuracy=None, both_correct=None, ranking_accuracy=None)
    n = len(predictions)
    correct, both, ranking, ties = 0, 0, 0, 0
    global_delta, probability_delta = 0., 0.
    baselines = {name: dict(correct=0, ties=0) for name in ('character_length', 'token_length', 'visible_token_length')}
    for row in predictions:
        before, after = row['local_logits']
        states = int(before >= 0), int(after < 0)
        correct += sum(states)
        both += all(states)
        ranking += before > after
        ties += before == after
        gb, ga = row['global_logits']
        global_delta += gb - ga
        probability_delta += float(torch.sigmoid(torch.tensor(gb)) - torch.sigmoid(torch.tensor(ga)))
        for name, field in [('character_length', 'source_chars'), ('token_length', 'actual_tokens'),
                            ('visible_token_length', 'visible_source_tokens')]:
            b, a = row[field]
            baselines[name]['correct'] += b > a
            baselines[name]['ties'] += b == a
    return dict(pairs=n, local_state_accuracy=correct / (2 * n), both_correct=both / n,
                ranking_accuracy=ranking / n, ranking_ties=ties,
                ranking_accuracy_half_ties=(ranking + .5 * ties) / n,
                global_diagnostic=dict(mean_before_minus_after_logit=global_delta / n,
                                       mean_before_minus_after_probability=probability_delta / n,
                                       definition='score-change diagnosis only; no whole-function repair labels'),
                length_baselines={key: dict(ranking_accuracy=value['correct'] / n, ties=value['ties'],
                                           ranking_accuracy_half_ties=(value['correct'] + .5 * value['ties']) / n,
                                           shorter_is_riskier_accuracy=(n - value['correct'] - value['ties']) / n,
                                           shorter_is_riskier_accuracy_half_ties=(n - value['correct'] - .5 * value['ties']) / n)
                                  for key, value in baselines.items()})


def local_report(checkpoint_path, checkpoint, pairs, predictions):
    """Recompute diagnostics from fixed scores; lengths never select members."""
    if [row['sample_id'] for row in predictions] != [pair['sample_id'] for pair in pairs]:
        raise ValueError('cached local prediction members differ')
    groups = defaultdict(list)
    lengths = {name: defaultdict(list) for name in ('source_chars', 'actual_tokens', 'visible_source_tokens')}
    for pair, row in zip(pairs, predictions):
        expected = dict(source_chars=[r['source_chars'] for r in pair['regions']],
                        actual_tokens=[r['actual_tokens'] for r in pair['regions']],
                        visible_source_tokens=[len(r['visible_ids']) for r in pair['regions']])
        for field, values in expected.items():
            if field in row and row[field] != values:
                raise ValueError(f'cached local diagnosis source length differs: {row["sample_id"]}/{field}')
            row[field] = values
        groups['/'.join(row['group']) if row['group'] else 'unclassified'].append(row)
        for field, group in lengths.items():
            b, a = row[field]
            group['after_longer' if a > b else 'after_shorter' if a < b else 'equal'].append(row)
    return dict(checkpoint=str(checkpoint_path), selected_epoch=checkpoint['selected_epoch'],
                global_threshold=checkpoint['decision_threshold'], local_threshold=.5,
                members=[pair['sample_id'] for pair in pairs], overall=local_summary(predictions),
                mechanisms={key: local_summary(value) for key, value in groups.items()},
                **{name: {key: local_summary(lengths[field][key]) for key in ('after_longer', 'after_shorter', 'equal')}
                   for name, field in [('after_character_length', 'source_chars'), ('after_token_length', 'actual_tokens'),
                                       ('after_visible_token_length', 'visible_source_tokens')]},
                length_definitions=dict(character_length='complete original source characters',
                                        token_length='complete untruncated source tokenizer length',
                                        visible_token_length='actual source IDs entering native windows, counted once; excludes prefix/BOS/EOS',
                                        directions='longer-is-riskier and shorter-is-riskier both reported; no direction selection'),
                role='independent official PrimeVul Valid-source local diagnosis; local states are not global labels')


def evaluate_local(checkpoint_path, pairs, output, *, device):
    from . import model as core
    from .artifacts import write_json
    checkpoint_path, output = Path(checkpoint_path), Path(output)
    predictions_path = output.with_name(output.name.replace('.metrics.json', '.predictions.jsonl'))
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if output.exists() != predictions_path.exists():
        raise FileExistsError('incomplete local diagnosis artifacts')
    if output.exists():
        result = json.loads(output.read_text())
        if (result['members'] != [pair['sample_id'] for pair in pairs]
                or result['checkpoint'] != str(checkpoint_path)
                or result['selected_epoch'] != checkpoint['selected_epoch']
                or result['global_threshold'] != checkpoint['decision_threshold']):
            raise ValueError('cached local diagnosis differs')
        predictions = [json.loads(line) for line in predictions_path.open()]
        refreshed = local_report(checkpoint_path, checkpoint, pairs, predictions)
        if refreshed != result:
            # Only deterministic derived length summaries are supplemented;
            # fixed scores, selected epoch, threshold and members are retained.
            temporary = predictions_path.with_suffix(predictions_path.suffix + '.tmp')
            with temporary.open('x') as handle:
                for row in predictions:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            temporary.replace(predictions_path)
            write_json(output, refreshed)
        return refreshed
    resolved = core._resolve_device(device)
    builder = core.InputBuilder(core.AutoTokenizer.from_pretrained(checkpoint['model_path']),
                                source_max_length=checkpoint['source_max_length'], context_max_length=checkpoint['context_max_length'])
    model = core._load_model(checkpoint, device=resolved)
    predictions = []
    with torch.no_grad(), core._auxiliary_mode(model):
        for pair in pairs:
            _, z, g, _ = region_readouts(model, builder, [pair], device=resolved)
            predictions.append(dict(sample_id=pair['sample_id'], group=pair['group'],
                                    local_logits=z.cpu().tolist(), global_logits=g.cpu().tolist(),
                                    source_chars=[r['source_chars'] for r in pair['regions']],
                                    actual_tokens=[r['actual_tokens'] for r in pair['regions']]))
    result = local_report(checkpoint_path, checkpoint, pairs, predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open('x') as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(output, result)
    return result
