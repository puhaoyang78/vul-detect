"""Prepare auditable C/C++ benchmark candidates without training any model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from importlib.metadata import version
import itertools
import json
from pathlib import Path
import re
import sys
import subprocess
from urllib.parse import urlsplit

from .prepare_primevul import single_function_language


# Preserve literals and token boundaries: deleting all whitespace would conflate
# `a + +b` with `a++b`, and deleting comments with a regex would damage strings.
TOKEN = re.compile(
    r'(?P<space>\s+)|(?P<comment>//(?:\\\r?\n|[^\n])*|/\*[\s\S]*?\*/)'
    r'|(?:u8|u|U|L)?R"(?P<delimiter>[^\s()\\]{0,16})\([\s\S]*?\)(?P=delimiter)"'
    r'|(?:u8|u|U|L)?"(?:\\[\s\S]|[^"\\])*"'
    r"|(?:u8|u|U|L)?'(?:\\[\s\S]|[^'\\])*'"
    r'|[a-zA-Z_$\u0080-\uffff][\w$]*'
    r"|(?:\d|\.\d)(?:[eEpP][+-]|[\w.'])*"
    r'|>>=|<<=|<=>|->\*|\.\.\.|##|::|\.\*|->|\+\+|--|<<|>>|<=|>='
    r'|==|!=|&&|\|\||\*=|/=|%=|\+=|-=|&=|\^=|\|=|[^\s]'
)
SPLITS = ('train', 'valid', 'test')
PRIORITY = (('sven', 'external_test'), ('primevul', 'test'), ('cleanvul', 'test'),
            ('primevul', 'valid'), ('cleanvul', 'valid'),
            ('primevul', 'train'), ('cleanvul', 'train'))


def normalized_source(source: str) -> str:
    return json.dumps([m.group() for m in TOKEN.finditer(source)
                       if m.lastgroup not in {'space', 'comment'}], ensure_ascii=False)


@lru_cache(maxsize=32768)
def shingles(source: str) -> frozenset[bytes]:
    tokens = [m.group().encode('utf-8') for m in TOKEN.finditer(source)
              if m.lastgroup not in {'space', 'comment'}]
    # Length framing is unambiguous even when a literal contains NUL. Encode
    # each token once rather than serializing five overlapping tokens per gram.
    tokens = [len(token).to_bytes(4, 'big') + token for token in tokens]
    width = min(5, len(tokens))
    return frozenset(hashlib.blake2b(b''.join(tokens[i:i + width]),
                                   digest_size=16).digest()
                     for i in range(max(1, len(tokens) - width + 1)))


class NearIndex:
    """Exact prefix-filtered Jaccard join at 0.90; no randomized LSH."""
    def __init__(self):
        self.postings = defaultdict(list)
        self.entries = []

    @staticmethod
    def prefix(signature):
        return sorted(signature)[:len(signature) - (9 * len(signature) + 9) // 10 + 1]

    def matches(self, item):
        matches = []
        for row in item['rows']:
            signature = shingles(row['function'])
            candidates = set(i for token in self.prefix(signature) for i in self.postings[token])
            for i in sorted(candidates):
                other, previous, partition, label, groups = self.entries[i]
                if (partition == (item['dataset'], item['split']) and label != row['label']
                        and groups.intersection(row.get('counterpart_groups', []))):
                    continue  # Declared repair counterparts are intentionally similar.
                if min(len(signature), len(other)) * 10 < max(len(signature), len(other)) * 9:
                    continue
                common = len(signature.intersection(other))
                union = len(signature) + len(other) - common
                if common * 10 >= union * 9:
                    matches.append(dict(kind='near_duplicate', retained_unit=previous,
                                        similarity_numerator=common, similarity_denominator=union))
                    break
        return matches

    def add(self, item):
        for row in item['rows']:
            signature = shingles(row['function'])
            index = len(self.entries)
            self.entries.append((signature, item['id'], (item['dataset'], item['split']),
                                 row['label'], set(row.get('counterpart_groups', []))))
            for token in self.prefix(signature):
                self.postings[token].append(index)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def repository(value: str) -> str:
    if not value or value == 'None':
        return ''
    if '://' not in value:
        value = 'https://' + value
    parsed = urlsplit(value)
    path = parsed.path.rstrip('/').removesuffix('.git')
    # GitHub host and owner/repository names are case insensitive.
    if parsed.hostname in {'github.com', 'www.github.com'}:
        return 'github.com/' + path.strip('/').lower()
    return (parsed.hostname or '').lower() + path


def commit_link(value: str) -> tuple[str, str]:
    if '/commit/' not in value:
        raise ValueError(f'unsupported commit URL: {value!r}')
    repo, commit = value.split('/commit/', 1)
    commit = commit.split('?', 1)[0].split('#', 1)[0].rstrip('/').lower()
    if not re.fullmatch(r'[0-9a-f]{7,40}', commit):
        raise ValueError(f'expected commit SHA or hexadecimal prefix: {value!r}')
    return repository(repo), commit


def prime_provenance(row: dict) -> tuple[str, str, dict]:
    repo = repository(row['project_url'])
    url = row.get('commit_url') or ''
    extra = {}
    if not repo and '/+/' in url:
        repo = repository(url.split('/+/', 1)[0])
        extra['repo_evidence'] = 'gitiles_commit_url'
    original = row['commit_id']
    if not isinstance(original, str) or not original:
        raise ValueError('PrimeVul commit_id must be a nonempty Git revision')
    commit = original.lower() if re.fullmatch(r'[0-9a-fA-F]{7,40}', original) else original
    for marker in ('/commit/', '/+/'):
        if marker in url:
            revision = url.split(marker, 1)[1].split('?', 1)[0].split('#', 1)[0].rstrip('/')
            if re.fullmatch(r'[0-9a-fA-F]{40}', revision) and len(commit) != 40:
                if not re.fullmatch(r'[0-9a-f]{7,40}', commit) or revision.lower().startswith(commit):
                    commit = revision.lower()
                    extra['source_commit'] = original
    return repo, commit, extra


def source_language(source: str, file_name: str) -> tuple[str | None, str]:
    suffix = Path(file_name).suffix
    if suffix == '.c':
        return 'c', 'extension'
    if suffix in {'.C', '.cc', '.cpp', '.cxx', '.c++', '.hpp', '.hh', '.hxx'}:
        return 'cpp', 'extension'
    if suffix == '.h':
        # .h is C-family evidence, not proof of C versus C++. This choice is
        # only a parser hint; retain that ambiguity in language_evidence.
        return single_function_language(source, file_name) or 'cpp', 'c_family_header'
    if suffix:
        return None, 'non_c_cpp_extension'
    return single_function_language(source), 'syntax_inference_missing_filename'


def record(dataset: str, key: str, code: str, label: int, language: str,
           repo: str, commit: str, path: Path, line: int, **extra) -> dict:
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f'{path}:{line}: empty/non-string function')
    return dict(dataset=dataset, sample_key=key, function=code, label=label,
                language=language, repo=repo, commit=commit,
                source_file=str(path), source_row=line, **extra)


def unit(rows: list[dict]) -> dict:
    keys = set()
    normalized = []
    for row in rows:
        keys.add(('exact_source', digest(row['function'])))
        normalized.append(digest(normalized_source(row['function'])))
        keys.add(('normalized_source', normalized[-1]))
        if row['repo'] and row['commit']:
            keys.add(('repo_commit', row['repo'] + '@' + row['commit']))
        keys.update(('pair_counterpart', group) for group in row.get('counterpart_groups', []))
    return {'id': rows[0].get('pair_id', rows[0]['sample_key']), 'rows': rows,
            'keys': keys, 'normalized': normalized,
            'dataset': rows[0]['dataset'], 'split': rows[0]['split']}


def count_units(units: list[dict]) -> dict:
    counts = Counter()
    for item in units:
        for row in item['rows']:
            counts[f"{row['language']}_{'vulnerable' if row['label'] else 'benign'}"] += 1
    return {'units': len(units), 'functions': sum(len(u['rows']) for u in units),
            'labels_by_language': dict(sorted(counts.items()))}


def jsonl_rows(path: Path):
    with path.open(encoding='utf-8') as handle:
        for line, text in enumerate(handle, 1):
            yield line, json.loads(text)


def load_sources(root: Path) -> tuple[list[dict], dict, list[dict]]:
    units, audit, excluded = [], {}, []
    prime = root / 'PrimeVul_v0.1'
    metadata = json.loads((prime / 'file_info.json').read_text())
    for split in SPLITS:
        path = prime / f'primevul_{split}.jsonl'
        counts = Counter()
        start = len(units)
        for line, row in jsonl_rows(path):
            if line % 25000 == 0:
                print(f'Reading PrimeVul {split}: {line} records', flush=True)
            label = row['target']
            if type(label) is not int or label not in (0, 1):
                raise ValueError(f'{path}:{line}: invalid target')
            counts[f'raw_{label}'] += 1
            filename = row.get('file_name') or ''
            if filename == 'None':
                filename = ''
            if not filename:
                filename = metadata.get(str(row['func_hash']), {}).get('file_name', '')
                counts['filename_restored_from_file_info'] += bool(filename)
            language, evidence = source_language(row['func'], filename)
            if language is None and not filename:
                # Upstream explicitly defines PrimeVul as C/C++. Failed parsing
                # of extracted/macro-containing snippets cannot disprove that
                # provenance. Do not invent a C-versus-C++ distinction here.
                language, evidence = 'c_cpp', 'upstream_c_cpp_unspecified'
            key = f"primevul:{row['idx']}"
            if language is None or not row['func'].strip():
                counts[f'unresolved_or_non_c_cpp_{label}'] += 1
                excluded.append(dict(unit_id=key, dataset='primevul', split=split, labels=[label],
                                     reason='unresolved_or_non_c_cpp',
                                     source_file=str(path.relative_to(root)), source_row=line))
                continue
            counts[f'{evidence}_{label}'] += 1
            repo, commit, provenance = prime_provenance(row)
            r = record('primevul', key, row['func'], label, language, repo, commit,
                       path.relative_to(root), line, split=split, official_split=split,
                       file_name=filename, language_evidence=evidence, idx=row['idx'], **provenance)
            units.append(unit([r]))
        audit[f'primevul_{split}'] = dict(counts=dict(sorted(counts.items())),
                                          eligible=count_units(units[start:]))
        print(f'PrimeVul {split}: {json.dumps(audit[f"primevul_{split}"])}', flush=True)
    del metadata
    for score in (4, 3):
        path = root / 'cleanvul' / f'vulnerability_score_{score}.csv'
        counts = Counter()
        start = len(units)
        with path.open(encoding='utf-8', newline='') as handle:
            for line, row in enumerate(csv.DictReader(handle), 1):
                if int(row['vulnerability_score']) != score:
                    raise ValueError(f'{path}: record {line}: score/file mismatch')
                language = row['extension']
                counts[f"raw_language_{language}"] += 1
                if language not in {'c', 'cpp'}:
                    continue
                pair_id = f'cleanvul:{score}:{line}'
                if not row['date']:
                    counts['missing_date_pairs'] += 1
                    excluded.append(dict(unit_id=pair_id, dataset='cleanvul', labels=[1, 0],
                                         score=score, language=language, reason='missing_commit_date'))
                    continue
                date = datetime.fromisoformat(row['date'].replace('Z', '+00:00'))
                if date.tzinfo is None:
                    raise ValueError(f'{pair_id}: timezone missing')
                date = date.astimezone(timezone.utc).isoformat()
                repo, commit = commit_link(row['commit_url'])
                rows = [record('cleanvul', f'{pair_id}:{side}', row[field], label,
                               language, repo, commit, path.relative_to(root), line,
                               pair_id=pair_id, score=score, commit_time=date,
                               file_name=row['file_name'], split='unassigned',
                               language_evidence='explicit_extension')
                        for side, field, label in [('before', 'func_before', 1),
                                                   ('after', 'func_after', 0)]]
                if normalized_source(rows[0]['function']) == normalized_source(rows[1]['function']):
                    counts['normalized_no_change_pairs'] += 1
                    excluded.append(dict(unit_id=pair_id, dataset='cleanvul', labels=[1, 0],
                                         score=score, language=language, reason='normalized_no_change'))
                    continue
                units.append(unit(rows))
        audit[f'cleanvul_score_{score}'] = dict(counts=dict(sorted(counts.items())),
                                                eligible=count_units(units[start:]))
    counts = Counter()
    start = len(units)
    for path in sorted((root / 'sven' / 'data_train_val').glob('*/*.jsonl')):
        for line, row in jsonl_rows(path):
            suffix = Path(row['file_name']).suffix
            counts[f'raw_extension_{suffix}'] += 1
            language, evidence = source_language(row['func_src_before'], row['file_name'])
            if language is None:
                continue
            pair_id = f'sven:{path.parent.name}:{path.stem}:{line}'
            counts[f'c_cpp_pairs_{language}'] += 1
            repo, commit = commit_link(row['commit_link'])
            rows = [record('sven', f'{pair_id}:{side}', row[field], label, language,
                           repo, commit, path.relative_to(root), line, pair_id=pair_id,
                           split='external_test', original_split=path.parent.name,
                           file_name=row['file_name'], language_evidence=evidence,
                           function_name=row['func_name'], cwe=row['vul_type'])
                    for side, field, label in [('before', 'func_src_before', 1),
                                               ('after', 'func_src_after', 0)]]
            if normalized_source(rows[0]['function']) == normalized_source(rows[1]['function']):
                counts['normalized_no_change_pairs'] += 1
                excluded.append(dict(unit_id=pair_id, dataset='sven', split='external_test',
                                     labels=[1, 0], language=language, reason='normalized_no_change'))
                continue
            units.append(unit(rows))
    audit['sven'] = dict(counts=dict(sorted(counts.items())), eligible=count_units(units[start:]))
    if not audit['sven']['eligible']['units']:
        raise ValueError('no SVEN pairs found')
    if len({u['id'] for u in units}) != len(units):
        raise ValueError('duplicate unit IDs')
    return units, audit, excluded


def official_counterparts(root: Path) -> tuple[list[tuple[str, str]], dict]:
    edges, counts = [], {}
    for split in SPLITS:
        path = root / 'PrimeVul_v0.1' / f'primevul_{split}_paired.jsonl'
        iterator = iter(jsonl_rows(path))
        count = 0
        for _, before in iterator:
            following = next(iterator, None)
            if following is None or before['target'] != 1 or following[1]['target'] != 0:
                raise ValueError(f'{path}: expected adjacent vulnerable/benign pairs')
            after = following[1]
            edges.append((digest(normalized_source(before['func'])),
                          digest(normalized_source(after['func']))))
            count += 1
        counts[split] = count
    return edges, counts


def canonicalize_commits(units: list[dict]) -> dict:
    by_repo = defaultdict(set)
    for item in units:
        for row in item['rows']:
            by_repo[row['repo']].add(row['commit'])
    resolved, unresolved, revisions = {}, [], []
    for repo, commits in sorted(by_repo.items()):
        for commit in sorted(commits):
            if not re.fullmatch(r'[0-9a-f]{7,40}', commit):
                resolved[(repo, commit)] = commit
                revisions.append(dict(repo=repo, revision=commit))
                continue
            if len(commit) == 40:
                resolved[(repo, commit)] = commit
                continue
            matches = sorted((c for c in commits if re.fullmatch(r'[0-9a-f]{7,40}', c)
                              and c.startswith(commit)), key=lambda c: (-len(c), c))
            longest = matches[0]
            if any(not longest.startswith(c) for c in matches):
                raise ValueError(f'ambiguous local commit prefix: {repo}@{commit}')
            resolved[(repo, commit)] = longest
            if len(longest) != 40:
                unresolved.append(dict(repo=repo, source_commit=commit, local_commit=longest))
    expanded = 0
    for item in units:
        item['keys'] = {key for key in item['keys'] if key[0] != 'repo_commit'}
        for row in item['rows']:
            original = row['commit']
            row['commit'] = resolved[(row['repo'], original)]
            if row['commit'] != original:
                row.setdefault('source_commit', original)
                expanded += 1
            if row['repo']:
                item['keys'].add(('repo_commit', row['repo'] + '@' + row['commit']))
    return dict(expanded_function_rows=expanded, unresolved_prefixes=unresolved,
                unresolved_git_revisions=revisions,
                missing_repo_function_rows=sum(not r['repo'] for u in units for r in u['rows']))


def link_counterparts(units: list[dict], official_edges: list[tuple[str, str]]) -> list[dict]:
    parents = {}

    def find(x):
        parents.setdefault(x, x)
        root = x
        while parents[root] != root:
            root = parents[root]
        while parents[x] != x:
            previous = parents[x]
            parents[x] = root
            x = previous
        return root

    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    for a, b in official_edges:
        union(a, b)
    for item in units:
        if item['dataset'] != 'primevul':
            hashes = sorted(value for kind, value in item['keys'] if kind == 'normalized_source')
            for a, b in zip(hashes, hashes[1:]):
                union(a, b)
    linked = []
    for item in units:
        groups = sorted({find(value) for kind, value in item['keys']
                         if kind == 'normalized_source' and value in parents})
        linked.append(dict(item, rows=[dict(r, counterpart_groups=groups) for r in item['rows']],
                           keys=item['keys'] | {('pair_counterpart', group) for group in groups}))
    return linked


def clean_label_conflicts(units: list[dict]) -> tuple[list[dict], list[dict]]:
    labels = defaultdict(set)
    for item in units:
        for row, key in zip(item['rows'], item['normalized']):
            labels[key].add(row['label'])
    conflicts = {key for key, values in labels.items() if len(values) > 1}
    bad = {u['id'] for u in units if any(k == 'normalized_source' and v in conflicts for k, v in u['keys'])}
    blocked_groups = {key for u in units if u['id'] in bad for key in u['keys'] if key[0] == 'pair_counterpart'}
    rejected = [u for u in units if u['id'] in bad or u['keys'].intersection(blocked_groups)]
    removed = {u['id'] for u in rejected}
    return [u for u in units if u['id'] not in removed], [
        dict(unit_id=u['id'], dataset=u['dataset'], split=u['split'],
             labels=[r['label'] for r in u['rows']],
             reason='label_conflict' if u['id'] in bad else 'label_conflict_counterpart')
        for u in rejected]


def temporal_split(units: list[dict]) -> tuple[list[dict], dict]:
    """Use nearest 70%/85% cumulative timestamp boundaries, never split ties."""
    by_time = defaultdict(list)
    commit_dates = {}
    for item in units:
        row = item['rows'][0]
        date = row['commit_time']
        key = (row['repo'], row['commit'])
        if key in commit_dates and commit_dates[key] != date:
            raise ValueError(f'inconsistent commit dates: {key}')
        commit_dates[key] = date
        by_time[date].append(item)
    dates = sorted(by_time)
    cumulative = [0]
    for date in dates:
        cumulative.append(cumulative[-1] + len(by_time[date]))
    n = len(units)
    first = min(range(len(cumulative)), key=lambda i: (abs(cumulative[i] * 100 - n * 70), i))
    second = min(range(first, len(cumulative)), key=lambda i: (abs(cumulative[i] * 100 - n * 85), i))
    assigned = []
    for i, date in enumerate(dates):
        split = 'train' if i < first else 'valid' if i < second else 'test'
        for item in sorted(by_time[date], key=lambda u: u['id']):
            assigned.append(unit([dict(row, split=split) for row in item['rows']]))
    return assigned, dict(target_ratios=[0.70, 0.15, 0.15],
                          train_end=dates[first - 1] if first else None,
                          valid_end=dates[second - 1] if second else None,
                          before_leakage_filter={s: count_units([u for u in assigned if u['split'] == s])
                                                 for s in SPLITS})


def overlap_report(units: list[dict]) -> dict:
    """Count shared keys and affected units, not quadratic matching row pairs."""
    indexes = defaultdict(lambda: defaultdict(list))
    commit_sources = defaultdict(lambda: defaultdict(set))
    for item in units:
        partition = f"{item['dataset']}/{item['split']}"
        for key in item['keys']:
            indexes[key][partition].append(item['id'])
            if key[0] == 'repo_commit':
                commit_sources[key[1]][partition].update(item['normalized'])
    counts = Counter()
    affected = defaultdict(set)
    examples = defaultdict(list)
    commit_only = Counter()
    for (kind, value), partitions in sorted(indexes.items()):
        for a, b in itertools.combinations(sorted(partitions), 2):
            name = f'{a} <-> {b}: {kind}'
            counts[name] += 1
            if kind == 'repo_commit' and not commit_sources[value][a].intersection(commit_sources[value][b]):
                commit_only[name] += 1
            affected[name].update(partitions[a] + partitions[b])
            if len(examples[name]) < 3:
                examples[name].append([partitions[a][0], partitions[b][0]])
    return {name: dict(shared_keys=counts[name], affected_units=len(affected[name]),
                       example_unit_ids=examples[name],
                       **({'commits_without_shared_normalized_source': commit_only[name]}
                          if name.endswith(': repo_commit') else {})) for name in sorted(counts)}


def select(units: list[dict]) -> tuple[list[dict], list[dict]]:
    """Holdout first; source/counterpart/near duplicates, never commit-only purge."""
    selected, excluded = [], []
    seen = {}
    near = NearIndex()

    def matches(item):
        found = [dict(kind=key[0], retained_unit=seen[key]) for key in sorted(item['keys'])
                 if key in seen and key[0] != 'repo_commit']
        return found or near.matches(item)

    def reject(item, reason, found=None):
        excluded.append(dict(unit_id=item['id'], dataset=item['dataset'], split=item['split'],
                             reason=reason, labels=[r['label'] for r in item['rows']],
                             matches=found or []))

    for dataset, split in PRIORITY:
        pool = sorted((u for u in units if (u['dataset'], u['split']) == (dataset, split)),
                      key=lambda u: (-u['rows'][0]['label'], digest('benchmark-benign-v1:' + u['id']), u['id']))
        # Check complete declared counterpart families against higher-priority
        # partitions before sampling any of their members.
        blocked_groups, direct = {}, {}
        for item in pool:
            groups = [key for key in item['keys'] if key[0] == 'pair_counterpart']
            if groups:
                found = matches(item)
                if found:
                    direct[item['id']] = found
                    for key in groups:
                        blocked_groups[key] = found
        chosen = []
        vulnerable_count = benign_count = 0
        partition_keys = set()
        for item in pool:
            found = direct.get(item['id'])
            if not found:
                found = next((blocked_groups[k] for k in sorted(item['keys']) if k in blocked_groups), None)
            if found:
                reject(item, 'duplicate_or_counterpart', found)
                continue
            label = item['rows'][0]['label']
            if dataset == 'primevul' and label == 0 and benign_count >= vulnerable_count:
                reject(item, 'benign_balance_not_selected')
                continue
            # Pair-group equality inside this partition is allowed, while
            # exact/normalized repeats and unrelated near clones are removed.
            found = [dict(kind=k[0], retained_unit=seen[k]) for k in sorted(item['keys'])
                     if k in seen and k[0] != 'repo_commit'
                     and not (k[0] == 'pair_counterpart' and k in partition_keys)]
            found = found or near.matches(item)
            if found:
                reject(item, 'duplicate_or_counterpart', found)
                continue
            chosen.append(item)
            vulnerable_count += label == 1
            benign_count += label == 0
            for key in sorted(item['keys']):
                seen.setdefault(key, item['id'])
                partition_keys.add(key)
            near.add(item)
        if dataset == 'primevul' and vulnerable_count != benign_count:
            raise ValueError(f'{split}: insufficient benign rows after duplicate removal')
        selected.extend(chosen)
        print(f'In-memory selection {dataset}/{split}: {len(chosen)} units', flush=True)
    return selected, excluded


def validate(units: list[dict]) -> dict:
    overlaps = overlap_report(units)
    if any(not name.endswith(': repo_commit') for name in overlaps):
        raise ValueError('remaining cross-partition leakage')
    near = NearIndex()
    for item in units:
        if near.matches(item):
            raise ValueError('remaining near duplicate outside declared counterpart pair')
        near.add(item)
    rows = [r for u in units for r in u['rows']]
    if len({r['sample_key'] for r in rows}) != len(rows):
        raise ValueError('duplicate sample keys')
    labels = defaultdict(set)
    for item in units:
        for row, key in zip(item['rows'], item['normalized']):
            labels[key].add(row['label'])
    if any(len(values) > 1 for values in labels.values()):
        raise ValueError('remaining normalized-source label conflict')
    for item in units:
        if item['dataset'] == 'primevul':
            if any(r['split'] != r['official_split'] for r in item['rows']):
                raise ValueError('PrimeVul official split changed')
        else:
            if (len(item['rows']) != 2 or {r['label'] for r in item['rows']} != {0, 1}
                    or len({r['split'] for r in item['rows']}) != 1):
                raise ValueError('broken pair')
        if item['dataset'] == 'sven' and item['split'] != 'external_test':
            raise ValueError('SVEN is external-test-only')
    for split in SPLITS:
        labels = Counter(r['label'] for r in rows if r['dataset'] == 'primevul' and r['split'] == split)
        if labels[0] != labels[1]:
            raise ValueError('unbalanced PrimeVul split')
    clean_dates = [[r['commit_time'] for r in rows if r['dataset'] == 'cleanvul' and r['split'] == s]
                   for s in SPLITS]
    for a, b in itertools.combinations(clean_dates, 2):
        if a and b and max(a) >= min(b):
            raise ValueError('CleanVul temporal order violation')
    return dict(cross_partition_exact_source=0, cross_partition_normalized_source=0,
                cross_partition_pair_counterpart=0, near_duplicates_outside_counterparts=0,
                repo_commit_shared_keys=sum(v['shared_keys'] for k, v in overlaps.items()
                                           if k.endswith(': repo_commit')), broken_pairs=0,
                label_conflicts=0, sven_outside_external_test=0, primevul_balanced=True)


def commit_disjoint(units: list[dict], candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    commits = {r['commit'] for u in units if u['dataset'] == 'sven' for r in u['rows']}
    short_commits = [c for c in sorted(commits) if len(c) < 40]

    def shared_commit(item):
        for row in item['rows']:
            sha = row['commit']
            if not sha:
                continue
            if sha in commits or any(sha.startswith(c) for c in short_commits):
                return True
            if len(sha) < 40 and any(c.startswith(sha) for c in commits):
                return True
        return False

    unknown = {u['id'] for u in candidates if u['split'] in {'train', 'valid'}
               and any(not re.fullmatch(r'[0-9a-f]{7,40}', r['commit']) for r in u['rows'])}
    removed = unknown | {u['id'] for u in candidates if u['split'] in {'train', 'valid'} and shared_commit(u)}
    groups = {k for u in candidates if u['id'] in removed for k in u['keys'] if k[0] == 'pair_counterpart'}
    removed.update(u['id'] for u in candidates if u['split'] in {'train', 'valid'} and u['keys'] & groups)
    exclusions = [dict(unit_id=u['id'], dataset=u['dataset'], split=u['split'],
                       reason='unresolved_commit_revision' if u['id'] in unknown else 'sven_commit_or_counterpart',
                       labels=[r['label'] for r in u['rows']])
                  for u in candidates if u['id'] in removed]
    # Refill benign rows from the same official split using the same ranking;
    # never discard a remaining vulnerable merely to restore class balance.
    retained, selection_exclusions = select([u for u in candidates if u['id'] not in removed])
    exclusions.extend(selection_exclusions)
    if any(shared_commit(u) for u in retained if u['split'] in {'train', 'valid'}):
        raise ValueError('strict experiment has SVEN commit overlap')
    for partition in PRIORITY[:3]:
        if ({u['id'] for u in units if (u['dataset'], u['split']) == partition}
                != {u['id'] for u in retained if (u['dataset'], u['split']) == partition}):
            raise ValueError('strict experiment changed a test cohort')
    return retained, exclusions


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def deletion_statistics(rows: list[dict], exclusions: list[dict]) -> dict:
    """Reconcile persisted units, using disjoint reason and evidence-set bins."""
    retained = {}
    for row in rows:
        key = row.get('pair_id') or row['sample_key']
        retained[key] = (row['dataset'], row['split'])
    groups = {}
    for dataset, split in PRIORITY:
        labels = (1, 0) if dataset == 'primevul' else (None,)
        for label in labels:
            name = f'{dataset}/{split}' + (f'/label_{label}' if label is not None else '/pairs')
            kept = {r.get('pair_id') or r['sample_key'] for r in rows
                    if (r['dataset'], r['split']) == (dataset, split)
                    and (label is None or r['label'] == label)}
            deleted = [e for e in exclusions if (e['dataset'], e.get('split')) == (dataset, split)
                       and (label is None or e['labels'] == [label])]
            reasons = Counter(e['reason'] for e in deleted)
            evidence_sets, evidence_targets = Counter(), Counter()
            for entry in deleted:
                matches = entry.get('matches', [])
                if not matches:
                    continue
                evidence_sets['+'.join(sorted({m['kind'] for m in matches}))] += 1
                targets = set()
                for match in matches:
                    target = retained[match['retained_unit']]
                    targets.add('/'.join(target))
                evidence_targets[' + '.join(sorted(targets))] += 1
            groups[name] = dict(input_units=len(kept) + len(deleted), retained_units=len(kept),
                                deleted_units=len(deleted), reasons=dict(sorted(reasons.items())),
                                duplicate_evidence_sets=dict(sorted(evidence_sets.items())),
                                duplicate_retained_partitions=dict(sorted(evidence_targets.items())))
    source = Counter(f"{e['dataset']}/{e['reason']}" for e in exclusions if not e.get('split'))
    return dict(unit='PrimeVul functions; CleanVul/SVEN complete pairs',
                evidence_note='Evidence-set bins are disjoint; multiple kinds within one bin are not additive. Matches may be inherited from a counterpart. Near matching runs only if exact/normalized/counterpart lookup found nothing.',
                groups=groups, source_exclusions=dict(sorted(source.items())))


def audit_deletions(output: Path) -> dict:
    report = {}
    for score in (4, 3):
        folder = output / f'score_{score}'
        rows = [row for _, row in jsonl_rows(folder / 'manifest.jsonl')]
        exclusions = [row for _, row in jsonl_rows(folder / 'exclusions.jsonl')]
        report[str(score)] = deletion_statistics(rows, exclusions)
    (output / 'deletion_statistics.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('DELETION AUDIT ' + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return report


def prepare(root: Path, output: Path, final_score: int | None = None) -> dict:
    if final_score not in {None, 3, 4}:
        raise ValueError('final_score must be 3, 4, or None')
    if output.resolve() == root.resolve() or root.resolve() in output.resolve().parents:
        raise ValueError('output must be outside the raw source directory')
    csv.field_size_limit(sys.maxsize)
    shingles.cache_clear()
    units, raw, source_exclusions = load_sources(root)
    commit_resolution = canonicalize_commits(units)
    official_edges, paired_counts = official_counterparts(root)
    sven_git = root / 'sven' / '.git'
    revision = subprocess.run(['git', '-C', str(root / 'sven'), 'rev-parse', 'HEAD'],
                              check=True, capture_output=True, text=True).stdout.strip() if sven_git.exists() else None
    report = dict(schema_version=1, raw=raw,
                  environment=dict(python=sys.version.split()[0],
                                   packages={name: version(name) for name in ('tree-sitter', 'tree-sitter-c', 'tree-sitter-cpp')},
                                   sven_revision=revision),
                  source_exclusion_counts=dict(sorted(Counter(
        e['reason'] for e in source_exclusions).items())),
        policy=dict(priority=[list(p) for p in PRIORITY],
                    benign_selection='SHA256(benchmark-benign-v1:<sample_key>), then sample_key',
                    normalization='C/C++ lexical tokens; comments/formatting ignored; literals and identifiers preserved',
                    near_duplicate='token 5-gram set Jaccard >= 0.90; deterministic exact prefix join; 128-bit shingle hashes',
                    near_counterpart_exception='different labels in the same declared counterpart group and same partition',
                    label_conflict='quarantine all exact/normalized conflicting labels and their counterpart groups',
                    repo_commit='report only in main benchmark; strict variant purges SVEN commit SHA/prefix overlap from train/valid across repository aliases and forks',
                    strict_benign_selection='same official split and deterministic ranking, refill after commit filtering',
                    language='known extension/explicit metadata; syntax inference for missing filenames; unresolved PrimeVul retains upstream C/C++ family as c_cpp',
                    primevul_language_reference='https://github.com/DLVulDet/PrimeVul#-overview',
                    source_row='1-based JSONL line or CSV data-record ordinal (excluding header)',
                    missing_cleanvul_dates='excluded; never imputed',
                    pair_labels='dataset before=1/after=0, not independently adjudicated',
                    primevul_chronology='official split membership retained; no commit dates in JSONL'),
        commit_prefix_resolution=commit_resolution, official_primevul_pair_counts=paired_counts, scenarios={})
    print('RAW AUDIT ' + json.dumps(raw, ensure_ascii=False, sort_keys=True), flush=True)
    scenario_results = {}
    for score in (4, 3):
        clean = [u for u in units if u['dataset'] == 'cleanvul' and u['rows'][0]['score'] >= score]
        clean, temporal = temporal_split(clean)
        candidate = [u for u in units if u['dataset'] != 'cleanvul'] + clean
        candidate = link_counterparts(candidate, official_edges)
        overlaps = overlap_report(candidate)
        consistent, conflicts = clean_label_conflicts(candidate)
        selected, excluded = select(consistent)
        excluded = conflicts + excluded
        checks = validate(selected)
        rows = sorted((r for u in selected for r in u['rows']),
                      key=lambda r: (PRIORITY.index((r['dataset'], r['split'])), r['sample_key']))
        stats = {f'{d}/{s}': count_units([u for u in selected if (u['dataset'], u['split']) == (d, s)])
                 for d, s in PRIORITY}
        losses = Counter(e['split'] for e in excluded if e['dataset'] == 'primevul' and e['labels'] == [1])
        strict, strict_excluded = commit_disjoint(selected, consistent)
        strict_checks = validate(strict)
        strict_stats = {f'{d}/{s}': count_units([u for u in strict if (u['dataset'], u['split']) == (d, s)])
                        for d, s in PRIORITY}
        scenario = dict(cleanvul_min_score=score, temporal=temporal, final=stats,
                        primevul_vulnerable_removed=dict(sorted(losses.items())),
                        overlap_before_selection=overlaps, overlap_after_selection=overlap_report(selected), checks=checks,
                        sven_commit_disjoint=dict(final=strict_stats, checks=strict_checks,
                                                 train_valid_shared_sven_commits=0,
                                                 exclusion_counts=dict(sorted(Counter(e['reason'] for e in strict_excluded).items()))),
                        exclusion_counts=dict(sorted(Counter(e['reason'] for e in excluded).items())))
        scenario['duplicate_match_counts'] = dict(sorted(Counter(
            kind for e in excluded for kind in {m['kind'] for m in e.get('matches', [])}).items()))
        report['scenarios'][str(score)] = scenario
        strict_rows = sorted((r for u in strict for r in u['rows']),
                             key=lambda r: (PRIORITY.index((r['dataset'], r['split'])), r['sample_key']))
        scenario_results[score] = (rows, strict_rows)
        folder = output / f'score_{score}'
        write_jsonl(folder / 'manifest.jsonl', rows)
        write_jsonl(folder / 'manifest_sven_commit_disjoint.jsonl', strict_rows)
        write_jsonl(folder / 'exclusions_sven_commit_disjoint.jsonl', sorted(strict_excluded, key=lambda e: e['unit_id']))
        applicable_source_exclusions = [e for e in source_exclusions if e.get('score', 4) >= score]
        write_jsonl(folder / 'exclusions.jsonl', sorted(applicable_source_exclusions + excluded,
                                                       key=lambda e: e['unit_id']))
        print(f'FINAL CANDIDATE score>={score}: ' + json.dumps(stats, ensure_ascii=False, sort_keys=True), flush=True)
        print(f'PrimeVul vulnerable removed score>={score}: {dict(losses)}', flush=True)
        print(f'SVEN COMMIT-DISJOINT score>={score}: ' + json.dumps(strict_stats, sort_keys=True), flush=True)
    if final_score is not None:
        write_jsonl(output / 'manifest.jsonl', scenario_results[final_score][0])
        write_jsonl(output / 'manifest_sven_commit_disjoint.jsonl', scenario_results[final_score][1])
    report['selected_cleanvul_min_score'] = final_score
    output.mkdir(parents=True, exist_ok=True)
    (output / 'statistics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    audit_deletions(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('/home/PublicData/PHY-data/vul_detect/data'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/benchmark'))
    parser.add_argument('--final-score', type=int, choices=(3, 4),
                        help='Choose only after reviewing both printed candidate counts.')
    parser.add_argument("--audit-deletions", action="store_true",
                        help="Reconcile existing manifests and exclusion ledgers without rebuilding splits.")
    args = parser.parse_args()
    csv.field_size_limit(sys.maxsize)
    if args.audit_deletions:
        audit_deletions(args.output_dir)
    else:
        prepare(args.source_dir, args.output_dir, args.final_score)


if __name__ == '__main__':
    main()
