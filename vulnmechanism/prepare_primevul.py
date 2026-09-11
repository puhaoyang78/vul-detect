"""Sample original PrimeVul binary labels within its official splits.

Run: python -m vulnmechanism.prepare_primevul
Language follows known source extensions, otherwise strict syntax parsing.
Exact source duplicates keep their first occurrence in train/valid/test order;
records are never moved between splits. No paired files are read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from .syntax import identifier, parser_for, walk


def single_function_language(source: str, file_name: str = '') -> str | None:
    encoded = source.encode('utf-8')
    suffix = Path(file_name).suffix
    languages = ('c',) if suffix == '.c' else (
        ('cpp',) if suffix in {'.C', '.cc', '.cpp', '.cxx', '.c++', '.hpp', '.hh', '.hxx'}
        else ('c', 'cpp')
    )
    for language in languages:
        root = parser_for(language).parse(encoded).root_node
        if root.has_error:
            continue
        nodes = [node for node in root.named_children if node.type != 'comment']
        if len(nodes) != 1:
            continue
        top = nodes[0]
        while top.type == 'template_declaration':
            children = [n for n in top.named_children
                        if n.type not in {'template_parameter_list', 'comment'}]
            if len(children) != 1:
                break
            top = children[0]
        if top.type != 'function_definition':
            continue
        # C++ grammar also accepts arbitrary typeless names as constructors.
        # A standalone constructor/destructor must identify its owning class.
        if top.child_by_field_name('type') is None:
            declarator = top.child_by_field_name('declarator')
            while declarator is not None and declarator.type != 'qualified_identifier':
                declarator = declarator.child_by_field_name('declarator')
            if language != 'cpp' or declarator is None:
                continue
            scope = declarator.child_by_field_name('scope')
            name = declarator.child_by_field_name('name')
            if scope is None or name is None:
                continue
            while scope.type == 'qualified_identifier':
                scope = scope.child_by_field_name('name')
            if scope.type == 'template_type':
                scope = scope.child_by_field_name('name')
            owner = encoded[scope.start_byte:scope.end_byte].decode()
            method = encoded[name.start_byte:name.end_byte].decode()
            if name.type != 'operator_cast' and method not in {owner, '~' + owner}:
                continue
        if sum(n.type == 'function_definition' for n in walk(root)) != 1:
            continue
        if identifier(top.child_by_field_name('declarator'), encoded):
            return language
    return None


def prepare(source_dir: Path, output: Path, seed: int = 42) -> dict:
    seen: dict[str, tuple[str, int]] = {}
    selected = []
    stats = {}
    for split, per_class in (('train', 1000), ('valid', 200), ('test', 200)):
        counts = Counter()
        pools = {0: [], 1: []}
        path = source_dir / f'primevul_{split}.jsonl'
        with path.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                label, function, idx = row['target'], row['func'], row['idx']
                if type(label) is not int or label not in (0, 1):
                    raise ValueError(f'{path}:{line_number}: invalid target {label!r}')
                if not isinstance(function, str) or type(idx) is not int:
                    raise ValueError(f'{path}:{line_number}: invalid func or idx type')
                counts[f'raw_{label}'] += 1
                if not function.strip():
                    counts['empty'] += 1
                    continue
                if function in seen:
                    previous_split, previous_label = seen[function]
                    counts['duplicates'] += 1
                    counts['cross_split_duplicates'] += previous_split != split
                    counts['conflicting_label_duplicates'] += previous_label != label
                    continue
                seen[function] = (split, label)
                language = single_function_language(function, row.get('file_name') or '')
                if language is None:
                    counts['invalid_single_function'] += 1
                    continue
                record = {'sample_key': f'primevul:{idx}', 'function': function,
                          'label': label, 'language': language, 'split': split}
                for key in ('idx', 'project', 'commit_id', 'func_hash', 'file_name'):
                    if key in row:
                        record[key] = row[key]
                pools[label].append(record)
        rng = random.Random(seed)
        for label in (0, 1):
            counts[f'eligible_{label}'] = len(pools[label])
            amount = min(per_class, len(pools[label]))
            counts[f'selected_{label}'] = amount
            counts[f'shortfall_{label}'] = per_class - amount
            selected.extend(rng.sample(pools[label], amount))
        stats[split] = dict(counts)
        print(f'{split}: {dict(counts)}', flush=True)
    if len({r['sample_key'] for r in selected}) != len(selected):
        raise ValueError('selected sample_key values are not unique')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    with output.open(encoding='utf-8') as handle:
        restored = [json.loads(line) for line in handle]
    assert restored == selected
    assert len({r['function'] for r in restored}) == len(restored)
    assert all(set(('sample_key', 'function', 'label', 'language', 'split')) <= r.keys()
               for r in restored)
    for split in stats:
        for label in (0, 1):
            assert sum(r['split'] == split and r['label'] == label for r in restored) == stats[split][f'selected_{label}']
    return {'seed': seed, 'total': len(restored), 'splits': stats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path,
                        default=Path('/home/PublicData/PHY-data/vul_detect/data/PrimeVul_v0.1'))
    parser.add_argument('--output', type=Path, default=Path('data/functions.jsonl'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_dir, args.output, args.seed), indent=2))


if __name__ == '__main__':
    main()
