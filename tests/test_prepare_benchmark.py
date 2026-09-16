import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from vulnmechanism.prepare_benchmark import (
    NearIndex, canonicalize_commits, clean_label_conflicts, commit_disjoint, commit_link, digest,
    link_counterparts, normalized_source, overlap_report, prepare, prime_provenance, repository,
    select, shingles, source_language, temporal_split, unit, validate,
)


def make(dataset, key, split, code, label=1, commit='', date='2000-01-01T00:00:00+00:00'):
    row = dict(dataset=dataset, sample_key=key, split=split, function=code,
               label=label, language='c', repo='github.com/o/r', commit=commit,
               official_split=split, commit_time=date)
    if dataset == 'primevul':
        return unit([row])
    row['pair_id'] = key
    fixed = dict(row, sample_key=key + ':after', label=0,
                 function=code.replace('return 1;', 'return 0;'))
    return unit([row, fixed])


class BenchmarkTests(unittest.TestCase):
    def test_training_split_boundary(self):
        from vulnmechanism.model import _record_split

        records = [dict(sample_key='heldout', split='external_test'),
                   dict(sample_key='train', split='train'),
                   dict(sample_key='valid', split='valid')]
        self.assertEqual(_record_split(records[0]), 'external_test')
        self.assertEqual([r['sample_key'] for r in records if _record_split(r) in {'train', 'valid'}],
                         ['train', 'valid'])
        with self.assertRaisesRegex(ValueError, 'unknown explicit dataset split'):
            _record_split(dict(sample_key='typo', split='external_typo'))

    def test_lexical_normalization(self):
        self.assertEqual(normalized_source('int f(){ /* x */ return 1; }'),
                         normalized_source('int\r\nf ( ) {return 1 ;}// tail'))
        for a, b in [('"a b"', '"ab"'), ('"// x"', '"// y"'),
                     ('a + +b', 'a++b'), ('foo bar', 'foobar'),
                     ('R"tag(a /* b */)tag"', 'R"tag(a b)tag"'),
                     ("' '", "''"), ('return 1;', 'return 2;'), ('1e+2', '1e + 2')]:
            self.assertNotEqual(normalized_source(a), normalized_source(b))

    def test_repository_identity(self):
        self.assertEqual(repository('https://GitHub.com/Owner/Repo.git/'), 'github.com/owner/repo')
        self.assertEqual(commit_link('github.com/Owner/Repo/commit/' + 'a' * 40),
                         ('github.com/owner/repo', 'a' * 40))
        with self.assertRaises(ValueError):
            commit_link('github.com/o/r/commit/short')

    def test_short_commit_resolution_and_ambiguity(self):
        self.assertEqual(commit_link('https://github.com/o/r/commit/abcdef1')[1], 'abcdef1')
        a = make('primevul', 'a', 'train', 'int a(){}', commit='abcdef1' + '0' * 33)
        b = make('cleanvul', 'b', 'test', 'int b(){return 1;}', commit='abcdef1')
        report = canonicalize_commits([a, b])
        self.assertEqual(report['expanded_function_rows'], 2)
        self.assertEqual(b['rows'][0]['source_commit'], 'abcdef1')
        self.assertEqual(a['rows'][0]['commit'], b['rows'][0]['commit'])
        c = make('primevul', 'c', 'test', 'int c(){}', commit='abcdef1' + '1' * 33)
        b['rows'][0]['commit'] = 'abcdef1'
        with self.assertRaises(ValueError):
            canonicalize_commits([a, b, c])
        missing = make('primevul', 'missing', 'train', 'int missing(){}', commit='b' * 40)
        missing['rows'][0]['repo'] = ''
        missing = unit(missing['rows'])
        report = canonicalize_commits([missing])
        self.assertEqual(report['missing_repo_function_rows'], 1)
        self.assertFalse(any(k[0] == 'repo_commit' for k in missing['keys']))

    def test_gitiles_and_revision_metadata(self):
        sha = 'a' * 40
        row = dict(project_url='None', commit_url='https://android.googlesource.com/platform/x/+/' + sha,
                   commit_id=sha)
        repo, commit, evidence = prime_provenance(row)
        self.assertEqual(repo, 'android.googlesource.com/platform/x')
        self.assertEqual(commit, sha)
        self.assertEqual(evidence['repo_evidence'], 'gitiles_commit_url')
        item = make('primevul', 'revision', 'train', 'int rev(){return 1;}', commit='Release~2')
        audit = canonicalize_commits([item])
        self.assertEqual(audit['unresolved_git_revisions'][0]['revision'], 'Release~2')
        ext = make('sven', 'ext', 'external_test', 'int ext(){return 1;}', commit=sha)
        benign = make('primevul', 'b', 'train', 'int b(){return 0;}', label=0, commit='b' * 40)
        main, _ = select([item, ext, benign])
        strict, excluded = commit_disjoint(main, [item, ext, benign])
        self.assertFalse(any(u['id'] == 'revision' for u in strict))
        self.assertTrue(any(e['reason'] == 'unresolved_commit_revision' for e in excluded))

    def test_language_evidence_and_macro_retention(self):
        self.assertEqual(source_language('EXPORT int f() {}', 'x.c'), ('c', 'extension'))
        self.assertEqual(source_language('int f() {}', 'x.C'), ('cpp', 'extension'))
        self.assertIsNone(source_language('int f() {}', 'x.java')[0])
        self.assertIsNone(source_language('def f(): pass', '')[0])

    def test_temporal_ties_and_input_order(self):
        rows = [make('cleanvul', f'p{i}', 'unassigned', f'int f{i}() {{return 1;}}',
                     commit=str(i // 2), date=f'2000-01-{i // 2 + 1:02}T00:00:00+00:00')
                for i in range(20)]
        first, report = temporal_split(rows)
        second, other = temporal_split(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(report, other)
        memberships = {}
        for item in first:
            commit = item['rows'][0]['commit']
            memberships.setdefault(commit, set()).add(item['split'])
        self.assertTrue(all(len(s) == 1 for s in memberships.values()))
        self.assertEqual([report['before_leakage_filter'][s]['units'] for s in ('train', 'valid', 'test')],
                         [14, 2, 4])
        rows[1]['rows'][0]['commit_time'] = '2000-02-01T00:00:00+00:00'
        with self.assertRaises(ValueError):
            temporal_split(rows)

    def test_test_first_pair_atomicity_and_balance(self):
        ext = make('sven', 'external', 'external_test', 'int ext(){return 1;}', commit='ext')
        pair = make('cleanvul', 'pair', 'train', 'int pair(){return 1;}')
        pair['rows'][1]['function'] = 'int ext() { /* identical tokens */ return 1; }'
        pair = unit(pair['rows'])
        rows = [ext, pair,
                make('primevul', 'conflict_commit', 'train', 'int other(){return 1;}', commit='ext'),
                make('primevul', 'v', 'train', 'int v(){return 1;}', commit='v'),
                make('primevul', 'b1', 'train', 'int b1(){return 0;}', label=0),
                make('primevul', 'b2', 'train', 'int b2(){return 0;}', label=0)]
        chosen, removed = select(rows)
        self.assertEqual({e['unit_id'] for e in removed if e['reason'] == 'duplicate_or_counterpart'}, {'pair'})
        self.assertEqual(len(chosen), 5)
        self.assertEqual(select(list(reversed(rows))), (chosen, removed))
        self.assertTrue(all(name.endswith(': repo_commit') for name in overlap_report(chosen)))
        validate(chosen)
        self.assertTrue(overlap_report(rows))
        strict, dropped = commit_disjoint(chosen, rows)
        self.assertNotIn('conflict_commit', {u['id'] for u in strict})
        self.assertTrue(dropped)
        validate(strict)

    def test_official_split_conflict_and_validator(self):
        rows = [make('primevul', 'vtrain', 'train', 'int b(){return 1;}', commit='same'),
                make('primevul', 'vtest', 'test', 'int b(){return 1;}', commit='same'),
                make('primevul', 'btest', 'test', 'int c(){return 0;}', label=0)]
        chosen, removed = select(rows)
        self.assertEqual({u['id'] for u in chosen}, {'vtest', 'btest'})
        self.assertEqual(removed[0]['unit_id'], 'vtrain')
        validate(chosen)
        chosen[0]['rows'][0]['official_split'] = 'valid'
        with self.assertRaises(ValueError):
            validate(chosen)

    def test_strict_commit_identity_across_repo_alias(self):
        sha = 'a' * 40
        ext = make('sven', 'ext', 'external_test', 'int ext(){return 1;}', commit=sha)
        v = make('primevul', 'v', 'train', 'int other(){return 1;}', commit=sha)
        v['rows'][0]['repo'] = 'github.com/old-owner/r'
        v = unit(v['rows'])
        benign = make('primevul', 'b', 'train', 'int benign(){return 0;}', label=0, commit='b' * 40)
        candidates = [ext, v, benign]
        main, _ = select(candidates)
        self.assertEqual(len(main), 3)
        strict, excluded = commit_disjoint(main, candidates)
        self.assertEqual([u['id'] for u in strict], ['ext'])
        self.assertTrue(any(e['unit_id'] == 'v' and e['reason'] == 'sven_commit_or_counterpart' for e in excluded))

    def test_counterpart_and_label_conflict_quarantine(self):
        a = 'int a(){return 1;}'
        b = 'int a(){return 0;}'
        edges = [(digest(normalized_source(a)), digest(normalized_source(b)))]
        rows = [make('primevul', 'v', 'train', a),
                make('primevul', 'b', 'train', b, label=0),
                make('primevul', 'contradiction', 'test', a, label=0)]
        linked = link_counterparts(rows, edges)
        remaining, removed = clean_label_conflicts(linked)
        self.assertEqual(remaining, [])
        self.assertEqual({r['reason'] for r in removed}, {'label_conflict', 'label_conflict_counterpart'})
        # Only the repaired counterpart is present in test; its vulnerable
        # train version must still be excluded despite different source text.
        linked = link_counterparts([rows[0], dict(rows[1], split='test',
                                  rows=[dict(rows[1]['rows'][0], split='test')])], edges)
        self.assertTrue(any(k.endswith(': pair_counterpart') for k in overlap_report(linked)))

    def test_near_prefix_join_against_bruteforce(self):
        base = 'int f(int x){' + ''.join(f'x += {i};' for i in range(80)) + 'return x;}'
        index = NearIndex()
        previous = []
        for i in range(20):
            code = base.replace('x += 40;', f'x += {100 + i};') if i else base
            item = make('primevul', str(i), 'train', code)
            signature = shingles(code)
            brute = any(len(signature & other) * 10 >= len(signature | other) * 9 for other in previous)
            self.assertEqual(bool(index.matches(item)), brute)
            index.add(item)
            previous.append(signature)
        before = make('sven', 'p', 'external_test', base.replace('return x;', 'return 1;'))
        # Atomic pair itself is never compared internally for near duplication.
        index = NearIndex()
        self.assertEqual(index.matches(before), [])
        index.add(make('primevul', 'base', 'train', base))
        base_signature = shingles(base)
        outcomes = set()
        for changes in (0, 1, 2, 5, 10, 30, 80):
            code = base
            for j in range(changes):
                code = code.replace(f'x += {j};', f'x -= {500 + j};')
            signature = shingles(code)
            expected = len(signature & base_signature) * 10 >= len(signature | base_signature) * 9
            outcomes.add(expected)
            self.assertEqual(bool(index.matches(make('primevul', 'query', 'test', code))), expected)
        self.assertEqual(outcomes, {False, True})

    def test_declared_near_counterparts_stay_together(self):
        before = 'int f(int x){' + ''.join(f'x += {i};' for i in range(80)) + 'return 1;}'
        after = before.replace('return 1;', 'return 0;')
        rows = [make('primevul', 'before', 'train', before),
                make('primevul', 'after', 'train', after, label=0)]
        edges = [(digest(normalized_source(before)), digest(normalized_source(after)))]
        linked = link_counterparts(rows, edges)
        selected, excluded = select(linked)
        self.assertEqual(len(selected), 2)
        self.assertEqual(excluded, [])
        validate(selected)

    def test_real_schema_fixture_repeatability(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'raw'
            pv = root / 'PrimeVul_v0.1'
            pv.mkdir(parents=True)
            (pv / 'file_info.json').write_text('{}')
            for si, split in enumerate(('train', 'valid', 'test')):
                rows = [dict(idx=si * 10 + i, func=f'int f{si}_{i}() {{return {i % 2};}}',
                             target=i % 2, file_name='x.c', func_hash=i,
                             project_url='https://github.com/o/prime', commit_id=str(si) * 40)
                        for i in range(4)]
                # A missing filename with an extracted, unparsable snippet is
                # still upstream C/C++; its vulnerable label must not vanish.
                if split == 'train':
                    rows[1]['file_name'] = 'None'
                    rows[1]['func'] = 'EXPORT missing_type(arg) { unmatched_macro(arg);'
                (pv / f'primevul_{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
                (pv / f'primevul_{split}_paired.jsonl').write_text('')
            clean = root / 'cleanvul'
            clean.mkdir()
            columns = ['func_before', 'func_after', 'vulnerability_score', 'extension', 'date',
                       'commit_url', 'file_name']
            for score in (4, 3):
                with (clean / f'vulnerability_score_{score}.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=columns)
                    writer.writeheader()
                    for i in range(10):
                        writer.writerow(dict(func_before=f'int c{score}_{i}() {{return 1;}}',
                                             func_after=f'int c{score}_{i}() {{return 0;}}',
                                             vulnerability_score=score, extension='c',
                                             date=f'200{i}-01-01T00:00:00Z',
                                             commit_url=f'https://github.com/o/clean/commit/{score}{i:039x}',
                                             file_name='x.c'))
            sven = root / 'sven/data_train_val/train'
            sven.mkdir(parents=True)
            row = dict(func_src_before='int ext(){return 1;}', func_src_after='int ext(){return 0;}',
                       file_name='x.cpp', func_name='ext', vul_type='cwe-125',
                       commit_link='github.com/o/sven/commit/' + 'a' * 40)
            (sven / 'cwe-125.jsonl').write_text(json.dumps(row) + '\n')
            output = Path(tmp) / 'output'
            with contextlib.redirect_stdout(io.StringIO()):
                first = prepare(root, output, 4)
                files = {p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()}
                second = prepare(root, output, 4)
            self.assertEqual(first, second)
            self.assertEqual(files, {p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()})
            self.assertEqual((output / 'manifest.jsonl').read_bytes(),
                             (output / 'score_4/manifest.jsonl').read_bytes())
            raw = first['raw']['primevul_train']
            self.assertEqual(raw['counts']['upstream_c_cpp_unspecified_1'], 1)


if __name__ == '__main__':
    unittest.main()

class DeletionStatisticsTests(unittest.TestCase):
    def test_counts_are_units_and_evidence_bins_do_not_double_count(self):
        from vulnmechanism.prepare_benchmark import deletion_statistics
        rows = [dict(dataset='cleanvul', split='test', pair_id='pair', sample_key=f'pair:{label}', label=label)
                for label in (1, 0)]
        excluded = [dict(dataset='primevul', split='train', unit_id='gone', labels=[1],
                         reason='duplicate_or_counterpart', matches=[
                             dict(kind='exact_source', retained_unit='pair'),
                             dict(kind='normalized_source', retained_unit='pair')])]
        stats = deletion_statistics(rows, excluded)['groups']
        self.assertEqual(stats['cleanvul/test/pairs']['retained_units'], 1)
        deleted = stats['primevul/train/label_1']
        self.assertEqual(deleted['input_units'], 1)
        self.assertEqual(deleted['duplicate_evidence_sets'], {'exact_source+normalized_source': 1})
        self.assertEqual(deleted['duplicate_retained_partitions'], {'cleanvul/test': 1})
