import json
from pathlib import Path
import tempfile
import unittest

from vulnmechanism.prepare_primevul import prepare, single_function_language


class PreparePrimeVulTests(unittest.TestCase):
    def test_strict_single_function(self):
        self.assertEqual(single_function_language('int f(void) { return 0; }'), 'c')
        self.assertEqual(single_function_language('Foo::Foo() : x(0) {}'), 'cpp')
        self.assertIsNone(single_function_language('missing_type(int x) { return x; }'))
        self.assertIsNone(single_function_language('Foo::bar() {}'))
        self.assertIsNone(single_function_language('Foo::Foo() {}', 'source.c'))
        self.assertEqual(single_function_language('int f() {}', 'source.cpp'), 'cpp')
        self.assertEqual(single_function_language('template<class T> T f(T x) { return x; }'), 'cpp')
        for source in ('', 'int f();', 'int f() {',
                       'int f() {} int g() {}', 'int x; int f() {}'):
            with self.subTest(source=source):
                self.assertIsNone(single_function_language(source))

    def test_original_labels_splits_duplicates_and_shortfall(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = '  int f(void) { return 0; }\n'
            rows = {
                'train': [{'idx': 1, 'func': original, 'target': 1},
                          {'idx': 2, 'func': '', 'target': 0},
                          {'idx': 3, 'func': 'int f();', 'target': 0}],
                'valid': [{'idx': 4, 'func': original, 'target': 0},
                          {'idx': 5, 'func': 'int g() {}', 'target': 0}],
                'test': [{'idx': 6, 'func': 'int h() {}', 'target': 1}],
            }
            for split, records in rows.items():
                (root / f'primevul_{split}.jsonl').write_text(
                    ''.join(json.dumps(r) + '\n' for r in records))
            output = root / 'functions.jsonl'
            stats = prepare(root, output)
            first = output.read_bytes()
            self.assertEqual(stats['total'], 3)
            self.assertEqual(stats['splits']['train']['empty'], 1)
            self.assertEqual(stats['splits']['train']['invalid_single_function'], 1)
            self.assertEqual(stats['splits']['valid']['cross_split_duplicates'], 1)
            self.assertEqual(stats['splits']['valid']['conflicting_label_duplicates'], 1)
            self.assertEqual(stats['splits']['test']['shortfall_0'], 200)
            result = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(result[0]['function'], original)
            self.assertEqual([(r['idx'], r['label'], r['split']) for r in result],
                             [(1, 1, 'train'), (5, 0, 'valid'), (6, 1, 'test')])
            self.assertEqual(prepare(root, output), stats)
            self.assertEqual(output.read_bytes(), first)


if __name__ == '__main__':
    unittest.main()
