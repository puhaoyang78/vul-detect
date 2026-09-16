import json
from pathlib import Path
import tempfile
import unittest
import subprocess
from unittest.mock import patch

from vulnmechanism.cpg import CPGError, FunctionGraph, GraphNode, GraphEdge
from vulnmechanism.dataset import build_function_dataset


class BuildResumeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.samples = Path(self.directory.name) / 'samples.jsonl'
        self.output = Path(self.directory.name) / 'graphs.jsonl'
        self.rows = [{'sample_key': str(i), 'function': f'void f{i}(void) {{ free(0); }}',
                      'label': i % 2, 'language': 'c', 'split': 'train'} for i in range(3)]
        self.samples.write_text(''.join(json.dumps(r) + '\n' for r in self.rows))
        self.graph = FunctionGraph('f', {'1': GraphNode('1', 'METHOD', 'f'),
                                       '2': GraphNode('2', 'free', 'free(0)')},
                                   (GraphEdge('AST', '1', '2'),))

    def test_failure_preserves_successes_and_resume_skips_them(self):
        with patch('vulnmechanism.dataset.extract_function_cpg',
                   side_effect=[self.graph, CPGError('failure'), self.graph]):
            result = build_function_dataset(self.samples, self.output)
        self.assertEqual([r['sample_key'] for r in result], ['0', '2'])
        errors = self.output.with_suffix('.errors.jsonl')
        self.assertEqual(json.loads(errors.read_text())['sample_key'], '1')
        self.assertEqual(len(self.output.read_text().splitlines()), 2)
        with patch('vulnmechanism.dataset.extract_function_cpg', return_value=self.graph) as extract:
            records = build_function_dataset(self.samples, self.output)
            self.assertEqual([call.args[1] for call in extract.call_args_list], ['f1'])
        self.assertEqual({r['sample_key'] for r in records}, {'0', '1', '2'})
        self.assertEqual(errors.read_text(), '')
        before = self.output.read_bytes()
        with patch('vulnmechanism.dataset.extract_function_cpg') as extract:
            self.assertEqual(build_function_dataset(self.samples, self.output), records)
            extract.assert_not_called()
        self.assertEqual(self.output.read_bytes(), before)

    def test_interruption_and_partial_last_line_resume(self):
        with patch('vulnmechanism.dataset.extract_function_cpg',
                   side_effect=[self.graph, KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                build_function_dataset(self.samples, self.output)
        self.assertEqual(len(self.output.read_text().splitlines()), 1)
        with self.output.open('ab') as handle:
            handle.write(b'{"sample_key": "1", "raw_source": "\xe4')
        with patch('vulnmechanism.dataset.extract_function_cpg', return_value=self.graph) as extract:
            records = build_function_dataset(self.samples, self.output)
            self.assertEqual([call.args[1] for call in extract.call_args_list], ['f1', 'f2'])
        self.assertEqual([r['sample_key'] for r in records], ['0', '1', '2'])
        before = self.output.read_bytes()
        with patch('vulnmechanism.dataset.extract_function_cpg') as extract:
            self.assertEqual(build_function_dataset(self.samples, self.output), records)
            extract.assert_not_called()
        self.assertEqual(self.output.read_bytes(), before)

    def test_changed_input_and_corrupt_complete_line_are_rejected(self):
        with patch('vulnmechanism.dataset.extract_function_cpg', return_value=self.graph):
            build_function_dataset(self.samples, self.output)
        before = self.output.read_bytes()
        self.rows[0]['label'] = 1
        self.samples.write_text(''.join(json.dumps(r) + '\n' for r in self.rows))
        with self.assertRaisesRegex(ValueError, 'does not match input'):
            build_function_dataset(self.samples, self.output)
        self.assertEqual(self.output.read_bytes(), before)
        self.output.write_bytes(b'broken\n')
        with self.assertRaisesRegex(ValueError, 'invalid saved JSON'):
            build_function_dataset(self.samples, self.output)
        self.assertEqual(self.output.read_bytes(), b'broken\n')

    def test_complete_record_without_newline_is_preserved(self):
        with patch('vulnmechanism.dataset.extract_function_cpg',
                   side_effect=[self.graph, KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                build_function_dataset(self.samples, self.output)
        self.output.write_bytes(self.output.read_bytes().rstrip(b'\n'))
        with patch('vulnmechanism.dataset.extract_function_cpg', return_value=self.graph):
            build_function_dataset(self.samples, self.output)
        self.assertEqual(len([json.loads(line) for line in self.output.read_bytes().splitlines()]), 3)

    def test_syntax_failure_and_timeout_are_logged_and_do_not_stop_run(self):
        self.rows[0]['function'] = 'int f(void);'
        self.samples.write_text(''.join(json.dumps(r) + '\n' for r in self.rows))
        with patch('vulnmechanism.dataset.extract_function_cpg',
                   side_effect=[subprocess.TimeoutExpired('joern', 1), self.graph]):
            records = build_function_dataset(self.samples, self.output)
        self.assertEqual([r['sample_key'] for r in records], ['2'])
        failures = [json.loads(line) for line in self.output.with_suffix('.errors.jsonl').read_text().splitlines()]
        self.assertEqual([r['stage'] for r in failures], ['syntax', 'joern'])
        self.assertEqual(failures[1]['error_type'], 'TimeoutExpired')

    def test_programming_error_is_not_hidden(self):
        with patch('vulnmechanism.dataset.extract_function_cpg', side_effect=TypeError('bug')):
            with self.assertRaisesRegex(TypeError, 'bug'):
                build_function_dataset(self.samples, self.output)

    def test_c_family_external_roundtrip_and_failure(self):
        from vulnmechanism.dataset import _sample_fields, _resolve_sample
        from vulnmechanism.model import _record_split
        from vulnmechanism.cli import build_parser
        for name, source, expected in [
            ('x.c', 'void f(void) {}', 'c'),
            ('x.cpp', 'void f(void) {}', 'cpp'),
            ('x.C', 'void f(void) {}', 'cpp'),
            ('', 'void f(void) {}', 'c'),
            ('', 'void f() { auto x = []() { return 1; }; }', 'cpp'),
        ]:
            row = dict(sample_key='x', function=source, label=1,
                       language='c_cpp', file_name=name, split='external_test')
            self.assertEqual(_resolve_sample(_sample_fields(row, 1))[0], expected)
        row['function'] = 'not a function'
        with self.assertRaisesRegex(ValueError, r'C/C\+\+ resolution failed'):
            _resolve_sample(_sample_fields(row, 1))
        self.rows[0].update(language='c_cpp', file_name='x.cpp', split='external_test')
        self.rows[1].update(language='c_cpp', function='not a function', split='external_test')
        self.samples.write_text(''.join(json.dumps(r) + '\n' for r in self.rows))
        with patch('vulnmechanism.dataset.extract_function_cpg', return_value=self.graph) as extract:
            records = build_function_dataset(self.samples, self.output)
            self.assertEqual(extract.call_args_list[0].kwargs['language'], 'cpp')
        self.assertEqual(records[0]['resolved_language'], 'cpp')
        self.assertEqual(records[0]['language'], 'c_cpp')
        with patch('vulnmechanism.model._split_for_key', side_effect=AssertionError('hash called')):
            self.assertEqual(_record_split(records[0]), 'external_test')
            self.assertFalse(any(_record_split(r) == 'valid' for r in records))
            self.assertEqual([r['sample_key'] for r in records if _record_split(r) == 'train'], ['2'])
        failure = json.loads(self.output.with_suffix('.errors.jsonl').read_text())
        self.assertEqual(failure['split'], 'external_test')
        self.assertIsNone(failure['resolved_language'])
        with patch('vulnmechanism.dataset.extract_function_cpg') as extract:
            self.assertEqual(build_function_dataset(self.samples, self.output), records)
            extract.assert_not_called()
        args = build_parser().parse_args(['eval', '--checkpoint', 'unused', '--split', 'external_test'])
        self.assertEqual(args.split, 'external_test')
