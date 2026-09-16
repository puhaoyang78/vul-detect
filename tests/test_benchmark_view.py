import unittest

from vulnmechanism.benchmark_view import record_dataset, record_pair_id, record_split, select_source_records


class BenchmarkViewTests(unittest.TestCase):
    def test_multiple_sources_require_explicit_selection(self):
        records = []
        for split in ('train', 'valid', 'test'):
            records.extend([
                dict(sample_key=f'primevul:{split}:v', split=split, label=1),
                dict(sample_key=f'primevul:{split}:b', split=split, label=0),
            ])
        records.extend([
            dict(sample_key='cleanvul:4:1:before', split='train', label=1),
            dict(sample_key='cleanvul:4:1:after', split='train', label=0),
        ])
        with self.assertRaisesRegex(ValueError, 'multiple formal benchmark sources'):
            select_source_records(records, None)
        selected, summary = select_source_records(records, 'primevul')
        self.assertEqual(len(selected), 6)
        self.assertTrue(all(record_dataset(r) == 'primevul' for r in selected))
        self.assertEqual(summary['source_dataset'], 'primevul')

    def test_primevul_build_success_is_balanced_per_split(self):
        records = []
        for split in ('train', 'valid', 'test'):
            records.extend([
                dict(sample_key=f'primevul:{split}:v1', split=split, label=1),
                dict(sample_key=f'primevul:{split}:v2', split=split, label=1),
                dict(sample_key=f'primevul:{split}:b1', split=split, label=0),
                dict(sample_key=f'primevul:{split}:b2', split=split, label=0),
                dict(sample_key=f'primevul:{split}:b3', split=split, label=0),
            ])
        selected, summary = select_source_records(records, 'primevul')
        for split in ('train', 'valid', 'test'):
            labels = [r['label'] for r in selected if record_split(r) == split]
            self.assertEqual(labels.count(1), 2)
            self.assertEqual(labels.count(0), 2)
        self.assertEqual(summary['dropped_primevul_benign_records'], 3)
        again, _ = select_source_records(list(reversed(records)), 'primevul')
        self.assertEqual({r['sample_key'] for r in selected}, {r['sample_key'] for r in again})

    def test_incomplete_pairs_are_removed_atomically(self):
        records = [
            dict(sample_key='cleanvul:4:1:before', split='train', label=1),
            dict(sample_key='cleanvul:4:1:after', split='train', label=0),
            dict(sample_key='cleanvul:4:2:before', split='valid', label=1),
        ]
        selected, summary = select_source_records(records, 'cleanvul')
        self.assertEqual({r['sample_key'] for r in selected}, {
            'cleanvul:4:1:before', 'cleanvul:4:1:after'
        })
        self.assertEqual(summary['dropped_incomplete_pair_records'], 1)
        self.assertEqual(record_pair_id(selected[0]), 'cleanvul:4:1')

    def test_dataset_and_split_inference(self):
        self.assertEqual(record_dataset({'sample_key': 'sven:train:cwe-125:1:before'}), 'sven')
        self.assertEqual(record_split({'sample_key': 'x', 'split': 'validation'}), 'valid')
        with self.assertRaisesRegex(ValueError, 'unknown explicit dataset split'):
            record_split({'sample_key': 'x', 'split': 'typo'})


if __name__ == '__main__':
    unittest.main()
