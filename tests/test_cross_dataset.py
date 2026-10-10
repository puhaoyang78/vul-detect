import unittest
import numpy as np
from vulnmechanism.cross_dataset import mega_source, project_split, summarize_predictions, balanced_keys


class CrossDatasetTests(unittest.TestCase):
    def test_native_labels_and_existing_splits(self):
        self.assertEqual(mega_source(dict(is_vul=True, func_before='bad', func='fix')), ('bad', 1))
        self.assertEqual(mega_source(dict(is_vul=False, func_before='unused', func='normal')), ('normal', 0))
        with self.assertRaises(ValueError): mega_source(dict(is_vul=True, func='fix'))
        self.assertEqual(project_split('p', {'p': 'valid'}), 'valid')
        self.assertEqual(project_split('new', {}), project_split('new', {}))

    def test_metrics_and_fixed_source_threshold(self):
        rows = [dict(label=0, score=.2), dict(label=1, score=.4)]
        r = summarize_predictions(rows, .3)
        self.assertEqual((r['fp'], r['fn'], r['auc']), (0, 0, 1))
        self.assertEqual(summarize_predictions(rows, .5)['fn'], 1)
        self.assertAlmostEqual(r['bce'], -np.log(.8 * .4) / 2)

    def test_balanced_members_are_fixed_without_predictions(self):
        rows = [dict(sample_key=str(i), label=int(i < 3)) for i in range(20)]
        keys = balanced_keys(rows)
        self.assertEqual(len(keys), 6)
        self.assertEqual(keys, balanced_keys(list(reversed(rows))))
        self.assertEqual(sum(r['label'] for r in rows if r['sample_key'] in keys), 3)
        self.assertTrue({'0', '1', '2'}.issubset(keys))


if __name__ == '__main__': unittest.main()
