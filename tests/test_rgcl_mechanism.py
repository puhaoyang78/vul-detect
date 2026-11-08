"""Coverage background and exact RQ3 classification/error accounting."""

import copy
import unittest

from tests.test_losver import Tokens
from vulnmechanism.data import paired_features
from vulnmechanism.rgcl_mechanism import (align_predictions, coverage_record, line_counts,
    classification_records, summarize_coverage, summarize_classification, aggregate_classification,
    pair_state, STATES)


class RGCLMechanismTests(unittest.TestCase):
    def record(self, code, gt, i=0):
        row = dict(func=code, lines_ground_truth=gt, target=1-i % 2,
                   sample_key=f"s:{i}", pair_key=f"full:split:{i // 2}")
        feature = paired_features(row, Tokens(), "baseline")
        return row, feature, coverage_record(row, feature)

    def test_empty_diff_is_separate_from_invisible_and_all_region(self):
        _, _, empty = self.record("a\nb\n", [])
        _, _, full = self.record("a\nb\n", [0])
        _, _, all_region = self.record("a\nb\n", [0, 1])
        _, _, invisible = self.record("x" * 511 + "\nb\n", [1])
        result = summarize_coverage([empty, full, all_region, invisible])["all"]
        self.assertEqual(result["category_counts"], dict(full=2, partial=0, invisible=1, no_direct_diff=1))
        self.assertEqual(result["eligible_functions"], 1)
        self.assertEqual(result["all_visible_code_is_region"], 1)
        self.assertEqual(empty["effective_region_token_mass"], 0)
        self.assertEqual(all_region["effective_region_token_mass"], 0)
        self.assertEqual(full["effective_region_token_mass"], 1)
        self.assertEqual(sum(result["category_counts"].values()), result["functions"])

    def test_partial_final_line_keeps_historical_primevul_supervision(self):
        _, _, record = self.record("a\n" + "x" * 520 + "\nz\n", [1, 2])
        self.assertEqual(record["visibility"], "partial")
        self.assertEqual(record["visible_diff_lines"], [1])
        self.assertEqual(record["partial_diff_lines"], [1])
        self.assertTrue(record["eligible"])
        self.assertEqual(record["visible_region_token_mass"], 508)
        self.assertEqual(record["code_tokens"], 509)

    def test_cross_line_mass_matches_bounded_pool(self):
        feature = dict(code_mask=[False, True, True, True, False],
                       line_token_indices=[[0, 1, 2], [2, 3, 4]])
        self.assertEqual(line_counts(feature), [1.5, 1.5])
        feature["code_mask"][4] = True
        feature["line_token_indices"][1].remove(4)
        with self.assertRaisesRegex(ValueError, "cover"):
            line_counts(feature)

    def fixture(self):
        rows, coverage, pred = [], [], []
        for i in range(4):
            r, _, c = self.record("a\nb\n", [0], i)
            rows.append(r)
            coverage.append(c)
            p = [.8, .6, .5, .4][i]
            pred.append(dict(sample_key=r["sample_key"], pair_key=r["pair_key"], target=r["target"],
                probability=p, prediction=int(p > .5), line_numbers=c["line_numbers"],
                line_token_counts=c["line_token_counts"]))
        return rows, coverage, pred

    def test_alignment_checks_full_key_and_never_truncates_missing_records(self):
        rows, cov, pred = self.fixture()
        self.assertEqual(align_predictions(rows, pred[::-1], cov), pred)
        for changed in (pred[:-1], pred+[pred[0]]):
            with self.assertRaises(ValueError):
                align_predictions(rows, changed, cov)
        changed = copy.deepcopy(pred)
        changed[0]["pair_key"] = "another:split:0"
        with self.assertRaisesRegex(ValueError, "pair_key"):
            align_predictions(rows, changed, cov)
        changed = copy.deepcopy(pred)
        changed[0]["line_token_counts"][0] += 1
        with self.assertRaisesRegex(ValueError, "Token counts"):
            align_predictions(rows, changed, cov)

    def test_strict_threshold_and_pair_classification_changes(self):
        rows, cov, context = self.fixture()
        rgcl = copy.deepcopy(context)
        for p, value in zip(rgcl, [.8, .4, .4, .6]):
            p.update(probability=value, prediction=int(value > .5))
        records = classification_records(rows, dict(context_only=context, rgcl=rgcl), 42)
        summary = summarize_classification(records)
        self.assertEqual(summary["methods"]["context_only"]["pair_counts"]["p_c"], 0)
        self.assertEqual(summary["methods"]["context_only"]["pair_counts"]["p_b"], 1)
        self.assertEqual(summary["methods"]["rgcl"]["pair_counts"]["p_c"], 1)
        self.assertEqual(summary["methods"]["rgcl"]["pair_counts"]["p_r"], 1)
        self.assertEqual(summary["pc"]["gained"], 1)
        self.assertEqual(summary["pc"]["lost"], 0)
        self.assertEqual(summary["errors"]["corrected_fp"]["count"], 1)
        self.assertEqual(summary["errors"]["new_fp"]["count"], 1)
        for probabilities, state in (((.5,.5),'p_b'), ((.500001,.5),'p_c'),
                                     ((.5,.500001),'p_r'), ((.500001,.500001),'p_v')):
            self.assertEqual(pair_state(*probabilities), state)

    def all_transitions(self, seed):
        probabilities = dict(p_c=(.8,.2), p_v=(.8,.8), p_b=(.2,.2), p_r=(.2,.8))
        rows, predictions = [], dict(context_only=[], rgcl=[])
        for before in STATES:
            for after in STATES:
                key = f'full:test:{len(rows)//2}'
                pair = [dict(sample_key=f'{key}:{i}', pair_key=key, target=1-i) for i in (0,1)]
                rows.extend(pair)
                for method,state in [('context_only',before),('rgcl',after)]:
                    predictions[method].extend(dict(r, probability=p, prediction=int(p>.5))
                                               for r,p in zip(pair,probabilities[state]))
        return classification_records(rows,predictions,seed)

    def test_all_16_transitions_reconcile_pair_and_function_counts(self):
        result = summarize_classification(self.all_transitions(42))
        self.assertEqual(result['matrix'], [[1]*4 for _ in range(4)])
        self.assertEqual(result['row_totals'], [4]*4)
        self.assertEqual(result['column_totals'], [4]*4)
        self.assertEqual(result['pc']['gained'], 3)
        self.assertEqual(result['pc']['lost'], 3)
        self.assertEqual(result['pc']['net'], 0)
        self.assertEqual(result['pc']['gain_source_shares'], dict(p_v=1/3,p_b=1/3,p_r=1/3))
        self.assertTrue(all(v['count']==4 for v in result['errors'].values()))
        self.assertEqual(result['net_fp'], 0)
        self.assertEqual(result['net_fn'], 0)
        self.assertEqual(result['fraction_all'][0][0], 1/16)
        self.assertEqual(result['fraction_source'][0][0], 1/4)
        for method in ['context_only','rgcl']:
            self.assertEqual(result['methods'][method]['recall'], .5)
            self.assertEqual(result['methods'][method]['patched_fpr'], .5)

    def test_seed_repetitions_are_never_pooled_as_independent_pairs(self):
        records = [r for seed in [42,1234,123456] for r in self.all_transitions(seed)]
        summaries = [summarize_classification([r for r in records if r['seed']==seed]) for seed in [42,1234,123456]]
        aggregate = aggregate_classification(summaries,records)
        self.assertEqual(aggregate['pairs_per_seed'],16)
        self.assertEqual(aggregate['matrix'][0][0],dict(mean=1,std=0))
        with self.assertRaisesRegex(ValueError,'one nonempty seed'):
            summarize_classification(records)
        changed = copy.deepcopy(records)
        changed[-1]['pair_key'] = 'another:full:pair'
        with self.assertRaisesRegex(ValueError,'same cohort'):
            aggregate_classification(summaries,changed)

    def test_zero_opportunity_rates_are_undefined_and_duplicate_pairs_rejected(self):
        records = self.all_transitions(42)[:1]
        result = summarize_classification(records)
        self.assertEqual(result['fraction_source'][1],[None]*4)
        self.assertIsNone(result['errors']['corrected_fp']['fraction_opportunity'])
        self.assertIsNone(result['pc']['gain_source_shares']['p_v'])
        with self.assertRaisesRegex(ValueError,'duplicate repair pairs'):
            summarize_classification(records*2)


if __name__ == "__main__":
    unittest.main()
