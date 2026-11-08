"""Post-hoc denominators, truncation, cross-line mass and pair-ID alignment."""

import copy
import unittest

from tests.test_losver import Tokens
from vulnmechanism.data import paired_features
from vulnmechanism.rgcl_mechanism import (align_predictions, coverage_record, line_counts,
                                         paired_records, summarize_coverage, summarize_gaps)


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

    def test_ordering_is_not_both_correct_and_threshold_remains_strict(self):
        rows, cov, context = self.fixture()
        rgcl = copy.deepcopy(context)
        for p, value in zip(rgcl, [.8, .4, .4, .6]):
            p.update(probability=value, prediction=int(value > .5))
        records = paired_records(rows, dict(context_only=context, rgcl=rgcl), cov, [False, False], 42, "test")
        summary = summarize_gaps(records)
        self.assertEqual(summary["methods"]["context_only"]["ordered"], 2)
        self.assertEqual(summary["methods"]["context_only"]["pair_counts"]["p_c"], 0)
        self.assertEqual(summary["methods"]["context_only"]["pair_counts"]["p_b"], 1)
        self.assertEqual(summary["methods"]["rgcl"]["pair_counts"]["p_c"], 1)
        self.assertEqual(summary["methods"]["rgcl"]["pair_counts"]["p_r"], 1)
        self.assertEqual(summary["gained_p_c"], 1)
        self.assertEqual(summary["lost_p_c"], 0)
        self.assertEqual(summary["lost_ordering"], 1)
        self.assertAlmostEqual(summary["gap_change"]["mean"], -.05)


if __name__ == "__main__":
    unittest.main()
