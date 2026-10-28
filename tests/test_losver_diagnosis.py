import copy
import math
import unittest

import torch

from tests.test_losver import Tokens, model
from vulnmechanism.data import paired_encoding, paired_features
from vulnmechanism.losver_diagnosis import (
    POLICIES, binary_summary, intervened_forward, pair_states, pair_summary,
    policy_weights, selected_mass, transition,
)
from vulnmechanism.training import paired_collate


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_softmax_amplifies_by_exp_four_not_five(self):
        value = selected_mass(1, 100)
        self.assertAlmostEqual(value/(1-value)*99, math.exp(4))
        self.assertGreater(value, .35)
        self.assertEqual(selected_mass(0, 100), 0)
        self.assertEqual(selected_mass(100, 100), 1)
        self.assertAlmostEqual(selected_mass(10, 100, 1), .1)

    def test_strict_boundary_and_pair_denominators(self):
        rows = [dict(target=y) for y in [1, 0]*4]
        p = [.8, .2, .9, .7, .4, .1, .5, .8]
        summary = pair_summary(rows, p)
        self.assertEqual(pair_states(p), ["C", "V", "B", "R"])
        self.assertEqual(summary["pair_counts"], dict(p_c=1, p_v=1, p_b=1, p_r=1))
        self.assertEqual(summary["ranking_accuracy"], .75)
        self.assertEqual(summary["recall"], summary["p_c"]+summary["p_v"])
        self.assertEqual(summary["accuracy"], summary["p_c"]+(summary["p_v"]+summary["p_b"])/2)
        self.assertEqual(binary_summary([1, 0], [.5, .5])["tp"], 0)
        self.assertEqual(sum(map(sum, transition(p, p)["counts"])), 4)

    def test_interventions_reproduce_all_canonical_pools_and_isolate_diff(self):
        rows = [dict(func="int f(){\n"+" x += 1;\n"*75+"return x;\n}", target=1,
                     lines_predicted=[1, 2], lines_ground_truth=[2, 3], source_row=1),
                dict(func="int g(){\n return 0;\n}", target=0,
                     lines_predicted=[], lines_ground_truth=[1], source_row=2)]
        for mode in ("512", "l1"):
            with self.subTest(mode=mode):
                t = Tokens()
                fs = []
                for row in rows:
                    encoding = paired_encoding(row["func"], t, mode)
                    fs.append(paired_features(dict(row, line_numbers=encoding["line_numbers"]),
                                              t, "predicted", encoding=encoding, input_mode=mode))
                batch = paired_collate(fs)
                weights = policy_weights(fs, batch, rows, "train")
                other = copy.deepcopy(rows)
                other[0]["lines_ground_truth"] = [0]
                other[1]["lines_ground_truth"] = []
                changed = policy_weights(fs, batch, other, "train")
                for policy in ("equal_1", "predicted_2", "predicted_3", "predicted_5",
                               "predicted_10", "random_42", "random_43", "random_44"):
                    self.assertTrue(torch.equal(weights[POLICIES.index(policy)],
                                                changed[POLICIES.index(policy)]))
                detector = model().eval()
                detector.input_mode = mode
                detector.window_micro_batch = 1
                with torch.no_grad():
                    result = intervened_forward(detector, batch, weights, torch.device("cpu")).sigmoid()
                    for i in range(len(POLICIES)):
                        normal = detector(input_ids=batch["input_ids"],
                                          attention_mask=batch["attention_mask"],
                                          code_mask=batch["code_mask"],
                                          window_owners=batch["window_owners"],
                                          token_weights=weights[i])[:, 0]
                        torch.testing.assert_close(result[:, i], normal, atol=1e-6, rtol=1e-6)
                    subset = intervened_forward(detector, batch, weights[:3], torch.device("cpu")).sigmoid()
                    torch.testing.assert_close(subset, result[:, :3], atol=1e-6, rtol=1e-6)
                self.assertTrue(torch.equal(weights, policy_weights(fs, batch, rows, "train")))


if __name__ == "__main__":
    unittest.main()
