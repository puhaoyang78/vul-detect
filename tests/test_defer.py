import unittest
import numpy as np
from vulnmechanism.defer import threshold_rows, matched_points, partition, router_inputs, joint_outcomes, gain_from_probabilities, allocate

class DeferTests(unittest.TestCase):
    def test_curve_preserves_ties(self):
        rows=[dict(label=y, score=p, prediction=0) for y,p in [(1,.8),(0,.8),(1,.2),(0,.1)]]
        curve=threshold_rows(rows)
        self.assertEqual([(r['tp'],r['fp']) for r in curve],[(0,0),(1,1),(2,1),(2,2)])
        match=matched_points(curve,dict(fp=1,tp=2))
        self.assertEqual(match['at_most_same_fp']['tp'],2)
        self.assertEqual(match['at_least_same_recall']['fp'],1)

    def test_supervision_and_cost(self):
        z=joint_outcomes([1,0,1,0,1],[0,1,1,0,1],[1,0,0,1,1])
        np.testing.assert_array_equal(z,[1,2,3,4,0])
        np.testing.assert_array_equal(gain_from_probabilities(np.eye(5),[0,1,2,3,4],3),[0,3,1,-3,-1])
        self.assertEqual(partition('owner/repo'),partition('owner/repo'))
        with self.assertRaises(ValueError):joint_outcomes([2],[0],[1])

    def test_only_upstream_features(self):
        self.assertEqual(router_inputs([.1,.8],[20,200]).shape,(2,4))
        with self.assertRaises(TypeError):router_inputs([.1],[20],qwen_scores=[.2])
        with self.assertRaises(ValueError):router_inputs([np.nan],[20])
        with self.assertRaises(ValueError):router_inputs([.1],[-1])

    def test_budget_and_ties(self):
        keys=['b','a','c','d'];scores=[1,1,0,-1]
        self.assertEqual(allocate(keys,scores,25),{1})
        self.assertEqual(allocate(keys,scores,50),{0,1})
        self.assertEqual(allocate(keys,scores,50,costs=[3,2,1,1],cap=4),{1})

if __name__=='__main__':unittest.main()
