import unittest
from vulnmechanism.cascade import route, BUDGETS, STRATEGIES, decision_metrics, changes


class CascadeTests(unittest.TestCase):
    def test_actual_boundary_not_half(self):
        keys=['near_threshold','near_half','negative','positive']
        scores=[.151,.499,.01,.9]
        self.assertEqual(route(keys,scores,.15,'margin',25),{0})
        self.assertEqual(route(keys,scores,.15,'probability_uncertainty',25),{1})

    def test_budgets_nested_and_no_labels_or_downstream_inputs(self):
        keys=[f'k{i:04}' for i in range(741)];scores=[(i%101)/100 for i in range(741)]
        for strategy in STRATEGIES:
            previous=set()
            for budget in BUDGETS:
                selected=route(keys,scores,.15,strategy,budget)
                self.assertEqual(len(selected),741*budget//100)
                self.assertTrue(previous<=selected);previous=selected
        # The routing API accepts only keys, upstream probabilities, threshold and fixed policy.
        with self.assertRaises(TypeError):route(keys,scores,.15,'margin',25,labels=[0]*741)
        with self.assertRaises(TypeError):route(keys,scores,.15,'margin',25,qwen_scores=scores)

    def test_direction_and_tie_order(self):
        keys=['b','a','c','d'];scores=[.2,.2,.149,.01]
        self.assertEqual(route(keys,scores,.15,'positive_first',25),{1})
        self.assertEqual(route(keys,scores,.15,'negative_first',25),{2})
        self.assertEqual(route(keys,scores,.15,'positive_first',75),{0,1,2})
        perm=[3,1,0,2]
        selected=route([keys[i] for i in perm],[scores[i] for i in perm],.15,'positive_first',25)
        self.assertEqual({keys[perm[i]] for i in selected},{'a'})

    def test_invalid(self):
        for keys,scores in [(['x','x'],[.1,.2]),(['a'],[float('nan')]),(['a'],[1.1]),(['a'],[])]:
            with self.assertRaises(ValueError):route(keys,scores,.15,'margin',25)

    def test_decisions_and_paired_counts(self):
        rows=[dict(sample_key=str(i),dataset='primevul',split='valid',source_sha256=str(i),label=y,prediction=p) for i,(y,p) in enumerate([(0,1),(1,0),(0,0),(1,1)])]
        other=[dict(r,prediction=1-r['prediction']) for r in rows]
        d=changes(rows,other)
        self.assertEqual((d['corrected'],d['damaged'],d['net_corrected']),(2,2,0))
        m=decision_metrics(rows)
        self.assertEqual((m['tp'],m['tn'],m['fp'],m['fn']),(1,1,1,1))
        self.assertNotIn('auc',m);self.assertNotIn('bce',m)
        self.assertEqual(decision_metrics([]),{'samples':0})


if __name__=='__main__':unittest.main()
