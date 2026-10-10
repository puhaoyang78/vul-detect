import unittest
import numpy as np
from vulnmechanism.audit_statistics import graph_counts, syntax_counts, summary, feature_blocks, fit_predict, conditional_auc, SIZE, SYNTAX, GRAPH, SPECS


class StatisticsAuditTests(unittest.TestCase):
    def test_conditional_auc_does_not_credit_between_project_priors(self):
        y=[0,0,0,1,0,1,1,1,1]
        p=[.1]*4+[.9]*4+[1.]
        groups=['a']*4+['b']*4+['single_class']
        result=conditional_auc(y,p,groups)
        self.assertEqual(result['auc'],.5)
        self.assertEqual(result['eligible_functions'],8)
        self.assertEqual(result['comparable_pairs'],6)
        self.assertGreater(summary(y,p)['auc'],.5)

    def test_all_probe_paths_fit_without_validation_labels(self):
        rng=np.random.default_rng(42)
        blocks={k:rng.normal(size=(14,4)) for k in ('size','syntax','structure')}
        blocks['labels']=np.array([0,1]*7)
        texts=np.array(['int f() { return '+str(i%2)+'; }' for i in range(14)])
        projects=np.array(['p'+str(i%3) for i in range(14)])
        frozen=rng.normal(size=(14,6));tr=np.arange(10);va=np.arange(10,14)
        for spec in SPECS:
            a,_,_=fit_predict(spec,tr,va,blocks,texts,projects,frozen)
            blocks['labels'][va]=1-blocks['labels'][va]
            b,_,_=fit_predict(spec,tr,va,blocks,texts,projects,frozen)
            np.testing.assert_array_equal(a,b)

    def test_visible_graph_is_induced_and_cycle_rank_not_path_count(self):
        graph={'nodes':[{'id':str(i)} for i in range(4)],'edges':[
            {'kind':'CFG','source':'0','target':'1'},
            {'kind':'CFG','source':'1','target':'2'},
            {'kind':'CFG','source':'2','target':'0'},
            {'kind':'CFG','source':'2','target':'3'},
            {'kind':'DDG','source':'0','target':'3'}]}
        full=graph_counts(graph);visible=graph_counts(graph,{'0','1','2'})
        self.assertEqual(full['CFG_cycle_rank'],1)
        self.assertEqual(full['CFG_components'],1)
        self.assertEqual(visible['CFG_edges'],3)
        self.assertEqual(visible['DDG_edges'],0)
        self.assertEqual(graph_counts(graph,set())['CFG_components'],0)

    def test_utf8_cutoff_does_not_include_invisible_calls(self):
        src='int f(int x) { /* 中文 */ x = x + 1; hidden(); return x; }'
        full,error=syntax_counts(src,'c',len(src))
        visible,_=syntax_counts(src,'c',src.index('hidden'))
        self.assertFalse(error)
        self.assertEqual(full['calls'],1)
        self.assertEqual(visible['calls'],0)
        self.assertEqual(visible['assignments'],1)
        self.assertEqual(full['parameters'],1)

    def test_summary_and_label_not_part_of_features(self):
        m=summary([0,1],[.25,.75]);self.assertEqual(m['auc'],1.)
        self.assertAlmostEqual(m['bce'],-np.log(.75))
        self.assertEqual(summary([0],[.2])['auc'],None)
        f={k:1 for k in SIZE+SYNTAX+['g_'+k for k in GRAPH]}
        a=feature_blocks([{'label':0,'features':f}]);b=feature_blocks([{'label':1,'features':f}])
        for k in a:np.testing.assert_array_equal(a[k],b[k])

if __name__=='__main__':unittest.main()
