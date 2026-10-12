import unittest
from unittest.mock import patch
import numpy as np
from vulnmechanism import audit_statistics as audit
from vulnmechanism.audit_statistics import graph_counts, syntax_counts, summary, feature_blocks, fit_predict, conditional_auc, SIZE, SYNTAX, GRAPH, SPECS


class StatisticsAuditTests(unittest.TestCase):
    def test_visible_statistics_use_character_offsets_and_actual_budget(self):
        class Tokenizer:
            is_fast=True
            def __call__(self, source, **kwargs):
                n=min(len(source),kwargs['max_length'])
                return dict(input_ids=[1]*n,offset_mapping=[(i,i+1) for i in range(n)])
        visible,features=audit.visible_source_statistics('a\n中文 hidden',Tokenizer(),4)
        self.assertEqual(visible,'a\n中文')
        self.assertEqual(features,dict(visible_tokens=4,visible_chars=4,visible_lines=2))

    def test_dependency_groups_link_transitive_repairs_and_visible_duplicates(self):
        rows=[dict(sample_key=str(i),raw_source=f'int f{i}(){{return {i};}}') for i in range(6)]
        rows[0]['counterpart_groups']=['a'];rows[1]['counterpart_groups']=['a','b']
        rows[2]['counterpart_groups']=['b'];rows[3]['raw_source']=rows[0]['raw_source']+' /* same */'
        visible=[r['raw_source'] for r in rows];visible[5]=visible[4]
        groups=audit.dependency_groups(rows,visible)
        self.assertEqual(len(set(groups[:4])),1)
        self.assertEqual(groups[4],groups[5])
        self.assertNotEqual(groups[0],groups[4])

    def test_oof_fits_fold_local_vocab_scaler_and_rejects_valid(self):
        class Tokenizer:
            is_fast=True
            def __call__(self, source, **kwargs):
                n=min(len(source),kwargs['max_length'])
                return dict(input_ids=[1]*n,offset_mapping=[(i,i+1) for i in range(n)])
        rows=[dict(sample_key=str(i),split='train',label=i%2,counterpart_groups=[f'repair_{i//3}'],
            raw_source=f'int unique_{i}(){{int term_{i//3}=0; return {i};}}') for i in range(18)]
        vecs=[];scalers=[]
        def vectorizer(**kwargs):
            obj=original_vec(**kwargs);vecs.append(obj);return obj
        def scaler():
            obj=original_scale();scalers.append(obj);return obj
        original_vec,original_scale=audit.TfidfVectorizer,audit.StandardScaler
        with patch.object(audit,'TfidfVectorizer',side_effect=vectorizer),patch.object(audit,'StandardScaler',side_effect=scaler):
            cache,report=audit.shallow_oof_weights(rows,Tokenizer(),2048)
        self.assertEqual(len(cache),18);self.assertEqual(len(vecs),3);self.assertEqual(len(scalers),3)
        self.assertEqual(cache[0]['fold'],cache[1]['fold'])
        self.assertEqual({r['fold'] for r in cache},{1,2,3})
        sizes=audit.size_block([{'features':r} for r in cache])
        for fold,(vec,scale) in enumerate(zip(vecs,scalers),1):
            fit=[i for i,r in enumerate(cache) if r['fold']!=fold]
            np.testing.assert_allclose(scale.mean_,sizes[fit].mean(0))
            for r in cache:
                token='term_'+str(int(r['sample_key'])//3)
                if r['fold']==fold:self.assertNotIn(token,vec.vocabulary_)
                else:self.assertIn(token,vec.vocabulary_)
        for y in (0,1):
            self.assertAlmostEqual(np.mean([r['weight'] for r in cache if r['label']==y]),1.)
        with self.assertRaisesRegex(ValueError,'Train'):
            audit.shallow_oof_weights(rows+[dict(rows[0],sample_key='valid',split='valid')],Tokenizer(),2048)

    def test_hard_weights_use_fixed_half_threshold_and_label_means(self):
        hard,raw,weights=audit.normalize_hard_weights([0,0,1,1],[.5,.1,.2,.8])
        np.testing.assert_array_equal(hard,[True,False,True,False])
        np.testing.assert_array_equal(raw,[3,1,3,1])
        np.testing.assert_array_equal(weights,[1.5,.5,1.5,.5])

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
