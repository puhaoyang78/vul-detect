import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from vulnmechanism.cfg_transfer_diagnostics import executable_tree,comment_variants,natural_pairs,summarize,COMMENTS

class Builder:
    source_max_length=2048
    def _encode(self,source):
        # Three suffixes have equal formal tokenizer length, checked on real data separately.
        for suffix in COMMENTS.values():
            if source.endswith(suffix):return list(range(len(source[:-len(suffix)])+7))
        return list(range(len(source)))

class TransferTests(unittest.TestCase):
    def test_fixed_branch_protocol_arguments(self):
        from vulnmechanism.cfg_transfer_diagnostics import topology_diagnosis
        from vulnmechanism.cfg_experiment import parser
        args=parser().parse_args(['diagnose-origin','--phase','branches','--variant',
                                 'lm_pretrain_cfg','--split','test','--output-dir','unused'])
        self.assertEqual((args.variant,args.split),('lm_pretrain_cfg','test'))
        for kwargs in ({'split':'train','branches_only':True}, {'variant':'baseline','branches_only':True},
                       {'split':'test'}, {'variant':'lm_pretrain_cfg'}):
            with self.assertRaises(ValueError):
                topology_diagnosis('missing','unused',device='cpu',**kwargs)

    def test_dependency_pair_keeps_reverse_order_and_checks_visibility(self):
        import torch
        from vulnmechanism.cfg_transfer_diagnostics import dependency_pair_features
        hidden=torch.arange(32).reshape(4,8)
        query=dict(definition_token=2,use_token=0,label=1)
        torch.testing.assert_close(dependency_pair_features(hidden,query),hidden[[2,0]])
        query['label']=0
        torch.testing.assert_close(dependency_pair_features(hidden,query),hidden[[2,0]])
        for position in (-1,4):
            with self.assertRaisesRegex(ValueError,'outside visible'):
                dependency_pair_features(hidden,dict(definition_token=position,use_token=0))

    def test_degree_preserving_topology_intervention(self):
        from collections import Counter
        from vulnmechanism.cfg_transfer_diagnostics import rewire_cfg
        edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,0)]
        before=list(edges)
        changed=rewire_cfg(edges,seed=42,key='fixture')
        self.assertEqual(edges,before)
        self.assertEqual(Counter(a for a,b in edges),Counter(a for a,b in changed))
        self.assertEqual(Counter(b for a,b in edges),Counter(b for a,b in changed))
        self.assertEqual(len(changed),len(set(changed)))
        self.assertIn((0,0),changed)
        self.assertNotEqual(set(edges),set(changed))
        self.assertEqual(changed,rewire_cfg(edges,seed=42,key='fixture'))
        self.assertEqual(rewire_cfg([],seed=42,key='empty'),[])

    def test_frozen_probe_train_only_statistics_and_roundtrip(self):
        import torch
        from vulnmechanism.cfg_transfer_diagnostics import fit_origin_probe
        torch.manual_seed(3)
        cache=dict(pool=torch.randn(8,8),pairs=torch.randn(12,2,8),
            labels=torch.tensor([0,1]*4),splits=['train']*6+['valid']*2,
            relation_labels=torch.tensor([0,1]*6),relation_splits=['train']*8+['valid']*4,
            relation_keys=['a','a','b','b','c','c','d','d','e','e','f','f'])
        for task in ('classification','dependency'):
            result=fit_origin_probe(cache,task=task,epochs=2)
            self.assertTrue(all(torch.isfinite(v).all() for v in result['selected']['head_state'].values()))
            self.assertTrue(torch.equal(result['mean'],(cache['pool'][:6] if task=='classification' else cache['pairs'][:8]).mean(0)))
            self.assertEqual(len(result['history']),2)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'probe.pt';torch.save(result,path)
                loaded=torch.load(path,weights_only=False)
                self.assertEqual(loaded['selected']['scores'],result['selected']['scores'])

    def test_scope_summary_refuses_partial_responses(self):
        from vulnmechanism.cfg_transfer_diagnostics import summarize_scope
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'function_readout';root.mkdir()
            protocol=dict(models={'m':{}}, cases=[dict(sample_key='a',case=c) for c in ('original','newline')])
            (root/'protocol.json').write_text(json.dumps(protocol))
            (root/'responses.jsonl').write_text(json.dumps(dict(model='m',sample_key='a',case='original'))+'\n')
            with self.assertRaisesRegex(ValueError,'incomplete or duplicated'):summarize_scope(tmp)

    def test_original_valid_summary_is_covered_subset_at_fixed_threshold(self):
        from vulnmechanism.cfg_transfer_diagnostics import summarize_scope
        from vulnmechanism.cfg_data import file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'function_readout';root.mkdir();folder=Path(tmp)/'m';folder.mkdir()
            (folder/'valid.predictions.jsonl').write_text('original full valid')
            (folder/'complete.json').write_text(json.dumps(dict(validation={},
                validation_predictions_sha256=file_sha256(folder/'valid.predictions.jsonl'))))
            protocol=dict(models={'m':dict(directory=tmp,variant='m')},
                cases=[dict(sample_key='a',case='original'),dict(sample_key='b',case='original')],
                coverage={'valid':dict(members=3,resolved=2,unresolved=1)},training_allowed=False)
            (root/'protocol.valid.json').write_text(json.dumps(protocol))
            records=[dict(model='m',sample_key=k,case='original',split='valid',label=y,threshold=.5,
                graph_logit=0,readouts={name:dict(score=score,logit=z) for name,score,z in
                    [('full',.7,1.),('function_core',.3,-1.)]}) for k,y in [('a',0),('b',1)]]
            (root/'responses.valid.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            report=summarize_scope(tmp,original_valid=True)
            self.assertEqual(report['m/valid/function_core']['selected_changes'],{'corrected':['a'],'introduced':['b']})
            self.assertEqual(report['m/valid/full']['selected_originals']['samples'],2)
            self.assertNotIn('m/train/full',report)

    def test_fixed_evaluation_no_grad_no_parameter_update(self):
        import torch
        from vulnmechanism.cfg_transfer_diagnostics import fixed_logits
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1));self.calls=0
            def forward(self,ids,mask):
                assert not torch.is_grad_enabled() and not self.training
                self.calls+=1
                return ids.float().sum(1)*self.weight
        m=Model();before=m.weight.detach().clone()
        a,b=fixed_logits(m,(torch.ones(1,3),torch.ones(1,3)),baseline=True)
        self.assertEqual(m.calls,1);self.assertFalse(a.requires_grad)
        self.assertTrue(torch.equal(before,m.weight));self.assertIsNone(m.weight.grad)

    def test_positional_reference_is_train_only(self):
        from vulnmechanism.cfg_transfer_diagnostics import position_reference
        def q(split,y):return dict(split=split,label=y,sample_key='a',condition_token=2,use_token=4,branch=0)
        with tempfile.TemporaryDirectory() as tmp,patch('vulnmechanism.cfg_transfer_diagnostics.iter_jsonl',
                side_effect=[[q('train',1),q('train',1)],[q('valid',0),q('valid',1)]]):
            report=position_reference(tmp)
        self.assertEqual(report['cells'],[dict(key=[0,1,1],positive=2,total=2)])
        with tempfile.TemporaryDirectory() as tmp,patch('vulnmechanism.cfg_transfer_diagnostics.iter_jsonl',return_value=[q('valid',1)]):
            with self.assertRaisesRegex(ValueError,'train-only'):position_reference(tmp)

    def test_append_comments_preserve_ast_and_input_is_unchanged(self):
        row=dict(raw_source='int f(int x) { return x+1; }',language='c',label=1)
        before=dict(row);variants=comment_variants(row,Builder())
        self.assertEqual(row,before)
        self.assertEqual(len({r['source_tokens'] for r in variants.values()}),1)
        for v in variants.values():self.assertEqual(executable_tree(row['raw_source'],'c'),executable_tree(v['raw_source'],'c'))
        self.assertNotEqual(executable_tree('int f(){return 1;}','c'),executable_tree('int f(){return 0;}','c'))
        for source in ('int f(){return X;}\n#define X 1','int f(){', 'int f(){return 1;} int g(){return 2;}'):
            with self.assertRaises(ValueError):comment_variants(dict(row,raw_source=source),Builder())
        builder=Builder();builder.source_max_length=2
        with self.assertRaisesRegex(ValueError,'truncation'):comment_variants(row,builder)

    def test_pairs_do_not_use_test_or_scores_or_invent_patch_labels(self):
        rows=[dict(function_name='f',sample_key=str(i),label=i%2,split='valid',raw_source=f'int f(){{return {i};}}') for i in range(2)]
        pairs=natural_pairs(rows)
        self.assertEqual(len(pairs),1)
        self.assertIn('not verified',pairs[0]['provenance'])
        self.assertEqual(natural_pairs([dict(r,score=1-r['label']) for r in rows]),pairs)
        self.assertEqual(natural_pairs([dict(r,split='test') for r in rows]),[])

    def test_summary_keeps_threshold_and_independent_function_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            records=[]
            for key in ('a','b'):
                for case,z in [('original',-.2),('neutral',-.1),('risk_cue',.2),('safe_cue',-.3)]:
                    records.append(dict(model='m',sample_key=key,case=case,logit=z,graph_logit=.1,prediction=int(z>0),label=0))
            report=summarize(tmp,records)['m']
            self.assertEqual(report['functions'],2);self.assertEqual(report['cue_flips'],2)
            self.assertAlmostEqual(report['mean_cue_logit_gap'],.5)
            self.assertEqual(report['variants']['neutral']['flips'],0)
