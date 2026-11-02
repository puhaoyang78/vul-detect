"""Check the label-free frozen readout interventions and their upper bounds."""
import unittest
import copy

import torch

from vulnmechanism.mil_diagnosis import pooling_interventions,bounded_logit_extrema


class MILDiagnosisTests(unittest.TestCase):
    def test_box_capacity_matches_all_vertices_and_zero_floor_is_stable(self):
        import itertools
        z=torch.tensor([-4.,2.,8.,1.]);n=torch.tensor([.5,5.,2.,1.])
        expected=[]
        for values in itertools.product((1.,5.),repeat=4):
            mass=n.double()*torch.tensor(values,dtype=torch.double)
            expected.append((mass*z.double()).sum()/mass.sum())
        low,high=bounded_logit_extrema(z,n)
        torch.testing.assert_close(low,min(expected),rtol=0,atol=1e-12)
        torch.testing.assert_close(high,max(expected),rtol=0,atol=1e-12)
        scores,_=pooling_interventions(z,torch.ones(4),n,torch.full((4,),-1000.))
        self.assertTrue(torch.isfinite(scores['zero_floor']))
        torch.testing.assert_close(scores['zero_floor'],(z*n).sum()/n.sum(),rtol=1e-4,atol=1e-4)
    def test_identical_inputs_cannot_be_fixed_by_label_informed_pooling_bound(self):
        from vulnmechanism.mil_diagnosis import summarize
        scores,stats=pooling_interventions(torch.tensor([-1.,1.]),torch.ones(2),torch.ones(2))
        rows=[dict(target=i%2==0,pair_key=str(i//2),lines_ground_truth=[]) for i in range(4)]
        for r in rows:r['target']=int(r['target'])
        records=[dict(target=r['target'],source_tokens=2,visible_inputs_differ=i>=2,
                      probabilities={k:float(v.sigmoid()) for k,v in scores.items()},**stats)
                 for i,r in enumerate(rows)]
        s=summarize(rows,records)
        self.assertEqual(s['convex_pooling_bounds']['all']['possible_p_c_pairs'],1)
        self.assertEqual(s['convex_pooling_bounds']['identical_input']['possible_p_c_pairs'],0)

    def test_input_marginalization_exact_inputs_rng_and_hard_label_preservation(self):
        from tests.test_bounded_attention import batch, rows
        from tests.test_losver import Tokens
        from vulnmechanism.data import paired_features
        from vulnmechanism.training import marginalize_input_targets,paired_collate
        rs=rows()
        rs[1]['func']=rs[0]['func']
        f=[paired_features(r,Tokens(),'baseline') for r in rs]
        extra=copy.deepcopy(f[0]); f.append(extra)
        unique=copy.deepcopy(f[1]); unique['line_token_indices'][0]=unique['line_token_indices'][0][1:]; f.append(unique)
        original=copy.deepcopy(f);rng=torch.get_rng_state()
        summary=marginalize_input_targets(f)
        torch.testing.assert_close(torch.get_rng_state(),rng,atol=0,rtol=0)
        self.assertEqual(summary['conflicting_functions'],3)
        self.assertEqual([x['classification_target'] for x in f],[2/3,2/3,2/3,0.])
        for a,b in zip(f,original):
            self.assertEqual({k:v for k,v in a.items() if k!='classification_target'},b)
        data=paired_collate(f[:3])
        torch.testing.assert_close(data['labels'],torch.tensor([1,0,1]))
        torch.testing.assert_close(data['classification_targets'],torch.full((3,),2/3))

    def test_conditional_bce_is_exact_average_of_label_assignments(self):
        import torch.nn.functional as F
        w=torch.tensor([.2,-.3],requires_grad=True)
        h=torch.tensor([[.7,2.],[2.,1.]])
        z=h@w
        hard=F.binary_cross_entropy_with_logits(z,torch.tensor([1.,0.]))
        swapped=F.binary_cross_entropy_with_logits(z,torch.tensor([0.,1.]))
        soft=F.binary_cross_entropy_with_logits(z,torch.full((2,),.5))
        torch.testing.assert_close(soft,(hard+swapped)/2)
        g=torch.autograd.grad(soft,w,retain_graph=True)[0]
        avg=torch.autograd.grad((hard+swapped)/2,w)[0]
        torch.testing.assert_close(g,avg)

    def test_soft_targets_only_enter_training_and_gradients_flow(self):
        from tests.test_bounded_attention import detector,batch
        from vulnmechanism.training import paired_forward
        torch.set_num_threads(2)
        m=detector(context=True,region='diff')
        b=batch();b['classification_targets']=torch.tensor([.5,.5])
        m.eval()
        with torch.no_grad():
            before=paired_forward(m,b,'baseline','cpu',with_labels=False)
            _,_,d=paired_forward(m,b,'baseline','cpu',return_attention=True)
        torch.testing.assert_close(d['bce'],torch.nn.functional.binary_cross_entropy_with_logits(d['logits'],b['labels'].float()))
        m.train()
        loss,_,d=paired_forward(m,b,'baseline','cpu',return_attention=True)
        torch.testing.assert_close(d['bce'],torch.nn.functional.binary_cross_entropy_with_logits(d['logits'],b['classification_targets']))
        loss.backward()
        self.assertGreater(float(m.classifier.weight.grad.abs().sum()),0.)
        self.assertGreater(float(m.encoder.embeddings.word_embeddings.weight.grad.abs().sum()),0.)
        m.eval();b['classification_targets'].fill_(float('nan'));b['line_labels'].fill_(float('nan'))
        with torch.no_grad():
            after=paired_forward(m,b,'baseline','cpu',with_labels=False)
        torch.testing.assert_close(after,before,atol=0,rtol=0)

    def test_training_resume_keeps_targets_rng_optimizer_and_selection(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tests.test_resume import ResumeTests,Interrupted
        from tests.test_bounded_attention import detector
        from tests.test_losver import Tokens
        from vulnmechanism import training,model as core
        from vulnmechanism.data import read_jsonl,write_jsonl
        helper=ResumeTests();helper.setUp()
        def factory(*args,**kwargs):
            return detector(context=True,region='diff'),Tokens()
        with tempfile.TemporaryDirectory() as tmp,patch.object(core,'losver_model',side_effect=factory):
            root=Path(tmp);helper.paired_fixture(root,'512')
            path=root/'data/primevul_paired/train.lines.jsonl';rows=read_jsonl(path)
            rows[1]['func']=rows[0]['func'];write_jsonl(path,rows)
            settings=dict(training.LOSVER_CONFIG,batch_size=4,eval_batch_size=2,
                          epochs=dict(baseline=2,localizer=2,predicted=2,oracle=2))
            options=dict(config=settings,device='cpu',bounded_attention=True,attention_context=True,
                         attention_region='diff',marginalize_inputs=True,seed=123457,save_every=1)
            training.train_losver(root/'data',root/'control','fixture','unixcoder',**options)
            with helper.interruption(epoch=2,step=4,after_save=True),self.assertRaises(Interrupted):
                training.train_losver(root/'data',root/'resumed','fixture','unixcoder',**options)
            training.train_losver(root/'data',root/'resumed','fixture','unixcoder',resume=True,**options)
            helper.compare_runs(root/'control/primevul_paired/unixcoder',root/'resumed/primevul_paired/unixcoder',['best-p_c.pt'])
            with self.assertRaisesRegex(ValueError,'completed training configuration'):
                training.train_losver(root/'data',root/'control','fixture','unixcoder',resume=True,
                                     **dict(options,marginalize_inputs=False))

    def test_linear_decomposition_padding_and_fixed_interventions(self):
        h = torch.tensor([[1., 2.], [-2., 1.], [3., -1.], [0., 2.], [999., 999.]])
        w, bias = torch.tensor([.3, -.7]), .2
        n, r = torch.tensor([2., 5., .5, 4., 0.]), torch.tensor([1., 5., 2., 3., 999.])
        z = h @ w + bias
        scores, bounds = pooling_interventions(z, r, n)
        mass = n[:4] * r[:4]
        expected = (h[:4] * (mass / mass.sum())[:,None]).sum(0) @ w + bias
        torch.testing.assert_close(scores['original'], expected)
        self.assertAlmostEqual(bounds['min_line_logit'], float(z[:4].min()))
        self.assertAlmostEqual(bounds['max_line_logit'], float(z[:4].max()))
        torch.testing.assert_close(scores['remove_attention_top3'], z[2])
        for s in scores.values():
            self.assertGreaterEqual(float(s), bounds['min_line_logit'] - 1e-6)
            self.assertLessEqual(float(s), bounds['max_line_logit'] + 1e-6)

    def test_one_line_and_empty_mapping(self):
        scores, bounds = pooling_interventions(torch.tensor([2.]), torch.tensor([4.]), torch.tensor([3.]))
        self.assertTrue(all(float(v)==2 for v in scores.values()))
        self.assertEqual(bounds['attention_effective_fraction'],1.)
        with self.assertRaises(ValueError):
            pooling_interventions(torch.tensor([2.]),torch.tensor([4.]),torch.tensor([0.]))

    def test_intermediate_layers_share_forward_and_keep_original_prediction(self):
        from tests.test_bounded_attention import detector, batch
        from vulnmechanism.training import paired_forward
        torch.set_num_threads(2)
        model=detector(context=True).eval()
        b=batch()
        calls=[]
        handle=model.encoder.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            expected=paired_forward(model,b,'baseline','cpu',with_labels=False)
            actual,d=paired_forward(model,b,'baseline','cpu',with_labels=False,return_attention=True,
                                    return_line_states=True,diagnostic_layers=(0,1))
        handle.remove()
        self.assertEqual(len(calls),2)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        torch.testing.assert_close(d['diagnostic_line_states'][1],d['line_states'],atol=0,rtol=0)
        valid=d['line_counts']>0
        torch.testing.assert_close(d['line_rates'][valid],1+4*d['attention_scores'][valid].sigmoid(),rtol=0,atol=0)
        self.assertFalse(torch.equal(d['diagnostic_line_states'][0],d['line_states']))
        with self.assertRaises(ValueError):
            paired_forward(model,b,'baseline','cpu',with_labels=False,return_attention=True,diagnostic_layers=(9,))
        model.train()
        with self.assertRaises(ValueError):
            paired_forward(model,b,'baseline','cpu',return_attention=True,diagnostic_layers=(1,))


if __name__ == '__main__':
    unittest.main()
