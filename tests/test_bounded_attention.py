"""Bounded MIL, source-only inference, function-balanced edit-region margin."""

import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import RobertaModel

from tests.test_losver import Tokens, config
from vulnmechanism.data import paired_features
from vulnmechanism.model import LOSVERDetector, predict_checkpoint
from vulnmechanism.training import paired_collate, paired_forward, shuffle_line_labels
from vulnmechanism.evaluation import validate_paired
from vulnmechanism.evaluation import bounded_region_record


def detector(bounded=True, region=None):
    c = config()
    return LOSVERDetector(RobertaModel(c,add_pooling_layer=False),c,
        SimpleNamespace(class_weight=1.,bounded_attention=bounded,attention_region=region))


def rows():
    return [dict(func=s,target=y,lines_ground_truth=gt,sample_key=f"valid:{i}",pair_key="valid:0")
            for i,(s,y,gt) in enumerate([
                ("long_name(p, count);\r\n\r\n x++;\n return x;",1,[2]),
                ("check();\n safe();\n return;",0,[]),
            ])]


def batch(records=None):
    return paired_collate([paired_features(r,Tokens(),"baseline") for r in (rows() if records is None else records)])


class BoundedAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_zero_head_preserves_initialization_rng_sampler_and_mean(self):
        torch.manual_seed(123456)
        original=detector(False).eval()
        state=torch.get_rng_state()
        order=torch.randperm(30)
        torch.manual_seed(123456)
        model=detector().eval()
        torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)
        torch.testing.assert_close(torch.randperm(30),order,rtol=0,atol=0)
        for name,value in original.state_dict().items():
            torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
        self.assertFalse(model.line_attention.weight.any())
        data=batch()
        expected=paired_forward(original,data,"baseline","cpu",with_labels=False)
        with patch.object(model.encoder,"forward",wraps=model.encoder.forward) as enc:
            actual,details=paired_forward(model,data,"baseline","cpu",with_labels=False,return_attention=True)
            self.assertEqual(enc.call_count,1)
        torch.testing.assert_close(expected,actual,atol=1e-7,rtol=1e-6)
        for bias in (-4.,2.):
            model.line_attention.bias.data.fill_(bias)
            actual=paired_forward(model,data,"baseline","cpu",with_labels=False)
            torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)

    def test_all_tokens_padding_cross_line_mass_and_bound(self):
        model=detector().eval()
        states=torch.randn(1,5,16,requires_grad=True)
        valid=torch.tensor([[False,True,True,True,False]])
        membership=torch.tensor([[[True,True,True,False,False],[False,False,True,True,True]]])
        h,details=model.bounded_pool(states,valid,membership)
        torch.testing.assert_close(h,states[:,1:4].mean(1))
        torch.testing.assert_close(details["line_counts"].sum(),torch.tensor(3.))
        torch.testing.assert_close(details["token_attention"].sum(),torch.tensor(1.))
        self.assertTrue((details["token_attention"][valid]>0).all())
        self.assertFalse(details["token_attention"][~valid].any())
        model.line_attention.weight.data.normal_(std=10)
        _,details=model.bounded_pool(states,valid,membership)
        weights=details["token_attention"][valid]
        self.assertLessEqual(float(weights.max()/weights.min()),5.+1e-6)
        h.sum().backward()
        self.assertTrue((states.grad[:,1:4].abs().sum(-1)>0).all())
        self.assertFalse(states.grad[:,[0,4]].any())

    def test_missing_valid_token_is_error(self):
        model=detector()
        with self.assertRaises(ValueError):
            model.bounded_pool(torch.randn(1,3,16),torch.ones(1,3,dtype=torch.bool),
                               torch.tensor([[[True,False,True]]]))

    def test_region_mean_uses_tokens_and_eligible_function_denominator(self):
        rates=torch.tensor([[4.,1.,1.],[1.,2.,4.],[2.,2.,2.],[2.,2.,2.]],requires_grad=True)
        counts=torch.tensor([[1.,8.,1.],[2.,3.,5.],[1.,1.,1.],[1.,1.,1.]])
        labels=torch.tensor([[1,0,0],[1,0,0],[0,0,0],[1,1,1]])
        loss,n=LOSVERDetector.attention_region_loss(rates,counts,labels)
        expected=torch.relu(torch.log(torch.tensor(2.))-torch.log(torch.tensor(1.))+torch.log(torch.tensor(3.25)))/2
        torch.testing.assert_close(loss,expected)
        self.assertEqual(int(n),2)
        loss.backward()
        self.assertFalse(rates.grad[2:].any())
        self.assertLess(float(rates.grad[1,0]),0)
        self.assertGreater(float(rates.grad[1,1]),0)

    def test_region_gradient_encoder_after_first_attention_update(self):
        model=detector(region="diff").eval()
        data=batch()
        total,prob,details=paired_forward(model,data,"baseline","cpu",return_attention=True)
        torch.testing.assert_close(total,details["bce"]+.1*details["region_loss"])
        self.assertEqual(int(details["region_functions"]),1)
        details["region_loss"].backward()
        self.assertGreater(float(model.line_attention.weight.grad.abs().sum()),0.)
        enc=model.encoder.embeddings.word_embeddings.weight
        self.assertEqual(float(enc.grad.abs().sum()),0.) # required by zero initialization
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        optimizer.step();optimizer.zero_grad()
        _,_,details=paired_forward(model,data,"baseline","cpu",return_attention=True)
        details["region_loss"].backward()
        self.assertTrue(torch.isfinite(enc.grad).all())
        self.assertGreater(float(enc.grad.abs().sum()),0.)

    def test_empty_and_all_token_regions_keep_bce(self):
        records=rows()
        records[0]["lines_ground_truth"]=[]
        records[1]["lines_ground_truth"]=list(range(3))
        model=detector(region="diff").eval()
        total,prob,details=paired_forward(model,batch(records),"baseline","cpu",return_attention=True)
        self.assertEqual(int(details["region_functions"]),0)
        torch.testing.assert_close(total,details["bce"],rtol=0,atol=0)
        total.backward()
        self.assertGreater(float(model.encoder.embeddings.word_embeddings.weight.grad.abs().sum()),0.)

    def test_diff_never_changes_pooling_and_uniform_intervention(self):
        model=detector(region="diff").eval()
        model.line_attention.weight.data.normal_(std=.1)
        data=batch()
        _,p,_=paired_forward(model,data,"baseline","cpu",return_attention=True)
        changed=copy.deepcopy(data)
        changed["line_labels"]=1-changed["line_labels"]
        _,q,_=paired_forward(model,changed,"baseline","cpu",return_attention=True)
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        del changed["line_labels"]
        q=paired_forward(model,changed,"baseline","cpu",with_labels=False)
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        model.force_uniform_attention=True
        uniform=paired_forward(model,changed,"baseline","cpu",with_labels=False)
        original=detector(False).eval()
        original.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith("line_attention.")})
        expected=paired_forward(original,data,"baseline","cpu",with_labels=False)
        torch.testing.assert_close(uniform,expected,atol=1e-7,rtol=1e-6)

    def test_random_rows_preserve_availability_counts_and_global_rng(self):
        fs=[dict(line_labels=[1,0,0,0]),dict(line_labels=[0,0]),dict(line_labels=[1,1,1])]
        a,b=copy.deepcopy(fs),copy.deepcopy(fs)
        py,th=random.getstate(),torch.get_rng_state()
        shuffle_line_labels(a,123456);shuffle_line_labels(b,123456)
        self.assertEqual(a,b);self.assertEqual(random.getstate(),py)
        torch.testing.assert_close(torch.get_rng_state(),th,rtol=0,atol=0)
        for original,actual in zip(fs,a):
            self.assertEqual(sum(original["line_labels"]),sum(actual["line_labels"]))
            self.assertEqual(0<sum(original["line_labels"])<len(original["line_labels"]),
                             0<sum(actual["line_labels"])<len(actual["line_labels"]))

    def test_validation_no_diff_forward_and_real_save_reload(self):
        model=detector(region="diff").eval()
        data=batch()
        with patch.object(model,"forward",wraps=model.forward) as forward:
            metrics,predictions=validate_paired(model,rows(),Tokens(),"baseline","cpu",loader=[data])
            self.assertNotIn("labels",forward.call_args.kwargs)
            self.assertNotIn("line_labels",forward.call_args.kwargs)
        self.assertEqual(metrics["pairs"],1)
        self.assertIn("line_rates",predictions[0])
        cp=dict(protocol="losver",line_mapping="source_offsets",model_path="fixture",variant="baseline",input_mode="512",
                pooling="bounded_attention",bounded_attention=dict(region_supervision="diff"),state_dict=model.state_dict())
        def factory(path,variant,**kwargs):
            return detector(kwargs.get("bounded_attention",False),kwargs.get("attention_region")),Tokens()
        with tempfile.TemporaryDirectory() as d,patch("vulnmechanism.model.losver_model",side_effect=factory):
            p=Path(d)/"model.pt";torch.save(cp,p)
            z=predict_checkpoint(p,[dict(raw_source=rows()[0]["func"])],device="cpu")
        torch.testing.assert_close(z.sigmoid(),torch.tensor([predictions[0]["probability"]]),atol=1e-7,rtol=1e-6)

    def test_post_prediction_diagnostics_distinguish_coverage_and_rate_ratio(self):
        annotation=dict(sample_key="x",pair_key="p",target=1,lines_ground_truth=[1])
        prediction=dict(sample_key="x",line_numbers=[0,1],line_token_counts=[9,1],line_rates=[1,4])
        r=bounded_region_record(annotation,prediction)
        self.assertEqual(r["mean_rate_ratio"],4)
        self.assertEqual(r["token_weight_ratio"],4)
        self.assertEqual(r["token_coverage"],.1)
        self.assertAlmostEqual(r["region_attention_mass"],4/13)
        for selected in ([],[0,1]):
            r=bounded_region_record(dict(annotation,lines_ground_truth=selected),prediction)
            self.assertFalse(r["eligible"])
            self.assertIsNone(r["mean_rate_ratio"])


if __name__=="__main__":
    unittest.main()
