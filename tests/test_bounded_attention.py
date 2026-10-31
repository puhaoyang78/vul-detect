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
from vulnmechanism.training import (
    paired_collate, paired_forward, shuffle_line_labels, random_attention_suppression,
    make_attention_ema, update_attention_ema, attention_ema_loss,
)
from vulnmechanism.evaluation import validate_paired
from vulnmechanism.evaluation import bounded_region_record, attention_distribution_record, attention_epoch_diagnostics


def detector(bounded=True, region=None, suppression=None):
    c = config()
    return LOSVERDetector(RobertaModel(c,add_pooling_layer=False),c,
        SimpleNamespace(class_weight=1.,bounded_attention=bounded,attention_region=region,
                        attention_suppression=suppression))


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

    def test_high_suppresses_rates_not_length_weighted_attention(self):
        model=detector(region="diff",suppression="high")
        states=torch.zeros(1,49,16,requires_grad=True)
        states.data[0,0,0]=4.
        states.data[0,9:,0]=.5
        membership=torch.zeros(1,10,49,dtype=torch.bool)
        for i in range(9): membership[0,i,i]=True
        membership[0,9,9:]=True
        model.line_attention.weight.data.zero_();model.line_attention.weight.data[0,0]=1.
        _,d=model.bounded_pool(states,torch.ones(1,49,dtype=torch.bool),membership,suppress=True)
        self.assertEqual(int(d["line_attention"].argmax(-1)),9)
        self.assertEqual(d["suppression_mask"].nonzero().tolist(),[[0,0]])
        self.assertEqual(float(d["suppressed_rates"][0,0]),1.)
        self.assertTrue((d["suppressed_token_attention"]>0).all())
        self.assertLessEqual(float(d["suppressed_token_attention"].max()/d["suppressed_token_attention"].min()),5.+1e-6)
        torch.testing.assert_close(d["suppressed_token_attention"].sum(),torch.tensor(1.))

    def test_random_suppression_rng_counts_padding_and_small_functions(self):
        data=dict(line_numbers=torch.tensor([list(range(23))+[-1]*3,list(range(9))+[-1]*17]))
        py,th=random.getstate(),torch.get_rng_state()
        a=random_attention_suppression(data,random.Random(123456))
        b=random_attention_suppression(data,random.Random(123456))
        torch.testing.assert_close(a,b)
        self.assertEqual(a.sum(-1).tolist(),[2,0])
        self.assertFalse((a & (data["line_numbers"]<0)).any())
        self.assertEqual(random.getstate(),py)
        torch.testing.assert_close(torch.get_rng_state(),th,rtol=0,atol=0)

    def test_suppression_two_bce_one_encoder_original_region_and_inference(self):
        records=rows();records[0]["func"]="\n".join(f"x += {i};" for i in range(12))
        data=batch(records)
        for mode in ("random","high"):
            with self.subTest(mode=mode):
                model=detector(region="diff",suppression=mode).train()
                mask=random_attention_suppression(data,random.Random(4)) if mode=="random" else None
                with patch.object(model.encoder,"forward",wraps=model.encoder.forward) as enc:
                    loss,p,d=paired_forward(model,data,"baseline","cpu",return_attention=True,suppression_mask=mask)
                self.assertEqual(enc.call_count,1)
                torch.testing.assert_close(d["bce"],(d["bce_normal"]+d["bce_suppressed"])/2)
                expected,n=model.attention_region_loss(d["line_rates"],d["line_counts"],data["line_labels"])
                torch.testing.assert_close(loss,d["bce"]+.1*expected)
                self.assertEqual(int(n),int(d["region_functions"]))
                self.assertEqual(d["suppression_mask"].sum(-1).tolist(),[1,0])
                loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                model.eval()
                with torch.no_grad():
                    p,d=paired_forward(model,data,"baseline","cpu",with_labels=False,return_attention=True)
                    self.assertNotIn("suppressed_rates",d)
                    model.attention_suppression=None
                    q=paired_forward(model,dict(data,line_labels=1-data["line_labels"]),"baseline","cpu",with_labels=False)
                torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_ema_full_encoder_update_kl_gradients_and_padding(self):
        model=detector(region="diff").train()
        model.line_attention.weight.data.normal_(std=.2)
        py,th=random.getstate(),torch.get_rng_state()
        ema=make_attention_ema(model)
        self.assertEqual(set(ema),{"encoder","line_attention"})
        self.assertFalse(ema.training);self.assertTrue(all(not p.requires_grad for p in ema.parameters()))
        self.assertEqual(random.getstate(),py)
        torch.testing.assert_close(torch.get_rng_state(),th,rtol=0,atol=0)
        before={k:v.clone() for k,v in ema.state_dict().items()}
        model.line_attention.weight.data.add_(.3)
        model.encoder.embeddings.word_embeddings.weight.data[5:25].add_(.2)
        data=batch()
        _,_,d=paired_forward(model,data,"baseline","cpu",return_attention=True)
        kl=attention_ema_loss(model,ema,data,d["line_attention"],"cpu")
        self.assertGreater(float(kl.detach()),0.)
        padded=dict(data,line_token_mask=torch.cat([data["line_token_mask"],torch.zeros(len(data["labels"]),2,512,dtype=torch.bool)],1))
        padded_kl=attention_ema_loss(model,ema,padded,torch.cat([d["line_attention"],torch.zeros(len(data["labels"]),2)],1),"cpu")
        torch.testing.assert_close(kl,padded_kl)
        kl.backward()
        self.assertGreater(float(model.line_attention.weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.encoder.embeddings.word_embeddings.weight.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in ema.parameters()))
        update_attention_ema(ema,model)
        for k,v in ema.state_dict().items():
            online=dict(encoder=model.encoder,line_attention=model.line_attention)[k.split('.')[0]].state_dict()[k.split('.',1)[1]]
            expected=before[k]*.99+online.detach()*.01 if v.is_floating_point() else online
            torch.testing.assert_close(v,expected)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in ema.state_dict().items() if k.startswith("encoder.")))

    def test_diagnostics_uniform_token_reference_and_rng_preservation(self):
        ann=dict(sample_key="x",pair_key="p",target=1,lines_ground_truth=[1])
        pred=dict(sample_key="x",line_numbers=[0,1],line_token_counts=[3,1],line_rates=[2.,2.])
        first=attention_distribution_record(ann,pred)
        self.assertAlmostEqual(first["token_mean_jsd"],0.)
        self.assertLess(first["normalized_line_entropy"],1.)
        self.assertIsNone(first["consecutive_jsd"])
        other=dict(pred,line_rates=[1.,3.])
        second=attention_distribution_record(ann,other,pred)
        self.assertGreater(second["consecutive_jsd"],0.)
        self.assertGreater(second["token_mean_jsd"],0.)
        model=detector(region="diff").train();data=batch()
        py,th=random.getstate(),torch.get_rng_state()
        with tempfile.TemporaryDirectory() as tmp:
            summary,previous=attention_epoch_diagnostics(model,{s:rows() for s in ("train","valid")},
                {s:[data] for s in ("train","valid")},Tokens(),"cpu",1,{},Path(tmp)/"diag.jsonl")
            again,_=attention_epoch_diagnostics(model,{s:rows() for s in ("train","valid")},
                {s:[data] for s in ("train","valid")},Tokens(),"cpu",2,previous,Path(tmp)/"diag.jsonl")
        self.assertTrue(model.training)
        self.assertEqual(random.getstate(),py)
        torch.testing.assert_close(torch.get_rng_state(),th,rtol=0,atol=0)
        self.assertIsNone(summary["valid"]["consecutive_jsd"])
        self.assertAlmostEqual(again["valid"]["consecutive_jsd"],0.)


if __name__=="__main__":
    unittest.main()
