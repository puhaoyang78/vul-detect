"""Actual dropout, decoupled decay and independent-line R-Drop gradients."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from transformers import RobertaModel

from tests.test_losver import Tokens, config
from vulnmechanism.data import paired_features
from vulnmechanism.model import LOSVERLocalizer
from vulnmechanism.training import (
    LOSVERAdamW, losver_parameter_groups, paired_collate, paired_forward, paired_rdrop_loss,
)
from vulnmechanism.cli import build_parser


def localizer(dropout=.1):
    c = config()
    return LOSVERLocalizer(RobertaModel(c, add_pooling_layer=False), c,
                          SimpleNamespace(class_weight=1., dropout_probability=dropout))


def batch():
    return paired_collate([
        paired_features(dict(func=code, target=y, lines_ground_truth=[1]), Tokens(), "localizer")
        for code, y in (("int f(){\n use();\n}", 1), ("int g(){\n check();\n use();\n}", 0))
    ])


class LocalizerRegularizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_dropout_changes_only_stochastic_behavior_not_initialization(self):
        torch.manual_seed(123456)
        original = localizer(0)
        torch.manual_seed(123456)
        regularized = localizer(.1)
        for k, value in original.state_dict().items():
            torch.testing.assert_close(value, regularized.state_dict()[k], rtol=0, atol=0)
        self.assertEqual(regularized.line_attention.dropout, .1)
        self.assertEqual([m.p for m in regularized.line_classifier.modules() if isinstance(m, nn.Dropout)], [.1, .1])
        self.assertEqual(regularized.config.hidden_dropout_prob, original.config.hidden_dropout_prob)
        regularized.train()
        regularized.encoder.eval()  # Isolate ONLY the newly configurable layers.
        x = torch.cat(paired_forward(regularized, batch(), "localizer", "cpu", with_labels=False))
        y = torch.cat(paired_forward(regularized, batch(), "localizer", "cpu", with_labels=False))
        self.assertGreater(float((x-y).abs().max()), 0)
        regularized.eval()
        a = torch.cat(paired_forward(regularized, batch(), "localizer", "cpu", with_labels=False))
        b = torch.cat(paired_forward(regularized, batch(), "localizer", "cpu", with_labels=False))
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_decay_excludes_numeric_layernorm_and_attention_bias(self):
        m = localizer()
        groups = losver_parameter_groups(m, .01)
        decay = {id(p) for p in groups[0]["params"]}
        excluded = {id(p) for p in groups[1]["params"]}
        self.assertFalse(decay & excluded)
        self.assertEqual(decay | excluded, {id(p) for p in m.parameters()})
        for module in m.modules():
            if isinstance(module, nn.LayerNorm):
                self.assertTrue(all(id(p) in excluded for p in module.parameters()))
        self.assertIn(id(m.line_attention.in_proj_bias), excluded)
        self.assertIn(id(m.line_classifier[0].weight), decay)
        with torch.no_grad():
            for p in m.parameters():
                p.fill_(1.)
                p.grad = torch.zeros_like(p)
        opt = LOSVERAdamW(groups, lr=.1)
        opt.step()
        for p in groups[0]["params"]:
            torch.testing.assert_close(p, torch.full_like(p, .999), atol=1e-7, rtol=0)
        for p in groups[1]["params"]:
            torch.testing.assert_close(p, torch.ones_like(p), atol=0, rtol=0)
        for state in opt.state.values():
            self.assertEqual(float(state["exp_avg"].abs().max()), 0)
            self.assertEqual(float(state["exp_avg_sq"].abs().max()), 0)

    def test_decay_follows_unchanged_adaptive_update(self):
        p = nn.Parameter(torch.tensor([.5, -.2], dtype=torch.float64))
        gradient = torch.tensor([.3, -.4], dtype=torch.float64)
        p.grad = gradient.clone()
        expected = p.detach().clone()
        avg = .1 * gradient
        sq = .001 * gradient.square()
        expected.addcdiv_(avg, sq.sqrt() + 1e-8, value=-.003 * (.001**.5) / .1)
        expected.add_(expected, alpha=-.003*.01)
        LOSVERAdamW([p], lr=.003, weight_decay=.01).step()
        torch.testing.assert_close(p, expected, atol=1e-14, rtol=0)

    def test_original_zero_decay_checkpoint_still_resumes(self):
        p = nn.Parameter(torch.tensor([.5, -.2]))
        original = LOSVERAdamW([p], lr=.003)
        p.grad = torch.tensor([.3, -.4])
        original.step()
        saved = copy.deepcopy(original.state_dict())
        # Existing unregularized checkpoints have no weight_decay group key.
        del saved["param_groups"][0]["weight_decay"]
        q = nn.Parameter(p.detach().clone())
        resumed = LOSVERAdamW([q], lr=.003)
        resumed.load_state_dict(saved)
        p.grad = torch.tensor([-.1, .2])
        q.grad = p.grad.clone()
        original.step()
        resumed.step()
        torch.testing.assert_close(p, q, atol=0, rtol=0)

    def test_bernoulli_kl_matches_two_class_formula_and_both_gradients(self):
        a = torch.tensor([[.2, -.7, float('inf')], [1.2, 1000., -1000.]], requires_grad=True)
        b = torch.tensor([[-.3, .4, -float('inf')], [.1, -1000., 1000.]], requires_grad=True)
        valid = torch.tensor([[True, True, False], [True, False, False]])
        loss = LOSVERLocalizer.symmetric_bernoulli_kl(a, b, valid)
        x,y=a[valid],b[valid]
        logp=torch.stack((nn.functional.logsigmoid(x),nn.functional.logsigmoid(-x)),-1)
        logq=torch.stack((nn.functional.logsigmoid(y),nn.functional.logsigmoid(-y)),-1)
        expected=((logp.exp()*(logp-logq)).sum(-1)+(logq.exp()*(logq-logp)).sum(-1)).mean()/2
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue((a.grad[valid].abs()>0).all())
        self.assertTrue((b.grad[valid].abs()>0).all())
        self.assertEqual(float(a.grad[~valid].abs().sum()+b.grad[~valid].abs().sum()),0)
        duplicated=LOSVERLocalizer.symmetric_bernoulli_kl(a.repeat(2,1), b.repeat(2,1), valid.repeat(2,1))
        torch.testing.assert_close(duplicated,loss)

    def test_kl_extremes_and_empty_valid_mask(self):
        a = torch.tensor([1000., -1000.], requires_grad=True)
        b = -a.detach().clone().requires_grad_()
        loss = LOSVERLocalizer.symmetric_bernoulli_kl(a,b,torch.ones(2,dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        empty = LOSVERLocalizer.symmetric_bernoulli_kl(a,b,torch.zeros(2,dtype=torch.bool))
        self.assertEqual(float(empty),0)

    def test_two_focals_and_kl_only_differ_by_coefficient(self):
        m = localizer().train()
        data = batch()
        torch.manual_seed(42)
        with patch.object(m,"forward",wraps=m.forward) as forward:
            zero,parts0=paired_rdrop_loss(m,data,"cpu",0.)
            self.assertEqual(forward.call_count,2)
        torch.testing.assert_close(zero,parts0["focal"])
        self.assertGreater(float(parts0["rdrop_kl"]),0)
        torch.manual_seed(42)
        loss,parts=paired_rdrop_loss(m,data,"cpu",.1)
        torch.testing.assert_close(parts["focal"],parts0["focal"],atol=0,rtol=0)
        torch.testing.assert_close(parts["rdrop_kl"],parts0["rdrop_kl"],atol=0,rtol=0)
        torch.testing.assert_close(loss,zero+.1*parts0["rdrop_kl"])
        loss.backward()
        for parameter in (m.encoder.embeddings.word_embeddings.weight,m.line_attention.in_proj_weight,m.line_classifier[-1].weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.norm()),0)

    def test_cli_omitted_versus_explicit_zero(self):
        parser=build_parser()
        self.assertIsNone(parser.parse_args(["train"]).localizer_rdrop)
        args=parser.parse_args(["train","--localizer-dropout",".1","--localizer-weight-decay",".01","--localizer-rdrop","0"])
        self.assertEqual((args.localizer_dropout,args.localizer_weight_decay,args.localizer_rdrop),(.1,.01,0.))


if __name__ == "__main__":
    unittest.main()
