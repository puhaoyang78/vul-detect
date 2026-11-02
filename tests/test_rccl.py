"""Reliable equal-line alignment, paired controls and training-only RCCL."""

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
from tests.test_insertion_localization import records
from vulnmechanism import model as core, training
from vulnmechanism.cli import build_parser
from vulnmechanism.data import _line_diff, paired_features, rccl_annotations, rccl_coverage, write_jsonl
from vulnmechanism.evaluation import load_paired


def detector(mode="diff", **options):
    c = config()
    args = dict(class_weight=1., bounded_attention=True, attention_context=True,
                attention_region="diff", rccl=mode)
    args.update(options)
    return core.LOSVERDetector(RobertaModel(c, add_pooling_layer=False), c, SimpleNamespace(**args))


def features(rows=None, mode="diff"):
    rows = records() if rows is None else rows
    fs = [paired_features(r, Tokens(), "baseline") for r in rows]
    annotations = rccl_annotations(rows, fs, mode)
    return fs, annotations


class RCCLTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_original_diff_unchanged_and_edit_neighbor_coordinates(self):
        rows = records()
        deleted, edits = _line_diff(rows[0]["func"], rows[1]["func"])
        a, b, alignment = _line_diff(rows[0]["func"], rows[1]["func"], return_alignment=True)
        self.assertEqual((a, b), (deleted, edits))
        self.assertEqual(alignment["blocks"], [[1, 1, 1, 2], [2, 2, 3, 4]])
        fs, rs = features(rows)
        self.assertEqual(rs[0]["near_lines"], [[0, 0], [1, 2], [2, 4]])
        self.assertEqual(fs[0]["line_labels"], [0]*4)
        self.assertEqual(fs[1]["line_labels"], [0, 1, 0, 1, 0, 0])
        self.assertTrue(rs[0]["pure_addition"])
        self.assertEqual(rccl_coverage(rs)["pure_addition"]["pair_coverage"], 1.)

    def test_duplicate_line_ambiguity_and_blank_neighbors(self):
        old = "a();\n\n same();\n b();\n same();\n c();\n"
        new = old.replace(" b();", " check();\n b();")
        fs, rs = features(records([(old, new)]))
        self.assertEqual(rs[0]["near_lines"], [[3, 4]])
        self.assertEqual(rs[0]["ambiguous_equal_lines"], 2)
        old = "a();\r\n\r\n b();\r\n"
        _, rs = features(records([(old, old.replace(" b();", " check();\r\n b();"))]))
        self.assertEqual(rs[0]["near_lines"], [[0, 0], [2, 3]])

    def test_truncated_partial_lines_equal_visible_inputs_and_empty(self):
        prefix = "a();\n" + "z"*700 + ";\n"
        _, rs = features(records([(prefix + "old();\n", prefix + "new();\n")]))
        self.assertFalse(rs[0]["visible_inputs_differ"])
        self.assertFalse(rs[0]["eligible"])
        old = "a();\n" + "z"*700 + ";\n"
        fs, rs = features(records([(old, "check();\n"+old)]))
        self.assertTrue(rs[0]["eligible"])
        self.assertEqual(rs[0]["near_lines"], [[0, 1]])
        self.assertFalse(fs[0]["line_complete"][-1])
        for old, new in (("old();\n", "new();\n"), ("a();\n", "a();\n")):
            fs, rs = features(records([(old, new)]))
            self.assertFalse(rs[0]["eligible"])
            b = training.paired_collate(fs)
            self.assertEqual(tuple(b["rccl_matches"].shape), (0, 4))
            model = detector().train()
            loss, _, d = training.paired_forward(model, b, "baseline", "cpu", return_attention=True, rccl_dropout_seed=1)
            self.assertEqual(float(d["rccl_loss"]), 0.)
            torch.testing.assert_close(loss, d["bce"] + .1*d["region_loss"])
            loss.backward()

    def test_random_counts_reproducibility_and_all_original_rng_unchanged(self):
        rows = records([("head();\n a();\n b();\n tail();\n", "head();\n a();\n check();\n b();\n tail();\n")]*3)
        state, py = torch.get_rng_state().clone(), random.getstate()
        a, ar = features(rows)
        b, br = features(rows, "random")
        c, cr = features(rows, "random")
        self.assertEqual(br, cr)
        for x, y in zip(a, b):
            self.assertEqual(len(x["rccl_selected"]), len(y["rccl_selected"]))
            self.assertEqual(x["line_labels"], y["line_labels"])
            self.assertTrue(set(map(tuple, y["rccl_selected"])).issubset(set(map(tuple, x["rccl_alignment"]))))
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(py, random.getstate())
        with self.assertRaises(ValueError):
            training.paired_collate(a[:1])
        with self.assertRaises(ValueError):
            rccl_annotations(rows[::-1], b, "diff")

    def test_initial_original_parameters_rng_and_matched_paired_sampler(self):
        torch.manual_seed(123456)
        original = detector(None).eval()
        state = torch.get_rng_state().clone()
        order = list(training.RepairPairSampler([None]*18))
        for mode in ("baseline", "diff", "random"):
            torch.manual_seed(123456); model = detector(mode).eval()
            torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
            self.assertEqual(list(training.RepairPairSampler([None]*18)), order)
            self.assertEqual(sorted(order), list(range(18)))
            self.assertTrue(all(order[i+1] == order[i]+1 and order[i] % 2 == 0 for i in range(0,18,2)))
            for k, v in original.state_dict().items():
                torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
            b = training.paired_collate(features()[0])
            a = training.paired_forward(original,b,"baseline","cpu",with_labels=False)
            p = training.paired_forward(model,b,"baseline","cpu",with_labels=False)
            torch.testing.assert_close(a,p,rtol=0,atol=0)

    def test_four_view_loss_matches_manual_formula_and_private_dropout(self):
        model = detector()
        lines = torch.randn(4,3,16, requires_grad=True)
        matches = torch.tensor([[0,0,1,1],[0,2,1,2],[2,0,3,0]])
        owners = torch.tensor([0,0,1])
        state = torch.get_rng_state().clone()
        loss, count = model.rccl_loss(lines,matches,owners,73)
        torch.testing.assert_close(state,torch.get_rng_state(),rtol=0,atol=0)
        views = torch.stack([lines[matches[:,0],matches[:,1]]]*2 + [lines[matches[:,2],matches[:,3]]]*2,1)
        keep = (torch.rand(views.shape,generator=torch.Generator().manual_seed(73)) >= .1)
        z = torch.nn.functional.normalize(model.rccl_projection(views*keep/.9),dim=-1)
        manual = []
        for row in z:
            terms = []
            for anchor, positive in enumerate((1,0,3,2)):
                scores = (row[anchor]*row).sum(-1)/.1
                terms.append(torch.logsumexp(scores[torch.arange(4) != anchor],0)-scores[positive])
            manual.append(torch.stack(terms).mean())
        expected = (torch.stack(manual[:2]).mean()+manual[2])/2
        torch.testing.assert_close(loss,expected)
        self.assertEqual(int(count),2)
        loss.backward()
        self.assertGreater(float(model.rccl_projection.weight.grad.abs().sum()),0)
        self.assertEqual(float(lines.grad[2,1:].abs().sum()),0.)
        different, _ = model.rccl_loss(lines,matches,owners,74)
        self.assertNotEqual(float(loss),float(different))

    def test_one_encoder_gradients_parameter_update_padding_and_inference(self):
        model = detector().train()
        fs, _ = features(); b = training.paired_collate(fs)
        calls = []; handle = model.encoder.register_forward_pre_hook(lambda *_: calls.append(1))
        loss,p,d = training.paired_forward(model,b,"baseline","cpu",return_attention=True,rccl_dropout_seed=123456)
        handle.remove(); self.assertEqual(len(calls),1)
        torch.testing.assert_close(loss,d["bce"]+.1*d["region_loss"]+.05*d["rccl_loss"])
        before = model.rccl_projection.weight.detach().clone()
        d["rccl_loss"].backward()
        self.assertGreater(float(model.encoder.embeddings.word_embeddings.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.rccl_projection.weight.grad.abs().sum()),0)
        torch.optim.SGD(model.parameters(),lr=.1).step()
        self.assertFalse(torch.equal(before,model.rccl_projection.weight))
        model.eval()
        p = training.paired_forward(model,b,"baseline","cpu",with_labels=False)
        b = {k:v for k,v in b.items() if k not in ("labels","line_labels","rccl_matches","rccl_owners")}
        b["input_ids"] = b["input_ids"].masked_fill(~b["attention_mask"].bool(),9)
        q = training.paired_forward(model,b,"baseline","cpu",with_labels=False)
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        one = paired_features(dict(func=records()[0]["func"],target=0),Tokens(),"baseline")
        q = training.paired_forward(model,training.paired_collate([one]),"baseline","cpu",with_labels=False)
        torch.testing.assert_close(p[:1],q,rtol=1e-6,atol=1e-7)
        cp = dict(protocol="losver",line_mapping="source_offsets",input_mode="512",variant="baseline",model_path="fixture",
                  pooling="bounded_attention",state_dict=model.state_dict(),
                  bounded_attention=dict(region_supervision="diff",context=True),rccl=dict(mode="diff"))
        with patch.object(core,"losver_model",return_value=(detector(None),Tokens())):
            restored,_ = load_paired(cp,"cpu")
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/"smoke.pt"
                torch.save(cp,path)
                single=core.predict_checkpoint(path,[dict(raw_source=records()[0]["func"])],device="cpu").sigmoid()
                torch.testing.assert_close(single,p[0,0].reshape(1),rtol=1e-6,atol=1e-7)
        self.assertFalse(hasattr(restored,"rccl_projection"))
        r = training.paired_forward(restored,b,"baseline","cpu",with_labels=False)
        torch.testing.assert_close(p,r,rtol=0,atol=0)

    def test_cli_incompatible_combinations_and_exact_resume(self):
        self.assertEqual(build_parser().parse_args(["train","--rccl","random"]).rccl,"random")
        for options in (dict(attention_region="random"),dict(attention_context=False),dict(attention_local=True),
                        dict(insertion_supervision="diff"),dict(attention_context_scale=True)):
            with self.assertRaises(ValueError): detector(**options)
        from tests import test_resume
        checker = test_resume.ResumeTests()
        pairs = [(f"head{i}();\n use{i}();\n return {i};\n", f"head{i}();\n check{i}();\n use{i}();\n return {i};\n") for i in range(5)]
        def factory(path,variant,**kwargs):
            return detector(kwargs.get("rccl")),Tokens()
        for mode in ("baseline","diff","random"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, patch.object(core,"losver_model",side_effect=factory):
                root=Path(tmp)
                for split, ps in (("train",pairs),("valid",pairs[:2])):
                    write_jsonl(root/"data/primevul_paired"/f"{split}.lines.jsonl",records(ps,split))
                settings=dict(training.LOSVER_CONFIG,batch_size=4,eval_batch_size=2,
                              epochs=dict(training.LOSVER_CONFIG["epochs"],baseline=2))
                options=dict(device="cpu",config=settings,bounded_attention=True,attention_context=True,
                             attention_region="diff",rccl=mode,function_micro_batch=2,save_every=1)
                training.train_losver(root/"data",root/"control","fixture","unixcoder",**options)
                with checker.interruption(epoch=1,step=1,after_save=True), self.assertRaises(test_resume.Interrupted):
                    training.train_losver(root/"data",root/"resumed","fixture","unixcoder",**options)
                training.train_losver(root/"data",root/"resumed","fixture","unixcoder",resume=True,**options)
                checker.compare_runs(root/"control/primevul_paired/unixcoder",root/"resumed/primevul_paired/unixcoder",["best-p_c.pt"])
                self.assertEqual((root/"control/primevul_paired/unixcoder/train.rccl_alignment.jsonl").read_bytes(),
                                 (root/"resumed/primevul_paired/unixcoder/train.rccl_alignment.jsonl").read_bytes())

    def test_diagnostics_keep_source_only_classification_and_fixed_cohorts(self):
        from vulnmechanism.rccl import diagnostics
        from vulnmechanism.evaluation import validate_paired
        rows = records()
        model = detector(None).eval()
        with tempfile.TemporaryDirectory() as tmp:
            value = diagnostics(model,Tokens(),rows,tmp,"valid",device="cpu",projection=torch.randn(32,16))
            expected, _ = validate_paired(model,rows,Tokens(),"baseline","cpu")
        self.assertEqual(value["metrics"]["all"]["pair_counts"],expected["pair_counts"])
        self.assertEqual(value["similarity"]["all"]["near_raw"]["pairs"],1)
        self.assertEqual(value["similarity"]["all"]["near_raw"]["lines"],3)
        self.assertEqual(value["similarity"]["all"]["random_raw"]["lines"],3)
        self.assertFalse(value["classification_projection_used"])


if __name__ == "__main__":
    unittest.main()
