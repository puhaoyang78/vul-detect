"""Old-side gap alignment, one-pass auxiliary ranking and exact training resume."""

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
from vulnmechanism import model as core, training
from vulnmechanism.cli import build_parser
from vulnmechanism.data import insertion_annotations, insertion_features, modified_lines, paired_features, write_jsonl
from vulnmechanism.evaluation import load_paired, insertion_localization_metrics


def detector(mode="diff", **options):
    c = config()
    args = dict(class_weight=1., bounded_attention=True, attention_context=True,
                attention_region="diff", insertion_supervision=mode)
    args.update(options)
    return core.LOSVERDetector(RobertaModel(c, add_pooling_layer=False), c, SimpleNamespace(**args))


def records(pairs=None, split="train"):
    if pairs is None:
        pairs = [("int f(){\n use(x);\n return x;\n}",
                  "int f(){\n check(x);\n use(x);\n guard(x);\n return x;\n}")]
    result = []
    for i, (old, new) in enumerate(pairs):
        for j, (source, other) in enumerate(((old, new), (new, old))):
            result.append(dict(func=source, target=1-j, pair_key=f"{split}:{i}", sample_key=f"{split}:{2*i+j}",
                               project="fixture", lines_ground_truth=modified_lines(source, other)))
    return result


def features(rows=None):
    return [paired_features(r, Tokens(), "baseline", insertion=True)
            for r in insertion_annotations(records() if rows is None else rows)]


class InsertionLocalizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_shared_diff_multiple_positions_and_unchanged_line_labels(self):
        rows = records(); annotated = insertion_annotations(rows)
        self.assertEqual(rows[0]["lines_ground_truth"], [])
        self.assertTrue(annotated[0]["pure_addition_pair"])
        self.assertTrue(annotated[1]["pure_addition_pair"])
        self.assertEqual([e["start"] for e in annotated[0]["insertion_edits"]], [1, 2])
        fs = features(rows)
        self.assertEqual(fs[0]["insertion_truth"].count(1), 2)
        self.assertEqual(set(fs[0]["line_labels"]), {0})
        self.assertEqual(fs[1]["insertion_truth"].count(1), 0)
        self.assertNotIn("insertion_edits", rows[0])

    def test_replacement_and_blank_only_addition_are_not_positive_insertions(self):
        for old, new in (("return a;\n", "return b;\n"), ("a();\n", "a();\n\n")):
            rows = insertion_annotations(records([(old, new)]))
            self.assertFalse(rows[0]["pure_addition_pair"])
            self.assertEqual(rows[0]["insertion_edits"], [])

    def test_blank_run_interval_and_crlf_are_gap_targets_not_neighboring_lines(self):
        old = "int f(){\r\n\r\n\r\n a();\r\n}\r\n"
        new = old.replace(" a();", " check();\r\n a();")
        f = features(records([(old, new)]))[0]
        positive = f["insertion_truth"].index(1)
        self.assertEqual(f["insertion_ranges"][positive], [1, 3])
        self.assertEqual((f["insertion_left"][positive], f["insertion_right"][positive]), (0, 1))
        self.assertEqual(f["line_numbers"], [0, 3, 4])
        synthetic = insertion_features(dict(line_numbers=[0,3,4], line_complete=[True]*3,
            source_tokens=20, source_lines=5), [dict(start=1,end=3), dict(start=2,end=2)])
        self.assertEqual(synthetic["insertion_truth"].count(1), 1)

    def test_actual_start_end_boundaries_and_all_positive_no_rank_loss(self):
        for new, target in (("check();\na();\n", 0), ("a();\ncheck();\n", 1)):
            f = features(records([("a();\n", new)]))[0]
            self.assertEqual(f["insertion_ranges"], [[0,0],[1,1]])
            self.assertEqual(f["insertion_truth"].index(1), target)
            self.assertEqual(f["insertion_left"], [-1,0])
            self.assertEqual(f["insertion_right"], [0,-1])
        z = torch.tensor([[1.,2.]], requires_grad=True)
        loss, count = core.LOSVERDetector.insertion_ranking_loss(z, torch.ones_like(z, dtype=torch.bool), torch.ones_like(z))
        self.assertEqual(float(loss), 0.)
        self.assertEqual(int(count), 0)
        loss.backward(); self.assertEqual(float(z.grad.abs().sum()), 0.)

    def test_partial_line_and_truncated_prefix_never_become_eof(self):
        old = "a();\nb();\n" + "z"*700 + ";\n"
        new = "check();\n" + old + "end_check();\n"
        f = features(records([(old,new)]))[0]
        self.assertTrue(f["source_tokens"] > 510)
        self.assertFalse(f["line_complete"][-1])
        self.assertEqual(f["insertion_ranges"], [[0,0],[1,1]])
        self.assertEqual(f["insertion_truth"], [1,0])
        self.assertTrue(all(r >= 0 for r in f["insertion_right"]))
        old = "z"*700 + ";\n"
        f = features(records([(old, "check();\n"+old)]))[0]
        self.assertEqual(f["insertion_ranges"], [])
        b = training.paired_collate([f]); model = detector().eval()
        loss, _, d = training.paired_forward(model, b, "baseline", "cpu", return_attention=True)
        self.assertEqual(int(d["insertion_functions"]), 0)
        self.assertEqual(float(d["insertion_loss"]), 0.)
        torch.testing.assert_close(loss, d["bce"] + .1*d["region_loss"])

    def test_pair_alignment_rejects_mismatched_existing_diff(self):
        rows = records(); rows[0]["lines_ground_truth"] = [0]
        with self.assertRaises(ValueError): insertion_annotations(rows)
        with self.assertRaises(ValueError): insertion_annotations(records()[:1])

    def test_random_control_counts_truth_and_global_rng(self):
        original = features(); shuffled = copy.deepcopy(original)
        state, torch_state = random.getstate(), torch.get_rng_state().clone()
        training.shuffle_insertion_labels(shuffled, 123456)
        again = copy.deepcopy(original); training.shuffle_insertion_labels(again, 123456)
        self.assertEqual(shuffled, again)
        for a,b in zip(original, shuffled):
            self.assertEqual(a["insertion_truth"], b["insertion_truth"])
            self.assertEqual(a["insertion_labels"].count(1), b["insertion_labels"].count(1))
            self.assertEqual(a["line_labels"], b["line_labels"])
        self.assertEqual(random.getstate(), state)
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)

    def test_initial_original_parameters_outputs_and_sampler_rng_equal(self):
        torch.manual_seed(123456); original = detector(None).eval()
        state = torch.get_rng_state().clone(); order = torch.randperm(27)
        torch.manual_seed(123456); model = detector().eval()
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        torch.testing.assert_close(torch.randperm(27), order, rtol=0, atol=0)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in model.insertion_head.parameters()), 1600)
        b = training.paired_collate(features())
        a = training.paired_forward(original, b, "baseline", "cpu", with_labels=False)
        p, d = training.paired_forward(model, b, "baseline", "cpu", with_labels=False, return_attention=True)
        torch.testing.assert_close(a, p, rtol=0, atol=0)
        self.assertEqual(int(d["insertion_mask"].sum()), sum(len(f["insertion_ranges"]) for f in features()))

    def test_auxiliary_gradients_one_encoding_and_loss_identity(self):
        model = detector().eval(); b = training.paired_collate(features())
        calls = []; handle = model.encoder.register_forward_pre_hook(lambda *_: calls.append(1))
        loss, _, d = training.paired_forward(model, b, "baseline", "cpu", return_attention=True)
        handle.remove(); self.assertEqual(len(calls), 1)
        torch.testing.assert_close(loss, d["bce"] + .1*d["region_loss"] + .1*d["insertion_loss"])
        self.assertEqual(int(d["insertion_functions"]), 1)
        before = model.insertion_head[0].weight.detach().clone()
        d["insertion_loss"].backward()
        self.assertTrue(all(p.grad.abs().sum() > 0 for p in model.insertion_head.parameters()))
        self.assertGreater(float(model.encoder.embeddings.word_embeddings.weight.grad.abs().sum()), 0)
        torch.optim.SGD(model.parameters(), lr=.1).step()
        self.assertFalse(torch.equal(before, model.insertion_head[0].weight))

    def test_global_representation_affects_within_function_ranking(self):
        torch.manual_seed(61); model = detector()
        left, right = torch.randn(1,3,16), torch.randn(1,3,16)
        g = torch.randn(1,1,16, requires_grad=True)
        scores = model.insertion_head(torch.cat([left,right,g.expand(-1,3,-1)],-1)).squeeze(-1)
        loss, _ = model.insertion_ranking_loss(scores, torch.ones(1,3,dtype=torch.bool), torch.tensor([[1,0,0]]))
        loss.backward(); self.assertGreater(float(g.grad.abs().sum()), 0)

    def test_function_average_not_number_of_negatives_and_multiple_positives(self):
        z = torch.tensor([[1.,0.,0.,0.,0.,0.],[-1.,0.,0.,0.,0.,0.]], requires_grad=True)
        mask = torch.tensor([[1,1,0,0,0,0],[1,1,1,1,1,1]], dtype=torch.bool)
        targets = torch.tensor([[1,0,-1,-1,-1,-1],[1,0,0,0,0,0]])
        value, count = detector().insertion_ranking_loss(z, mask, targets)
        expected = (torch.nn.functional.softplus(torch.tensor(-1.)) + torch.nn.functional.softplus(torch.tensor(1.)))/2
        torch.testing.assert_close(value, expected); self.assertEqual(int(count), 2)
        value.backward(); self.assertEqual(float(z.grad[0,2:].abs().sum()), 0.)
        value, _ = detector().insertion_ranking_loss(torch.tensor([[2.,1.,0.]]), torch.ones(1,3,dtype=torch.bool), torch.tensor([[1,1,0]]))
        torch.testing.assert_close(value, torch.nn.functional.softplus(torch.tensor([-2.,-1.])).mean())

    def test_label_free_padding_and_single_function_inference(self):
        model = detector().eval(); fs = features(); b = training.paired_collate(fs)
        p,d = training.paired_forward(model,b,"baseline","cpu",with_labels=False,return_attention=True)
        no_labels = {k:v for k,v in b.items() if k not in ('labels','line_labels','insertion_labels','insertion_truth')}
        no_labels['input_ids'] = b['input_ids'].masked_fill(~b['attention_mask'].bool(), 9)
        q,other = training.paired_forward(model,no_labels,"baseline","cpu",with_labels=False,return_attention=True)
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        torch.testing.assert_close(d['insertion_logits'],other['insertion_logits'],rtol=0,atol=0)
        one = paired_features(dict(func=records()[0]['func'], target=0),Tokens(),'baseline',insertion=True)
        q = training.paired_forward(model,training.paired_collate([one]),'baseline','cpu',with_labels=False)
        torch.testing.assert_close(p[:1],q,rtol=1e-6,atol=1e-7)
        with patch.object(core,'losver_model',return_value=(detector(),Tokens())):
            restored,_=load_paired(dict(line_mapping='source_offsets',input_mode='512',variant='baseline',model_path='fixture',
                pooling='bounded_attention',state_dict=model.state_dict(),bounded_attention=dict(region_supervision='diff',context=True),
                insertion_localization=dict(supervision='diff')), 'cpu')
        r=training.paired_forward(restored,b,'baseline','cpu',with_labels=False)
        torch.testing.assert_close(r,p,rtol=0,atol=0)

    def test_cli_and_incompatible_combinations(self):
        a = build_parser().parse_args(['train','--insertion-supervision','random'])
        self.assertEqual(a.insertion_supervision, 'random')
        for options in (dict(attention_context=False),dict(attention_region='random'),dict(attention_local=True),dict(attention_context_scale=True)):
            with self.assertRaises(ValueError): detector(**options)

    def test_localization_metrics_multiple_positions_and_invisible_denominator(self):
        old, new = records()[0]['func'], records()[1]['func']
        long_source = 'z'*700+';\n'
        rows = records([(old,new),(long_source,'check();\n'+long_source)])
        fs = features(rows)
        predicted = []
        for r,f in zip(rows,fs):
            scores = [float(i+1) if y == 1 else -1. for i,y in enumerate(f['insertion_truth'])]
            predicted.append(dict(sample_key=r['sample_key'],insertion_ranges=f['insertion_ranges'],
                insertion_truth=f['insertion_truth'],insertion_scores=scores))
        metrics, _ = insertion_localization_metrics(rows,predicted)
        self.assertEqual(metrics['pure_functions'],2)
        self.assertEqual(metrics['mapped_functions'],1)
        self.assertEqual(metrics['hit_at_1'],1.)
        self.assertEqual(metrics['hit_at_1_all_pure'],.5)
        self.assertEqual(metrics['positive_gap_recall_at_1'],.5)
        self.assertEqual(metrics['positive_gap_recall_at_3'],1.)
        predicted[0]['insertion_truth'] = [0]*len(predicted[0]['insertion_truth'])
        with self.assertRaisesRegex(ValueError,'source Diff'):
            insertion_localization_metrics(rows,predicted)

    def test_exact_training_resume_for_real_and_random_insertion_supervision(self):
        from tests import test_resume
        checker = test_resume.ResumeTests()
        pairs = [("int f0(){\n use(x);\n return x;\n}","int f0(){\n check(x);\n use(x);\n guard(x);\n return x;\n}"),
                 ("int f1(){\n use(y);\n}","int f1(){\n check(y);\n use(y);\n}"),
                 ("int f2(){\n bad();\n}","int f2(){\n safe();\n}"),
                 ("int f3(){\n"+"z"*550+";\n use(x);\n}","int f3(){\n"+"z"*550+";\n check(x);\n use(x);\n}")]
        def factory(path, variant, **kwargs):
            return detector(kwargs.get('insertion_supervision')), Tokens()
        for mode in ('diff','random'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, patch.object(core,'losver_model',side_effect=factory):
                root=Path(tmp)
                for split, ps in (('train',pairs),('valid',pairs[:2])):
                    write_jsonl(root/'data/primevul_paired'/f'{split}.lines.jsonl', records(ps,split))
                settings=dict(training.LOSVER_CONFIG,batch_size=4,eval_batch_size=2,
                              epochs=dict(training.LOSVER_CONFIG['epochs'],baseline=2))
                options=dict(device='cpu',config=settings,bounded_attention=True,attention_context=True,
                             attention_region='diff',insertion_supervision=mode,function_micro_batch=2,save_every=1)
                training.train_losver(root/'data',root/'control','fixture','unixcoder',**options)
                with checker.interruption(epoch=1,step=1,after_save=True), self.assertRaises(test_resume.Interrupted):
                    training.train_losver(root/'data',root/'resumed','fixture','unixcoder',**options)
                training.train_losver(root/'data',root/'resumed','fixture','unixcoder',resume=True,**options)
                checker.compare_runs(root/'control/primevul_paired/unixcoder',root/'resumed/primevul_paired/unixcoder',['best-p_c.pt'])
                self.assertEqual((root/'control/primevul_paired/unixcoder/train.insertion_positions.jsonl').read_bytes(),
                                 (root/'resumed/primevul_paired/unixcoder/train.insertion_positions.jsonl').read_bytes())


if __name__ == '__main__':
    unittest.main()
