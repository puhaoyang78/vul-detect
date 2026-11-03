"""Global/local MIL controls share encoding, initialization and training protocol."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import RobertaModel

from tests.test_bounded_attention import batch
from tests.test_losver import config, Tokens
from vulnmechanism.model import LOSVERDetector
from vulnmechanism.training import paired_forward


def detector(mode=None, **extra):
    c = config()
    args = dict(class_weight=1., bounded_attention=True, attention_context=True,
                attention_region='diff', local_mil=mode)
    args.update(extra)
    return LOSVERDetector(RobertaModel(c, add_pooling_layer=False), c, SimpleNamespace(**args))


class DualStreamTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_initialization_preserves_original_weights_rng_and_prediction(self):
        torch.manual_seed(123456)
        original = detector().eval()
        rng = torch.get_rng_state()
        order = torch.randperm(30)
        heads = []
        for mode in ('mean', 'lse'):
            torch.manual_seed(123456)
            model = detector(mode).eval()
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            torch.testing.assert_close(torch.randperm(30), order, rtol=0, atol=0)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            self.assertEqual(set(model.state_dict()) - set(original.state_dict()),
                             {'local_classifier.weight', 'local_classifier.bias'})
            self.assertNotIn('local_fusion', dict(model.named_parameters()))
            self.assertFalse(model.local_classifier.weight.any())
            self.assertFalse(model.local_classifier.bias.any())
            self.assertEqual(sum(p.numel() for p in model.local_classifier.parameters()), 34)
            heads.append(copy.deepcopy(model.local_classifier.state_dict()))
            b = batch()
            expected = paired_forward(original, b, 'baseline', 'cpu', with_labels=False)
            actual, d = paired_forward(model, b, 'baseline', 'cpu', with_labels=False, return_attention=True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertFalse(d['local_residual'].any())
        for k in heads[0]:
            torch.testing.assert_close(heads[0][k], heads[1][k], rtol=0, atol=0)

    def test_first_step_head_gradients_updates_and_original_losses(self):
        for mode in ('mean', 'lse'):
            torch.manual_seed(123456)
            model = detector(mode)
            opt = torch.optim.SGD(model.parameters(), lr=.02)
            original_head = model.local_classifier.weight.detach().clone()
            for step in range(2):
                loss, p, d = paired_forward(model, batch(), 'baseline', 'cpu', return_attention=True)
                torch.testing.assert_close(loss, d['bce'] + .1*d['region_loss'])
                torch.testing.assert_close(d['logits'], d['global_logits'] + d['local_residual'])
                torch.testing.assert_close(p[:, 0], d['logits'].sigmoid())
                loss.backward()
                head_grad = float(model.local_classifier.weight.grad.abs().sum())
                self.assertGreater(head_grad, 0.)
                if step == 1:
                    self.assertGreater(float(d['local_residual'].detach().abs().sum()), 0.)
                self.assertGreater(float(model.encoder.embeddings.word_embeddings.weight.grad.abs().sum()), 0.)
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                opt.step()
                opt.zero_grad()
            self.assertFalse(torch.equal(original_head, model.local_classifier.weight))
            model.eval()
            _, details = paired_forward(model, batch(), 'baseline', 'cpu', with_labels=False, return_attention=True)
            details['local_residual'].square().mean().backward()
            self.assertGreater(float(model.encoder.embeddings.word_embeddings.weight.grad.abs().sum()), 0.)
            self.assertGreater(float(model.local_classifier.weight.grad.abs().sum()), 0.)
            self.assertIsNone(model.classifier.weight.grad)

    def test_stable_length_normalized_lse_mean_and_padding(self):
        scores = torch.tensor([[[10000., 10001.], [9999., 10003.], [float('nan'), float('nan')]]], requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        for mode in ('mean', 'lse'):
            actual = LOSVERDetector.pool_local_classes(scores, mask, mode)
            selected = scores[:, :2]
            expected = selected.mean(1) if mode == 'mean' else torch.logsumexp(selected, 1)-torch.tensor(2.).log()
            torch.testing.assert_close(actual, expected)
            self.assertTrue(torch.isfinite(actual).all())
            actual.sum().backward(retain_graph=True)
            self.assertFalse(scores.grad[:, 2].any())
            scores.grad.zero_()
            duplicated = LOSVERDetector.pool_local_classes(selected.repeat(1, 3, 1), torch.ones(1, 6, dtype=torch.bool), mode)
            torch.testing.assert_close(duplicated, actual)
            one = LOSVERDetector.pool_local_classes(selected[:, :1], mask[:, :1], mode)
            torch.testing.assert_close(one, selected[:, 0])
        with self.assertRaises(ValueError):
            LOSVERDetector.pool_local_classes(scores, torch.zeros_like(mask), 'lse')
        with self.assertRaises(ValueError):
            LOSVERDetector.pool_local_classes(scores, mask, 'topk')

    def test_both_local_classes_change_the_signed_residual(self):
        valid = torch.ones(1, 2, dtype=torch.bool)
        for mode in ('mean', 'lse'):
            vulnerable = torch.tensor([[[0., 4.], [0., 0.]]])
            safe = vulnerable.flip(-1)
            a = LOSVERDetector.pool_local_classes(vulnerable, valid, mode)
            b = LOSVERDetector.pool_local_classes(safe, valid, mode)
            self.assertGreater(float(a[0, 1]-a[0, 0]), 0.)
            torch.testing.assert_close(a[:, 1]-a[:, 0], -(b[:, 1]-b[:, 0]))

    def test_shared_physical_line_mapping_conserves_cross_line_token_mass(self):
        states = torch.randn(1, 5, 16)
        code = torch.tensor([[False, True, True, True, False]])
        membership = torch.tensor([[[False, True, True, False, False],
                                    [False, False, True, True, False],
                                    [False, False, False, False, True]]])
        weighted = (membership & code[:, None]).float()
        weighted /= weighted.sum(1).clamp_min(1)[:, None]
        counts = weighted.sum(-1)
        lines = torch.bmm(weighted, states)/counts.clamp_min(1e-12)[..., None]
        for mode in ('mean', 'lse'):
            model = detector(mode).eval()
            _, d = model.bounded_pool(states, code, membership, return_line_states=True)
            torch.testing.assert_close(d['line_counts'], counts)
            torch.testing.assert_close(d['line_counts'].sum(), code.sum().float())
            torch.testing.assert_close(d['line_states'], lines)
            expected = model.pool_local_classes(model.local_classifier(lines), counts > 0, mode)
            torch.testing.assert_close(d['local_class_scores'], expected)
            self.assertFalse(d['local_line_scores'][:, 2].any())

    def test_one_encoding_and_diff_does_not_enter_inference(self):
        model = detector('lse').eval()
        torch.nn.init.normal_(model.local_classifier.weight, std=.1)
        b = batch()
        calls = []
        handle = model.encoder.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            before = paired_forward(model, b, 'baseline', 'cpu', with_labels=False)
        handle.remove()
        self.assertEqual(calls, [1])
        b['line_labels'].fill_(-999)
        b['labels'].fill_(0)
        with torch.no_grad():
            after = paired_forward(model, b, 'baseline', 'cpu', with_labels=False)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        from vulnmechanism.data import paired_features
        from vulnmechanism.training import paired_collate
        f = paired_features(dict(func='x();\n return;', target=0), Tokens(), 'baseline')
        one = paired_collate([f])
        with torch.no_grad():
            self.assertEqual(paired_forward(model, one, 'baseline', 'cpu', with_labels=False).shape, (1, 1))

    def test_checkpoint_and_single_function_api(self):
        from vulnmechanism.evaluation import load_paired
        from vulnmechanism.model import predict_checkpoint
        from vulnmechanism.training import paired_collate
        from vulnmechanism.data import paired_features
        def factory(*args, **kwargs):
            return detector(kwargs.get('local_mil'), local_mil_version=kwargs.get('local_mil_version', 2)), Tokens()
        source = 'x();\n return;'
        for mode, version in ((None, 2), ('mean', 2), ('lse', 2), ('mean', 1), ('lse', 1)):
            m = detector(mode, local_mil_version=version).eval()
            if mode:
                torch.nn.init.normal_(m.local_classifier.weight, std=.1)
                if version == 1: m.local_fusion.data.fill_(.2)
            cp = dict(protocol='losver', model_path='fixture', dataset='primevul_paired', variant='baseline', input_mode='512',
                      line_mapping='source_offsets', pooling='bounded_attention',
                      bounded_attention=dict(context=True, region_supervision='diff'), state_dict=m.state_dict())
            if mode:
                cp['local_mil'] = dict(pooling=mode)
                if version == 2: cp['local_mil']['version'] = 2
            data = paired_collate([paired_features(dict(func=source, target=0), Tokens(), 'baseline')])
            with patch('vulnmechanism.model.losver_model', side_effect=factory), torch.no_grad():
                restored, _ = load_paired(cp, 'cpu')
                expected = paired_forward(m, data, 'baseline', 'cpu', with_labels=False)[:, 0]
                actual = paired_forward(restored, data, 'baseline', 'cpu', with_labels=False)[:, 0]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp)/'model.pt'
                    torch.save(cp, path)
                    out = predict_checkpoint(path, [dict(raw_source=source)], device='cpu')
                torch.testing.assert_close(out.sigmoid(), expected)

    def test_training_resume_and_pooling_mismatch(self):
        from tests.test_resume import ResumeTests, Interrupted
        from vulnmechanism import training, model as core
        def factory(*args, **kwargs):
            return detector(kwargs.get('local_mil'), local_mil_version=kwargs.get('local_mil_version', 2)), Tokens()
        helper = ResumeTests(); helper.setUp()
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'losver_model', side_effect=factory):
            root = Path(tmp); helper.paired_fixture(root, '512')
            settings = dict(training.LOSVER_CONFIG, batch_size=4, eval_batch_size=2,
                            epochs=dict(baseline=2, localizer=2, predicted=2, oracle=2))
            options = dict(config=settings, device='cpu', bounded_attention=True, attention_context=True,
                           attention_region='diff', local_mil='lse', save_every=1)
            training.train_losver(root/'data', root/'control', 'fixture', 'unixcoder', **options)
            with helper.interruption(epoch=2, step=4, after_save=True), self.assertRaises(Interrupted):
                training.train_losver(root/'data', root/'resumed', 'fixture', 'unixcoder', **options)
            training.train_losver(root/'data', root/'resumed', 'fixture', 'unixcoder', resume=True, **options)
            helper.compare_runs(root/'control/primevul_paired/unixcoder', root/'resumed/primevul_paired/unixcoder', ['best-p_c.pt'])
            with self.assertRaisesRegex(ValueError, 'completed training configuration'):
                training.train_losver(root/'data', root/'control', 'fixture', 'unixcoder', resume=True,
                                     **dict(options, local_mil='mean'))
            import json
            path = root/'control/primevul_paired/unixcoder/training.metrics.json'
            old = json.loads(path.read_text()); old['local_mil'].pop('version')
            path.write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError, 'completed training configuration'):
                training.train_losver(root/'data', root/'control', 'fixture', 'unixcoder', resume=True, **options)

    def test_rejects_stacked_modules_and_cli_defaults(self):
        from vulnmechanism.cli import build_parser
        args = build_parser().parse_args(['train'])
        self.assertIsNone(args.local_mil)
        self.assertEqual(build_parser().parse_args(['train', '--local-mil', 'lse']).local_mil, 'lse')
        for extra in (dict(attention_context=False), dict(attention_local=True),
                      dict(attention_context_scale=True), dict(insertion_supervision='diff'), dict(rccl='baseline')):
            with self.assertRaises(ValueError):
                detector('lse', **extra)

    def test_paired_gain_loss_and_repair_visibility_cohorts(self):
        from vulnmechanism.dual_stream import pair_changes, cohort_indices
        rows = [dict(target=i % 2 == 0, pair_key=str(i//2),
                     lines_ground_truth=gt) for i, gt in enumerate(([], [1], [2], [3]))]
        features = [dict(input_ids=ids, line_numbers=[0, 1]) for ids in ([1], [1], [1], [2])]
        groups = cohort_indices(rows, features)
        self.assertEqual(groups['pure_addition'], [0, 1])
        self.assertEqual(groups['replacement'], [2, 3])
        self.assertEqual(groups['visible_difference'], [2, 3])
        changes = pair_changes(rows, [.6, .4, .7, .8], [.8, .9, .6, .3])
        self.assertEqual((changes['gains'], changes['losses']), (1, 1))
        self.assertEqual(changes['gained_pairs'], ['0'])
        self.assertEqual(changes['lost_pairs'], ['1'])

    def test_historical_join_uses_complete_pair_key_and_rejects_missing_or_duplicates(self):
        from vulnmechanism.dual_stream import align_predictions, saved_logit, pair_category
        rows = [dict(pair_key=f'primevul_paired:{s}:0', target=y, sample_key=f'{s}:{y}')
                for s in ('train','valid') for y in (1,0)]
        pred = [dict(r, probability=.6) for r in reversed(rows)]
        self.assertEqual([r['sample_key'] for r in align_predictions(rows, pred)],
                         [r['sample_key'] for r in rows])
        for invalid in (pred[:-1], pred+[pred[0]], [dict(p, pair_key='0') for p in pred]):
            with self.assertRaises(ValueError): align_predictions(rows, invalid)
        self.assertIsNone(saved_logit(dict(probability=1.))[0])
        self.assertAlmostEqual(saved_logit(dict(probability=.5))[0], 0.)
        self.assertEqual(saved_logit(dict(global_logit=2., local_residual=-.1, probability=.9))[0], 1.9)
        self.assertEqual([pair_category(a,b) for a,b in ((.6,.5),(.6,.6),(.5,.5),(.5,.6))],
                         ['p_c','p_v','p_b','p_r'])

    def test_formal_validation_exports_both_contributions_and_fp_changes(self):
        from tests.test_bounded_attention import rows
        from vulnmechanism.evaluation import validate_paired
        from vulnmechanism.dual_stream import branch_summary
        model = detector('lse').eval()
        torch.nn.init.normal_(model.local_classifier.weight, std=.1)
        metrics, pred = validate_paired(model, rows(), Tokens(), 'baseline', 'cpu')
        self.assertEqual(set(metrics['local_mil']['by_target']), {'0', '1'})
        for p in pred:
            self.assertIn('global_probability', p)
            self.assertIn('scaled_local_probability', p)
            self.assertEqual(len(p['local_line_scores']), len(p['line_numbers']))
            self.assertAlmostEqual(p['local_residual'], p['local_logit'], places=6)
        summary = branch_summary(rows(), pred)
        self.assertIn('introduced_patched_false_positives', summary)
        self.assertEqual(summary['global_only_metrics']['samples'], 2)


if __name__ == '__main__':
    unittest.main()
