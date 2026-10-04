import copy
import unittest
import torch
from vulnmechanism.syntax import function_readout_span
from vulnmechanism.model import InputBuilder
from vulnmechanism.cfg_network import SourceGraphClassifier, collate_graphs
from tests.test_cfg_alignment import CharacterTokenizer
from tests.test_cfg_regions import HPSource
from tests.test_cfg_joint_control import JointControlTests


class FunctionReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.builder = InputBuilder(CharacterTokenizer(), source_max_length=2048, context_max_length=384)

    def row(self, source, language='cpp'):
        return dict(raw_source=source, language=language)

    def test_function_extent_unicode_template_member_and_literals(self):
        sources = ['int f(){ const char *s="} /* 😀 */"; /*keep*/ return 2; }',
                   'template<class T> T A<T>::f(T x) { return x; }',
                   'A::A() : x(1) { auto f = [](){return 0;}; }',
                   'f(int x) { return x+1; }']
        for source in sources:
            full = '// 外部\r\n' + source + '\n/* tail } */'
            # Missing-return-type fragments have a strict, explicit grammar.
            if source.startswith('f('): full = source + '\n/* tail } */'
            a, b = function_readout_span(full, 'cpp')
            self.assertEqual(full[a:b], source)
        for source in ['int f(){', 'int f(){} int g(){}', 'return 3;', 'int f(){if(1){}',
                       'if (x) {}', 'else if (x > 0) { return 1; }']:
            with self.assertRaises(ValueError): function_readout_span(source, 'c')

    def test_masks_keep_input_comments_signature_truncation_and_padding(self):
        rows = [self.row('/*头*/ int f(){ /*keep*/ return 123; }\n/*tail*/'), self.row('int g(){return 2;}')]
        ids, attention, pooling = self.builder.source_function_batch(rows, device='cpu')
        original = self.builder.sequence_batch(rows, variant='baseline', excluded_groups=(), device='cpu')
        self.assertTrue(torch.equal(ids, original[0])); self.assertTrue(torch.equal(attention, original[1]))
        prefix = len(self.builder.source_prefix)
        self.assertFalse(pooling[:, :prefix].any())
        for i, row in enumerate(rows):
            a, b = function_readout_span(row['raw_source'], 'cpp')
            self.assertEqual(int(pooling[i].sum()), b-a)
            self.assertFalse(pooling[i, int(attention[i].sum())-1:].any())
        self.builder.source_max_length = 15
        ids, attention, pooling = self.builder.source_function_batch([rows[1]], device='cpu')
        self.assertEqual(int(pooling.sum()), 15)
        with self.assertRaisesRegex(ValueError, 'no visible'):
            self.builder.source_function_batch([self.row('/*' + ' ' * 30 + '*/ int f(){}')], device='cpu')

    def test_key_code_changes_are_retained_tail_is_not_directly_pooled(self):
        rows = [self.row(s) for s in ['int f(){return 1;}', 'int f(){return 1;}\n/* unrelated */', 'int f(){return 0;}']]
        ids, attention, pool = self.builder.source_function_batch(rows, device='cpu')
        self.assertTrue(torch.equal(ids[0][pool[0].bool()], ids[1][pool[1].bool()]))
        self.assertFalse(torch.equal(ids[0][pool[0].bool()], ids[2][pool[2].bool()]))
        self.assertGreater(int(attention[1].sum()), int(attention[0].sum()))

    def test_boundary_rule_does_not_discard_mixed_operator_tokens(self):
        class MergedTokenizer(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                if text.endswith('++;}'):
                    result['input_ids'] = result['input_ids'][:-4] + [30]
                    if kwargs.get('return_offsets_mapping'):
                        result['offset_mapping'] = result['offset_mapping'][:-4] + [(len(text)-4, len(text))]
                return result
        builder = InputBuilder(MergedTokenizer(), source_max_length=2048, context_max_length=384)
        row = self.row('int f(){ int x=0; return x++;}')
        _, attention, mask = builder.source_function_batch([row], device='cpu', exclude_boundary_token=True)
        self.assertEqual(int(mask[0, int(attention.sum())-2]), 1)
        _, attention, mask = self.builder.source_function_batch(
            [self.row('int f(){return 0;}')], device='cpu', exclude_boundary_token=True)
        self.assertEqual(int(mask[0, int(attention.sum())-2]), 0)

    def test_model_legacy_single_forward_gradients_reload_and_invalid_masks(self):
        JointControlTests.setUpClass(); fixture = JointControlTests(); graph = fixture.batch()
        model = SourceGraphClassifier(HPSource(), [3]*4, hidden_size=8, steps=5, mode='joint',
                                      device='cpu', behavior_sizes=fixture.vocab.sizes(), source_pool='structure')
        ids = torch.tensor([[1,2,3,4]]); attention = torch.ones_like(ids); pooling = torch.tensor([[0,1,1,0]])
        self.assertTrue(torch.equal(model(ids, attention, graph), model(ids, attention, graph, attention)))
        before = model.encoder.calls
        with torch.no_grad(): model.task_modules['source_pool'].score.weight.normal_()
        loss = model(ids, attention, graph, pooling).sum(); loss.backward()
        self.assertEqual(model.encoder.calls, before+1)
        self.assertGreater(float(model.encoder.adapter.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.task_modules['cfg_encoder'].message.weight.grad.abs().sum()), 0)
        clone = copy.deepcopy(model); clone.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(ids, attention, graph, pooling), clone(ids, attention, graph, pooling))
        for bad in [pooling*0, torch.ones(1,3), pooling*2]:
            with self.assertRaises(ValueError): model(ids, attention, graph, bad)
        snapshot = {n: p.detach().clone() for n,p in model.named_parameters()}
        model.eval()
        with torch.no_grad(): model(ids, attention, graph, pooling)
        for n,p in model.named_parameters(): self.assertTrue(torch.equal(snapshot[n], p))

    def test_batch_isolation_and_small_fit(self):
        JointControlTests.setUpClass(); fixture = JointControlTests()
        graphs = collate_graphs([fixture.view]*4,
            [[[0]*4]*len(fixture.view['node_ids'])]*4,
            behavior=[fixture.vocab.encode(fixture.item)]*4)
        model = SourceGraphClassifier(HPSource(), [3]*4, hidden_size=8, steps=5, mode='joint',
            device='cpu', behavior_sizes=fixture.vocab.sizes(), source_pool='structure')
        ids = torch.tensor([[1,2,3,4], [1,5,6,4], [1,2,3,9], [1,5,6,10]])
        mask = torch.ones_like(ids); pooling = torch.tensor([[0,1,1,0]]*4)
        labels = torch.tensor([0.,1.,0.,1.])
        optimizer = torch.optim.Adam(model.parameters(), lr=.02)
        for _ in range(100):
            optimizer.zero_grad()
            logits = model(ids, mask, graphs, pooling)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward(); optimizer.step()
        self.assertLess(float(loss.detach()), .02)
        model.eval()
        combined = model(ids, mask, graphs, pooling)
        separate = torch.cat([model(ids[i:i+1], mask[i:i+1], fixture.batch(), pooling[i:i+1]) for i in range(4)])
        torch.testing.assert_close(combined, separate)
        torch.testing.assert_close(combined[:2], combined[2:])
        self.assertEqual((combined > 0).tolist(), labels.bool().tolist())
