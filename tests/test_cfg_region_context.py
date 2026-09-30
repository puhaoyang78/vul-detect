"""Node-preserving region residuals; no Qwen training required."""
import copy
import unittest

import torch
from vulnmechanism.cfg_network import AttributeCFGEncoder, collate_graphs
from vulnmechanism.cfg_data import cfg_regions


class RegionContextTests(unittest.TestCase):
    def setUp(self):
        self.view = {"node_ids": list('abcdef'), "edges": [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (5, 5)]}
        self.values = [[0, 0, 0, 0], [2, 3, 2, 3], [3, 2, 3, 2], [1]*4, [2]*4, [0]*4]
        self.partition = cfg_regions(self.view)
        self.batch = collate_graphs([self.view], [self.values], regions=[self.partition])

    def model(self, mode):
        torch.manual_seed(42)
        return AttributeCFGEncoder([4]*4, hidden_size=8, mode=mode).double()

    def test_zero_projection_exact_c_and_common_gradients_rng(self):
        c = self.model('cfg')
        after = torch.rand(7)
        for mode in ('region_local', 'region_context'):
            m = self.model(mode)
            self.assertTrue(torch.equal(after, torch.rand(7)))
            self.assertEqual(sum(p.numel() for p in m.parameters())-sum(p.numel() for p in c.parameters()), 64)
            for key, value in c.state_dict().items():
                self.assertTrue(torch.equal(value, m.state_dict()[key]))
            self.assertEqual(torch.count_nonzero(m.region_projection.weight).item(), 0)
            self.assertTrue(torch.equal(c(self.batch), m(self.batch)))
            c.zero_grad(); m.zero_grad()
            c(self.batch).square().sum().backward(); m(self.batch).square().sum().backward()
            for name, parameter in c.named_parameters():
                torch.testing.assert_close(parameter.grad, dict(m.named_parameters())[name].grad, rtol=0, atol=0)
            self.assertGreater(m.region_projection.weight.grad.abs().sum().item(), 0)
            with self.assertRaises(RuntimeError):
                m.load_state_dict(self.model('cfg_hierarchical').state_dict())

    def test_classifier_common_initialization_single_forward_and_bce(self):
        from tests.test_cfg_regions import HPSource
        from vulnmechanism.cfg_network import SourceGraphClassifier
        models, rng = [], []
        for mode in ('cfg', 'region_local', 'region_context'):
            torch.manual_seed(42)
            model = SourceGraphClassifier(HPSource(), [4]*4, hidden_size=8,
                steps=5, mode=mode, device=torch.device('cpu'))
            models.append(model)
            rng.append(torch.rand(9))
        for model, random_after in zip(models[1:], rng[1:]):
            self.assertTrue(torch.equal(rng[0], random_after))
            for name, value in models[0].state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]))
            with torch.no_grad():
                model.task_modules['cfg_classifier'].weight.fill_(.1)
                model.task_modules['cfg_encoder'].region_projection.weight.copy_(torch.eye(8)*.2)
            ids = torch.tensor([[2, 3, 4]])
            logits = model(ids, torch.ones_like(ids), self.batch)
            self.assertEqual(model.encoder.calls, 1)
            torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
            self.assertGreater(model.encoder.adapter.weight.grad.abs().sum().item(), 0)
            for parameter in model.task_modules['cfg_encoder'].parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_manual_pool_delta_broadcast_and_gradient_paths(self):
        m = self.model('region_context')
        with torch.no_grad():
            m.region_projection.weight.copy_(torch.eye(8)*.3)
        actual = m(self.batch)
        x = torch.cat([e(self.batch.attributes[:, i]) for i, e in enumerate(m.embedding)], -1)
        x.retain_grad()
        def updates(state, edges, steps):
            for _ in range(steps):
                messages = m.message(state)
                incoming = torch.stack([messages[j] + sum((messages[a] for a, b in edges if b == j and a != b),
                                                          torch.zeros_like(messages[j])) for j in range(len(state))])
                state = m.update(incoming, state)
            return state
        h = updates(x, self.view['edges'], 5); h.retain_grad()
        weights = m.pool_gate(torch.cat([h, x], -1)).squeeze(-1)
        r0 = torch.stack([(weights[ns].softmax(0).unsqueeze(-1)*h[ns]).sum(0) for ns in self.partition['members']])
        r0.retain_grad()
        r3 = updates(r0, self.partition['edges'], 3); r3.retain_grad()
        context = (r3-r0)[self.partition['node_to_region']]; context.retain_grad()
        enhanced = h+m.region_projection(context)
        combined = torch.cat([enhanced, x], -1)
        expected = (m.pool_gate(combined).squeeze(-1).softmax(0).unsqueeze(-1)*combined).sum(0).unsqueeze(0)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        actual.square().sum().backward()
        gradients = {n: p.grad.clone() for n, p in m.named_parameters()}
        m.zero_grad()
        expected.square().sum().backward()
        for tensor in (x, h, r0, r3, context):
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        for name, p in m.named_parameters():
            torch.testing.assert_close(p.grad, gradients[name], atol=1e-12, rtol=1e-12)
        self.assertEqual(combined.shape[0], len(self.values))
        self.assertTrue(torch.equal(combined[:, 8:], x))

    def test_local_context_without_cross_edges_self_loop_and_parameters(self):
        local, context = self.model('region_local'), self.model('region_context')
        with torch.no_grad():
            local.region_projection.weight.copy_(torch.eye(8))
        context.load_state_dict(local.state_dict())
        partition = copy.deepcopy(self.partition); partition['edges'] = []
        batch = collate_graphs([self.view], [self.values], regions=[partition])
        torch.testing.assert_close(context(batch), local(batch), rtol=0, atol=0)
        self.assertFalse(torch.allclose(context(self.batch), local(self.batch)))
        partition['edges'] = [[i, i] for i in range(len(partition['members']))]
        loops = collate_graphs([self.view], [self.values], regions=[partition])
        torch.testing.assert_close(context(loops), context(batch), rtol=0, atol=0)
        self.assertEqual(context.state_dict().keys(), local.state_dict().keys())

    def test_node_region_permutations_and_batch_isolation(self):
        for mode in ('region_local', 'region_context'):
            m = self.model(mode)
            with torch.no_grad():
                m.region_projection.weight.copy_(torch.eye(8)*.2)
            output = m(self.batch)
            order = [4, 2, 0, 5, 1, 3]; inverse = {old: new for new, old in enumerate(order)}
            view = {'node_ids': [self.view['node_ids'][i] for i in order],
                    'edges': [(inverse[a], inverse[b]) for a, b in self.view['edges']]}
            partition = copy.deepcopy(self.partition)
            region_order = list(reversed(range(len(partition['members']))))
            region_inverse = {old: new for new, old in enumerate(region_order)}
            partition['members'] = [[inverse[n] for n in partition['members'][i]] for i in region_order]
            partition['edges'] = [[region_inverse[a], region_inverse[b]] for a, b in partition['edges']]
            permuted = collate_graphs([view], [[self.values[i] for i in order]], regions=[partition])
            torch.testing.assert_close(m(permuted), output, atol=1e-12, rtol=1e-12)
            other = [[3]*4]*6
            joined = collate_graphs([self.view]*2, [self.values, other], regions=[self.partition]*2)
            single_other = collate_graphs([self.view], [other], regions=[self.partition])
            torch.testing.assert_close(m(joined), torch.cat([output, m(single_other)]), atol=1e-12, rtol=1e-12)
            m.zero_grad(); m(joined)[0].sum().backward()
            joint_grad = {n:p.grad.clone() for n,p in m.named_parameters()}
            m.zero_grad(); output.sum().backward()
            for n,p in m.named_parameters():
                torch.testing.assert_close(joint_grad[n], p.grad, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
