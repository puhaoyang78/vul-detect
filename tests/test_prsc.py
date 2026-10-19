"""Detached pair masks, binary RSC, eligibility and real encoder gradients."""

import tempfile
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from tests.test_codebert_source import CodeTokenizer
from tests.test_surface_vat import tiny_model
from vulnmechanism import model as core
from vulnmechanism.prsc import feature_mask, masked_bce, pair_eligibility, training_masks


class PRSCTests(unittest.TestCase):
    def test_contributions_identity_exact_mask_count_and_detachment(self):
        h = torch.tensor([[1., 2., 3., 4.]], requires_grad=True)
        changed = torch.tensor([[2., 2., 4., -1.]], requires_grad=True)
        head = nn.Linear(4, 1)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[2., 1., -3., 4.]]))
        contributions = head.weight * (changed - h)
        torch.testing.assert_close(head(changed) - head(h), contributions.sum(1, keepdim=True))
        mask = feature_mask("prsc", head.weight, h, changed, torch.tensor([1.]), .25)
        self.assertFalse(mask.requires_grad)
        self.assertEqual(mask.tolist(), [[1., 1., 1., 0.]])
        self.assertEqual(int((mask == 0).sum()), 1)
        # Mask gradients reach encoder representations and linear parameters,
        # but not the detached pair-difference selection computation.
        masked_bce(head, h, h.new_tensor([1.]), mask, h.new_ones(1)).backward()
        self.assertIsNone(changed.grad)
        self.assertEqual(float(h.grad[0, 3]), 0.)
        self.assertGreater(float(h.grad[:, :3].abs().sum()), 0.)
        self.assertGreater(float(head.weight.grad.abs().sum()), 0.)

    def test_original_rsc_signed_true_class_gradient_and_linear_degeneracy(self):
        weight = torch.tensor([[3., -4., 1., -2.]], requires_grad=True)
        h = torch.randn(4, 4, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1])
        z = F.linear(h, weight).squeeze(-1)
        logits = torch.stack([-z / 2, z / 2], 1)
        gradient = torch.autograd.grad(logits[torch.arange(4), labels].sum(), h)[0]
        torch.testing.assert_close(gradient, (2 * labels - 1)[:, None] * weight / 2)
        mask = feature_mask("rsc", weight, h, h + torch.randn_like(h), labels, .25)
        self.assertEqual(mask.tolist(), [[1., 0., 1., 1.], [0., 1., 1., 1.]] * 2)
        torch.testing.assert_close(F.cross_entropy(logits, labels),
                                   F.binary_cross_entropy_with_logits(z, labels.float()))

    def test_random_private_rng_reproducible_and_stable_ties(self):
        h, weight = torch.zeros(2, 20), torch.zeros(1, 20)
        labels = torch.tensor([0, 1])
        state = torch.get_rng_state().clone()
        first = feature_mask("random", weight, h, h, labels, .10, seeds=[42, 43])
        second = feature_mask("random", weight, h, h, labels, .10, seeds=[42, 43])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first[0], first[1]))
        tied = feature_mask("prsc", weight, h, h, labels, .10)
        self.assertEqual(tied[0, :3].tolist(), [0., 0., 1.])

    def test_strict_and_actual_input_gating_dropout_restore_and_reload(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            path, model = tiny_model(Path(tmp))
            builder = core.InputBuilder(CodeTokenizer())
            rows = [dict(sample_key=str(i), raw_source="int f(int x){return x;}", label=i % 2)
                    for i in range(3)]
            selected = [dict(raw_source="int f(int y){return y;}", strict_semantic=True),
                        dict(raw_source="int g(int x){return x;}", strict_semantic=False), None]
            self.assertEqual(pair_eligibility(rows[0], dict(selected[0], raw_source=rows[0]["raw_source"]),
                                              builder), "identical_encoder_input")
            long = dict(rows[0], raw_source="x" * 2500)
            self.assertEqual(pair_eligibility(long, dict(selected[0], raw_source=long["raw_source"] + "/* hi */"),
                                              builder), "identical_encoder_input")
            model.train()
            flags = [m.training for m in model.modules()]
            state = torch.get_rng_state().clone()
            reference_inputs = builder.codebert_batch([rows[0], dict(rows[0], **selected[0])], device="cpu")
            with torch.no_grad(), core.auxiliary_mode(model):
                _, h = model(*reference_inputs, return_representation=True)
            references = {(rows[0]["sample_key"], rows[0]["raw_source"]): h[0],
                          (rows[0]["sample_key"], selected[0]["raw_source"]): h[1]}
            masks, reasons, costs = training_masks(
                model, rows, [dict(selected=s) for s in selected], builder, device="cpu",
                method="prsc", fraction=.10, seed=42, epoch=1,
                references=references,
            )
            self.assertEqual(set(masks), {"0"})
            self.assertEqual(reasons, dict(eligible=1, input_only_stress=1, no_legal_view=1))
            self.assertEqual(costs.get("function_forwards", 0), 0)
            self.assertEqual(costs["reused_search_representations"], 2)
            self.assertEqual([m.training for m in model.modules()], flags)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            inputs = builder.codebert_batch(rows[:1] + [dict(rows[0], **selected[0])], device="cpu")
            with core.auxiliary_mode(model):
                logits, h = model(*inputs, return_representation=True)
            labels = logits.new_zeros(2)
            base = F.binary_cross_entropy_with_logits(logits, labels)
            extra = masked_bce(model.task_modules["classifier"], h, labels,
                               masks["0"].expand(2, -1), h.new_full((2,), .5))
            (base + .2 * extra).backward()
            for parameter in (model.encoder.embeddings.word_embeddings.weight,
                              model.task_modules["classifier"].weight):
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.norm()), 0.)
            model.eval()
            with torch.no_grad():
                before = model(*inputs)
                reported, h = model(*inputs, return_representation=True)
            torch.testing.assert_close(before, reported, rtol=0, atol=0)
            cp = dict(model_path=str(path), state_dict=model.state_dict())
            target = Path(tmp) / "check.pt"
            torch.save(cp, target)
            loaded = core.load_model(torch.load(target, weights_only=False), "cpu")
            with torch.no_grad():
                after = loaded(*inputs)
            torch.testing.assert_close(before, after, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
