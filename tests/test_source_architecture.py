import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from tests.test_training_progress import TinyTokenizer
from vulnmechanism.model import (
    BidirectionalSourceAdapter, InputBuilder, SourceAttentionPool,
    SequenceVulnerabilityClassifier, _load_model, _validate_variant, train_model,
)


VARIANTS = ('baseline', 'source_attention', 'source_bidirectional',
            'source_bidirectional_attention')


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 12)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def tiny_encoder(*args, **kwargs):
    return TinyEncoder(), 12


class SourceArchitectureTests(unittest.TestCase):
    def test_identical_source_inputs_without_reading_cpg(self):
        builder = InputBuilder(TinyTokenizer(), source_max_length=8, context_max_length=8)
        row = {'raw_source': 'int f() { return 0; }'}
        reference = builder.sequence_ids(row, variant='baseline', excluded_groups=())
        for variant in VARIANTS:
            self.assertEqual(builder.sequence_ids(row, variant=variant, excluded_groups=()), reference)
            with self.assertRaisesRegex(ValueError, 'only valid for mechanism'):
                _validate_variant(variant, ('mechanism',))

    def test_pool_ignores_padding_and_has_trainable_queries(self):
        torch.manual_seed(1)
        pool = SourceAttentionPool(12)
        h = torch.randn(2, 5, 12, requires_grad=True)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
        result = pool(h, mask)
        torch.testing.assert_close(result[:1], pool(h[:1, :2], mask[:1, :2]))
        result.square().sum().backward()
        self.assertGreater(pool.queries.grad.abs().sum().item(), 0)
        self.assertEqual(h.grad[0, 2:].abs().sum().item(), 0)
        with torch.no_grad():
            pool.queries.zero_()
        torch.testing.assert_close(pool(h, mask)[0], h[0, :2].mean(0))

    def test_adapter_identity_then_future_context_and_padding(self):
        torch.manual_seed(2)
        adapter = BidirectionalSourceAdapter(12).eval()
        h = torch.randn(1, 4, 12, requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 0]])
        torch.testing.assert_close(adapter(h, mask), h)
        with torch.no_grad():
            nn.init.normal_(adapter.up.weight, std=0.02)
        result = adapter(h, mask)
        torch.testing.assert_close(result[:, :3], adapter(h[:, :3], mask[:, :3]), atol=1e-6, rtol=1e-5)
        result[0, 0].square().sum().backward()
        self.assertGreater(h.grad[0, 2].abs().sum().item(), 0)
        self.assertEqual(h.grad[0, 3].abs().sum().item(), 0)

    @patch('vulnmechanism.model._build_lora_encoder', side_effect=tiny_encoder)
    def test_common_initialization_and_rng_are_preserved(self, _):
        states = []
        for variant in VARIANTS:
            torch.manual_seed(42)
            model = SequenceVulnerabilityClassifier('tiny', device=torch.device('cpu'),
                lora_r=2, lora_alpha=4, lora_dropout=0.0, target_modules=(),
                gradient_checkpointing=False, variant=variant)
            states.append((model, torch.get_rng_state()))
        for model, rng in states[1:]:
            torch.testing.assert_close(rng, states[0][1])
            torch.testing.assert_close(model.encoder.embedding.weight, states[0][0].encoder.embedding.weight)
            torch.testing.assert_close(model.task_modules['classifier'].weight,
                                       states[0][0].task_modules['classifier'].weight)
        for index, name in [(1, 'source_pool'), (2, 'source_adapter')]:
            for key, value in states[index][0].task_modules[name].state_dict().items():
                torch.testing.assert_close(value, states[3][0].task_modules[name].state_dict()[key])

    @patch('vulnmechanism.model.AutoTokenizer.from_pretrained', return_value=TinyTokenizer())
    @patch('vulnmechanism.model._build_lora_encoder', side_effect=tiny_encoder)
    @patch('vulnmechanism.model.get_peft_model_state_dict', side_effect=lambda model: model.state_dict())
    @patch('vulnmechanism.model.set_peft_model_state_dict', side_effect=lambda model, state: model.load_state_dict(state))
    def test_train_and_reload_every_variant(self, *_):
        rows = [dict(sample_key=str(i), dataset='primevul', split='train',
                     label=i % 2, raw_source=f'int x{i};') for i in range(4)]
        with tempfile.TemporaryDirectory() as temp:
            for variant in VARIANTS:
                path = Path(temp) / f'{variant}.pt'
                saved = train_model(None, path, records=rows, variant=variant, model_path='tiny',
                    device='cpu', epochs=1, batch_size=2, gradient_accumulation=1,
                    source_max_length=16, fixed_epochs=True)
                model = _load_model(saved, device=torch.device('cpu'))
                restored = _load_model(torch.load(path, weights_only=False), device=torch.device('cpu'))
                ids = torch.tensor([[2, 3, 1], [4, 1, 0]])
                mask = (ids != 0).long()
                torch.testing.assert_close(model(ids, mask), restored(ids, mask))
                self.assertTrue(torch.isfinite(restored(ids, mask)).all())
                if 'source_adapter' in model.task_modules:
                    self.assertGreater(model.task_modules['source_adapter'].up.weight.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
