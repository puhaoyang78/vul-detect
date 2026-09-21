import json
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from vulnmechanism.model import InputBuilder, _predict_logits, train_model
from vulnmechanism.progress import print_table


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def __call__(self, text, **kwargs):
        ids = [ord(c) % 14 + 2 for c in text]
        if kwargs.get("max_length"):
            ids = ids[:kwargs["max_length"]]
        return {"input_ids": ids}


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Embedding(16, 4)
        self.task_modules = nn.ModuleDict({"classifier": nn.Linear(4, 1)})
    def forward(self, ids, mask):
        h = self.encoder(ids)
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.task_modules["classifier"](pooled).squeeze(-1)


class TrainingTests(unittest.TestCase):
    def test_accumulation_weights_short_batch_by_sample_count(self):
        records = [dict(sample_key=str(i), dataset="primevul", split="train", label=i%2,
                        raw_source=f"int x{i};") for i in range(5)]
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            stack.enter_context(patch("vulnmechanism.model.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()))
            stack.enter_context(patch("vulnmechanism.model._build_model", side_effect=lambda *a, **kw: TinyClassifier()))
            stack.enter_context(patch("vulnmechanism.model.get_peft_model_state_dict", side_effect=lambda model: model.state_dict()))
            common = dict(records=records, variant="baseline", model_path="tiny", fixed_epochs=True,
                          device="cpu", epochs=1, seed=42,
                          sample_weights={str(i): float(i+1) for i in range(5)})
            a = train_model(None, Path(temp)/"a.pt", batch_size=2, gradient_accumulation=3, **common)
            b = train_model(None, Path(temp)/"b.pt", batch_size=5, gradient_accumulation=1, **common)
            for state in ("adapter_state", "task_state"):
                for key in a[state]:
                    torch.testing.assert_close(a[state][key], b[state][key], atol=1e-6, rtol=1e-6)

    def test_continuation_restores_adapter_and_head(self):
        records = [dict(sample_key=str(i), dataset='primevul', split='train', label=i%2,
                        raw_source=f'int x{i};') for i in range(4)]
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            stack.enter_context(patch('vulnmechanism.model.AutoTokenizer.from_pretrained', return_value=TinyTokenizer()))
            stack.enter_context(patch('vulnmechanism.model._build_model', side_effect=lambda *a, **kw: TinyClassifier()))
            stack.enter_context(patch('vulnmechanism.model.get_peft_model_state_dict', side_effect=lambda model: model.state_dict()))
            setter = stack.enter_context(patch('vulnmechanism.model.set_peft_model_state_dict', side_effect=lambda model, state: model.load_state_dict(state)))
            common = dict(records=records, variant='baseline', model_path='tiny', fixed_epochs=True,
                          device='cpu', epochs=1, seed=42)
            first = Path(temp)/'first.pt'
            a = train_model(None, first, **common)
            with patch('torch.optim.AdamW.step'):
                b = train_model(None, Path(temp)/'next.pt', initial_checkpoint=first, **common)
            setter.assert_called_once()
            for state in ('adapter_state', 'task_state'):
                for key in a[state]:
                    torch.testing.assert_close(a[state][key], b[state][key])
            with self.assertRaisesRegex(ValueError, 'source_max_length'):
                train_model(None, Path(temp)/'bad.pt', initial_checkpoint=first, source_max_length=99, **common)
            with self.assertRaisesRegex(ValueError, 'cover unique'):
                train_model(None, Path(temp)/'bad.pt', sample_weights={'0': 2.0}, **common)

    def test_fixed_epoch_training_progress_and_checkpoint_selection(self):
        records = [dict(sample_key=str(i), dataset="primevul", split="train", label=i%2,
                        raw_source=f"int x{i};") for i in range(5)]
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            stack.enter_context(patch("vulnmechanism.model.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()))
            stack.enter_context(patch("vulnmechanism.model._build_model", side_effect=lambda *a, **kw: TinyClassifier()))
            stack.enter_context(patch("vulnmechanism.model.get_peft_model_state_dict", side_effect=lambda model: model.state_dict()))
            path = Path(temp) / "tiny.pt"
            result = train_model(None, path, records=records, variant="baseline", model_path="tiny",
                                 fixed_epochs=True, device="cpu", epochs=2, batch_size=2,
                                 gradient_accumulation=2, log_every=1)
            self.assertEqual(result["selected_epoch"], 2)
            self.assertEqual(result["selection"], "fixed_epochs")
            self.assertIsNone(result["validation"])
            self.assertTrue(path.exists())
            history = [json.loads(l) for l in path.with_suffix(".training.jsonl").read_text().splitlines()]
            self.assertEqual([r["step"] for r in history if r["event"] == "step"], [1, 2, 3, 4])
            self.assertFalse(path.with_suffix(".training.html").exists())
            with self.assertRaisesRegex(ValueError, "training records only"):
                train_model(None, path, records=records + [dict(records[0], split="valid")],
                            variant="baseline", model_path="tiny", fixed_epochs=True, device="cpu")
            normal = train_model(None, Path(temp)/"normal.pt", records=records+[
                dict(records[i], sample_key=f"v{i}", split="valid") for i in range(2)],
                variant="baseline", model_path="tiny", device="cpu", epochs=1, log_every=1)
            self.assertEqual(normal["selection"], "validation_mcc")
            self.assertIsNotNone(normal["validation"])

    def test_prediction_keeps_raw_extreme_logits(self):
        class Constant(nn.Module):
            def forward(self, ids, mask):
                return torch.full((len(ids),), 40.0)
        builder = InputBuilder(TinyTokenizer(), source_max_length=16, context_max_length=16)
        logits = _predict_logits(Constant(), [{"raw_source": "x"}], builder, variant="baseline",
                                 excluded_groups=(), batch_size=1, device=torch.device("cpu"))
        self.assertEqual(logits.tolist(), [40.0])

    def test_terminal_table_is_printed(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_table("Epoch complete", ["Loss", "MCC"], [[0.4, 0.5]])
        self.assertIn("Epoch complete", buffer.getvalue())
        self.assertIn("0.4", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
