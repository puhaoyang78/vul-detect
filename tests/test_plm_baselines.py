"""CodeT5 defect readout and shared three-seed baseline protocol."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers import T5Config, T5Model

from tests.test_losver import Tokens
from tests.test_rgcl import record
from vulnmechanism.data import paired_features, write_jsonl
from vulnmechanism.evaluation import load_paired
from vulnmechanism.model import LOSVERDetector, losver_model, predict_checkpoint
from vulnmechanism.training import LOSVER_CONFIG, paired_collate, paired_forward, train_losver
from vulnmechanism import rgcl_study


class T5Tokens(Tokens):
    cls_token_id, pad_token_id, sep_token_id, unk_token_id = 1, 0, 2, 3
    all_special_ids = [0, 1, 2, 3]


def factory(path, variant, **kwargs):
    c = T5Config(vocab_size=128, d_model=16, d_ff=32, d_kv=8, num_layers=1,
                 num_decoder_layers=1, num_heads=2, pad_token_id=0,
                 eos_token_id=2, decoder_start_token_id=0, dropout_rate=.1)
    args = SimpleNamespace(class_weight=kwargs.get("class_weight", 1.), window_micro_batch=8,
        cls_baseline=kwargs.get("cls_baseline", True), bounded_attention=kwargs.get("bounded_attention", False),
        attention_context=kwargs.get("attention_context", False), attention_region=kwargs.get("attention_region"))
    return LOSVERDetector(T5Model(c), c, args), T5Tokens()


class PLMBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_t5_official_decoder_eos_readout_weighted_ce_gradient_and_padding(self):
        torch.manual_seed(42)
        model, tokens = factory(None, "baseline", class_weight=2.)
        # Literal EOS in source and truncated source still use the final,
        # appended EOS. Padding and extra EOS counts must not change readout.
        rows = [record(0, 1, code="int f(){return 1;}"),
                record(1, 0, code="int g(){\n" + "x;\n" * 300 + "}")]
        features = [paired_features(r, tokens, "baseline") for r in rows]
        batch = paired_collate(features)
        batch["input_ids"][0, 3] = model.config.eos_token_id
        model.eval()
        loss, probability = paired_forward(model, batch, "baseline", "cpu")
        ids, mask = batch["input_ids"], batch["attention_mask"]
        output = model.encoder(input_ids=ids, attention_mask=mask,
            decoder_input_ids=model.encoder._shift_right(ids), decoder_attention_mask=mask,
            use_cache=False, output_hidden_states=True)
        last = mask.sum(-1) - 1
        states = output.decoder_hidden_states[-1][torch.arange(2), last]
        logits = model.classifier(states)
        expected = F.cross_entropy(logits, batch["labels"], reduction="none")
        expected = (expected * torch.tensor([2., 1.])).mean()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(probability[:, 0], logits.softmax(-1)[:, 1])
        changed = {**batch, "input_ids": ids.clone(), "token_weights": torch.randn_like(batch["token_weights"])}
        changed["input_ids"][~mask.bool()] = 91
        torch.testing.assert_close(paired_forward(model, changed, "baseline", "cpu")[1], probability,
                                   rtol=0, atol=0)
        loss.backward()
        for parameters in (model.encoder.encoder.parameters(), model.encoder.decoder.parameters(), model.classifier.parameters()):
            gradients = [p.grad for p in parameters if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        with self.assertRaisesRegex(ValueError, "visible source EOS"):
            paired_forward(model, {**batch, "input_ids": ids.masked_fill(ids == 2, 3)}, "baseline", "cpu")

    def test_t5_strict_checkpoint_single_function_and_diff_independent_inference(self):
        model, tokens = factory(None, "baseline")
        model.eval()
        row = record(0, 1)
        features = [paired_features(row, tokens, "baseline")]
        batch = paired_collate(features)
        expected = paired_forward(model, batch, "baseline", "cpu", with_labels=False)
        changed = {**batch, "line_labels": 1 - batch["line_labels"], "labels": 1 - batch["labels"]}
        torch.testing.assert_close(expected, paired_forward(model, changed, "baseline", "cpu", with_labels=False), rtol=0, atol=0)
        cp = dict(protocol="rgcl_native_binary", model_path="tiny", variant="baseline",
            line_mapping="source_offsets", input_mode="512", native_binary=True,
            cls_baseline=True, pooling="hard_weight", training_config=dict(class_weight=1.),
            classification_architecture="codet5_defect", state_dict=model.state_dict())
        with tempfile.TemporaryDirectory() as tmp, patch("vulnmechanism.model.losver_model", side_effect=factory):
            path = Path(tmp)/"model.pt"
            torch.save(cp, path)
            restored, _ = load_paired(cp, "cpu")
            torch.testing.assert_close(expected, paired_forward(restored, batch, "baseline", "cpu", with_labels=False), rtol=0, atol=0)
            probability = predict_checkpoint(path, [dict(raw_source=row["func"])], device="cpu").sigmoid()
            torch.testing.assert_close(probability, expected[:, 0])

    def test_t5_uses_existing_full_epoch_trainer_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp, patch("vulnmechanism.model.losver_model", side_effect=factory):
            root = Path(tmp)
            data = root/"data/secvuleval"
            train = [record(i, i % 2) for i in range(4)]
            write_jsonl(data/"train.lines.jsonl", train)
            write_jsonl(data/"valid.jsonl", [{k:v for k,v in r.items() if k not in ("lines_ground_truth", "diff_reliable")} for r in train])
            (data/"statistics.json").write_text(json.dumps(dict(classification_positive_weight=1.)))
            cfg = dict(LOSVER_CONFIG, epochs=dict(LOSVER_CONFIG["epochs"], baseline=2), batch_size=2, eval_batch_size=2)
            original_save = __import__("vulnmechanism.training", fromlist=["save_training_state"]).save_training_state
            def interrupt(run, model, optimizer, metadata, loop, **kwargs):
                original_save(run, model, optimizer, metadata, loop, **kwargs)
                if loop["global_step"] == 1:
                    raise RuntimeError("intentional interruption")
            args = (root/"data", root/"out", "tiny", "plm_baseline")
            with patch("vulnmechanism.training.save_training_state", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "intentional interruption"):
                    train_losver(*args, dataset="secvuleval", config=cfg, seed=42, device="cpu", save_every=1)
            result = train_losver(*args, dataset="secvuleval", config=cfg, seed=42, device="cpu", save_every=1, resume=True)
            reference = train_losver(root/"data", root/"reference", "tiny", "plm_baseline", dataset="secvuleval", config=cfg, seed=42, device="cpu", save_every=1)
            self.assertEqual(result["completed_epochs"], 2)
            self.assertEqual(result["optimizer_steps"], 4)
            self.assertEqual(result["selected_epochs"], reference["selected_epochs"])
            a = torch.load(root/"out/secvuleval/plm_baseline/last.pt", weights_only=False)
            b = torch.load(root/"reference/secvuleval/plm_baseline/last.pt", weights_only=False)
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
            torch.testing.assert_close(a["rng"]["torch"], b["rng"]["torch"], rtol=0, atol=0)

    def test_backbone_factory_and_study_paths_preserve_history(self):
        c = T5Config(d_model=16)
        with patch("vulnmechanism.model.AutoConfig.from_pretrained", return_value=c), \
             patch("vulnmechanism.model.RobertaTokenizerFast.from_pretrained", return_value=T5Tokens()):
            with self.assertRaisesRegex(ValueError, "requires the standard 512"):
                losver_model("tiny", "baseline")
        for backbone in ("codebert", "codet5"):
            self.assertEqual(rgcl_study.study_methods(backbone), ("plm_baseline", "context_only", "rgcl"))
            self.assertEqual(rgcl_study.run_path("out", "secvuleval", 123456, "plm_baseline", backbone=backbone),
                             Path("out/seed123456/secvuleval/plm_baseline"))
            with tempfile.TemporaryDirectory() as tmp:
                inventory = rgcl_study.inventory("data", tmp, "tiny", datasets=rgcl_study.SUPPORTED_DATASETS, backbone=backbone)
                self.assertEqual(inventory["missing"], 27)
                self.assertEqual(inventory["reused"], 0)

    def test_t5_context_rgcl_share_decoder_states_initialization_and_sampler_rng(self):
        rows = [record(0, 1), record(1, 0)]
        initial, rng = None, None
        for method in ("plm_baseline", "context_only", "rgcl"):
            torch.manual_seed(123456)
            model, tokens = factory(None, "baseline", cls_baseline=method == "plm_baseline",
                bounded_attention=method != "plm_baseline", attention_context=method != "plm_baseline",
                attention_region="diff" if method == "rgcl" else None)
            if rng is None:
                rng = torch.get_rng_state()
            else:
                torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
            if method == "plm_baseline":
                continue
            batch = paired_collate([paired_features(r, tokens, "baseline") for r in rows])
            model.eval()
            calls = []
            hook = model.encoder.register_forward_hook(lambda *args: calls.append(1))
            loss, probability, details = paired_forward(model, batch, "baseline", "cpu",
                                                       return_attention=True, return_line_states=True)
            hook.remove()
            self.assertEqual(len(calls), 1)
            with torch.no_grad():
                decoded = model.encoder(batch["input_ids"], attention_mask=batch["attention_mask"],
                    decoder_input_ids=model.encoder._shift_right(batch["input_ids"]),
                    decoder_attention_mask=batch["attention_mask"], use_cache=False)[0]
                pooled, direct = model.bounded_pool(decoded, batch["code_mask"], batch["line_token_mask"], return_line_states=True)
            torch.testing.assert_close(details["line_states"], direct["line_states"])
            torch.testing.assert_close(probability, model.classifier(pooled).sigmoid())
            if initial is None:
                initial = probability.detach()
            else:
                torch.testing.assert_close(initial, probability, rtol=0, atol=0)
                torch.testing.assert_close(loss, details["bce"] + .1 * details["region_loss"])
                loss.backward()
                self.assertGreater(float(model.line_attention.weight.grad.abs().sum()), 0)
                self.assertGreater(float(model.context_v.weight.grad.abs().sum()), 0)
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.encoder.decoder.parameters() if p.grad is not None), 0)


if __name__ == "__main__":
    unittest.main()
