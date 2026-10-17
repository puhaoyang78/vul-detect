"""Function pooling, fresh six-method training, VAT gradients and strict reload."""

import io
from contextlib import redirect_stdout
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import RobertaConfig, RobertaModel

from vulnmechanism import model as core
from vulnmechanism.data import write_jsonl
from vulnmechanism.training import METHODS, TRAINING, train
from vulnmechanism.vat import vat_loss


class CodeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    all_special_ids = [0, 1, 2]
    is_fast = True

    def __call__(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
        max_length=None,
        return_offsets_mapping=False,
        **kwargs,
    ):
        if text == "FORBIDDEN_HELDOUT":
            raise AssertionError("Test reached training or model selection")
        ids = [2 + ord(char) % 14 for char in text]
        if truncation:
            ids = ids[:max_length]
        result = dict(input_ids=ids)
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(ids))]
        return result


def tiny_base(path):
    model = RobertaModel(
        RobertaConfig(
            vocab_size=16,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=514,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=1,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        ),
        add_pooling_layer=False,
    )
    model.save_pretrained(path)
    return model


class CodeBERTTests(unittest.TestCase):
    def test_csd_shared_gate_changes_only_specific_common_term(self):
        torch.manual_seed(42)
        baseline = core.CSD(3, domains=2)
        torch.manual_seed(42)
        head = core.CSD(3, domains=2, shared_gate_init=.1)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(head.state_dict()[name], value, rtol=0, atol=0)
        self.assertAlmostEqual(float(head.shared_gate.detach()), .1, places=7)
        with torch.no_grad():
            head.common.weight.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
            head.common.bias.copy_(torch.tensor([.3, -.2]))
            head.specific_weight.copy_(torch.tensor([[[0., 1., 0.], [1., 0., 0.]]]))
            head.specific_bias.copy_(torch.tensor([[.1, -.05]]))
            head.coefficients.weight.copy_(torch.tensor([[.2], [-.3]]))
        features = torch.tensor([[1., 2., 3.], [-1., 1., 2.]], requires_grad=True)
        domains = torch.tensor([0, 1])
        common, specific = head(features, domains)
        full_common = head.common(features)
        residual = torch.stack([
            head.coefficients.weight[d, 0] * (features[i] @ head.specific_weight[0].T + head.specific_bias[0])
            for i, d in enumerate(domains)
        ])
        expected = head.shared_gate.tanh() * full_common + residual
        torch.testing.assert_close(common, full_common[:, 1] - full_common[:, 0])
        torch.testing.assert_close(specific, expected[:, 1] - expected[:, 0])
        self.assertIsNone(torch.autograd.grad(head.orthonormality(), head.shared_gate, allow_unused=True)[0])
        torch.nn.functional.binary_cross_entropy_with_logits(specific, torch.tensor([0., 1.])).backward()
        self.assertGreater(float(head.shared_gate.grad.abs()), 0.)
        self.assertGreater(float(head.coefficients.weight.grad.norm()), 0.)
        before = head(features).detach()
        regularizer = head.orthonormality().detach()
        with torch.no_grad():
            head.shared_gate.fill_(-2.)
        torch.testing.assert_close(head(features), before, rtol=0, atol=0)
        torch.testing.assert_close(head.orthonormality(), regularizer, rtol=0, atol=0)

    def test_csd_decomposition_cross_entropy_and_orthonormality(self):
        head = core.CSD(3, domains=2)
        with torch.no_grad():
            head.common.weight.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
            head.common.bias.zero_()
            head.specific_weight.copy_(torch.tensor([[[0., 1., 0.], [1., 0., 0.]]]))
            head.specific_bias.zero_()
            head.coefficients.weight.copy_(torch.tensor([[.2], [-.3]]))
        features = torch.tensor([[1., 2., 3.], [-1., 1., 2.]], requires_grad=True)
        domains, labels = torch.tensor([0, 1]), torch.tensor([0, 1])
        common, specific = head(features, domains)
        full_common = features @ head.common.weight.T
        full_specific = torch.stack([
            features[i] @ (head.common.weight + head.coefficients.weight[d, 0] * head.specific_weight[0]).T
            for i, d in enumerate(domains)
        ])
        torch.testing.assert_close(common, full_common[:, 1] - full_common[:, 0])
        torch.testing.assert_close(specific, full_specific[:, 1] - full_specific[:, 0])
        self.assertEqual(float(head.orthonormality().detach()), 0.)
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        ce = torch.nn.functional.cross_entropy
        torch.testing.assert_close(bce(specific, labels.float()), ce(full_specific, labels))
        with torch.no_grad():
            head.common.weight[0, 0] = 1.1
        expected = sum(
            ((torch.stack((head.common.weight[y], head.specific_weight[0, y])) @
              torch.stack((head.common.weight[y], head.specific_weight[0, y])).T - torch.eye(2)) ** 2).sum()
            for y in range(2)
        )
        torch.testing.assert_close(head.orthonormality(), expected / 8)
        actual_gradient = torch.autograd.grad(head.orthonormality(), head.common.weight, retain_graph=True)[0]
        sum_gradient = torch.autograd.grad(expected, head.common.weight, retain_graph=True)[0]
        torch.testing.assert_close(actual_gradient, sum_gradient / 8)
        common, specific = head(features, domains)
        loss = bce(common, labels.float()) + bce(specific, labels.float()) + head.orthonormality()
        loss.backward()
        for parameter in head.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertGreater(float(features.grad.abs().sum()), 0.)

    def test_csd_trains_both_sources_and_infers_common_head_only(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = tiny_base(root / "encoder")
            for dataset in ("primevul", "secvuleval"):
                rows = [dict(sample_key=str(i), dataset=dataset, split="train", label=i % 2,
                             project=f"p{i // 2}", raw_source=f"int f{i}(){{return {i};}}") for i in range(4)]
                write_jsonl(root / "data" / dataset / "train.jsonl", rows)
                write_jsonl(root / "data" / dataset / "valid.jsonl", [
                    dict(rows[i], sample_key=f"v{i}", split="valid", project="VALID_ONLY") for i in (0, 1)
                ])
                write_jsonl(root / "data" / dataset / "test.jsonl", [dict(raw_source="FORBIDDEN_HELDOUT")])
                with patch.object(core.AutoTokenizer, "from_pretrained", return_value=CodeTokenizer()), redirect_stdout(io.StringIO()):
                    summary = train(root / "data", root / "runs", str(root / "encoder"), dataset,
                                    "csd", device="cpu", config=dict(TRAINING, batch_size=2))
                    self.assertEqual(summary["completed_epochs"], 3)
                    self.assertEqual(summary["optimizer_steps"], 3)
                    self.assertEqual(summary["csd"]["projects"], ["p0", "p1"])
                    self.assertEqual(summary["csd"]["orthonormality_reduction"], "mean")
                    self.assertEqual(summary["csd"]["shared_gate_init"], .1)
                    self.assertIsNone(summary["augmentation"])
                    self.assertIsNone(summary["vat"])
                    target = root / "runs" / dataset / "csd/best.pt"
                    cp = torch.load(target, map_location="cpu", weights_only=False)
                    self.assertFalse(torch.equal(cp["state_dict"]["encoder.embeddings.word_embeddings.weight"],
                                                 base.embeddings.word_embeddings.weight))
                    model = core.load_model(cp, torch.device("cpu"))
                    builder = core.InputBuilder(CodeTokenizer())
                    expected = core.predict_logits(model, rows, builder, device="cpu")
                    actual = core.predict_checkpoint(target, rows, device="cpu")
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    head = model.task_modules["classifier"]
                    with torch.no_grad():
                        head.specific_weight.fill_(100.)
                        head.specific_bias.fill_(100.)
                        head.coefficients.weight.fill_(100.)
                        head.shared_gate.fill_(-100.)
                    torch.testing.assert_close(core.predict_logits(model, rows, builder, device="cpu"),
                                               expected, rtol=0, atol=0)
                    model.train()
                    outputs = model(*builder.codebert_batch(rows, device="cpu"), project_ids=torch.tensor([0, 0, 1, 1]))
                    (outputs[0].square().mean() + outputs[1].square().mean()).backward()
                    self.assertGreater(float(model.encoder.embeddings.word_embeddings.weight.grad.norm()), 0.)
                    records = [json.loads(line) for line in target.with_suffix(".training.jsonl").read_text().splitlines()]
                    epochs = [r for r in records if r["event"] == "epoch"]
                    self.assertEqual(len(epochs), 3)
                    self.assertTrue(all(set(r["csd_losses"]) == {"common_loss", "specific_loss", "orthonormality"} for r in epochs))
                    self.assertTrue(all(set(r["csd_gate"]) == {"parameter", "scale"} for r in epochs))
                    self.assertNotEqual(epochs[-1]["csd_gate"]["parameter"], float(torch.tensor(.1)))

    def test_window_coverage_and_function_normalization(self):
        builder = core.InputBuilder(CodeTokenizer())
        records = [dict(raw_source="a" * 31), dict(raw_source="abcd" * 800)]
        ids, mask, owners, weights, count = builder.codebert_batch(records, device="cpu")
        self.assertLessEqual(ids.shape[1], 512)
        self.assertEqual(count, 2)
        self.assertEqual(int((owners == 0).sum()), 1)
        self.assertGreater(int((owners == 1).sum()), 1)
        self.assertTrue(torch.all(weights[mask == 0] == 0))
        embedding = nn.Embedding(16, 12)
        pooled = core._window_mean(embedding(ids), mask, owners, weights, count)
        expected = []
        for row in records:
            source = builder._encode(row["raw_source"], 2048 - len(builder.source_prefix) - 2)
            direct = torch.tensor([[2, *builder.source_prefix, *source, 1]])
            expected.append(embedding(direct).mean(1)[0])
        torch.testing.assert_close(pooled, torch.stack(expected), rtol=1e-5, atol=1e-6)
        pooled.square().sum().backward()
        self.assertGreater(float(embedding.weight.grad.abs().sum()), 0.0)
        with self.assertRaisesRegex(ValueError, "every function"):
            core._window_mean(
                torch.ones(1, 3, 2), torch.ones(1, 3), torch.tensor([0]), torch.ones(1, 3), 2
            )

    def test_twelve_fresh_paths_updates_and_strict_save_load(self):
        from tests.test_surface_vat import MATERIALS

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = tiny_base(root / "encoder")
            for dataset in ("primevul", "megavul"):
                rows = [
                    dict(
                        sample_key=str(i),
                        dataset=dataset,
                        split="train",
                        label=i % 2,
                        language="c",
                        raw_source=f"int f{i}(int count){{/* ordinary comment */return api(count)+{i};}}",
                    )
                    for i in range(8)
                ]
                rows[0]["raw_source"] = "return external_api(1)+GLOBAL;"
                write_jsonl(root / "data" / dataset / "train.jsonl", rows)
                write_jsonl(
                    root / "data" / dataset / "valid.jsonl",
                    [dict(rows[i], sample_key=f"v{i}", split="valid") for i in (1, 2)],
                )
                write_jsonl(
                    root / "data" / dataset / "test.jsonl",
                    [dict(rows[0], split="test", raw_source="FORBIDDEN_HELDOUT")],
                )
                config = dict(TRAINING, epochs=1, batch_size=2, gradient_accumulation=4)
                for method in METHODS:
                    with self.subTest(dataset=dataset, method=method), patch.object(
                        core.AutoTokenizer, "from_pretrained", return_value=CodeTokenizer()
                    ), redirect_stdout(io.StringIO()):
                        summary = train(
                            root / "data",
                            root / "runs",
                            str(root / "encoder"),
                            dataset,
                            method,
                            device="cpu",
                            materials=MATERIALS,
                            config=config,
                        )
                        self.assertEqual(summary["optimizer_steps"], 1)
                        target = root / "runs" / dataset / method / "best.pt"
                        cp = torch.load(target, map_location="cpu", weights_only=False)
                        self.assertEqual(cp["training_config"], config)
                        self.assertEqual(cp["dataset"], dataset)
                        self.assertTrue(cp["selection"].startswith(dataset))
                        self.assertFalse(
                            torch.equal(
                                cp["state_dict"]["encoder.embeddings.word_embeddings.weight"],
                                base.embeddings.word_embeddings.weight,
                            )
                        )
                        model = core.load_model(cp, torch.device("cpu"))
                        builder = core.InputBuilder(CodeTokenizer())
                        expected = core.predict_logits(model, rows[:2], builder, device="cpu")
                        actual = core.predict_checkpoint(target, rows[:2], device="cpu")
                        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                        broken = dict(cp, state_dict=dict(cp["state_dict"]))
                        broken["state_dict"].pop("encoder.embeddings.word_embeddings.weight")
                        with self.assertRaisesRegex(RuntimeError, "Missing key"):
                            core.load_model(broken, torch.device("cpu"))
                        epoch = [
                            json.loads(line)
                            for line in target.with_suffix(".training.jsonl")
                            .read_text()
                            .splitlines()
                            if json.loads(line)["event"] == "epoch"
                        ][0]
                        if method.startswith("random_"):
                            self.assertEqual(epoch["candidate_queries"], 0)
                        if method.startswith("hard_"):
                            self.assertGreater(epoch["candidate_queries"], 0)
                        if method in ("vat", "random_aug_vat", "hard_aug_vat"):
                            self.assertEqual(epoch["vat_eligible_functions"], 8)
                        with self.assertRaises(FileExistsError):
                            train(
                                root / "data",
                                root / "runs",
                                str(root / "encoder"),
                                dataset,
                                method,
                                config=config,
                            )

    def test_global_vat_signal_dropout_gradient_and_radius(self):
        from tests.test_surface_vat import tiny_model

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            _, model = tiny_model(Path(tmp))
            builder = core.InputBuilder(CodeTokenizer())
            rows = [
                dict(raw_source="return external_api(1)+GLOBAL;"),
                dict(raw_source="int f(int x){return api(x);}"),
            ]
            rng = torch.get_rng_state().clone()
            terms, info, delta = vat_loss(
                model,
                rows,
                builder,
                device=torch.device("cpu"),
                noise_seed=42,
                return_delta=True,
            )
            self.assertGreater(float(terms.detach().sum()), 0.0)
            self.assertEqual(info["eligible"], 2)
            self.assertEqual(info["zero_direction"], 0)
            self.assertAlmostEqual(info["norm_min"], 1.0, places=6)
            self.assertAlmostEqual(info["norm_max"], 1.0, places=6)
            terms.mean().backward()
            self.assertGreater(float(model.task_modules["classifier"].weight.grad.norm()), 0.0)
            self.assertGreater(
                float(model.encoder.embeddings.word_embeddings.weight.grad.norm()), 0.0
            )
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            again, _, repeat = vat_loss(
                model,
                rows,
                builder,
                device=torch.device("cpu"),
                noise_seed=42,
                return_delta=True,
            )
            torch.testing.assert_close(terms, again, rtol=0, atol=0)
            torch.testing.assert_close(delta, repeat, rtol=0, atol=0)
