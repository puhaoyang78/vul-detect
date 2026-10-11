import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2Model, RobertaConfig, RobertaModel

from tests.test_codebert_source import CodeTokenizer
from vulnmechanism import model as core
from vulnmechanism.change_learning import (
    OBJECTIVES, check_pair, invariance_loss, prepare, random_pair_control,
    ranking_loss, read_rows, score_metrics, validate_supervision,
)
from vulnmechanism.syntax import scope_safe_rename, semantic_preserving_view


def row(key, source, label, split="train"):
    return dict(sample_key=key, dataset="primevul", split=split,
                language="c", label=label, raw_source=source)


def training_rows():
    pairs = []
    for i in range(2):
        before = row(f"p{i}:before", f"int f{i}(int n){{return 12/n;}}", 1)
        after = row(f"p{i}:after", f"int f{i}(int n){{if(n==0)return 0;return 12/n;}}", 0)
        pairs.append((before, after))
    rows = [r for p in pairs for r in p]
    rows += [row("v0", "int safe(int n){return n+1;}", 0, "valid"),
             row("v1", "int bad(int n){return 10/n;}", 1, "valid")]
    views = [(r, dict(r, raw_source=scope_safe_rename(r["raw_source"], "c")[0])) for r in rows[:4]]
    return rows, pairs, views


class ChangeLearningTests(unittest.TestCase):
    def test_scope_shadowing_loops_literals_and_members(self):
        src = 'int f(int x){ int y=x; {int x=3;y+=x;} for(int x=0;x<2;x++){y+=x;} return y+x;}'
        renamed, reason = scope_safe_rename(src, "c")
        self.assertEqual(reason, "renamed")
        self.assertEqual(renamed, 'int f(int local_value){ int y=local_value; {int x=3;y+=x;} for(int x=0;x<2;x++){y+=x;} return y+local_value;}')
        renamed, _ = scope_safe_rename('int f(int x){/* x 中文 */ const char *s="x";return x;}', "c")
        self.assertIn('/* x 中文 */ const char *s="x";', renamed)
        renamed, _ = scope_safe_rename('int f(T obj){return obj.obj;}', "cpp")
        self.assertEqual(renamed, 'int f(T local_value){return local_value.obj;}')
        for src in ['int f(int x){return MACRO(x);}', 'int f(int x){return global+x;}',
                    'int f(int x){auto g=[x](){return x;};return x;}',
                    'int f(int x){asm("" : "+r"(x));return x;}', 'int f(int x){']:
            self.assertIsNone(scope_safe_rename(src, "cpp")[0])

    def test_stable_losses_and_gradient_direction(self):
        z = torch.tensor([0., 0.], requires_grad=True)
        loss = ranking_loss(z)
        loss.backward()
        self.assertLess(float(z.grad[0]), 0)
        self.assertGreater(float(z.grad[1]), 0)
        self.assertLess(float(ranking_loss(torch.tensor([2., -2.]))), float(loss))
        self.assertEqual(float(invariance_loss(torch.tensor([1., 1.]))), 0.)
        a = invariance_loss(torch.tensor([1., -2.]))
        self.assertAlmostEqual(float(a), float(invariance_loss(torch.tensor([-2., 1.]))))
        z = torch.tensor([1000., -1000.], requires_grad=True)
        loss = invariance_loss(z)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_comment_view_preserves_preprocessor_tokens_and_line_numbers(self):
        from vulnmechanism.prepare_benchmark import normalized_source
        src = 'int f(int x){\nreturn CALL(x) + __LINE__;\n}'
        transformed, kind = semantic_preserving_view(src, "c")
        self.assertEqual(kind, "comment")
        self.assertEqual(normalized_source(src), normalized_source(transformed))
        self.assertEqual(src.count("\n"), transformed.count("\n"))
        self.assertIn('{ /* code */ ', transformed)
        self.assertIsNone(semantic_preserving_view('int f(){return __builtin_COLUMN();}', "c")[0])
        self.assertIsNone(semantic_preserving_view('int f(){', "c")[0])

    def test_compiled_transforms_match_original_behavior(self):
        sources = [
            'int f(int x){int y=x;{int x=3;y+=x;}for(int x=0;x<2;x++){y+=x;}return y+x;}',
            'int f(int x){\nif(x>0)return __LINE__;\nreturn __LINE__;\n}',
        ]
        main = '\nint main(void){for(int n=-5;n<=5;n++)printf("%d ",f(n));return 0;}\n'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for source in sources:
                transformed, _ = semantic_preserving_view(source, "c")
                outputs = []
                for i, code in enumerate((source, transformed)):
                    path = root / f"program{i}.c"
                    path.write_text('#include <stdio.h>\n' + code + main)
                    target = root / f"program{i}"
                    subprocess.run(["cc", "-std=c11", "-Wall", "-Werror", str(path), "-o", str(target)],
                                   check=True, capture_output=True)
                    outputs.append(subprocess.check_output([str(target)]))
                self.assertEqual(*outputs)

    def test_pair_identity_labels_leakage_and_random_marginals(self):
        rows, pairs, views = training_rows()
        validate_supervision(rows[:4], pairs, views)
        self.assertEqual(check_pair(*pairs[0]), "accepted")
        self.assertEqual(check_pair(pairs[0][0], dict(pairs[0][1], raw_source="int unrelated(int n){return n;}")), "function_name_mismatch")
        self.assertEqual(check_pair(pairs[0][0], dict(pairs[0][1], raw_source="int f0(double n){return n;}")), "function_signature_mismatch")
        with self.assertRaises(ValueError):
            validate_supervision(rows[:4], [(pairs[0][0], dict(pairs[0][1], split="valid"))], views)
        control = random_pair_control(pairs, 42)
        self.assertEqual({a["sample_key"] for a, _ in control}, {a["sample_key"] for a, _ in pairs})
        self.assertEqual({b["sample_key"] for _, b in control}, {b["sample_key"] for _, b in pairs})
        self.assertTrue(all(a[1]["sample_key"] != b[1]["sample_key"] for a, b in zip(control, pairs)))
        repeated = pairs * 3
        control = random_pair_control(repeated, 42)
        from collections import Counter
        self.assertEqual(Counter(b["sample_key"] for _, b in repeated), Counter(b["sample_key"] for _, b in control))
        self.assertTrue(all(a[1]["sample_key"] != b[1]["sample_key"] for a, b in zip(control, repeated)))
        with self.assertRaises(ValueError):
            random_pair_control(pairs[:1], 42)

    def test_metrics_keep_function_pair_and_robustness_denominators(self):
        rows, pairs, views = training_rows()
        result = score_metrics(rows[:4], torch.tensor([2., -2., -1., -2.]), .5,
                               pairs, views[:2], torch.tensor([-2., -2.]))
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["repair"]["pairwise_correct"], .5)
        self.assertEqual(result["repair"]["ranking_accuracy"], 1.)
        self.assertEqual(result["robustness"]["prediction_consistency"], .5)
        self.assertEqual(result["robustness"]["robust_accuracy"], .5)
        self.assertEqual(result["robustness"]["coverage"], .5)
        self.assertIsNotNone(result["pr_auc"])
        result = score_metrics([rows[0], rows[3]], torch.tensor([2., -2.]), .5,
                               pairs[:1], [], torch.empty(0), [rows[1]], torch.tensor([-2.]))
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["repair"]["pairwise_correct"], 1.)
        self.assertIsNone(result["robustness"]["robust_accuracy"])

    def test_input_identical_repairs_are_ties_despite_batch_numerical_noise(self):
        before = row("before", "int f(int n){return n;}", 1, "test")
        after = row("after", before["raw_source"], 0, "test")
        result = score_metrics([before, after], torch.tensor([.0001, -.0001]), .5,
                               [(before, after)], [], torch.empty(0),
                               input_identical_pairs={("before", "after")})
        self.assertEqual(result["accuracy"], 1.)
        self.assertEqual(result["repair"]["pairwise_correct"], 0.)
        self.assertEqual(result["repair"]["ranking_accuracy"], 0.)
        self.assertEqual(result["repair"]["ties"], 1)
        self.assertEqual(result["repair"]["input_identical_pairs"], 1)

    def test_official_train_fixed_ends_are_recovered_for_every_bce(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / "PrimeVul_v0.1"
            raw.mkdir(parents=True)
            rows, pairs, _ = training_rows()
            native = []
            manifest = []
            for i, (before, after) in enumerate(pairs):
                for j, r in enumerate((before, after)):
                    native.append(dict(idx=2*i+j, target=r["label"], func=r["raw_source"],
                        project_url="https://github.com/o/r", commit_id="a"*40,
                        commit_url="https://github.com/o/r/commit/" + "a"*40, file_name="f.c"))
                manifest.append(dict(before, sample_key=f"primevul:{2*i}", idx=2*i,
                    function=before["raw_source"], repo="github.com/o/r", commit="a"*40,
                    official_split="train", counterpart_groups=[f"group{i}"]))
            for i in range(2):
                src = f"int other{i}(int k){{ int a[20]; a[0]=k;return a[0]+2;}}"
                native.append(dict(native[0], idx=10+i, target=0, func=src))
                manifest.append(dict(manifest[0], sample_key=f"primevul:{10+i}", idx=10+i,
                                     raw_source=src, function=src, label=0, counterpart_groups=[]))
            for split in ("train", "valid", "test"):
                (raw / f"primevul_{split}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in native) if split=="train" else "")
                (raw / f"primevul_{split}_paired.jsonl").write_text("".join(json.dumps(r)+"\n" for r in native[:4]) if split=="train" else "")
            path = root / "manifest.jsonl"
            path.write_text("".join(json.dumps(r)+"\n" for r in manifest))
            result = prepare(path, root / "raw", root / "prepared")
            self.assertEqual(result["repair_pairs"]["primevul/train"]["accepted"], 2)
            self.assertEqual(result["repair_pairs"]["primevul/train"]["recovered_fixed_ends"], 2)
            self.assertEqual(len(read_rows(root / "prepared" / "records.jsonl")), 6)
            self.assertEqual(len(read_rows(root / "prepared" / "pair_members.jsonl")), 2)

    def test_real_encoder_training_save_reload_all_ablations(self):
        torch.set_num_threads(2)
        rows, pairs, views = training_rows()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for variant in ("codebert_source", "baseline"):
                modelpath = root / variant
                if variant == "codebert_source":
                    model = RobertaModel(RobertaConfig(vocab_size=16, hidden_size=16,
                        num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                        max_position_embeddings=514, pad_token_id=0, bos_token_id=2,
                        eos_token_id=1, hidden_dropout_prob=0., attention_probs_dropout_prob=0.))
                else:
                    model = Qwen2Model(Qwen2Config(vocab_size=16, hidden_size=16,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                        intermediate_size=32, max_position_embeddings=1024, pad_token_id=0,
                        eos_token_id=1, attention_dropout=0.))
                model.save_pretrained(modelpath)
                budgets = []
                with patch.object(core.AutoTokenizer, "from_pretrained", return_value=CodeTokenizer()):
                    for objective in OBJECTIVES:
                        path = root / variant / objective / "best.pt"
                        cp = core.train_model(None, path, records=rows, variant=variant,
                            model_path=str(modelpath), source_max_length=512, device="cpu",
                            epochs=1, batch_size=2, gradient_accumulation=1,
                            lora_r=2, lora_alpha=4, lora_dropout=0., change_objective=objective,
                            repair_pairs=pairs, invariant_views=views)
                        budgets.append((cp["trainable_parameters"], cp["change_supervision"]["optimizer_steps_per_epoch"],
                                        cp["change_supervision"]["auxiliary_function_forwards_per_epoch"]))
                        logits = core.predict_checkpoint(path, rows[-2:], device="cpu", batch_size=2)
                        self.assertTrue(torch.isfinite(logits).all())
                        self.assertEqual(cp["selection"], "validation_mcc")
                        self.assertEqual(cp["seed"], 42)
                    self.assertEqual(len(set(budgets)), 1)


if __name__ == "__main__":
    unittest.main()
