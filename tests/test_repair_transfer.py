"""Frozen extraction, isolation, deterministic cosine ranks and reviewed denominators."""

import copy
import csv
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_bounded_attention import detector, batch
from vulnmechanism.cli import build_parser
from vulnmechanism.repair_transfer import (
    METHODS, source_identity, frozen_line_batch, reference_groups, allowed_references,
    cosine_nearest, top_lines, rank_query, metric_rows, load_inputs, retrieve, report,
)
from vulnmechanism.data import write_jsonl


def reference(code, label, *, pair="train:1", group=1):
    return dict(split="train", target=1, code=code, label=label, pair_key=pair,
                source_group=group, sample_key=pair+":v", project="p", line=1, token_count=2.)


class RepairTransferTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_cli_offline_stages(self):
        for stage in ("extract", "retrieve", "report"):
            args = build_parser().parse_args(["repair-transfer", stage])
            self.assertEqual(args.stage, stage)
            self.assertIn("context_diff", str(args.checkpoint))

    def test_source_identity_excludes_formatting_and_comment_copies(self):
        self.assertEqual(source_identity("int f(){return 1;}"), source_identity("int f ( ) { /* x */ return 1; }"))
        self.assertNotEqual(source_identity('return "a b";'), source_identity('return "ab";'))

    def test_observe_exact_pool_states_without_model_changes(self):
        model = detector(context=True, region="diff").eval().requires_grad_(False)
        before = copy.deepcopy(model.state_dict()); data = batch()
        # One code Token belongs to two physical lines, retaining total mass one.
        token = data["code_mask"][0].nonzero()[0, 0]
        data["line_token_mask"][0, 1, token] = True
        rng = torch.get_rng_state().clone()
        probability, vectors, details = frozen_line_batch(model, data, "cpu")
        with torch.no_grad():
            states = model.encoder(data["input_ids"], attention_mask=data["attention_mask"])[0]
        membership = (data["line_token_mask"] & data["code_mask"][:, None]).float()
        membership /= membership.sum(1).clamp_min(1)[:, None]
        expected = torch.bmm(membership, states) / membership.sum(-1).clamp_min(1e-12)[..., None]
        torch.testing.assert_close(vectors, expected, rtol=0, atol=0)
        torch.testing.assert_close(details["line_counts"].sum(-1), data["code_mask"].sum(-1).float())
        torch.testing.assert_close(details["line_attention"].sum(-1), torch.ones(2))
        self.assertEqual(vectors.shape[-1], 16)
        self.assertFalse(model.line_attention._forward_pre_hooks)
        for key, value in before.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        other, other_vectors, _ = frozen_line_batch(model, data, "cpu")
        torch.testing.assert_close(probability, other, rtol=0, atol=0)
        torch.testing.assert_close(vectors, other_vectors, rtol=0, atol=0)

    def test_extraction_ignores_function_and_diff_labels(self):
        model = detector(context=True, region="diff").eval().requires_grad_(False)
        data = batch(); a, av, _ = frozen_line_batch(model, data, "cpu")
        data["labels"] = 1-data["labels"]
        data["line_labels"] = torch.ones_like(data["line_numbers"])
        b, bv, _ = frozen_line_batch(model, data, "cpu")
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(av, bv, rtol=0, atol=0)

    def test_rejects_training_or_unfrozen_model(self):
        model = detector(context=True).eval()
        with self.assertRaises(ValueError): frozen_line_batch(model, batch(), "cpu")
        model.requires_grad_(False).train()
        with self.assertRaises(ValueError): frozen_line_batch(model, batch(), "cpu")

    def test_reference_source_isolation_and_conflicting_duplicate_text(self):
        refs = [reference("  x++; ", 1), reference("x++;", 1, pair="train:2", group=2),
                reference("return 0;", 1), reference("return 0;", 0),
                reference("background();", 0, pair="train:3", group=3)]
        groups, info = reference_groups(refs)
        self.assertEqual(groups, [[0, 1], [4]])
        self.assertEqual(info["conflicting_texts_removed"], 1)
        selected, _ = allowed_references(refs, groups, dict(pair_key="train:1", source_group=1))
        self.assertEqual(selected, {1:[1], 0:[4]})

    def test_reference_train_and_vulnerable_only(self):
        for key, value in (("split", "valid"), ("target", 0)):
            row = reference("x;", 1); row[key] = value
            with self.assertRaises(ValueError): reference_groups([row])

    def test_excludes_duplicate_function_even_under_different_pair(self):
        refs = [reference("p;", 1, pair="train:8", group=3),
                reference("p;", 1, pair="train:9", group=4), reference("b;", 0, group=5)]
        groups, _ = reference_groups(refs)
        selected, counts = allowed_references(refs, groups, dict(pair_key="valid:2", source_group=3))
        self.assertEqual(selected[1], [1])
        self.assertEqual(counts["excluded_source_occurrences"], 1)

    def test_empty_isolated_class_is_an_error(self):
        refs = [reference("p;", 1), reference("b;", 0)]
        groups, _ = reference_groups(refs)
        with self.assertRaises(ValueError):
            allowed_references(refs, groups, dict(pair_key="train:1", source_group=2))

    def test_cosine_ties_across_reference_chunks(self):
        query = torch.tensor([[1.,0.], [0.,1.]])
        bank = torch.tensor([[1.,0.]]).expand(8193, -1).clone()
        score, ids = cosine_nearest(query, bank, list(range(len(bank))), "cpu")
        self.assertEqual(ids, [0,0]); self.assertEqual(score, [1.,0.])

    def test_direct_and_contrastive_have_the_specified_different_ranks(self):
        refs = [reference("positive;", 1), reference("background;", 0)]
        groups, _ = reference_groups(refs)
        query = dict(pair_key="valid:1", source_group=None, line_numbers=[5,9], line_token_counts=[1.,1.],
                     line_attention=[.1,.9])
        q = torch.eye(2); bank = torch.tensor([[1.,.5],[1.,0.]])
        rng = random.getstate()
        result = rank_query(query, q, bank, refs, groups, "cpu", 0)
        self.assertEqual(result["methods"]["positive_nn"][0]["line"], 5)
        self.assertEqual(result["methods"]["contrastive_nn"][0]["line"], 9)
        self.assertEqual(result["methods"]["context_diff_attention"][0]["line"], 9)
        self.assertEqual(result, rank_query(query, q, bank, refs, groups, "cpu", 0))
        self.assertEqual(random.getstate(), rng)

    def test_line_ties_use_physical_order(self):
        self.assertEqual(top_lines([20,2,7], [.5,.5,.5]), [1,2,0])

    def test_unknown_and_unreviewed_denominators(self):
        q = dict(split="valid", sample_key="v:1", methods={m:[dict(line=i) for i in (1,2,3)] for m in METHODS})
        labels = {"v:1:line:1":"unknown", "v:1:line:2":"related", "v:1:line:3":"unrelated"}
        rows = metric_rows([q], labels)
        top = next(r for r in rows if r["method"]=="positive_nn" and r["k"]==3)
        self.assertEqual(top["selected_lines"], 3)
        self.assertEqual(top["related_fraction"], 1/3)
        self.assertEqual(top["unknown_fraction"], 1/3)
        self.assertEqual(top["hit_fraction"], 1.)
        del labels["v:1:line:2"]
        with self.assertRaises(KeyError): metric_rows([q], labels)

    def test_input_reader_never_opens_test(self):
        def fixture(path):
            path = Path(path)
            self.assertNotIn("test", path.name)
            split = "train" if "train" in path.name else "valid"
            if path.name.startswith("pairs."):
                return [dict(pair_key=f"primevul_paired:{split}:{i}") for i in range(24)]
            result = []
            for i in range((8708 if split=="train" else 1124)//2):
                for j,target in enumerate((1,0)):
                    result.append(dict(pair_key=f"primevul_paired:{split}:{i}", target=target,
                        sample_key=f"primevul_paired:{split}:{2*i+j+1}", lines_ground_truth=[] if target else [0]))
            return result
        with patch("vulnmechanism.repair_transfer.read_jsonl", side_effect=fixture) as reader:
            rows, cohorts = load_inputs("data", "audit")
        self.assertEqual(reader.call_count, 4)
        self.assertEqual(len(rows["train"]), 8708)
        self.assertEqual(len(cohorts["valid"]), 24)

    def test_report_requires_complete_unchanged_source_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            q = dict(split="valid", sample_key="valid:v", pair_key="valid:p", cohort="fixed",
                project="p", source_tokens=600, source_lines=80, truncated=True, static_recovered=False,
                line_numbers=[1,2,3], methods={m:[dict(rank=i, line=i,
                    positive_reference=dict(project="p"), background_reference=dict(project="other"))
                    for i in (1,2,3)] for m in METHODS})
            sample = [dict(review_id=f"valid:v:line:{i}", sample_key=q["sample_key"], pair_key=q["pair_key"],
                split="valid", line=i, code=f"line{i};", judgment=value, note="synthetic source review")
                for i,value in enumerate(("related","unrelated","pending"),1)]
            write_jsonl(root/"rankings.jsonl", [q]); write_jsonl(root/"review_sample.jsonl", sample)
            write_jsonl(root/"review.jsonl", sample)
            with self.assertRaises(ValueError): report(root)
            sample[-1]["judgment"] = "unknown"
            write_jsonl(root/"review.jsonl", sample)
            result = report(root)
            self.assertEqual(len(result), 8)
            self.assertEqual(next(r for r in result if r["method"]=="positive_nn" and r["k"]==3)["related_fraction"], 1/3)
            self.assertTrue((root/"length_bias.csv").exists())
            with (root/"paired_hits.csv").open() as handle:
                paired = list(csv.DictReader(handle))
            self.assertEqual(len(paired), 8)
            self.assertTrue(all(r["gained"] == r["lost"] == "0" for r in paired))
            sample[0]["code"] = "different_source;"
            write_jsonl(root/"review.jsonl", sample)
            with self.assertRaises(ValueError): report(root)

    def test_retrieval_preserves_existing_review_before_any_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "review.jsonl").write_text("existing manual review\n")
            with patch("vulnmechanism.repair_transfer.load_inputs") as reader:
                with self.assertRaisesRegex(ValueError, "preserve completed judgments"):
                    retrieve("data", "audit", "checkpoint", root, "cpu")
            reader.assert_not_called()
            self.assertEqual((root / "review.jsonl").read_text(), "existing manual review\n")


if __name__ == "__main__":
    unittest.main()
