"""Cohort isolation, relation controls and actual encoder/readout gradients."""

from collections import Counter
from pathlib import Path
import tempfile
import unittest

import torch

from tests.test_codebert_source import CodeTokenizer, tiny_base
from vulnmechanism import model as core
from vulnmechanism.relations import RelationPool, relation_tokens
from vulnmechanism.static_analysis import align_graph, candidate_kinds, run_process
from vulnmechanism.static_data import remove_duplicates, isolate_families, near_overlaps


class StaticTests(unittest.TestCase):
    def test_process_monitor_handles_child_exit_and_nonzero_status(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "process.log"
            result = run_process([sys.executable, "-c", "import subprocess; p=[subprocess.Popen(['true']) for _ in range(20)]; [x.wait() for x in p]"], log)
            self.assertGreater(result["seconds"], 0)
            with self.assertRaisesRegex(RuntimeError, "Joern exit 3"):
                run_process([sys.executable, "-c", "raise SystemExit(3)"], log)
            with self.assertRaises(TimeoutError):
                run_process([sys.executable, "-c", "import time; time.sleep(10)"], log, timeout=.1)

    def test_conflicts_duplicates_and_official_commit_isolation(self):
        def row(key, source, label, split):
            return dict(sample_key=key, raw_source=source, label=label, split=split,
                        project="p", commit_id="c")
        rows = [row("a", "int f(){return 0;}", 0, "train"),
                row("b", "int f() { /*comment*/ return 0; }", 0, "test"),
                row("c", "int g(){return 1;}", 0, "train"),
                row("d", "int g(){return 1;}", 1, "valid"),
                row("e", "int fixed(){return 2;}", 0, "train")]
        kept, removed, counts = remove_duplicates(rows)
        self.assertEqual({r["sample_key"] for r in kept}, {"b", "e"})
        self.assertEqual(counts["conflicting_source_or_token_labels"], 2)
        kept, removed = isolate_families(kept, "devign")
        self.assertEqual([r["sample_key"] for r in kept], ["b"])
        self.assertEqual(removed[0]["reason"], "cross_split_commit_family")

    def test_cross_split_near_filter_removes_lower_priority_only(self):
        source = "int f(int x) { " + "; ".join(f"x += {i}" for i in range(40)) + "; return x; }"
        rows = [dict(sample_key="test", split="test", raw_source=source),
                dict(sample_key="train", split="train", raw_source=source.replace("x += 39", "x += 99")),
                dict(sample_key="valid", split="valid", raw_source="int different(){return 0;}")]
        kept, pairs = near_overlaps(rows)
        self.assertEqual({r["sample_key"] for r in kept}, {"test", "valid"})
        self.assertEqual(pairs[0]["removed"], "train")
        self.assertGreaterEqual(pairs[0]["jaccard"], .9)

    def test_candidate_annotations_do_not_claim_verified_vulnerability(self):
        self.assertEqual(candidate_kinds(dict(name="memcpy")), ["sensitive_api"])
        self.assertEqual(candidate_kinds(dict(name="<operator>.lessThan")), ["guard"])
        graph = dict(nodes=[dict(id="0", name="f", kind="CALL", code="wrong", line=1, column=1)], edges=[])
        result = align_graph(graph, "f(x);")
        self.assertEqual(result["status"], "incomplete_graph")
        self.assertEqual(result["nodes"][0]["alignment"], "location_token")

    def test_shuffle_preserves_marginals_and_ignores_labels(self):
        row = dict(raw_source="abcdefghij", program_relations=dict(status="ok",
                   nodes=[dict(span=[i, i + 1]) for i in range(10)],
                   edges=[[i, i + 1, i % 3] for i in range(9)]))
        real, _ = relation_tokens(row, CodeTokenizer(), 2025, "real")
        shuffled, info = relation_tokens(row, CodeTokenizer(), 2025, "shuffled")
        self.assertEqual(Counter(a for a, _, _ in real), Counter(a for a, _, _ in shuffled))
        self.assertEqual(Counter(b for _, b, _ in real), Counter(b for _, b, _ in shuffled))
        self.assertEqual(Counter(k for _, _, k in real), Counter(k for _, _, k in shuffled))
        self.assertGreater(info["changed_endpoints"], 0)
        self.assertEqual(shuffled, relation_tokens(dict(row, label=1), CodeTokenizer(), 2025, "shuffled")[0])
        self.assertTrue(relation_tokens(dict(raw_source="abc"), CodeTokenizer(), 2025, "sequence")[0])

    def test_overlapping_window_positions_have_unit_weight(self):
        torch.manual_seed(42)
        module = RelationPool(8)
        torch.nn.init.normal_(module.output.weight)
        hidden = torch.randn(1, 3, 8, requires_grad=True)
        owners = torch.tensor([0])
        positions = torch.tensor([[0, 1, 2]])
        edges = torch.tensor([[0, 0, 2, 0], [0, 1, 2, 1]])
        expected = module(hidden, positions, owners, torch.ones(1, 3), edges, 1)
        repeated = module(hidden.expand(2, -1, -1), positions.expand(2, -1), torch.tensor([0, 0]),
                          torch.full((2, 3), .5), edges, 1)
        torch.testing.assert_close(expected, repeated)
        repeated.square().sum().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertGreater(float(hidden.grad.abs().sum()), 0)

    def test_real_model_forward_backward_and_strict_reload(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "base";tiny_base(path)
            row = dict(sample_key="a", raw_source="int f(int x){return x+1;}",
                       program_relations=dict(status="ok", nodes=[dict(span=[4, 5]), dict(span=[20, 21])],
                                              edges=[[0, 1, 0], [1, 0, 3]]))
            core.seed_everything(42);plain = core.CodeBERT(path, "cpu").eval()
            reference = plain(*core.InputBuilder(CodeTokenizer()).codebert_batch([row], device="cpu"))
            initial, counts = [], []
            for mode in ("sequence", "shuffled", "real"):
                core.seed_everything(42);model = core.CodeBERT(path, "cpu", relation_mode=mode).eval()
                builder = core.InputBuilder(CodeTokenizer(), relation_mode=mode)
                inputs = builder.codebert_batch([row], device="cpu")
                torch.testing.assert_close(model(*inputs), reference, atol=0, rtol=0)
                initial.append({k: v.clone() for k, v in model.task_modules["relations"].state_dict().items()})
                counts.append(sum(p.numel() for p in model.parameters()))
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                for _ in range(2):
                    optimizer.zero_grad();logits = model(*inputs)
                    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                    optimizer.step()
                self.assertGreater(float(model.task_modules["relations"].source.weight.grad.abs().sum()), 0)
                self.assertGreater(float(model.encoder.embeddings.word_embeddings.weight.grad.abs().sum()), 0)
                saved = Path(tmp) / "checkpoint.pt"
                checkpoint = dict(model_path=str(path), state_dict=model.state_dict(), static_relations=dict(mode=mode))
                torch.save(checkpoint, saved)
                restored = core.load_model(torch.load(saved, weights_only=False), "cpu")
                torch.testing.assert_close(model(*inputs), restored(*inputs), atol=0, rtol=0)
            self.assertEqual(len(set(counts)), 1)
            for other in initial[1:]:
                for key in initial[0]:
                    torch.testing.assert_close(initial[0][key], other[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
