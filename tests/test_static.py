"""Cohort isolation, relation controls and actual encoder/readout gradients."""

from collections import Counter
from pathlib import Path
import tempfile
import unittest

import torch

from tests.test_codebert_source import CodeTokenizer, tiny_base
from vulnmechanism import model as core
from vulnmechanism.relations import StaticRelationEncoder, relation_tokens
from vulnmechanism.static_analysis import align_graph, attach_patterns, run_process, RELATIONS
from vulnmechanism.static_data import remove_duplicates, isolate_families, near_overlaps
from vulnmechanism.static_patterns import extract_patterns


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

    def test_vulnerability_pattern_families_are_typed_and_auditable(self):
        code = """void f(int i) {
          char dst[8]; char src[4]; int n=9;
          memcpy(dst,src,n); memcpy(dst,src,sizeof(src));
          if(i >= 0 && i < 8) dst[i]=1;
          dst[-1]=2;
          char *p=malloc(8); free(p); free(p); p[0]=1;
        }"""
        patterns = extract_patterns(code)
        types = {p["kind"] for p in patterns}
        self.assertTrue({"buffer_size_mismatch", "buffer_overread",
                         "off_by_one", "source_size_copy", "double_free",
                         "use_after_free", "negative_index_write",
                         "sensitive_write", "allocation", "release"}.issubset(types), types)
        self.assertTrue(all(p["anchor"][0] < p["anchor"][1] for p in patterns))
        self.assertTrue(all(p["kind"] in RELATIONS for p in patterns))
        self.assertTrue(all(p["status"] in {"candidate", "constant_comparison"} for p in patterns))
        negative = "void g(int i){ char dst[8]; int r=dst[i]; }"
        self.assertIn("negative_index_read", {p["kind"] for p in extract_patterns(negative)})

    def test_source_length_copy_and_destination_length_copy(self):
        source = """void f() {
          char d[8]; char s[4];
          memcpy(d, s, strlen(s));
          memcpy(d, s, sizeof(d));
          strcpy(d, s);
        }"""
        patterns = extract_patterns(source)
        names = {p["kind"] for p in patterns}
        self.assertIn("source_size_copy", names)
        self.assertIn("destination_size_copy", names)
        self.assertTrue(all(p["status"] == "candidate" for p in patterns
                            if p["kind"] in ("source_size_copy", "destination_size_copy")))

    def test_existing_joern_cache_upgrades_without_reparsing(self):
        source = "void f(){char d[8];char s[4];memcpy(d,s,9);}"
        graph = dict(status="ok", nodes=[dict(span=[5, 6], kind="METHOD"),
                       dict(span=[18, 19], kind="IDENTIFIER")],
                     edges=[[0, 1, 0], [1, 0, 1], [0, 0, 6]],
                     counts={"AST": 1, "CFG": 1, "memory": 1})
        upgraded = attach_patterns(graph, source)
        self.assertEqual(upgraded["schema_version"], 4)
        self.assertNotIn("memory", upgraded["counts"])
        self.assertGreater(upgraded["counts"].get("buffer_size_mismatch", 0), 0)
        self.assertEqual(upgraded, attach_patterns(upgraded, source))
        call_at = source.index("memcpy")
        graph2 = dict(status="ok", nodes=[dict(span=[call_at, call_at + 6], kind="CALL"),
                                          dict(span=[0, 4], kind="IDENTIFIER")],
                      edges=[[0, 1, 0], [0, 1, 1], [0, 1, 3]], counts={"AST": 1, "CFG": 1, "REACHING_DEF": 1})
        extended = attach_patterns(graph2, source)
        family = RELATIONS.index("buffer_size_mismatch")
        self.assertTrue(any(e[2] == family and e[1] == 1 for e in extended["edges"]))

        incomplete = dict(nodes=[dict(id="0", name="f", kind="CALL", code="wrong", line=1, column=1)], edges=[])
        result = align_graph(incomplete, "f(x);")
        self.assertEqual(result["status"], "incomplete_graph")
        self.assertEqual(result["nodes"][0]["alignment"], "location_token")

    def test_copy_arguments_and_byte_buffer_sizes(self):
        read = "void f(int fd){char dst[8];read(fd,dst,9);}"
        p = extract_patterns(read)
        self.assertIn("buffer_size_mismatch", {x["kind"] for x in p})
        other = "void f(){int dst[8];char src[16];memcpy(dst,src,10);}"
        # Ten bytes do not exceed the capacity of an array of eight ints.
        self.assertNotIn("buffer_size_mismatch", {x["kind"] for x in extract_patterns(other)})
        unsigned = "void f(unsigned int i){char b[8];int x=b[i];}"
        self.assertNotIn("negative_index_read", {x["kind"] for x in extract_patterns(unsigned)})

    def test_shuffle_preserves_window_marginals_and_ignores_labels(self):
        row = dict(sample_key="pair", raw_source="abcdefghij",
                   program_relations=dict(status="ok",
                       nodes=[dict(span=[i, i + 1]) for i in range(10)],
                       edges=[[i, i + 1, 3] for i in range(9)]))
        real_pairs, _ = relation_tokens(row, CodeTokenizer(), 2025, "real")
        shuffled_pairs, _ = relation_tokens(row, CodeTokenizer(), 2025, "shuffled")
        self.assertEqual(real_pairs, shuffled_pairs)
        real = core.InputBuilder(CodeTokenizer(), relation_mode="real")
        shuffled = core.InputBuilder(CodeTokenizer(), relation_mode="shuffled")
        a = real.codebert_batch([row], device="cpu")[-1].tolist()
        b = shuffled.codebert_batch([row], device="cpu")[-1].tolist()
        self.assertEqual(Counter((w, q, typ) for w, q, k, typ in a),
                         Counter((w, q, typ) for w, q, k, typ in b))
        self.assertEqual(Counter((w, k, typ) for w, q, k, typ in a),
                         Counter((w, k, typ) for w, q, k, typ in b))
        self.assertGreater(shuffled.relation_cache["pair"][1]["changed_endpoints"], 0)
        self.assertEqual(b, shuffled.codebert_batch([dict(row, label=1)], device="cpu")[-1].tolist())
        self.assertTrue(relation_tokens(dict(raw_source="abc"), CodeTokenizer(), 2025, "sequence")[0])


    def test_relation_attention_is_per_query_and_changes_encoder_state(self):
        torch.manual_seed(42)
        module = StaticRelationEncoder(8, [0, 1])
        hidden = torch.randn(1, 4, 8, requires_grad=True)
        # Two distinct definitions compete for the same use's attention.
        edges = torch.tensor([[0, 0, 1, 3], [0, 0, 3, 3], [0, 2, 3, 2]])
        actual = module(hidden, edges, 0)
        rewired = edges.clone()
        rewired[0, 2], rewired[1, 2] = edges[1, 2], edges[0, 2]
        # Swapping keys within one query group alone preserves an attention sum.
        # Change the pairing across two query sites, preserving key marginals.
        rewired[1, 2], rewired[2, 2] = edges[2, 2], edges[1, 2]
        changed = module(hidden, rewired, 0)
        self.assertGreater(float((actual - changed).abs().max()), 1e-6)
        self.assertGreater(float((actual - hidden).abs().max()), 1e-6)
        torch.testing.assert_close(module(hidden, edges[:0], 0), hidden)
        actual.square().sum().backward()
        self.assertIsNotNone(module.adapters["0"].query.weight.grad)
        self.assertGreater(float(module.adapters["0"].query.weight.grad.abs().sum()), 0)
        self.assertGreater(float(hidden.grad.abs().sum()), 0)
        self.assertEqual(tuple(module.layer_indices), (0, 1))

    def test_joern_failure_still_keeps_conservative_syntax_candidates(self):
        src = "void f(){char dst[8];char src[4];memcpy(dst,src,9);}"
        result = attach_patterns(dict(status="no_definition", nodes=[], edges=[], counts={}), src)
        self.assertEqual(result["status"], "no_definition")
        self.assertEqual(result["analysis_source"], "syntax_only")
        self.assertGreater(result["pattern_counts"]["buffer_size_mismatch"], 0)
        row = dict(raw_source=src, program_relations=result)
        pair, info = relation_tokens(row, CodeTokenizer(), 2025, "real")
        self.assertTrue(pair)
        self.assertEqual(info["graph_status"], "no_definition")

    def test_relation_attention_repeats_exactly_with_shared_query_edges(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)
        module = StaticRelationEncoder(16, [0]).to(device).eval()
        hidden = torch.randn(2, 32, 16, device=device, requires_grad=True)
        edges = torch.stack((torch.randint(2, (1024,)),
                             torch.randint(4, (1024,)),
                             torch.randint(32, (1024,)),
                             torch.randint(len(RELATIONS), (1024,))), dim=1).to(device)
        actual = module(hidden, edges, 0)
        for _ in range(5):
            torch.testing.assert_close(module(hidden, edges, 0), actual, atol=0, rtol=0)
        # Query groups remain independent, including positions with no edges.
        isolated = edges[edges[:, 0] == 0].clone()
        torch.testing.assert_close(module(hidden[:1], isolated, 0), actual[:1])
        torch.testing.assert_close(actual[:, 4:], hidden[:, 4:], atol=0, rtol=0)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in module.parameters()))

    def test_windows_include_only_co_visible_relation_endpoints(self):
        from vulnmechanism.model import InputBuilder
        src = "int f() { return 0; }"
        row = dict(sample_key="w", raw_source=src,
                   program_relations=dict(status="ok", nodes=[
                       dict(span=[4, 5]), dict(span=[10, 16])],
                       edges=[[0, 1, 3], [1, 0, 2]]))
        builder = InputBuilder(CodeTokenizer(), relation_mode="real")
        batch = builder.codebert_batch([row], device="cpu")
        self.assertEqual(batch[-1].shape[1], 4)
        self.assertGreater(int(batch[-1].shape[0]), 0)
        self.assertGreaterEqual(int(batch[-1][:, 1:3].min()), 1)
        self.assertGreaterEqual(builder.relation_cache["w"][1]["pairs_in_shared_windows"], 1)

    def test_checkpointed_codebert_relations_receive_gradients(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "base"; tiny_base(path)
            row = dict(
                sample_key="ck", raw_source="int f(int x){int n=x+1;return n;}",
                program_relations=dict(status="ok", nodes=[
                    dict(span=[4, 5]), dict(span=[16, 17]),
                    dict(span=[21, 22]), dict(span=[30, 31])],
                    edges=[[0, 1, 3], [0, 2, 3], [1, 3, 2], [2, 3, 3]]),
            )
            model = core.CodeBERT(path, "cpu", gradient_checkpointing=True,
                                  relation_mode="real").train()
            builder = core.InputBuilder(CodeTokenizer(), relation_mode="real")
            inputs = builder.codebert_batch([row], device="cpu")
            self.assertTrue(inputs[-1].numel() > 0)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(*inputs), torch.ones(1))
            loss.backward()
            adapters = model.task_modules["relations"].adapters
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in adapters.parameters()))
            self.assertGreater(float(model.encoder.embeddings.word_embeddings.weight.grad.abs().sum()), 0)

    def test_real_model_forward_backward_and_strict_reload(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "base"; tiny_base(path)
            source = "int f(int x){int z=x+1;return z;}"
            row = dict(sample_key="a", raw_source=source,
                       program_relations=dict(status="ok", nodes=[
                           dict(span=[4, 5]), dict(span=[10, 11]),
                           dict(span=[13, 16]), dict(span=[22, 23])],
                           edges=[[0, 1, 0], [2, 3, 0], [0, 3, 3], [2, 1, 3]]))
            core.seed_everything(42)
            reference = core.CodeBERT(path, "cpu").eval()(
                *core.InputBuilder(CodeTokenizer()).codebert_batch([row], device="cpu"))
            initial, counts = [], []
            for mode in ("sequence", "shuffled", "real"):
                core.seed_everything(42)
                model = core.CodeBERT(path, "cpu", relation_mode=mode).eval()
                builder = core.InputBuilder(CodeTokenizer(), relation_mode=mode)
                inputs = builder.codebert_batch([row], device="cpu")
                observed = model(*inputs)
                self.assertTrue(torch.isfinite(observed).all())
                self.assertTrue(model.task_modules["relations"].layer_indices)
                self.assertGreater(inputs[-1].shape[0], 0)
                # Unlike the removed zero-residual model, relations affect initial predictions.
                self.assertGreater(float((observed - reference).abs().max()), 1e-8)
                initial.append({k: v.clone() for k, v in model.task_modules["relations"].state_dict().items()})
                counts.append(sum(p.numel() for p in model.parameters()))
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                for _ in range(2):
                    optimizer.zero_grad()
                    logits = model(*inputs)
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, torch.ones_like(logits)).backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                    optimizer.step()
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.task_modules["relations"].parameters() if p.grad is not None), 0)
                self.assertGreater(float(model.encoder.embeddings.word_embeddings.weight.grad.abs().sum()), 0)
                saved = Path(tmp) / "checkpoint.pt"
                checkpoint = dict(model_path=str(path), state_dict=model.state_dict(),
                                  static_relations=dict(mode=mode))
                torch.save(checkpoint, saved)
                restored = core.load_model(torch.load(saved, weights_only=False), "cpu")
                torch.testing.assert_close(model(*inputs), restored(*inputs), atol=0, rtol=0)
            self.assertEqual(len(set(counts)), 1)
            for other in initial[1:]:
                for key in initial[0]:
                    torch.testing.assert_close(initial[0][key], other[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
