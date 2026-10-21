"""Cohort isolation, relation controls and actual encoder/readout gradients."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import torch

from tests.test_codebert_source import CodeTokenizer, tiny_base
from vulnmechanism import model as core
from vulnmechanism.relations import StaticRelationEncoder, relation_tokens
from vulnmechanism.static_analysis import align_graph, attach_patterns, run_process, RELATIONS, PATTERN_SCHEMA
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
        diagnostics = {}
        patterns = extract_patterns(code, diagnostics=diagnostics)
        types = {p["kind"] for p in patterns}
        self.assertTrue({"buffer_size_mismatch", "buffer_overread",
                         "off_by_one", "source_size_copy", "negative_index_write",
                         "sensitive_write", "allocation", "release"}.issubset(types), types)
        self.assertTrue(all(p["anchor"][0] < p["anchor"][1] for p in patterns))
        self.assertTrue(all(p["kind"] in RELATIONS for p in patterns))
        self.assertTrue(all(p["status"] in {"candidate", "constant_comparison"} for p in patterns))
        # Without a CPG, the lifetime path is unknown, not a claimed UAF/DF.
        self.assertNotIn("use_after_free", types)
        self.assertNotIn("double_free", types)
        self.assertGreater(diagnostics["rule_stages"]["double_free"]["unknown"], 0)
        negative = "void g(){ char dst[8]; int r=dst[-1]; }"
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
        self.assertEqual(upgraded["schema_version"], PATTERN_SCHEMA)
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
        # Element counts alone do not prove byte overflow without the ABI size.
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


class PatternRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = json.loads((Path(__file__).parent / "fixtures/static_patterns.json").read_text())

    def analyze(self, key, transform=None):
        row = self.samples[key]
        source = row["raw_source"] if transform is None else transform(row["raw_source"])
        diag = {}
        # Edited fixtures must never reuse a CPG extracted from different source.
        graph = row.get("graph") if source == row["raw_source"] else None
        patterns = extract_patterns(source, language=row["language"], graph=graph, diagnostics=diag)
        return source, patterns, diag

    def test_real_cxx_subscripts_and_unknown_signed_ranges(self):
        key = "reveal:0:10800"
        _, patterns, diag = self.analyze(key)
        self.assertNotIn("negative_index_read", {p["kind"] for p in patterns})
        self.assertGreater(diag["rule_stages"]["negative_index_read"]["unknown:loop_variant_definition"], 0)
        self.assertNotIn("unknown:subscript_ast_unsupported", diag["rule_stages"]["negative_index_read"])
        source, patterns, _ = self.analyze(key, lambda s: s.replace("env -> fptags [ i ]", "env -> fptags [ - 1 ]"))
        hits = [p for p in patterns if p["kind"] == "negative_index_read"]
        self.assertEqual(len(hits), 1)
        self.assertIn("[ - 1 ]", source[slice(*hits[0]["anchor"])])

    def test_real_alias_assignment_order_and_sizeof_provenance(self):
        key = "bigvul:180876"
        old = 'memcpy(xbuf, "perm", 5);'
        _, patterns, _ = self.analyze(key, lambda s: s.replace(old,
            '{ char *p=xbuf; int n=sizeof(xbuf); n=n+1; memset(p,0,n); }'))
        kinds = {p["kind"] for p in patterns}
        self.assertTrue({"buffer_size_mismatch", "off_by_one", "destination_size_copy"} <= kinds)
        _, patterns, _ = self.analyze(key, lambda s: s.replace(old,
            '{ char *p=xbuf; int n=100; n=5; memcpy(p,"perm",n); }'))
        self.assertFalse({"buffer_size_mismatch", "off_by_one", "buffer_overread"} & {p["kind"] for p in patterns})
        _, patterns, _ = self.analyze(key, lambda s: s.replace(old,
            '{ int n=17; memcpy(xbuf,"perm",n); n=5; }'))
        self.assertIn("off_by_one", {p["kind"] for p in patterns})

    def test_real_typed_capacity_and_offset_without_assumed_abi(self):
        key = "bigvul:178941"
        _, patterns, diag = self.analyze(key)
        self.assertNotIn("buffer_size_mismatch", {p["kind"] for p in patterns})
        self.assertGreaterEqual(diag["rule_stages"]["buffer_size_mismatch"]["resolved_extents"], 1)
        _, patterns, _ = self.analyze(key, lambda s: s.replace("sizeof(__u32) * 7", "sizeof(__u32) * 8"))
        hits = [p for p in patterns if p["kind"] == "buffer_size_mismatch"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["evidence"]["certainty"], "symbolic_relation")
        self.assertIsNone(hits[0]["evidence"]["excess"])
        self.assertNotIn("off_by_one", {p["kind"] for p in patterns})

    def test_real_string_terminators_and_pointer_size_are_distinct(self):
        key = "devign:14017"
        _, patterns, _ = self.analyze(key)
        self.assertNotIn("buffer_size_mismatch", {p["kind"] for p in patterns})
        _, patterns, _ = self.analyze(key, lambda s: s.replace("proto_str[128]", "proto_str[4]"))
        self.assertTrue({"buffer_size_mismatch", "off_by_one"} <= {p["kind"] for p in patterns})
        _, patterns, _ = self.analyze(key, lambda s: s.replace("proto_str[128]", "proto_str[4]").replace('"file"', '"fi\\0le"'))
        self.assertFalse({"buffer_size_mismatch", "off_by_one", "buffer_overread"} & {p["kind"] for p in patterns})
        _, patterns, diag = self.analyze(key, lambda s: s.replace('strcpy(proto_str, "file");',
            '{ char *p=proto_str; int n=sizeof(p); memcpy(proto_nested,p,n); }'))
        self.assertNotIn("source_size_copy", {p["kind"] for p in patterns})
        self.assertGreater(diag["rule_stages"]["buffer_size_mismatch"]["unknown:sizeof_pointer_is_not_extent"], 0)

    def test_real_array_parameters_do_not_define_caller_extents(self):
        key = "reveal:0:5743"
        _, patterns, diag = self.analyze(key)
        self.assertNotIn("buffer_size_mismatch", {p["kind"] for p in patterns})
        self.assertGreater(diag["rule_stages"]["buffer_size_mismatch"]["unknown:array_parameter_decay"], 0)
        source, patterns, _ = self.analyze(key, lambda s: s.replace("{\n",
            "{\nreal local[6]; memset(local,0,sizeof(local)+1);\n", 1))
        self.assertTrue({"buffer_size_mismatch", "off_by_one"} <= {p["kind"] for p in patterns})

    def test_real_returning_failure_branch_is_not_a_later_uaf(self):
        key = "bigvul:177924"
        self.assertGreater(self.samples[key]["baseline_patterns"]["use_after_free"], 0)
        _, patterns, diag = self.analyze(key)
        self.assertNotIn("use_after_free", {p["kind"] for p in patterns})
        self.assertGreater(diag["rule_stages"]["use_after_free"]["filtered:unconditional_return_before_use"], 0)

    def test_actual_joern_paths_on_controlled_real_failure_branch(self):
        # Both derivative sources were separately parsed by Joern, never with
        # an old CPG. These controls are excluded from dataset coverage.
        for mode, expected in (("returning", False), ("continuing", True)):
            key = "control:bigvul:177924:" + mode
            self.assertEqual(self.samples[key]["graph"]["status"], "ok")
            _, patterns, _ = self.analyze(key)
            self.assertEqual("use_after_free" in {p["kind"] for p in patterns}, expected)

    def test_real_array_address_preserves_the_whole_storage_extent(self):
        _, patterns, diag = self.analyze("devign:14075")
        self.assertNotIn("buffer_size_mismatch", {p["kind"] for p in patterns})
        self.assertGreaterEqual(diag["rule_stages"]["buffer_size_mismatch"]["resolved_extents"], 2)
        _, patterns, _ = self.analyze("devign:14075", lambda s: s.replace("&buf_in, 1024", "&buf_in, 1025"))
        self.assertTrue({"buffer_size_mismatch", "off_by_one"} <= {p["kind"] for p in patterns})

    def test_real_reassigned_pointer_does_not_reuse_field_identity(self):
        _, patterns, _ = self.analyze("reveal:0:4495")
        self.assertFalse({"double_free", "use_after_free"} & {p["kind"] for p in patterns})

    def test_real_copy_site_length_changes_through_aliases(self):
        key, old = "devign:14017", 'strcpy(proto_str, "file");'
        for content, unsafe in (("123456", True), ("x", False)):
            _, patterns, _ = self.analyze(key, lambda s: s.replace("proto_str[128]", "proto_str[4]").replace(old,
                '{ char src[8]="a"; char *q=src; strcpy(q,"' + content + '"); strcpy(proto_str,src); }'))
            self.assertEqual("buffer_size_mismatch" in {p["kind"] for p in patterns}, unsafe)
        _, patterns, diag = self.analyze(key, lambda s: s.replace("proto_str[128]", "proto_str[4]").replace(old,
            '{ char src[8]="a"; char *q=src; get_input(q); strcpy(proto_str,src); }'))
        self.assertNotIn("buffer_size_mismatch", {p["kind"] for p in patterns})
        self.assertGreater(diag["rule_stages"]["buffer_size_mismatch"]["unknown"], 0)

    def test_real_cfg_blocks_loop_redefinition_and_terminating_branches(self):
        for key in ("devign:18180", "reveal:0:20026"):
            with self.subTest(key=key):
                _, patterns, diag = self.analyze(key)
                self.assertFalse({"use_after_free", "double_free"} & {p["kind"] for p in patterns})
                self.assertGreater(sum(values.get("filtered:no_redefinition_free_cfg_path", 0)
                                       for kind, values in diag["rule_stages"].items()
                                       if kind in ("double_free", "use_after_free")), 0)

    def test_real_cpp_new_missing_nodes_and_utf8_spans(self):
        key = "bigvul:188106"
        _, patterns, diag = self.analyze(key)
        allocations = sum(p["kind"] == "allocation" for p in patterns)
        self.assertGreater(allocations, 0)
        self.assertGreater(diag["rule_stages"]["allocation"]["missing_syntax_operands"], 0)
        source, changed, _ = self.analyze(key, lambda s: '/* 字符坐标 😀 */\n' + s + '\n/* new int[99]; delete p; */')
        self.assertEqual(sum(p["kind"] == "allocation" for p in changed), allocations)
        for pattern in changed:
            for a, b in [pattern["anchor"], *pattern["related"], *pattern["evidence"].get("definitions", [])]:
                self.assertTrue(0 <= a < b <= len(source))
                self.assertTrue(source[a:b])

    def test_real_unexpanded_macro_invalidates_prior_constant(self):
        _, patterns, diag = self.analyze("reveal:1:1668")
        self.assertNotIn("negative_index_read", {p["kind"] for p in patterns})
        self.assertGreater(diag["rule_stages"]["negative_index_read"].get(
            "unknown:intervening_parse_error_or_macro", 0), 0)

    def test_real_copy_site_calloc_realloc_and_null_padding_semantics(self):
        key = "bigvul:180876"
        old = 'memcpy(xbuf, "perm", 5);'
        for allocate, expected in (("calloc(2,4)", True), ("realloc(p,16)", False)):
            source, patterns, _ = self.analyze(key, lambda s: s.replace(old,
                '{ char *p=malloc(8); p=' + allocate + '; memset(p,0,9); }'))
            self.assertEqual("buffer_size_mismatch" in {p["kind"] for p in patterns}, expected)
        for operation, unsafe in (('strncpy(d,s,8)', True), ('strncpy(d,s,4)', False), ('strcat(d,"x")', True),
                                  ('strncat(d,"x",0)', False)):
            source, patterns, _ = self.analyze(key, lambda s: s.replace(old,
                '{ char d[4]="abc"; char s[4]="a"; ' + operation + '; }'))
            self.assertEqual("buffer_size_mismatch" in {p["kind"] for p in patterns}, unsafe)
            self.assertNotIn("buffer_overread", {p["kind"] for p in patterns})

    def test_lifetime_requires_same_object_and_cfg_path(self):
        code = "void f(){char *p=malloc(8);char *q=p;free(p);q[0]=1;free(q);}"
        first, use, last = code.index("free(p)"), code.index("q[0]"), code.index("free(q)")
        graph = dict(status="ok", nodes=[dict(kind="CALL", span=[a, b]) for a, b in (
            (first, first+7), (use, use+4), (last, last+7))],
            edges=[[0, 1, 1], [1, 2, 1]])
        patterns = extract_patterns(code, graph=graph)
        self.assertTrue({"use_after_free", "double_free"} <= {p["kind"] for p in patterns})
        blocked = dict(graph, edges=[])
        patterns = extract_patterns(code, graph=blocked)
        self.assertFalse({"use_after_free", "double_free"} & {p["kind"] for p in patterns})
        # A field with the same spelling is a distinct object.
        code = "void f(struct X *x){char *p=malloc(8);free(p);x->p[0]=1;}"
        self.assertNotIn("use_after_free", {p["kind"] for p in extract_patterns(code)})
        # A proven local alias refers to the same field, but an intervening
        # field write through another alias makes its old object unknown.
        for update, expected in (("", True), ("q->p=malloc(8);", False)):
            code = "void f(struct X *x){struct X *q=x;free(x->p);" + update + "q->p[0]=1;}"
            first, use = code.index("free(x->p)"), code.index("q->p[0]")
            graph = dict(status="ok", nodes=[dict(kind="CALL", span=[a,b])
                for a,b in ((first,first+10),(use,use+7))], edges=[[0,1,1]])
            patterns = extract_patterns(code, graph=graph)
            self.assertEqual("use_after_free" in {p["kind"] for p in patterns}, expected)


if __name__ == "__main__":
    unittest.main()
