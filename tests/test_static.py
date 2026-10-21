"""Leakage controls, vulnerability-pattern rules, directed attention and gradients."""

from collections import Counter
from pathlib import Path
import tempfile
import unittest

import torch

from tests.test_codebert_source import CodeTokenizer, tiny_base
from vulnmechanism import model as core
from vulnmechanism.relations import StaticRelationAttention, relation_tokens
from vulnmechanism.static_analysis import align_graph, candidate_kinds, run_process
from vulnmechanism.static_data import remove_duplicates, isolate_families, near_overlaps
from vulnmechanism.vulnerability_patterns import extract_patterns, split_arguments


def graph(nodes, edges=()):
    return dict(nodes=nodes, edges=list(edges))


def cnode(id, name, code, line, column=1, kind="CALL"):
    return dict(id=str(id), name=name, kind=kind, code=code,
                line=line, column=column)


class StaticTests(unittest.TestCase):
    def test_process_monitor_handles_child_exit_and_nonzero_status(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "process.log"
            result = run_process([sys.executable, "-c",
                "import subprocess; p=[subprocess.Popen(['true']) for _ in range(20)]; [x.wait() for x in p]"], log)
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

    def test_call_arguments_handle_nested_calls_and_strings(self):
        self.assertEqual(split_arguments('memcpy(dst, src, f(x, "a,b"))'),
                         ["dst", "src", 'f(x, "a,b")'])

    def test_buffer_vulnerability_patterns_are_distinct(self):
        source = "void f() {\n char dst[8];\n char src[4];\n memcpy(dst, src, 9);\n}"
        g = graph([cnode(1, "memcpy", "memcpy(dst, src, 9)", 4, 2)])
        patterns = extract_patterns(g, source)
        kinds = {p["kind"] for p in patterns}
        self.assertTrue({"incorrect_size", "off_by_one", "buffer_overread",
                         "write_api", "copy_length_relation"} <= kinds)
        self.assertEqual(next(p for p in patterns if p["kind"] == "off_by_one")["status"],
                         "numeric_match")
        g2 = graph([cnode(2, "memcpy", "memcpy(dst, src, n)", 4, 2)])
        patterns2 = extract_patterns(g2, source)
        self.assertFalse(any(p["kind"] == "off_by_one" for p in patterns2))
        self.assertEqual(next(p for p in patterns2 if p["kind"] == "incorrect_size")["status"],
                         "unknown_bound")
        src3 = 'void f(){char dst[8];memcpy(dst, src, strlen(src));}'
        self.assertIn("source_size_copy",
                      {p["kind"] for p in extract_patterns(graph([
                        cnode(3,"memcpy","memcpy(dst, src, strlen(src))",1)]),src3)})

    def test_lifetime_patterns_and_path_uncertainty(self):
        src = "void f() {\n free(p);\n p[0] = 1;\n free(p);\n}"
        g = graph([cnode(1,"free","free(p)",2,2),
                   cnode(2,"p","p",3,2,"IDENTIFIER"),
                   cnode(3,"free","free(p)",4,2)])
        patterns = extract_patterns(g,src)
        kinds = {p["kind"] for p in patterns}
        self.assertIn("double_free",kinds)
        self.assertIn("use_after_free",kinds)
        self.assertTrue(all(p["status"] == "path_unchecked"
                            for p in patterns if p["kind"] in ("double_free","use_after_free")))

    def test_index_access_classifies_read_write_and_guard_candidate(self):
        src = "void f() {\n buf[i] = 1;\n int y = buf[j];\n}"
        g = graph([cnode(1,"<operator>.indexAccess","buf[i]",2,2),
                   cnode(2,"<operator>.indexAccess","buf[j]",3,10)])
        kinds = {p["kind"] for p in extract_patterns(g,src)}
        self.assertIn("buffer_underwrite",kinds)
        self.assertIn("buffer_underread",kinds)
        self.assertEqual(candidate_kinds(dict(name="memcpy")), ["sensitive_api"])

    def test_align_patterns_and_keep_ambiguous_status(self):
        src = "int f(){\n memcpy(dst, src, 9);\n return 0;\n}"
        g = graph([cnode(1,"f","f",1,5,"METHOD"),
                   cnode(2,"memcpy","memcpy(dst, src, 9)",2,2),
                   cnode(3,"src","src",2,14,"IDENTIFIER")],
                  [("1","2","AST"),("1","2","CFG"),("2","3","AST")])
        result = align_graph(g,src)
        self.assertEqual(result["status"], "ok")
        self.assertIn("copy_length_relation",result["pattern_counts"])
        self.assertIn("unknown_bound", {p["status"] for p in result["patterns"]})
        self.assertTrue(result["counts"]["copy_length_relation"] > 0)
        failed = align_graph(graph([cnode(4,"f","WRONG",1,5)]),"f(x);")
        self.assertEqual(failed["status"],"incomplete_graph")
        self.assertEqual(failed["nodes"][0]["alignment"],"location_token")

    def test_shuffle_preserves_per_type_marginals(self):
        row = dict(raw_source="abcdefghijklmnopqrst",program_relations=dict(status="ok",
            nodes=[dict(span=[i,i+1]) for i in range(20)],
            edges=[[i,i+1,i%3] for i in range(12)]))
        real,_ = relation_tokens(row,CodeTokenizer(),2025,"real")
        shuf,info = relation_tokens(row,CodeTokenizer(),2025,"shuffled")
        for k in range(3):
            self.assertEqual(Counter(a for a,b,t in real if t==k),
                             Counter(a for a,b,t in shuf if t==k))
            self.assertEqual(Counter(b for a,b,t in real if t==k),
                             Counter(b for a,b,t in shuf if t==k))
        self.assertGreater(info["changed_endpoints"],0)
        self.assertEqual(shuf,relation_tokens(dict(row,label=1),CodeTokenizer(),2025,"shuffled")[0])

    def test_pair_attention_changes_under_rewiring(self):
        torch.manual_seed(42)
        module=StaticRelationAttention(8)
        hidden=torch.randn(1,6,8,requires_grad=True)
        owners=torch.tensor([0])
        positions=torch.arange(6).unsqueeze(0)
        base=hidden.mean(dim=1)
        pooling=torch.ones(1,6)
        real=torch.tensor([[0,0,1,0],[0,2,3,0]],dtype=torch.long)
        shuffled=torch.tensor([[0,0,3,0],[0,2,1,0]],dtype=torch.long)
        orig=module(hidden,positions,owners,pooling,real,1,base)
        moved=module(hidden,positions,owners,pooling,shuffled,1,base)
        self.assertGreater(float((orig-moved).abs().max().detach()),1e-6)
        self.assertGreater(float((orig-base).norm().detach()),1e-3)
        (orig.square().sum()).backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertGreater(float(module.pair.weight.grad.abs().sum()),0)

    def test_overlapping_window_positions_have_unit_weight(self):
        torch.manual_seed(42)
        module=StaticRelationAttention(8)
        hidden=torch.randn(1,3,8,requires_grad=True)
        owners=torch.tensor([0])
        positions=torch.tensor([[0,1,2]])
        edges=torch.tensor([[0,0,2,0],[0,1,2,1]])
        base=torch.randn(1,8)
        orig=module(hidden,positions,owners,torch.ones(1,3),edges,1,base)
        repeated=module(hidden.expand(2,-1,-1),positions.expand(2,-1),
                        torch.tensor([0,0]),torch.full((2,3),.5),edges,1,base)
        torch.testing.assert_close(orig,repeated)
        repeated.square().sum().backward()
        self.assertGreater(float(hidden.grad.abs().sum()),0)

    def test_real_model_forward_backward_and_strict_reload(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"base";tiny_base(path)
            row=dict(sample_key="a",raw_source="int f(int x){return x+1;}",
                program_relations=dict(status="ok",
                    nodes=[dict(span=[4,5]),dict(span=[20,21])],
                    edges=[[0,1,0],[1,0,3]]))
            core.seed_everything(42)
            plain=core.CodeBERT(path,"cpu").eval()
            initial=[]
            counts=[]
            for mode in ("sequence","shuffled","real"):
                core.seed_everything(42)
                model=core.CodeBERT(path,"cpu",relation_mode=mode).eval()
                builder=core.InputBuilder(CodeTokenizer(),relation_mode=mode)
                inputs=builder.codebert_batch([row],device="cpu")
                initial.append({k:v.detach().clone() for k,v in
                                model.task_modules["relations"].state_dict().items()})
                counts.append(sum(p.numel() for p in model.parameters()))
                optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
                for _ in range(2):
                    optimizer.zero_grad()
                    logits=model(*inputs)
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits,torch.ones_like(logits)).backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                        for p in model.parameters()))
                    optimizer.step()
                self.assertGreater(float(model.task_modules["relations"].pair.weight.grad.abs().sum()),0)
                self.assertGreater(float(model.encoder.embeddings.word_embeddings.weight.grad.abs().sum()),0)
                saved=Path(tmp)/"checkpoint.pt"
                checkpoint=dict(model_path=str(path),state_dict=model.state_dict(),
                                static_relations=dict(mode=mode))
                torch.save(checkpoint,saved)
                restored=core.load_model(torch.load(saved,weights_only=False),"cpu")
                torch.testing.assert_close(model(*inputs),restored(*inputs),atol=0,rtol=0)
            self.assertEqual(len(set(counts)),1)
            for other in initial[1:]:
                for key in initial[0]:
                    torch.testing.assert_close(initial[0][key],other[key],atol=0,rtol=0)

    def test_cpg_cache_reuses_raw_graph_when_matching_rules_change(self):
        import hashlib
        import json
        from unittest.mock import patch
        from vulnmechanism import static_analysis as analysis

        source = "int f(){return 0;}"
        row = dict(sample_key="bigvul:1", raw_source=source, file_name="example.c")
        script = Path(analysis.__file__).with_name("static_extract.sc")
        namespace = hashlib.sha256(analysis.JOERN_VERSION.encode() + script.read_bytes()).hexdigest()[:12]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "cache" / "cpg" / namespace
            directory.mkdir(parents=True)
            raw = graph([
                cnode(1, "f", "f", 1, 5, "METHOD"),
                cnode(2, "<operator>.return", "return 0", 1, 9)
            ], [("1", "2", "AST"), ("1", "2", "CFG")])
            (directory / (analysis.cache_key(row) + ".json")).write_text(json.dumps(raw))
            with patch.object(analysis, "run_process",
                              side_effect=AssertionError("Joern should not run for a cached CPG")):
                parsed, costs = analysis.extract([row], root)
                self.assertEqual(costs["raw_cpg_hits"], 1)
                self.assertEqual(costs["unique_uncached"], 0)
                self.assertEqual(parsed["bigvul:1"]["status"], "ok")
                again, costs2 = analysis.extract([row], root)
                self.assertEqual(costs2["aligned_hits"], 1)
                self.assertEqual(again, parsed)

    def test_relation_training_updates_encoder_and_attention_without_test(self):
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch
        from vulnmechanism.data import write_jsonl
        from vulnmechanism.training import TRAINING, train
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiny_base(root / "base")
            rows = []
            for i in range(4):
                src = f"int f{i}(int x){{return x + {i};}}"
                rows.append(dict(sample_key=f"train:{i}", dataset="bigvul",
                                 split="train", label=i % 2, raw_source=src,
                                 program_relations=dict(
                                     status="ok",
                                     nodes=[dict(span=[4,6]), dict(span=[13,14])],
                                     edges=[[0,1,0],[1,0,3]])))
            write_jsonl(root / "data" / "bigvul" / "train.jsonl", rows)
            write_jsonl(root / "data" / "bigvul" / "valid.jsonl", [
                dict(rows[i], sample_key=f"valid:{i}", split="valid") for i in range(4)
            ])
            write_jsonl(root / "data" / "bigvul" / "test.jsonl",
                        [dict(raw_source="FORBIDDEN_HELDOUT")])
            with patch.object(core.AutoTokenizer, "from_pretrained",
                              return_value=CodeTokenizer()), redirect_stdout(io.StringIO()):
                summary = train(root / "data", root / "runs", root / "base",
                                "bigvul", "bce", device="cpu", relation_mode="real",
                                config=dict(TRAINING, epochs=1, batch_size=2,
                                            gradient_accumulation=1))
            self.assertEqual(summary["optimizer_steps"], 2)
            cp = torch.load(root / "runs" / "bigvul" / "bce" / "best.pt",
                            map_location="cpu", weights_only=False)
            self.assertEqual(cp["static_relations"]["mode"], "real")
            model = core.load_model(cp, torch.device("cpu"))
            self.assertIn("relations", model.task_modules)


if __name__ == "__main__":
    unittest.main()
