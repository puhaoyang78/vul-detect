"""CPU tests: real graph math/backprop, mocked Qwen/Joern boundaries.

Run: python -m unittest discover -s tests -p 'test_cfg*.py' -v
These tests do not download a model, run Joern, or execute dataset C/C++ code.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import random
import re
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from vulnmechanism import cfg_data as data
from vulnmechanism import cfg_experiment as exp
from vulnmechanism import cfg_metrics as metric
from vulnmechanism.cfg_network import (AttributeCFGEncoder, GraphBatch,
                                      SourceGraphClassifier, build_model, collate_graphs)
from vulnmechanism.rdrop import binary_rdrop_loss
from vulnmechanism.cpg import (
    FunctionGraph, GraphEdge, GraphNode, _prepare_joern_source, resolve_target_graph,
)


def fixture_graph(literal="4"):
    def n(key, kind, code, name=None, order=None, dtype=None):
        props = dict(kind=kind, LINE_NUMBER=int(key), COLUMN_NUMBER=1)
        if name is not None:
            props["NAME"] = name
        if order is not None:
            props.update(ORDER=order, ARGUMENT_INDEX=order)
        if dtype is not None:
            props["TYPE_FULL_NAME"] = dtype
        return dict(id=key, label=name or kind, code=code, properties=props)
    nodes = [n("0", "METHOD", "foo", "foo"),
             n("1", "CALL", f"x=malloc({literal})", "<operator>.assignment"),
             n("2", "IDENTIFIER", "x", "x", 1, "char *"),
             n("3", "CALL", f"malloc({literal})", "malloc", 2, "void *"),
             n("4", "LITERAL", literal, order=1, dtype="int"),
             n("5", "CALL", "y=x+1", "<operator>.assignment"),
             n("6", "IDENTIFIER", "y", "y", 1, "int"),
             n("7", "CALL", "x+1", "<operator>.addition", 2, "int"),
             n("8", "IDENTIFIER", "x", "x", 1, "int"),
             n("9", "LITERAL", "1", order=2, dtype="int"),
             n("10", "RETURN", "return y"), n("11", "METHOD_RETURN", "RET")]
    ast = [("0", "1"), ("1", "2"), ("1", "3"), ("3", "4"),
           ("0", "5"), ("5", "6"), ("5", "7"), ("7", "8"), ("7", "9"),
           ("0", "10"), ("0", "11")]
    cfg = [("0", "1"), ("1", "5"), ("5", "10"), ("10", "11")]
    return dict(nodes=nodes, edges=[dict(kind=kind, source=s, target=t)
                                    for kind, pairs in (("AST", ast), ("CFG", cfg)) for s, t in pairs])


def positioned_graph(source):
    begin = source.index("return")
    code = source[begin:source.index(";", begin) + 1]
    nodes = [
        dict(id="0", label="METHOD", code="f", properties=dict(kind="METHOD")),
        dict(id="1", label="RETURN", code=code,
             properties=dict(kind="RETURN", OFFSET=begin, OFFSET_END=begin+len(code),
                             LINE_NUMBER=1, COLUMN_NUMBER=begin+1)),
        dict(id="2", label="METHOD_RETURN", code="RET", properties=dict(kind="METHOD_RETURN")),
    ]
    edges = [dict(kind="AST", source="0", target=x) for x in ("1", "2")]
    edges += [dict(kind="CFG", source=a, target=b) for a,b in (("0", "1"), ("1", "2"))]
    return dict(nodes=nodes, edges=edges)


def raw_graph(literal="4"):
    g = fixture_graph(literal)
    return FunctionGraph("foo", {n["id"]: GraphNode(n["id"], n["label"], n["code"], n["properties"])
                                 for n in g["nodes"]},
                         tuple(GraphEdge(e["kind"], e["source"], e["target"]) for e in g["edges"]),
                         {"status": "accepted"})


def fixture_rows():
    rows = []
    for i, split in enumerate(["train"]*5 + ["valid"]*2 + ["test"]*2):
        rows.append(dict(schema_version=9, sample_key=f"primevul:{i}", dataset="primevul", split=split,
                         label=i % 2, raw_source=f"int f{i}(){{return {i};}}", resolved_language="c",
                         language="c", function_name=f"f{i}", cpg_relations="CFG|a|b",
                         mechanism_context="none", mechanism_items=[], cpg_quality={}))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))


def graph_views(rows):
    return {r["sample_key"]: data.abstract_cfg(fixture_graph(str(i+4))) for i, r in enumerate(rows)}


class DataTests(unittest.TestCase):
    def test_abstract_attributes_are_assignment_local(self):
        view = data.abstract_cfg(fixture_graph())
        index = view["node_ids"].index("1")
        self.assertEqual(view["signatures"][index], ['["malloc"]', '["char *"]', '["4"]', '[]'])
        index = view["node_ids"].index("5")
        self.assertEqual(view["signatures"][index], ['[]', '["int"]', '["1"]', '["addition"]'])
        self.assertEqual(view["definition_count"], 2)
        self.assertEqual(view["signatures"][0], ["[]"]*4)

    def test_ddg_uses_cfg_nodes_and_reports_each_skip_reason(self):
        graph = fixture_graph()
        graph["edges"].extend(dict(kind="DDG", source=source, target=target)
                              for source, target in (("1", "5"), ("2", "5"),
                                                     ("5", "3"), ("2", "3")))
        view = data.abstract_cfg(graph)
        self.assertEqual(view["ddg_edges"], [(view["node_ids"].index("1"),
                                              view["node_ids"].index("5"))])
        self.assertEqual(view["ddg_audit"], {
            "raw_edges": 4, "usable_edges": 1, "skipped_source_outside_cfg": 1,
            "skipped_target_outside_cfg": 1, "skipped_both_outside_cfg": 1,
        })

    def test_literals_do_not_collapse(self):
        left, right = [data.abstract_cfg(fixture_graph(x)) for x in ("4", "999")]
        self.assertNotEqual(left["signatures"], right["signatures"])

    def test_no_definition_is_not_discarded(self):
        g = fixture_graph()
        for n in g["nodes"]:
            if n["properties"].get("NAME") == "<operator>.assignment":
                n["properties"]["NAME"] = "ordinary_call"
        v = data.abstract_cfg(g)
        self.assertEqual(v["definition_count"], 0)
        self.assertTrue(v["node_ids"])

    def test_repeated_text_nodes_stay_distinct(self):
        g = fixture_graph()
        g["nodes"][5]["code"] = g["nodes"][1]["code"]
        view = data.abstract_cfg(g)
        self.assertIn("1", view["node_ids"])
        self.assertIn("5", view["node_ids"])
        self.assertEqual(len(view["node_ids"]), 5)

    def test_raw_properties_and_all_current_edges_roundtrip(self):
        g = raw_graph()
        self.assertEqual(data.serialize_graph(g), fixture_graph())
        left = GraphNode("1", "IDENTIFIER", "x")
        right = GraphNode("1", "IDENTIFIER", "x", {"TYPE_FULL_NAME": "int"})
        self.assertEqual(left, right)
        self.assertEqual(hash(left), hash(right))

    def test_graph_validation_missing_properties_and_dangling_edges(self):
        for mutation in ("properties", "edges", "ids"):
            g = fixture_graph()
            if mutation == "properties":
                g["nodes"][0]["properties"] = {}
            elif mutation == "edges":
                g["edges"][0]["target"] = "missing"
            else:
                g["nodes"][1]["id"] = "0"
            with self.assertRaises(ValueError):
                data.validate_graph(g)

    def test_vocab_train_only_and_unknown(self):
        rows = fixture_rows()
        views = graph_views(rows)
        vocab = data.AttributeVocabulary.fit(rows[:5], views)
        self.assertEqual(vocab.values, data.AttributeVocabulary(vocab.values).values)
        valid = vocab.encode(views[rows[5]["sample_key"]])
        self.assertIn(1, [x[2] for x in valid])
        with self.assertRaises(ValueError):
            data.AttributeVocabulary.fit(rows, views)

    def test_vocab_does_not_use_metadata_or_node_ids(self):
        rows = fixture_rows()[:5]
        views = graph_views(rows)
        expected = data.AttributeVocabulary.fit(rows, views).values
        other = copy.deepcopy(rows)
        for r in other:
            r["label"] = 1-r["label"]
            r["cve_id"] = "CVE-SHOULD-NOT-BE-A-FEATURE"
        self.assertEqual(expected, data.AttributeVocabulary.fit(other, views).values)
        self.assertNotIn("CVE-SHOULD", json.dumps(expected))

    def test_cpg_resolution_retains_original_node_metadata(self):
        source = "int f(){return 0;}"
        nodes = {"f": dict(kind="FILE", NAME="one.c", CONTENT=source),
                 "0": dict(kind="METHOD", NAME="f", FILENAME="one.c", CODE=source,
                           LINE_NUMBER=1, LINE_NUMBER_END=1),
                 "1": dict(kind="BLOCK", CODE="{return 0;}"),
                 "2": dict(kind="RETURN", CODE="return 0;", LINE_NUMBER=1, COLUMN_NUMBER=9),
                 "3": dict(kind="LITERAL", CODE="0", TYPE_FULL_NAME="int", ORDER=1),
                 "4": dict(kind="METHOD_RETURN", CODE="RET")}
        edges = [("AST", "0", "1"), ("AST", "1", "2"), ("AST", "2", "3"),
                 ("AST", "0", "4"), ("CFG", "0", "2"), ("CFG", "2", "4")]
        # Isolate the existing syntax dependency; this is a metadata-retention test,
        # not a substitute for Joern/tree-sitter integration validation.
        syntax = types.ModuleType("vulnmechanism.syntax")
        syntax.source_tokens = lambda text: tuple(re.findall(r"\w+|[^\w\s]", text))
        with patch.dict("sys.modules", {"vulnmechanism.syntax": syntax}):
            g = resolve_target_graph(nodes, edges, filename="one.c", source=source)
        self.assertEqual(g.nodes["3"].properties["TYPE_FULL_NAME"], "int")
        self.assertEqual(g.nodes["2"].properties["COLUMN_NUMBER"], 9)

    def test_standalone_cpp_member_specifiers_are_syntax_aware_and_position_preserving(self):
        cases = [
            "void f(int x) override /* rule (1) */ { return; }\n",
            "void f(int x) /* { */ override { return; }\n",
            "void f(int x = int{}) override { return; }\n",
            'void f(const char *s = "{") override { return; }\n',
            "void f() const noexcept override final { return; }\n",
            "void f() & override { return; }\n",
        ]
        for source in cases:
            with self.subTest(source=source):
                prepared = _prepare_joern_source(source, language="cpp", standalone=True)
                self.assertNotEqual(prepared, source)
                self.assertEqual(len(prepared.encode("utf-8")), len(source.encode("utf-8")))
                self.assertEqual(
                    [i for i, byte in enumerate(prepared.encode("utf-8")) if byte == 10],
                    [i for i, byte in enumerate(source.encode("utf-8")) if byte == 10],
                )
                self.assertTrue(prepared.endswith("{ return; }\n"))

        parameter_name = "void f(int final) override { (void)final; }\n"
        prepared = _prepare_joern_source(parameter_name, language="cpp", standalone=True)
        self.assertIn("int final", prepared)
        self.assertIn("(void)final", prepared)
        self.assertNotIn(") override", prepared)

        comment_words = "void f() override /* override final */ { return; }\n"
        prepared = _prepare_joern_source(comment_words, language="cpp", standalone=True)
        self.assertIn("/* override final */", prepared)
        self.assertNotIn("f() override", prepared)

        for source in (
            "struct final {};\nauto make() -> final { return {}; }\n",
            "auto make() -> Box<final> { return {}; }\n",
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    _prepare_joern_source(source, language="cpp", standalone=True),
                    source,
                )

        source = "void f() override { return; }\n"
        self.assertEqual(_prepare_joern_source(source, language="cpp", standalone=False), source)
        self.assertEqual(_prepare_joern_source(source, language="c", standalone=True), source)

    def test_build_resume_retries_failed_only_and_preserves_dataset(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); src, out = root/"source.jsonl", root/"graphs.jsonl"
            rows = fixture_rows()[:3]; write_jsonl(src, rows); before = src.read_bytes()
            calls = []
            def first(requests, **kwargs):
                calls.append(len(requests))
                return [raw_graph(), subprocess.TimeoutExpired("joern", 1), raw_graph()]
            report = data.build_graphs(str(src), str(out), batch_size=3, extractor=first)
            self.assertEqual(report["failed"], 1)
            def retry(requests, **kwargs):
                calls.append(len(requests)); return [raw_graph() for _ in requests]
            report = data.build_graphs(str(src), str(out), batch_size=3, extractor=retry)
            self.assertTrue(report["complete"]); self.assertEqual(calls, [3, 1])
            self.assertEqual(src.read_bytes(), before)
            exported = data.read_jsonl(out)
            self.assertEqual(len(data.load_graphs(out, rows)), 3)
            self.assertTrue(all(
                row["preprocessing_version"] == data.JOERN_SOURCE_PREPROCESSING_VERSION
                for row in exported
            ))
            self.assertTrue(all(
                row["original_source_sha256"] == row["parsed_source_sha256"]
                and row["preprocessing_applied"] is False
                for row in exported
            ))
            meta = json.loads(Path(str(out) + ".meta.json").read_text())
            self.assertEqual(
                meta["preprocessing_version"], data.JOERN_SOURCE_PREPROCESSING_VERSION
            )
            data.build_graphs(str(src), str(out), extractor=lambda *a, **k: self.fail("no reparse"))

    def test_build_catches_raised_timeout_and_refuses_incomplete_training_cohort(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp); src, out = root/"src.jsonl", root/"g.jsonl"
            rows=fixture_rows()[:2];write_jsonl(src,rows)
            def fail(*a, **kw):
                raise subprocess.TimeoutExpired("joern", 1)
            r=data.build_graphs(str(src),str(out),extractor=fail)
            self.assertEqual(r["failed"], 2)
            with self.assertRaises(ValueError):
                data.load_graphs(out, rows)

    def test_cache_identity_and_original_output_protection(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp); src, out = root/"src.jsonl",root/"g.jsonl"
            rows=fixture_rows()[:1];write_jsonl(src,rows)
            with self.assertRaises(ValueError):
                data.build_graphs(str(src),str(src))
            data.build_graphs(str(src),str(out),extractor=lambda req, **k: [raw_graph()])
            changed=copy.deepcopy(rows);changed[0]["label"]=1-changed[0]["label"]
            with self.assertRaises(ValueError):
                data.load_graphs(out,changed)
            write_jsonl(src,changed)
            with self.assertRaises(ValueError):
                data.build_graphs(str(src),str(out),extractor=lambda *a,**k: self.fail("stale"))

    def test_tail_recovery_only_recovers_partial_last_line(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path=Path(tmp)/"g.jsonl";path.write_bytes(b'{"a":1}\n{"part')
            data.repair_tail(path);self.assertEqual(path.read_bytes(),b'{"a":1}\n')
            path.write_bytes(b'{"a":1}');data.repair_tail(path)
            self.assertEqual(path.read_bytes(),b'{"a":1}\n')
            path.write_bytes(b'{"broken\n{"a":2}\n')
            with self.assertRaises(json.JSONDecodeError):
                data.repair_tail(path)


class MetricTests(unittest.TestCase):
    def test_perfect_tied_and_single_class_auc(self):
        self.assertEqual(metric.metrics([0,1],[0.1,0.9])["auc"],1.0)
        self.assertEqual(metric.metrics([0,1,0,1],[0.5]*4)["auc"],0.5)
        self.assertIsNone(metric.metrics([0,0],[0.1,0.2])["auc"])

    def test_float32_boundary_matches_original_tensor_comparison(self):
        labels=[0,1,0,1,1,0]
        scores=torch.tensor([.05,.1,.3,.7,.9,.95],dtype=torch.float32)
        for value in range(5,96):
            threshold=value/100
            expected=scores >= threshold
            got=metric.metrics(labels,scores.tolist(),threshold)
            self.assertEqual(got["tp"],sum(int(p) and y for p,y in zip(expected,labels)))
            self.assertEqual(got["fp"],sum(int(p) and not y for p,y in zip(expected,labels)))

    def test_threshold_selection_and_invalid_inputs(self):
        threshold, m=metric.select_threshold([0,0,1,1],[.1,.2,.3,.4])
        self.assertEqual(m["mcc"],1.0);self.assertGreater(threshold,.2)
        for labels,scores in [([],[]),([0],[float("nan")]),([2],[.2]),([0],[2.0])]:
            with self.assertRaises(ValueError):
                metric.metrics(labels,scores)

    def test_all_four_changes_and_identity_checks(self):
        b=[];c=[]
        for i,(y,old,new) in enumerate([(1,0,1),(0,1,0),(1,1,0),(0,0,1)]):
            common=dict(sample_key=str(i),label=y,split="valid",source_sha256=str(i),threshold=.5)
            b.append(dict(common,prediction=old,score=.8 if old else .2))
            c.append(dict(common,prediction=new,score=.8 if new else .2))
        report=metric.paired_changes(b,c)
        self.assertEqual(set(report["counts"].values()),{1})
        self.assertEqual(report["net_corrected"],0)
        self.assertEqual(metric.paired_changes(b,list(reversed(c))),report)
        with self.assertRaises(ValueError):
            metric.paired_changes(b,c[:-1])
        c[0]["label"]=0
        with self.assertRaises(ValueError):
            metric.paired_changes(b,c)


class RDropLossTests(unittest.TestCase):
    def test_symmetric_bernoulli_kl_matches_independent_reference(self):
        first = torch.tensor([-3.0, -0.4, 0.8, 2.5], dtype=torch.float16)
        second = torch.tensor([1.5, 0.2, -1.1, 2.0], dtype=torch.float16)
        labels = torch.tensor([0, 1, 1, 0])
        total, bce, kl = binary_rdrop_loss(first, second, labels, alpha=1.0)
        reference = 0.5 * (
            torch.distributions.kl_divergence(
                torch.distributions.Bernoulli(logits=first.double()),
                torch.distributions.Bernoulli(logits=second.double())) +
            torch.distributions.kl_divergence(
                torch.distributions.Bernoulli(logits=second.double()),
                torch.distributions.Bernoulli(logits=first.double()))).mean()
        expected_bce = 0.5 * (
            nn.functional.binary_cross_entropy_with_logits(first.float(), labels.float()) +
            nn.functional.binary_cross_entropy_with_logits(second.float(), labels.float()))
        self.assertEqual(total.dtype, torch.float32)
        torch.testing.assert_close(kl.double(), reference, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(bce, expected_bce)
        torch.testing.assert_close(total, bce + kl)
        control, control_bce, control_kl = binary_rdrop_loss(first, second, labels, alpha=0.0)
        torch.testing.assert_close(control, expected_bce)
        torch.testing.assert_close(control_bce, expected_bce)
        torch.testing.assert_close(control_kl, kl)
        _, _, zero_kl = binary_rdrop_loss(first, first, labels, alpha=1.0)
        self.assertEqual(zero_kl.item(), 0.0)
        extreme, _, extreme_kl = binary_rdrop_loss(
            torch.tensor([-100.0, 100.0]), torch.tensor([100.0, -100.0]), labels[:2], alpha=1.0)
        self.assertTrue(torch.isfinite(extreme))
        self.assertTrue(torch.isfinite(extreme_kl))
        self.assertGreater(extreme_kl.item(), 0)

    def test_both_branches_receive_bce_and_kl_gradients(self):
        first = torch.tensor([-1.4, 0.7, 2.1], requires_grad=True)
        second = torch.tensor([0.5, -0.8, 1.3], requires_grad=True)
        labels = torch.tensor([0.0, 1.0, 1.0])
        total, bce, kl = binary_rdrop_loss(first, second, labels, alpha=1.0)
        total.backward(retain_graph=True)
        self.assertGreater(first.grad.abs().sum().item(), 0)
        self.assertGreater(second.grad.abs().sum().item(), 0)
        first.grad = second.grad = None
        kl.backward(retain_graph=True)
        self.assertGreater(first.grad.abs().sum().item(), 0)
        self.assertGreater(second.grad.abs().sum().item(), 0)
        first.grad = second.grad = None
        control, _, _ = binary_rdrop_loss(first, second, labels, alpha=0.0)
        control.backward()
        self.assertGreater(first.grad.abs().sum().item(), 0)
        self.assertGreater(second.grad.abs().sum().item(), 0)

    def test_dual_loss_preserves_sample_weighted_gradient_accumulation(self):
        torch.manual_seed(42)
        full = nn.Linear(3, 1)
        micro = copy.deepcopy(full)
        x = torch.randn(5, 3)
        perturbation = torch.randn(5, 3) * 0.2
        labels = torch.tensor([0., 1., 1., 0., 1.])
        def loss(model, start, end):
            first = model(x[start:end]).squeeze(-1)
            second = model(x[start:end] + perturbation[start:end]).squeeze(-1)
            return binary_rdrop_loss(first, second, labels[start:end], alpha=1.0)[0]
        loss(full, 0, 5).backward()
        for start in range(0, 5, 2):
            end = min(start + 2, 5)
            (loss(micro, start, end) * (end-start)/5).backward()
        torch.testing.assert_close(full.weight.grad, micro.weight.grad)
        torch.testing.assert_close(full.bias.grad, micro.bias.grad)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(32,8);self.adapter=nn.Linear(8,8,bias=False)
        self.embedding.weight.requires_grad_(False)
        self.calls = 0
    def forward(self,input_ids,attention_mask,use_cache=False):
        self.calls += 1
        h=self.embedding(input_ids);return types.SimpleNamespace(last_hidden_state=h+self.adapter(h))


class TinySource(nn.Module):
    def __init__(self,*args,device="cpu",**kwargs):
        super().__init__();self.encoder=TinyEncoder()
        self.task_modules=nn.ModuleDict({"classifier":nn.Linear(8,1)})
        self.to(device)
    def forward(self,input_ids,attention_mask):
        hidden=self.encoder(input_ids,attention_mask).last_hidden_state
        mask=attention_mask.unsqueeze(-1).to(hidden.dtype)
        return self.task_modules["classifier"]((hidden*mask).sum(1)/mask.sum(1).clamp_min(1)).squeeze(-1)


class DropoutTinyEncoder(TinyEncoder):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.draws = []

    def forward(self, input_ids, attention_mask, use_cache=False):
        self.calls += 1
        hidden = self.embedding(input_ids)
        dropped = self.dropout(hidden)
        self.draws.append(dropped.detach().clone())
        return types.SimpleNamespace(last_hidden_state=hidden + self.adapter(dropped))


class DropoutTinySource(TinySource):
    created = []

    def __init__(self, *args, device="cpu", **kwargs):
        super().__init__(*args, device=device, **kwargs)
        self.encoder = DropoutTinyEncoder()
        self.to(device)
        self.initial_state = {k: value.detach().clone() for k, value in self.state_dict().items()}
        self.created.append(self)


class TinyTokenizer:
    pad_token_id=0
    eos_token_id=1
    is_fast=True
    @classmethod
    def from_pretrained(cls,*args,**kwargs):
        return cls()
    def __call__(self,text,*,add_special_tokens=False,truncation=False,max_length=None,
                 return_offsets_mapping=False):
        ids=[ord(c)%30+2 for c in text]
        if truncation:ids=ids[:max_length]
        result={"input_ids":ids}
        if return_offsets_mapping:result["offset_mapping"]=[(i,i+1) for i in range(len(ids))]
        return result


class TinyInputBuilder:
    allow_test=False
    def __init__(self,tokenizer,*,source_max_length,context_max_length):
        self.tokenizer=tokenizer
        self.source_max_length=source_max_length
        self.source_prefix=[29,28]
    def _encode(self,text,max_length=None):
        return [ord(c)%30+2 for c in text][:max_length]
    def sequence_batch(self,records,*,variant,excluded_groups,device):
        if not self.allow_test and any(r["split"]=="test" for r in records):
            raise AssertionError("test records reached model during training")
        ids=[self.source_prefix+self._encode(r["raw_source"],self.source_max_length)+[1]
             for r in records]
        tensor=torch.zeros((len(ids),max(map(len,ids))),dtype=torch.long,device=device)
        for i,seq in enumerate(ids):tensor[i,:len(seq)]=torch.tensor(seq,device=device)
        return tensor,tensor.ne(0).long()
    def source_alignment_batch(self,records,*,device):
        ids,mask=self.sequence_batch(records,variant="baseline",excluded_groups=(),device=device)
        offsets=[[(i,i+1) for i in range(min(len(r["raw_source"]),self.source_max_length))]
                 for r in records]
        return ids,mask,offsets


def fake_base():
    def seed(n):random.seed(n);torch.manual_seed(n)
    def cpu(s):return {k:v.detach().cpu().clone() for k,v in s.items()}
    api=types.SimpleNamespace(SequenceVulnerabilityClassifier=TinySource,AutoTokenizer=TinyTokenizer,
                              InputBuilder=TinyInputBuilder,_seed_everything=seed,_resolve_device=torch.device,
                              _cpu_state=cpu,get_peft_model_state_dict=lambda e:e.state_dict(),
                              set_peft_model_state_dict=lambda e,s:e.load_state_dict(s))
    def train(dataset,output,**kw):
        # Fixture for checking delegation/orchestration only; not a Qwen training substitute.
        seed(kw["seed"]);model=TinySource();optimizer=torch.optim.SGD(model.parameters(),lr=.03)
        builder=TinyInputBuilder(None,source_max_length=kw["source_max_length"],context_max_length=384)
        rows=[r for r in kw["records"] if r["split"]=="train"]
        for _ in range(kw["epochs"]):
            ids,mask=builder.sequence_batch(rows,variant="baseline",excluded_groups=(),device="cpu")
            loss=nn.functional.binary_cross_entropy_with_logits(model(ids,mask),torch.tensor([r["label"] for r in rows]).float())
            optimizer.zero_grad();loss.backward();optimizer.step()
        checkpoint=dict(adapter_state=cpu(model.encoder.state_dict()),task_state=cpu(model.task_modules.state_dict()),
                        decision_threshold=.5,source_max_length=kw["source_max_length"])
        torch.save(checkpoint,output);return checkpoint
    api.train_model=unittest.mock.Mock(side_effect=train)
    def predict(path,rows,**kw):
        cp=torch.load(path,map_location="cpu",weights_only=False);model=TinySource()
        model.encoder.load_state_dict(cp["adapter_state"]);model.task_modules.load_state_dict(cp["task_state"])
        builder=TinyInputBuilder(None,source_max_length=cp["source_max_length"],context_max_length=384)
        ids,mask=builder.sequence_batch(rows,variant="baseline",excluded_groups=(),device="cpu")
        with torch.no_grad():return model(ids,mask)
    api.predict_checkpoint=predict
    return api


def tiny_config(variant="cfg"):
    return dict(variant=variant,model_path="tiny-test-only",source_max_length=8,graph_hidden_size=8,
                graph_steps=2,lora_r=2,lora_alpha=4,lora_dropout=0.0,seed=42,epochs=2,
                batch_size=2,gradient_accumulation=2,learning_rate=.01,graph_learning_rate=.02,
                weight_decay=.01,log_every=1)


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        self.rows=fixture_rows();self.views=graph_views(self.rows)
        self.vocab=data.AttributeVocabulary.fit(self.rows[:5],self.views)
        self.view=self.views[self.rows[0]["sample_key"]]
        self.batch=collate_graphs([self.view],[self.vocab.encode(self.view)])
        torch.manual_seed(42)

    def test_b_c_have_identical_parameters(self):
        b=AttributeCFGEncoder(self.vocab.sizes(),hidden_size=8,steps=2,mode="attributes")
        torch.manual_seed(42)
        c=AttributeCFGEncoder(self.vocab.sizes(),hidden_size=8,steps=2,mode="cfg")
        self.assertEqual(sum(p.numel() for p in b.parameters()),sum(p.numel() for p in c.parameters()))
        for key,value in b.state_dict().items():self.assertTrue(torch.equal(value,c.state_dict()[key]))

    def test_attributes_ignore_edges_but_cfg_uses_direction(self):
        empty=GraphBatch(self.batch.attributes,torch.empty((2,0),dtype=torch.long),self.batch.ptr)
        reverse=GraphBatch(self.batch.attributes,self.batch.edges.flip(0),self.batch.ptr)
        b=AttributeCFGEncoder(self.vocab.sizes(),hidden_size=8,steps=2,mode="attributes")
        self.assertTrue(torch.equal(b(self.batch),b(empty)))
        c=AttributeCFGEncoder(self.vocab.sizes(),hidden_size=8,steps=2,mode="cfg")
        self.assertFalse(torch.allclose(c(self.batch),c(empty)))
        self.assertFalse(torch.allclose(c(self.batch),c(reverse)))

    def test_batching_does_not_exchange_between_graphs(self):
        v2=self.views[self.rows[1]["sample_key"]]
        batch=collate_graphs([self.view,v2],[self.vocab.encode(self.view),self.vocab.encode(v2)])
        c=AttributeCFGEncoder(self.vocab.sizes(),hidden_size=8,steps=2)
        individual=torch.cat([c(self.batch),c(collate_graphs([v2],[self.vocab.encode(v2)]))])
        torch.testing.assert_close(c(batch),individual)

    def test_node_permutation_invariance(self):
        batch=self.batch;n=batch.attributes.shape[0];perm=torch.randperm(n);inverse=torch.argsort(perm)
        other=GraphBatch(batch.attributes[perm],inverse[batch.edges],batch.ptr)
        c=AttributeCFGEncoder(self.vocab.sizes(),hidden_size=8,steps=2)
        torch.testing.assert_close(c(batch),c(other))

    def test_ddg_shuffle_is_function_local_with_exact_degree_multiset(self):
        views = []
        for i in range(2):
            graph = fixture_graph()
            graph["edges"].extend(dict(kind="DDG", source=source, target=target)
                                  for source, target in (("1", "5"), ("5", "10"), ("1", "10")))
            view = data.abstract_cfg(graph)
            before_cfg = list(view["edges"])
            shuffled = exp.shuffle_ddg_edges(view, 42, f"function-{i}")
            self.assertEqual(view["edges"], before_cfg)
            self.assertEqual(len(shuffled), len(view["ddg_edges"]))
            self.assertEqual(exp._ddg_degree_multiset(shuffled, len(view["node_ids"])),
                             exp._ddg_degree_multiset(view["ddg_edges"], len(view["node_ids"])))
            view["ddg_shuffled_edges"] = shuffled
            views.append(view)
        batch = collate_graphs(views, [self.vocab.encode(v) for v in views],
                               ddg_edges=[v["ddg_shuffled_edges"] for v in views])
        self.assertTrue(torch.equal(batch.edges[:, :len(views[0]["edges"])], self.batch.edges))
        for source, target in batch.ddg_edges.T.tolist():
            self.assertEqual(source < batch.ptr[1], target < batch.ptr[1])
        self.assertEqual(batch.ddg_edges.shape[1], 6)

    def test_f_g_share_initialization_and_ddg_direction_backpropagates(self):
        graph = fixture_graph()
        graph["edges"].extend(dict(kind="DDG", source=source, target=target)
                              for source, target in (("1", "5"), ("5", "10")))
        view = data.abstract_cfg(graph)
        batch = collate_graphs([view], [self.vocab.encode(view)], ddg_edges=[view["ddg_edges"]])
        reverse = GraphBatch(batch.attributes, batch.edges, batch.ptr,
                             ddg_edges=batch.ddg_edges.flip(0))
        api = fake_base()
        models = {}
        for variant in ("cfg", "cfg_ddg", "cfg_ddg_shuffled"):
            api._seed_everything(42)
            models[variant] = build_model(api, tiny_config(variant), self.vocab.sizes(),
                                          torch.device("cpu"), training=True)
        c = models["cfg"].state_dict()
        f = models["cfg_ddg"].state_dict()
        g = models["cfg_ddg_shuffled"].state_dict()
        self.assertEqual(sum(p.numel() for p in models["cfg_ddg"].parameters()),
                         sum(p.numel() for p in models["cfg_ddg_shuffled"].parameters()))
        for key, value in f.items():
            self.assertTrue(torch.equal(value, g[key]), key)
        for key, value in c.items():
            self.assertTrue(torch.equal(value, f[key]), key)
        encoder = models["cfg_ddg"].task_modules["cfg_encoder"]
        self.assertFalse(torch.allclose(encoder(batch), encoder(reverse)))
        model = models["cfg_ddg"]
        with torch.no_grad():
            model.task_modules["cfg_classifier"].weight.fill_(.1)
        ids = torch.tensor([[2, 3, 1]])
        mask = torch.ones_like(ids)
        nn.functional.binary_cross_entropy_with_logits(model(ids, mask, batch),
                                                       torch.ones(1)).backward()
        self.assertEqual(model.encoder.calls, 1)
        self.assertGreater(encoder.ddg_message.weight.grad.abs().sum().item(), 0)
        self.assertGreater(encoder.message.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.encoder.adapter.weight.grad.abs().sum().item(), 0)

    def test_jk_five_round_values_and_original_cfg_last_state(self):
        outputs = {}
        initial = None
        for mode in ("cfg", *exp.JK_VARIANTS):
            torch.manual_seed(42)
            encoder = AttributeCFGEncoder(self.vocab.sizes(), hidden_size=8, steps=5, mode=mode)
            if initial is None:
                initial = {key: value.clone() for key, value in encoder.state_dict().items()}
            else:
                for key, value in initial.items():
                    self.assertTrue(torch.equal(value, encoder.state_dict()[key]), key)
            states = []
            handle = encoder.update.register_forward_hook(lambda _module, _inputs, output: states.append(output))
            output = encoder(self.batch)
            handle.remove()
            self.assertEqual(len(states), 5)
            x = torch.cat([layer(self.batch.attributes[:, i])
                           for i, layer in enumerate(encoder.embedding)], dim=-1)
            if mode == "cfg":
                selected = states[-1]
            elif mode == "cfg_jk_mean":
                selected = (states[0] + states[1] + states[2] + states[3] + states[4]) / 5
            else:
                selected = states[0]
                for state in states[1:]:
                    selected = torch.maximum(selected, state)
            combined = torch.cat((selected, x), dim=-1)
            scores = encoder.pool_gate(combined).squeeze(-1).softmax(dim=0)
            expected = (scores.unsqueeze(-1) * combined).sum(dim=0, keepdim=True)
            torch.testing.assert_close(output, expected)
            outputs[mode] = output
        self.assertFalse(torch.allclose(outputs["cfg"], outputs["cfg_jk_mean"]))
        self.assertFalse(torch.allclose(outputs["cfg"], outputs["cfg_jk_max"]))

    def test_jk_one_round_equals_cfg_and_full_model_initialization(self):
        one_round = {}
        for mode in ("cfg", *exp.JK_VARIANTS):
            torch.manual_seed(42)
            one_round[mode] = AttributeCFGEncoder(self.vocab.sizes(), hidden_size=8,
                                                  steps=1, mode=mode)(self.batch)
        for mode in exp.JK_VARIANTS:
            torch.testing.assert_close(one_round[mode], one_round["cfg"], rtol=0, atol=0)
        api = fake_base()
        models = {}
        for mode in ("cfg", *exp.JK_VARIANTS):
            api._seed_everything(42)
            models[mode] = build_model(api, dict(tiny_config(mode), graph_steps=5),
                                       self.vocab.sizes(), torch.device("cpu"), training=True)
        c = models["cfg"]
        for mode in exp.JK_VARIANTS:
            other = models[mode]
            self.assertEqual(sum(p.numel() for p in c.parameters()),
                             sum(p.numel() for p in other.parameters()))
            self.assertEqual(c.state_dict().keys(), other.state_dict().keys())
            for key, value in c.state_dict().items():
                self.assertTrue(torch.equal(value, other.state_dict()[key]), key)
            self.assertEqual(other.task_modules["cfg_encoder"].steps, 5)

    def test_jk_mean_and_max_backpropagate_through_shared_cfg_and_source(self):
        api = fake_base()
        ids = torch.tensor([[2, 3, 1]])
        mask = torch.ones_like(ids)
        for mode in exp.JK_VARIANTS:
            api._seed_everything(42)
            model = build_model(api, dict(tiny_config(mode), graph_steps=5),
                                self.vocab.sizes(), torch.device("cpu"), training=True)
            with torch.no_grad():
                model.task_modules["cfg_classifier"].weight.fill_(.1)
            encoder = model.task_modules["cfg_encoder"]
            states = []
            def retain_state(_module, _inputs, output):
                output.retain_grad()
                states.append(output)
            handle = encoder.update.register_forward_hook(retain_state)
            logits = model(ids, mask, self.batch)
            handle.remove()
            nn.functional.binary_cross_entropy_with_logits(logits, torch.ones(1)).backward()
            self.assertEqual(model.encoder.calls, 1)
            self.assertEqual(len(states), 5)
            self.assertTrue(any(state.grad is not None and state.grad.abs().sum() > 0
                                for state in states[:-1]))
            self.assertGreater(encoder.embedding[0].weight.grad.abs().sum().item(), 0)
            self.assertGreater(encoder.message.weight.grad.abs().sum().item(), 0)
            self.assertGreater(encoder.update.weight_hh.grad.abs().sum().item(), 0)
            self.assertGreater(model.encoder.adapter.weight.grad.abs().sum().item(), 0)

    def test_dual_configs_use_exact_c_model_and_independent_dropout(self):
        api = fake_base()
        api.SequenceVulnerabilityClassifier = DropoutTinySource
        states = {}
        for variant in ("cfg", "cfg_double_ce", "cfg_rdrop"):
            api._seed_everything(42)
            model = build_model(api, tiny_config(variant), self.vocab.sizes(),
                                torch.device("cpu"), training=True)
            self.assertEqual(model.task_modules["cfg_encoder"].mode, "cfg")
            states[variant] = {k: value.clone() for k, value in model.state_dict().items()}
        self.assertEqual(states["cfg"].keys(), states["cfg_double_ce"].keys())
        for key, value in states["cfg"].items():
            self.assertTrue(torch.equal(value, states["cfg_double_ce"][key]), key)
            self.assertTrue(torch.equal(value, states["cfg_rdrop"][key]), key)
        model.train()
        ids = torch.tensor([[2, 3, 4, 1]])
        mask = torch.ones_like(ids)
        first = model(ids, mask, self.batch)
        second = model(ids, mask, self.batch)
        self.assertEqual(model.encoder.calls, 2)
        self.assertFalse(torch.equal(model.encoder.draws[-2], model.encoder.draws[-1]))
        binary_rdrop_loss(first, second, torch.ones(1), alpha=1.0)[0].backward()
        self.assertGreater(model.encoder.adapter.weight.grad.abs().sum().item(), 0)
        builder = TinyInputBuilder(None, source_max_length=2048, context_max_length=384)
        before = model.encoder.calls
        scores = exp._graph_scores(model, self.rows[:2], builder, self.views,
                                   {r["sample_key"]: self.vocab.encode(self.views[r["sample_key"]])
                                    for r in self.rows[:2]}, batch_size=2, device=torch.device("cpu"))
        self.assertEqual(len(scores), 2)
        self.assertEqual(model.encoder.calls, before + 1)

    def test_invalid_endpoint_fails(self):
        v=copy.deepcopy(self.view);v["edges"].append((0,999))
        with self.assertRaises(ValueError):collate_graphs([v],[self.vocab.encode(v)])

    def test_source_initialization_and_rng_exactly_preserved(self):
        api=fake_base();cfg=tiny_config();api._seed_everything(42);base=TinySource()
        original_rng=torch.get_rng_state().clone()
        ids=torch.tensor([[2,3,1]]);mask=torch.ones_like(ids);expected=base(ids,mask).detach()
        wrapped=SourceGraphClassifier(base,self.vocab.sizes(),hidden_size=8,steps=2,mode="cfg",device="cpu")
        self.assertTrue(torch.equal(original_rng,torch.get_rng_state()))
        self.assertTrue(torch.equal(expected,wrapped(ids,mask,self.batch)))
        api._seed_everything(42);a=build_model(api,dict(cfg,variant="baseline"),None,torch.device("cpu"),training=True)
        api._seed_everything(42);c=build_model(api,cfg,self.vocab.sizes(),torch.device("cpu"),training=True)
        for key,value in a.encoder.state_dict().items():self.assertTrue(torch.equal(value,c.encoder.state_dict()[key]))

    def test_joint_training_gradients_and_checkpoint_roundtrip(self):
        api=fake_base();cfg=tiny_config();model=build_model(api,cfg,self.vocab.sizes(),torch.device("cpu"),training=True)
        ids=torch.tensor([[2,3,1]]);mask=torch.ones_like(ids)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.03)
        previous=model.encoder.adapter.weight.detach().clone()
        for _ in range(2):
            optimizer.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(ids,mask,self.batch),torch.ones(1))
            loss.backward();optimizer.step()
        self.assertFalse(torch.equal(previous,model.encoder.adapter.weight))
        self.assertGreater(model.task_modules["cfg_encoder"].message.weight.grad.abs().sum().item(),0)
        other=build_model(api,cfg,self.vocab.sizes(),torch.device("cpu"),training=False)
        other.load_state_dict(model.state_dict());torch.testing.assert_close(model(ids,mask,self.batch),other(ids,mask,self.batch))

    def test_d_e_identical_parameters_and_directed_behavior(self):
        torch.manual_seed(42)
        d=AttributeCFGEncoder(self.vocab.sizes(),hidden_size=8,steps=2,
                              mode="aligned_attributes",source_hidden_size=8)
        torch.manual_seed(42)
        e=AttributeCFGEncoder(self.vocab.sizes(),hidden_size=8,steps=2,
                              mode="aligned_cfg",source_hidden_size=8)
        self.assertEqual(d.steps,e.steps)
        self.assertEqual(sum(p.numel() for p in d.parameters()),sum(p.numel() for p in e.parameters()))
        for key,value in d.state_dict().items():self.assertTrue(torch.equal(value,e.state_dict()[key]))
        api=fake_base()
        api._seed_everything(42)
        full_d=build_model(api,tiny_config("aligned_attributes"),self.vocab.sizes(),
                           torch.device("cpu"),training=True)
        api._seed_everything(42)
        full_e=build_model(api,tiny_config("aligned_cfg"),self.vocab.sizes(),
                           torch.device("cpu"),training=True)
        self.assertEqual(sum(p.numel() for p in full_d.parameters()),
                         sum(p.numel() for p in full_e.parameters()))
        for key,value in full_d.state_dict().items():
            self.assertTrue(torch.equal(value,full_e.state_dict()[key]))
        batch=collate_graphs([self.view],[self.vocab.encode(self.view)],
                             alignments=[[(0,2),(1,3)]])
        empty=GraphBatch(batch.attributes,torch.empty((2,0),dtype=torch.long),batch.ptr,batch.token_pairs)
        reverse=GraphBatch(batch.attributes,batch.edges.flip(0),batch.ptr,batch.token_pairs)
        hidden=torch.randn(1,6,8)
        no_alignment=GraphBatch(batch.attributes,batch.edges,batch.ptr,
                                torch.empty((0,3),dtype=torch.long))
        original=AttributeCFGEncoder(self.vocab.sizes(),hidden_size=8,steps=2,mode="attributes")
        original.load_state_dict({key:value for key,value in d.state_dict().items()
                                  if key in original.state_dict()})
        torch.testing.assert_close(d(no_alignment,hidden),original(self.batch))
        torch.testing.assert_close(d(batch,hidden),d(empty,hidden))
        self.assertFalse(torch.allclose(e(batch,hidden),e(empty,hidden)))
        self.assertFalse(torch.allclose(e(batch,hidden),e(reverse,hidden)))

    def test_aligned_graph_backpropagates_through_one_source_forward(self):
        api=fake_base();cfg=tiny_config("aligned_cfg")
        model=build_model(api,cfg,self.vocab.sizes(),torch.device("cpu"),training=True)
        batch=collate_graphs([self.view],[self.vocab.encode(self.view)],
                             alignments=[[(0,2),(1,3)]])
        with torch.no_grad():
            model.task_modules["classifier"].weight.zero_()
            model.task_modules["classifier"].bias.zero_()
            model.task_modules["cfg_classifier"].weight.fill_(.1)
        ids=torch.tensor([[29,28,2,3,1]]);mask=torch.ones_like(ids)
        model(ids,mask,batch).sum().backward()
        self.assertEqual(model.encoder.calls,1)
        self.assertGreater(model.task_modules["cfg_encoder"].source_projection.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.encoder.adapter.weight.grad.abs().sum().item(),0)

    def test_partial_accumulation_matches_full_batch_gradient(self):
        # Same sample-weighting formula used by the existing and new trainers.
        model=nn.Linear(3,1);other=copy.deepcopy(model);x=torch.randn(5,3);y=torch.arange(5).float()%2
        nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1),y).backward()
        n_batches=3;accumulation=3;batch_size=2
        for batch_index,start in enumerate(range(0,5,batch_size)):
            end=min(start+batch_size,5)
            group_start=(batch_index//accumulation)*accumulation
            group_end=min(group_start+accumulation,n_batches)
            group_samples=min(group_end*batch_size,5)-group_start*batch_size
            loss=nn.functional.binary_cross_entropy_with_logits(other(x[start:end]).squeeze(-1),y[start:end])
            (loss*(end-start)/group_samples).backward()
        torch.testing.assert_close(model.weight.grad,other.weight.grad)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_full_abc_runner_resume_eval_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp);src=root/"source.jsonl";graphs=root/"graphs.jsonl";run=root/"run"
            rows=fixture_rows();write_jsonl(src,rows)
            exports=[
                dict(data.identity(r), graph_schema_version=data.GRAPH_SCHEMA,
                     preprocessing_version=data.JOERN_SOURCE_PREPROCESSING_VERSION,
                     preprocessing_applied=False,
                     original_source_sha256=data.source_hash(r["raw_source"]),
                     parsed_source_sha256=data.source_hash(r["raw_source"]),
                     graph=fixture_graph(str(i+4)))
                for i, r in enumerate(rows)
            ]
            write_jsonl(graphs,exports)
            api=fake_base();TinyInputBuilder.allow_test=False
            args=exp.parser().parse_args(["run","--dataset",str(src),"--graphs",str(graphs),"--output-dir",str(run),
                "--model-path","tiny-test-only","--device","cpu","--epochs","2","--batch-size","2",
                "--gradient-accumulation","2","--source-max-length","8","--graph-hidden-size","8","--graph-steps","2"])
            output=exp.run_experiment(args,base=api)
            self.assertEqual(set(output["metrics"]),set(exp.DEFAULT_VARIANTS))
            self.assertEqual(api.train_model.call_count,1)
            self.assertEqual(api.train_model.call_args.kwargs["records"],rows)
            self.assertEqual(api.train_model.call_args.kwargs["variant"],"baseline")
            self.assertEqual(api.train_model.call_args.kwargs["source_max_length"],8)
            for variant in exp.DEFAULT_VARIANTS:
                self.assertTrue((run/variant/"best.pt").exists())
                self.assertFalse((run/variant/"test.predictions.jsonl").exists())
                p=data.read_jsonl(run/variant/"valid.predictions.jsonl")
                self.assertEqual({r["split"] for r in p},{"valid"})
            args.resume=True;exp.run_experiment(args,base=api);self.assertEqual(api.train_model.call_count,1)
            # Explicit held-out evaluation only; threshold is loaded, never selected again.
            evalargs=exp.parser().parse_args(["eval","--run-dir",str(run),"--split","test","--device","cpu"])
            TinyInputBuilder.allow_test=True
            try:
                with patch.object(exp,"select_threshold",side_effect=AssertionError("test threshold tuning")):
                    tested=exp.evaluate_run(evalargs,base=api)
                self.assertEqual(set(tested["metrics"]),set(exp.DEFAULT_VARIANTS))
                for variant in exp.DEFAULT_VARIANTS:
                    cp=torch.load(run/variant/"best.pt",weights_only=False)
                    preds=data.read_jsonl(run/variant/"test.predictions.jsonl")
                    self.assertEqual({p["threshold"] for p in preds},{cp["decision_threshold"]})
                with self.assertRaises(FileExistsError):exp.evaluate_run(evalargs,base=api)
            finally:TinyInputBuilder.allow_test=False
            cp_path=run/"cfg"/"best.pt"
            with cp_path.open("ab") as f:f.write(b"tampered")
            with self.assertRaises(ValueError):exp.run_experiment(args,base=api)

    def test_de_runner_coverage_and_fixed_test_threshold(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp);src=root/"source.jsonl";graphs=root/"graphs.jsonl";run=root/"de"
            rows=fixture_rows();write_jsonl(src,rows)
            exports=[dict(data.identity(r), graph_schema_version=data.GRAPH_SCHEMA,
                          preprocessing_version=data.JOERN_SOURCE_PREPROCESSING_VERSION,
                          preprocessing_applied=False,
                          original_source_sha256=data.source_hash(r["raw_source"]),
                          parsed_source_sha256=data.source_hash(r["raw_source"]),
                          graph=positioned_graph(r["raw_source"])) for i,r in enumerate(rows)]
            write_jsonl(graphs,exports)
            api=fake_base()
            args=exp.parser().parse_args(["run","--dataset",str(src),"--graphs",str(graphs),
                "--output-dir",str(run),"--model-path","tiny-test-only","--device","cpu",
                "--variants","aligned_attributes","aligned_cfg","--epochs","1","--batch-size","2",
                "--graph-hidden-size","8","--graph-steps","2"])
            result=exp.run_experiment(args,base=api)
            self.assertEqual(set(result["metrics"]),{"aligned_attributes","aligned_cfg"})
            self.assertIn("changes_aligned_cfg_vs_aligned_attributes",result)
            self.assertEqual(api.train_model.call_count,0)
            for variant in ("aligned_attributes","aligned_cfg"):
                coverage=json.loads((run/variant/"alignment_coverage.json").read_text())
                self.assertEqual(set(coverage),{"train","valid"})
                self.assertGreater(coverage["valid"]["total_nodes"],0)
                self.assertGreater(coverage["valid"]["aligned_nodes"],0)
                self.assertEqual(json.loads((run/variant/"valid.metrics.json").read_text())
                                 ["alignment_coverage"],coverage["valid"])
            testargs=exp.parser().parse_args(["eval","--run-dir",str(run),"--split","test",
                "--variants","aligned_attributes","aligned_cfg","--device","cpu"])
            TinyInputBuilder.allow_test=True
            try:
                with patch.object(exp,"select_threshold",side_effect=AssertionError("test tuning")):
                    tested=exp.evaluate_run(testargs,base=api)
                self.assertEqual(set(tested["metrics"]),{"aligned_attributes","aligned_cfg"})
                for variant in ("aligned_attributes","aligned_cfg"):
                    cp=torch.load(run/variant/"best.pt",weights_only=False)
                    preds=data.read_jsonl(run/variant/"test.predictions.jsonl")
                    self.assertEqual({p["threshold"] for p in preds},{cp["decision_threshold"]})
                    self.assertIn("alignment_coverage",json.loads((run/variant/"test.metrics.json").read_text()))
            finally:TinyInputBuilder.allow_test=False
            args.source_max_length=8
            with self.assertRaisesRegex(ValueError,"2048-token"):
                exp.run_experiment(args,base=api)

    def test_compare_new_de_with_saved_matching_abc(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp);abc=root/"abc";de=root/"de"
            abc.mkdir();de.mkdir()
            (abc/"config.json").write_text(json.dumps({"seed":42,"source_max_length":2048}))
            (de/"config.json").write_text((abc/"config.json").read_text())
            rows=fixture_rows()[5:7]
            for variant in exp.VARIANTS:
                folder=(abc if variant in exp.DEFAULT_VARIANTS else de)/variant
                folder.mkdir()
                predictions=[dict(data.identity(r),score=.8 if r["label"] else .2,
                                  prediction=r["label"],threshold=.5) for r in rows]
                write_jsonl(folder/"valid.predictions.jsonl",predictions)
            compared=exp.compare_run(de,reference_root=abc)
            self.assertEqual(set(compared["metrics"]),set(exp.VARIANTS))
            self.assertIn("aligned_cfg",compared["changes_vs_baseline"])
            self.assertIn("cfg_double_ce",compared["changes_vs_cfg"])
            self.assertIn("cfg_rdrop",compared["changes_vs_cfg"])
            (abc/"config.json").write_text(json.dumps({"seed":7,"source_max_length":2048}))
            with self.assertRaisesRegex(ValueError,"different dataset"):
                exp.compare_run(de,reference_root=abc)

    def test_jk_runner_uses_c_settings_and_valid_threshold_for_test(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source, graphs, run, prior = (root/name for name in
                                          ("source.jsonl", "graphs.jsonl", "jk", "prior"))
            rows = fixture_rows()
            write_jsonl(source, rows)
            write_jsonl(graphs, [dict(data.identity(row), graph_schema_version=data.GRAPH_SCHEMA,
                                      preprocessing_version=data.JOERN_SOURCE_PREPROCESSING_VERSION,
                                      preprocessing_applied=False,
                                      original_source_sha256=data.source_hash(row["raw_source"]),
                                      parsed_source_sha256=data.source_hash(row["raw_source"]),
                                      graph=fixture_graph()) for row in rows])
            api = fake_base()
            args = exp.parser().parse_args([
                "run", "--dataset", str(source), "--graphs", str(graphs), "--output-dir", str(run),
                "--model-path", "tiny-test-only", "--device", "cpu", "--variants", *exp.JK_VARIANTS,
                "--epochs", "1", "--batch-size", "2", "--graph-hidden-size", "8"])
            self.assertEqual(args.source_max_length, 2048)
            self.assertEqual(args.seed, 42)
            self.assertEqual(args.graph_steps, 5)
            trained = exp.run_experiment(args, base=api)
            self.assertEqual(set(trained["metrics"]), set(exp.JK_VARIANTS))
            self.assertEqual(api.train_model.call_count, 0)
            for mode in exp.JK_VARIANTS:
                predictions = data.read_jsonl(run/mode/"valid.predictions.jsonl")
                self.assertEqual([p["sample_key"] for p in predictions],
                                 [r["sample_key"] for r in rows if r["split"] == "valid"])
                self.assertFalse((run/mode/"test.predictions.jsonl").exists())
                checkpoint = torch.load(run/mode/"best.pt", weights_only=False)
                self.assertEqual(checkpoint["model_config"]["graph_steps"], 5)
                self.assertNotIn("training_loss", checkpoint)
            self.assertFalse((run/"ddg_audit.json").exists())
            with self.assertRaisesRegex(ValueError, "2048-token"):
                args.source_max_length = 8
                exp.run_experiment(args, base=api)
            prior.mkdir()
            (prior/"config.json").write_text((run/"config.json").read_text())
            for split in ("valid", "test"):
                selected = [row for row in rows if row["split"] == split]
                write_jsonl(prior/"cfg"/f"{split}.predictions.jsonl", [
                    dict(data.identity(row), score=.8 if row["label"] else .2,
                         prediction=row["label"], threshold=.5) for row in selected])
            valid = exp.compare_run(run, "valid", reference_root=prior)
            self.assertEqual(set(valid["changes_vs_cfg"]), set(exp.JK_VARIANTS))
            evaluation = exp.parser().parse_args([
                "eval", "--run-dir", str(run), "--split", "test", "--variants", *exp.JK_VARIANTS,
                "--device", "cpu"])
            TinyInputBuilder.allow_test = True
            try:
                with patch.object(exp, "select_threshold", side_effect=AssertionError("test tuning")):
                    tested = exp.evaluate_run(evaluation, base=api)
                self.assertEqual(set(tested["metrics"]), set(exp.JK_VARIANTS))
                for mode in exp.JK_VARIANTS:
                    checkpoint = torch.load(run/mode/"best.pt", weights_only=False)
                    predictions = data.read_jsonl(run/mode/"test.predictions.jsonl")
                    self.assertEqual({p["threshold"] for p in predictions},
                                     {checkpoint["decision_threshold"]})
                    self.assertEqual([p["sample_key"] for p in predictions],
                                     [r["sample_key"] for r in selected])
            finally:
                TinyInputBuilder.allow_test = False
            compared = exp.compare_run(run, "test", reference_root=prior)
            self.assertEqual(set(compared["changes_vs_cfg"]), set(exp.JK_VARIANTS))
            for mode in exp.JK_VARIANTS:
                self.assertIn("validation_selected_thresholds", compared["changes_vs_cfg"][mode])

    def test_dual_forward_runner_records_losses_and_keeps_test_single_pass(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source, graphs, run, prior = root/"source.jsonl", root/"graphs.jsonl", root/"dual", root/"prior"
            rows = fixture_rows()
            write_jsonl(source, rows)
            write_jsonl(graphs, [dict(data.identity(row), graph_schema_version=data.GRAPH_SCHEMA,
                                      preprocessing_version=data.JOERN_SOURCE_PREPROCESSING_VERSION,
                                      preprocessing_applied=False,
                                      original_source_sha256=data.source_hash(row["raw_source"]),
                                      parsed_source_sha256=data.source_hash(row["raw_source"]),
                                      graph=fixture_graph()) for row in rows])
            api = fake_base()
            api.SequenceVulnerabilityClassifier = DropoutTinySource
            DropoutTinySource.created.clear()
            args = exp.parser().parse_args([
                "run", "--dataset", str(source), "--graphs", str(graphs), "--output-dir", str(run),
                "--model-path", "tiny-test-only", "--device", "cpu", "--variants",
                "cfg_double_ce", "cfg_rdrop", "--epochs", "1", "--batch-size", "2",
                "--gradient-accumulation", "2", "--graph-hidden-size", "8", "--graph-steps", "2"])
            trained = exp.run_experiment(args, base=api)
            self.assertEqual(set(trained["metrics"]), set(exp.DUAL_FORWARD_ALPHA))
            self.assertEqual(api.train_model.call_count, 0)
            self.assertEqual(len(DropoutTinySource.created), 2)
            first, second = DropoutTinySource.created
            for key, value in first.initial_state.items():
                self.assertTrue(torch.equal(value, second.initial_state[key]), key)
            for source_model in (first, second):
                self.assertEqual(source_model.encoder.calls, 7)  # 3 train batches x 2, 1 valid batch x 1
                self.assertFalse(torch.equal(source_model.encoder.draws[0], source_model.encoder.draws[1]))
            self.assertTrue(torch.equal(first.encoder.draws[0], second.encoder.draws[0]))
            self.assertTrue(torch.equal(first.encoder.draws[1], second.encoder.draws[1]))
            for variant, alpha in exp.DUAL_FORWARD_ALPHA.items():
                history = data.read_jsonl(run/variant/"history.jsonl")
                steps = [item for item in history if item["event"] == "step"]
                epochs = [item for item in history if item["event"] == "epoch"]
                self.assertTrue(steps)
                self.assertEqual(len(epochs), 1)
                self.assertTrue(all({"bce", "kl", "alpha"} <= item.keys() for item in steps))
                self.assertEqual({item["alpha"] for item in steps}, {alpha})
                self.assertEqual(epochs[0]["training_loss"]["alpha"], alpha)
                self.assertIn("validation", epochs[0])
                checkpoint = torch.load(run/variant/"best.pt", weights_only=False)
                self.assertEqual(checkpoint["training_loss"], epochs[0]["training_loss"])
            with self.assertRaisesRegex(ValueError, "2048-token"):
                args.source_max_length = 8
                exp.run_experiment(args, base=api)
            evaluation = exp.parser().parse_args([
                "eval", "--run-dir", str(run), "--split", "test", "--variants",
                "cfg_double_ce", "cfg_rdrop", "--device", "cpu", "--batch-size", "2"])
            TinyInputBuilder.allow_test = True
            try:
                with patch.object(exp, "select_threshold", side_effect=AssertionError("test tuning")):
                    tested = exp.evaluate_run(evaluation, base=api)
                self.assertEqual(set(tested["metrics"]), set(exp.DUAL_FORWARD_ALPHA))
                for source_model in DropoutTinySource.created[2:]:
                    self.assertEqual(source_model.encoder.calls, 1)
                for variant in exp.DUAL_FORWARD_ALPHA:
                    checkpoint = torch.load(run/variant/"best.pt", weights_only=False)
                    predictions = data.read_jsonl(run/variant/"test.predictions.jsonl")
                    self.assertEqual({p["threshold"] for p in predictions},
                                     {checkpoint["decision_threshold"]})
            finally:
                TinyInputBuilder.allow_test = False
            prior.mkdir()
            (prior/"config.json").write_text((run/"config.json").read_text())
            for split in ("valid", "test"):
                selected = [row for row in rows if row["split"] == split]
                write_jsonl(prior/"cfg"/f"{split}.predictions.jsonl", [
                    dict(data.identity(row), score=.8 if row["label"] else .2,
                         prediction=row["label"], threshold=.5) for row in selected])
                comparison = exp.compare_run(run, split, reference_root=prior)
                self.assertEqual(set(comparison["changes_vs_cfg"]), set(exp.DUAL_FORWARD_ALPHA))

    def test_f_g_runner_reuses_graph_cache_and_validation_threshold(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source, graphs, run = root/"source.jsonl", root/"graphs.jsonl", root/"fg"
            rows = fixture_rows()
            write_jsonl(source, rows)
            exports = []
            for row in rows:
                graph = fixture_graph()
                graph["edges"].append(dict(kind="DDG", source="1", target="5"))
                exports.append(dict(data.identity(row), graph_schema_version=data.GRAPH_SCHEMA,
                                    preprocessing_version=data.JOERN_SOURCE_PREPROCESSING_VERSION,
                                    preprocessing_applied=False,
                                    original_source_sha256=data.source_hash(row["raw_source"]),
                                    parsed_source_sha256=data.source_hash(row["raw_source"]),
                                    graph=graph))
            write_jsonl(graphs, exports)
            api = fake_base()
            args = exp.parser().parse_args([
                "run", "--dataset", str(source), "--graphs", str(graphs), "--output-dir", str(run),
                "--model-path", "tiny-test-only", "--device", "cpu", "--variants",
                "cfg_ddg", "cfg_ddg_shuffled", "--epochs", "1", "--batch-size", "2",
                "--graph-hidden-size", "8", "--graph-steps", "2"])
            result = exp.run_experiment(args, base=api)
            self.assertEqual(set(result["metrics"]), {"cfg_ddg", "cfg_ddg_shuffled"})
            self.assertEqual(api.train_model.call_count, 0)
            audit = json.loads((run/"ddg_audit.json").read_text())
            self.assertEqual(audit["splits"]["train"]["raw_edges"], 5)
            self.assertEqual(audit["splits"]["train"]["usable_edges"], 5)
            self.assertEqual(audit["splits"]["train"]["same_degree_multiset_functions"], 5)
            self.assertEqual(audit["splits"]["train"]["same_edge_count_functions"], 5)
            with self.assertRaisesRegex(ValueError, "2048-token"):
                args.source_max_length = 8
                exp.run_experiment(args, base=api)
            evaluation = exp.parser().parse_args([
                "eval", "--run-dir", str(run), "--split", "test", "--variants",
                "cfg_ddg", "cfg_ddg_shuffled", "--device", "cpu"])
            TinyInputBuilder.allow_test = True
            try:
                with patch.object(exp, "select_threshold", side_effect=AssertionError("test tuning")):
                    tested = exp.evaluate_run(evaluation, base=api)
                self.assertEqual(set(tested["metrics"]), {"cfg_ddg", "cfg_ddg_shuffled"})
                for variant in exp.DDG_VARIANTS:
                    checkpoint = torch.load(run/variant/"best.pt", weights_only=False)
                    predictions = data.read_jsonl(run/variant/"test.predictions.jsonl")
                    self.assertEqual({p["threshold"] for p in predictions},
                                     {checkpoint["decision_threshold"]})
            finally:
                TinyInputBuilder.allow_test = False

    def test_ddg_audit_and_compare_f_g_against_saved_c(self):
        rows = fixture_rows()
        views = graph_views(rows)
        for view in views.values():
            view["ddg_edges"] = [(0, 1)]
            view["ddg_audit"] = dict(raw_edges=3, usable_edges=1,
                                     skipped_source_outside_cfg=1,
                                     skipped_target_outside_cfg=1,
                                     skipped_both_outside_cfg=0)
        report = exp.ddg_audit_report(rows, views, 42)
        self.assertEqual(report["splits"]["valid"]["raw_edges"], 6)
        self.assertEqual(report["splits"]["valid"]["usable_edges"], 2)
        self.assertEqual(report["splits"]["valid"]["usable_rate"], 1/3)
        self.assertEqual(report["splits"]["valid"]["same_edge_count_functions"], 2)
        self.assertEqual(report["splits"]["valid"]["same_degree_multiset_functions"], 2)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            old, new = root/"c", root/"fg"
            old.mkdir(); new.mkdir()
            config = {"seed": 42, "source_max_length": 2048}
            (old/"config.json").write_text(json.dumps(config))
            (new/"config.json").write_text(json.dumps(config))
            for split in ("valid", "test"):
                selected = [r for r in rows if r["split"] == split]
                for variant in ("cfg", "cfg_ddg", "cfg_ddg_shuffled"):
                    folder = (old if variant == "cfg" else new)/variant
                    folder.mkdir(exist_ok=True)
                    predictions = [dict(data.identity(r), score=.8 if r["label"] else .2,
                                        prediction=r["label"], threshold=.5) for r in selected]
                    write_jsonl(folder/f"{split}.predictions.jsonl", predictions)
                result = exp.compare_run(new, split, reference_root=old)
                self.assertEqual(set(result["metrics"]), {"cfg", "cfg_ddg", "cfg_ddg_shuffled"})
                self.assertEqual(set(result["changes_vs_cfg"]), {"cfg_ddg", "cfg_ddg_shuffled"})

    def test_cli_help_and_compare_without_model_dependencies(self):
        for command in ([],["build"],["run"],["eval"],["compare"],["audit-alignment"],["audit-ddg"]):
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
                exp.parser().parse_args(command+["--help"])
            self.assertEqual(cm.exception.code,0)


if __name__=="__main__":
    unittest.main()
