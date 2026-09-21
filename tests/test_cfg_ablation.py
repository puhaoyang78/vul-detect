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
from vulnmechanism.cpg import FunctionGraph, GraphEdge, GraphNode, resolve_target_graph


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
            self.assertEqual(len(data.load_graphs(out, rows)), 3)
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


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(32,8);self.adapter=nn.Linear(8,8,bias=False)
        self.embedding.weight.requires_grad_(False)
    def forward(self,input_ids,attention_mask,use_cache=False):
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


class TinyTokenizer:
    pad_token_id=0
    eos_token_id=1
    @classmethod
    def from_pretrained(cls,*args,**kwargs):
        return cls()


class TinyInputBuilder:
    allow_test=False
    def __init__(self,tokenizer,*,source_max_length,context_max_length):
        self.source_max_length=source_max_length
    def _encode(self,text,max_length=None):
        return [ord(c)%30+2 for c in text][:max_length]
    def sequence_batch(self,records,*,variant,excluded_groups,device):
        if not self.allow_test and any(r["split"]=="test" for r in records):
            raise AssertionError("test records reached model during training")
        ids=[self._encode(r["raw_source"],self.source_max_length)+[1] for r in records]
        tensor=torch.zeros((len(ids),max(map(len,ids))),dtype=torch.long,device=device)
        for i,seq in enumerate(ids):tensor[i,:len(seq)]=torch.tensor(seq,device=device)
        return tensor,tensor.ne(0).long()


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
            exports=[dict(data.identity(r),graph_schema_version=1,graph=fixture_graph(str(i+4))) for i,r in enumerate(rows)]
            write_jsonl(graphs,exports)
            api=fake_base();TinyInputBuilder.allow_test=False
            args=exp.parser().parse_args(["run","--dataset",str(src),"--graphs",str(graphs),"--output-dir",str(run),
                "--model-path","tiny-test-only","--device","cpu","--epochs","2","--batch-size","2",
                "--gradient-accumulation","2","--source-max-length","8","--graph-hidden-size","8","--graph-steps","2"])
            output=exp.run_experiment(args,base=api)
            self.assertEqual(set(output["metrics"]),set(exp.VARIANTS))
            self.assertEqual(api.train_model.call_count,1)
            self.assertEqual(api.train_model.call_args.kwargs["records"],rows)
            self.assertEqual(api.train_model.call_args.kwargs["variant"],"baseline")
            self.assertEqual(api.train_model.call_args.kwargs["source_max_length"],8)
            for variant in exp.VARIANTS:
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
                self.assertEqual(set(tested["metrics"]),set(exp.VARIANTS))
                for variant in exp.VARIANTS:
                    cp=torch.load(run/variant/"best.pt",weights_only=False)
                    preds=data.read_jsonl(run/variant/"test.predictions.jsonl")
                    self.assertEqual({p["threshold"] for p in preds},{cp["decision_threshold"]})
                with self.assertRaises(FileExistsError):exp.evaluate_run(evalargs,base=api)
            finally:TinyInputBuilder.allow_test=False
            cp_path=run/"cfg"/"best.pt"
            with cp_path.open("ab") as f:f.write(b"tampered")
            with self.assertRaises(ValueError):exp.run_experiment(args,base=api)

    def test_cli_help_and_compare_without_model_dependencies(self):
        for command in ([],["build"],["run"],["eval"],["compare"]):
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
                exp.parser().parse_args(command+["--help"])
            self.assertEqual(cm.exception.code,0)


if __name__=="__main__":
    unittest.main()
