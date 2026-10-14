"""Dynamic legal search, global soft-weighted VAT and the public training path."""
import io
import json
import random
import tempfile
import unittest
from argparse import Namespace
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import RobertaConfig, RobertaModel
from vulnmechanism import model as core, syntax, augmentation as data
from tests.test_codebert_source import CodeTokenizer


class OffsetTokenizer(CodeTokenizer):
    is_fast=True
    def __call__(self,text,return_offsets_mapping=False,**kwargs):
        if text=='FORBIDDEN_HELDOUT':raise AssertionError('Test reached tokenizer')
        kwargs.pop('verbose',None)
        result=super().__call__(text,**kwargs)
        if return_offsets_mapping:result['offset_mapping']=[(i,i+1) for i in range(len(result['input_ids']))]
        return result

def tiny_model(root):
    path=root/'encoder'
    RobertaModel(RobertaConfig(vocab_size=16,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
        intermediate_size=32,max_position_embeddings=514,pad_token_id=0,bos_token_id=2,eos_token_id=1,
        hidden_dropout_prob=.1,attention_probs_dropout_prob=.1)).save_pretrained(path)
    core._seed_everything(42)
    model=core._build_model('codebert_source',str(path),device=torch.device('cpu'),lora_r=16,lora_alpha=32,
        lora_dropout=.05,target_modules=('attention.self.query','attention.self.key','attention.self.value','attention.output.dense'),
        gradient_checkpointing=False,fusion_dim=256,fusion_heads=8,finetuning_mode='full')
    return path,model

MATERIALS=dict(source='unit fixture',train=dict(names=['count','item','array','buffer','total','entry'],
    words=['ordinary','section','value','later','routine','return']),
    heldout=dict(names=['orchard','passage'],words=['linen','cedar']),rules=syntax.SURFACE_RULES)

def proposals(source,kind=None,seed=42,**kwargs):
    return syntax.surface_preserving_candidates(source,'c',rng=random.Random(seed),materials=MATERIALS,
        tokenizer=OffsetTokenizer(),kinds=[kind] if kind else None,**kwargs)


class SurfaceVATTests(unittest.TestCase):
    def test_binding_shadowing_utf8_members_macros_and_self_references(self):
        source='int f(int 变量){int y=fetch(变量);{int 变量=2;use(变量);}return y+变量+GLOBAL;}'
        layout=syntax.surface_binding_layout(source,'c')
        self.assertEqual([b['name'] for b in layout['bindings']],['变量','y','变量'])
        selected=set()
        for seed in range(15):
            for candidate in proposals(source,'local_name',seed)[0]:
                selected.add(candidate['binding_token_index']);changed=candidate['raw_source']
                binding=next(b for b in layout['bindings'] if b['declaration']==candidate['edits'][0][0])
                self.assertEqual([(a,b) for a,b,v in candidate['edits']],list(map(tuple,binding['spans'])))
                self.assertIn('fetch(',changed);self.assertIn('GLOBAL',changed)
        self.assertEqual(len(selected),3)
        for c in proposals('int f(int x){return obj.x+x;}','local_name')[0]:self.assertIn('obj.x+',c['raw_source'])
        layout=syntax.surface_binding_layout('int f(int x,int y){CHECK(x);use(y);return y;}','c')
        self.assertEqual([b['name'] for b in layout['bindings']],['y'])
        self.assertGreater(layout['reasons']['macro_ambiguous_binding'],0)
        layout=syntax.surface_binding_layout('int f(int x){{int (*x)(int);x(2);}return x;}','c')
        self.assertEqual(len(layout['bindings'][0]['spans']),2)
        for c in proposals('int f(int x){return x ? f(x-1) : 0;}','function_name')[0]:
            self.assertEqual(len(c['edits']),2);self.assertFalse(c['strict_semantic'])
            self.assertTrue(c['function_name_input_only']);self.assertNotIn('f(',c['raw_source'])
        self.assertFalse(proposals('int f(int x){log(__func__);return x;}','function_name')[0])

    def test_gradient_copy_metric_and_expanded_radius(self):
        ids=torch.tensor([[0,0,1,-1],[2,2,2,-1]]);owners=torch.tensor([0,0,1]);windows=torch.tensor([0,1])
        values=torch.ones(3,5,requires_grad=True)
        core.expand_vat_groups(values,ids).sum().backward();counts=torch.bincount(ids[ids>=0],minlength=3)
        torch.testing.assert_close(values.grad/counts[:,None],torch.ones_like(values))
        for magnitude in (1.,1e-22,1e-23,1e-40):
            normalized,norms=core.normalize_vat_groups(torch.full((3,5),magnitude),ids,owners,windows,2,1.)
            actual=core.expand_vat_groups(normalized,ids)
            self.assertTrue(torch.all(norms>0));self.assertTrue(torch.all(actual[ids<0]==0))
            torch.testing.assert_close(actual.double().square().sum((1,2)).sqrt(),torch.ones(2,dtype=torch.float64))
        normalized,norms=core.normalize_vat_groups(torch.zeros(3,5),ids,owners,windows,2,1.)
        self.assertTrue(torch.all(norms==0));self.assertTrue(torch.all(normalized==0))

    def test_comment_edits_and_visibility(self):
        source='int f(int x){/* ordinary\n section */return api(x);}'
        for kind in ('comment_insert','comment_delete','comment_replace'):
            candidates,reasons=proposals(source,kind);self.assertTrue(candidates,(kind,reasons))
            for c in candidates:
                self.assertEqual(syntax.source_tokens(source),syntax.source_tokens(c['raw_source']))
                self.assertEqual(source.count('\n'),c['raw_source'].count('\n'));self.assertLessEqual(c['added_comment_tokens'],64)
        for c in proposals('int/**/f(int x){return x;}','comment_delete')[0]:self.assertIn('int f(',c['raw_source'])
        tokenizer=OffsetTokenizer();source='int f(int x){/* ordinary section */return x+2;}'
        kept,reasons=data.visible_semantic_candidates(source,'c',tokenizer,len(source),proposals=proposals(source,'comment_insert')[0])
        self.assertFalse(kept);self.assertGreater(reasons['visible_execution_removed'],0)
        deleted=proposals(source,'comment_delete')[0]
        kept,reasons=data.visible_semantic_candidates(source,'c',tokenizer,source.index('return')+3,proposals=deleted)
        self.assertFalse(kept);self.assertGreater(reasons['extra_execution_exposed'],0)
        self.assertTrue(data.visible_semantic_candidates(source,'c',tokenizer,2048,proposals=deleted)[0])

    def test_dynamic_reproducibility_holdout_and_hardest_search(self):
        source='int f(int x){/* ordinary section */int y=api(x);return x+y;}'
        row=dict(sample_key='stable',raw_source=source,language='c',label=1);tokenizer=OffsetTokenizer()
        parameter=torch.tensor([.7],requires_grad=True);parameter.grad=torch.tensor([2.]);before=parameter.detach().clone()
        def score(rows):return [float(parameter.detach())-len(r['raw_source'])/100 for r in rows]
        a=data.search_surface_views([row],score,tokenizer,2025,MATERIALS,epoch=1)[0]
        self.assertEqual(a,data.search_surface_views([row],score,tokenizer,2025,MATERIALS,epoch=1)[0])
        self.assertNotEqual(a['visited'],data.search_surface_views([row],score,tokenizer,2025,MATERIALS,epoch=2)[0]['visited'])
        self.assertLessEqual(a['candidate_queries'],12);self.assertEqual(a['original_queries'],1)
        self.assertEqual(a['selected']['loss'],max(v['loss'] for v in a['visited']))
        self.assertTrue(any(len(v['units'])>1 for v in a['visited']))
        for v in a['visited']:
            self.assertLessEqual(len(v['units']),3);self.assertLessEqual(v['added_comment_tokens'],64)
            self.assertEqual(v['raw_source'],data.replay_surface_units(source,v['units']))
            for combo in syntax.SURFACE_RULES['heldout_combinations']:self.assertTrue(Counter(combo)-Counter(u['kind'] for u in v['units']))
            for u in v['units']:
                self.assertNotIn(u['rule'],syntax.SURFACE_RULES['heldout_names']+syntax.SURFACE_RULES['heldout_comments'])
                for aa,bb,value in u['edits']:self.assertNotIn('orchard',value);self.assertNotIn('cedar',value)
        torch.testing.assert_close(before,parameter);torch.testing.assert_close(parameter.grad,torch.tensor([2.]))
        improved=data.search_surface_views([row],lambda batch:[10. if r['raw_source']!=source else -10. for r in batch],tokenizer,2025,MATERIALS)[0]
        self.assertIsNotNone(improved['selected']);self.assertLess(improved['loss_delta'],0)
        random_view=data.search_surface_views([{k:v for k,v in row.items() if k!='label'}],
            lambda batch:(_ for _ in ()).throw(AssertionError('random ablation scored candidates')),tokenizer,2025,MATERIALS,random_selection=True)[0]
        self.assertIsNotNone(random_view['selected']);self.assertEqual(random_view['candidate_queries'],0)
        attack=data.search_surface_views([row],score,tokenizer,2025,MATERIALS,query_limit=48,purpose='adaptive')[0]
        self.assertLessEqual(attack['candidate_queries'],48)
        self.assertEqual(len(set(v['raw_source'] for v in attack['visited'])),len(attack['visited']))

    def test_function_weights_and_complete_robustness_denominators(self):
        rows=[dict(sample_key=str(i),raw_source='original',label=i%2) for i in range(3)]
        counts={str(i):0. for i in range(3)}
        for batch,weights in core.function_view_batches(rows,{'0':['changed'],'1':[],'2':['changed']},2):
            for r,w in zip(batch,weights):counts[r['sample_key']]+=w
        self.assertEqual(counts,{'0':1.,'1':1.,'2':1.})
        from vulnmechanism.cli import _robustness_summary
        originals=[dict(r,logit=-1.,prediction=0) for r in rows]
        report=_robustness_summary(originals,{'0':[dict(logit=1.,prediction=1)],'1':[dict(logit=1.,prediction=1)]},.5)
        self.assertEqual(report['total'],3);self.assertEqual(report['eligible'],2)
        self.assertEqual(report['asr']['denominator'],1);self.assertEqual(report['asr']['count'],1)
        self.assertEqual(report['wrong_to_correct']['count'],1);self.assertEqual(report['worst_complete']['accuracy'],1/3)

    def test_evaluation_uses_the_saved_float32_threshold_boundary(self):
        from vulnmechanism.cli import _robustness_evaluation
        from vulnmechanism.metrics import decision_boundary
        z=torch.logit(torch.tensor(.29))
        self.assertEqual(torch.sigmoid(z).item(),decision_boundary(.29))
        class Constant(torch.nn.Module):
            def forward(self,*batch):return z.expand(batch[-1])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'bce';run.mkdir();target=run/'best.pt'
            source='int f(int x){return x;}'
            rows=[dict(sample_key=str(i),raw_source=source,label=i,language='c',split='valid') for i in (0,1)]
            originals=[dict(sample_key=r['sample_key'],label=r['label'],logit=z.item(),prediction=1) for r in rows]
            (run/'valid.predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in originals))
            entries=[dict(sample_key=r['sample_key'],groups={'comment_insert':[dict(strict_semantic=True,
                units=[dict(edits=[[source.index('{')+1,source.index('{')+1,' /* note */ ']])])]},reasons={}) for r in rows]
            for bank in ('fixed','heldout'):
                (root/f'{bank}.candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in entries))
            tokenizer=OffsetTokenizer();builder=core.InputBuilder(tokenizer,source_max_length=2048,context_max_length=384)
            cp=dict(decision_threshold=.29,selected_epoch=3,input_protocol=dict(source_budget=2048-len(builder.source_prefix)-2))
            args=Namespace(device='cpu',batch_size=4,model='fixture')
            with patch.object(core.AutoTokenizer,'from_pretrained',return_value=tokenizer),\
                    patch.object(core,'_load_model',return_value=Constant()),\
                    patch('vulnmechanism.cli._finetuning_evaluate',return_value={}):
                result=_robustness_evaluation(rows,run,target,cp,root,MATERIALS,args)
            self.assertEqual(result['fixed']['comment_insert']['all']['perturbed']['recall'],1.)
            self.assertEqual(result['fixed']['comment_insert']['all']['all_candidates_correct'],1)
            with (run/'fixed.predictions.jsonl').open() as handle:
                cached=[json.loads(l) for l in handle]
            self.assertTrue(all(v['prediction']==1 for r in cached for v in r['groups']['comment_insert']))
