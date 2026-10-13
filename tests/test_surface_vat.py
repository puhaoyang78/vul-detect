"""Binding correspondence, constrained embedding VAT and the common experiment entry."""
import io
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import RobertaConfig,RobertaModel
from vulnmechanism import model as core,syntax
from tests.test_codebert_source import CodeTokenizer


class OffsetTokenizer(CodeTokenizer):
    is_fast=True
    def __call__(self,text,return_offsets_mapping=False,**kwargs):
        if text=='FORBIDDEN_HELDOUT':raise AssertionError('Test reached tokenizer')
        kwargs.pop('verbose',None)
        result=super().__call__(text,**kwargs)
        if return_offsets_mapping:result['offset_mapping']=[(i,i+1) for i in range(len(result['input_ids']))]
        return result


class MixedNameTokenizer(OffsetTokenizer):
    """Reproduce CodeBERT's real mixed `(_` token without a model download."""
    def __call__(self,text,return_offsets_mapping=False,**kwargs):
        encoded=super().__call__(text,return_offsets_mapping=True,**kwargs)
        ids=[];offsets=[];index=0
        while index<len(encoded['input_ids']):
            a,b=encoded['offset_mapping'][index]
            if text[a:b]=='(' and index+1<len(encoded['input_ids']) and text[b:b+1]=='_':
                b=encoded['offset_mapping'][index+1][1];index+=1
            ids.append(encoded['input_ids'][index]);offsets.append((a,b));index+=1
        result=dict(input_ids=ids)
        if return_offsets_mapping:result['offset_mapping']=offsets
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


class SurfaceVATTests(unittest.TestCase):
    def test_per_binding_calls_shadowing_unicode_macros_and_barriers(self):
        source='int f(int 变量){int y=fetch(变量);{int 变量=2;use(变量);}return y+变量+GLOBAL;}'
        layout=syntax.surface_binding_layout(source,'c')
        self.assertEqual([b['name'] for b in layout['bindings']],['变量','y','变量'])
        names=[source[a:b] for binding in layout['bindings'] for a,b in binding['spans']]
        self.assertNotIn('fetch',names);self.assertNotIn('GLOBAL',names)
        proposals,_=syntax.surface_preserving_candidates(source,'c')
        self.assertEqual(len(proposals),8)
        for candidate in proposals:
            changed=syntax.apply_source_edits(source,candidate['edits'])
            self.assertIn('{int 变量=2;use(变量);}',changed)
            self.assertIn('fetch(',changed);self.assertIn('GLOBAL',changed)
            self.assertEqual(source.count('\n'),changed.count('\n'))
            if candidate['kind']=='comment':
                self.assertEqual(syntax.source_tokens(source),syntax.source_tokens(changed))
                self.assertGreaterEqual(len(candidate['inserted_comment_spans']),2)
        macro='int f(int x,int y){CHECK(x);use(y);return y;}'
        layout=syntax.surface_binding_layout(macro,'c')
        self.assertEqual([b['name'] for b in layout['bindings']],['y'])
        self.assertGreater(layout['reasons']['macro_ambiguous_binding'],0)
        barrier='int f(int x){{int (*x)(int);x(2);}return x;}'
        layout=syntax.surface_binding_layout(barrier,'c')
        outer=next(b for b in layout['bindings'] if b['name']=='x')
        self.assertEqual(len(outer['spans']),2)
        ranged='int f(int x){for(auto x: items){use(x);}return x;}'
        layout=syntax.surface_binding_layout(ranged,'cpp')
        self.assertEqual([len(b['spans']) for b in layout['bindings']],[2,2])
        self.assertEqual([b['kind'] for b in layout['bindings']],['parameter','local'])
        for candidate in syntax.surface_preserving_candidates(ranged,'cpp')[0]:
            changed=syntax.apply_source_edits(ranged,candidate['edits'])
            if candidate['kind']=='local_name':
                self.assertFalse('for(auto surface_item:' in changed and 'return surface_item;' in changed)
        self.assertFalse(syntax.surface_binding_layout('int f(){','c')['bindings'])

    def test_surface_offsets_frozen_execution_and_window_duplicates(self):
        source='int f(int x){/* original */ int y=api(x);{int x=2;y+=x;}return y+x;}'
        proposals,_=syntax.surface_preserving_candidates(source,'c')
        candidate=next(c for c in proposals if c['kind']=='combined')
        changed=syntax.apply_source_edits(source,candidate['edits'])
        row=dict(raw_source=changed,language='c',inserted_comment_spans=candidate['inserted_comment_spans'])
        builder=core.InputBuilder(OffsetTokenizer(),source_max_length=2048,context_max_length=384)
        keys,valid=builder.surface_tokens(changed,'c','',tuple(map(tuple,row['inserted_comment_spans'])))
        for word in ('api','return','original','int','2','+'):
            at=changed.index(word)
            self.assertTrue(all(key is None for key in keys[at:at+len(word)]))
        for a,b in row['inserted_comment_spans']:
            self.assertTrue(any(k is not None for k in keys[a:b]))
        # Same spelling in two scopes must map to different declaration groups.
        keys,_=builder.surface_tokens(source,'c','',())
        occurrences=[i for i,c in enumerate(source) if c=='x']
        self.assertEqual(keys[occurrences[0]],keys[occurrences[-1]])
        self.assertNotEqual(keys[occurrences[0]],keys[occurrences[2]])
        long='int f(int x){'+'/* filler */ '*60+'return x;}'
        batch,groups,owners,eligible=builder.vat_batch([dict(raw_source=long,language='c')],mode='binding',device='cpu')
        self.assertEqual(eligible,[True]);self.assertGreater(len(batch[0]),1)
        source_positions=builder.codebert_batch([dict(raw_source=long)],device='cpu',return_source_positions=True)[-1]
        for position in source_positions[source_positions>=0].unique():
            selected=groups[source_positions==position]
            self.assertEqual(len(selected.unique()),1)
        self.assertTrue(torch.all(groups[source_positions<0]==-1))

    def test_shared_gradient_and_actual_token_norm(self):
        ids=torch.tensor([[0,0,1,-1],[2,2,2,-1]])
        owners=torch.tensor([0,0,1]);windows=torch.tensor([0,1])
        values=torch.ones(3,5,requires_grad=True)
        expanded=core.expand_vat_groups(values,ids)
        expanded.sum().backward()
        torch.testing.assert_close(values.grad,torch.tensor([[2.]*5,[1.]*5,[3.]*5]))
        normalized,_=core.normalize_vat_groups(values.detach(),ids,owners,windows,2,1.)
        actual=core.expand_vat_groups(normalized,ids)
        torch.testing.assert_close(actual.square().sum((1,2)).sqrt(),torch.ones(2))
        self.assertTrue(torch.all(actual[ids<0]==0))
        # A real run exposed float32 norm underflow for tiny VAT gradients.
        # Such a direction is nonzero and must still expand to epsilon=1.
        for magnitude in (1e-22,1e-23,1e-40):
            tiny=torch.full((3,5),magnitude)
            normalized,norms=core.normalize_vat_groups(tiny,ids,owners,windows,2,1.)
            self.assertTrue(torch.all(norms>0))
            actual=core.expand_vat_groups(normalized,ids)
            torch.testing.assert_close(actual.double().square().sum((1,2)).sqrt(),torch.ones(2,dtype=torch.float64))
        normalized,norms=core.normalize_vat_groups(torch.zeros(3,5),ids,owners,windows,2,1.)
        self.assertTrue(torch.all(norms==0));self.assertTrue(torch.all(normalized==0))

    def test_mixed_execution_token_freezes_the_entire_binding(self):
        source='int f(int _x,int y){use(_x);return _x+y;}'
        tokenizer=MixedNameTokenizer()
        builder=core.InputBuilder(tokenizer,source_max_length=2048,context_max_length=384)
        keys,_=builder.surface_tokens(source,'c','',())
        offsets=tokenizer(source,return_offsets_mapping=True)['offset_mapping']
        layout=syntax.surface_binding_layout(source,'c')
        for binding in layout['bindings']:
            touched=[keys[i] for i,(a,b) in enumerate(offsets) if any(a<z and b>x for x,z in binding['spans'])]
            if binding['name']=='_x':self.assertTrue(all(key is None for key in touched))
            else:self.assertTrue(all(key==('binding',binding['declaration']) for key in touched))
        row=dict(raw_source=source,language='c')
        surface=builder.vat_batch([row],mode='surface',device='cpu')[1]
        binding=builder.vat_batch([row],mode='binding',device='cpu')[1]
        self.assertTrue(torch.equal(surface>=0,binding>=0))

    def test_vat_kl_signal_dropout_rng_and_zero_embedding_equivalence(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            _,model=tiny_model(Path(tmp))
            builder=core.InputBuilder(OffsetTokenizer(),source_max_length=2048,context_max_length=384)
            rows=[dict(raw_source='int f(int x){int y=api(x);return y+x;}',language='c')]
            batch=builder.codebert_batch(rows,device='cpu')
            with core._auxiliary_mode(model):
                ref=model(*batch,force_fp32=True)
                zero=torch.zeros((*batch[0].shape,16))
                torch.testing.assert_close(ref,model(*batch,embedding_delta=zero,force_fp32=True),rtol=0,atol=0)
            before=torch.get_rng_state().clone()
            for mode in ('vat','surface','binding'):
                cfg=dict(core.SURFACE_VAT_CONFIG,mode=mode)
                values,info,delta=core.surface_vat_loss(model,rows,builder,cfg,device=torch.device('cpu'),noise_seed=42,return_delta=True)
                self.assertGreater(float(values.sum()),0.)
                gradient=torch.autograd.grad(values.sum(),model.task_modules['classifier'].weight)[0]
                self.assertGreater(float(gradient.norm()),0.)
                self.assertEqual(info['eligible'],1);self.assertEqual(info['zero_direction'],0)
                torch.testing.assert_close(delta.square().sum().sqrt(),torch.tensor(1.),rtol=1e-5,atol=1e-5)
                self.assertTrue(torch.equal(before,torch.get_rng_state()))
                self.assertTrue(model.training)
            ref=torch.tensor([-2.,2.],requires_grad=True);perturbed=torch.tensor([-1.,1.],requires_grad=True)
            loss=core.bernoulli_vat_kl(ref,perturbed).sum();loss.backward()
            self.assertIsNone(ref.grad);self.assertGreater(float(perturbed.grad.abs().sum()),0.)
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            with core._auxiliary_mode(model):
                values,info=core.surface_vat_loss(model,rows,builder,dict(core.SURFACE_VAT_CONFIG,mode='binding'),
                    device=torch.device('cpu'),noise_seed=42)
                values.sum().backward()
            self.assertGreater(float(model.encoder.embeddings.word_embeddings.weight.grad.norm()),0.)

    def test_formal_entry_shared_budget_reload_and_reuse_without_shallow_models(self):
        from vulnmechanism.cli import _codebert_finetuning,_finetuning_evaluate
        from vulnmechanism import audit_statistics
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);modelpath,_=tiny_model(root);manifest=root/'manifest.jsonl';baseline=root/'baseline/best.pt'
            rows=[dict(dataset='primevul',sample_key=str(i),split='train',label=i%2,language='c',
                function=f'int f{i}(int x){{int y=api(x);return y+{i};}}') for i in range(8)]
            rows+=[dict(rows[i],sample_key=f'v{i}',split='valid') for i in range(2)]
            rows+=[dict(rows[0],sample_key='heldout',split='test',function='FORBIDDEN_HELDOUT')]
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            args=Namespace(manifest=str(manifest),model=str(modelpath),output_dir=str(root/'surface'),stage='all',
                finetuning_mode=None,device='cpu',batch_size=4,gradient_accumulation=2,surface_vat='all',baseline_checkpoint=str(baseline))
            with patch.object(core.AutoTokenizer,'from_pretrained',return_value=MixedNameTokenizer()),redirect_stdout(io.StringIO()),\
                    patch.object(audit_statistics,'shallow_oof_weights',side_effect=AssertionError('shallow model used')):
                data=[r for r in core._read_jsonl(manifest,source_only=True) if r['split'] in ('train','valid')]
                cp=core.train_model(None,baseline,records=data,variant='codebert_source',model_path=str(modelpath),source_max_length=2048,
                    epochs=3,batch_size=4,gradient_accumulation=2,learning_rate=1e-5,device='cpu',finetuning_mode='full')
                for split in ('train','valid'):_finetuning_evaluate(data,baseline.parent,split,cp,args)
                result=_codebert_finetuning(args)
                self.assertEqual(set(result['valid']),{'bce',*core.SURFACE_VAT_MODES})
                self.assertEqual(len(result['matched_training_budgets']),3)
                plain=result['valid']['bce']
                self.assertEqual(plain['checkpoint'],str(baseline))
                self.assertTrue(plain['reused'])
                self.assertFalse(plain['data_augmentation']);self.assertFalse(plain['vat'])
                self.assertEqual(plain['bce_views_per_function'],1)
                self.assertEqual(plain['threshold'],cp['decision_threshold'])
                self.assertEqual(plain['selected_epoch'],cp['selected_epoch'])
                self.assertEqual(result['matched_optimizer_budgets'],[(i,1) for i in (1,2,3)])
                self.assertFalse((root/'surface/bce/best.pt').exists())
                for kind in ('comment','local_name','function_name','combined'):
                    self.assertEqual(plain['robustness'][kind]['eligible'],2)
                    self.assertTrue((root/f'surface/bce/unseen_{kind}.predictions.jsonl').exists())
                for mode in core.SURFACE_VAT_MODES:
                    saved=torch.load(root/f'surface/{mode}/best.pt',weights_only=False)
                    self.assertEqual(saved['training_config'],cp['training_config'])
                    self.assertEqual(saved['surface_vat_config']['mode'],mode)
                    self.assertEqual(saved['trainable_parameters'],cp['trainable_parameters'])
                with patch.object(core,'train_model',side_effect=AssertionError('completed experiment retrained')):
                    _codebert_finetuning(args)


if __name__=='__main__':unittest.main()
