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
from vulnmechanism import model as core, syntax, audit_statistics as data
from vulnmechanism.change_learning import bernoulli_js
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

    def test_soft_weights_mixed_bpe_specials_and_window_copies(self):
        source='int f(int _x,int y){/* existing */ use(_x);return _x+y;}'
        tokenizer=MixedNameTokenizer();builder=core.InputBuilder(tokenizer,source_max_length=2048,context_max_length=384)
        weights=builder.surface_tokens(source,'c','',3.);offsets=tokenizer(source,return_offsets_mapping=True)['offset_mapping']
        mixed=next(i for i,(a,b) in enumerate(offsets) if source[a:b]=='(_')
        self.assertEqual(weights[mixed],2.5);self.assertTrue(all(1<=w<=4 for w in weights))
        for word in ('use','return','int','+'):
            a=source.index(word)
            for i,(left,right) in enumerate(offsets):
                if a<=left<right<=a+len(word):self.assertEqual(weights[i],1.)
        comment=source.index('existing')
        self.assertTrue(all(w==4 for w,(a,b) in zip(weights,offsets) if comment<=a<b<=comment+8))
        long='int f(int x){'+'/* filler */ '*60+'return x;}';row=dict(raw_source=long,language='c')
        batch,groups,owners,eligible,weights=builder.vat_batch([row],alpha=3.,device='cpu')
        positions=builder.codebert_batch([row],device='cpu',return_source_positions=True)[-1]
        self.assertEqual(eligible,[True]);self.assertGreater(len(batch[0]),1)
        for position in positions[positions>=0].unique():self.assertEqual(len(groups[positions==position].unique()),1)
        self.assertTrue(torch.all(groups[positions<0]==-1))
        xx=[i for i,c in enumerate(long) if c=='x']
        self.assertNotEqual(groups[positions==xx[0]][0].item(),groups[positions==xx[-1]][0].item())
        self.assertTrue(torch.all(weights>=1))

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

    def test_finite_difference_analytical_direction_and_weights_once(self):
        ids=torch.tensor([[0,0,1,-1]]);owners=torch.tensor([0,0]);windows=torch.tensor([0])
        class ToyBuilder:
            def vat_batch(self,rows,alpha,device):
                batch=(torch.ones(1,4,dtype=torch.long),torch.ones(1,4),windows,None,1)
                return batch,ids,owners,[True],torch.tensor([1.+alpha,1.])
        class Toy(torch.nn.Module):
            def __init__(self,zero=False):
                super().__init__();self.encoder=torch.nn.Identity();self.encoder.config=SimpleNamespace(hidden_size=3)
                self.coefficients=torch.nn.Parameter(torch.zeros(2,3) if zero else torch.tensor([[.2,-.3,.4],[.5,.2,-.1]]))
            def forward(self,*batch,embedding_delta=None,force_fp32=False):
                delta=torch.zeros(1,4,3) if embedding_delta is None else embedding_delta
                return .2+(delta[:,:3].double()*self.coefficients[[0,0,1]].double()).sum((1,2))
        for alpha in (0.,3.):
            model=Toy()
            loss,info,delta=core.vat_loss(model,[{}],ToyBuilder(),core.VAT_CONFIG,alpha=alpha,
                device=torch.device('cpu'),noise_seed=42,return_delta=True,diagnose=True)
            expected,_=core.normalize_vat_groups(model.coefficients.detach()*torch.tensor([1+alpha,1.])[:,None],ids,owners,windows,1,1.)
            expected=core.expand_vat_groups(expected,ids);sign=torch.sign((delta*expected).sum())
            torch.testing.assert_close(delta,sign*expected,rtol=1e-4,atol=1e-5)
            self.assertEqual(info['zero_direction'],0);self.assertEqual(info['signal_loss'],0)
            self.assertGreater(info['adversarial_kl'][0],info['random_kl'][0])
            self.assertGreater(float(torch.autograd.grad(loss.sum(),model.coefficients)[0].norm()),0)
        _,info=core.vat_loss(Toy(zero=True),[{}],ToyBuilder(),core.VAT_CONFIG,alpha=0.,device=torch.device('cpu'),noise_seed=42,diagnose=True)
        self.assertEqual(info['true_zero'],1);self.assertEqual(info['signal_loss'],0);self.assertEqual(info['zero_direction'],1)
        class QuantizedToy(Toy):
            def forward(self,*batch,embedding_delta=None,force_fp32=False):
                delta=torch.zeros(1,4,3) if embedding_delta is None else embedding_delta
                # Finite precision loses the probe value, while the Jacobian exists.
                return (.2+(delta[:,:3]*self.coefficients[[0,0,1]]*1e-6).sum((1,2))).double()
        _,info=core.vat_loss(QuantizedToy(),[{}],ToyBuilder(),core.VAT_CONFIG,alpha=0.,device=torch.device('cpu'),noise_seed=42,diagnose=True)
        self.assertEqual(info['signal_loss'],1);self.assertEqual(info['true_zero'],0)

    def test_js_both_gradients_kl_reference_detach_dropout_and_alpha_zero(self):
        torch.set_num_threads(2)
        first=torch.tensor([-2.,2.],requires_grad=True);second=torch.tensor([-1.,1.],requires_grad=True)
        bernoulli_js(first,second).sum().backward()
        self.assertTrue(torch.all(first.grad.abs()>0));self.assertTrue(torch.all(second.grad.abs()>0))
        # logsumexp/subtraction has float32 roundoff even at equal logits.
        torch.testing.assert_close(bernoulli_js(first,first),torch.zeros_like(first),rtol=0,atol=torch.finfo(first.dtype).eps)
        first.grad=None;second.grad=None;core.bernoulli_vat_kl(first,second).sum().backward()
        self.assertIsNone(first.grad);self.assertTrue(torch.isfinite(second.grad).all())
        self.assertTrue(torch.all(core.bernoulli_vat_kl(first,first)==0))
        with tempfile.TemporaryDirectory() as tmp:
            _,model=tiny_model(Path(tmp));builder=core.InputBuilder(OffsetTokenizer(),source_max_length=2048,context_max_length=384)
            rows=[dict(raw_source='int f(int x){int y=api(x);return y+x;}',language='c')];batch=builder.codebert_batch(rows,device='cpu')
            with core._auxiliary_mode(model):
                ref=model(*batch,force_fp32=True)
                torch.testing.assert_close(ref,model(*batch,embedding_delta=torch.zeros((*batch[0].shape,16)),force_fp32=True),rtol=0,atol=0)
            rng=torch.get_rng_state().clone()
            a,ai,ad=core.vat_loss(model,rows,builder,core.VAT_CONFIG,alpha=0.,device=torch.device('cpu'),noise_seed=42,return_delta=True)
            b,bi,bd=core.vat_loss(model,rows,builder,core.robustness_config('ours_uniform')['vat'],alpha=0.,
                device=torch.device('cpu'),noise_seed=42,return_delta=True)
            torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(ad,bd,rtol=0,atol=0)
            self.assertEqual(ai,bi);self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(model.training)
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            with core._auxiliary_mode(model):
                values,info=core.vat_loss(model,rows,builder,core.VAT_CONFIG,alpha=3.,device=torch.device('cpu'),noise_seed=42)
                values.sum().backward()
            self.assertGreater(float(model.encoder.embeddings.word_embeddings.weight.grad.norm()),0)

    def test_pure_vat_without_syntax_gate(self):
        builder=core.InputBuilder(OffsetTokenizer(),source_max_length=2048,context_max_length=384)
        rows=[dict(raw_source='return external_api(1)+GLOBAL;',language='c'),dict(raw_source='int f(){return 1;}',language='c')]
        with patch.object(builder,'surface_tokens',side_effect=AssertionError('pure VAT parsed a function')):
            batch,groups,owners,eligible,weights=builder.vat_batch(rows,alpha=0.,device='cpu')
        positions=builder.codebert_batch(rows,device='cpu',return_source_positions=True)[-1]
        self.assertTrue(torch.equal(groups>=0,(positions>=0)&(batch[0]!=builder.tokenizer.bos_token_id)))
        self.assertEqual(eligible,[True,True]);self.assertTrue(torch.all(weights==1))
        self.assertTrue(all(w==1 for w in builder.surface_tokens(rows[0]['raw_source'],'c','',3.)))

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

    def test_public_three_paths_updates_reload_and_reuse(self):
        from vulnmechanism.cli import _codebert_finetuning, _finetuning_evaluate
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);modelpath,_=tiny_model(root);manifest=root/'manifest.jsonl';baseline=root/'baseline/best.pt'
            names=['count','item','array','buffer','total','entry','length','offset']
            rows=[dict(dataset='primevul',sample_key=str(i),split='train',label=i%2,language='c',
                function=f'int f{i}(int {name}){{/* ordinary section value later routine return */return api({name})+{i};}}') for i,name in enumerate(names)]
            rows[0]['function']='return external_api(1)+GLOBAL;'
            rows+=[dict(rows[i],sample_key=f'v{i}',split='valid') for i in (1,2)]
            rows+=[dict(rows[0],sample_key='heldout',split='test',function='FORBIDDEN_HELDOUT')]
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            args=Namespace(manifest=str(manifest),model=str(modelpath),output_dir=str(root/'robustness'),stage='all',
                finetuning_mode=None,device='cpu',batch_size=4,gradient_accumulation=2,robustness='all',baseline_checkpoint=str(baseline))
            with patch.object(core.AutoTokenizer,'from_pretrained',return_value=MixedNameTokenizer()),redirect_stdout(io.StringIO()), \
                    patch.object(data,'shallow_oof_weights',side_effect=AssertionError('shallow model used')):
                parsed=[r for r in core._read_jsonl(manifest,source_only=True) if r['split'] in ('train','valid')]
                with patch.object(data,'search_surface_views',side_effect=AssertionError('BCE searched')), \
                        patch.object(core,'vat_loss',side_effect=AssertionError('BCE used VAT')), \
                        patch('vulnmechanism.change_learning.bernoulli_js',side_effect=AssertionError('BCE used JS')):
                    cp=core.train_model(None,baseline,records=parsed,variant='codebert_source',model_path=str(modelpath),source_max_length=2048,
                        epochs=3,batch_size=4,gradient_accumulation=2,learning_rate=1e-5,device='cpu',finetuning_mode='full')
                for split in ('train','valid'):_finetuning_evaluate(parsed,baseline.parent,split,cp,args)
                result=_codebert_finetuning(args)
                self.assertEqual(set(result),{'bce','vat','ours'});self.assertEqual(result['bce']['checkpoint'],str(baseline))
                self.assertFalse((root/'robustness/bce/best.pt').exists())
                smoke=json.loads((root/'robustness/smoke.metrics.json').read_text())
                self.assertEqual(smoke['untransformable'],'0')
                for mode in ('bce','vat','ours'):
                    self.assertTrue(smoke['checks'][mode]['encoder_updated']);self.assertTrue(smoke['checks'][mode]['save_reload_exact'])
                for mode in ('vat','ours'):
                    saved=torch.load(root/f'robustness/{mode}/best.pt',weights_only=False)
                    self.assertEqual(saved['training_config'],cp['training_config']);self.assertEqual(saved['robustness_options']['mode'],mode)
                    self.assertEqual([r['optimizer_steps'] for r in result[mode]['training']],[1,1,1])
                    for epoch in result[mode]['training']:
                        self.assertEqual(epoch['vat_eligible_functions'],8);self.assertGreater(epoch['vat_loss'],0)
                        self.assertEqual(epoch['augmented_functions'],0 if mode=='vat' else 7)
                        self.assertEqual(epoch['bce_function_forwards'],8 if mode=='vat' else 15)
                        if mode=='vat':self.assertEqual(epoch['candidate_queries'],0);self.assertEqual(epoch['real_js'],0)
                    self.assertEqual(result[mode]['adaptive']['all']['total'],2)
                    for k in ('fixed','heldout'):self.assertEqual(len(result[mode][k]),6)
                args.stage='train'
                with patch.object(core,'train_model',side_effect=AssertionError('completed experiment retrained')):_codebert_finetuning(args)
                bad_input=dict(saved,input_protocol=dict(saved['input_protocol'],stride=999))
                with self.assertRaisesRegex(ValueError,'input protocol differs'):
                    core._load_model(bad_input,device=torch.device('cpu'))
                corrupted=dict(saved,task_state=dict(saved['task_state']))
                del corrupted['task_state'][next(iter(corrupted['task_state']))]
                with self.assertRaises(RuntimeError):core._load_model(corrupted,device=torch.device('cpu'))

    def test_evaluation_uses_the_saved_float32_threshold_boundary(self):
        from vulnmechanism.cli import _robustness_evaluation
        from vulnmechanism.cfg_metrics import decision_boundary
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


if __name__=='__main__':unittest.main()
