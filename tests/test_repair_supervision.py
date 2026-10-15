"""Audited regions, event-safe contrast and the shared function trainer."""
import copy
import io
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch

from vulnmechanism import model as core, repair_supervision as audit
from vulnmechanism.cli import build_parser, _codebert_finetuning
from tests.test_surface_vat import OffsetTokenizer, tiny_model, MATERIALS


def location(source, left=0, right=None, *, byte_columns=False):
    right = len(source) if right is None else right
    values = []
    for position in (left, right):
        line = source.count('\n', 0, position) + 1
        beginning = source.rfind('\n', 0, position) + 1
        column = len(source[beginning:position].encode()) + 1 if byte_columns else position - beginning + 1
        values.append((line, column))
    return dict(start_line=values[0][0], start_column=values[0][1], end_line=values[1][0], end_column=values[1][1],
                source_text=source[left:right].replace('\r\n', '\n'))


def sample(key='a', before='int f(int n){return api(n);}', after='int f(int n){if(n<0)return 0;return api(n);}'):
    return dict(dataset='MegaVul', sample_id=key, project='project_' + key, cve='', fix_commit='',
                code_before=before, code_after=after, vulnerability_type='bounds_read',
                unsafe_behavior='The operation lacks the reviewed bound.', security_change='Add the reviewed bound.',
                keep_reason='Retain this local mechanism; no whole-function safety conclusion.',
                review_annotations=dict(repair_pattern='bounds_check', locations=dict(
                    before=[location(before)], after=[location(after)])),
                pair_provenance=dict(source_path='megavul/megavul.json'))


def builder():
    return core.InputBuilder(OffsetTokenizer(), source_max_length=2048, context_max_length=384)


class RepairDataTests(unittest.TestCase):
    def test_exact_source_split_and_conflicts_not_name_guessing(self):
        row = sample('primevul_valid_000001')
        row['dataset'] = 'PrimeVul';row['pair_provenance']['source_path'] = r'PrimeVul_v0.1\primevul_valid_paired.jsonl'
        self.assertEqual(audit.source_partition(row), 'valid')
        row['pair_provenance']['source_path'] = 'custom.jsonl'
        self.assertIsNone(audit.source_partition(row))
        row['pair_provenance']['source_path'] = 'primevul_train_paired.jsonl'
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            audit.source_partition(row)

    def test_copy_filters_whole_pairs_preserves_source_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);neighbor = root / 'neighbor';neighbor.mkdir();raw = root / 'raw/PrimeVul_v0.1';raw.mkdir(parents=True)
            samples = []
            for split, indices in [('train', [11, 12]), ('valid', [21, 22]), ('test', [31, 32])]:
                row = sample(f'primevul_{split}_000001');row['dataset'] = 'PrimeVul'
                row['pair_provenance'] = dict(source_path=f'PrimeVul_v0.1/primevul_{split}_paired.jsonl', source_row_indices=indices)
                samples.append(row)
                (raw / f'primevul_{split}_paired.jsonl').write_text(''.join(json.dumps(dict(idx=index, target=1-side,
                    func=row['code_' + ('before' if side == 0 else 'after')])) + '\n' for side, index in enumerate(indices)))
            # Different sources avoid accidental near/equal-event exclusions.
            samples[0]['code_before'] = 'void train(){write(35);}'
            samples[0]['code_after'] = 'void train(){check(35);write(35);}'
            (raw / 'primevul_train_paired.jsonl').write_text(''.join(json.dumps(dict(idx=index,target=1-side,
                 func=samples[0]['code_' + ('before' if side == 0 else 'after')]))+'\n' for side,index in enumerate([11,12])))
            document = dict(schema_version=2, rag_knowledge_text_generated=False, heldout_exclusion={'old': 15}, samples=samples)
            source = neighbor / 'high_quality_repair_pairs.json';source.write_text(json.dumps(document))
            original = source.read_bytes();destination = root / 'high_quality_repair_pairs.json'
            retained, valid, stats = audit.copy_training_source(neighbor, destination, [], raw_root=root/'raw')
            self.assertEqual(retained, samples[:1]);self.assertEqual(valid, samples[1:2])
            self.assertEqual(stats['official_removed'], dict(valid=1, test=1))
            self.assertEqual(json.loads(destination.read_text()), {k:v if k!='samples' else samples[:1] for k,v in document.items() if k!='heldout_exclusion'})
            before_time=destination.stat().st_mtime_ns
            audit.copy_training_source(neighbor,destination,[],raw_root=root/'raw')
            self.assertEqual(destination.stat().st_mtime_ns,before_time);self.assertEqual(source.read_bytes(),original)
            changed=json.loads(destination.read_text());changed['samples'][0]['keep_reason']='USER EDIT';destination.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'preserve user edits'):
                audit.copy_training_source(neighbor,destination,[],raw_root=root/'raw')
            self.assertEqual(json.loads(destination.read_text()),changed)

    def test_either_endpoint_cve_commit_and_native_id_exclude_whole_pair(self):
        heldout = dict(dataset='primevul',split='valid',sample_key='primevul:42',idx=42,label=0,
                       raw_source='void protected(){read(200);}',repo='same_project',commit='a'*40,cve='CVE-2024-1234')
        pairs = [sample(str(i),before=f'void f{i}(){{read({i});}}',after=f'void f{i}(){{check({i});read({i});}}') for i in range(5)]
        pairs[0]['code_after'] = heldout['raw_source'];pairs[1]['cve'] = 'CVE-2024-1234'
        pairs[2]['fix_commit'] = 'a'*12
        pairs[3]['dataset']='PrimeVul';pairs[3]['sample_id']='primevul_train_000003'
        pairs[3]['pair_provenance']=dict(source_path='primevul_train_paired.jsonl',source_row_indices=[42,43])
        pairs[4]['project'] = 'same_project'
        retained, excluded=audit.isolate_samples(pairs,[heldout],[])
        self.assertEqual(retained,pairs[4:]);self.assertEqual(len(excluded),4)
        self.assertIn('protected_native_sample', [e['reason'] for e in excluded[3]['evidence']])

    def test_existing_near_duplicate_gate(self):
        code='int f(){'+''.join(f'a{i}={i};' for i in range(100))+'return a1;}'
        protected=dict(dataset='sven',split='external_test',sample_key='sv',label=1,raw_source=code,repo='',commit='')
        row=sample(before=code.replace('a99=99','a99=98'))
        retained,excluded=audit.isolate_samples([row],[protected],[])
        self.assertFalse(retained)
        self.assertIn('protected_near_duplicate',[e['reason'] for e in excluded[0]['evidence']])

    def test_different_source_files_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);(p/'one').mkdir();(p/'two').mkdir()
            (p/'one/high_quality_repair_pairs.json').write_text('{"samples":[]}')
            (p/'two/high_quality_repair_pairs.json').write_text('{"samples":[1]}')
            with self.assertRaisesRegex(ValueError,'different repair sources'):
                audit.find_source(p)


class RepairRegionTests(unittest.TestCase):
    def test_crlf_unicode_multiline_and_byte_coordinates(self):
        code='int 函数(){\r\n  int 变量=2;\r\n  return 变量;\r\n}'
        left,right=code.index('变量'),code.index(';')+1
        for byte in (False,True):
            loc=location(code,left,right,byte_columns=byte)
            self.assertEqual(audit.location_span(code,loc),(left,right))
        loc=location(code,code.index('int 变量'),code.index('return 变量')+len('return 变量;'))
        self.assertEqual(audit.location_span(code,loc),(code.index('int 变量'),code.index('return 变量')+len('return 变量;')))
        loc['source_text']='WRONG'
        with self.assertRaisesRegex(ValueError,'source_text mismatch'):
            audit.location_span(code,loc)

    def test_overlap_union_comments_and_padding_not_in_pool(self):
        code='int f(){/* ordinary */return 4;}'
        loc=location(code,code.index('return'),code.index('4')+1)
        whole=audit.source_region(code,[loc],builder())
        repeated=audit.source_region(code,[loc,loc,location(code,code.index('return'),code.index('return')+3)],builder())
        self.assertEqual(whole['token_weights'],repeated['token_weights'])
        self.assertEqual(sum(whole['token_weights']),7)
        full=audit.source_region(code,[location(code)],builder())
        self.assertTrue(all(weight==0 for weight in full['token_weights'][code.index('/*'):code.index('*/')+2]))
        with self.assertRaisesRegex(ValueError,'empty_code_region'):
            audit.source_region(code,[location(code,code.index('/*'),code.index('*/')+2)],builder())

    def test_cross_line_token_fraction(self):
        class CrossLine(OffsetTokenizer):
            def __call__(self,text,return_offsets_mapping=False,**kwargs):
                if text=='a\nb':
                    result={'input_ids':[5]}
                    if return_offsets_mapping:result['offset_mapping']=[(0,3)]
                    return result
                return super().__call__(text,return_offsets_mapping=return_offsets_mapping,**kwargs)
        b=core.InputBuilder(CrossLine(),source_max_length=2048,context_max_length=384)
        value=audit.source_region('a\nb',[location('a\nb',2,3)],b)
        self.assertEqual(value['token_weights'],[1/3])

    def test_window_dedup_hand_tensor_and_gradient(self):
        hidden=torch.tensor([[[99.],[2.],[4.],[88.]],[[99.],[6.],[8.],[88.]]],requires_grad=True)
        mask=torch.ones(2,4,dtype=torch.long);owners=torch.tensor([0,0])
        weights=torch.tensor([[.5,1.,.5,.5],[.5,.5,1.,.5]])
        positions=torch.tensor([[-1,0,1,-1],[-1,1,2,-1]])
        actual=core.region_mean(hidden,mask,owners,weights,1,positions,[[0.,1.,1.]])
        self.assertEqual(actual.item(),6.5)
        actual.sum().backward();self.assertEqual(hidden.grad[0,0].item(),0)
        self.assertEqual(hidden.grad[0,2].item(),.25);self.assertEqual(hidden.grad[1,2].item(),.5)
        with self.assertRaisesRegex(ValueError,'visible pooling tokens'):
            core.region_mean(hidden,mask,owners,weights,1,positions,[[0.,0.,0.]])
        with self.assertRaisesRegex(ValueError,'actual source positions'):
            core.region_mean(hidden,mask,owners,weights,1,positions,[[1.]])

    def test_truncation_identical_input_and_unchanged_operation_context(self):
        b=builder()
        before='int f(int n){return api(n);}'
        after='int f(int n){if(n<0)return 0;return api(n);}'
        row=sample(before=before,after=after)
        for side in ('before','after'):
            code=row['code_'+side];start=code.index('api(n)')
            row['review_annotations']['locations'][side]=[location(code,start,start+6)]
        prepared,stats=audit.prepare_regions([row],b)
        self.assertEqual(stats['usable_pairs'],1)
        row=sample(before=before,after=before)
        self.assertEqual(audit.prepare_regions([row],b)[1]['skip_counts'],{'identical_actual_encoder_input':1})
        after=before+'/*'+'x'*2100+'*/ return changed;'
        row=sample(before=before,after=after)
        row['review_annotations']['locations']['after']=[location(after,0,len(before))]
        self.assertIn('changed_code_truncated_or_offset_uncovered',audit.prepare_regions([row],b)[1]['skip_counts'])
        with self.assertRaisesRegex(ValueError,'region_truncated'):
            audit.source_region(after,[location(after)],b)


class RepairLossTests(unittest.TestCase):
    def pairs(self):
        return [dict(sample_id=str(i),group=('bounds_read','bounds_check'),event_id=str(i),project=str(i)) for i in range(3)]

    def test_stable_manual_formula_and_gradients(self):
        vectors=torch.tensor([[1.,0.],[-1.,0.],[.9,.1],[-.9,.1]],requires_grad=True)
        logits=torch.tensor([.2,-.3,.4,-.5],requires_grad=True)
        pairs=self.pairs()[:2]
        state,contrast,info=audit.local_losses(vectors,logits,pairs,beta=.1,tau=.1)
        sim=torch.nn.functional.normalize(vectors,dim=1)@torch.nn.functional.normalize(vectors,dim=1).T/.1
        expected=[]
        for i in range(4):
            positive=2 if i==0 else 3 if i==1 else 0 if i==2 else 1
            negatives=[j for j in range(4) if j%2!=i%2]
            expected.append(torch.logaddexp(sim[i,positive],torch.logsumexp(sim[i,negatives],0))-sim[i,positive])
        torch.testing.assert_close(contrast,torch.stack(expected).reshape(2,2).mean(1))
        self.assertEqual(info['contrast_anchors'],4)
        contrast.mean().backward();self.assertGreater(float(vectors.grad.norm()),0);self.assertIsNone(logits.grad)
        state.mean().backward();self.assertGreater(float(logits.grad.norm()),0)

    def test_group_event_self_exclusions_and_beta_zero(self):
        pairs=self.pairs();pairs[1]['event_id']='0';pairs[2]['group']=('other_group','input_validation')
        vectors=torch.randn(6,5,requires_grad=True);logits=torch.randn(6,requires_grad=True)
        state,contrast,info=audit.local_losses(vectors,logits,pairs,beta=.1,tau=.1)
        self.assertEqual(info['contrast_anchors'],0);self.assertEqual(float(contrast.sum()),0)
        self.assertTrue(torch.isfinite(state).all())
        with patch.object(torch.nn.functional,'normalize',side_effect=AssertionError('contrast ran')):
            _,contrast,info=audit.local_losses(vectors,logits,pairs,beta=0,tau=.1)
        self.assertFalse(contrast.requires_grad);self.assertTrue(info['contrast_disabled'])

    def test_alignment_separation_reduces_loss(self):
        bad=torch.tensor([[1.,0.],[.9,.1],[-1.,0.],[-.9,.1]])
        good=torch.tensor([[1.,0.],[-1.,0.],[.9,.1],[-.9,.1]])
        p=self.pairs()[:2];z=torch.zeros(4)
        self.assertLess(float(audit.local_losses(good,z,p,beta=.1,tau=.1)[1].mean()),
                        float(audit.local_losses(bad,z,p,beta=.1,tau=.1)[1].mean()))

    def test_schedule_once_tail_and_epoch_scaling_without_main_division(self):
        pairs=self.pairs()+[dict(self.pairs()[0],sample_id='extra',event_id='extra')]
        pairs[-1]['group']=None
        random.seed(19);before=random.getstate()
        schedule=audit.auxiliary_schedule(pairs,5,seed=42,epoch=1)
        self.assertEqual(random.getstate(),before)
        flattened=[i for batches in schedule.values() for batch in batches for i in batch]
        self.assertEqual(sorted(flattened),list(range(len(pairs))))
        self.assertEqual(schedule,audit.auxiliary_schedule(pairs,5,seed=42,epoch=1))
        scale=audit.auxiliary_scale(5,len(pairs),.1)
        self.assertAlmostEqual(sum(scale for _ in flattened)/5,.1)

    def test_length_directions_and_visible_budget_ties(self):
        rows=[dict(local_logits=[1.,-1.],global_logits=[1.,0.],source_chars=[100,200],
                   actual_tokens=[2000,2200],visible_source_tokens=[2000,2025]),
              dict(local_logits=[0.,0.],global_logits=[0.,0.],source_chars=[300,400],
                   actual_tokens=[2400,2600],visible_source_tokens=[2025,2025])]
        result=audit.local_summary(rows)
        self.assertEqual(result['length_baselines']['token_length']['shorter_is_riskier_accuracy'],1)
        self.assertEqual(result['length_baselines']['visible_token_length']['ties'],1)
        self.assertEqual(result['length_baselines']['visible_token_length']['shorter_is_riskier_accuracy_half_ties'],.75)
        self.assertEqual(result['ranking_ties'],1)


class RepairTrainingTests(unittest.TestCase):
    def test_public_none_never_reads_audit_and_incompatible_modes_reject(self):
        args=build_parser().parse_args(['codebert-finetuning','--repair-pairs','DO_NOT_READ'])
        with (patch('vulnmechanism.cli._codebert_robustness',return_value='original') as old,
              patch.object(audit,'copy_training_source',side_effect=AssertionError('audit accessed'))):
            self.assertEqual(_codebert_finetuning(args),'original');old.assert_called_once_with(args)
        args.repair_supervision='state';args.robustness='hard_aug_vat'
        with self.assertRaisesRegex(ValueError,'no VAT'):_codebert_finetuning(args)
        args.robustness='hard_aug_bce';args.stage='test'
        with self.assertRaisesRegex(ValueError,'no VAT/Test'):_codebert_finetuning(args)

    def test_shared_encoder_head_gradients_dropout_rng_and_reload(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path,model=tiny_model(root);b=builder()
            pairs,stats=audit.prepare_regions([sample('first'),sample('second',before='int g(int n){return read(n);}',
                    after='int g(int n){if(n<2)return 0;return read(n);}')],b)
            self.assertEqual(stats['usable_pairs'],2)
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            model.train();torch.manual_seed(10);rng=torch.random.get_rng_state().clone()
            with core._auxiliary_mode(model):
                vectors,logits,_,_=audit.region_readouts(model,b,pairs,device='cpu')
                state,contrast,_=audit.local_losses(vectors,logits,pairs,beta=.1,tau=.1)
                (state+.1*contrast).sum().backward()
                self.assertTrue(all(not module.training for module in model.modules() if isinstance(module,torch.nn.Dropout)))
            self.assertTrue(model.training);torch.testing.assert_close(rng,torch.random.get_rng_state(),rtol=0,atol=0)
            self.assertGreater(float(model.encoder.get_input_embeddings().weight.grad.norm()),0)
            self.assertGreater(float(model.task_modules['classifier'].weight.grad.norm()),0)
            original=model.encoder.get_input_embeddings().weight.detach().clone()
            rows=[dict(dataset='primevul',split='train' if i<3 else 'valid',sample_key=str(i),label=i%2,
                       raw_source=f'int f{i}(int n){{return api(n)+{i};}}',language='c') for i in range(5)]
            with patch.object(core.AutoTokenizer,'from_pretrained',return_value=OffsetTokenizer()),redirect_stdout(io.StringIO()):
                cp=core.train_model(None,root/'best.pt',records=rows,variant='codebert_source',model_path=str(path),
                    finetuning_mode='full',source_max_length=2048,batch_size=2,gradient_accumulation=2,epochs=1,
                    learning_rate=1e-5,device='cpu',robustness_options=core.robustness_config('hard_aug_bce'),surface_materials=MATERIALS,
                    repair_supervision=audit.supervision_config('state_contrast'),repair_pairs=pairs,repair_pairs_path='NOT_REQUIRED_FOR_INFERENCE')
                loaded=core._load_model(cp,device=torch.device('cpu'))
                expected=core._predict_logits(loaded,rows,b,variant='codebert_source',excluded_groups=(),batch_size=2,device=torch.device('cpu'))
                actual=core.predict_checkpoint(root/'best.pt',rows,device='cpu',batch_size=2)
            torch.testing.assert_close(expected,actual,rtol=0,atol=0)
            self.assertFalse(torch.equal(original,cp['encoder_state']['embeddings.word_embeddings.weight']))
            history=[json.loads(line) for line in (root/'best.training.jsonl').open() if json.loads(line)['event']=='epoch'][0]
            self.assertEqual(history['optimizer_steps'],1);self.assertEqual(history['audit_pairs'],2)
            self.assertEqual(history['audit_contrast_anchors'],4);self.assertGreater(history['audit_contrast_loss'],0)
            self.assertAlmostEqual(history['audit_scale_per_update'],.05)
            with self.assertRaisesRegex(ValueError,'configuration'):
                core._load_model(dict(cp,repair_supervision=dict(cp['repair_supervision'],tau=.2)),device=torch.device('cpu'))


if __name__=='__main__':
    unittest.main()
