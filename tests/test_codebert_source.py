"""Native position limits and function-normalized sliding CodeBERT readout."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import RobertaConfig, RobertaModel
from vulnmechanism import model as core
from tests.test_graph_training import TinyTokenizer, TinyEncoder
from tests.test_graph_features import sample_row


class CodeTokenizer(TinyTokenizer):
    bos_token_id = 2


class CodeBERTSourceTests(unittest.TestCase):
    def test_function_view_bce_normalizes_functions_not_flat_views(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);modelpath=root/'encoder'
            RobertaModel(RobertaConfig(vocab_size=16,hidden_size=16,num_hidden_layers=1,
                num_attention_heads=2,intermediate_size=32,max_position_embeddings=514,
                pad_token_id=0,bos_token_id=2,eos_token_id=1,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)).save_pretrained(modelpath)
            rows=[dict(dataset='primevul',sample_key=str(i),split='train',label=i%2,
                raw_source=f'int f{i}(int x){{return x+{i};}}') for i in range(8)]
            rows += [dict(rows[i],sample_key=f'v{i}',split='valid') for i in range(2)]
            views={r['sample_key']:([r['raw_source'].replace('x','item'),r['raw_source'].replace('x','data')] if i%3 else []) for i,r in enumerate(rows[:8])}
            core._seed_everything(42)
            fresh=core._build_model('codebert_source',str(modelpath),device=torch.device('cpu'),
                lora_r=16,lora_alpha=32,lora_dropout=.05,
                target_modules=('attention.self.query','attention.self.key','attention.self.value','attention.output.dense'),
                gradient_checkpointing=False,fusion_dim=256,fusion_heads=8,finetuning_mode='full')
            initial=core._cpu_state(fresh.encoder.state_dict())
            builder=core.InputBuilder(CodeTokenizer(),source_max_length=2048,context_max_length=384)
            expected=0.
            for batch,coefficients in core.function_view_batches(rows[:8],views,4):
                logits=fresh(*builder.codebert_batch(batch,device='cpu'))
                losses=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([r['label'] for r in batch],dtype=torch.float32),reduction='none')
                expected+=float((losses*losses.new_tensor(coefficients)).sum().detach())/8
            with patch.object(core.AutoTokenizer,'from_pretrained',return_value=CodeTokenizer()):
                cp=core.train_model(None,root/'best.pt',records=rows,variant='codebert_source',model_path=str(modelpath),
                    source_max_length=2048,epochs=1,batch_size=2,gradient_accumulation=4,learning_rate=1e-3,
                    device='cpu',finetuning_mode='full',function_views=views,augmentation_config=dict(mode='random'))
                epoch=[json.loads(l) for l in (root/'best.training.jsonl').open() if json.loads(l)['event']=='epoch'][0]
                self.assertAlmostEqual(epoch['train_loss'],expected,places=6)
                self.assertEqual(epoch['optimizer_steps'],1)
                self.assertEqual(epoch['bce_function_forwards'],sum(1+len(v) for v in views.values()))
                self.assertEqual(cp['augmentation_config'],dict(mode='random'))
                self.assertFalse(torch.equal(cp['encoder_state']['embeddings.word_embeddings.weight'],initial['embeddings.word_embeddings.weight']))
                loaded=core._load_model(torch.load(root/'best.pt',weights_only=False),device=torch.device('cpu'))
                before=core._predict_logits(loaded,rows[-2:],builder,variant='codebert_source',excluded_groups=(),batch_size=2,device=torch.device('cpu'))
                torch.testing.assert_close(before,core.predict_checkpoint(root/'best.pt',rows[-2:],batch_size=2,device='cpu'),rtol=0,atol=0)

    def test_augmentation_entry_uses_shared_trainer_and_matches_forward_budgets(self):
        from argparse import Namespace
        from contextlib import redirect_stdout
        import io
        from vulnmechanism.cli import _codebert_finetuning,_finetuning_evaluate
        torch.set_num_threads(2)
        class OffsetTokenizer(CodeTokenizer):
            is_fast=True
            def __call__(self,text,return_offsets_mapping=False,**kwargs):
                if text=='FORBIDDEN_HELDOUT':raise AssertionError('Test tokenized')
                result=super().__call__(text,**kwargs)
                if return_offsets_mapping:result['offset_mapping']=[(i,i+1) for i in range(len(result['input_ids']))]
                return result
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);modelpath=root/'encoder';manifest=root/'manifest.jsonl';baseline=root/'baseline/best.pt'
            RobertaModel(RobertaConfig(vocab_size=16,hidden_size=16,num_hidden_layers=1,
                num_attention_heads=2,intermediate_size=32,max_position_embeddings=514,
                pad_token_id=0,bos_token_id=2,eos_token_id=1,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)).save_pretrained(modelpath)
            rows=[dict(dataset='primevul',sample_key=str(i),split='train',label=i%2,language='c',
                       function=f'int f{i}(int x){{return x+{i};}}') for i in range(12)]
            rows += [dict(rows[i],sample_key=f'v{i}',split='valid') for i in range(2)]
            rows += [dict(rows[0],sample_key='test',split='test',function='FORBIDDEN_HELDOUT')]
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            args=Namespace(manifest=str(manifest),model=str(modelpath),output_dir=str(root/'augmented'),stage='all',
                finetuning_mode=None,device='cpu',batch_size=4,gradient_accumulation=2,
                shallow_augmentation='both',baseline_checkpoint=str(baseline))
            with patch.object(core.AutoTokenizer,'from_pretrained',return_value=OffsetTokenizer()),redirect_stdout(io.StringIO()):
                data=[r for r in core._read_jsonl(manifest,source_only=True) if r['split'] in ('train','valid')]
                cp=core.train_model(None,baseline,records=data,variant='codebert_source',model_path=str(modelpath),
                    source_max_length=2048,epochs=3,batch_size=4,gradient_accumulation=2,learning_rate=1e-5,device='cpu',finetuning_mode='full')
                for split in ('train','valid'):_finetuning_evaluate(data,baseline.parent,split,cp,args)
                result=_codebert_finetuning(args)
                self.assertEqual(set(result['valid']),{'bce','random','high_low'})
                self.assertEqual(len(result['matched_training_budgets']),3)
                self.assertEqual(result['valid']['random']['robustness']['rename']['eligible'],2)
                self.assertTrue((root/'augmented/oof.models.pkl').exists())
                for mode in ('random','high_low'):
                    saved=torch.load(root/f'augmented/{mode}/best.pt',weights_only=False)
                    self.assertEqual(saved['cohort_members'],cp['cohort_members'])
                    self.assertEqual(saved['training_config'],cp['training_config'])
                    self.assertIsNone(saved['sample_weights'])
                    self.assertIsNone(saved['initial_checkpoint'])
                    self.assertEqual(saved['augmentation_config']['mode'],mode)
                with patch.object(core,'train_model',side_effect=AssertionError('completed experiment retrained')):
                    _codebert_finetuning(args)

    def test_full_weighted_bce_uses_sample_mean_and_saves_fixed_weights(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);modelpath=root/'encoder'
            RobertaModel(RobertaConfig(vocab_size=16,hidden_size=16,num_hidden_layers=1,
                num_attention_heads=2,intermediate_size=32,max_position_embeddings=514,
                pad_token_id=0,bos_token_id=2,eos_token_id=1,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)).save_pretrained(modelpath)
            rows=[dict(dataset='primevul',sample_key=str(i),split='train',label=i%2,
                       raw_source=f'int f{i}(){{return {i};}}') for i in range(8)]
            rows += [dict(rows[i],sample_key=f'v{i}',split='valid') for i in range(2)]
            weights={str(i):w for i,w in enumerate([.5,.5,.5,.5,1.5,1.5,1.5,1.5])}
            core._seed_everything(42)
            fresh=core._build_model('codebert_source',str(modelpath),device=torch.device('cpu'),
                lora_r=16,lora_alpha=32,lora_dropout=.05,
                target_modules=('attention.self.query','attention.self.key','attention.self.value','attention.output.dense'),
                gradient_checkpointing=False,fusion_dim=256,fusion_heads=8,finetuning_mode='full')
            builder=core.InputBuilder(CodeTokenizer(),source_max_length=2048,context_max_length=384)
            initial=core._cpu_state(fresh.encoder.state_dict())
            logits=fresh(*builder.codebert_batch(rows[:8],device='cpu'))
            losses=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([r['label'] for r in rows[:8]],dtype=torch.float32),reduction='none')
            expected=float((losses*losses.new_tensor(list(weights.values()))).mean().detach())
            with patch.object(core.AutoTokenizer,'from_pretrained',return_value=CodeTokenizer()):
                cp=core.train_model(None,root/'best.pt',records=rows,variant='codebert_source',model_path=str(modelpath),
                    source_max_length=2048,epochs=1,batch_size=2,gradient_accumulation=4,
                    learning_rate=1e-3,device='cpu',finetuning_mode='full',sample_weights=weights)
                epoch=[json.loads(l) for l in (root/'best.training.jsonl').open() if json.loads(l)['event']=='epoch'][0]
                self.assertAlmostEqual(epoch['train_loss'],expected,places=6)
                self.assertEqual(epoch['optimizer_steps'],1)
                self.assertEqual(cp['sample_weights'],weights)
                self.assertFalse(torch.equal(cp['encoder_state']['embeddings.word_embeddings.weight'],initial['embeddings.word_embeddings.weight']))
                loaded=core._load_model(torch.load(root/'best.pt',weights_only=False),device=torch.device('cpu'))
                predicted=core._predict_logits(loaded,rows[-2:],builder,variant='codebert_source',excluded_groups=(),batch_size=2,device=torch.device('cpu'))
                torch.testing.assert_close(predicted,core.predict_checkpoint(root/'best.pt',rows[-2:],batch_size=2,device='cpu'),rtol=0,atol=0)

    def test_shallow_weighted_entry_runs_one_matching_full_model(self):
        from argparse import Namespace
        from contextlib import redirect_stdout
        import io
        from vulnmechanism.cli import _codebert_finetuning,_finetuning_evaluate
        torch.set_num_threads(2)
        class OffsetTokenizer(CodeTokenizer):
            is_fast=True
            def __call__(self,text,return_offsets_mapping=False,**kwargs):
                if text=='FORBIDDEN_HELDOUT':raise AssertionError('Test tokenized')
                result=super().__call__(text,**kwargs)
                if return_offsets_mapping:result['offset_mapping']=[(i,i+1) for i in range(len(result['input_ids']))]
                return result
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);modelpath=root/'encoder';manifest=root/'manifest.jsonl';baseline=root/'baseline/best.pt'
            RobertaModel(RobertaConfig(vocab_size=16,hidden_size=16,num_hidden_layers=1,
                num_attention_heads=2,intermediate_size=32,max_position_embeddings=514,
                pad_token_id=0,bos_token_id=2,eos_token_id=1,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)).save_pretrained(modelpath)
            rows=[dict(dataset='primevul',sample_key=str(i),split='train',label=i%2,
                       function=f'int f{i}(){{return {i};}}') for i in range(12)]
            rows += [dict(rows[i],sample_key=f'v{i}',split='valid') for i in range(2)]
            rows += [dict(rows[0],sample_key='test',split='test',function='FORBIDDEN_HELDOUT')]
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            args=Namespace(manifest=str(manifest),model=str(modelpath),output_dir=str(root/'weighted'),
                stage='all',finetuning_mode=None,device='cpu',batch_size=4,gradient_accumulation=2,
                shallow_reweighting=True,baseline_checkpoint=str(baseline))
            with patch.object(core.AutoTokenizer,'from_pretrained',return_value=OffsetTokenizer()),redirect_stdout(io.StringIO()):
                data=[r for r in core._read_jsonl(manifest,source_only=True) if r['split'] in ('train','valid')]
                cp=core.train_model(None,baseline,records=data,variant='codebert_source',model_path=str(modelpath),
                    source_max_length=2048,epochs=3,batch_size=4,gradient_accumulation=2,learning_rate=1e-5,device='cpu',finetuning_mode='full')
                for split in ('train','valid'):_finetuning_evaluate(data,baseline.parent,split,cp,args)
                result=_codebert_finetuning(args)
                self.assertEqual(result['config']['training'],cp['training_config'])
                self.assertEqual(result['oof']['samples'],12)
                self.assertEqual(set(result['valid']),{'bce','weighted'})
                self.assertEqual(result['valid']['weighted']['short_positive']['samples'],1)
                self.assertEqual(result['valid']['weighted']['long_negative']['samples'],0)
                weighted=torch.load(root/'weighted/best.pt',weights_only=False)
                self.assertIsNone(weighted['initial_checkpoint'])
                self.assertEqual(len(weighted['sample_weights']),12)
                self.assertEqual(weighted['input_protocol'],cp['input_protocol'])
                with patch.object(core,'train_model',side_effect=AssertionError('completed experiment retrained')):
                    _codebert_finetuning(args)

    def test_manifest_reader_preserves_native_labels_without_cpg(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'manifest.jsonl'
            rows = [dict(dataset='primevul', sample_key=str(i), split='train' if i<2 else 'valid',
                         label=i%2, function=f'int f{i}(){{return {i};}}') for i in range(4)]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            actual = core._read_jsonl(path, source_only=True)
            self.assertEqual([(r['sample_key'],r['label'],r['raw_source']) for r in actual],
                             [(r['sample_key'],r['label'],r['function']) for r in rows])
            self.assertTrue(all('cpg_relations' not in r for r in actual))
            with self.assertRaisesRegex(ValueError, 'schema'):
                core._read_jsonl(path)
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows+[rows[0]]))
            with self.assertRaisesRegex(ValueError, 'duplicate sample_key'):
                core._read_jsonl(path, source_only=True)

    def test_six_run_entry_preserves_train_valid_budget_and_reuses_results(self):
        from argparse import Namespace
        from contextlib import redirect_stdout
        import io
        from vulnmechanism.cli import _codebert_finetuning
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);modelpath=root/'encoder';manifest=root/'manifest.jsonl'
            RobertaModel(RobertaConfig(vocab_size=16,hidden_size=16,num_hidden_layers=1,
                num_attention_heads=2,intermediate_size=32,max_position_embeddings=514,
                pad_token_id=0,bos_token_id=2,eos_token_id=1,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)).save_pretrained(modelpath)
            rows=[dict(dataset='primevul',sample_key=str(i),split='train',label=i%2,
                       function=f'int f{i}(){{return {i};}}') for i in range(8)]
            rows += [dict(rows[i],sample_key=f'v{i}',split='valid') for i in range(2)]
            rows += [dict(rows[0],sample_key='heldout',split='test',function='FORBIDDEN_HELDOUT')]
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            class GuardTokenizer(CodeTokenizer):
                def __call__(self,text,**kwargs):
                    if text=='FORBIDDEN_HELDOUT':
                        raise AssertionError('Test reached tokenizer')
                    return super().__call__(text,**kwargs)
            args=Namespace(manifest=str(manifest),model=str(modelpath),output_dir=str(root/'results'),
                           stage='all',finetuning_mode=None,device='cpu',batch_size=4,gradient_accumulation=2)
            with patch.object(core.AutoTokenizer,'from_pretrained',return_value=GuardTokenizer()), redirect_stdout(io.StringIO()):
                result=_codebert_finetuning(args)
                self.assertEqual(len(result['candidates']),6)
                self.assertEqual(set(result['selected']),{'lora','full'})
                self.assertEqual(result['config']['counts'],{'train/label_0':4,'train/label_1':4,'valid/label_0':1,'valid/label_1':1})
                for selected in result['selected'].values():
                    self.assertEqual(selected['train']['samples'],8)
                    self.assertEqual(selected['valid']['samples'],2)
                histories=list((root/'results').glob('*/lr_*/best.training.jsonl'))
                self.assertEqual(len(histories),6)
                for path in histories:
                    epochs=[r for r in (json.loads(l) for l in path.open()) if r['event']=='epoch']
                    self.assertEqual([(r['epoch'],r['optimizer_steps']) for r in epochs],[(1,1),(2,1),(3,1)])
                with patch.object(core,'train_model',side_effect=AssertionError('completed run retrained')):
                    _codebert_finetuning(args)
                rows[8]['function']='int changed(){return 0;}'
                manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                args.stage='report'
                with self.assertRaisesRegex(ValueError,'members/source'):
                    _codebert_finetuning(args)

    def test_matched_modes_share_head_rng_and_initial_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = RobertaModel(RobertaConfig(vocab_size=16,hidden_size=16,num_hidden_layers=1,
                num_attention_heads=2,intermediate_size=32,max_position_embeddings=514,
                pad_token_id=0,bos_token_id=2,eos_token_id=1,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.))
            base.save_pretrained(tmp)
            models, rng = [], []
            for mode in ('lora', 'full'):
                core._seed_everything(42)
                m = core._build_model('codebert_source',tmp,device=torch.device('cpu'),
                    lora_r=16,lora_alpha=32,lora_dropout=.05,
                    target_modules=('attention.self.query','attention.self.key','attention.self.value','attention.output.dense'),
                    gradient_checkpointing=False,fusion_dim=256,fusion_heads=8,finetuning_mode=mode)
                models.append(m.eval()); rng.append(torch.rand(8))
                names = [n for n,p in m.encoder.named_parameters() if p.requires_grad]
                self.assertTrue(names)
                self.assertTrue(all('lora_' in n for n in names) if mode=='lora' else
                                all(p.requires_grad for p in m.encoder.parameters()))
            for k,v in models[0].task_modules.state_dict().items():
                torch.testing.assert_close(v,models[1].task_modules.state_dict()[k],rtol=0,atol=0)
            torch.testing.assert_close(rng[0],rng[1],rtol=0,atol=0)
            builder=core.InputBuilder(CodeTokenizer(),source_max_length=2048,context_max_length=8)
            rows=[{'raw_source':'int f(){return 0;}'},{'raw_source':'a'*1800}]
            inputs=builder.codebert_batch(rows,device='cpu')
            with torch.no_grad():
                torch.testing.assert_close(models[0](*inputs),models[1](*inputs),rtol=1e-5,atol=1e-6)
            with self.assertRaisesRegex(ValueError,'CodeBERT'):
                core._build_model('baseline',tmp,device=torch.device('cpu'),lora_r=16,lora_alpha=32,
                    lora_dropout=.05,target_modules=(),gradient_checkpointing=False,
                    fusion_dim=256,fusion_heads=8,finetuning_mode='full')

    def test_matched_modes_update_correct_parameters_and_strictly_reload(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);modelpath=root/'encoder'
            base=RobertaModel(RobertaConfig(vocab_size=16,hidden_size=16,num_hidden_layers=1,
                num_attention_heads=2,intermediate_size=32,max_position_embeddings=514,
                pad_token_id=0,bos_token_id=2,eos_token_id=1,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.))
            base.save_pretrained(modelpath)
            rows=[dict(dataset='primevul',sample_key=str(i),split='train',label=i%2,
                       raw_source=f'int f{i}(){{return {i};}}') for i in range(8)]
            rows[0]['raw_source']='int f(){int x=0;'+'x++;'*200+'return x;}'
            rows += [dict(rows[i], sample_key=f'v{i}', split='valid') for i in range(2)]
            for mode in ('lora','full'):
                with self.subTest(mode=mode), patch.object(core.AutoTokenizer,'from_pretrained',return_value=CodeTokenizer()):
                    cp=core.train_model(None,root/f'{mode}.pt',records=rows,variant='codebert_source',
                        model_path=str(modelpath),source_max_length=2048,epochs=1,batch_size=2,
                        gradient_accumulation=4,learning_rate=1e-3,device='cpu',finetuning_mode=mode)
                    self.assertEqual(cp['finetuning_mode'],mode)
                    self.assertEqual('encoder_state' in cp,mode=='full')
                    self.assertEqual('adapter_state' in cp,mode=='lora')
                    m=core._load_model(cp,device=torch.device('cpu'))
                    if mode=='full':
                        self.assertFalse(torch.equal(cp['encoder_state']['embeddings.word_embeddings.weight'],
                                                     base.state_dict()['embeddings.word_embeddings.weight']))
                        self.assertFalse(torch.equal(cp['encoder_state']['encoder.layer.0.attention.self.query.weight'],
                                                     base.state_dict()['encoder.layer.0.attention.self.query.weight']))
                        broken=dict(cp,encoder_state=dict(cp['encoder_state']))
                        broken['encoder_state'].pop('embeddings.word_embeddings.weight')
                        with self.assertRaisesRegex(RuntimeError,'Missing key'):
                            core._load_model(broken,device=torch.device('cpu'))
                        with self.assertRaisesRegex(ValueError,'state'):
                            core._load_model(dict(cp,finetuning_mode='lora'),device=torch.device('cpu'))
                    else:
                        self.assertTrue(any('lora_B' in k and torch.count_nonzero(v)>0 for k,v in cp['adapter_state'].items()))
                        torch.testing.assert_close(m.encoder.get_base_model().embeddings.word_embeddings.weight,
                                                   base.embeddings.word_embeddings.weight,rtol=0,atol=0)
                    builder=core.InputBuilder(CodeTokenizer(),source_max_length=2048,context_max_length=384)
                    expected=core._predict_logits(m,rows[-2:],builder,variant='codebert_source',
                        excluded_groups=(),batch_size=2,device=torch.device('cpu'))
                    actual=core.predict_checkpoint(root/f'{mode}.pt',rows[-2:],batch_size=2,device='cpu')
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    result=core.evaluate_model(None,root/f'{mode}.pt',records=rows,split='valid',device='cpu',
                        prediction_path=root/f'{mode}.valid.predictions.jsonl')
                    self.assertEqual(result['samples'],2)
                    self.assertIn('fp',result['metrics']);self.assertIn('fn',result['metrics'])
                    self.assertEqual(result['mechanism_subgroups'],{})

    def test_original_cohort_preserves_imbalance_and_order(self):
        from vulnmechanism.benchmark_view import select_source_records
        rows=[sample_row('b',label=0),sample_row('a',label=0),sample_row('c',label=1),
              sample_row('v0','valid',0),sample_row('v1','valid',1)]
        selected,info=select_source_records(rows,'primevul',preserve_members=True)
        self.assertEqual(selected,rows)
        self.assertEqual(info['dropped_primevul_balance_records'],0)
        self.assertEqual(info['split_counts'],{'train':3,'valid':2})

    def test_window_coverage_padding_and_short_equivalence(self):
        tok=CodeTokenizer();rows=[{'raw_source':'a'*31},{'raw_source':'abcd'*800}]
        builder=core.InputBuilder(tok,source_max_length=2048,context_max_length=8)
        ids,mask,owners,weights,count=builder.codebert_batch(rows,device='cpu')
        self.assertLessEqual(ids.shape[1],512)
        self.assertEqual(count,2)
        self.assertEqual(int((owners==0).sum()),1)
        self.assertGreater(int((owners==1).sum()),1)
        self.assertTrue(torch.all(weights[mask==0]==0))
        encoder=TinyEncoder()
        pooled=core._window_mean(encoder(ids).last_hidden_state,mask,owners,weights,count)
        expected=[]
        for row in rows:
            src=builder._encode(row['raw_source'],2048-len(builder.source_prefix)-2)
            direct=torch.tensor([[tok.bos_token_id,*builder.source_prefix,*src,tok.eos_token_id]])
            expected.append(encoder(direct).last_hidden_state.mean(1)[0])
        torch.testing.assert_close(pooled,torch.stack(expected),rtol=1e-5,atol=1e-6)
        native=core.InputBuilder(tok,source_max_length=512,context_max_length=8)
        short=native.codebert_batch(rows[:1],device='cpu')
        self.assertTrue(torch.equal(ids[:1,:short[0].shape[1]],short[0]))
        self.assertEqual(int(native.codebert_batch(rows[1:],device='cpu')[1].sum()),512)
        pooled.square().sum().backward()
        self.assertGreater(float(encoder.embedding.weight.grad.abs().sum()),0.)

    def test_missing_function_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError,'every function'):
            core._window_mean(torch.ones(1,3,2),torch.ones(1,3),torch.tensor([0]),torch.ones(1,3),2)
        builder=core.InputBuilder(CodeTokenizer(),source_max_length=1024,context_max_length=8)
        with self.assertRaises(ValueError):builder.codebert_batch([{'raw_source':'a'}],device='cpu')

    def test_real_roberta_lora_train_save_load_and_evaluate(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);modelpath=root/'encoder'
            base=RobertaModel(RobertaConfig(vocab_size=16,hidden_size=16,num_hidden_layers=1,
                num_attention_heads=2,intermediate_size=32,max_position_embeddings=514,
                pad_token_id=0,bos_token_id=2,eos_token_id=1,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.))
            base.save_pretrained(modelpath)
            rows=[sample_row(str(i),label=i%2) for i in range(4)]
            rows += [sample_row('v0','valid',0),sample_row('v1','valid',1)]
            rows[0]['raw_source']='int f(){int x=0;'+ 'x++;'*200+'return x;}'
            with patch.object(core.AutoTokenizer,'from_pretrained',return_value=CodeTokenizer()):
                cp=core.train_model(None,root/'best.pt',records=rows,variant='codebert_source',
                    model_path=str(modelpath),source_max_length=2048,epochs=1,batch_size=2,
                    gradient_accumulation=2,lora_r=2,lora_alpha=4,lora_dropout=0.,device='cpu')
                self.assertEqual(cp['training_config']['epochs'],1)
                self.assertEqual(cp['input_protocol']['native_window'],512)
                self.assertTrue(any('attention.output.dense' in k for k in cp['adapter_state']))
                self.assertTrue(any('lora_B' in k and torch.count_nonzero(v)>0 for k,v in cp['adapter_state'].items()))
                scores=core.predict_checkpoint(root/'best.pt',rows[-2:],batch_size=2,device='cpu')
                self.assertEqual(tuple(scores.shape),(2,))
                self.assertTrue(torch.isfinite(scores).all())
                records=root/'rows.jsonl'
                records.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                result=core.evaluate_model(records,root/'best.pt',split='valid',device='cpu',
                    prediction_path=root/'valid.predictions.jsonl')
                pred=[json.loads(x) for x in (root/'valid.predictions.jsonl').read_text().splitlines()]
                torch.testing.assert_close(scores,torch.tensor([r['logit'] for r in pred]))
                self.assertEqual(result['samples'],2)
                self.assertGreater(result['bce'],0.)
                self.assertTrue(all(r['windows']==1 for r in pred))
                with self.assertRaises(FileExistsError):core.evaluate_model(records,root/'best.pt',split='valid',
                    prediction_path=root/'valid.predictions.jsonl')

if __name__=='__main__':unittest.main()
