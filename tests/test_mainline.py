"""Active public modes, frozen cache reuse and ordinary global VAT."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
from vulnmechanism import augmentation, model as core, syntax
from vulnmechanism.cli import build_parser, _codebert_finetuning, _finetuning_evaluate
from tests.test_surface_vat import OffsetTokenizer, tiny_model


class MainlineTests(unittest.TestCase):
    def test_single_function_cli_emits_probability_without_labels(self):
        from vulnmechanism.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'function.c';source.write_text('int f(){return 0;}')
            args=['vulnmechanism','infer','--checkpoint','fixture.pt','--source-file',str(source),'--device','cpu']
            output=io.StringIO()
            with patch('sys.argv',args),patch('torch.load',return_value={'decision_threshold':.7}),\
                    patch.object(core,'predict_checkpoint',return_value=torch.tensor([2.])) as predictor,redirect_stdout(output):
                main()
            result=json.loads(output.getvalue());self.assertEqual(result['threshold'],.7);self.assertEqual(result['prediction'],1)
            self.assertAlmostEqual(result['probability'],float(torch.sigmoid(torch.tensor(2.))),places=7)
            row=predictor.call_args.args[1][0]
            self.assertEqual(row['raw_source'],source.read_text());self.assertNotIn('label',row)

    def test_default_modes_and_retired_training_rejected_before_io(self):
        parser=build_parser();args=parser.parse_args(['codebert-finetuning'])
        self.assertEqual(args.robustness,'hard_aug_bce')
        self.assertEqual(set(core.ROBUSTNESS_MODES),{'hard_aug_bce','hard_aug_vat','bce','vat','random_aug_bce','random_aug_vat'})
        args=parser.parse_args(['codebert-finetuning','--robustness','all','--include-history','--stage','train'])
        with patch.object(core,'_read_jsonl',side_effect=AssertionError('retired training accessed data')):
            with self.assertRaisesRegex(ValueError,'cannot start training'):_codebert_finetuning(args)

    def test_global_vat_signal_dropout_norm_and_no_syntax_gate(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            _,model=tiny_model(Path(tmp));builder=core.InputBuilder(OffsetTokenizer(),source_max_length=2048,context_max_length=384)
            rows=[dict(raw_source='return external_api(1)+GLOBAL;',language='c'),dict(raw_source='int f(int x){return api(x);}',language='c')]
            state=torch.get_rng_state().clone()
            with patch.object(syntax,'surface_binding_layout',side_effect=AssertionError('global VAT parsed bindings')):
                loss,info,delta=core.vat_loss(model,rows,builder,core.VAT_CONFIG,alpha=0.,device=torch.device('cpu'),noise_seed=42,return_delta=True)
            self.assertGreater(float(loss.detach().sum()),0);self.assertEqual(info['eligible'],2)
            self.assertEqual(info['weighted_source_tokens'],0);self.assertEqual(info['zero_direction'],0)
            self.assertAlmostEqual(info['norm_min'],1.,places=6);self.assertAlmostEqual(info['norm_max'],1.,places=6)
            loss.mean().backward();self.assertGreater(float(model.task_modules['classifier'].weight.grad.norm()),0)
            self.assertGreater(float(model.encoder.embeddings.word_embeddings.weight.grad.norm()),0)
            self.assertTrue(model.training);self.assertTrue(torch.equal(state,torch.get_rng_state()))
            again,_,repeat=core.vat_loss(model,rows,builder,core.VAT_CONFIG,alpha=0.,device=torch.device('cpu'),noise_seed=42,return_delta=True)
            torch.testing.assert_close(loss,again,rtol=0,atol=0);torch.testing.assert_close(delta,repeat,rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'global VAT'):
                core.vat_loss(model,rows,builder,core.VAT_CONFIG,alpha=3.,device=torch.device('cpu'),noise_seed=42)

    def test_six_public_paths_budget_strict_reload_and_frozen_test(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);modelpath,_=tiny_model(root);manifest=root/'manifest.jsonl';baseline=root/'baseline/best.pt'
            names=['count','item','array','buffer','total','entry','length','offset']
            rows=[dict(dataset='primevul',sample_key=str(i),split='train',label=i%2,language='c',
                function=f'int f{i}(int {name}){{/* ordinary section value later routine return */return api({name})+{i};}}') for i,name in enumerate(names)]
            rows[0]['function']='return external_api(1)+GLOBAL;'
            rows += [dict(rows[i],sample_key=f'v{i}',split='valid') for i in (1,2)]
            rows += [dict(rows[0],sample_key='heldout',split='test',function='FORBIDDEN_HELDOUT')]
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            args=build_parser().parse_args(['codebert-finetuning','--manifest',str(manifest),'--model',str(modelpath),
                '--output-dir',str(root/'runs'),'--robustness','all','--baseline-checkpoint',str(baseline),'--device','cpu'])
            with patch.object(core.AutoTokenizer,'from_pretrained',return_value=OffsetTokenizer()),redirect_stdout(io.StringIO()):
                parsed=[r for r in core._read_jsonl(manifest,source_only=True) if r['split'] in ('train','valid')]
                cp=core.train_model(None,baseline,records=parsed,variant='codebert_source',model_path=str(modelpath),source_max_length=2048,
                    epochs=3,batch_size=4,gradient_accumulation=2,learning_rate=1e-5,device='cpu',finetuning_mode='full')
                for split in ('train','valid'):_finetuning_evaluate(parsed,baseline.parent,split,cp,args)
                results=_codebert_finetuning(args)
                self.assertEqual(set(results),set(core.ROBUSTNESS_MODES))
                smoke=json.loads((root/'runs/mainline.smoke.metrics.json').read_text())
                self.assertTrue(all(check['save_reload_exact'] and check['encoder_updated'] for check in smoke['checks'].values()))
                for mode,r in results.items():
                    self.assertEqual([e['optimizer_steps'] for e in r['training']],[1,1,1])
                    if mode=='bce':continue
                    saved=torch.load(root/f'runs/{mode}/best.pt',weights_only=False)
                    opts=saved['robustness_options'];self.assertEqual(opts['alpha'],0);self.assertEqual(opts['lambda_real'],0)
                    self.assertEqual(saved['cohort_members'],cp['cohort_members']);self.assertEqual(set(saved['task_state']),set(cp['task_state']))
                    for e in r['training']:
                        self.assertEqual(e['real_js'],0);self.assertEqual(e['real_function_forwards'],0)
                        self.assertEqual(e['augmented_functions'],0 if mode=='vat' else 7)
                        if mode in ('vat','random_aug_vat','hard_aug_vat'):self.assertGreater(e['vat_loss'],0)
                        else:self.assertEqual(e['vat_loss'],0)
                        if not mode.startswith('hard_'):self.assertEqual(e['candidate_queries'],0)
                originals=(root/'runs/mainline.comparison.metrics.json').read_bytes()
                args.stage='report'
                with patch.object(core,'train_model',side_effect=AssertionError('completed mode retrained')):_codebert_finetuning(args)
                self.assertEqual((root/'runs/mainline.comparison.metrics.json').read_bytes(),originals)
                args.include_incremental=True
                with self.assertRaisesRegex(FileNotFoundError,'saved evaluation caches'):
                    _codebert_finetuning(args)
                args.include_incremental=False
                rows[-1]['function']='int heldout(int orchard){return api(orchard);}'
                rows.append(dict(rows[-1],sample_key='heldout_positive',label=1))
                manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows));args.stage='test'
                with patch.object(core,'train_model',side_effect=AssertionError('Test trained')),patch.object(core,'_select_validation_threshold',side_effect=AssertionError('Test tuned threshold')):
                    tested=_codebert_finetuning(args)
                self.assertEqual(set(tested),set(results))
                for mode,r in tested.items():
                    self.assertEqual(r['threshold'],results[mode]['threshold']);self.assertEqual(r['selected_epoch'],results[mode]['selected_epoch'])
                    self.assertEqual(r['clean']['samples'],2);self.assertEqual(r['adaptive']['all']['total'],2)
                with patch.object(core,'_load_model',side_effect=AssertionError('cached Test ran inference')):_codebert_finetuning(args)
