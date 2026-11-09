"""Author loss, original members, fixed decisions and callback resume checks."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import RobertaModel

from tests.test_losver import config
from vulnmechanism.official_baselines import (UPSTREAM, native_begin, native_batches,
    native_step, native_epoch, native_predictions, official_rows)


class OfficialBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_real_members_and_format_exports_keep_all_three_splits(self):
        import pandas as pd
        from vulnmechanism.data import read_jsonl
        for split, pairs in [('train',4354),('valid',562),('test',564)]:
            rows=official_rows('data',split)
            csv_rows=pd.read_csv(f'data/official_baselines/linevul/{split}.csv').to_dict('records')
            self.assertEqual(len(csv_rows),2*pairs)
            self.assertEqual([r['func'] for r in rows],[r['processed_func'] for r in csv_rows])
            for method in ('svuld','regvd'):
                exported=read_jsonl(f'data/official_baselines/{method}/{split}.jsonl')
                self.assertEqual([r['sample_key'] for r in rows],[r['sample_key'] for r in exported])
                self.assertEqual([r['target'] for r in rows],[r['label' if method=='svuld' else 'target'] for r in exported])
                self.assertEqual([r['func'] for r in rows],[r['code' if method=='svuld' else 'func'] for r in exported])
                if method=='svuld':
                    self.assertEqual([r['contrast'] for r in exported],
                        [rows[i ^ 1]['func'] if split=='train' else r['func'] for i,r in enumerate(rows)])

    def test_author_svuld_rdrop_loss_and_views_are_preserved(self):
        path=Path('external/SVulD/Source_Code/SVulD/model.py')
        spec=importlib.util.spec_from_file_location('author_svuld_model',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        c=config()
        args=SimpleNamespace(r_drop=True,do_train=True,simcse=False,simct=False)
        model=module.Model(RobertaModel(c),c,None,args).train()
        ids=torch.tensor([[0,5,8,2,1],[0,9,7,2,1]])
        labels=torch.tensor([1,0])
        torch.manual_seed(13)
        loss,prob=model(ids,ids.flip(0),labels)
        torch.manual_seed(13)
        a=model.classifier(model.get_xcode_vec(ids));b=model.classifier(model.get_xcode_vec(ids))
        expected=.5*(nn.functional.cross_entropy(a,labels)+nn.functional.cross_entropy(b,labels))+.1*module.compute_kl_loss(a,b)
        torch.testing.assert_close(loss,expected,rtol=0,atol=0)
        torch.testing.assert_close(prob,a.softmax(-1),rtol=0,atol=0)
        loss.backward()
        self.assertGreater(float(model.encoder.embeddings.word_embeddings.weight.grad.norm()),0)
        torch.manual_seed(13)
        other,_=model(ids,ids.roll(1,1),labels)
        torch.testing.assert_close(other,loss,rtol=0,atol=0)

    def test_original_budgets_probability_columns_and_models_are_not_unified(self):
        original=('linevul','svuld','regvd')
        self.assertEqual([UPSTREAM[m]['epochs'] for m in original],[10,20,100])
        self.assertEqual([UPSTREAM[m]['block_size'] for m in original],[512,400,400])
        self.assertEqual([UPSTREAM[m]['probability_column'] for m in original],[1,1,0])
        self.assertEqual(UPSTREAM['svuld']['sampler'],'SequentialSampler')
        self.assertTrue(UPSTREAM['svuld']['r_drop'])
        self.assertEqual(UPSTREAM['regvd']['model'],'graphcodebert-base')
        for method in UPSTREAM:
            path=Path('external')/UPSTREAM[method]['directory']
            import subprocess
            self.assertEqual(subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip(),UPSTREAM[method]['commit'])

    def test_vulmixer_author_budget_static_embedding_and_unchanged_model(self):
        import subprocess
        import sys
        from vulnmechanism.official_baselines import mixer_embedding_cache
        spec=UPSTREAM['vulmixer']
        self.assertEqual((spec['epochs'],spec['block_size'],spec['batch_size']), (20,1024,128))
        self.assertEqual((spec['learning_rate'],spec['weight_decay'],spec['gradient_clip']),(.0005,.01,None))
        self.assertTrue(spec['fp16'])
        repo=Path('external/VulMixer').resolve()
        self.assertEqual(subprocess.check_output(['git','-C',str(repo),'diff','--','models','cl.py','cl_dataset.py','dataset.py'],text=True),'')
        sys.path.insert(0,str(repo))
        try:
            from util import load_config
            from models.fire.model import FireSVD
            cfg=load_config(repo/'config/model/vulmixer.yml')
            model=FireSVD(cfg)
            x=torch.randn(2,1024,768)
            torch.manual_seed(8)
            out=model(x)
            self.assertEqual(tuple(out.shape),(2,2))
            out.square().mean().backward()
            for name in ('statement_info','function_info','block_info'):
                branch=getattr(model.mixers[0].layers[0].token_mixer,name)
                self.assertGreater(float(branch.weight.grad.norm()),0.)
            model.eval()
            state=torch.get_rng_state()
            a=model(x)
            torch.set_rng_state(state)
            torch.testing.assert_close(a,model(x),rtol=0,atol=0)
        finally:
            sys.path.remove(str(repo))

    def test_vulmixer_exports_keep_every_original_id_label_and_author_format(self):
        import subprocess
        import sys
        from vulnmechanism.data import read_jsonl
        for split in ('train','valid','test'):
            original=official_rows('data',split)
            exported=read_jsonl(f'data/official_baselines/vulmixer/{split}.jsonl')
            keys=lambda rows:[(r['sample_key'],r['pair_key'],r['target']) for r in rows]
            self.assertEqual(keys(original),keys(exported))
            for index in (0,len(original)-1):
                expected=subprocess.check_output([str(Path(sys.executable).with_name('clang-format')),
                    '-style=file:external/VulMixer/config/clang-format.yml'],input=original[index]['func'],text=True)
                self.assertEqual(exported[index]['func'],expected)

    def test_vulmixer_resume_preserves_full_framework_state_and_truncates_history(self):
        from vulnmechanism.official_baselines import mixer_callback
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);log=run/'best.training.jsonl'
            log.write_text('committed\n')
            args=SimpleNamespace(reproduction_run=run,reproduction_metadata={'seed':42})
            callback=mixer_callback(args)
            args.reproduction_loop.update(epoch=3,global_step=207)
            torch.manual_seed(12)
            checkpoint={};callback.on_save_checkpoint(None,None,checkpoint)
            expected_rng=torch.get_rng_state().clone()
            log.write_text('committed\nuncommitted\n')
            args2=SimpleNamespace(reproduction_run=run,reproduction_metadata={'seed':42})
            resumed=mixer_callback(args2)
            resumed.on_load_checkpoint(None,None,checkpoint)
            torch.manual_seed(999)
            resumed.on_fit_start(None,None)
            torch.testing.assert_close(torch.get_rng_state(),expected_rng,rtol=0,atol=0)
            self.assertEqual(args2.reproduction_loop['global_step'],207)
            self.assertEqual(log.read_text(),'committed\n')
            checkpoint['official_reproduction']['metadata']={'seed':1234}
            with self.assertRaisesRegex(ValueError,'configuration'):
                resumed.on_load_checkpoint(None,None,checkpoint)

    def test_vulberta_uses_original_notebook_clang_bpe_and_dataset(self):
        from vulnmechanism.official_baselines import vulberta_notebook_components
        namespace,book=vulberta_notebook_components()
        clean=namespace['cleaner']
        self.assertEqual(clean('int f(){\n// comment\nreturn 1;\t}'),'int f(){return 1;}')
        texts=['int f(int *p) { return *p; }','int g(void) { return 0; }']
        encoded=namespace['my_tokenizer'].encode_batch(texts)
        for e in encoded:
            self.assertEqual(e.ids[0],0)
            self.assertEqual(e.ids[sum(e.attention_mask)-1],2)
        dataset=namespace['MyCustomDataset'](namespace['process_encodings'](encoded),[1,0])
        self.assertEqual([int(dataset[i]['labels']) for i in range(2)],[1,0])
        self.assertIn('RobertaForSequenceClassification.from_pretrained', ''.join(book['cells'][27]['source']))
        self.assertEqual((UPSTREAM['vulberta']['epochs'],UPSTREAM['vulberta']['block_size']), (10,1024))

    def test_original_adamw_accepts_accelerate_closure_without_changing_updates(self):
        from vulnmechanism.training import LOSVERAdamW
        p=nn.Parameter(torch.tensor([.2,-.4]))
        q=nn.Parameter(p.detach().clone())
        a,b=LOSVERAdamW([p],lr=.003),LOSVERAdamW([q],lr=.003)
        for step in range(3):
            p.grad=torch.tensor([.1+step,.2])
            q.grad=p.grad.clone()
            self.assertIsNone(a.step())
            self.assertIsNone(b.step(None))
            torch.testing.assert_close(p,q,rtol=0,atol=0)
        def closure():
            self.assertTrue(torch.is_grad_enabled())
            b.zero_grad();loss=q.square().sum();loss.backward();return loss
        before=q.detach().square().sum()
        torch.testing.assert_close(b.step(closure),before,rtol=0,atol=0)

    def test_real_lightning_epoch_resume_matches_shuffle_dropout_and_optimizer(self):
        import lightning.pytorch as pl
        from vulnmechanism.official_baselines import mixer_callback
        from vulnmechanism.progress import TrainingProgress

        class Module(pl.LightningModule):
            def __init__(self):
                super().__init__();self.model=nn.Sequential(nn.Dropout(.25),nn.Linear(4,2));self.order=[]
            def training_step(self,batch,index):
                x,y,ids=batch;self.order.extend(ids.tolist())
                return {'loss':nn.functional.cross_entropy(self.model(x),y)}
            def validation_step(self,batch,index):
                return {'all':self.model(batch['input'])}
            def configure_optimizers(self):
                return torch.optim.AdamW(self.parameters(),lr=.01)

        class Data(pl.LightningDataModule):
            def train_dataloader(self):
                return DataLoader(TensorDataset(torch.arange(64,dtype=torch.float32).reshape(16,4)/64,
                    torch.arange(16)%2,torch.arange(16)),batch_size=4,shuffle=True)
            def val_dataloader(self):
                return DataLoader([{'input':torch.ones(4),'target':1-i%2,'index':i} for i in range(1124)],batch_size=1124)

        class Stop(pl.Callback):
            def on_train_epoch_end(self,trainer,module):
                trainer.should_stop=True

        def fit(path,stop=False,resume=False):
            args=SimpleNamespace(reproduction_run=path,reproduction_metadata={'data_dir':'data','method':'vulmixer'},
                reproduction_split='valid',reproduction_evaluate=False)
            cb=mixer_callback(args)
            args.reproduction_progress=TrainingProgress(path/'best.pt',resume=resume)
            module=Module()
            trainer=pl.Trainer(accelerator='cpu',max_epochs=2,reload_dataloaders_every_n_epochs=1,
                callbacks=[cb,*([Stop()] if stop else [])],logger=False,enable_checkpointing=False,
                enable_progress_bar=False,enable_model_summary=False)
            trainer.fit(module,Data(),ckpt_path=str(path/'framework-last.ckpt') if resume else None,weights_only=False)
            return module,trainer

        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            torch.manual_seed(77)
            complete,trainer=fit(Path(a))
            expected=torch.get_rng_state().clone()
            torch.manual_seed(77)
            first,_=fit(Path(b),stop=True)
            resumed,resumed_trainer=fit(Path(b),resume=True)
            self.assertEqual(first.order+resumed.order,complete.order)
            self.assertEqual(resumed_trainer.global_step,trainer.global_step)
            for x,y in zip(complete.parameters(),resumed.parameters()):
                torch.testing.assert_close(x,y,rtol=0,atol=0)
            a_state=trainer.optimizers[0].state_dict()['state']
            b_state=resumed_trainer.optimizers[0].state_dict()['state']
            self.assertEqual(a_state.keys(),b_state.keys())
            for key in a_state:
                for name in ('step','exp_avg','exp_avg_sq'):
                    torch.testing.assert_close(a_state[key][name],b_state[key][name],rtol=0,atol=0)
            torch.testing.assert_close(torch.get_rng_state(),expected,rtol=0,atol=0)

    def test_cached_metadata_keeps_full_cohort_and_strict_threshold(self):
        rows=official_rows('data','valid')
        args=SimpleNamespace(reproduction_metadata=dict(data_dir='data'),reproduction_cohorts={},
                             reproduction_split='valid',reproduction_evaluate=False)
        labels=[r['target'] for r in rows]
        a=native_predictions(args,[.500001,.5]*562,labels)
        self.assertEqual(a['p_c'],1.)
        self.assertEqual(a['patched_fpr'],0.)
        cached=args.reproduction_cohorts['valid']
        b=native_predictions(args,[.5,.500001]*562,labels)
        self.assertEqual(b['p_r'],1.)
        self.assertIs(args.reproduction_cohorts['valid'],cached)
        self.assertEqual(len(cached),1124)
        self.assertTrue(all(set(r)=={'sample_key','pair_key','target'} for r in cached))
        self.assertEqual([r['sample_key'] for r in args.reproduction_predictions],
                         [r['sample_key'] for r in rows])

    def resume_fixture(self, root, resume):
        meta=dict(method='linevul',checkpoint_interval=1)
        cfg=root/'config.json';cfg.write_text(json.dumps(meta))
        args=SimpleNamespace(reproduction_config=str(cfg),output_dir=str(root),reproduction_resume=resume)
        model=nn.Sequential(nn.Dropout(.2),nn.Linear(4,1))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1/(1+step))
        native_begin(args,model,optimizer,scheduler)
        return args,model,optimizer,scheduler

    def train_fixture(self, state, stop=None, workers=0, sampler_class=RandomSampler):
        args,model,optimizer,scheduler=state
        features=torch.arange(40,dtype=torch.float32).reshape(10,4)/40
        dataset=TensorDataset(features,torch.arange(10))
        loader=DataLoader(dataset,sampler=sampler_class(dataset),batch_size=2,num_workers=workers)
        order=[]
        for step,(x,ids) in native_batches(args,loader,0):
            model.train();loss=model(x).square().mean();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();optimizer.zero_grad();scheduler.step()
            native_step(args,model,optimizer,scheduler,0,step,loss)
            order.extend(ids.tolist())
            if stop is not None and step==stop:
                break
        return order

    def test_mid_epoch_callbacks_preserve_sampler_dropout_parameters_and_rng(self):
        for workers,sampler in ((0,RandomSampler),(4,RandomSampler),(4,SequentialSampler)):
            with self.subTest(workers=workers,sampler=sampler.__name__), \
                    tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
                options=dict(workers=workers,sampler_class=sampler)
                torch.manual_seed(77)
                complete=self.resume_fixture(Path(a),False)
                order=self.train_fixture(complete,**options)
                rng=torch.get_rng_state().clone()
                torch.manual_seed(77)
                interrupted=self.resume_fixture(Path(b),False)
                first=self.train_fixture(interrupted,stop=1,**options)
                restored=self.resume_fixture(Path(b),True)
                remaining=self.train_fixture(restored,**options)
                self.assertEqual(first+remaining,order)
                self.assertEqual(restored[0].reproduction_loop['global_step'],5)
                for x,y in zip(complete[1].parameters(),restored[1].parameters()):
                    torch.testing.assert_close(x,y,rtol=0,atol=0)
                torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)

    def test_zero_pc_still_saves_first_checkpoint_and_ties_keep_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=self.resume_fixture(Path(tmp),False)
            args,model,optimizer,scheduler=state
            args.reproduction_validation=dict(p_c=0.)
            args.reproduction_predictions=[]
            for epoch in (0,1):
                args.reproduction_loop.update(loss_sum=.7,loss_steps=1)
                native_epoch(args,model,optimizer,scheduler,epoch)
            self.assertEqual(json.loads((Path(tmp)/'best-p_c.json').read_text())['epoch'],1)
            self.assertTrue((Path(tmp)/'best-p_c.pt').is_file())


if __name__=='__main__':
    unittest.main()
