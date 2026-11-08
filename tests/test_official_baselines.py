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
        self.assertEqual([UPSTREAM[m]['epochs'] for m in UPSTREAM],[10,20,100])
        self.assertEqual([UPSTREAM[m]['block_size'] for m in UPSTREAM],[512,400,400])
        self.assertEqual([UPSTREAM[m]['probability_column'] for m in UPSTREAM],[1,1,0])
        self.assertEqual(UPSTREAM['svuld']['sampler'],'SequentialSampler')
        self.assertTrue(UPSTREAM['svuld']['r_drop'])
        self.assertEqual(UPSTREAM['regvd']['model'],'graphcodebert-base')
        for method in UPSTREAM:
            path=Path('external')/UPSTREAM[method]['directory']
            import subprocess
            self.assertEqual(subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip(),UPSTREAM[method]['commit'])

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
