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
