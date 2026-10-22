"""Line alignment, overlap averaging, shared-risk ranking and matched training."""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from tests.test_codebert_source import CodeTokenizer,tiny_base
from vulnmechanism import model as core
from vulnmechanism.data import write_jsonl
from vulnmechanism.evaluation import evaluate,paired_report
from vulnmechanism.pair_representation import (
    METHODS,attach_line_metadata,matched_random_lines,patch_regions,region_pair_ranking,
)
from vulnmechanism.training import TRAINING,train


class BatchTokenizer(CodeTokenizer):
    def __call__(self,text,**kwargs):
        if isinstance(text,list):
            encoded=[super(BatchTokenizer,self).__call__(t,**kwargs) for t in text]
            return {key:[r[key] for r in encoded] for key in encoded[0]}
        return super().__call__(text,**kwargs)


class MILTests(unittest.TestCase):
    def test_single_function_cli_preserves_crlf_and_unicode(self):
        from vulnmechanism.cli import main
        source='int café(){\r\n  return 1;\r\n}\r\n'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'function.c';path.write_bytes(source.encode('utf-8'))
            args=['vulnmechanism','infer','--checkpoint',str(Path(tmp)/'dummy.pt'),
                  '--source-file',str(path),'--device','cpu']
            seen=[]
            def predict(_checkpoint,records,**_kwargs):
                seen.extend(records);return torch.tensor([.2])
            with patch.object(sys,'argv',args),patch('torch.load',return_value=dict(decision_threshold=.5)),\
                 patch.object(core,'predict_checkpoint',side_effect=predict),redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(seen,[dict(raw_source=source)])

    def test_untrimmed_bytelevel_offsets_keep_indentation_without_changing_ids(self):
        from tokenizers import ByteLevelBPETokenizer
        from transformers import RobertaTokenizerFast
        source='int f(){\n  if(x){\n    use(x);\n  }\n}'
        with tempfile.TemporaryDirectory() as tmp:
            bytelevel=ByteLevelBPETokenizer()
            bytelevel.train_from_iterator([source],vocab_size=270,
                special_tokens=['<s>','<pad>','</s>','<unk>','<mask>'])
            bytelevel.save_model(tmp)
            arguments=dict(vocab_file=str(Path(tmp)/'vocab.json'),merges_file=str(Path(tmp)/'merges.txt'))
            trimmed=RobertaTokenizerFast(**arguments,trim_offsets=True)
            untrimmed=RobertaTokenizerFast(**arguments,trim_offsets=False)
            a=trimmed(source,add_special_tokens=False,return_offsets_mapping=True)
            b=untrimmed(source,add_special_tokens=False,return_offsets_mapping=True)
            self.assertEqual(a['input_ids'],b['input_ids'])
            self.assertTrue(any(i==j for i,j in a['offset_mapping']))
            self.assertFalse(any(i==j for i,j in b['offset_mapping']))
            row=dict(raw_source=source)
            attach_line_metadata(row,b['input_ids'],b['offset_mapping'],set(untrimmed.all_special_ids))
            self.assertEqual(row['_line_numbers'],list(range(5)))
            self.assertEqual(row['_alignment']['mapped_tokens'],len(b['input_ids']))

    def test_insertion_deletion_anchors_unicode_crlf_and_source_only_alignment(self):
        a="int café(){\r\n  use(x);\r\n}\r\n"
        b="int café(){\r\n  if (!x) return;\r\n  use(x);\r\n}\r\n"
        regions,hunks=patch_regions(a,b,context=0)
        self.assertEqual(hunks[0]['kind'],'insert')
        self.assertIn('use(x)',a[slice(*regions[0][0])])
        self.assertIn('if (!x)',b[slice(*regions[1][0])])
        reverse,_=patch_regions(b,a,context=0)
        self.assertEqual(reverse,regions[::-1])
        self.assertEqual(patch_regions(a,a),([[],[]],[]))
        source="<s>int a;\n\n  use(a);\n"
        # A source special token, end-of-line token, zero offset and code token.
        row=dict(raw_source=source)
        ids=[2,10,11,12,13,14]
        offsets=[(0,3),(3,6),(9,11),(11,11),(13,16),(0,0)]
        attach_line_metadata(row,ids,offsets,{2})
        self.assertEqual(row['_line_numbers'],[0,2])
        self.assertEqual(row['_token_lines'],(-1,0,0,-1,1,-1))
        self.assertEqual(row['_alignment']['mapped_tokens'],3)

    def test_random_union_preserves_physical_line_size_span_and_shape(self):
        numbers=list(range(20))
        true=[2,3,8,9]
        result,forced=matched_random_lines(numbers,true,seed='42:function')
        self.assertFalse(forced)
        self.assertNotEqual(result,true)
        self.assertEqual(len(result),len(true))
        self.assertEqual([x-min(result) for x in result],[x-min(true) for x in true])
        self.assertEqual(matched_random_lines(numbers,true,seed='42:function')[0],result)
        full,forced=matched_random_lines([0,2,4],[0,1,2],seed='42')
        self.assertTrue(forced);self.assertEqual(full,[0,1,2])
        self.assertEqual(matched_random_lines(numbers,[],seed='42'),([],True))

    def test_topk_ranking_gradient_direction_exclusion_and_empty_region(self):
        z=torch.tensor([.1,.2,.4,.6,.0,.3],dtype=torch.float64,requires_grad=True)
        owners=torch.tensor([0,0,0,1,1,1])
        rows=[dict(pair_key='p',label=1,patch_eligible=True,candidate_lines=[0,2]),
              dict(pair_key='p',label=0,patch_eligible=True,candidate_lines=[1,2])]
        actual=region_pair_ranking(z,owners,rows)
        expected=torch.nn.functional.softplus((z[4]+z[5])/2-(z[0]+z[2])/2)
        torch.testing.assert_close(actual[0],expected)
        self.assertTrue(torch.autograd.gradcheck(lambda x:region_pair_ranking(x,owners,rows),(z,)))
        actual.mean().backward()
        self.assertTrue((z.grad[:3]<=0).all() and (z.grad[3:]>=0).all())
        self.assertEqual(z.grad[1].item(),0);self.assertEqual(z.grad[3].item(),0)
        rows[0]['patch_eligible']=False
        self.assertEqual(region_pair_ranking(z,owners,rows).numel(),0)
        rows[0]['patch_eligible']=True;rows[0]['candidate_lines']=[]
        with self.assertRaises(ValueError):region_pair_ranking(z,owners,rows)

    def test_overlap_states_average_before_lines_and_gated_sum_identity(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            tiny_base(Path(tmp)/'base')
            model=core.CodeBERT(Path(tmp)/'base','cpu',readout='line_mil')
            class EncodedStates(nn.Module):
                def __init__(self):
                    super().__init__();self.scale=nn.Parameter(torch.ones(16));self.config=SimpleNamespace(hidden_size=16)
                def forward(self,input_ids,**kwargs):
                    return SimpleNamespace(last_hidden_state=input_ids.float()[...,None]*self.scale)
            model.encoder=EncodedStates()
            with torch.no_grad():
                model.task_modules['classifier'].weight.zero_();model.task_modules['classifier'].weight[0,0]=1
                model.task_modules['classifier'].bias.zero_()
            # Source token 0 has contextual states 10 and 20; token 1 has state 0.
            # Its line mean must be ((10+20)/2+0)/2=7.5, rather than mean(window means).
            ids=torch.tensor([[99,10,0],[99,20,0]])
            mask=torch.tensor([[1,1,1],[1,1,0]])
            owners=torch.tensor([0,0]);weights=torch.tensor([[.5,.5,1],[.5,.5,0]])
            mapping=torch.tensor([[-1,0,0],[-1,0,-1]])
            positions=torch.tensor([[-1,0,1],[-1,0,-1]])
            output=model(ids,mask,owners,weights,1,positions,line_ids=mapping,line_owners=torch.tensor([0]),return_line_scores=True,window_batch_size=1)
            self.assertAlmostEqual(output[0].item(),7.5)
            torch.testing.assert_close(output[0],output[1]*output[2])
            self.assertAlmostEqual(output[2].sum().item(),1.)
            output[0].sum().backward()
            self.assertTrue(torch.isfinite(model.encoder.scale.grad).all())
            self.assertGreater(model.encoder.scale.grad.abs().sum(),0)

    def test_effective_pair_normalization_is_invariant_to_gradient_accumulation(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            base=tiny_base(root/'base')
            base.config.hidden_dropout_prob=0.;base.config.attention_probs_dropout_prob=0.
            # Reload constructs dropout modules from these zero probabilities.
            base.save_pretrained(root/'base')
            raw=root/'raw/PrimeVul_v0.1'
            def row(i,code,y):return dict(idx=i,func=code,target=y,project='p',commit_id='c',cve='x')
            training=[row(1,'int f(){use(x);}',1),row(2,'int f(){if(x) use(x);}',0),
                      row(3,'int identical(){return 7;}',1),row(4,'int identical(){return 7;}',0)]
            valid=[row(5,'int valid_only(){return 8;}',1),row(6,'int valid_only(){return 9;}',0)]
            for name,rows in (('train_paired',training),('valid_paired',valid),('valid',valid)):
                write_jsonl(raw/f'primevul_{name}.jsonl',rows)
            gradients=[]
            clip=torch.nn.utils.clip_grad_norm_
            for batch,accum in ((2,2),(4,1)):
                saved=[]
                def capture(parameters,*args,**kwargs):
                    parameters=list(parameters)
                    saved.append([p.grad.detach().clone() if p.grad is not None else None for p in parameters])
                    return clip(parameters,*args,**kwargs)
                with patch.object(core.AutoTokenizer,'from_pretrained',return_value=BatchTokenizer()),redirect_stdout(io.StringIO()),\
                     patch('vulnmechanism.training.torch.nn.utils.clip_grad_norm_',side_effect=capture):
                    train(root/'data',root/f'runs{batch}',str(root/'base'),'primevul_paired','mil_patch_rank',device='cpu',
                          config=dict(TRAINING,epochs=1,batch_size=batch,gradient_accumulation=accum),
                          full_source=True,pair_raw_root=root/'raw',train_pair_limit=2)
                self.assertEqual(len(saved),1);gradients.append(saved[0])
            for a,b in zip(*gradients):
                if a is None or b is None:self.assertIs(a,b)
                else:torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

    def test_four_methods_pr_auc_selection_no_test_single_function_reload(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tiny_base(root/'base');raw=root/'raw/PrimeVul_v0.1'
            def native(idx,code,label,commit='c'):
                return dict(idx=idx,func=code,target=label,project='p',commit_id=commit,cve='x')
            train_rows=[native(1,'int f(){\nuse(x);\n}',1),native(2,'int f(){\nif (!x) return;\nuse(x);\n}',0),
                        native(3,'int q(){return 4;}',1),native(4,'int q(){return 4;}',0),
                        native(5,'int z(){return 9;}',1),native(6,'int z(){return 1;}',0,'other')]
            valid=[native(7,'int valid_only(){return 7;}',1),native(8,'int valid_only(){return 8;}',0)]
            write_jsonl(raw/'primevul_train_paired.jsonl',train_rows)
            write_jsonl(raw/'primevul_valid_paired.jsonl',valid)
            write_jsonl(raw/'primevul_valid.jsonl',valid+[native(9,'int safe_only(){return 2;}',0)])
            (raw/'primevul_test.jsonl').write_text('FORBIDDEN_HELDOUT')
            (raw/'primevul_test_paired.jsonl').write_text('FORBIDDEN_HELDOUT')
            initial=[];constructor=core.CodeBERT
            def capture(*args,**kwargs):
                model=constructor(*args,**kwargs)
                initial.append(model.task_modules['classifier'].weight.detach().clone())
                return model
            for method in METHODS:
                with self.subTest(method=method),patch.object(core.AutoTokenizer,'from_pretrained',return_value=BatchTokenizer()),redirect_stdout(io.StringIO()),\
                     patch('vulnmechanism.training.CodeBERT',side_effect=capture),\
                     patch('vulnmechanism.training.select_threshold',side_effect=AssertionError('fixed threshold')):
                    result=train(root/'data',root/'runs',str(root/'base'),'primevul_paired',method,device='cpu',
                                 config=dict(TRAINING,epochs=2,batch_size=2),full_source=True,pair_raw_root=root/'raw',train_pair_limit=3)
                run=root/'runs/primevul_paired'/method
                cp=torch.load(run/'best.pt',weights_only=False,map_location='cpu')
                self.assertEqual(result['optimizer_steps'],4)
                self.assertEqual(cp['decision_threshold'],.5)
                self.assertEqual(cp['paired']['eligible_train_pairs'],1)
                self.assertEqual(cp['paired']['selection_keys'],['auprc','mcc'])
                coverage=json.loads((run/'input.coverage.json').read_text())
                self.assertEqual(coverage['exclusion_reasons']['identical_encoder_input'],1)
                self.assertEqual(coverage['exclusion_reasons']['metadata_anomaly'],1)
                epochs=[r for r in map(json.loads,(run/'best.training.jsonl').read_text().splitlines()) if r['event']=='epoch']
                selected=max(epochs,key=lambda r:(r['validation']['auprc'],r['validation']['mcc']))
                self.assertEqual(selected['epoch'],cp['selected_epoch'])
                self.assertFalse((run/'test.metrics.json').exists())
                with patch.object(core.AutoTokenizer,'from_pretrained',return_value=BatchTokenizer()):
                    z=core.predict_checkpoint(run/'best.pt',[dict(raw_source=valid[0]['func'])],device='cpu')
                    self.assertTrue(torch.isfinite(z).all())
                    with patch('vulnmechanism.model.load_model',side_effect=AssertionError('cache should suffice')):
                        evaluate(root/'data',root/'runs','primevul_paired',(method,),device='cpu',robust=False,splits=('valid',))
            for weight in initial[1:]:torch.testing.assert_close(weight,initial[0],atol=0,rtol=0)
            report=paired_report(root/'runs',root/'data');self.assertEqual(len(report['metrics']),4)
            saved=(root/'runs/results.json').read_bytes();paired_report(root/'runs',root/'data')
            self.assertEqual((root/'runs/results.json').read_bytes(),saved)

if __name__=='__main__':unittest.main()
