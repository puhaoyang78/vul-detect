"""Repair candidates, source-only line pooling, and the common staged trainer."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import Qwen2Config, Qwen2Model, RobertaConfig, RobertaModel

from vulnmechanism import model as core
from vulnmechanism.change_learning import line_invariance_loss, region_loss
from vulnmechanism.cli import build_parser
from vulnmechanism.syntax import (
    byte_character_map, code_line_spans, repair_candidate_lines,
    semantic_preserving_view, source_token_spans,
)


class CharacterTokenizer:
    is_fast = True
    pad_token_id, eos_token_id, bos_token_id = 0, 1, 2
    all_special_ids = [0, 1, 2]

    def __call__(self, source, *, add_special_tokens=False, truncation=False,
                 max_length=None, return_offsets_mapping=False, **kwargs):
        n = min(len(source), max_length) if truncation and max_length is not None else len(source)
        result = {'input_ids': [3+ord(c)%13 for c in source[:n]]}
        if return_offsets_mapping:
            result['offset_mapping'] = [(i, i+1) for i in range(n)]
        return result


def record(key, source, label, split='train'):
    return dict(sample_key=key, raw_source=source, label=label, dataset='primevul', split=split, language='c')


def examples():
    rows, views, regions = [], [], {}
    for i in range(2):
        old = f'int f{i}(int n) {{\n int x=n;\n consume(n);\n return x;\n}}'
        new = old.replace(' consume(n);', ' if(n<0)return 0;\n consume(n);')
        before, after = record(f'{i}:v', old, 1), record(f'{i}:f', new, 0)
        rows.extend((before, after))
        transformed, _ = semantic_preserving_view(old, 'c')
        views.append((before, dict(before, raw_source=transformed)))
        regions[before['sample_key']] = [3]
    rows.extend((record('v0', 'int safe(int n){\n return n+1;\n}', 0, 'valid'),
                 record('v1', 'int bad(int n){\n return 4/n;\n}', 1, 'valid')))
    return rows, regions, views


class LocalRiskTests(unittest.TestCase):
    def test_added_guard_finds_unchanged_protected_call_with_local_bindings(self):
        old = 'int f(int *p, int n) {\n prepare();\n notify();\n consume(p,n);\n return n;\n}'
        new = old.replace(' prepare();', ' if(!p)return -1;\n prepare();')
        result = repair_candidate_lines(old, new, 'c')
        self.assertEqual(result['lines'], [2, 4])
        self.assertIn('guard_first_bound_use_within_three', {x['rule'] for x in result['events']})
        self.assertNotIn(1, result['lines'])
        self.assertTrue(result['equal_token_sections'])
        # Ordinary calls do not disable the binding analysis.
        old = 'int f(int n){\n int size=n;\n prepare();\n consume(size);\n return n;\n}'
        new = old.replace(' prepare();', ' if(size<0)return -1;\n prepare();')
        self.assertIn(4, repair_candidate_lines(old, new, 'c')['lines'])

    def test_replace_delete_multiple_hunks_and_control_header_only(self):
        old = 'int f(int n){\n int x=n;\n if(n>2){\n consume(n);\n }\n cleanup(x);\n return n;\n}'
        new = old.replace('n>2', 'n>4').replace(' cleanup(x);\n', '')
        result = repair_candidate_lines(old, new, 'c')
        self.assertIn(3, result['lines'])
        self.assertIn(6, result['lines'])
        self.assertNotIn(4, result['lines'])
        self.assertNotIn(7, result['lines'])

    def test_shadowing_same_block_and_equal_declaration_requirement(self):
        old = 'int f(int n){\n prepare();\n {\n int n=2;\n consume(n);\n }\n consume(n);\n return 0;\n}'
        new = old.replace(' prepare();', ' if(n<0)return -1;\n prepare();')
        result = repair_candidate_lines(old, new, 'c')
        self.assertEqual(result['lines'], [2, 7])
        self.assertNotIn(5, result['lines'])
        old = 'int f(int n){\n int size=n;\n prepare();\n consume(size);\n return n;\n}'
        new = old.replace('int size=n;', 'int size=n+1;\n if(size<0)return -1;')
        result = repair_candidate_lines(old, new, 'c')
        self.assertNotIn('guard_first_bound_use_within_three', {x['rule'] for x in result['events']})

    def test_format_comments_and_unreliable_parse_do_not_guess_insertions(self):
        old = 'int f(int n){return n+1;}'
        result = repair_candidate_lines(old, 'int  f(int n) { /* guard? */ return n + 1; }', 'c')
        self.assertFalse(result['changed'])
        self.assertEqual(result['lines'], [])
        result = repair_candidate_lines('int f(){\n return x;\n', 'int f(){\n if(x)stop();\n return x;\n', 'c')
        self.assertEqual(result['lines'], [])
        self.assertIn('unreliable_parse_direct_edits_only', result['skip_reasons'])

    def test_unicode_coordinates_lexical_lines_and_truncation(self):
        src = 'int f(){\n /* 中文 */\n const char *s="中文";\n {\n }\n return 1;\n}'
        self.assertEqual(list(code_line_spans(src)), [1, 3, 6])
        mapping = byte_character_map(src)
        self.assertEqual(mapping[len(src.encode())], len(src))
        self.assertTrue(all(src[a:b] == value for value, a, b in source_token_spans(src)))
        old = 'int f(int n){\n' + ' consume(n);\n'*250 + ' return n;\n}'
        new = old.replace(' return n;', ' if(n<0)return -1;\n return n;')
        raw = repair_candidate_lines(old, new, 'c')['lines']
        layout = core.InputBuilder(CharacterTokenizer(), source_max_length=2048, context_max_length=8).local_layout(old, 'codebert')
        self.assertFalse(layout['complete'])
        self.assertFalse(set(raw) & set(layout['full_lines']))
        self.assertGreater(min(raw), max(layout['full_lines']))

    def test_multiline_bpe_fraction_and_comment_exclusion_and_offset_errors(self):
        source = 'ab\n/* note */\ncde'
        class CrossLine(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                if text == source:
                    return {'input_ids': [3], **({'offset_mapping': [(0,len(text))]} if kwargs.get('return_offsets_mapping') else {})}
                return super().__call__(text, **kwargs)
        builder = core.InputBuilder(CrossLine(), source_max_length=2048, context_max_length=8)
        layout = builder.local_layout(source, 'qwen')
        self.assertEqual(layout['lines'], [1, 3])
        self.assertEqual(layout['line_weights'], [[(0,.4)], [(0,.6)]])
        class BadOffsets(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                out = super().__call__(text, **kwargs)
                if kwargs.get('return_offsets_mapping'):
                    out['offset_mapping'] = [(1,1)]*len(out['input_ids'])
                return out
        builder = core.InputBuilder(BadOffsets(), source_max_length=2048, context_max_length=8)
        with self.assertRaisesRegex(ValueError, 'offsets'):
            builder.local_layout('int f(){return 1;}', 'qwen')
        with self.assertRaisesRegex(ValueError, 'm=0'):
            core.InputBuilder(CharacterTokenizer(),source_max_length=2048,context_max_length=8).local_layout('{\n// only\n}', 'qwen')

    def test_special_token_inside_literal_has_valid_offset_but_no_local_weight(self):
        import re
        class Reserved(CharacterTokenizer):
            def __call__(self,source,**kwargs):
                pieces=list(re.finditer(r'</s>|.',source,re.S))
                ids=[2 if m.group()=='</s>' else 3+ord(m.group())%13 for m in pieces]
                out={'input_ids':ids}
                if kwargs.get('return_offsets_mapping'):out['offset_mapping']=[m.span() for m in pieces]
                return out
        builder=core.InputBuilder(Reserved(),source_max_length=2048,context_max_length=8)
        layout=builder.local_layout('int f(){\n emit("</s>");\n return 0;\n}', 'codebert')
        self.assertTrue(layout['complete'])
        self.assertEqual(layout['lines'],[1,2,3])
        special=layout['ids'].index(2)
        self.assertFalse(any(index==special for weights in layout['line_weights'] for index,_ in weights))

    def test_windows_deduplicate_token_states_before_lines_and_source_only_ids(self):
        hidden = torch.tensor([[[99.],[2.],[4.]], [[99.],[8.],[10.]]], requires_grad=True)
        positions = torch.tensor([[-1,0,1],[-1,1,2]])
        layout = dict(ids=[3,4,5], lines=[1,2,3], line_weights=[[(0,1.)],[(1,1.)],[(2,1.)]])
        lines = core._pool_source_lines(hidden, torch.tensor([0,0]), positions, [layout])[0]
        torch.testing.assert_close(lines, torch.tensor([[2.],[6.],[10.]]))
        lines.sum().backward()
        self.assertEqual(float(hidden.grad[0,2]), .5)
        self.assertEqual(float(hidden.grad[1,1]), .5)
        source = 'int f(){\n' + 'x++;\n'*800 + 'return x;\n}'
        builder = core.InputBuilder(CharacterTokenizer(),source_max_length=2048,context_max_length=8)
        basic = builder.codebert_batch([{'raw_source':source}], device='cpu')
        local = builder.local_batch([{'raw_source':source, 'label':1, 'diff':'secret', 'candidate_lines':[1]}], encoder='codebert', device='cpu')
        for a,b in zip(basic[:4],local[:4]):
            self.assertTrue(torch.equal(a,b))
        self.assertEqual(int(local[5].max())+1, len(local[6][0]['ids']))
        self.assertEqual(int(local[5][0,0]), -1)

    def test_empty_full_regions_gradients_and_line_alignment(self):
        scores = torch.tensor([1.,2.,3.],requires_grad=True)
        self.assertIsNone(region_loss(scores, [1,3,4], []))
        self.assertIsNone(region_loss(scores, [1,3,4], [1,3,4]))
        loss = region_loss(scores, [1,3,4], [3]);loss.backward()
        self.assertLess(float(scores.grad[1]),0)
        self.assertGreater(float(scores.grad[0]),0)
        z = torch.tensor([1.,-2.,0.],requires_grad=True)
        js = line_invariance_loss([([1,3,4],scores),([1,3,4],z)])
        js.backward(); self.assertGreater(float(z.grad.abs().sum()),0)
        with self.assertRaises(ValueError):line_invariance_loss([([1],z[:1]),([2],z[:1])])
        head=core.LocalRiskHead(4)
        out,local=head(torch.randn(3,4),torch.randn(4))
        torch.testing.assert_close(out,torch.logsumexp(local,0)-torch.tensor(3.).log())
        self.assertEqual(out.dtype,torch.float32)
        self.assertEqual(head.local.out_features,64)

    def test_qwen_global_context_is_valid_eos_not_padding(self):
        class Encoder(nn.Module):
            def __init__(self):super().__init__();self.scale=nn.Parameter(torch.ones(4))
            def forward(self,input_ids,**kwargs):
                from types import SimpleNamespace
                h=(input_ids+torch.arange(input_ids.shape[1])).float()
                return SimpleNamespace(last_hidden_state=h[:,:,None]*self.scale)
        with patch.object(core,'_build_lora_encoder',return_value=(Encoder(),4)):
            model=core.SequenceVulnerabilityClassifier('',device=torch.device('cpu'),lora_r=16,lora_alpha=32,lora_dropout=.05,
                target_modules=('q_proj',),gradient_checkpointing=False,classifier_head='local_risk')
        captured=[]
        class Capture(nn.Module):
            def forward(self,lines,global_context):
                captured.append(global_context)
                scores=lines[:,0]*0+global_context[0]
                return scores.mean(),scores
        model.task_modules['classifier']=Capture()
        builder=core.InputBuilder(CharacterTokenizer(),source_max_length=2048,context_max_length=8)
        batch=builder.local_batch([{'raw_source':'int f(){return 1;}'},{'raw_source':'int f(){ /* longer */ return 1;}'}],encoder='qwen',device='cpu')
        model(*batch)
        ends=batch[1].sum(1)-1
        for i,g in enumerate(captured):self.assertEqual(float(g[0].detach()),1+int(ends[i]))

    def test_common_real_tiny_encoders_train_save_reload_and_strict_loading(self):
        torch.set_num_threads(2)
        rows,regions,views=examples()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for variant in ('codebert_source','baseline'):
                path=root/variant
                if variant=='codebert_source':
                    base=RobertaModel(RobertaConfig(vocab_size=32,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
                        intermediate_size=32,max_position_embeddings=514,pad_token_id=0,bos_token_id=2,eos_token_id=1))
                else:
                    base=Qwen2Model(Qwen2Config(vocab_size=32,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
                        num_key_value_heads=2,intermediate_size=32,max_position_embeddings=4096,pad_token_id=0,eos_token_id=1))
                base.save_pretrained(path)
                common=dict(records=rows,variant=variant,model_path=str(path),source_max_length=2048,epochs=3,
                    batch_size=2,gradient_accumulation=4,lora_r=16,lora_alpha=32,lora_dropout=.05,device='cpu')
                with patch.object(core.AutoTokenizer,'from_pretrained',return_value=CharacterTokenizer()):
                    initial=root/variant/'init.pt'
                    core.train_model(None,initial,**common)
                    budgets=[]
                    for objective in ('A','B','C','D'):
                        target=root/variant/objective/'best.pt'
                        cp=core.train_model(None,target,**common,initial_checkpoint=initial,
                            classifier_head='linear' if objective=='A' else 'local_risk',local_objective=objective,
                            local_regions=regions,invariant_views=views)
                        events=[json.loads(s) for s in target.with_suffix('.training.jsonl').read_text().splitlines()]
                        epochs=[r for r in events if r['event']=='epoch']
                        self.assertEqual([r['phase'] for r in epochs],['head_bce','joint','joint'])
                        self.assertEqual(epochs[0]['auxiliary_function_forwards'],0)
                        budgets.append([r['optimizer_steps'] for r in epochs])
                        logits=core.predict_checkpoint(target,rows[-2:],batch_size=2,device='cpu')
                        self.assertTrue(torch.isfinite(logits).all())
                        if objective=='D':
                            self.assertGreater(epochs[-1]['region_loss'],0)
                            self.assertGreater(epochs[-1]['invariance_loss'],0)
                            bad=dict(cp,task_state=dict(cp['task_state']))
                            bad['task_state'].pop(next(iter(bad['task_state'])))
                            with self.assertRaises(RuntimeError):core._load_model(bad,device=torch.device('cpu'))
                            bad=dict(cp,adapter_state=dict(cp['adapter_state']))
                            bad['adapter_state'].pop(next(iter(bad['adapter_state'])))
                            with self.assertRaises(ValueError):core._load_model(bad,device=torch.device('cpu'))
                    self.assertTrue(all(b==budgets[0] for b in budgets))

    def test_cli_defaults_and_opt_in_random_region(self):
        args=build_parser().parse_args(['local-risk','--stage','train'])
        self.assertEqual((args.source_max_length,args.epochs,args.seed),(2048,3,42))
        self.assertIsNone(args.objective)
        self.assertEqual(build_parser().parse_args(['local-risk','--stage','train','--objective','C_random']).objective,'C_random')

    def test_multiple_training_repairs_union_and_auxiliary_dropout_mode(self):
        from types import SimpleNamespace
        from vulnmechanism.change_learning import _local_data
        old='int f(int n){\n consume(n);\n keep();\n second(n);\n return n;\n}'
        rows=[record('old',old,1), record('fix1',old.replace(' consume(n);',' if(n<0)return -1;\n consume(n);'),0),
              record('fix2',old.replace('second(n)','second(n-1)'),0),
              record('v0','int safe(){return 0;}',0,'valid'),record('v1','int bad(){return 1;}',1,'valid')]
        transformed,kind=semantic_preserving_view(old,'c')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            artifacts={'records':rows,'pair_members':[],
                'pairs':[dict(dataset='primevul',split='train',before_key='old',after_key=key) for key in ('fix1','fix2')],
                'transforms':[dict(sample_key='old',dataset='primevul',split='train',raw_source=transformed,transformation=kind)]}
            for name,values in artifacts.items():
                (root/(name+'.jsonl')).write_text(''.join(json.dumps(value)+'\n' for value in values))
            args=SimpleNamespace(data_dir=str(root),source_dataset='primevul',encoder='codebert',smoke_samples=0)
            with patch.object(core.AutoTokenizer,'from_pretrained',return_value=CharacterTokenizer()):
                selected,regions,views,cache,coverage=_local_data(args,'unused')
            self.assertEqual(regions['old'],[2,4])
            self.assertEqual(len(views),1)
            self.assertEqual(len(selected),len(rows))
        container=nn.Sequential(nn.Dropout(.9),nn.Linear(2,2))
        container.gradient_checkpointing=True
        container.train()
        with core._auxiliary_mode(container):
            self.assertTrue(container.training)
            self.assertFalse(container[0].training)
            output=container(torch.ones(1,2));output.sum().backward()
        self.assertTrue(container[0].training)
        self.assertGreater(float(container[1].weight.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
