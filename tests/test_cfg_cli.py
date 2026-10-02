import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from vulnmechanism.cfg_experiment import parser, short_command
from vulnmechanism.progress import TrainingProgress, training_bar, show_training_event


class ShortCommandTests(unittest.TestCase):
    def test_joint_control_commands_keep_shared_settings_and_stage_roles(self):
        config=dict(dataset='d',graphs='g',source_max_length=2048,epochs=3,seed=42,
                    supervision_dir='saved_dependencies')
        with patch.object(Path,'exists',return_value=False), patch.object(Path,'read_text',return_value=json.dumps(config)):
            joint=parser().parse_args(short_command(parser().parse_args(['joint','train'])))
            control=parser().parse_args(short_command(parser().parse_args(['control','train'])))
            stage=parser().parse_args(short_command(parser().parse_args(['control','pretrain'])))
        self.assertEqual(joint.variants,['joint_nodes','joint_edges','joint','joint_shuffled'])
        self.assertEqual(control.variants,['control_cfg','control_joint'])
        self.assertEqual(control.pretrain_dir,'results/cfg_control_pretrain_seed42')
        self.assertEqual(stage.modes,['control_pretrain'])
        self.assertEqual(stage.supervision_dir,'saved_dependencies')
        self.assertEqual(stage.reference_pretrain_dir,joint.pretrain_dir)
        self.assertEqual((joint.seed,joint.epochs,joint.source_max_length),(42,3,2048))

    def test_short_training_preserves_saved_hyperparameters(self):
        config = dict(dataset='data/input.jsonl', graphs='data/graphs.jsonl',
                      source_max_length=2048, epochs=3, seed=42, batch_size=1,
                      gradient_accumulation=8, learning_rate=0.0002,
                      graph_learning_rate=0.001, lora_dropout=0.05)
        with patch.object(Path, 'exists', return_value=False), patch.object(Path, 'read_text', return_value=json.dumps(config)):
            args = parser().parse_args(short_command(parser().parse_args(['behavior', 'train'])))
        for name, value in config.items():
            self.assertEqual(getattr(args, name), value)
        self.assertEqual(args.variants, ['behavior_nodes','behavior_edges','behavior_joint','behavior_masked'])
        self.assertEqual(args.pretrain_dir, 'results/cfg_dep_pretrain_windowfix_seed42')
        self.assertTrue(args.resume)

    def test_existing_run_cache_is_reused_and_evaluation_does_not_train(self):
        saved = {'behavior_dir': '/chosen/cache', 'pretrain_run_dir':'/chosen/p0'}
        def read(path, *args, **kwargs):
            return json.dumps(saved if 'cfg_behavior' in str(path) else {'dataset':'d','graphs':'g'})
        with patch.object(Path, 'exists', return_value=True), patch.object(Path, 'read_text', read):
            for action, command in [('valid','compare'),('test','eval')]:
                args=parser().parse_args(short_command(parser().parse_args(['behavior',action])))
                self.assertEqual(args.command,command)
                self.assertEqual(args.split,'valid' if action=='valid' else 'test')
            args=parser().parse_args(short_command(parser().parse_args(['behavior','train','joint'])))
            self.assertEqual(args.behavior_dir,'/chosen/cache')
            self.assertEqual(args.pretrain_dir,'/chosen/p0')
            self.assertEqual(args.variants,['behavior_joint'])

    def test_ablation_paths_and_families_match_for_train_and_eval(self):
        with patch.object(Path,'exists',return_value=False), patch.object(Path,'read_text',return_value='{"dataset":"d","graphs":"g"}'):
            train=parser().parse_args(short_command(parser().parse_args(['behavior','ablate','guard'])))
            test=parser().parse_args(short_command(parser().parse_args(['behavior','test','guard'])))
            self.assertEqual(train.output_dir,test.run_dir)
            self.assertEqual(train.disable_behavior_family,['guard'])
            self.assertEqual(test.variants,['behavior_joint'])
            with self.assertRaises(ValueError):
                short_command(parser().parse_args(['behavior','ablate','nonsense']))

    def test_history_unchanged_and_no_step_spam(self):
        row={'event':'step','epoch':1,'step':1,'loss':0.2,'learning_rate':0.001,'samples_seen':8}
        epoch={'event':'epoch','epoch':1,'train_loss':0.2,'optimizer_steps':1,
               'validation_threshold':0.5,'validation':{'mcc':0.6,'f1':0.7,'auc':0.8}}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as output:
            history=TrainingProgress(Path(tmp)/'best.pt')
            history.add(row)
            self.assertEqual(output.getvalue(),'')
            history.add(epoch)
            self.assertEqual([json.loads(line) for line in history.path.read_text().splitlines()],[row,epoch])
            self.assertIn('0.6000',output.getvalue())
            self.assertNotIn('"event"',output.getvalue())

    def test_terminal_and_redirected_progress_same_iteration_and_rng(self):
        class Terminal(io.StringIO):
            def isatty(self): return True
        random.seed(42); before=random.getstate()
        for target, enabled in [(io.StringIO(),False),(Terminal(),True)]:
            with redirect_stdout(target):
                with training_bar(range(3),desc='Train',mininterval=0) as bar:
                    self.assertEqual(not bar.disable,enabled)
                    self.assertEqual(list(bar),[0,1,2])
            if not enabled:self.assertEqual(target.getvalue(),'')
        self.assertEqual(random.getstate(),before)

    def test_rendering_does_not_change_trained_weights_or_selection(self):
        import torch
        from contextlib import ExitStack
        from tests.test_training_progress import TinyTokenizer, TinyClassifier
        from vulnmechanism.model import train_model
        class Terminal(io.StringIO):
            def isatty(self): return True
        records=[dict(sample_key=str(i),dataset='primevul',split='train' if i<4 else 'valid',
                      label=i%2,raw_source=f'int x{i};') for i in range(6)]
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch('vulnmechanism.model.AutoTokenizer.from_pretrained',return_value=TinyTokenizer()))
            stack.enter_context(patch('vulnmechanism.model._build_model',side_effect=lambda *a,**kw:TinyClassifier()))
            stack.enter_context(patch('vulnmechanism.model.get_peft_model_state_dict',side_effect=lambda model:model.state_dict()))
            results=[]; histories=[]
            for i,output in enumerate((io.StringIO(),Terminal())):
                path=Path(tmp)/f'{i}.pt'
                with redirect_stdout(output):
                    results.append(train_model(None,path,records=records,variant='baseline',model_path='tiny',
                        device='cpu',epochs=1,batch_size=1,gradient_accumulation=2,seed=42,log_every=1))
                histories.append(path.with_suffix('.training.jsonl').read_text())
            for field in ('adapter_state','task_state'):
                for name,value in results[0][field].items():
                    self.assertTrue(torch.equal(value,results[1][field][name]),name)
            for field in ('selected_epoch','decision_threshold','validation'):
                self.assertEqual(results[0][field],results[1][field])
            self.assertEqual(histories[0],histories[1])
