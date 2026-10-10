import unittest
from vulnmechanism.egcl import generate, outcomes, render

class EGCLTests(unittest.TestCase):
    def test_semantics_reassignment_and_cross_conditions(self):
        for family in ('Bounds','Pointer','Lifetime'):
            s=dict(family=family,capacity=4,offset=0,access='read',s=0,r=0,t=0,u=0,composition=False,predicate='bit',control='if',initial=0,first=1,second=0,reverse=False)
            self.assertTrue(any(x['violation'] for x in outcomes(s)))
            self.assertFalse(any(x['violation'] for x in outcomes(dict(s,t=1))))
            self.assertFalse(any(x['violation'] for x in outcomes(dict(s,composition=True,u=0))))
            self.assertTrue(any(x['violation'] for x in outcomes(dict(s,composition=True,u=1))))

    def test_structural_splits_labels_pairs_and_controls(self):
        rows,pairs,controls=generate()
        structures={s:{tuple(r['structure_family']) for r in rows if r['split']==s} for s in ('train','valid')}
        self.assertFalse(structures['train']&structures['valid'])
        self.assertEqual(len({r['source'] for r in rows}),len(rows))
        for pair in pairs:
            a,b=rows[pair['safe']],rows[pair['unsafe']]
            self.assertEqual((a['label'],b['label']),(0,1))
            self.assertEqual(a['split'],b['split'])
            self.assertTrue(b['witness'])
        for c in controls:
            a,b=rows[c['original']],rows[c['changed']]
            self.assertEqual(a['label'],b['label'])
            self.assertEqual(a['split'],b['split'])
        for row in rows:
            self.assertEqual(row['label'],int(any(x['violation'] for x in row['outcomes'])))
            self.assertTrue(any(x['target_executed'] for x in row['outcomes']))

    def test_full_legal_combinations_and_training_members(self):
        from collections import Counter
        from vulnmechanism.egcl import training_groups
        rows,pairs,controls=generate()
        kinds=Counter(c['kind'] for c in controls)
        self.assertGreater(kinds['safe_safe'],0)
        self.assertGreater(kinds['unsafe_unsafe'],0)
        self.assertGreater(len(pairs),0)
        groups=training_groups(rows,pairs,controls)
        self.assertEqual(Counter(i for group in groups for i in group),
                         Counter(r['id'] for r in rows if r['split']=='train'))
        for family in ('Bounds','Pointer','Lifetime'):
            cases=[r for r in rows if r['family']==family and r['variant']=='original']
            for equal in (False,True):
                self.assertEqual({r['label'] for r in cases if (r['spec']['s']==r['spec']['t'])==equal},{0,1})
        # Changing assignment order changes the target state for this witness.
        spec=dict(family='Pointer',capacity=4,offset=0,access='read',s=0,r=0,t=0,u=0,
                  composition=False,predicate='bit',control='if',initial=0,first=1,second=0,reverse=False)
        self.assertFalse(outcomes(spec)[0]['violation'])
        self.assertTrue(outcomes(dict(spec,reverse=True))[0]['violation'])

    def test_admission_precedes_any_model_or_device_access(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from vulnmechanism.egcl import frozen_experiment,transfer_experiment
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'structural_shortcut.json').write_text(json.dumps(dict(admission=False,failures=['shortcut'])))
            with patch('vulnmechanism.cfg_experiment._base_module',side_effect=AssertionError('encoder accessed')):
                with self.assertRaisesRegex(ValueError,'admission failed'):
                    frozen_experiment(root,root/'run')
                with self.assertRaisesRegex(ValueError,'admission failed'):
                    transfer_experiment(root,root/'run','bce_rank')
            self.assertFalse((root/'run').exists())

    def test_bce_is_never_replaced_and_same_labels_are_not_ranked(self):
        import torch
        from vulnmechanism.model import synthetic_supervision_loss
        z=torch.tensor([.1,.2],requires_grad=True)
        bce=synthetic_supervision_loss(z,'bce')
        rank=synthetic_supervision_loss(z,'bce_rank')
        self.assertGreater(float(rank),float(bce))
        for labels in ([0,0],[1,1]):
            self.assertEqual(float(synthetic_supervision_loss(z,'bce',labels)),
                             float(synthetic_supervision_loss(z,'bce_rank',labels)))
        self.assertNotEqual(float(synthetic_supervision_loss(z+3,'bce_rank')),float(rank))
        with self.assertRaisesRegex(ValueError,'unknown synthetic objective'):
            synthetic_supervision_loss(z,'egcl')

    def test_same_token_multiset_can_require_different_safety_labels(self):
        import re
        from collections import Counter
        for family in ('Bounds','Pointer','Lifetime'):
            spec=dict(family=family,capacity=4,offset=0,access='read',s=0,r=0,t=0,u=0,
                      composition=True,predicate='xor',control='switch',initial=0,first=1,second=0,reverse=False)
            changed=dict(spec,reverse=True)
            self.assertEqual(Counter(re.findall(r'\w+|[^\s]',render(spec))),
                             Counter(re.findall(r'\w+|[^\s]',render(changed))))
            self.assertFalse(any(o['violation'] for o in outcomes(spec)))
            self.assertEqual([o['input'] for o in outcomes(changed) if o['violation']],[0])

    def test_frozen_head_roundtrip_includes_same_label_members(self):
        import tempfile,json
        from pathlib import Path
        import torch
        from vulnmechanism.egcl import fit_heads
        rows=[];pairs=[];controls=[]
        for split in ('train','valid'):
            for family in ('Bounds','Pointer','Lifetime'):
                for labels in ((0,0),(0,1),(1,1)):
                    i=len(rows)
                    rows.extend(dict(id=i+j,label=y,split=split,family=family,scenario=split)
                                for j,y in enumerate(labels))
                    if labels==(0,1):pairs.append(dict(safe=i,unsafe=i+1,split=split))
                    else:
                        controls.append(dict(original=i,changed=i+1,split=split,
                                             kind='safe_safe' if labels==(0,0) else 'unsafe_unsafe'))
                        controls.append(dict(original=i,changed=i+1,split=split,
                                             kind='rename' if labels==(0,0) else 'neutral'))
        # Sentinel features test only the plumbing, not language-model ability.
        features=torch.tensor([[r['label']*2.-1.,(r['id']%3)*.1] for r in rows])
        initial=dict(weight=torch.zeros(1,2),bias=torch.zeros(1))
        with tempfile.TemporaryDirectory() as d:
            reports=fit_heads(features,rows,pairs,controls,initial,d)
            self.assertEqual(set(reports),{'bce','bce_rank'})
            for mode in reports:
                cp=torch.load(Path(d)/(mode+'.pt'),weights_only=False)
                head=torch.nn.Linear(2,1);head.load_state_dict(cp['head_state'])
                with torch.no_grad():predicted=head((features-cp['mean'])/cp['std']).squeeze(-1)
                saved=[json.loads(l)['logit'] for l in (Path(d)/(mode+'.predictions.jsonl')).read_text().splitlines()]
                torch.testing.assert_close(predicted,torch.tensor(saved))
                self.assertEqual(reports[mode]['train']['samples'],18)
                self.assertTrue(all(v['pairs']>0 for v in reports[mode]['stability'].values()))

    def test_property_specific_and_family_audits_cannot_be_averaged_away(self):
        from vulnmechanism.egcl import structural_shortcut_audit
        # Fixed negative fixture, independent of the current generator: most
        # Bounds cases have equal lexical counts but opposite order-sensitive
        # labels; Pointer/Lifetime carry a perfect assignment-count shortcut.
        rows=[]
        for split,control,capacities in (('train','if',range(4,14)),('valid','switch',range(14,24))):
            for family in ('Bounds','Pointer','Lifetime'):
                for n in capacities if family=='Bounds' else (4,):
                    for label in (0,1):
                        spec=dict(family=family,capacity=n,offset=0,access='read',s=0,r=0,t=0,u=0,
                                  composition=True,predicate='xor',control=control,
                                  initial=0,first=1,second=0,reverse=bool(label))
                        if family!='Bounds':spec.update(initial=label,first=label,second=label,reverse=False)
                        self.assertEqual(int(any(o['violation'] for o in outcomes(spec))),label)
                        rows.append(dict(id=len(rows),split=split,family=family,spec=spec,
                                         label=label,source=render(spec),structure_family=[control,'xor',True]))
        audit=structural_shortcut_audit(rows)
        self.assertFalse(audit['admission'])
        self.assertLess(audit['lexical']['auc'],audit['limits']['proxy_auc'])
        self.assertGreaterEqual(audit['lexical']['families']['Pointer']['auc'],audit['limits']['proxy_auc'])
        self.assertEqual(audit['unseen_object_state_combinations'],[])
        self.assertIn('shared_core_program_instances',audit['failures'])

    def test_ranking_gradient_and_original_trainer_roundtrip(self):
        import tempfile
        from pathlib import Path
        import torch
        from vulnmechanism import model as core
        from tests.test_graph_training import mock_models
        from tests.test_graph_features import sample_row
        for mode in ('bce','bce_rank'):
            logits=torch.tensor([.1,.2],requires_grad=True)
            loss=core.synthetic_supervision_loss(logits,mode);loss.backward()
            self.assertGreater(float(logits.grad[0]),0)
            self.assertLess(float(logits.grad[1]),0)
            rows=[sample_row(str(i),label=i%2) for i in range(4)]
            rows += [sample_row('v0','valid',0),sample_row('v1','valid',1)]
            pair=[dict(raw_source='int f(){return 0;}',label=0,split='train'),
                  dict(raw_source='int f(){return 1;}',label=1,split='train')]
            with tempfile.TemporaryDirectory() as d, mock_models():
                path=Path(d)/'best.pt'
                cp=core.train_model(None,path,records=rows,variant='baseline',model_path='tiny',
                    device='cpu',epochs=1,batch_size=1,gradient_accumulation=3,source_max_length=32,
                    synthetic_pairs=[pair],synthetic_objective=mode)
                self.assertEqual(cp['synthetic_supervision']['objective'],mode)
                scores=core.predict_checkpoint(path,rows[-2:],device='cpu')
                self.assertTrue(torch.isfinite(scores).all())
                self.assertEqual(tuple(scores.shape),(2,))

    def test_transfer_rejects_synthetic_valid(self):
        from vulnmechanism.model import train_model
        with self.assertRaisesRegex(ValueError,'training pairs'):
            train_model(None,'unused.pt',records=[],variant='baseline',model_path='unused',
                synthetic_pairs=[[dict(label=0,split='valid'),dict(label=1,split='train')]],
                synthetic_objective='bce_rank')

if __name__=='__main__':unittest.main()
