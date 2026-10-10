import io
import unittest
import torch
from vulnmechanism.cgvl import local_operations, verified_pairs, OperationConstraintAlignment, local_rank_loss


class CGVLTests(unittest.TestCase):
    def test_three_properties_and_reverified_pairs(self):
        samples=[('Bounds','void f(){int a[4]; a[1]=0;}'),
                 ('Pointer','void f(){int x; int *p=&x; *p=0;}'),
                 ('Lifetime','void f(){int *p=malloc(4); if(p){free(p);}}')]
        for family,source in samples:
            with self.subTest(family=family):
                ops=local_operations(source,'c')
                self.assertEqual([x['state'] for x in ops if x['family']==family],['Satisfied'])
                pairs=verified_pairs(source,'c',ops)
                unsafe=[p for p in pairs if p['kind']=='unsafe']
                self.assertEqual(len(unsafe),1)
                self.assertEqual(unsafe[0]['changed_state'],'Violated')
                self.assertIsNone(unsafe[0]['function_label'])

    def test_verdicts_are_not_model_inputs_and_controls_remain_safe(self):
        from vulnmechanism.cgvl import model_spans
        class Characters:
            def __call__(self,source,**kwargs):
                return {'offset_mapping':[(i,i+1) for i in range(min(len(source),kwargs['max_length']))]}
        source='void f(){int a[4]; a[1]=0;}'
        ops=local_operations(source,'c')
        features=model_spans(source,ops,Characters())
        for op in ops:op.update(state='Violated',reason='different truth',edits=[])
        self.assertEqual(features,model_spans(source,ops,Characters()))
        self.assertEqual(set(features[0]),{'family','operation','context'})
        ops=local_operations(source,'c');pairs=verified_pairs(source,'c',ops)
        self.assertEqual({p['kind'] for p in pairs},{'unsafe','safe_change','rename','unrelated_comment'})
        for p in pairs:
            if p['kind']!='unsafe':self.assertEqual(p['changed_state'],'Satisfied')
        self.assertFalse(model_spans(source,ops,Characters(),max_length=5))

    def test_immediate_parameter_guard_is_a_nullness_proof_only(self):
        source='int f(int *p){if(p){return *p;} return 0;}'
        ops=local_operations(source,'c')
        self.assertEqual(ops[0]['state'],'Satisfied')
        self.assertEqual(ops[0]['reason'],'guarded_parameter_nullness')
        pairs=verified_pairs(source,'c',ops)
        self.assertTrue(any(p['kind']=='unsafe' and p['changed_state']=='Violated' for p in pairs))
        for source in ['int f(int*p){if(p){change(&p); return *p;}return 0;}',
                       'int f(int*p){if(p){return (p=0, *p);}return 0;}',
                       'int f(int*p){if(p){return sizeof(*p);}return 0;}',
                       'int f(int*p){if(!p){return &*p;}return 0;}',
                       'int f(int*p){if(!p){return 0 && *p;}return 0;}']:
            self.assertEqual(local_operations(source,'c')[0]['state'],'Unknown')
        self.assertEqual(local_operations('int f(int*p){if(!p){return __alignof__(*p);}return 0;}','c'),[])

    def test_unknown_is_not_violated(self):
        for source in ['void f(int *p){*p=0;}',
                       'void f(){int a[4]; a[i]=0;}',
                       'void f(){int x; int *p=&x; change(&p); *p=0;}',
                       'void f(){int *p=malloc(4); free(p); free(p);}',
                       'void f(){T a[4]; a[1]=0;}']:
            ops=local_operations(source,'c')
            self.assertTrue(any(x['state']=='Unknown' for x in ops),source)
            self.assertFalse(any(p['kind']=='unsafe' for p in verified_pairs(source,'c',ops)),source)

    def test_shadow_sizeof_address_and_parse_are_not_proofs(self):
        for source in ['void f(){int a[4]; {int a[2]; a[3]=0;}}',
                       'void f(){int a[4]; sizeof(a[4]);}',
                       'void f(){int a[4]; int *p=&a[4];}',
                       'void f(){int a[4]; a[1]=0; BAD @; }']:
            self.assertFalse(any(x['state']!='Unknown' for x in local_operations(source,'c')),source)

    def test_readout_gradient_correspondence_and_roundtrip(self):
        torch.manual_seed(42)
        m=OperationConstraintAlignment(8,4)
        h=torch.randn(8,8,requires_grad=True); mask=torch.tensor([1]*7+[0])
        items=[dict(operation=[2],context=[[0],[1]]),dict(operation=[5],context=[[3],[4]])]
        pooled,risk=m(h,mask,items);shuffled,_=m(h,mask,items,shuffled=True)
        self.assertFalse(torch.allclose(pooled,shuffled))
        classifier=torch.nn.Linear(8,1)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(classifier(pooled),torch.ones(1))
        loss.backward(retain_graph=True)
        self.assertGreater(float(m.interaction[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(m.risk.weight.grad.abs().sum()),0)
        self.assertGreater(float(h.grad.abs().sum()),0)
        rank=local_rank_loss(risk[:1],risk[1:]);rank.backward()
        stream=io.BytesIO();torch.save(m.state_dict(),stream);stream.seek(0)
        loaded=OperationConstraintAlignment(8,4);loaded.load_state_dict(torch.load(stream,weights_only=True))
        torch.testing.assert_close(loaded(h,mask,items)[0],pooled)
        empty,_=m(h,mask,[]);torch.testing.assert_close(empty,h[:7].mean(0))



class RealSourceConstructionTests(unittest.TestCase):
    def test_entry_guard_and_early_return_have_null_witness(self):
        from vulnmechanism.cgvl import construction_sites, constructed_variants
        for source in ['int f(int *p){if(p){return *p;}return 0;}',
                       'int f(int *p){if(!p)return 0;return *p;}']:
            variants = constructed_variants(source,'c')
            self.assertEqual([v['verified']['state'] for v in variants],['Safe','Unsafe','Safe'])
            self.assertEqual(variants[1]['verified']['witness'],{'p':'NULL'})
            self.assertNotEqual(variants[0]['source'],variants[1]['source'])
            self.assertEqual(construction_sites(source,'c')[0]['state'],'Safe')

    def test_loop_crosses_capacity_and_iteration_bound(self):
        from vulnmechanism.cgvl import constructed_variants
        source='void f(){int a[4];int i;for(i=0;i<4;i++){a[i]=0;}}'
        v=constructed_variants(source,'c')
        self.assertEqual([x['verified']['state'] for x in v],['Safe','Unsafe','Safe','Unsafe','Safe'])
        self.assertEqual(v[1]['verified']['bound'],v[2]['verified']['bound'])
        self.assertNotEqual(v[1]['verified']['capacity'],v[2]['verified']['capacity'])
        self.assertEqual(v[1]['verified']['witness'],{'iteration':4})
        source='void f(){char a[4];for(int i=0;i<4;++i)a[i]=0;}'
        self.assertEqual(len(constructed_variants(source,'c')),5)

    def test_unreachable_or_unbound_or_side_effecting_is_not_admitted(self):
        from vulnmechanism.cgvl import construction_sites
        sources=['int f(int*p){return 0;if(p)return *p;}',
            'int f(int*p){call();if(p)return *p;}',
            'int f(int*p){if(p){call();return *p;}return 0;}',
            'int f(int*p){if(p)return p[0];return 0;}',
            'int f(int*p){if(p)return sizeof(*p);return 0;}',
            'int f(int*p){if(p)return 0 && *p;return 0;}',
            'void f(){int a[4];int i;for(i=0;i<4;i++){call();a[i]=0;}}',
            'void f(){int a[4];int i;for(i=0;i<4;i++)a[i++]=0;}',
            'void f(){int a[4];for(int i=0;i<4;i++){int a[4];a[i]=0;}}',
            'void f(int*p){free(p);}',
            'void f(){int a[4];int x;int intx;for(intx=0;x<4;x++)a[x]=0;}',
            'void f(){int a[4];for(int *i=0;i<4;i++)a[i]=0;}',
            'void f(){int a[4];for(int i=0;i<4;i++)a[i]=1e100;}',
            'void f(){int a[4];for(int i=0;i<4;i++)a[i]=g();}']
        for source in sources:
            self.assertEqual(construction_sites(source,'c'),[],source)
        self.assertEqual(construction_sites('int f(int*p){if(p)return *p;return 0;}','cpp'),[])

    def test_unicode_offsets_and_preserved_source(self):
        from vulnmechanism.cgvl import constructed_variants
        source='/* 中文 */ int f(int*p){if(!p)return 0;return *p;}'
        for v in constructed_variants(source,'c')[:2]:
            raw=source.encode()
            for e in sorted(v['edits'],key=lambda x:x['span'][0],reverse=True):
                a,b=e['span'];raw=raw[:a]+e['text'].encode()+raw[b:]
            self.assertEqual(raw.decode(),v['source'])
            a,b=v['verified']['operation']
            self.assertEqual(v['source'].encode()[a:b],b'*p')

if __name__ == '__main__':
    unittest.main()
