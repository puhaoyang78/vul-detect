import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch
from vulnmechanism.cfg_behavior import (extract_behavior, literal, LocalGraph, loop_roles,
    BehaviorVocabulary, NODE_FAMILIES, EDGE_FAMILIES, FAMILIES)
from vulnmechanism.cfg_data import abstract_cfg, serialize_graph
from vulnmechanism.cfg_network import AttributeCFGEncoder, collate_graphs
from vulnmechanism.cpg import read_neo4jcsv, GraphEdge


class BehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads((Path(__file__).parent/'fixtures/joern_behavior_c.json').read_text())
        cls.graph=cls.fixture['graph'];cls.features=extract_behavior(cls.graph)
        cls.vocab=BehaviorVocabulary.fit([{'sample_key':'train','split':'train'}],{'train':cls.features})
    def batch(self,graph=None):
        graph=graph or self.graph;view=abstract_cfg(graph);item=extract_behavior(graph)
        return collate_graphs([view],[[[0]*4 for _ in view['node_ids']]],behavior=[self.vocab.encode(item)])
    def model(self,mode):
        torch.manual_seed(42)
        return AttributeCFGEncoder([3]*4,hidden_size=8,mode=mode,behavior_sizes=self.vocab.sizes()).double()
    def test_all_relations_and_edge_fields_survive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name,header,rows in [('nodes_IDENTIFIER',[':ID',':LABEL','CODE:string'],[['1','IDENTIFIER','x']]),
                 ('edges_CONDITION',[':START_ID',':END_ID',':TYPE','INDEX:int','FLAG:boolean','LABEL:string'],[['1','1','CONDITION','2','false','a\\\\b']])]:
                with (root/(name+'_header.csv')).open('w') as f:csv.writer(f).writerow(header)
                with (root/(name+'_data.csv')).open('w') as f:csv.writer(f).writerows(rows)
            _,edges=read_neo4jcsv(root)
            self.assertEqual(edges[0].kind,'CONDITION')
            self.assertEqual(edges[0].properties,{'INDEX':2,'FLAG':False,'LABEL':'a\\b'})
        self.assertTrue({'CONDITION','ARGUMENT','REF'}<={e['kind'] for e in self.graph['edges']})
    def test_literal_no_cutoff_suffix_sign_base_and_null(self):
        self.assertIn('value=18446744073709551615',literal('0xffffffffffffffffULL'))
        self.assertIn('suffix=ull',literal('0xffffffffffffffffULL'))
        self.assertIn('value=-8',literal('-010L'))
        self.assertNotEqual(literal('0'),literal('nullptr'))
        self.assertIn('length=2',literal('"a\\n"'))
        self.assertIn('length=UNKNOWN',literal('L"\\u1234"'))
        self.assertIn('length=3',literal('u8"\\u1234"'))
        self.assertIn('length=2',literal('u"\\U0001f600"'))
        self.assertIn('length=1',literal('U"\\U0001f600"'))
        self.assertIn('length=1',literal('u8"\\xff"'))
        self.assertIn('length=3',literal('"a" "bc"'))
        self.assertIn('length=2',literal('R"(\\n)"'))
    def test_symbolic_type_extents_do_not_embed_variable_names(self):
        from vulnmechanism.cfg_behavior import type_features
        values=type_features('const unsigned int *[private_variable + 1]', 'array_base')
        self.assertFalse(any('private_variable' in t for t in values))
        self.assertIn('array_base:type=const unsigned int *[UNKNOWN]',values)
        self.assertIn('array_base:pointer_depth=1',values)
        self.assertIn('array_base:qualifier=const',values)
        self.assertIn('array_base:signedness=unsigned',values)
        self.assertIn('array_base:width=UNKNOWN',values)

    def test_real_control_structures_guards_and_transfers(self):
        item=self.features;g=LocalGraph(self.graph);seen=set()
        for (a,b),alts in zip(item['cfg_edges'],item['edges']):
            src=item['node_ids'][a];dst=item['node_ids'][b]
            seen.update(t for alt in alts for t in alt[2])
            if g.kind(dst)=='JUMP_TARGET' and g.nodes[dst]['code'].startswith('case') and g.kind(src)=='IDENTIFIER':
                self.assertEqual(alts[0][0],['case'])
            if any(alt[0][0] in ('true','false') for alt in alts):self.assertNotEqual(alts[0][1],['UNKNOWN'])
        self.assertTrue({'break','continue','goto','return','switch_fallthrough'}<=seen)
        branches={a[0][0] for alts in item['edges'] for a in alts}
        self.assertTrue({'true','false','case','default','unconditional'}<=branches)
        cpp=json.loads((Path(__file__).parent/'fixtures/joern_behavior_cpp.json').read_text())
        self.assertTrue(any('explicit_exception' in a[2] for alts in extract_behavior(cpp['graph'])['edges'] for a in alts))
    def test_native_and_ast_recovered_conditions_agree(self):
        stripped=copy.deepcopy(self.graph);stripped['edges']=[e for e in stripped['edges'] if e['kind'] not in ('CONDITION','ARGUMENT','REF')]
        self.assertEqual(extract_behavior(stripped),self.features)
    def test_empty_loops_and_declaration_initializer_native_roles(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/joern_behavior_empty_loops.json').read_text())
        native=extract_behavior(fixture['graph'])
        stripped=copy.deepcopy(fixture['graph'])
        stripped['edges']=[e for e in stripped['edges'] if e['kind']!='CONDITION']
        self.assertEqual(native,extract_behavior(stripped))
        g=LocalGraph(fixture['graph'])
        byid=dict(zip(native['node_ids'],range(len(native['node_ids']))))
        for (a,b),alts in zip(native['cfg_edges'],native['edges']):
            u,v=native['node_ids'][a],native['node_ids'][b]
            if u==v and g.nodes[u]['code']=='n':
                self.assertEqual([a[0][0] for a in alts],['true'])
            if g.nodes[u]['code']=='i<n':
                self.assertTrue(all(a[0][0] in ('true','false') for a in alts))

    def test_access_roles_and_no_control_body_copy(self):
        g=LocalGraph(self.graph)
        index_nodes=[n for n in g.nodes if g.op(n)=='indexAccess']
        self.assertTrue(index_nodes)
        for n in index_nodes:
            self.assertEqual(g.role(g.argument(n,1)),'array_base')
            self.assertEqual(g.role(g.argument(n,2)),'array_index')
            ancestors=set();todo=list(g.parents[n])
            while todo:
                parent=todo.pop()
                if parent in ancestors:continue
                ancestors.add(parent);todo.extend(g.parents[parent])
            if any(g.op(parent)=='sizeOf' for parent in ancestors):
                # This actual Joern node has type ANY, so VLA evaluation cannot be excluded.
                self.assertEqual(g.access_mode(g.argument(n,1)),'UNKNOWN')
            else:
                self.assertEqual(g.access_mode(g.argument(n,1)),'READ_ADDRESS_OPERAND')
        entry=next(n for n in g.nodes if g.kind(n)=='METHOD')
        self.assertEqual(list(g.local(entry)),[entry])
        ids=self.features['node_ids'];self.assertEqual(len(ids),len(set(ids)))
        self.assertGreater(sum(g.nodes[n]['code']=='y' for n in ids),1)
    def test_real_memory_forms_shadowing_pointer_call_and_unevaluated(self):
        from vulnmechanism.cfg_behavior import node_features
        fixture=json.loads((Path(__file__).parent/'fixtures/joern_behavior_access.json').read_text())
        g=LocalGraph(fixture['graph']);features=extract_behavior(fixture['graph'])
        forms={t for row in features['nodes'] for t in row[5]}
        self.assertTrue({'scalar','dereference','array_subscript','direct_field','indirect_field','address_of'}<=forms)
        locals_x=[n for n in g.nodes if g.kind(n)=='LOCAL' and g.prop(n).get('NAME')=='x']
        self.assertEqual(len(locals_x),2)
        for n in g.nodes:
            if g.op(n)=='pointerCall':
                attrs=node_features(g,n)
                self.assertIn('dispatch=DYNAMIC_DISPATCH',attrs[1])
                self.assertEqual(g.access_mode(n),'UNKNOWN')
                self.assertFalse(any('fp' in t for f in attrs for t in f))
                self.assertIn('call_receiver',attrs[7]);self.assertIn('argument:1',attrs[7])
            if g.op(n)=='sizeOf':
                self.assertEqual(g.access_mode(g.argument(n,1)),'UNEVALUATED')

    def test_loop_dominators_not_scc_and_irreducible(self):
        view={'node_ids':list('abcde'),'edges':[(0,1),(1,2),(2,1),(1,3),(3,4)]}
        roles=loop_roles(view,['METHOD']+['CALL']*4)
        self.assertNotIn('back',roles[1]);self.assertIn('back',roles[2]);self.assertIn('exit',roles[3])
        view['edges']=[(0,1),(0,2),(1,2),(2,1),(2,3),(3,4)]
        roles=loop_roles(view,['METHOD']+['CALL']*4)
        self.assertTrue(any('irreducible' in r for r in roles))
        self.assertFalse(any('back' in r for r in roles))
    def test_numbering_identity_and_train_only_vocabulary(self):
        changed=copy.deepcopy(self.graph);mapping={n['id']:str(100000-i) for i,n in enumerate(changed['nodes'])}
        for n in changed['nodes']:n['id']=mapping[n['id']]
        for e in changed['edges']:e['source']=mapping[e['source']];e['target']=mapping[e['target']]
        other=extract_behavior(changed)
        old_nodes=dict(zip(self.features['node_ids'],self.features['nodes']));new_nodes=dict(zip(other['node_ids'],other['nodes']))
        for n,value in old_nodes.items():self.assertEqual(value,new_nodes[mapping[n]])
        old_edges={(mapping[self.features['node_ids'][a]],mapping[self.features['node_ids'][b]]):v for (a,b),v in zip(self.features['cfg_edges'],self.features['edges'])}
        new_edges={(other['node_ids'][a],other['node_ids'][b]):v for (a,b),v in zip(other['cfg_edges'],other['edges'])}
        self.assertEqual(old_edges,new_edges)
        with self.assertRaises(ValueError):BehaviorVocabulary.fit([{'split':'valid'}],{})
    def test_masked_same_capacity_no_guard_leak_rng_and_gradients(self):
        joint=self.model('behavior_joint');after=torch.rand(4);masked=self.model('behavior_masked')
        edges=self.model('behavior_edges')
        for name,value in edges.state_dict().items():
            self.assertTrue(torch.equal(value,joint.state_dict()[name]),name)
        self.assertTrue(torch.equal(after,torch.rand(4)))
        self.assertEqual(joint.state_dict().keys(),masked.state_dict().keys())
        batch=self.batch();changed=copy.deepcopy(batch)
        for edge in changed.behavior[0]['edges']:
            for alt in edge:
                for ts in alt:ts[:]=[2]
        torch.testing.assert_close(masked(batch),masked(changed),rtol=0,atol=0)
        self.assertFalse(torch.allclose(joint(batch),joint(changed)))
        joint(batch).square().sum().backward()
        for name in ('message.weight','update.weight_hh','edge_hidden.weight','edge_output.weight','node_projection.weight'):
            self.assertGreater(dict(joint.named_parameters())[name].grad.abs().sum().item(),0)
    def test_c_regression_zero_edge_residual_and_batch_isolation(self):
        c=self.model('cfg');edge=self.model('behavior_edges')
        with torch.no_grad():edge.edge_output.weight.zero_()
        batch=self.batch()
        torch.testing.assert_close(c(batch),edge(batch),rtol=0,atol=0)
        c(batch).sum().backward();edge(batch).sum().backward()
        for name,p in c.named_parameters():torch.testing.assert_close(p.grad,dict(edge.named_parameters())[name].grad,rtol=0,atol=0)
        joint=self.model('behavior_joint');view=abstract_cfg(self.graph);encoded=self.vocab.encode(self.features)
        together=collate_graphs([view,view],[[[0]*4]*len(view['node_ids'])]*2,behavior=[encoded,encoded])
        torch.testing.assert_close(joint(together),joint(batch).repeat(2,1),atol=1e-12,rtol=1e-12)
    def test_actual_merged_branch_is_disjunction_not_conjunction(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/joern_behavior_merged.json').read_text())
        item=extract_behavior(fixture['graph']);g=LocalGraph(fixture['graph'])
        merged=[alts for (a,b),alts in zip(item['cfg_edges'],item['edges'])
                if g.nodes[item['node_ids'][a]]['code']=='n']
        self.assertEqual(len(merged),1)
        self.assertEqual({tuple(alt[0]) for alt in merged[0]},{('true',),('false',)})
        self.assertEqual(len(merged[0]),2)

    def test_alternative_average_and_self_loop_attribute(self):
        model=self.model('behavior_edges');batch=self.batch();duplicated=copy.deepcopy(batch)
        for alts in duplicated.behavior[0]['edges']:alts.extend(copy.deepcopy(alts))
        torch.testing.assert_close(model(batch),model(duplicated),atol=1e-12,rtol=1e-12)
        view={'node_ids':['one'],'edges':[(0,0)]};encoded=self.vocab.encode(self.features)
        edge=encoded['edges'][0]
        b=collate_graphs([view],[[[0]*4]],behavior=[{'nodes':[encoded['nodes'][0]],'edges':[edge]}])
        changed=copy.deepcopy(b);changed.behavior[0]['edges'][0]=[[[2],[2],[2],[2]]]
        self.assertFalse(torch.allclose(model(b),model(changed)))


if __name__=='__main__':unittest.main()
