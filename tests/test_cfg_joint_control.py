import copy
import io
import json
from pathlib import Path
import unittest
import torch
from vulnmechanism.cfg_behavior import (LocalGraph,BoundVocabulary,extract_bound_behavior,
    JOINT_VARIANTS,FAMILIES,extract_behavior)
from vulnmechanism.cfg_control import postdominators,control_queries
from vulnmechanism.cfg_data import abstract_cfg
from vulnmechanism.cfg_network import AttributeCFGEncoder,collate_graphs
from vulnmechanism.cfg_dependency import ControlRelationHead,relation_function_loss,accumulation_window_loss

class CharacterTokenizer:
    is_fast=True
    def __call__(self,source,**kwargs):
        length=min(len(source),kwargs.get('max_length',len(source)))
        return {'input_ids':list(range(length)), 'offset_mapping':[(i,i+1) for i in range(length)]}
class Builder:
    tokenizer=CharacterTokenizer();source_prefix=[1];source_max_length=2048

class JointControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.fixture=json.loads(Path('tests/fixtures/joern_control_relations.json').read_text())
        cls.graph=cls.fixture['graph'];cls.item=extract_bound_behavior(cls.graph)
        cls.vocab=BoundVocabulary.fit([{'sample_key':'t','split':'train'}],{'t':cls.item})
        cls.view=abstract_cfg(cls.graph)
    def model(self,mode):
        torch.manual_seed(42)
        return AttributeCFGEncoder([3]*4,hidden_size=8,mode=mode,behavior_sizes=self.vocab.sizes()).double()
    def batch(self,item=None):
        return collate_graphs([self.view],[[[0]*4]*len(self.view['node_ids'])],behavior=[self.vocab.encode(item or self.item)])
    def test_initial_equivalence_and_shared_gradients(self):
        batch=self.batch();c=self.model('cfg');expected=c(batch);expected.sum().backward()
        for mode in JOINT_VARIANTS:
            model=self.model(mode);actual=model(batch)
            self.assertTrue(torch.equal(expected,actual),mode);actual.sum().backward()
            # x_C has a separate readout path: autograd addition order differs by
            # one double-precision ULP (measured max 2.22e-16), not the derivative.
            for name,p in c.named_parameters():
                torch.testing.assert_close(p.grad,dict(model.named_parameters())[name].grad,rtol=0,atol=5e-15)
    def test_zero_start_then_upstream_gradients_and_disabled_equivalence(self):
        model=self.model('joint');batch=self.batch()
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        model(batch).square().sum().backward()
        self.assertEqual(model.bound_nodes.values[0].weight.grad.abs().sum().item(),0)
        self.assertEqual(model.semantic_hidden.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.node_residual.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.semantic_output.weight.grad.abs().sum().item(),0)
        optimizer.step();optimizer.zero_grad();model(batch).square().sum().backward()
        self.assertGreater(model.bound_nodes.values[0].weight.grad.abs().sum().item(),0)
        self.assertGreater(model.semantic_hidden.weight.grad.abs().sum().item(),0)
        model.disabled_families=set(FAMILIES)
        c=self.model('cfg');c.load_state_dict({k:model.state_dict()[k] for k in c.state_dict()})
        self.assertTrue(torch.equal(c(batch),model(batch)))
    def test_role_binding_direction_and_count(self):
        g=LocalGraph(self.graph);root=next(n for n in g.nodes if g.nodes[n]['code']=='x<16')
        original=copy.deepcopy(self.graph)
        children=set(g.children(root))
        for n in original['nodes']:
            if n['id'] in children:n['properties']['TYPE_FULL_NAME']='int'
        swapped=copy.deepcopy(original)
        for n in swapped['nodes']:
            if n['id'] in children:n['properties']['ARGUMENT_INDEX']=3-n['properties']['ARGUMENT_INDEX']
        a=extract_bound_behavior(original);b=extract_bound_behavior(swapped);i=a['node_ids'].index(root)
        self.assertEqual([sorted(v) for v in a['nodes'][i]],[sorted(v) for v in b['nodes'][i]])
        self.assertNotEqual(a['node_facts'][i],b['node_facts'][i])
        vocab=BoundVocabulary.fit([{'sample_key':'a','split':'train'},{'sample_key':'b','split':'train'}],{'a':a,'b':b})
        model=AttributeCFGEncoder([3]*4,hidden_size=8,mode='joint',behavior_sizes=vocab.sizes())
        encoded=[vocab.encode(x)['nodes'][i] for x in (a,b)]
        vectors=model.bound_nodes(encoded);self.assertFalse(torch.allclose(vectors[0],vectors[1]))
        repeated=copy.deepcopy(encoded[0]);repeated[3]+=copy.deepcopy(repeated[3])
        counts=model.bound_nodes([encoded[0],repeated]);self.assertFalse(torch.allclose(counts[0],counts[1]))
    def test_complete_real_cfg_labels_and_no_complement(self):
        row={'sample_key':'real','split':'train','raw_source':self.fixture['source']}
        queries,counts=control_queries(row,self.graph,Builder(),limit=10000)
        self.assertGreater(counts['selected_positive'],0);self.assertGreater(counts['selected_negative'],0)
        kinds={q['branch'] for q in queries};self.assertTrue({0,1,2,3}<=kinds)
        bypair={}
        for q in queries:bypair.setdefault((q['condition_node'],q['use_node']),{})[q['branch']]=q['label']
        self.assertTrue(any(v.get(0)==v.get(1)==0 for v in bypair.values()))
        self.assertTrue(all(q['condition_token']<2049 and q['use_token']<2049 for q in queries))
        with self.assertRaises(ValueError):control_queries(dict(row,split='test'),self.graph,Builder())
    def test_postdominance_and_incomplete_graph(self):
        view={'edges':[(0,1),(1,2),(1,3),(2,4),(3,4),(4,5)]}
        pd,reason=postdominators(view,['METHOD','CALL','CALL','CALL','RETURN','METHOD_RETURN'])
        self.assertIsNone(reason);self.assertTrue(pd[2] & (1<<4));self.assertFalse(pd[1] & (1<<2))
        view['edges']=[(0,1),(1,2),(1,3),(2,2),(3,4),(4,5)]
        self.assertIsNone(postdominators(view,['METHOD','CALL','CALL','CALL','RETURN','METHOD_RETURN'])[0])
    def test_context_query_fit_label_not_input_and_gradient(self):
        torch.manual_seed(11);hidden=torch.randn(2,3,8,requires_grad=True)
        queries=[[{'condition_token':0,'use_token':2,'branch':b,'case_token':None,'label':int(b==state)} for b in (0,1)] for state in (0,1)]
        head=ControlRelationHead(8,4);optimizer=torch.optim.Adam(head.parameters(),lr=.02)
        for _ in range(150):
            optimizer.zero_grad();loss,_,_=relation_function_loss(hidden.detach(),queries,head);loss.backward();optimizer.step()
        loss,labels,logits=relation_function_loss(hidden,queries,head)
        self.assertLess(loss.item(),.02);self.assertEqual([int(x>0) for x in logits],labels)
        changed=copy.deepcopy(queries[0]);changed[0]['label']=1-changed[0]['label'];changed[0]['reason']='fake'
        torch.testing.assert_close(head(hidden[0],queries[0]),head(hidden[0],changed))
        loss.backward();self.assertGreater(hidden.grad.abs().sum().item(),0)
    def test_function_denominator_and_microbatch_gradients(self):
        torch.manual_seed(42);head=ControlRelationHead(8,4)
        hidden=torch.randn(3,3,8,requires_grad=True)
        q={'condition_token':0,'use_token':2,'branch':0,'case_token':None,'label':1}
        queries=[[q],[],[dict(q,label=0),dict(q,branch=1)]]
        loss,_,_=relation_function_loss(hidden,queries,head);loss.backward();expected=hidden.grad.clone()
        hidden.grad=None
        for i in range(3):
            value,_,_=relation_function_loss(hidden[i:i+1],queries[i:i+1],head)
            (value*bool(queries[i])/2).backward()
        torch.testing.assert_close(expected,hidden.grad,atol=1e-7,rtol=1e-6)

    def test_nonzero_semantics_unknowns_self_loops_and_alternative_mean(self):
        model=self.model('joint_edges');batch=self.batch()
        with torch.no_grad():model.semantic_output.weight.fill_(.02)
        changed=copy.deepcopy(batch)
        changed.behavior[0]['edges']=[[copy.deepcopy(changed.behavior[0]['neutral'])] for _ in changed.behavior[0]['edges']]
        self.assertFalse(torch.allclose(model(batch),model(changed)))
        c=self.model('cfg');self.assertTrue(torch.equal(c(changed),model(changed)))
        unknown=copy.deepcopy(batch)
        for alts in unknown.behavior[0]['edges']:
            for alt in alts:
                for facts in alt:
                    for fact in facts:fact[2]=0
        self.assertTrue(torch.equal(c(unknown),model(unknown)))
        duplicated=copy.deepcopy(batch)
        duplicated.behavior[0]['edges']=[a+copy.deepcopy(a) for a in duplicated.behavior[0]['edges']]
        torch.testing.assert_close(model(batch),model(duplicated),rtol=1e-12,atol=1e-12)
        view=copy.deepcopy(self.view);view['edges']=list(view['edges'])+[(0,0)]
        encoded=self.vocab.encode(self.item)
        semantic=next(a for a in encoded['edges'] if a[0]!=encoded['neutral'])
        encoded['edges'].append(copy.deepcopy(semantic))
        self_loop=collate_graphs([view],[[[0]*4]*len(view['node_ids'])],behavior=[encoded])
        neutral_loop=copy.deepcopy(self_loop);neutral_loop.behavior[0]['edges'][-1]=[copy.deepcopy(encoded['neutral'])]
        self.assertFalse(torch.allclose(model(self_loop),model(neutral_loop)))

    def test_renumbering_batch_isolation_and_control_sampling(self):
        changed=copy.deepcopy(self.graph);mapping={n['id']:str(900000-i) for i,n in enumerate(changed['nodes'])}
        for n in changed['nodes']:n['id']=mapping[n['id']]
        for e in changed['edges']:e['source']=mapping[e['source']];e['target']=mapping[e['target']]
        row={'sample_key':'real','split':'train','raw_source':self.fixture['source']}
        q,_=control_queries(row,self.graph,Builder());other,_=control_queries(row,changed,Builder())
        def inputs(queries):return sorted((r['condition_token'],r['use_token'],r['branch'],r['case_token'] or -1,r['label']) for r in queries)
        self.assertEqual(inputs(q),inputs(other))
        item=extract_bound_behavior(changed);view=abstract_cfg(changed)
        batch=collate_graphs([view],[[[0]*4]*len(view['node_ids'])],behavior=[self.vocab.encode(item)])
        model=self.model('joint')
        with torch.no_grad():model.node_residual.weight.fill_(.01);model.semantic_output.weight.fill_(.02)
        expected=model(self.batch());torch.testing.assert_close(expected,model(batch),rtol=1e-10,atol=1e-10)
        together=collate_graphs([self.view,view],[[[0]*4]*len(view['node_ids'])]*2,
            behavior=[self.vocab.encode(self.item),self.vocab.encode(item)])
        torch.testing.assert_close(model(together),expected.expand(2,-1),rtol=1e-10,atol=1e-10)

    def test_real_positive_control_dependencies_crosscheck_joern(self):
        for name in ('joern_control_relations.json','joern_behavior_c.json','joern_behavior_cpp.json'):
            fixture=json.loads(Path('tests/fixtures',name).read_text())
            q,_=control_queries({'sample_key':'x','split':'train','raw_source':fixture['source']},fixture['graph'],Builder(),limit=10000)
            cdg={(e['source'],e['target']) for e in fixture['graph']['edges'] if e['kind']=='CDG'}
            self.assertTrue(q)
            self.assertTrue({(r['condition_node'],r['use_node']) for r in q if r['label']}<=cdg)

    def test_unreliable_position_truncation_and_missing_branch_are_not_negatives(self):
        row={'sample_key':'x','split':'valid','raw_source':self.fixture['source']}
        builder=Builder();builder.source_max_length=2
        queries,stats=control_queries(row,self.graph,builder)
        self.assertEqual(queries,[]);self.assertGreater(stats['alignment_outside_visible_source'],0)
        changed=copy.deepcopy(self.graph)
        for node in changed['nodes']:
            if node['properties']['kind']=='CALL':node['properties']['OFFSET']=1000000
        queries,stats=control_queries(row,changed,Builder())
        broken={n['id'] for n in changed['nodes'] if n['properties']['kind']=='CALL'}
        # Reliable RETURN targets and identifier conditions must survive;
        # corrupting CALL positions is not permission to drop the function.
        self.assertTrue(queries)
        self.assertTrue(all(q['condition_node'] not in broken and q['use_node'] not in broken for q in queries))
        self.assertGreater(stats['alignment_position_mismatch'],0)
        original=copy.deepcopy(self.graph);item=extract_behavior(original)
        g=LocalGraph(original);condition=next(n for n in g.nodes if g.nodes[n]['code']=='x<16')
        removed=next((item['node_ids'][a],item['node_ids'][b]) for (a,b),alts in zip(item['cfg_edges'],item['edges'])
                     if item['node_ids'][a]==condition and alts[0][0]==['true'])
        original['edges']=[e for e in original['edges'] if not (e['kind']=='CFG' and (e['source'],e['target'])==removed)]
        queries,stats=control_queries(row,original,Builder())
        self.assertTrue(all(q['condition_node']!=condition for q in queries))
        self.assertGreater(stats['unknown_incomplete_decisions'],0)

    def test_formal_shuffle_keeps_records_and_head_gets_no_answers(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from vulnmechanism.cfg_experiment import _graph_inputs
        encoded=self.vocab.encode(self.item)
        views={'a':dict(self.view,behavior=encoded),'b':dict(self.view,behavior=copy.deepcopy(encoded))}
        model=SimpleNamespace(task_modules={'cfg_encoder':SimpleNamespace(mode='joint_shuffled')})
        builder=SimpleNamespace(sequence_batch=lambda batch,**kwargs:(torch.zeros((len(batch),1),dtype=torch.long),torch.ones((len(batch),1))))
        rows=[{'sample_key':'a'},{'sample_key':'b'}];original=[[[0]*4]*len(self.view['node_ids'])]*2
        inputs=_graph_inputs(model,rows,builder,views,dict(zip(('a','b'),original)),'cpu')[2]
        again=_graph_inputs(model,rows,builder,views,dict(zip(('a','b'),original)),'cpu')[2]
        for shuffled in inputs.behavior:
            self.assertEqual(sorted(map(json.dumps,shuffled['edges'])),sorted(map(json.dumps,encoded['edges'])))
            self.assertEqual(shuffled['nodes'],encoded['nodes'])
        self.assertEqual(inputs.behavior,again.behavior)
        self.assertNotEqual(inputs.behavior[0]['edges'],encoded['edges'])
        head=ControlRelationHead(8,4);hidden=torch.randn(1,3,8)
        query={'condition_token':0,'use_token':2,'case_token':None,'branch':0,'label':1,'reason':'answer','use_node':'id'}
        original_forward=head.forward
        def forward(states,descriptions):
            self.assertEqual(set(descriptions[0]),{'condition_token','use_token','case_token','branch'})
            return original_forward(states,descriptions)
        with patch.object(head,'forward',side_effect=forward):
            relation_function_loss(hidden,[[query]],head)

    def test_unknown_literal_lexeme_is_not_a_known_feature(self):
        item=copy.deepcopy(self.item)
        item['node_facts'][0][3]=[{'object':'synthetic','path':['SELF'],'value':v}
                                 for v in ('UNKNOWN','lexeme=<global>')]
        vocab=BoundVocabulary.fit([{'sample_key':'t','split':'train'}],{'t':item})
        self.assertEqual([f[2] for f in vocab.encode(item)['nodes'][0][3]],[0,0])
