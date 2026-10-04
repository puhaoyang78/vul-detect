import copy
import unittest
import torch
from torch import nn
from vulnmechanism.cfg_network import ConditionalSourcePool,SourceGraphClassifier,collate_graphs
from tests import test_cfg_joint_control as joint_fixture
from tests.test_cfg_regions import HPSource


class SourcePoolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_uniform_padding_and_identical_initialization(self):
        hidden=torch.randn(2,5,4);mask=torch.tensor([[1,1,0,0,0],[1,1,1,1,1]])
        mean=(hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
        graph=torch.randn(2,6)
        for structural in (False,True):
            pool=ConditionalSourcePool(4,6,structural=structural)
            self.assertTrue(torch.equal(pool(hidden,mask,graph,mean),mean))
            low=hidden.to(torch.bfloat16);low_mask=mask.unsqueeze(-1).to(low.dtype)
            low_mean=((low*low_mask).sum(1)/low_mask.sum(1)).float()
            self.assertTrue(torch.equal(pool(low,mask,graph,low_mean),low_mean))
            self.assertEqual(pool.weights(hidden,mask,graph)[0,2:].sum(),0)
            changed=hidden.clone();changed[0,2:]=1000
            self.assertTrue(torch.equal(pool(hidden,mask,graph,mean),pool(changed,mask,graph,mean)))
            with torch.no_grad():pool.score.weight.normal_()
            self.assertTrue(torch.equal(pool(hidden,mask,graph,mean),pool(changed,mask,graph,mean)))
            with self.assertRaises(ValueError):pool.weights(hidden,mask*0,graph)

    def test_content_structure_interaction_fit_and_batch_isolation(self):
        hidden=torch.tensor([[[-1.,-1.],[1.,1.]],[[-1.,-1.],[1.,1.]],
                             [[-1.,1.],[1.,-1.]],[[-1.,1.],[1.,-1.]]])
        graph=torch.tensor([[-1.],[1.],[-1.],[1.]],requires_grad=True)
        mask=torch.ones(4,2,dtype=torch.long);mean=hidden.mean(1)
        pool=ConditionalSourcePool(2,1,structural=True,rank=8)
        classifier=nn.Linear(2,1)
        optimizer=torch.optim.Adam([*pool.parameters(),*classifier.parameters()],lr=.03)
        target=torch.tensor([0.,1.,1.,0.])
        for _ in range(350):
            optimizer.zero_grad()
            logits=classifier(pool(hidden,mask,graph,mean)).squeeze(-1)
            loss=nn.functional.binary_cross_entropy_with_logits(logits,target)
            loss.backward();optimizer.step()
        self.assertLess(float(loss.detach()),.03)
        self.assertEqual((logits>0).tolist(),target.bool().tolist())
        weights=pool.weights(hidden,mask,graph)
        self.assertGreater(float((weights[0]-weights[1]).detach().abs().max()),.2)
        self.assertGreater(float(graph.grad.abs().sum()),0)
        split=torch.cat([pool.weights(hidden[i:i+1],mask[i:i+1],graph[i:i+1]) for i in range(4)])
        torch.testing.assert_close(weights,split)
        order=torch.tensor([3,1,0,2])
        torch.testing.assert_close(weights[order],pool.weights(hidden[order],mask[order],graph[order]))
        source=copy.deepcopy(pool);source.structural=False
        self.assertTrue(torch.equal(source.weights(hidden,mask,graph),source.weights(hidden,mask,graph*10)))
        clone=ConditionalSourcePool(2,1,structural=True,rank=8);clone.load_state_dict(pool.state_dict())
        self.assertTrue(torch.equal(weights,clone.weights(hidden,mask,graph)))

    def test_joint_single_forward_gradients_shared_rng_and_reload(self):
        joint_fixture.JointControlTests.setUpClass();fixture=joint_fixture.JointControlTests();batch=fixture.batch()
        ids=torch.tensor([[1,2,3,4]]);mask=torch.tensor([[1,1,1,0]])
        models=[];states=[]
        for mode in ('mean','source','structure'):
            torch.manual_seed(42)
            models.append(SourceGraphClassifier(HPSource(),[3]*4,hidden_size=8,steps=5,mode='joint',
                device='cpu',behavior_sizes=fixture.vocab.sizes(),source_pool=mode))
            states.append(torch.get_rng_state())
        for model,state in zip(models[1:],states[1:]):
            self.assertTrue(torch.equal(states[0],state))
            for name,p in models[0].named_parameters():self.assertTrue(torch.equal(p,dict(model.named_parameters())[name]))
            self.assertTrue(torch.equal(models[0](ids,mask,batch),model(ids,mask,batch)))
        self.assertEqual(sum(p.numel() for p in models[1].parameters()),sum(p.numel() for p in models[2].parameters()))
        for key,value in models[1].task_modules['source_pool'].state_dict().items():
            self.assertTrue(torch.equal(value,models[2].task_modules['source_pool'].state_dict()[key]))
        model=models[2];pool=model.task_modules['source_pool']
        calls=model.encoder.calls
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        nn.functional.binary_cross_entropy_with_logits(model(ids,mask,batch),torch.ones(1)).backward()
        self.assertEqual(model.encoder.calls,calls+1)
        self.assertGreater(float(pool.score.weight.grad.abs().sum()),0)
        self.assertEqual(float(pool.tokens.weight.grad.abs().sum()),0)
        optimizer.step();optimizer.zero_grad()
        # Disable the graph classifier to prove a source-readout-to-graph gradient path.
        with torch.no_grad():model.task_modules['cfg_classifier'].weight.zero_()
        loss=nn.functional.binary_cross_entropy_with_logits(model(ids,mask,batch),torch.ones(1));loss.backward()
        for p in (pool.tokens.weight,pool.condition.weight,
                  model.task_modules['cfg_encoder'].message.weight,model.encoder.adapter.weight):
            self.assertGreater(float(p.grad.abs().sum()),0)
        clone=copy.deepcopy(model);clone.load_state_dict(model.state_dict())
        self.assertTrue(torch.equal(model(ids,mask,batch),clone(ids,mask,batch)))
