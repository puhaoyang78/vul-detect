import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import RobertaConfig, RobertaModel
from vulnmechanism import model as core
from tests.test_codebert_source import CodeTokenizer
from tests.test_graph_features import sample_row

class PrototypeTests(unittest.TestCase):
    def test_k1_and_identical_normalization(self):
        x=torch.randn(3,8,requires_grad=True)
        head=core.MultiPrototypeHead(8,1)
        torch.testing.assert_close(head(x),head.prototypes(x))
        four=core.MultiPrototypeHead(8,4)
        with torch.no_grad():
            four.prototypes.weight.copy_(head.prototypes.weight.expand(4,-1))
            four.prototypes.bias.copy_(head.prototypes.bias.expand(4))
        torch.testing.assert_close(four(x),head(x))
        torch.nn.functional.binary_cross_entropy_with_logits(four(x).squeeze(-1),torch.tensor([0.,1.,1.])).backward()
        self.assertTrue(torch.all(four.prototypes.weight.grad.norm(dim=1)>0))
        self.assertGreater(x.grad.norm(),0)

    def test_rng_encoder_and_end_to_end(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'encoder'
            RobertaModel(RobertaConfig(vocab_size=16,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,max_position_embeddings=514,pad_token_id=0,bos_token_id=2,eos_token_id=1)).save_pretrained(base)
            rows=[sample_row(str(i),label=i%2) for i in range(4)]+[sample_row('v0','valid',0),sample_row('v1','valid',1)]
            states=[];streams=[]
            for head in ('linear','mlp4','prototype4'):
                core._seed_everything(42)
                model=core._build_model('codebert_source',str(base),device=torch.device('cpu'),lora_r=2,lora_alpha=4,lora_dropout=.05,target_modules=('attention.self.query',),gradient_checkpointing=False,fusion_dim=256,fusion_heads=8,classifier_head=head)
                states.append({k:v.clone() for k,v in model.encoder.state_dict().items()});streams.append(torch.random.get_rng_state())
                with patch.object(core.AutoTokenizer,'from_pretrained',return_value=CodeTokenizer()):
                    cp=core.train_model(None,root/(head+'.pt'),records=rows,variant='codebert_source',model_path=str(base),source_max_length=512,epochs=1,batch_size=2,gradient_accumulation=2,lora_r=2,lora_alpha=4,device='cpu',classifier_head=head)
                    scores=core.predict_checkpoint(root/(head+'.pt'),rows[-2:],device='cpu')
                    self.assertTrue(torch.isfinite(scores).all())
                    self.assertEqual(cp['classifier_head'],head)
            for other,stream in zip(states[1:],streams[1:]):
                for k,v in states[0].items():torch.testing.assert_close(v,other[k])
                self.assertTrue(torch.equal(stream,streams[0]))

if __name__=='__main__':unittest.main()
