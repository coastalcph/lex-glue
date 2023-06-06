from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import numpy as np
from torch import nn
from transformers.file_utils import ModelOutput


@dataclass
class SimpleOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    logits:torch.FloatTensor = None




class HierarchicalConvolutionalBert(nn.Module):
    def __init__(self, encoder, max_segments=[64, 32, 16], max_segment_length=[128, 256, 512],num_labels=None):
        super(HierarchicalConvolutionalBert, self).__init__()

        supported_models = ['bert']
        assert encoder.config.model_type in supported_models 

        self.encoder = encoder

        self.hidden_size = encoder.config.hidden_size
        self.max_segments = max_segments
        self.max_segment_length = max_segment_length
        self.num_labels=num_labels
        self.seg_encoder = nn.Transformer(d_model=encoder.config.hidden_size,
                                           nhead=encoder.config.num_attention_heads,
                                           batch_first=True, dim_feedforward=encoder.config.intermediate_size,
                                           activation=encoder.config.hidden_act,
                                           dropout=encoder.config.hidden_dropout_prob,
                                           layer_norm_eps=encoder.config.layer_norm_eps,
                                           num_encoder_layers=2, num_decoder_layers=0).encoder
        self.fc = nn.Linear(encoder.config.hidden_size, self.num_labels)

    def forward(self, input_ids_1=None, attention_mask_1=None, token_type_ids_1=None, 
                input_ids_2=None, attention_mask_2=None, token_type_ids_2=None,
                input_ids_3=None, attention_mask_3=None, token_type_ids_3=None,labels=None,**kwargs):
        output_list = []
        for i in range(1,4):
            input_ids_i = eval("input_ids_" + str(i))
            #batch=input_ids_i.size(0)
            attention_mask_i = eval("attention_mask_" + str(i))
            token_type_ids_i = eval("token_type_ids_" + str(i))          
            input_ids_i = input_ids_i.contiguous().view(-1, input_ids_i.size(-1))
            attention_mask_i = attention_mask_i.contiguous().view(-1, attention_mask_i.size(-1))
            token_type_ids_i = token_type_ids_i.contiguous().view(-1, token_type_ids_i.size(-1))

            encoder_outputs = self.encoder(input_ids=input_ids_i,
                                           attention_mask=attention_mask_i,
                                           token_type_ids=token_type_ids_i)[0]

            encoder_outputs = encoder_outputs.contiguous().view(-1, self.max_segments[i-1],
                                                                self.max_segment_length[i-1],
                                                                self.hidden_size)

            encoder_outputs = encoder_outputs[:, :, 0]

            seg_encoder_outputs = self.seg_encoder(encoder_outputs)

            outputs, _ = torch.max(seg_encoder_outputs, 1)
            output_list.append(outputs)

        output = torch.mean(torch.stack(output_list), dim=0)
        logits = self.fc(output)
        return SimpleOutput(logits=logits, last_hidden_state=logits)


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Use as a stand-alone encoder
    bert = AutoModel.from_pretrained('bert-base-uncased')
    model = HierarchicalConvolutionalBert(encoder=bert,num_labels=10)

    fake_inputs = {'input_ids_1': [], 'attention_mask_1': [], 'token_type_ids_1': [],
                   'input_ids_2': [], 'attention_mask_2': [], 'token_type_ids_2': [],
                   'input_ids_3': [], 'attention_mask_3': [], 'token_type_ids_3': []}
    for i in range(4):
        #batch=4
        # Tokenize segment

        temp_inputs = tokenizer(['dog ' * 126] * 64)
        fake_inputs['input_ids_1'].append(temp_inputs['input_ids'])
        fake_inputs['attention_mask_1'].append(temp_inputs['attention_mask'])
        fake_inputs['token_type_ids_1'].append(temp_inputs['token_type_ids'])
        temp_inputs = tokenizer(['cat ' * 254] * 32)
        fake_inputs['input_ids_2'].append(temp_inputs['input_ids'])
        fake_inputs['attention_mask_2'].append(temp_inputs['attention_mask'])
        fake_inputs['token_type_ids_2'].append(temp_inputs['token_type_ids'])
        temp_inputs = tokenizer(['snake ' * 510] * 16)
        fake_inputs['input_ids_3'].append(temp_inputs['input_ids'])
        fake_inputs['attention_mask_3'].append(temp_inputs['attention_mask'])
        fake_inputs['token_type_ids_3'].append(temp_inputs['token_type_ids'])        

    for key in fake_inputs:
            fake_inputs[key] = torch.as_tensor(fake_inputs[key])
    input=fake_inputs['input_ids_3']
    print(input.size(0))
    print(len(fake_inputs['input_ids_3']))
    output = model(input_ids_1=fake_inputs['input_ids_1'], attention_mask_1=fake_inputs['attention_mask_1'], token_type_ids_1=fake_inputs['token_type_ids_1'],
                   input_ids_2=fake_inputs['input_ids_2'], attention_mask_2=fake_inputs['attention_mask_2'], token_type_ids_2=fake_inputs['token_type_ids_2'],
                   input_ids_3=fake_inputs['input_ids_3'], attention_mask_3=fake_inputs['attention_mask_3'], token_type_ids_3=fake_inputs['token_type_ids_3'])

    # 4 document representations of 768 features are expected
    assert output[0].shape == torch.Size([4, 10])





'''  # Use with HuggingFace AutoModelForSequenceClassification and Trainer API

    # Init Classifier
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)
    # Replace flat BERT encoder with hierarchical BERT encoder
    model.bert = HierarchicalConvolutionalBert(encoder=model.bert)
    output = model(input_ids_1=fake_inputs['input_ids_1'], attention_mask_1=fake_inputs['attention_mask_1'], token_type_ids_1=fake_inputs['token_type_ids_1'],
                   input_ids_2=fake_inputs['input_ids_2'], attention_mask_2=fake_inputs['attention_mask_2'], token_type_ids_2=fake_inputs['token_type_ids_2'],
                   input_ids_3=fake_inputs['input_ids_3'], attention_mask_3=fake_inputs['attention_mask_3'], token_type_ids_3=fake_inputs['token_type_ids_3'])
    # 4 document outputs with 10 (num_labels) logits are expected
    assert output.logits.shape == torch.Size([4, 10])
'''
