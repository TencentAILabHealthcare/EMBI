from pickle import TRUE
from turtle import forward
from matplotlib.pyplot import axis
from transformers import ESMForMaskedLM
from tokenize import Token
import torch
import torch.nn as nn
from . import base_model
import numpy as np

class ESM2Decoder(base_model.BaseModel):
    def __init__(self, ntokens=33):
        super().__init__()
        self.esm_encoder = ESMForMaskedLM.from_pretrained("facebook/esm-1b", local_files_only=True)
        esm_1b_dim = 1280
        # self.decoder = nn.Sequential(
        #     nn.Linear(in_features=tcrbert_emb_dim, out_features=tcrbert_emb_dim),
        #     nn.LayerNorm(normalized_shape=tcrbert_emb_dim),
        #     nn.GELU(),
        #     nn.Linear(in_features=tcrbert_emb_dim, out_features=tcrbert_emb_dim),
        #     nn.LayerNorm(normalized_shape=tcrbert_emb_dim),
        #     nn.GELU(),
        #     nn.Linear(in_features=tcrbert_emb_dim, out_features=tcrbert_emb_dim)            
        # )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=esm_1b_dim, out_features=int(esm_1b_dim /2)),
            nn.ReLU(),
            nn.Linear(in_features=int(esm_1b_dim /2), out_features=int(esm_1b_dim /4)),
            nn.ReLU(),
            nn.Linear(in_features=int(esm_1b_dim /4), out_features=1), 
        )
        self.activation = nn.Sigmoid()


    
    def forward(self, x_input_ids, x_attention_mask):
        x_input_ids = torch.squeeze(x_input_ids)
        x_attention_mask = torch.squeeze(x_attention_mask)
        # print('x_input_ids',x_input_ids.shape)
        # print('x_attention_mask',x_attention_mask.shape)

        encoder_embedding = self.esm_encoder.base_model(input_ids = x_input_ids, attention_mask = x_attention_mask).last_hidden_state
        
        output = self.decoder(encoder_embedding)

        # print('output_embedding shape:', output.shape)

        # print('output_embedding:', output )
        output = torch.sum(torch.squeeze(output),axis=1)
        output = self.activation(output)
        # output = torch.tensor(np.round_(output.cpu().detach().numpy())).to('cuda:0')
        # output = self.esm_encoder(input_ids = x_input_ids, attention_mask = x_attention_mask)
        # print(output.keys())
        # output = self.decoder(output)
        # print('output',output)
        return output


