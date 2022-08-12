from turtle import forward
from transformers import BertModel
from tokenize import Token
import torch
import torch.nn as nn
from . import base_model

class TCRBert2Decoder(base_model.BaseModel):
    def __init__(self,ntokens=26) -> None:
        super().__init__()
        self.tcrbert_encoder = BertModel.from_pretrained("wukevin/tcr-bert", local_files_only=True)
        tcrbert_emb_dim = 768
        self.decoder = nn.Sequential(
            nn.Linear(in_features=tcrbert_emb_dim, out_features=tcrbert_emb_dim),
            nn.LayerNorm(normalized_shape=tcrbert_emb_dim),
            nn.GELU(),
            nn.Linear(in_features=tcrbert_emb_dim, out_features=tcrbert_emb_dim),
            nn.LayerNorm(normalized_shape=tcrbert_emb_dim),
            nn.GELU(),
            nn.Linear(in_features=tcrbert_emb_dim, out_features=1)            
        )
        self.activation = nn.Sigmoid()
    
    def forward(self, x_input_ids, x_attention_mask):
        x_input_ids = torch.squeeze(x_input_ids)
        x_attention_mask = torch.squeeze(x_attention_mask)
        print('x_input_ids',x_input_ids.shape)
        print('x_attention_mask',x_attention_mask.shape)
        encoder_embedding = self.tcrbert_encoder.base_model(input_ids = x_input_ids, attention_mask = x_attention_mask).last_hidden_state
            
        output = self.decoder(encoder_embedding)

        output = torch.sum(torch.squeeze(output),axis=1)
        output = self.activation(output)

        return output