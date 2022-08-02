from turtle import forward
from transformers import ESMForMaskedLM
from tokenize import Token
import torch
import torch.nn as nn
import base_model

class ESM2Decoder(base_model.BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.esm_encoder = ESMForMaskedLM.from_pretrained("facebook/esm-1b", local_files_only=True)
        # tcrbert_emb_dim = 768
        # self.decoder = nn.Sequential(
        #     nn.Linear(in_features=tcrbert_emb_dim, out_features=tcrbert_emb_dim),
        #     nn.LayerNorm(normalized_shape=tcrbert_emb_dim),
        #     nn.GELU(),
        #     nn.Linear(in_features=tcrbert_emb_dim, out_features=tcrbert_emb_dim),
        #     nn.LayerNorm(normalized_shape=tcrbert_emb_dim),
        #     nn.GELU(),
        #     nn.Linear(in_features=tcrbert_emb_dim, out_features=tcrbert_emb_dim)            
        # )
    
    def forward(self, x_input_ids, x_attention_mask):
        x_input_ids = torch.squeeze(x_input_ids)
        x_attention_mask = torch.squeeze(x_attention_mask)

        output = self.esm_encoder(input_ids = x_input_ids, attention_mask = x_attention_mask).logits

        return output