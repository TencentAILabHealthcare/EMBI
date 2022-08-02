from turtle import forward
from transformers import BertModel
from tokenize import Token
import torch
import torch.nn as nn
import base_model

class TCRBert2Decoder(base_model.BaseModel):
    def __init__(self) -> None:
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
            nn.Linear(in_features=tcrbert_emb_dim, out_features=tcrbert_emb_dim)            
        )
    
    def forward(self, ):
        pass
