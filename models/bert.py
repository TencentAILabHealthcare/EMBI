
from asyncio import base_tasks
import math
from turtle import forward, pos, position
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .base_model import BaseModel


class PositionalEmbedding(nn.Module):

    def __init__(self,d_model, dropout=0.1, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)       

    def forward(self, x):
        batch_size = x.size(0)
        position = self.pe.repeat(batch_size, 1, 1)
        return position

class BERT(BaseModel):
    def __init__(self, ntoken, hidden, heads, n_layers, n_blocks, max_len, dropout):
        super().__init__()

        # embeddings
        self.token_embedding = nn.Embedding(ntoken, hidden)
        self.positional_embedding = PositionalEmbedding(d_model=hidden, dropout=dropout, max_len=max_len)

        encoder_layer = TransformerEncoderLayer(d_model=hidden, 
                                                nhead=heads,
                                                dim_feedforward=hidden*4,
                                                dropout=dropout,
                                                activation="gelu")
        self.bert = nn.ModuleList([TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)
            for i in range(n_blocks)])

        # Decoder
        self.decoder_MHC = nn.Linear(hidden, ntoken)
        # self.decoder_chain2 = nn.Linear(hidden, ntoken)
        # self.decoder_activation = nn.Softmax(dim=1)
    def forward(self, input):

        tokens_embedding = self.token_embedding(input)
        tokens_position = self.positional_embedding(input)

        embedding = tokens_embedding + tokens_position
        embedding = torch.transpose(embedding, 0, 1)

        x = embedding
        for te in self.bert:
            x = te(x)
        
        x = torch.transpose(x, 0, 1)

        MHC_predict = self.decoder_MHC(x)
        output = MHC_predict

        return output
