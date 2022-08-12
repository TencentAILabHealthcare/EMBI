
from turtle import forward
import torch
import torch.nn as nn

class DSSM(torch.nn.Module):
    def __init__(self, input_dim, output_dim,temperature=1.0,sim_func="cosine") -> None:
        super().__init__()
        # self.epitope_features = epitope_features
        # self.MHC_features = MHC_features
        # self.sim_func = sim_func

        self.temperature = temperature
        # self.user_dims = sum([fea.embed_dim for fea in user_features])
        # self.embedding = EmbeddingLayer(epitope_features + MHC_features)
        # self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)
        # self.mode = None
        
        self.m1 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features = 250),
            nn.ReLU(),
            nn.Linear(in_features=250, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features= output_dim),
            nn.ReLU()
        )
        self.m2 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features = 250),
            nn.ReLU(),
            nn.Linear(in_features=250, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features= output_dim),
            nn.ReLU()
        )
        self.relu = nn.ReLU()
    
    def forward(self,epitope_features, MHC_features):
        epitope_embedding = self.m1(epitope_features)
        #shape batch * 34 * 10
        # print('shape of epitope_embedding',epitope_embedding.shape)
        MHC_embedding = self.m2(MHC_features)
        # print('shape of MHC_embedding',MHC_embedding.shape)
        cosine = torch.cosine_similarity(epitope_embedding, MHC_embedding, dim=2, eps=1e-8)
        cosine_sum = torch.sum(cosine, axis = 1)
        # cosine = self.relu(cosine)
        # print('1 cosine relu', cosine)
        # cosine = torch.clamp(cosine, 0, 1)
        output = torch.sigmoid(cosine_sum)
        # print('cosine after cosine', cosine.shape)
        # output = torch.squeeze(cosine)
        # print('output:',output)
        return output     


