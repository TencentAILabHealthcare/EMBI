import torch
import torch.nn as nn
from transformers import BertModel

class EpitopeBertMHC(nn.Module):
    def __init__(self, EpitopeBert_dir, emb_dim):
        super().__init__()
        self.EpitopeBert = BertModel.from_pretrained(EpitopeBert_dir)
        self.decoder = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=256), 
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
        )
        self.activation = nn.Sigmoid()


    
    def forward(self, epitope, MHC_encoding):

        epitope_encoded = self.EpitopeBert(**epitope).last_hidden_state

        epitope_cls = epitope_encoded[:, 0, :]
        concated_encoded = torch.concat((epitope_cls, MHC_encoding), dim=1)  
        # print("concated_encoded", concated_encoded.shape)   
        output = self.decoder(concated_encoded)
        # print('output_embedding shape:', output.shape)

        # output = torch.sum(torch.squeeze(output),axis=1)
        output = self.activation(output)
        output = torch.squeeze(output)
        # print('output_embedding:', output )

        return output


