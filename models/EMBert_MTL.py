import torch
import torch.nn as nn
from transformers import BertModel

class EpitopeMHCBertMTL(nn.Module):
    def __init__(self, EpitopeBert_dir, MHCBert_dir, emb_dim, dropout):
        super().__init__()
        self.EpitopeBert = BertModel.from_pretrained(EpitopeBert_dir)
        self.MHCBert = BertModel.from_pretrained(MHCBert_dir)
        if dropout == "":
            self.sharedlayer = nn.Sequential(
                nn.Linear(in_features=emb_dim*2, out_features=emb_dim),
                nn.ReLU(),
                nn.Linear(in_features=emb_dim, out_features=int(emb_dim/2)), 
                nn.ReLU(),
                nn.Linear(in_features=int(emb_dim/2), out_features=int(emb_dim/4))
            )
        else:
            self.sharedlayer = nn.Sequential(
                nn.Linear(in_features=emb_dim*2, out_features=emb_dim),
                nn.ReLU(),
                nn.Linear(in_features=emb_dim, out_features=int(emb_dim/2)), 
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=int(emb_dim/2), out_features=int(emb_dim/4))
            )
        self.immulayer = nn.Sequential(
            nn.Linear(in_features=int(emb_dim/4), out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        self.BAlayer = nn.Sequential(
            nn.Linear(in_features=int(emb_dim/4), out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        self.APlayer = nn.Sequential(
            nn.Linear(in_features=int(emb_dim/4), out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )

        self.activation = nn.Sigmoid()


    
    def forward(self, epitope, MHC):
        epitope_encoded = self.EpitopeBert(**epitope).last_hidden_state
        MHC_encoded = self.MHCBert(**MHC).last_hidden_state
        # print('x_input_ids',x_input_ids.shape)
        # print('x_attention_mask',x_attention_mask.shape)

        epitope_cls = epitope_encoded[:, 0, :]
        MHC_cls = MHC_encoded[:, 0, :]   
        concated_encoded = torch.concat((epitope_cls, MHC_cls), dim=1)     
        shared_output = self.sharedlayer(concated_encoded)

        
        immu_output = torch.squeeze(self.activation(self.immulayer(shared_output)))
        BA_output = torch.squeeze(self.activation(self.BAlayer(shared_output)))
        AP_output = torch.squeeze(self.activation(self.APlayer(shared_output)))

        return immu_output, BA_output, AP_output


