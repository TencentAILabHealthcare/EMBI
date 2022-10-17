import torch
import torch.nn as nn
from transformers import BertModel

class EpitopeMHCBert(nn.Module):
    def __init__(self, EpitopeBert_dir, MHCBert_dir, emb_dim, dropout):
        super().__init__()
        self.EpitopeBert = BertModel.from_pretrained(EpitopeBert_dir)
        self.MHCBert = BertModel.from_pretrained(MHCBert_dir)
        if dropout == "":
            self.decoder = nn.Sequential(
                nn.Linear(in_features=emb_dim*2, out_features=emb_dim),
                nn.ReLU(),
                nn.Linear(in_features=emb_dim, out_features=1), 
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(in_features=emb_dim*2, out_features=emb_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=emb_dim, out_features=1), 
            )
        self.activation = nn.Sigmoid()


    
    def forward(self, ba_relu_output, ap_relu_output,epitope, MHC):
        epitope_encoded = self.EpitopeBert(**epitope).last_hidden_state
        MHC_encoded = self.MHCBert(**MHC).last_hidden_state
        # print('x_input_ids',x_input_ids.shape)
        # print('x_attention_mask',x_attention_mask.shape)

        epitope_cls = epitope_encoded[:, 0, :]
        MHC_cls = MHC_encoded[:, 0, :]   

        concated_encoded = torch.concat((ba_relu_output, ap_relu_output, epitope_cls, MHC_cls), dim=1)     
        # output = self.decoder(concated_encoded)
        # for i in range(len(self.decoder)):
        #     print('i',i)
        #     concated_encoded = self.decoder[i](concated_encoded)
        #     print('concated_encoded.shape',concated_encoded.shape)
        #     if i == 1:
        #         ReLU_output = concated_encoded
        output = self.decoder(concated_encoded)

        # output = torch.sum(torch.squeeze(output),axis=1)
        output = self.activation(output)
        output = torch.squeeze(output)
        # print('output_embedding:', output )

        return output


