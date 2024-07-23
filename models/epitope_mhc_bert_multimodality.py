import torch
import torch.nn as nn
# from transformers import BertModel
from .BertModel_Noise import BertModelNoise



class EpitopeMHCBert(nn.Module):
    def __init__(self, EpitopeBert_dir, MHCBert_dir, emb_dim, dropout, Add_noise_sep = 5, aug_N = 2, aug_M = 10, kernel_set = 0):
        super().__init__()
        self.EpitopeBert = BertModelNoise.from_pretrained(EpitopeBert_dir)
        self.MHCBert = BertModelNoise.from_pretrained(MHCBert_dir)
        if dropout == "":
            self.decoder = nn.Sequential(
                nn.Linear(in_features=emb_dim*2, out_features=emb_dim),
                nn.ReLU(),
                nn.Linear(in_features=emb_dim, out_features=1)
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(in_features=emb_dim*2, out_features=emb_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=emb_dim, out_features=1), 

            )
        self.activation = nn.Sigmoid()
        self.Add_noise_sep = Add_noise_sep
        self.aug_N = aug_N
        self.aug_M = aug_M
        self.kernel_set = kernel_set
        
    def forward(self, ba_relu_output, ap_relu_output,epitope, MHC, batch_idx = 1, transform_proportion = 0.0):
        noise_add_position = "BeforeBert" if batch_idx % self.Add_noise_sep == 0 else None
        epitope_encoded = self.EpitopeBert(**epitope,noise_add_position = noise_add_position, noise_add_params = [self.aug_N,self.aug_M],transform_proportion=transform_proportion).last_hidden_state
        MHC_encoded = self.MHCBert(**MHC,noise_add_position = noise_add_position, noise_add_params = [self.aug_N,self.aug_M],transform_proportion=transform_proportion).last_hidden_state
        # print('x_input_ids',x_input_ids.shape)
        # print('x_attention_mask',x_attention_mask.shape)

        epitope_cls = epitope_encoded[:, 0, :]
        MHC_cls = MHC_encoded[:, 0, :]   

        concated_encoded = torch.concat((ba_relu_output, ap_relu_output, epitope_cls, MHC_cls), dim=1) 
        output = self.decoder(concated_encoded)

        # output = torch.sum(torch.squeeze(output),axis=1)
        output = self.activation(output)
        output = torch.squeeze(output)
        # print('output_embedding:', output )

        return output


