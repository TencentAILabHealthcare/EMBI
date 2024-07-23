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
        self.Add_noise_sep = Add_noise_sep
        self.aug_N = aug_N
        self.aug_M = aug_M
        self.kernel_set = kernel_set
        # print("=========Initial EpitopeMHCBert ks={}=========".format(self.kernel_set))


    
    def forward(self, epitope, MHC, batch_idx = 1):
        
        noise_add_position = "BeforeBert" if batch_idx % self.Add_noise_sep == 0 else None
        # epitope_encoded = self.EpitopeBert(**epitope,noise_add_position = noise_add_position, noise_add_params = [self.aug_N,self.aug_M]).last_hidden_state
        # MHC_encoded = self.MHCBert(**MHC,noise_add_position = noise_add_position, noise_add_params = [self.aug_N,self.aug_M]).last_hidden_state
        
        epitope_encoded = self.EpitopeBert(**epitope,noise_add_position = noise_add_position, kernel_set=self.kernel_set).last_hidden_state
        MHC_encoded = self.MHCBert(**MHC,noise_add_position = noise_add_position, kernel_set=self.kernel_set).last_hidden_state

        epitope_cls = epitope_encoded[:, 0, :]
        MHC_cls = MHC_encoded[:, 0, :]   
        concated_encoded = torch.concat((epitope_cls, MHC_cls), dim=1) 
        concated_encoded_return =  concated_encoded
        # output = self.decoder(concated_encoded)
        # To save the output of ReLU Layer, run following loops
        for i in range(len(self.decoder)):
            # print('i',i)
            concated_encoded = self.decoder[i](concated_encoded)
            # print('concated_encoded.shape',concated_encoded.shape)
            if i == 1:
                ReLU_output = concated_encoded
        output = concated_encoded

        # output = torch.sum(torch.squeeze(output),axis=1)
        output = self.activation(output)
        output = torch.squeeze(output)
        # print('output_embedding:', output )
        # print('concated_encoded shape',concated_encoded_return.shape)   

        return output, ReLU_output #, concated_encoded_return


