from . import base_model
import torch.nn as nn
import torch

class MLP(base_model.BaseModel):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()

        self.m = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features = 250),
            nn.ReLU(),
            nn.Linear(in_features=250, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features= output_dim),
            nn.Sigmoid()
        )
        # self.activation = nn.Sigmoid()

    def forward(self, ba_output, ap_output, x):
        x = x.view(x.shape[0],-1)
        # print('x',x.shape)
        x = torch.squeeze(x)
        concated_encoded = torch.concat((ba_output, ap_output, x), dim=1) 
        output = self.m(concated_encoded)
        # print('output:',output.shape)
        # output = torch.sum(torch.squeeze(output),axis = 1)
        # print('output sum',output)
        # output = self.activation(output)
        output = torch.squeeze(output)
        return output
