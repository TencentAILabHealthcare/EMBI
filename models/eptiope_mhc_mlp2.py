import torch
import torch.nn as nn

class EpitopeMHCMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=4),
            nn.ReLU(),
            nn.Linear(in_features=4, out_features=2),
            nn.ReLU(),
            nn.Linear(in_features=2, out_features=output_dim),
            nn.Sigmoid()
        )


    
    def forward(self, p1, p2):
        # print('shape of p1',p1.shape)
        # print('shape of p2',p1.shape)

        concated_encoded = torch.concat((p1.view(-1,1), p2.view(-1,1)), dim=1)
        # print('shape of concated', concated_encoded.shape)  
        output = self.mlp(concated_encoded)
        # print('shape of output', output.shape)

        output = torch.squeeze(output)
        # print('shape of output', output.shape)

        return output


