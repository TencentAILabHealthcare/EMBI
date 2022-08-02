import torch.nn as nn
import numpy as np
from abc import abstractmethod

class BaseModel(nn.Module):
    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError
    
    def __str__(self) -> str:
        params = sum([np.prod(p.size()) for p in self.parameters() if p.requires_grad])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

        