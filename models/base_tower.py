import torch.nn as nn
import torch

class BaseTower(nn.Module):
    def __init__(self, epitope, MHC) -> None:
        super().__init__()

        # self.reg_loss = torch.zeros((1,))

