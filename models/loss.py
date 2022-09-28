
import torch.nn.functional as F
import torch

def CrossEntropyLoss(output, target):
    return F.cross_entropy(input=output, target=target)


def BCELoss(output, target):
    return F.binary_cross_entropy(input=output, target=target)

# def BCELoss_weighted(output, target):
#     input = torch.clamp(output, min=1e-7, max=1-1e-7)
#     bce = - 0.8 * target * torch.log(input) - 0.2 * torch.log(1 - input)
#     return bce

def BCELoss_weighted(output, target, class_weights=[1,6]):
    weight = torch.zeros_like(target)
    weight[target==0] = class_weights[0]
    weight[target==1] = class_weights[1]
    loss = F.binary_cross_entropy(input=output, target=target, weight=weight)
    return loss

