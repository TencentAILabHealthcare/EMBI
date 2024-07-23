
import torch.nn.functional as F
import torch

def CrossEntropyLoss(output, target):
    return F.cross_entropy(input=output, target=target)


def BCELoss(output, target):
    # return F.binary_cross_entropy(input=output, target=target)
    if output.size() != target.size():
        output = output.unsqueeze(0)
    return F.binary_cross_entropy(input=output, target=target)


def BCELoss_weighted(output, target, class_weights=[1,5]):
    weight = torch.zeros_like(target)
    # print('class weight',class_weights)
    weight[target==0] = class_weights[0]
    weight[target==1] = class_weights[1]
    if output.size() != target.size():
        output = output.unsqueeze(0)
    loss = F.binary_cross_entropy(input=output, target=target, weight=weight)
    return loss

