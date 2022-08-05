
import torch.nn.functional as F

def CrossEntropyLoss(output, target):
    return F.cross_entropy(input=output, target=target)


def BCELoss(output, target):
    return F.binary_cross_entropy(input=output, target=target)
