
import torch.nn.functional as F

def CrossEntropyLoss(output, target):
    return F.cross_entropy(input=output, target=target, ignore_index=21)