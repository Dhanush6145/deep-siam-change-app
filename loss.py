import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def forward(self, pred, target):
        smooth = 1
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

bce = nn.BCELoss()
dice = DiceLoss()

def hybrid_loss(pred, target):
    return bce(pred, target) + dice(pred, target)
