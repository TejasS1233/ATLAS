import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        inter = (pred * target).sum()
        dice = (2.*inter + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCELoss()
        self.alpha, self.gamma, self.dice_weight = alpha, gamma, dice_weight
    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='mean')
        # focal weighting
        pt = torch.exp(-bce)
        focal = self.alpha * (1-pt)**self.gamma * bce
        return self.dice_weight * dice_loss + (1-self.dice_weight)*focal

def dice_coef(pred, target, smooth=1e-6, thresh=0.5):
    pred = (pred > thresh).float().view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    return (2*inter + smooth)/(pred.sum() + target.sum() + smooth)

def iou_score(pred, target, thresh=0.5, smooth=1e-6):
    pred = (pred > thresh).float().view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + smooth)/(union + smooth)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
