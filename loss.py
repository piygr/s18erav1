import torch.nn as nn

def dice_loss(pred, target):
    smooth = 1e-5

    # flatten predictions and targets
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice

def bce_loss(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(pred, target)
    return loss