import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import device

def dice_loss(pred, target):
    smooth = 1e-5

    channels = target.size(1)

    #y = pred.view(pred.size(0), -1)
    y = F.softmax(pred, dim=1)
    #y = y.view(y.size(0), channels, pred.size(-2), pred.size(-1))

    #channel wise loss

    # flatten predictions and targets
    loss = torch.tensor(0.).to(device)
    for i in range(channels):
        yi = y[:, i, :, :]
        ti = target[:, i, :, :]

        yi = yi.view(y.size(0), -1)
        ti = ti.view(target.size(0), -1)

        intersection = (yi * ti).sum()
        union = yi.sum() + ti.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        loss += dice

    return 1 - (loss / channels)

def bce_loss(pred, target):
    #y = pred.view(pred.size(0), -1)
    y = F.softmax(pred, dim=1)

    #y = y.view(y.size(0), -1)
    #target = target.view(target.size(0), -1)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(y, target)

    return loss