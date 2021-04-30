import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class Dice(nn.Module):

    def __init__(self, eps=1e-5, **kwargs):
        super(Dice, self).__init__()
        self.eps = eps
        self.kwargs = kwargs

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = F.one_hot(y_pred, num_classes=21)
        y_true = F.one_hot(y_true, num_classes=21)

        n_dims = y_true.ndimension()
        numerator = 2 * torch.sum(y_true * y_pred, dim=list(range(0, n_dims - 1)))
        denominator = torch.sum(y_true**2 + y_pred**2, dim=list(range(0, n_dims - 1)))
        dice = torch.mean((numerator + self.eps) / (denominator + self.eps))

        return dice


def _fast_hist(label, pred, n_class):
    mask = (label >= 0) & (label < n_class)
    hist = np.bincount(
        n_class * label[mask].astype(int) +
        pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def eval_score(labels, preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(labels, preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    
    # pixel accuracy
    PA = np.diag(hist).sum() / hist.sum()

    # mean pixel accuracy
    MPA = np.diag(hist) / hist.sum(axis=1)
    MPA = np.nanmean(MPA)

    # mean intersection over union
    IoU = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    MIoU = np.nanmean(IoU)

    # frequency weighted intersection over union
    freq = hist.sum(axis=1) / hist.sum()
    FWIoU = (freq[freq > 0] * IoU[freq > 0]).sum()
    
    return PA, MPA, MIoU, FWIoU
