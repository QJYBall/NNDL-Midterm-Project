import torch.nn as nn
import torch
import numpy as np


class Dice(nn.Module):

    def __init__(self, eps=1e-5, **kwargs):
        super(Dice, self).__init__()
        self.eps = eps
        self.kwargs = kwargs

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        n_dims = y_true.ndimension()
        numerator = 2 * torch.sum(y_true * y_pred, dim=[0,] + list(range(2, n_dims)))
        denominator = torch.sum(y_true ** 2 + y_pred ** 2, dim=[0,] + list(range(2, n_dims)))
        dice = torch.mean(numerator / (denominator + self.eps))

        return dice
