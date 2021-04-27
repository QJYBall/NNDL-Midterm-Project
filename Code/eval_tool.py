import torch.nn as nn
import torch


class Dice(nn.Module):
    """
    The Dice computed between probabilistic predictions and the ground truth.

    """
    def __init__(self, eps=1e-5, **kwargs):
        super(Dice, self).__init__()
        self.eps = eps
        self.kwargs = kwargs

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size(), "The prediction and ground truth must be of the same size!"
        n_dims = y_true.ndimension()
        numerator = 2 * torch.sum(y_true * y_pred, dim=[0,] + list(range(2, n_dims)))
        denominator = torch.sum(y_true ** 2 + y_pred ** 2, dim=[0,] + list(range(2, n_dims)))
        dice = torch.mean(numerator / (denominator + self.eps))

        return dice