import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.basic_loss import L1Loss, l1_loss


@LOSS_REGISTRY.register()
class CycleLoss(L1Loss):
    """
    MultiScaleGANLoss accepts a list of predictions
    """

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)