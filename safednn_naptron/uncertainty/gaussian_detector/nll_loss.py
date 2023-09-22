"""
* Based on https://github.com/open-mmlab/mmdetection/tree/v2.23.0
* Apache License
* Copyright (c) 2018-2023 OpenMMLab
* Copyright (c) SafeDNN group 2023
"""
import mmcv
import torch
import torch.nn as nn
import numpy as np

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def nll_loss(pred, target, sigma):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.abs(-torch.log(_gaussian_dist_pdf(pred, target, sigma) + 1e-9) / 2.0)
    return loss


def _gaussian_dist_pdf(val, mean, var):
    return torch.exp(- (val - mean) ** 2.0 / var / 2.0) / torch.sqrt(2.0 * np.pi * var)



@LOSSES.register_module()
class NLLLoss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, **kwargs):
        super(NLLLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                sigma,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * nll_loss(
            pred,
            target,
            sigma=sigma,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_bbox