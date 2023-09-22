"""
* Based on https://github.com/dimitymiller/openset_detection
* BSD 3-Clause License
* Copyright (c) 2021, Dimity Miller
* Based on https://github.com/open-mmlab/mmdetection/tree/v2.23.0
* Apache License
* Copyright (c) 2018-2023 OpenMMLab
* Copyright (c) SafeDNN group 2023
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from mmdet.models import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss


# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          distances,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None,
                          n_classes=20,
                          anchor_weight=0.1):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    if len(target.size()) > 1:
        target = torch.reshape(target, (-1, n_classes))
        target = torch.argmax(target, dim=1)
        target = target.long()

    # don't apply to background classes
    mask = target != n_classes
    if torch.sum(mask) != 0:
        label = target[mask]
        distances = distances[mask]

        loss_a = torch.gather(distances, 1, label.view(-1, 1)).view(-1)

        if weight != None:
            weight = weight.reshape(-1)[mask]
            loss_a *= weight

        if reduction == 'mean':
            avg_factor = torch.sum(mask)
            if avg_factor is not None:
                loss_a = loss_a.sum() / avg_factor
            else:
                loss_a = torch.mean(loss_a)
        else:
            loss_a = torch.sum(loss_a)
    else:
        loss_a = 0

    return loss + (anchor_weight * loss_a)


def py_focal_loss_with_prob(pred,
                            target,
                            distances,
                            weight=None,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            avg_factor=None,
                            n_classes=20,
                            anchor_weight=0.1):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.

    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    num_classes = pred.size(1)
    target = F.one_hot(target, num_classes=num_classes + 1)
    target = target[:, :num_classes]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    if len(target.size()) > 1:
        target = torch.reshape(target, (-1, num_classes))
        target = torch.argmax(target, dim=1)
        target = target.long()

    # don't apply to background classes
    mask = target != n_classes
    if torch.sum(mask) != 0:
        label = target[mask]
        distances = distances[mask]

        loss_a = torch.gather(distances, 1, label.view(-1, 1)).view(-1)

        if weight != None:
            weight = weight.reshape(-1)[mask]
            loss_a *= weight

        if reduction == 'mean':
            avg_factor = torch.sum(mask)
            if avg_factor is not None:
                loss_a = loss_a.sum() / avg_factor
            else:
                loss_a = torch.mean(loss_a)
        else:
            loss_a = torch.sum(loss_a)
    else:
        loss_a = 0

    return loss + (anchor_weight * loss_a)


def sigmoid_focal_loss(pred,
                       target,
                       distances,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None,
                       n_classes=20,
                       anchor_weight=0.1):
    r"""A warpper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                               alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

    if len(target.size()) > 1:
        target = torch.reshape(target, (-1, n_classes))
        target = torch.argmax(target, dim=1)
        target = target.long()

    # don't apply to background classes
    mask = target != n_classes
    if torch.sum(mask) != 0:
        label = target[mask]
        distances = distances[mask]

        loss_a = torch.gather(distances, 1, label.view(-1, 1)).view(-1)

        if weight != None:
            weight = weight.reshape(-1)[mask]
            loss_a *= weight

        if reduction == 'mean':
            avg_factor = torch.sum(mask)
            if avg_factor is not None:
                loss_a = loss_a.sum() / avg_factor
            else:
                loss_a = torch.mean(loss_a)
        else:
            loss_a = torch.sum(loss_a)
    else:
        loss_a = 0

    return loss + (anchor_weight * loss_a)


@LOSSES.register_module()
class AnchorFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 activated=False,
                 class_weight=None,
                 loss_weight=1.0, anchor_weight=0.1, n_classes=15, background_flag=False, *args, **kwargs):
        super(AnchorFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        self.n_classes = n_classes
        self.anchor_weight = anchor_weight
        self.background_flag = background_flag
        # plus one to account for background class
        anch = torch.diag(torch.ones(n_classes) * 5)
        anch = torch.where(anch != 0, anch, torch.Tensor([-5]))
        self.anchors = nn.Parameter(anch, requires_grad=False).cuda()

    def euclideanDistance(self, logits):
        # plus one to account for background clss logit
        logits = logits.view(-1, self.n_classes)
        n = logits.size(0)
        m = self.anchors.size(0)
        d = logits.size(1)

        x = logits.unsqueeze(1).expand(n, m, d)
        anchors = self.anchors.unsqueeze(0).expand(n, m, d)

        dists = torch.norm(x - anchors, 2, 2)

        return dists

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                if torch.cuda.is_available() and pred.is_cuda:
                    calculate_loss_func = sigmoid_focal_loss
                else:
                    num_classes = pred.size(1)
                    target = F.one_hot(target, num_classes=num_classes + 1)
                    target = target[:, :num_classes]
                    calculate_loss_func = py_sigmoid_focal_loss

            distances = self.euclideanDistance(pred)


            pred = pred.reshape(-1, pred.shape[-1])
            target = target.flatten().long()

            loss_cls = self.loss_weight * calculate_loss_func(pred,
                                                              target,
                                                              distances,
                                                              weight=weight,
                                                              n_classes=self.n_classes if not self.background_flag else self.n_classes - 1,
                                                              anchor_weight=self.anchor_weight,
                                                              reduction=reduction,
                                                              avg_factor=avg_factor,
                                                              gamma=self.gamma,
                                                              alpha=self.alpha)

        else:
            raise NotImplementedError
        return loss_cls
