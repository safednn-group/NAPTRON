"""
* Based on https://github.com/open-mmlab/mmdetection/tree/v2.23.0
* Apache License
* Copyright (c) 2018-2023 OpenMMLab
* Copyright (c) SafeDNN group 2023
"""
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import build_linear_layer
from mmdet.models.roi_heads import Shared2FCBBoxHead
from mmdet.core import multiclass_nms
from mmcv.runner import force_fp32
from mmdet.models.losses import accuracy

import torch.nn.functional as F
import torch


@HEADS.register_module()
class GaussianBBoxHead(Shared2FCBBoxHead):

    def __init__(self, *args, **kwargs):
        super(GaussianBBoxHead, self).__init__(
            *args,
            **kwargs)

        if self.with_reg:
            self.out_dim_reg = 8 if self.reg_class_agnostic else 8 * self.num_classes
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=self.out_dim_reg)

    def forward(self, x):
        # shared part

        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)  # (256, 7, 7) -> 12544

            for fc in self.shared_fcs:
                x = self.relu(fc(x))  # 12544 -> 1024

        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)

        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)

            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)

        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        bbox_pred, sigma = bbox_pred[..., :int(self.out_dim_reg/2)], bbox_pred[..., int(self.out_dim_reg/2):]
        return cls_score, bbox_pred, torch.sigmoid(sigma)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None,
                   ret_keep_ids=False):
        """Transform network output for a batch into bbox predictions.
        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None
            ret_keep_ids (bool): If True, return ids of class scores to be kept. ***SAFEDNN modification***
        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels, keep = multiclass_nms(bboxes, scores,
                                                          cfg.score_thr, cfg.nms,
                                                          cfg.max_per_img, return_inds=True)
            if ret_keep_ids:
                num_classes = scores.size(1) - 1
                keep = (keep / num_classes).long()
                return det_bboxes, det_labels, keep
            else:
                return det_bboxes, det_labels

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'sigma'))
    def loss(self,
             cls_score,
             bbox_pred,
             sigma,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                    pos_bbox_sigma = sigma.view(
                        sigma.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                    pos_bbox_sigma = sigma.view(
                        sigma.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]

                loss_xy = self.loss_bbox(
                    pos_bbox_pred[..., :2],
                    bbox_targets[pos_inds.type(torch.bool), ..., :2],
                    pos_bbox_sigma[..., :2],
                    bbox_weights[pos_inds.type(torch.bool), :2],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
                loss_wh = self.loss_bbox(
                    pos_bbox_pred[..., 2:4],
                    bbox_targets[pos_inds.type(torch.bool), ..., 2:4],
                    pos_bbox_sigma[..., 2:4],
                    bbox_weights[pos_inds.type(torch.bool), 2:4],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
                losses['loss_bbox'] = loss_xy + loss_wh

            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses