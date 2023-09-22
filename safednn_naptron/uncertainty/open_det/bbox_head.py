"""
* Based on https://github.com/open-mmlab/mmdetection/tree/v2.23.0 and https://github.com/csuhan/opendet2
* Apache License
* Copyright (c) 2018-2023 OpenMMLab
* Copyright (c) SafeDNN group 2023
"""
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import ConvFCBBoxHead
from mmdet.core import multiclass_nms
from mmcv.runner import force_fp32

import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init


@HEADS.register_module()
class ODBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, scale=20, ic_loss_out_dim=128, *args, **kwargs):
        super(ODBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=0,
            num_cls_convs=0,
            num_cls_fcs=2,
            num_reg_convs=0,
            num_reg_fcs=2,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        self.scale = scale
        self.encoder = MLP(self.fc_cls.in_features, ic_loss_out_dim)


    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
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

        x_norm = torch.norm(x_cls, p=2, dim=1).unsqueeze(1).expand_as(x_cls)
        x_normalized = x_cls.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = (
            torch.norm(self.fc_cls.weight.data, p=2, dim=1)
                .unsqueeze(1)
                .expand_as(self.fc_cls.weight.data)
        )

        self.fc_cls.weight.data = self.fc_cls.weight.data.div(
            temp_norm + 1e-5
        )
        cos_dist = self.fc_cls(x_normalized)
        scores = self.scale * cos_dist
        # encode feature with MLP
        mlp_feat = self.encoder(x_cls)

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return scores, bbox_pred, mlp_feat

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

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if not hidden_dim:
            hidden_dim = in_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal(layer.weight)

    def forward(self, x):
        feat = self.head(x)
        feat_norm = F.normalize(feat, dim=1)
        return feat_norm
