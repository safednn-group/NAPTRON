"""
* Based on https://github.com/open-mmlab/mmdetection/tree/v2.23.0
* Apache License
* Copyright (c) 2018-2023 OpenMMLab
* Copyright (c) SafeDNN group 2023
"""
import torch

from mmdet.utils import util_mixins
from mmdet.core.bbox.samplers import SamplingResult

class OWODSamplingResult(SamplingResult):
    """Bbox sampling result.

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
        >>> self = SamplingResult.random(rng=10)
        >>> print(f'self = {self}')
        self = <SamplingResult({
            'neg_bboxes': torch.Size([12, 4]),
            'neg_inds': tensor([ 0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
            'num_gts': 4,
            'pos_assigned_gt_inds': tensor([], dtype=torch.int64),
            'pos_bboxes': torch.Size([0, 4]),
            'pos_inds': tensor([], dtype=torch.int64),
            'pos_is_gt': tensor([], dtype=torch.uint8)
        })>
    """

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags, num_classes):
        super().__init__(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)
        self.neg_inds = neg_inds
        self.neg_bboxes = bboxes[neg_inds]
        neg_scores = self.neg_bboxes[:, 4]
        max_ind = neg_inds[neg_scores.argmax()].item()
        bboxes = bboxes[:, :4]
        gt_bboxes = gt_bboxes[:, :4]

        neg_inds = neg_inds.tolist()
        neg_inds.remove(max_ind)
        self.neg_inds = torch.tensor(neg_inds)
        self.neg_bboxes = bboxes[self.neg_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)

            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds.long(), :]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

        pos_inds = pos_inds.tolist() + [max_ind]
        self.pos_inds = torch.tensor(pos_inds)
        self.pos_is_gt = gt_flags[self.pos_inds]
        self.pos_bboxes = bboxes[self.pos_inds]
        self.pos_gt_labels = torch.cat((self.pos_gt_labels, torch.tensor([num_classes - 1], device=self.pos_gt_labels.device)))
        self.pos_gt_bboxes = torch.cat((self.pos_gt_bboxes, bboxes[max_ind].unsqueeze(0)), dim=0)
