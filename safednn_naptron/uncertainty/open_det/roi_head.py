"""
* Based on https://github.com/open-mmlab/mmdetection/tree/v2.23.0 and https://github.com/csuhan/opendet2
* Apache License
* Copyright (c) 2018-2023 OpenMMLab
* Copyright (c) SafeDNN group 2023
"""
from mmdet.core import (bbox2roi, bbox2result)
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as dists


@HEADS.register_module()
class ODRoiHead(StandardRoIHead):

    def __init__(self, num_known_classes=20,
                 max_iters=90000,
                 up_loss_start_iter=100,
                 up_loss_sampling_metric="min_score",
                 up_loss_topk=3,
                 up_loss_alpha=1.,
                 up_loss_weight=.2,
                 ic_loss_out_dim=128,
                 ic_loss_queue_size=256,
                 ic_loss_in_queue_size=16,
                 ic_loss_batch_iou_thr=.5,
                 ic_loss_queue_iou_thr=.7,
                 ic_loss_queue_tau=.1,
                 ic_loss_weight=.1, *args, **kwargs):
        super(ODRoiHead, self).__init__(
            *args,
            **kwargs)
        self.curr_iter = 0
        self.num_known_classes = num_known_classes
        self.max_iters = max_iters

        self.up_loss = UPLoss(
            self.bbox_head.num_classes,
            sampling_metric=up_loss_sampling_metric,
            topk=up_loss_topk,
            alpha=up_loss_alpha
        )
        self.up_loss_start_iter = up_loss_start_iter
        self.up_loss_weight = up_loss_weight

        self.ic_loss_loss = ICLoss(tau=ic_loss_queue_tau)
        self.ic_loss_out_dim = ic_loss_out_dim
        self.ic_loss_queue_size = ic_loss_queue_size
        self.ic_loss_in_queue_size = ic_loss_in_queue_size
        self.ic_loss_batch_iou_thr = ic_loss_batch_iou_thr
        self.ic_loss_queue_iou_thr = ic_loss_queue_iou_thr
        self.ic_loss_weight = ic_loss_weight

        self.queue = torch.zeros((self.num_known_classes, ic_loss_queue_size, ic_loss_out_dim), device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.queue_label = torch.empty((self.num_known_classes, ic_loss_queue_size), device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")).fill_(-1).long()
        self.queue_ptr = torch.zeros(self.num_known_classes, dtype=torch.long, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, mlp_feats = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, mlp_feats=mlp_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        self.curr_iter += 1
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)

        ious = torch.cat([torch.cat((p.pos_assigned_ious, torch.zeros(p.neg_inds.shape[0], device=rois.device))) for p in sampling_results], dim=0)
        gt_labels = torch.cat([torch.cat((p.pos_gt_labels, torch.ones(p.neg_inds.shape[0], device=rois.device, dtype=torch.int64) * self.num_known_classes)) for p in
                          sampling_results], dim=0)
        # gt_labels = torch.cat([p.pos_gt_labels for p in sampling_results], dim=0)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        self._dequeue_and_enqueue(
            bbox_results['mlp_feats'], gt_labels, ious, iou_thr=self.ic_loss_queue_iou_thr)
        losses = {
            "loss_cls": loss_bbox["loss_cls"],
            "loss_ic": self.get_ic_loss(bbox_results['mlp_feats'], gt_labels, ious),
            "loss_up": self.get_up_loss(bbox_results['cls_score'], gt_labels),
            "loss_box_reg": loss_bbox["loss_bbox"]
        }
        bbox_results.update(loss_bbox=losses)
        return bbox_results

    def get_up_loss(self, scores, gt_classes):
        # start up loss after several warmup iters
        if self.curr_iter > self.up_loss_start_iter:
            loss_cls_up = self.up_loss(scores, gt_classes)
        else:
            loss_cls_up = scores.new_tensor(0.0)

        return self.up_loss_weight * loss_cls_up

    def get_ic_loss(self, feat, gt_classes, ious):
        # select foreground and iou > thr instance in a mini-batch
        pos_inds = (ious > self.ic_loss_batch_iou_thr) & (
                gt_classes != self.bbox_head.num_classes)
        feat, gt_classes = feat[pos_inds], gt_classes[pos_inds]

        queue = self.queue.reshape(-1, self.ic_loss_out_dim)
        queue_label = self.queue_label.reshape(-1)
        queue_inds = queue_label != -1  # filter empty queue
        queue, queue_label = queue[queue_inds], queue_label[queue_inds]

        loss_ic_loss = self.ic_loss_loss(feat, gt_classes, queue, queue_label)
        # loss decay
        decay_weight = 1.0 - self.curr_iter / self.max_iters
        return self.ic_loss_weight * decay_weight * loss_ic_loss

    def _dequeue_and_enqueue(self, feat, gt_classes, ious, iou_thr=0.7):
        # 2. filter by iou and obj, remove bg
        keep = (ious > iou_thr) & (gt_classes != self.bbox_head.num_classes)
        feat, gt_classes = feat[keep], gt_classes[keep]

        for i in range(self.num_known_classes):
            ptr = int(self.queue_ptr[i])
            cls_ind = gt_classes == i
            cls_feat, cls_gt_classes = feat[cls_ind], gt_classes[cls_ind]
            # 3. sort by similarity, low sim ranks first
            cls_queue = self.queue[i, self.queue_label[i] != -1]
            _, sim_inds = F.cosine_similarity(
                cls_feat[:, None], cls_queue[None, :], dim=-1).mean(dim=1).sort()
            top_sim_inds = sim_inds[:self.ic_loss_in_queue_size]
            cls_feat, cls_gt_classes = cls_feat[top_sim_inds], cls_gt_classes[top_sim_inds]
            # 4. in queue
            batch_size = cls_feat.size(
                0) if ptr + cls_feat.size(0) <= self.ic_loss_queue_size else self.ic_loss_queue_size - ptr
            self.queue[i, ptr:ptr + batch_size] = cls_feat[:batch_size].detach()
            self.queue_label[i, ptr:ptr + batch_size] = cls_gt_classes[:batch_size].detach()

            ptr = ptr + batch_size if ptr + batch_size < self.ic_loss_queue_size else 0
            self.queue_ptr[i] = ptr

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        uncertainty = torch.softmax(cls_score, dim=1)[..., -2].unsqueeze(1)
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        uncertainty = uncertainty.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label, keep = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg,
                    ret_keep_ids=True)
                u = uncertainty[i][keep]

                det_bbox = torch.cat((det_bbox, u), 1)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

class ICLoss(nn.Module):
    """ Instance Contrastive Loss
    """
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau

    def forward(self, features, labels, queue_features, queue_labels):
        device = features.device
        mask = torch.eq(labels[:, None], queue_labels[:, None].T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, queue_features.T), self.tau)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.ones_like(logits)
        # mask itself
        logits_mask[logits == 0] = 0

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - mean_log_prob_pos.mean()
        # trick: avoid loss nan
        return loss if not torch.isnan(loss) else features.new_tensor(0.0)


class UPLoss(nn.Module):
    """Unknown Probability Loss
    """

    def __init__(self,
                 num_classes: int,
                 sampling_metric: str = "min_score",
                 topk: int = 3,
                 alpha: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        assert sampling_metric in ["min_score", "max_entropy", "random"]
        self.sampling_metric = sampling_metric
        # if topk==-1, sample len(fg)*2 examples
        self.topk = topk
        self.alpha = alpha

    def _soft_cross_entropy(self, input, target):
        logprobs = F.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]

    def _sampling(self, scores, labels):
        fg_inds = labels != (self.num_classes - 1)
        fg_scores, fg_labels = scores[fg_inds], labels[fg_inds]
        bg_scores, bg_labels = scores[~fg_inds], labels[~fg_inds]

        # remove unknown classes
        _fg_scores = torch.cat(
            [fg_scores[:, :self.num_classes-2], fg_scores[:, -2:]], dim=1)
        _bg_scores = torch.cat(
            [bg_scores[:, :self.num_classes-2], bg_scores[:, -2:]], dim=1)

        num_fg = fg_scores.size(0)
        topk = num_fg if (self.topk == -1) or (num_fg <
                                               self.topk) else self.topk
        num_bg = bg_scores.size(0)
        if num_bg < topk:
            topk = num_bg
        # use maximum entropy as a metric for uncertainty
        # we select topk proposals with maximum entropy
        if self.sampling_metric == "max_entropy":
            pos_metric = dists.Categorical(
                _fg_scores.softmax(dim=1)).entropy()
            neg_metric = dists.Categorical(
                _bg_scores.softmax(dim=1)).entropy()
        # use minimum score as a metric for uncertainty
        # we select topk proposals with minimum max-score
        elif self.sampling_metric == "min_score":
            pos_metric = -_fg_scores.max(dim=1)[0]
            neg_metric = -_bg_scores.max(dim=1)[0]
        # we randomly select topk proposals
        elif self.sampling_metric == "random":
            pos_metric = torch.rand(_fg_scores.size(0),).to(scores.device)
            neg_metric = torch.rand(_bg_scores.size(0),).to(scores.device)

        _, pos_inds = pos_metric.topk(topk)
        _, neg_inds = neg_metric.topk(topk)
        fg_scores, fg_labels = fg_scores[pos_inds], fg_labels[pos_inds]
        bg_scores, bg_labels = bg_scores[neg_inds], bg_labels[neg_inds]

        return fg_scores, bg_scores, fg_labels, bg_labels

    def forward(self, scores, labels):
        fg_scores, bg_scores, fg_labels, bg_labels = self._sampling(
            scores, labels)
        # sample both fg and bg
        scores = torch.cat([fg_scores, bg_scores])
        labels = torch.cat([fg_labels, bg_labels])

        num_sample, num_classes = scores.shape
        if num_sample == 0:
            return scores.new_tensor(0.)

        mask = torch.arange(num_classes).repeat(
            num_sample, 1).to(scores.device)
        inds = mask != labels[:, None].repeat(1, num_classes)
        mask = mask[inds].reshape(num_sample, num_classes-1)

        gt_scores = torch.gather(
            F.softmax(scores, dim=1), 1, labels[:, None]).squeeze(1)
        mask_scores = torch.gather(scores, 1, mask)

        gt_scores[gt_scores < 0] = 0.0
        targets = torch.zeros_like(mask_scores)
        num_fg = fg_scores.size(0)
        targets[:num_fg, self.num_classes-2] = gt_scores[:num_fg] * \
            (1-gt_scores[:num_fg]).pow(self.alpha)
        targets[num_fg:, self.num_classes-1] = gt_scores[num_fg:] * \
            (1-gt_scores[num_fg:]).pow(self.alpha)

        return self._soft_cross_entropy(mask_scores, targets.detach())
