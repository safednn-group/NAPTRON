"""
* Based on https://github.com/open-mmlab/mmdetection/tree/v2.23.0
* Apache License
* Copyright (c) 2018-2023 OpenMMLab
* Copyright (c) SafeDNN group 2023
"""
from mmdet.core import (bbox2roi, bbox2result)
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead
from .store import Store

import torch
import torch.nn as nn

from typing import List
import os
import shortuuid


@HEADS.register_module()
class OWODRoiHead(StandardRoIHead):

    def __init__(self,
                 num_classes,
                 clustering_items_per_class,
                 clustering_start_iter,
                 clustering_update_mu_iter,
                 clustering_momentum,
                 enable_clustering,
                 prev_intro_cls,
                 curr_intro_cls,
                 max_iterations,
                 output_dir,
                 feat_store_path,
                 margin,
                 compute_energy,
                 energy_save_path,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(OWODRoiHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None

        self.enable_threshold_autolabelling = True
        self.unk_k = 1
        self.compute_energy_flag = compute_energy
        self.energy_save_path = energy_save_path
        self.num_classes = num_classes
        self.clustering_start_iter = clustering_start_iter
        self.clustering_update_mu_iter = clustering_update_mu_iter
        self.clustering_momentum = clustering_momentum

        self.hingeloss = nn.HingeEmbeddingLoss(2)
        self.enable_clustering = enable_clustering

        self.prev_intro_cls = prev_intro_cls
        self.curr_intro_cls = curr_intro_cls
        self.seen_classes = self.prev_intro_cls + self.curr_intro_cls
        self.invalid_class_range = list(range(self.seen_classes, self.num_classes-1))
        self.curr_iteration = 0
        self.max_iterations = max_iterations
        self.feature_store_is_stored = False
        self.output_dir = output_dir
        self.feat_store_path = feat_store_path
        self.feature_store_save_loc = os.path.join(self.output_dir, self.feat_store_path, 'feat.pt')

        if os.path.isfile(self.feature_store_save_loc):
            self.feature_store = torch.load(self.feature_store_save_loc)
        else:
            self.feature_store = Store(num_classes + 1, clustering_items_per_class)
        self.means = [None for _ in range(num_classes + 1)]
        self.margin = margin

    def compute_energy(self, predictions, proposals):
        gt_classes = torch.cat([torch.cat((p.pos_gt_labels, p.neg_bboxes.new_full((p.neg_inds.size()[0],),
                                                                                  self.num_classes,
                                                                                  dtype=torch.long))) for p in
                                proposals])
        data = (predictions, gt_classes)
        location = os.path.join(self.energy_save_path, shortuuid.uuid() + '.pkl')
        torch.save(data, location)

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.curr_iteration += 1
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                    num_classes=self.num_classes
                )
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def update_feature_store(self, features, proposals):
        gt_classes = torch.cat([torch.cat((p.pos_gt_labels, p.neg_bboxes.new_full((p.neg_inds.size()[0],),
                                                                                  self.num_classes,
                                                                                  dtype=torch.long))) for p in
                                proposals])
        self.feature_store.add(features, gt_classes)

        if self.curr_iteration == self.max_iterations - 1 and not self.feature_store_is_stored:
            torch.save(self.feature_store, self.feature_store_save_loc)
            self.feature_store_is_stored = True

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""

        rois = bbox2roi([res.bboxes for res in sampling_results])

        bbox_results = self._bbox_forward(x, rois)

        scores, proposal_deltas, box_features = \
            bbox_results["cls_score"], bbox_results["bbox_pred"], bbox_results["bbox_feats"]

        box_features = self.bbox_head.forward_shared(box_features)

        if self.enable_clustering:
            self.update_feature_store(box_features, sampling_results)
        if self.compute_energy_flag:
            self.compute_energy(scores, sampling_results)
            losses = {
                "loss_cls": torch.zeros(1, requires_grad=True).cuda(),
                "lr_reg_loss": torch.zeros(1, requires_grad=True).cuda(),
                "loss_rpn_cls": torch.zeros(1, requires_grad=True).cuda(),
                "loss_rpn_bbox": torch.zeros(1, requires_grad=True).cuda(),
                "loss_box_reg": torch.zeros(1, requires_grad=True).cuda()
            }
            bbox_results.update(loss_bbox=losses)
            return bbox_results

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        losses = {
            "loss_cls": loss_bbox["loss_cls"],
            "lr_reg_loss": 0.1 * self.get_clustering_loss(box_features, sampling_results),
            "loss_box_reg": loss_bbox["loss_bbox"]

        }
        bbox_results.update(loss_bbox=losses)
        return bbox_results

    def clstr_loss_l2_cdist(self, input_features, proposals):
        """
        Get the foreground input_features, generate distributions for the class,
        get probability of each feature from each distribution;
        Compute loss: if belonging to a class -> likelihood should be higher
                      else -> lower
        :param input_features:
        :param proposals:
        :return:
        """
        gt_classes = torch.cat([torch.cat((p.pos_gt_labels, p.neg_bboxes.new_full((p.neg_inds.size()[0],),
                                                                                  self.num_classes,
                                                                                  dtype=torch.long))) for p in
                                proposals])
        mask = gt_classes != self.num_classes
        fg_features = input_features[mask]
        classes = gt_classes[mask]
        # fg_features = F.normalize(fg_features, dim=0)
        # fg_features = self.ae_model.encoder(fg_features)

        all_means = self.means
        for item in all_means:
            if item != None:
                length = item.shape
                break

        for i, item in enumerate(all_means):
            if item == None:
                all_means[i] = torch.zeros((length))

        distances = torch.cdist(fg_features, torch.stack(all_means).cuda(), p=self.margin)
        labels = []

        for index, feature in enumerate(fg_features):
            for cls_index, mu in enumerate(self.means):
                if mu is not None and feature is not None:
                    if  classes[index] ==  cls_index:
                        labels.append(1)
                    else:
                        labels.append(-1)
                else:
                    labels.append(0)

        loss = self.hingeloss(distances, torch.tensor(labels).reshape((-1, self.num_classes+1)).cuda())

        return loss

    def get_clustering_loss(self, input_features, proposals):
        if not self.enable_clustering:
            return torch.zeros(1).cuda()

        c_loss = torch.zeros(1).cuda()
        if self.curr_iteration == self.clustering_start_iter:
            items = self.feature_store.retrieve(-1)
            for index, item in enumerate(items):
                if len(item) == 0:
                    self.means[index] = None
                else:
                    mu = torch.tensor(item).mean(dim=0)
                    self.means[index] = mu
            c_loss = self.clstr_loss_l2_cdist(input_features, proposals)
            # Freeze the parameters when clustering starts
            # for param in self.ae_model.parameters():
            #     param.requires_grad = False
        elif self.curr_iteration > self.clustering_start_iter:
            if self.curr_iteration % self.clustering_update_mu_iter == 0:
                # Compute new MUs
                items = self.feature_store.retrieve(-1)
                new_means = [None for _ in range(self.num_classes + 1)]
                for index, item in enumerate(items):
                    if len(item) == 0:
                        new_means[index] = None
                    else:
                        new_means[index] = torch.tensor(item).mean(dim=0)
                # Update the MUs
                for i, mean in enumerate(self.means):
                    if(mean) is not None and new_means[i] is not None:
                        self.means[i] = self.clustering_momentum * mean + \
                                        (1 - self.clustering_momentum) * new_means[i]

            c_loss = self.clstr_loss_l2_cdist(input_features, proposals)
        return c_loss


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
            ***SAFEDNN modification***
            tuple[list[Tensor], list[Tensor], list[Tensor]]: The last list
                contains the class scores of the corresponding detected bboxes.

        """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0,), dtype=torch.long)
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
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cs_zeros = torch.zeros_like(cls_score[0])
        cls_score = cls_score.split(num_proposals_per_img, 0)

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
            bbox_pred = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        det_cs = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0,), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))
                cs = cs_zeros
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
                cs = cls_score[i][keep]

            energy_score = torch.logsumexp(cs[:, :-2], dim=1).unsqueeze(1)
            det_bbox = torch.cat((det_bbox, energy_score), 1)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        return det_bboxes, det_labels
