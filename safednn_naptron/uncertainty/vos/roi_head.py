"""
* Based on https://github.com/open-mmlab/mmdetection/tree/v2.23.0 and https://github.com/deeplearning-wisc/vos
* Apache License
* Copyright (c) 2018-2023 OpenMMLab
* Copyright (c) SafeDNN group 2023
"""
from mmdet.core import (bbox2roi, bbox2result)
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead

import torch
import torch.nn.functional as F

from typing import List


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


@HEADS.register_module()
class VOSRoiHead(StandardRoIHead):

    def __init__(self,
                 sample_number,
                 starting_iter,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(VOSRoiHead, self).__init__(
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

        self.sample_number = sample_number
        self.start_iter = starting_iter

        self.logistic_regression = torch.nn.Linear(1, 2)
        self.logistic_regression.cuda()
        # torch.nn.init.xavier_normal_(self.logistic_regression.weight)
        self.curr_iteration = 0
        self.select = 1
        self.sample_from = 10000
        self.loss_weight = 0.1
        self.num_classes = self.bbox_head.num_classes
        self.weight_energy = torch.nn.Linear(self.num_classes, 1).cuda()
        torch.nn.init.uniform_(self.weight_energy.weight)
        self.data_dict = torch.zeros(self.num_classes, self.sample_number, 1024).cuda()
        self.number_dict = {}
        self.eye_matrix = torch.eye(1024, device='cuda')
        self.trajectory = torch.zeros((self.num_classes, 900, 3)).cuda()
        for i in range(self.num_classes):
            self.number_dict[i] = 0
        self.cos = torch.nn.MSELoss()

    def log_sum_exp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation

        value.exp().sum(dim, keepdim).log()
        """
        import math
        # TODO: torch.max(value, dim=None) threw an error at time of writing
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(
                F.relu(self.weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            # if isinstance(sum_exp, Number):
            #     return m + math.log(sum_exp)
            # else:
            return m + torch.log(sum_exp)

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
                    feats=[lvl_feat[i][None] for lvl_feat in x])
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

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """
        Run forward function and calculate loss for box head in training.
        ***SAFEDNN modification***
        The following is mmdetection version of code from: https://github.com/deeplearning-wisc/vos
        It is as little modified as possible.
        """
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        scores, proposal_deltas, box_features = \
            bbox_results["cls_score"], bbox_results["bbox_pred"], bbox_results["bbox_feats"]

        box_features = self.bbox_head.forward_shared(box_features)
        # parse classification outputs
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)

        gt_classes = bbox_targets[0]

        # parse box regression outputs
        if len(gt_bboxes):
            # proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            # assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # # If "gt_boxes" does not exist, the proposals must be all negative and
            # # should not be included in regression loss computation.
            # # Here we just use proposal_boxes as an arbitrary placeholder because its
            # # value won't be used in self.box_reg_loss().
            # gt_boxes = cat(
            #     [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
            #     dim=0,
            # )

            sum_temp = 0
            for index in range(self.num_classes):
                sum_temp += self.number_dict[index]
            lr_reg_loss = torch.zeros(1).cuda()
            if sum_temp == self.num_classes * self.sample_number and self.curr_iteration < self.start_iter:
                selected_fg_samples = (gt_classes != scores.shape[1] - 1).nonzero().view(-1)
                indices_numpy = selected_fg_samples.cpu().numpy().astype(int)
                gt_classes_numpy = gt_classes.cpu().numpy().astype(int)
                # maintaining an ID data queue for each class.
                for index in indices_numpy:
                    dict_key = gt_classes_numpy[index]
                    self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                          box_features[index].detach().view(1, -1)), 0)
            elif sum_temp == self.num_classes * self.sample_number and self.curr_iteration >= self.start_iter:
                selected_fg_samples = (gt_classes != scores.shape[1] - 1).nonzero().view(-1)
                indices_numpy = selected_fg_samples.cpu().numpy().astype(int)
                gt_classes_numpy = gt_classes.cpu().numpy().astype(int)
                # maintaining an ID data queue for each class.
                for index in indices_numpy:
                    dict_key = gt_classes_numpy[index]
                    self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                          box_features[index].detach().view(1, -1)), 0)
                # the covariance finder needs the data to be centered.
                for index in range(self.num_classes):
                    if index == 0:
                        X = self.data_dict[index] - self.data_dict[index].mean(0)
                        mean_embed_id = self.data_dict[index].mean(0).view(1, -1)
                    else:
                        X = torch.cat((X, self.data_dict[index] - self.data_dict[index].mean(0)), 0)
                        mean_embed_id = torch.cat((mean_embed_id,
                                                   self.data_dict[index].mean(0).view(1, -1)), 0)

                # add the variance.
                temp_precision = torch.mm(X.t(), X) / len(X)
                # for stable training.
                temp_precision += 0.0001 * self.eye_matrix

                for index in range(self.num_classes):
                    new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                        mean_embed_id[index], covariance_matrix=temp_precision)
                    negative_samples = new_dis.rsample((self.sample_from,))
                    prob_density = new_dis.log_prob(negative_samples)

                    # keep the data in the low density area.
                    cur_samples, index_prob = torch.topk(- prob_density, self.select)
                    if index == 0:
                        ood_samples = negative_samples[index_prob]
                    else:
                        ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
                    del new_dis
                    del negative_samples

                if len(ood_samples) != 0:
                    # add some gaussian noise
                    # ood_samples = self.noise(ood_samples)
                    energy_score_for_fg = self.log_sum_exp(scores[selected_fg_samples][:, :-1], 1)
                    predictions_ood = self.bbox_head.forward_separate(ood_samples)
                    energy_score_for_bg = self.log_sum_exp(predictions_ood[0][:, :-1], 1)

                    input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                    labels_for_lr = torch.cat((torch.ones(len(selected_fg_samples)).cuda(),
                                               torch.zeros(len(ood_samples)).cuda()), -1)

                    criterion = torch.nn.CrossEntropyLoss()  # weight=weights_fg_bg)
                    output = self.logistic_regression(input_for_lr.view(-1, 1))
                    lr_reg_loss = criterion(output, labels_for_lr.long())

                del ood_samples

            else:

                selected_fg_samples = (gt_classes != scores.shape[1] - 1).nonzero().view(-1)
                indices_numpy = selected_fg_samples.cpu().numpy().astype(int)
                gt_classes_numpy = gt_classes.cpu().numpy().astype(int)
                for index in indices_numpy:
                    dict_key = gt_classes_numpy[index]
                    if self.number_dict[dict_key] < self.sample_number:
                        self.data_dict[dict_key][self.number_dict[dict_key]] = box_features[index].detach()
                        self.number_dict[dict_key] += 1
            # create a dummy in order to have all weights to get involved in for a loss.
            loss_dummy = self.cos(self.logistic_regression(torch.zeros(1).cuda()), self.logistic_regression.bias)
            loss_dummy1 = self.cos(self.weight_energy(torch.zeros(self.num_classes).cuda()), self.weight_energy.bias)
            del box_features

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        if sum_temp == self.num_classes * self.sample_number:
            losses = {
                "loss_cls": loss_bbox["loss_cls"],
                "lr_reg_loss": self.loss_weight * lr_reg_loss,
                "loss_dummy": loss_dummy,
                "loss_dummy1": loss_dummy1,
                "loss_box_reg": loss_bbox["loss_bbox"]

            }
        else:
            losses = {
                "loss_cls": loss_bbox["loss_cls"],
                "lr_reg_loss": torch.zeros(1).cuda(),
                "loss_dummy": loss_dummy,
                "loss_dummy1": loss_dummy1,
                "loss_box_reg": loss_bbox["loss_bbox"]
            }

        bbox_results.update(loss_bbox=losses)
        return bbox_results

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

            energy_score = torch.logsumexp(cs[:, :-1], dim=1).unsqueeze(1)
            det_bbox = torch.cat((det_bbox, energy_score), 1)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        return det_bboxes, det_labels

