"""
* Based on https://github.com/open-mmlab/mmdetection/tree/v2.23.0
* Apache License
* Copyright (c) 2018-2023 OpenMMLab
* Copyright (c) SafeDNN group 2023
"""
from mmdet.models.builder import DETECTORS
from mmdet.core import bbox2result
from mmdet.models.detectors import RetinaNet
import torch


@DETECTORS.register_module()
class NAPTRONRetinaNet(RetinaNet):

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        ret = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        if len(ret[0]) == 4:
            det_bboxes, det_labels, det_nap, nap_shapes = ret[0]
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            ]
            return bbox_results, det_nap, nap_shapes, det_labels
        else:
            det_bboxes, det_labels = ret[0]
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            ]
            return bbox_results, torch.tensor([[]]), torch.tensor([]), det_labels

