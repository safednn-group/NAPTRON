from mmdet.core.bbox.samplers import SamplingResult


class ODSamplingResult(SamplingResult):
    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags):
        super().__init__(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)
        self.pos_assigned_ious = assign_result.max_overlaps[pos_inds]
