_base_ = [
    '../oln/oln_box.py',
    '../../_base_/datasets/coco.py',
    '../../_base_/runtimes/default_runtime.py'
]

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.coco_eval_ood',
             'safednn_naptron.uncertainty.coco_ood_dataset',
             'safednn_naptron.uncertainty.oln.convfc_bbox_score_head',
             'safednn_naptron.uncertainty.oln.oln_roi_head',
             'safednn_naptron.uncertainty.oln.oln_rpn_head'],
    allow_failed_imports=False)

output_handler = dict(
    type="simple_dump"
)


dataset_type = 'CocoOODDataset'
data = dict(
    val=dict(type=dataset_type, filter_empty_gt=False),
    test=dict(type=dataset_type, filter_empty_gt=False))

