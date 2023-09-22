_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/coco.py',
    '../../_base_/runtimes/default_runtime.py'
]

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.coco_eval_ood',
             'safednn_naptron.uncertainty.coco_ood_dataset',
             'safednn_naptron.uncertainty.mcd.roi_head',
             'safednn_naptron.uncertainty.mcd.bbox_head'],
    allow_failed_imports=False)

output_handler = dict(
    type="simple_dump"
)

score_thr = 0.01

model = dict(
    roi_head=dict(
        type="MCDRoiHead",
        num_forward_passes=10,
        bbox_head=dict(
            type="MCDBBoxHead",
            num_classes=20,
            dropout_rates=(0., 0.1, 0.)
        )
    ),
    test_cfg=dict(rcnn=dict(score_thr=score_thr))
)

dataset_type = 'CocoOODDataset'
data = dict(
    val=dict(type=dataset_type, filter_empty_gt=False),
    test=dict(type=dataset_type, filter_empty_gt=False))

