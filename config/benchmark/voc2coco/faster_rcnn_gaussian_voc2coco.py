_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/coco.py',
    '../../_base_/runtimes/default_runtime.py'
]

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.coco_eval_ood',
             'safednn_naptron.uncertainty.coco_ood_dataset',
             'safednn_naptron.uncertainty.gaussian_detector.faster_rcnn.bbox_head',
             'safednn_naptron.uncertainty.gaussian_detector.faster_rcnn.roi_head'],
    allow_failed_imports=False)

output_handler = dict(
    type="simple_dump"
)

score_thr = 0.01
model = dict(
    roi_head=dict(
        type='GaussianRoiHead',
        bbox_head=dict(
            type='GaussianBBoxHead',
            reg_class_agnostic=True,
            num_classes=20,
        )
    ),
    test_cfg=dict(rcnn=dict(score_thr=score_thr))
)



dataset_type = 'CocoOODDataset'
data = dict(
    val=dict(type=dataset_type, filter_empty_gt=False),
    test=dict(type=dataset_type, filter_empty_gt=False))


