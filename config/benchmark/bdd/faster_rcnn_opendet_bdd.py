_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/bdd100k_cocofmt.py',
    '../../_base_/runtimes/default_runtime.py',
]

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.coco_eval_ood',
             'safednn_naptron.uncertainty.coco_ood_dataset',
             'safednn_naptron.uncertainty.open_det.roi_head',
             'safednn_naptron.uncertainty.open_det.bbox_head',
             'safednn_naptron.uncertainty.open_det.random_sampler'],
    allow_failed_imports=False)

output_handler = dict(
    type="simple_dump",
    chunks_count=5
)

score_thr = 0.01

model = dict(
    roi_head=dict(
        type="ODRoiHead",
        num_known_classes=4,
        bbox_head=dict(
            type="ODBBoxHead",
            reg_class_agnostic=True,
            num_classes=5)
    ),
    test_cfg=dict(rcnn=dict(score_thr=score_thr)),
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(
                type='ODSampler',
                pos_fraction=0.5,
            ),
        )
    ),
)


dataset_type = 'CocoOODDataset'
data = dict(
    val=dict(type=dataset_type, filter_empty_gt=False),
    test=dict(type=dataset_type, filter_empty_gt=False))
