_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../bdd/bdd100k_cocofmt_half.py',
    '../_base_/runtimes/default_runtime.py',
    '../_base_/schedules/default_schedule.py'
]

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.open_det.roi_head',
             'safednn_naptron.uncertainty.open_det.bbox_head',
             'safednn_naptron.uncertainty.open_det.random_sampler',
             'safednn_naptron.uncertainty.coco_ood_dataset'],
    allow_failed_imports=False)
score_thr = 0.01


output_handler = dict(
    type="simple_dump"
)

model = dict(
    roi_head=dict(
        type="ODRoiHead",
        num_known_classes=4,
        max_iters=60000,
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

CLASSES = ('pedestrian', 'bicycle', 'car', 'traffic sign', 'unknown')

data = dict(
    train=dict(
        dataset=dict(
            classes=CLASSES)),
    val=dict(classes=CLASSES),
    test=dict(classes=CLASSES)
)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

