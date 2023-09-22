_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712_cocofmt.py',
    '../_base_/runtimes/default_runtime.py',
    '../_base_/schedules/default_schedule.py'
]

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.open_det.roi_head',
             'safednn_naptron.uncertainty.open_det.bbox_head',
             'safednn_naptron.uncertainty.open_det.random_sampler'],
    allow_failed_imports=False)
score_thr = 0.01


output_handler = dict(
    type="simple_dump"
)

model = dict(
    roi_head=dict(
        type="ODRoiHead",
        num_known_classes=20,
        max_iters=50000,
        bbox_head=dict(
            type="ODBBoxHead",
            reg_class_agnostic=True,
            num_classes=21)
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
CLASSES = ('airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle',
           'person', 'potted plant', 'sheep', 'couch', 'train', 'tv', 'unknown')

data = dict(
    train=dict(
        dataset=dict(
            classes=CLASSES)),
    val=dict(classes=CLASSES),
    test=dict(classes=CLASSES)
)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

