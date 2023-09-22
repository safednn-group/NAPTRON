_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/voc0712_cocofmt.py',
    '../_base_/runtimes/default_runtime.py',
    '../_base_/schedules/default_schedule.py'
]

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.open_det.retina_head',
             'safednn_naptron.uncertainty.open_det.pseudo_sampler'],
    allow_failed_imports=False)

output_handler = dict(
    type="simple_dump"
)

score_thr = 0.01

model = dict(
    bbox_head=dict(
        type="ODRetinaHead",
        num_known_classes=20,
        max_iters=50000,
        num_classes=21
    ),
    test_cfg=dict(score_thr=score_thr),
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
