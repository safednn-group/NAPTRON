_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../bdd/bdd100k_cocofmt_half.py',
    '../_base_/runtimes/default_runtime.py',
    '../_base_/schedules/default_schedule.py'
]

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.open_det.retina_head',
             'safednn_naptron.uncertainty.open_det.pseudo_sampler',
             'safednn_naptron.uncertainty.coco_ood_dataset'],
    allow_failed_imports=False)

output_handler = dict(
    type="simple_dump"
)

score_thr = 0.01

model = dict(
    bbox_head=dict(
        type="ODRetinaHead",
        num_known_classes=4,
        max_iters=50000,
        num_classes=5
    ),
    test_cfg=dict(score_thr=score_thr),
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
