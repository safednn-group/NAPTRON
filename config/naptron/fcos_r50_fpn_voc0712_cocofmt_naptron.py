_base_ = [
    '../_base_/models/fcos_r50_fpn.py',
    '../_base_/datasets/voc0712_cocofmt.py',
    '../_base_/runtimes/default_runtime.py',
    '../_base_/schedules/default_schedule.py'
]

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.naptron.fcos_head',
             'safednn_naptron.uncertainty.naptron.fcos'],
    allow_failed_imports=False)

output_handler = dict(
    type="simple_dump"
)

score_thr = 0.01

model = dict(
    type="NAPTRONFCOS",
    bbox_head=dict(
        type='NAPTRONFCOSHead',
        num_classes=20,
    ),
    test_cfg=dict(score_thr=score_thr)
)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
