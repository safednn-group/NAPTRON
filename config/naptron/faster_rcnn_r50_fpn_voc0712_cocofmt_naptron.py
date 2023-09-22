_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712_cocofmt.py',
    '../_base_/runtimes/default_runtime.py',
    '../_base_/schedules/default_schedule.py'
]

output_handler = dict(
    type="simple_dump"
)
score_thr = 0.01

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.naptron.roi_head',
             'safednn_naptron.uncertainty.naptron.bbox_head'],
    allow_failed_imports=False)

model = dict(
    roi_head=dict(
        type="NAPTRONRoiHead",
        bbox_head=dict(
            type="NAPTRONBBoxHead",
            num_classes=20)
    ),
    test_cfg=dict(rcnn=dict(score_thr=score_thr))
)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
