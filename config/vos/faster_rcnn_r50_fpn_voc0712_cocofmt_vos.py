_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712_cocofmt.py',
    '../_base_/runtimes/default_runtime.py',
    '../_base_/schedules/default_schedule.py'
]

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.vos.roi_head',
             'safednn_naptron.uncertainty.vos.bbox_head'],
    allow_failed_imports=False)

score_thr = 0.01

model = dict(
    roi_head=dict(
        type="VOSRoiHead",
        sample_number=1000,
        starting_iter=10000,
        bbox_head=dict(
            type="VOSBBoxHead",
            num_classes=20)
    ),
    test_cfg=dict(rcnn=dict(score_thr=score_thr))
)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

