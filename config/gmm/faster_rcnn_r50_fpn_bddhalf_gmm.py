_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../bdd/bdd100k_cocofmt_half.py',
    '../_base_/runtimes/default_runtime.py',
    '../_base_/schedules/default_schedule.py'
]

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.gmm.anchor_cross_entropy_loss',
             'safednn_naptron.uncertainty.gmm.faster_rcnn.bbox_head',
             'safednn_naptron.uncertainty.gmm.faster_rcnn.roi_head',
             'safednn_naptron.uncertainty.coco_ood_dataset'],
    allow_failed_imports=False)

output_handler = dict(
    type="simple_dump"
)

score_thr = 0.01

model = dict(
    roi_head=dict(
        type='GMMRoiHead',
        bbox_head=dict(
            type='LogitsBBoxHead',
            num_classes=4,
            loss_cls=dict(type='AnchorwCrossEntropyLoss', loss_weight=1.0, anchor_weight=0.1, n_classes=5,
                          background_flag=True))
    ),
    test_cfg=dict(rcnn=dict(score_thr=score_thr))
)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
