_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712_cocofmt.py',
    '../_base_/runtimes/default_runtime.py',
    '../_base_/schedules/default_schedule.py'
]

CLASSES = ('airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle',
           'person', 'potted plant', 'sheep', 'couch', 'train', 'tv', 'unknown')

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.owod.roi_head',
             'safednn_naptron.uncertainty.owod.bbox_head',
             'safednn_naptron.uncertainty.owod.random_sampler'],
    allow_failed_imports=False)

score_thr = 0.01


model = dict(
    roi_head=dict(
        type="OWODRoiHead",
        num_classes=21,
        clustering_items_per_class=20,
        clustering_start_iter=2000,
        clustering_update_mu_iter=3000,
        clustering_momentum=0.99,
        enable_clustering=True,
        prev_intro_cls=0,
        curr_intro_cls=20,
        max_iterations=90000,
        output_dir='work_dirs/owod',
        feat_store_path='feature_store',
        margin=10,
        compute_energy=False,
        energy_save_path='work_dirs/owod/energy',
        bbox_head=dict(
            type="OWODBBoxHead",
            num_classes=21)
    ),
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(
                type='OWODSampler',
            ),
        )
    ),
    test_cfg=dict(rcnn=dict(score_thr=score_thr))
)

data = dict(
    train=dict(
        dataset=dict(
            classes=CLASSES)),
    val=dict(classes=CLASSES),
    test=dict(classes=CLASSES)
)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
