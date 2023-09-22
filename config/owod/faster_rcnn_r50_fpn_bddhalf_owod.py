_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../bdd/bdd100k_cocofmt_half.py',
    '../_base_/runtimes/default_runtime.py',
    '../_base_/schedules/default_schedule.py'
]

CLASSES = ('pedestrian', 'bicycle', 'car', 'traffic sign', 'unknown')

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.owod.roi_head',
             'safednn_naptron.uncertainty.owod.bbox_head',
             'safednn_naptron.uncertainty.owod.random_sampler',
             'safednn_naptron.uncertainty.coco_ood_dataset'],
    allow_failed_imports=False)

score_thr = 0.01


model = dict(
    roi_head=dict(
        type="OWODRoiHead",
        num_classes=5,
        clustering_items_per_class=4,
        clustering_start_iter=2000,
        clustering_update_mu_iter=3000,
        clustering_momentum=0.99,
        enable_clustering=True,
        prev_intro_cls=0,
        curr_intro_cls=4,
        max_iterations=60000,
        output_dir='work_dirs/owod_bdd',
        feat_store_path='feature_store',
        margin=10,
        compute_energy=False,
        energy_save_path='work_dirs/owod_bdd/energy',
        bbox_head=dict(
            type="OWODBBoxHead",
            num_classes=5)
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
