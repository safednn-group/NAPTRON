_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/coco.py',
    '../../_base_/runtimes/default_runtime.py'
]

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.coco_eval_ood',
             'safednn_naptron.uncertainty.coco_ood_dataset',
             'safednn_naptron.uncertainty.owod.roi_head',
             'safednn_naptron.uncertainty.owod.bbox_head'],
    allow_failed_imports=False)

output_handler = dict(
    type="simple_dump"
)

model = dict(
    roi_head=dict(
        type="OWODRoiHead",
        num_classes=21,
        clustering_items_per_class=20,
        clustering_start_iter=2000,
        clustering_update_mu_iter=3000,
        clustering_momentum=0.99,
        enable_clustering=False,
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
    test_cfg=dict(rcnn=dict(score_thr=0.01))
)

dataset_type = 'CocoOODDataset'
data = dict(
    val=dict(type=dataset_type, filter_empty_gt=False),
    test=dict(type=dataset_type, filter_empty_gt=False))
