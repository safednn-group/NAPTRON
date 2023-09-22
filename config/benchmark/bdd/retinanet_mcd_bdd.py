_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py',
    '../../_base_/datasets/bdd100k_cocofmt.py',
    '../../_base_/runtimes/default_runtime.py'
]

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.coco_eval_ood',
             'safednn_naptron.uncertainty.coco_ood_dataset',
             'safednn_naptron.uncertainty.mcd.retina_head'],
    allow_failed_imports=False)

output_handler = dict(
    type="simple_dump",
    chunks_count=5
)

score_thr = 0.01

model = dict(
    bbox_head=dict(
        type="MCDRetinaHead",
        num_classes=4,
        num_forward_passes=10,
        dropout_rates=(0., 0., 0., 0.2, 0.2),
    ),
    test_cfg=dict(score_thr=score_thr)
)

dataset_type = 'CocoOODDataset'
data = dict(
    val=dict(type=dataset_type, filter_empty_gt=False),
    test=dict(type=dataset_type, filter_empty_gt=False))

