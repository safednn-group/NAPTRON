
_base_ = [
    './oln_box.py',
    '../bdd/bdd100k_cocofmt_half.py',
    '../_base_/runtimes/default_runtime.py',
    '../_base_/schedules/default_schedule.py'
]

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.coco_eval_ood',
             'safednn_naptron.uncertainty.coco_ood_dataset',
             'safednn_naptron.uncertainty.oln.convfc_bbox_score_head',
             'safednn_naptron.uncertainty.oln.oln_roi_head',
             'safednn_naptron.uncertainty.oln.oln_rpn_head'],
    allow_failed_imports=False)


output_handler = dict(
    type="simple_dump"
)

dataset_type = 'CocoOODDataset'

data = dict(
    train=dict(
        dataset=dict(
            type=dataset_type,
            is_class_agnostic=True)),
    val=dict(type=dataset_type,
             is_class_agnostic=False),
    test=dict(type=dataset_type,
              is_class_agnostic=False)
)


# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)