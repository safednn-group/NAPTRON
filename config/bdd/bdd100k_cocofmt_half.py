_base_ = [
    '../_base_/datasets/bdd100k_cocofmt.py',
]

CLASSES = ('pedestrian', 'bicycle', 'car', 'traffic sign')

dataset_type = 'CocoOODDataset'

custom_imports = dict(
    imports=['safednn_naptron.uncertainty.coco_ood_dataset'],
    allow_failed_imports=False)

data = dict(
    train=dict(dataset=dict(type=dataset_type, classes=CLASSES, filter_unknown_imgs=True)),
    val=dict(type=dataset_type, classes=CLASSES, filter_unknown_imgs=True),
    test=dict(type=dataset_type, classes=CLASSES, filter_unknown_imgs=True))
