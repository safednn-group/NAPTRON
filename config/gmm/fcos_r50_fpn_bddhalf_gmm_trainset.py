_base_ = [
    './fcos_r50_fpn_bddhalf_gmm.py'
]

data_root = 'data/bdd100k/'

data = dict(
    val=dict(ann_file=data_root + 'labels/det_train_coco.json',
             img_prefix=data_root + 'images/100k/train', ),
    test=dict(ann_file=data_root + 'labels/det_train_coco.json',
              img_prefix=data_root + 'images/100k/train', )
)