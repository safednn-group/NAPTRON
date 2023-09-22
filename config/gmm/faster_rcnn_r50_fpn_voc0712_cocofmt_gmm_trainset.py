_base_ = [
    './faster_rcnn_r50_fpn_voc0712_cocofmt_gmm.py'
]

data_root = 'data/VOC_0712_converted/'


data = dict(
    val=dict(ann_file=data_root + 'voc0712_train_all.json'),
    test=dict(ann_file=data_root + 'voc0712_train_all.json')
)