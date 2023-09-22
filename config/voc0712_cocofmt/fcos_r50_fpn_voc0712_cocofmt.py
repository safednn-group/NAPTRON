_base_ = [
    '../_base_/models/fcos_r50_fpn.py',
    '../_base_/datasets/voc0712_cocofmt.py',
    '../_base_/runtimes/default_runtime.py',
    '../_base_/schedules/default_schedule.py'
]

model = dict(bbox_head=dict(num_classes=20))

# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
