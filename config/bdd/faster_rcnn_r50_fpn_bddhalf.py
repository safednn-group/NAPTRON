_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    './bdd100k_cocofmt_half.py',
    '../_base_/runtimes/default_runtime.py',
    '../_base_/schedules/default_schedule.py'

]
model = dict(roi_head=dict(bbox_head=dict(num_classes=4)))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
