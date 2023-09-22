NAP usage

1. Generate bboxes and corresponding activations for training set images (only those that had been used for training the detector that is being evaluated; not the entire original training dataset). It can be done using config/naptron/faster_rcnn_r50_fpn_1x_voc0712_cocofmt_naptron_trainset.py configuration file after correctly specifying data_root and ann_file path.
2. Code below `if __name__ == '__main__':` in naptron/compute_uncertainty_bboxes.py can be used to perform whole process of computing NAPTRON uncertainty and appending it to each bbox in a detector output list. One needs to provide the following args:
    parser.add_argument('train', help='train results path')
    parser.add_argument('train_config', help='train config path')
    parser.add_argument('test', help='test results path')

for example:
train_config = config/naptron/faster_rcnn_r50_fpn_1x_voc0712_cocofmt_naptron_trainset.py
train = work_dirs/outputs_dump/faster_rcnn_r50_fpn_1x_voc0712_cocofmt_naptron_trainset_model_outputs.pkl
test = work_dirs/outputs_dump/`name of the config file that were used for generating test results`_model_outputs.pkl

Notes: trainset annotations must be in COCO format (json file); generated NAPTRON results may take a lot of disc space (couple of GBs); preparing the NAPTRON monitor takes couple of minutes.

Full working example:

1. Generate bboxes and activations for test samples (change paths to dataset annotation file and model checkpoint if needed):

`python safednn/utils/test.py config/benchmark/voc2coco/faster_rcnn_naptron_voc2coco.py work_dirs/faster_rcnn_r50_fpn_1x_voc0712_cocofmt/latest.pth --eval bbox`

The generated bboxes and activations by default are stored in work_dirs/outputs_dump/faster_rcnn_naptron_voc2coco_model_ouputs.pkl

2. Generate bboxes and activations for training samples

`python safednn_naptron/utils/test.py config/naptron/faster_rcnn_r50_fpn_1x_voc0712_cocofmt_naptron_trainset.py work_dirs/faster_rcnn_r50_fpn_1x_voc0712_cocofmt/latest.pth --eval bbox`

3. Compute hamming distances for all test samples generated during the previous step.
(uncertainty score is appended to every bbox vector)

`python nap/compute_uncertainty_bboxes.py work_dirs/outputs_dump/faster_rcnn_r50_fpn_1x_voc0712_cocofmt_naptron_trainset_model_outputs.pkl config/naptron/faster_rcnn_r50_fpn_1x_voc0712_cocofmt_naptron_trainset.py work_dirs/outputs_dump/faster_rcnn_naptron_voc2coco_model_outputs.pkl 2 -mean;`

The resulting bboxes with appended uncertainty scores are stored in work_dirs/outputs_dump/faster_rcnn_naptron_voc2coco_model_outputs_uncertainty.pkl
