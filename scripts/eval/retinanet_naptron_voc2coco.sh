#!/bin/bash
echo Evaluating algorithm from config file: config/benchmark/voc2coco/retinanet_naptron_voc2coco.py

if [ ! -f work_dirs/outputs_dump/retinanet_r50_fpn_voc0712_cocofmt_naptron_trainset_model_outputs.pkl ]; then
  python safednn_naptron/utils/test.py config/naptron/retinanet_r50_fpn_voc0712_cocofmt_naptron_trainset.py work_dirs/retinanet_r50_fpn_voc0712_cocofmt/latest.pth --override-batch-size --eval bbox;
fi

if [ ! -f work_dirs/outputs_dump/retinanet_naptron_voc2coco_model_outputs_uncertainty.pkl ]; then
  python scripts/naptron/compute_uncertainty_bboxes.py work_dirs/outputs_dump/retinanet_r50_fpn_voc0712_cocofmt_naptron_trainset_model_outputs.pkl config/naptron/retinanet_r50_fpn_voc0712_cocofmt_naptron_trainset.py work_dirs/outputs_dump/retinanet_naptron_voc2coco_model_outputs.pkl 3;
fi

if [ ! -f work_dirs/outputs_dump/retinanet_naptron_voc2coco_model_outputs_uncertainty_certainties.pkl ]; then
  python eval.py work_dirs/outputs_dump/retinanet_naptron_voc2coco_model_outputs_uncertainty.pkl config/benchmark/voc2coco/retinanet_naptron_voc2coco.py config/voc0712_cocofmt/faster_rcnn_r50_fpn_voc0712_cocofmt.py;
fi