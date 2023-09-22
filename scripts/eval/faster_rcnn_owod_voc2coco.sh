#!/bin/bash
echo Evaluating algorithm from config file: config/benchmark/voc2coco/faster_rcnn_owod_voc2coco.py
if [ ! -f work_dirs/outputs_dump/faster_rcnn_owod_voc2coco_model_outputs_uncertainty_certainties.pkl ]; then
  python safednn_naptron/utils/train.py config/owod/faster_rcnn_r50_fpn_voc0712_cocofmt_owod_val.py --no-validate;

  python scripts/owod/analyze_energy_scores.py config/owod/faster_rcnn_r50_fpn_voc0712_cocofmt_owod_val.py work_dirs/outputs_dump/faster_rcnn_owod_voc2coco_model_outputs.pkl -use-fit;

  python eval.py work_dirs/outputs_dump/faster_rcnn_owod_voc2coco_model_outputs_uncertainty.pkl config/benchmark/voc2coco/faster_rcnn_owod_voc2coco.py config/uncertainty/owod/faster_rcnn_r50_fpn_voc0712_cocofmt_owod.py;
fi