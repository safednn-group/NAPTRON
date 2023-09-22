#!/bin/bash
echo Evaluating algorithm from config file: config/benchmark/voc2coco/faster_rcnn_gmm_voc2coco.py
if [ ! -f work_dirs/outputs_dump/faster_rcnn_r50_fpn_voc0712_cocofmt_gmm_trainset_model_outputs_associated.pkl ]; then
    python safednn_naptron/utils/test.py config/gmm/faster_rcnn_r50_fpn_voc0712_cocofmt_gmm_trainset.py work_dirs/faster_rcnn_r50_fpn_voc0712_cocofmt_gmm/latest.pth --eval bbox;
    python scripts/gmm/associate_data.py work_dirs/outputs_dump/faster_rcnn_r50_fpn_voc0712_cocofmt_gmm_trainset_model_outputs.pkl config/gmm/faster_rcnn_r50_fpn_voc0712_cocofmt_gmm_trainset.py -train;
fi

if [ ! -f work_dirs/outputs_dump/faster_rcnn_r50_fpn_voc0712_cocofmt_gmm_model_outputs_associated.pkl ]; then
    python safednn_naptron/utils/test.py config/gmm/faster_rcnn_r50_fpn_voc0712_cocofmt_gmm.py work_dirs/faster_rcnn_r50_fpn_voc0712_cocofmt_gmm/latest.pth --eval bbox;
    python scripts/gmm/associate_data.py work_dirs/outputs_dump/faster_rcnn_r50_fpn_voc0712_cocofmt_gmm_model_outputs.pkl config/gmm/faster_rcnn_r50_fpn_voc0712_cocofmt_gmm.py ;
fi

if [ ! -f work_dirs/outputs_dump/faster_rcnn_gmm_voc2coco_model_outputs_uncertainty.pkl ]; then
  python scripts/gmm/uncertainty.py work_dirs/outputs_dump/faster_rcnn_r50_fpn_voc0712_cocofmt_gmm_trainset_model_outputs_associated.pkl work_dirs/outputs_dump/faster_rcnn_r50_fpn_voc0712_cocofmt_gmm_model_outputs_associated.pkl work_dirs/outputs_dump/faster_rcnn_gmm_voc2coco_model_outputs.pkl work_dirs/gmms/faster_rcnn_voc_cocofmt_gmms.pkl -use-fit;
fi

if [ ! -f work_dirs/outputs_dump/faster_rcnn_gmm_voc2coco_model_outputs_uncertainty_certainties.pkl ]; then
  python eval.py work_dirs/outputs_dump/faster_rcnn_gmm_voc2coco_model_outputs_uncertainty.pkl config/benchmark/voc2coco/faster_rcnn_gmm_voc2coco.py config/voc0712_cocofmt/faster_rcnn_r50_fpn_voc0712_cocofmt.py;
fi