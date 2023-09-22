#!/bin/bash
echo Evaluating algorithm from config file: config/benchmark/bdd/faster_rcnn_naptron_bdd.py
if [ ! -f work_dirs/outputs_dump/faster_rcnn_r50_fpn_bddhalf_naptron_trainset_model_outputs.pkl ]; then
  python safednn_naptron/utils/test.py config/naptron/faster_rcnn_r50_fpn_bddhalf_naptron_trainset.py work_dirs/faster_rcnn_r50_fpn_bdd100k_half/latest.pth --eval bbox --override-batch-size;
fi

if [ ! -f work_dirs/outputs_dump/faster_rcnn_naptron_bdd_model_outputs_uncertainty.pkl ]; then
  python scripts/naptron/compute_uncertainty_bboxes.py work_dirs/outputs_dump/faster_rcnn_r50_fpn_bddhalf_naptron_trainset_model_outputs.pkl config/naptron/faster_rcnn_r50_fpn_bddhalf_naptron_trainset.py work_dirs/outputs_dump/faster_rcnn_naptron_bdd_model_outputs.pkl 2;
fi
if [ ! -f work_dirs/outputs_dump/faster_rcnn_naptron_bdd_model_outputs_uncertainty_certainties.pkl ]; then
  python eval.py work_dirs/outputs_dump/faster_rcnn_naptron_bdd_model_outputs_uncertainty.pkl config/benchmark/bdd/faster_rcnn_naptron_bdd.py config/bdd/faster_rcnn_r50_fpn_bdd100k_half.py;
fi