#!/bin/bash
echo Evaluating algorithm from config file: config/benchmark/bdd/retinanet_naptron_bdd.py

if [ ! -f work_dirs/outputs_dump/retinanet_r50_fpn_bddhalf_naptron_trainset_model_outputs.pkl ]; then
  python safednn_naptron/utils/test.py config/naptron/retinanet_r50_fpn_bddhalf_naptron_trainset.py work_dirs/retinanet_r50_fpn_bdd100k_half/latest.pth --override-batch-size --eval bbox;
fi

if [ ! -f work_dirs/outputs_dump/retinanet_naptron_bdd_model_outputs_uncertainty.pkl ]; then
  python scripts/naptron/compute_uncertainty_bboxes.py work_dirs/outputs_dump/retinanet_r50_fpn_bddhalf_naptron_trainset_model_outputs.pkl config/naptron/retinanet_r50_fpn_bddhalf_naptron_trainset.py work_dirs/outputs_dump/retinanet_naptron_bdd_model_outputs.pkl 3;
fi

if [ ! -f work_dirs/outputs_dump/retinanet_naptron_bdd_model_outputs_uncertainty_certainties.pkl ]; then
  python eval.py work_dirs/outputs_dump/retinanet_naptron_bdd_model_outputs_uncertainty.pkl config/benchmark/bdd/retinanet_naptron_bdd.py config/bdd/faster_rcnn_r50_fpn_bdd100k_half.py;
fi