#!/bin/bash
echo Evaluating algorithm from config file: config/benchmark/bdd/faster_rcnn_owod_bdd.py
if [ ! -f work_dirs/outputs_dump/faster_rcnn_owod_bdd_model_outputs_uncertainty_certainties.pkl ]; then
  python safednn_naptron/utils/train.py config/owod/faster_rcnn_r50_fpn_bddhalf_owod_val.py --no-validate;

  python scripts/owod/analyze_energy_scores.py config/owod/faster_rcnn_r50_fpn_bddhalf_owod_val.py work_dirs/outputs_dump/faster_rcnn_owod_bdd_model_outputs.pkl -use-fit;

  python eval.py work_dirs/outputs_dump/faster_rcnn_owod_bdd_model_outputs_uncertainty.pkl config/benchmark/bdd/faster_rcnn_owod_bdd.py config/owod/faster_rcnn_r50_fpn_bddhalf_owod.py;
fi