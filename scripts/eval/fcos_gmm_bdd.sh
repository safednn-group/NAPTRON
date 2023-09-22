#!/bin/bash

echo Evaluating algorithm from config file: config/benchmark/bdd/fcos_gmm_bdd.py

if [ ! -f work_dirs/outputs_dump/fcos_r50_fpn_bddhalf_gmm_trainset_model_outputs_associated.pkl ]; then
    python safednn_naptron/utils/test.py config/gmm/fcos_r50_fpn_bddhalf_gmm_trainset.py work_dirs/fcos_r50_fpn_bddhalf_gmm/latest.pth --eval bbox;
    python scripts/gmm/associate_data.py work_dirs/outputs_dump/fcos_r50_fpn_bddhalf_gmm_trainset_model_outputs.pkl config/gmm/fcos_r50_fpn_bddhalf_gmm_trainset.py -train;
fi

if [ ! -f work_dirs/outputs_dump/fcos_r50_fpn_bddhalf_gmm_model_outputs_associated.pkl ]; then
    python safednn_naptron/utils/test.py config/gmm/fcos_r50_fpn_bddhalf_gmm.py work_dirs/fcos_r50_fpn_bddhalf_gmm/latest.pth --eval bbox;
    python scripts/associate_data.py work_dirs/outputs_dump/fcos_r50_fpn_bddhalf_gmm_model_outputs.pkl config/gmm/fcos_r50_fpn_bddhalf_gmm.py ;
fi

if [ ! -f work_dirs/outputs_dump/fcos_gmm_bdd_model_outputs_uncertainty.pkl ]; then
  python scripts/gmm/uncertainty.py work_dirs/outputs_dump/fcos_r50_fpn_bddhalf_gmm_trainset_model_outputs_associated.pkl work_dirs/outputs_dump/fcos_r50_fpn_bddhalf_gmm_model_outputs_associated.pkl work_dirs/outputs_dump/fcos_gmm_bdd_model_outputs.pkl work_dirs/gmms/fcos_bddhalf_gmms.pkl -use-fit;
fi
if [ ! -f work_dirs/outputs_dump/fcos_gmm_bdd_model_outputs_uncertainty_certainties.pkl ]; then
  python eval.py work_dirs/outputs_dump/fcos_gmm_bdd_model_outputs_uncertainty.pkl config/benchmark/bdd/fcos_gmm_bdd.py config/bdd/faster_rcnn_r50_fpn_bdd100k_half.py;
fi