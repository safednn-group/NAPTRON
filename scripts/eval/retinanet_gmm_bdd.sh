#!/bin/bash

echo Evaluating algorithm from config file: config/benchmark/bdd/retinanet_gmm_bdd.py

if [ ! -f work_dirs/outputs_dump/retinanet_r50_fpn_bddhalf_gmm_trainset_model_outputs_associated.pkl ]; then
    python safednn_naptron/utils/test.py config/gmm/retinanet_r50_fpn_bddhalf_gmm_trainset.py work_dirs/retinanet_r50_fpn_bddhalf_gmm/latest.pth --eval bbox;
    python scripts/gmm/associate_data.py work_dirs/outputs_dump/retinanet_r50_fpn_bddhalf_gmm_trainset_model_outputs.pkl config/gmm/retinanet_r50_fpn_bddhalf_gmm_trainset.py -train;
fi

if [ ! -f work_dirs/outputs_dump/retinanet_r50_fpn_bddhalf_gmm_model_outputs_associated.pkl ]; then
    python safednn_naptron/utils/test.py config/gmm/retinanet_r50_fpn_bddhalf_gmm.py work_dirs/retinanet_r50_fpn_bddhalf_gmm/latest.pth --eval bbox;
    python scripts/gmm/associate_data.py work_dirs/outputs_dump/retinanet_r50_fpn_bddhalf_gmm_model_outputs.pkl config/gmm/retinanet_r50_fpn_bddhalf_gmm.py ;
fi

if [ ! -f work_dirs/outputs_dump/retinanet_gmm_bdd_model_outputs_uncertainty.pkl ]; then
  python scripts/gmm/uncertainty.py work_dirs/outputs_dump/retinanet_r50_fpn_bddhalf_gmm_trainset_model_outputs_associated.pkl work_dirs/outputs_dump/retinanet_r50_fpn_bddhalf_gmm_model_outputs_associated.pkl work_dirs/outputs_dump/retinanet_gmm_bdd_model_outputs.pkl work_dirs/gmms/retinanet_bddhalf_gmms.pkl -use-fit;
fi

if [ ! -f work_dirs/outputs_dump/retinanet_gmm_bdd_model_outputs_uncertainty_certainties.pkl ]; then
  python eval.py work_dirs/outputs_dump/retinanet_gmm_bdd_model_outputs_uncertainty.pkl config/benchmark/bdd/retinanet_gmm_bdd.py config/bdd/faster_rcnn_r50_fpn_bdd100k_half.py;
fi