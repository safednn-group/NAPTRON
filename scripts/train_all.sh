#!/bin/bash

#run this script from project root dir

declare -a dirs=("config/voc0712_cocofmt" "config/bdd" "config/gmm"
 "config/gaussian_detectors" "config/vos" "config/oln/" "config/owod/" "config/open_det/")

# Train all methods
for dir in "${dirs[@]}"; do
  for file in `find $dir -name "*.py" -type f -not \( -path "config/oln/oln_box.py" -prune \) -not \( -path "config/owod/*val*" -prune \) -not \( -path "config/anchor_gmm/*trainset*" -prune \) -not \( -path "config/bdd/*cocofmt*" -prune \)`; do
    echo " training: $file"
    python safednn_naptron/utils/train.py $file --no-validate
  done
done

