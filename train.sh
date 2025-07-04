#!/usr/bin/env bash
# run_train.sh – two-phase LP-DSS training

python train.py \
    --train   dataset/dummy/train \
    --val     dataset/dummy/val \
    --stage   AB              \
    --frames  100              \
    --bits    8              \
    --batch   32              \
    --epochsA 3              \
    --epochsB 4 \
    --lporder 12 \
    --name "dummy1" > dummy_040725.log
