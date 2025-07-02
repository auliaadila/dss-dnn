#!/usr/bin/env bash
# run_train.sh – two-phase LP-DSS training

python train.py \
    --train   /home/adila/Data/research/dss-dnn/dataset/dummy/train \
    --val     /home/adila/Data/research/dss-dnn/dataset/dummy/val \
    --phase   AB              \
    --frames  100              \
    --bits    8              \
    --batch   32              \
    --epochsA 2              \
    --epochsB 3 > dummy_030725.log