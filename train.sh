#!/usr/bin/env bash
# run_train.sh – two-phase LP-DSS training

python train.py \
    --train   /home/adila/Data/audio/Libri1h/train \
    --val     /home/adila/Data/audio/Libri1h/val \
    --stage   AB              \
    --frames  15              \
    --bits    64              \
    --batch   32              \
    --epochsA 8              \
    --epochsB 12 \
    --lporder 12 \
    --name "proto1h" > proto1h_040725.log
