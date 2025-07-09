#!/usr/bin/env bash
# run_train.sh – two-phase LP-DSS training

# python train.py \
#     --train   ./dataset/dummy/train/ \
#     --val     ./dataset/dummy/train \
#     --stage   AB              \
#     --frames  15              \
#     --bits    64              \
#     --batch   32              \
#     --epochsA 8              \
#     --epochsB 12 \
#     --lporder 12 \
#     --name "proto1h" > proto1h_040725.log

python train.py \
    --train   ./dataset/Libri1h/train \
    --val     ./dataset/Libri1h/val \
    --stage   AB              \
    --frames  15              \
    --bits    64              \
    --batch   32              \
    --epochsA 8              \
    --epochsB 12 \
    --lporder 12 \
    --name "proto1h" | tee log.txt
