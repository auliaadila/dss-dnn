#!/usr/bin/env bash
# GPU-optimized training script for LP-DSS

python train_gpu.py \
    --train   ./dataset/Libri2h/train \
    --val     ./dataset/Libri2h/val \
    --stage   AB              \
    --frames  15              \
    --bits    64              \
    --batch   32              \
    --epochsA 15              \
    --epochsB 30 \
    --lporder 12 \
    --name "proto1h" | tee log-gpu.txt
