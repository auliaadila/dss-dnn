#!/usr/bin/env bash
# ------------------------------------------------------------
# Launch training for every (spread, alpha, resid) combo.
# Each run gets its own descriptive model name and its own
# TensorBoard log folder.
#
# USAGE:
#   ./train.sh  /path/to/train  /path/to/val  [GPU_ID]
#   ./train.sh  ./dataset/Libri2h/train  ./dataset/Libri2h/val  0
# ------------------------------------------------------------

set -euo pipefail     # safer bash

TRAIN_DIR=$1          # e.g. ./dataset/Libri2h/train
VAL_DIR=$2            # e.g. ./dataset/Libri2h/val
GPU_ID=${3:-0}        # default GPU 0

mkdir -p checkpoints logs

PY=python3            # adjust if you use a venv
SCRIPT=train.py       # <- name of the main script

# Hyper-params shared by every run ----------------------------------------
COMMON_ARGS="
  --train     ${TRAIN_DIR}
  --val       ${VAL_DIR}
  --frames    15
  --bits      64
  --batch     32
  --epochsA   8
  --epochsB   12
  --lporder   12
  --stage     AB
"

# Enumerate all combinations ----------------------------------------------
SPREAD=(dss lpdss)
ALPHA=(constant adaptive)
RESID=(sign whiten)           # only used for LP-DSS

for s in "${SPREAD[@]}"; do
  for a in "${ALPHA[@]}"; do

    # choose appropriate residual loop
    if [[ "$s" == "lpdss" ]]; then
      resid_list=("${RESID[@]}")
    else
      resid_list=("sign")     # dummy value – will be ignored by DSS
    fi

    for r in "${resid_list[@]}"; do
      MODEL_NAME="${s}_${a}_${r}"

      echo -e "\n============================================================"
      echo   "Training model: ${MODEL_NAME}"
      echo   "  spread=${s}  alpha=${a}  resid=${r}"
      echo   "============================================================"

      CUDA_VISIBLE_DEVICES=${GPU_ID} \
      ${PY} ${SCRIPT} \
          ${COMMON_ARGS} \
          --spread       "${s}" \
          --alphaset     "${a}" \
          --residmethod  "${r}" \
          --name         "${MODEL_NAME}" \
          2>&1 | tee "logs/${MODEL_NAME}.log"
    done
  done
done