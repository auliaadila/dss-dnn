#!/usr/bin/env bash
set -euo pipefail

# Move to the directory this script is in (tools/)
cd "$(dirname "$0")"

# Now ./embedder.py is right here
python embedder.py \
    --in "../dataset/host/19-198-0000.wav" \
    --out "../dataset/wm_out/WM_19-198-0000.wav" \
    --wm 0123456789ABCDEF \
    --alpha 0.00005

python extract.py \
    --in "../dataset/host/19-198-0000.wav" \
    --model "../checkpoints/proto1h_stageA.h5" \
    --wm 0123456789ABCDEF