MODEL_NAME="lptest"

python trainlp.py \
  --train ./dataset/Libri1h/train \
  --val   ./dataset/Libri1h/val \
  --alphaset adaptive \
  --frames 15 \
  --bits   64 \
  --batch  32 \
  --name "${MODEL_NAME}"
  2>&1 | tee "logslp/${MODEL_NAME}.log"