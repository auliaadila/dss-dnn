MODEL_NAME="lptest"

python trainlp_test.py \
  --train ./dataset/Libri1h/train \
  --val   ./dataset/Libri1h/val \
  --frames 15 \
  --bits   64 \
  --batch  32 \
  --name "${MODEL_NAME}"
  2>&1 | tee "logslp/${MODEL_NAME}.log"