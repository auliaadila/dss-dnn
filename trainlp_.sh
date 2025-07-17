MODEL_NAME="lptest"

python trainlp_.py \
  --train ./dataset/Libri1h/train \
  --val   ./dataset/Libri1h/val \
  --stage AB \
  --alphaset adaptive \
  --residsign True \
  --frames 15 \
  --bits   64 \
  --batch  32 \
  --name "${MODEL_NAME}"
  2>&1 | tee "logslp/${MODEL_NAME}.log"