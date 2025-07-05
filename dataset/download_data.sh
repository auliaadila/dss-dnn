#!/usr/bin/env bash
# Download librispeech-clean

set -euo pipefail

ROOT="LibriSpeech"          # destination folder
mkdir -p "$ROOT"
cd "$ROOT"

BASE_URL="https://www.openslr.org/resources/12"

# plain indexed arrays – same order in both
SPLIT_KEYS=(
    train-clean-100
    dev-clean
    test-clean
    )
SPLIT_FILES=(
  train-clean-100.tar.gz
  dev-clean.tar.gz
  test-clean.tar.gz
)

download_if_missing () {
  local url=$1
  local file=$2
  if [[ -f $file ]]; then
    echo " $file already present – skip download"
  else
    echo " Downloading $file …"
    wget -c "$url/$file"
  fi
}

extract_and_flatten () {
  local tarfile=$1
  local target=$2
  mkdir -p "$target"

  echo "Extracting $tarfile …"
  tmp=$(mktemp -d)
  tar -xzf "$tarfile" -C "$tmp"

  echo " Moving .flac files → $target/"
  find "$tmp" -type f -name '*.flac' -exec mv -t "$target" {} +

  rm -rf "$tmp"
}

# ---------------- download -----------------
for ((i=0;i<${#SPLIT_KEYS[@]};i++)); do
  download_if_missing "$BASE_URL" "${SPLIT_FILES[$i]}"
done

# ---------------- extract ------------------
for ((i=0;i<${#SPLIT_KEYS[@]};i++)); do
  extract_and_flatten "${SPLIT_FILES[$i]}" "${SPLIT_KEYS[$i]}"
done

echo " Finished.  Created $(printf '%s ' \"${SPLIT_KEYS[@]}\") directories."