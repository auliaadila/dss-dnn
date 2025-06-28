#!/usr/bin/env python3
# make_subset.py ---------------------------------------------------------
# Usage:  python make_subset.py  /path/LIBRISPEECH  /path/target_60h
# -----------------------------------------------------------------------

import os, random, shutil, argparse, csv
from pathlib import Path
import soundfile as sf              # pip install soundfile

TARGET_HOURS = 60.0
TOLERANCE    = 0.60                 # ±1 %   (0.6 h)

def collect_wavs(root: Path):
    """Return list of all .wav files under root."""
    return sorted([p for p in root.rglob("*.wav")])

def wav_duration(path: Path):
    """Return duration in seconds without loading whole file."""
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate

def main(src_root: Path, dst_root: Path):
    splits = ["train-clean-100", "train-other-500"]
    random.seed(42)

    # ------------------------------------------------------------------
    # 1. collect candidate wavs & shuffle
    # ------------------------------------------------------------------
    candidates = []
    for s in splits:
        candidates += collect_wavs(src_root / s)

    if not candidates:
        raise RuntimeError("No .wav files found under", src_root)

    random.shuffle(candidates)

    # ------------------------------------------------------------------
    # 2. sample until target duration reached
    # ------------------------------------------------------------------
    chosen, tot_sec = [], 0.0
    for wav in candidates:
        dur = wav_duration(wav)
        if tot_sec + dur > (TARGET_HOURS + TOLERANCE) * 3600:
            continue
        chosen.append((wav, dur))
        tot_sec += dur
        if tot_sec >= (TARGET_HOURS - TOLERANCE) * 3600:
            break

    hrs = tot_sec / 3600
    print(f"Selected {len(chosen)} files, total {hrs:.2f} h")

    # ------------------------------------------------------------------
    # 3. copy into flat dst folder  (keeps original speakerID-uttID name)
    # ------------------------------------------------------------------
    dst_root.mkdir(parents=True, exist_ok=True)
    for wav, _ in chosen:
        shutil.copy(wav, dst_root / wav.name)

    # optional: write a manifest CSV (filepath,duration_sec)
    with open(dst_root / "manifest_60h.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["wav", "seconds"])
        for wav, dur in chosen:
            wr.writerow([wav.name, f"{dur:.2f}"])

    print("Done →", dst_root)

# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("librispeech_root", type=Path,
                        help="root dir that contains train-clean-100 / ...")
    parser.add_argument("output_dir",       type=Path,
                        help="where to copy the 60 h subset")
    args = parser.parse_args()

    main(args.librispeech_root, args.output_dir)