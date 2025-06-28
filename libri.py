#!/usr/bin/env python3
"""make_subset.py – create a ≈60-hour LibriSpeech subset
-------------------------------------------------------
• Walks through <root>/{train-clean-100,train-other-500}/reader/chapter/*.flac
• Randomly selects files until 60 h ±1 % reached
• Converts each FLAC to either 16-kHz mono WAV **or** raw 16-bit PCM (*.s16)
  (choice via --format {wav,s16})
• Writes all files into a flat output directory + a manifest CSV

# 60-hour subset as raw 16-bit PCM
python make_subset.py  /data/LibriSpeech  /data/Libri60h_s16  --format s16

"""
import argparse, csv, os, random, warnings
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_HOURS  = 60.0
TOLERANCE_H   = 0.60          # ±1 %  (0.6 h)
FS_OUT        = 16_000

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
# def collect_flacs(root: Path, splits=("train-clean-100", "train-other-500")):
def collect_flacs(root: Path, splits=("train-clean-100")):
    flacs = []
    for s in splits:
        flacs.extend((root / s).rglob("*.flac"))
    return sorted(flacs)

def flac_duration(path: Path):
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate

def _load_resample_mono(path: Path):
    audio, fs = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if fs != FS_OUT:
        import librosa
        audio = librosa.resample(audio, orig_sr=fs, target_sr=FS_OUT)
    return audio

def flac_to_wav(src: Path, dst: Path):
    sf.write(dst, _load_resample_mono(src), FS_OUT)

def flac_to_s16(src: Path, dst: Path):
    pcm = _load_resample_mono(src)
    pcm16 = np.clip(pcm * 32768.0, -32768, 32767).astype("<i2")
    with open(dst, "wb") as f:
        pcm16.tofile(f)

# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------

def main(src_root: Path, dst_root: Path, fmt: str):
    random.seed(42)
    flacs = collect_flacs(src_root)
    if not flacs:
        raise SystemExit(f"No .flac files found under {src_root}")
    random.shuffle(flacs)

    chosen, tot_sec = [], 0.0
    for f in flacs:
        dur = flac_duration(f)
        if tot_sec + dur > (TARGET_HOURS + TOLERANCE_H) * 3600:
            continue
        chosen.append((f, dur))
        tot_sec += dur
        if tot_sec >= (TARGET_HOURS - TOLERANCE_H) * 3600:
            break

    print(f"Selected {len(chosen)} files → {tot_sec/3600:.2f} h")

    dst_root.mkdir(parents=True, exist_ok=True)
    for src, _ in chosen:
        base = src.stem     # 19-198-0001
        if fmt == "wav":
            flac_to_wav(src, dst_root / f"{base}.wav")
        else:
            flac_to_s16(src, dst_root / f"{base}.s16")

    with open(dst_root / "manifest_60h.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["file", "seconds"])
        ext = ".wav" if fmt == "wav" else ".s16"
        for src, dur in chosen:
            wr.writerow([src.stem + ext, f"{dur:.2f}"])

    print("Done →", dst_root)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("librispeech_root", type=Path,
                   help="LibriSpeech root dir (contains train-clean-100/ …)")
    p.add_argument("output_dir", type=Path,
                   help="Destination folder for 60 h subset")
    p.add_argument("--format", choices=["wav", "s16"], default="wav",
                   help="Output audio format (default: wav)")
    args = p.parse_args()

    main(args.librispeech_root.expanduser(),
         args.output_dir.expanduser(),
         args.format)