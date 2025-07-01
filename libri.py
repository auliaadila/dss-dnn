#!/usr/bin/env python3
"""make_subset.py – build train/val/test subsets from LibriSpeech archives
---------------------------------------------------------------------------
* Auto‑detects `<root>/<split>.tar.gz`, extracts if the split folder is
  missing, then collects the `.flac`s.
* Converts to mono 16‑kHz **WAV** or **.s16** (raw int16) via `--format`.
* Picks exactly `--train-minutes`,  `--val-minutes`, `--test-minutes`.
* Outputs three flat dirs (`train/`, `val/`, `test/`) + manifest CSVs.
"""
import argparse, csv, random, tarfile, warnings
from pathlib import Path
import os

import numpy as np
import soundfile as sf

FS_OUT = 16_000
RNG_SEED = 42
SPLITS = {
    "train": "train-clean-100",
    "val"  : "dev-clean",
    "test" : "test-clean",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_split_extracted(root: Path, split_name: str):
    """If `<split>.tar.gz` exists and folder is absent, extract it."""
    folder = root / split_name
    if folder.exists():
        return folder
    tar_path = root / f"{split_name}.tar.gz"
    if not tar_path.exists():
        raise FileNotFoundError(f"Missing {folder} and {tar_path}")
    print(f"Extracting {tar_path} …")
    with tarfile.open(tar_path) as tar:
        def is_within_directory(directory, target):
            import os
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            prefix = os.path.commonprefix([abs_directory, abs_target])
            return prefix == abs_directory
        def safe_extract(tar_obj, path="."):
            for member in tar_obj.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
            tar_obj.extractall(path)
        safe_extract(tar, root)
    return folder


def collect_flacs(root: Path, split_key: str):
    split_dir = ensure_split_extracted(root, SPLITS[split_key])
    return sorted(split_dir.rglob("*.flac"))


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

# ---------------------------------------------------------------------------
# Sampling routine
# ---------------------------------------------------------------------------

def sample_split(flacs, minutes_target, fmt, out_dir: Path):
    random.shuffle(flacs)
    goal_sec = minutes_target * 60
    chosen, total = [], 0.0
    for f in flacs:
        d = flac_duration(f)
        if total + d > goal_sec:
            continue
        chosen.append((f, d))
        total += d
        if total >= goal_sec:
            break
    if total < 0.99 * goal_sec:
        warnings.warn(f"Only {total/60:.1f} min collected vs {minutes_target} min")

    out_dir.mkdir(parents=True, exist_ok=True)
    to_fn = flac_to_wav if fmt == "wav" else flac_to_s16
    ext   = ".wav" if fmt == "wav" else ".s16"
    for src, _ in chosen:
        to_fn(src, out_dir / (src.stem + ext))

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        csv.writer(f).writerows([["file","seconds"]] +
                                [[src.stem + ext, f"{d:.2f}"] for src,d in chosen])
    print(f"{out_dir.name}: {total/60:.1f} min  ({len(chosen)} files)")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("librispeech_root", type=Path)
    arg.add_argument("output_dir",       type=Path)
    arg.add_argument("--train-minutes", type=float, default=60.0)
    arg.add_argument("--val-minutes",   type=float, default=4.0)
    arg.add_argument("--test-minutes",  type=float, default=4.0)
    arg.add_argument("--format", choices=["wav","s16"], default="wav")
    cfg = arg.parse_args()

    random.seed(RNG_SEED)
    root = cfg.librispeech_root.expanduser()
    out  = cfg.output_dir.expanduser(); out.mkdir(parents=True, exist_ok=True)

    sample_split(collect_flacs(root, "train"), cfg.train_minutes, cfg.format, out/"train")
    sample_split(collect_flacs(root, "val"  ), cfg.val_minutes,   cfg.format, out/"val")
    sample_split(collect_flacs(root, "test" ), cfg.test_minutes,  cfg.format, out/"test")

    print("All splits done→", out)
