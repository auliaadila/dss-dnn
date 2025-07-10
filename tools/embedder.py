#!/usr/bin/env python3
import argparse
import binascii
import os
import sys

import numpy as np
import soundfile as sf
import tensorflow as tf

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from components.residualembedding_gpu import ResidualEmbeddingGPU

FRAME = 160
K = 15
SEG_LEN = FRAME * K  # 2400
P = 64  # bits per payload
FS = 16_000


def hex_to_bits(hex_str):
    """Convert 16-digit hex → 64-element numpy array of 0/1."""
    value = int(hex_str, 16)
    bits = [(value >> (63 - i)) & 1 for i in range(P)]
    return np.array(bits, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Watermark a WAV (classic model)")
    ap.add_argument("--in", required=True, dest="inp", help="input wav")
    ap.add_argument("--out", required=True, dest="out", help="output watermarked wav")
    ap.add_argument("--wm", dest="wm", help="16-digit hex watermark")
    ap.add_argument("--alpha", type=float, default=0.5, help="embedding strength")
    args = ap.parse_args()

    pcm, sr = sf.read(args.inp, dtype="float32")
    assert sr == FS, "sample-rate harus 16 kHz"
    pcm = pcm.reshape(-1, 1)
    n_samples = len(pcm)

    if args.wm:
        assert len(args.wm) == 16, "hex string harus 16 digit (64 bit)"
        bits = hex_to_bits(args.wm)
    else:
        bits = np.random.randint(0, 2, P).astype(np.float32)

    embedder = ResidualEmbeddingGPU(frame_size=FRAME, K=K, P=P, trainable_alpha=False)
    embedder.set_alpha_value(args.alpha)

    out = np.zeros_like(pcm)
    for start in range(0, n_samples, SEG_LEN):
        seg = pcm[start : start + SEG_LEN]
        if len(seg) < SEG_LEN:
            break
        seg_tf = tf.constant(seg[None, :, :])  # (1,T,1)
        bits_tf = tf.constant(bits[None, :])  # (1,64)
        wm_seg = embedder([bits_tf, seg_tf]).numpy()[0]
        out[start : start + SEG_LEN] = wm_seg

    sf.write(args.out, out.squeeze(), sr)
    print("Watermarked saved to", args.out)


if __name__ == "__main__":
    main()
