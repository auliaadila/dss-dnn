#!/usr/bin/env python3
# inference_demo.py (for 1 audio) --------------------------------------------------------
# * loads the end-to-end LP-DSS model (Stage-B weights)
# * embeds one 64-bit payload word into EVERY 2 400-sample window
# * saves the water-marked WAV
# * immediately extracts the bits and prints BER
# --------------------------------------------------------------------------

import argparse
import os
import sys
import numpy as np
import soundfile as sf
import tensorflow as tf
from pathlib import Path

# ---------- import your custom layers ------------------------------------
from components.residualembedding import ResidualEmbedding
from components.extractor          import DSSExtractor
from components.attacks            import AttackPipeline, AdditiveNoise, CuttingSamples, ButterworthIIR
from components.dataloader         import FrameSequencePayload               # helper only
from components.lossfunc             import bit_consistency_loss               # not used, but lets TF resolve
from components.metrics            import BitErrorRate                       # not used

CUSTOM = {  # for load_model()
    "ResidualEmbedding": ResidualEmbedding,
    "AttackPipeline"   : AttackPipeline,
    "DSSExtractor"     : DSSExtractor,
    "AdditiveNoise"    : AdditiveNoise,
    "CuttingSamples"   : CuttingSamples,
    "ButterworthIIR"   : ButterworthIIR,
    "bit_consistency_loss": bit_consistency_loss,
    "BitErrorRate"        : BitErrorRate,
}

# -------------------------------------------------------------------------
def slice_windows(x, win):
    """Split 1-D signal into ⌊N/W⌋ windows (discard tail)."""
    n_win = len(x) // win
    return x[: n_win * win].reshape(n_win, win)

def main(args):
    fs          = 16_000
    frame_size  = 160
    K           = args.frames                      # windows = 15 frames
    T           = frame_size * K          # 2 400 samples
    P           = args.bits

    # ---------- 1. load host audio ---------------------------------------
    wav_in, sr = sf.read(args.wav_in, always_2d=False)
    if sr != fs:
        sys.exit(f"Expected {fs} Hz WAV but got {sr} Hz.")
    wav_in = wav_in.astype("float32")
    if wav_in.ndim == 2:                  # stereo → mono
        wav_in = wav_in.mean(axis=1)
    windows = slice_windows(wav_in, T)    # shape (N, 2400)
    n_win   = windows.shape[0]
    if n_win == 0:
        sys.exit("Audio shorter than one 2 400-sample window.")
    
    print("Wav in shape:", windows.shape) #Wav in shape: (16, 2400)

    # ---------- 2. prepare random payload bits ---------------------------
    rng   = np.random.default_rng(42)
    bits  = rng.integers(0, 2, size=(n_win, P), dtype=np.uint8).astype("float32") #same for each n_win?
    print("Bits in shape:", bits.shape) #Bits in shape: (16, 64)
    print(bits)

    # ---------- 3. load model & sub-graphs -------------------------------
    full   = tf.keras.models.load_model(args.model, custom_objects=CUSTOM,
                                        compile=False) #stage B

    embedder  = full.get_layer("embedder")
    extractor = full.get_layer("extractor")

    # ---------- 4. embedding loop  ---------------------------------------
    wm_windows = []
    for win_idx in range(n_win):
        pcm  = windows[win_idx][None, :, None]          # (1,2400,1)
        word = bits[win_idx][None, :]                  # (1,64)
        wm   = embedder([word, pcm], training=False)
        wm_ = wm.numpy().squeeze()
        # print("wm_ shape per win_idx:", wm_.shape) #wm_ shape per win_idx: (2400,)
        wm_windows.append(wm_)
    
    print("wm_windows shape:", len(wm_windows))

    wav_wm = np.concatenate(wm_windows, axis=0)
    sf.write(args.wav_out, wav_wm, fs)
    print(f"✅ Water-marked file written → {args.wav_out}")

    # ---------- 5. extraction  -------------------------------------------
    preds = []
    for win_idx in range(n_win):
        pcm_wm = wm_windows[win_idx][None, :, None]    # (1,2400,1)
        pred   = extractor(pcm_wm, training=False).numpy().squeeze()
        pred_ = (pred > 0.5).astype(np.float32)
        preds.append(pred_)

    preds = np.vstack(preds)
    ber   = np.mean(np.not_equal(bits, preds))
    print(f"BER on {n_win} windows: {ber:.4f}")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", required=True, type=Path,
                    help="Stage-B .h5 checkpoint")
    pa.add_argument("--wav_in",   required=True, type=Path,
                    help="Clean host WAV (16-kHz mono)")
    pa.add_argument("--wav_out",   required=True, type=Path,
                    help="Output water-marked WAV path")
    pa.add_argument(
        "--frames", type=int, default=15, help="frames per payload window (K)"
    )
    pa.add_argument(
        "--bits", type=int, default=64, help="payload length (P) per window"
    )
    args = pa.parse_args()
    main(args)