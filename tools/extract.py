import argparse
import os
import sys

import numpy as np
import soundfile as sf
import tensorflow as tf

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from components.attack_pipeline_fixed import AttackPipelineFixed
from components.attacks_gpu import (
    AdditiveNoiseGPU,
    ButterworthIIRGPU,
    CuttingSamplesGPU,
    LowpassFIRGPU,
)
from components.extractor_fixed import DSSExtractorFixed
from components.residualembedding_gpu import ResidualEmbeddingGPU
from embedder import hex_to_bits

FS = 16_000  # sample-rate expected by the model
P = 64  # payload bits


# ------------------------------------------------------------
def bits_to_hex(bits_bin):
    """Convert array 64×{0,1} → 16-digit hex string."""
    val = 0
    for b in bits_bin:
        val = (val << 1) | int(b)
    return f"{val:016X}"


def main():
    ap = argparse.ArgumentParser(description="Classic LP-DSS inference")
    ap.add_argument("--in", required=True, dest="inp", help="water-marked wav")
    ap.add_argument("--model", required=True, help="trained .h5 model")
    ap.add_argument("--wm", help="16-digit hex ground truth watermark")
    args = ap.parse_args()

    # -------------------------------------------------------- #
    # 1.  Load Keras model
    # -------------------------------------------------------- #
    custom_objects = {
        "ResidualEmbeddingGPU": ResidualEmbeddingGPU,
        "AttackPipelineFixed": AttackPipelineFixed,
        "LowpassFIRGPU": LowpassFIRGPU,
        "ButterworthIIRGPU": ButterworthIIRGPU,
        "AdditiveNoiseGPU": AdditiveNoiseGPU,
        "CuttingSamplesGPU": CuttingSamplesGPU,
        "DSSExtractorFixed": DSSExtractorFixed,
    }
    model = tf.keras.models.load_model(
        args.model,
        custom_objects=custom_objects,
        compile=False,
    )

    extractor = None
    for layer in model.layers:
        if isinstance(layer, DSSExtractorFixed):
            extractor = layer
            break
    if extractor is None:
        raise RuntimeError("Extractor layer DSSExtractorFixed not found!")

    seg_len = extractor.len  # 2400 sampel (frame_size*K)
    print(f"Extractor context length : {seg_len}")

    pcm, sr = sf.read(args.inp, dtype="float32")
    if sr != FS:
        raise RuntimeError(f"Sample-rate {sr} Hz ≠ {FS} Hz")
    pcm = pcm.reshape(-1, 1)
    n_samples = len(pcm)

    # -------------------------------------------------------- #
    # 3.  Sliding-window inference
    # -------------------------------------------------------- #
    votes = []
    for start in range(0, n_samples - seg_len + 1, seg_len):
        seg = pcm[start : start + seg_len]

        bits_prob = extractor(tf.constant(seg[None, :, :]), training=False).numpy()[
            0
        ]  # (64,)
        votes.append(bits_prob)

    votes = np.stack(votes)  # (Nwin, 64) #(13, 64)
    prob = votes.mean(axis=0)  # majority / mean #(64,)
    bits = (prob > 0.5).astype(int) #(64,)
    hex_out = bits_to_hex(bits) #16

    print("\nRecovered watermark hex :", hex_out)
    print("Bit probabilities         :", np.round(prob, 3))
    
    if args.wm:
        bits_in = hex_to_bits(args.wm).astype(int) #(64, )
        errors = bits_in != bits
        ber = np.mean(errors)
        print(f"Bit Error Rate (BER)     : {ber:.4f} ({ber*100:.1f}%)")
        print("Ground truth watermark hex: ", args.wm)


if __name__ == "__main__":
    main()
