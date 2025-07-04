import glob
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import tensorflow as tf

# -----------------------------------------------------------------------------
# 1.  FrameSequencePayload – yields K‑frame windows + one payload vector
# -----------------------------------------------------------------------------


class FrameSequencePayload(tf.keras.utils.Sequence):
    """Generate training windows for *payload* strategy.

    Each sample is **K frames** of PCM (K×160) and **one payload vector**
    of length `payload_bits` that will be copied to every frame in the
    SpreadLayer.

    Output:
        ([bits, pcm], bits) where
            bits : (B, payload_bits)
            pcm  : (B, K*160, 1)
    """

    def __init__(
        self,
        root: str,
        *,
        frame_size=160,
        frames_per_payload=15,
        payload_bits=64,
        batch=32,
        fs=16_000,
        shuffle=True,
        **kwargs,
    ):
        self.F = frame_size
        self.K = frames_per_payload
        self.W = frame_size * frames_per_payload  # (160 * 15 = 2400 samples =~ 150 ms)
        self.P = payload_bits
        self.B = batch
        self.fs = fs
        self.shuffle = shuffle

        self.paths = self._discover(root)
        self.index: List[Tuple[int, int]] = []  # (file_id, start)
        for fi, p in enumerate(self.paths):
            n = sf.info(p).frames
            # n = sf.info(p).frames if p.endswith('.wav') else os.path.getsize(p)//2
            for s in range(0, n - self.W + 1, self.W):  # per window
                self.index.append((fi, s))  # 1 index correspond to 1 window
        if shuffle:
            random.shuffle(self.index)

    def _discover(self, root):
        wavs = glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
        # s16  = glob.glob(os.path.join(root,'**','*.s16'),recursive=True)
        if not wavs:
            raise RuntimeError(f"No audio in {root}")
        # if not (wavs or s16):
        #     raise RuntimeError(f"No audio in {root}")
        paths = sorted(wavs)
        # paths = sorted(wavs+s16)
        random.shuffle(paths)
        return paths

    def __len__(self):
        return math.ceil(len(self.index) / self.B)

    def _read_slice(self, p, start):
        stop = start + self.W
        if p.endswith(".wav"):
            wav, _ = sf.read(p, start=start, stop=stop, dtype="float32")
        # else:
        #     with open(p,'rb') as f:
        #         f.seek(start*2); buf=f.read((stop-start)*2)
        #     wav=np.frombuffer(buf,dtype='<i2').astype(np.float32)/32768
        return wav

    def __getitem__(self, i):
        segs = self.index[i * self.B : (i + 1) * self.B]  # take one batch (32 index(?))
        pcm_batch, bit_batch = [], []
        for fi, st in segs:
            # pcm
            wav = self._read_slice(self.paths[fi], st)
            pcm_batch.append(wav.reshape(-1, 1))
            # bits
            bits = np.random.randint(
                0, 2, (self.P,), dtype=np.uint8
            )  # generate new payload per batch (1 batch = 1 window)
            bit_batch.append(bits.astype(np.float32))
        output = {
            "bits_pred":np.stack(bit_batch),
            "wm_pcm":np.stack(pcm_batch)
        }
        return [np.stack(bit_batch), np.stack(pcm_batch)], output
        # [bits, pcm], bits

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.index)
