import math
import random
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from .preprocessor import AudioPreprocessor


class FrameSequencePayloadCached(tf.keras.utils.Sequence):
    """
    Cached version of FrameSequencePayload that avoids repeated soundfile calls.

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
        cache_dir="./cache",
        force_refresh=False,
        **kwargs,
    ):
        self.F = frame_size
        self.K = frames_per_payload
        self.W = frame_size * frames_per_payload  # (160 * 15 = 2400 samples =~ 150 ms)
        self.P = payload_bits
        self.B = batch
        self.fs = fs
        self.shuffle = shuffle

        # Initialize preprocessor and load cached audio
        self.preprocessor = AudioPreprocessor(cache_dir)
        print(f"Loading audio data from {root}...")
        self.audio_cache = self.preprocessor.preprocess_and_cache(root, force_refresh)

        # Build index from cached audio
        self.paths = list(self.audio_cache.keys())
        self.index: List[Tuple[int, int]] = []  # (file_id, start)

        print("Building audio index...")
        for fi, path in enumerate(self.paths):
            n = self.audio_cache[path]["frames"]
            for s in range(0, n - self.W + 1, self.W):  # per window
                self.index.append((fi, s))  # 1 index correspond to 1 window

        if shuffle:
            random.shuffle(self.index)

        print(f"Dataset ready: {len(self.paths)} files, {len(self.index)} windows")

    def __len__(self):
        return math.ceil(len(self.index) / self.B)

    def _read_slice(self, file_id: int, start: int) -> np.ndarray:
        """Read audio slice from cached data (much faster than soundfile)."""
        path = self.paths[file_id]
        audio_data = self.audio_cache[path]["data"]
        stop = start + self.W
        return audio_data[start:stop].astype(np.float32)

    def __getitem__(self, i):
        segs = self.index[i * self.B : (i + 1) * self.B]  # take one batch
        pcm_batch, bit_batch = [], []

        for fi, st in segs:
            # pcm - read from cache (no soundfile call!)
            wav = self._read_slice(fi, st)
            pcm_batch.append(wav.reshape(-1, 1))

            # bits - generate random payload
            bits = np.random.randint(0, 2, (self.P,), dtype=np.uint8).astype(np.float32)
            bit_batch.append(bits)

        output = {"bits_pred": np.stack(bit_batch), 
                "wm_pcm": np.stack(pcm_batch)}
        return [np.stack(bit_batch), np.stack(pcm_batch)], output
    

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.index)


# Backward compatibility alias
FrameSequencePayload = FrameSequencePayloadCached
