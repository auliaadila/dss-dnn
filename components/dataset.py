import os, glob, math, argparse
from typing import List, Tuple
import warnings

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_io as tfio

class FrameSequence(tf.keras.utils.Sequence):
    """Streams clean PCM + random 64‑bit word repeated over `seq_frames`."""
    def __init__(self, wav_folder: str, frame_size=160, seq_frames=15,
                 bits_per_frame=64, bits_in=None, batch_size=32, shuffle=True, fs=16_000):
        self.paths = sorted(glob.glob(os.path.join(wav_folder, "**", "*.wav"), recursive=True))
        if not self.paths:
            raise RuntimeError("No WAVs found under", wav_folder)
        self.frame_size = frame_size
        self.seq_frames = seq_frames
        self.window = frame_size * seq_frames #2400
        self.bits_per_frame = bits_per_frame
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.fs = fs
        self.indices = []  # (file_idx, start_sample)
        self.bits_in = np.random.randint(0, 2, size=(self.batch_size, self.bits_per_frame), dtype='int32')
        for fi, p in enumerate(self.paths):
            n = sf.info(p).frames
            for s in range(0, n - self.window + 1, self.window):
                self.indices.append((fi, s)) #len?
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx): #one-batch-of-windows-at-a-time
        batch_idx = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        pcm = []
        for fi, st in batch_idx:
            p = self.paths[fi]
            # Get audio
            raw, _ = sf.read(p, start=st, stop=st+self.window, dtype="float32")
            raw_ = np.expand_dims(raw, -1) # (2400, 1)
            pcm.append(raw_)

        pcm = np.stack(pcm)
        actual_batch_size = len(pcm)
        bits_in = np.random.randint(0, 2, size=(actual_batch_size, self.bits_per_frame), dtype='int32')

        dummy = np.zeros_like(pcm, dtype="float32")
        inputs = [bits_in, pcm]
        outputs = {
            "res_w": dummy,
            "pcm_w": pcm,
            "bits_pred": bits_in,

        }

        return (inputs, outputs)  # labels = bits (unused)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)