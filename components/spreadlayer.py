import glob
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import tensorflow as tf


class SpreadLayer(tf.keras.layers.Layer):
    """Map payload vector to bipolar chip stream over K frames."""

    def __init__(self, frame_size=160, frames_per_payload=15, **kw):
        super().__init__(**kw)
        self.F = frame_size
        self.K = frames_per_payload

    def call(self, bits, total_len):
        # bits: (B, P)
        # total_len: int (should be frame_size * frames_per_payload) (160 * 15)

        B = tf.shape(bits)[0]  # batch
        P = tf.shape(bits)[1]  # payload

        spread_factor = total_len // P  # multiplier, how many repeats
        expected_len = spread_factor * P

        tf.print("→ Debug SpreadLayer:", "B =", B, "P =", P, "total_len =", total_len, "expected_len =", expected_len)
        # → Debug SpreadLayer: B = 32 P = 64 total_len = 2400 expected_len = 2368

        repeated = tf.repeat(bits, repeats=spread_factor, axis=1)  # (B, total_len)

        chips = 2 * tf.cast(repeated, tf.float32) - 1  # map {0,1} → {-1,+1}
        # chips = tf.reshape(chips, (B, total_len, 1))  # (32, 2400, 1)
        chips = tf.reshape(chips, (B, expected_len, 1))  # (32, 2400, 1)
        return chips

    # def call(self,bits,pcm_len):
    #     # bits: (B, payload_bits)
    #     B = tf.shape(bits)[0]
    #     chips=tf.repeat(bits[:,None,:], repeats=self.K, axis=1)   # (B,K,P)

    #     # print("Repeated chips:", chips) #shape=(None, 100, 8)

    #     chips=tf.reshape(chips, (B, self.K*self.F, -1))           # (B,W,P)
    #     # print(chips) #shape=(None, 16000, None)
    #     chips=chips[:,:,0]                                        # use col-0 per time step ??what is col-0 represents

    #     # print("Use col-0 per time step:", chips) #shape=(None, 16000)

    #     chips=tf.cast(2*chips-1, tf.float32)
    #     return chips[:,:,None]            # (B,W,1)
