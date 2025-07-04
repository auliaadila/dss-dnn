import glob
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import tensorflow as tf

from components.spreadlayer import SpreadLayer


class ResidualEmbedding(tf.keras.layers.Layer):
    """
    Embed watermark by *multiplying* LPC residual with the chip sequence.

    Inputs
    -------
    bits : (B, P)                # payload word, P bits
    pcm  : (B, T, 1)             # host window, T = K·160 samples

    Output
    -------
    watermarked_pcm : (B, T, 1)
    """

    def __init__(
        self,
        frame_size=160,
        frames_per_payload=15,
        lpc_order=12,
        alpha_init=0.05,
        trainable_alpha=False,
        **kw,
    ):
        super().__init__(**kw)
        self.frame = frame_size
        self.K = frames_per_payload
        self.order = lpc_order
        self.spread = SpreadLayer(frame_size, frames_per_payload)
        self.alpha_init = alpha_init
        self.trainable_alpha = trainable_alpha

    # -------- per-frame LPC residual ---------------------------------
    ## wrong version
    # def _residual(self, pcm):
    #     pcm = tf.squeeze(pcm, -1)                       # (B, T)
    #     B    = tf.shape(pcm)[0]
    #     frames = tf.reshape(pcm, (B, self.K, self.frame))  # (B,K,160)

    #     def _lpc(frame_np):
    #         import librosa, numpy as np
    #         a = librosa.lpc(frame_np, self.order)
    #         pred = np.convolve(frame_np, a[1:], mode='full')[: len(frame_np)]
    #         return frame_np - pred

    #     res = tf.numpy_function(
    #           lambda f: np.stack([_lpc(fr) for fr in f], 0),
    #           [tf.reshape(frames, (-1, self.frame))],
    #           tf.float32)
    #     return tf.reshape(res, (B, self.K * self.frame, 1))

    def _residual(self, pcm):
        """
        Parameters
        ----------
        pcm : (B, K*frame_size, 1) float32  in [-1,1]

        Returns
        -------
        residual : (B, K*frame_size, 1) float32
        """
        pcm = tf.squeeze(pcm, -1)  # (B, T)
        B = tf.shape(pcm)[0]
        frames = tf.reshape(pcm, (-1, self.frame))  # (B*K, 160) (32 * 15, 160)

        # Extract constants
        frame_size = self.frame  # 160
        lpc_order = self.order  # 16 -> 12

        def _lpc_residual(batch_2d, frame_size, lpc_order):
            """NumPy batch → NumPy batch (vectorised)."""
            import librosa
            import numpy as np
            from scipy.signal import lfilter

            out = np.empty_like(batch_2d, dtype=np.float32)
            for i, frame in enumerate(batch_2d):
                frame_win = frame * np.hanning(frame_size)
                a = librosa.lpc(frame_win, order=int(lpc_order))  # Fix: pass explicitly
                out[i] = lfilter(a, [1.0], frame).astype(np.float32)
            return out

        # Wrap in tf.numpy_function (non-diff)
        res = tf.numpy_function(
            func=_lpc_residual,
            inp=[frames, tf.constant(frame_size), tf.constant(lpc_order)],
            Tout=tf.float32,
            name="lpc_residual_np",
        )

        # Restore shape
        res = tf.reshape(
            res, (B, self.K * self.frame, 1)
        )  # (32, 15 * 160, 1) = (32, 2400, 1)
        return res

    # -------- build & call -------------------------------------------
    def build(self, _):
        init = tf.constant_initializer(self.alpha_init)
        self.alpha = self.add_weight(
            "alpha", shape=(), initializer=init, trainable=self.trainable_alpha
        )  # gradient flow through alpha, causing change of value in alpha (adaptive)
        super().build(_)

    def call(self, inputs):
        bits, pcm = inputs  # bits (B,P)  pcm (B,T,1)
        residual = self._residual(pcm)  # (B,T,1)
        chips = self.spread(bits, tf.shape(pcm)[1])  # (B,T,1) ±1
        wm_term = self.alpha * residual * chips
        return pcm + wm_term  # water-marked PCM (simple addition)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "frame_size": self.frame,
                "frames_per_payload": self.K,
                "lpc_order": self.order,
                "alpha_init": self.alpha_init,
                "trainable_alpha": self.trainable_alpha,
            }
        )
        return config
