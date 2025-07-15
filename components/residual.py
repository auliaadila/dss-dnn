# components/layers.py
import tensorflow as tf
import numpy as np
from scipy.signal import lfilter
import librosa

class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, use_full_lpc=False, frame_size=160, total_len=2400, lpc_order=12, **kwargs):
        super().__init__(**kwargs)
        self.use_full_lpc = use_full_lpc
        self.frame = frame_size
        self.T = total_len
        self.ord = lpc_order

    def call(self, pcm):
        if not self.use_full_lpc:
            return self._gpu_residual(pcm)
        return self._lpc_residual(pcm)

    @staticmethod
    def _gpu_residual(pcm, z=0.95):
        pcm_p = tf.pad(pcm, [[0,0],[1,0],[0,0]])
        return pcm - z * pcm_p[:, :-1, :]

    def _lpc_residual(self, pcm):
        B = tf.shape(pcm)[0]
        win = tf.signal.hann_window(self.frame, periodic=False)
        pcm2d = tf.reshape(pcm, [-1, self.frame]) * win
        res2d = tf.numpy_function(self._lpc_residual_numpy, [pcm2d, self.ord], tf.float32)
        res2d.set_shape(pcm2d.shape)
        return tf.reshape(res2d, [B, self.T, 1])

    @staticmethod
    def _lpc_residual_numpy(frames, order):
        out = np.empty_like(frames, dtype=np.float32)
        for i, fr in enumerate(frames):
            a = librosa.lpc(fr, order=int(order))
            out[i] = lfilter(a, 1.0, fr).astype(np.float32)
        return out

    def get_config(self):
        return {
            "use_full_lpc": self.use_full_lpc,
            "frame_size": self.frame,
            "total_len": self.T,
            "lpc_order": self.ord
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)