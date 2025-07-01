import os, glob, math, argparse
from typing import List, Tuple
import warnings

import numpy as np
import tensorflow as tf
import scipy.signal as ss

class LPAnalysis(tf.keras.layers.Layer):
    def __init__(self, order=16, frame_size=160, **kw):
        super().__init__(trainable=False, **kw)
        self.order = order
        self.frame_size = frame_size

    def _lpc_residual(self, x):
        
        B, T = x.shape
        n_frames = T // self.frame_size
        x = x.reshape(B, n_frames, self.frame_size)
        res = np.zeros_like(x)

        for b in range(B):
            for f in range(n_frames):
                frm = x[b, f]
                r = np.correlate(frm, frm, mode='full')[self.frame_size - 1 : self.frame_size + self.order]
                
                a = np.zeros(self.order + 1, dtype=np.float64)
                a[0] = 1.0
                e = r[0] if r[0] != 0 else 1e-8  # avoid division by zero

                for i in range(1, self.order + 1):
                    acc = r[i] - np.dot(a[1:i], r[1:i][::-1])
                    k = acc / e if e != 0 else 0
                    a_new = a[1:i] - k * a[i-1:0:-1]
                    a[1:i] = a_new
                    a[i] = k
                    e *= (1.0 - k * k)
                    if e <= 1e-8:  # prevent numerical instability
                        e = 1e-8

                res[b, f] = ss.lfilter(a, [1.0], frm)

        return res.reshape(B, -1).astype(np.float32)

    def call(self, x):
        if x.shape.rank == 3:
            x = tf.squeeze(x, -1)  # (B, T)
        r = tf.numpy_function(self._lpc_residual, [x], tf.float32)
        r.set_shape(x.shape)  # match input shape (B, T)
        return tf.expand_dims(r, -1)  # output shape (B, T, 1)