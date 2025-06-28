# attack.py
from random import randint

import numpy as np
import tensorflow as tf
from scipy.signal import butter
from tensorflow.keras.backend import stop_gradient
from tensorflow.keras.layers import Conv1D, Layer
from tensorflow.keras.callbacks import Callback

# ------------------------- constants -------------------------------- #
CHUNK_SAMPLES = 2400  # 15 LPC frames × 160
FS = 16_000  # LPCNet default
NUM_SAMPLES_CUT = 200  # default for CuttingSamples
NOISE_STRENGTH = 0.005
PROB_COEFF = 30  # percentage probability a distortion fires


# -------------------------------------------------------------------- #
#  1) LOW-PASS FIR FILTER (non-causal, linear-phase)                   #
# -------------------------------------------------------------------- #
def design_lowpass(cutoff=4_000, order=64):
    """Return a 1-D symmetrical FIR kernel for tf.nn.conv1d."""
    nyq = 0.5 * FS
    taps = cutoff / nyq
    b = np.sinc(2 * taps * (np.arange(order) - (order - 1) / 2))
    w = np.hamming(order)
    kernel = b * w
    kernel /= kernel.sum()
    return kernel.astype("float32")[::-1]  # flip for conv1d (causal)


class LowpassFilter(Layer):
    def __init__(self, cutoff=4_000, **kw):
        super().__init__(**kw)
        kernel = design_lowpass(cutoff)
        self.kernel = tf.Variable(kernel[:, None, None], trainable=False)  # (K,1,1)

    def call(self, x):
        # x: (B, 2400, 1)
        y = tf.nn.conv1d(x, self.kernel, stride=1, padding="SAME")
        return y


# -------------------------------------------------------------------- #
#  2) ADDITIVE WHITE NOISE                                             #
# -------------------------------------------------------------------- #
class AdditiveNoise(Layer):
    def __init__(self, strength=NOISE_STRENGTH, prob=PROB_COEFF, **kw):
        super().__init__(**kw)
        self.strength = strength
        self.prob = prob

    def call(self, x):
        # decide per-batch whether to apply noise
        rnd = tf.random.uniform((), 0, 100)

        def _add():
            noise = tf.random.uniform(tf.shape(x), -self.strength, self.strength)
            return x + noise

        return tf.cond(rnd < self.prob, _add, lambda: x)


# -------------------------------------------------------------------- #
#  3) RANDOM SAMPLE CUT (zero-out short bursts)                        #
# -------------------------------------------------------------------- #
class CuttingSamples(Layer):
    def __init__(self, num_samples=NUM_SAMPLES_CUT, prob=PROB_COEFF, **kw):
        super().__init__(**kw)
        self.num = num_samples
        self.prob = prob

    def call(self, x):
        # B, T, _ = x.shape #shape statik
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        rnd = tf.random.uniform((), 0, 100)

        def _cut():
            # build a mask of ones then zero out random indices
            idx = tf.random.uniform((B, self.num), minval=0, maxval=T, dtype=tf.int32)
            mask = tf.ones_like(x)
            flat = tf.reshape(mask, (-1,))
            coef = tf.range(0, B)[:, None] * T + idx  # global idx
            coef = tf.reshape(coef, (-1,))
            flat = tf.tensor_scatter_nd_update(
                flat, tf.expand_dims(coef, 1), tf.zeros_like(coef, tf.float32)
            )
            mask = tf.reshape(flat, (B, T, 1))
            return x * mask

        return tf.cond(rnd < self.prob, _cut, lambda: x)


# -------------------------------------------------------------------- #
#  4) BUTTERWORTH IIR (approximated via causal filtering)              #
# -------------------------------------------------------------------- #
class ButterworthFilter(Layer):
    def __init__(self, cutoff=4_000, order=4, prob=PROB_COEFF, **kw):
        super().__init__(**kw)
        self.b, self.a = butter(order, cutoff / (0.5 * FS), "low")
        self.prob = prob

    @tf.function
    def _tf_filt(self, sig):
        # tf.numpy_function wrapper because tf.signal.lfilter has no grad
        def _np(x):
            from scipy.signal import lfilter

            return lfilter(self.b, self.a, x).astype("float32")

        return tf.numpy_function(_np, [sig], tf.float32)

    def call(self, x):
        rnd = tf.random.uniform((), 0, 100)

        def _filt():
            # apply per batch item
            parts = [self._tf_filt(xi[:, 0]) for xi in tf.unstack(x, axis=0)]
            return tf.stack([tf.expand_dims(p, -1) for p in parts], 0)

        return tf.cond(rnd < self.prob, _filt, lambda: x)