# ChannelSim

import numpy as np
import tensorflow as tf
from scipy.signal import butter

FS = 16_000
# CHUNK_SAMPLES = 2400

# -----------------------------------------------------------------------------
# 1. Helper: low-pass FIR kernel design (linear-phase)
# -----------------------------------------------------------------------------


def fir_lowpass(cutoff=4000, order=64):  # ok
    nyq = 0.5 * FS
    taps = cutoff / nyq
    b = np.sinc(2 * taps * (np.arange(order) - (order - 1) / 2))
    w = np.hamming(order)
    kernel = b * w
    kernel /= kernel.sum()
    return kernel.astype("float32")[::-1]  # flip for causal conv1d


# -----------------------------------------------------------------------------
# 2. Individual attack layers
# -----------------------------------------------------------------------------


class LowpassFIR(tf.keras.layers.Layer):
    """Linear-phase FIR low-pass filter via 1-D convolution."""

    def __init__(self, cutoff=4000, prob=30.0, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.prob = prob
        kern = fir_lowpass(cutoff)
        self.kernel = tf.constant(kern[:, None, None])  # (K,1,1)

    def call(self, x):
        cond = tf.random.uniform([]) * 100 < self.prob

        def _f():
            return tf.nn.conv1d(x, self.kernel, stride=1, padding="SAME")

        return tf.cond(cond, _f, lambda: x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(cutoff=self.cutoff, prob=self.prob))
        return cfg


class AdditiveNoise(tf.keras.layers.Layer):
    """Add uniform noise ±strength with given probability."""

    def __init__(self, strength=0.005, prob=30.0, **kw):
        super().__init__(**kw)
        self.strength = strength
        self.prob = prob

    def call(self, x):
        cond = tf.random.uniform([]) * 100 < self.prob

        def _f():
            n = tf.random.uniform(tf.shape(x), -self.strength, self.strength)
            return x + n

        return tf.cond(cond, _f, lambda: x)

    def get_config(self):
        c = super().get_config()
        c.update(dict(strength=self.strength, prob=self.prob))
        return c


class CuttingSamples(tf.keras.layers.Layer):
    """Zero out *num* random samples in each chunk (burst erasure)."""

    def __init__(self, num=200, prob=30.0, **kw):
        super().__init__(**kw)
        self.num = num
        self.prob = prob

    def call(self, x):
        cond = tf.random.uniform([]) * 100 < self.prob
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        def _f():
            idx = tf.random.uniform((B, self.num), 0, T, dtype=tf.int32)
            flat = tf.reshape(x, (-1,))
            base = tf.range(B)[:, None] * T
            pos = tf.reshape(base + idx, (-1,))
            mask = tf.tensor_scatter_nd_update(
                tf.ones_like(flat),
                tf.expand_dims(pos, 1),
                tf.zeros_like(pos, tf.float32),
            )
            return tf.reshape(flat * mask, (B, T, 1))

        return tf.cond(cond, _f, lambda: x)

    def get_config(self):
        c = super().get_config()
        c.update(dict(num=self.num, prob=self.prob))
        return c


class ButterworthIIR(tf.keras.layers.Layer):
    """Apply causal Butterworth low-pass IIR (Scipy) via tf.numpy_function."""

    def __init__(self, cutoff=4000, order=4, prob=30.0, **kw):
        super().__init__(**kw)
        from scipy.signal import butter

        b, a = butter(order, cutoff / (0.5 * FS), "low")
        self.b = b.astype("float32")
        self.a = a.astype("float32")
        self.prob = prob

    def _np_filt(self, sig):
        from scipy.signal import lfilter

        return lfilter(self.b, self.a, sig).astype("float32")

    def call(self, x):
        cond = tf.random.uniform([]) * 100 < self.prob

        def _f():
            # x shape: [B, T, 1]
            x_squeezed = tf.squeeze(x, -1)  # [B, T]
            outputs = tf.map_fn(
                lambda sig: tf.numpy_function(self._np_filt, [sig], tf.float32),
                x_squeezed,
                fn_output_signature=tf.TensorSpec(shape=(None,), dtype=tf.float32),
            )
            return tf.expand_dims(outputs, -1)  # [B, T, 1]

        return tf.cond(cond, _f, lambda: x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(cutoff=4000, order=4, prob=self.prob))
        return cfg


# -----------------------------------------------------------------------------
# 3. AttackPipeline – chain attacks lazily (non-diff)
# -----------------------------------------------------------------------------
class AttackPipeline(tf.keras.layers.Layer):
    """
    Stochastic channel simulation.
    • During training  -> apply attacks   (gradients stopped)
    • During inference -> pass-through
    """

    def __init__(self, attacks=None, name="attack_pipe", **kwargs):
        super().__init__(name=name, **kwargs)
        self.attacks = attacks or [
            ButterworthIIR(),
            AdditiveNoise(),
            CuttingSamples(),
            # LowpassFIR(), AdditiveNoise(), CuttingSamples(), ButterworthIIR()
        ]

    def call(self, x, training=None):
        # Resolve the learning phase if `training` is None
        # training=None → look up Keras’ global learning-phase so attacks run
        # during model.fit() but not during model.predict().
        if training is None:
            training = tf.keras.backend.learning_phase()

        def _apply():
            y = x
            for layer in self.attacks:
                y = layer(y, training=training)  # pass flag downstream
            return tf.stop_gradient(y)  # block grads

        # Use tf.cond so it works with symbolic `training`
        return tf.cond(tf.cast(training, tf.bool), true_fn=_apply, false_fn=lambda: x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            dict(
                attacks=[tf.keras.utils.serialize_keras_object(a) for a in self.attacks]
            )
        )
        return cfg

    @classmethod
    def from_config(cls, config):
        attacks = [tf.keras.utils.deserialize_keras_object(a) for a in config.pop('attacks', [])]
        return cls(attacks=attacks, **config)


"""
from attack_layers import AttackPipeline

pcm_attacked = AttackPipeline()(pcm_watermarked, training=True)
"""
