import numpy as np
import tensorflow as tf
from pesq import pesq


class DeltaPESQMetric(tf.keras.metrics.Metric):
    def __init__(self, fs=16000, name="delta_pesq", **kwargs):
        super().__init__(name=name, **kwargs)
        self.fs = fs
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        def pesq_np(ref, deg):
            # import numpy as np
            # from pesq import pesq
            ref = np.asarray(ref).flatten()
            deg = np.asarray(deg).flatten()
            return np.float32(pesq(self.fs, ref, deg, "wb"))

        delta = tf.numpy_function(
            lambda ref, deg: pesq_np(ref, deg),
            [tf.squeeze(y_true), tf.squeeze(y_pred)],
            tf.float32,
        )
        self.total.assign_add(delta)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


import numpy as np
import tensorflow as tf
from pesq import pesq as pesq_score  # ITU-T WB implementation


class WatermarkedPESQMetric(tf.keras.metrics.Metric):
    """
    Average PESQ-WB (MOS-LQO) of the **water-marked signal** vs. the clean
    reference.  Higher = better quality.

    Expected shapes
    ---------------
    y_true : clean/host  (B, T, 1)  float32  [-1, 1]
    y_pred : water-marked (B, T, 1) same dtype/shape
    """

    def __init__(self, fs: int = 16_000, name: str = "pesq", **kwargs):
        super().__init__(name=name, **kwargs)
        self.fs = fs
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    # --------------------------------------------------------------
    def update_state(self, y_true, y_pred, sample_weight=None):
        def pesq_np(ref, deg):
            ref = np.asarray(ref).flatten()
            deg = np.asarray(deg).flatten()
            return np.float32(pesq_score(self.fs, ref, deg, "wb"))

        pesq_val = tf.numpy_function(
            func=lambda ref, deg: pesq_np(ref, deg),
            inp=[tf.squeeze(y_true), tf.squeeze(y_pred)],
            Tout=tf.float32,
        )

        self.total.assign_add(pesq_val)
        self.count.assign_add(1.0)

    # --------------------------------------------------------------
    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


# -----------------------------  helpers  ---------------------------------
class BitErrorRate(tf.keras.metrics.Metric):
    """BER = fraction of bits decoded wrong (lower is better)"""

    def __init__(self, name="ber", **kw):
        super().__init__(name=name, **kw)
        self.err = self.add_weight("err", initializer="zeros")
        self.total = self.add_weight("tot", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)  # threshold
        errors = tf.reduce_sum(tf.abs(y_true - y_pred_bin))
        bits = tf.cast(tf.size(y_true), tf.float32)
        self.err.assign_add(errors)
        self.total.assign_add(bits)

    def result(self):
        return self.err / self.total

    def reset_state(self):
        self.err.assign(0.0)
        self.total.assign(0.0)
