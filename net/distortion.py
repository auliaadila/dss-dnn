import tensorflow as tf

# ---------------------------------------------------------------------------
#  Channel simulator layer – adds random Gaussian noise and optional resample
# ---------------------------------------------------------------------------

class ChannelSim(tf.keras.layers.Layer):
    """Differentiable channel with AWGN + random gain. Extend as needed."""

    def __init__(self, snr_db_min=10.0, snr_db_max=30.0, **kwargs):
        super().__init__(**kwargs)
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max

    def call(self, x, training=False):
        if not training:
            return x  # no distortion in inference / validation

        snr_db = tf.random.uniform([], self.snr_db_min, self.snr_db_max)
        signal_pow = tf.reduce_mean(tf.square(x))
        noise_pow = signal_pow / (10.0 ** (snr_db / 10.0))
        noise = tf.random.normal(tf.shape(x), stddev=tf.sqrt(noise_pow))
        return x + noise