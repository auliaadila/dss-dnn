import tensorflow as tf

class LPDSExtractor(tf.keras.Model):
    """
    Minimal patch over DSSExtractorFixed so it works with
    whitened (not clipped) LP–DSS carriers and variable chip length.
    """

    def __init__(self, payload_bits: int, context_len: int,
                 frame_size: int = 160,  # so we can derive chips_per_bit
                 base_filters: int = 64, **kw):
        super().__init__(**kw)
        self.B = payload_bits
        self.T = context_len            # e.g. 2400
        self.frame = frame_size         # 160
        self.base_filters = base_filters
        F = base_filters

        # ---------- 1)  LP–residual whitening  ---------------------------
        def _whiten(x, eps=1e-8):
            # x: (B, T, 1)
            m   = tf.reduce_mean(x, axis=1, keepdims=True)
            rms = tf.sqrt(tf.reduce_mean(tf.square(x - m), axis=1, keepdims=True))
            return (x - m) / (rms + eps)
        self.pre = tf.keras.layers.Lambda(_whiten, name="whiten")
        self.norm0 = tf.keras.layers.LayerNormalization()  # fixes α drift

        # ---------- 2)  backbone identical to old version ----------------
        self.backbone = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=F, kernel_size=3, strides=1, padding="same", activation="relu", dilation_rate=1),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv1D(filters=F, kernel_size=3, strides=1, padding="same", activation="relu", dilation_rate=2),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv1D(filters=F, kernel_size=3, strides=1, padding="same", activation="relu", dilation_rate=4),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv1D(filters=F, kernel_size=3, strides=1, padding="same", activation="relu", dilation_rate=8),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv1D(filters=F, kernel_size=3, strides=1, padding="same", activation="relu", dilation_rate=16),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv1D(filters=F, kernel_size=3, strides=1, padding="same", activation="relu", dilation_rate=32),
            tf.keras.layers.LayerNormalization(),
        ])

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(F*8, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(payload_bits, activation="sigmoid"),
        ])

    # ------------------------------------------------------------------
    def _avg_pool_dynamic(self, x):
        """
        Average-pool with stride = chips_per_bit  (= ceil(T / B)).
        Fallback to stride=1 when chips<bit (rare).
        """
        chips_per_bit = (self.T + self.B - 1) // self.B
        if chips_per_bit < 2:                         # nothing to pool
            return x
        return tf.keras.layers.AveragePooling1D(
            pool_size=chips_per_bit,
            strides=chips_per_bit)(x)

    # ------------------------------------------------------------------
    def call(self, pcm, training=False):
        """
        pcm : (B, T, 1) float32  whitened LP residual expected
        """
        z = self.pre(pcm)            # whitening
        z = self.norm0(z, training=training)

        z = self.backbone(z, training=training)
        z = self._avg_pool_dynamic(z)   # stride = chips/bit

        z = tf.keras.layers.GlobalAveragePooling1D()(z)
        return self.classifier(z, training=training)

    # ---------- (de)serialisation  ------------------------------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(payload_bits=self.B,
                        context_len=self.T,
                        frame_size=self.frame,
                        base_filters=self.base_filters))
        return cfg

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)