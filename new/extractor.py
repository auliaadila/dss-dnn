import tensorflow as tf

# lighter version than components/WatermarkExtractor
class DSSExtractor(tf.keras.Model):
    """1-D CNN extractor for LP-DSS payload.

    Parameters
    ----------
    payload_bits : int
        Length of the payload vector (e.g. 64 or 8).
    context_len  : int
        Number of time-samples per training window (frame_size * frames_per_payload).
    base_filters : int, optional
        # of filters in the first Conv layer.
    """

    def __init__(self, payload_bits: int, context_len: int,
                 base_filters: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.bits = payload_bits
        self.len  = context_len
        F = base_filters

        self.backbone = tf.keras.Sequential([
            tf.keras.layers.Conv1D(F, 5, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool1D(2),                     # L/2

            tf.keras.layers.Conv1D(F*2, 5, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool1D(2),                     # L/4

            tf.keras.layers.Conv1D(F*4, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),         # (B, F*4)
        ])
        self.out_dense = tf.keras.layers.Dense(payload_bits, activation='sigmoid')

    # ------------------------------------------------------------------
    def call(self, x, training=False):
        """x : (B, context_len, 1) float32 in [-1,1]"""
        h = self.backbone(x, training=training)
        return self.out_dense(h)

    # ------------------------------------------------------------------
    def get_config(self):
        return dict(payload_bits=self.bits,
                    context_len=self.len,
                    name=self.name)

# ----------------------------------------------------------------------
# Convenience builder (mirrors previous build_extractor() function)
# ----------------------------------------------------------------------

def build_extractor(payload_bits: int, seg_len: int, base_filters: int =32):
    pcm_in = tf.keras.Input((seg_len,1))
    ext    = DSSExtractor(payload_bits, seg_len, base_filters)
    bits   = ext(pcm_in)
    return tf.keras.Model(pcm_in, bits, name='extractor_cnn')

'''
from extractor import DSSExtractor, build_extractor
extractor = build_extractor(payload_bits=64, seg_len=2400)
'''