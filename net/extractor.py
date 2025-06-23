import tensorflow as tf

__all__ = [
    "build_extractor",
]

# -----------------------------------------------------------------------------
#  Simple 1‑D CNN extractor for DSS watermarks
#  ----------------------------------------------------------
#  * Accepts variable‑length 16‑kHz waveform frames (B, T, 1)
#  * Outputs a vector of length `bits_per_frame` with probabilities ∈ [0,1]
# -----------------------------------------------------------------------------

def _conv_block(x, filters, kernel_size=3, dilation_rate=1, dropout_rate=0.1, name_prefix="cb"):
    """Conv → BN → ReLU → Dropout"""
    x = tf.keras.layers.Conv1D(filters, kernel_size,
                               dilation_rate=dilation_rate,
                               padding="same",
                               use_bias=False,
                               name=f"{name_prefix}_conv")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = tf.keras.layers.Activation("relu", name=f"{name_prefix}_relu")(x)
    if dropout_rate > 0.0:
        x = tf.keras.layers.Dropout(dropout_rate, name=f"{name_prefix}_drop")(x)
    return x


def build_extractor(bits_per_frame: int,
                    conv_filters=(64, 128, 256),
                    kernel_size=3,
                    dilation_rates=(1, 2, 4),
                    dropout_rate=0.1) -> tf.keras.Model:
    """Create a lightweight extractor network.

    Parameters
    ----------
    bits_per_frame : int
        Target length of the recovered bit vector.
    conv_filters : tuple[int]
        Number of filters per conv block.
    kernel_size : int
        Width of 1‑D convolution kernels.
    dilation_rates : tuple[int]
        Dilation factor for successive blocks (same length as *conv_filters*).
    dropout_rate : float
        Dropout applied after each block during training.
    """

    assert len(conv_filters) == len(dilation_rates), "conv_filters and dilation_rates must match in length"

    inp = tf.keras.Input(shape=(None, 1), dtype=tf.float32, name="attacked_pcm")
    x = inp

    # Stack of dilated conv blocks – captures context up to (kernel_size * dilation)
    for i, (f, d) in enumerate(zip(conv_filters, dilation_rates)):
        x = _conv_block(x, f, kernel_size=kernel_size,
                        dilation_rate=d,
                        dropout_rate=dropout_rate,
                        name_prefix=f"block{i+1}")

    # Global Average Pool to summarise temporal dimension
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)

    # Final dense with sigmoid to get bit probs
    out = tf.keras.layers.Dense(bits_per_frame, activation="sigmoid", name="bit_probs")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="extractor")
    return model
