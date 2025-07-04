import tensorflow as tf
from dss.lpanalysis import LPAnalysis

__all__ = [
    "WatermarkEmbedding",
    "WatermarkAddition",
]

# ---------------------------------------------------------------------------
#  WatermarkEmbedding Layer
# ---------------------------------------------------------------------------
class WatermarkEmbedding(tf.keras.layers.Layer):
    """Embed DSS‑spread bits into an LP residual **or** raw PCM.

    If `compute_residual=True`, the second input is assumed to be a *host PCM*
    waveform; an internal `LPAnalysis` layer converts it to residual form before
    embedding.
    """

    def __init__(
        self,
        frame_size: int = 160,
        bits_per_frame: int = 64,
        alpha_init: float = 0.04,
        trainable_alpha: bool = True,
        compute_residual: bool = False,
        lpc_order: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frame_size = int(frame_size)
        self.bits_per_frame = int(bits_per_frame)
        self.alpha_init = float(alpha_init)
        self.trainable_alpha = bool(trainable_alpha)
        self.compute_residual = bool(compute_residual)
        self.lpc_order = int(lpc_order)
        if self.compute_residual:
            self.lp_layer = LPAnalysis(order=lpc_order, frame_size=frame_size, name="lp_analysis")

    def build(self, input_shapes):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.alpha_init),
            constraint=tf.keras.constraints.NonNeg(),
            trainable=self.trainable_alpha,
        )
        super().build(input_shapes)

    # -------------------------------------------------------------------
    def _spread_bits(self, bits, n_samples):
        chips_per_bit = self.frame_size // self.bits_per_frame
        chips = 2.0 * bits - 1.0
        chips = tf.repeat(chips, chips_per_bit, axis=-2)
        return tf.reshape(chips, (-1, n_samples, 1))

    # -------------------------------------------------------------------
    def call(self, inputs, **kwargs):
        bits, x = inputs  # x = residual **or** host PCM

        if self.compute_residual:
            residual = self.lp_layer(x)
        else:
            residual = x
            if residual.shape.rank == 2:
                residual = tf.expand_dims(residual, -1)

        n_samples = tf.shape(residual)[1]

        # Spread bits to chip sequence of same length
        if tf.shape(bits)[1] == n_samples:
            chips = 2.0 * bits - 1.0
        else:
            chips = self._spread_bits(bits, n_samples)

        return residual + self.alpha * chips

    # -------------------------------------------------------------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            frame_size=self.frame_size,
            bits_per_frame=self.bits_per_frame,
            alpha_init=self.alpha_init,
            trainable_alpha=self.trainable_alpha,
            compute_residual=self.compute_residual,
            lpc_order=self.lpc_order,
        )
        return cfg
'''
# -----------------------------------------------------------------------------
#  WatermarkEmbedding Layer
# -----------------------------------------------------------------------------
class WatermarkEmbedding(tf.keras.layers.Layer):
    """Embed DSS‑spread bits into an LP residual (or any host frame).

    Parameters
    ----------
    frame_size : int
        Number of audio samples per frame (e.g. 160 for 10 ms @ 16 kHz).
    bits_per_frame : int
        How many bits are embedded within one *frame_size* window.
    alpha_init : float, default 0.04
        Initial embedding strength (scales the DSS chips).
    trainable_alpha : bool, default True
        Whether *alpha* is a learnable parameter during back‑prop.

    Input
    -----
    A list/tuple of two tensors `[bits, residual]`.
        • bits      : (B, T, 1) float32 in {0,1} – 1 per audio sample **OR**
                      (B, ⌈T / frame_size⌉, bits_per_frame)
        • residual  : (B, T, 1) float32 – LP residual / host signal

    Output
    ------
    residual_w     : (B, T, 1) float32 – watermarked residual
    """

    def __init__(self, frame_size=160, bits_per_frame=64,
                 alpha_init=0.04, trainable_alpha=True, **kwargs):
        super().__init__(**kwargs)
        self.frame_size = int(frame_size)
        self.bits_per_frame = int(bits_per_frame)
        self.alpha_init = float(alpha_init)
        self.trainable_alpha = bool(trainable_alpha)

    def build(self, input_shapes):
        # Single scalar α shared for the whole model; clip to [0, 0.3]
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.alpha_init),
            constraint=tf.keras.constraints.NonNeg(),
            trainable=self.trainable_alpha,
        )
        super().build(input_shapes)

    # ---------------------------------------------------------------------
    #  Helper – spread bits to chips of length `frame_size` (Walsh/Hadamard‑like)
    # ---------------------------------------------------------------------
    def _spread_bits(self, bits, n_samples):
        """Maps bits (0/1) → DSS chips (‑1/+1) and repeats to frame_size chips."""
        chips_per_bit = self.frame_size // self.bits_per_frame
        # Scale bits to ±1 and repeat
        chips = 2.0 * bits - 1.0                      # (0,1) → (‑1,+1)
        chips = tf.repeat(chips, chips_per_bit, axis=-2)  # time‑axis repeat
        return tf.reshape(chips, (-1, n_samples, 1))

    # ------------------------------------------------------------------
    def call(self, inputs, **kwargs):
        bits, residual = inputs
        # Ensure residual has shape (B, T, 1)
        if residual.shape.rank == 2:
            residual = tf.expand_dims(residual, -1)

        n_samples = tf.shape(residual)[1]

        # If bits are already per‑sample, scale & broadcast.
        if tf.shape(bits)[1] == n_samples:
            chips = 2.0 * bits - 1.0  # (B, T, 1)
        else:
            # Otherwise bits are per‑frame – spread them.
            chips = self._spread_bits(bits, n_samples)

        # Embed: r_w = r + α ⋅ chips
        return residual + self.alpha * chips

    # ------------------------------------------------------------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            frame_size=self.frame_size,
            bits_per_frame=self.bits_per_frame,
            alpha_init=self.alpha_init,
            trainable_alpha=self.trainable_alpha,
        )
        return cfg


'''
# -----------------------------------------------------------------------------
#  WatermarkAddition Layer
# -----------------------------------------------------------------------------
class WatermarkAddition(tf.keras.layers.Layer):
    """Adds the watermarked residual back into the host PCM waveform.

    Parameters
    ----------
    learnable_mask : bool, default False
        If *True*, introduce a per‑frequency mask `M(f)` (currently scalar β).
    beta : float, default 0.10
        Initial mixture coefficient.
    """

    def __init__(self, learnable_mask: bool = False, beta: float = 0.10, **kwargs):
        super().__init__(**kwargs)
        self.learnable_mask = bool(learnable_mask)
        self.beta_init = float(beta)

    def build(self, input_shapes):
        # Scalar β (or small vector if you later want a mask per Bark band)
        self.beta = self.add_weight(
            name="beta",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.beta_init),
            constraint=tf.keras.constraints.NonNeg(),
            trainable=self.learnable_mask,
        )
        super().build(input_shapes)

    def call(self, inputs, **kwargs):
        pcm, residual_w = inputs
        if pcm.shape.rank == 2:
            pcm = tf.expand_dims(pcm, -1)
        if residual_w.shape.rank == 2:
            residual_w = tf.expand_dims(residual_w, -1)
        return pcm + self.beta * residual_w

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            learnable_mask=self.learnable_mask,
            beta=self.beta_init,
        )
        return cfg
