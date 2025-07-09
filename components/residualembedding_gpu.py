import tensorflow as tf


class ResidualEmbeddingGPU(tf.keras.layers.Layer):
    """
    Watermark-embedding layer that adds the host-signal residual
    (simple high-pass or full LPC) multiplied by a chip sequence.

    Parameters
    ----------
    frame_size : int
        Samples per LPC frame.
    K : int
        Frames per utterance (total samples = K * frame_size).
    P : int
        Number of payload bits per utterance.
    order : int
        LPC order (only used if use_full_lpc=True).
    alpha : float
        Embedding strength.
    trainable_alpha : bool
        Whether alpha is trainable.
    use_full_lpc : bool
        True  → residual = LPC residual (slow, accurate)
        False → residual = 1-pole high-pass (fast, robust).
    """

    def __init__(
        self,
        frame_size=160,
        K=15,
        P=64,
        order=12,
        alpha=0.1,
        trainable_alpha=False,
        use_full_lpc=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # ------------------------------------------------------------------ #
        # Static hyper-parameters
        # ------------------------------------------------------------------ #
        self.frame = int(frame_size)
        self.K = int(K)
        self.P = int(P)
        self.order = int(order)
        self.use_full_lpc = bool(use_full_lpc)

        # ------------------------------------------------------------------ #
        # Trainable embedding strength α  (0.001 … 1.0, soft-clamped)
        # ------------------------------------------------------------------ #
        self._alpha_raw = tf.Variable(
            initial_value=tf.math.log(tf.exp(alpha) - 1.0),
            trainable=trainable_alpha,
            dtype=tf.float32,
            name="alpha_raw",
        )

        # ------------------------------------------------------------------ #
        # Constant Hann window (shape 1 × frame) - not needed for simple residual
        # ------------------------------------------------------------------ #
        # Removed to simplify implementation

    # ====================================================================== #
    # Helper: α in a stable, positive range via soft-plus
    # ====================================================================== #
    @property
    def alpha(self):
        return 0.001 + 0.099 * tf.nn.softplus(self._alpha_raw)

    def set_alpha_value(self, value):
        """Set alpha value (compatible with old interface)."""
        # Clamped to the new desired range: 0.002 to 0.1
        clamped_value = tf.clip_by_value(float(value), 0.002, 0.1)
        # Adjust the multiplier for the inverse softplus transformation
        softplus_input = (clamped_value - 0.001) / 0.099

        alpha_raw_value = tf.math.log(tf.exp(softplus_input) - 1.0)
        self._alpha_raw.assign(alpha_raw_value)

    def set_alpha_trainable(self, trainable):
        """Set alpha variable trainability (compatible with old interface)."""
        # In TensorFlow 2.x, we need to recreate the variable with new trainable setting
        current_value = self._alpha_raw.numpy()
        self._alpha_raw = tf.Variable(
            initial_value=current_value,
            trainable=bool(trainable),
            dtype=tf.float32,
            name="alpha_raw",
        )

    # Removed autocorrelation - not needed for simple residual

    # Removed Levinson-Durbin - not needed for simple residual

    # Removed full LPC residual - using simple 1-pole filter only

    # ====================================================================== #
    # Helper: 1-pole high-pass residual (very fast)
    # ====================================================================== #
    @staticmethod
    def _gpu_residual_simple(pcm):
        pcm_padded = tf.pad(pcm, [[0, 0], [1, 0], [0, 0]])
        return pcm - 0.95 * pcm_padded[:, :-1, :]

    # ====================================================================== #
    # Residual computation (simple 1-pole high-pass only)
    # ====================================================================== #
    def _residual(self, pcm):
        return self._gpu_residual_simple(pcm)

    # ====================================================================== #
    # Keras call
    # ====================================================================== #
    def call(self, inputs):
        """
        inputs = [bits, pcm]
            bits : (B, P)  float32 or int32 in {0,1}
            pcm  : (B, K*frame, 1)  float32 in [-1,1]
        """
        bits, pcm = inputs
        bits = tf.cast(bits, tf.float32)

        # ------------------------------------------------------------ #
        # 1. residual
        # ------------------------------------------------------------ #
        resid = self._residual(pcm)

        # ------------------------------------------------------------ #
        # 2. chip sequence (proper PN spread spectrum)
        # ------------------------------------------------------------ #
        total = self.K * self.frame

        # Convert bits to bipolar
        bits_bipolar = bits * 2.0 - 1.0  # (B, P)

        # Generate proper PN sequence for each bit
        # Use a fixed seed for reproducibility
        seed = [42, 123]  # Fixed seed for consistent PN sequences

        # Calculate chips per bit (spreading factor)
        chips_per_bit = total // self.P
        if total % self.P != 0:
            chips_per_bit += 1

        # Generate PN sequence for each bit position
        # Each bit gets a unique PN sequence
        pn_sequences = []
        for bit_idx in range(self.P):
            # Use different seeds for different bits
            bit_seed = [seed[0] + bit_idx, seed[1] + bit_idx * 17]
            pn_seq = tf.random.stateless_uniform(
                shape=[chips_per_bit], seed=bit_seed, minval=0, maxval=1
            )
            # Convert to bipolar ±1
            pn_seq = tf.where(pn_seq > 0.5, 1.0, -1.0)
            pn_sequences.append(pn_seq)

        # Stack all PN sequences
        pn_matrix = tf.stack(pn_sequences, axis=0)  # (P, chips_per_bit)

        # Spread each bit with its corresponding PN sequence
        batch_size = tf.shape(bits)[0]

        # Expand bits_bipolar to match PN matrix
        bits_expanded = tf.expand_dims(bits_bipolar, -1)  # (B, P, 1)

        # Multiply each bit with its PN sequence
        spread_matrix = bits_expanded * pn_matrix[None, :, :]  # (B, P, chips_per_bit)

        # Flatten to get chip sequence
        chips_flat = tf.reshape(
            spread_matrix, [batch_size, -1]
        )  # (B, P * chips_per_bit)

        # Truncate to exact total length
        chips = chips_flat[:, :total]  # (B, total)

        chips = tf.expand_dims(chips, -1)  # (B, total, 1)

        # ------------------------------------------------------------ #
        # 3. embed (direct addition for reliable correlation)
        # ------------------------------------------------------------ #
        # Direct addition works perfectly for correlation
        # Use smaller alpha to preserve audio quality
        watermarked = pcm + self.alpha * chips
        watermarked = tf.clip_by_value(watermarked, -1.0, 1.0)
        return watermarked

    def get_config(self):
        """Return configuration for model serialization."""
        config = super().get_config()
        config.update(
            {
                "frame_size": self.frame,
                "K": self.K,
                "P": self.P,
                "order": self.order,
                "alpha": 0.1,  # Default alpha value
                "trainable_alpha": False,  # Default trainable setting
                "use_full_lpc": self.use_full_lpc,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from configuration, with defaults for missing parameters."""
        # Handle old models that don't have these parameters saved
        frame_size = config.get("frame_size", 160)
        K = config.get("K", 15)
        P = config.get("P", 64)
        order = config.get("order", 12)
        alpha = config.get("alpha", 0.1)
        trainable_alpha = config.get("trainable_alpha", False)
        use_full_lpc = config.get("use_full_lpc", False)

        # Remove these from config to avoid passing them twice
        config = config.copy()
        for key in [
            "frame_size",
            "K",
            "P",
            "order",
            "alpha",
            "trainable_alpha",
            "use_full_lpc",
        ]:
            config.pop(key, None)

        return cls(
            frame_size=frame_size,
            K=K,
            P=P,
            order=order,
            alpha=alpha,
            trainable_alpha=trainable_alpha,
            use_full_lpc=use_full_lpc,
            **config,
        )
