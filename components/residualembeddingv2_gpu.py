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
    alpha_min, alpha_max : float
        Embedding strength.
    use_full_lpc : bool
        True  → residual = LPC residual (slow, accurate)
        False → residual = 1-pole high-pass (fast, robust).
    """
    """
    Water-marking layer that
        ① extracts a per-window residual r(t)
        ② turns sign(r(t)) into the chip sequence
        ③ chooses α *per window* with a tiny network
        ④ adds α · chips · bits  to the host

    inputs  : [bits, pcm]
        bits ∈ {0,1}          – (B,  P)
        pcm  ∈ [-1,1] float32 – (B,  T, 1)   with  T = K·frame
    output  : watermarked pcm – (B,  T, 1)
    """

    def __init__(
        self,
        frame_size=160,
        K=15,
        P=64,
        order=12,
        alpha_min=1e-3,
        alpha_max=2e-1,
        use_full_lpc=False,
        trainable_alpha=False,
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
        self.T = self.frame * self.K
        self.a_min = alpha_min
        self.a_max = alpha_max
        self.trainable_alpha = bool(trainable_alpha)

        # very small "alpha-net": GAP → two Dense → sigmoid  → scalar α
        self.alpha_net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense( 1, activation='sigmoid')
        ])
        self.alpha_net.trainable = self.trainable_alpha

        # ------------------------------------------------------------------ #
        # Constant Hann window (shape 1 × frame) - not needed for simple residual
        # ------------------------------------------------------------------ #
        # Removed to simplify implementation

    # full LPC residual
    # @staticmethod
    def _lpc_residual_numpy(frames, order):
        """
        frames : (N, frame) np.float32  – already windowed
        return : (N, frame) np.float32  – LPC residual
        """
        import numpy as np
        from scipy.signal import lfilter
        import librosa                                # fast Levinson solver

        out = np.empty_like(frames, dtype=np.float32)
        for i, fr in enumerate(frames):
            a = librosa.lpc(fr, order=int(order))     # length = order+1
            out[i] = lfilter(a, [1.0], fr).astype(np.float32)
        return out

    # ====================================================================== #
    # Helper: 1-pole high-pass residual (very fast)
    # ====================================================================== #
    @staticmethod
    def _gpu_residual_simple(pcm, z=0.95):
        pcm_padded = tf.pad(pcm, [[0, 0], [1, 0], [0, 0]])
        return pcm - z * pcm_padded[:, :-1, :]

    # ====================================================================== #
    # Residual computation (choose fast HPF   *or*   full LPC)               #
    # ====================================================================== #
    def _residual(self, pcm):
        """
        Return (B, T, 1) residual according to `self.use_full_lpc`
        """
        if not self.use_full_lpc:
            # ── cheap & cheerful 1-pole HPF (GPU-friendly) ──────────────────
            return self._gpu_residual_simple(pcm)

        # ── accurate frame-wise LPC residual (CPU) ──────────────────────────
        # shape prep:   (B, T, 1) → (B*K, frame)
        B      = tf.shape(pcm)[0]
        pcm_2d = tf.reshape(pcm, [-1, self.frame])             # (B*K, frame)

        # Hann window (broadcasted)
        win    = tf.signal.hann_window(self.frame, periodic=False)
        pcm_win= pcm_2d * win

        # numpy_function -> (B*K, frame)  float32
        res = tf.numpy_function(
            func=self._lpc_residual_numpy,
            inp=[pcm_win, self.order],
            Tout=tf.float32,
            name='lpc_residual_numpy'
        )

        # restore (B, T, 1)
        res = tf.reshape(res, [B, self.T, 1])
        return res
    
    def _pn(self, pcm):
        B = tf.shape(pcm)[0]
        T = self.T
        rnd = tf.random.stateless_uniform(
                shape=(B, T, 1), seed=[123, 456], dtype=tf.float32)
        return tf.where(rnd > 0.5, 1.0, -1.0)          # (B,T,1)
    
    # ----------  build chip train from residual + bits  -------------------
    def _chips_from_residual(self, bits, resid):
        """
        • sign(resid) ∈ {-1,+1}  becomes the chip waveform
        • each payload bit stretches over ⌈T/P⌉ samples
        """
        chips_host = tf.sign(resid)
        chips_host = tf.where(chips_host == 0,
                              tf.ones_like(chips_host), chips_host)

        # repeat payload bits along time axis
        chips_per_bit = self.T // self.P #floor
        if self.T % self.P != 0:
            chips_per_bit += 1
        
        bits_b = bits * 2. - 1.                        # bipolar
        bits_rep = tf.repeat(bits_b, repeats=chips_per_bit, axis=1)[:, :self.T]  # (B,T)

        chips = chips_host[:, :, 0] * bits_rep         # (B,T)
        return tf.expand_dims(chips, -1)               # (B,T,1)

    # ====================================================================== #
    # Keras call
    # ====================================================================== #
    def call(self, inputs, training=False):
        bits, pcm = inputs
        bits = tf.cast(bits, tf.float32)

        # 1) random ±1 PN
        pn = self._pn(pcm)

        # 2) window-wise α using host loudness
        host_mag   = tf.reduce_mean(tf.abs(pcm), axis=1)          # (B,1)
        alpha_norm = self.alpha_net(host_mag,
                                    training=self.trainable_alpha)
        alpha      = self.a_min + (self.a_max - self.a_min) * alpha_norm
        alpha      = tf.expand_dims(alpha, 1)                     # (B,1,1)

        # 3) spread bits with PN
        chips_per_bit = tf.cast(tf.math.ceil(self.T / self.P), tf.int32)
        bits_bip      = bits * 2. - 1.                            # bipolar
        bits_rep      = tf.repeat(bits_bip,
                                repeats=chips_per_bit,
                                axis=1)[:, :self.T]             # (B,T)
        chips         = pn[:, :, 0] * bits_rep                    # (B,T)
        chips         = tf.expand_dims(chips, -1)                 # (B,T,1)

        # 4) embed
        watermarked = tf.clip_by_value(pcm + alpha * chips, -1., 1.)
        return watermarked
    
    '''
    ## Residual
    
    def call(self, inputs, training=False):
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

        # ── α per window ────────────────────────────────────────────────
        #  Use |resid| statistics as masking clue
        feat = tf.reduce_mean(tf.abs(resid), axis=1)      # (B,1)
        # feat = tf.expand_dims(feat,-1) 
        alpha_norm = self.alpha_net(feat, 
                                    training=self.trainable_alpha)  # (B,1)
        alpha = self.a_min + (self.a_max - self.a_min) * alpha_norm
        alpha = tf.expand_dims(alpha, 1)                  # (B,1,1) for broadcast

        # ── chips built from residual sign + payload bits ───────────────
        chips = self._chips_from_residual(bits, resid)    # (B,T,1)

        # ------------------------------------------------------------ #
        # 3. embed (direct addition for reliable correlation)
        # ------------------------------------------------------------ #
        # Direct addition works perfectly for correlation
        # Use smaller alpha to preserve audio quality
        watermarked = pcm + alpha * chips
        watermarked = tf.clip_by_value(watermarked, -1.0, 1.0)
        return watermarked
    '''
    
    def get_config(self):
        """Return configuration for model serialization."""
        config = super().get_config()
        config.update(
            {
                "frame_size": self.frame,
                "K": self.K,
                "P": self.P,
                "order": self.order,
                "alpha_min": self.a_min,  # Default alpha value
                "alpha_max": self.a_max,  # Default trainable setting
                "trainable_alpha": self.trainable_alpha,
                "use_full_lpc": self.use_full_lpc,
            }
        )
        return config
    
    @classmethod
    def from_config(cls, config):
        """
        Rebuild the layer from a saved `config` dict – regardless of whether it
        comes from the *current* version (alpha_min / alpha_max) or an older
        one that still used the single scalar `alpha` (+ `trainable_alpha`).

        Any unknown / legacy keys are silently ignored after we have copied
        out the values we still care about.
        """
        cfg = config.copy()                     # don’t mutate the caller’s dict

        # ―― mandatory / existing hyper-params (with fall-backs) ―――――――――――――――――――
        frame_size   = cfg.pop("frame_size",   160)
        K            = cfg.pop("K",            15)
        P            = cfg.pop("P",            64)
        order        = cfg.pop("order",        12)
        use_full_lpc = cfg.pop("use_full_lpc", False)

        # ―― new style: per-window alpha range ――――――――――――――――――――――――――――――――――
        alpha_min    = cfg.pop("alpha_min", 1e-3)
        alpha_max    = cfg.pop("alpha_max", 2e-1)
        trainable_alpha = cfg.pop("trainable_alpha", False)

        # ―― legacy style: single scalar alpha――――――
        # We *map* it onto the new range by setting both min & max to the same
        # value; that reproduces the old behaviour.
        if "alpha" in cfg:
            legacy_alpha = cfg.pop("alpha")
            alpha_min = alpha_max = float(legacy_alpha)
        cfg.pop("trainable_alpha", None)   # remove leftovers if any

        # ―― Any other leftover items in `cfg` are forwarded to `__init__` so
        #     custom kwargs are preserved. ―――――――――――――――――――――――――――――――――――――
        return cls(
            frame_size   = frame_size,
            K            = K,
            P            = P,
            order        = order,
            alpha_min    = alpha_min,
            alpha_max    = alpha_max,
            trainable_alpha = trainable_alpha,
            use_full_lpc = use_full_lpc,
            **cfg,                     # forward-compat custom args
        )
