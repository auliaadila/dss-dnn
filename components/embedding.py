import tensorflow as tf

class Embedder(tf.keras.layers.Layer):
    def __init__(self,
                 frame_size=160, K=15, P=64,
                 spread_method="dss",
                 alpha_settings="constant",
                 alpha=0.1,
                 a_min=1e-3, a_max=2e-1,
                 trainable_alpha=False,
                 lporder=12,
                 resid_method="sign",
                 use_full_lpc=False,
                 target_rms=0.3,          # for whiten variant
                 **kwargs):
        super().__init__(**kwargs)

        # ─── static hyper-params ───────────────────────────────────────
        self.frame, self.K, self.P = int(frame_size), int(K), int(P)
        self.T        = self.frame * self.K
        self.order    = int(lporder)
        self.use_full_lpc = bool(use_full_lpc)

        self.spread_method   = spread_method.lower()
        self.alpha_settings  = alpha_settings.lower()
        self.resid_method    = resid_method.lower()
        self.trainable_alpha = bool(trainable_alpha)

        self.a_min, self.a_max = a_min, a_max
        self.target_rms = target_rms                       # used by _resid_whiten

        # ── 1) GLOBAL (constant) α  ───────────────────────────────────
        self._alpha_raw = tf.Variable(
            tf.math.log(tf.exp(alpha) - 1.0),
            trainable=(self.alpha_settings == "constant"
                       and self.trainable_alpha),     
            dtype=tf.float32, name="alpha_raw")

        # ── 2) ADAPTIVE tiny net  ─────────────────────────────────────
        self.alpha_net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense( 1, activation='sigmoid')
        ])
        # Only train this net if we are in *adaptive* mode **and**
        # the user asked for a trainable α.
        self.alpha_net.trainable = (self.alpha_settings == "adaptive"
                                    and self.trainable_alpha)          # ★ FIXED

    # ---------- alpha helpers -----------------------------------------------
    @property
    def alpha_const(self):                 # scalar in (0.001,0.1)
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

    # ---------- residual helpers --------------------------------------------
    @staticmethod
    def _gpu_residual_simple(pcm, z=0.95):
        pcm_padded = tf.pad(pcm, [[0,0],[1,0],[0,0]])
        return pcm - z * pcm_padded[:, :-1, :]
    
    @staticmethod
    def _norm_residual(resid, eps=1e-8):
        """
        Per-window RMS normalisation so that           E[ r² ] = 1
        This keeps α_w in a predictable ≈dB range.
        """
        rms = tf.sqrt(tf.reduce_mean(tf.square(resid), axis=1, keepdims=True))
        return resid / (rms + eps) 
    
    @staticmethod
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

    # sign & whiten carriers
    def _resid_sign(self, r):       # ±1
        r = self._norm_residual(r)
        c = tf.sign(r)
        return tf.where(c == 0., 1., c)

    def _resid_whiten(self, r):     # zero-mean → unit-RMS → fixed RMS → ±1
        m   = tf.reduce_mean(r, axis=1, keepdims=True)
        std = tf.math.reduce_std(r - m, axis=1, keepdims=True) + 1e-6
        w   = (r - m) / std
        w   = w * self.target_rms #keep the RMS at 1 (don’t multiply by target_rms).
        c   = tf.sign(w)
        return tf.where(c == 0., 1., c)

    # ---------- PN helper ---------------------------------------------------
    def _chips_from_pn(self, bits):
        T = self.T
        chips_pb = tf.cast(bits * 2. - 1., tf.float32)     # (B,P)
        chips_per_bit = (T + self.P - 1) // self.P         # ceil

        # build per-bit PN codes (constant graph)
        seed = [42, 123]
        codes = []
        for i in range(self.P):
            rnd = tf.random.stateless_uniform([chips_per_bit],
                                               seed=[seed[0]+i, seed[1]+17*i])
            codes.append(tf.where(rnd > .5, 1., -1.))
        codes = tf.stack(codes, axis=0)                    # (P, chips_per_bit)

        spread = tf.expand_dims(chips_pb, -1) * codes[None, :, :]  # (B,P,L)
        spread = tf.reshape(spread, [tf.shape(bits)[0], -1])[:, :T]
        return tf.expand_dims(spread, -1)                  # (B,T,1)

    # ---------- residual spreading helper -----------------------------------
    # def _chips_from_resid(self, bits, carrier):
    #     chips_per_bit = (self.T + self.P - 1) // self.P
    #     bits_bip = bits * 2. - 1.
    #     bits_rep = tf.repeat(bits_bip, repeats=chips_per_bit, axis=1)[:, :self.T]
    #     return tf.expand_dims(carrier[:, :, 0] * bits_rep, -1) #break orthogonality

    def _carrier_blocks(self, resid):
        """
        Slice residual into P equal blocks, then (optionally) whiten each block.
        Returns (B, P, L) real-valued carrier.
        """
        # global RMS normalisation keeps α_k in a narrow dB range
        rms   = tf.math.reduce_std(resid, axis=[1,2], keepdims=True) + 1e-6
        resid = resid / rms

        # slice
        L   = (self.T + self.P - 1) // self.P
        pad = L * self.P - self.T
        blocks = tf.reshape(tf.pad(resid, [[0,0],[0,pad],[0,0]]),
                            [-1, self.P, L])                      # (B,P,L)

        if self.resid_method == "whiten":       # default for LP-DSS
            mu  = tf.reduce_mean(blocks, 2, keepdims=True)
            std = tf.math.reduce_std(blocks - mu, 2, keepdims=True) + 1e-6
            blocks = (blocks - mu) / std * self.target_rms          # real values

        # no tf.sign → keep analogue samples
        return blocks                                               # (B,P,L)
    
    def _chips_from_resid(self, bits, carrier):
        # carrier: (B,P,L)  – produced by _carrier_blocks
        bits_b = tf.expand_dims(bits * 2. - 1., -1)      # (B,P,1)
        chips  = carrier * bits_b                        # (B,P,L)
        chips  = tf.reshape(chips, [tf.shape(bits)[0], -1, 1])[:, :self.T, :]
        return chips 
    
    '''
    def _chips_from_resid(self, bits, resid):
        # 1) slice residual into P equal-length blocks
        L   = (self.T + self.P - 1) // self.P          # ceil
        pad = L * self.P - self.T
        blocks = tf.reshape(tf.pad(resid, [[0,0],[0,pad],[0,0]]),
                            [-1, self.P, L])           # (B,P,L)

        # 2) optional per-block whitening / sign
        if self.resid_method == "sign":
            mu  = tf.reduce_mean(blocks, 2, keepdims=True)
            std = tf.math.reduce_std(blocks - mu, 2, keepdims=True) + 1e-6
            blocks = tf.sign((blocks - mu) / std)
            blocks = tf.where(blocks == 0., 1., blocks)

        # 3) spread: each bit multiplies its *own* residual block
        bits_b = tf.expand_dims(bits * 2. - 1., -1)    # (B,P,1)
        chips  = blocks * bits_b                       # (B,P,L)

        # 4) back to (B,T,1)
        chips  = tf.reshape(chips, [tf.shape(bits)[0], -1, 1])[:, :self.T, :]
        return chips
    '''

    # ---------- forward -----------------------------------------------------
    '''
    def call(self, inputs):
        bits, pcm = inputs
        bits = tf.cast(bits, tf.float32)

        # build chips
        if self.spread_method == "lpdss":
            resid   = self._residual(pcm)
            if self.resid_method=="sign":
                carrier = self._resid_sign(resid)
            else:
                carrier = self._resid_whiten(resid)
            chips   = self._chips_from_resid(bits, carrier)
        
        else:                                  # classic DSSS with PN
            chips   = self._chips_from_pn(bits)

        # choose α
        if self.alpha_settings == "adaptive":                       # per-window
            host_mag = tf.reduce_mean(tf.abs(chips), axis=1)        # (B,1)
            a_norm   = self.alpha_net(host_mag,                      # will honour
                                      training=self.alpha_net.trainable)
            alpha    = self.a_min + (self.a_max - self.a_min) * a_norm
            alpha    = tf.expand_dims(alpha, 1)                      # (B,1,1)

        else:                                                       # constant
            # If trainable_alpha==True the scalar will still receive
            # gradients; otherwise it just stays where `set_alpha_value`
            # puts it.
            alpha = tf.reshape(self.alpha_const, [1, 1, 1])         # ★ FIXED
        
        # embed
        watermarked = tf.clip_by_value(pcm + alpha * chips, -1., 1.)
        return watermarked
    '''
    
    def call(self, inputs, training=False):
        """
        bits : (B, P)  {0,1}
        pcm  : (B, T, 1)  float32  host signal
        """
        bits = tf.cast(inputs[0], tf.float32)     # ensure float for maths
        pcm  = inputs[1]

        # ─── build chips (orthogonal LP-DSS or classical DSSS) ──────────
        if self.spread_method == "lpdss":
            resid   = self._residual(pcm)                     # (B,T,1)
            carrier = self._carrier_blocks(resid)             # (B,P,L)  real-valued
            chips   = self._chips_from_resid(bits, carrier)   # (B,T,1)
        else:                                                 # PN DSSS
            chips   = self._chips_from_pn(bits)               # (B,T,1)

        # ─── choose α (constant or adaptive) ────────────────────────────
        if self.alpha_settings == "adaptive":
            # use host-frame magnitude as feature; keep α_net frozen if not trainable
            host_mag = tf.reduce_mean(tf.abs(chips), axis=1, keepdims=True)  # (B,1)
            a_norm   = self.alpha_net(host_mag, training=self.alpha_net.trainable)
            alpha    = self.a_min + (self.a_max - self.a_min) * a_norm       # (B,1)
            alpha    = tf.expand_dims(alpha, 1)                              # (B,1,1)
        else:                                           # constant scalar α
            alpha = tf.reshape(self.alpha_const, [1, 1, 1])                  # (1,1,1)

        # ─── embed & clip  ────────────────────────────────────────────────
        watermarked = tf.clip_by_value(pcm + alpha * chips, -1.0, 1.0)
        return watermarked

    # ---------- (de)serialisation -------------------------------------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(frame_size=self.frame, K=self.K, P=self.P,
                        spread_method=self.spread_method,
                        alpha_settings=self.alpha_settings,
                        alpha=float(self.alpha_const.numpy()),
                        a_min=self.a_min, a_max=self.a_max,
                        trainable_alpha=self.trainable_alpha,
                        lporder=self.order,
                        resid_method=self.resid_method,
                        use_full_lpc=self.use_full_lpc,
                        target_rms=self.target_rms))
        return cfg

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)