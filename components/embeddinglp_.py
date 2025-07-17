# embedder.py
# ───────────
import tensorflow as tf
import math


class LPDSSEmbedder(tf.keras.layers.Layer):
    """
    DNN–LP-DSS embedder with orthogonal time slicing and hybrid α_k = a_k·g_k

    Args
    ----
    frame_size   : samples per LPC frame   (default 160 → 10 ms @16 kHz)
    K            : frames per payload      (default 15  → 240 ms window)
    P            : payload bits            (e.g. 64)
    lporder      : LPC order (full-LPC mode)
    ssl_db       : strength–setting level (MATLAB ‘ssl’, dB rel. host)
    resid_whiten : True or False
    alpha_settings : "adaptive" (g_k learnable)  or  "constant"
    use_full_lpc : True ⇒ exact LPC residual  /  False ⇒ fast 1-pole HPF
    """

    def __init__(self,
                 frame_size=160, K=15, P=64,
                 lporder=12,
                 ssl_db=-10,
                 alpha_settings='adaptive',
                 trainable_alpha = False,
                 resid_sign=False,
                 resid_whiten = True,
                 use_full_lpc=True,
                 alpha_gate_init=0.5,
                 **kw):
        super().__init__(**kw)

        # ---------- hyper-params -----------------------------------------
        self.frame   = int(frame_size)
        self.K       = int(K)
        self.P       = int(P)
        self.T       = self.frame * self.K
        self.L       = math.ceil(self.T / self.P)       # samples per bit block
        self.ord     = int(lporder)
        self.ssl_db  = float(ssl_db)
        self.alpha_settings = alpha_settings
        self.use_full_lpc   = bool(use_full_lpc)
        self.resid_whiten = bool(resid_whiten)
        self.trainable_alpha = trainable_alpha
        self.resid_sign = bool(resid_sign)

        # ---------- learnable gate g_k  ----------------------------------
        # bias init so that sigmoid(b) ≈ alpha_gate_init
        b0 = math.log(alpha_gate_init / max(1e-6, 1-alpha_gate_init))
        self.alpha_net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid',
                bias_initializer=tf.keras.initializers.Constant(b0))
        ])
        # You can still ignore it by setting .trainable = False
        if self.alpha_settings != "adaptive":
            self.alpha_net.trainable = False
        else:
            self.alpha_net.trainable = True

        # expose constant α so it can be set / trained if desired
        target_linear = 10**(ssl_db / 20)
        self.alpha_const = tf.Variable(target_linear, dtype=tf.float32,
                                       trainable=(alpha_settings == "constant" and trainable_alpha),
                                       name='alpha_const')


    # ---------- helpers ----------------------------------------------------
    @staticmethod
    def _gpu_residual(pcm, z=0.95):
        pcm_p = tf.pad(pcm, [[0,0],[1,0],[0,0]])
        return pcm - z * pcm_p[:, :-1, :]
    
    @staticmethod
    def _lpc_residual_numpy(frames, order):
        """
        frames : (N, frame) float32  – windowed speech frames
        return : (N, frame) float32  – LPC residual e[n]
        """
        import numpy as np
        from scipy.signal import lfilter
        import librosa                                # fast Levinson solver

        out = np.empty_like(frames, dtype=np.float32)
        for i, fr in enumerate(frames):
            a = librosa.lpc(fr, order=int(order))          # a[0]=1, a[1:] = –α_k
            out[i] = lfilter(a, 1.0, fr).astype(np.float32)
        return out

    def _residual(self, pcm):
        if not self.use_full_lpc:
            return self._gpu_residual(pcm)

        B = tf.shape(pcm)[0]
        win = tf.signal.hann_window(self.frame, periodic=False) #any choice except hann_window?
        pcm2d = tf.reshape(pcm, [-1, self.frame]) * win            # (B*K, frame)
        res2d = tf.numpy_function(self._lpc_residual_numpy,
                                  [pcm2d, self.ord], tf.float32)
        res2d.set_shape(pcm2d.shape) #added
        return tf.reshape(res2d, [B, self.T, 1])
    
    def _carrier_blocks(self, resid): #support orthogonality
        """
        Slice residual into P equal-length blocks (length L = ceil(T/P)).
        Optionally whiten each block; *no* hard sign → keep
        carrier’s energy distribution.
        """
        # (a) global RMS normalisation (optional but keeps α in dB range)
        rms = tf.math.reduce_std(resid, axis=[1,2], keepdims=True) + 1e-6
        resid = resid / rms                                         # global norm

        # (b) slice to (B, P, L)
        L   = self.L
        pad = L * self.P - self.T
        blocks = tf.reshape(tf.pad(resid, [[0, 0], [0, pad], [0, 0]]),
                            [-1, self.P, L])                   # (B,P,L)

        # (c) per-block processing
        if self.resid_whiten:                    # ◀ only if chosen
            mu  = tf.reduce_mean(blocks, 2, keepdims=True)
            std = tf.math.reduce_std(blocks-mu, 2, keepdims=True)+1e-6
            blocks = (blocks - mu) / std 

        if self.resid_sign:
            # hard sign
            b = tf.sign(blocks)
            blocks = tf.where(b == 0., 1., b)
            
        return blocks                                          # (B,P,L)

    def _chips_from_resid(self, bits, resid):
        """Spread each bit with its own residual block."""
        carrier = self._carrier_blocks(resid)                       # (B,P,L)
        bits_b  = bits * 2.0 - 1.0                                  # bipolar

        chips   = carrier * tf.expand_dims(bits_b, -1)              # (B,P,L)
        chips   = tf.reshape(chips, [tf.shape(bits)[0], -1])        # (B,P*L)
        return tf.expand_dims(chips[:, :self.T], -1)                # (B,T,1)
    
    def _alpha_hybrid(self, pcm, resid, training=False):
        B = tf.shape(pcm)[0]

        pad_len = self.P * self.L - self.T
        pcm_padded   = tf.pad(pcm, [[0, 0], [0, pad_len], [0, 0]])
        resid_padded = tf.pad(resid, [[0, 0], [0, pad_len], [0, 0]])

        s = tf.reshape(pcm_padded,  [B, self.P, self.L])
        r = tf.reshape(resid_padded,[B, self.P, self.L])

        # a_k  from power ratio  +  ssl offset (dB)
        s_rms = tf.sqrt(tf.reduce_mean(s**2, axis=2, keepdims=True)+1e-8)
        r_rms = tf.sqrt(tf.reduce_mean(r**2, axis=2, keepdims=True)+1e-8)
        a_k   = (s_rms / r_rms) * 10**(self.ssl_db/20)    # (B,P,1)

        # g_k gate (only if adaptive)
        if self.alpha_settings == 'adaptive':
            feat = tf.math.log(s_rms + 1e-6)                        # (B,P,1)
            g_k  = self.alpha_net(tf.reshape(feat, [-1,1]), training=training)
            g_k  = tf.reshape(g_k, tf.shape(a_k))
            alpha_k = a_k * g_k
        else:
            alpha_k = tf.ones_like(a_k) * self.alpha_const

        # alpha = tf.reshape(alpha_k, [B, self.P*self.L, 1])[:, :self.T, :]
        alpha = tf.repeat(alpha_k, repeats=self.L, axis=1)[:, :self.T, :]
        return alpha

    # ---------- forward ----------------------------------------------------
    def call(self, inputs, training=False):
        bits, pcm = inputs
        bits = tf.cast(bits, tf.float32)
        
        resid = self._residual(pcm)
        chips = self._chips_from_resid(bits, resid)            # (B,T,1)
        alpha = self._alpha_hybrid(pcm, resid, training)           # (B,T,1)
        return tf.clip_by_value(pcm + alpha*chips, -1., 1.)


    # ---------- (de)serialization -----------------------------------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(frame_size=self.frame, K=self.K, P=self.P,
                        lporder=self.ord,
                        ssl_db=self.ssl_db,
                        resid_whiten=self.resid_whiten,
                        alpha_settings=self.alpha_settings,
                        trainable_alpha = self.trainable_alpha,
                        use_full_lpc=self.use_full_lpc))
        return cfg

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)