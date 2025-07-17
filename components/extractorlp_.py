import tensorflow as tf
import math


class LPDSBlockExtractor(tf.keras.Model):
    """
    LP-DSS extractor that mirrors the embedder's orthogonal block layout.

    Pipeline
    --------
    y'  ->  residual (LPC or 1-pole HPF) -> global whiten
        ->  dilated Conv1D backbone (local features)
        ->  pad & reshape to (B,P,L,F) -> mean over L  (block binning)
        ->  per-bit MLP  -> sigmoid probs (B,P)

    Args
    ----
    payload_bits : int  (P)
    context_len  : int  (T samples per example)
    frame_size   : int  (embedder frame; used to recompute L = ceil(T/P))
    lporder      : int  LPC order (if use_full_lpc=True)
    use_full_lpc : bool replicate embedder's residual (slow CPU fallback)
    base_filters : int  conv channel count
    """

    def __init__(self,
                 payload_bits: int,
                 context_len: int,
                 frame_size: int = 160,
                 lporder: int = 12,
                 use_full_lpc: bool = False,
                 base_filters: int = 64,
                 **kw):
        super().__init__(**kw)
        self.P   = int(payload_bits)
        self.T   = int(context_len)
        self.frm = int(frame_size)
        self.L   = math.ceil(self.T / self.P)   # chips per bit
        self.ord = int(lporder)
        self.use_full_lpc = bool(use_full_lpc)
        self.F   = int(base_filters)

        # ── whitening lambda (global, per example) ───────────────────
        def _whiten(x, eps=1e-8):
            m   = tf.reduce_mean(x, axis=1, keepdims=True)
            rms = tf.sqrt(tf.reduce_mean(tf.square(x - m), axis=1, keepdims=True))
            return (x - m) / (rms + eps)
        self.pre = tf.keras.layers.Lambda(_whiten, name="whiten_global")

        # optional norm to stabilise α drift & batch loudness skew
        self.norm0 = tf.keras.layers.LayerNormalization(name="ln0")

        # ── Conv backbone (dilated stack) ─────────────────────────────
        F = self.F
        self.backbone = tf.keras.Sequential([
            tf.keras.layers.Conv1D(F, 3, padding="same", activation="relu", dilation_rate=1),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv1D(F, 3, padding="same", activation="relu", dilation_rate=2),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv1D(F, 3, padding="same", activation="relu", dilation_rate=4),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv1D(F, 3, padding="same", activation="relu", dilation_rate=8),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv1D(F, 3, padding="same", activation="relu", dilation_rate=16),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv1D(F, 3, padding="same", activation="relu", dilation_rate=32),
            tf.keras.layers.LayerNormalization(name="backbone_out_norm"),
        ], name="backbone")

        # ── per-bit classifier head (TimeDistributed MLP) ─────────────
        head = tf.keras.Sequential([
            tf.keras.layers.Dense(F*4, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ], name="bit_head")
        self.head = tf.keras.layers.TimeDistributed(head, name="td_head")


    # ------------------------------------------------------------------
    # Residual helpers (must mirror embedder)
    # ------------------------------------------------------------------
    @staticmethod
    def _gpu_residual_simple(pcm, z=0.95):
        pcm_p = tf.pad(pcm, [[0,0],[1,0],[0,0]])
        return pcm - z * pcm_p[:, :-1, :]

    @staticmethod
    def _lpc_residual_numpy(frames, order):
        import numpy as np
        from scipy.signal import lfilter
        import librosa
        out = np.empty_like(frames, dtype=np.float32)
        for i, fr in enumerate(frames):
            a = librosa.lpc(fr, order=int(order))
            out[i] = lfilter(a, [1.0], fr).astype(np.float32)
        return out

    def _residual(self, pcm):
        """
        pcm: (B,T,1)
        return: (B,T,1) LPC (or HPF) residual
        """
        if not self.use_full_lpc:
            return self._gpu_residual_simple(pcm)

        B = tf.shape(pcm)[0]
        pcm2d = tf.reshape(pcm, [-1, self.frm])    # (B*K, frm) but K = T/frm may not be int; we floor
        # To guard mismatch, trim to multiple of frame:
        frames_tot = (self.T // self.frm) * self.frm
        pcm_trunc = pcm[:, :frames_tot, :]
        B = tf.shape(pcm_trunc)[0]
        pcm2d = tf.reshape(pcm_trunc, [-1, self.frm])
        win = tf.signal.hann_window(self.frm, periodic=False)
        pcm_win = pcm2d * win
        res2d = tf.numpy_function(self._lpc_residual_numpy,
                                  [pcm_win, self.ord], tf.float32)
        res2d.set_shape(pcm2d.shape)
        res = tf.reshape(res2d, [B, -1, 1])
        # pad tail back if truncated
        pad = self.T - tf.shape(res)[1]
        res = tf.pad(res, [[0,0],[0,pad],[0,0]])
        return res


    # ------------------------------------------------------------------
    # Block binning
    # ------------------------------------------------------------------
    def _block_pool(self, feats):
        """
        feats: (B,T,F)
        Return: (B,P,F) by zero-pad to P*L then mean over L.
        """
        B = tf.shape(feats)[0]
        pad = self.P * self.L - self.T
        feats_pad = tf.pad(feats, [[0,0],[0,pad],[0,0]])
        feats_blk = tf.reshape(feats_pad, [B, self.P, self.L, self.F])
        return tf.reduce_mean(feats_blk, axis=2)   # (B,P,F)


    # ------------------------------------------------------------------
    def call(self, y_prime, training=False):
        """
        y_prime : (B,T,1) attacked watermarked audio segment
        returns : (B,P) bit probabilities
        """
        # 1) residual + global whiten
        r   = self._residual(y_prime)                      # (B,T,1)
        r_w = self.pre(r)                                  # (B,T,1)
        r_w = self.norm0(r_w, training=training)

        # 2) CNN backbone
        z = self.backbone(r_w, training=training)          # (B,T,F)

        # 3) orthogonal block binning
        z_blk = self._block_pool(z)                        # (B,P,F)

        # 4) per-bit classifier
        probs = self.head(z_blk, training=training)        # (B,P,1)
        return tf.squeeze(probs, axis=-1)                  # (B,P)


    # ------------------------------------------------------------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(payload_bits=self.P,
                        context_len=self.T,
                        frame_size=self.frm,
                        lporder=self.ord,
                        use_full_lpc=self.use_full_lpc,
                        base_filters=self.F))
        return cfg

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)