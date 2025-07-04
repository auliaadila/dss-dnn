import tensorflow as tf

__all__ = [
    "LPAnalysis"
]

# ---------------------------------------------------------------------------
#  LPAnalysis – compute per‑frame LPC residual on the fly
# ---------------------------------------------------------------------------
class LPAnalysis(tf.keras.layers.Layer):
    """Frame‑wise LPC analysis that outputs the prediction residual.

    *This layer is NOT trainable.*  By default it runs the Levinson–Durbin
    algorithm inside a `tf.numpy_function`, which breaks the gradient – we
    generally **do not** want gradients to flow through the LPC step anyway.

    Parameters
    ----------
    order : int
        LPC order (typ. 16 for 16 kHz speech).
    frame_size : int
        Number of samples per analysis frame (e.g. 160 → 10 ms @ 16 kHz).
    """

    def __init__(self, order: int = 16, frame_size: int = 160, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.order = int(order)
        self.frame_size = int(frame_size)

    # -------------------------------------------------------------------
    #  Numpy helper – executed in graph via tf.numpy_function
    # -------------------------------------------------------------------
    def _lpc_residual_np(self, pcm_np):
        import numpy as np
        from scipy.signal import lfilter

        N = pcm_np.shape[-1]
        n_frames = N // self.frame_size
        pcm_np = pcm_np[:, : n_frames * self.frame_size]  # truncate to multiple
        pcm_np = pcm_np.reshape(-1, n_frames, self.frame_size)

        residual_buf = np.zeros_like(pcm_np)
        for b in range(pcm_np.shape[0]):
            for t in range(n_frames):
                frame = pcm_np[b, t]
                # 1) autocorrelation
                r = np.correlate(frame, frame, mode="full")[self.frame_size - 1 : self.frame_size + self.order]
                # 2) Levinson–Durbin recursion
                a = np.zeros(self.order + 1)
                a[0] = 1.0
                e = r[0]
                for i in range(1, self.order + 1):
                    acc = r[i] - np.dot(a[1:i], r[1:i][::-1])
                    k = acc / e
                    a[1:i] -= k * a[i - 1 : 0 : -1]
                    a[i] = k
                    e *= 1.0 - k * k
                # 3) residual via inverse filter
                residual_buf[b, t] = lfilter(a, [1.0], frame)
        return residual_buf.reshape(residual_buf.shape[0], -1).astype(np.float32)

    # -------------------------------------------------------------------
    def call(self, pcm):
        # Ensure shape (B, T)
        if pcm.shape.rank == 3:  # (B, T, 1)
            pcm = tf.squeeze(pcm, -1)
        residual = tf.numpy_function(self._lpc_residual_np, [pcm], tf.float32)
        residual.set_shape(pcm.shape)
        residual = tf.expand_dims(residual, -1)  # restore channel dim
        return residual

    def get_config(self):
        cfg = super().get_config()
        cfg.update(order=self.order, frame_size=self.frame_size)
        return cfg
