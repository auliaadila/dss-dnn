import tensorflow as tf
import tensorflow.signal as tfs


def spectral_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Log-magnitude STFT loss for float32 PCM already in [-1,1].
    """
    y_true = tf.squeeze(y_true, -1)  # (B,T)
    y_pred = tf.squeeze(y_pred, -1)

    stft_args = dict(frame_length=512, frame_step=256, window_fn=tf.signal.hann_window)
    S_true = tfs.stft(y_true, **stft_args)
    S_pred = tfs.stft(y_pred, **stft_args)

    mag_true = tf.abs(S_true) + 1e-7
    mag_pred = tf.abs(S_pred) + 1e-7
    log_diff = tf.math.log(mag_true) - tf.math.log(mag_pred)
    return tf.reduce_mean(tf.square(log_diff))


def bit_consistency_loss(bits: tf.Tensor, logits: tf.Tensor, lam: float = 0.5) -> tf.Tensor:
    """
    BCE + penalty for predictions close to 0.5.
    """
    bce = tf.keras.losses.binary_crossentropy(bits, logits)  # (B,P)
    bce = tf.reduce_mean(bce)

    uncertainty = tf.abs(logits - 0.5)  # 0 … 0.5
    penalty = tf.reduce_mean(1.0 - 2.0 * uncertainty)
    return bce + lam * penalty


def perceptual_loss_fixed(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    MSE loss for float32 PCM already in [-1,1] (fixed scaling).
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))


def snr_loss_fixed(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    SNR-based loss for float32 PCM already in [-1,1] (fixed scaling).
    """
    eps = 1e-8
    signal_power = tf.reduce_mean(tf.square(y_true), axis=-1, keepdims=True)
    noise_power = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1, keepdims=True)
    snr = 10.0 * tf.math.log(signal_power / (noise_power + eps)) / tf.math.log(10.0)
    return tf.reduce_mean(-snr)  # Negative because we want to maximize SNR


def residual_distribution_loss_fixed(
    residual: tf.Tensor, target_var: float = 0.01
) -> tf.Tensor:
    """
    Encourage residual to have target variance (fixed reduction).
    """
    actual_var = tf.reduce_mean(tf.square(residual))
    return tf.square(actual_var - target_var)
