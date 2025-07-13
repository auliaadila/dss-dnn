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

def frequency_masking_loss(y_true, y_pred):
    """Perceptual loss with frequency masking (simpler version)"""
    y_true = tf.cast(y_true, "float32") / 32768.0
    y_pred = y_pred / 32768.0

    if len(y_true.shape) == 3:
        y_true = tf.squeeze(y_true, -1)
    if len(y_pred.shape) == 3:
        y_pred = tf.squeeze(y_pred, -1)

    # Compute STFT
    stft_true = tf.signal.stft(y_true, frame_length=512, frame_step=256)
    stft_pred = tf.signal.stft(y_pred, frame_length=512, frame_step=256)

    # Magnitude and phase
    mag_true = tf.abs(stft_true)
    mag_pred = tf.abs(stft_pred)

    # Weight loss by magnitude (louder frequencies matter more)
    weights = mag_true + 1e-8
    weighted_loss = weights * tf.square(mag_true - mag_pred)

    return tf.reduce_mean(weighted_loss) / tf.reduce_mean(weights)

def l1_loss(y_true, y_pred):
    """L1 (MAE) loss for time-domain signals"""
    y_true = tf.cast(y_true, "float32") / 32768.0
    y_pred = y_pred / 32768.0

    return tf.reduce_mean(tf.abs(y_true - y_pred))

def snr_loss(clean_signal, watermarked_signal, eps=1e-8):
    clean_signal = tf.cast(clean_signal, "float32") / 32768.0
    watermarked_signal = tf.cast(watermarked_signal, "float32") / 32768.0

    noise = watermarked_signal - clean_signal
    p_sig = tf.reduce_mean(tf.square(clean_signal))
    p_noise = tf.reduce_mean(tf.square(noise))

    # NEW: avoid log(0) and division by 0
    p_sig = tf.maximum(p_sig, eps)
    p_noise = tf.maximum(p_noise, eps)

    snr = 10.0 * tf.math.log(p_sig / p_noise) / tf.math.log(10.0)
    return -snr

# audio–quality loss  (pcm head)
def pcm_loss(y_true, y_pred, λ_l1, λ_snr, λ_mask, gamma_bits):
    l1 = λ_l1 * l1_loss(y_true, y_pred)
    snr = λ_snr * snr_loss(y_true, y_pred)
    fmsk = λ_mask * frequency_masking_loss(y_true, y_pred)
    return (1.0 - gamma_bits) * (l1 + snr + fmsk)

