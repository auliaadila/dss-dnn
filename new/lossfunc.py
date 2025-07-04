"""
Custom Loss functions and metrics for training/analysis
"""

import numpy as np
import tensorflow as tf

# from tf_funcs import *
from pesq import pesq as pesq_score

# The following loss functions all expect the lpcnet model to output the lpc prediction


def perceptual_loss(y_true, y_pred):
    y_true = tf.cast(y_true, "float32")
    # Normalize by dividing by 32768 to bring PCM values to [-1, 1] range
    y_true = y_true / 32768.0
    y_pred = y_pred / 32768.0
    return tf.reduce_mean(tf.square(y_true - y_pred))


def spectral_loss(y_true, y_pred):
    """Spectral loss using STFT magnitude difference"""
    y_true = tf.cast(y_true, "float32") / 32768.0
    y_pred = y_pred / 32768.0

    # Handle different tensor shapes - extract only the first channel if multi-channel
    if len(y_true.shape) == 3 and y_true.shape[-1] > 1:
        y_true = y_true[:, :, 0:1]
    if len(y_pred.shape) == 3 and y_pred.shape[-1] > 1:
        y_pred = y_pred[:, :, 0:1]

    # Ensure we have 2D tensors for STFT (batch, time)
    if len(y_true.shape) == 3:
        y_true = tf.squeeze(y_true, -1)
    if len(y_pred.shape) == 3:
        y_pred = tf.squeeze(y_pred, -1)

    # Compute STFT
    stft_true = tf.signal.stft(y_true, frame_length=512, frame_step=256)
    stft_pred = tf.signal.stft(y_pred, frame_length=512, frame_step=256)

    # Magnitude loss
    mag_true = tf.abs(stft_true)
    mag_pred = tf.abs(stft_pred)

    return tf.reduce_mean(tf.square(mag_true - mag_pred))


def watermark_robustness_loss(y_true, y_pred):
    """Encourages consistent watermark extraction under attacks"""
    # This expects y_true and y_pred to be the extracted bits
    # Penalize differences between original and extracted bits
    return tf.reduce_mean(tf.square(y_true - y_pred))


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


def bit_consistency_loss(original_bits, extracted_bits):
    """Binary cross-entropy with consistency regularization"""
    # Standard BCE
    bce = tf.keras.losses.binary_crossentropy(original_bits, extracted_bits)

    # Consistency term - penalize bits that are close to 0.5 (uncertain)
    uncertainty = tf.abs(extracted_bits - 0.5)
    consistency_penalty = tf.reduce_mean(
        1.0 - 2.0 * uncertainty
    )  # Higher when bits are near 0.5

    return bce + 0.1 * consistency_penalty


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


def residual_distribution_loss(_, y_pred):
    """Encourage Gaussian-like distribution of watermark residuals"""
    y_pred = tf.cast(y_pred, "float32") / 32768.0

    # Compute statistics
    mean = tf.reduce_mean(y_pred)
    var = tf.reduce_mean(tf.square(y_pred - mean))

    # Penalize non-zero mean and encourage controlled variance
    mean_penalty = tf.square(mean)
    var_penalty = tf.square(var - 0.01)  # Target small variance

    return mean_penalty + var_penalty


def l1_loss(y_true, y_pred):
    """L1 (MAE) loss for time-domain signals"""
    y_true = tf.cast(y_true, "float32") / 32768.0
    y_pred = y_pred / 32768.0

    return tf.reduce_mean(tf.abs(y_true - y_pred))


# not used
def delta_pesq(clean_pcm, wm_pcm, fs=16000):
    """
    Returns a *callable metric* that ignores y_true/y_pred —
    we bind clean & watermarked signals via a closure.
    """

    def _metric(_, __):
        # NOTE: Assumes pcm tensors are (B,T,1) float32 in [-1,1]
        def _batch_pesq(c, w):
            c = c.numpy().squeeze()
            w = w.numpy().squeeze()
            return np.float32(pesq_score(fs, c, c, "wb") - pesq_score(fs, c, w, "wb"))

        return tf.numpy_function(
            _batch_pesq, [clean_pcm, wm_pcm], tf.float32
        )  # Autograd is automatically stopped by tf.numpy_function

    _metric.__name__ = "delta_pesq"
    return _metric
