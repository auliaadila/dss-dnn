import numpy as np
import tensorflow as tf


class LowpassFIRGPU(tf.keras.layers.Layer):
    """GPU-optimized FIR lowpass filter."""

    def __init__(self, cutoff=0.8, taps=64, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.taps = taps

        # Pre-compute filter coefficients
        self.filter_coeffs = self._design_lowpass_fir(cutoff, taps)

    def _design_lowpass_fir(self, cutoff, taps):
        """Design FIR lowpass filter coefficients."""
        # Simple windowed sinc filter
        n = tf.range(taps, dtype=tf.float32)
        center = (taps - 1) / 2.0

        # Sinc function
        sinc_arg = 2.0 * cutoff * (n - center)
        sinc_arg = tf.where(tf.abs(sinc_arg) < 1e-6, 1e-6, sinc_arg)
        h = tf.sin(tf.constant(np.pi) * sinc_arg) / (tf.constant(np.pi) * sinc_arg)

        # Apply Hamming window
        window = 0.54 - 0.46 * tf.cos(2.0 * tf.constant(np.pi) * n / (taps - 1))
        h = h * window

        # Normalize
        h = h / tf.reduce_sum(h)

        return h

    def call(self, x):
        """Apply FIR filter using GPU-native convolution."""
        # x: (B, T, 1)
        # Reshape filter for conv1d
        filter_kernel = tf.reshape(self.filter_coeffs, [self.taps, 1, 1])

        # Apply padding for same-size output
        padding_left = self.taps // 2
        padding_right = self.taps - padding_left - 1  # Ensure exact same size
        x_padded = tf.pad(
            x, [[0, 0], [padding_left, padding_right], [0, 0]], mode="SYMMETRIC"
        )

        # Convolve
        output = tf.nn.conv1d(x_padded, filter_kernel, stride=1, padding="VALID")

        return output


class ButterworthIIRGPU(tf.keras.layers.Layer):
    """GPU-optimized Butterworth IIR filter approximation."""

    def __init__(self, cutoff=0.8, order=2, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.order = order

        # Pre-compute filter coefficients
        self.b, self.a = self._design_butterworth_iir(cutoff, order)

    def _design_butterworth_iir(self, cutoff, order):
        """Design Butterworth IIR filter coefficients."""
        # Simplified Butterworth design for GPU
        # This is an approximation - for exact results, use scipy on CPU

        # Simple 2nd order lowpass
        if order == 2:
            # Bilinear transform approximation
            wc = tf.constant(np.pi, dtype=tf.float32) * tf.constant(
                cutoff, dtype=tf.float32
            )
            k = tf.tan(wc / 2.0)
            sqrt2 = tf.constant(np.sqrt(2.0), dtype=tf.float32)
            norm = 1.0 / (1.0 + sqrt2 * k + k * k)

            b = tf.stack([k * k * norm, 2.0 * k * k * norm, k * k * norm])
            a = tf.stack(
                [1.0, (2.0 * (k * k - 1.0)) * norm, (1.0 - sqrt2 * k + k * k) * norm]
            )
        else:
            # Fallback to simple 1st order
            pi_f32 = tf.constant(np.pi, dtype=tf.float32)
            cutoff_f32 = tf.constant(cutoff, dtype=tf.float32)
            alpha = tf.exp(-2.0 * pi_f32 * cutoff_f32)
            b = tf.stack([1.0 - alpha, 0.0])
            a = tf.stack([1.0, -alpha])

        return b, a

    def call(self, x):
        """Apply IIR filter using GPU-native operations."""
        # x: (B, T, 1)
        # Use simple moving average as GPU-friendly approximation
        x_squeezed = tf.squeeze(x, -1)  # (B, T)

        # Simple moving average filter (approximates lowpass behavior)
        window_size = 5

        # Pad the signal
        padding = window_size // 2
        x_padded = tf.pad(x_squeezed, [[0, 0], [padding, padding]], mode="SYMMETRIC")

        # Apply moving average using conv1d
        kernel = tf.ones([window_size, 1, 1]) / window_size
        x_expanded = tf.expand_dims(x_padded, -1)
        y = tf.nn.conv1d(x_expanded, kernel, stride=1, padding="VALID")

        return y


class AdditiveNoiseGPU(tf.keras.layers.Layer):
    """GPU-optimized additive noise layer."""

    def __init__(self, snr_db=20.0, **kwargs):
        super().__init__(**kwargs)
        self.snr_db = snr_db

    def call(self, x, training=None):
        """Add noise with specified SNR."""
        if training is False:
            return x
        if training is None:
            training = tf.keras.backend.learning_phase()

        def _noisy():
            # Compute signal power
            signal_power = tf.reduce_mean(tf.square(x), axis=1, keepdims=True)
            # Compute noise power from SNR
            snr_linear = tf.pow(10.0, self.snr_db / 10.0)
            noise_power = signal_power / snr_linear
            # Generate noise
            noise = tf.random.normal(tf.shape(x), stddev=tf.sqrt(noise_power))
            return x + noise

        return tf.cond(tf.cast(training, tf.bool), _noisy, lambda: x)


class CuttingSamplesGPU(tf.keras.layers.Layer):
    """GPU-optimized sample cutting layer."""

    def __init__(self, max_cut_ratio=0.1, **kwargs):
        super().__init__(**kwargs)
        self.max_cut_ratio = max_cut_ratio

    def call(self, x, training=None):
        """Randomly cut samples from the signal."""
        if training is False:
            return x
        if training is None:
            training = tf.keras.backend.learning_phase()

        def _cutting():
            B, T, C = tf.unstack(tf.shape(x))
            # Random cut length (up to max_cut_ratio of signal length)
            max_cut = tf.cast(tf.cast(T, tf.float32) * self.max_cut_ratio, tf.int32)
            # Handle edge case where max_cut_ratio is 0
            max_cut = tf.maximum(max_cut, 1)  # Ensure at least 1 for random.uniform
            cut_length = tf.random.uniform([], 0, max_cut, dtype=tf.int32)
            # If max_cut_ratio was 0, force cut_length to 0
            cut_length = tf.cond(
                tf.equal(max_cut, 1),
                lambda: tf.constant(0, dtype=tf.int32),
                lambda: cut_length,
            )
            # Random cut position
            cut_start = tf.random.uniform(
                [], 0, tf.maximum(T - cut_length, 1), dtype=tf.int32
            )
            # Create mask
            indices = tf.range(T)
            mask = tf.logical_or(indices < cut_start, indices >= cut_start + cut_length)
            # Apply mask
            x_cut = tf.boolean_mask(x, mask, axis=1)
            # Pad back to original length
            pad_length = T - tf.shape(x_cut)[1]
            x_padded = tf.pad(x_cut, [[0, 0], [0, pad_length], [0, 0]], mode="CONSTANT")
            return x_padded

        return tf.cond(tf.cast(training, tf.bool), _cutting, lambda: x)


class AttackPipelineGPU(tf.keras.layers.Layer):
    """GPU-optimized attack pipeline for training robustness."""

    def __init__(
        self, fir_cutoff=0.8, iir_cutoff=0.8, snr_db=20.0, max_cut_ratio=0.1, **kwargs
    ):
        super().__init__(**kwargs)

        self.fir_filter = LowpassFIRGPU(cutoff=fir_cutoff)
        self.iir_filter = ButterworthIIRGPU(cutoff=iir_cutoff)
        self.noise = AdditiveNoiseGPU(snr_db=snr_db)
        self.cutting = CuttingSamplesGPU(max_cut_ratio=max_cut_ratio)

    def call(self, x, training=None):
        """Apply attack pipeline."""
        # Apply attacks in sequence
        x = self.fir_filter(x)
        x = self.iir_filter(x)
        x = self.noise(x, training=training)
        x = self.cutting(x, training=training)

        return x


# Create backward compatibility aliases
LowpassFIR = LowpassFIRGPU
ButterworthIIR = ButterworthIIRGPU
AdditiveNoise = AdditiveNoiseGPU
CuttingSamples = CuttingSamplesGPU
AttackPipeline = AttackPipelineGPU
