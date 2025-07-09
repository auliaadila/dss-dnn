import tensorflow as tf

from components.attacks_gpu import (
    AdditiveNoiseGPU,
    ButterworthIIRGPU,
    CuttingSamplesGPU,
    LowpassFIRGPU,
)


class AttackPipelineFixed(tf.keras.layers.Layer):
    """Fixed attack pipeline with proper training flag propagation."""

    def __init__(
        self, fir_cutoff=0.8, iir_cutoff=0.8, snr_db=20.0, max_cut_ratio=0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.fir = LowpassFIRGPU(cutoff=fir_cutoff)
        self.iir = ButterworthIIRGPU(cutoff=iir_cutoff)
        self.noise = AdditiveNoiseGPU(snr_db=snr_db)
        self.cut = CuttingSamplesGPU(max_cut_ratio=max_cut_ratio)
        self._warm_up = False  # new flag

    def call(self, x, training=None):
        """Apply attack pipeline with proper training flag forwarding."""
        # During warm-up return x unchanged - bypass ALL attacks completely
        # This prevents noise/cutting layers from seeing training=True
        if self._warm_up:
            return x

        x = self.fir(x)
        x = self.iir(x)
        x = self.noise(x, training=training)  # Training-dependent
        x = self.cut(x, training=training)
        return x

    def set_warm_up_mode(self, warm_up=True):
        """Enable / disable all attacks."""
        self._warm_up = bool(warm_up)
        if warm_up:
            # later re-enable values will be restored in caller
            return
        # restore default strengths once warm-up ends
        self.noise.snr_db = 20.0
        self.cut.max_cut_ratio = 0.1

    def set_gradual_attacks(self, phase="mild"):
        """Set gradual attack strength for smoother transition."""
        self._warm_up = False
        if phase == "mild":
            # Weaker attacks for initial transition
            self.noise.snr_db = 35.0  # Very mild noise
            self.cut.max_cut_ratio = 0.05  # Light cutting
        elif phase == "moderate":
            # Moderate attacks
            self.noise.snr_db = 25.0
            self.cut.max_cut_ratio = 0.08
        elif phase == "full":
            # Full strength
            self.noise.snr_db = 20.0
            self.cut.max_cut_ratio = 0.1
