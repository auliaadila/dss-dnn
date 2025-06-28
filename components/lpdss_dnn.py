# model definition

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from components.watermark import WatermarkSpreading, WatermarkAddition
from components.extractor import WatermarkExtractor
from components.attack import *
from tensorflow.keras.constraints import Constraint

# BITS_PER_FRAME=64

class AttackScheduler(Callback):
    def __init__(
        self,
        model,
        stage1_end=40,
        stage2_end=80,
        stage3_end=120,
        max_strength_multiplier=3.0,
    ):
        super(AttackScheduler, self).__init__()
        self.model_ref = model
        self.stage1_end = stage1_end
        self.stage2_end = stage2_end
        self.stage3_end = stage3_end
        self.max_strength_multiplier = max_strength_multiplier

    def on_epoch_begin(self, epoch, logs=None):
        # Stage 1 (epochs 0-39): No attacks, multiplier = 0
        if epoch < self.stage1_end:
            strength_multiplier = 0.0  # No attacks
            stage = 1

        # Stage 2 (epochs 40-79): Medium attacks, multiplier 0.5 to 2.0
        elif epoch < self.stage2_end:
            progress = (epoch - self.stage1_end) / (self.stage2_end - self.stage1_end)
            strength_multiplier = 0.5 + progress * 1.5  # 0.5 -> 2.0
            stage = 2

        # Stage 3 (epochs 80-119): Strong attacks, multiplier 2.0 to 3.0
        else:
            progress = (epoch - self.stage2_end) / (self.stage3_end - self.stage2_end)
            strength_multiplier = 2.0 + progress * 1.0  # 2.0 -> 3.0
            stage = 3

        # Update attack layers in the model
        self._update_attack_layers(strength_multiplier)

        print(
            f"Epoch {epoch + 1} (Stage {stage}): Attack strength multiplier = {strength_multiplier:.3f}"
        )

    def _update_attack_layers(self, multiplier):
        # Find and update attack layers
        attack_logs = []

        # Special case: if multiplier is 0, disable all attacks
        if multiplier == 0.0:
            for layer in self.model_ref.layers:
                class_name = layer.__class__.__name__

                if class_name == "AdditiveNoise":
                    layer.strength = 0.0
                    layer.prob = 0.0
                    attack_logs.append(
                        f"  AdditiveNoise: DISABLED (strength=0.000000, prob=0.0%)"
                    )

                elif class_name == "CuttingSamples":
                    layer.num = 0
                    layer.prob = 0.0
                    attack_logs.append(
                        f"  CuttingSamples: DISABLED (num_samples=0, prob=0.0%)"
                    )

                elif class_name == "LowpassFilter":
                    # Set cutoff very high to effectively disable filtering
                    high_cutoff = 8000  # Nyquist frequency for 16kHz sampling
                    from attack import design_lowpass

                    kernel = design_lowpass(high_cutoff)
                    layer.kernel.assign(kernel[:, None, None])
                    attack_logs.append(
                        f"  LowpassFilter: DISABLED (cutoff={high_cutoff:.0f}Hz)"
                    )

                elif class_name == "ButterworthFilter":
                    layer.prob = 0.0
                    attack_logs.append(f"  ButterworthFilter: DISABLED (prob=0.0%)")
        else:
            # Normal case: scale attacks based on multiplier
            for layer in self.model_ref.layers:
                class_name = layer.__class__.__name__

                if class_name == "AdditiveNoise":
                    # Update noise strength
                    original_strength = 0.005  # NOISE_STRENGTH from attacks_.py
                    max_strength = original_strength * self.max_strength_multiplier
                    layer.strength = original_strength * multiplier

                    # Also update probability
                    original_prob = 30  # PROB_COEFF from attacks_.py
                    max_prob = min(100, original_prob * self.max_strength_multiplier)
                    layer.prob = min(100, original_prob * multiplier)

                    attack_logs.append(
                        f"  AdditiveNoise: strength={layer.strength:.6f} (max={max_strength:.6f}), prob={layer.prob:.1f}% (max={max_prob:.1f}%)"
                    )

                elif class_name == "CuttingSamples":
                    # Update number of samples to cut
                    original_num = 200  # NUM_SAMPLES_CUT from attacks_.py
                    max_num = int(original_num * self.max_strength_multiplier)
                    layer.num = int(original_num * multiplier)

                    # Also update probability
                    original_prob = 30  # PROB_COEFF from attacks_.py
                    max_prob = min(100, original_prob * self.max_strength_multiplier)
                    layer.prob = min(100, original_prob * multiplier)

                    attack_logs.append(
                        f"  CuttingSamples: num_samples={layer.num} (max={max_num}), prob={layer.prob:.1f}% (max={max_prob:.1f}%)"
                    )

                elif class_name == "LowpassFilter":
                    # Update cutoff frequency (lower cutoff = stronger attack)
                    original_cutoff = 4000  # Default cutoff from attacks_.py
                    min_cutoff = 2000  # Minimum cutoff frequency
                    max_cutoff_reduction = max(
                        min_cutoff, original_cutoff / self.max_strength_multiplier
                    )

                    # Inverse relationship: higher multiplier = lower cutoff
                    new_cutoff = max(min_cutoff, original_cutoff / multiplier)

                    # Recreate kernel with new cutoff
                    from attack import design_lowpass

                    kernel = design_lowpass(new_cutoff)
                    layer.kernel.assign(kernel[:, None, None])

                    attack_logs.append(
                        f"  LowpassFilter: cutoff={new_cutoff:.0f}Hz (min={max_cutoff_reduction:.0f}Hz)"
                    )

                elif class_name == "ButterworthFilter":
                    # For weak attacks, use high cutoff (less filtering)
                    # For strong attacks, use low cutoff (more filtering)
                    original_cutoff = 4000
                    min_cutoff = 1000  # Strongest attack
                    max_cutoff = 7000  # Weakest attack (almost no filtering)

                    # Linear interpolation: weak attacks = high cutoff
                    new_cutoff = (
                        max_cutoff
                        - (multiplier - 0.5) * (max_cutoff - min_cutoff) / 2.5
                    )
                    new_cutoff = max(min_cutoff, min(max_cutoff, new_cutoff))

                    # Recreate filter coefficients with proper normalization
                    nyquist = 0.5 * 16000  # 8000 Hz
                    normalized_cutoff = new_cutoff / nyquist

                    order = 4  # Default order from attacks_.py
                    layer.b, layer.a = butter(order, normalized_cutoff, "low")

                    # Also update probability
                    original_prob = 30  # PROB_COEFF from attacks_.py
                    max_prob = min(100, original_prob * self.max_strength_multiplier)
                    layer.prob = min(100, original_prob * multiplier)

                    attack_logs.append(
                        f"  ButterworthFilter: cutoff={new_cutoff:.0f}Hz prob={layer.prob:.1f}% (max={max_prob:.1f}%)"
                    )

        # Print all attack parameter updates
        if attack_logs:
            print("  Attack parameters updated:")
            for log in attack_logs:
                print(log)

'''
class WeightClip(Constraint):
    """Clips the weights incident to each hidden unit to be inside a range"""

    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        # Ensure that abs of adjacent weights don't sum to more than 127. Otherwise there's a risk of
        # saturation when implementing dot products with SSSE3 or AVX2.
        return (
            self.c
            * p
            / tf.maximum(
                self.c, tf.repeat(tf.abs(p[:, 1::2]) + tf.abs(p[:, 0::2]), 2, axis=1)
            )
        )
        # return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {"name": self.__class__.__name__, "c": self.c}


constraint = WeightClip(0.992)
'''

def build_model(
        batch_size=32,
        lp_order=16,
        bits_per_frame=64,
        frame_size=160,
        seq_frames=15,
        training=False
):
    bits_in = tf.keras.Input(shape=bits_per_frame, 
                             batch_size=batch_size, name="bits")
    # pcm_in = tf.keras.Input(shape=(None,1), batch_size=batch_size, name="pcm_clean") #dynamic
    pcm_in  = tf.keras.Input(shape=(frame_size*seq_frames, 1), 
                             batch_size=batch_size, name="pcm_clean")

    # Watermark embedding
    wm_embed   = WatermarkSpreading(frame_size=frame_size,
                                    lp_order=lp_order,
                                #  alpha=0.05, # trainable?
                                 name="wm_embed")
    res_w = wm_embed([bits_in, pcm_in])
    wm_add   = WatermarkAddition(beta=0.10, 
                                name="wm_add")
    pcm_w = wm_add(pcm_in, res_w)
    
    # Attacks
    attacked_w = LowpassFilter()(pcm_w)  # ⇽ water-marked signal
    attacked_w = AdditiveNoise()(attacked_w)
    attacked_w = CuttingSamples()(attacked_w)

    # Watermark extraction
    wm_extract = WatermarkExtractor(
        time_len=None, bits_per_frame=64, use_global_pool=True
    )
    attacked_w = tf.ensure_shape(attacked_w, [None, None, 1])
    bits_pred = wm_extract(attacked_w)

    model = Model(
            [bits_in, pcm_in],
            outputs={
                "res_w": res_w,
                "pcm_w": pcm_w,
                "bits_pred": bits_pred,
            },
        )
    
    return model, wm_embed, wm_add, wm_extract