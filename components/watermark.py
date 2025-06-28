from typing import List, Tuple

import tensorflow as tf
from components.lpanalysis import LPAnalysis

class WatermarkSpreading(tf.keras.layers.Layer):
    def __init__(self, frame_size=160, alpha=0.05, lp_order=16, **kwargs):
        super().__init__(**kwargs)
        self.frame_size = frame_size
        self.alpha_init = alpha
        self.lp = LPAnalysis(order=lp_order, frame_size=frame_size)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=tf.constant_initializer(self.alpha_init),
            trainable=True,
        )

    def call(self, inputs):
        bits, pcm = inputs
        res = self.lp(pcm)
        time_steps = tf.shape(res)[1] 

        # expand bits to res shape
        bits_expanded = tf.expand_dims(bits, axis=1) 
        bits_tiled = tf.tile(bits_expanded, [1, time_steps, 1])
        bits = 2.*bits_tiled-1.

        # watermark spreading
        wm_per_bit = bits * res                      # broadcast mult 
        wm_single  = self.alpha * tf.reduce_sum(
                        wm_per_bit, axis=-1, keepdims=True)  # (B,T,1)
        return wm_single


    def get_config(self):
        config = super().get_config()
        config.update({
            "frame_size": self.frame_size,
            "alpha": self.alpha_init
        })
        return config
    
class WatermarkAddition(tf.keras.layers.Layer):
    def __init__(self, frame_size=160, beta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.frame_size = frame_size
        self.beta_init = beta

    def build(self, input_shape):
        self.beta = self.add_weight(
            name="beta",
            shape=(),
            initializer=tf.constant_initializer(self.beta_init),
            trainable=False,
        )

    def call(self, pcm, res_w):
        # beta = tf.clip_by_value(self.beta, 0.0, 1.0)  # 👈 adjust bounds as needed
        return pcm + self.beta * res_w

    def get_config(self):
        config = super().get_config()
        config.update({
            "frame_size": self.frame_size,
            "beta": self.beta_init
        })
        return config