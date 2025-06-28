import os, glob, math, argparse
from typing import List, Tuple
import warnings

import numpy as np
import soundfile as sf
import tensorflow as tf
# import tensorflow_io as tfio

class LPAnalysis(tf.keras.layers.Layer):
    def __init__(self, order=16, frame_size=160, **kw):
        super().__init__(trainable=False, **kw)
        self.order = order
        self.frame_size = frame_size

    def _lpc_residual(self, x):
        import numpy as np
        import scipy.signal as ss

        B, T = x.shape
        n_frames = T // self.frame_size
        x = x.reshape(B, n_frames, self.frame_size)
        res = np.zeros_like(x)
        for b in range(B):
            for f in range(n_frames): #process per frame
                frm = x[b, f]
                r = np.correlate(frm, frm, mode='full')[self.frame_size-1 : self.frame_size+self.order]
                # Levinson‑Durbin
                a = np.zeros(self.order+1)
                a[0]=1
                e=r[0]
                for i in range(1, self.order+1):
                    acc = r[i] - np.dot(a[1:i], r[1:i][::-1])
                    k = acc/e
                    a[1:i] -= k*a[i-1:0:-1]
                    a[i]=k
                    e *= (1-k*k)
                res[b, f] = ss.lfilter(a, [1], frm)
        res = res.reshape(B, -1).astype(np.float32)
        
        return res

    def call(self, x):
        if x.shape.rank==3: x=tf.squeeze(x,-1)
        r = tf.numpy_function(self._lpc_residual, [x], tf.float32) #check this: shape, logic
        r.set_shape(x.shape) #check this: shape
        r = tf.expand_dims(r, -1)
        print("LPC RESIDUAL:", r.shape)
        return r
    
class WatermarkEmbedding(tf.keras.layers.Layer):
   
    def __init__(self,
                 frame_size     = 160,
                 bits_per_frame = 64,
                 alpha_init     = 0.05,
                 trainable_alpha= False,
                 **kwargs):
        super().__init__(**kwargs)
        self.frame_size      = frame_size
        self.bits_per_frame = bits_per_frame
        self.alpha_init      = alpha_init
        self.trainable_alpha = trainable_alpha
        self.lp = LPAnalysis(frame_size=frame_size)

    def build(self, input_shape):
        if self.trainable_alpha:
            self.alpha = self.add_weight(name='alpha',
                                         shape=(),
                                         initializer=tf.constant_initializer(
                                             self.alpha_init),
                                         trainable=True)
        else:
            self.alpha = tf.constant(self.alpha_init, dtype=tf.float32)
        super().build(input_shape)

    # def call(self, inputs):
    #     bits_in, pcm = inputs            # unpack
    #     # LP analysis
    #     residual = self.lp(pcm)
    #     print("EMBEDDING residual:", residual.shape) #EMBEDDING residual: (None, 2400, 1)
    #     # process WM (B, 64) -> (B, None, 1)
    #     time_steps = tf.shape(residual)[1] 
    #     bits_expanded = tf.expand_dims(bits_in, axis=1)  # (B, 64) -> (B, 1, 64) #(128, 1, 64)
    #     bits_tiled = tf.tile(bits_expanded, [1, time_steps, 1])  # (B, 1, 64) -> (B, time, 64) #(128, None, 64)

    #     bits_in = tf.cast(bits_tiled * 2 - 1, tf.float32)      # (B,T,bits_per_frame) -> {-1,+1}

    #     print("bits_in:", bits_in.shape)
    #     # print("residual:", residual.shape)

    #     # wm_bits_per_frame: (128, None, 64)
    #     # residual: (128, None, 1)
        
    #     # Element-wise multiplication with broadcasting (B,T,64) * (B,T,1) = (B,T,64)
    #     wm_per_bit = bits_in * residual

    #     # Sum across bits and apply alpha scaling
    #     wm_single = self.alpha * tf.reduce_sum(wm_per_bit, axis=-1, keepdims=True)  # (B,T,1)
        
    #     return wm_single
    def call(self, inputs):
        bits_in, pcm = inputs                     # bits_in: (B, 64)
        residual = self.lp(pcm)                   # (B, T, 1)

        # ---- bit spreading --------------------------------------------------
        # time_steps = tf.shape(residual, out_type=tf.int32)[1]
        # bits_expanded = tf.expand_dims(bits_in, axis=1)
        # # bits_tiled = tf.tile(bits_in, [1, time_steps, 1]) 
        # bits_tiled = tf.repeat(bits_expanded[:, tf.newaxis, :],  # (B,1,64)
        #                     repeats=time_steps,
        #                     axis=1)                     # (B,T,64)
        # chips = tf.cast(bits_tiled * 2 - 1, tf.float32)    # {-1,+1}

        print("=====> WATERMARK EMBEDDING")
        print("bits_in:", bits_in.shape) #(None, 15, 64)
        print("residual:", residual.shape) #(None, 2400, 1)

        wm_per_bit = bits_in * residual                      # broadcast mult
        wm_single  = self.alpha * tf.reduce_sum(
                        wm_per_bit, axis=-1, keepdims=True)  # (B,T,1)
        return wm_single


    
    def get_config(self):
        base = super().get_config()
        base.update(dict(frame_size=self.frame_size,
                         bits_per_frame=self.bits_per_frame,
                         alpha_init=self.alpha_init,
                         trainable_alpha=self.trainable_alpha))
#         return base

class WatermarkEmbedding(tf.keras.layers.Layer):
    def __init__(self, frame_size=160, alpha_init=0.05, **kw):
        super().__init__(**kw)
        self.frame_size=frame_size
        self.alpha = tf.Variable(alpha_init, trainable=True, dtype=tf.float32) #change this into adaptive, or learnable
        # self.spread = SpreadLayer(frame_size) #bits_in (-1, pcm_len, 1)
        self.lp = LPAnalysis(frame_size=frame_size)
        print("alpha:", self.alpha.shape)
    def call(self, inputs):
        bits, pcm = inputs
        res = self.lp(pcm)
        # chips = self.spread(bits, tf.shape(pcm)[1])  # (-1, T, 1)
        print("=====> WATERMARK SHAPE")
        tf.print("pcm shape:", tf.shape(pcm))
        tf.print("res shape:", tf.shape(res))
        tf.print("bits shape:", tf.shape(bits))

        # print("pcm shape:", tf.shape(pcm))
        # print("res shape:", tf.shape(res))
        # print("call chips shape:", tf.shape(chips))

        return res + self.alpha * bits


# ---------------------------------------------------------------------------
# 4) WatermarkAddition, ChannelSim, Extractor
# ---------------------------------------------------------------------------
# class WatermarkAddition(tf.keras.layers.Layer):
#     def __init__(self, frame_size=160, beta=0.1, **kw):
#         super().__init__(**kw)
#         self.frame_size=frame_size
#         self.beta = tf.Variable(beta, trainable=False, dtype=tf.float32) #change this into adaptive, or learnable
#     def call(self, pcm, res_w):
#         return pcm + self.beta*res_w
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "beta": self.beta
#         })
#         return config

## learnable
import tensorflow as tf

class WatermarkAddition(tf.keras.layers.Layer):
    def __init__(self, frame_size=160, beta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.frame_size = frame_size
        self.beta_val = beta  # raw float, for config export

    def build(self, input_shape):
        # Learnable scalar weight
        self.beta = self.add_weight(
            name="beta",
            shape=(),
            initializer=tf.constant_initializer(self.beta_val),
            trainable=True,
            dtype=tf.float32
        )

    def call(self, pcm, res_w):
        return pcm + self.beta * res_w

    def get_config(self):
        config = super().get_config()
        config.update({
            "frame_size": self.frame_size,
            "beta": self.beta_val  # raw float, not tf.Variable
        })
        return config