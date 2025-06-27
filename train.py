#!/usr/bin/env python
"""wm_dnn_pipeline.py – *One‑file* demo of a Deep‑Learning LP‑DSS watermark
pipeline.

Run:
  python wm_dnn_pipeline.py  \
         --train_dir data/train --val_dir data/val \
         --epochs 10 --batch 16

What it contains
================
1. **FrameSequence** – streams 150‑ms windows + random bits.
2. **LPAnalysis**    – fixed Levinson–Durbin residual via tf.numpy_function.
3. **SpreadLayer**   – repeats bits across their frames.
4. **WatermarkEmbedding** – LP residual → tanh carrier • α + host.
5. **ChannelSim**    – AWGN + random resample/MP3 (train‑time only).
6. **Extractor**     – tiny ResNet returning per‑frame bits.
7. **Model builder** – stitches everything into Keras graph.
8. **Training stub** – ModelCheckpoint + EarlyStopping.

This is *minimal but runnable*; you can break it into modules later.
"""
import os, glob, math, argparse
from typing import List, Tuple
import warnings

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_io as tfio

# ---------------------------------------------------------------------------
# 1) Data loader – FrameSequence
# ---------------------------------------------------------------------------
class FrameSequence(tf.keras.utils.Sequence):
    """Streams clean PCM + random 64‑bit word repeated over `seq_frames`."""
    # change seq_frames? no need
    def __init__(self, wav_folder: str, frame_size=160, seq_frames=15,
                 bits_per_frame=64, bits_in=None, batch_size=32, shuffle=True, fs=16_000):
        self.paths = sorted(glob.glob(os.path.join(wav_folder, "**", "*.wav"), recursive=True))
        if not self.paths:
            raise RuntimeError("No WAVs found under", wav_folder)
        self.frame_size = frame_size
        self.seq_frames = seq_frames
        self.window = frame_size * seq_frames #2400
        self.bits_per_frame = bits_per_frame
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.fs = fs
        self.indices = []  # (file_idx, start_sample)
        self.bits_in = np.random.randint(0, 2, size=(self.batch_size, self.bits_per_frame), dtype='int32')
        for fi, p in enumerate(self.paths):
            n = sf.info(p).frames
            for s in range(0, n - self.window + 1, self.window):
                self.indices.append((fi, s)) #len?
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx): #one-batch-of-windows-at-a-time
        batch_idx = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        pcm = []
        for fi, st in batch_idx:
            p = self.paths[fi]
            # Get audio
            raw, _ = sf.read(p, start=st, stop=st+self.window, dtype="float32")
            raw_ = np.expand_dims(raw, -1) # (2400, 1)
            pcm.append(raw_)
            # print("DATALOADER raw pcm:", raw_.shape)
            
            # Generate bits
            # Different 64-bit word in each 15 frames
            # bits.append(np.random.randint(0, 2, (self.seq_frames, self.bits_per_frame)))
            
            # one 64-bit word, shared by all 15 frames
            # [later used in SpreadLayer] bits: (B, seq_frames, bits_per_frame)
            # word = np.random.randint(0, 2, (self.batch_size, self.bits_per_frame))      # (1, 64)
            # bits_full = np.repeat(word, self.seq_frames, axis=0)          # (15, 64) #?
            # bits.append(bits_full)

        pcm = np.stack(pcm)
        # bits = np.stack(bits).astype(np.float32)
        ''' PREVIOUS
        print("DATALOADER bits:", bits.shape)
        print("DATALOADER pcm:", pcm.shape)
        DATALOADER bits: (13, 15, 64)
        DATALOADER pcm: (13, 2400, 1)
        DATALOADER bits: (32, 15, 64)
        DATALOADER pcm: (32, 2400, 1)
        '''
        actual_batch_size = len(pcm)
        bits_in = np.random.randint(0, 2, size=(actual_batch_size, self.bits_per_frame), dtype='int32')

        # CURRENT
        print("DATALOADER bits:", bits_in.shape)
        print("DATALOADER pcm:", pcm.shape)
        # DATALOADER bits: (13, 64)
        # DATALOADER pcm: (13, 2400, 1)
        
        # input: [bits_in, pcm], output: bits_in

        return [bits_in, pcm], bits_in  # labels = bits (unused)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ---------------------------------------------------------------------------
# 2) LPAnalysis – residual via numpy_function (no gradients)
# ---------------------------------------------------------------------------
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
            for f in range(n_frames):
                frm = x[b, f]
                r = np.correlate(frm, frm, mode='full')[self.frame_size - 1 : self.frame_size + self.order]
                
                a = np.zeros(self.order + 1, dtype=np.float64)
                a[0] = 1.0
                e = r[0] if r[0] != 0 else 1e-8  # avoid division by zero

                for i in range(1, self.order + 1):
                    acc = r[i] - np.dot(a[1:i], r[1:i][::-1])
                    k = acc / e if e != 0 else 0
                    a_new = a[1:i] - k * a[i-1:0:-1]
                    a[1:i] = a_new
                    a[i] = k
                    e *= (1.0 - k * k)
                    if e <= 1e-8:  # prevent numerical instability
                        e = 1e-8

                res[b, f] = ss.lfilter(a, [1.0], frm)

        return res.reshape(B, -1).astype(np.float32)

    def call(self, x):
        if x.shape.rank == 3:
            x = tf.squeeze(x, -1)  # (B, T)
        r = tf.numpy_function(self._lpc_residual, [x], tf.float32)
        r.set_shape(x.shape)  # match input shape (B, T)
        return tf.expand_dims(r, -1)  # output shape (B, T, 1)

# ---------------------------------------------------------------------------
# 3) SpreadLayer & WatermarkEmbedding (+α)
# ---------------------------------------------------------------------------
'''
class SpreadLayer(tf.keras.layers.Layer):
    def __init__(self, frame_size=160, **kw):
        super().__init__(**kw)
        self.frame_size=frame_size
    def call(self, bits, pcm_len):
        # print("SPREAD LAYER")
        # print("bits:", bits.shape)
        # print("pcm_len:", pcm_len.shape)
        time_steps = tf.shape(residual)[1] 
        bits_expanded = tf.expand_dims(bits_in, axis=1)  # (B, 64) -> (B, 1, 64)
        bits_tiled = tf.tile(bits_expanded, [1, time_steps, 1])  # (B, 1, 64) -> (B, time, 64)
        # chips = tf.repeat(bits, repeats=self.frame_size, axis=1)
        # chips = tf.reshape(chips, (-1, pcm_len, 1))
        print("chips:", chips.shape)
        return 2.*chips-1.
'''

class WatermarkEmbedding(tf.keras.layers.Layer):
    def __init__(self, frame_size=160, alpha_init=0.05, **kw):
        super().__init__(**kw)
        self.frame_size=frame_size
        self.alpha_init = tf.Variable(alpha_init, trainable=True, dtype=tf.float32) #change this into adaptive, or learnable
        # self.spread = SpreadLayer(frame_size) #bits_in (-1, pcm_len, 1)
        self.lp = LPAnalysis(frame_size=frame_size)
        # print("alpha:", self.alpha.shape)
    def call(self, inputs):
        bits, pcm = inputs
        res = self.lp(pcm)
        # chips = self.spread(bits, tf.shape(pcm)[1])  # (-1, T, 1)
        # print("=====> WATERMARK SHAPE")

        # import IPython
        # IPython.embed()

        # tf.print("pcm shape:", tf.shape(pcm))
        # print("res shape:", res.shape)
        # print("init bits shape:", bits.shape)
        

        # print("pcm shape:", tf.shape(pcm))
        # print("res shape:", tf.shape(res))
        # print("call chips shape:", tf.shape(chips))

        # Incompatible shapes: [32,64] vs. [32,2400,1]
        time_steps = tf.shape(res)[1] 
        bits_expanded = tf.expand_dims(bits, axis=1) 
        # print("expanded bits shape:", bits_expanded.shape)
        bits_tiled = tf.tile(bits_expanded, [1, time_steps, 1])
        # print("tiled bits shape:", bits_tiled.shape)
        bits = 2.*bits_tiled-1.

        wm_per_bit = bits * res                      # broadcast mult 
        # print("wm_per_bit shape:", wm_per_bit.shape)
        wm_single  = self.alpha_init * tf.reduce_sum(
                        wm_per_bit, axis=-1, keepdims=True)  # (B,T,1)
        # print("wm_single shape:", wm_single.shape)
        return wm_single
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "frame_size": self.frame_size,
            "alpha_init": self.alpha_init,
        })
        return config


# ---------------------------------------------------------------------------
# 4) WatermarkAddition, ChannelSim, Extractor
# ---------------------------------------------------------------------------
class WatermarkAddition(tf.keras.layers.Layer):
    def __init__(self, beta=0.1, **kw):
        super().__init__(**kw); self.beta=tf.Variable(beta, trainable=False) #change this into adaptive, or learnable
    def call(self, pcm, res_w):
        return pcm + self.beta*res_w
    def get_config(self):
        config = super().get_config()
        config.update({
            "beta": self.beta
        })
        return config

import tensorflow as tf
import tensorflow_io as tfio

class ChannelSim(tf.keras.layers.Layer):
    """
    Differentiable distortion block.
    ---------------------------------
    Acts only when `training=True`.

    Attacks implemented
    -------------------
    1. AWGN  (always on; SNR sampled from `snr_db`)
    2. Resample 16 kHz → mid­-rate → 16 kHz  (prob = `resample_prob`)
    3. MP3   round-trip @ 128 kbps          (prob = `mp3_prob`)
    4. Opus  round-trip @ 24 kbps           (prob = `opus_prob`)

    Parameters
    ----------
    fs : int
        Native sample-rate of the signal fed to the layer.
    snr_db : tuple(float,float)
        Min/max SNR for AWGN in dB (uniformly sampled).
    resample_rates : list[int]
        Candidate mid-rates for the resample attack.
    Optional prob args : float in [0,1]
        resample_prob, mp3_prob, opus_prob
    clip_dbfs : float or None
        If set, peak-normalises to this dBFS *after* all attacks.
    """

    def __init__(
        self,
        fs: int = 16_000,
        snr_db: tuple = (10.0, 30.0),
        resample_rates = (11025, 22050, 44100),
        resample_prob: float = 0.3,
        mp3_prob: float = 0.3,
        opus_prob: float = 0.0,
        # clip_dbfs: float | None = None,
        **kwargs,
    ):
        super().__init__(trainable=False, **kwargs)
        self.fs  = tf.cast(fs, tf.int64)
        self.snr = snr_db
        self.resample_rates = list(resample_rates)
        self.rp  = resample_prob
        self.mp3 = mp3_prob
        self.opu = opus_prob
        # self.clip_dbfs = clip_dbfs

    # ---------------------------------------------------------
    # helpers
    def _awgn(self, x):
        snr = tf.random.uniform([], *self.snr)  # dB
        pwr = tf.reduce_mean(tf.square(x))
        noise_pwr = pwr / (10.0 ** (snr / 10.0))
        return x + tf.random.normal(tf.shape(x), stddev=tf.sqrt(noise_pwr))

    def _resample_once(self, y, rate_mid):
        rate_mid = tf.cast(rate_mid, tf.int64)
        y = tfio.audio.resample(y, rate_in=self.fs, rate_out=rate_mid)
        return tfio.audio.resample(y, rate_in=rate_mid, rate_out=self.fs)

    def _mp3_rt(self, y):
        return tfio.audio.decode_mp3(tfio.audio.encode_mp3(y, rate=self.fs))

    def _opus_rt(self, y):
        try:
            return tfio.audio.decode_opus(tfio.audio.encode_opus(y, rate=self.fs))
        except AttributeError:
            # fallback: no-op if tensorflow-io built without Opus
            return y

    def _peak_norm(self, y):
        if self.clip_dbfs is None:
            return y
        peak = tf.reduce_max(tf.abs(y)) + 1e-12
        scale = (10.0 ** (-self.clip_dbfs / 20.0)) / peak
        return y * tf.minimum(1.0, scale)

    # ---------------------------------------------------------
    # main call
    def call(self, x, training=False):
        """
        x: (B, T, 1) float32 in [-1,1]
        """
        if not training:
            return x

        y = self._awgn(x)

        # Resample attack
        if tf.random.uniform([]) < self.rp:
            rate_mid = tf.random.shuffle(self.resample_rates)[0]
            y = tf.squeeze(y, -1)
            y = self._resample_once(y, rate_mid)
            y = tf.expand_dims(y, -1)
            y = tf.stop_gradient(y) 

        # MP3 attack
        if tf.random.uniform([]) < self.mp3:
            y = tf.squeeze(y, -1)
            y = self._mp3_rt(y)
            y = tf.expand_dims(y, -1)
            y = tf.stop_gradient(y) 

        # Opus attack
        if tf.random.uniform([]) < self.opu:
            y = tf.squeeze(y, -1)
            y = self._opus_rt(y)
            y = tf.expand_dims(y, -1)
            y = tf.stop_gradient(y) 

        # Optional peak clipping / normalisation
        # y = self._peak_norm(y)
        return y

    # ---------------------------------------------------------
    # convenience: change global strength at runtime
    def set_strength(self, preset: str):
        """
        Quickly scale all attack probabilities and SNR range.

        preset in {'low','default','high'}
        """
        if preset == 'low':
            self.snr = (15.0, 35.0)
            self.rp, self.mp3, self.opu = 0.1, 0.1, 0.0
        elif preset == 'high':
            self.snr = (0.0, 20.0)
            self.rp, self.mp3, self.opu = 0.6, 0.6, 0.3
        else:  # 'default'
            self.snr = (10.0, 30.0)
            self.rp, self.mp3, self.opu = 0.3, 0.3, 0.0

def build_extractor(bits_per_message=64, filters=32):
    inp = tf.keras.Input(shape=(None, 1))  # (B, T, 1)

    x = tf.keras.layers.Conv1D(filters, 7, 2, padding='same')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv1D(filters * 2, 5, 2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    # Collapse time dimension — entire audio sequence pooled
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Final output: 64-bit prediction for entire message
    out = tf.keras.layers.Dense(bits_per_message, activation='sigmoid')(x)

    return tf.keras.Model(inp, out, name='extractor')

# ---------------------------------------------------------------------------
# 5) ΔPESQ metric / loss helper
# ---------------------------------------------------------------------------
    
class DeltaPESQ(tf.keras.layers.Layer):
    def __init__(self, fs=16000, name="delta_pesq", **kw):
        super().__init__(name=name, trainable=False, **kw)
        self.fs = fs

    def call(self, inputs):
        clean, deg = inputs
        def _pesq_np(c, d):
            try:
                from pesq import pesq as pesq_score
                sc = pesq_score(self.fs, c, c, 'wb')
                sd = pesq_score(self.fs, c, d, 'wb')
                return np.float32(sc - sd)
            except Exception:
                warnings.warn('PESQ unavailable — ΔPESQ=0')
                return np.float32(0)
        delta = tf.numpy_function(_pesq_np,
                                  [tf.squeeze(clean), tf.squeeze(deg)],
                                  tf.float32)
        self.add_loss(delta)
        self.add_metric(delta, name=self.name)
        return deg

    def get_config(self):
        config = super().get_config()
        config.update({
            "fs": self.fs,
            "name": self.name
        })
        return config


# ---------------------------------------------------------------------------
# 6) build_model()
# ---------------------------------------------------------------------------
# def build_model(frame_size=160, seq_frames=15, bits_per_frame=64):
#     bits_in=tf.keras.Input(shape=(seq_frames,bits_per_frame), name='bits')
#     pcm_in =tf.keras.Input(shape=(frame_size*seq_frames,1), name='pcm')

#     res_w = WatermarkEmbedding(frame_size)([bits_in, pcm_in])
#     pcm_w = WatermarkAddition()(pcm_in, res_w)
#     attacked = ChannelSim()(pcm_w)
#     bits_out = build_extractor(bits_per_frame, seq_frames)(attacked)

#     model=tf.keras.Model([bits_in, pcm_in], bits_out, name='dl_lp_dss')
#     ber_metric=tf.keras.metrics.BinaryAccuracy(name='acc') # update loss
#     model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
#                   loss=tf.keras.losses.BinaryCrossentropy(),
#                   metrics=[ber_metric])
#     return model

def build_model(frame_size=160, seq_frames=15, bits_per_frame=64,
                resample_p=0.2, resample_r=(11025, 22050, 44100), mp3_p=0.2, opus_p=0.0, snr=(10.,30.), 
                # clip_d=0.0
                ): #attack purpose
    bits_in = tf.keras.Input(shape=bits_per_frame, name="bits")
    # bits_in = tf.keras.Input(shape=(seq_frames, bits_per_frame), name="bits") #(15,64)
    pcm_in  = tf.keras.Input(shape=(frame_size*seq_frames, 1), name="pcm_clean")

    print("bits_in:", bits_in.shape)
    print("pcm_in:", pcm_in.shape)
    '''
    bits_in: (None, 15, 64) -> (None, 64)
    pcm_in: (None, 2400, 1)
    '''

    # Watermark path
    res_w   = WatermarkEmbedding(frame_size, name="wm_embed")([bits_in, pcm_in])
    pcm_w   = WatermarkAddition(beta=0.10, name="wm_add")(pcm_in, res_w)

    # ΔPESQ branch (only adds loss/metric, returns pcm_w unchanged)
    pcm_w   = DeltaPESQ(name="delta_pesq")( [pcm_in, pcm_w] )

    # Channel attacks
    '''
    attacked = ChannelSim(resample_prob=resample_p, 
                          resample_rates=resample_r,
                          mp3_prob=mp3_p, 
                          opus_prob=opus_p, 
                          snr_db=snr,
                        #   clip_dbfs=clip_d, 
                          name="chan")(pcm_w)
    '''

    # Extract bits
    bits_out = build_extractor(bits_per_frame, seq_frames)(pcm_w)

    model=tf.keras.Model([bits_in, pcm_in], bits_out, name='dl_lp_dss')

    return model

# ---------------------------------------------------------------------------
# 7  Validation BER under attack
# ---------------------------------------------------------------------------

# later
class ValAttackBER(tf.keras.callbacks.Callback):
    def __init__(self, val_seq, name="val_acc_attack"):
        super().__init__(); self.val_seq=val_seq; self.name=name
        self.metric=tf.keras.metrics.BinaryAccuracy()
    def on_epoch_end(self, epoch, logs=None):
        self.metric.reset_states()
        for (bits, pcm), lbl in self.val_seq:
            y_pred = self.model([bits, pcm], training=True)  # attacks ON
            self.metric.update_state(lbl, y_pred)
        logs[self.name] = float(self.metric.result().numpy())
        print(f"\n{self.name}: {logs[self.name]:.4f}")

# ---------------------------------------------------------------------------
# 8  Training helpers
# ---------------------------------------------------------------------------

def compile_phase_A(model):
    # freeze embedder & adder
    model.get_layer("wm_embed").trainable = False
    model.get_layer("wm_add").trainable  = False
    # model.get_layer("chan").rp = 0.2
    # model.get_layer("chan").mp = 0.2 
    # model.get_layer("chan").snr = (10.,30.)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")])


def compile_phase_B(model, lambda_=0.75):
    model.get_layer("wm_embed").alpha.trainable = True       # unfreeze α
    # stronger attacks
    # model.get_layer("chan").set_strength('high')

    # model.get_layer("chan").rp = 0.5
    # model.get_layer("chan").mp = 0.5
    # model.get_layer("chan").snr = (5.,25.)

    bce = tf.keras.losses.BinaryCrossentropy()
    def composite(bits_true, bits_pred):
        acc = bce(bits_true, bits_pred)
        pesq_pen = tf.reduce_mean(model.get_layer("delta_pesq").losses)  # scalar ΔPESQ
        return lambda_ * acc + (1 - lambda_) * pesq_pen

    model.compile(optimizer=tf.keras.optimizers.Adam(2e-5),
                  loss=composite,
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),
                           tf.keras.metrics.Mean(name="delta_pesq", dtype=tf.float32)])

# ---------------------------------------------------------------------------
# 9  Fit wrappers
# ---------------------------------------------------------------------------

def fit_model(model, seq_train, seq_val, ckpt_path, epochs):
    """Train `model` and save the best weights to `ckpt_path`."""
    cbs = [
        # 1) save only the best model (highest val_ber)
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor='val_acc', mode='max', save_best_only=True
        ),

        # 2) stop early if val_ber hasn't improved for 5 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor='val_acc', mode='max', patience=5, restore_best_weights=True
        ),

        # 3) extra metric: BER on validation set *with* ChannelSim attacks
        # ValAttackBER(seq_val),
    ]

    model.fit(
        seq_train,
        validation_data=seq_val,
        epochs=epochs,
        callbacks=cbs,
        workers=4,
        use_multiprocessing=True,
    )
    
# ---------------------------------------------------------------------------
# 10 Training stub
# ---------------------------------------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train_dir',required=True)
    ap.add_argument('--val_dir',required=True)
    ap.add_argument('--phase',choices=['A','B','AB'],default='A')
    ap.add_argument('--ckpt',help='Phase‑A checkpoint for Phase B')
    ap.add_argument('--epochsA',type=int,default=15)
    ap.add_argument('--epochsB',type=int,default=20)
    ap.add_argument('--batch', type=int, default=32)
    args=ap.parse_args()

    os.makedirs('checkpoints',exist_ok=True)

    seq_train=FrameSequence(args.train_dir, batch_size=args.batch)
    seq_val  =FrameSequence(args.val_dir,   batch_size=args.batch, shuffle=False)

    # custom objs for model reload
    cust={'WatermarkEmbedding': WatermarkEmbedding,
          'WatermarkAddition': WatermarkAddition,
        #   'ChannelSim': ChannelSim,
          'DeltaPESQ': DeltaPESQ}
    
    if args.phase == 'A':
        model = build_model()
        compile_phase_A(model)
        fit_model(model,
                  seq_train, seq_val,
                  'checkpoints/phaseA_best.h5',
                  args.epochsA)
        
    elif args.phase == 'B':
        if not args.ckpt: 
            raise SystemExit('Phase B needs --ckpt from Phase A')
        model=tf.keras.models.load_model(args.ckpt, 
                                         custom_objects=cust, 
                                         compile=False)
        compile_phase_B(model)
        fit_model(model,
                  seq_train, seq_val,
                  'checkpoints/phaseB_best.h5',
                  args.epochsB)
        
    else: #AB chain
        model = build_model()
        compile_phase_A(model)
        fit_model(model,
                  seq_train, seq_val,
                  'checkpoints/phaseA_best.h5',
                  args.epochsA)
        model=tf.keras.models.load_model('checkpoints/phaseA_best.h5', 
                                         custom_objects=cust, 
                                         compile=False)
        compile_phase_B(model)
        fit_model(model,
                  seq_train, seq_val,
                  'checkpoints/phaseB_best.h5',
                  args.epochsB)

    print('Training complete. Best checkpoints saved in ./checkpoints/')


if __name__=='__main__':
    main()
