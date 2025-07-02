#!/usr/bin/env python3
"""train_payload_phased.py  –  Two-phase training for residual-carrier LP-DSS
###########################################################################
* Embedding  : ResidualEmbedding (frame-by-frame LPC, payload repetition)
* Attack     : AttackPipeline (FIR + Butterworth + noise + cut)
* Extractor  : DSSExtractor (small 1-D CNN)
* Phase A    : freeze α, moderate attacks, BCE loss
* Phase B    : unfreeze α, stronger attacks, λ·BER + (1-λ)·ΔPESQ loss

Usage
-----
Phase A only:
    python train_payload_phased.py --phase A --train data/train --val data/val
Phase B continue:
    python train_payload_phased.py --phase B --ckpt checkpoints/pA.h5 \
           --train data/train --val data/val
Both phases back-to-back:
    python train_payload_phased.py --phase AB --train data/train --val data/val
"""
from pathlib import Path
import argparse, os
import tensorflow as tf

from dataloader import FrameSequencePayload
from residualembedding import ResidualEmbedding
from extractor import DSSExtractor
from attacks import AttackPipeline, LowpassFIR, AdditiveNoise, CuttingSamples, ButterworthIIR
from attacks import fir_lowpass
from loss import DeltaPESQ, PESQLogger, DeltaPESQMetric

# ------------------------ build end-to-end graph --------------------------- #

def build_model(frame=160, K=15, P=64, lpc_order=16, trainable_alpha=False):  ### CHANGED
    """Return uncompiled Keras model (bits, pcm) → (bits_pred)."""
    T = frame * K
    bits_in = tf.keras.Input((P,), name='payload_bits')
    pcm_in  = tf.keras.Input((T,1),  name='pcm_window')

    wm_pcm = ResidualEmbedding(
        frame_size=frame,
        frames_per_payload=K,
        lpc_order=lpc_order,
        trainable_alpha=trainable_alpha,  ### CHANGED
        name='embed'
    )([bits_in, pcm_in])

    pcm_qual = DeltaPESQ(name='pesq_layer')([pcm_in, wm_pcm])
    attacks = AttackPipeline(name='attacks')
    pcm_ch  = attacks(pcm_qual)

    extractor = DSSExtractor(payload_bits=P, context_len=T, name='extractor')
    bits_out  = extractor(pcm_ch)

    return tf.keras.Model([bits_in, pcm_in], bits_out, name='LPDSS_Payload')

# ---------------------- compile helpers ----------------------------------- #

def compile_phase_A(model):
    embed = model.get_layer('embed')
    embed.alpha.assign(0.05)  # Only assign value
    # Do NOT set embed.alpha.trainable = False

    chan = model.get_layer('attacks')
    for lay in chan.attacks:
        if hasattr(lay, 'prob'):
            lay.prob = 20.

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='ber'),
            # tf.keras.metrics.Mean(name='delta_pesq') #log pesq
            DeltaPESQMetric(fs=16000)
        ]
    )

# ------------------------------------------------- Phase B
def compile_phase_B(model, lam=0.75, lam_pesq=0.3):
    # DO NOT modify embed.alpha.trainable
    chan  = model.get_layer('attacks')
    for lay in chan.attacks:
        if hasattr(lay, 'prob'):
            lay.prob = 40.
        if isinstance(lay, AdditiveNoise):
            lay.strength *= 2

    bce = tf.keras.losses.BinaryCrossentropy()
    pesq_layer = model.get_layer('pesq_layer')

    def composite(y_true, y_pred):
        ber  = bce(y_true, y_pred)
        d_pq = tf.add_n(pesq_layer.losses) # scalar ΔPESQ (mean over batch)
        return lam * ber + (1 - lam) * d_pq * lam_pesq

    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-5),
        loss=composite,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='ber'),
            # tf.keras.metrics.Mean(name='delta_pesq')
            DeltaPESQMetric(fs=16000)
        ]
    )

# ---------------------- training util ------------------------------------- #
import datetime  # add this import at the top if not already

def fit_model(model, train_ds, val_ds, ckpt, epochs, log_prefix=""):
    # log_dir = f"logs/{log_prefix}" if log_prefix else "logs/train"
    log_dir = os.path.join("logs", log_prefix + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt, monitor='val_ber', mode='max', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_ber', mode='max', patience=5, restore_best_weights=False),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        PESQLogger(log_dir=log_dir, log_prefix=log_prefix)
    ]

    model.fit(train_ds, validation_data=val_ds,
              epochs=epochs, callbacks=cbs)

# def fit_model(model, train_ds, val_ds, ckpt, epochs, log_prefix="run"):
#     log_dir = os.path.join("logs", log_prefix + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#     tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#     cbs = [
#         tensorboard_cb,
#         tf.keras.callbacks.ModelCheckpoint(
#             ckpt, monitor='val_ber', mode='max', save_best_only=True),
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_ber', mode='max', patience=5, restore_best_weights=False),
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_delta_pesq', mode='min', patience=3,
#             restore_best_weights=True)
#     ]
#     model.fit(train_ds, validation_data=val_ds,
#               epochs=epochs, callbacks=cbs)

# ---------------------- CLI ------------------------------------------------ #
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True, type=Path)
    ap.add_argument('--val',   required=True, type=Path)
    ap.add_argument('--phase', choices=['A','B','AB'], default='A')
    ap.add_argument('--ckpt',  type=Path, help='phase-A checkpoint for phase-B')
    ap.add_argument('--epochsA', type=int, default=15)
    ap.add_argument('--epochsB', type=int, default=20)
    ap.add_argument('--batch',   type=int, default=32)
    ap.add_argument('--frames',  type=int, default=15, help='frames per payload (K)')
    ap.add_argument('--bits',    type=int, default=64, help='payload length (P)')
    ap.add_argument('--name', type=str)
    args = ap.parse_args()

    # ---------------- data ----------------
    train_seq = FrameSequencePayload(args.train, frames_per_payload=args.frames,
                                     payload_bits=args.bits, batch=args.batch)
    val_seq   = FrameSequencePayload(args.val, frames_per_payload=args.frames,
                                     payload_bits=args.bits, batch=args.batch, shuffle=False)

    os.makedirs('checkpoints', exist_ok=True)
    custom = {'ResidualEmbedding': ResidualEmbedding, 'AttackPipeline': AttackPipeline,
              'DSSExtractor': DSSExtractor, 'LowpassFIR': LowpassFIR,
              'AdditiveNoise': AdditiveNoise, 'CuttingSamples': CuttingSamples,
              'ButterworthIIR': ButterworthIIR, 'fir_lowpass': fir_lowpass}

    if args.phase == 'A':
        model = build_model(K=args.frames, P=args.bits, trainable_alpha=False)  ### CHANGED
        compile_phase_A(model)
        fit_model(model, train_seq, val_seq, f"checkpoints/pA_{args.name}.h5", args.epochsA, log_prefix=f"A_{args.name}")

    elif args.phase == 'B':
        if not args.ckpt:
            raise SystemExit('Require --ckpt from Phase A')
        model = tf.keras.models.load_model(args.ckpt, custom_objects=custom, compile=False)
        # Rebuild model with alpha trainable, then load weights
        model_b = build_model(K=args.frames, P=args.bits, trainable_alpha=True)  ### CHANGED
        model_b.set_weights(model.get_weights())  ### Transfer weights
        compile_phase_B(model_b)
        fit_model(model_b, train_seq, val_seq, f"checkpoints/pB_{args.name}.h5", args.epochsB, log_prefix=f"B_{args.name}")

    else:  # AB chain
        model_a = build_model(K=args.frames, P=args.bits, trainable_alpha=False)  ### CHANGED
        compile_phase_A(model_a)
        fit_model(model_a, train_seq, val_seq, f"checkpoints/pA_{args.name}.h5", args.epochsA, log_prefix=f"A_{args.name}")

        model_a = tf.keras.models.load_model(f"checkpoints/pA_{args.name}.h5", custom_objects=custom, compile=False)
        model_b = build_model(K=args.frames, P=args.bits, trainable_alpha=True)  ### CHANGED
        model_b.set_weights(model_a.get_weights())
        compile_phase_B(model_b)
        fit_model(model_b, train_seq, val_seq, f"checkpoints/pB_{args.name}.h5", args.epochsB, log_prefix=f"B_{args.name}")

    print("✅ Training finished. Checkpoints in ./checkpoints/")