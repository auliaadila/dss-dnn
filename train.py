#!/usr/bin/env python
"""train.py – End‑to‑end training script for DSS speech watermarking with a DNN extractor.

Assumptions
===========
* The following helper modules exist in the same repo:
  ▸ dss/layers.py → `WatermarkEmbedding`, `WatermarkAddition`
  ▸ net/extractor.py → `build_extractor(bits_per_frame)` returning a Keras model
  ▸ dataset.py → `FrameSequence` that yields `(bits, pcm)` pairs and uses bits as labels
* All audio is 16 kHz mono. Each training sample is a *frame* of 160 samples (10 ms).
* Training is run inside the Conda env specified in dss-wm.yml (Python 3.8, TF 2.10 GPU).

Run:
    python train.py --train_dir data/train --val_dir data/val \
                    --chkpt_dir checkpoints

"""
import os
import argparse
import tensorflow as tf
from datetime import datetime


# ---------------------------------------------------------------------------
#  Model assembly
# ---------------------------------------------------------------------------

def build_wm_model(frame_size=160, bits_per_frame=64, alpha_init=0.04, beta=0.10):
    """Constructs the DSS embedder + channel + extractor graph."""

    # Lazy imports – avoid circulars if modules import TF too
    from dss.layers import WatermarkEmbedding, WatermarkAddition
    # from dss.lpanalysis import LPAnalysis
    from net.extractor import build_extractor
    from net.distortion import ChannelSim

    bits_in = tf.keras.Input(shape=(None, 1), dtype=tf.float32, name="bits_in")
    pcm_in  = tf.keras.Input(shape=(None, 1), dtype=tf.float32, name="pcm_in")

    # 1 ‑ Embed watermark in residual
    residual_w = WatermarkEmbedding(frame_size=frame_size,
                                    bits_per_frame=bits_per_frame,
                                    alpha_init=alpha_init,
                                    trainable_alpha=False,
                                    compute_residual=False,
                                    name="wm_embed")([bits_in, pcm_in])

    # 2 ‑ Add residual back to waveform
    pcm_w = WatermarkAddition(beta=beta, learnable_mask=False,
                              name="wm_add")([pcm_in, residual_w])

    # 3 ‑ Simulated attack channel
    # attacked = ChannelSim(name="channel_sim")(pcm_w)

    # 4 ‑ Extract bits
    bits_out = build_extractor(bits_per_frame=bits_per_frame)(pcm_w)

    model = tf.keras.Model(inputs=[bits_in, pcm_in], outputs=bits_out, name="dss_wm")

    # Loss & metrics - TO DO: refine
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    acc     = tf.keras.metrics.BinaryAccuracy(name="acc")  # frame‑level ACC proxy

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                  loss=loss_fn,
                  metrics=[acc])
    return model

# ---------------------------------------------------------------------------
#  Training utility
# ---------------------------------------------------------------------------

def get_sequences(train_dir, val_dir, frame_size, bits_per_frame, batch_size):
    """Instantiate FrameSequence generators for train / val."""
    from dataset import FrameSequence  # local import to avoid heavy deps at module load

    seq_train = FrameSequence(wav_folder=train_dir,
                              frame_size=frame_size,
                              bpf=bits_per_frame,
                              batch_size=batch_size,
                              shuffle=True)

    seq_val = FrameSequence(wav_folder=val_dir,
                            frame_size=frame_size,
                            bpf=bits_per_frame,
                            batch_size=batch_size,
                            shuffle=False)
    return seq_train, seq_val


# ---------------------------------------------------------------------------
#  Main entry‑point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DSS watermarking model")
    parser.add_argument("--train_dir", required=True, help="Folder with training WAVs")
    parser.add_argument("--val_dir",   required=True, help="Folder with validation WAVs")
    parser.add_argument("--chkpt_dir", default="checkpoints", help="Where to save models")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch",  type=int, default=32)
    parser.add_argument("--frame_size", type=int, default=160)
    parser.add_argument("--bits_per_frame", type=int, default=64)
    parser.add_argument("--alpha_init", type=float, default=0.04)
    parser.add_argument("--beta", type=float, default=0.10)
    args = parser.parse_args()

    # TO DO: add args phase

    os.makedirs(args.chkpt_dir, exist_ok=True)

    # 1 ‑ Data
    seq_train, seq_val = get_sequences(args.train_dir, args.val_dir,
                                       args.frame_size, args.bits_per_frame,
                                       args.batch)

    # 2 ‑ Model
    model = build_wm_model(args.frame_size, args.bits_per_frame,
                           args.alpha_init, args.beta)

    model.summary(line_length=120)

    # 3 ‑ Callbacks
    ckpt_path = os.path.join(args.chkpt_dir,
                             "dsswm_{epoch:02d}_{val_acc:.4f}.h5")
    cb_chkpt  = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                   monitor="val_acc",
                                                   mode="max",
                                                   save_best_only=True,
                                                   save_weights_only=False)
    cb_es     = tf.keras.callbacks.EarlyStopping(monitor="val_acc", mode="max",
                                                patience=5, restore_best_weights=True)
    log_path  = os.path.join(args.chkpt_dir, "training_log.csv")
    cb_csv    = tf.keras.callbacks.CSVLogger(log_path, append=True)

    # 4 ‑ Fit
    model.fit(seq_train,
              validation_data=seq_val,
              epochs=args.epochs,
              callbacks=[cb_chkpt, cb_es, cb_csv],
              workers=4, use_multiprocessing=True)

    print("Training complete. Best model saved to:", args.chkpt_dir)


if __name__ == "__main__":
    main()
