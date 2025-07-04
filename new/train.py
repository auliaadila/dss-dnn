#!/usr/bin/env python3
"""train_payload_staged.py  –  Two-stage training for residual-carrier LP-DSS
###########################################################################
* Embedding  : ResidualEmbedding (frame-by-frame LPC, payload repetition)
* Attack     : AttackPipeline (FIR + Butterworth + noise + cut)
* Extractor  : DSSExtractor (small 1-D CNN)
* stage A    : freeze α, moderate attacks, BCE loss
* stage B    : unfreeze α, stronger attacks, λ·BER + (1-λ)·ΔPESQ loss

Usage
-----
stage A only:
    python train_payload_staged.py --stage A --train data/train --val data/val
stage B continue:
    python train_payload_staged.py --stage B --ckpt checkpoints/pA.h5 \
           --train data/train --val data/val
Both stages back-to-back:
    python train_payload_staged.py --stage AB --train data/train --val data/val
"""

import argparse
import datetime
import os
from pathlib import Path

import tensorflow as tf

from attacks import (
    AdditiveNoise,
    AttackPipeline,
    ButterworthIIR,
    CuttingSamples,
    LowpassFIR,
    fir_lowpass,
)
from dataloader import FrameSequencePayload
from extractor import DSSExtractor

# from new.pesqloss import DeltaPESQ, PESQLogger, DeltaPESQMetric
from lossfunc import *
from metrics import *
from residualembedding import ResidualEmbedding

# ------------------------ build end-to-end graph --------------------------- #


def build_model(
    frame=160, K=15, P=64, lpc_order=16, trainable_alpha=False
):  ### CHANGED
    """Return uncompiled Keras model (bits, pcm) → (bits_pred)."""
    T = frame * K
    bits_in = tf.keras.Input((P,), name="payload_bits")
    pcm_in = tf.keras.Input((T, 1), name="pcm_window")

    ## Components: embedder, attacks, extractor
    embedder = ResidualEmbedding(
        frame_size=frame,
        frames_per_payload=K,
        lpc_order=lpc_order,
        trainable_alpha=trainable_alpha,  ### CHANGED
        name="embedder",
    )

    # pcm_qual = DeltaPESQ(name='pesq_layer')([pcm_in, wm_pcm]) #delete
    attacks = AttackPipeline(name="attacks")
    extractor = DSSExtractor(payload_bits=P, context_len=T, name="extractor")

    ## Data flow: wm_pcm, pcm_ch, bits_out
    wm_pcm = embedder([bits_in, pcm_in])
    pcm_ch = attacks(wm_pcm)
    bits_out = extractor(pcm_ch)

    return tf.keras.Model(
        inputs=[bits_in, pcm_in],
        outputs={"bits_pred": bits_out, "wm_pcm": wm_pcm},
        name="LPDSS_Payload",
    )


# ---------------------- compile helpers ----------------------------------- #


def compile_stage_A(model, lr=1e-4):
    # Focus: BER
    # [from build_model] return tf.keras.Model([bits_in, pcm_in], bits_out, name='LPDSS_Payload')
    embed = model.get_layer("embedder")
    embed.alpha.assign(0.05)  # Only assign value
    # Do NOT set embed.alpha.trainable = False

    chan = model.get_layer("attacks")
    for lay in chan.attacks:
        # lay.trainable = False
        if hasattr(lay, "prob"):
            lay.prob = 20.0  # set low strength

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={"bits_pred": bit_consistency_loss, "wm_pcm": None},
        metrics={"bits_pred": [BitErrorRate(name="ber")], "wm_pcm": None},
    )


def compile_stage_B(
    model,
    gamma_bits=0.7,  # weight on robustness vs audio
    λ_l1=0.2,
    λ_snr=0.2,
    λ_mask=0.2,
    lr=2e-5,
):
    """
    Make embedder’s α trainable, harden channel, and compile with
    composite loss:  γ·BER  +  (1−γ)·(λ1·L1 + λ2·SNR + λ3·Mask)
    """

    # 1) unfreeze alpha
    embed = model.get_layer("embedder")
    embed.trainable = True

    # 2) harden attacks
    chan = model.get_layer("attacks")
    for lay in chan.attacks:
        # lay.trainable = False
        if hasattr(lay, "prob"):
            lay.prob = 40.0
        if isinstance(lay, AdditiveNoise):
            lay.strength *= 2.0

    # ------------------------------------------------------------------
    # robustness loss  (bits head)
    def bits_loss(y_true, y_pred):
        return bit_consistency_loss(y_true, y_pred) * gamma_bits

    # audio–quality loss  (pcm head)
    def pcm_loss(y_true, y_pred):
        l1 = λ_l1 * l1_loss(y_true, y_pred)
        snr = λ_snr * snr_loss(y_true, y_pred)
        fmsk = λ_mask * frequency_masking_loss(y_true, y_pred)
        return (1.0 - gamma_bits) * (l1 + snr + fmsk)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={"bits_pred": bits_loss, 
              "wm_pcm": pcm_loss},
        metrics={
            "bits_pred": [BitErrorRate(name="ber")],
            "wm_pcm": pcm_loss,
        },  #  audio head purely for loss
    )


def callbacks_stage_A(ckpt_path, patience=5, log_prefix="stageA"):
    """Create standard callbacks for BER-centred training."""
    # TensorBoard run folder
    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(
        "logs", f"{log_prefix}_{run_name}" if log_prefix else run_name
    )
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_extractor_ber",  # define later
            mode="min",  # << because lower BER is better
            save_best_only=True,  # error
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_extractor_ber",
            mode="min",  # << same reasoning
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    ]

    return callbacks


def callbacks_stage_B(ckpt_path, patience_bit=5, patience_audio=3, log_prefix="stageB"):
    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(
        "logs", f"{log_prefix}_{run_name}" if log_prefix else run_name
    )
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        # Save best-BER model
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_extractor_ber",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        # Stop if BER plateaus
        tf.keras.callbacks.EarlyStopping(
            monitor="val_extractor_ber",
            mode="min",
            patience=patience_bit,
            restore_best_weights=False,
            verbose=1,
        ),
        # Extra guard: stop if audio quality degrades for 3 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor="val_embedder_pcm_loss",  # your combined audio loss
            mode="min",
            patience=patience_audio,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    ]

    return callbacks


# ---------------------- training util ------------------------------------- #
# import datetime  # add this import at the top if not already


def fit_model(
    model,
    train_ds,
    val_ds,
    epochs: int,
    ckpt_path: str,
    stage: str,
    log_prefix: str = "",
):
    """
    Wrapper around model.fit that chooses the proper callback set
    based on training stage.

    Parameters
    ----------
    model        : tf.keras.Model   – already compiled network
    train_ds     : tf.data.Dataset / Keras Sequence
    val_ds       : tf.data.Dataset / Keras Sequence
    epochs       : int              – max epochs to run
    ckpt_path    : str | Path       – where ModelCheckpoint stores .h5
    stage        : {"A", "B"}       – training stage selector
    log_prefix   : str              – sub-folder name inside ./logs
    """
    if stage.upper() == "A":
        cbs = callbacks_stage_A(ckpt_path, log_prefix=log_prefix)
    elif stage.upper() == "B":
        cbs = callbacks_stage_B(ckpt_path, log_prefix=log_prefix)
    else:
        raise ValueError("stage must be 'A' or 'B' (got %s)" % stage)

    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs)


# ---------------------- CLI ------------------------------------------------ #
if __name__ == "__main__":
    import argparse
    import datetime
    import os
    from pathlib import Path

    # ---------------- CLI --------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--val", required=True, type=Path)
    ap.add_argument(
        "--stage",
        choices=["A", "B", "AB"],
        default="A",
        help="training stage: A, B, or AB (both)",
    )
    ap.add_argument(
        "--ckpt", type=Path, help="stage-A checkpoint to start stage-B from"
    )
    ap.add_argument("--epochsA", type=int, default=15)
    ap.add_argument("--epochsB", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument(
        "--frames", type=int, default=15, help="frames per payload window (K)"
    )
    ap.add_argument(
        "--bits", type=int, default=64, help="payload length (P) per window"
    )
    ap.add_argument(
        "--name", type=str, default="run", help="experiment tag used in filenames/logs"
    )
    args = ap.parse_args()

    print("==== NAME:", args.name)

    # ---------------- data -------------------------------------------------
    train_seq = FrameSequencePayload(
        args.train,
        frames_per_payload=args.frames,
        payload_bits=args.bits,
        batch=args.batch,
    )
    val_seq = FrameSequencePayload(
        args.val,
        frames_per_payload=args.frames,
        payload_bits=args.bits,
        batch=args.batch,
        shuffle=False,
    )

    os.makedirs("checkpoints", exist_ok=True)

    custom = {  # for model reload
        "ResidualEmbedding": ResidualEmbedding,
        "AttackPipeline": AttackPipeline,
        "DSSExtractor": DSSExtractor,
        "AdditiveNoise": AdditiveNoise,
        "CuttingSamples": CuttingSamples,
        "ButterworthIIR": ButterworthIIR,
        "fir_lowpass": fir_lowpass,
    }

    # ---------------- stage logic -----------------------------------------
    if args.stage == "A":
        model_a = build_model(K=args.frames, P=args.bits, trainable_alpha=False)
        compile_stage_A(model_a)
        fit_model(
            model_a,
            train_seq,
            val_seq,
            ckpt_path=f"checkpoints/sA_{args.name}.h5",
            epochs=args.epochsA,
            stage="A",
            log_prefix=f"A_{args.name}",
        )

    elif args.stage == "B":
        if not args.ckpt:
            raise SystemExit("--ckpt must point to a stage-A checkpoint")

        # load A-weights, rebuild with alpha trainable
        base = tf.keras.models.load_model(
            args.ckpt, custom_objects=custom, compile=False
        )
        model_b = build_model(K=args.frames, P=args.bits, trainable_alpha=True)
        model_b.set_weights(base.get_weights())
        compile_stage_B(model_b)
        fit_model(
            model_b,
            train_seq,
            val_seq,
            ckpt_path=f"checkpoints/sB_{args.name}.h5",
            epochs=args.epochsB,
            stage="B",
            log_prefix=f"B_{args.name}",
        )

    else:  # "AB"  – run both stages sequentially
        # ----- stage A -----
        model_a = build_model(K=args.frames, P=args.bits, trainable_alpha=False)
        compile_stage_A(model_a)
        fit_model(
            model_a,
            train_seq,
            val_seq,
            ckpt_path=f"checkpoints/sA_{args.name}.h5",
            epochs=args.epochsA,
            stage="A",
            log_prefix=f"A_{args.name}",
        )

        # ----- stage B -----
        base = tf.keras.models.load_model(
            f"checkpoints/sA_{args.name}.h5", custom_objects=custom, compile=False
        )
        model_b = build_model(K=args.frames, P=args.bits, trainable_alpha=True)
        model_b.set_weights(base.get_weights())
        compile_stage_B(model_b)
        fit_model(
            model_b,
            train_seq,
            val_seq,
            ckpt_path=f"checkpoints/sB_{args.name}.h5",
            epochs=args.epochsB,
            stage="B",
            log_prefix=f"B_{args.name}",
        )

    print("✅ Training finished.  Checkpoints are in ./checkpoints/")
