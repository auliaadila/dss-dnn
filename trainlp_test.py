#!/usr/bin/env python3
"""
GPU-optimized training script for LP-DSS watermarking.
This version removes CPU bottlenecks and maximizes GPU utilization.
"""

import argparse
import datetime
import os
from pathlib import Path

import tensorflow as tf

# Use GPU-optimized components
from components.attack_pipeline_fixed import AttackPipelineFixed as AttackPipeline
from components.attacks_gpu import (
    AdditiveNoise,
    ButterworthIIR,
    CuttingSamples,
    LowpassFIR,
)
from components.dataloader_cached import FrameSequencePayload
from components.extractorlp_ import LPDSBlockExtractor as LPDSSExtractor
from components.lossfunc_fixed import bit_consistency_loss, frequency_masking_loss, l1_loss, snr_loss
from components.metrics import BitErrorRate
from components.embeddinglp_ import LPDSSEmbedder as Embedder

def build_model_gpu(frame=160, K=15, P=64,
                    lpc_order=12,
                    alpha_settings="constant",
                    resid_sign=False,
                    trainable_alpha=False):
    """Return GPU-optimized uncompiled Keras model (bits, pcm) → (bits_pred)."""

    # Input layers
    bits_input = tf.keras.Input(shape=(P,), name="bits_input")
    pcm_input = tf.keras.Input(shape=(K * frame, 1), name="pcm_input")
    
    embedding = Embedder(frame_size = frame, K = K, P = P,
                        lporder = lpc_order,
                        ssl_db  = -10,
                        alpha_settings  = alpha_settings,
                        trainable_alpha = trainable_alpha,
                        use_full_lpc    = False,
                        resid_whiten    = True,
                        resid_sign=resid_sign)

    # Apply embedding
    watermarked = embedding([bits_input, pcm_input])

    # GPU-optimized attack pipeline with fixed training flag propagation
    attacks = AttackPipeline(
        fir_cutoff=0.8, iir_cutoff=0.8, snr_db=20.0, max_cut_ratio=0.1
    )

    # Apply attacks (training flag will be automatically forwarded via learning_phase)
    attacked = attacks(watermarked)

    extractor = LPDSSExtractor(payload_bits=P, 
                               context_len=K*frame, 
                               frame_size=frame, 
                               lporder=12,
                               use_full_lpc=False, 
                               base_filters=64)
    
    bits_pred = extractor(attacked)

    # Create model
    model = tf.keras.Model(
        inputs=[bits_input, pcm_input], 
        outputs={"bits_pred": bits_pred, "wm_pcm": watermarked}, 
        name="LPDSS_GPU"
    )

    return model

def configure_gpu():
    """Configure GPU for optimal performance."""
    print("Configuring GPU for optimal performance...")

    # Get GPU devices
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Enable memory growth to avoid OOM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            print(f"Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")

            print("Mixed precision disabled - using float32")

        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPUs found - using CPU")

def create_optimized_dataset(dataloader):
    """Return dataloader (already optimized with caching)."""
    return dataloader

def train_stage_gpu(
    model,
    train_ds,
    val_ds,
    epochs,
    stage,
    ckpt_path,
    log_prefix="",
    batch_size=32,
    extra_callbacks=None,
):
    """GPU-optimized training stage."""

    # Configure optimizer with gradient clipping
    lr = 1e-4 if stage == "B" else 5e-4
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        clipnorm=1.0,
    )

    # Stage A setup
    if stage == "A":
        lambda_bits = 1.0

        def bits_loss(y_true, y_pred):
            return bit_consistency_loss(y_true, y_pred) * lambda_bits

        loss_fn = {
            "bits_pred": bits_loss,
            "wm_pcm": None
        }
        metrics = {
            "bits_pred": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                BitErrorRate(name="ber"),
            ],
            "wm_pcm": None
        }

    # Stage B setup
    elif stage == "B":
        lambda_bits_var = tf.Variable(0.7, dtype=tf.float32, trainable=False)

        lambda_pcm = dict(λ_l1=0.3, λ_snr=0.1, λ_mask=0.2)

        def bits_loss(y_true, y_pred):
            return bit_consistency_loss(y_true, y_pred) * lambda_bits_var

        def pcm_loss(y_true, y_pred):
            l1   = lambda_pcm["λ_l1"] * l1_loss(y_true, y_pred)
            snr  = lambda_pcm["λ_snr"] * snr_loss(y_true, y_pred)
            fmsk = lambda_pcm["λ_mask"]* frequency_masking_loss(y_true, y_pred)
            return (1.0 - lambda_bits_var) * (l1 + snr + fmsk)
        
        # linear decay 0.7 → 0.5 over the whole Stage-B run
        def schedule_lambda_bits(epoch, epochs_total):
            t = epoch / max(1, epochs_total-1)
            return 0.7 - 0.2 * t           # 0.7 → 0.5

        class LambdaBitsScheduler(tf.keras.callbacks.Callback):
            def __init__(self, var, epochs_total):
                super().__init__(); self.var = var; self.T = epochs_total
            def on_epoch_begin(self, epoch, logs=None):
                new_val = schedule_lambda_bits(epoch, self.T)
                self.var.assign(new_val)
                print(f"\nλ_bits set to {new_val:.3f}")

        lambda_cb = LambdaBitsScheduler(lambda_bits_var, epochs)
        
        class PCMLossMetric(tf.keras.metrics.Metric):
            def __init__(self, name='pcm_loss', **kwargs):
                super().__init__(name=name, **kwargs)
                self.total = self.add_weight(name='total', initializer='zeros')
                self.count = self.add_weight(name='count', initializer='zeros')
            
            def update_state(self, y_true, y_pred, sample_weight=None):
                values = pcm_loss(y_true, y_pred)
                self.total.assign_add(tf.reduce_sum(values))
                self.count.assign_add(tf.cast(tf.size(values), tf.float32))
            
            def result(self):
                return self.total / self.count
            
            def reset_state(self):
                self.total.assign(0.)
                self.count.assign(0.)

        loss_fn = {
            "bits_pred": bits_loss,
            "wm_pcm": pcm_loss
        }
        metrics = {
            "bits_pred": [BitErrorRate(name="ber")],
            "wm_pcm": [PCMLossMetric()],
        }

    else:
        raise ValueError(f"Invalid training stage: {stage}")

    # Compile model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, save_best_only=True, monitor="val_loss", verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor="val_loss"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f"./logs/{log_prefix}", histogram_freq=1
        ),
    ]

    if stage == "B":
        callbacks.append(lambda_cb)

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    print(f"Starting Stage {stage} training...")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return history

# ─────────────────────  DEBUG / SANITY HELPERS  ────────────────────────
class AlphaMonitor(tf.keras.callbacks.Callback):
    def __init__(self, emb): super().__init__(); self.emb = emb
    def on_epoch_end(self, epoch, logs=None):
        if self.emb.alpha_settings == "constant":
            val = float(self.emb.alpha_const.numpy())
            print(f"[α] epoch {epoch+1}: {val:.4f}")
        else:
            g = self.emb.alpha_net(tf.ones((1,1)))
            print(f"[g_k mean] epoch {epoch+1}: {float(tf.reduce_mean(g)):.3f}")

class GradNorm(tf.keras.callbacks.Callback):
    def __init__(self, model): super().__init__(); self.model = model
    def on_train_batch_end(self, batch, logs=None):
        if batch % 200: return
        with tf.GradientTape() as tape:
            y_pred = self.model(logs["x"], training=True)
            loss   = sum(self.model.losses)
        grads = tape.gradient(loss, self.model.trainable_weights)
        norm  = tf.linalg.global_norm([g for g in grads if g is not None])
        print(f"[grad-norm] batch {batch}: {norm.numpy():.2f}")

def inspect_alpha(embedder, pcm_batch, bits_batch):
    """
    1️⃣  Print min / max / mean α on a single forward pass.
    """
    resid  = embedder._residual(pcm_batch)
    alpha  = embedder._alpha_hybrid(pcm_batch, resid, training=False)  # (B,T,1)
    print("\n[α-inspect]  mean={:.4f}  min={:.4f}  max={:.4f}".format(
          tf.reduce_mean(alpha).numpy(),
          tf.reduce_min(alpha).numpy(),
          tf.reduce_max(alpha).numpy()))


def check_orthogonality(embedder, bits_batch, pcm_batch):
    """
    2️⃣  Verify that different bit blocks are time-orthogonal (zero dot-product).
       Assumes resid_sign option already applied inside _carrier_blocks().
    """
    resid   = embedder._residual(pcm_batch)
    chips   = embedder._chips_from_resid(bits_batch, resid)  # (B,T,1)
    B, P, L = bits_batch.shape[0], embedder.P, embedder.L

    # extract two random blocks from first sample
    import numpy as np, random
    i, j = random.sample(range(P), 2)
    c_i = chips[0, i*L : (i+1)*L, 0]
    c_j = chips[0, j*L : (j+1)*L, 0]
    dp  = tf.reduce_sum(c_i * c_j).numpy()
    print(f"[orthogonality] block {i}·{j} dot-product = {dp:.4e}")


def confirm_residual_match(embedder, extractor, pcm_batch):
    """
    3️⃣  Make sure embedder & extractor produce *identical* residuals
        when both are set to the same residual mode.
    """
    r_emb = embedder._residual(pcm_batch)
    r_ext = extractor._residual(pcm_batch)

    err = tf.reduce_mean(tf.abs(r_emb - r_ext)).numpy()
    print(f"[residual-match]  mean |Δ| = {err:.4e}")

# ------------------ grab one batch in a robust way ------------------
def get_sample_batch(dataset):
    """
    Works with (inputs, labels) tuples or plain input-dicts.
    Returns: bits_batch, pcm_batch  (both tensors)
    """
    sample = next(iter(dataset))          # one mini-batch

    # Keras Sequence may give (inp, lbl) or (inp,) etc.
    if isinstance(sample, tuple):
        inp = sample[0]
    else:                                 # tf.data w/ dict only
        inp = sample

    # Functional-API input dict: keys match Input layer names
    if isinstance(inp, dict):
        bits_batch = inp["bits_input"]
        pcm_batch  = inp["pcm_input"]
    else:                                 # pure tuple order
        bits_batch, pcm_batch = inp

    return bits_batch, pcm_batch

def main():
    parser = argparse.ArgumentParser(description="GPU-optimized LP-DSS training")
    parser.add_argument("--train", required=True, help="Training data directory")
    parser.add_argument("--val", required=True, help="Validation data directory")
    parser.add_argument(
        "--stage", default="AB", choices=["A", "B", "AB"], help="Training stage"
    )
    parser.add_argument("--frames", type=int, default=15, help="Frames per payload")
    parser.add_argument("--bits", type=int, default=64, help="Payload bits")
    parser.add_argument(
        "--batch", type=int, default=64, help="Batch size (increased for GPU)"
    )
    parser.add_argument("--epochsA", type=int, default=8, help="Stage A epochs")
    parser.add_argument("--epochsB", type=int, default=12, help="Stage B epochs")
    parser.add_argument("--lporder", type=int, default=12, help="LPC order")
    parser.add_argument("--name", default="gpu_model", help="Model name")
    parser.add_argument("--ckpt", help="Checkpoint path to continue from")
    parser.add_argument("--alphaset", default="constant", choices=["constant", "adaptive"], help="Alpha strategy")
    parser.add_argument("--residsign", type=bool, default=False, help="Carrier built from LP-residual: plain sign() or whiten-&-quantise")
    args = parser.parse_args()

    # Configure GPU
    configure_gpu()

    # Create data loaders with larger batch size for GPU
    print("Loading training data...")
    train_loader = FrameSequencePayload(
        args.train,
        frames_per_payload=args.frames,
        payload_bits=args.bits,
        batch=args.batch,
        shuffle=True,
    )

    print("Loading validation data...")
    val_loader = FrameSequencePayload(
        args.val,
        frames_per_payload=args.frames,
        payload_bits=args.bits,
        batch=args.batch,
        shuffle=False,
    )

    # Create optimized datasets
    train_ds = create_optimized_dataset(train_loader)
    val_ds = create_optimized_dataset(val_loader)

    # Build GPU-optimized model
    print("Building GPU-optimized model...")
    model = build_model_gpu(
        frame        = 160,
        K            = args.frames,
        P            = args.bits,
        lpc_order    = args.lporder,
        alpha_settings  = args.alphaset, 
        resid_sign    = args.residsign,    
        trainable_alpha = False)

    # Print model summary
    model.summary()

    # Create checkpoint directory
    ckpt_dir = Path("./checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # TRAINING STAGES
    # ------------------------------------------------------------------ #
    # 1. constant, not trainable, no attack, 3 epoch
    # ------------------------------------------------------------------ #
    ckpt_path_A1 = ckpt_dir / f"{args.name}_stageA1.h5"
    num_epochs = 3
    embedding_layer = next(l for l in model.layers if isinstance(l, Embedder))
    extractor_layer = next(l for l in model.layers if isinstance(l, LPDSSExtractor))
    attack_layer = next(l for l in model.layers if isinstance(l, AttackPipeline))

    if embedding_layer:
        embedding_layer.alpha_settings = "constant" #{adaptive, constant}
        embedding_layer.resid_sign = False
        embedding_layer.trainable_alpha = False # False by defaul
        embedding_layer.ssl_db = -6

    if attack_layer:
        # Warm-up mode: bypass ALL attacks including filters
        attack_layer.set_warm_up_mode(True)
        
    # pull one mini-batch from the training loader
    '''
    bits_batch, pcm_batch = get_sample_batch(train_ds)   # shapes (B,P) , (B,T,1)

    inspect_alpha(embedding_layer, pcm_batch, bits_batch)
    check_orthogonality(embedding_layer, bits_batch, pcm_batch)
    confirm_residual_match(embedding_layer, extractor_layer, pcm_batch)
    '''

    train_stage_gpu(
        model=model, train_ds=train_ds, val_ds=val_ds,
        epochs=num_epochs, stage="A",
        ckpt_path=str(ckpt_path_A1),
        log_prefix=f"{args.name}_stageA_phase1",
        batch_size=args.batch,
    )

    # ------------------------------------------------------------------ #
    # 2. constant, trainable, gradual attack, 3 epoch
    # ------------------------------------------------------------------ #
   



if __name__ == "__main__":
    main()