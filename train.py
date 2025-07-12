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
from components.extractor_fixed import DSSExtractorFixed as DSSExtractor
from components.lossfunc_fixed import bit_consistency_loss
from components.metrics import BitErrorRate
from components.embedding import Embedder


def build_model_gpu(frame=160, K=15, P=64,
                    lpc_order=12,
                    spread_method="dss",
                    alpha_settings="constant",
                    resid_method="sign",
                    trainable_alpha=False):
    """Return GPU-optimized uncompiled Keras model (bits, pcm) → (bits_pred)."""

    # Input layers
    bits_input = tf.keras.Input(shape=(P,), name="bits_input")
    pcm_input = tf.keras.Input(shape=(K * frame, 1), name="pcm_input")

    # GPU-optimized embedding
    embedding = Embedder(frame_size=frame,
                         K=K, P=P,
                         lporder=lpc_order,
                         spread_method   = spread_method,
                         resid_method = resid_method,
                         alpha_settings  = alpha_settings,
                         trainable_alpha = trainable_alpha,
                         use_full_lpc    = True)

    # Apply embedding
    watermarked = embedding([bits_input, pcm_input])

    # GPU-optimized attack pipeline with fixed training flag propagation
    attacks = AttackPipeline(
        fir_cutoff=0.8, iir_cutoff=0.8, snr_db=20.0, max_cut_ratio=0.1
    )

    # Apply attacks (training flag will be automatically forwarded via learning_phase)
    attacked = attacks(watermarked)

    # Extractor (already GPU-optimized)
    extractor = DSSExtractor(payload_bits=P, context_len=K * frame)

    # Extract bits
    bits_pred = extractor(attacked)

    # Create model
    model = tf.keras.Model(
        inputs=[bits_input, pcm_input], outputs=bits_pred, name="LPDSS_GPU"
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

            # Use mixed precision for better performance (disabled for now)
            # policy = tf.keras.mixed_precision.Policy('mixed_float16')
            # tf.keras.mixed_precision.set_global_policy(policy)
            # print("Mixed precision enabled (float16)")
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
    # Use lower learning rate for Stage B to prevent instability
    lr = 1e-4 if stage == "B" else 5e-4
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        clipnorm=1.0,  # Clip gradients to prevent explosion
    )
    # optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)  # Disabled for now

    # Configure loss and metrics
    # Use bit-wise BCE + temperature-sharpening loss for better performance
    loss_fn = lambda y_true, y_pred: bit_consistency_loss(y_true, y_pred, lam=0.2)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        BitErrorRate(name="ber"),
    ]

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

    # Add extra callbacks if provided
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    # Train
    print(f"Starting Stage {stage} training...")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1
    )

    return history


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
    parser.add_argument("--spread", default="dss", choices=["dss", "lpdss"], help="Spread method: DSS or LPDSS")
    parser.add_argument("--alphaset", default="constant", choices=["constant", "adaptive"], help="Alpha strategy")
    parser.add_argument("--residmethod", default="sign", choices=["sign", "whiten"], help="Carrier built from LP-residual: plain sign() or whiten-&-quantise")
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
        spread_method   = args.spread,       
        alpha_settings  = args.alphaset, 
        resid_method    = args.residmethod if args.spread == "lpdss" else "sign",    
        trainable_alpha = False)

    # Print model summary
    model.summary()

    # Create checkpoint directory
    ckpt_dir = Path("./checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # Training stages
    if args.stage in ["A", "AB"]:
        print("\n" + "=" * 60)
        print("STAGE A: Fixed alpha, moderate attacks")
        print("=" * 60)

        # Stage A: Fixed alpha
        # Find the embedding layer
        embedding_layer = None
        for layer in model.layers:
            if isinstance(layer, Embedder):
                embedding_layer = layer
                break

        # Find attack pipeline for warm-up configuration
        attack_layer = None
        for layer in model.layers:
            if isinstance(layer, AttackPipeline):
                attack_layer = layer
                break


        if embedding_layer:

            FIXED_ALPHA = float(embedding_layer.alpha_const.numpy())
            embedding_layer.alpha_settings = args.alphaset #{adaptive, constant}
            embedding_layer.spread_method  = args.spread #{dss, lpdss}

            if args.spread == "lpdss":
                embedding_layer.resid_method = args.residmethod

            if args.alphaset == "constant":
                embedding_layer.set_alpha_value(FIXED_ALPHA)
                embedding_layer.set_alpha_trainable(False)     # stays frozen
            else:   # "adaptive"
                embedding_layer.a_min = embedding_layer.a_max = FIXED_ALPHA
                embedding_layer.alpha_net.trainable = False    # α-net frozen in Stage-A

        if attack_layer:
            # Warm-up mode: bypass ALL attacks including filters
            attack_layer.set_warm_up_mode(True)

        ckpt_path_A1 = ckpt_dir / f"{args.name}_stageA1.h5"
        ckpt_path_A2 = ckpt_dir / f"{args.name}_stageA2.h5"
        ckpt_path_A3 = ckpt_dir / f"{args.name}_stageA3.h5"
        ckpt_path_A = ckpt_dir / f"{args.name}_stageA.h5"  # Final Stage A checkpoint

        no_attack_a_epochs = 3
        # Stage A Phase 1: No attacks at all
        print("Phase 1: No attacks - letting extractor learn basic watermark")
        train_stage_gpu(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=no_attack_a_epochs,
            stage="A",
            ckpt_path=str(ckpt_path_A1),
            log_prefix=f"{args.name}_stageA_phase1",
            batch_size=args.batch,
        )

        # Load Phase 1 weights to continue training
        print(f"Loading Phase 1 weights from {ckpt_path_A1}") #ok
        try:
            model.load_weights(str(ckpt_path_A1))
            print(f"✓ Phase 1 checkpoint loaded successfully") #ok
        except Exception as e:
            print(f"❌ Error loading Phase 1 checkpoint: {e}")
            raise

        # Stage A Phase 2: Gradual attack introduction
        remaining_epochs = max(1, args.epochsA - no_attack_a_epochs)
        if remaining_epochs > 0 and attack_layer:
            mild_epochs = max(1, remaining_epochs // 3)
            mod_epochs = max(1, remaining_epochs // 3)
            full_epochs = remaining_epochs - mild_epochs - mod_epochs

            print(f"Phase 2a: Mild attacks ({mild_epochs} epochs)")
            attack_layer.set_gradual_attacks("mild")
            train_stage_gpu(
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                epochs=mild_epochs,
                stage="A",
                ckpt_path=str(ckpt_path_A2),
                log_prefix=f"{args.name}_stageA_phase2a",
                batch_size=args.batch,
            )

            # Load weights from Phase 2a
            print(f"Loading Phase 2a weights from {ckpt_path_A2}")
            try:
                model.load_weights(str(ckpt_path_A2))
                print(f"✓ Phase 2a checkpoint loaded successfully") #ok
            except Exception as e:
                print(f"❌ Error loading Phase 2a checkpoint: {e}")
                raise

            print(f"Phase 2b: Moderate attacks ({mod_epochs} epochs)")
            attack_layer.set_gradual_attacks("moderate")
            train_stage_gpu(
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                epochs=mod_epochs,
                stage="A",
                ckpt_path=str(ckpt_path_A3),
                log_prefix=f"{args.name}_stageA_phase2b",
                batch_size=args.batch,
            )

            # Load weights from Phase 2b
            print(f"Loading Phase 2b weights from {ckpt_path_A3}")
            try:
                model.load_weights(str(ckpt_path_A3))
                print(f"✓ Phase 2b checkpoint loaded successfully")
            except Exception as e:
                print(f"❌ Error loading Phase 2b checkpoint: {e}")
                raise

            print(f"Phase 2c: Full attacks ({full_epochs} epochs)")
            attack_layer.set_gradual_attacks("full")
            train_stage_gpu(
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                epochs=full_epochs,
                stage="A",
                ckpt_path=str(ckpt_path_A),
                log_prefix=f"{args.name}_stageA_phase2c",
                batch_size=args.batch,
            )

            # Load final Phase 2c weights
            print(f"Loading Phase 2c weights from {ckpt_path_A}")
            try:
                model.load_weights(str(ckpt_path_A))
                print(f"✓ Phase 2c checkpoint loaded successfully")
            except Exception as e:
                print(f"❌ Error loading Phase 2c checkpoint: {e}")
                raise

        print(f"Stage A complete. Checkpoint saved to {ckpt_path_A}")

    if args.stage in ["B", "AB"]:
        print("\n" + "=" * 60)
        print("STAGE B: Trainable alpha, stronger attacks")
        print("=" * 60)

        # Stage B: Trainable alpha
        if args.stage == "B" and args.ckpt:
            print(f"Loading checkpoint from {args.ckpt}")
            model.load_weights(args.ckpt)
        elif args.stage == "AB":
            # Load Stage A weights to continue training
            print(f"Loading Stage A weights from {ckpt_path_A}")
            try:
                model.load_weights(str(ckpt_path_A))
                print(f"✓ Stage A checkpoint loaded successfully for Stage B")
            except Exception as e:
                print(f"❌ Error loading Stage A checkpoint: {e}")
                raise
        
        # Find the embedding layer
        embedding_layer = None
        for layer in model.layers:
            if isinstance(layer, Embedder):  
                embedding_layer = layer
                
                break

        # Find attack pipeline to re-enable normal attacks
        attack_layer = None
        for layer in model.layers:
            if isinstance(layer, AttackPipeline):
                attack_layer = layer
                break

        if embedding_layer:
            if embedding_layer.alpha_settings == "constant":
                embedding_layer.set_alpha_trainable(True)        # learn scalar α
            else:                                               # adaptive
                embedding_layer.a_min, embedding_layer.a_max = 1e-3, 2e-1
                embedding_layer.alpha_net.trainable = True       # learn α-net
                
            print("trainable?", embedding_layer.alpha_net.trainable,
                        "a_min/max:", embedding_layer.a_min, embedding_layer.a_max)

        if attack_layer:
            # Normal mode: full attack strength
            attack_layer.set_warm_up_mode(False)

        ckpt_path_B = ckpt_dir / f"{args.name}_stageB.h5"

        # # Custom callback to print alpha during Stage B
        # class AlphaPrintCallback(tf.keras.callbacks.Callback):
        #     def __init__(self, emb):
        #         super().__init__()
        #         self.emb = emb
        #     def on_batch_end(self, batch, logs=None):
        #         a = self.emb.current_alpha      # (B,1,1) tensor you expose from call()
        #         if batch % 100 == 0:            # print every 100 batches
        #             mn = tf.reduce_min(a).numpy()
        #             mx = tf.reduce_max(a).numpy()
        #             sd = tf.math.reduce_std(a).numpy()
        #             print(f"[batch {batch}] α min={mn:.4f}  max={mx:.4f}  std={sd:.4f}")   

        class AlphaPrintCallback(tf.keras.callbacks.Callback):
            def __init__(self, emb): super().__init__(); self.emb = emb

            def on_epoch_end(self, batch, epoch, logs=None):
                if self.emb is None: return
                if self.emb.alpha_settings == "constant":
                    print(f"[ep {epoch+1}] α = {self.emb.alpha_const.numpy():.4f}")
                else:
                    α_hat = self.emb.alpha_net(tf.ones((1,1))).numpy()[0,0]
                    α     = self.emb.a_min + (self.emb.a_max-self.emb.a_min)*α_hat
                    print(f"[ep {epoch+1}] α̂ = {α:.4f} "
                        f"(norm={α_hat:.3f}, range {self.emb.a_min:.3f}…{self.emb.a_max:.3f})")

        alpha_callback = AlphaPrintCallback(embedding_layer)

        train_stage_gpu(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=args.epochsB,
            stage="B",
            ckpt_path=str(ckpt_path_B),
            log_prefix=f"{args.name}_stageB",
            batch_size=args.batch,
            extra_callbacks=[alpha_callback],
        )

        print(f"Stage B complete. Checkpoint saved to {ckpt_path_B}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()