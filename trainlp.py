#!/usr/bin/env python3
"""
GPU-optimised training script for LP-DSS watermarking
that matches LPDSSEmbedder + LPDSExtractor (July-2025 APIs).
"""

import argparse, os, datetime
from pathlib import Path
import tensorflow as tf

# ─── your own project modules ────────────────────────────────────────────
from components.attack_pipeline_fixed import AttackPipelineFixed as AttackPipeline
from components.dataloader_cached    import FrameSequencePayload
from components.embeddinglp          import LPDSSEmbedder  as Embedder
from components.extractorlp          import LPDSExtractor  as Extractor
from components.lossfunc_fixed       import (
    bit_consistency_loss, frequency_masking_loss, l1_loss, snr_loss)
from components.metrics              import BitErrorRate
from components.residual import ResidualLayer
# ─────────────────────────────────────────────────────────────────────────



# ════════════════════════════════════════════════════════════════════════
# 1.  MODEL CONSTRUCTION
# ════════════════════════════════════════════════════════════════════════
def build_model_gpu(frame=160, K=15, P=64,
                    lpc_order=12,
                    alpha_settings="constant",
                    trainable_alpha=False):

    bits_in = tf.keras.Input((P,),         name="bits_input")
    pcm_in  = tf.keras.Input((K*frame, 1), name="pcm_input")

    # — embedder —
    embedder = Embedder(frame_size = frame, K = K, P = P,
                        lporder = lpc_order,
                        ssl_db  = -10,
                        alpha_settings  = alpha_settings,
                        trainable_alpha = trainable_alpha,
                        use_full_lpc    = True,
                        resid_whiten    = True)

    wm_pcm = embedder([bits_in, pcm_in])             # (B,T,1)

    # — attacks —
    attacks = AttackPipeline(fir_cutoff=0.8, iir_cutoff=0.8,
                             snr_db=20.0, max_cut_ratio=0.1)
    attacked_pcm = attacks(wm_pcm)

    # — residual for extractor —
    resid = ResidualLayer(
        use_full_lpc=True,
        frame_size=frame,
        total_len=frame * K,
        lpc_order=lpc_order,
        name="residual"
    )(attacked_pcm)

        # — extractor —
    extractor = Extractor(payload_bits=P, context_len=K*frame, base_filters=64)
    bits_pred = extractor(resid)

    return tf.keras.Model(
        inputs  = [bits_in, pcm_in],
        outputs = {"bits_pred": bits_pred, "wm_pcm": wm_pcm},
        name="LPDSS_GPU"
    )

# ════════════════════════════════════════════════════════════════════════
# 2.  GPU & DATA HELPERS   (unchanged except cosmetic)
# ════════════════════════════════════════════════════════════════════════
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if not gpus:
        print("⚠  No GPU detected – running on CPU."); return
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    print(f"✓  {len(gpus)} GPU(s) found.")
    # mixed precision off by default; turn on if Tensor Cores help you.

def create_optimized_dataset(dataloader):
    """Return dataloader (already optimized with caching)."""
    return dataloader

# ════════════════════════════════════════════════════════════════════════
# 3.  TRAINING STAGE WRAPPER  (API unchanged, only docstring snippets)
#    — Stage A  : freeze embedder α, light/no attack → train extractor
#    — Stage B  : un-freeze α (or α-net), enable full attacks
# ════════════════════════════════════════════════════════════════════════
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
    create_new_optimizer=True
):
    """GPU-optimized training stage."""

    # Configure optimizer with gradient clipping
    if create_new_optimizer or model.optimizer is None:
        lr = 1e-4 if stage == "B" else 5e-4
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    else:
        optimizer = model.optimizer        # keep old Adam state

    # lr = 1e-4 if stage == "B" else 5e-4
    # optimizer = tf.keras.optimizers.Adam(
    #     learning_rate=lr,
    #     clipnorm=1.0,
    # )

    # Stage A setup
    if stage == "A":          #  ← keep the if/elif structure
        lambda_bits = 1.0       # FULL BCE gradient – extractor learns faster

        def bits_loss(y_true, y_pred):
            return bit_consistency_loss(y_true, y_pred) * lambda_bits

        loss_fn = {"bits_pred": bits_loss, "wm_pcm": None}
        metrics = {"bits_pred": [tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                                BitErrorRate(name="ber")],
                "wm_pcm": None}

    # Stage B setup
    elif stage == "B":
        # mutable scalar
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
            ckpt_path, save_best_only=True, save_weights_only = False, monitor="val_loss", verbose=1
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

# ════════════════════════════════════════════════════════════════════════
# 4.  CLI / MAIN
# ════════════════════════════════════════════════════════════════════════
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--train",   required=True)
    pa.add_argument("--val",     required=True)
    pa.add_argument("--frames",  type=int, default=15)
    pa.add_argument("--bits",    type=int, default=64)
    pa.add_argument("--batch",   type=int, default=32)
    pa.add_argument("--lporder", type=int, default=12)
    pa.add_argument("--alphaset",default="constant",
                    choices=["constant", "adaptive"])
    pa.add_argument("--name",    default="gpu_model")
    # pa.add_argument("--ckpt")            # not used but kept for compat
    args = pa.parse_args()

    tf.random.set_seed(42)

    configure_gpu()

    train_loader = FrameSequencePayload(args.train, frames_per_payload=args.frames,
                                        payload_bits=args.bits, batch=args.batch, shuffle=True)
    val_loader   = FrameSequencePayload(args.val,   frames_per_payload=args.frames,
                                        payload_bits=args.bits, batch=args.batch, shuffle=False)

    train_ds = create_optimized_dataset(train_loader)
    val_ds   = create_optimized_dataset(val_loader)

    model = build_model_gpu(frame=160, K=args.frames, P=args.bits,
                            lpc_order=args.lporder,
                            alpha_settings="constant",     # start constant
                            trainable_alpha=False)         # frozen scalar

    model.summary(line_length=110)
    ckpt_dir = Path("./checkpoints"); ckpt_dir.mkdir(exist_ok=True)

    # ckpt path
    ckpt_path_P0 = ckpt_dir/f"{args.name}_P0"
    ckpt_path_P1 = ckpt_dir/f"{args.name}_P1"
    ckpt_path_P2 = ckpt_dir/f"{args.name}_P2"
    ckpt_path_P3 = ckpt_dir/f"{args.name}_P3"

    def reload_full_model(model_path):
        print(f"Reloading full model from {model_path}")
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={
                "Embedder": Embedder,
                "AttackPipeline": AttackPipeline,
                "Extractor": Extractor,
                "ResidualLayer": ResidualLayer,
            }
        )
 
    # ------------------------------------------------------------------ #
    # PHASE 0 : constant α, frozen, NO attacks 
    # ------------------------------------------------------------------ #­­­­­­­­­­
    print("\n─── Phase 0 | constant α (frozen) | no attack ───")
    embedder = next(l for l in model.layers if isinstance(l, Embedder))
    attacks  = next(l for l in model.layers if isinstance(l, AttackPipeline))

    attacks.set_warm_up_mode(True)                 # bypass everything
    train_stage_gpu(model, train_ds, val_ds,
                    epochs=1, stage="A",
                    ckpt_path=str(ckpt_path_P0),
                    log_prefix=f"{args.name}_P0",
                    batch_size=args.batch,
                    )
    
    model.save(ckpt_path_P0)

    # ------------------------------------------------------------------ #
    # PHASE 1 : constant α, trainable scalar ─­­­­­­­­­
    # ------------------------------------------------------------------ #­­­­­­­­­­
    print("\n─── Phase 1 | constant α (trainable) | mild+mod attack ───")

    
    model = tf.keras.models.load_model(ckpt_path_P0)
    embedder = next(l for l in model.layers if isinstance(l, Embedder))
    attacks  = next(l for l in model.layers if isinstance(l, AttackPipeline))

    # model = build_model_gpu(..., trainable_alpha=True, ...)

    embedder.alpha_const.trainable = True
    # Start with 'mild'
    attacks.set_gradual_attacks("mild")      

    # ── Phase-1 setup ──────────────────────────────────────────────
    phase1_epochs = 6
    switch_epoch  = 3           # epoch index at which to bump to "moderate"
    # define the callback that will change it to “moderate” later
    class AttackRamp(tf.keras.callbacks.Callback):
        """Flip mild → moderate at a chosen epoch."""
        def __init__(self, attack_layer, switch_epoch):
            super().__init__()
            self.attack       = attack_layer
            self.switch_epoch = switch_epoch
        def on_epoch_begin(self, epoch, logs=None):
            if epoch == self.switch_epoch:
                print("\n⇧ Switching attacks: mild → moderate\n")
                self.attack.set_gradual_attacks("moderate")

    attack_callback = AttackRamp(attacks, switch_epoch)

    train_stage_gpu(
        model, train_ds, val_ds,
        epochs       = phase1_epochs,
        stage        = "A",
        ckpt_path    = str(ckpt_path_P1),
        log_prefix   = f"{args.name}_P1",
        batch_size   = args.batch,
        extra_callbacks=[attack_callback],
        create_new_optimizer=False
    )

    # Decide path: constant OR adaptive
    if args.alphaset == "constant":
        print("\n==== Constant-α flavour selected; stopping after Phase 1 ====")
        return

    # ------------------------------------------------------------------ #
    # PHASE 2 : adaptive structure, gate frozen ­­­­­­­­­­
    # ------------------------------------------------------------------ #­­­­­­­­­­
    print("\n─── Phase 2 | adaptive gate (frozen) | full attack ───")
    model = reload_full_model(ckpt_path_P1)   # replaces load_weights(...)
    embedder = next(l for l in model.layers if isinstance(l, Embedder))
    attacks  = next(l for l in model.layers if isinstance(l, AttackPipeline))

    attacks.set_warm_up_mode(False)                # enable full attacks
    
    embedder.alpha_settings      = "adaptive"
    # make sure gate exists (if model was built with alpha_settings="constant")
    if not hasattr(embedder, "alpha_net"):
        b0 = 0.0
        embedder.alpha_net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid",
                bias_initializer=tf.keras.initializers.Constant(b0))
        ])
        embedder.alpha_net.build((None,1))
        embedder.alpha_net.trainable = False

    embedder.alpha_net.trainable = False
    embedder.alpha_const.trainable = False
    
    train_stage_gpu(model, train_ds, val_ds,
                    epochs=1, stage="B",
                    ckpt_path=str(ckpt_path_P2),
                    log_prefix=f"{args.name}_P2",
                    batch_size=args.batch,
                    create_new_optimizer=False)

    # ------------------------------------------------------------------ #
    # PHASE 3 : adaptive gate trainable full run ­­­­­­­­­­
    # ------------------------------------------------------------------ #­­­­­­­­­­
    print("\n─── Phase 3 | adaptive gate (trainable) | full attack ───")
    model = reload_full_model(ckpt_path_P2)   # replaces load_weights(...)
    embedder = next(l for l in model.layers if isinstance(l, Embedder))
    attacks  = next(l for l in model.layers if isinstance(l, AttackPipeline))

    embedder.alpha_net.trainable = True
    train_stage_gpu(model, train_ds, val_ds,
                    epochs=10, stage="B",
                    ckpt_path=str(ckpt_path_P3),
                    log_prefix=f"{args.name}_P3",
                    batch_size=args.batch,
                    create_new_optimizer=False)

    print("\n✓ TRAINING COMPLETE – final checkpoint:",
          ckpt_dir/f"{args.name}_P3.h5")

if __name__ == "__main__":
    main()