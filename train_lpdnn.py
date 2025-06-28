# Train LP-DSS + DNN
import os, glob, math, argparse, re
from typing import List, Tuple
import warnings

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_io as tfio
import importlib
from components.lpdss_dnn import *
from components.lossfunc import *

from components.dataset import FrameSequence


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train_dir',required=True)
    ap.add_argument('--val_dir',required=True)
    ap.add_argument('--epoch', type=int, default=2)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--output', metavar="<output>", help="trained model file (.h5)")
    ap.add_argument("--retrain", metavar="<input weights>", help="continue training model")
    ap.add_argument(
        "--auto-resume",
        action="store_true",
        help="automatically resume from latest checkpoint",
    )
    ap.add_argument(
        "--checkpoint-freq",
        type=int,
        default=1,
        help="save checkpoint every N epochs (default: 1)",
    )
    ap.add_argument(
        "--cuda-devices",
        metavar="<cuda devices>",
        type=str,
        default=None,
        help="string with comma separated cuda device ids",
    )
    ap.add_argument(
        "--logdir", metavar="<log dir>", help="directory for tensorboard log files"
    )

    args=ap.parse_args()

    # set visible cuda devices
    if args.cuda_devices != None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # input data preprocessing
    seq_train=FrameSequence(args.train_dir, batch_size=args.batch)
    seq_val  =FrameSequence(args.val_dir,   batch_size=args.batch, shuffle=False)

    # build model
    model, wm_embed, wm_add, wm_extract = build_model(
        batch_size=args.batch,
        training=True
    )
    opt = tf.keras.optimizers.Adam(1e-4)
    
    # compile model: define loss
    model.compile(
        optimizer=opt,
        loss={  # utk gradient
            "residual_w": residual_distribution_loss,
            "pcm_w": [l1_loss, spectral_loss, snr_loss],
            "bits_pred": bit_consistency_loss,
        },
        loss_weights={
            "residual_w": 0.01,  # Lower weight for residual regularization
            "pcm_w": [0.1, 0.1, 0.05],  # Much lower for imperceptibility
            "bits_pred": 2.0,  # Higher weight for bit accuracy
        },
        metrics={  # utk logging
            "residual_w": None,
            "pcm_w": [perceptual_loss, spectral_loss],
            "bits_pred": ["accuracy", bit_consistency_loss],
        },
        run_eagerly=True,
    )

    model.summary()


    # Create checkpoint directory
    checkpoint_dir = f"checkpoints_{args.output}"

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save both model weights and optimizer state
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_epoch_{epoch:02d}.ckpt")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_weights_only=True,
        save_freq="epoch",  # Save every epoch, not every batch
        verbose=1,
    )


    # Function to find latest checkpoint
    def find_latest_checkpoint(checkpoint_dir):
        pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.ckpt.index")
        checkpoints = glob.glob(pattern)
        if not checkpoints:
            return None, 0

        # Extract epoch numbers and find the latest
        epochs = []
        for ckpt in checkpoints:
            match = re.search(r"checkpoint_epoch_(\d+)\.ckpt\.index", ckpt)
            if match:
                epochs.append(int(match.group(1)))

        if epochs:
            latest_epoch = max(epochs)
            latest_ckpt = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{latest_epoch:02d}.ckpt"
            )
            return latest_ckpt, latest_epoch
        return None, 0


    # Auto-resume from latest checkpoint or use --retrain flag
    initial_epoch = 0
    if args.retrain is not None:
        print(f"Loading weights from specified checkpoint: {args.retrain}")
        model.load_weights(args.retrain)
    elif args.auto_resume:
        # Try to auto-resume from latest checkpoint
        latest_ckpt, latest_epoch = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt:
            print(f"Auto-resuming from checkpoint: {latest_ckpt} (epoch {latest_epoch})")
            model.load_weights(latest_ckpt)
            initial_epoch = latest_epoch
        else:
            print("No checkpoint found - starting training from scratch")
    else:
        print("Starting training from scratch")


    model.save_weights("{}_{}_initial.h5".format(args.output))


    attack_scheduler = AttackScheduler(
        model, stage1_end=40, stage2_end=80, stage3_end=120, max_strength_multiplier=3.0
    )
    callbacks = [checkpoint, attack_scheduler]
    if args.logdir is not None:
        logdir = "{}/{}_{}_logs".format(args.logdir, args.output)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        callbacks.append(tensorboard_callback)

    model.fit(
        seq_train,
        validation_data=seq_val,
        epochs=args.epoch,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True,
    )
