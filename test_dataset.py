import numpy as np
from dataset import FrameSequence
import soundfile as sf
import os 
from dss.dss import embed_dss

def test_dataloader():
    # Adjust these paths/params to match your setup
    wav_folder = "dataset/host"
    bps        = 4
    ssl_db     = 3
    frame_size = 160
    fs         = 16000
    batch_size = 8
    max_files  = 2    # limit for a quick smoke‐test

    seq = FrameSequence(
        wav_folder=wav_folder,
        bps=bps,
        ssl_db=ssl_db,
        frame_size=frame_size,
        fs=fs,
        batch_size=batch_size,
        max_files=max_files,
        shuffle=False
    )

    print(f"→ Total batches per epoch: {len(seq)}")
    print(f"→ Total frames indexed:      {len(seq.indices)}\n")

    # Pull the first batch
    (frames, carriers), labels = seq[0]
    print("Batch #0:")
    print(" frames shape:  ", frames.shape)    # (batch_size, frame_size)
    print(" carriers shape:", carriers.shape)  # (batch_size, frame_size)
    print(" labels shape:  ", labels.shape)    # (batch_size,)
    
    print(frames)
    print(carriers)
    print(labels)

    '''
    print(seq) #<dataset.FrameSequence object at 0x793209793cd0>

    print("Batch #0:")
    print(" frames shape:  ", frames.shape)    # (batch_size, frame_size)
    print(" carriers shape:", carriers.shape)  # (batch_size, frame_size)
    print(" labels shape:  ", labels.shape)    # (batch_size,)
    print(labels)

    # Quick value checks
    print("\nLabel distribution in batch:", np.unique(labels, return_counts=True))
    print("Frame value range:  ", frames.min(), "→", frames.max())
    print("Carrier value range:", carriers.min(), "→", carriers.max())

    # Try iterating a couple more batches
    for batch_idx in range(1, min(3, len(seq))):
        (f2, c2), l2 = seq[batch_idx]
        print(f"\nBatch #{batch_idx}: frames {f2.shape}, labels unique {np.unique(l2)}")
    

    
    # pick the first host file in your sequence
    first_host_idx, _ = seq.indices[0]
    host_sig = seq.hosts[first_host_idx]  # full 1D array
    print(first_host_idx)
    print(host_sig.shape)
    # generate a random wm_bps of the right length
    duration   = len(host_sig) / fs
    total_bits = int(np.ceil(bps * duration))
    wm_bps     = np.random.randint(0, 2, size=total_bits)

    # embed the entire signal
    watermarked, wm_bits_used, bit_s, carrier = embed_dss(
        host_sig, fs, wm_bps, ssl_db, frame_size
    )

    # ensure output directory exists
    out_folder = "dataset/dataset-dss-wm"
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, "host0_watermarked.wav")

    # write to disk
    sf.write(out_path, watermarked, fs)
    print(f"Exported watermarked file to: {out_path}")

    '''
    

if __name__ == "__main__":
    test_dataloader()