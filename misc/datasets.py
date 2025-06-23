# datasets.py
# in torch
import os, numpy as np, torch
from torch.utils.data import Dataset
from dss.dss import embed_dss, generate_bps  # your existing embed function
import soundfile as sf

class FrameDataset(Dataset):
    def __init__(self, wav_folder, bps, ssl_db, frame_size, fs, max_files=None):
        self.frame_size = frame_size
        self.fs = fs
        self.bps = bps
        self.ssl_db = ssl_db

        # 1) Load all hosts
        self.hosts = []
        for i, fname in enumerate(os.listdir(wav_folder)):
            if max_files and i >= max_files:
                break
            if not fname.lower().endswith(".wav"):
                continue
            path = os.path.join(wav_folder, fname)
            sig, fs_read = sf.read(path)
            assert fs_read == fs, "Sampling rate mismatch"
            self.hosts.append(sig.flatten())

        # 2) Build an index: list of (host_idx, frame_idx)
        self.indices = []
        for h_idx, host in enumerate(self.hosts):
            nframes = len(host) // self.frame_size
            for f in range(nframes):
                self.indices.append((h_idx, f))
        
        print("Index")
        print(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        host_idx, frame_idx = self.indices[idx]
        host_sig = self.hosts[host_idx]

        # 3) Generate a random wm_bps of length ceil(bps*duration)
        # duration = len(host_sig) / self.fs
        # total_bits = int(np.ceil(self.bps * duration))
        # wm_bps = np.random.randint(0, 2, size=total_bits).tolist()

        wm_bps = generate_bps(self.bps)

        # 4) Embed the *entire* signal once
        watermarked, _, embed_bits_extended, c = embed_dss(
            host_signal=host_sig,
            fs=self.fs,
            wm_bps=wm_bps,
            ssl_db=self.ssl_db,
            frame_size=self.frame_size
        )

        # # Group frames into bits
        # frames_per_sec = fs / frame_size
        # frames_per_bit = int(np.ceil(frames_per_sec / bps))
        # total_bits    = int(np.ceil(bps * (N / fs)))

        # 5) Slice out that one frame and its carrier
        start = frame_idx * self.frame_size
        frame   = watermarked[start : start + self.frame_size]
        carrier = c[:, frame_idx]
        bit     = embed_bits_extended[frame_idx]

        # 6) Return tensors
        return (
            torch.from_numpy(frame).float(),
            torch.from_numpy(carrier).float(),
            torch.tensor(bit, dtype=torch.float32)
        )