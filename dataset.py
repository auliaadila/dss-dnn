# in keras
import os
import numpy as np
import tensorflow as tf
import soundfile as sf
from dss import embed_dss, generate_bps  # existing DSS functions

	# •	frame_size = How long each audio example is (160 samples/frame)
	# •	batch_size = How many of those examples you process together in each training step


class FrameSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence for frame-wise DSS watermark detection.
    Each item is a batch of (frames, carriers) with labels.
    """
    def __init__(self, wav_folder, bps, ssl_db, frame_size, fs,
                 batch_size=32, max_files=None, shuffle=True):
        self.frame_size = frame_size
        self.fs = fs
        self.bps = bps
        self.ssl_db = ssl_db
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 1) Load all host signals
        self.hosts = []
        for i, fname in enumerate(os.listdir(wav_folder)):
            if max_files and i >= max_files:
                break
            if not fname.lower().endswith('.wav'):
                continue
            path = os.path.join(wav_folder, fname)
            sig, fs_read = sf.read(path)
            assert fs_read == fs, f"Expected fs={fs}, got {fs_read}"
            self.hosts.append(sig.flatten())

        # 2) Build index list: (host_idx, frame_idx)
        self.indices = []
        for h_idx, host in enumerate(self.hosts):
            nframes = len(host) // self.frame_size
            for f in range(nframes):
                self.indices.append((h_idx, f))
                # host_index, frame_index (per host)

        # print("Index")
        # print(self.indices)

        self.on_epoch_end()

    def __len__(self):
        # number of batches per epoch (self.indices is generated per epoch)
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_idx):
        # fetch batch indices
        start = batch_idx * self.batch_size
        end   = start + self.batch_size
        batch_indices = self.indices[start:end]
        print("batch_indices:", len(batch_indices)) #8
        print(batch_indices)

        # prepare containers
        wm_frames = np.zeros((len(batch_indices), self.frame_size), dtype=np.float32)
        carriers = np.zeros_like(wm_frames)
        labels = np.zeros((len(batch_indices),), dtype=np.int32)

        # fill batch
        for i, (host_idx, frame_idx) in enumerate(batch_indices):
            host_sig = self.hosts[host_idx]
            # generate or reuse a bit-per-second pattern
            wm_bps = generate_bps(self.bps)
            print("host idx | frame idx:", host_idx, frame_idx)
            print("generated bps:", wm_bps)
            # embed entire signal once
            watermarked, _, embed_bits_extended, c = embed_dss(
                host_sig, self.fs, wm_bps,
                self.ssl_db, self.frame_size
            )
            # slice frame
            s = frame_idx * self.frame_size
            wm_frames[i] = watermarked[s:s+self.frame_size]
            carriers[i] = c[:, frame_idx]
            labels[i] = embed_bits_extended[frame_idx]

        # return as tuple of arrays
        return [wm_frames, carriers], labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
