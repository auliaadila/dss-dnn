from torch.utils.data import DataLoader
from datasets import FrameDataset

# 1) Instantiate with a small folder of WAVs
ds = FrameDataset(
    wav_folder="dataset/host",
    bps=4,
    ssl_db=3,
    frame_size=160,
    fs=16000,
    max_files=2       # limit for quick test
)

print("Dataset length (frames):", len(ds))

# 2) Grab a few samples
for i in [0, len(ds)//2, len(ds)-1]:
    frame, carrier, bit = ds[i]
    print(f"\nSample idx={i}:")
    print("  frame.shape  =", frame.shape)    # expect torch.Size([160])
    print("  carrier.shape=", carrier.shape)  # expect torch.Size([160])
    print("  bit          =", bit.item())      # expect 0 or 1

# 3) Batch test via DataLoader
loader = DataLoader(ds, batch_size=8, shuffle=True)
batch = next(iter(loader))
frames, carriers, bits = batch
print("\nBatch shapes:")
print("  frames  =", frames.shape)    # expect [8, 160]
print("  carriers=", carriers.shape)  # expect [8, 160]
print("  bits    =", bits.shape)      # expect [8]
print("  bits unique:", bits.unique())