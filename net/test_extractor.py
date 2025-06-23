from net.extractor import build_extractor
model = build_extractor(bits_per_frame=64)
model.summary()