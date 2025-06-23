import os
import numpy as np
import soundfile as sf
from pesq import pesq as pesq_score

import dss

# Configuration
FOLDER_IN = 'dataset/host'           # folder with input .wav files
FOLDER_OUT = 'dataset/dss-wm'
BPS = 64                   # bits per second
SSL_DB = -25                # embedding strength in dB
FRAME_SIZE = 160

os.makedirs(FOLDER_OUT, exist_ok=True)

# Iterate WAV files
results = []
for fname in os.listdir(FOLDER_IN):
    if not fname.lower().endswith('.wav'):
        continue
    path = os.path.join(FOLDER_IN, fname)

    # Read host
    host, fs = sf.read(path)
    host = host if host.ndim == 1 else host[:,0]

    # Generate random watermark bits
    # duration = len(host) / fs
    # total_bits = int(np.ceil(BPS * duration))
    wm_bps = dss.generate_bps(BPS)

    # Embed
    watermarked, embed_bits, _ , _ = dss.embed_dss(host, fs, wm_bps, SSL_DB, FRAME_SIZE)
    print("embedded WM:",embed_bits.shape)
    # print(embed_bits)

    # export watermarked signal
    output_path = os.path.join(FOLDER_OUT, f"WM_{fname}.wav")
    sf.write(output_path, watermarked, fs)

    

    # Detect
    detect_way = [0,1,2]
    ber_ways = []
    for d in detect_way:
        detected_bits = dss.detect_dss(watermarked, fs, BPS, FRAME_SIZE, detect_way=d)
        print("detected WM:", detected_bits.shape)
        # print(detected_bits)
        # ber = dss.compute_ber(wm_bps[:len(detected_bps)], detected_bps)
        ber = dss.compute_ber(embed_bits, detected_bits)
        ber_ways.append({
            'detect_way': d,
            'ber': ber
        })
        print(ber_ways)
        # import IPython
        # IPython.embed()


    # Compute PESQ (wideband)
    pesq_score_wb = dss.compute_pesq(host, watermarked, fs, 'wb')
    pypesq_score = dss.compute_pypesq(host, watermarked, fs)

    # list to dict
    ber_dict = {entry['detect_way']: entry['ber'] for entry in ber_ways}

    # Store results
    results.append({
        'file': fname,
        # 'duration': duration,
        'ber_fft': ber_dict.get(0,None),
        'ber_simple': ber_dict.get(1,None),
        'ber_corr': ber_dict.get(2,None),
        'pesq_wb': pesq_score_wb,
        'pypesq': pypesq_score,
        # 'bps':BPS,
        # 'ssl_db':SSL_DB
    })

    
print(results)

# Print summary
print("BPS:", BPS)
print("SSL DB:", SSL_DB)
print(f"{'File':30} {'BER_FFT':>8} {'BER_SimpleCorr':>12} {'BER_PearsonCorr':>10} {'PESQ':>6} {'PYPESQ':>6}")
for r in results:
    print(f"{r['file']:30} {r['ber_fft']:8.3f} {r['ber_simple']:12.3f} {r['ber_corr']:10.3f} {r['pesq_wb']:6.2f} {r['pypesq']:6.2f}")

# ## Export to CSV
# import csv

# csv_path = f"dss_classic_{BPS}bps_{SSL_DB}db.csv"
# with open(csv_path, mode="w", newline='', encoding="utf-8") as f:
#     writer = csv.DictWriter(f, fieldnames=["file", "ber_fft", "ber_simple", "ber_corr", "pesq_wb", "pypesq", "bps", "ssl_db"])
#     writer.writeheader()
#     for r in results:
#         writer.writerow(r)

# print(f"\nResults exported to: {csv_path}")