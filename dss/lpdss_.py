# -*- coding: utf-8 -*-
"""Linear‑Prediction Direct Spread‐Spectrum (LP‑DSS) baseline implementation
==========================================================================
This script provides *both* the embedder and the detector for a classical
frame‑based LP‑DSS audio watermark.  It follows the original Cox et al. idea
but derives the pseudo‑noise (PN) carrier in each frame from its linear‑
prediction (LP) residual so that no explicit key is needed.

The code is deliberately kept self‑contained and dependency‑light; only
``numpy`` and ``scipy`` are required.

API
---
``embed_lpdss(host_signal, fs, wm_bits, bitrate_bps, ssl_db, frame_size=160,
              lpc_order=16)``
    Embed the watermark into *host_signal*.

``detect_lpdss(wm_signal, fs, bitrate_bps, frame_size=160, detect_mode=1,
               lpc_order=16)``
    Recover the embedded bits from *wm_signal*.

Utilities ``compute_ber``, ``compute_pesq`` (if the optional *pesq* or
*pypesq* package is installed) are also included.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz, toeplitz
from scipy.signal import lfilter
from lpdss import detect_fft

# ---------------------------------------------------------------------------
#  Core DSP helpers
# ---------------------------------------------------------------------------

# check whether we use this or the 
def levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
    """Return LPC coefficients *a[0]=1, a[1:]* (positive sign convention).

    Parameters
    ----------
    r : ndarray, shape (order+1,)
        Autocorrelation sequence (lag 0 … lag *order*).
    order : int
        Prediction order.
    """
    # Solve Ra = −r[1:]
    # Toeplitz is symmetric, positive‑definite for speech so solve_toeplitz is OK
    a_rest = solve_toeplitz((r[:-1], r[:-1]), -r[1:])
    return np.concatenate(([1.0], a_rest))

# ---------------------------------------------------------------------------
#  Embedding
# ---------------------------------------------------------------------------

def embed_lpdss(
    host_signal: np.ndarray,
    fs: int,
    wm_bits: np.ndarray,
    bitrate_bps: float,
    ssl_db: float,
    *,
    frame_size: int = 160,
    lpc_order: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed *wm_bits* into *host_signal* using LP‑DSS.

    Parameters
    ----------
    host_signal : ndarray, shape (N,)
        PCM float32/float64 audio in range ±1.
    fs : int
        Sampling rate [Hz].
    wm_bits : ndarray of 0/1
        Watermark payload.  If shorter than needed it repeats cyclically.
    bitrate_bps : float
        Net bit rate [bit/s].
    ssl_db : float
        Spread‑spectrum level in dB *below* host RMS (negative number).
    frame_size : int, default 160
        Samples per analysis frame (10 ms at 16 kHz).
    lpc_order : int, default 16
        LPC analysis order.

    Returns
    -------
    y : ndarray
        Watermarked signal, same length as *host_signal*.
    used_bits : ndarray
        The exact bitstring actually embedded (after repetition).
    """
    x = host_signal.astype(np.float64).flatten()
    N = x.size

    # --- Prepare bitstream --------------------------------------------------
    duration_s = N / fs
    print("duration:", duration_s)
    total_bits = int(np.ceil(duration_s * bitrate_bps))
    print("total bits:", total_bits)

    print("WM bits:", wm_bits)
    wm_bits = wm_bits.astype(int).flatten() & 1
    print("WM bits after &1:", wm_bits)

    if wm_bits.size == 0:
        raise ValueError("wm_bits must contain at least one bit.")

    if wm_bits.size < total_bits:
        reps = int(np.ceil(total_bits / wm_bits.size))
        wm_bits = np.tile(wm_bits, reps)
    used_bits = wm_bits[:total_bits] #whole bits across host signal
    print("used bits:", used_bits)
    used_bits_bipolar = 2 * used_bits - 1  # 0→−1, 1→+1
    print("used bits bipolar:", used_bits_bipolar)
    print(used_bits_bipolar.shape)

    # Frame/bit mapping
    frames_per_sec = fs / frame_size #10
    frames_per_bit = int(np.ceil(frames_per_sec / bitrate_bps)) #10 / 64 = 1
    print("frames per bit:", frames_per_bit)

    # Signal framing (no overlap)
    num_frames = N // frame_size
    print("num_frames:", num_frames)
    tail = x[num_frames * frame_size :]
    frames = x[: num_frames * frame_size].reshape(num_frames, frame_size).T  # (F, K)

    # Expand bitstream to frame length
    bit_per_frame = np.repeat(used_bits_bipolar, frames_per_bit)[:num_frames]
    print("bit per frame:", bit_per_frame)
    print(bit_per_frame.shape)

    # --- Generate PN carrier per frame -------------------------------------
    eps = np.finfo(float).eps
    carriers = np.empty_like(frames)
    for k in range(num_frames):
        frame = frames[:, k]
        # autocorr lags 0…p
        r = np.correlate(frame, frame, mode="full")
        mid = r.size // 2
        r = r[mid : mid + lpc_order + 1]
        a = levinson_durbin(r, lpc_order)
        # residual e[n] = x[n] − Σ a_i x[n‑i]  (note the minus sign!)
        residual = lfilter([1.0] + list(-a[1:]), 1.0, frame)
        residual_zm   = residual - residual.mean()          # zero-mean
        rms           = np.sqrt(np.mean(residual_zm**2) + 1e-12)
        pn            = residual_zm / rms                   # unit-power but real-valued
        carriers[:,k] = pn
        # pn = np.sign(residual)
        # pn[pn == 0] = 1
        # carriers[:, k] = pn

    # --- Embed --------------------------------------------------------------
    y_frames = np.empty_like(frames)
    for k in range(num_frames):
        frame = frames[:, k]
        pn = carriers[:, k]
        rms_frame = np.sqrt(np.mean(frame**2) + eps) #host
        embed_amp = rms_frame * 10 ** (ssl_db / 20.0)
        y_frames[:, k] = frame + embed_amp * pn * bit_per_frame[k]

    y = np.concatenate((y_frames.T.flatten(), tail))

    # Peak normalise to ±1 if needed
    max_val = np.max(np.abs(y))
    if max_val > 1.0:
        y /= max_val

    return y.astype(host_signal.dtype), used_bits

# ---------------------------------------------------------------------------
#  Detection
# ---------------------------------------------------------------------------

def detect_lpdss(
    wm_signal: np.ndarray,
    fs: int,
    bitrate_bps: float,
    *,
    frame_size: int = 160,
    lpc_order: int = 16,
    detect_mode: int = 1,
) -> np.ndarray:
    """Detect watermark bits from *wm_signal*.

    Parameters and return are analogous to *embed_lpdss*.
    """
    y = wm_signal.astype(np.float64).flatten()
    N = y.size

    num_frames = N // frame_size
    frames = y[: num_frames * frame_size].reshape(num_frames, frame_size).T

    # --- Regenerate PN carriers -------------------------------------------
    carriers = np.empty_like(frames)
    for k in range(num_frames):
        frame = frames[:, k]
        r = np.correlate(frame, frame, mode="full")
        mid = r.size // 2
        r = r[mid : mid + lpc_order + 1]
        a = levinson_durbin(r, lpc_order)
        residual = lfilter([1.0] + list(-a[1:]), 1.0, frame)
        residual_zm   = residual - residual.mean()          # zero-mean
        rms           = np.sqrt(np.mean(residual_zm**2) + 1e-12)
        pn            = residual_zm / rms                   # unit-power but real-valued
        carriers[:,k] = pn
        # pn = np.sign(residual)
        # pn[pn == 0] = 1
        # carriers[:, k] = pn

    # --- Per‑frame decision statistic -------------------------------------
    if detect_mode == 1:  # simple dot‑product
        corr = np.sum(frames * carriers, axis=0)
        bipolar_frames = np.sign(corr)
        bipolar_frames[bipolar_frames == 0] = 1

    elif detect_mode == 2:  # Pearson correlation
        f_zm = frames - frames.mean(axis=0, keepdims=True)
        c_zm = carriers - carriers.mean(axis=0, keepdims=True)
        denom = np.sqrt(np.sum(f_zm**2, axis=0) * np.sum(c_zm**2, axis=0) + 1e-12)
        corr = np.sum(f_zm * c_zm, axis=0) / denom
        bipolar_frames = np.sign(corr)
        bipolar_frames[bipolar_frames == 0] = 1

    else: #detect_mode == 0; detect_fft
        bipolar_frames = detect_fft(frames, carriers)

    # --- Majority vote over frames → bits ----------------------------------
    frames_per_sec = fs / frame_size
    frames_per_bit = int(np.ceil(frames_per_sec / bitrate_bps))
    total_bits = int(np.ceil((N / fs) * bitrate_bps))

    detected_bits = np.empty(total_bits, dtype=int)
    for k in range(total_bits):
        start = k * frames_per_bit
        end = min((k + 1) * frames_per_bit, num_frames)
        group = bipolar_frames[start:end]

        if (detect_mode == 0):
            bit = 1 if np.sum(group) >= 1 else 0
        else:
            bit = 1 if np.sum(group) >= 0 else 0
        
        detected_bits[k] = bit

    return detected_bits

# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------

def compute_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """Bit‑Error Rate."""
    tx = tx_bits.flatten()
    rx = rx_bits.flatten()
    if tx.size != rx.size:
        raise ValueError("Bitvectors must have same length for BER.")
    return np.mean(tx != rx)

# Optional PESQ wrappers -----------------------------------------------------

def _import_pesq():
    try:
        from pesq import pesq as pesq_score  # type: ignore
        return pesq_score
    except ImportError:  # pragma: no cover
        return None

def compute_pesq(ref: np.ndarray, deg: np.ndarray, fs: int, mode: str = "wb") -> float:
    pesq_score = _import_pesq()
    if pesq_score is None:
        raise ImportError("pip install pesq or pypesq to use compute_pesq()")
    return float(pesq_score(fs, ref.flatten(), deg.flatten(), mode))

# pypesq

# ---------------------------------------------------------------------------
#  Tiny smoke‑test -----------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import soundfile as sf

    # Dummy 3‑second white‑noise ‘speech’ for a quick self‑test
    # fs = 16000
    path = "/home/adila/Data/research/dss-dnn/dataset/host/19-198-0000.wav"
    x, fs = sf.read(path)
    # x = np.random.randn(fs * 3).astype(np.float32) * 0.05

    payload = np.random.randint(0, 2, 64)
    y, used = embed_lpdss(x, fs, payload, bitrate_bps=8.0, ssl_db=-25)

    rx = detect_lpdss(y, fs, bitrate_bps=8.0)
    ber = compute_ber(used, rx[: used.size])
    print(f"BER = {ber:.4f}  (should be < 0.01 at –25 dB for random noise host)")

    # Listen / save for manual inspection (if desired)
    # sf.write("host.wav", x, fs)
    sf.write("wm.wav",   y, fs)
