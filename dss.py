import numpy as np

def generate_bitpattern(bps):
    """
    Generate a random bipolar bit pattern of length `bps`.
    If bps == 4 and the pattern isn’t balanced (equal +1 and –1),
    pick one at random from the six balanced 4-bit patterns.
    """
    # random uniform in {0, +1}
    bitpattern = np.random.randint(0, 2, size=bps)

    # if bps==4, enforce balance
    if bps == 4:
        if np.sum(bitpattern == 0) != np.sum(bitpattern == 1):
            # all 6 possible balanced patterns of length 4
            bitpattern_list = np.array([
                [0, 0,  1,  1],
                [0,  1, 0,  1],
                [0,  1,  1, 0],
                [ 1, 0,  1, 0],
                [ 1, 0, 0,  1],
                [ 1,  1, 0, 0],
            ])
            idx = np.random.randint(0, len(bitpattern_list))
            bitpattern = bitpattern_list[idx]

    print(bitpattern)
    return bitpattern


def embed_dss(host_signal, fs, wm_bps, ssl_db, frame_size):
    """
    Embed a binary watermark into a host signal using frame-based Direct Spread Spectrum
    with per-frame adaptive amplitude based on host and carrier power levels.

    Parameters
    ----------
    host_signal : array_like, shape (N,)
        Input audio samples, normalized between -1 and 1.
    fs : int
        Sampling rate in Hz.
    wm_bps : array_like, shape (K,)
        Binary watermark bits (0 or 1).
    bps : float
        Bits per second to embed.
    ssl_db : float
        Embedding strength level in dB (analogous to MATLAB's ssl).
    frame_size : int
        Number of samples per frame (e.g., 160).

    Returns
    -------
    watermarked_signal : ndarray, shape (N,)
        Watermarked audio signal.
    """

    host = np.asarray(host_signal).flatten()
    N = host.size
    duration = N / fs
    bps = wm_bps.shape[0]
    total_bits = int(np.ceil(bps * duration))  # overall bits to be embedded

    # Prepare watermark bits: repeat if shorter
    wm = np.asarray(wm_bps, dtype=int).flatten()
    if wm.size < total_bits:
        reps = int(np.ceil(total_bits / wm.size))
        wm = np.tile(wm, reps)
    wm = wm[:total_bits]
    bipolar = 2 * wm - 1  # {0,1} -> {-1,+1}  

    # Frame and bit calculations
    num_frames = N // frame_size
    frames_per_sec = fs / frame_size
    frames_per_bit = int(np.ceil(frames_per_sec / bps))

    # Expand bits to frames
    # bit_stream = np.repeat(bipolar, frames_per_bit)
    # bit_s = np.repeat(wm, frames_per_bit)[:num_frames]
    bit_stream = np.repeat(bipolar, frames_per_bit)[:num_frames]

    print("Total bits:", total_bits)
    # print("WM:", wm)
    # print("Bit stream:", bit_stream)
    # print(bit_stream)

    # Segment host into frames
    X = host[:num_frames * frame_size].reshape(num_frames, frame_size).T
    tail = host[num_frames * frame_size:]

    # Generate PN codes
    np.random.seed(0)
    PN = np.sign(np.random.randn(frame_size, num_frames))

    # embed with adaptive per-frame amplitude
    eps = np.finfo(float).eps
    X_emb = np.zeros_like(X)

    # import IPython
    # IPython.embed()

    for k in range(num_frames):
        frame = X[:, k]
        # host power (dB)
        slev = 10 * np.log10(np.mean(frame**2) + eps)
        # carrier (PN) power (dB)
        clev = 10 * np.log10(np.mean(PN[:, k]**2) + eps)
        # compute embedding level (dB)
        lev = slev - clev + ssl_db
        # convert to linear amplitude
        amp = 10**(lev / 20)
        # watermark for this frame
        wl = PN[:, k] * amp * bit_stream[k]
        # embed
        X_emb[:, k] = frame + wl

    # reconstruct and normalize
    y = np.concatenate((X_emb.T.flatten(), tail))
    max_val = np.max(np.abs(y))
    if max_val > 1:
        y = y / max_val

    return y, wm



def detect_dss(watermarked_signal, fs, bps, frame_size, detect_way=None):
    """
    Detect watermark bits from a frame-based DSS-watermarked signal.

    Parameters
    ----------
    watermarked_signal : array_like, shape (N,)
        Input watermarked audio samples.
    fs : int
        Sampling rate in Hz.
    bps : float
        Bits per second used during embedding.
    frame_size : int
        Number of samples per frame (e.g., 160).

    Returns
    -------
    detected_bits : ndarray, shape (L,)
        Recovered binary watermark bits (0 or 1).
    """
    # Ensure 1D numpy array
    y = np.asarray(watermarked_signal).flatten()
    N = y.shape[0]

    # Frame calculations
    num_frames = N // frame_size
    X = y[:frame_size*num_frames].reshape(num_frames, frame_size).T

    # Regenerate PN codes
    np.random.seed(0)
    PN = np.sign(np.random.randn(frame_size, num_frames))

    if detect_way == 1:
        print("detect way 1")
        # [opt 1] Correlate per frame: simple dot product
        corr_vals = np.sum(X * PN, axis=0)
        bipolar_frames = np.sign(corr_vals)
        bipolar_frames[bipolar_frames == 0] = 1

    elif detect_way == 2:
        print("detect way 2")
        # [opt 2] Correlate per frame: Pearson correlation
        X_mean = X.mean(axis=0)
        PN_mean = PN.mean(axis=0)
        X_zm = X - X_mean
        PN_zm = PN - PN_mean
        num = np.sum(X_zm * PN_zm, axis=0)
        denom = np.sqrt(np.sum(X_zm**2, axis=0) * np.sum(PN_zm**2, axis=0))
        denom[denom == 0] = np.finfo(float).eps
        corr_vals = num / denom

        bipolar_frames = np.sign(corr_vals)
        bipolar_frames[bipolar_frames == 0] = 1
    
    else:
        print("detect way 0")
        bipolar_frames = detect_dss_fft(X,PN)
    
    # print("bipolar frames")
    # print(bipolar_frames)
    
    # Group frames into bits
    frames_per_sec = fs / frame_size
    frames_per_bit = int(np.ceil(frames_per_sec / bps))
    total_bits    = int(np.ceil(bps * (N / fs)))

    # print(num_frames)
    # print(frames_per_bit)

    detected = []
    for k in range(total_bits):
        start = k * frames_per_bit
        end   = min((k+1) * frames_per_bit, num_frames)
        group = bipolar_frames[start:end]
        # print("detection result")
        # print(group)
        # Majority vote
        if (detect_way == 0):
            bit = 1 if np.sum(group) >= 1 else 0
        else:
            bit = 1 if np.sum(group) >= 0 else 0
        # print(bit)
        detected.append(bit)

    # print("detected WM:", detected)

    return np.array(detected, dtype=int)

def detect_dss_fft(watermarked_signal, PN):
    """
    Detect watermark bits from a DSS-watermarked signal using FFT-based decision per frame.

    Inputs:
      watermarked_signal : array_like
      e                  : ndarray (frame_size x num_frames), carrier matrix from embed

    Returns:
      detected_bits : ndarray of 0/1 bits per frame
    """
    X_wm = watermarked_signal
    num_frames = PN.shape[1]

    detected = []
    for k in range(num_frames):
        wm = X_wm[:, k] * PN[:, k]
        c = np.fft.fft(wm)
        a_k = np.round(c[0], 12)
        bit = 1 if a_k > 0 else 0
        detected.append(bit)

    # print(detected.shape[0])

    return np.array(detected, dtype=int)

def compute_ber(original_bits, detected_bits):
    """
    Compute Bit Error Rate (BER) between original and detected bit sequences.

    Returns:
      ber : float, fraction of bits in error
    """
    orig = np.asarray(original_bits).flatten()
    det  = np.asarray(detected_bits).flatten()
    if orig.size != det.size:
        raise ValueError("Original and detected bit arrays must have the same length.")
    errors = np.sum(orig != det)
    return errors / orig.size


def compute_pesq(ref_signal, deg_signal, fs, mode='wb'):
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality) score.

    Requires the `pesq` package (pip install pesq).

    Inputs:
      ref_signal : array_like, reference clean signal
      deg_signal : array_like, degraded (watermarked) signal
      fs         : int, sampling frequency
      mode       : str, 'wb' or 'nb' for wide/narrow band

    Returns:
      score : float, PESQ MOS-LQO score
    """
    try:
        from pesq import pesq as pesq_score
    except ImportError:
        raise ImportError("Please install the 'pesq' package: pip install pesq")
    
    return pesq_score(fs, ref_signal.flatten(), deg_signal.flatten(), mode)

def compute_pypesq(ref_signal, deg_signal, fs):
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality) score.

    Requires the `pesq` package (pip install pesq).

    Inputs:
      ref_signal : array_like, reference clean signal
      deg_signal : array_like, degraded (watermarked) signal
      fs         : int, sampling frequency
      mode       : str, 'wb' or 'nb' for wide/narrow band

    Returns:
      score : float, PESQ MOS-LQO score
    """
    # import soundfile as sf

    try:
        from pypesq import pesq as pesq_score
    except ImportError:
        raise ImportError("Please install the 'pypesq' package: pip install https://github.com/vBaiCai/python-pesq/archive/master.zip")
    
    return pesq_score(ref_signal.flatten(), deg_signal.flatten(), fs)
