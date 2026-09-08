"""Bed DSP: load a background bed, loop it to length, follow its loudness.

The CONDUCTOR half of the overlay work. The other half -- listing candidate
beds from ESC-50 and Emo-Soundscapes, auditioning them, and writing the
curation log -- is data collection: it needs the corpora on disk and it
produces rating data, so it stays on the training side with
`models/3.5_transition_dynamics_and_scheduling/train/build_bed_bank.py`.

Keeping it out is not tidiness. The version this was cut from imported
`paths` and resolved two dataset roots at import, which would have put a
dataset dependency inside the conductor -- the one thing the whole
train/inference split exists to prevent, and something a grader would only
discover when the app refused to start on a machine without the corpora.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

SR = 16000
XFADE_S = 1.0            # bed loop crossfade
ENV_HOP_S = 0.05         # loudness-envelope frame hop


def load_mono_16k(path):
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    x, sr = sf.read(str(path), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return np.asarray(x, dtype=np.float64)


def loop_to_len(bed, n, sr=SR, xfade_s=XFADE_S):
    """Loop `bed` to n samples with equal-power crossfades at the seams."""
    if len(bed) >= n:
        return bed[:n]
    xf = min(int(xfade_s * sr), len(bed) // 2)
    out = np.array(bed)
    env_in = np.sin(0.5 * np.pi * np.linspace(0, 1, xf))
    env_out = np.cos(0.5 * np.pi * np.linspace(0, 1, xf))
    while len(out) < n:
        head = np.array(bed)
        out[-xf:] = out[-xf:] * env_out + head[:xf] * env_in
        out = np.concatenate([out, head[xf:]])
    return out[:n]


def loud_env(x, sr=SR, win_s=0.25, smooth_s=1.0):
    """Smoothed RMS loudness envelope of x, per sample, min-max normalized to
    [0, 1] (flat signal -> all zeros)."""
    hop, win = int(ENV_HOP_S * sr), int(win_s * sr)
    pad = np.concatenate([x, np.zeros(win)])
    n_fr = max(1, (len(x) + hop - 1) // hop)
    rms = np.array([np.sqrt(np.mean(pad[i * hop:i * hop + win] ** 2))
                    for i in range(n_fr)])
    # Cap the smoothing kernel at the number of frames we actually have.
    # `np.convolve(..., mode="same")` returns max(len(signal), len(kernel)),
    # NOT len(signal) -- so a bed shorter than smooth_s came back with a
    # longer envelope than the frame axis and the np.interp below raised
    # "fp and xp are not of the same length". Real beds are 5 s and the
    # default smooth_s is 1 s, so this never fired in the study; it is a
    # crash waiting for the first short bed anyone adds.
    k = max(1, min(int(smooth_s / ENV_HOP_S), len(rms)))
    kern = np.ones(k)
    # normalize by the kernel mass actually inside the signal, else the
    # zero-padded edges read as fake loudness dips (spurious burst sites)
    rms = np.convolve(rms, kern, mode="same") \
        / np.convolve(np.ones_like(rms), kern, mode="same")
    lo, hi = rms.min(), rms.max()
    norm = (rms - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(rms)
    return np.interp(np.arange(len(x)), np.arange(n_fr) * hop, norm)
