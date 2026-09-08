"""A-weighted loudness: one definition, shared by everything that levels audio.

Perceived loudness, not raw level. The ear is far more sensitive around 2-4 kHz
than at the extremes, so two signals at the same RMS can differ by many decibels
in how loud they sound. A melody moved up an octave at constant RMS reads
noticeably louder; a dark reverb tail reads quieter. Weighting the spectrum by
the A curve before taking the level is what lets ONE gain setting hold across
register and timbre.

This lives in `common/` because it is a PROTOCOL, not a component. Section 3.7's
melody runtime and the conductor's arranger both normalise against it, and the
same target value appears in the rated stimuli. Two definitions that drifted
apart would silently mean two different loudnesses, and every level comparison
made against them would be wrong in a way no test would catch.

`AW_TARGET` is frozen. It is the value the rated clips were normalised to, so
changing it does not merely rescale new audio -- it breaks comparability with
every rating already collected.
"""

from __future__ import annotations

import numpy as np

SR = 16000
AW_TARGET = 0.015          # FROZEN: the level the rated stimuli were set to


def a_weighting_db(f):
    """A-weighting sensitivity in dB at frequency ``f`` (IEC 61672).

    Accepts a scalar or an array. Below 10 Hz the curve is clamped, because the
    analytic form dives toward negative infinity and contributes nothing a
    listener could hear anyway.
    """
    f = np.maximum(np.asarray(f, dtype=np.float64), 10.0)
    f2 = f * f
    c1, c2, c3, c4 = 12194.217 ** 2, 20.598997 ** 2, 107.65265 ** 2, 737.86223 ** 2
    num = c1 * f2 * f2
    den = (f2 + c2) * np.sqrt((f2 + c3) * (f2 + c4)) * (f2 + c1)
    return 20.0 * np.log10(num / den) + 2.0


def aw_rms(x, sr=SR, n_fft=2048):
    """A-weighted RMS of a signal: the level a listener perceives.

    Computed in the frequency domain and returned to a time-domain amplitude by
    Parseval, so it is directly comparable with a plain RMS and can be used as
    a gain denominator.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    seg = x[:int(min(len(x), n_fft * 8))]
    X = np.abs(np.fft.rfft(seg))
    f = np.fft.rfftfreq(len(seg), 1.0 / sr)
    w = 10.0 ** (a_weighting_db(f) / 20.0)
    return float(np.sqrt(np.sum((X * w) ** 2) / (len(seg) ** 2 / 2.0 + 1e-12)))


def normalize_aw(x, sr=SR, target=AW_TARGET, peak_guard=0.9):
    """Scale to the frozen A-weighted target, without clipping.

    The peak guard is applied AFTER the loudness match, so a signal that would
    clip comes back down and is then quieter than the target. That is the right
    trade for a single clip, but it does mean the match no longer holds -- see
    Section 3.6's `_match_level`, where exactly this costs one condition several
    decibels. Callers comparing ACROSS conditions should apply one common
    headroom scalar instead.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    gain = target / (aw_rms(x, sr) + 1e-12)
    peak = np.max(np.abs(x * gain))
    if peak > peak_guard:
        gain *= peak_guard / peak
    return (x * gain).astype(np.float32)
