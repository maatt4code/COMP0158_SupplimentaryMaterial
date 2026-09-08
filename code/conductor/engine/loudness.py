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
from scipy.signal import bilinear, lfilter

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


def _a_weighting_ba(sr):
    """Bilinear-transformed IEC 61672 A-weighting filter coefficients."""
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    A1000 = 1.9997
    nums = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
    dens = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                      [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
    dens = np.polymul(np.polymul(dens, [1, 2 * np.pi * f3]),
                      [1, 2 * np.pi * f2])
    return bilinear(nums, dens, sr)


def aw_rms(x, sr=SR):
    """A-weighted RMS of a signal: the level a listener perceives.

    Filters in the TIME domain, over the WHOLE signal. Both matter, and this
    is the definition every rated stimulus was normalised with -- so it is
    not an implementation detail that may be swapped for an equivalent.

    An FFT version that weights a spectrum and returns a Parseval amplitude
    looks equivalent and is not. Measured against this one:

      * a 1 kHz tone agrees to 0.04 dB -- which is why such a substitution
        survives casual testing;
      * a 30 Hz tone reads **6.1 dB louder**, and since this value is used as
        a GAIN DENOMINATOR that is a 6 dB level error on exactly the dark
        drone material this system produces;
      * a signal that is silent for its first seconds reads as **zero**, if
        the FFT version truncates to a leading window -- a zero denominator
        on any segment with a slow entry, and on the masked melody audio in
        Section 3.7's level matching.

    Whole-signal, time-domain, one definition. See the module docstring.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    b, a = _a_weighting_ba(sr)
    return float(np.sqrt(np.mean(lfilter(b, a, x) ** 2)))
