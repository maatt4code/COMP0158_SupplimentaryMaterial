"""
distance.py -- move the drone TOWARD and AWAY from the listener.

WHY THIS RATHER THAN PANNING. The per-voice panning spike sounded, in the brief's
words, "barely noticeable", and measurement says that is structural, not a
tuning failure. At the rated melody anchor the energy splits:

    sub_bass  60.5%   (centred -- panning bass collapses a mix)
    pad_root  31.9%   (centred)
    air        7.6%   <- the only voice that actually moved
    min3/maj3/fifth 0.00%  (gains below threshold, voices skipped entirely)

So panning moved 7.6% of the signal, and the remaining 92% sits at 63-126 Hz
where human azimuth localisation is poor regardless. Width is the wrong axis for
this instrument.

Distance is not. Its cues act on the WHOLE signal, at any frequency:

  1. DIRECT-TO-REVERBERANT RATIO -- the dominant one. A distant source is not
     merely quieter, it is proportionally WETTER. This is why loudness alone
     reads as someone turning a volume knob rather than as something receding:
     listeners discount absolute level, but they cannot discount D/R.
  2. AIR ABSORPTION -- distance rolls off highs.
  3. LOUDNESS -- real, but only convincing when 1 and 2 move with it.
  4. INITIAL TIME DELAY GAP -- the space between the direct sound and its first
     reflection. LARGE when you are close to a source in a room, SMALL when far.
     Note the sign: pre-delay SHRINKS as distance grows. Getting this backwards
     is the usual way a "distance" effect ends up sounding like a plain reverb
     send.

The brief's professor described it as "the same sounds slightly out of sync with
proportionate loudness". The out-of-sync part is cue 4, plus a small left/right
offset on the REFLECTIONS ONLY -- never on the direct signal, which would comb
filter the moment anyone folds it to mono.

MOTION IS OCCASIONAL BY DESIGN. feedback: "obviously don't do it constantly coz
that will make people dizzy". `distance_curve` therefore spends most of its time
STATIC and moves rarely, on a slow raised-cosine ramp -- the shape matters, since
a linear ramp has a velocity discontinuity at each end that reads as a lurch.

INTERACTION WITH THE LOUDNESS PROTOCOL -- read before wiring this anywhere.
Every rated stimulus is A-weighted-normalised, and the conductor levels each
segment. That normalisation would REMOVE cue 3 and partially cue 1. Distance has
to be applied AFTER levelling, or the level pass has to be told to leave it
alone; otherwise the effect is silently flattened, which is precisely how the
long-track judge traces went null.

NOTHING HERE IS RATED -- authored numbers for auditioning.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.signal as sps

HERE = Path(__file__).resolve().parent
# The arranger lives in the conductor's engine/ directory, one level up.
ENGINE = HERE.parent / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from arranger import SR                                        # noqa: E402

# Cue ranges, near (d=0) -> far (d=1).
NEAR_FAR = dict(
    gain_db=(0.0, -11.0),        # cue 3: level
    wet=(0.12, 0.72),            # cue 1: D/R ratio -- the load-bearing one
    lp_hz=(7200.0, 900.0),       # cue 2: air absorption
    predelay_ms=(45.0, 6.0),     # cue 4: ITDG, SHRINKS with distance
)
# left/right offset on the reflections only, so the room has a side to it
# without the direct signal ever being duplicated
REFLECT_LR_OFFSET_MS = 7.0


def _lerp(pair, d):
    return pair[0] + (pair[1] - pair[0]) * float(np.clip(d, 0.0, 1.0))


def _dark_ir(tail_s=6.0, cutoff_hz=600.0, seed=1234):
    """The project's reverb IR, same construction as the frozen
    `convolve_reverb` (fixed seed, exponential decay, dark low-pass) -- rebuilt
    here only because we need it as a separate WET signal to control the D/R
    ratio, which the frozen helper mixes internally and does not expose."""
    n = int(SR * tail_s)
    ir = np.random.default_rng(seed).normal(0, 1, n) * np.exp(-np.linspace(0, 6, n))
    sos = sps.butter(2, cutoff_hz / (SR / 2), btype="low", output="sos")
    ir = sps.sosfilt(sos, ir)
    return ir / (np.sqrt(np.sum(ir ** 2)) + 1e-8)


def _delay(x, ms):
    n = int(round(SR * ms / 1000.0))
    return x if n <= 0 else np.concatenate([np.zeros(n), x])[:len(x)]


def place(audio, distance=0.0, ir_l=None, ir_r=None):
    """Render `audio` (mono) as if it sat at `distance` in [0, 1].

    Returns (left, right). Static distance -- see `move` for a travelling one.
    """
    x = np.asarray(audio, dtype=np.float64)
    d = float(np.clip(distance, 0.0, 1.0))
    g = 10.0 ** (_lerp(NEAR_FAR["gain_db"], d) / 20.0)
    wet = _lerp(NEAR_FAR["wet"], d)
    lp = _lerp(NEAR_FAR["lp_hz"], d)
    pre = _lerp(NEAR_FAR["predelay_ms"], d)

    # cue 2: air absorption on the DIRECT path
    sos = sps.butter(2, min(lp, SR / 2 * 0.99) / (SR / 2), btype="low", output="sos")
    direct = sps.sosfilt(sos, x) * g * (1.0 - wet)

    ir_l = _dark_ir(seed=1234) if ir_l is None else ir_l
    ir_r = _dark_ir(seed=8765) if ir_r is None else ir_r
    wl = sps.fftconvolve(x, ir_l, mode="full")[:len(x)]
    wr = sps.fftconvolve(x, ir_r, mode="full")[:len(x)]
    # match wet energy to dry so `wet` is a genuine RATIO knob and not a
    # loudness knob wearing a hat
    for i, w in enumerate((wl, wr)):
        w *= (np.sqrt(np.mean(x ** 2)) + 1e-9) / (np.sqrt(np.mean(w ** 2)) + 1e-9)
    # cue 4: pre-delay, plus the small L/R offset. REFLECTIONS ONLY.
    wl = _delay(wl, pre) * wet * g
    wr = _delay(wr, pre + REFLECT_LR_OFFSET_MS) * wet * g
    return direct + wl, direct + wr


def place_stereo(l, r, distance=0.0, narrow_far=True):
    """Distance applied to an ALREADY-PANNED stereo pair, so width and depth
    combine instead of competing (2026-08-01, feedback: "why not have both?!").

    They are orthogonal cues -- panning is azimuth, distance is depth -- but
    they are not independent in the real world, and the coupling is worth
    keeping: a source that recedes subtends a SMALLER ANGLE, so the image should
    narrow as it goes away. Without that, a far source that is still fully wide
    reads as "quiet and wet" rather than "further off". `narrow_far` folds the
    frozen arranger's own mid/side maths in to do it.
    """
    from stereo_pad import ms_width
    l = np.asarray(l, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    d = float(np.clip(distance, 0.0, 1.0))
    if narrow_far:
        # 1.0 up close, 0.45 at the far end -- narrowed, never collapsed to
        # mono, since a hard collapse is audible as the image "snapping"
        l, r = ms_width(l, r, 1.0 - 0.55 * d)
    # each side keeps its own direct path (that is the width), and gets its own
    # decorrelated reflections (that is the room)
    ll, lr_ = place(l, d, ir_l=_dark_ir(seed=1234), ir_r=_dark_ir(seed=4321))
    rl, rr = place(r, d, ir_l=_dark_ir(seed=8765), ir_r=_dark_ir(seed=5678))
    return 0.5 * (ll + rl), 0.5 * (lr_ + rr)


def distance_curve(n_samples, moves=((0.35, 0.9, 0.35, 12.0),), start=0.15):
    """A mostly-STATIC distance track with occasional slow travels.

    Each move is (start_frac, end_distance, ..., seconds) -- expressed as a
    fraction of the clip so a curve can be reused at any length. The ramp is a
    raised cosine: a linear ramp arrives and departs with a velocity step, which
    is heard as a lurch at each end, and lurching is exactly the dizziness
    the brief warned against."""
    d = np.full(n_samples, float(start))
    cur = float(start)
    for (t_frac, target, _hold, secs) in moves:
        i0 = int(np.clip(t_frac, 0, 1) * n_samples)
        i1 = min(n_samples, i0 + int(secs * SR))
        if i1 <= i0:
            continue
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, i1 - i0)))
        d[i0:i1] = cur + (target - cur) * ramp
        d[i1:] = target
        cur = target
    return d


def move(audio, curve, n_levels=9):
    """Travelling distance, blended SAMPLE-WISE across a pre-rendered ladder.

    Convolving with a time-varying IR is where clicks normally come from, so the
    distance is rendered at a ladder of fixed values and interpolated between.
    The interpolation is per SAMPLE, which matters -- an earlier version blended
    in 64 blocks and had two defects the brief heard immediately:

      * CLICKS. The blocks did not overlap, but each was Hann-windowed and
        divided by the window sum. Inside a block that is w/w = 1, but at the
        two ends w -> 0 and the divide-by-zero guard passed the near-silent
        value straight through -- a notch to zero at all 64 joins.
      * STEPPED LOUDNESS. Each block used one constant distance (the block
        mean), so the level moved in 64 stairs instead of a ramp.

    Blending per sample removes the blocks entirely, so neither failure has
    anywhere to live.

    The crossfade is LINEAR, not equal-power. Equal-power (cos/sin) is correct
    for UNCORRELATED signals; these two renders share the same dry source and
    are highly correlated, so an equal-power blend would bulge about +3 dB at
    the midpoint -- which would itself read as a loudness step.
    """
    x = np.asarray(audio, dtype=np.float64)
    curve = np.clip(np.asarray(curve, dtype=np.float64), 0.0, 1.0)
    if len(curve) != len(x):
        curve = np.interp(np.linspace(0, 1, len(x)),
                          np.linspace(0, 1, len(curve)), curve)
    rendered = [place(x, lv) for lv in np.linspace(0.0, 1.0, n_levels)]
    L = np.stack([p[0] for p in rendered])
    R = np.stack([p[1] for p in rendered])

    pos = curve * (n_levels - 1)
    k = np.clip(np.floor(pos).astype(int), 0, n_levels - 2)
    f = pos - k
    i = np.arange(len(x))
    out_l = L[k, i] * (1.0 - f) + L[k + 1, i] * f
    out_r = R[k, i] * (1.0 - f) + R[k + 1, i] * f
    return out_l, out_r


def dr_ratio_db(l, r, x):
    """Direct-to-reverberant estimate, for checking the cue is really moving:
    correlation of the output with the dry signal explains the direct part, the
    residual is reverberant."""
    y = 0.5 * (np.asarray(l) + np.asarray(r))
    x = np.asarray(x, dtype=np.float64)[:len(y)]
    a = float(np.dot(y, x) / (np.dot(x, x) + 1e-12))
    direct = a * x
    resid = y - direct
    return 10.0 * np.log10((np.mean(direct ** 2) + 1e-20)
                           / (np.mean(resid ** 2) + 1e-20))
