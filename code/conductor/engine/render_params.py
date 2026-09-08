"""The frozen render settings every voiced path in the system shares.

These constants are a PROTOCOL, not preferences. The arc pool, the long-track
renders, the rated stimuli and the live conductor all render through the same
values, which is the only reason a preference learned on a rated clip means
anything when the conductor voices it. Change one and every level comparison
against the existing ratings is quietly wrong -- the ratings do not become
noisy, they become measurements of a different instrument.

They were extracted here from the arc-pool builder (`s07_arc_pool.py`), which
is training-side code the conductor should not import. The dates and the
reasoning are kept because they are the evidence for the values; the people
and tools that argued for them are not.

`aw_rms` and `AW_TARGET` live in `common/loudness.py` rather than here, since
the melody runtime needs them too and two definitions of a loudness protocol
would be two different loudnesses.
"""
from __future__ import annotations

# ---------------------------------------------------------------- chord axis

# A minor third is 3 semitones, a major third 4. The conductor moves along
# this axis to change chord quality without re-retrieving a preset.
THIRD_INTERVAL = {"min": 3.0, "maj": 4.0}

# Retrieved presets often carry third_gain near zero, which would make the
# chord axis inaudible -- the manipulation would be real in the parameters and
# absent in the audio. The floor guarantees the axis can be heard.
THIRD_GAIN_FLOOR = 0.35

CHORD_PATHS = [(s, e) for s in ("min", "maj") for e in ("min", "maj")]


def apply_chord(theta, quality):
    """Set a preset's chord quality, raising the third into audibility.

    Returns a copy: the caller's theta is a retrieved row and callers reuse
    it, so mutating in place would make chord choice order-dependent.
    """
    theta = dict(theta)
    theta["third_interval"] = THIRD_INTERVAL[quality]
    theta["third_gain"] = max(float(theta.get("third_gain", 0.0)),
                              THIRD_GAIN_FLOOR)
    return theta


# ------------------------------------------------------------------- timing

# Approved 2026-07-04. The corpus's own held-state durations have a median of
# 9 s, so a hold wants to be long; 5 s is the most the stimulus-length by
# listener-fatigue budget allows. It is a compromise against the corpus, and
# the corpus is the thing being approximated.
HOLD_S = 5.0

# Approved 2026-07-05 from the stratified six-artist transition typology (946
# state transitions, k=5). The knots sit at the measured type-median clusters:
# 3 s (quick shifts 2.75 / warm swells 3.25), 6 s (releases 5.6 / post-hold
# moves 6.5), 12 s (slow section morphs 11), plus 16 s as a tail anchor so the
# preference GP interpolates rather than extrapolates at the slow end.
#
# Supersedes three earlier sets, and why each failed is the argument for this
# one: {4, 8} predates the corpus measurement entirely; {6, 12, 16} predates
# the cleanup and leans slow, when 67% of state transitions actually finish
# within 6 s; {3, 5, 8} leaves the 11 s section-morph type unsampled.
RAMP_CHOICES = (3.0, 6.0, 12.0, 16.0)


# ------------------------------------------------------------ renderer state

# The renderer's frozen defaults.
#
# `param_wander_std=0.15`. An earlier revert to 0.05 argued that the breathing
# micro-events were noise-floor flatlines. The post-floor measurement
# contradicts it: 541 micro-events survive the absolute perceptual floors
# (>=1.5 dB loudness, >=10% centroid), median magnitude ~1.0 -- one of the six
# corpus artists averages 3.5 audible micro-events per held minute. A listener
# also reported hearing no wandering at 0.05. The 0.30 probe below keeps a 2x
# contrast so ratings can still reject the raise rather than being asked to
# confirm it.
#
# `xfade_mode="layered"` (2026-07-05, from a smoke listen). Each waypoint
# renders as its own static pad with an equal-power gain crossfade between
# pads. This kills the root-f0 glide -- audible note motion during a
# transition, which in a drone reads as a mistake. `"interp"` is the legacy
# path and is kept only so old renders reproduce.
RENDER_KW = dict(breath_period_s=10.0, pitch_drift_cents=2.0,
                 drift_tau_s=12.0, param_wander_std=0.15, seed=0,
                 xfade_mode="layered")

# Single-knob contrasts the factorial design cannot express, rendered
# explicitly so each knob can be rated against its own default.
RENDER_PROBES = [dict(param_wander_std=0.30), dict(pitch_drift_cents=8.0)]

# Ceiling applied after loudness normalisation. Normalising to a fixed
# A-weighted RMS can push peaks past full scale on spectrally narrow material,
# so the gain is reduced until the peak fits and the reduction is reported
# rather than hidden.
PEAK_GUARD = 0.9
