"""
stereo_pad.py -- per-voice stereo placement for the arranger's five-voice stack.

WHY THIS FILE EXISTS AND WHY IT IS NOT AN EDIT TO THE ARRANGER.
`Phase2/the arranger` is FROZEN, and not merely by convention: its
sha256 is recorded in `arc_pool_meta.json` as the enforcement artifact, so an
edit -- even one that leaves the mono path bit-identical -- trips the freeze
check and puts every rated stimulus in question.

`s03_longtrack_render.py` already established the legitimate route for exactly
this shape of problem (it needed per-boundary crossfades the frozen renderer
could not express): RE-DERIVE the loop outside the frozen file, calling only
that file's own primitives, and PROVE equivalence. Same thing here. Everything
that makes sound -- the DDSP synth instance, `apply_spectral_tilt`, `_ou_drift`,
`convolve_reverb`, `WAYPOINT_VOICES`, every constant -- is imported, never
reimplemented. The only new code is where the six voices land in the field.

THE CHANGE IS ONE LINE, CONCEPTUALLY. `_render_static_pad` already renders each
voice as its own buffer and sums them:

    mixed[:m] += raw[:m] * g_audio[:m]

so per-voice access is not something that has to be built -- it is already
there, one accumulator. Stereo is two accumulators and a pan gain.

DESIGN DECISIONS, and the reasons, since none of these are rated:

  * CONSTANT-POWER panning, not HRTF/binaural. 2026-08-01, feedback: "we want to
    create directionality rather than messing with people's heads" -- which is
    precisely the distinction. HRTF externalises a source but is headphone-only
    and colours badly over speakers; the listening test plays back on unknown
    hardware. Amplitude panning reads as direction on both.
  * BASS STAYS CENTRED. `sub_bass` and `pad_root` are pinned to the middle in
    every preset. Panning low frequencies is the classic way to make a mix
    collapse in mono and wander on small speakers, and this synth's energy is
    heavily bass-weighted (the sub-bass voice is the one that had to be cut to
    0.25x for the "vibrating" bed).
  * MONO COMPATIBILITY IS A CONSTRAINT, not a nice-to-have. Every preset here
    is amplitude-only: summing L+R can attenuate a voice but can never comb
    filter it, because no channel is ever a DELAYED copy of the other. That is
    the whole reason the Haas/chorus option was not taken.
  * The reverb IR may be shared (centred reverb, conservative) or decorrelated
    per channel (wide). Decorrelated is the more natural-sounding one and is
    still delay-free in the comb-filtering sense, but it is a bigger departure
    from what every rated clip has heard, so it is a switch and not a default.

NOTHING HERE IS RATED. Pan positions, LFO depth and period, and width are
authored numbers, exactly like the envelope constants that produced the
"attack not finished" complaint. They are for auditioning; if stereo ships they
become a rated factor or a declared convention, not a silent default.

Usage:
  python stereo_pad.py --verify     # replication proof, no audio written
  python make_demos.py              # renders the audition set
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# The arranger lives in the conductor's engine/ directory, one level up.
ENGINE = HERE.parent / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import arranger as arr                                         # noqa: E402
from arranger import (SR, N_NOISE_BANDS, WAYPOINT_VOICES,        # noqa: E402
                      apply_spectral_tilt, convolve_reverb, _ou_drift)

CTRL_HZ = getattr(arr, "CTRL_HZ", 50)


# --------------------------------------------------------------- pan presets
# pan is in [-1, +1]: -1 hard left, 0 centre, +1 hard right.
#
# `lfo` is the DEPTH of that voice's slow drift around its base position, in the
# same units. Phases are spread so the voices do not swing as one rigid block --
# a field that rotates as a unit reads as the room turning, which is the
# "messing with heads" failure mode; independent slow drifts read as players
# shifting slightly, which is the directionality asked for.
PAN_PRESETS = {
    # the control: bit-identical to the frozen mono render, in both channels
    "mono":   {"sub_bass": 0.0, "pad_root": 0.0, "pad_min3": 0.0,
               "pad_maj3": 0.0, "pad_fifth": 0.0, "air": 0.0},
    # conservative: chord voices just off-centre, bass pinned
    "narrow": {"sub_bass": 0.0, "pad_root": 0.0, "pad_min3": -0.30,
               "pad_maj3": +0.30, "pad_fifth": +0.50, "air": -0.50},
    # the one to judge "directionality" on
    "wide":   {"sub_bass": 0.0, "pad_root": 0.0, "pad_min3": -0.60,
               "pad_maj3": +0.60, "pad_fifth": +0.85, "air": -0.85},
}

# LFO depth per voice for the drifting preset (base positions = "wide").
# Bass depth is 0 on purpose -- see the bass-stays-centred note above.
LFO_DEPTH = {"sub_bass": 0.0, "pad_root": 0.0, "pad_min3": 0.25,
             "pad_maj3": 0.25, "pad_fifth": 0.30, "air": 0.30}
LFO_PHASE = {"sub_bass": 0.0, "pad_root": 0.0, "pad_min3": 0.0,
             "pad_maj3": 2.09, "pad_fifth": 4.19, "air": 1.05}
LFO_PERIOD_S = 45.0        # slow: a drift you notice over a phrase, not a wobble


def pan_gains(pan):
    """Constant-power pan law. Centre gives 0.7071 per side, so the SUM keeps
    the same power as the mono signal rather than the same amplitude -- which is
    what stops a centred voice jumping in level the moment stereo is enabled."""
    theta = (np.clip(pan, -1.0, 1.0) + 1.0) * (np.pi / 4.0)
    return np.cos(theta), np.sin(theta)


def voice_pan_curve(vc_name, n_samples, preset, lfo=False, t0=0.0):
    """Per-sample pan position for one voice. Constant unless `lfo` is on."""
    base = PAN_PRESETS[preset][vc_name] if preset in PAN_PRESETS else 0.0
    if not lfo or LFO_DEPTH.get(vc_name, 0.0) <= 0.0:
        return np.full(n_samples, base)
    t = t0 + np.arange(n_samples) / float(SR)
    return np.clip(base + LFO_DEPTH[vc_name]
                   * np.sin(2 * np.pi * t / LFO_PERIOD_S + LFO_PHASE[vc_name]),
                   -1.0, 1.0)


# ------------------------------------------------------- stereo static pad
def render_static_pad_stereo(renderer, wp, n_ctrl, breath_period_s,
                             pitch_drift_cents, drift_tau_s, param_wander_std,
                             seed, ctrl_hz=CTRL_HZ, preset="wide", lfo=False,
                             t0=0.0):
    """Line-for-line replication of `ArrangedRenderer._render_static_pad`,
    summing into TWO accumulators instead of one.

    Every number, every RNG draw and every call order is preserved -- that is
    what `--verify` checks, and it is the only reason this is safe to use for
    rated stimuli. If the frozen file is ever changed, this WILL drift silently
    and the verification is what catches it, so run it after any arranger touch.
    """
    import torch

    duration_s = n_ctrl / ctrl_hz
    num_samples = int(duration_s * SR)
    t_ctrl = np.linspace(0, duration_s, n_ctrl)
    t_audio = np.linspace(0, duration_s, num_samples)

    rng = np.random.default_rng(seed)
    base = {k: float(wp.get(k, 0.0)) for k in
            ["f0_hz", "swell_rate", "swell_depth", "noise_level",
             "noise_cutoff_hz", "third_interval", "third_gain",
             "fifth_gain", "octave_gain"]}
    base["swell_depth"] = min(base["swell_depth"], arr.DDSP_SWELL_CAP)
    base["swell_rate"] = min(base["swell_rate"], 0.15)
    base["noise_level"] = min(base["noise_level"], 0.08)

    sc = {k: np.full(n_ctrl, v) for k, v in base.items()}
    if param_wander_std > 0:
        for k in ["swell_rate", "swell_depth", "noise_level"]:
            wander = _ou_drift(n_ctrl, 1.0 / ctrl_hz, drift_tau_s, rng)
            sc[k] = sc[k] * np.exp(param_wander_std * wander)
        f0_wander = _ou_drift(n_ctrl, 1.0 / ctrl_hz, drift_tau_s, rng)
        sc["f0_hz"] = sc["f0_hz"] * np.exp(
            min(param_wander_std, 0.05) * 0.1 * f0_wander)
    rng = np.random.default_rng(seed)          # frozen file re-seeds here
    if pitch_drift_cents > 0:
        drift = _ou_drift(n_ctrl, 1.0 / ctrl_hz, drift_tau_s, rng)
        f0_factor = 2.0 ** (pitch_drift_cents / 1200.0 * drift)
    else:
        f0_factor = np.ones(n_ctrl)

    def ctrl_to_audio(a):
        return np.interp(t_audio, t_ctrl, a)

    f0_base_audio = np.exp(np.interp(
        t_audio, t_ctrl, np.log(np.maximum(sc["f0_hz"], 1.0))))
    f0_factor_audio = np.exp(np.interp(
        t_audio, t_ctrl, np.log(np.maximum(f0_factor, 1e-6))))
    rate_audio = ctrl_to_audio(sc["swell_rate"])
    depth_audio = ctrl_to_audio(sc["swell_depth"])
    n_lvl_audio = ctrl_to_audio(sc["noise_level"])
    n_cut_audio = ctrl_to_audio(sc["noise_cutoff_hz"])

    dt = 1.0 / SR
    swell_phase = 2.0 * np.pi * np.cumsum(rate_audio * dt)
    shimmer_env = (1.0 - depth_audio * 0.5 * (1.0 + np.sin(swell_phase))) * 0.8

    harm = np.asarray(wp["harm_dist"], dtype=float)
    harm = harm / (harm.sum() + 1e-8)

    n_frames = num_samples // (N_NOISE_BANDS - 1)
    t_frames = np.linspace(0, duration_s, n_frames)
    n_lvl_frames = np.interp(t_frames, t_audio, n_lvl_audio)
    n_cut_frames = np.interp(t_frames, t_audio, n_cut_audio)
    band_freqs = np.linspace(0, SR / 2, N_NOISE_BANDS)

    t_env = np.linspace(0, duration_s, n_ctrl)
    maj_frac = float(np.clip(base["third_interval"] - 3.0, 0.0, 1.0))
    static_gain = {"pad_min3": base["third_gain"] * (1.0 - maj_frac),
                   "pad_maj3": base["third_gain"] * maj_frac,
                   "pad_fifth": base["fifth_gain"],
                   "air": base["octave_gain"]}

    left = np.zeros(num_samples)
    right = np.zeros(num_samples)
    for vc in WAYPOINT_VOICES:
        g_static = static_gain.get(vc["name"], 1.0)
        if g_static <= 1e-4:
            continue
        role_harm = apply_spectral_tilt(harm, vc["bias"])
        detune_mult = 2.0 ** (vc["detune"] / 1200.0)
        voice_f0 = f0_base_audio * vc["mult"] * detune_mult * f0_factor_audio
        voice_amp = shimmer_env * g_static

        f0_t = torch.tensor(voice_f0, dtype=torch.float32,
                            device=renderer.device).view(1, -1, 1)
        amp_t = torch.tensor(voice_amp, dtype=torch.float32,
                             device=renderer.device).view(1, -1, 1)
        hd_t = torch.tensor(np.tile(role_harm, (num_samples, 1)),
                            dtype=torch.float32,
                            device=renderer.device).unsqueeze(0)

        nl_f = n_lvl_frames * vc["noise_scale"] * g_static
        freqs_g, cuts_g = np.meshgrid(band_freqs, n_cut_frames, indexing="ij")
        fmags = (1.0 / (1.0 + (freqs_g / (cuts_g + 1e-8)) ** 4)) \
            * nl_f[np.newaxis, :]
        noise_t = torch.tensor(fmags.T, dtype=torch.float32,
                               device=renderer.device).view(
                                   1, n_frames, N_NOISE_BANDS)
        with torch.no_grad():
            raw = renderer.synth(f0_t, amp_t, hd_t, noise_t)\
                .squeeze(0).cpu().numpy()

        cos_curve = np.cos(vc["breath_phase"]
                           + 2.0 * np.pi * t_env / breath_period_s)
        genv = vc["gain"] * (1.0 - vc["breath_depth"] * 0.5 * (1.0 + cos_curve))
        g_audio = np.interp(t_audio, t_env, genv)
        m = min(len(raw), num_samples)

        # THE ONLY DEPARTURE from the frozen loop: one accumulator becomes two.
        pan = voice_pan_curve(vc["name"], m, preset, lfo=lfo, t0=t0)
        gl, gr = pan_gains(pan)
        voiced = raw[:m] * g_audio[:m]
        left[:m] += voiced * gl
        right[:m] += voiced * gr
    return left, right


# --------------------------------------------------------- stereo waypoints
def render_waypoints_stereo(renderer, theta_waypoints, hold_s=1.0, xfade_s=0.75,
                            breath_period_s=10.0, pitch_drift_cents=2.0,
                            drift_tau_s=12.0, param_wander_std=0.05, seed=0,
                            preset="wide", lfo=False, reverb_mode="mono",
                            width=1.0):
    """Stereo twin of `_render_waypoints_layered`, same overlap-add timeline.

    `t0` is threaded per waypoint so the LFO runs on ABSOLUTE track time -- a
    per-pad phase reset would step the image at every waypoint boundary, which
    is audible as a click in the stereo field even when the mono sum is smooth.
    """
    n_wp = len(theta_waypoints)
    hold_n = max(1, int(round(hold_s * CTRL_HZ)))
    xfade_n = max(1, int(round(xfade_s * CTRL_HZ)))
    n_ctrl = n_wp * (hold_n + xfade_n)
    num_samples = int(n_ctrl / CTRL_HZ * SR)
    spc = SR / CTRL_HZ

    ml = np.zeros(num_samples)
    mr = np.zeros(num_samples)
    for i, wp in enumerate(theta_waypoints):
        fade_in = xfade_n if i > 0 else 0
        seg_ctrl = fade_in + hold_n + xfade_n
        start = int((i * (hold_n + xfade_n) - fade_in) * spc)
        rl, rr = render_static_pad_stereo(
            renderer, wp, seg_ctrl, breath_period_s, pitch_drift_cents,
            drift_tau_s, param_wander_std, seed + i, CTRL_HZ,
            preset=preset, lfo=lfo, t0=max(0, start) / float(SR))
        env = np.ones(len(rl))
        fi = int(fade_in * spc)
        fo = int((xfade_n if i < n_wp - 1 else 0) * spc)
        if fi:
            env[:fi] = np.sin(np.linspace(0, np.pi / 2, fi))
        if fo:
            env[-fo:] = np.cos(np.linspace(0, np.pi / 2, fo))
        end = min(start + len(rl), num_samples)
        ml[start:end] += (rl * env)[: end - start]
        mr[start:end] += (rr * env)[: end - start]

    ml, mr = stereo_reverb(ml, mr, mode=reverb_mode)
    if width != 1.0:
        ml, mr = ms_width(ml, mr, width)
    # JOINT peak normalisation. Per-channel would scale the sides differently
    # and shift the image whenever one side happened to peak higher -- the
    # stereo equivalent of the conditional peak guards that cause the
    # conductor's segment loudness jumps.
    peak = max(np.max(np.abs(ml)), np.max(np.abs(mr))) + 1e-8
    return ml / peak * 0.88, mr / peak * 0.88


def stereo_reverb(l, r, mode="mono"):
    """`mono`: one shared IR, so the reverb sits dead centre -- conservative,
    and the closest thing to what every rated clip has heard.
    `wide`: a decorrelated IR per channel, which is what makes a reverb sound
    like a room rather than a centred effect. Still delay-free in the
    comb-filtering sense, since neither channel is a shifted copy of the other.
    """
    if mode == "mono":
        return convolve_reverb(l), convolve_reverb(r)
    # decorrelate by reseeding the IR generator the frozen helper fixes at 1234
    saved = np.random.default_rng
    out = []
    for ch, s in ((l, 1234), (r, 8765)):
        np.random.default_rng = lambda _s=None, _fixed=s: saved(_fixed)
        try:
            out.append(convolve_reverb(ch))
        finally:
            np.random.default_rng = saved
    return out[0], out[1]


def autopan(mono, depth=0.7, period_s=40.0, t0=0.0):
    """Pan the WHOLE mix slowly, bass included.

    The per-voice presets deliberately pin the bass, which at the rated melody
    anchor left only 7.6% of the energy free to move. This moves 100% of it, so
    it is the one option that is unmistakably audible on any anchor -- at the
    cost of the thing the pinning was protecting against (a wandering low end,
    and a weaker mono fold-down). Included because it should be judged by ear
    rather than ruled out on principle; `period_s` is deliberately long, since
    the dizziness risk lives in the RATE, not the depth."""
    x = np.asarray(mono, dtype=np.float64)
    t = t0 + np.arange(len(x)) / float(SR)
    pan = depth * np.sin(2 * np.pi * t / period_s)
    gl, gr = pan_gains(pan)
    return x * gl, x * gr


def ms_width(l, r, width=1.0):
    """Mid/side width. 1.0 = unchanged, >1 wider, 0 = mono."""
    mid = 0.5 * (l + r)
    side = 0.5 * (l - r) * float(width)
    return mid + side, mid - side


# ------------------------------------------------------------------ verify
def verify(device="cpu"):
    """Prove the replication, because everything else here rests on it.

    HARD-LEFT is the clean test: at pan = -1 the constant-power law is exactly
    (1.0, 0.0), so the left channel must reproduce the frozen mono pad
    BIT-FOR-BIT. Centre pan is then only that result times cos(pi/4), which is
    arithmetic and does not need its own proof.

    TORCH MUST BE SEEDED FIRST, and this is not a formality. The frozen renderer
    is NOT deterministic call-to-call: `ddsp_synth.py`'s filtered-noise source
    draws from torch's GLOBAL RNG, which nothing seeds. Measured here, the
    frozen pad differs from ITSELF by 6.8e-3 across two identical calls -- the
    same order as a genuine replication bug, so without the seed this test
    reports failure whether or not anything is wrong, and (worse) could report
    success by luck. `s03_longtrack_render.py` pins the seed for the same
    reason. Any future bit-identity claim in this project has to do likewise.
    """
    import torch
    from step02_arranger import ArrangedRenderer
    rng = np.random.default_rng(0)
    harm = np.abs(rng.normal(size=32)); harm /= harm.sum()
    wp = dict(f0_hz=110.0, swell_rate=0.05, swell_depth=0.05,
              noise_level=0.02, noise_cutoff_hz=2000.0, third_interval=3.0,
              third_gain=0.5, fifth_gain=0.5, octave_gain=0.35, harm_dist=harm)
    r = ArrangedRenderer(device=device)
    kw = dict(breath_period_s=10.0, pitch_drift_cents=2.0, drift_tau_s=12.0,
              param_wander_std=0.05, seed=7, ctrl_hz=CTRL_HZ)

    # sanity: show the nondeterminism the seed is protecting against
    a = r._render_static_pad(wp, 120, **kw)
    b = r._render_static_pad(wp, 120, **kw)
    print(f"frozen vs itself, UNSEEDED     : max|diff| = "
          f"{float(np.max(np.abs(a - b))):.3e}  (why the seed below matters)")

    torch.manual_seed(0)
    mono = r._render_static_pad(wp, 120, **kw)
    PAN_PRESETS["_hardleft"] = {v["name"]: -1.0 for v in WAYPOINT_VOICES}
    torch.manual_seed(0)
    L, R = render_static_pad_stereo(r, wp, 120, preset="_hardleft", **kw)
    d = float(np.max(np.abs(L - mono)))
    print(f"pad, hard-left vs frozen mono : max|diff| = {d:.3e}  "
          f"({'BIT-IDENTICAL' if d == 0.0 else 'DRIFT'})")
    print(f"pad, right channel silent      : max|R| = {float(np.max(np.abs(R))):.3e}")

    # centred must be perfectly symmetric, and carry mono's POWER not amplitude
    torch.manual_seed(0)
    Lc, Rc = render_static_pad_stereo(r, wp, 120, preset="mono", **kw)
    sym = float(np.max(np.abs(Lc - Rc)))
    ratio = float(np.sqrt(np.mean((Lc + Rc) ** 2)) / (np.sqrt(np.mean(mono ** 2)) + 1e-20))
    print(f"pad, centre L vs R             : max|L-R| = {sym:.3e}")
    print(f"pad, centre (L+R) power vs mono: ratio = {ratio:.6f}  (want sqrt(2)={np.sqrt(2):.6f})")

    # hard-left must be EXACT (it is the replication proof); centre symmetry
    # only has to be at float precision, since cos(pi/4) and sin(pi/4) are not
    # the same double even though the maths says they are.
    ok = (d == 0.0) and (sym < 1e-12) and abs(ratio - np.sqrt(2)) < 1e-6
    print("\nREPLICATION", "OK -- safe for rated stimuli" if ok else "FAILED -- do not use")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.verify:
        raise SystemExit(0 if verify(a.device) else 1)
    print(__doc__)
