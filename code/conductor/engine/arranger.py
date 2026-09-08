"""The arranged renderer: continuous parameters in, ambient audio out.

This is the conductor's voice. Every other module decides WHAT to play -- a
retrieved preset, a chord, an arc between two points -- and this turns that
decision into sound, through a fixed stack of detuned DDSP voices.

It renders:
1. Role Stack: Sub-bass, Pad root, Pad fifth, and Air voices with static spectral tilts.
2. Detuning: Symmetrical cents offsets to build a rich chorused texture.
3. Continuous Swell integration: Integrates swell_rate over time via cumsum
   to prevent phase tearing during frequency modulation.
4. Dark Reverb: Master convolution reverb matching the project's production chain.

The render settings it is called with are frozen in `render_params.py`, and
that file explains why they are not tunable.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from scipy import signal as sps

# The conductor is self-contained: every module it needs sits beside it, so
# this resolves within `engine/` rather than reaching up into the repo.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ddsp_synth import DifferentiableDDSPSynth                    # noqa: E402

SR = 16000
N_HARM = 32
N_NOISE_BANDS = 65

# Ratios and configuration
FIFTH = 2.0 ** (7.0 / 12.0)
OCTAVE = 2.0
SUBOCT = 0.5

# Symmetrical detuning offsets (in cents).
# ±4 cents was too much: sub_bass harmonics overlap pad_root/pad_fifth harmonics,
# creating 0.2–0.4 Hz beats (audible tremolo). Reduced to ±1 cent.
DETUNES = {
    "sub_bass": 0.0,
    "pad_root": -1.0,
    "pad_fifth": +1.0,
    "air": -0.5
}

ROLES = [
    {"name": "sub_bass",  "mult": SUBOCT, "gain": 0.70, "bias": +0.10, "detune": DETUNES["sub_bass"],  "noise_scale": 0.0},
    {"name": "pad_root",  "mult": 1.0,    "gain": 0.80, "bias": +0.03, "detune": DETUNES["pad_root"],  "noise_scale": 1.0},
    {"name": "pad_fifth", "mult": FIFTH,  "gain": 0.50, "bias": 0.00,  "detune": DETUNES["pad_fifth"], "noise_scale": 1.0},
    {"name": "air",       "mult": OCTAVE, "gain": 0.35, "bias": -0.06, "detune": DETUNES["air"],       "noise_scale": 1.0},
]


def apply_spectral_tilt(harm, bias):
    """Adjusts harmonic weights to make them darker (bias > 0) or brighter (bias < 0)."""
    n = np.arange(1, len(harm) + 1, dtype=np.float64)
    # Apply tilt factor n^(-bias)
    harm_tilted = harm * (n ** (-bias * 2.0))
    return harm_tilted / (harm_tilted.sum() + 1e-12)


def convolve_reverb(audio, tail_sec=6.0, wet=0.35):
    """Energy-normalized dark convolution reverb matching the project baseline."""
    audio = np.asarray(audio, dtype=np.float64)
    ir_len = int(SR * tail_sec)
    decay = np.exp(-np.linspace(0, 6, ir_len))
    
    # Fixed seed for consistent space acoustics
    rng_ir = np.random.default_rng(1234)
    ir = rng_ir.normal(0, 1, ir_len) * decay
    
    # Low-pass filter the IR to make it warm/dark (cutoff = 600 Hz)
    sos = sps.butter(2, 600.0 / (SR / 2), btype="low", output="sos")
    ir = sps.sosfilt(sos, ir)
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-8
    
    rev = sps.fftconvolve(audio, ir, mode="full")[:len(audio)]
    
    dry_rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
    wet_rms = np.sqrt(np.mean(rev ** 2)) + 1e-8
    rev *= dry_rms / wet_rms
    
    out = audio * (1 - wet) + rev * wet
    peak = np.max(np.abs(out)) + 1e-8
    if peak > 0.95:
        out *= 0.95 / peak
    return out


# Voice roles for the waypoint renderer — 6 voices including dynamic chord thirds.
# We explicitly separate minor and major thirds to allow GAIN CROSSFADES between them
# (tension->release arcs) rather than pitch-gliding the third, which creates
# microtonal smearing (see resolution_experiment_v2).
WAYPOINT_VOICES = [
    {"name": "sub_bass",  "mult": SUBOCT, "gain": 0.70, "bias": +0.10,
     "detune": 0.0,  "noise_scale": 0.0, "breath_depth": 0.12, "breath_phase": 0.0},
    {"name": "pad_root",  "mult": 1.0,    "gain": 0.80, "bias": +0.03,
     "detune": -1.0, "noise_scale": 1.0, "breath_depth": 0.55, "breath_phase": 0.0},                          
    {"name": "pad_min3",  "mult": 2.0**(3.0/12.0), "gain": 0.60, "bias": +0.01,
     "detune": +0.5, "noise_scale": 0.0, "breath_depth": 0.55, "breath_phase": 1.047}, # π/3
    {"name": "pad_maj3",  "mult": 2.0**(4.0/12.0), "gain": 0.60, "bias": +0.01,
     "detune": -0.5, "noise_scale": 0.0, "breath_depth": 0.55, "breath_phase": 1.047}, # π/3
    {"name": "pad_fifth", "mult": FIFTH,  "gain": 0.50, "bias": 0.00,
     "detune": +1.0, "noise_scale": 0.0, "breath_depth": 0.55, "breath_phase": 2.094}, # 2π/3
    {"name": "air",       "mult": OCTAVE, "gain": 0.35, "bias": -0.06,
     "detune": -0.5, "noise_scale": 0.0, "breath_depth": 0.55, "breath_phase": 4.189}, # 4π/3
]

# Swell cap on the DDSP path: keeps it a surface shimmer.
# Macro breathing comes from the per-voice gain envelope instead (v3 lesson).
DDSP_SWELL_CAP = 0.10


def _ou_drift(n, dt, tau_s, rng):
    """Zero-mean OU walk, unit stationary variance (from conductor_common)."""
    a = np.exp(-dt / max(tau_s, 1e-6))
    s = np.sqrt(max(1.0 - a * a, 0.0))
    x = np.empty(n, dtype=np.float64)
    x[0] = rng.standard_normal()
    for i in range(1, n):
        x[i] = a * x[i - 1] + s * rng.standard_normal()
    return x


class ArrangedRenderer:
    def __init__(self, device="cpu"):
        self.device = device
        self.synth = DifferentiableDDSPSynth(sample_rate=SR, num_harmonics=N_HARM, num_filter_bands=N_NOISE_BANDS).to(device)
        self.synth.eval()

    def render_trajectory(self, theta_series, duration_s, hold_timbre=False):
        """
        Renders a continuous audio segment from a time-series of theta dictionaries.
        
        Args:
            theta_series: List of theta dicts, sampled uniformly (e.g. at 20 Hz control rate).
            duration_s: Total duration in seconds.
        """
        num_samples = int(duration_s * SR)
        control_steps = len(theta_series)
        
        # 1. Unpack and interpolate base parameters over time
        f0_base = np.array([t["f0_hz"] for t in theta_series])
        swell_rate = np.array([t["swell_rate"] for t in theta_series])
        swell_depth = np.array([t["swell_depth"] for t in theta_series])
        noise_level = np.array([t["noise_level"] for t in theta_series])
        noise_cutoff = np.array([t["noise_cutoff_hz"] for t in theta_series])
        
        # Unpack chord parameters (default to inactive if not in bank)
        third_interval = np.array([t.get("third_interval", 4.0) for t in theta_series])
        third_gain = np.array([t.get("third_gain", 0.0) for t in theta_series])
        fifth_gain = np.array([t.get("fifth_gain", 0.0) for t in theta_series])
        octave_gain = np.array([t.get("octave_gain", 0.0) for t in theta_series])
        
        # Interp functions to map control frames to audio samples
        x_ctrl = np.linspace(0, duration_s, control_steps)
        x_audio = np.linspace(0, duration_s, num_samples)
        
        f0_audio = np.exp(np.interp(x_audio, x_ctrl, np.log(f0_base)))
        rate_audio = np.interp(x_audio, x_ctrl, swell_rate)
        depth_audio = np.interp(x_audio, x_ctrl, swell_depth)
        n_lvl_audio = np.interp(x_audio, x_ctrl, noise_level)
        # Cap the noise floor to prevent low-frequency wind rumble
        n_lvl_audio = np.minimum(n_lvl_audio, 0.08)
        n_cut_audio = np.interp(x_audio, x_ctrl, noise_cutoff)
        
        # Interpolate chord features
        third_int_audio = np.interp(x_audio, x_ctrl, third_interval)
        third_g_audio = np.interp(x_audio, x_ctrl, third_gain)
        fifth_g_audio = np.interp(x_audio, x_ctrl, fifth_gain)
        octave_g_audio = np.interp(x_audio, x_ctrl, octave_gain)
        
        # Integrate swell rate over time to compute continuous phase (prevents phase tearing)
        dt = 1.0 / SR
        swell_phase = 2.0 * np.pi * np.cumsum(rate_audio * dt)
        amplitude_envelope = 1.0 - depth_audio * 0.5 * (1.0 + np.sin(swell_phase))
        # Scale amplitude envelope to reasonable peak levels
        amplitude_envelope = amplitude_envelope * 0.8
        
        # Harmonic distribution: either sample-and-hold (one timbre per control frame,
        # no cross-frame blending) or linear interpolation between frames.
        harm_stack = np.stack([t["harm_dist"] for t in theta_series], axis=0)  # (ctrl_steps, N_HARM)
        if hold_timbre:
            # Each audio sample inherits the timbre of its nearest control frame (no smearing)
            ctrl_indices = np.clip(
                np.floor(x_audio / duration_s * control_steps).astype(int),
                0, control_steps - 1
            )
            harm_audio = harm_stack[ctrl_indices]  # (num_samples, N_HARM)
        else:
            harm_audio = np.zeros((num_samples, N_HARM))
            for h_idx in range(N_HARM):
                harm_audio[:, h_idx] = np.interp(x_audio, x_ctrl, harm_stack[:, h_idx])
        # Renormalize each sample's harmonics
        harm_audio = harm_audio / (np.sum(harm_audio, axis=1, keepdims=True) + 1e-8)
        
        # Define dynamic chord voices based on the theta configuration
        # sub_bass (anchor) and pad_root are always on. Third, fifth, and air voices
        # scale dynamically in the mix. Noise rides on the root voice only (noise_scale=1.0).
        dynamic_voices = [
            {"name": "sub_bass",  "mult": SUBOCT, "detune": DETUNES["sub_bass"],  "gain": 0.70, "bias": +0.10, "noise_scale": 0.0, "active_gain": np.ones(num_samples)},
            {"name": "pad_root",  "mult": 1.0,    "detune": DETUNES["pad_root"],  "gain": 0.80, "bias": +0.03, "noise_scale": 1.0, "active_gain": np.ones(num_samples)},
            {"name": "pad_third", "mult": 2.0 ** (third_int_audio / 12.0), "detune": +0.5, "gain": 0.60, "bias": +0.01, "noise_scale": 0.0, "active_gain": third_g_audio},
            {"name": "pad_fifth", "mult": 1.5,    "detune": DETUNES["pad_fifth"], "gain": 0.50, "bias": 0.00,  "noise_scale": 0.0, "active_gain": fifth_g_audio},
            {"name": "air",       "mult": OCTAVE, "detune": DETUNES["air"],       "gain": 0.35, "bias": -0.06, "noise_scale": 0.0, "active_gain": octave_g_audio},
        ]
        
        # 2. Render each dynamic voice
        mixed_audio = np.zeros(num_samples)
        
        for role in dynamic_voices:
            # Apply spectral tilt bias to the harmonic distribution
            role_harm = np.stack([apply_spectral_tilt(h, role["bias"]) for h in harm_audio], axis=0)
            
            # Apply register multiplier and symmetrical detuning (mult can be time-varying array now)
            detune_mult = 2.0 ** (role["detune"] / 1200.0)
            voice_f0 = f0_audio * role["mult"] * detune_mult
            
            # Construct time-varying tensors. Internal amplitude envelope is scaled by active gain
            # to make sure noise and harmonics remain proportional inside the synthesizer.
            voice_amp = amplitude_envelope * role["active_gain"]
            
            f0_t = torch.tensor(voice_f0, dtype=torch.float32, device=self.device).view(1, -1, 1)
            amp_t = torch.tensor(voice_amp, dtype=torch.float32, device=self.device).view(1, -1, 1)
            hd_t = torch.tensor(role_harm, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Filtered noise component (noise rides on voices with scale > 0)
            n_frames = num_samples // (N_NOISE_BANDS - 1)
            x_noise = np.linspace(0, duration_s, n_frames)
            
            # Compute time-varying noise filter response (scaled by active gain and noise scale)
            active_g_noise = np.interp(x_noise, x_audio, role["active_gain"])
            n_lvl_noise = np.interp(x_noise, x_audio, n_lvl_audio) * role["noise_scale"] * active_g_noise
            n_cut_noise = np.interp(x_noise, x_audio, n_cut_audio)
            
            band_freqs = np.linspace(0, SR / 2, N_NOISE_BANDS)
            freqs_grid, cutoffs_grid = np.meshgrid(band_freqs, n_cut_noise, indexing='ij')
            
            # Butter-like filter shape: 1 / (1 + (f / cutoff)^4)
            filter_mags = 1.0 / (1.0 + (freqs_grid / cutoffs_grid) ** 4)
            filter_mags = filter_mags * n_lvl_noise[np.newaxis, :]
            filter_mags = filter_mags.T
            
            noise_t = torch.tensor(filter_mags, dtype=torch.float32, device=self.device).view(1, n_frames, N_NOISE_BANDS)
            
            # Differentiable Synthesis of the voice
            with torch.no_grad():
                voice_audio = self.synth(f0_t, amp_t, hd_t, noise_t).squeeze(0).cpu().numpy()
            
            # Mix the voice with its specific role gain and active gain (to prevent silent voice boosting)
            mixed_audio += voice_audio * (role["gain"] * role["active_gain"])
            
        # 3. Master Section (Convolution Reverb)
        master_audio = convolve_reverb(mixed_audio)

        # Final normalization
        peak = np.max(np.abs(master_audio)) + 1e-8
        return master_audio / peak * 0.9

    def _render_static_pad(self, wp, n_ctrl, breath_period_s,
                           pitch_drift_cents, drift_tau_s, param_wander_std,
                           seed, ctrl_hz):
        """One waypoint as a constant-theta five-voice pad (pre-reverb,
        un-normalized). Same breathing / OU wander / shimmer machinery as
        the trajectory path, but f0, chord and timbre never move -- motion
        between waypoints is the caller's gain crossfade, not parameter
        interpolation."""
        duration_s = n_ctrl / ctrl_hz
        num_samples = int(duration_s * SR)
        t_ctrl = np.linspace(0, duration_s, n_ctrl)
        t_audio = np.linspace(0, duration_s, num_samples)

        rng = np.random.default_rng(seed)
        base = {k: float(wp.get(k, 0.0)) for k in
                ["f0_hz", "swell_rate", "swell_depth", "noise_level",
                 "noise_cutoff_hz", "third_interval", "third_gain",
                 "fifth_gain", "octave_gain"]}
        base["swell_depth"] = min(base["swell_depth"], DDSP_SWELL_CAP)
        base["swell_rate"] = min(base["swell_rate"], 0.15)
        base["noise_level"] = min(base["noise_level"], 0.08)

        sc = {k: np.full(n_ctrl, v) for k, v in base.items()}
        if param_wander_std > 0:
            for k in ["swell_rate", "swell_depth", "noise_level"]:
                wander = _ou_drift(n_ctrl, 1.0 / ctrl_hz, drift_tau_s, rng)
                sc[k] = sc[k] * np.exp(param_wander_std * wander)
            # f0 wander capped at the 0.05-era depth regardless of the
            # texture-wander raise: at std 0.15 the old *0.1 coupling gives
            # ~±26 cents of pitch drift -- audibly out of tune on a drone.
            f0_wander = _ou_drift(n_ctrl, 1.0 / ctrl_hz, drift_tau_s, rng)
            sc["f0_hz"] = sc["f0_hz"] * np.exp(
                min(param_wander_std, 0.05) * 0.1 * f0_wander)
        rng = np.random.default_rng(seed)
        if pitch_drift_cents > 0:
            drift = _ou_drift(n_ctrl, 1.0 / ctrl_hz, drift_tau_s, rng)
            f0_factor = 2.0 ** (pitch_drift_cents / 1200.0 * drift)
        else:
            f0_factor = np.ones(n_ctrl)

        def ctrl_to_audio(arr):
            return np.interp(t_audio, t_ctrl, arr)

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

        mixed = np.zeros(num_samples)
        for vc in WAYPOINT_VOICES:
            g_static = static_gain.get(vc["name"], 1.0)
            if g_static <= 1e-4:
                continue
            role_harm = apply_spectral_tilt(harm, vc["bias"])
            detune_mult = 2.0 ** (vc["detune"] / 1200.0)
            voice_f0 = f0_base_audio * vc["mult"] * detune_mult * f0_factor_audio
            voice_amp = shimmer_env * g_static

            f0_t = torch.tensor(voice_f0, dtype=torch.float32,
                                device=self.device).view(1, -1, 1)
            amp_t = torch.tensor(voice_amp, dtype=torch.float32,
                                 device=self.device).view(1, -1, 1)
            hd_t = torch.tensor(np.tile(role_harm, (num_samples, 1)),
                                dtype=torch.float32,
                                device=self.device).unsqueeze(0)

            nl_f = n_lvl_frames * vc["noise_scale"] * g_static
            freqs_g, cuts_g = np.meshgrid(band_freqs, n_cut_frames,
                                          indexing="ij")
            fmags = (1.0 / (1.0 + (freqs_g / (cuts_g + 1e-8)) ** 4)) \
                * nl_f[np.newaxis, :]
            noise_t = torch.tensor(fmags.T, dtype=torch.float32,
                                   device=self.device).view(
                                       1, n_frames, N_NOISE_BANDS)
            with torch.no_grad():
                raw = self.synth(f0_t, amp_t, hd_t, noise_t)\
                    .squeeze(0).cpu().numpy()

            cos_curve = np.cos(vc["breath_phase"]
                               + 2.0 * np.pi * t_env / breath_period_s)
            genv = vc["gain"] * (1.0 - vc["breath_depth"] * 0.5
                                 * (1.0 + cos_curve))
            g_audio = np.interp(t_audio, t_env, genv)
            m = min(len(raw), num_samples)
            mixed[:m] += raw[:m] * g_audio[:m]
        return mixed

    def _render_waypoints_layered(self, theta_waypoints, hold_s, xfade_s,
                                  breath_period_s, pitch_drift_cents,
                                  drift_tau_s, param_wander_std, seed,
                                  ctrl_hz):
        """Overlap-add of static per-waypoint pads under equal-power gain
        crossfades. Timeline matches the interp path exactly:
        n_wp * (hold_s + xfade_s) total, waypoint i fully present on
        [i*(hold+xfade), i*(hold+xfade)+hold]."""
        n_wp = len(theta_waypoints)
        hold_n = max(1, int(round(hold_s * ctrl_hz)))
        xfade_n = max(1, int(round(xfade_s * ctrl_hz)))
        n_ctrl = n_wp * (hold_n + xfade_n)
        num_samples = int(n_ctrl / ctrl_hz * SR)
        spc = SR / ctrl_hz                       # samples per ctrl frame

        mixed = np.zeros(num_samples)
        for i, wp in enumerate(theta_waypoints):
            fade_in = xfade_n if i > 0 else 0
            fade_out = xfade_n if i < n_wp - 1 else xfade_n  # last wp rings
            seg_ctrl = fade_in + hold_n + fade_out
            raw = self._render_static_pad(
                wp, seg_ctrl, breath_period_s, pitch_drift_cents,
                drift_tau_s, param_wander_std, seed + i, ctrl_hz)
            env = np.ones(len(raw))
            fi = int(fade_in * spc)
            fo = int((xfade_n if i < n_wp - 1 else 0) * spc)
            if fi:
                env[:fi] = np.sin(np.linspace(0, np.pi / 2, fi)) ** 1.0
            if fo:
                env[-fo:] = np.cos(np.linspace(0, np.pi / 2, fo)) ** 1.0
            start = int((i * (hold_n + xfade_n) - fade_in) * spc)
            end = min(start + len(raw), num_samples)
            mixed[start:end] += (raw * env)[: end - start]

        out = convolve_reverb(mixed)
        peak = np.max(np.abs(out)) + 1e-8
        return out / peak * 0.88

    def render_waypoints(self, theta_waypoints, hold_s=4.0, xfade_s=1.5,
                         breath_period_s=10.0, pitch_drift_cents=2.0,
                         drift_tau_s=12.0, param_wander_std=0.05, seed=0,
                         xfade_mode="layered"):
        """
        v3-style rendering from sparse VA waypoints.

        Each waypoint is held for hold_s seconds then crossfaded to the next
        over xfade_s seconds. Macro breathing lives on a per-voice gain
        envelope (decorrelated: voices crest in rotation). The DDSP swell is
        capped to a shallow surface shimmer. A shared gentle pitch OU drift
        keeps all voices in tune while adding life.

        xfade_mode (2026-07-05, from an arc-pool smoke listen reporting
        "cents too high during transition" -- audible note/pitch motion):
          - "layered" (default): each waypoint renders as its OWN static pad
            (constant f0/chord/timbre) and adjacent pads overlap under an
            equal-power gain crossfade -- no scalar interpolation between
            anchors, so NO pitch glide; the two drones coexist and trade
            places, per the resolution_experiment_v2 rule (gain crossfades,
            never pitch glides) which the smoothstep path only honored for
            the chord third, not the root f0.
          - "interp": the legacy single-trajectory smoothstep path (kept for
            reproducing pre-2026-07-05 renders).
          - "glide": deliberate portamento. Kept in the RATED test as a
            probe (2026-07-05) rather than banned outright by the Phase 1
            rule, so listeners could reject it rather than it being assumed. Same single-trajectory path as "interp" but f0
            blends in LOG domain (perceptually even sweep; linear-Hz blend
            front-loads the pitch motion). The caller must slew-limit:
            xfade_s proportional to the octave difference (the arc-pool
            builder does this).

        Why this sounds better than render_trajectory at 20 Hz:
          - f0 / timbre only change every (hold_s + xfade_s) seconds → no wobble
          - Breathing is on the GAIN path, not the DDSP swell path → no tremolo
          - Voices peak in rotation → natural 1-2 voice prominence at any moment
        """
        CTRL_HZ = 10.0
        if xfade_mode == "layered":
            return self._render_waypoints_layered(
                theta_waypoints, hold_s, xfade_s, breath_period_s,
                pitch_drift_cents, drift_tau_s, param_wander_std, seed,
                CTRL_HZ)
        n_wp = len(theta_waypoints)
        hold_n = max(1, int(round(hold_s * CTRL_HZ)))
        xfade_n = max(1, int(round(xfade_s * CTRL_HZ)))
        n_ctrl = n_wp * (hold_n + xfade_n)
        duration_s = n_ctrl / CTRL_HZ

        # ── 1. Build hold+smoothstep control trajectory ──────────────────────
        keys = ["f0_hz", "swell_rate", "swell_depth", "noise_level", "noise_cutoff_hz",
                "third_interval", "third_gain", "fifth_gain", "octave_gain"]
        sc = {k: np.empty(n_ctrl) for k in keys}
        harm_ctrl = np.empty((n_ctrl, N_HARM))

        frame = 0
        for i, wp in enumerate(theta_waypoints):
            nxt = theta_waypoints[i + 1] if i + 1 < n_wp else wp
            for _ in range(hold_n):
                for k in keys:
                    sc[k][frame] = wp.get(k, 0.0)
                harm_ctrl[frame] = wp["harm_dist"]
                frame += 1
            for j in range(xfade_n):
                t = (j + 1) / xfade_n
                a = t * t * (3.0 - 2.0 * t)          # smoothstep
                for k in keys:
                    if xfade_mode == "glide" and k == "f0_hz":
                        # log-domain pitch blend: constant cents/second
                        sc[k][frame] = np.exp(
                            (1.0 - a) * np.log(max(wp.get(k, 1.0), 1.0))
                            + a * np.log(max(nxt.get(k, 1.0), 1.0)))
                    else:
                        sc[k][frame] = (1.0 - a) * wp.get(k, 0.0) + a * nxt.get(k, 0.0)
                harm_ctrl[frame] = (1.0 - a) * wp["harm_dist"] + a * nxt["harm_dist"]
                frame += 1

        # Cap swell on the DDSP path — macro breath goes on the gain envelope
        sc["swell_depth"] = np.minimum(sc["swell_depth"], DDSP_SWELL_CAP)
        sc["swell_rate"]  = np.minimum(sc["swell_rate"],  0.15)
        sc["noise_level"] = np.minimum(sc["noise_level"], 0.08)
        harm_ctrl = harm_ctrl / (harm_ctrl.sum(axis=1, keepdims=True) + 1e-8)
        
        # ── 1b. Add continuous parameter wandering (OU drift) ────────────────
        rng = np.random.default_rng(seed)
        if param_wander_std > 0:
            # Independent OU walks for key parameters so they don't sound completely static
            # during the hold segments.
            for k in ["swell_rate", "swell_depth", "noise_level"]:
                wander = _ou_drift(n_ctrl, 1.0 / CTRL_HZ, drift_tau_s, rng)
                sc[k] = sc[k] * np.exp(param_wander_std * wander)
            # f0_hz gets an independent slight drift as well (in addition to shared pitch drift below)
            f0_wander = _ou_drift(n_ctrl, 1.0 / CTRL_HZ, drift_tau_s, rng)
            sc["f0_hz"] = sc["f0_hz"] * np.exp(param_wander_std * 0.1 * f0_wander)

        # ── 2. Shared pitch OU drift (all voices move together → stay in tune) ─
        rng = np.random.default_rng(seed)
        if pitch_drift_cents > 0:
            drift = _ou_drift(n_ctrl, 1.0 / CTRL_HZ, drift_tau_s, rng)
            f0_factor = 2.0 ** (pitch_drift_cents / 1200.0 * drift)
        else:
            f0_factor = np.ones(n_ctrl)

        # ── 3. Interpolation grids ─────────────────────────────────────────────
        num_samples = int(duration_s * SR)
        t_ctrl  = np.linspace(0, duration_s, n_ctrl)
        t_audio = np.linspace(0, duration_s, num_samples)

        def ctrl_to_audio(arr):
            return np.interp(t_audio, t_ctrl, arr)

        f0_base_audio  = np.exp(np.interp(t_audio, t_ctrl, np.log(np.maximum(sc["f0_hz"], 1.0))))
        rate_audio     = ctrl_to_audio(sc["swell_rate"])
        depth_audio    = ctrl_to_audio(sc["swell_depth"])
        n_lvl_audio    = ctrl_to_audio(sc["noise_level"])
        n_cut_audio    = ctrl_to_audio(sc["noise_cutoff_hz"])
        f0_factor_audio = np.exp(np.interp(t_audio, t_ctrl, np.log(np.maximum(f0_factor, 1e-6))))

        # DDSP surface-shimmer amplitude (shallow swell only)
        dt = 1.0 / SR
        swell_phase = 2.0 * np.pi * np.cumsum(rate_audio * dt)
        shimmer_env = (1.0 - depth_audio * 0.5 * (1.0 + np.sin(swell_phase))) * 0.8

        # Harm at audio rate (smoothstep crossfades already baked in at ctrl rate)
        harm_audio = np.zeros((num_samples, N_HARM))
        for h in range(N_HARM):
            harm_audio[:, h] = np.interp(t_audio, t_ctrl, harm_ctrl[:, h])
        harm_audio = harm_audio / (harm_audio.sum(axis=1, keepdims=True) + 1e-8)

        # Noise frames
        n_frames = num_samples // (N_NOISE_BANDS - 1)
        t_frames = np.linspace(0, duration_s, n_frames)
        n_lvl_frames = np.interp(t_frames, t_audio, n_lvl_audio)
        n_cut_frames = np.interp(t_frames, t_audio, n_cut_audio)
        band_freqs   = np.linspace(0, SR / 2, N_NOISE_BANDS)

        # ── 4. Per-voice breathing gain envelope ──────────────────────────────
        t_env = np.linspace(0, duration_s, n_ctrl)

        def breath_env(voice_cfg):
            depth = voice_cfg["breath_depth"]
            phase = voice_cfg["breath_phase"]
            # cosine: starts at crest (gain=base) when cos=1, trough when cos=-1
            # env = base * (1 - depth * 0.5 * (1 + cos(phase + 2π t / T)))
            cos_curve = np.cos(phase + 2.0 * np.pi * t_env / breath_period_s)
            return voice_cfg["gain"] * (1.0 - depth * 0.5 * (1.0 + cos_curve))

        # Split the generic `third_gain` and `third_interval` into distinct minor/major gain envelopes
        # This implements true gain crossfades for chord transitions (arcs) instead of pitch smearing.
        t_int = ctrl_to_audio(sc["third_interval"])
        t_gain = ctrl_to_audio(sc["third_gain"])
        
        # Mapping: 3.0 -> pure minor, 4.0 -> pure major. 
        # Interpolate gain based on the interval value
        maj_frac = np.clip(t_int - 3.0, 0.0, 1.0)
        min_frac = 1.0 - maj_frac
        
        min3_audio = t_gain * min_frac
        maj3_audio = t_gain * maj_frac
        fifth_audio = ctrl_to_audio(sc["fifth_gain"])
        octave_audio = ctrl_to_audio(sc["octave_gain"])

        # ── 5. Render voices and mix ──────────────────────────────────────────
        mixed = np.zeros(num_samples)

        for vc in WAYPOINT_VOICES:
            # Determine dynamic active gain for the voice
            active_g_audio = np.ones(num_samples)
            if vc["name"] == "pad_min3":
                active_g_audio = min3_audio
            elif vc["name"] == "pad_maj3":
                active_g_audio = maj3_audio
            elif vc["name"] == "pad_fifth":
                active_g_audio = fifth_audio
            elif vc["name"] == "air":
                active_g_audio = octave_audio
                
            # Spectral tilt per sample
            role_harm = np.stack([apply_spectral_tilt(h, vc["bias"]) for h in harm_audio])

            detune_mult = 2.0 ** (vc["detune"] / 1200.0)
            voice_f0 = f0_base_audio * vc["mult"] * detune_mult * f0_factor_audio

            # Apply active gain to the shimmer envelope to ensure internal DDSP noise is proportional
            voice_amp = shimmer_env * active_g_audio
            
            f0_t  = torch.tensor(voice_f0,    dtype=torch.float32, device=self.device).view(1, -1, 1)
            amp_t = torch.tensor(voice_amp, dtype=torch.float32, device=self.device).view(1, -1, 1)
            hd_t  = torch.tensor(role_harm,   dtype=torch.float32, device=self.device).unsqueeze(0)

            n_scale = vc["noise_scale"]
            nl_f = n_lvl_frames * n_scale * np.interp(t_frames, t_audio, active_g_audio)
            freqs_g, cuts_g = np.meshgrid(band_freqs, n_cut_frames, indexing="ij")
            fmags = (1.0 / (1.0 + (freqs_g / (cuts_g + 1e-8)) ** 4)) * nl_f[np.newaxis, :]
            noise_t = torch.tensor(fmags.T, dtype=torch.float32, device=self.device).view(1, n_frames, N_NOISE_BANDS)

            with torch.no_grad():
                raw = self.synth(f0_t, amp_t, hd_t, noise_t).squeeze(0).cpu().numpy()

            # Apply decorrelated macro breath envelope * role gain
            genv = breath_env(vc)
            g_audio = np.interp(t_audio, t_env, genv)
            m = min(len(raw), num_samples)
            mixed[:m] += raw[:m] * g_audio[:m]

        # ── 6. Master (reverb + peak normalize) ──────────────────────────────
        out = convolve_reverb(mixed)
        peak = np.max(np.abs(out)) + 1e-8
        return out / peak * 0.88
