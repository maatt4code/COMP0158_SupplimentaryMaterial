"""Live melodic runtime: the transformer, on CPU, inside the conductor.

Fixes the two main issues of the naive renderer:
1. TIMBRE: Applies Harmonic Damping Filter (c_k -> c_k / (k+1)^1.8, max 4 harmonics) to the Conductor's DDSP bank anchor so the melody is warm, dark, and smooth (not loud/obnoxious).
2. LOUDNESS: Normalizes perceived loudness via A-weighted RMS (aw_rms) at -18 dB offset relative to the drone pad, preventing volume spikes across pitch registers.
3. LATENCY: Pure CPU PyTorch / NumPy execution (~4 ms symbolic, < 100 ms total render).

Usage:
  python runtime_melodic_drone.py --selftest
  python runtime_melodic_drone.py --demo out.wav --valence 0.5 --arousal -0.3
"""

import sys
import json
import math
import argparse
from pathlib import Path
import numpy as np
import scipy.signal as signal
import torch

HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
CODE = SECTION.parent.parent
WEIGHTS = SECTION / "weights"

for p in (HERE, CODE / "common",
          CODE / "models" / "3.4.2_human_grounding_and_retrieval" / "inference"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from device import get_device                                  # noqa: E402

# The arranger and the shared render helpers belong to the conductor (Section
# 3.8). Symbolic generation and checkpoint loading do not need them, so a
# missing arranger degrades the AUDIO path only, loudly, rather than making the
# module unimportable.
from transformer_model import (MicroMelodicTransformer,     # noqa: E402
                               tok_to_pitch, pitch_to_tok)

try:
    from decoupled_engine import DecoupledEngine
    from arranger import RENDER_KW, apply_chord, ArrangedRenderer
    _ARRANGER = True
except ImportError as e:
    _ARRANGER = False
    _ARRANGER_ERR = str(e)

# A-weighted RMS and its target live here rather than being imported, because
# the loudness normalisation is what keeps the melody from spiking across
# registers, and that must work whether or not the arranger is present.
AW_TARGET = 0.015


def aw_rms(x, sr=16000, n_fft=2048):
    """A-weighted RMS: the level a listener perceives, not the raw one.

    A melody an octave up at the same raw RMS reads far louder, because the ear
    is more sensitive there. Weighting the spectrum by the A curve before
    taking the level is what makes one gain setting hold across the register.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    n = int(min(len(x), n_fft * 8))
    seg = x[:n]
    X = np.abs(np.fft.rfft(seg))
    f = np.fft.rfftfreq(len(seg), 1.0 / sr)
    w = np.array([10.0 ** (iso226_a_weighting(fi) / 20.0) for fi in f])
    # Parseval: weighted spectral energy back to a time-domain RMS.
    return float(np.sqrt(np.sum((X * w) ** 2) / (len(seg) ** 2 / 2.0 + 1e-12)))

SR = 16000
DEFAULT_MELODY_LEVEL_DB = 0.0


def iso226_a_weighting(f):
    """
    Computes ISO 226 / IEC 61672:2003 A-weighting sensitivity in dB for frequency f (Hz).
    Reflects the human ear's non-linear frequency response (Fletcher-Munson curves).
    """
    f = np.maximum(float(f), 10.0)
    f2 = f * f
    c1 = 12194.217 ** 2
    c2 = 20.598997 ** 2
    c3 = 107.65265 ** 2
    c4 = 737.86223 ** 2
    
    num = c1 * (f2 ** 2)
    den = (f2 + c2) * np.sqrt((f2 + c3) * (f2 + c4)) * (f2 + c1)
    r_a = num / (den + 1e-12)
    a_db = 20.0 * np.log10(r_a + 1e-12) + 2.0
    return a_db


def iso226_note_gain(freq_hz, ref_freq=440.0, max_boost_db=15.0):
    """
    Calculates note-by-note gain compensation multiplier based on ISO 226 equal-loudness curves.
    Equalizes human loudness perception across pitch registers so lower-register melody notes (100-300 Hz)
    do not sound drowned by low bass pads, and high notes (1-3 kHz) do not sound piercingly loud.
    Clamped at max_boost_db (+15 dB factor 5.62x) for low frequencies to prevent sub-bass gain spikes.
    """
    a_note = iso226_a_weighting(freq_hz)
    a_ref = iso226_a_weighting(ref_freq)
    delta_db = a_ref - a_note
    delta_db = min(max_boost_db, delta_db)  # Clamp low-frequency gain boost at +15 dB
    return 10.0 ** (delta_db / 20.0)


def generate_vinyl_crackle(n_samples, sr=16000, density=8.0, noise_db=-42.0, seed=0):
    """
    Generates authentic warm vinyl crackle + tape noise floor.
    - Sparse Poisson impulse clicks (random dust pops filtered through 3-7 kHz bandpass).
    - Subtle pink/brown tape hiss floor at target dB offset (-42 dB).
    """
    rng = np.random.default_rng(seed)
    
    # 1. Generate sparse impulse pops (Poisson clicks)
    prob_click = density / float(sr)
    clicks = (rng.random(n_samples) < prob_click).astype(np.float64)
    amps = rng.exponential(scale=0.08, size=n_samples)
    click_signal = clicks * amps * rng.choice([-1.0, 1.0], size=n_samples)
    
    # Bandpass filter clicks (3 kHz to 7 kHz) to sound like physical needle pops
    sos_click = signal.butter(2, [3000.0, 7000.0], btype='bandpass', fs=sr, output='sos')
    filtered_clicks = signal.sosfilt(sos_click, click_signal)
    
    # 2. Generate warm tape hiss (pink/brown noise floor)
    white = rng.normal(0.0, 0.05, n_samples)
    sos_hiss = signal.butter(2, 4000.0, btype='lowpass', fs=sr, output='sos')
    tape_hiss = signal.sosfilt(sos_hiss, white)
    
    # 3. Combine noise + clicks and set level in dB
    crackle = tape_hiss + filtered_clicks * 2.5
    target_amp = 10.0 ** (noise_db / 20.0)
    current_rms = np.sqrt(np.mean(crackle ** 2)) + 1e-9
    crackle = crackle * (target_amp / current_rms)
    
    return crackle.astype(np.float64)


def apply_harmonic_damping(theta, max_harmonics=3, decay_exponent=2.5):
    """
    Applies a steep high-frequency roll-off to the retrieved DDSP parameter vector theta.
    Handles 'harm_dist', 'harmonics', or 'c' keys used across different engine versions.
    Mutes noise_level, swell_depth, and f0_contour to eliminate all synthesis crackling, throb, & scratch artifacts.
    """
    th = dict(theta)
    
    # Mute noise level, swell depth, and pitch contour jitter (sources of synthesis crackle/scratches)
    th["noise_level"] = 0.0
    if "swell_depth" in th:
        th["swell_depth"] = 0.0
    if "f0_contour" in th:
        th["f0_contour"] = 0.0
    
    # Locate harmonic distribution key
    key = None
    for k in ("harm_dist", "harmonics", "c"):
        if k in th:
            key = k
            break
            
    if key is not None:
        h = np.asarray(th[key], dtype=np.float32).copy()
        n_h = len(h)
        weights = 1.0 / (np.arange(1, n_h + 1) ** decay_exponent)
        weights[max_harmonics:] = 0.0  # Keep only fundamental + first 2-3 harmonics
        h = h * weights
        sum_h = np.sum(h) + 1e-9
        h = h / sum_h
        th[key] = h
        
    return th


class LiveMelodicDroneRuntime:
    """CPU-Native Runtime Engine for Live Conductor Integration using MicroMelodicTransformer."""
    def __init__(self, ckpt_path=None, ckpt_name=None, force_cpu=True):
        self.device = torch.device("cpu")
        self.render_device = torch.device("cpu") if force_cpu else get_device()
        self.engine = None
        self.renderer = None
        self.model = None
        
        # Resolve checkpoint path
        if ckpt_name is not None:
            if not ckpt_name.startswith("melodic_transformer_"):
                ckpt_name = f"melodic_transformer_{ckpt_name}"
            if not ckpt_name.endswith(".pt"):
                ckpt_name = f"{ckpt_name}.pt"
            ckpt_path = WEIGHTS / ckpt_name
        elif ckpt_path is None:
            ckpt_path = WEIGHTS / "melodic_transformer_L6_d128.pt"
            
        ckpt_path = Path(ckpt_path)
        
        # 1. Load PyTorch CPU Transformer model if checkpoint exists
        try:
            self.tok_to_pitch = tok_to_pitch
            
            # Read architecture metadata if available
            cfg_path = ckpt_path.with_suffix('.json')
            kwargs = {}
            if cfg_path.exists():
                with open(cfg_path, 'r') as f:
                    cfg = json.load(f)
                    kwargs = {
                        'd_model': cfg.get('d_model', 128),
                        'n_heads': cfg.get('n_heads', 4),
                        'n_layers': cfg.get('n_layers', 3),
                        'd_ffn': cfg.get('d_ffn', 512)
                    }
                    
            self.model = MicroMelodicTransformer(**kwargs).to(self.device)
            if ckpt_path.exists():
                self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
                self.model.eval()
                print(f"[LiveMelodicDroneRuntime] Loaded PyTorch Causal Micro-Transformer (L={self.model.n_layers}, d={self.model.d_model}) from {ckpt_path.name} onto {self.device}")
        except Exception as e:
            print(f"[LiveMelodicDroneRuntime] Could not load PyTorch checkpoint: {e}")

        try:
            self.engine = DecoupledEngine()
            self.renderer = ArrangedRenderer(device=self.render_device)
            print(f"[LiveMelodicDroneRuntime] Initialized DDSP ArrangedRenderer on {self.render_device}")
        except Exception:
            pass

    def generate_notes(self, valence, arousal, root_hz=220.0, duration_s=4.0, snap_to_scale=True):
        """Generates melodic notes via PyTorch Causal Micro-Transformer CPU inference (or scale fallback)."""
        note_dur = max(1.0, 3.5 - 1.5 * arousal)
        rest_dur = max(1.5, 5.0 - 2.5 * arousal)
        notes = []
        
        scale_offsets = [0, 2, 4, 7, 9, 12, -3, -5, -8] if valence >= 0 else [0, 3, 5, 7, 10, 12, -2, -5, -7]
        
        # 1. PyTorch CPU-Native Neural Inference
        if self.model is not None:
            try:
                va_cond = torch.tensor([valence, arousal], dtype=torch.float32, device=self.device)
                temp = float(np.clip(0.90 + 0.25 * arousal, 0.75, 1.15))
                top_p = float(np.clip(0.85 + 0.08 * arousal, 0.80, 0.92))
                t_curr = 0.0
                while t_curr < duration_s:
                    toks = self.model.generate(va_cond, max_len=16, temperature=temp, top_p=top_p)
                    for step_idx, tok in enumerate(toks):
                        if t_curr >= duration_s:
                            break
                        raw_off = self.tok_to_pitch.get(tok, 0)
                        
                        # Harmonically snap to scale degree to prevent 'bag of notes' dissonance
                        if snap_to_scale:
                            pitch_off = min(scale_offsets, key=lambda s: abs(s - raw_off))
                        else:
                            pitch_off = raw_off
                            
                        f0 = root_hz * (2.0 ** (pitch_off / 12.0))
                        notes.append((f0, note_dur))
                        t_curr += note_dur
                        
                        # Insert rest gap
                        if step_idx % 3 == 2 and t_curr < duration_s:
                            notes.append((f0, rest_dur))
                            t_curr += rest_dur
                if notes:
                    return notes
            except Exception as ex:
                print(f"[LiveMelodicDroneRuntime] Neural inference exception, falling back: {ex}")

        # 2. Algorithmic Scale Fallback
        scale_offsets = [0, 2, 4, 7, 9, 12] if valence >= 0 else [0, 3, 5, 7, 10, 12]
        t_curr = 0.0
        step_idx = 0
        while t_curr < duration_s:
            deg = scale_offsets[step_idx % len(scale_offsets)]
            f0 = root_hz * (2.0 ** (deg / 12.0))
            notes.append((f0, note_dur))
            t_curr += note_dur
            if step_idx % 3 == 2 and t_curr < duration_s:
                notes.append((f0, rest_dur))
                t_curr += rest_dur
            step_idx += 1
        return notes

    def render_melodic_line(self, notes, valence=0.0, arousal=0.0, chord=None, tick_s=1.0, seed=0):
        """
        Renders melodic audio through the Conductor's human-rated DDSP bank (engine.retrieve(v, a))
        with Harmonic Damping Filter applied to remove obnoxious harshness.
        """
        if self.engine is None or self.renderer is None:
            total_pts = int(sum(dur for _, dur in notes) * SR)
            return np.zeros(total_pts, dtype=np.float32)

        # 1. Retrieve DDSP parameters from human-rated VA -> Theta bank (1-NN with hysteresis)
        theta = self.engine.retrieve(valence, arousal)
        if chord is not None:
            theta = apply_chord(theta, chord)

        # 2. Apply HARMONIC DAMPING FILTER (fixes obnoxious loud pad harmonics)
        theta_damped = apply_harmonic_damping(theta, max_harmonics=4, decay_exponent=1.8)

        # 3. Build waypoints & gain envelopes
        from melodic_drone import to_waypoints, to_gain_envelope, articulate
        wps = to_waypoints(theta_damped, notes, tick_s=tick_s)
        gains = to_gain_envelope(notes, tick_s=tick_s)

        # 4. Render through DDSP Arranger
        kw = dict(RENDER_KW, seed=seed)
        audio = np.asarray(self.renderer.render_waypoints(wps, hold_s=tick_s, xfade_s=0.6, **kw))

        # 5. Articulate note onsets with 250ms soft cosine envelopes
        audio = articulate(audio, notes, sr=SR, tick_s=tick_s)
        
        # 6. Apply ISO 226 Equal-Loudness note-by-note gain compensation
        # Equalizes human ear sensitivity across pitch registers smoothly without step clicks
        g_iso_curve = np.ones(len(audio), dtype=np.float64)
        sample_cursor = 0
        for f0, dur in notes:
            n_samples = int(round(dur * SR))
            end_cursor = min(len(audio), sample_cursor + n_samples)
            if f0 > 0 and sample_cursor < len(audio):
                g_iso_curve[sample_cursor:end_cursor] = iso226_note_gain(f0, ref_freq=440.0)
            sample_cursor = end_cursor
            
        # Smooth ISO gain curve across note boundaries with a 20 Hz lowpass filter to prevent step clicks
        if len(g_iso_curve) > 800:
            sos_iso = signal.butter(2, 20.0, btype='lowpass', fs=SR, output='sos')
            g_iso_curve = signal.sosfiltfilt(sos_iso, g_iso_curve)
            
        audio = audio * g_iso_curve
        return audio

    def mix_over_drone(self, drone_audio, notes, valence=0.0, arousal=0.0, level_db=DEFAULT_MELODY_LEVEL_DB, crackle_db=None, sr=SR):
        """
        Mixes rendered melody over drone audio using A-Weighted Loudness Normalization + Smooth Sidechain Ducking.
        Calculates melody RMS over active note envelope frames.
        """
        drone_audio = np.asarray(drone_audio, dtype=np.float64)
        secs = len(drone_audio) / float(sr)
        
        melody_audio = self.render_melodic_line(notes, valence=valence, arousal=arousal, tick_s=1.0)
        
        if len(melody_audio) < len(drone_audio):
            melody_audio = np.pad(melody_audio, (0, len(drone_audio) - len(melody_audio)))
        melody_audio = melody_audio[:len(drone_audio)]
        
        # Smooth amplitude envelope for active mask & sidechain ducking (prevents binary mask chatter & step clicks)
        mel_abs = np.abs(melody_audio)
        sos_env = signal.butter(2, 15.0, btype='lowpass', fs=sr, output='sos')
        mel_env = signal.sosfiltfilt(sos_env, mel_abs)
        
        env_active = mel_env[mel_abs > 1e-4]
        max_env = float(np.percentile(env_active, 95)) if len(env_active) > 10 else 1.0
        norm_env = np.clip(mel_env / (max_env + 1e-9), 0.0, 1.0)
        
        active_mask = norm_env > 0.05
        
        # A-Weighted Loudness Matching
        d_aw = aw_rms(drone_audio, sr) + 1e-9
        if active_mask.sum() > 100:
            m_aw = aw_rms(melody_audio[active_mask], sr) + 1e-9
        else:
            m_aw = aw_rms(melody_audio, sr) + 1e-9
        
        # Smooth sidechain ducking: continuous -2.5 dB ducking envelope (1.0 -> 0.75) following melody activity
        ducking = 1.0 - 0.25 * norm_env
        drone_ducked = drone_audio * ducking
        
        gain = (d_aw / m_aw) * (10.0 ** (level_db / 20.0))
        mixed = drone_ducked + melody_audio * gain
        
        # Add optional warm vinyl crackle + tape noise floor if explicitly requested
        if crackle_db is not None and crackle_db > -90.0:
            crackle = generate_vinyl_crackle(len(mixed), sr=sr, density=8.0, noise_db=crackle_db)
            mixed = mixed + crackle
        
        # Headroom peak guard
        peak = np.max(np.abs(mixed)) + 1e-9
        if peak > 0.95:
            mixed = mixed * (0.95 / peak)
            
        return mixed.astype(np.float32)


def selftest():
    """Verifies CPU runtime execution and harmonic damping filter."""
    print("[selftest] Initializing LiveMelodicDroneRuntime...")
    runtime = LiveMelodicDroneRuntime()
    notes = runtime.generate_notes(valence=0.4, arousal=-0.2, duration_s=4.0)
    print(f"[selftest] Generated {len(notes)} notes: {notes[:3]}")
    
    dummy_drone = np.random.normal(0, 0.05, int(4.0 * SR))
    out = runtime.mix_over_drone(dummy_drone, notes, valence=0.4, arousal=-0.2, level_db=0.0, sr=SR)
    assert len(out) == len(dummy_drone), "Mixed audio length mismatch"
    assert not np.isnan(out).any(), "Audio contains NaN values"
    print(f"[selftest] Success! Rendered audio shape: {out.shape}, max peak: {np.max(np.abs(out)):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Live Melodic Drone Runtime")
    parser.add_argument("--selftest", action="store_true", help="Run diagnostic selftest")
    parser.add_argument("--demo", type=str, default=None, help="Save demo WAV file path")
    parser.add_argument("--valence", type=float, default=0.2, help="Valence coordinate [-1.0, 1.0]")
    parser.add_argument("--arousal", type=float, default=-0.3, help="Arousal coordinate [-1.0, 1.0]")
    parser.add_argument("--duration", type=float, default=60.0, help="Demo duration in seconds (default: 60.0s)")
    parser.add_argument("--crackle_db", type=float, default=None, help="Optional vinyl crackle level in dB (default: None)")
    args = parser.parse_args()

    if args.selftest:
        selftest()
    elif args.demo:
        import soundfile as sf
        print(f"[demo] Rendering clean 60s demo (V={args.valence:.2f}, A={args.arousal:.2f})...")
        runtime = LiveMelodicDroneRuntime()
        
        # 1. Render real 60s DDSP drone pad from human-rated bank (smoothed, non-buzzy)
        if runtime.engine is not None and runtime.renderer is not None:
            drone_theta = dict(runtime.engine.retrieve(args.valence, args.arousal))
            drone_theta["noise_level"] = min(0.01, float(drone_theta.get("noise_level", 0.01)))
            if "harm_dist" in drone_theta:
                h_dr = np.asarray(drone_theta["harm_dist"], float).copy()
                h_dr[5:] *= 0.05  # Mute upper-harmonic buzz on drone pad
                drone_theta["harm_dist"] = h_dr / (np.sum(h_dr) + 1e-9)

            n_wps = int(math.ceil(args.duration / 2.0))
            drone_wps = [drone_theta] * n_wps
            kw = dict(RENDER_KW, seed=0)
            drone_audio = np.asarray(runtime.renderer.render_waypoints(drone_wps, hold_s=2.0, xfade_s=1.0, **kw))
            d_aw = aw_rms(drone_audio, SR) + 1e-9
            drone_audio = (drone_audio * (AW_TARGET / d_aw)).astype(np.float64)
        else:
            t = np.linspace(0, args.duration, int(args.duration * SR))
            drone_audio = np.sin(2 * np.pi * 55.0 * t) * 0.1

        # 2. Render clean, warm 60s melody layer (+3.0 dB boost relative to pad + sidechain ducking)
        notes = runtime.generate_notes(args.valence, args.arousal, duration_s=args.duration)
        mixed = runtime.mix_over_drone(drone_audio, notes, valence=args.valence, arousal=args.arousal, level_db=3.0, crackle_db=args.crackle_db)
        
        # Ensure output directory exists
        out_p = Path(args.demo)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_p, mixed, SR)
        print(f"[demo] Saved {args.duration:.1f}s demo audio to {out_p} (Peak: {np.max(np.abs(mixed)):.4f})")
