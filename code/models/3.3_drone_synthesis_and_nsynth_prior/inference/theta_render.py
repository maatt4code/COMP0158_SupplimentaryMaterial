"""Turn a theta parameter set into audio.

This is the render path, split out of the old ``s03_generate_ddsp_dataset.py``
where generation and rendering were one file. Generation samples new theta and
belongs to training; rendering turns an existing theta into sound and is what
the runtime does. Only the second half is here.

Theta is the 41-dimensional synthesis vector of Section 3.3: 32 harmonic
weights plus nine scalars (f0, swell rate, swell depth, noise level, noise
cutoff, third interval, third gain, fifth gain, octave gain). ``tilt`` is
applied to the harmonic distribution when a preset is generated, so it is not
carried as a separate coordinate here.

Nothing in this module samples, trains, or touches a dataset root.

Reproducibility. The ``rng`` argument controls the f0 wander only. The
filtered-noise floor is drawn by ``torch.randn`` inside the synth, which takes
no generator, so a caller that needs a repeatable render must also call
``torch.manual_seed``. ``generate_preset_bank.py`` does this.

Used as a library:
    from theta_render import render_theta, SR
    audio = render_theta(synth, theta, n_samples, device, rng)

Rendered to a file directly, for a quick listen:
    python theta_render.py --out /tmp/demo.wav --f0 55.0 --duration 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from ddsp_synth import DifferentiableDDSPSynth  # noqa: E402
from device import get_device                   # noqa: E402

SR = 16000
N_HARMONICS = 32
N_NOISE_BANDS = 65


def chord_voices(theta: dict) -> list[tuple[float, float]]:
    """[(frequency multiplier, gain), ...] for every active voice, root first.

    A voice is skipped when its gain is effectively zero, which is how plain
    single-voice drones stay in the bank alongside chordal ones.
    """
    voices = [(1.0, 1.0)]
    if theta.get("third_gain", 0.0) > 1e-3:
        voices.append((2.0 ** (theta["third_interval"] / 12.0), theta["third_gain"]))
    if theta.get("fifth_gain", 0.0) > 1e-3:
        voices.append((2.0 ** (7.0 / 12.0), theta["fifth_gain"]))
    if theta.get("octave_gain", 0.0) > 1e-3:
        voices.append((2.0, theta["octave_gain"]))
    return voices


def theta_to_synth_inputs(theta: dict, n_samples: int, device, rng: np.random.Generator):
    """Expand a static theta into the time-varying tensors DDSP expects."""
    t = np.arange(n_samples) / SR

    # f0: static pitch plus a very slow wander within +-2.9%, about a
    # quartertone. Enough that the clip is alive, small enough that it still
    # reads as one note.
    wander = np.cumsum(rng.normal(0, 1, n_samples)).astype(np.float64)
    wander = wander / (np.abs(wander).max() + 1e-8) * 0.029
    f0 = theta["f0_hz"] * (1.0 + wander)
    f0_t = torch.tensor(f0, dtype=torch.float32, device=device).view(1, -1, 1)

    # amplitude: slow swell, never fully silent
    swell = 1.0 - theta["swell_depth"] * 0.5 * (1.0 + np.sin(2 * np.pi * theta["swell_rate"] * t))
    amp_t = torch.tensor(swell, dtype=torch.float32, device=device).view(1, -1, 1) * 0.8

    # harmonic distribution: one frame, held for the whole clip
    hd = torch.tensor(theta["harm_dist"], dtype=torch.float32, device=device)
    hd_t = hd.view(1, 1, N_HARMONICS).expand(1, n_samples, N_HARMONICS)

    # noise: static lowpass-shaped band gains scaled by level
    n_frames = n_samples // (N_NOISE_BANDS - 1)
    band_freqs = np.linspace(0, SR / 2, N_NOISE_BANDS)
    band_gains = 1.0 / (1.0 + (band_freqs / theta["noise_cutoff_hz"]) ** 4)
    band_gains = band_gains * theta["noise_level"]
    noise_t = torch.tensor(band_gains, dtype=torch.float32, device=device)
    noise_t = noise_t.view(1, 1, N_NOISE_BANDS).expand(1, n_frames, N_NOISE_BANDS).contiguous()

    return f0_t, amp_t, hd_t, noise_t


def render_theta(synth, theta: dict, n_samples: int, device, rng) -> np.ndarray:
    """Render a possibly chordal theta.

    Every voice shares the timbre and the swell, transposed by its interval
    ratio. Noise rides on the root only, so adding voices does not multiply the
    noise floor. The sum is peak-normalised to 0.9.
    """
    total = None
    for i, (mult, gain) in enumerate(chord_voices(theta)):
        t_voice = dict(theta)
        t_voice["f0_hz"] = theta["f0_hz"] * mult
        if i > 0:
            t_voice["noise_level"] = 0.0
        f0_t, amp_t, hd_t, noise_t = theta_to_synth_inputs(t_voice, n_samples, device, rng)
        with torch.no_grad():
            audio = synth(f0_t, amp_t, hd_t, noise_t).squeeze(0).cpu().numpy() * gain
        total = audio if total is None else total + audio
    peak = np.max(np.abs(total)) + 1e-8
    return total / peak * 0.9


def polish(audio: np.ndarray, sr: int = SR, tail_sec: float = 6.0,
           wet: float = 0.35) -> np.ndarray:
    """Fixed production chain applied before labelling.

    Dark convolution reverb with an ENERGY-normalised impulse response (unit
    sum of squares; peak normalisation multiplies the gain about 74x and
    clips), RMS-matched wet/dry blend, then a peak guard.

    This chain is constant and is not part of theta. The affect estimators
    should see audio that resembles installation output rather than a dry stem.
    The impulse response is seeded, so every clip gets the same room.
    """
    audio = np.asarray(audio, dtype=np.float64)
    ir_len = int(sr * tail_sec)
    decay = np.exp(-np.linspace(0, 6, ir_len))
    rng_ir = np.random.default_rng(1234)
    ir = rng_ir.normal(0, 1, ir_len) * decay
    sos = sps.butter(2, 600.0 / (sr / 2), btype="low", output="sos")
    ir = sps.sosfilt(sos, ir)
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-8

    rev = sps.fftconvolve(audio, ir, mode="full")[: len(audio)]
    dry_rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
    wet_rms = np.sqrt(np.mean(rev ** 2)) + 1e-8
    rev *= dry_rms / wet_rms

    out = audio * (1 - wet) + rev * wet
    peak = np.max(np.abs(out)) + 1e-8
    if peak > 0.95:
        out *= 0.95 / peak
    return out.astype(np.float32)


def _demo_theta(f0_hz: float) -> dict:
    """A flat-ish single-voice theta. For the CLI demo and the smoke test only;
    real theta comes from the preset bank."""
    harm = np.arange(1, N_HARMONICS + 1, dtype=np.float32) ** -1.2
    return {"harm_dist": harm / harm.sum(), "f0_hz": f0_hz,
            "swell_rate": 0.08, "swell_depth": 0.3,
            "noise_level": 0.05, "noise_cutoff_hz": 1200.0,
            "third_interval": 4.0, "third_gain": 0.0,
            "fifth_gain": 0.0, "octave_gain": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description="Render one demo theta to a wav.")
    ap.add_argument("--out", required=True, help="output .wav path")
    ap.add_argument("--f0", type=float, default=55.0, help="fundamental in Hz")
    ap.add_argument("--duration", type=float, default=6.0, help="seconds")
    ap.add_argument("--polish", action="store_true", help="apply the fixed reverb chain")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import soundfile as sf
    device = get_device()
    synth = DifferentiableDDSPSynth(sample_rate=SR).to(device)
    rng = np.random.default_rng(args.seed)
    audio = render_theta(synth, _demo_theta(args.f0), int(args.duration * SR), device, rng)
    if args.polish:
        audio = polish(audio)
    sf.write(args.out, audio.astype(np.float32), SR)
    print(f"wrote {args.out}  ({args.duration:.1f}s at {SR} Hz)")


if __name__ == "__main__":
    main()
