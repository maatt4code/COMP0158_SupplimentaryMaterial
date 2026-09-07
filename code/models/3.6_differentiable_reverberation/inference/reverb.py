"""Apply reverb. Numpy only, no torch, no fitting.

Runtime code, and the half of the reverb work the conductor actually loads.
`../train/model.py` holds the differentiable model that LEARNS the parameters;
this module only applies them, which is why it can stay a small numpy file with
no training dependency.

Two ways to apply a space:

  * CONVOLVE A MEASURED IMPULSE RESPONSE. Used wherever a real IR exists, so no
    fitting error enters the audio at all -- the fitted parameters are used only
    to CHOOSE which IR represents a category, never to reproduce it.
  * SYNTHESISE AN EXPONENTIAL-DECAY TAIL from fitted parameters. Used where no
    dry IR exists, such as acoustics measured from finished recordings.

Two details that matter more than they look:

  `tail=True` lets the reverb ring out PAST the input. Without it the decay is
  truncated the moment the input ends, which is precisely the part of a reverb
  a listener notices, and the effect becomes nearly inaudible.

  The wet level is set by RMS-MATCHING the wet signal to the dry over the
  region where the dry exists, rather than by a raw gain. The IR is
  energy-normalised, so absolute IR gain cancels and `wet` means the same
  perceptual thing across spaces of very different lengths.
"""

from __future__ import annotations

import numpy as np

SR = 16000

# Hand-set presets, kept for reference and for callers that want a quick knob.
# The measured ladder in `reverb_bank.py` supersedes these: every condition
# there comes from a bank fitted to data rather than from knob-guessing.
REVERB_PRESETS = {
    "dry": None,
    "subtle":    dict(decay_time_s=0.6, damping_hz=2500.0, wet=0.25, pre_delay_s=0.01),
    "room":      dict(decay_time_s=1.2, damping_hz=1500.0, wet=0.35, pre_delay_s=0.02),
    "hall":      dict(decay_time_s=2.5, damping_hz=1000.0, wet=0.45, pre_delay_s=0.04),
    "cathedral": dict(decay_time_s=4.5, damping_hz=700.0,  wet=0.55, pre_delay_s=0.08),
}


def load_mono(path, sr=SR, max_s=None):
    """Load audio as mono float32 at ``sr``."""
    import soundfile as sf
    x, srr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    if max_s:
        x = x[: int(max_s * srr)]
    if srr != sr and len(x):
        import scipy.signal as sps
        x = sps.resample(x, max(1, int(len(x) * sr / srr))).astype("float32")
    return np.asarray(x, dtype=np.float32)


def load_ir(path, sr=SR, max_s=6.0):
    """Load an impulse response, conditioned exactly as the fit conditions it.

    Trims leading silence to the direct-sound peak, crops, and normalises to
    unit energy. Lives on the inference side deliberately: the runtime needs
    this and must not have to import a fitting script to get it.
    """
    x = load_mono(path, sr=sr, max_s=None)
    if x.size == 0:
        return x
    x = x[int(np.argmax(np.abs(x))):]          # start at the direct sound
    x = x[: int(max_s * sr)]
    e = np.sqrt((x.astype(np.float64) ** 2).sum()) + 1e-12
    return (x / e).astype(np.float32)


def apply_reverb_np(audio, sr=SR, decay_time_s=1.2, damping_hz=1500.0, wet=0.30,
                    pre_delay_s=0.0, tail=True, seed=1234):
    """Synthesise and apply an exponential-decay reverb.

    The same IR form the differentiable model uses, so a fitted decay and
    damping render here as they did during fitting. ``pre_delay_s`` gaps the
    tail, which is a stronger cue of space. Returns float32, peak-safe.
    """
    import scipy.signal as sps

    audio = np.asarray(audio, dtype=np.float64)
    if wet <= 0 or audio.size == 0:
        return audio.astype(np.float32)
    n_in = len(audio)
    # Cap the tail at 5 s so an extreme preset cannot produce an absurd clip.
    ir_len = min(int(sr * 5.0), max(int(sr * 0.2), int(sr * decay_time_s * 3)))
    t = np.arange(ir_len) / sr
    ir = (np.random.default_rng(seed).normal(0, 1, ir_len)
          * np.exp(-t / max(decay_time_s, 1e-3)))
    sos = sps.butter(2, min(damping_hz, sr / 2 * 0.99) / (sr / 2),
                     btype="low", output="sos")
    ir = sps.sosfilt(sos, ir)
    fade = max(1, int(0.12 * ir_len))          # no click where the tail is capped
    ir[-fade:] *= np.linspace(1.0, 0.0, fade)
    pd = int(max(0.0, pre_delay_s) * sr)
    if pd:
        ir = np.concatenate([np.zeros(pd), ir])
    ir /= (np.sqrt((ir ** 2).sum()) + 1e-8)

    rev = sps.fftconvolve(audio, ir)           # full length, so it rings out
    rev *= ((np.sqrt(np.mean(audio ** 2)) + 1e-9)
            / (np.sqrt(np.mean(rev[:n_in] ** 2)) + 1e-9))
    n_out = len(rev) if tail else n_in
    dry = np.zeros(n_out)
    dry[:n_in] = audio
    out = dry * (1 - wet) + rev[:n_out] * wet
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 0.95:
        out *= 0.95 / peak
    return out.astype(np.float32)


def apply_ir_np(audio, ir, wet=0.5, tail=True):
    """Convolve with a measured impulse response, RMS-matched and peak-safe."""
    import scipy.signal as sps

    audio = np.asarray(audio, dtype=np.float64)
    ir = np.asarray(ir, dtype=np.float64)
    if wet <= 0 or audio.size == 0 or ir.size == 0:
        return audio.astype(np.float32)
    n_in = len(audio)
    rev = sps.fftconvolve(audio, ir)
    rev *= ((np.sqrt(np.mean(audio ** 2)) + 1e-9)
            / (np.sqrt(np.mean(rev[:n_in] ** 2)) + 1e-9))
    n_out = len(rev) if tail else n_in
    dry = np.zeros(n_out)
    dry[:n_in] = audio
    out = dry * (1 - wet) + rev[:n_out] * wet
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 0.95:
        out *= 0.95 / peak
    return out.astype(np.float32)


def apply_preset(audio, preset, sr=SR):
    """Apply a named entry from REVERB_PRESETS. 'dry' passes through."""
    p = REVERB_PRESETS.get(preset)
    if p is None:
        return np.asarray(audio, dtype=np.float32)
    return apply_reverb_np(audio, sr=sr, **p)
