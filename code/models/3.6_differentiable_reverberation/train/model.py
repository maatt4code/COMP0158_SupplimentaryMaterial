"""A differentiable reverb in PyTorch, and the loss that fits it.

Training code. `../inference/reverb.py` applies the parameters this learns; it
carries no torch dependency, which is why the split exists.

This project's DDSP layer is a torch reimplementation and Magenta's TensorFlow
`ddsp` is not installed, so the trainable reverb is written here directly: a
differentiable FFT-convolution reverb whose impulse response is a learnable
parameter, fitted to a target by a multi-scale STFT loss. It is the torch
analogue of `ddsp.effects.Reverb`, with the same objective.

Two parameterisations, for two different jobs:

  ExpDecayReverb  interpretable {gain, decay_time_s, damping_hz}. The learned
                  numbers are the reverb control space -- they mean something
                  a person can read, and they map onto the knobs the numpy
                  applier exposes.
  FreeIRReverb    a fully free impulse response of fixed length. More
                  expressive, and the literal analogue of ddsp's Reverb, but
                  its parameters are not interpretable.

An identifiability note worth knowing before trusting a fit: magnitude-only
STFT supervision leaves a gain/damping degeneracy, because making the tail
quieter and making it darker both reduce high-frequency energy. Direct
supervision in IR space breaks it, which is what `fit_ir` is for and what the
selftest checks. The general, tail-agnostic objective is still the STFT loss.

Run:
  python model.py --selftest
  python model.py --distil-baseline out.pt
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000


def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


def _inv_softplus(y):
    """Stable inverse of softplus. Avoids overflow for large y, such as a
    damping cutoff in the kilohertz."""
    return y + math.log(-math.expm1(-y))


def fft_convolve(dry, ir):
    """Linear convolution via rFFT, truncated to the input length."""
    n = dry.shape[-1] + ir.shape[-1] - 1
    nfft = _next_pow2(n)
    Y = torch.fft.rfft(dry, nfft) * torch.fft.rfft(ir, nfft)
    return torch.fft.irfft(Y, nfft)[..., : dry.shape[-1]]


def multiscale_stft_loss(a, b, fft_sizes=(256, 512, 1024, 2048)):
    """L1 on linear and log STFT magnitudes, summed over several window sizes.

    The log term is what makes the tail matter: a reverb's decay lives tens of
    dB below the direct sound, and a purely linear magnitude loss would be
    dominated by the onset and barely see the part being fitted.
    """
    loss = a.new_zeros(())
    for n in fft_sizes:
        win = torch.hann_window(n, device=a.device)
        A = torch.stft(a, n, hop_length=n // 4, window=win,
                       return_complex=True).abs()
        B = torch.stft(b, n, hop_length=n // 4, window=win,
                       return_complex=True).abs()
        loss = loss + (A - B).abs().mean()
        loss = loss + (torch.log(A + 1e-5) - torch.log(B + 1e-5)).abs().mean()
    return loss


class ExpDecayReverb(nn.Module):
    """Interpretable reverb: gain * exp(-t/decay) * noise, soft-lowpassed at a
    learnable damping cutoff.

    The noise buffer is fixed and seeded: it is the diffuse tail's "room shape",
    not something to learn. Learning it as well is what FreeIRReverb does, at
    the cost of interpretability.

    All three parameters are stored unconstrained and mapped through
    softplus/sigmoid, so the optimiser never has to respect a bound. Damping is
    carried on a kilohertz scale so its raw parameter is order 1, matching the
    others' gradient scale -- in raw hertz it would train far too slowly under
    any single learning rate.
    """

    def __init__(self, ir_len_s=0.5, sr=SR, seed=1234,
                 init_decay_s=0.3, init_damp_hz=1500.0, init_gain=0.5):
        super().__init__()
        self.sr = sr
        self.ir_len = int(sr * ir_len_s)
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("noise", torch.randn(self.ir_len, generator=g))
        self.register_buffer("t", torch.arange(self.ir_len) / sr)
        self.register_buffer("freqs", torch.fft.rfftfreq(self.ir_len, 1.0 / sr))
        self.raw_decay = nn.Parameter(torch.tensor(_inv_softplus(init_decay_s)))
        self.raw_damp = nn.Parameter(
            torch.tensor(_inv_softplus(init_damp_hz / 1000.0)))
        self.raw_gain = nn.Parameter(
            torch.tensor(math.log(init_gain / (1 - init_gain))))

    @property
    def decay_time_s(self):
        return F.softplus(self.raw_decay).item()

    @property
    def damping_hz(self):
        return (1000.0 * F.softplus(self.raw_damp)).item()

    @property
    def gain(self):
        return torch.sigmoid(self.raw_gain).item()

    def ir(self):
        decay = F.softplus(self.raw_decay) + 1e-3
        x = self.noise * torch.exp(-self.t / decay)
        cutoff = 1000.0 * F.softplus(self.raw_damp) + 50.0
        mask = 1.0 / torch.sqrt(1.0 + (self.freqs / cutoff) ** 4)   # ~2nd order
        x = torch.fft.irfft(torch.fft.rfft(x) * mask, n=self.ir_len)
        return torch.sigmoid(self.raw_gain) * x

    def wet(self, dry):
        return fft_convolve(dry, self.ir())

    def forward(self, dry, wet_mix=0.35):
        rev = self.wet(dry)
        rev = rev * (dry.pow(2).mean().sqrt() / (rev.pow(2).mean().sqrt() + 1e-8))
        return dry * (1 - wet_mix) + rev * wet_mix


class FreeIRReverb(nn.Module):
    """A fully free impulse response. A light exponential window keeps the tail
    causal and decaying at initialisation."""

    def __init__(self, ir_len_s=0.5, sr=SR, seed=1234):
        super().__init__()
        self.sr = sr
        self.ir_len = int(sr * ir_len_s)
        g = torch.Generator().manual_seed(seed)
        t = torch.arange(self.ir_len) / sr
        self.ir_raw = nn.Parameter(
            torch.randn(self.ir_len, generator=g) * torch.exp(-t / 0.3) * 0.1)

    def ir(self):
        return self.ir_raw

    def wet(self, dry):
        return fft_convolve(dry, self.ir())

    def forward(self, dry, wet_mix=0.35):
        rev = self.wet(dry)
        rev = rev * (dry.pow(2).mean().sqrt() / (rev.pow(2).mean().sqrt() + 1e-8))
        return dry * (1 - wet_mix) + rev * wet_mix


def fit(model, dry, target_wet, steps=200, lr=0.05, verbose=False):
    """Fit so model.wet(dry) matches target_wet. Returns the loss history."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for i in range(steps):
        opt.zero_grad()
        loss = multiscale_stft_loss(model.wet(dry), target_wet)
        loss.backward()
        opt.step()
        hist.append(loss.item())
        if verbose and i % 25 == 0:
            print(f"  step {i:4d}  loss {loss.item():.4f}")
    return hist


def fit_ir(model, target_ir, steps=400, lr=0.05):
    """Fit by direct MSE in IR space.

    Only meaningful when the model and the target share the same diffuse tail,
    such as an identifiability check or distilling a reverb whose noise seed is
    known. The general, tail-agnostic objective is the STFT loss in `fit`.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        loss = (model.ir() - target_ir).pow(2).mean()
        loss.backward()
        opt.step()
        hist.append(loss.item())
    return hist


def scipy_baseline_wet(dry_np, sr=SR):
    """The fixed scipy reverb used elsewhere in the pipeline, as a teacher."""
    import numpy as np
    import scipy.signal as sps
    tail = np.exp(-np.linspace(0, 6, int(sr * 6)))
    ir = np.random.default_rng(1234).normal(0, 1, int(sr * 6)) * tail
    sos = sps.butter(2, 600.0 / (sr / 2), btype="low", output="sos")
    ir = sps.sosfilt(sos, ir)
    ir /= (np.sqrt((ir ** 2).sum()) + 1e-8)
    return sps.fftconvolve(dry_np, ir)[: len(dry_np)]


def selftest():
    import numpy as np
    torch.manual_seed(0)
    sr = SR
    # Excite with an impulse, so wet(dry) IS the impulse response. That is how
    # a reverb is identified in practice, and it is far better conditioned than
    # noise excitation.
    dry = torch.zeros(int(sr * 0.5))
    dry[0] = 1.0

    true = ExpDecayReverb(ir_len_s=0.4, sr=sr, seed=99, init_decay_s=0.25,
                          init_damp_hz=900.0, init_gain=0.6)
    with torch.no_grad():
        target = true.wet(dry)
        target_ir = true.ir().detach()

    m = ExpDecayReverb(ir_len_s=0.4, sr=sr, seed=99, init_decay_s=0.6,
                       init_damp_hz=3000.0, init_gain=0.3)
    with torch.no_grad():
        loss0 = multiscale_stft_loss(m.wet(dry), target).item()
    loss1 = fit(m, dry, target, steps=400, lr=0.1)[-1]
    assert math.isfinite(loss1), "loss went non-finite"
    assert loss1 < 0.7 * loss0, f"STFT loss barely moved: {loss0:.3f} -> {loss1:.3f}"
    print(f"  PASS  the shipped STFT objective trains: {loss0:.2f} -> {loss1:.2f} "
          f"({100 * (1 - loss1 / loss0):.0f}% down)")

    # Identifiability: with the tail shared, IR-space supervision recovers all
    # three interpretable parameters. Magnitude-only STFT alone would not.
    m_id = ExpDecayReverb(ir_len_s=0.4, sr=sr, seed=99, init_decay_s=0.6,
                          init_damp_hz=3000.0, init_gain=0.3)
    fit_ir(m_id, target_ir, steps=500, lr=0.05)
    assert abs(m_id.decay_time_s - 0.25) < 0.06, m_id.decay_time_s
    assert abs(m_id.damping_hz - 900.0) < 300.0, m_id.damping_hz
    assert abs(m_id.gain - 0.60) < 0.15, m_id.gain
    print(f"  PASS  IR-supervised fit recovers decay {m_id.decay_time_s:.2f}s, "
          f"damping {m_id.damping_hz:.0f}Hz, gain {m_id.gain:.2f} "
          f"(true 0.25s / 900Hz / 0.60)")

    free = FreeIRReverb(ir_len_s=0.4, sr=sr)
    fh = fit(free, dry, target, steps=200, lr=0.02)
    assert fh[-1] < fh[0], "FreeIRReverb did not improve"
    print(f"  PASS  the free-IR parameterisation also trains: "
          f"{fh[0]:.2f} -> {fh[-1]:.2f}")
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser(description="Differentiable reverb in torch.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--distil-baseline", dest="distil_baseline", metavar="OUT.pt",
                    help="fit this reverb to the fixed scipy baseline and save")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.distil_baseline:
        import numpy as np
        dry_np = np.zeros(int(SR * 6.0), dtype=np.float32)
        dry_np[0] = 1.0
        dry = torch.tensor(dry_np)
        target = torch.tensor(scipy_baseline_wet(dry_np), dtype=torch.float32)
        model = ExpDecayReverb(ir_len_s=6.0)
        fit(model, dry, target, steps=args.steps, lr=0.1, verbose=True)
        torch.save({"raw_decay": model.raw_decay, "raw_damp": model.raw_damp,
                    "raw_gain": model.raw_gain,
                    "decay_time_s": model.decay_time_s,
                    "damping_hz": model.damping_hz, "gain": model.gain},
                   args.distil_baseline)
        print(f"distilled -> {args.distil_baseline}: "
              f"decay {model.decay_time_s:.2f}s, "
              f"damping {model.damping_hz:.0f}Hz, gain {model.gain:.2f}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
