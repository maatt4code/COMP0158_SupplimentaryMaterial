"""Closed-loop perceptual training of the (valence, arousal) -> theta mapper.

Step 4 of Section 3.4.1, and the experiment Section 4.1 reports. Instead of
minimising error in theta space, as the CVAE does, this trains the mapper so
that the RENDERED AUDIO judges back as the requested emotion:

    loss = || Proxy( Polish( DDSP( g(v, a) ) ) ) - (v, a) ||^2

Everything between g and the loss is differentiable: the DDSP synthesiser, a
torch port of the fixed-IR polish reverb, and the frozen judge proxy. Gradients
flow end to end into g.

This is the loop that reward-hacks. Optimising hard against a learned proxy
finds the regions where the proxy is confidently wrong -- a high-harmonic comb
that scores as "aroused" and sounds like a buzz. Three defences are implemented,
all on by default, and turning them off reproduces the failure:

  --anchor        pulls predicted theta toward the theta of the nearest labelled
                  clip in VA space, keeping g on the measured manifold. 0 disables.
  --theta-noise   jitters theta before rendering, so imperceptible micro-
                  modulation of the proxy can no longer carry the loss.
  --ensemble-var  a MOPO-style uncertainty penalty: the mean variance of the
                  ensemble's predictions. Members disagree where they are
                  extrapolating, so that region costs. Needs an ensemble from
                  train_judge_proxy.py --n-models N; a no-op with one member.

Design notes:
  - g is deterministic, with no latent, for a clean comparison against k-NN.
  - Targets are sampled over the REACHABLE region measured at labelling time
    (v in [-1, 0.3], a in [-1, 1]). Asking for unreachable coordinates would
    only teach g to saturate.
  - f0 is constant per clip. The quarter-tone wander in the preset bank is
    ornamental, not differentiable-friendly, and the judge barely hears it.
  - The saved checkpoint is the best 100-step moving average, not the final
    model: closed-loop runs peak mid-training and drift afterwards.

Final evaluation MUST use the real judge, never the proxy. The proxy is the
thing being gamed, so it cannot also be the referee:

  python evaluate_cycle_consistency.py --mode render --inverse mlp --data-dir ../data/cycle_mlp
  python label_dataset.py --data-dir ../data/cycle_mlp
  python evaluate_cycle_consistency.py --mode report --data-dir ../data/cycle_mlp

Output (in --out-dir):
  closed_loop_mapper.pt        best-moving-average checkpoint
  closed_loop_mapper_loss.png  perceptual-loss curve

Default output is ../data/mapper, not ../weights. See train_judge_proxy.py.

Run:
  python train_closed_loop.py --bank-dirs ../data/chord10k

Previous: train_judge_proxy.py
Next:     evaluate_cycle_consistency.py, then dagger_render.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sps

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "common"))
sys.path.insert(0, str(_HERE.parent))
from device import get_device                        # noqa: E402
from ddsp_synth import DifferentiableDDSPSynth       # noqa: E402
from train_judge_proxy import JudgeProxy             # noqa: E402

SR = 16000
CLIP_SEC = 9.0          # matches the proxy's crop length
N_HARM = 32
N_NOISE_BANDS = 65

# Reachable VA region measured at labelling time (left half-plane plus a sliver).
V_RANGE = (-1.0, 0.3)
A_RANGE = (-1.0, 1.0)

# Denormalisation ranges. Must match generate_preset_bank.py and train_cvae.py.
BASE_SCALAR_RANGES = [
    ("f0_hz",           32.7, 261.6, "log"),
    ("swell_rate",      0.01, 2.0,   "log"),
    ("swell_depth",     0.05, 0.9,   "lin"),
    ("noise_level",     0.0,  0.4,   "lin"),
    ("noise_cutoff_hz", 200.0, 6000.0, "log"),
]

CHORD_SCALAR_DEFS = [
    ("third_interval",  3.0, 4.0, "lin"),
    ("third_gain",      0.0, 1.0, "lin"),
    ("fifth_gain",      0.0, 0.8, "lin"),
    ("octave_gain",     0.0, 0.6, "lin"),
]


def load_proxies(weights_dir, device):
    """Load the proxy ensemble (judge_proxy_e*.pt) if present, else the single
    judge_proxy.pt. Returns frozen, eval-mode modules."""
    weights_dir = Path(weights_dir)
    member_paths = sorted(weights_dir.glob("judge_proxy_e*.pt"))
    if not member_paths:
        single = weights_dir / "judge_proxy.pt"
        if not single.exists():
            raise SystemExit(
                f"\nno judge proxy found in:\n    {weights_dir}\n\n"
                "Train one with train_judge_proxy.py, or point --proxy-dir at "
                "../weights, which ships the fitted proxy.\n")
        member_paths = [single]
    proxies = []
    for p in member_paths:
        ckpt = torch.load(p, map_location=device)
        m = JudgeProxy(size=ckpt.get("size", "small")).to(device)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        for prm in m.parameters():
            prm.requires_grad_(False)
        proxies.append(m)
    print(f"Proxy      : {len(proxies)} member(s) {[p.name for p in member_paths]}")
    return proxies


class Mapper(nn.Module):
    """g(v, a) -> theta. Same head structure as the CVAE decoder, no latent.

    Imported by evaluate_cycle_consistency.py and dagger_render.py.
    """

    def __init__(self, hidden=256, n_scalars=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_harm = nn.Linear(hidden, N_HARM)
        num_sc = n_scalars if n_scalars is not None else len(BASE_SCALAR_RANGES)
        self.head_scal = nn.Linear(hidden, num_sc)

    def forward(self, c):
        h = self.net(c)
        harm = torch.softmax(self.head_harm(h), dim=-1)
        scal01 = torch.sigmoid(self.head_scal(h))
        return harm, scal01


def denorm_scalars_torch(scal01: torch.Tensor, ranges) -> dict:
    out = {}
    for i, (name, lo, hi, scale) in enumerate(ranges):
        x = scal01[:, i]
        if scale == "log":
            out[name] = torch.exp(np.log(lo) + x * (np.log(hi) - np.log(lo)))
        else:
            out[name] = lo + x * (hi - lo)
    return out


def render_batch(synth, harm, scal01, n_samples, device, ranges):
    """Differentiable render: theta tensors -> audio batch (B, n_samples)."""
    B = harm.shape[0]
    sc = denorm_scalars_torch(scal01, ranges)
    t = torch.arange(n_samples, device=device, dtype=torch.float32) / SR

    f0 = sc["f0_hz"].view(B, 1, 1).expand(B, n_samples, 1)

    phase = 2 * np.pi * sc["swell_rate"].view(B, 1) * t.view(1, -1)
    swell = 1.0 - sc["swell_depth"].view(B, 1) * 0.5 * (1.0 + torch.sin(phase))
    amp = (swell * 0.8).unsqueeze(-1)

    hd = harm.view(B, 1, N_HARM).expand(B, n_samples, N_HARM)
    harmonic_part = synth.synthesize_harmonic(f0, amp, hd)

    if "third_gain" in sc:
        f0_third = f0 * (2.0 ** (sc["third_interval"].view(B, 1, 1) / 12.0))
        harmonic_part = harmonic_part + synth.synthesize_harmonic(
            f0_third, amp * sc["third_gain"].view(B, 1, 1), hd)
    if "fifth_gain" in sc:
        harmonic_part = harmonic_part + synth.synthesize_harmonic(
            f0 * (2.0 ** (7.0 / 12.0)), amp * sc["fifth_gain"].view(B, 1, 1), hd)
    if "octave_gain" in sc:
        harmonic_part = harmonic_part + synth.synthesize_harmonic(
            f0 * 2.0, amp * sc["octave_gain"].view(B, 1, 1), hd)

    # Noise band gains: a differentiable lowpass shape from cutoff and level.
    n_frames = n_samples // (N_NOISE_BANDS - 1)
    band_freqs = torch.linspace(0, SR / 2, N_NOISE_BANDS, device=device)
    gains = 1.0 / (1.0 + (band_freqs.view(1, -1)
                          / sc["noise_cutoff_hz"].view(B, 1)) ** 4)
    gains = gains * sc["noise_level"].view(B, 1)
    noise = gains.view(B, 1, N_NOISE_BANDS).expand(
        B, n_frames, N_NOISE_BANDS).contiguous()
    noise_part = synth.synthesize_noise(noise)

    # synthesize_noise can differ in length by a hop; trim both to match.
    min_len = min(harmonic_part.shape[1], noise_part.shape[1])
    harmonic_part = harmonic_part[:, :min_len]
    noise_part = noise_part[:, :min_len] * amp[:, :min_len, 0] * 0.20

    mixed = harmonic_part + noise_part
    return mixed / (torch.max(torch.abs(mixed), dim=1, keepdim=True)[0] + 1e-8)


class TorchPolish(nn.Module):
    """The fixed dark-reverb polish chain as a differentiable torch op.

    Same IR seed (1234) as Section 3.3's ``polish``, so the mapper is trained
    through the same virtual room the training data was rendered in.
    """

    def __init__(self, device, tail_sec=6.0, wet=0.35):
        super().__init__()
        self.wet = wet
        ir_len = int(SR * tail_sec)
        decay = np.exp(-np.linspace(0, 6, ir_len))
        rng = np.random.default_rng(1234)
        ir = rng.normal(0, 1, ir_len) * decay
        sos = sps.butter(2, 600.0 / (SR / 2), btype="low", output="sos")
        ir = sps.sosfilt(sos, ir)
        ir /= np.sqrt(np.sum(ir ** 2)) + 1e-8
        self.register_buffer("ir", torch.tensor(ir, dtype=torch.float32, device=device))

    def forward(self, audio):                       # (B, n)
        n = audio.shape[1]
        L = n + self.ir.shape[0]
        A = torch.fft.rfft(audio, n=L)
        H = torch.fft.rfft(self.ir, n=L)
        rev = torch.fft.irfft(A * H.unsqueeze(0), n=L)[:, :n]
        dry_rms = torch.sqrt(torch.mean(audio ** 2, dim=1, keepdim=True)) + 1e-8
        wet_rms = torch.sqrt(torch.mean(rev ** 2, dim=1, keepdim=True)) + 1e-8
        rev = rev * dry_rms / wet_rms
        out = audio * (1 - self.wet) + rev * self.wet
        peak = torch.amax(torch.abs(out), dim=1, keepdim=True) + 1e-8
        return torch.where(peak > 0.95, out * 0.95 / peak, out)


def normalize_scalars_np(df: pd.DataFrame, ranges) -> np.ndarray:
    cols = []
    for name, lo, hi, scale in ranges:
        x = df[name].to_numpy(dtype=np.float64)
        if scale == "log":
            x = (np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo))
        else:
            x = (x - lo) / (hi - lo)
        cols.append(np.clip(x, 0.0, 1.0))
    return np.stack(cols, axis=1)


def load_anchor_bank(dirs, ranges, device):
    """Labelled (VA, theta) bank for the manifold anchor, all on device."""
    frames = []
    needed = [name for name, *_ in ranges]
    for d in dirs:
        idx = Path(d) / "labeled_index.csv"
        if not idx.exists():
            raise SystemExit(
                f"\nlabelled index not found at:\n    {idx}\n\n"
                "Run label_dataset.py on that directory, or pass --anchor 0 to "
                "train without the manifold anchor (which reward-hacks).\n")
        f = pd.read_csv(idx)
        missing = [c for c in needed if c not in f.columns]
        if missing:
            raise SystemExit(
                f"\nanchor bank {d} lacks {missing}.\n"
                "Do not mix drone-only and chord-era banks in --bank-dirs.\n")
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    va = torch.tensor(df[["valence", "arousal"]].to_numpy(dtype=np.float32),
                      device=device)
    harm = torch.tensor(df[[f"h{k+1:02d}" for k in range(N_HARM)]]
                        .to_numpy(dtype=np.float32), device=device)
    scal = torch.tensor(normalize_scalars_np(df, ranges).astype(np.float32),
                        device=device)
    print(f"Anchor bank: {len(df)} labelled presets")
    return va, harm, scal


def main():
    ap = argparse.ArgumentParser(
        description="Train the VA->theta mapper against the frozen judge proxy.")
    ap.add_argument("--bank-dirs", dest="bank_dirs", nargs="+", required=True,
                    metavar="DIR", help="labelled bank(s) for the manifold anchor")
    ap.add_argument("--proxy-dir", dest="proxy_dir", default=None, metavar="DIR",
                    help="where to read the judge proxy (default ../weights)")
    ap.add_argument("--out-dir", dest="out_dir", default=None, metavar="DIR",
                    help="checkpoint directory (default ../data/mapper)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor", type=float, default=0.5,
                    help="weight of the manifold anchor. 0 disables it, which "
                         "is how the reward-hacking failure is reproduced")
    ap.add_argument("--theta-noise", dest="theta_noise", type=float, default=0.02,
                    help="std of jitter applied to theta before rendering. "
                         "0 disables")
    ap.add_argument("--ensemble-var", dest="ensemble_var", type=float, default=1.0,
                    help="weight of the ensemble-disagreement penalty. "
                         "No-op with a single proxy. 0 disables")
    args = ap.parse_args()

    proxy_dir = Path(args.proxy_dir) if args.proxy_dir else _HERE.parents[1] / "weights"
    out_dir = Path(args.out_dir) if args.out_dir else _HERE.parents[1] / "data" / "mapper"
    out_dir.mkdir(parents=True, exist_ok=True)
    mapper_ckpt = out_dir / "closed_loop_mapper.pt"

    # Chord columns are detected from the bank, never assumed. The mapper's
    # output width follows, so this must happen before the model is built.
    scalar_ranges = list(BASE_SCALAR_RANGES)
    for d in args.bank_dirs:
        idx_path = Path(d) / "labeled_index.csv"
        if idx_path.exists():
            cols = pd.read_csv(idx_path, nrows=1).columns
            detected = [item for item in CHORD_SCALAR_DEFS if item[0] in cols]
            if detected:
                print(f"Chord-era bank detected; adding {[x[0] for x in detected]}")
                scalar_ranges.extend(detected)
                break

    device = get_device()
    torch.manual_seed(args.seed)

    proxies = load_proxies(proxy_dir, device)
    use_var = args.ensemble_var > 0 and len(proxies) > 1
    if args.ensemble_var > 0 and len(proxies) == 1:
        print("NOTE: --ensemble-var is set but only one proxy was found, so the "
              "variance penalty is a no-op.\n      Train an ensemble with "
              "train_judge_proxy.py --n-models N to enable it.")

    bank_va = bank_harm = bank_scal = None
    if args.anchor > 0:
        bank_va, bank_harm, bank_scal = load_anchor_bank(
            args.bank_dirs, scalar_ranges, device)

    synth = DifferentiableDDSPSynth(sample_rate=SR).to(device)
    polish = TorchPolish(device).to(device)
    mapper = Mapper(n_scalars=len(scalar_ranges)).to(device)
    opt = torch.optim.Adam(mapper.parameters(), lr=args.lr)
    # Cosine decay: at constant LR the mapper overshoots and drifts off-manifold.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    n_samples = int(CLIP_SEC * SR)
    hist, hist_var = [], []
    best_ma = float("inf")
    print(f"Output     : {out_dir}")

    for step in range(1, args.steps + 1):
        v = torch.rand(args.batch, 1, device=device) * (V_RANGE[1] - V_RANGE[0]) + V_RANGE[0]
        a = torch.rand(args.batch, 1, device=device) * (A_RANGE[1] - A_RANGE[0]) + A_RANGE[0]
        c = torch.cat([v, a], dim=1)

        harm, scal01 = mapper(c)

        # Render a jittered copy; the anchor below still sees the clean theta.
        harm_r, scal_r = harm, scal01
        if args.theta_noise > 0:
            scal_r = (scal01 + torch.randn_like(scal01) * args.theta_noise).clamp(0.0, 1.0)
            harm_r = torch.softmax(torch.log(harm + 1e-8)
                                   + torch.randn_like(harm) * args.theta_noise * 5.0,
                                   dim=-1)

        audio = polish(render_batch(synth, harm_r, scal_r, n_samples, device,
                                    scalar_ranges))

        # Every member judges the same audio. The mean drives the perceptual
        # loss; cross-member variance is the uncertainty penalty.
        member_preds = torch.stack([p(audio) for p in proxies], dim=0)   # (M,B,2)
        va_pred = member_preds.mean(dim=0)
        loss_percep = nn.functional.mse_loss(va_pred, c)
        loss_var = member_preds.var(dim=0, unbiased=False).mean()

        loss = loss_percep
        if use_var:
            loss = loss + args.ensemble_var * loss_var
        if args.anchor > 0:
            d = torch.cdist(c, bank_va)
            nn_idx = d.argmin(dim=1)
            loss_anchor = (nn.functional.mse_loss(harm, bank_harm[nn_idx]) * N_HARM
                           + nn.functional.mse_loss(scal01, bank_scal[nn_idx])
                           * len(scalar_ranges))
            loss = loss + args.anchor * loss_anchor

        opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        hist.append(loss_percep.item())
        hist_var.append(loss_var.item())

        # Checkpoint the best moving average on the COMBINED objective, so a
        # low-MSE-but-high-disagreement model is not saved -- that is exactly
        # the out-of-distribution corner being avoided.
        if step >= 100:
            sel = np.mean(hist[-100:])
            if use_var:
                sel = sel + args.ensemble_var * np.mean(hist_var[-100:])
            if float(sel) < best_ma:
                best_ma = float(sel)
                torch.save({"state_dict": mapper.state_dict(),
                            "scalar_ranges": scalar_ranges, "clip_sec": CLIP_SEC,
                            "best_ma": best_ma, "at_step": step}, mapper_ckpt)

        if step == 1 or step % 100 == 0:
            var_str = f"  ens-var {np.mean(hist_var[-100:]):.4f}" if use_var else ""
            ma_str = f"{best_ma:.4f}" if best_ma < float("inf") else "n/a"
            print(f"step {step:5d}  perceptual MSE {np.mean(hist[-100:]):.4f}"
                  f"{var_str}  (best MA {ma_str})")

    if best_ma == float("inf"):
        print(f"\nNo checkpoint written: --steps was {args.steps}, and the best "
              "moving average is only tracked from step 100.")
        return

    print(f"\nSaved best checkpoint (MA {best_ma:.4f}): {mapper_ckpt}")
    plt.figure(figsize=(7, 4))
    plt.plot(np.convolve(hist, np.ones(50) / 50, mode="valid"))
    plt.xlabel("step"); plt.ylabel("perceptual MSE (50-step MA)"); plt.grid(alpha=0.3)
    plt.title("Closed-loop training against the frozen judge proxy")
    plt.savefig(str(mapper_ckpt).replace(".pt", "_loss.png"), dpi=150,
                bbox_inches="tight")


if __name__ == "__main__":
    main()
