"""Inverse CVAE: (valence, arousal) -> theta, trained in parameter space.

Step 2 of Section 3.4.1, and the first of the two inverse models the section
compares. It is trained to match theta directly, which is the objective mismatch
that ``train_closed_loop.py`` then tries to fix: matching parameters is not the
same as sounding like the requested emotion.

Why a CVAE rather than a regressor: the inverse problem is one-to-many. Many
different parameter sets render to audio the judge scores at the same
coordinate. An MLP trained with MSE averages those modes into one blurred
preset; a CVAE keeps them, so distinct plausible presets can be drawn per
coordinate by sampling z.

theta is split by type:
  harm_dist (32)  a point on the simplex   -> softmax head
  scalars         normalised to [0,1] with the SAME ranges the preset bank was
                  sampled from (log scale for f0_hz, swell_rate, noise_cutoff_hz)
                  -> sigmoid head

Chord-era columns are picked up automatically when the index has them, so a
drone-only bank still trains.

Output (in --out-dir):
  inverse_cvae.pt   weights, config and the normalisation metadata
  cvae_loss.png     training curves

The default output is this section's ``data/``, not ``weights/``. See the note
in train_judge_proxy.py: ``weights/`` holds only what ships.

Run:
  python train_cvae.py --data-dir ../data/chord10k

  # sanity check a trained model: print presets across the reachable manifold
  python train_cvae.py --data-dir ../data/chord10k --sample-only

Previous: label_dataset.py
Next:     evaluate_cycle_consistency.py
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
matplotlib.use("Agg")   # headless: avoids a Qt/OpenMP clash with torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "common"))
from device import get_device                       # noqa: E402

N_HARM = 32
SCALARS = ["f0_hz", "tilt", "swell_rate", "swell_depth", "noise_level", "noise_cutoff_hz"]
# Chord parameters, present only in chord-era banks.
CHORD_SCALARS = ["third_interval", "third_gain", "fifth_gain", "octave_gain"]

# MUST match the sampling ranges in Section 3.3's generate_preset_bank.py.
SCALAR_RANGES = {
    "f0_hz":           (32.7, 261.6, "log"),
    "tilt":            (0.0, 1.5, "lin"),
    "swell_rate":      (0.01, 2.0, "log"),
    "swell_depth":     (0.05, 0.9, "lin"),
    "noise_level":     (0.0, 0.4, "lin"),
    "noise_cutoff_hz": (200.0, 6000.0, "log"),
    "third_interval":  (3.0, 4.0, "lin"),
    "third_gain":      (0.0, 1.0, "lin"),
    "fifth_gain":      (0.0, 0.8, "lin"),
    "octave_gain":     (0.0, 0.6, "lin"),
}


def normalize_scalars(df: pd.DataFrame, scalars=None) -> np.ndarray:
    cols = []
    for name in (scalars or SCALARS):
        lo, hi, scale = SCALAR_RANGES[name]
        x = df[name].to_numpy(dtype=np.float64)
        if scale == "log":
            x = (np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo))
        else:
            x = (x - lo) / (hi - lo)
        cols.append(np.clip(x, 0.0, 1.0))
    return np.stack(cols, axis=1)


def denormalize_scalars(x01: np.ndarray, scalars=None) -> dict:
    out = {}
    for i, name in enumerate(scalars or SCALARS):
        lo, hi, scale = SCALAR_RANGES[name]
        if scale == "log":
            out[name] = np.exp(np.log(lo) + x01[..., i] * (np.log(hi) - np.log(lo)))
        else:
            out[name] = lo + x01[..., i] * (hi - lo)
    return out


class InverseCVAE(nn.Module):
    def __init__(self, latent_dim=8, cond_dim=2, hidden=256, n_scalars=len(SCALARS)):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_scalars = n_scalars
        in_dim = N_HARM + n_scalars + cond_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_harm = nn.Linear(hidden, N_HARM)       # -> softmax (simplex)
        self.head_scalar = nn.Linear(hidden, n_scalars)  # -> sigmoid ([0,1])

    def encode(self, theta, c):
        h = self.encoder(torch.cat([theta, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, c):
        h = self.decoder(torch.cat([z, c], dim=1))
        harm = torch.softmax(self.head_harm(h), dim=-1)
        scal = torch.sigmoid(self.head_scalar(h))
        return harm, scal

    def forward(self, theta, c):
        mu, logvar = self.encode(theta, c)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        harm, scal = self.decode(z, c)
        return harm, scal, mu, logvar


def loss_fn(harm_pred, scal_pred, harm_true, scal_true, mu, logvar, beta):
    # The 32 harmonic dims are small-valued; upweighting keeps them from being
    # drowned out by the handful of scalars.
    l_harm = nn.functional.mse_loss(harm_pred, harm_true, reduction="mean") * N_HARM
    l_scal = nn.functional.mse_loss(scal_pred, scal_true, reduction="mean") * scal_true.shape[1]
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return l_harm + l_scal + beta * kld, l_harm, l_scal, kld


def main():
    ap = argparse.ArgumentParser(
        description="Train the inverse CVAE mapping (valence, arousal) to theta.")
    ap.add_argument("--data-dir", dest="data_dir", required=True, metavar="DIR")
    ap.add_argument("--index-name", dest="index_name", default="labeled_index.csv")
    ap.add_argument("--out-dir", dest="out_dir", default=None, metavar="DIR",
                    help="checkpoint directory (default ../data/cvae). Not "
                         "../weights: that holds the checkpoints that ship")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint to read for --sample-only "
                         "(default ../weights/inverse_cvae.pt)")
    ap.add_argument("--latent-dim", dest="latent_dim", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-only", dest="sample_only", action="store_true",
                    help="load a trained model and print presets on a VA grid")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _HERE.parents[1] / "data" / "cvae"
    device = get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    index_path = Path(args.data_dir) / args.index_name
    if not index_path.exists():
        raise SystemExit(
            f"\nlabelled index not found at:\n    {index_path}\n\n"
            "Run label_dataset.py on that directory first.\n")

    df = pd.read_csv(index_path)
    harm = df[[f"h{k+1:02d}" for k in range(N_HARM)]].to_numpy(dtype=np.float32)
    scalars = SCALARS + [c for c in CHORD_SCALARS if c in df.columns]
    if len(scalars) > len(SCALARS):
        print(f"Chord-era bank: theta includes {scalars[len(SCALARS):]}")
    scal = normalize_scalars(df, scalars).astype(np.float32)
    cond = df[["valence", "arousal"]].to_numpy(dtype=np.float32)
    print(f"Training pairs: {len(df)}  (VA coverage: "
          f"v [{cond[:,0].min():+.2f},{cond[:,0].max():+.2f}] "
          f"a [{cond[:,1].min():+.2f},{cond[:,1].max():+.2f}])")

    if args.sample_only:
        ckpt_path = (Path(args.ckpt) if args.ckpt
                     else _HERE.parents[1] / "weights" / "inverse_cvae.pt")
        if not ckpt_path.exists():
            raise SystemExit(f"\ncheckpoint not found at:\n    {ckpt_path}\n")
        ckpt = torch.load(ckpt_path, map_location=device)
        scalars = ckpt.get("scalars", SCALARS)
        model = InverseCVAE(latent_dim=ckpt["latent_dim"],
                            n_scalars=len(scalars)).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"\nPresets across the reachable manifold, z=0 ({ckpt_path.name}):")
        for v in [-0.9, -0.5, -0.1]:
            for a in [-0.8, 0.0, 0.8]:
                c = torch.tensor([[v, a]], dtype=torch.float32, device=device)
                z = torch.zeros(1, ckpt["latent_dim"], device=device)
                with torch.no_grad():
                    h, s = model.decode(z, c)
                sc = denormalize_scalars(s.cpu().numpy(), scalars)
                chord = (f", third {sc['third_interval'].item():.1f}st "
                         f"g{sc['third_gain'].item():.2f}"
                         if "third_gain" in sc else "")
                print(f"  V{v:+.1f} A{a:+.1f} -> f0 {sc['f0_hz'].item():6.1f} Hz, "
                      f"tilt {sc['tilt'].item():.2f}, "
                      f"swell {sc['swell_rate'].item():.3f} Hz, "
                      f"noise {sc['noise_level'].item():.2f}, "
                      f"h1 {h[0,0].item():.2f}{chord}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "inverse_cvae.pt"
    print(f"Output     : {out_dir}")
    model = InverseCVAE(latent_dim=args.latent_dim, n_scalars=len(scalars)).to(device)

    n = len(df)
    idx = np.random.permutation(n)
    n_val = max(1, int(0.1 * n))
    tr, va = idx[n_val:], idx[:n_val]

    def to_loader(ix, shuffle):
        ds = TensorDataset(torch.tensor(harm[ix]), torch.tensor(scal[ix]),
                           torch.tensor(cond[ix]))
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle)

    train_loader, val_loader = to_loader(tr, True), to_loader(va, False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist_tr, hist_va = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for h_b, s_b, c_b in train_loader:
            h_b, s_b, c_b = h_b.to(device), s_b.to(device), c_b.to(device)
            theta_b = torch.cat([h_b, s_b], dim=1)
            hp, sp, mu, logvar = model(theta_b, c_b)
            loss, *_ = loss_fn(hp, sp, h_b, s_b, mu, logvar, args.beta)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(h_b)
        hist_tr.append(tot / len(tr))

        model.eval()
        tot = 0.0
        with torch.no_grad():
            for h_b, s_b, c_b in val_loader:
                h_b, s_b, c_b = h_b.to(device), s_b.to(device), c_b.to(device)
                theta_b = torch.cat([h_b, s_b], dim=1)
                hp, sp, mu, logvar = model(theta_b, c_b)
                loss, *_ = loss_fn(hp, sp, h_b, s_b, mu, logvar, args.beta)
                tot += loss.item() * len(h_b)
        hist_va.append(tot / len(va))

        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch {epoch:3d}/{args.epochs}  train {hist_tr[-1]:.4f}  "
                  f"val {hist_va[-1]:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "latent_dim": args.latent_dim,
        "scalar_ranges": SCALAR_RANGES,
        "scalars": scalars,
        "n_harm": N_HARM,
        "beta": args.beta,
        "index_used": str(index_path),
    }, ckpt_path)
    print(f"\nSaved: {ckpt_path}")

    plt.figure(figsize=(7, 4))
    plt.plot(hist_tr, label="train")
    plt.plot(hist_va, label="val", ls="--")
    plt.xlabel("epoch"); plt.ylabel("loss (recon + beta*KL)")
    plt.title("Inverse CVAE training")
    plt.legend(); plt.grid(alpha=0.3)
    loss_png = out_dir / "cvae_loss.png"
    plt.savefig(loss_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {loss_png}")


if __name__ == "__main__":
    main()
