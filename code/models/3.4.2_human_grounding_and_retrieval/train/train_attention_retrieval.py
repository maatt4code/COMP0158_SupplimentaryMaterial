"""Train the learned-attention retrieval engine, self-supervised on the bank.

Step 5 of Section 3.4.2, and the third of the continuous retrieval engines that
``../inference/retrieval_engines.py`` exposes. It produces
``../weights/attn_retrieval.pt``, which ships.

The engine learns a VA-similarity metric and a soft assignment over the bank's
rows. The supervision is the BANK ITSELF, not any external labels: for each row
i, feed its VA, attend over all OTHER rows with row i masked out, blend their
theta, and train the blend to reconstruct row i's theta.

Leave-self-out is the whole design. Without the mask the network would learn to
retrieve the row it was given and score perfectly while learning nothing; with
it, the metric has to generalise, so at deploy time a genuinely new VA target
returns an interpolation rather than one memorised clip.

theta is standardised per column before the loss, because the 32 harmonic
weights sit around 0.03 while f0 is in the tens of hertz -- unstandardised, the
scalars would dominate the gradient entirely. The mean and standard deviation go
into the checkpoint and are undone at inference by AttnEngine.

Validation rows are used only as QUERIES and are never masked out of the
attended set, which is what makes the validation loss a measure of new-query
interpolation rather than of reconstruction.

Output (in --out-dir):
  attn_retrieval.pt   weights, key/query dimension, and the theta normalisation

Default output is ../data/attention, not ../weights. See the note in Section
3.4.1's training scripts: weights/ holds the artefacts that ship.

Run:
  python train_attention_retrieval.py \
      --bank-index <bank>/labeled_index.csv <bank2>/labeled_index.csv \
      --epochs 60

Previous: propagate_labels_krr.py (optional; the engine works on either axis)
Next:     ../inference/retrieval_engines.py loads the checkpoint
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_SECTION = _HERE.parents[1]
sys.path.insert(0, str(_HERE.parents[3] / "common"))
sys.path.insert(0, str(_SECTION / "inference"))
from device import get_device                       # noqa: E402
import retrieval_engines as engines                 # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Train the learned-attention retrieval engine.")
    ap.add_argument("--bank-index", dest="bank_index", nargs="+", required=True,
                    metavar="CSV",
                    help="labelled_index.csv file(s) forming the retrieval bank")
    ap.add_argument("--d", type=int, default=64, help="key/query dimension")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.1,
                    help="bank rows held out as queries only; they stay in the "
                         "attended set, so this measures interpolation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", dest="out_dir", default=None,
                    help="default ../data/attention. Not ../weights")
    args = ap.parse_args()

    present = [p for p in args.bank_index if Path(p).exists()]
    if not present:
        raise SystemExit(
            "\nno bank index found. Tried:\n"
            + "\n".join(f"    {p}" for p in args.bank_index) + "\n")
    out_dir = (Path(args.out_dir) if args.out_dir
               else _SECTION / "data" / "attention")

    device = get_device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    bank = engines.build_bank(present)
    print(f"Attention bank: {bank['n']} clips; scalars={bank['scalar_keys']}")

    theta = np.concatenate([bank["harm"], bank["scal"]], axis=1)
    y_mean, y_std = theta.mean(0), theta.std(0) + 1e-8
    theta_norm = (theta - y_mean) / y_std

    va_t = torch.tensor(bank["va"], dtype=torch.float32, device=device)
    theta_t = torch.tensor(theta_norm, dtype=torch.float32, device=device)

    n = bank["n"]
    perm = rng.permutation(n)
    n_val = int(args.val_frac * n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    net = engines._build_attn_net(args.d).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def batch_loss(q_idx, mask_self):
        q = torch.as_tensor(q_idx, device=device)
        pred = net(va_t[q], va_t, theta_t, mask_idx=q if mask_self else None)
        return ((pred - theta_t[q]) ** 2).mean()

    steps = max(1, len(train_idx) // args.batch)
    for ep in range(args.epochs):
        net.train()
        ep_loss = 0.0
        order = rng.permutation(train_idx)
        for s in range(steps):
            q = order[s * args.batch:(s + 1) * args.batch]
            if len(q) == 0:
                continue
            opt.zero_grad()
            loss = batch_loss(q, mask_self=True)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if ep % 5 == 0 or ep == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                vl = batch_loss(val_idx, mask_self=False).item()
            print(f"  ep {ep:3d}  train {ep_loss / steps:.4f}  val {vl:.4f}  "
                  f"temp {float(torch.exp(net.log_temp).item()):.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "attn_retrieval.pt"
    torch.save({
        "d": args.d,
        "state_dict": net.state_dict(),
        "y_mean": y_mean,
        "y_std": y_std,
        "bank_index": [Path(p).parent.name + "/" + Path(p).name for p in present],
        "scalar_keys": bank["scalar_keys"],
    }, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
