"""Distil the frozen judge into a small differentiable CNN proxy.

Step 3 of Section 3.4.1. The frozen judge is MERT plus a ridge head: accurate,
but far too heavy to backpropagate through once per training step. This trains a
compact log-mel CNN to reproduce its outputs on rendered drones, giving a
differentiable audio -> (valence, arousal) function that ``train_closed_loop.py``
can optimise against.

This is the substitution that Section 4.1 is about. The proxy is a stand-in for
a stand-in, and the closed loop that optimises against it finds its blind spots.

Details that matter for reproducing the result:
  - Targets are the unclipped ``valence_raw`` / ``arousal_raw`` columns when
    present, so ordering beyond +1 survives.
  - Crops are 9 s of the 10 s clip. The judge scored the whole clip including
    its slow swell cycle; short crops land on swell peaks or troughs and inject
    label noise that is not in the target.
  - No gain augmentation. The judges are not level-invariant -- loudness is a
    primary arousal cue -- so normalising it away would teach the proxy a
    different function from the one it is imitating.
  - The train/validation split is fixed independently of the member seed, so
    per-member validation metrics are comparable across an ensemble.

An ensemble is worth training. ``train_closed_loop.py`` penalises the mapper by
member disagreement, which is what makes the out-of-distribution region
unattractive without a heavy manifold anchor.

Output (in --out-dir):
  judge_proxy_e0.pt .. judge_proxy_e{N-1}.pt   one per ensemble member
  judge_proxy.pt                               a copy of member 0

Note the default output is this section's ``data/`` directory, NOT ``weights/``.
``weights/`` holds the checkpoints that ship and reproduce the reported results;
a short or exploratory fit written there would be indistinguishable from them.
Copy a checkpoint into ``weights/`` deliberately, never by default.

Run:
  # single proxy, the quick path
  python train_judge_proxy.py --data-dir ../data/chord10k

  # the ensemble the closed loop wants, over two banks
  python train_judge_proxy.py --data-dir ../data/chord10k ../data/mixed10k --n-models 4

Previous: label_dataset.py
Next:     train_closed_loop.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "common"))
from device import get_device                       # noqa: E402

SR = 16000
CROP_SEC = 9.0
N_MELS = 96


def load_audio_matrix(df, clip_len, device):
    """Load every clip into one device tensor (N, clip_len).

    Kept as a single on-device buffer because slicing crops there removes the
    per-batch host-to-device copy that otherwise dominates the step time.
    Stored as fp16: quantisation noise is about -66 dB, irrelevant against the
    label noise, and it halves a buffer that can exceed a single-allocation cap.
    """
    mat = np.zeros((len(df), clip_len), dtype=np.float32)
    for i, fname in enumerate(df["filename"]):
        x, _ = sf.read(fname, dtype="float32", always_2d=True)
        x = x.mean(axis=1)[:clip_len]
        mat[i, :len(x)] = x
    audio = torch.from_numpy(mat).to(device, dtype=torch.float16)
    targets = torch.from_numpy(df[["v_t", "a_t"]].to_numpy(dtype=np.float32)).to(device)
    return audio, targets


def crop_batch(audio, idx, crop, train, device):
    """Random (train) or centre (eval) crops, gathered on-device."""
    full = audio.shape[1]
    if train:
        starts = torch.randint(0, max(1, full - crop), (len(idx),), device=device)
    else:
        starts = torch.full((len(idx),), (full - crop) // 2, device=device)
    gather_idx = starts.unsqueeze(1) + torch.arange(crop, device=device).unsqueeze(0)
    return torch.gather(audio[idx], 1, gather_idx).float()


class JudgeProxy(nn.Module):
    """Log-mel CNN -> (valence, arousal). Differentiable wrt the input audio.

    Imported by ``train_closed_loop.py`` and ``evaluate_cycle_consistency.py``
    to rebuild the architecture before loading a checkpoint.
    """

    def __init__(self, size="small"):
        super().__init__()
        self.size = size
        # hop 512 halves the default time resolution: drones carry no fast
        # transients, and it halves the CNN's work.
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=1024, hop_length=512, n_mels=N_MELS, power=2.0)

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                nn.ReLU(), nn.MaxPool2d(2))

        # small: ~0.3M params, for the 2k-clip regime. large: ~2.5M, for 10k+.
        chans = [32, 64, 128, 128] if size == "small" else [48, 96, 192, 256, 256]
        blocks, cin = [], 1
        for c in chans:
            blocks.append(block(cin, c))
            cin = c
        self.cnn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(cin, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, audio):                    # (B, n_samples)
        m = self.mel(audio)                      # (B, n_mels, frames)
        m = torch.log(m + 1e-5).unsqueeze(1)     # (B, 1, n_mels, frames)
        return self.head(self.cnn(m))            # (B, 2) -> [valence, arousal]


def train_one(seed, ckpt_path, tr_audio, tr_targets, va_audio, va_targets,
              n_tr, n_va, args, device):
    """Train one proxy, saving its best-validation checkpoint. Ensemble members
    differ only in this seed (weight init and batch order)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    crop = int(CROP_SEC * SR)
    model = JudgeProxy(size=args.model_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    hist_tr, hist_va = [], []
    best_va = float("inf")
    metrics = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        perm = torch.randperm(n_tr, device=device)
        for b0 in range(0, n_tr, args.batch):
            bidx = perm[b0:b0 + args.batch]
            x = crop_batch(tr_audio, bidx, crop, True, device)
            y = tr_targets[bidx]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = model(x)
            loss = nn.functional.mse_loss(pred.float(), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(bidx)
        sched.step()
        hist_tr.append(tot / n_tr)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for b0 in range(0, n_va, args.batch):
                bidx = torch.arange(b0, min(b0 + args.batch, n_va), device=device)
                x = crop_batch(va_audio, bidx, crop, False, device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    p = model(x)
                preds.append(p.float().cpu().numpy())
                trues.append(va_targets[bidx].cpu().numpy())
        preds, trues = np.concatenate(preds), np.concatenate(trues)
        va_loss = float(np.mean((preds - trues) ** 2))
        hist_va.append(va_loss)

        if va_loss < best_va:
            best_va = va_loss
            torch.save({"state_dict": model.state_dict(), "size": args.model_size,
                        "crop_sec": CROP_SEC, "sr": SR, "seed": seed}, ckpt_path)

        if epoch == 1 or epoch % 5 == 0:
            rv = np.corrcoef(preds[:, 0], trues[:, 0])[0, 1]
            ra = np.corrcoef(preds[:, 1], trues[:, 1])[0, 1]
            mv = np.mean(np.abs(preds[:, 0] - trues[:, 0]))
            ma = np.mean(np.abs(preds[:, 1] - trues[:, 1]))
            metrics = {"rv": rv, "ra": ra, "mv": mv, "ma": ma}
            print(f"  epoch {epoch:3d}  train {hist_tr[-1]:.4f}  val {va_loss:.4f}  "
                  f"| val r: v {rv:.3f} a {ra:.3f}  | val MAE: v {mv:.3f} a {ma:.3f}")

    return best_va, metrics


def main():
    ap = argparse.ArgumentParser(
        description="Distil the frozen judge into a differentiable CNN proxy.")
    ap.add_argument("--data-dir", dest="data_dir", nargs="+", required=True,
                    metavar="DIR", help="one or more labelled dataset directories")
    ap.add_argument("--index-name", dest="index_name", default="labeled_index.csv")
    ap.add_argument("--out-dir", dest="out_dir", default=None, metavar="DIR",
                    help="checkpoint directory (default ../data/proxy). Not "
                         "../weights: that holds the checkpoints that ship")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-size", dest="model_size",
                    choices=["small", "large"], default="small")
    ap.add_argument("--n-models", dest="n_models", type=int, default=1,
                    help="train an ensemble of N proxies, seeds seed..seed+N-1. "
                         "train_closed_loop.py penalises the mapper by ensemble "
                         "disagreement, so N>1 makes the uncertainty penalty live")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _HERE.parents[1] / "data" / "proxy"
    device = get_device()

    frames = []
    for d in args.data_dir:
        idx = Path(d) / args.index_name
        if not idx.exists():
            raise SystemExit(
                f"\nlabelled index not found at:\n    {idx}\n\n"
                "Run label_dataset.py on that directory first.\n")
        sub = pd.read_csv(idx)
        # Prefer unclipped targets: they preserve ordering past +1.
        sub["v_t"] = sub["valence_raw"] if "valence_raw" in sub.columns else sub["valence"]
        sub["a_t"] = sub["arousal_raw"] if "arousal_raw" in sub.columns else sub["arousal"]
        sub["filename"] = sub["filename"].apply(lambda f: str(Path(d) / f))
        frames.append(sub[["filename", "v_t", "a_t"]])
        print(f"  {d}: {len(sub)} clips "
              f"({'raw' if 'valence_raw' in sub.columns else 'clipped'} targets)")
    df = pd.concat(frames, ignore_index=True)
    print(f"Total clips: {len(df)}")
    print(f"Output     : {out_dir}")

    # Fixed split, independent of the member seed, so members compare.
    split_rng = np.random.default_rng(12345)
    idx = split_rng.permutation(len(df))
    n_val = max(1, int(0.1 * len(df)))
    df_tr = df.iloc[idx[n_val:]].reset_index(drop=True)
    df_va = df.iloc[idx[:n_val]].reset_index(drop=True)

    clip_len = int(10.0 * SR)
    print("Loading audio onto device (one copy, shared across members)...")
    tr_audio, tr_targets = load_audio_matrix(df_tr, clip_len, device)
    va_audio, va_targets = load_audio_matrix(df_va, clip_len, device)
    n_tr, n_va = len(df_tr), len(df_va)

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(args.n_models):
        seed = args.seed + i
        ckpt_path = out_dir / f"judge_proxy_e{i}.pt"
        tag = (f"[member {i+1}/{args.n_models}, seed {seed}]"
               if args.n_models > 1 else "")
        print(f"\n=== Training proxy {tag} -> {ckpt_path.name} ===")
        best_va, metrics = train_one(seed, ckpt_path, tr_audio, tr_targets,
                                     va_audio, va_targets, n_tr, n_va, args, device)
        # Member 0 doubles as the single-proxy checkpoint the loaders default to.
        if i == 0:
            shutil.copyfile(ckpt_path, out_dir / "judge_proxy.pt")
        results.append((i, seed, best_va, metrics))

    print("\n=== Ensemble summary ===")
    for i, seed, best_va, m in results:
        print(f"  member {i} (seed {seed}): best val MSE {best_va:.4f}"
              + (f"  | r: v {m['rv']:.3f} a {m['ra']:.3f}"
                 f"  MAE: v {m['mv']:.3f} a {m['ma']:.3f}" if m else ""))
    print(f"\nSaved to {out_dir}")
    print("To use these for the reported pipeline, copy them into ../weights/ "
          "deliberately.")


if __name__ == "__main__":
    main()
