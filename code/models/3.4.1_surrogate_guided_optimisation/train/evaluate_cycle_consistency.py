"""Cycle consistency: ask for a coordinate, render it, measure what comes back.

Step 5 of Section 3.4.1, and the measurement every inverse model in this thesis
is compared under. Runs in two passes, with the real judge in between, because
the proxy cannot referee a contest it is the target of.

  --mode render   sample presets from an inverse model on a VA grid, render each
                  through DDSP and polish, write the wavs and a theta_index.csv
                  that records the REQUESTED (v_target, a_target).
  (label)         score those renders with the frozen judge:
                      python label_dataset.py --data-dir <the same dir>
  --mode report   compare requested against read-back: per-axis MAE and bias, a
                  per-grid-point breakdown CSV, and a displacement plot.

Inverse models available to --inverse:
  cvae   the parameter-space CVAE            (train_cvae.py)
  mlp    the closed-loop mapper              (train_closed_loop.py)
  knn    hard k-NN over a labelled bank; near-optimal by construction, since
         each retrieved theta was already judged to sit at that coordinate
  soft | gp | attn
         the continuous retrieval engines from Section 3.4.2. These import that
         section, so they only work once it is present; the import is deferred
         so the three modes above never depend on it.

Output (in --data-dir):
  cyc_XXXX.wav, theta_index.csv     from --mode render
  cycle_report.csv, cycle_consistency.png   from --mode report

Run:
  python evaluate_cycle_consistency.py --mode render --inverse mlp --data-dir ../data/cycle_mlp
  python label_dataset.py --data-dir ../data/cycle_mlp
  python evaluate_cycle_consistency.py --mode report --data-dir ../data/cycle_mlp

Previous: train_cvae.py or train_closed_loop.py
Next:     dagger_render.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve()
_SECTION = _HERE.parents[1]
_CODE = _HERE.parents[3]
sys.path.insert(0, str(_CODE / "common"))
sys.path.insert(0, str(_HERE.parent))
# Section 3.3 owns the renderer. Importing it here rather than copying keeps one
# definition of how a theta becomes audio.
sys.path.insert(0, str(_CODE / "models" / "3.3_drone_synthesis_and_nsynth_prior"
                       / "inference"))
from device import get_device                                   # noqa: E402
from ddsp_synth import DifferentiableDDSPSynth                   # noqa: E402
from theta_render import render_theta, polish, SR, N_HARMONICS   # noqa: E402
from train_cvae import InverseCVAE, denormalize_scalars, SCALARS  # noqa: E402

# Grid over the reachable manifold; valence stays in the dark half plus a sliver.
V_GRID = [-0.9, -0.6, -0.3, 0.0]
A_GRID = [-0.8, -0.4, 0.0, 0.4, 0.8]
SAMPLES_PER_POINT = 3   # z draws per grid point: z=0 plus two random


def make_cvae_inverse(device, ckpt_path):
    """fn(v, a, sample_idx, rng) -> theta, from the parameter-space CVAE."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise SystemExit(f"\nCVAE checkpoint not found at:\n    {ckpt_path}\n")
    ckpt = torch.load(ckpt_path, map_location=device)
    scalars = ckpt.get("scalars", SCALARS)
    model = InverseCVAE(latent_dim=ckpt["latent_dim"],
                        n_scalars=len(scalars)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # tilt is recorded in theta but folded into harm_dist by the renderer.
    render_keys = [k for k in scalars if k != "tilt"]

    def fn(v, a, s, rng):
        c = torch.tensor([[v, a]], dtype=torch.float32, device=device)
        z = (torch.zeros(1, ckpt["latent_dim"], device=device) if s == 0
             else torch.randn(1, ckpt["latent_dim"], device=device))
        with torch.no_grad():
            harm, scal = model.decode(z, c)
        sc = denormalize_scalars(scal.cpu().numpy(), scalars)
        return {"harm_dist": harm.squeeze(0).cpu().numpy().astype(np.float32),
                **{k: sc[k].item() for k in render_keys}}
    return fn


def make_mlp_inverse(device, ckpt_path):
    """fn(v, a, s, rng) -> theta, from the closed-loop mapper. Deterministic:
    every draw at a grid point returns the same preset."""
    from train_closed_loop import Mapper, denorm_scalars_torch

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise SystemExit(f"\nmapper checkpoint not found at:\n    {ckpt_path}\n")
    ckpt = torch.load(ckpt_path, map_location=device)
    ranges = ckpt["scalar_ranges"]
    mapper = Mapper(n_scalars=len(ranges)).to(device)
    mapper.load_state_dict(ckpt["state_dict"])
    mapper.eval()

    def fn(v, a, s, rng):
        c = torch.tensor([[v, a]], dtype=torch.float32, device=device)
        with torch.no_grad():
            harm, scal01 = mapper(c)
            sc = denorm_scalars_torch(scal01, ranges)
        return {"harm_dist": harm.squeeze(0).cpu().numpy().astype(np.float32),
                **{k: float(t.item()) for k, t in sc.items()}}
    return fn


def make_knn_inverse(train_index: str, k: int):
    """fn(v, a, s, rng) -> theta, by retrieving the theta of the labelled clip
    whose MEASURED VA is nearest the target.

    Near-optimal by construction: each retrieved theta has already been judged
    to sit at approximately the requested coordinate. sample_idx 0 takes the
    nearest neighbour; later draws sample from the k nearest.
    """
    train_index = Path(train_index)
    if not train_index.exists():
        raise SystemExit(
            f"\nk-NN bank index not found at:\n    {train_index}\n\n"
            "Pass --train-index at a labelled bank.\n")
    df = pd.read_csv(train_index)
    va = df[["valence", "arousal"]].to_numpy()
    harm_cols = [f"h{j+1:02d}" for j in range(N_HARMONICS)]

    def fn(v, a, s, rng):
        d = np.sqrt((va[:, 0] - v) ** 2 + (va[:, 1] - a) ** 2)
        nearest = np.argsort(d)[:k]
        row = df.iloc[nearest[0] if s == 0 else rng.choice(nearest)]
        keys = ["f0_hz", "swell_rate", "swell_depth", "noise_level", "noise_cutoff_hz"]
        keys += [key for key in ["third_interval", "third_gain", "fifth_gain",
                                 "octave_gain"] if key in df.columns]
        return {"harm_dist": row[harm_cols].to_numpy(dtype=np.float32),
                **{key: float(row[key]) for key in keys}}
    return fn


def make_engine_inverse(name, args, device):
    """fn(v, a, s, rng) -> theta from one of Section 3.4.2's retrieval engines.

    Imported lazily: Section 3.4.2 owns these, and the cvae/mlp/knn modes must
    keep working whether or not that section is present.
    """
    engines_dir = (_CODE / "models" / "3.4.2_human_grounding_and_retrieval"
                   / "inference")
    if not (engines_dir / "retrieval_engines.py").exists():
        raise SystemExit(
            f"\n--inverse {name} needs Section 3.4.2's retrieval engines, expected at:\n"
            f"    {engines_dir / 'retrieval_engines.py'}\n\n"
            "Use --inverse cvae, mlp or knn, which are self-contained in this "
            "section.\n")
    sys.path.insert(0, str(engines_dir))
    import retrieval_engines as re_mod

    present = [p for p in args.bank_index if Path(p).exists()]
    if not present:
        raise SystemExit(
            f"\nnone of the bank indexes exist:\n    " +
            "\n    ".join(str(p) for p in args.bank_index) + "\n")
    bank = re_mod.build_bank(present)
    engine = re_mod.make_engine(
        name, bank, bandwidth=args.bandwidth, soft_k=args.soft_k,
        gp_subsample=args.gp_subsample, gp_length_scale=args.gp_length_scale,
        gp_optimize=args.gp_optimize, attn_ckpt=args.attn_ckpt,
        device=device, seed=args.seed)
    print(f"Inverse    : {name} engine over a {bank['n']}-clip bank "
          f"({len(present)} index file(s))")

    def fn(v, a, s, rng):
        return engine.blend(v, a, rng=None)
    return fn


def render(args):
    device = get_device()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)   # the noise floor is drawn inside the synth
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.inverse == "knn":
        inverse_fn = make_knn_inverse(args.train_index, args.k)
        print(f"Inverse    : k-NN (k={args.k}) over {args.train_index}")
    elif args.inverse in ("soft", "gp", "attn"):
        inverse_fn = make_engine_inverse(args.inverse, args, device)
    elif args.inverse == "mlp":
        ckpt = args.ckpt or _SECTION / "weights" / "closed_loop_mapper.pt"
        inverse_fn = make_mlp_inverse(device, ckpt)
        print(f"Inverse    : closed-loop mapper ({ckpt})")
    else:
        ckpt = args.ckpt or _SECTION / "weights" / "inverse_cvae.pt"
        inverse_fn = make_cvae_inverse(device, ckpt)
        print(f"Inverse    : CVAE ({ckpt})")

    synth = DifferentiableDDSPSynth(sample_rate=SR).to(device)
    n_samples = int(args.duration * SR)

    rows, i = [], 0
    for v in V_GRID:
        for a in A_GRID:
            for s in range(SAMPLES_PER_POINT):
                theta = inverse_fn(v, a, s, rng)
                audio = polish(render_theta(synth, theta, n_samples, device, rng))

                fname = f"cyc_{i:04d}.wav"
                sf.write(data_dir / fname, audio.astype(np.float32), SR)

                row = {"filename": fname, "v_target": v, "a_target": a,
                       "z_sample": s,
                       **{k: theta[k] for k in theta if k != "harm_dist"},
                       "tilt": theta.get("tilt", float("nan"))}
                for k in range(N_HARMONICS):
                    row[f"h{k+1:02d}"] = float(theta["harm_dist"][k])
                rows.append(row)
                i += 1

    pd.DataFrame(rows).to_csv(data_dir / "theta_index.csv", index=False)
    print(f"Rendered {i} clips ({len(V_GRID)}x{len(A_GRID)} grid "
          f"x {SAMPLES_PER_POINT} samples) to {data_dir}")
    print(f"Next: python label_dataset.py --data-dir {data_dir}")
    print(f"then: python {Path(__file__).name} --mode report --data-dir {data_dir}")


def report(args):
    data_dir = Path(args.data_dir)
    labelled = data_dir / "labeled_index.csv"
    if not labelled.exists():
        raise SystemExit(
            f"\nlabelled index not found at:\n    {labelled}\n\n"
            f"Label the renders first:\n"
            f"    python label_dataset.py --data-dir {data_dir}\n")
    df = pd.read_csv(labelled)

    df["v_err"] = df["valence"] - df["v_target"]
    df["a_err"] = df["arousal"] - df["a_target"]
    v_mae = df["v_err"].abs().mean()
    a_mae = df["a_err"].abs().mean()

    print(f"Cycle consistency over {len(df)} clips:")
    print(f"  Valence MAE: {v_mae:.3f}   bias: {df['v_err'].mean():+.3f}")
    print(f"  Arousal MAE: {a_mae:.3f}   bias: {df['a_err'].mean():+.3f}")
    print(f"  Euclidean mean error: "
          f"{np.sqrt(df['v_err']**2 + df['a_err']**2).mean():.3f}")

    grp = df.groupby(["v_target", "a_target"]).agg(
        v_read=("valence", "mean"), a_read=("arousal", "mean"),
        v_mae=("v_err", lambda x: x.abs().mean()),
        a_mae=("a_err", lambda x: x.abs().mean())).reset_index()
    grp.to_csv(data_dir / "cycle_report.csv", index=False)

    plt.figure(figsize=(7, 7))
    plt.axhline(0, color="gray", ls=":", alpha=0.5)
    plt.axvline(0, color="gray", ls=":", alpha=0.5)
    for _, r in grp.iterrows():
        plt.annotate("", xy=(r["v_read"], r["a_read"]),
                     xytext=(r["v_target"], r["a_target"]),
                     arrowprops=dict(arrowstyle="->", color="#673ab7", alpha=0.8))
        plt.scatter([r["v_target"]], [r["a_target"]], color="green", s=30, zorder=5)
        plt.scatter([r["v_read"]], [r["a_read"]], color="red", s=20, zorder=5)
    plt.scatter([], [], color="green", label="requested")
    plt.scatter([], [], color="red", label="read back")
    plt.xlim(-1.05, 1.05); plt.ylim(-1.05, 1.05)
    plt.xlabel("valence"); plt.ylabel("arousal")
    plt.title(f"VA cycle consistency (V MAE {v_mae:.2f}, A MAE {a_mae:.2f})")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    out = data_dir / "cycle_consistency.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}\n       {data_dir / 'cycle_report.csv'}")


def main():
    ap = argparse.ArgumentParser(
        description="Measure VA cycle consistency of an inverse model.")
    ap.add_argument("--mode", choices=["render", "report"], required=True)
    ap.add_argument("--inverse", choices=["cvae", "knn", "soft", "gp", "attn", "mlp"],
                    default="cvae")
    ap.add_argument("--data-dir", dest="data_dir", default=None, metavar="DIR",
                    help="working directory (default ../data/cycle_eval)")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint override for --inverse cvae or mlp "
                         "(default: the fitted one in ../weights)")
    ap.add_argument("--train-index", dest="train_index", default=None,
                    help="labelled bank index for --inverse knn")
    ap.add_argument("--k", type=int, default=5)
    # Bank and hyperparameters for the Section 3.4.2 engines.
    ap.add_argument("--bank-index", dest="bank_index", nargs="+", default=[],
                    help="labelled_index.csv file(s) forming the bank for "
                         "--inverse soft/gp/attn")
    ap.add_argument("--bandwidth", type=float, default=0.15)
    ap.add_argument("--soft-k", dest="soft_k", type=int, default=64)
    ap.add_argument("--gp-subsample", dest="gp_subsample", type=int, default=1500)
    ap.add_argument("--gp-length-scale", dest="gp_length_scale", type=float, default=0.25)
    ap.add_argument("--gp-optimize", dest="gp_optimize", action="store_true")
    ap.add_argument("--attn-ckpt", dest="attn_ckpt", default=None,
                    help="attention retrieval checkpoint (Section 3.4.2)")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.data_dir is None:
        args.data_dir = _SECTION / "data" / "cycle_eval"
    if args.mode == "render":
        render(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
