"""Fit the boundary guard and freeze it to a JSON the runtime can load.

Step 4 of Section 3.4.2, and one of the pre-fits that lets the conductor ship
with no rating data.

A Gaussian process is fitted over the coordinates where clips were actually
rated. Its posterior sigma(v, a) is a distance-from-rated-territory field, and
``../inference/guard.py`` turns that field into two drift terms that keep a
wandering walk inside the region a listener has heard.

What gets frozen into ``../weights/boundary_guard.json``:

  X             the rated COORDINATES (not the ratings)
  length_scale  fitted RBF length scale
  noise_level   fitted white-noise level
  K_inv         inverse of the training kernel matrix
  y_scale       the normalisation factor sklearn applied to the targets
  tau           the derived trigger threshold
  safe_points   the rated points the GP is most certain about

**No rating values are written.** The posterior variance depends only on the
coordinates and the kernel, so the fence needs nothing anyone scored. That is
what makes the shipped guard rating-free.

The length-scale floor is 0.25, not the 0.1 used elsewhere. The marginal-
likelihood fit pins the length scale to whatever floor it is given, because the
ratings are largely noise with respect to the judge-labelled coordinates, and
at 0.1 the sigma field becomes a pincushion the guard cannot descend -- a
guarded stress walk stayed above tau on 99% of steps. At 0.25 grad sigma^2
stays informative at the waypoint distances the walk actually takes.

tau is DERIVED: the tau-quantile of posterior sigma at the rated points.

Label space. The guard must be fitted in the SAME valence axis the walk
navigates. ``--label-space human`` re-expresses the same ratings at the same
clips' propagated human coordinates, which requires propagate_labels_krr.py to
have written a bank index carrying valence_human. A guard fitted in judge space
fences a human-space walk out of exactly the territory the human axis opens up.

Run:
  python fit_guard.py                       # judge space, the default
  python fit_guard.py --label-space human --bank-index ../data/chord10k/labeled_index_human.csv
  python fit_guard.py --self-test           # guarded vs unguarded stress walk

Previous: rating_agreement.py
Next:     ../inference/guard.py loads what this writes
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

_HERE = Path(__file__).resolve()
_SECTION = _HERE.parents[1]
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_SECTION / "inference"))
from rating_agreement import CUTOVER, PRIMARY_RATER, clip_name   # noqa: E402

RATINGS = _SECTION / "human_ratings" / "valence_ratings.csv"
POOL_META = _SECTION / "human_ratings" / "pool_meta.json"
OUT = _SECTION / "weights" / "boundary_guard.json"


def load_rated_coords(csv_path, rater=PRIMARY_RATER):
    """(coords (n,2), ratings (n,)) for one rater: achieved VA, latest rating
    per clip, the same hygiene rules rating_agreement.py applies."""
    latest = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["timestamp"] < CUTOVER or r["rater"] != rater:
                continue
            key = clip_name(r["clip_path"])
            if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
                latest[key] = r
    if not latest:
        raise SystemExit(f"\nno ratings for rater {rater!r} in {csv_path}\n")
    rows = list(latest.values())
    X = np.array([[float(r["achieved_v"]), float(r["achieved_a"])] for r in rows])
    y = np.array([float(r["rating"]) for r in rows])
    return X, y, rows


def human_coords(rows, X, pool_meta_path, bank_index):
    """Re-express the rated clips on the propagated human valence axis.

    Arousal is untouched: it is judge-labelled on both axes by design, because
    the arousal judge is the valid one. Only valence moves.

    The anchor join is VALIDATED rather than trusted. Every clip records the
    bank row it was rendered from, and that row's judge labels must reproduce
    the clip's recorded achieved VA. If they do not, the bank's row order has
    drifted since the pool was built and the human coordinates would be
    silently wrong, so this raises instead of returning a plausible answer.
    """
    import pandas as pd

    META_TOL, CSV_TOL = 1e-9, 2e-4      # pool_meta is full precision; the
                                        # ratings CSV writes 4 decimal places
    meta = json.loads(Path(pool_meta_path).read_text())
    by_clip = {clip_name(m["clip_path"]): m for m in meta}

    bank = Path(bank_index)
    if not bank.exists():
        raise SystemExit(
            f"\nbank index not found at:\n    {bank}\n\n"
            "The human label space needs a bank carrying valence_human.\n"
            "Run propagate_labels_krr.py --propagate first.\n")
    df = pd.read_csv(bank)
    if "valence_human" not in df.columns:
        raise SystemExit(f"\n{bank} has no valence_human column.\n")
    vj = df["valence"].to_numpy(np.float64)
    aj = df["arousal"].to_numpy(np.float64)
    vh = df["valence_human"].to_numpy(np.float64)
    if not np.isfinite(vh).all():
        raise SystemExit(f"\nvalence_human has "
                         f"{int((~np.isfinite(vh)).sum())} missing values\n")

    out = X.copy()
    worst_meta = worst_csv = 0.0
    for i, r in enumerate(rows):
        key = clip_name(r["clip_path"])
        m = by_clip.get(key)
        if m is None or m.get("anchor_idx") is None:
            raise SystemExit(f"\n{key} has no anchor_idx in the pool metadata; "
                             f"cannot place it on the human axis\n")
        idx = int(m["anchor_idx"])
        if not 0 <= idx < len(vj):
            raise SystemExit(f"\n{key}: anchor_idx {idx} outside the "
                             f"{len(vj)}-row bank\n")
        worst_meta = max(worst_meta, abs(vj[idx] - float(m["achieved_v"])),
                         abs(aj[idx] - float(m["achieved_a"])))
        worst_csv = max(worst_csv, abs(vj[idx] - X[i, 0]),
                        abs(aj[idx] - X[i, 1]))
        out[i, 0] = vh[idx]
    if worst_meta > META_TOL or worst_csv > CSV_TOL:
        raise SystemExit(
            f"\nanchor join FAILED: achieved VA disagrees with the bank at "
            f"anchor_idx by up to {worst_meta:.2e} (pool metadata, tol "
            f"{META_TOL:.0e}) and {worst_csv:.2e} (ratings CSV, tol "
            f"{CSV_TOL:.0e}).\nThe bank's row order has changed since the pool "
            f"was generated; the human coordinates would be wrong. Refusing.\n")
    return out


def fit(X, y, tau_quantile=0.90, safe_quantile=0.50, seed=0):
    """Fit the GP and derive tau and the safe set. Returns a JSON-ready dict."""
    kernel = (RBF(length_scale=0.5, length_scale_bounds=(0.25, 1.0))
              + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 0.5)))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                  normalize_y=True, random_state=seed)
    gp.fit(X, y)

    length_scale = float(gp.kernel_.k1.length_scale)
    noise_level = float(gp.kernel_.k2.noise_level)
    # normalize_y scales the posterior std by the target std.
    y_scale = float(getattr(gp, "_y_train_std", 1.0))

    K = gp.kernel_(X) + np.eye(len(X)) * gp.alpha
    K_inv = np.linalg.inv(K)

    _, sig_rated = gp.predict(X, return_std=True)
    tau = float(np.quantile(sig_rated, tau_quantile))
    safe = X[sig_rated <= np.quantile(sig_rated, safe_quantile)]
    return gp, {
        "X": X.tolist(),
        "length_scale": length_scale,
        "noise_level": noise_level,
        "y_scale": y_scale,
        "K_inv": K_inv.tolist(),
        "tau": tau,
        "tau_quantile": float(tau_quantile),
        "safe_points": safe.tolist(),
        "n_rated": int(len(X)),
        "gamma": 0.08, "beta": 1.0, "fd_step": 0.02,
        "k_safe": 5, "ramp_band": 0.05, "hard_step_cap": 0.08,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Fit the boundary guard and freeze it for the runtime.")
    ap.add_argument("--csv", default=None, help=f"ratings (default {RATINGS})")
    ap.add_argument("--pool-meta", dest="pool_meta", default=None)
    ap.add_argument("--rater", default=PRIMARY_RATER)
    ap.add_argument("--label-space", dest="label_space",
                    choices=["judge", "human"], default="judge")
    ap.add_argument("--bank-index", dest="bank_index", default=None,
                    help="bank index carrying valence_human, for --label-space human")
    ap.add_argument("--tau-quantile", dest="tau_quantile", type=float, default=0.90)
    ap.add_argument("--safe-quantile", dest="safe_quantile", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help=f"default {OUT}")
    ap.add_argument("--self-test", dest="self_test", action="store_true",
                    help="run guarded and unguarded stress walks and compare")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else RATINGS
    meta_path = Path(args.pool_meta) if args.pool_meta else POOL_META
    out_path = Path(args.out) if args.out else OUT
    if not csv_path.exists():
        raise SystemExit(f"\nratings not found at:\n    {csv_path}\n")

    X, y, rows = load_rated_coords(csv_path, args.rater)
    print(f"Rated clips: {len(X)} from {args.rater}")
    if args.label_space == "human":
        if not args.bank_index:
            raise SystemExit("\n--label-space human needs --bank-index\n")
        X = human_coords(rows, X, meta_path, args.bank_index)
        print("Coordinates re-expressed on the propagated human valence axis")

    gp, payload = fit(X, y, args.tau_quantile, args.safe_quantile, args.seed)
    payload["label_space"] = args.label_space
    print(f"kernel: RBF(length_scale={payload['length_scale']:.3f}) + "
          f"White(noise_level={payload['noise_level']:.3f})")
    print(f"tau (q={args.tau_quantile}): {payload['tau']:.4f}   "
          f"safe points: {len(payload['safe_points'])}")

    # The runtime recomputes the posterior in numpy, without sklearn. Verify
    # the two agree before freezing, or the shipped fence is not the fitted one.
    from guard import BoundaryGuard
    g = BoundaryGuard(**{k: v for k, v in payload.items()
                         if k in ("X", "length_scale", "noise_level", "K_inv",
                                  "tau", "safe_points", "label_space", "gamma",
                                  "beta", "fd_step", "k_safe", "ramp_band",
                                  "hard_step_cap", "tau_quantile", "y_scale")})
    probe = np.random.default_rng(0).uniform(-1, 1, (200, 2))
    _, sk = gp.predict(probe, return_std=True)
    mine = np.asarray(g.sigma(probe))
    err = float(np.max(np.abs(sk - mine)))
    print(f"runtime posterior vs sklearn, 200 probes: max|diff| {err:.3e}")
    if err > 1e-9:
        raise SystemExit("runtime posterior does NOT match the fitted GP; "
                         "refusing to write a guard the runtime cannot honour")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1))
    print(f"written -> {out_path}")

    if args.self_test:
        from guard import ou_walk
        for use in (False, True):
            path, sig = ou_walk(g, start=(-0.6, 0.0), target=(0.9, 0.9),
                                use_guard=use)
            above = float(np.mean(sig > payload["tau"]))
            print(f"  {'guarded  ' if use else 'unguarded'}: "
                  f"final ({path[-1][0]:+.2f}, {path[-1][1]:+.2f})  "
                  f"max sigma {sig.max():.3f}  "
                  f"steps above tau {above*100:.0f}%")


if __name__ == "__main__":
    main()
