"""Fuse pairwise and absolute transition ratings into one preference GP.

Step 1 of Section 3.5. Two rating studies are combined into a single latent
utility over the arc-parameter vocabulary, and the fitted posterior is frozen
to `../weights/preference_gp.npz` for `../inference/arc_policy.py` to query.

The model:

  - Every pool arc gets a latent utility f_i = u(x_i), jointly Gaussian with
    f ~ N(0, K) and K an ARD-RBF kernel over z-scored arc parameters.
    Comparisons therefore SHARE STRENGTH across arcs, rather than each arc
    getting an isolated Bradley-Terry score -- which matters enormously at
    roughly one comparison per arc.
  - Pairwise "which transition feels better" comparisons enter through a
    Davidson (1970) tie-aware likelihood in the score difference. Ties are
    informative data here, not coin-flips to be dropped: the rating app gave
    raters a "same" button precisely so indifference could be recorded.
  - Absolute valence ratings enter through a linear-link Gaussian likelihood,
    y = a + b*f_i + scene_offset + noise. The per-scene offset soaks up the
    pad-level valence baseline from the blocked-scene design, so b*f_i
    isolates the TRANSITION's contribution instead of the pad it happened on.
  - Both terms share the same f. The posterior is a Laplace approximation --
    MAP plus Hessian at the mode -- which is the standard Chu and Ghahramani
    (2005) preference-GP construction, extended with Davidson ties in place of
    a plain probit comparison.

Honesty note, kept deliberately: the kernel hyperparameters (length scales,
signal variance, tie parameter nu) are FIXED sensible defaults on z-scored
features, not marginal-likelihood optimised. Full type-II ML would need a
nested Newton-in-the-loop fit, which is not worth the risk at this number of
comparisons. Flags override them, and the report always prints the tie rate,
comparisons per arc and in-sample sign accuracy, which is what you would need
to judge when that becomes worth revisiting.

Three rating sources feed it, all shipped pseudonymised in `../human_ratings/`:
`arc_ratings.csv` (pairwise), `arc_valence_ratings.csv` (absolute), and the
texture round's `arc`-kind control rows from `texture_ratings.csv`, which are
honest replicate observations on already-pooled arcs. Rows flagged
`bad_texture` are bounds evidence rather than utility observations and are
excluded.

Output (in --out-dir):
  preference_gp.npz   the frozen Laplace posterior, everything the policy needs

Default output is ../data/preference_gp, not ../weights. See Section 3.4.1's
training scripts for why.

Run:
  python fit_preference_gp.py                 # fit and report
  python fit_preference_gp.py --freeze        # also write the npz

Next: ../inference/arc_policy.py loads the frozen posterior.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_SECTION = _HERE.parents[1]
sys.path.insert(0, str(_SECTION / "inference"))
# ONE canonical texture ladder, defined on the INFERENCE side because the
# conductor renders with it. Importing it back here is what guarantees the
# level this GP learns a preference for is voiced identically live.
from arc_policy import (TEXTURE_GAINS, texture_overrides,      # noqa: E402
                        texture_signed, load_pool)

RATINGS = _SECTION / "human_ratings"
POOL_META = RATINGS / "arc_pool_meta.json"
PAIR_CSV = RATINGS / "arc_ratings.csv"
VALENCE_CSV = RATINGS / "arc_valence_ratings.csv"
TEXTURE_CSV = RATINGS / "texture_ratings.csv"

DTYPE = torch.float64
JITTER = 1e-4

FEATURE_NAMES = ["context_v", "context_a", "audible_dist", "ramp_s",
                 "start_is_maj", "end_is_maj", "is_glide", "glide_semitones",
                 "texture_probe", "param_wander_std", "pitch_drift_cents"]


def arc_features(m):
    ov_render = m.get("render_overrides") or {}
    rk = m.get("render_kw") or {}
    return [
        float(m.get("context_v", 0.0)), float(m.get("context_a", 0.0)),
        float(m.get("audible_dist", 0.0)), float(m["ramp_s"]),
        1.0 if m.get("chord_start") == "maj" else 0.0,
        1.0 if m.get("chord_end") == "maj" else 0.0,
        1.0 if ov_render.get("xfade_mode") == "glide" else 0.0,
        float(m.get("glide_semitones", 0.0) or 0.0),
        texture_signed(m),
        float(rk.get("param_wander_std", 0.15)),
        float(rk.get("pitch_drift_cents", 2.0)),
    ]


def scene_of(m):
    return m.get("scene_id") or f"scene_{m.get('start_anchor')}"


def build_feature_matrix(pool):
    ids = sorted(pool)
    X = np.array([arc_features(pool[i]) for i in ids], dtype=np.float64)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-8] = 1.0
    return ids, X, (X - mean) / std, mean, std


def kernel_matrix(Xz, lengthscales, signal_var):
    Xzs = torch.as_tensor(Xz, dtype=DTYPE) / lengthscales
    K = signal_var * torch.exp(-0.5 * torch.cdist(Xzs, Xzs) ** 2)
    return K + JITTER * torch.eye(Xzs.shape[0], dtype=DTYPE)


def load_pairwise(idx_of, csv_path=PAIR_CSV, exclude_raters=()):
    """(i_idx, j_idx, outcome); outcome 0 = i beat j, 1 = j beat i, 2 = tie.

    Catch trials where arc_a == arc_b are KEPT: the score difference is zero by
    construction, so they carry pure information about the tie parameter and
    none about f.
    """
    rows = []
    if Path(csv_path).exists():
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                if r.get("rater") in exclude_raters:
                    continue
                a, b, choice = r["arc_a"], r["arc_b"], r["choice"]
                if a not in idx_of or b not in idx_of or choice not in ("A", "B", "same"):
                    continue
                rows.append((idx_of[a], idx_of[b],
                             0 if choice == "A" else (1 if choice == "B" else 2)))
    if not rows:
        e = torch.empty(0, dtype=torch.long)
        return e, e, e
    i, j, o = zip(*rows)
    return (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long),
            torch.tensor(o, dtype=torch.long))


def load_absolute(idx_of, scene_lookup=None, valence_csv=VALENCE_CSV,
                  texture_csv=TEXTURE_CSV, exclude_raters=()):
    """(arc_idx, scene_idx, y, scene_names) for the absolute-valence term.

    Repeat trials are included: they are honest replicate observations that
    inform the noise, not double-counted arcs. The texture round's `arc`-kind
    control rows join on the pool manifest's scene, because that file logs no
    scene id and the manifest is the source of truth -- so an arc rated in both
    studies shares ONE scene offset.
    """
    rows = []
    if Path(valence_csv).exists():
        with open(valence_csv) as fh:
            for r in csv.DictReader(fh):
                if r.get("rater") in exclude_raters:
                    continue
                if r["arc_id"] in idx_of:
                    rows.append((idx_of[r["arc_id"]],
                                 r.get("scene_id") or "unknown", float(r["rating"])))
    if scene_lookup is not None and Path(texture_csv).exists():
        with open(texture_csv) as fh:
            for r in csv.DictReader(fh):
                if r.get("rater") in exclude_raters:
                    continue
                if r.get("kind") != "arc":          # skip the wander-pool clips
                    continue
                if r.get("bad_texture") == "1":     # bounds evidence, not utility
                    continue
                aid, rv = r.get("clip_id"), r.get("rating")
                if not aid or aid not in idx_of or rv in (None, ""):
                    continue
                rows.append((idx_of[aid],
                             scene_lookup.get(aid) or "texture_control", float(rv)))
    if not rows:
        return (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=DTYPE), [])
    arc_idx, names, y = zip(*rows)
    scenes = sorted(set(names))
    k_of = {s: k for k, s in enumerate(scenes)}
    return (torch.tensor(arc_idx, dtype=torch.long),
            torch.tensor([k_of[s] for s in names], dtype=torch.long),
            torch.tensor(y, dtype=DTYPE), scenes)


def pairwise_nll(f, i_idx, j_idx, outcome, log_nu):
    if i_idx.numel() == 0:
        return torch.zeros((), dtype=DTYPE)
    nu = torch.exp(log_nu)
    z = f[i_idx] - f[j_idx]
    logZ = torch.log(2 * torch.cosh(z / 2) + nu)
    logp = torch.where(outcome == 0, z / 2 - logZ,
                       torch.where(outcome == 1, -z / 2 - logZ,
                                   torch.log(nu) - logZ))
    return -logp.sum()


def absolute_nll(f, arc_idx, scene_idx, y, a, b, scene_offsets, sigma_obs):
    if arc_idx.numel() == 0:
        return torch.zeros((), dtype=DTYPE)
    resid = (y - (a + b * f[arc_idx] + scene_offsets[scene_idx])) / sigma_obs
    return 0.5 * (resid ** 2).sum()


def fit_laplace(K, pairwise, absolute, valence_noise, ridge_ab=1.0,
                ridge_scene=1.0, nu_init=1.0):
    n = K.shape[0]
    Lk = torch.linalg.cholesky(K)
    Kinv = torch.cholesky_inverse(Lk)
    i_idx, j_idx, outcome = pairwise
    arc_idx, scene_idx, y, scenes = absolute
    has_absolute = arc_idx.numel() > 0

    f = torch.zeros(n, dtype=DTYPE, requires_grad=True)
    log_nu = torch.tensor(float(np.log(nu_init)), dtype=DTYPE, requires_grad=True)
    params = [f, log_nu]
    if has_absolute:
        a = torch.zeros((), dtype=DTYPE, requires_grad=True)
        b = torch.zeros((), dtype=DTYPE, requires_grad=True)
        scene_offsets = torch.zeros(len(scenes), dtype=DTYPE, requires_grad=True)
        params += [a, b, scene_offsets]
    else:
        a = b = torch.zeros((), dtype=DTYPE)
        scene_offsets = torch.zeros(0, dtype=DTYPE)

    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=200, history_size=20,
                            line_search_fn="strong_wolfe")

    def total_loss():
        loss = 0.5 * f @ torch.cholesky_solve(f.unsqueeze(1), Lk).squeeze(1)
        loss = loss + pairwise_nll(f, i_idx, j_idx, outcome, log_nu)
        if has_absolute:
            loss = loss + absolute_nll(f, arc_idx, scene_idx, y, a, b,
                                       scene_offsets, valence_noise)
            loss = loss + 0.5 * (a ** 2 + b ** 2) / ridge_ab
            loss = loss + 0.5 * (scene_offsets ** 2).sum() / ridge_scene
        return loss

    def closure():
        opt.zero_grad()
        loss = total_loss()
        loss.backward()
        return loss

    for _ in range(3):
        opt.step(closure)

    f_hat = f.detach()
    nu_hat = float(torch.exp(log_nu.detach()))
    a_hat = float(a.detach()) if has_absolute else 0.0
    b_hat = float(b.detach()) if has_absolute else 0.0
    scene_hat = scene_offsets.detach() if has_absolute else torch.zeros(0, dtype=DTYPE)

    def data_nll(f_):
        loss = pairwise_nll(f_, i_idx, j_idx, outcome,
                            torch.tensor(float(np.log(nu_hat)), dtype=DTYPE))
        if has_absolute:
            loss = loss + absolute_nll(f_, arc_idx, scene_idx, y,
                                       torch.tensor(a_hat, dtype=DTYPE),
                                       torch.tensor(b_hat, dtype=DTYPE),
                                       scene_hat, valence_noise)
        return loss

    W = torch.autograd.functional.hessian(data_nll, f_hat)
    Sigma = torch.linalg.inv(Kinv + W + 1e-6 * torch.eye(n, dtype=DTYPE))
    # Lk is carried so the policy can query the posterior at NEW points without
    # rebuilding or refactorising K.
    return dict(f=f_hat, Sigma=Sigma, nu=nu_hat, a=a_hat, b=b_hat,
                scenes=scenes, scene_offsets=scene_hat,
                loss=float(total_loss().detach()), Lk=Lk)


def report(pool, ids, feat_names, X, pairwise, absolute, fit):
    i_idx, j_idx, outcome = pairwise
    arc_idx, _, _, scenes = absolute
    n_pair = i_idx.numel()
    print(f"\n--- preference GP report ---")
    print(f"arcs in pool: {len(ids)}")
    print(f"pairwise comparisons: {n_pair} "
          f"(A {int((outcome == 0).sum())}, B {int((outcome == 1).sum())}, "
          f"ties {int((outcome == 2).sum())})")
    if n_pair:
        print(f"tie rate: {float((outcome == 2).float().mean()):.2%}  |  "
              f"fitted Davidson nu: {fit['nu']:.3f}")
        counts = np.zeros(len(ids))
        for i in i_idx.tolist():
            counts[i] += 1
        for j in j_idx.tolist():
            counts[j] += 1
        touched = counts[counts > 0]
        if len(touched):
            print(f"comparisons per touched arc: min {touched.min():.0f} "
                  f"median {np.median(touched):.0f} max {touched.max():.0f} "
                  f"({len(touched)}/{len(ids)} arcs touched)")
        non_tie = outcome != 2
        if non_tie.any():
            z = (fit["f"][i_idx] - fit["f"][j_idx])[non_tie]
            acc = float(((z > 0) == (outcome[non_tie] == 0)).float().mean())
            print(f"in-sample sign accuracy (non-tie): {acc:.2%} -- NOT held out")
    print(f"absolute valence ratings: {arc_idx.numel()} across {len(scenes)} scenes")
    if arc_idx.numel():
        print(f"fitted link: valence ~= {fit['a']:.3f} + {fit['b']:.3f}*utility "
              f"+ scene_offset")

    f = fit["f"].numpy()
    sd = np.sqrt(np.diag(fit["Sigma"].numpy()))
    order = np.argsort(-f)
    for label, sel in (("top 5 arcs by posterior mean utility", order[:5]),
                       ("bottom 5", order[-5:])):
        print(f"\n{label}:")
        for k in sel:
            m = pool[ids[k]]
            print(f"  {ids[k]}: f={f[k]:+.3f}+-{sd[k]:.3f}  ramp={m['ramp_s']}s "
                  f"dist={m.get('dist_bin')} "
                  f"chord={m.get('chord_start')}->{m.get('chord_end')}")

    if n_pair >= 5:
        print("\nposterior-mean correlation with each raw feature "
              "(interpretability only -- length scales are fixed, not learned):")
        for name, col in zip(feat_names, X.T):
            if col.std() >= 1e-8:
                print(f"  {name:>18s}: r={float(np.corrcoef(f, col)[0, 1]):+.3f}")


def freeze(fit, ids, X, Xz, mean, std, lengthscales, signal_var, out_path):
    """Write everything the policy needs to query the posterior at new points."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             ids=np.array(ids, dtype=object).astype("U"),
             f=fit["f"].numpy(), Sigma=fit["Sigma"].numpy(), Lk=fit["Lk"].numpy(),
             X=X, Xz=Xz, mean=mean, std=std,
             lengthscales=lengthscales.numpy(),
             signal_var=np.asarray(float(signal_var)),
             nu=np.asarray(fit["nu"]), a=np.asarray(fit["a"]),
             b=np.asarray(fit["b"]),
             scenes=np.array(fit["scenes"], dtype=object).astype("U"),
             scene_offsets=fit["scene_offsets"].numpy(),
             feature_names=np.array(FEATURE_NAMES).astype("U"),
             context_v=np.array([0.0]), )
    print(f"\nfrozen -> {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Fit the arc preference GP and freeze its posterior.")
    ap.add_argument("--pool-meta", dest="pool_meta", default=None)
    ap.add_argument("--length-scale", dest="length_scale", type=float, default=1.0)
    ap.add_argument("--signal-var", dest="signal_var", type=float, default=1.0)
    ap.add_argument("--nu", type=float, default=1.0, help="Davidson tie parameter")
    ap.add_argument("--valence-noise", dest="valence_noise", type=float, default=0.4)
    ap.add_argument("--exclude-raters", dest="exclude_raters", default="",
                    help="comma-separated rater ids to exclude (sensitivity only)")
    ap.add_argument("--freeze", action="store_true",
                    help="write preference_gp.npz")
    ap.add_argument("--out-dir", dest="out_dir", default=None,
                    help="default ../data/preference_gp. Not ../weights")
    args = ap.parse_args()

    pool = load_pool(args.pool_meta or POOL_META)
    ids, X, Xz, mean, std = build_feature_matrix(pool)
    idx_of = {aid: i for i, aid in enumerate(ids)}
    lengthscales = torch.full((len(FEATURE_NAMES),), args.length_scale, dtype=DTYPE)
    K = kernel_matrix(Xz, lengthscales, args.signal_var)

    exclude = {r for r in args.exclude_raters.split(",") if r}
    scene_lookup = {aid: scene_of(pool[aid]) for aid in ids}
    pairwise = load_pairwise(idx_of, exclude_raters=exclude)
    absolute = load_absolute(idx_of, scene_lookup, exclude_raters=exclude)

    fit = fit_laplace(K, pairwise, absolute, args.valence_noise, nu_init=args.nu)
    report(pool, ids, FEATURE_NAMES, X, pairwise, absolute, fit)

    if args.freeze:
        out_dir = (Path(args.out_dir) if args.out_dir
                   else _SECTION / "data" / "preference_gp")
        freeze(fit, ids, X, Xz, mean, std, lengthscales, args.signal_var,
               out_dir / "preference_gp.npz")


if __name__ == "__main__":
    main()
