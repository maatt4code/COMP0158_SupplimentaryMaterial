"""Arc-selection policy: what transition should the conductor make right now?

Runtime code. It loads the frozen preference-GP posterior from
`../weights/preference_gp.npz` and answers queries against it. It never fits
anything, reads no ratings, and resolves no dataset root -- which is what lets
the conductor start with every dataset missing.

`../train/fit_preference_gp.py` fits a posterior mean utility for every arc
already in the rated pool, but has no way to answer "what is the best
transition at THIS live VA context", because the live conductor is not
restricted to replaying pool arcs verbatim. This module adds that query, in two
forms, deliberately kept side by side:

  1. `lookup_best_arc` -- the baseline. Filter the pool to arcs sharing the
     live context and return the one with the highest already-computed
     posterior mean. No new modelling, and honest at roughly one comparison
     per arc.
  2. `best_by_gp_predict` -- extends the fitted GP to a grid of NEVER-rated
     parameter combinations at the live context, through a standard Laplace-GP
     posterior-predictive query, and returns the argmax under a lower
     confidence bound.

Which parameters are policy axes, and which are not, is a real distinction and
not an implementation detail. context_v/context_a and audible_dist/
glide_semitones are CONSEQUENCES of where the walk currently is and which
anchor pair retrieval returned -- inputs to the policy, not outputs. The
policy-controllable axes are ramp_s (interpolated finely), chord_start and
chord_end, is_glide, and texture. param_wander_std and pitch_drift_cents are
NOT policy axes: they are voicing character, a uniform per-pad shimmer, so the
grid pins them at the frozen renderer defaults.

Texture caveat, worth knowing before trusting that axis: it is only as good as
the ratings behind it. Where the texture batch is thin the GP's texture utility
is prior-dominated, the lower-confidence-bound scores across the five levels
come out near-flat, and the argmax is effectively noise. That is acceptable by
design here -- every texture render is valid -- but the axis only becomes
evidence-driven once that batch is rated.

Run:
  python arc_policy.py --demo      # query the frozen posterior, print both methods
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

DTYPE = torch.float64
WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "preference_gp.npz"

RAMP_MIN, RAMP_MAX, RAMP_GRID_N = 3.0, 16.0, 20
CHORD_CHOICES = ("min", "maj")
# The signed voice-richness ladder, matching train/fit_preference_gp.py's
# TEXTURE_GAINS exactly, so the level scored here is voiced identically live.
TEXTURE_CHOICES = (-1.0, -0.5, 0.0, 0.5, 1.0)
TEXTURE_NAMES = {-1.0: "thin-strong", -0.5: "thin-mild", 0.0: "none",
                 0.5: "thick-mild", 1.0: "thick-strong"}

# The canonical voice-richness ladder. ONE source of truth, so the level the GP
# learns a preference for is voiced identically when the conductor renders it.
# Signed level in [-1, +1], a monotonic voice-count ladder on the END pad.
TEXTURE_GAINS = {
    1.0:  {"fifth_gain": 0.5, "octave_gain": 0.5},   # strong thick
    0.5:  {"octave_gain": 0.5},                       # mild thick
    0.0:  {},                                         # anchor baseline
    -0.5: {"fifth_gain": 0.0, "octave_gain": 0.0},    # mild thin
    -1.0: {"third_gain": 0.0, "fifth_gain": 0.0, "octave_gain": 0.0},
}


def texture_overrides(level):
    """Signed texture level -> the end-theta overrides that render it."""
    return dict(TEXTURE_GAINS[float(level)])


def texture_signed(m):
    """Signed texture level of an arc. Prefers the explicit field; otherwise
    reverse-detects it from the overrides, for the earlier binary probes."""
    if m.get("texture_signed") is not None:
        return float(m["texture_signed"])
    ov = m.get("theta_end_overrides") or {}
    if ov.get("fifth_gain") == 0.5 or ov.get("octave_gain") == 0.5:
        return 1.0
    if "third_gain" in ov and ov.get("third_gain") == 0.0:
        return -1.0
    return 0.0


def _pool_meta_path():
    """Where the arc metadata lives, in whichever tree this module sits in.

    It is METADATA -- 230 arcs and their parameters -- not rating data, and
    the runtime needs it to answer a policy query. In this section it ships
    beside the ratings it was staged with; in the conductor it ships in
    `weights/`, because the conductor has no `human_ratings/` at all and must
    not grow one. Checking both is what lets ONE copy of this module serve
    both trees, which is the condition for the conductor's engine copies
    staying byte-identical to their originals.
    """
    root = Path(__file__).resolve().parents[1]
    for cand in (root / "weights" / "arc_pool_meta.json",
                 root / "human_ratings" / "arc_pool_meta.json"):
        if cand.exists():
            return cand
    return root / "weights" / "arc_pool_meta.json"      # for the error message


def load_pool(meta_path=None):
    """Arc parameter metadata, keyed by arc id.

    Lives here rather than in the rating app because the RUNTIME needs arc
    metadata to answer a policy query, and it must not import a data-collection
    script to get it. No audio is needed or shipped.
    """
    import json
    p = Path(meta_path) if meta_path else _pool_meta_path()
    if not p.exists():
        raise SystemExit(f"\narc pool metadata not found at:\n    {p}\n")
    pool = {}
    for m in json.loads(p.read_text()):
        m["chord"] = f"{m['chord_start']}->{m['chord_end']}"
        pool[m["arc_id"]] = m
    return pool


class PreferenceGP:
    """The frozen Laplace posterior, queryable at new parameter vectors."""

    def __init__(self, npz):
        self.ids = [str(s) for s in npz["ids"]]
        self.X = np.asarray(npz["X"], float)
        self.Xz = torch.as_tensor(np.asarray(npz["Xz"], float), dtype=DTYPE)
        self.mean = np.asarray(npz["mean"], float)
        self.std = np.asarray(npz["std"], float)
        self.f = torch.as_tensor(np.asarray(npz["f"], float), dtype=DTYPE)
        self.Sigma = torch.as_tensor(np.asarray(npz["Sigma"], float), dtype=DTYPE)
        self.Lk = torch.as_tensor(np.asarray(npz["Lk"], float), dtype=DTYPE)
        self.lengthscales = torch.as_tensor(np.asarray(npz["lengthscales"], float),
                                            dtype=DTYPE)
        self.signal_var = float(npz["signal_var"])
        self.feature_names = [str(s) for s in npz["feature_names"]]
        self.idx_of = {a: i for i, a in enumerate(self.ids)}
        self.sd = np.sqrt(np.diag(self.Sigma.numpy()))

    @classmethod
    def load(cls, path=None):
        p = Path(path) if path else WEIGHTS
        if not p.exists():
            raise SystemExit(
                f"\nfrozen preference GP not found at:\n    {p}\n\n"
                "It ships with this section. Refit it with "
                "../train/fit_preference_gp.py --freeze.\n")
        return cls(np.load(p, allow_pickle=False))

    def predict_utility(self, feature_row):
        """Laplace-GP posterior mean and std at a NEW feature vector.

        The standard extension of the Laplace approximation to a held-out input
        (Rasmussen and Williams, sec 3.4.2). The MAP stationarity condition
        gives K^-1 f_hat = grad log p(y | f_hat), so the predictive mean is the
        textbook expression; the predictive variance is the matching projection
        of the already-fitted training covariance out to the new point.

        Self-check: querying at an existing training row's own feature vector
        must reproduce that row's f and diag(Sigma), because k_star is then a
        column of K itself.
        """
        xz = torch.as_tensor((np.asarray(feature_row, float) - self.mean) / self.std,
                             dtype=DTYPE) / self.lengthscales
        d2 = ((self.Xz / self.lengthscales - xz) ** 2).sum(dim=1)
        k_star = self.signal_var * torch.exp(-0.5 * d2)
        v = torch.cholesky_solve(k_star.unsqueeze(1), self.Lk).squeeze(1)
        mean = float(k_star @ torch.cholesky_solve(self.f.unsqueeze(1),
                                                   self.Lk).squeeze(1))
        var = float(self.signal_var - k_star @ v + v @ self.Sigma @ v)
        return mean, float(np.sqrt(max(var, 0.0)))

    def predict_utility_batch(self, feature_rows):
        """Vectorised `predict_utility`: score many points in one matrix pass.

        Exists because the candidate grid runs to hundreds of combinations, and
        a per-row Python loop would score them slowly on every live transition.
        Verified equal to `predict_utility` row for row.
        """
        F = torch.as_tensor((np.asarray(feature_rows, float) - self.mean) / self.std,
                            dtype=DTYPE) / self.lengthscales
        Xzs = self.Xz / self.lengthscales
        Kst = self.signal_var * torch.exp(-0.5 * torch.cdist(F, Xzs) ** 2)
        alpha = torch.cholesky_solve(self.f.unsqueeze(1), self.Lk).squeeze(1)
        means = Kst @ alpha
        V = torch.cholesky_solve(Kst.t(), self.Lk)
        quad1 = (Kst * V.t()).sum(dim=1)
        quad2 = (V * (self.Sigma @ V)).sum(dim=0)
        stds = torch.sqrt(torch.clamp(self.signal_var - quad1 + quad2, min=0.0))
        return means.numpy(), stds.numpy()

    def selfcheck(self, n=5):
        """Querying at existing training rows must reproduce f and diag(Sigma),
        and the batch path must agree with the row-at-a-time path."""
        idxs = np.linspace(0, len(self.X) - 1, min(n, len(self.X))).astype(int)
        f = self.f.numpy()
        max_mean = max(abs(self.predict_utility(self.X[i])[0] - f[i]) for i in idxs)
        max_std = max(abs(self.predict_utility(self.X[i])[1] - self.sd[i])
                      for i in idxs)
        bm, bs = self.predict_utility_batch(self.X[idxs])
        single = [self.predict_utility(self.X[i]) for i in idxs]
        batch_err = max(max(abs(bm[k] - single[k][0]), abs(bs[k] - single[k][1]))
                        for k in range(len(idxs)))
        # 1e-3, not machine precision: f_hat is an LBFGS-converged optimum
        # rather than closed form, and K carries a jitter term, so a residual
        # around 1e-4 is the expected noise floor.
        assert max_mean < 1e-3, "predictive mean does not reproduce f"
        assert max_std < 1e-3, "predictive std does not reproduce diag(Sigma)"
        assert batch_err < 1e-9, "batch path disagrees with the single-row path"
        return max_mean, max_std, batch_err


def context_of(m):
    return (float(m.get("context_v", 0.0)), float(m.get("context_a", 0.0)))


def pool_contexts(pool):
    """The distinct rated contexts actually present in the pool."""
    return sorted({context_of(m) for m in pool.values()})


def nearest_context(contexts, v, a):
    """Snap a live (v, a) to the nearest rated context."""
    d2 = [(cv - v) ** 2 + (ca - a) ** 2 for cv, ca in contexts]
    return contexts[int(np.argmin(d2))]


def lookup_candidates(pool, gp, context, tol=1e-6):
    """All pool arcs at this context as (arc_id, mean, std), best first.

    The coherence re-ranker needs the whole near-tie band, not just the top.
    """
    cv, ca = context
    out = []
    for k, aid in enumerate(gp.ids):
        if aid not in pool:
            continue
        acv, aca = context_of(pool[aid])
        if abs(acv - cv) > tol or abs(aca - ca) > tol:
            continue
        out.append((aid, float(gp.f[k]), float(gp.sd[k])))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def lookup_best_arc(pool, gp, context, tol=1e-6):
    """Baseline: the highest posterior mean among pool arcs at this context."""
    cands = lookup_candidates(pool, gp, context, tol)
    return cands[0] if cands else None


def candidate_grid(context, fixed_dist, fixed_glide_semitones=0.0):
    """Every policy-controllable combination at a fixed live context.

    Returns (choices, F) with F in feature order, so the whole grid scores in
    one batched pass. Wander and drift are pinned at the renderer defaults --
    voicing character, not policy.
    """
    cv, ca = context
    choices, rows = [], []
    for ramp_s in np.linspace(RAMP_MIN, RAMP_MAX, RAMP_GRID_N):
        for chord_start in CHORD_CHOICES:
            for chord_end in CHORD_CHOICES:
                for is_glide in (0.0, 1.0):
                    for texture in TEXTURE_CHOICES:
                        rows.append([cv, ca, fixed_dist, ramp_s,
                                     1.0 if chord_start == "maj" else 0.0,
                                     1.0 if chord_end == "maj" else 0.0,
                                     is_glide,
                                     fixed_glide_semitones if is_glide else 0.0,
                                     texture, 0.15, 2.0])
                        choices.append(dict(ramp_s=float(ramp_s),
                                            chord_start=chord_start,
                                            chord_end=chord_end,
                                            is_glide=bool(is_glide),
                                            texture=float(texture)))
    return choices, np.asarray(rows, dtype=np.float64)


def best_by_gp_predict(gp, context, fixed_dist, fixed_glide_semitones=0.0,
                       kappa=1.0):
    """Score every policy combination under a lower confidence bound
    (mean - kappa*std) and return the argmax."""
    choices, F = candidate_grid(context, fixed_dist, fixed_glide_semitones)
    means, stds = gp.predict_utility_batch(F)
    scores = means - kappa * stds
    i = int(np.argmax(scores))
    best = dict(choices[i])
    best.update(mean=float(means[i]), std=float(stds[i]), score=float(scores[i]))
    return best


def main():
    ap = argparse.ArgumentParser(description="Arc-selection policy.")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--kappa", type=float, default=1.0,
                    help="lower-confidence-bound penalty (score = mean - kappa*std)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--pool-meta", dest="pool_meta", default=None,
                    help="arc_pool_meta.json (default: ../weights/, else "
                         "../human_ratings/)")
    args = ap.parse_args()
    if not args.demo:
        ap.print_help()
        return

    import json
    gp = PreferenceGP.load(args.weights)
    meta_path = Path(args.pool_meta) if args.pool_meta else _pool_meta_path()
    pool = {m["arc_id"]: m for m in json.loads(Path(meta_path).read_text())}

    print(f"frozen posterior: {len(gp.ids)} arcs, "
          f"{len(gp.feature_names)} features")
    mm, ms, be = gp.selfcheck()
    print(f"self-check: max |mean err| {mm:.2e}, max |std err| {ms:.2e}, "
          f"batch-vs-single {be:.2e}")

    contexts = pool_contexts(pool)
    ctx = contexts[0]
    dists = [float(pool[a].get("audible_dist", 0.0)) for a in gp.ids
             if a in pool and context_of(pool[a]) == ctx]
    fixed_dist = float(np.median(dists)) if dists else 0.0
    print(f"\npool contexts: {len(contexts)}; querying {ctx}, "
          f"audible_dist={fixed_dist:.3f}")

    base = lookup_best_arc(pool, gp, ctx)
    if base is None:
        print("lookup_best_arc: no pool arc at this exact context")
    else:
        aid, m, s = base
        arc = pool[aid]
        print(f"lookup_best_arc  (baseline): {aid}  f={m:+.3f}+-{s:.3f}  "
              f"ramp={arc['ramp_s']}s "
              f"chord={arc.get('chord_start')}->{arc.get('chord_end')}")

    gpb = best_by_gp_predict(gp, ctx, fixed_dist, kappa=args.kappa)
    print(f"best_by_gp_predict (kappa={args.kappa}): ramp={gpb['ramp_s']:.2f}s "
          f"chord={gpb['chord_start']}->{gpb['chord_end']} "
          f"glide={gpb['is_glide']} "
          f"texture={TEXTURE_NAMES.get(gpb['texture'], gpb['texture'])}  "
          f"f={gpb['mean']:+.3f}+-{gpb['std']:.3f} score={gpb['score']:+.3f}")


if __name__ == "__main__":
    main()
