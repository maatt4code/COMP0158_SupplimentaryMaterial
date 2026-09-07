"""The control ridge: a human-grounded audio -> valence model, for contrast.

Step 3 of Section 3.4.2, and a deliberate control rather than a component.

It exists to disambiguate a result. If the propagated human labels produce a
better mapping than the automated surrogate did, is that because propagation
works, or because the mapping task is simply easier with valid labels? Training
a plain ridge DIRECTLY on the 150 raw rated clips -- no propagation anywhere --
separates the two:

  control good, propagated bad  ->  propagation is the bottleneck
  both bad                      ->  the mapping is hard even with valid labels

Without this the comparison is confounded.

It doubles as the first judge in the cross-judge comparison with a valid valence
signal for this synthesis class. The frozen MERT/DEAM judge of Section 3.4.1 is
flat on drones; this one is trained on human labels of the very clips in
question, so it is the human-grounded counterpart to report alongside.

The controlled contrast is exact by construction: the features are the SAME
MERT-v1-95M embedding family the frozen judge consumes. Same features, different
training labels. Anything that differs is the labels.

Three things are reported besides the ridge:
  - an RBF kernel ridge on the same embeddings, so a null result cannot be
    blamed on linearity
  - a permutation floor: the same model-selection procedure on shuffled labels,
    which says how large a correlation chance alone buys at n=150
  - the frozen judge's own valence on these clips, expected to be near null

Output (in --out-dir):
  control_ridge_valence.npz   scaler and ridge coefficients
  control_ridge_valence.json  alpha, the CV table, the feature protocol
  control_ridge_loocv.csv     per-clip LOOCV predictions

The model ships as an .npz of coefficients, not a pickle: a
StandardScaler-plus-Ridge pipeline is fully described by four arrays, and
reconstructing it from those neither pins a scikit-learn version nor asks
anyone to unpickle a downloaded object. Verified bit-identical to the pipeline's
own predictions on the shipped float64 features.

Run:
  python fit_control_ridge.py
  python fit_control_ridge.py --no-save

Previous: rating_agreement.py, extract_pool_mert.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_SECTION = _HERE.parents[1]
sys.path.insert(0, str(_HERE.parent))
from rating_agreement import load_ratings, clip_name, PRIMARY_RATER  # noqa: E402
from propagate_labels_krr import (evaluate, loocv, krr_fit_predict,   # noqa: E402
                                  spearman)

RATINGS = _SECTION / "human_ratings" / "valence_ratings.csv"
POOL_META = _SECTION / "human_ratings" / "pool_meta.json"
MERT_NPZ = _SECTION / "human_ratings" / "pool_mert.npz"
MERT_META = _SECTION / "human_ratings" / "pool_mert_meta.json"

ALPHA_GRID = [10.0 ** k for k in range(-1, 7)]

# The theta-KRR propagation model's LOOCV row, printed as the comparison
# reference. Reproduce it with propagate_labels_krr.py; it is not recomputed here.
THETA_KRR_REF = dict(mae=0.566, spearman=0.337, sign_acc=0.694)


def load_xy(csv_path, pool_meta_path, mert_npz, rater):
    """(names, embeddings, labels, judge valence), aligned by clip filename."""
    by_rater = load_ratings(csv_path)
    if rater not in by_rater:
        raise SystemExit(f"\nrater {rater!r} not in {csv_path}\n")
    meta = json.loads(Path(pool_meta_path).read_text())
    meta_by_name = {clip_name(e["clip_path"]): e for e in meta}
    labels = {c: v for c, v in by_rater[rater].items() if c in meta_by_name}

    if not Path(mert_npz).exists():
        raise SystemExit(
            f"\nMERT embeddings not found at:\n    {mert_npz}\n\n"
            "They ship with this section. Regenerate with extract_pool_mert.py "
            "if you have the pool audio.\n")
    z = np.load(mert_npz)
    names, X, y, judge_v = [], [], [], []
    for c in sorted(labels):
        fname = Path(meta_by_name[c]["clip_path"]).name
        if fname not in z:
            raise SystemExit(f"\n{fname} is rated but missing from {mert_npz}\n")
        names.append(c)
        X.append(z[fname])
        y.append(labels[c])
        judge_v.append(meta_by_name[c].get("achieved_v",
                                           meta_by_name[c].get("target_v")))
    return names, np.array(X), np.array(y, dtype=float), judge_v


def make_model(alpha):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def loocv_predict(X, y, alpha):
    preds = np.empty(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        preds[i] = make_model(alpha).fit(X[tr], y[tr]).predict(X[i:i + 1])[0]
    return preds


def main():
    ap = argparse.ArgumentParser(
        description="Fit the human-grounded control ridge on MERT embeddings.")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--pool-meta", dest="pool_meta", default=None)
    ap.add_argument("--embeddings", default=None, help=f"default {MERT_NPZ}")
    ap.add_argument("--rater", default=PRIMARY_RATER)
    ap.add_argument("--out-dir", dest="out_dir", default=None,
                    help="default ../data/control_ridge. Not ../weights: that "
                         "holds the fitted model that ships")
    ap.add_argument("--no-save", dest="no_save", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else RATINGS
    meta_path = Path(args.pool_meta) if args.pool_meta else POOL_META
    npz_path = Path(args.embeddings) if args.embeddings else MERT_NPZ
    out_dir = (Path(args.out_dir) if args.out_dir
               else _SECTION / "data" / "control_ridge")

    names, X, y, judge_v = load_xy(csv_path, meta_path, npz_path, args.rater)
    print(f"control ridge: {len(y)} rated clips, {X.shape[1]}-dim embeddings "
          f"({npz_path.name}, {X.dtype})")

    cv_table, preds_by_alpha = [], {}
    for alpha in ALPHA_GRID:
        preds = loocv_predict(X, y, alpha)
        mae, rho, sacc = evaluate(y, preds)
        cv_table.append(dict(alpha=alpha, mae=mae, spearman=rho, sign_acc=sacc))
        preds_by_alpha[alpha] = preds
        print(f"  alpha={alpha:>9.1f}  MAE={mae:.3f}  rho={rho:.3f}  "
              f"sign_acc={sacc:.3f}")
    best = min(cv_table, key=lambda r: r["mae"])
    print(f"\nLOOCV winner: alpha={best['alpha']} MAE={best['mae']:.3f} "
          f"rho={best['spearman']:.3f} sign_acc={best['sign_acc']:.3f}")

    # RBF kernel ridge on the same embeddings, so a null cannot be blamed on
    # linearity: this is the model family that succeeds on theta features.
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    med_d2 = float(np.median(((Xz[:, None] - Xz[None, :]) ** 2).sum(-1)))
    krr_table = []
    for alpha, gscale in itertools.product([0.1, 1.0, 10.0],
                                           [0.1, 0.3, 1.0, 3.0]):
        gamma = gscale / med_d2
        preds = loocv(Xz, y, lambda a, b, c, al=alpha, g=gamma:
                      krr_fit_predict(a, b, c, al, g))
        mae, rho, sacc = evaluate(y, preds)
        krr_table.append(dict(alpha=alpha, gamma=gamma, mae=mae,
                              spearman=rho, sign_acc=sacc))
    kb = min(krr_table, key=lambda r: r["mae"])
    print(f"RBF-KRR on the same embeddings, LOOCV best: alpha={kb['alpha']} "
          f"gamma={kb['gamma']:.2e} MAE={kb['mae']:.3f} rho={kb['spearman']:.3f} "
          f"sign_acc={kb['sign_acc']:.3f}")

    # Permutation floor: the same selection procedure on shuffled labels.
    rng = np.random.default_rng(0)
    null_rhos = []
    for _ in range(20):
        yp = rng.permutation(y)
        null_rhos.append(abs(evaluate(yp, loocv_predict(X, yp, best["alpha"]))[1]))
    print(f"permutation null (20 shuffles, |rho|): "
          f"median {np.median(null_rhos):.3f}, "
          f"p95 {np.percentile(null_rhos, 95):.3f}")
    print(f"theta-KRR propagation reference: MAE={THETA_KRR_REF['mae']} "
          f"rho={THETA_KRR_REF['spearman']} sign_acc={THETA_KRR_REF['sign_acc']}")

    jv = np.array([v for v in judge_v if v is not None], dtype=float)
    if len(jv) == len(y):
        print(f"frozen judge valence vs human on these clips: "
              f"spearman={spearman(jv, y):.3f} (the known-invalid contrast)")

    if args.no_save:
        return

    from sklearn.preprocessing import StandardScaler  # noqa: F401
    model = make_model(best["alpha"]).fit(X, y)
    sc = model.named_steps["standardscaler"]
    rg = model.named_steps["ridge"]

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_out = out_dir / "control_ridge_valence.npz"
    np.savez(npz_out, scaler_mean=sc.mean_, scaler_scale=sc.scale_,
             coef=np.asarray(rg.coef_), intercept=np.asarray(rg.intercept_),
             alpha=np.asarray(rg.alpha))

    # Never ship a reconstruction that does not reproduce the pipeline.
    Xs = (X - sc.mean_) / sc.scale_
    check = Xs @ np.asarray(rg.coef_) + float(rg.intercept_)
    err = float(np.max(np.abs(model.predict(X) - check)))
    print(f"\nnpz reconstruction vs pipeline: max|diff| {err:.3e}")
    if err > 1e-12:
        raise SystemExit("the npz does not reproduce the pipeline; not saving")

    mert_meta = json.loads(MERT_META.read_text()) if MERT_META.exists() else {}
    (out_dir / "control_ridge_valence.json").write_text(json.dumps(dict(
        generated=datetime.now().isoformat(timespec="seconds"),
        purpose="control model for the propagation comparison, and a "
                "human-grounded audio->valence scorer",
        alpha=best["alpha"], loocv=cv_table, loocv_krr_rbf=krr_table,
        permutation_null_abs_rho=dict(
            n_shuffles=20, median=float(np.median(null_rhos)),
            p95=float(np.percentile(null_rhos, 95))),
        theta_krr_reference=THETA_KRR_REF,
        n_labels=int(len(y)), rater=args.rater,
        labels="ternary {-1,0,+1}, one rater by design",
        feature_protocol=mert_meta,
        model="StandardScaler + Ridge, stored as coefficient arrays",
        scoring_note="reproduce the feature protocol above, then "
                     "(x - scaler_mean) / scaler_scale @ coef + intercept",
    ), indent=1))
    with open(out_dir / "control_ridge_loocv.csv", "w") as f:
        f.write("clip,human,loocv_pred\n")
        for n, yy, p in zip(names, y, preds_by_alpha[best["alpha"]]):
            f.write(f"{n},{yy},{p:.4f}\n")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
