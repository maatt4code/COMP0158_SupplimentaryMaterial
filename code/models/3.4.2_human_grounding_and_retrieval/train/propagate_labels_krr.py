"""Propagate one rater's 150 labels across the whole preset bank, via theta-KRR.

Step 2 of Section 3.4.2, and the step that produces the human valence axis the
deployed system navigates.

The problem this solves. Nine listeners rated 150 clips and agreed on almost
nothing (Krippendorff's alpha = 0.096, see rating_agreement.py), so a consensus
label does not exist to be learned. What does exist is one rater's internally
consistent 150 ternary judgements. Propagating those across the 20,000-preset
bank gives a coherent, navigable valence axis -- an individual curated prior,
explicitly not a claim about universal agreement. The report frames it that way
and so does this script.

What it does:

  1. Loads the ratings through the same hygiene rules as rating_agreement.py.
  2. Builds theta features for each rated clip: 9 scalars with log-f0 and
     log-cutoff, plus 32 normalised harmonics, z-scored against the bank. Each
     clip's stored theta is VERIFIED against the bank row it was rendered from,
     so a reordered bank aborts rather than corrupting every label.
  3. Runs leave-one-out cross-validation over candidate propagation models --
     k-NN across k and two metrics, and RBF kernel ridge across alpha and gamma
     grids -- scoring MAE, Spearman and sign accuracy against predict-zero and
     predict-mean baselines. LOOCV picks the model BEFORE it is used, which is
     the point: the winner's LOOCV error is the propagation-quality number the
     report quotes.
  4. Checks the winner's predicted signs against consensus sign on the
     consensus-extreme clips -- the only stratum where raters agree at all.
  5. With --propagate, refits the winner on all rated clips, predicts valence
     for every bank row, and writes labeled_index_human.csv NEXT TO each source
     index. The original file is never modified and an existing output is never
     overwritten. A JSON sidecar records the model and its LOOCV statistics.

MERT embeddings can enter the same LOOCV as a second feature family with
--embeddings. Propagating with them additionally needs embeddings for every
bank row; theta-space propagation has no such dependency, which is part of why
it wins.

Run:
  python propagate_labels_krr.py --bank-index <bank>/labeled_index.csv
  python propagate_labels_krr.py --bank-index <bank>/labeled_index.csv --propagate
  python propagate_labels_krr.py --bank-index <bank>/labeled_index.csv \
      --embeddings ../human_ratings/pool_mert.npz

Previous: rating_agreement.py
Next:     fit_control_ridge.py, and fit_guard.py --label-space human
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_SECTION = _HERE.parents[1]
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_SECTION / "inference"))
sys.path.insert(0, str(_HERE.parents[3] / "common"))
from rating_agreement import load_ratings, clip_name, PRIMARY_RATER  # noqa: E402
from decoupled_engine import DecoupledEngine                        # noqa: E402

RATINGS = _SECTION / "human_ratings" / "valence_ratings.csv"
POOL_META = _SECTION / "human_ratings" / "pool_meta.json"
LOG_KEYS = {"f0_hz", "noise_cutoff_hz"}


def bank_features(engine):
    """(n_bank, 41) features from the bank, plus the z-scoring statistics."""
    scal = engine.scal.copy()
    for j, key in enumerate(engine.scalar_keys):
        if key in LOG_KEYS:
            scal[:, j] = np.log(np.maximum(scal[:, j], 1e-6))
    harm = engine.harm / (engine.harm.sum(axis=1, keepdims=True) + 1e-8)
    feats = np.hstack([scal, harm])
    mu = feats.mean(axis=0)
    sd = feats.std(axis=0)
    sd[sd < 1e-4] = 1.0
    return (feats - mu) / sd, mu, sd


def pool_features(entries, engine, mu, sd):
    """Features for the rated clips, each verified against its bank anchor row.

    The verification is the important part: if the bank's row order has changed
    since the pool was generated, anchor_idx points at a different preset and
    every propagated label would be quietly wrong.
    """
    rows, names, n_checked = [], [], 0
    for e in entries:
        scal = []
        for key in engine.scalar_keys:
            v = float(e["theta_scalars"][key])
            scal.append(np.log(max(v, 1e-6)) if key in LOG_KEYS else v)
        harm = np.asarray(e["harm_dist"], float)
        harm = harm / (harm.sum() + 1e-8)

        idx = e.get("anchor_idx")
        if idx is not None and not engine.is_fallback_bank \
                and int(idx) < len(engine.scal):
            stored = np.array([float(e["theta_scalars"][k])
                               for k in engine.scalar_keys])
            if not np.allclose(engine.scal[int(idx)], stored,
                               rtol=1e-4, atol=1e-6):
                raise SystemExit(
                    f"\npool metadata theta for {clip_name(e['clip_path'])} does "
                    f"not match bank row {idx}.\nWrong bank files or a changed "
                    f"row order; refusing to propagate.\n")
            n_checked += 1
        rows.append(np.concatenate([scal, harm]))
        names.append(clip_name(e["clip_path"]))
    print(f"pool features: {len(rows)} clips, {n_checked} verified against "
          f"their bank anchor rows")
    return (np.asarray(rows) - mu) / sd, names


def knn_predict(train_X, train_y, test_X, k, metric):
    if metric == "cosine":
        a = train_X / (np.linalg.norm(train_X, axis=1, keepdims=True) + 1e-12)
        b = test_X / (np.linalg.norm(test_X, axis=1, keepdims=True) + 1e-12)
        d = 1.0 - b @ a.T
    else:
        d = ((test_X[:, None, :] - train_X[None, :, :]) ** 2).sum(-1)
    return train_y[np.argsort(d, axis=1)[:, :k]].mean(axis=1)


def krr_fit_predict(train_X, train_y, test_X, alpha, gamma):
    def rbf(A, B):
        return np.exp(-gamma * ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    K = rbf(train_X, train_X)
    coef = np.linalg.solve(K + alpha * np.eye(len(train_X)), train_y)
    return rbf(test_X, train_X) @ coef


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        order = np.argsort(v)
        r = np.empty(len(v)); r[order] = np.arange(len(v))
        out = np.empty(len(v))
        for val in np.unique(v):
            out[v == val] = np.mean(r[v == val])
        return out
    rx, ry = rank(x), rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def loocv(X, y, predict_fn):
    n = len(y)
    preds = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        preds[i] = predict_fn(X[mask], y[mask], X[i:i + 1])[0]
    return preds


def evaluate(y, preds):
    mae = float(np.mean(np.abs(preds - y)))
    rs = spearman(preds, y)
    nz = y != 0
    sign_acc = (float(np.mean(np.sign(preds[nz]) == np.sign(y[nz])))
                if nz.any() else float("nan"))
    return mae, rs, sign_acc


def main():
    ap = argparse.ArgumentParser(
        description="LOOCV model selection and human-label propagation.")
    ap.add_argument("--bank-index", dest="bank_index", nargs="+", required=True,
                    metavar="CSV", help="labelled_index.csv file(s) forming the bank")
    ap.add_argument("--csv", default=None, help=f"ratings (default {RATINGS})")
    ap.add_argument("--pool-meta", dest="pool_meta", default=None)
    ap.add_argument("--rater", default=PRIMARY_RATER,
                    help="label source. One rater by design: at the measured "
                         "agreement a consensus label is not meaningful")
    ap.add_argument("--consensus", action="store_true",
                    help="use mean-consensus labels instead of one rater. NOT "
                         "recommended: alpha across all raters is 0.096")
    ap.add_argument("--embeddings", default=None,
                    help="optional .npz of embeddings keyed by clip filename, "
                         "added to the LOOCV as a second feature family")
    ap.add_argument("--propagate", action="store_true",
                    help="refit the winner and write labeled_index_human.csv "
                         "next to each bank index")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else RATINGS
    meta_path = Path(args.pool_meta) if args.pool_meta else POOL_META
    for p in (csv_path, meta_path):
        if not p.exists():
            raise SystemExit(f"\nnot found:\n    {p}\n")

    by_rater = load_ratings(csv_path)
    meta = json.loads(meta_path.read_text())
    meta_by_name = {clip_name(e["clip_path"]): e for e in meta}

    if args.consensus:
        clips = sorted(set(c for d in by_rater.values() for c in d))
        labels = {c: float(np.mean([d[c] for d in by_rater.values() if c in d]))
                  for c in clips}
        label_desc = "consensus(mean)"
    else:
        if args.rater not in by_rater:
            raise SystemExit(f"\nrater {args.rater!r} not in {csv_path} "
                             f"(have {sorted(by_rater)})\n")
        labels = dict(by_rater[args.rater])
        label_desc = f"rater={args.rater}"
    labels = {c: v for c, v in labels.items() if c in meta_by_name}
    print(f"labels: {len(labels)} clips from {label_desc}, dist "
          f"{dict(sorted(collections.Counter(labels.values()).items()))}")

    engine = DecoupledEngine(args.bank_index)
    bank_X, mu, sd = bank_features(engine)
    entries = [meta_by_name[c] for c in sorted(labels)]
    pool_X, pool_names = pool_features(entries, engine, mu, sd)
    y = np.array([labels[c] for c in sorted(labels)])

    families = {"theta": (pool_X, True)}
    if args.embeddings:
        emb = np.load(args.embeddings)
        missing = [c for c in pool_names
                   if c not in emb and Path(c).name not in emb]
        if missing:
            raise SystemExit(f"\n{len(missing)} rated clips missing from "
                             f"{args.embeddings} (e.g. {missing[:3]})\n")
        E = np.stack([emb[c] if c in emb else emb[Path(c).name]
                      for c in pool_names])
        families["mert"] = ((E - E.mean(0)) / (E.std(0) + 1e-8), False)

    results = []
    for fam, (X, can_propagate) in families.items():
        for k, metric in itertools.product([1, 3, 5, 7, 10, 15],
                                           ["euclidean", "cosine"]):
            fn = (lambda a, b, c, k=k, m=metric: knn_predict(a, b, c, k, m))
            preds = loocv(X, y, fn)
            results.append((fam, f"knn k={k} {metric}", can_propagate, fn,
                            preds, *evaluate(y, preds)))
        for alpha, gamma in itertools.product([0.1, 1.0, 10.0],
                                              [0.003, 0.01, 0.03, 0.1]):
            fn = (lambda a, b, c, al=alpha, g=gamma:
                  krr_fit_predict(a, b, c, al, g))
            preds = loocv(X, y, fn)
            results.append((fam, f"krr alpha={alpha} gamma={gamma}",
                            can_propagate, fn, preds, *evaluate(y, preds)))

    base_zero = float(np.mean(np.abs(y)))
    base_mean = float(np.mean(np.abs(y - y.mean())))
    print(f"\nbaselines: predict-zero MAE={base_zero:.3f}, "
          f"predict-mean MAE={base_mean:.3f}")

    results.sort(key=lambda r: r[5])
    print(f"\n-- LOOCV (n={len(y)}), sorted by MAE, top 12 --")
    print(f"  {'family':6s} {'model':26s} {'MAE':>6s} {'rho':>7s} {'sign':>6s}")
    for fam, name, _, _, _, mae, rs, sacc in results[:12]:
        print(f"  {fam:6s} {name:26s} {mae:6.3f} {rs:7.3f} {sacc:6.2f}")

    cons = {}
    for c in labels:
        vals = [d[c] for d in by_rater.values() if c in d]
        if len(vals) >= 2:
            cons[c] = float(np.mean(vals))
    anchors = [c for c, m in cons.items() if abs(m) >= 0.5]
    fam, name, can_prop, predict_fn, best_pred, mae, rs, sacc = results[0]
    name_to_i = {c: i for i, c in enumerate(sorted(labels))}
    hits = [np.sign(best_pred[name_to_i[c]]) == np.sign(cons[c])
            for c in anchors if c in name_to_i]
    print(f"\nwinner: [{fam}] {name} -- MAE={mae:.3f} rho={rs:.3f} "
          f"sign_acc={sacc:.2f}")
    print(f"anchor check: LOOCV sign matches consensus sign on "
          f"{int(np.sum(hits))}/{len(hits)} consensus-extreme clips")

    if not args.propagate:
        print("\n(LOOCV only. Rerun with --propagate to write "
              "labeled_index_human.csv)")
        return
    if not can_prop:
        raise SystemExit(
            "\nthe winner uses embedding features, so propagation needs "
            "embeddings for every bank row.\nRerun without --embeddings to "
            "propagate in theta space.\n")

    import pandas as pd
    chord_defaults = {"third_interval": 3.0, "third_gain": 0.0,
                      "fifth_gain": 0.0, "octave_gain": 0.0}
    all_stats = []
    for src in engine.bank_files:
        out_path = Path(src).with_name("labeled_index_human.csv")
        if out_path.exists():
            raise SystemExit(f"\nrefusing to overwrite {out_path}\n")
        df = pd.read_csv(src)
        for col, dv in chord_defaults.items():
            if col not in df.columns:
                df[col] = dv
        feat = df.copy()
        feat[list(chord_defaults)] = feat[list(chord_defaults)].fillna(chord_defaults)
        scal = feat[engine.scalar_keys].to_numpy(float)
        for j, key in enumerate(engine.scalar_keys):
            if key in LOG_KEYS:
                scal[:, j] = np.log(np.maximum(scal[:, j], 1e-6))
        harm = feat[engine.harm_cols].to_numpy(float)
        harm = harm / (harm.sum(axis=1, keepdims=True) + 1e-8)
        X_file = (np.hstack([scal, harm]) - mu) / sd
        ok = ~np.isnan(X_file).any(axis=1)

        preds = np.full(len(df), np.nan)
        Xok = X_file[ok]
        out = np.empty(len(Xok))
        for lo in range(0, len(Xok), 2000):
            hi = min(lo + 2000, len(Xok))
            out[lo:hi] = predict_fn(pool_X, y, Xok[lo:hi])
        preds[ok] = np.clip(out, -1.0, 1.0)

        df["valence_human"] = preds
        df["valence"] = preds
        df.to_csv(out_path, index=False)
        print(f"written -> {out_path}  ({int(ok.sum())} labelled, "
              f"{int((~ok).sum())} without features)")
        all_stats.append((float(np.nanmean(preds)), float(np.nanstd(preds))))
    print("bank predictions: " + ", ".join(f"mean={m:.3f} std={s:.3f}"
                                           for m, s in all_stats))

    sidecar = _SECTION / "data" / "propagation_model.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({
        "label_source": label_desc, "n_labels": int(len(y)),
        "winner": f"[{fam}] {name}",
        "loocv_mae": mae, "loocv_spearman": rs, "loocv_sign_acc": sacc,
        "baseline_zero_mae": base_zero, "baseline_mean_mae": base_mean,
        "bank_files": [Path(f).parent.name + "/" + Path(f).name
                       for f in engine.bank_files],
    }, indent=2))
    print(f"model record -> {sidecar}")


if __name__ == "__main__":
    main()
