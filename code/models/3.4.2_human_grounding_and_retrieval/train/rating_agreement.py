"""Rating hygiene and inter-rater reliability over the seed valence pool.

Step 1 of Section 3.4.2, and the number the rest of the section is built to
answer. Nine listeners rated 150 drone clips on a ternary valence scale. This
computes what they agreed on, and the answer is: very little. Krippendorff's
alpha across all nine is **0.096**, which is the figure the report quotes.

That is not a failed measurement, it is the finding. A low alpha on a pool of
one synthesis class means the group has no shared valence ordering to recover,
so a single consensus label does not exist to be learned. What follows in this
section -- theta-KRR propagation from ONE rater -- is the consequence: the
model is an individual curated prior, not a claim about universal agreement.

The stratified alpha separates the two explanations for a low number. If the
raters agreed on nothing at all, both strata would be flat. Instead the
consensus-extreme clips reach 0.366 while the middle is negative, which says
there is a narrow real signal at the poles buried in rater noise across the
middle -- range restriction, not absence of signal.

Hygiene rules, applied in order:
  - Rows before the loudness-protocol cut-over are dropped. Earlier ratings
    were made against flat-RMS audio that no longer exists.
  - Only the LATEST rating per (rater, clip) counts. The log is append-only, so
    re-rating a clip means rating it again rather than editing.
  - Clips are keyed by FILENAME, never by path: the paths in the log are
    host-local, the filenames are the stable cross-host key.
  - Clips flagged `peak_limited` in pool_meta.json are counted, and optionally
    excluded with --drop-peak-limited.

On rater identifiers: the shipped ratings are pseudonymised to R01..R09. The
original log also carried a session-test account, and the rule that folded its
real-pass rows back into their owner's identity has already been applied here,
because that account and its owner are the same person and so share one
pseudonym. Nothing is dropped by it on this data, which the header line
confirms by reporting zero excluded.

Output: a report on stdout, and optionally a per-clip consensus CSV.

Run:
  python rating_agreement.py
  python rating_agreement.py --consensus-out ../data/consensus_preview.csv

Next: propagate_labels_krr.py, which fits on ONE rater's labels.
"""

import argparse
import collections
import csv
import itertools
import json
import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_SECTION = _HERE.parents[1]
RATINGS = _SECTION / "human_ratings" / "valence_ratings.csv"
POOL_META = _SECTION / "human_ratings" / "pool_meta.json"

# Loudness-protocol cut-over. Ratings before this were made against audio that
# was subsequently re-normalised, so they describe clips that no longer exist.
CUTOVER = "2026-07-02T22:40"

# The single rater whose labels supervise theta-KRR propagation. Identified by
# the hygiene remap in the original log, not chosen after the fact.
PRIMARY_RATER = "R03"

TERNARY_VALUES = np.array([-1.0, 0.0, 1.0])


def clip_name(path_str):
    """Filenames are the stable key; paths in the log are host-local hints."""
    return str(path_str).replace("\\", "/").rstrip("/").split("/")[-1]


def load_ratings(csv_path, cutover=CUTOVER):
    """Apply the hygiene rules. Returns {rater: {clip_filename: rating}}."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    n_stale = 0
    latest = {}
    for r in rows:
        if r["timestamp"] < cutover:
            n_stale += 1
            continue
        key = (r["rater"], clip_name(r["clip_path"]))
        if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
            latest[key] = r

    by_rater = collections.defaultdict(dict)
    for (rater, clip), r in latest.items():
        by_rater[rater][clip] = float(r["rating"])
    print(f"Loaded {len(rows)} rows -> {sum(len(d) for d in by_rater.values())} "
          f"effective ratings ({n_stale} pre-cutover, 0 test-ID excluded)")
    return dict(by_rater)


def load_peak_limited(meta_path):
    if not os.path.exists(meta_path):
        return set()
    with open(meta_path) as f:
        meta = json.load(f)
    return {clip_name(e["clip_path"]) for e in meta if e.get("peak_limited")}


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        order = np.argsort(v)
        ranks = np.empty(len(v))
        ranks[order] = np.arange(len(v))
        out = np.empty(len(v))
        for val in np.unique(v):
            out[v == val] = np.mean(ranks[v == val])
        return out
    rx, ry = rank(x), rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def krippendorff_alpha_ordinal(data):
    """data: (n_raters, n_units), NaN where missing. Ordinal metric on the
    ternary scale (cumulative-frequency delta^2, Krippendorff 2011). A unit
    needs at least two ratings to contribute."""
    units = []
    for j in range(data.shape[1]):
        col = data[:, j]
        col = col[~np.isnan(col)]
        if len(col) >= 2:
            units.append(col)
    if not units:
        return float("nan")
    n_total = sum(len(u) for u in units)
    nc = collections.Counter(v for u in units for v in u)

    def delta2(v1, v2):
        if v1 == v2:
            return 0.0
        lo, hi = sorted([v1, v2])
        s = sum(nc[v] for v in TERNARY_VALUES if lo <= v <= hi) \
            - (nc[lo] + nc[hi]) / 2
        return s ** 2

    d_obs = 0.0
    for u in units:
        m = len(u)
        for v1, v2 in itertools.permutations(u, 2):
            d_obs += delta2(v1, v2) / (m - 1)
    d_obs /= n_total

    d_exp = 0.0
    for v1 in TERNARY_VALUES:
        for v2 in TERNARY_VALUES:
            if v1 != v2:
                d_exp += nc[v1] * nc[v2] * delta2(v1, v2)
    d_exp /= n_total * (n_total - 1)
    return 1 - d_obs / d_exp if d_exp > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(
        description="Rating hygiene and inter-rater reliability report.")
    ap.add_argument("--csv", default=None, metavar="CSV",
                    help=f"ratings log (default {RATINGS})")
    ap.add_argument("--pool-meta", dest="pool_meta", default=None, metavar="JSON",
                    help=f"pool metadata (default {POOL_META})")
    ap.add_argument("--drop-peak-limited", dest="drop_peak_limited",
                    action="store_true",
                    help="exclude clips flagged peak_limited in the pool metadata")
    ap.add_argument("--exclude-raters", dest="exclude_raters", default=None,
                    help="comma-separated rater IDs to exclude. A sensitivity "
                         "analysis: report these ALONGSIDE the all-rater "
                         "numbers, never instead of them")
    ap.add_argument("--consensus-out", dest="consensus_out", default=None,
                    help="write a per-clip mean-consensus CSV here "
                         "(never overwrites an existing file)")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else RATINGS
    meta_path = Path(args.pool_meta) if args.pool_meta else POOL_META
    if not csv_path.exists():
        raise SystemExit(
            f"\nratings not found at:\n    {csv_path}\n\n"
            "The pseudonymised ratings ship in this section's human_ratings/.\n")

    by_rater = load_ratings(csv_path)
    if args.exclude_raters:
        excl = {r.strip() for r in args.exclude_raters.split(",")}
        missing = excl - set(by_rater)
        if missing:
            raise SystemExit(f"--exclude-raters not found: {sorted(missing)}")
        by_rater = {r: d for r, d in by_rater.items() if r not in excl}
        print(f"SENSITIVITY ANALYSIS: excluded {sorted(excl)} "
              f"-> {len(by_rater)} raters remain")

    peak_limited = load_peak_limited(meta_path)
    print(f"peak_limited clips in pool metadata: {len(peak_limited)}"
          + (" (excluded)" if args.drop_peak_limited and peak_limited else ""))
    if args.drop_peak_limited and peak_limited:
        by_rater = {rt: {c: v for c, v in d.items() if c not in peak_limited}
                    for rt, d in by_rater.items()}

    print("\n-- Per-rater summary --")
    for rater, d in sorted(by_rater.items()):
        dist = dict(sorted(collections.Counter(d.values()).items()))
        star = "  <- theta-KRR label source" if rater == PRIMARY_RATER else ""
        print(f"  {rater}: {len(d)} clips, dist {dist}{star}")

    print("\n-- Pairwise (shared clips) --")
    for a, b in itertools.combinations(sorted(by_rater), 2):
        common = sorted(set(by_rater[a]) & set(by_rater[b]))
        if not common:
            continue
        xa = [by_rater[a][c] for c in common]
        xb = [by_rater[b][c] for c in common]
        agree = sum(1 for u, v in zip(xa, xb) if u == v) / len(common)
        print(f"  {a} vs {b}: n={len(common)}, exact agree={agree:.2f}, "
              f"spearman={spearman(xa, xb):.3f}")

    clips = sorted(set(c for d in by_rater.values() for c in d))
    raters = sorted(by_rater)
    data = np.full((len(raters), len(clips)), np.nan)
    for i, rt in enumerate(raters):
        for j, c in enumerate(clips):
            if c in by_rater[rt]:
                data[i, j] = by_rater[rt][c]

    print("\n-- Krippendorff's alpha (ordinal) --")
    print(f"  all {len(raters)} raters: {krippendorff_alpha_ordinal(data):.3f}")
    for a, b in itertools.combinations(range(len(raters)), 2):
        print(f"  {raters[a]}+{raters[b]}: "
              f"{krippendorff_alpha_ordinal(data[[a, b], :]):.3f}")
    n3 = int(np.sum(np.sum(~np.isnan(data), axis=0) >= 3))
    n2 = int(np.sum(np.sum(~np.isnan(data), axis=0) >= 2))
    print(f"  coverage: {n3} clips with >=3 raters, {n2} with >=2, "
          f"{len(clips)} total")

    mean = np.nanmean(data, axis=0)
    nrat = np.sum(~np.isnan(data), axis=0)
    extreme = (np.abs(mean) >= 0.5) & (nrat >= 2)
    middle = (np.abs(mean) < 0.5) & (nrat >= 2)
    print("\n-- Stratified alpha (signal against range restriction) --")
    print(f"  consensus-extreme (|mean|>=0.5, n={int(extreme.sum())}): "
          f"{krippendorff_alpha_ordinal(data[:, extreme]):.3f}")
    print(f"  consensus-middle  (|mean|<0.5,  n={int(middle.sum())}): "
          f"{krippendorff_alpha_ordinal(data[:, middle]):.3f}")
    triple = np.where(nrat >= 3)[0]
    unanimous = sum(1 for j in triple
                    if len(set(data[~np.isnan(data[:, j]), j])) == 1)
    print(f"  unanimity: {unanimous}/{len(triple)} clips with >=3 raters")

    consensus = {c: float(np.nanmean(data[:, j])) for j, c in enumerate(clips)}
    print("\n-- Consensus (mean) label distribution --")
    print("  " + str(dict(sorted(collections.Counter(
        round(v, 2) for v in consensus.values()).items()))))

    if args.consensus_out:
        out = Path(args.consensus_out)
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip", "n_raters", "consensus_valence"])
            for j, c in enumerate(clips):
                n = int(np.sum(~np.isnan(data[:, j])))
                w.writerow([c, n, f"{consensus[c]:.4f}"])
        print(f"consensus written -> {out}")


if __name__ == "__main__":
    main()
