"""Fit an empirical semi-Markov model of arrangement timing on a corpus.

Step B of the scheduler's timing pipeline. It clusters the transition
descriptors that `extract_transition_typology.py` cached, then fits
state-conditional dwell distributions and a transition matrix by counting real
observation sequences. The result is `../weights/hsmm_transitions.json`, which
`../inference/scheduler.py` and `../inference/coherence_reranker.py` both read.

**Any corpus works.** Nothing here is specific to the audio this project used;
the fit ships, the corpus does not. Re-run the extractor on your own
directories and refit.

WHAT THIS TRAINS: state-conditional dwell windows and a transition matrix, by
maximum likelihood on real ordered sequences -- each track an ordered
hold-transition-hold sequence with a type label. It answers "train the model"
with zero new listening data.

WHAT IT DOES NOT CLAIM: corpus dynamics are not listener preference. This says
how the reference recordings arrange holds and transitions over time -- a
compositional prior, like the duration priors used to size the arc vocabulary
-- and never an emotion or preference model.

NAMING. Do not call this "a fitted HSMM" without qualification. Nothing is
hidden: the states are observed cluster labels, not decoded latents. The
accurate description is an EMPIRICAL SEMI-MARKOV PRIOR -- a transition matrix
plus state-conditional empirical duration bounds, fit by counting rather than
by Baum-Welch. A latent-state Gaussian HSMM would be the unsupervised
comparison, and is not built.

THE MAPPING, which is the honest core of this. Bridging the corpus types onto
the scheduler's three bed directives was originally a forced tercile split on
median hold length, and expanding the corpus PROVED that fragile: adding groups
reshuffled which cluster landed in which bed state, because the type
hold-medians clustered tightly and then jumped. Slicing a few seconds of spread
into three "levels" was quantizing noise. So the mapping now makes ONLY the
distinction the data supports, found by locating the dominant gap:

    SETTLED types (long holds)          -> drone_solo
    ACTIVE types (everything below it)  -> the bed-on states

and the corpus is declared SILENT on duck-versus-bursts. There is nothing in
album audio that distinguishes "duck the bed under the drone's swell" from
"burst the bed at its dips"; that is a synthesis-side directive with no corpus
correlate, so it stays on the hand-set ratio, labelled as authored. What the
corpus does inform, and what ships, is the RETURN RATE: how often a change goes
back to stillness rather than to another active arrangement. That is a real,
countable, defensible quantity.

STRONG and WEAK are reported separately so that neither launders the other:

  STRONG -- per-state dwell windows and the return rate. The settled state is
    several times longer-held than the active ones, which is a large and stable
    effect. An outlier-exclusion sensitivity check is run and reported, so the
    inherited veto's effect on the shipped numbers is a measured quantity
    rather than an argument.
  WEAK -- the duck-versus-bursts split, authored and corpus-silent.

ALSO EXPORTED: `corpus_type_model`, the full k-state semi-Markov over transition
TYPES -- a row-normalised transition matrix, per-type dwell windows, and the
centroids and scaling needed to classify a new rendered arc. This keeps the
resolution the 3x3 bed reduction throws away, and it is what the
sequential-coherence re-ranker consumes. The two are different layers and both
ship from this one fit.

DEPLOYMENT WARNING: a corpus's own pacing may be faster than a listening
application tolerates. This project's corpus has a q10 hold around 2 seconds,
while the scheduler's floor was chosen from listener pacing complaints. Feeding
a raw corpus window to a live system can reintroduce exactly the failure the
scheduler exists to fix, so the scheduler CLAMPS to a deployment floor and logs
both raw and clamped windows. That clamp is a stated design constraint from
listener evidence, NOT a data finding.

Run:
  python fit_corpus_hsmm.py --cache ../data/typology_transitions.json \
      --out ../weights/hsmm_transitions.json
  python fit_corpus_hsmm.py --selftest        # synthetic sequences, no corpus

Previous: extract_transition_typology.py
Next:     ../inference/scheduler.py and coherence_reranker.py read the output.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_SECTION = _HERE.parents[1]
sys.path.insert(0, str(_HERE.parent))
from extract_transition_typology import (   # noqa: E402
    DESC_NAMES, CLUSTER_DIMS, STATE_MAG_MIN, flag_outliers)

DEFAULT_CACHE = _SECTION / "data" / "typology_transitions.json"
DEFAULT_OUT = _SECTION / "weights" / "hsmm_transitions.json"

FORCED_K = 5    # the type count the shipped fit and the re-ranker both assume

# The scheduler's deployed three-state vocabulary, in the fixed order this
# module always reports against.
BED_STATES = ["drone_solo", "bed_duck", "bed_bursts"]
# The hand-set duck:bursts preference, reused verbatim because the corpus is
# SILENT on this split. Authored, not fit.
HANDSET_ACTIVE_SPLIT = {"bed_duck": 0.6, "bed_bursts": 0.4}


def fit_types(rows, k=FORCED_K, exclude_outliers=True, quiet=False, seed=0):
    """Cluster the transition descriptors into types.

    exclude_outliers=False keeps the flagged outliers in, and is used only for
    the sensitivity check that quantifies whether that veto biases the shipped
    dwell numbers.

    Returns (rows_state, labels, model), where model carries the z-scaling, the
    clustering dimensions and the centroids -- everything `classify_arc` needs
    to assign a NEW arc to the same types without refitting.
    """
    from sklearn.cluster import KMeans

    X_all = np.array([[r[key] if np.isfinite(r.get(key, np.nan)) else 0.0
                       for key in DESC_NAMES] for r in rows])
    i_mag = DESC_NAMES.index("total_magnitude")
    if exclude_outliers:
        ctx = contextlib.redirect_stdout(io.StringIO()) if quiet \
            else contextlib.nullcontext()
        with ctx:
            keep = flag_outliers(X_all, rows)
    else:
        keep = np.ones(len(rows), dtype=bool)
    # Only STATE transitions feed the typology; below the magnitude boundary
    # an event is breathing micro-motion, which is the wander prior.
    state = keep & (X_all[:, i_mag] >= STATE_MAG_MIN)
    rows_state = [r for r, m in zip(rows, state) if m]
    if len(rows_state) < k:
        raise SystemExit(
            f"\nonly {len(rows_state)} state transitions survive the filters, "
            f"which is fewer than k={k}.\nExtract more audio, or lower "
            f"--k.\n")
    X = X_all[state]
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xc = ((X - mu) / sd)[:, CLUSTER_DIMS]
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xc)
    model = dict(mu=mu, sd=sd,
                 cluster_dims=np.asarray(CLUSTER_DIMS, dtype=int),
                 centroids=km.cluster_centers_)
    return rows_state, km.labels_, model


def ordered_sequences(rows_state, labels):
    """Per-track type-label sequences, ordered by start time -- the
    observation sequences a semi-Markov fit actually needs."""
    by_track = defaultdict(list)
    for r, lab in zip(rows_state, labels):
        key = (r.get("group", r.get("artist")), r["track"])
        by_track[key].append((r["t_start_s"], int(lab), r))
    return {key: sorted(items, key=lambda x: x[0])
            for key, items in by_track.items()}


def fit_transition_counts(seqs, k=FORCED_K, alpha=1.0):
    """Laplace-smoothed consecutive-transition counts over types, counted
    within each track's own sequence."""
    C = np.full((k, k), float(alpha))
    for items in seqs.values():
        labs = [lab for _, lab, _ in items]
        for a, b in zip(labs, labs[1:]):
            C[a, b] += 1.0
    return C


def _holds(rows_state, idx):
    return np.array([rows_state[i]["pre_hold_s"] for i in idx
                     if np.isfinite(rows_state[i].get("pre_hold_s", np.nan))])


def hold_window(hs, lo=0.10, hi=0.90):
    return (float(np.quantile(hs, lo)), float(np.quantile(hs, hi)))


def split_settled_active(rows_state, labels, k=FORCED_K):
    """Split the types at the dominant gap in their median hold times.

    The report carries every median and gap, so the split's STABILITY is
    visible in the output rather than asserted. This replaces a forced tercile
    mapping that a corpus expansion showed to be unstable.
    """
    med = {}
    for c in range(k):
        hs = _holds(rows_state, np.where(labels == c)[0])
        med[c] = float(np.median(hs)) if len(hs) else float("nan")
    order = sorted((c for c in range(k) if np.isfinite(med[c])),
                   key=lambda c: med[c])
    gaps = [(med[b] - med[a], i) for i, (a, b) in enumerate(zip(order, order[1:]))]
    best_gap, gi = max(gaps) if gaps else (0.0, -1)
    active, settled = order[:gi + 1], order[gi + 1:]
    others = [g for g, _ in gaps if g != best_gap] or [0.0]
    report = dict(
        type_hold_median_s={f"type_{c}": round(med[c], 1) for c in order},
        sorted_gaps_s=[round(g, 1) for g, _ in gaps],
        dominant_gap_s=round(best_gap, 1),
        next_largest_gap_s=round(max(others), 1),
        gap_dominance_ratio=(round(best_gap / max(max(others), 1e-9), 1)
                             if gaps else None),
        settled_types=[f"type_{c}" for c in settled],
        active_types=[f"type_{c}" for c in active],
        note=("split at the dominant gap; a ratio well above 1 means the "
              "settled/active distinction is robust, while a ratio near 1 "
              "would mean the split is quantizing noise -- the failure mode "
              "of the superseded tercile mapping"))
    return settled, active, report


def fit_return_rate(seqs, settled, active):
    """P(next state is SETTLED | current is ACTIVE), counted over consecutive
    pairs and Laplace-smoothed. The one genuinely corpus-grounded number about
    bed sequencing."""
    to_settled = to_active = 1.0
    for items in seqs.values():
        labs = [lab for _, lab, _ in items]
        for a, b in zip(labs, labs[1:]):
            if a in active:
                if b in settled:
                    to_settled += 1.0
                else:
                    to_active += 1.0
    return to_settled / (to_settled + to_active)


def classify_arc(desc_vec, model):
    """Assign a descriptor vector to its nearest corpus type, using the
    corpus's own scaling, dimensions and centroids -- the exact projection
    `fit_types` clustered in, so a rendered arc lands in the type a corpus row
    with those features would. A pure function of the exported model."""
    z = (np.asarray(desc_vec, dtype=float) - np.asarray(model["mu"])) \
        / np.asarray(model["sd"])
    zc = z[np.asarray(model["cluster_dims"], dtype=int)]
    d = ((np.asarray(model["centroids"]) - zc) ** 2).sum(axis=1)
    return int(np.argmin(d))


def corpus_type_model(counts, rows_state, labels, model, k=FORCED_K):
    """The full k-state semi-Markov over types, for the re-ranker.

    Distinct from the 3x3 bed reduction: this keeps the full resolution. The
    matrix is row-normalised from Laplace-smoothed counts, so it is a proper
    conditional distribution with no zero rows and is safe to re-rank with.
    Per-type dwell is the hold quantile range within each type, which is what
    makes this genuinely semi-Markov rather than a plain first-order chain.
    """
    P = counts / counts.sum(axis=1, keepdims=True)
    dwell, occ = {}, {}
    for c in range(k):
        idx = np.where(labels == c)[0]
        hs = _holds(rows_state, idx)
        dwell[f"type_{c}"] = ([round(x, 1) for x in hold_window(hs)]
                              if len(hs) else None)
        occ[f"type_{c}"] = int(len(idx))
    return dict(
        k=k, desc_names=list(DESC_NAMES),
        cluster_dims=[int(i) for i in CLUSTER_DIMS],
        cluster_dim_names=[DESC_NAMES[i] for i in CLUSTER_DIMS],
        feature_mean=[float(x) for x in np.asarray(model["mu"])],
        feature_std=[float(x) for x in np.asarray(model["sd"])],
        centroids_z=np.asarray(model["centroids"]).round(4).tolist(),
        transition_matrix=P.round(4).tolist(),
        transition_counts=counts.round(3).tolist(),
        dwell_window_s_by_type=dwell, type_occupancy=occ,
        note=("k-state semi-Markov for the sequential-coherence re-ranker. To "
              "use: build an arc's descriptor vector, classify_arc(vec, model) "
              "for the previous and candidate types, then re-rank by "
              "transition_matrix[prev][cand]. Human ratings still decide what "
              "is GOOD; this only breaks ties toward what follows COHERENTLY, "
              "and never overrides a preference judgement. Not the 3x3 bed "
              "layer, which is the scheduler's synthesis control."))


def corpus_overlay_trans(corpus, bed_states=BED_STATES,
                         active_split=HANDSET_ACTIVE_SPLIT):
    """3x3 bed transition matrix: corpus-fit return rate, authored split."""
    p_ret = float(corpus["return_rate_active_to_settled"])
    i_solo = bed_states.index("drone_solo")
    actives = [b for b in bed_states if b != "drone_solo"]
    M = np.zeros((len(bed_states), len(bed_states)))
    tot = sum(active_split.get(b, 0.0) for b in actives) or 1.0
    for b in actives:
        M[i_solo, bed_states.index(b)] = active_split.get(b, 0.0) / tot
    for b in actives:
        i = bed_states.index(b)
        M[i, i_solo] = p_ret
        for o in actives:
            if o != b:
                M[i, bed_states.index(o)] = 1.0 - p_ret
    rs = M.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return (M / rs).tolist()


def fit(cache, out_path=None, k=FORCED_K, seed=0):
    cache = Path(cache)
    if not cache.exists():
        raise SystemExit(
            f"\ndescriptor cache not found at:\n    {cache}\n\n"
            "Build one first:\n"
            "    python extract_transition_typology.py --audio-dir <dir> ...\n")
    rows = json.loads(cache.read_text())
    rows_state, labels, model = fit_types(rows, k=k, seed=seed)
    seqs = ordered_sequences(rows_state, labels)
    settled, active, split_report = split_settled_active(rows_state, labels, k=k)
    counts = fit_transition_counts(seqs, k=k)
    p_ret = fit_return_rate(seqs, set(settled), set(active))
    type_model = corpus_type_model(counts, rows_state, labels, model, k=k)

    idx_s = [i for i, l in enumerate(labels) if l in settled]
    idx_a = [i for i, l in enumerate(labels) if l in active]
    h_settled, h_active = _holds(rows_state, idx_s), _holds(rows_state, idx_a)
    h_all = _holds(rows_state, range(len(rows_state)))

    # Sensitivity: does the inherited outlier veto bias the dwell window?
    rows_no, labels_no, _ = fit_types(rows, k=k, exclude_outliers=False,
                                      quiet=True, seed=seed)
    h_no = _holds(rows_no, range(len(rows_no)))

    out = dict(
        n_state_transitions=len(rows_state), n_tracks=len(seqs),
        n_consecutive_pairs=sum(len(v) - 1 for v in seqs.values()),
        n_groups=len({g for g, _ in seqs}),
        settled_active_split=split_report,
        dwell_window_s_by_group=dict(
            settled=[round(x, 1) for x in hold_window(h_settled)],
            active=[round(x, 1) for x in hold_window(h_active)]),
        dwell_window_s_pooled=[round(x, 1) for x in hold_window(h_all)],
        dwell_outlier_sensitivity=dict(
            with_exclusion=[round(x, 1) for x in hold_window(h_all)],
            without_exclusion=[round(x, 1) for x in hold_window(h_no)],
            note="near-identical means the outlier veto does not bias the "
                 "shipped dwell numbers"),
        return_rate_active_to_settled=round(float(p_ret), 3),
        type_transition_counts=counts.round(3).tolist(),
        corpus_type_model=type_model,
        authored_active_split=dict(HANDSET_ACTIVE_SPLIT),
        note=("STRONG (corpus-fit): per-group dwell windows and "
              "return_rate_active_to_settled. AUTHORED (the corpus is silent): "
              "authored_active_split, the duck-versus-bursts ratio. "
              "corpus_type_model is the full k-state semi-Markov consumed by "
              "the re-ranker; the 3x3 bed reduction is the scheduler's own "
              "layer."))

    print(f"\n{len(rows_state)} STATE transitions, {len(seqs)} tracks, "
          f"{out['n_groups']} groups ({out['n_consecutive_pairs']} pairs)")
    print("\ntype hold medians (s):", split_report["type_hold_median_s"])
    print(f"  gaps {split_report['sorted_gaps_s']} -> dominant "
          f"{split_report['dominant_gap_s']}s vs next "
          f"{split_report['next_largest_gap_s']}s "
          f"(x{split_report['gap_dominance_ratio']} dominance)")
    print(f"  SETTLED {split_report['settled_types']} -> drone_solo")
    print(f"  ACTIVE  {split_report['active_types']} -> bed-on states")
    print("\ndwell windows (q10-q90 of the preceding hold):")
    print(f"  settled {out['dwell_window_s_by_group']['settled']}   "
          f"active {out['dwell_window_s_by_group']['active']}   "
          f"pooled {out['dwell_window_s_pooled']}")
    print(f"  outlier sensitivity: "
          f"{out['dwell_outlier_sensitivity']['with_exclusion']} (shipped) vs "
          f"{out['dwell_outlier_sensitivity']['without_exclusion']} (veto off)")
    print(f"\nreturn rate active->settled: {p_ret:.3f} (corpus-fit); "
          f"duck:bursts {HANDSET_ACTIVE_SPLIT} (AUTHORED, corpus silent)")
    print(f"\nNOTE: the corpus active dwell starts at "
          f"{out['dwell_window_s_by_group']['active'][0]:.1f}s. The scheduler "
          "clamps to its deployment floor on use.")
    diag = [type_model["transition_matrix"][i][i] for i in range(k)]
    print(f"\n{k}-state type model: persistence (diagonal) "
          f"{[round(x, 2) for x in diag]}")
    print(f"  per-type dwell(s)  {type_model['dwell_window_s_by_type']}")
    print(f"  per-type occupancy {type_model['type_occupancy']}")

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {out_path}")
    return out


def selftest():
    """Synthetic sequences with a known answer. No corpus, no audio."""
    def mkrow(group, track, t, dur, hold):
        return dict(group=group, track=track, t_start_s=t, duration_s=dur,
                    pre_hold_s=hold)

    # Deliberately out of chronological order: ordering must come from the
    # start time, never from list position.
    rows_state = [
        mkrow("A", "t1", 40.0, 3.0, 8.0),
        mkrow("A", "t1", 10.0, 3.0, 9.0),
        mkrow("A", "t1", 70.0, 12.0, 90.0),
        mkrow("B", "t2", 90.0, 3.0, 7.0),
        mkrow("B", "t2", 5.0, 12.0, 95.0),
    ]
    labels = np.array([1, 0, 2, 0, 2])

    seqs = ordered_sequences(rows_state, labels)
    assert set(seqs) == {("A", "t1"), ("B", "t2")}
    assert [t for t, _, _ in seqs[("A", "t1")]] == [10.0, 40.0, 70.0], \
        "sequences must sort by start time, not input order"
    assert [lab for _, lab, _ in seqs[("A", "t1")]] == [0, 1, 2]
    print("  PASS  per-track sequences are ordered by time, not input order")

    counts = fit_transition_counts(seqs, k=3)
    assert counts.shape == (3, 3)
    assert counts[0, 1] > counts[0, 0] and counts[1, 2] > counts[1, 0]
    print("  PASS  transition counts follow the planted sequence")

    settled, active, rep = split_settled_active(rows_state, labels, k=3)
    assert settled == [2] and sorted(active) == [0, 1], (settled, active)
    assert rep["dominant_gap_s"] > rep["next_largest_gap_s"]
    assert rep["gap_dominance_ratio"] > 5
    print(f"  PASS  the split finds the dominant gap "
          f"({rep['dominant_gap_s']}s vs {rep['next_largest_gap_s']}s), "
          f"not the small one")

    # From active: t1 goes 0->1 (active) then 1->2 (settled); t2's 2->0 starts
    # settled and is not counted. So one of two active exits returns to
    # settled, and Laplace smoothing gives exactly 0.5.
    p = fit_return_rate(seqs, set(settled), set(active))
    assert abs(p - 0.5) < 1e-9, p
    print("  PASS  return rate counts only active exits, Laplace-smoothed")

    idx_s = [i for i, l in enumerate(labels) if l in settled]
    idx_a = [i for i, l in enumerate(labels) if l in active]
    ws = hold_window(_holds(rows_state, idx_s))
    wa = hold_window(_holds(rows_state, idx_a))
    assert ws[0] > wa[1], ("settled must out-dwell active", ws, wa)
    print(f"  PASS  settled dwell {tuple(round(x,1) for x in ws)} strictly "
          f"exceeds active {tuple(round(x,1) for x in wa)}")

    corpus = dict(return_rate_active_to_settled=0.7,
                  dwell_window_s_by_group=dict(settled=[30.0, 120.0],
                                               active=[5.0, 40.0]))
    M = np.asarray(corpus_overlay_trans(corpus))
    assert np.allclose(np.diag(M), 0.0), "diagonal must be zero"
    assert np.allclose(M.sum(axis=1), 1.0), "rows must sum to 1"
    i_solo = BED_STATES.index("drone_solo")
    for b in ("bed_duck", "bed_bursts"):
        assert abs(M[BED_STATES.index(b), i_solo] - 0.7) < 1e-9, \
            "the corpus return rate must appear verbatim"
    ratio = (M[i_solo, BED_STATES.index("bed_duck")]
             / M[i_solo, BED_STATES.index("bed_bursts")])
    assert abs(ratio - 0.6 / 0.4) < 1e-9, "the authored split must be preserved"
    print("  PASS  3x3 assembly honours the corpus return rate and keeps the "
          "authored duck:bursts split")

    # classify_arc must be a pure function of the exported model.
    model = dict(mu=np.zeros(len(DESC_NAMES)), sd=np.ones(len(DESC_NAMES)),
                 cluster_dims=np.asarray(CLUSTER_DIMS),
                 centroids=np.eye(3, len(CLUSTER_DIMS)))
    v = np.zeros(len(DESC_NAMES))
    v[CLUSTER_DIMS[1]] = 5.0
    assert classify_arc(v, model) == 1
    print("  PASS  classify_arc projects into the corpus's own space")

    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser(
        description="Fit an empirical semi-Markov timing model on a corpus.")
    ap.add_argument("--cache", default=None,
                    help=f"descriptor cache (default {DEFAULT_CACHE})")
    ap.add_argument("--out", default=None,
                    help="output JSON; omit to fit and report without writing")
    ap.add_argument("--k", type=int, default=FORCED_K,
                    help=f"number of transition types (default {FORCED_K})")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    fit(args.cache or DEFAULT_CACHE, args.out, k=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
