"""Long-track ratings: do the clip-level transition effects survive continuous
listening?

Step 1 of the long-track half of Section 3.9. Everything else in the thesis
that touches transitions is CLIP-level evidence -- 13-26 s journeys rated in
isolation. This asks the ecological question instead: listeners sat through
multi-minute tracks, and these are the three pre-registered contrasts.

  H1  MANIPULATION CHECK. Does the up-trajectory read higher valence than the
      down-trajectory? If not, the mid-track valence move was not perceptible
      and nothing below means anything.
  H2  TRANSITION PRESENCE. Does an arc boundary differ from a plain crossfade?
  H3  DISSOCIATION. Section 3.5 found a preference/valence flip on the
      chord-end axis: pairwise preference goes to the minor ending, absolute
      valence to the major. Does major-ending read more pleasant here too?

Model. Stimuli are a random effect, not fixed levels (Clark 1973; Baayen,
Davidson & Bates 2008):

    rating ~ C(condition) * probe_index + (1|rater) + (1|chain)

The rater intercept absorbs criterion shift; the chain intercept absorbs
pad-sequence idiosyncrasy, the three independent OU chains per trajectory
being the instantiation replication. `probe_index` is linear over the three
timed probes, which makes its slope the cumulative-effect test.

**Read the n before the p.** This is a pilot: five usable raters. The paired
per-rater contrasts are the primary read, because each rater contributes one
paired difference and is therefore weighted equally -- the correct
within-subject reduction. The crossed-random-effects mixed model is secondary
at this n and is reported with its convergence flag. Ternary {-1, 0, +1} SAM
ratings are treated as interval, which is standard for SAM and is flagged in
the output JSON.

Two session types are pooled in the shards and MUST be split rather than
merged. Timed-probe sessions (`probe1`/`probe2`/`probe3` + `overall`) feed the
mixed model; the sparse branch's free-timed `event` rows feed a separate
event-locked analysis around the arc transition.

Data only: writes a JSON. The figures are Section 3.9's `build_results.py` and
`analyse_recency.py`.

Run:
  python analyse_longtrack.py                      # the shipped ratings
  python analyse_longtrack.py --ratings DIR [DIR ...]
  python analyse_longtrack.py --drop-raters R12,R15
  python analyse_longtrack.py --use-meta-transitions
  python analyse_longtrack.py --selftest
"""

import argparse
import csv
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
RATINGS_DIRS = [SECTION / "human_ratings"]
META = SECTION / "human_ratings" / "longtrack_meta.json"
OUT = SECTION / "data" / "results" / "longtrack_analysis.json"

# Sessions that rated nothing: one row each, both from the deploy window.
# Always dropped, before --drop-raters is even consulted.
TEST_IDS = {"R18", "R10"}
TIMED_PROBES = ["probe1", "probe2", "probe3"]
PROBE_IDX = {"probe1": 1, "probe2": 2, "probe3": 3}
ARC_CONDS = ["arc_minend", "arc_majend"]


# --------------------------------------------------------------- load / clean

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_rows(dirs=RATINGS_DIRS, drop_raters=frozenset()):
    """Pool the boot-sharded CSVs, dedup exact re-uploads, drop test IDs, coerce
    numerics. Keeps a `_practice` flag rather than dropping practice rows (they
    are a scale-use check). Returns cleaned dict rows."""
    seen, raw = set(), []
    for d in dirs:
        # only long-track rating files -- logs/ also holds unrelated CSVs
        # (texture_ratings, conductor_log, ...) that share a `rating` column
        for f in sorted(Path(d).glob("longtrack_ratings*.csv")):
            try:
                rdr = csv.DictReader(open(f))
            except OSError:
                continue
            if not rdr.fieldnames or "condition" not in rdr.fieldnames:
                continue
            for r in rdr:
                if not any((v or "").strip() for v in r.values()):
                    continue
                key = (r.get("rater"), r.get("session_id"), r.get("track_id"),
                       r.get("stage"), r.get("timestamp"))
                if key in seen:
                    continue
                seen.add(key)
                raw.append(r)
    clean = []
    for r in raw:
        rt = r.get("rater")
        if rt in TEST_IDS or rt in drop_raters:
            continue
        rating = _f(r.get("rating"))
        if rating is None:
            continue
        r = dict(r)
        r["_rating"] = rating
        r["_elapsed"] = _f(r.get("elapsed_s"))
        r["_practice"] = (r.get("condition") == "practice"
                          or r.get("trajectory") == "practice")
        clean.append(r)
    return clean


# ------------------------------------------------------ per-rater contrasts

def _by_rater_cond_mean(rows, stages=None):
    """mean rating per (rater, condition), optionally restricted to `stages`."""
    acc = defaultdict(list)
    for r in rows:
        if r["_practice"]:
            continue
        if stages and r.get("stage") not in stages:
            continue
        acc[(r["rater"], r["condition"])].append(r["_rating"])
    return {k: float(np.mean(v)) for k, v in acc.items() if v}, \
           {k: len(v) for k, v in acc.items()}


def _by_rater_traj_mean(rows, stages=None):
    acc = defaultdict(list)
    for r in rows:
        if r["_practice"]:
            continue
        if stages and r.get("stage") not in stages:
            continue
        acc[(r["rater"], r["trajectory"])].append(r["_rating"])
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def paired_contrast(a_means, b_means, label_a, label_b):
    """Per-rater paired difference mean(a)-mean(b) over raters who have BOTH.
    Each rater contributes one paired diff -> equal weighting, the correct
    within-subject reduction. Returns mean diff, n, and a paired t + Wilcoxon."""
    from scipy import stats
    # a_means / b_means are {rater: mean rating}; pair over raters with BOTH
    common = sorted(set(a_means) & set(b_means))
    diffs = np.array([a_means[r] - b_means[r] for r in common])
    out = dict(contrast=f"{label_a} - {label_b}", n_raters=len(common),
               raters=common,
               mean_of_paired_diffs=round(float(np.mean(diffs)), 4) if len(diffs) else None,
               sd_of_paired_diffs=round(float(np.std(diffs, ddof=1)), 4) if len(diffs) > 1 else None,
               a_mean=round(float(np.mean([a_means[r] for r in common])), 4) if common else None,
               b_mean=round(float(np.mean([b_means[r] for r in common])), 4) if common else None)
    if len(diffs) >= 2 and np.std(diffs) > 0:
        t, p = stats.ttest_rel([a_means[r] for r in common],
                               [b_means[r] for r in common])
        out["paired_t"] = round(float(t), 3)
        out["paired_t_p"] = round(float(p), 4)
        try:
            w, pw = stats.wilcoxon(diffs)
            out["wilcoxon_p"] = round(float(pw), 4)
        except ValueError:
            out["wilcoxon_p"] = None
    else:
        out["paired_t"] = out["paired_t_p"] = out["wilcoxon_p"] = None
    return out


def _cond_means_by_rater(rows, cond, stages=None):
    """{rater: mean rating} for one condition."""
    acc = defaultdict(list)
    for r in rows:
        if r["_practice"] or r["condition"] != cond:
            continue
        if stages and r.get("stage") not in stages:
            continue
        acc[r["rater"]].append(r["_rating"])
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def _traj_means_by_rater(rows, traj, stages=None):
    acc = defaultdict(list)
    for r in rows:
        if r["_practice"] or r["trajectory"] != traj:
            continue
        if stages and r.get("stage") not in stages:
            continue
        acc[r["rater"]].append(r["_rating"])
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


# ------------------------------------------------------------- mixed model

def mixed_model(rows):
    """rating ~ C(condition)*probe_index + (1|rater) + (1|chain) on the timed
    probes. Crossed REs via statsmodels: groups=rater, chain as a variance
    component. Returns fixed effects + RE variances, or a reason it couldn't
    fit. Secondary evidence at pilot n -- convergence flagged."""
    timed = [r for r in rows if not r["_practice"]
             and r.get("stage") in TIMED_PROBES
             and r.get("chain") not in (None, "")]
    if len({r["rater"] for r in timed}) < 3 or len(timed) < 30:
        return dict(fitted=False,
                    reason=f"insufficient timed data (n={len(timed)}, "
                           f"raters={len({r['rater'] for r in timed})})")
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        df = pd.DataFrame(dict(
            rating=[r["_rating"] for r in timed],
            condition=[r["condition"] for r in timed],
            probe_index=[PROBE_IDX[r["stage"]] for r in timed],
            rater=[r["rater"] for r in timed],
            chain=[str(r["chain"]) for r in timed]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm(
                "rating ~ C(condition, Treatment('noarc')) * probe_index",
                df, groups=df["rater"],
                vc_formula={"chain": "0 + C(chain)"})
            res = md.fit(reml=False, method="lbfgs")
        fe = {name: dict(coef=round(float(res.params[name]), 4),
                         se=round(float(res.bse[name]), 4),
                         p=round(float(res.pvalues[name]), 4))
              for name in res.params.index if name != "Group Var"}
        return dict(fitted=True, converged=bool(res.converged),
                    n_obs=len(timed),
                    n_raters=int(df["rater"].nunique()),
                    formula="rating ~ C(condition)*probe_index + (1|rater) + (1|chain)",
                    fixed_effects=fe,
                    re_var=dict(rater_group=round(float(res.cov_re.iloc[0, 0]), 4)
                                if res.cov_re.size else None,
                                residual=round(float(res.scale), 4)),
                    note=("secondary at pilot n; lead with the paired contrasts. "
                          "condition coefs are vs noarc; probe_index linear = "
                          "cumulative-effect slope."))
    except Exception as e:   # noqa: BLE001 -- report, don't crash the analysis
        return dict(fitted=False, reason=f"{type(e).__name__}: {e}")


# ------------------------------------------------------------- peri-event

def _transition_times(meta_path=META):
    """Per-(trajectory, chain, condition) arc-transition time from the render
    metadata. Returns {} when it is unavailable, and `peri_event` then falls
    back to splitting at the median event time.

    The published run took that fallback, and not because the file was
    missing: it searched for `transition_s` / `arc_boundary_s` / `boundary_s`,
    and the renderer writes **`transition_start_s`**. So no key ever matched,
    the meta was silently treated as absent, and the reported peri-event split
    is the approximate one. The key list below is corrected, which is why
    using it is opt-in -- see `--use-meta-transitions`.
    """
    if not meta_path.exists():
        return {}
    try:
        meta = json.load(open(meta_path))
    except (OSError, ValueError):
        return {}
    tt = {}
    if isinstance(meta, dict):
        tracks = meta.get("tracks", [])
    elif isinstance(meta, list):
        tracks = meta
    else:
        tracks = []
    for t in (tracks.values() if isinstance(tracks, dict) else tracks):
        if not isinstance(t, dict):
            continue
        key = (t.get("trajectory"), str(t.get("chain")), t.get("condition"))
        tr = (t.get("transition_start_s") or t.get("transition_s")
              or t.get("arc_boundary_s") or t.get("boundary_s"))
        if tr is not None:
            tt[key] = float(tr)
    return tt


def peri_event(rows, use_meta=False):
    """Event-locked read of the sparse-mode rows: mean rating BEFORE vs AFTER
    the arc transition, per condition.

    `use_meta=False` reproduces the published numbers by splitting at the
    median event time. `use_meta=True` uses the real per-track transition
    times, which is the analysis that was intended -- see `_transition_times`
    for why the published run did not get them. Either way `approx_split`
    records which one ran."""
    ev = [r for r in rows if not r["_practice"] and r.get("stage") == "event"
          and r["_elapsed"] is not None]
    if not ev:
        return dict(available=False, reason="no sparse event rows")
    tt = _transition_times() if use_meta else {}
    approx = not tt
    med = float(np.median([r["_elapsed"] for r in ev])) if approx else None
    pre, post = defaultdict(list), defaultdict(list)
    for r in ev:
        tr = tt.get((r.get("trajectory"), str(r.get("chain")), r.get("condition")),
                    med)
        (pre if r["_elapsed"] < tr else post)[r["condition"]].append(r["_rating"])
    conds = sorted(set(pre) | set(post))
    return dict(available=True, approx_split=approx, n_events=len(ev),
                split_note=("per-track transition times from "
                            "longtrack_meta.json" if not approx
                            else f"median-elapsed split ~{med:.1f}s (approx; "
                                 "this is what the report's numbers used)"),
                by_condition={c: dict(
                    pre_mean=round(float(np.mean(pre[c])), 4) if pre[c] else None,
                    post_mean=round(float(np.mean(post[c])), 4) if post[c] else None,
                    n_pre=len(pre[c]), n_post=len(post[c]),
                    shift=round(float(np.mean(post[c]) - np.mean(pre[c])), 4)
                    if pre[c] and post[c] else None) for c in conds})


# ------------------------------------------------------------------ analyze

def analyze(rows, use_meta=False):
    non_practice = [r for r in rows if not r["_practice"]]
    raters = sorted({r["rater"] for r in non_practice})
    cell = Counter((r["rater"], r["condition"]) for r in non_practice)
    usable = sorted({rt for rt in raters
                     if len({c for (r2, c) in cell if r2 == rt
                             and c in (["noarc"] + ARC_CONDS)}) >= 2})

    # H1 manipulation check: up vs down trajectory (overall + all timed probes)
    up = _traj_means_by_rater(non_practice, "up")
    down = _traj_means_by_rater(non_practice, "down")
    h1 = paired_contrast(up, down, "trajectory_up", "trajectory_down")

    # H2 transition presence: arc (pooled minend+majend) vs noarc
    arc = defaultdict(list)
    for r in non_practice:
        if r["condition"] in ARC_CONDS:
            arc[r["rater"]].append(r["_rating"])
    arc_m = {k: float(np.mean(v)) for k, v in arc.items() if v}
    noarc_m = _cond_means_by_rater(non_practice, "noarc")
    h2 = paired_contrast(arc_m, noarc_m, "arc_any", "noarc")

    # H3 dissociation: majend vs minend
    maj = _cond_means_by_rater(non_practice, "arc_majend")
    minend = _cond_means_by_rater(non_practice, "arc_minend")
    h3 = paired_contrast(maj, minend, "arc_majend", "arc_minend")

    # cell means for the plot / sanity
    cmeans, cn = _by_rater_cond_mean(non_practice)
    cond_grand = {}
    for cond in ["noarc"] + ARC_CONDS:
        vals = [m for (rt, c), m in cmeans.items() if c == cond]
        cond_grand[cond] = round(float(np.mean(vals)), 4) if vals else None

    # scale-use check on practice
    prac = [r["_rating"] for r in rows if r["_practice"]]
    scale_use = dict(n=len(prac),
                     dist=dict(Counter(round(x) for x in prac)) if prac else {})

    return dict(
        meta=dict(
            n_rows=len(rows), n_experimental=len(non_practice),
            n_raters=len(raters), raters=raters,
            n_usable_raters=len(usable), usable_raters=usable,
            rating_scale="ternary {-1,0,+1}, treated as interval (SAM)",
            cell_counts={f"{rt}|{c}": n for (rt, c), n in sorted(cell.items())}),
        cond_grand_means=cond_grand,
        H1_manipulation_up_vs_down=h1,
        H2_transition_presence_arc_vs_noarc=h2,
        H3_dissociation_majend_vs_minend=h3,
        mixed_model=mixed_model(rows),
        peri_event_sparse=peri_event(rows, use_meta=use_meta),
        practice_scale_use=scale_use,
        caveats=[
            "pilot n (~5 usable raters): paired contrasts are the robust primary "
            "read; the mixed model is secondary and convergence-flagged.",
            "ternary ratings treated as interval; an ordinal mixed model is the "
            "confirmatory upgrade if more raters land.",
            "every rater who supplied ratings is kept unless named in "
            "--drop-raters; R18 and R10 are always dropped, each having "
            "supplied a single row.",
            "H3 tests whether s13's CLIP-level preference/valence dissociation "
            "reappears in continuous listening -- direction is the finding, not p."])


def _print(res):
    m = res["meta"]
    print(f"\nlong-track analysis: {m['n_experimental']} experimental ratings, "
          f"{m['n_raters']} raters ({m['n_usable_raters']} usable, >=2 conditions)")
    print(f"  condition grand means: {res['cond_grand_means']}")
    for key, tag in [("H1_manipulation_up_vs_down", "H1 up-vs-down"),
                     ("H2_transition_presence_arc_vs_noarc", "H2 arc-vs-noarc"),
                     ("H3_dissociation_majend_vs_minend", "H3 majend-vs-minend")]:
        c = res[key]
        print(f"  {tag:20} {c['contrast']:26} diff={c['mean_of_paired_diffs']} "
              f"(n={c['n_raters']}, t={c['paired_t']}, p={c['paired_t_p']}, "
              f"wilcoxon_p={c['wilcoxon_p']})")
    mm = res["mixed_model"]
    if mm.get("fitted"):
        print(f"  mixed model: converged={mm['converged']}, n={mm['n_obs']}; "
              "condition fixed effects (vs noarc):")
        for name, v in mm["fixed_effects"].items():
            if "condition" in name:
                print(f"      {name}: b={v['coef']} se={v['se']} p={v['p']}")
    else:
        print(f"  mixed model: NOT fitted ({mm['reason']})")
    pe = res["peri_event_sparse"]
    if pe.get("available"):
        print(f"  peri-event ({pe['split_note']}): "
              + "; ".join(f"{c} shift={d['shift']}"
                          for c, d in pe["by_condition"].items()))


# ----------------------------------------------------------------- testing

def _selftest():
    # synthetic: 4 raters, all 3 conditions, 3 probes, 2 trajectories, 3 chains,
    # with a planted majend>minend gap and up>down gap
    rng = np.random.default_rng(0)
    rows = []
    base = dict(noarc=0.0, arc_minend=-0.3, arc_majend=0.4)
    for rt in ["r1", "r2", "r3", "r4"]:
        bias = rng.normal(0, 0.05)
        for traj, tj in [("up", 0.3), ("down", -0.3)]:
            for chain in ["0", "1", "2"]:
                for cond in ["noarc", "arc_minend", "arc_majend"]:
                    for st in TIMED_PROBES + ["overall"]:
                        val = base[cond] + tj + bias + rng.normal(0, 0.05)
                        val = max(-1.0, min(1.0, round(val)))
                        rows.append(dict(rater=rt, session_id="s", track_id="t",
                                         stage=st, timestamp=f"{rt}{traj}{chain}{cond}{st}",
                                         condition=cond, trajectory=traj, chain=chain,
                                         rating=str(float(val)), elapsed_s="60",
                                         _rating=float(val), _elapsed=60.0,
                                         _practice=False))
    # a couple of sparse events + practice
    for st, el in [("event", "40"), ("event", "80")]:
        rows.append(dict(rater="r1", session_id="s", track_id="t", stage="event",
                         timestamp=f"ev{el}", condition="arc_majend", trajectory="up",
                         chain="0", rating="1.0", elapsed_s=el,
                         _rating=1.0, _elapsed=float(el), _practice=False))
    rows.append(dict(rater="r1", stage="probe1", condition="practice",
                     trajectory="practice", timestamp="prac", rating="0.0",
                     _rating=0.0, _elapsed=None, _practice=True))

    res = analyze(rows)
    h3 = res["H3_dissociation_majend_vs_minend"]
    assert h3["mean_of_paired_diffs"] > 0.3, ("planted majend>minend gap", h3)
    assert h3["n_raters"] == 4
    h1 = res["H1_manipulation_up_vs_down"]
    assert h1["mean_of_paired_diffs"] > 0.3, ("planted up>down gap", h1)
    assert res["meta"]["n_usable_raters"] == 4
    mm = res["mixed_model"]
    assert mm["fitted"], mm
    # majend coefficient (vs noarc) must be positive and minend negative
    fe = mm["fixed_effects"]
    majk = [k for k in fe if "arc_majend" in k and "probe" not in k][0]
    mink = [k for k in fe if "arc_minend" in k and "probe" not in k][0]
    assert fe[majk]["coef"] > 0 and fe[mink]["coef"] < 0, (fe[majk], fe[mink])
    pe = res["peri_event_sparse"]
    assert pe["available"] and pe["n_events"] == 2
    # load_rows dedup + test-ID drop
    print("selftest OK: paired contrasts recover planted majend>minend + "
          "up>down gaps, usable-rater detection, crossed-RE mixed model fits "
          "with correct condition-effect signs, peri-event split, scale-use")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ratings", nargs="+", type=Path, default=RATINGS_DIRS,
                    help="directories of longtrack_ratings_*.csv shards; "
                         "several may be given and are pooled, with exact "
                         "re-uploads deduplicated")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--drop-raters", default="",
                    help="comma-separated rater IDs to exclude")
    ap.add_argument("--use-meta-transitions", action="store_true",
                    help="split the peri-event analysis at each track's real "
                         "transition time instead of the median event time. "
                         "The report's numbers used the median; see "
                         "_transition_times for why.")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    drop = frozenset(x.strip() for x in args.drop_raters.split(",") if x.strip())
    rows = load_rows(dirs=args.ratings, drop_raters=drop)
    if not rows:
        raise SystemExit("no longtrack_ratings_*.csv under "
                         + ", ".join(str(d) for d in args.ratings))
    res = analyze(rows, use_meta=args.use_meta_transitions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    _print(res)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
