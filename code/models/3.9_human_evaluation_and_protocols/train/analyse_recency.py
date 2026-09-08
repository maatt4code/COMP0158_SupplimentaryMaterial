"""Does the final retrospective rating just follow the last thing heard?

Step 2 of the long-track half of Section 3.9, and the headline result of it.

Every long track carries three in-track probes ("rate it NOW") at roughly 40 s,
70 s and 115 s, then one final `overall` rating for the whole track. That makes
the recency question askable directly, on the same person and the same track:

    does `overall` agree with probe3 more than with probe1 or probe2?

If it tracks probe3, listeners are reporting the end rather than the whole.
Because the comparison is within-subject AND within-track, no amount of
between-rater disagreement can manufacture it -- which is what makes this the
one result in Section 3.9 that does not need hedging on sample size.

**This finding is why the thesis stopped trying to model valence from
movement.** A retrospective rating of a multi-minute generative track measures
its last few seconds, so it cannot serve as a target for the trajectory.

What it is NOT. Correlating each rating against the rendered valence
trajectory over sliding windows gives nothing at any window length from 4 s to
the whole track. So the claim is not "listeners track the last N seconds of
signal"; it is "the retrospective summary is the last judgement made". That is
a narrower claim and it is the one the probe design was built to test.

The sparse branch is excluded. Its sessions were free-timed and emit `event`
rows instead of scheduled probes, so they have no probe1/probe2/probe3 to
correlate; including them would only pad the `overall` count. Exclusion is by
DESIGN, not by filename: any session that emitted an `event` row is sparse.

Run:
  python analyse_recency.py                        # the shipped ratings
  python analyse_recency.py --ratings DIR [DIR ...]
  python analyse_recency.py --out ../data/results
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy.stats import spearmanr, wilcoxon                        # noqa: E402

HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
RATINGS_DIRS = [SECTION / "human_ratings"]
OUT = SECTION / "data" / "results"

PROBES = ["probe1", "probe2", "probe3"]
# The three probes' scheduled times, for axis labels only; the analysis never
# uses them, and the renderer's own per-track times are in longtrack_meta.json.
PROBE_LABELS = ["probe 1\n~40 s", "probe 2\n~70 s", "probe 3\n~112 s"]
GREY, MIDGREY, BLUE = "#cbd5e0", "#a0aec0", "#2b6cb0"


def load(dirs):
    """Pool the shards, drop exact duplicate rows, and keep the timed-probe
    sessions only. Returns one row per (rater, session, track) with the probe
    and overall ratings as columns."""
    frames = []
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / "longtrack_ratings_*.csv"))):
            try:
                df = pd.read_csv(f)
            except (OSError, ValueError, pd.errors.ParserError):
                continue
            if len(df) and "stage" in df.columns:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    lt = pd.concat(frames, ignore_index=True).drop_duplicates()

    # Drop the free-timed sparse sessions: they carry `event` rows rather than
    # scheduled probes, so they cannot answer this question.
    sparse = set(lt.loc[lt.stage == "event", "session_id"])
    lt = lt[~lt.session_id.isin(sparse)]

    lt = lt[lt.condition != "practice"].copy()
    lt["rating"] = pd.to_numeric(lt["rating"], errors="coerce")
    wide = lt.pivot_table(index=["rater", "session_id", "track_id", "condition"],
                          columns="stage", values="rating",
                          aggfunc="last").reset_index()
    return wide.dropna(subset=["overall"])


def probe_correlations(wide):
    """Per probe: Spearman with the overall rating, sign agreement, and mean
    absolute gap. Rising rho across the three probes is the result."""
    rows = []
    for p in PROBES:
        d = wide.dropna(subset=[p])
        rho, pv = spearmanr(d[p], d["overall"])
        rows.append(dict(probe=p, n=len(d), rho=rho, p=pv,
                         sign_agree=(np.sign(d[p]) == np.sign(d["overall"])).mean(),
                         mean_abs_gap=(d["overall"] - d[p]).abs().mean()))
    return pd.DataFrame(rows)


def within_track_gaps(wide):
    """The paired test. On tracks carrying both probe1 and probe3, is `overall`
    closer to probe3? Paired within the track, so track difficulty and rater
    criterion both cancel."""
    d = wide.dropna(subset=["probe1", "probe3"])
    g1 = (d["overall"] - d["probe1"]).abs()
    g3 = (d["overall"] - d["probe3"]).abs()
    try:
        _, pw = wilcoxon(g1, g3)
    except ValueError:                  # all differences zero
        pw = np.nan
    return d, g1, g3, pw


def _rho(x, y):
    """Spearman, returning NaN for a constant input rather than warning.

    A rater who gave the same value at every probe1 has no variance there, so
    the correlation is undefined -- not zero. Reporting it as NaN keeps that
    rater visible in the per-listener table, which matters: dropping them
    would quietly shrink the panel that the reader is using to check that the
    effect is not one person."""
    if len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return spearmanr(x, y).correlation


def per_rater(wide, min_tracks=4):
    """Per-listener rho against probe1 and probe3. Raters with fewer than
    `min_tracks` usable tracks are omitted rather than reported at n=1."""
    out = []
    for r, g in wide.groupby("rater"):
        g1, g3 = g.dropna(subset=["probe1"]), g.dropna(subset=["probe3"])
        if len(g1) >= min_tracks and len(g3) >= min_tracks:
            out.append(dict(rater=r, n=len(g),
                            rho1=_rho(g1["probe1"], g1["overall"]),
                            rho3=_rho(g3["probe3"], g3["overall"])))
    return pd.DataFrame(out)


def figure(res, g1, g3, n_paired, pw, per, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)

    ax = axes[0]
    ax.bar(range(3), res.rho, color=[GREY, MIDGREY, BLUE])
    for i, (rho, pv, n) in enumerate(zip(res.rho, res.p, res.n)):
        ax.text(i, rho + 0.02, f"{rho:+.2f}\np={pv:.3f}\nn={n}", ha="center",
                va="bottom", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(PROBE_LABELS)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel(r"Spearman $\rho$ with the overall rating")
    ax.set_ylim(-0.1, max(res.rho) * 1.45)
    ax.set_title("(a) the overall rating follows the last probe")

    ax = axes[1]
    ax.bar([0, 1], [g1.mean(), g3.mean()], color=[GREY, BLUE], width=0.6)
    ax.errorbar([0, 1], [g1.mean(), g3.mean()], yerr=[g1.sem(), g3.sem()],
                fmt="none", ecolor="k", capsize=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["probe 1", "probe 3"])
    ax.set_ylabel("mean |overall - probe|")
    ax.set_title(f"(b) distance to the overall rating\n"
                 f"(n={n_paired}, Wilcoxon p={pw:.3f})", fontsize=10)

    ax = axes[2]
    if len(per):
        x, w = np.arange(len(per)), 0.38
        ax.bar(x - w / 2, per.rho1, w, label="probe 1", color=GREY)
        ax.bar(x + w / 2, per.rho3, w, label="probe 3", color=BLUE)
        ax.set_xticks(x)
        ax.set_xticklabels(per.rater, fontsize=8)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_ylabel(r"$\rho$ with overall")
        ax.legend(fontsize=8)
    ax.set_title("(c) per listener")

    fig.suptitle("Recency in retrospective ratings of multi-minute tracks",
                 fontsize=13)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ratings", nargs="+", type=Path, default=RATINGS_DIRS,
                    help="directories of longtrack_ratings_*.csv shards")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="output directory; data/ and figs/ are created in it")
    args = ap.parse_args()

    wide = load(args.ratings)
    if not len(wide):
        raise SystemExit("no longtrack_ratings_*.csv under "
                         + ", ".join(str(d) for d in args.ratings))
    print(f"{len(wide)} completed tracks with an overall rating, "
          f"{wide.rater.nunique()} raters, {wide.track_id.nunique()} tracks\n")

    res = probe_correlations(wide)
    print(res.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    d, g1, g3, pw = within_track_gaps(wide)
    print(f"\nwithin-track gap to overall:  probe1 {g1.mean():.3f}   "
          f"probe3 {g3.mean():.3f}   (n={len(d)}, Wilcoxon p={pw:.4f})")

    per = per_rater(wide)
    if len(per):
        print("\nper rater:")
        print(per.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    data, figs = args.out / "data", args.out / "figs"
    data.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    figure(res, g1, g3, len(d), pw, per, figs / "fig_recency.png")
    res.to_csv(data / "recency_probe_vs_overall.csv", index=False)
    print(f"\nwrote {figs / 'fig_recency.png'}")
    print(f"wrote {data / 'recency_probe_vs_overall.csv'}")


if __name__ == "__main__":
    main()
