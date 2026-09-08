"""The cleaning pass over the component-preference ratings, plus its figures.

Step 2 of Section 3.9. `analyse_space_ratings.load()` is the merge; this is the
hygiene and reporting layer on top of it, and between them they produce the
dataset the report analysed. Neither is rewritten here -- see that module's
note on why re-deriving the merge would silently change the data.

Two hygiene rules, both dated and both consequential:

  * A per-tab audio CUT-OFF. Stimulus filenames are content-hashed, so a
    changed file set means a re-render; ratings made before that point describe
    audio that no longer exists and are dropped.
  * Deployment smoke sessions are excluded. They are short sessions from the
    deploy window, not listening data.

Re-runnable: it wipes and rewrites its own output directory and never touches
the shipped ratings.

Run:
  python build_results.py --out-dir ../data/results
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
RATINGS = SECTION / "human_ratings"
OUT = SECTION / "data" / "results"
DATA, FIGS = OUT / "data", OUT / "figs"
sys.path.insert(0, str(HERE))
from analyse_space_ratings import load  # noqa: E402

# ---- hygiene rules (derived 2026-08-19; see README for the derivation) -----
# Per tab, the last time its AUDIO changed, in UTC. Filenames are content
# hashed, so a changed file set = a re-render; ratings before that point were
# made on audio that no longer exists.
CUT = {"melody": "2026-08-01 00:01:16", "stereo": "2026-08-01 01:57:17"}
CUT_DEFAULT = "2026-07-26 16:55:07"
# Short sessions from the deploy window: smoke tests, not listening data.
TESTERS = {"rater_ef9f", "rater_57d5", "rater_9f44",
           "rater_65f1", "rater_4b0a", "rater_4a9b"}
# The three raters who completed all eight tabs, 156 rows each. Two carry
# the anonymous id the app minted; the third is pseudonymised like every
# other identified rater.
COMPLETERS = ["rater_22cd", "R21", "rater_7895"]

# ---- palette (dataviz skill reference instance; validated all-pairs light) --
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED = "#e34948"                      # diverging partner for BLUE
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
GRID, AXIS, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def bare(ax, xgrid=True):
    ax.grid(axis="x" if xgrid else "y", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def main(out_dir=OUT, ratings=RATINGS):
    # DATA and FIGS are module-level because the figure builders below all
    # write through them; rebinding here is what makes --out-dir work without
    # threading a path through fifteen call sites.
    global DATA, FIGS
    DATA, FIGS = out_dir / "data", out_dir / "figs"
    for d in (DATA, FIGS):
        d.mkdir(parents=True, exist_ok=True)

    df = load(ratings)
    df["ts"] = pd.to_datetime(df.timestamp, errors="coerce")
    df["cut"] = pd.to_datetime(df.test.map(CUT).fillna(CUT_DEFAULT))
    df["superseded_audio"] = df.ts < df.cut
    df["is_tester"] = df.rater.isin(TESTERS)
    df["clean"] = ~df.superseded_audio & ~df.is_tester

    keep = ["timestamp", "ts", "rater", "test", "item_id", "response",
            "response2", "elapsed_s", "note", "score", "use", "effect",
            "strength", "reverb", "treatment", "style", "level_db", "bed_type",
            "a_is_staged", "a_is_reverb", "pair_kind", "has_melody", "clip_id",
            "drone", "bed_id", "superseded_audio", "is_tester", "clean"]
    df[keep].to_csv(DATA / "ratings_all_flagged.csv", index=False)
    d = df[df.clean].copy()
    d[keep].to_csv(DATA / "ratings_clean.csv", index=False)

    # ---- summaries ---------------------------------------------------------
    rows = []
    for (test, cond) in (("modulation", "effect"), ("modulation", "strength"),
                         ("reverb_clips", "reverb"), ("stereo", "treatment"),
                         ("melody", "style"), ("melody", "level_db"),
                         ("pairing", "bed_type"), ("pairing", "level_db")):
        s = d[d.test == test]
        if s.empty:
            continue
        g = s.groupby(cond, dropna=False)
        for k, sub in g:
            rows.append(dict(test=test, factor=cond, level=k, n=len(sub),
                             mean_score=round(sub.score.mean(), 3),
                             sd_score=round(sub.score.std(), 3),
                             use_rate=round(sub.use.mean(), 3),
                             n_raters=sub.rater.nunique()))
    summary = pd.DataFrame(rows)
    summary.to_csv(DATA / "summary_by_condition.csv", index=False)

    # ---- paired contrasts vs a reference, within (rater, clip) -------------
    def paired(sub, cond, ref, unit):
        p = sub.pivot_table(index=unit, columns=cond, values="score",
                            aggfunc="mean")
        out = []
        if ref not in p:
            return out
        for c in p.columns:
            if c == ref:
                continue
            both = p[[c, ref]].dropna()
            if len(both) < 4:
                continue
            diff = both[c] - both[ref]
            pv = stats.wilcoxon(diff)[1] if diff.abs().sum() > 0 else 1.0
            out.append(dict(condition=c, reference=ref, n_pairs=len(both),
                            mean_advantage=round(diff.mean(), 3),
                            sd=round(diff.std(), 3), p_wilcoxon=round(pv, 4)))
        return out

    pc = []
    for test, cond, ref, unit in (
            ("modulation", "effect", "none", ["rater", "clip_id"]),
            ("reverb_clips", "reverb", "dry", ["rater", "clip_id"])):
        for r in paired(d[d.test == test], cond, ref, unit):
            pc.append(dict(test=test, **r))
    paired_df = pd.DataFrame(pc).sort_values(["test", "mean_advantage"],
                                             ascending=[True, False])
    paired_df.to_csv(DATA / "paired_contrasts.csv", index=False)

    # ---- forced-choice -----------------------------------------------------
    fc, fcr = [], []
    for test, flag, lt, lf in (("transitions", "a_is_reverb", "reverb", "dry"),
                               ("sensitivity", "a_is_staged", "staged",
                                "simultaneous")):
        s = d[d.test == test]
        r = s[s.response.isin(["A", "B"])]
        won = (((r.response == "A") & (r[flag] == True))      # noqa: E712
               | ((r.response == "B") & (r[flag] == False)))  # noqa: E712
        t, f = int(won.sum()), int(len(r) - won.sum())
        ties = int((s.response == "same").sum())
        pv = stats.binomtest(t, t + f, 0.5).pvalue if t + f else np.nan
        fc.append(dict(test=test, option_a=lt, option_b=lf, wins_a=t,
                       wins_b=f, ties=ties, p_binomial=round(pv, 5),
                       n_raters=r.rater.nunique()))
        for rater, gg in r.assign(w=won).groupby("rater"):
            fcr.append(dict(test=test, rater=rater, wins_a=int(gg.w.sum()),
                            n=len(gg), option_a=lt, option_b=lf))
    pd.DataFrame(fc).to_csv(DATA / "forced_choice.csv", index=False)
    fcr_df = pd.DataFrame(fcr)
    fcr_df.to_csv(DATA / "forced_choice_by_rater.csv", index=False)

    # ---- inter-rater agreement --------------------------------------------
    ag = []
    for test, cond in (("modulation", "effect"), ("reverb_clips", "reverb"),
                       ("melody", "style"), ("pairing", "bed_type")):
        piv = d[(d.test == test) & (d.rater.isin(COMPLETERS))].pivot_table(
            index=cond, columns="rater", values="score", aggfunc="mean")
        cols = list(piv.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                pair = piv[[a, b]].dropna()
                if len(pair) < 3 or pair[a].nunique() < 2 or pair[b].nunique() < 2:
                    continue
                ag.append(dict(test=test, rater_a=a, rater_b=b,
                               n_conditions=len(pair),
                               spearman=round(stats.spearmanr(pair[a],
                                                              pair[b])[0], 3)))
        piv.round(3).to_csv(DATA / f"rater_means_{test}.csv")
    agree = pd.DataFrame(ag)
    agree.to_csv(DATA / "rater_agreement.csv", index=False)

    # ================= FIGURES =============================================
    # Fig 1 -- forced choice. Job: polarity of a single headline each.
    fig, axes = plt.subplots(2, 1, figsize=(8, 3.4), constrained_layout=True)
    for ax, row in zip(axes, fc):
        tot = row["wins_a"] + row["wins_b"] + row["ties"]
        # Colour by OUTCOME, not by slot: the preferred option is always blue
        # and always on the left. Painting slot A blue in both panels would
        # make blue mean "winner" in one and "loser" in the other.
        pairs = [(row["wins_a"], row["option_a"]), (row["wins_b"], row["option_b"])]
        (wv, wl), (lv, ll) = sorted(pairs, key=lambda t: -t[0])
        left = 0
        for val, col, lab in ((wv, BLUE, wl),
                              (row["ties"], NEUTRAL, "no preference"),
                              (lv, RED, ll)):
            ax.barh(0, val, left=left, height=0.5, color=col, zorder=3)
            if val:
                ax.text(left + val / 2, 0, f"{lab}\n{val}", ha="center",
                        va="center", fontsize=9,
                        color=INK if col == NEUTRAL else "white",
                        fontweight="bold" if col != NEUTRAL else "normal")
            left += val
        ax.set_xlim(0, tot)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{row['test']}  —  {wl} preferred over {ll}"
                     f"   (p = {row['p_binomial']:.4f}, "
                     f"{row['n_raters']} listeners)", loc="left")
    fig.suptitle("Forced-choice tests: the only two questions that got a clear "
                 "answer", fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.savefig(FIGS / "fig1_forced_choice.png", dpi=160)
    plt.close(fig)

    # Fig 2 -- paired advantage. Job: polarity vs a zero reference.
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True,
                             sharex=True)
    for ax, (test, title) in zip(axes, (
            ("reverb_clips", "Reverb rooms vs no reverb\n(held drone)"),
            ("modulation", "Modulation effects vs dry"))):
        sub = paired_df[paired_df.test == test].sort_values("mean_advantage")
        y = np.arange(len(sub))
        # A zero advantage is neither better nor worse -- give it the neutral,
        # not the "worse" pole.
        cols = [NEUTRAL if abs(v) < 0.02 else (BLUE if v > 0 else RED)
                for v in sub.mean_advantage]
        ax.axvline(0, color=AXIS, lw=1.2, zorder=2)
        ax.hlines(y, 0, sub.mean_advantage, color=cols, lw=2, zorder=3)
        ax.scatter(sub.mean_advantage, y, color=cols, s=60, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{c}  (n={n})" for c, n in
                            zip(sub.condition, sub.n_pairs)])
        for i, (v, p) in enumerate(zip(sub.mean_advantage, sub.p_wilcoxon)):
            ax.text(v + (0.02 if v >= 0 else -0.02), i, f"p={p:.2f}",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=8, color=INK2)
        ax.set_title(title, loc="left")
        ax.set_xlabel("worse  ←   rating advantage   →  better")
        bare(ax)
        ax.set_xlim(-0.55, 0.45)
    fig.suptitle("Nothing beats the plain version once you compare like with "
                 "like", fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.savefig(FIGS / "fig2_paired_contrasts.png", dpi=160)
    plt.close(fig)

    # Fig 3 -- rater disagreement, small multiples. Job: identity per rater.
    tests = ["reverb_clips", "modulation", "melody", "pairing"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.1), constrained_layout=True,
                             sharey=True)
    hues = {r: c for r, c in zip(COMPLETERS, (BLUE, ORANGE, AQUA))}
    for ax, test in zip(axes, tests):
        piv = pd.read_csv(DATA / f"rater_means_{test}.csv", index_col=0)
        x = np.arange(len(piv))
        for j, r in enumerate([c for c in COMPLETERS if c in piv.columns]):
            ax.scatter(x + (j - 1) * 0.16, piv[r], s=55, color=hues[r],
                       label=r, zorder=4,
                       edgecolors=SURFACE, linewidths=1.2)
        ax.axhline(0, color=AXIS, lw=1.2, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(piv.index, rotation=40, ha="right", fontsize=8)
        agr = agree[agree.test == test].spearman
        lab = ", ".join(f"{v:+.2f}" for v in agr) if len(agr) else "n/a"
        ax.set_title(f"{test}\nagreement: {lab}", loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.9)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("mean rating  (−1 bad … +1 good)")
    # Legend goes OUTSIDE the panels: inside lower-left it landed on top of a
    # real data point in the reverb panel.
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, fontsize=9, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("The same three listeners, condition by condition — they agree "
                 "on reverb and contradict each other on melody and beds",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.savefig(FIGS / "fig3_rater_disagreement.png", dpi=160)
    plt.close(fig)

    # Fig 4 -- bed level. Job: magnitude of a count.
    b = d[d.test == "bed_level"]
    counts = b.response2.astype(str).value_counts()
    order = [x for x in ["-6", "-12", "-18", "-24", "-30"] if x in counts.index]
    fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    vals = [counts[o] for o in order]
    cols = [BLUE if v == max(vals) else "#9ec5f4" for v in vals]
    ax.bar(order, vals, color=cols, width=0.62, zorder=3)
    for o, v in zip(order, vals):
        ax.text(o, v + 0.6, str(v), ha="center", fontsize=9, color=INK)
    ax.set_xlabel("background bed level chosen (dB relative to the drone)")
    ax.set_ylabel("times chosen")
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_title("Listeners settle on a quiet bed, around −18 dB", loc="left")
    fig.savefig(FIGS / "fig4_bed_level.png", dpi=160)
    plt.close(fig)

    # Fig 5 -- data hygiene. Job: magnitude, kept vs discarded.
    per = df.groupby("test").agg(total=("clean", "size"),
                                 kept=("clean", "sum")).sort_values("total")
    per["dropped"] = per.total - per.kept
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    y = np.arange(len(per))
    ax.barh(y, per.kept, color=BLUE, height=0.6, zorder=3, label="usable")
    ax.barh(y, per.dropped, left=per.kept, color=NEUTRAL, height=0.6, zorder=3,
            label="discarded (re-rendered audio / smoke tests)")
    ax.set_yticks(y)
    ax.set_yticklabels(per.index)
    for i, (k, t) in enumerate(zip(per.kept, per.total)):
        ax.text(t + 2, i, f"{k}/{t}", va="center", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_xlabel("ratings")
    bare(ax)
    ax.set_title(f"What survived the hygiene filters "
                 f"({int(per.kept.sum())} of {int(per.total.sum())} rows)",
                 loc="left")
    fig.savefig(FIGS / "fig5_data_hygiene.png", dpi=160)
    plt.close(fig)

    meta = dict(
        built=date.today().isoformat(),
        hygiene_rules_derived="2026-08-19",
        # The upstream HuggingFace dataset the shards were pulled from. The
        # account handle is the author's, so only the shard set is recorded;
        # `human_ratings/` IS that snapshot.
        source_dataset="human_ratings/ (snapshot of the rating app's output)",
        # Section-relative: an absolute path here would leak the build
        # machine's layout into a committed artefact.
        snapshot_dir=_relative(ratings),
        rows_raw=int(len(df)), rows_clean=int(len(d)),
        raters_raw=int(df.rater.nunique()), raters_clean=int(d.rater.nunique()),
        date_range=[str(df.ts.min()), str(df.ts.max())],
        audio_cutovers_utc={**{k: v for k, v in CUT.items()},
                            "_all_other_tabs": CUT_DEFAULT},
        excluded_testers=sorted(TESTERS), completers=COMPLETERS)
    (DATA / "build_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"rows {len(df)} -> clean {len(d)}")
    print("data:", *[p.name for p in sorted(DATA.iterdir())], sep="\n  ")
    print("figs:", *[p.name for p in sorted(FIGS.iterdir())], sep="\n  ")


def _relative(p):
    """The ratings directory, relative to this section when it lives inside it.
    An absolute path here would bake the build machine's layout into a
    committed artefact."""
    p = Path(p).resolve()
    try:
        return p.relative_to(SECTION).as_posix()
    except ValueError:
        return p.name


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ratings", type=Path, default=RATINGS,
                    help="directory of space_ratings_*.csv shards")
    ap.add_argument("--out-dir", type=Path, default=OUT,
                    help="output directory; data/ and figs/ are created in it")
    args = ap.parse_args()
    main(out_dir=args.out_dir, ratings=args.ratings)
