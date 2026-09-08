# 3.9 Human Evaluation Methodology and Listening Protocols

Two listening studies and the analysis code for both. This is the only section
that measures rather than synthesises: nothing here runs at synthesis time and
no weight is fitted, so `inference/` and `weights/` ship a README each saying
why they are empty rather than being deleted. It exists so the report's
human-evaluation numbers can be recomputed from the ratings beside it.

## Datasets

No external dataset. Every input was collected for this project and ships
pseudonymised in [`human_ratings/`](human_ratings): 26 session shards holding
1,095 ratings, plus the two merged tables the analysis produces from them.
**Read [`human_ratings/README.md`](human_ratings/README.md) before touching
any of it.** Both rating apps wrote one CSV per boot, so sessions are split
across shards and naive concatenation double-counts.

Stimuli were rendered by this pipeline, so they are reproducible from
[`3.3`](../3.3_drone_synthesis_and_nsynth_prior) and the conductor rather than
distributed as audio. `human_ratings/longtrack_meta.json` records what each
long track was.

## The two studies

**Long-track recency**, 11–18 July 2026. Ten listeners rated whole
multi-minute tracks under three transition conditions, with three in-track
probes and one retrospective rating per track. Three pre-registered contrasts
plus a crossed-random-effects mixed model, and then the finding the design
made possible by accident.

**Component preference**, 26 July – 21 August 2026. Seventeen listeners
compared settings for one component at a time across eight tabs — melody,
reverb, background bed, modulation, stereo, transitions — which is where the
conductor's defaults come from.

## Run order

| # | Command | What it does |
|---|---|---|
| 1 | `python train/analyse_space_ratings.py` | the merge of record: 14 shards → 760 rated rows, and the per-tab winners |
| 2 | `python train/build_results.py` | hygiene flags, 11 tables and 5 figures into `data/results/` |
| 3 | `python train/analyse_longtrack.py` | the three long-track contrasts + mixed model → `data/results/longtrack_analysis.json` |
| 4 | `python train/analyse_recency.py` | the recency test and its figure |

All four default to the shipped ratings and need no arguments. Each takes a
`--ratings` directory (steps 3 and 4 take several and pool them) so the same
analysis can be re-run on a fresh collection.

`python smoke_test.py` runs all of it and asserts the numbers below.

## The recency result, and why it ended a line of work

Every long track carries three probes and then one `overall` rating, so the
question can be asked within-subject *and* within-track:

| probe | Spearman ρ with `overall` | p | sign agreement | mean gap |
|---|---|---|---|---|
| probe 1, ~40 s | +0.359 | 0.011 | 46% | 0.640 |
| probe 2, ~70 s | +0.457 | <0.001 | 60% | 0.520 |
| **probe 3, ~115 s** | **+0.744** | **<0.001** | **74%** | **0.280** |

Monotonic across the three, and the paired within-track test agrees: `overall`
sits 0.280 from the last probe and 0.640 from the first (n = 50 tracks, 7
raters, Wilcoxon p = 0.005). Per listener, the effect is clearest in the rater
who did the most tracks: +0.087 against the first probe, +0.870 against the
last.

Because the comparison is within-track and within-subject, between-rater
disagreement cannot have produced it. **It is the one result in this section
that does not need hedging on sample size**, and it is why the thesis stopped
trying to model valence from movement: a retrospective rating of a
multi-minute generative track measures its last few seconds, so it cannot
serve as a target for the whole trajectory.

What it is *not*: correlating ratings against the rendered valence trajectory
over sliding windows gives nothing, at every window length from 4 s to the
whole track. So the claim is not "listeners track the last N seconds of
signal". It is "the retrospective summary is the last judgement made" — a
narrower claim, and the one the probe design was built to test.

**On the old number.** Chapter 3 states ρ = +0.84 for recency. The
reproducible figure is **+0.744**; `train/analyse_recency.py` is what computes
it.

## The three long-track contrasts

| test | contrast | difference | paired t p | Wilcoxon p | raters |
|---|---|---|---|---|---|
| H1 manipulation check | up − down trajectory | −0.431 | 0.171 | 0.125 | 4 |
| H2 transition presence | arc − no arc | +0.171 | 0.517 | 0.625 | 4 |
| H3 dissociation | major − minor ending | +0.590 | 0.086 | 0.125 | 4 |

Five usable raters, so **read the n before the p**: direction is the finding,
not significance. H1 failing to reach significance is the one that constrains
the rest — it is the manipulation check, and at this n it cannot confirm the
mid-track valence move was perceptible.

The mixed model converges and is reported alongside, but it is secondary
evidence at pilot n. The per-rater paired contrasts lead because each rater
contributes one paired difference and is weighted equally, which is the
correct within-subject reduction.

## Two defects worth knowing about

**The peri-event split never used the stimulus metadata.** The original looked
for `transition_s` / `arc_boundary_s` / `boundary_s`; the renderer writes
`transition_start_s`. No key ever matched, so the meta was silently treated as
absent and the published peri-event numbers use a median-elapsed split. The
key list is corrected here, but the corrected path is **opt-in**
(`--use-meta-transitions`) so the default still reproduces the report.

**A session id was destroyed by a CSV reader.** `5e836502` is valid scientific
notation, so a dtype-inferring `read_csv` parsed it as a float that overflows
to `inf` — taking out one session's identity and half of the merge's dedup
key. The shipped shards are now written from text and match the report's own
pseudonymised copies field for field.

## What ships

| Path | What it is |
|---|---|
| `human_ratings/longtrack_ratings_*.csv` | 12 shards, 335 rows, 10 raters |
| `human_ratings/space_ratings_*.csv` | 14 shards, 760 rows, 17 raters |
| `human_ratings/ratings_all_flagged.csv` | the merged, flagged table (760 rows) |
| `human_ratings/ratings_clean.csv` | its analysed subset (668 rows, 8 raters) |
| `human_ratings/longtrack_meta.json` | the 19 rendered long tracks and their parameters |
| `train/` | the four analysis scripts |
| `inference/`, `weights/` | a README each: this section has no runtime half and fits nothing |

`data/results/` is generated and is not committed.

## Reference numbers

| Check | Value |
|---|---|
| Component-preference merge | 760 rows, 17 raters, 8 tabs |
| ...after hygiene | 668 rows, 8 raters (76 superseded, 24 tester rows) |
| Long-track experimental ratings | 255 from 8 raters, 5 usable |
| Recency ρ across probes | +0.359 → +0.457 → +0.744 |
| Recency paired test | n = 50 tracks, 7 raters, Wilcoxon p = 0.005 |
| H1 / H2 / H3 | −0.431 / +0.171 / +0.590 |
