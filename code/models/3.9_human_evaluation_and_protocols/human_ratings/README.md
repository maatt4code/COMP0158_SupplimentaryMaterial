# Human ratings — the two listening studies

Everything in this directory was collected for this project and is
pseudonymised. No participant supplied anything beyond their ratings.

Two studies ran, months apart, on two different rating apps. Both apps wrote
**one CSV per boot**, so a session that survived a restart is split across
shards. That is the single fact to hold onto before touching any of this:
naive concatenation is wrong, and the correct merge already exists in
[`../train/`](../train). Writing a new one produces a different dataset from
the one the thesis analysed.

## Identifiers

Two forms appear in the `rater` column and both are pseudonyms:

- `R03`, `R12`, `R21`, … — study participants, under the global map used
  across every section of this material.
- `rater_22cd`, `rater_7895`, … — the anonymous four-hex handle the
  component-preference app minted for a walk-up listener who never identified
  themselves. These were never linked to a person.

`session_id` is an opaque eight-hex boot handle. It is not a person: one
listener holds several, and it is used only to detect shards of the same
session. Note that some of them (`5e836502`) are valid scientific notation, so
**read these files as text**, not with a CSV reader that infers dtypes — a
naive `pd.read_csv` turns that one into `inf`.

## Long-track recency study — 12 shards, 335 rows, 10 raters

`longtrack_ratings_<session>.csv`, one schema throughout:

```
timestamp,rater,session_id,block,order_index,track_id,trajectory,chain,
condition,stage,scheduled_s,elapsed_s,rating
```

Listeners sat through whole multi-minute tracks, 11–18 July 2026. Three
trajectory conditions: `noarc` (a plain crossfade), `arc_majend` and
`arc_minend` (an arc boundary ending on a major or minor chord), plus
`practice` warm-ups that are excluded from every analysis but kept as a
scale-use check.

Each track carries three in-track probes at roughly 40 s, 70 s and 115 s
(`stage` = `probe1`/`probe2`/`probe3`) and one retrospective `overall`. **That
design is what makes the recency result possible**, because it puts the last
judgement and the summary judgement on the same track and the same person.

Two session types are pooled here and must be split, not merged:

| branch | `stage` values | feeds |
|---|---|---|
| timed probes | `probe1/2/3`, `overall` | the three contrasts and the recency test |
| sparse (free-timed) | `event`, `overall` | the peri-event analysis only |

`analyse_recency.py` excludes the sparse branch by design — any session that
emitted an `event` row — rather than by filename, so the split survives the
shards being reorganised.

**The data is thinner than 335 rows suggests.** Excluding practice, one
listener supplied 112 rows and the next 75; three supplied fewer than ten.
Five raters are usable for the condition contrasts. Report this as a pilot,
and never quote a group mean without saying who it came from.

`longtrack_meta.json` is the stimulus set: 19 rendered tracks with their
waypoints, probe times, transition times, loudness targets and the arranger's
code hash. Paths in it are basenames; the audio is not distributed, because it
is reproducible from [`3.3`](../../3.3_drone_synthesis_and_nsynth_prior) and
the conductor.

## Component preference study — 14 shards, 760 rows, 17 raters

`space_ratings_<session>.csv`, nine columns raw:

```
timestamp,rater,test,item_id,response,response2,elapsed_s,note,params
```

Eight tabs, 26 July – 21 August 2026, each asking listeners to compare
settings for one component:

| tab | rows | |
|---|---|---|
| `melody` | 207 | melody style and level |
| `reverb_clips` | 195 | the six reverb conditions |
| `pairing` | 113 | background bed type |
| `bed_level` | 65 | how loud the bed should sit |
| `modulation` | 64 | chorus / flange / phaser, mild or strong |
| `transitions` | 43 | staged vs direct |
| `stereo` | 43 | width treatments |
| `sensitivity` | 30 | scale-use check |

`params` is a JSON blob whose keys differ per tab; the merge expands it into
one column per key and maps `response`/`response2` onto a numeric `score` and
a boolean `use`. **Anything reading these raw shards directly is reading a
different, unusable shape** — the merged table has 28 columns.

### The merged tables, and the three flags

`ratings_all_flagged.csv` (760 rows, 17 raters) and its clean subset
`ratings_clean.csv` (668 rows, 8 raters) are shipped so a reader can start
from the analysed dataset without rebuilding it. Both are regenerated exactly
by `python ../train/build_results.py`.

Three derived columns drive the cleaning, and
[`../train/build_results.py`](../train/build_results.py) documents each:

- `superseded_audio` (76 rows) — the rating predates the last re-render of
  that tab's audio. Stimulus filenames are content-hashed, so a changed file
  set means the rated audio no longer exists.
- `is_tester` (24 rows) — one of six short sessions from the 26 July deploy
  window. Smoke tests, not listening data.
- `clean` — neither of the above.

### One condition was renamed

Two reverb conditions had been named after the commercial albums they were
measured from. They are re-keyed throughout this material onto descriptors of
the measurement itself: `sotl` → `long_bright`, `basinski` → `long_dark`. The
ratings and the reverb bank were re-keyed in the same commit, because a
stimulus id that no longer joins to its bank fails silently rather than
loudly. Reproducing a report table that names the old ids means mapping them
forward; nothing numeric changed.
