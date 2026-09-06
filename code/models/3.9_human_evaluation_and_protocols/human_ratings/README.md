# Human ratings — evaluation studies

Two studies, both fragmented across web-app restarts. **Read §0.1b of the
top-level README before touching either.** Naive concatenation is wrong in both
cases and the merge scripts already exist.

## Long-track recency study

12 session shards, `longtrack_ratings_<session>.csv`, one schema throughout
(17 columns, `timestamp,rater,session_id,block,order_index,track_id,trajectory,
chain,condition,stage,scheduled_s,elapsed_s,…`).

Source of truth is `longtrack_pool/huggingface_ratings/`. Its `logs/`
subdirectory holds byte-identical copies of 11 of the 12 and is **missing**
`longtrack_ratings_020fff1f.csv` (64 rows), so the parent is the superset. Both
carry a zero-byte `longtrack_ratings.csv`; drop it.

Three trajectory conditions: unchanged static baseline, major-ending arc,
minor-ending arc. Each track carries three in-track probes plus a retrospective
rating, which is what the recency analysis contrasts.

## Component preference study

14 session shards, `space_ratings_<session>.csv`, 9 columns raw.

Source of truth is `rating_space/downloaded_fresh/logs/`. The sibling
`downloaded/` folder is an **earlier pull and a subset**: `8c842679` grew from
156 to 312 rows between pulls and three sessions appear only in the fresh one.
Merging both double-counts. Discard `downloaded/`.

Merged output is `ratings_all_flagged.csv` (708 rows, 14 raters) and its clean
subset `ratings_clean.csv` (617 rows, 6 raters). Produced by
`analyze_space_ratings.load()` then `build_results.py`, both of which ship in
`../train/`. Verified 2026-09-06: the shards hold 709 rows across 15 raters, and
the one row and rater that do not survive is the `setup_check` deploy smoke row.

Eight tabs, by row count: melody 181, reverb_clips 170, pairing 113,
bed_level 64, modulation 64, transitions 43, stereo 43, sensitivity 30.

Two hygiene flags drive `clean`, and `build_results.py:27-33` documents how they
were derived. `superseded_audio` marks ratings made before that tab's audio was
last re-rendered, which matters because filenames are content hashed, so a
changed file set means the rated audio no longer exists. `is_tester` marks six
short smoke sessions from the 26 July deploy.
