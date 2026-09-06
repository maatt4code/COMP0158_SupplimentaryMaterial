# 3.9 Human Evaluation Methodology and Listening Protocols

The listening studies: long-track recency, and component preference across
eight tabs. Merge and hygiene scripts plus the rating data.

**Not yet migrated.** Sources and destinations are in the top-level
[`code/README.md`](../../README.md) §2.5 and §0.1b.

## Datasets

No external dataset. Every input was collected for this project and ships
pseudonymised in [`human_ratings/`](human_ratings).

Stimuli were rendered by this pipeline, so they are reproducible from
[`3.3`](../3.3_drone_synthesis_and_nsynth_prior) and the conductor rather than
distributed as audio.

## Read this before touching the data

Both studies are fragmented across web-app restarts, and merging them naively
double-counts. The traps and the existing merge scripts are documented in
[`human_ratings/README.md`](human_ratings/README.md) and in the top-level
[`code/README.md`](../../README.md) §0.1b. Do not write a new merge: it would
produce a different dataset from the one the report analysed.
