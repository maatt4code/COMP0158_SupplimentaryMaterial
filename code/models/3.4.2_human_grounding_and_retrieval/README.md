# 3.4.2 Human Grounding and Manifold Retrieval

Propagates 150 human seed ratings across the 20,000-preset bank with theta-KRR,
then serves 1-NN retrieval. This is what the conductor actually navigates.

**Not yet migrated.** Sources and destinations are in the top-level
[`code/README.md`](../../README.md) §2.4.

## Datasets

No external dataset. This section consumes two things produced upstream:

| Input | From |
|---|---|
| the preset bank and its theta index | [`3.3_drone_synthesis_and_nsynth_prior`](../3.3_drone_synthesis_and_nsynth_prior) |
| 150 human seed ratings | [`human_ratings/`](human_ratings), collected for this project |

The ratings are pseudonymised to `R01`…`R09`. See
[`human_ratings/README.md`](human_ratings/README.md) for the study design and
the two different `N` values it reports.
