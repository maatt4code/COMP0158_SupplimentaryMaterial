# Human ratings — seed valence calibration

| file | rows | raters | notes |
|---|---|---|---|
| `valence_ratings.csv` | 713 | 9 (`R01`…`R09`) | 712 unique rater-clip pairs; `R09` rated `pool_074` twice |
| `pool_meta.json` | 150 | — | one entry per seed preset: target and achieved (v, a), θ scalars, 32-bin harmonic distribution |

No merge needed. This study ran in one sitting per rater, so there are no boot
shards. An earlier snapshot exists in staging
(`valence_ratings__backup_20260703.csv`, 176 rows, same 9-column schema) but it
is superseded, not a fragment, and does not ship.

**Two different N values, both correct.** The θ-KRR model is fitted on `R01`
alone, which is the `N = 1` that Appendix A reports. The 9-rater figures are the
separate reliability study (Krippendorff's α = 0.096 overall, 0.366 on the 59
consensus-extreme clips, −0.135 on the 91 ambiguous middle ones). Do not
"reconcile" them.

Per-rater coverage: 150, 150, 100, 77, 68, 62, 52, 43, 11. Only two raters
completed all 150.

Names were replaced with `R01`…`R09` under the same mapping used for the
report's copy of this file, so the two stay consistent. The mapping is kept
outside this repository.
