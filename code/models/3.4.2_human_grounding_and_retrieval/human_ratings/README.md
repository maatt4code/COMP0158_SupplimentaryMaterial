# Human ratings — seed valence calibration

| file | rows | raters | notes |
|---|---|---|---|
| `valence_ratings.csv` | 713 | 9 (`R01`…`R09`) | 712 unique rater-clip pairs; `R09` rated `pool_074` twice |
| `pool_meta.json` | 150 | — | one entry per seed preset: target and achieved (v, a), θ scalars, 32-bin harmonic distribution |
| `pool_mert.npz` | 150 | — | MERT-v1-95M embeddings of the rated clips, 768-dim float64; the recipe is in `pool_mert_meta.json` |

No merge needed. This study ran in one sitting per rater, so there are no boot
shards. An earlier snapshot exists in staging
(`valence_ratings__backup_20260703.csv`, 176 rows, same 9-column schema) but it
is superseded, not a fragment, and does not ship.

**Two different N values, both correct.** The θ-KRR model is fitted on `R03`
alone, which is the `N = 1` that Appendix A reports. The 9-rater figures are the
separate reliability study (Krippendorff's α = 0.096 overall, 0.366 on the 59
consensus-extreme clips, −0.135 on the 91 ambiguous middle ones). Do not
"reconcile" them.

`R03`, not `R01`. The label source is identified by the hygiene rule in the
original log, which folded a session-test account's real-pass rows back into
their owner's identity; that account and its owner share one pseudonym. The
identification is checked, not assumed: propagating `R03` reproduces the
reported LOOCV row exactly (MAE 0.566, ρ 0.337, sign accuracy 0.69), while
`R01` — who also rated all 150 clips — gives 0.603 / 0.211 / 0.64.

Per-rater coverage: 150, 150, 100, 77, 68, 62, 52, 43, 11. Only two raters
completed all 150.

Names were replaced with `R01`…`R09` under the same mapping used for the
report's copy of this file, so the two stay consistent. The mapping was
recovered positionally — the report's copy has the same 713 rows in the same
order, so pairing the two rater columns row by row determines it exactly — and
it is kept outside this repository.

Clip paths are repository-relative. No absolute path, username or personal name
survives in any file here; the rated audio itself is not distributed.
