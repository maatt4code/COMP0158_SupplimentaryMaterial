# Human ratings — transition preference

| file | rows | raters | design |
|---|---|---|---|
| `arc_ratings.csv` | 120 | 8 | forced-choice A/B over transition pairs |
| `arc_valence_ratings.csv` | 199 | 4 | single-stimulus valence on arc scenes |

`arc_ratings.csv` is what the Bradley-Terry preference GP is fitted on
(`../train/fit_preference_gp.py`). Nine of the 120 comparisons are repeat trials
used to check within-rater consistency, leaving 111 that carry a contrast.

**What this study varied.** Three factors only: ramp duration (`ramp_s`), chord
mode (`chord`) and parameter distance (`dist_bin`). It did **not** compare
equal-power crossfades against parameter interpolation. Grepping the whole
parameter blob returns zero hits for `equal`, `crossfade`, `interp` or `linear`,
because every trial used the same fade.

**What it found: nothing significant.** Across the 111 contrast trials the split
is A 48 / B 51 / no-difference 12, and no single factor reaches significance
(best p = 0.58). That null is used deliberately in §3.5.2: because no archetype
was favoured, fixing one as a runtime default would have been arbitrary.

No merge needed; both files are single-shard.
