# Human ratings — reverb preference

| file | rows | raters |
|---|---|---|
| `reverb_ratings.csv` | 380 | several, heavily unbalanced |

**Read the balance before reading the numbers.** 348 of the 380 rows come from
a single rater (`R03`). This is not a study with a cohort behind it, and the
report does not present it as one: §3.6 documents the reverb as a measured
signal-processing component, not as a human-driven feature, and no result in
Chapter 4 rests on this file.

It ships because it is what the component's tuning was checked against, not as
evidence. An earlier draft of the thesis cited an "n = 92 listener test" from
this file; that claim was removed on 2026-09-01 and must not be reinstated. The
92 is `mode: clips = 92`, a clip count, not a participant count.

No merge needed; single shard. Names replaced under the shared `R01`…`R09` map.
