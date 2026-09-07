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

No merge needed; single shard.

**Identifiers.** Rater names were replaced under the same global map every other
section uses, so `R03` is the same person here as in Sections 3.4.2 and 3.5. The
second rater appears as `rater_0a2e`: that is not a name the map replaced but
the throwaway id the rating app itself minted, so an outside rater never had to
type anything identifying. The report's own copy of this file uses both ids
identically.

**The `reverb` column was re-keyed.** Two condition ids were the names of the
recording artists whose acoustics were measured; they are now `long_bright` and
`long_dark`, matching `weights/reverb_bank_measured.json`. The rename ran in the
same operation as the bank's, because these rows must join to their stimulus,
and it covers compound ids such as `bloom_vs_dry:long_bright:A=reverb`. No
rating value changed.
