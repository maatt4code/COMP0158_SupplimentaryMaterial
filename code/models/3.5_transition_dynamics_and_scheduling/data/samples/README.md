# Sample outputs — schedules and policy choices

No audio here. Section 3.5's batch renderer depended on the soundscape overlay
DSP and on bed audio that is not distributed, so what this section can
demonstrate honestly is its *decisions*, and those reproduce from the shipped
weights alone.

## `schedules.csv`

Three 600-second arrangement schedules from the same seed, differing only in
where the dwell windows come from and whether the VA gate is active.

| variant | segments | what it shows |
|---|---|---|
| `handset` | 17 | the hand-set dwell windows, the default |
| `corpus_dwell` | 14 | corpus-fitted per-state dwells, floored at 20 s |
| `handset_gated_v0.9` | 16 | the same as `handset`, with the VA gate at valence 0.9 |

The corpus-fitted variant produces **fewer, longer** segments. That is the
whole point of fitting dwell on a real ambient corpus: the hand-set windows
were a guess, and the corpus says settled states run longer than the guess
allowed. Filter `handset_gated_v0.9` for `bed_bursts` and you will find none —
the gate is a hard constraint, so bursts are unreachable at high valence.

Reproduce:

```bash
python ../../inference/scheduler.py --preset overlay --trans-source corpus_dwell
python ../../inference/scheduler.py --preset overlay --gate --valence 0.9
```

## `policy_choices.csv`

What the arc policy selects at each of the four rated contexts, by both
methods: `lookup_*` is the baseline (best already-rated arc at that context),
`gp_*` is the posterior-predictive argmax over all 800 policy-controllable
combinations under a lower confidence bound.

The two disagree, and the disagreement is informative. The lookup baseline is
restricted to arcs someone actually rated; the GP can propose ramps between the
rated values of {3, 6, 12, 16} seconds and is penalised for uncertainty, so it
tends toward the interior of the rated region rather than its edges.

Reproduce:

```bash
python ../../inference/arc_policy.py --demo
```

## What is not here

The arc-pool audio (230 clips) and the bed audio are not distributed. The
ratings they produced ship pseudonymised in `../../human_ratings/`, and the
fitted posterior over them ships in `../../weights/preference_gp.npz`.
