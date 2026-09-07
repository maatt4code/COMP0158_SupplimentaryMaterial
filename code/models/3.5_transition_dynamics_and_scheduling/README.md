# 3.5 Transition Dynamics and Scheduling

Two questions, kept deliberately separate: **which** transition to make, and
**when** to change the arrangement. The first is learned from human preference;
the second is a semi-Markov process whose timing is grounded in a corpus. They
meet only at the re-ranker, and only under a rule that stops the corpus
overruling the humans.

## Which transition: a preference GP over arc parameters

Two rating studies feed one latent utility over the arc-parameter vocabulary:
pairwise "which transition feels better" comparisons, and absolute valence
ratings on individual arcs. Both terms share the same latent f.

- Pairwise comparisons use a **Davidson tie-aware likelihood**. Ties are data,
  not coin-flips: the rating app offered a "same" button precisely so that
  indifference could be recorded rather than forced into a preference.
- Absolute ratings use a linear link with a **per-scene offset**, which soaks
  up the pad-level valence baseline so the fitted utility isolates the
  transition's contribution rather than the pad it happened on.
- The posterior is a **Laplace approximation** over a GP with an ARD-RBF kernel
  on z-scored parameters, so comparisons **share strength across arcs**. That
  matters enormously here: the median arc has 2 comparisons, and isolated
  per-item Bradley-Terry scores would be almost pure prior.

Kernel hyperparameters are fixed sensible defaults rather than
marginal-likelihood optimised. That is stated in the code, flagged in the
report it prints, and is the honest choice at this number of comparisons.

## When to change: a semi-Markov scheduler

The texture round's comments showed that what broke coherence was **pacing**,
not parameters — transitions arriving too soon after one another. A first-order
Markov chain has geometric, memoryless dwell times and will happily re-switch
seconds after a switch, which is exactly that failure. So every state carries an
explicit dwell **window**, making the process semi-Markov.

Uniform windows are a deliberate choice, not laziness: the bounded support *is*
the model. The upper bound is the anti-deadness guarantee, the lower bound the
anti-overload one, and the distribution's shape inside them is statistically
unidentifiable from the handful of segments a listener hears per track.

Naming honesty, and it matters for the viva: **this is not a hidden Markov
model.** Nothing is hidden and no emissions are decoded. It would become one
only if the states were later inferred from ratings.

## The re-ranker, and the rule that makes it safe

The policy is memoryless — it picks the best arc for the current context
independently every time — so nothing models order, and it can emit two large
brightenings back to back. The re-ranker fixes that with the corpus type
transition matrix, under one load-bearing rule:

> Human ratings decide what is GOOD; the corpus chain only breaks TIES toward
> what follows coherently.

It never selects outside a near-tie band around the best utility, so it cannot
override a preference judgement — only reorder arcs the GP is indifferent
between. That is why a corpus prior is acceptable here where a corpus prior over
the whole decision would not be: it sits on top of the validated preference
model rather than replacing it. The smoke test asserts the band rule directly.

## Run order

| # | Command | What it does |
|---|---|---|
| 0 | `python train/extract_transition_typology.py --audio-dir <dirs>` | audio → transition descriptors (any corpus) |
| 0b | `python train/fit_corpus_hsmm.py --cache <cache> --out weights/hsmm_transitions.json` | descriptors → the semi-Markov timing fit |
| 1 | `python train/arc_rating_app.py --selftest` | checks the rating protocol's queue logic |
| 2 | `python train/fit_preference_gp.py` | fits the preference GP and reports diagnostics |
| 3 | `python train/fit_preference_gp.py --freeze --out-dir weights` | freezes the posterior for the runtime |
| 4 | `python inference/arc_policy.py --demo` | queries the frozen posterior at each rated context |
| 5 | `python inference/scheduler.py --selftest` | checks the scheduler's contract |
| 6 | `python inference/coherence_reranker.py --selftest` | checks the near-tie band rule |

## What reaches the runtime

`inference/` is loaded by the conductor. None of it imports the path helper,
resolves a dataset root, refits anything, or reads any rating file — the smoke
test asserts all four. `arc_policy.py` loads the frozen posterior from
`weights/preference_gp.npz`, which is what lets the conductor start with every
dataset missing and no rating data present.

## Retraining the timing model on your own corpus

The timing pipeline ships in full and is **corpus-agnostic**. Point it at any
directories of audio; each becomes a group label and is walked recursively.

```bash
# A. audio -> transition descriptors (slow, cached, resumable per group)
python train/extract_transition_typology.py     --audio-dir /path/to/corpus/a /path/to/corpus/b     --out data/typology_transitions.json

# B. descriptors -> the semi-Markov fit the scheduler reads
python train/fit_corpus_hsmm.py     --cache data/typology_transitions.json     --out weights/hsmm_transitions.json
```

Step A finds structural boundaries by multi-scale Foote novelty and measures
what kind of transition each one is; step B clusters those descriptors into
types and fits state-conditional dwell windows and transition rates by counting
real ordered sequences. `python train/fit_corpus_hsmm.py --selftest` checks the
fit's properties on synthetic sequences, with no audio at all.

The shipped fit came from 1027 state transitions across 95 tracks in 14 groups.
Refitting from that descriptor cache reproduces `weights/hsmm_transitions.json`
with zero differing fields.

**The corpus itself is not redistributable, so it does not ship — but the
method does.** `weights/SHA256SUMS` records the fitted artefact's checksum so it
stays traceable:

```
f3367c866eff7d5af024c901a8ef23e31eccda4a8c1dc9c850297a640d730424  hsmm_transitions.json
```

Read carefully before citing any of it. The per-state dwell windows and the
active-to-settled return rate are **corpus-fit**; the duck-versus-bursts split
is **authored**, because the corpus has no correlate for it; and the whole thing
is a compositional prior, never a listener-preference model.

## What deliberately does not ship

**The corpus audio.** Not redistributable. The method that consumed it ships,
as above.

**The 20-second dwell floor is a design constraint, not a data finding.** The
corpus's active-regime holds start around 2 seconds, which would reintroduce
precisely the "re-switches too soon" failure the scheduler was built to fix. The
fitted window and the floored window are both reported, so the two never blur.

**The rating UI.** The protocol and the queue logic that implements it ship and
are self-tested; the interactive front end does not, since the arc audio it
would play is not distributed.

## Human ratings

`human_ratings/` ships three pseudonymised rating files and the arc parameter
vocabulary. Rater identifiers are consistent with every other section and with
the report: the same person is `R03` everywhere.

| file | rows | raters |
|---|---|---|
| `arc_ratings.csv` | 120 | 8 |
| `arc_valence_ratings.csv` | 199 | 4 |
| `texture_ratings.csv` | 122 | 1 |
| `arc_pool_meta.json` | 230 arcs | — |

The texture file's `arc`-kind rows feed the same absolute-valence term as
honest replicate observations. Rows flagged `bad_texture` are bounds evidence,
not utility observations, and are excluded.

## Reference numbers

| Check | Value |
|---|---|
| Arcs in the pool | 230 |
| Pairwise comparisons | 120, tie rate 10.8% |
| Comparisons per touched arc | min 1, median 2, max 4 (132/230 arcs touched) |
| Fitted Davidson tie parameter | 0.245 |
| In-sample sign accuracy, non-tie | 65.4% (NOT held out) |
| Absolute valence ratings | 219 across 51 scenes |
| Single-axis seed pairs | 293 (96 per axis, 5 probes) |
| Rating target / full pair space | 125 / 711 |
| Corpus fit | 1027 transitions, 95 tracks, 14 artists |
| Corpus active dwell, fitted then floored | (2.0, 28.6) s → (20.0, 28.6) s |
| Frozen posterior self-check | mean 4.9e-4, std 7.0e-4, batch-vs-single 3.6e-15 |

The comparisons-per-arc row is the one to read twice. At a median of two
comparisons per arc, per-item scores would be meaningless; the entire reason
this is a GP over parameters rather than a Bradley-Terry model over arcs is to
make that number workable.

## Smoke test

```bash
python smoke_test.py     # exit 0, no dataset, no audio, no network
```
