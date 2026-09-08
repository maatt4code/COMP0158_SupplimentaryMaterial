# 3.8 The Unfinished Conductor

The live application. This is where every other section becomes one thing you
can listen to: a session walks through valence–arousal space, a preset is
retrieved for wherever the walk currently is, a transition into it is chosen,
and audio is rendered ahead of playback.

## Run it

```bash
cd code/conductor
python app.py --no-share            # http://localhost:7860
```

That is the whole setup. It needs **no dataset, no network and no rating
data** — every weight it loads ships in [`weights/`](weights). The environment
is `env/env_linux_nvidia.yml` (or `env_windows_arc.yml`); nothing beyond it is
required.

A healthy start looks like this:

```
Engine: 20000 anchors, human label space available
  melody render engine ready
  transformer 'L3_d128' ready
  transformer 'L6_d128' ready
  transformer 'L6_d256' ready
  6 reverb IRs ready
[beds] 7 of 20 available (13 need the full corpora; set $DRONE_BEDS)
  7 of 7 beds decoded
* Running on local URL:  http://0.0.0.0:7860
```

Useful flags:

| Flag | What it does |
|---|---|
| `--no-share` | local only; without it a public relay link is created |
| `--port N` | serve on N. **Startup refuses a taken port** rather than quietly moving to the next one and leaving your browser on the old instance |
| `--label-space {judge,human}` | which valence axis retrieval matches against. Session-start only — see below |
| `--selftest` | control-flow check with a fake engine and renderer. No server, no audio, a few seconds |
| `--audio-format {mp3,ogg,flac,wav}` | container for segments sent to the browser. mp3 is ~7.3x smaller than wav, which is what closes the join over a relay; flac is the lossless fallback at 1.6x |
| `--audio-out` | stream to **this machine's** sound card instead of the browser. Removes the segment join entirely — but over SSH you will hear nothing |
| `--no-webaudio` | fall back to the plain `<audio>` element. Not gapless by construction; for debugging that path only |

`python smoke_test.py` runs everything below as assertions: 30 checks, no
server, no audio device.

## Why the label space is chosen at session start

The two valence axes are different instruments, not two calibrations of one.
Judge valence tops out at **+0.24** and human valence reaches **+0.57**, and
they disagree about the nearest anchor at 99.8% of targets. So the dropdown
picks which bank you are searching, and it cannot be changed mid-session
without changing what the music means halfway through.

Each axis has its own boundary guard, because a fence fitted in judge
coordinates keeps the walk out of exactly the region the human axis exists to
reach. Asked for v=+0.4, the judge-fitted fence settles at −0.31 and the
human-fitted one at +0.30. Both ship as frozen weights (τ = 0.5898 and
0.5505). An earlier version swapped the retrieval bank and left the fence
behind, which meant choosing "human labels" changed which anchors were
searched while still forbidding half the plane they live in.

## The surface

One skin ships: the **deck**. Five panels grouped by *provenance* rather than
by subsystem, each badged with where its behaviour actually came from —

| Badge | Meaning |
|---|---|
| **LEARNED** | fitted from listening ratings: a model, not a rule |
| **CORPUS** | measured from ambient recordings: a compositional prior, not a preference |
| **JUDGE** | frozen MERT+ridge labels. Valid on arousal; valence is the axis this thesis shows to be unreliable |
| **AUTHORED** | hand-set by ear. Not rated, and no emotion claim attaches to it |

That grouping is the point of the interface: it makes the evidential status of
every control visible while you are listening to it, instead of leaving the
reader to work out which knobs are backed by data.

A second "classic" surface existed and was dropped. It duplicated every
component inline and had to be kept in step by hand, and each drift between
the two was a bug that appeared under one skin only. The skin **contract**
stays: a skin returns its components in a dict keyed by name, `skins.validate`
enforces that every required control is present, and no handler ever refers to
a layout position. A skin that forgets a control therefore fails at
construction rather than rendering buttons that quietly do nothing.

## Layout

```
app.py          the session, the transition policy, the render loop
engine/         COPIES of the migrated inference modules, plus the renderer
UI/             the deck, drawn faders, reactive visuals, browser audio
runtime/        optional overlays: surface noise, near/far placement
weights/        every frozen artefact the app loads (50 MB)
assets/         the VA pad backgrounds, and the demo bed subset
```

### `engine/` is copied, and the copies are checked

The conductor is **self-contained** by decision: it carries its own copy of
every module and weight rather than importing across the tree, so this
directory can be lifted out and run. Fifteen of the eighteen files in
`engine/` are verbatim copies of a section's `inference/` module.

Duplication drifts, and a drift here is invisible — a fix made in a section
silently fails to reach the app, and the app keeps working, which is why
nobody notices. So [`engine/COPIED_FROM`](engine/COPIED_FROM) pins every copy
to its source by checksum and `code/verify.py` fails if any stops matching. It
distinguishes the two failures, because they need different fixes: a copy
edited in place (the edit will be lost on the next re-copy) versus a source
edited without re-copying (the app is running old code).

**Never edit a file listed in `COPIED_FROM`.** Edit the section original and
re-copy. `arranger.py`, `render_params.py` and `bed_bank.py` are the
conductor's own and have no section original.

### What comes from where

| Module | Section | What it does |
|---|---|---|
| `retrieval.py`, `decoupled_engine.py` | 3.4.2 | retrieval over 20,000 labelled anchors |
| `guard.py` | 3.4.2 | the boundary guard, one per label space |
| `arc_policy.py` | 3.5 | arcs, and the frozen preference GP |
| `scheduler.py` | 3.5 | the semi-Markov bed scheduler |
| `coherence_reranker.py` | 3.5 | sequential-coherence re-ranking |
| `reverb.py`, `reverb_bank.py` | 3.6 | the measured reverb spaces |
| `grammar.py`, `melody_markov.py`, `melody_transformer.py`, `transformer_model.py` | 3.7 | the melodic line |
| `arranger.py`, `render_params.py` | 3.8 | the voice, and the frozen render settings |
| `ddsp_synth.py` | 3.3 | the differentiable synthesiser |

## Nothing is fitted at startup

Four things used to be fitted when the app booted, which meant the conductor
had to carry rating data and that the policy a listener heard depended on a
fit that ran on their machine. All four are now frozen weights:

| Was fitted at boot | Now loads |
|---|---|
| the Bradley–Terry preference GP, from raw pairwise comparisons | `weights/preference_gp.npz` |
| the boundary guard, per label space (~2.3 s each) | `weights/boundary_guard*.json` |
| the melody timbre anchor, from the rated seed pool | `weights/melody_anchor.json` |
| the bed bank, from the curation log | `weights/bed_bank.json` |

The melody anchor is worth singling out. Choosing it needs Section 3.4.2's
rated pool, and the conductor ships none — so a runtime that refits does not
fail, it quietly returns something else. It did: the conductor resolved anchor
13161 while the section, the frozen weight and the report all say **4174**.
13161 is the fourth-ranked candidate, carrying `swell_depth` 0.567 against
4174's 0.011, so the deployed melody wobbled in exactly the way the gentleness
filter exists to prevent, and nothing reported a problem.

## How a transition is chosen

When the target's valence moves away from the current state by more than a
threshold — symmetric in both directions, because the pool holds real
preference data for darkening transitions too — or when you press **force**,
the conductor asks the arc policy for the best transition **by both methods**
and logs both, whichever one renders:

* `lookup_best_arc` — the honest baseline: the best already-rated arc at this
  context.
* `best_by_gp_predict` — the preference GP's posterior over a dense grid of
  policy-controllable parameters (~800 combinations), ranked by **lower
  confidence bound** rather than raw mean. Argmax of the mean was observed
  picking parameter values far outside anything ever rated; the uncertainty
  penalty is what stops the policy chasing GP noise in unrated territory.

Which one renders is switchable live, so the baseline can be *heard* rather
than only logged. Lookup mode replays the winning arc's transition **recipe**
on today's retrieved presets — not that arc's pre-rendered audio.

`lookup_best_arc` is memoryless: it picks the best arc for the current context
independently every time, so nothing models order and it will happily emit two
large brightenings back to back. In lookup mode the near-tie band is re-ranked
toward the corpus type that most coherently follows the last arc played.
**The discipline matters more than the mechanism: human ratings decide what is
GOOD, and the corpus chain only breaks TIES** — it never selects outside the
near-tie band, so a preference judgement is never overridden by a statistic.

## The VA pad is a measurement

`assets/pad_judge.png` and `assets/pad_human.png` are the backgrounds behind
the target you drag. They are not decoration: each shades out the part of the
plane its bank **cannot reach**, so the interface stops promising what the
instrument cannot deliver. Judge coverage leaves 38.5% of the square
unreachable, human 27.4%, which is why there is one per label space and why
the app swaps them with the dropdown.

They grey out the unreachable region rather than colouring the reachable one:
colouring the reachable part reads as "this area is blocked", the exact
opposite of what it means.

Both regenerate byte-identically from the shipped banks:

```bash
python ../models/3.4.2_human_grounding_and_retrieval/train/make_va_pads.py
```

## Beds

Bed audio is third-party and is **not** redistributed in full. It comes from
ESC-50 (CC BY-NC) and Emo-Soundscapes, so the supplementary material documents
and fetches them the way it does NSynth and Essen.

One bed per type ships in [`assets/beds/`](assets/beds) so the layer works out
of the box — chosen licence-first (six CC0, one CC-BY) with per-clip credit
read from the corpora themselves, in
[`assets/beds/ATTRIBUTION.md`](assets/beds/ATTRIBUTION.md). For the full bank,
fetch both corpora and set `$DRONE_BEDS`, or rebuild with
`models/3.5_transition_dynamics_and_scheduling/train/build_bed_bank.py`.

The bank is filtered to beds whose audio is actually present at load, and the
startup line tells you how many that is. `resolved` in the bank file records
whether the corpus existed when the bank was *built*, which is a different
question from whether the audio is here now.

## Reference numbers

| Check | Value |
|---|---|
| Retrieval bank | 20,000 anchors across two labelled banks |
| Valence range | judge −1.00 … +0.24, human −1.00 … +0.57 |
| Boundary guard τ | 0.5898 judge, 0.5505 human |
| Preference GP | frozen posterior over 230 rated arcs |
| Melody anchor | 4174 (frozen; 4 candidates cleared the filter) |
| Melody checkpoints | L3/d128, L6/d128, L6/d256 — all three selectable |
| Reverb | 6 conditions, 3 measured IRs |
| Beds | 20 in the bank, 7 shipped |
| Output ceiling | peak 0.95, A-weighted target 0.015 |
