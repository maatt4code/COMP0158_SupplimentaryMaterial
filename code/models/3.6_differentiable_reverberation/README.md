# 3.6 Differentiable Reverberation

A reverb you can fit by gradient descent, a bank of real spaces measured from
impulse responses, and a stimulus ladder built so that tail length and tone can
be told apart.

## The three pieces

**A differentiable reverb** (`train/model.py`). An FFT-convolution reverb whose
impulse response is a learnable parameter, fitted by multi-scale STFT loss — the
torch analogue of `ddsp.effects.Reverb`, written directly because this project's
DDSP layer is a torch reimplementation. Two parameterisations: an interpretable
one carrying `{gain, decay_time_s, damping_hz}`, and a fully free impulse
response. The interpretable one is what the rest of the section uses, because
its numbers mean something a person can read.

**Measurement and fitting** (`train/fit_from_irs.py`). For each impulse response,
measure RT60 by Schroeder backward integration and the spectral centroid of the
tail, then fit the differentiable model *initialised from those measurements*.
Measuring and fitting stay separate: the measurements are properties of the
recording and need no model.

**Application** (`inference/reverb.py`, `inference/reverb_bank.py`). Numpy only,
no torch. Where a real impulse response exists the runtime convolves it
directly, so no fitting error reaches the audio — the fitted parameters choose
which IR represents a category, they do not reproduce it.

## The ladder, and why it is built this way

```
dry -> nature 0.29s -> brutalism 0.95s -> stairwell 2.39s -> long_bright 5.05s
                                                          -> long_dark    5.16s
```

Tail length is the primary factor and the ladder spans the whole measured range.
At the long end sit a **tone-matched pair**: `long_bright` at 2434 Hz and
`long_dark` at 438 Hz, at nearly identical tail length. That pair is the design.
Without it, tail and tone are confounded — long spaces in this material also
tend to be dark — and no rating could distinguish "longer" from "darker".

**Wet is held constant** across every wet condition. The manipulation is which
space, not how much of it, so a shift cannot be read as "more effect produced
more effect".

The representative IR for a category is the one whose RT60 is closest to the
category median: deterministic, no cherry-picking.

## A known limitation of the loudness control

Conditions are RMS-matched to dry so that a level difference cannot masquerade
as a reverb effect. The peak guard, however, runs *after* the match, so a
condition whose matched output would clip is scaled back down — undoing the
control. On the smoke test's source this binds on `stairwell` alone, at −4.7 dB;
on a louder source it binds more widely.

This is preserved deliberately. These are the stimuli that were rated, and
silently re-levelling them now would mean the shipped code renders something
different from what the listeners heard. The fix for a future round is one
common headroom scalar across all conditions, so peak limiting cannot change
their relative levels. The smoke test asserts the mechanism — any deviation must
be downward and must coincide with the peak ceiling — so it cannot drift
unnoticed.

## Source release: a protocol constant, not a knob

Rendered clips stop dead at full amplitude, so the dry component vanishes in a
single sample. That step is an audible click *and* an abrupt loss of half the
mix, landing at the same moment in every clip — it would be rated as a property
of the reverb when it is a property of the source. The source is therefore
raised-cosine shaped before any reverb is applied, identically in every
condition, dry included.

The release is not tuned and not learned: there is no target to fit it against.
Its *shape* is taken from deployment (the same cos² ramp the scheduler uses for
crossfades) rather than invented, and its *duration* is bounded by the stimulus.
Report it alongside the ladder.

## Condition names

Two ladder conditions were originally named after the recording artists whose
acoustics were measured. They are re-keyed onto the measurement itself —
`long_bright` and `long_dark` — which is both what the experiment manipulates
and what the report describes.

The re-key is derived, not hand-assigned: each category is named from its own
measured medians, bucketed by tail (`short` under 1.5 s, `mid` under 3.5 s,
`long` above) and tone (`dark` under 1000 Hz, `mid` under 2300 Hz, `bright`
under 3000 Hz, `air` above). The boundaries are chosen so the measured
categories stay distinguishable, and the de-identification refuses to run if the
mapping is not one-to-one. `reverb_bank_measured.json` records the rule.

The ratings' `reverb` column was re-keyed in the same operation, because the
ratings must join to their stimulus. The measured numbers are unchanged
throughout: they are measurements of a recording's acoustics, not the recording.

**Note for the report:** the thesis's own data directory still ships
`reverb_ratings.csv` with the original `sotl` / `basinski` condition ids, which
`appendix_datasets.tex`'s scrubbing claim covers. That is a report-side fix, not
a code one.

## Datasets, and the audio this section redistributes

| Resource | Used for | Terms | Where to get it |
|---|---|---|---|
| **EchoThief Impulse Response Library** | the 115-space IR bank, and the three IRs shipped in `weights/irs/` | free to use with credit; see the site's terms | <http://www.echothief.com/> |

Set `$DRONE_ECHOTHIEF`, or pass `--ir-root`, to rebuild the bank from the full
library. Only `train/fit_from_irs.py` needs it; everything else runs from the
shipped bank.

**This section redistributes three impulse-response recordings.** They are
third-party audio, not project output, and they ship because the ladder cannot
be rendered without them — the whole point of the IR conditions is that the
runtime convolves the *actual measurement* rather than an approximation of it.
Credit belongs to the library's author, and `weights/irs/ATTRIBUTION.md` names
the source next to the files themselves so a copied `weights/` directory
carries its own attribution.

The repository-wide split applies here: `LICENSE` covers code only, and
attribution files govern redistributed audio.

## Retraining on your own impulse responses

```bash
python train/fit_from_irs.py --ir-root /path/to/impulse_responses \
    --out data/ir_reverb_bank.json
python train/fit_from_irs.py --ir-root <dir> --pilot        # 3 IRs, quick
python train/fit_from_irs.py --measure-only --ir-root <dir> # no fitting
```

Categories come from the containing directory names, so any library organised
that way works.

## What ships

| Path | What it is |
|---|---|
| `weights/ir_reverb_bank.json` | 115 measured spaces across 8 categories |
| `weights/reverb_bank_measured.json` | 76 acoustics fitted from finished recordings, re-keyed |
| `weights/irs/` | the three impulse responses the ladder actually uses, with their attribution |
| `human_ratings/reverb_ratings.csv` | 380 ratings, pseudonymised and re-keyed |
| `data/samples/` | the whole ladder rendered, plus the late-field variant |

There is no checkpoint: the fitted model is three scalars per impulse response,
and they live in the JSON banks.

## Reference numbers

| Check | Value |
|---|---|
| Impulse responses measured | 115, in 8 categories |
| Acoustics fitted from recordings | 76, in 5 re-keyed categories |
| Ladder | 6 conditions, tails 0.00 / 0.29 / 0.95 / 2.39 / 5.05 / 5.16 s |
| Tone-matched pair | 2434 Hz against 438 Hz at Δtail 0.11 s |
| Ratings | 380 rows, 2 raters |
| Level match | 4 of 5 wet conditions within 0.01 dB; `stairwell` −4.69 dB |
| IR-supervised parameter recovery | decay 0.25 s, damping 900 Hz, gain 0.60, all recovered |

## Smoke test

```bash
python smoke_test.py     # exit 0, no dataset, no network
```
