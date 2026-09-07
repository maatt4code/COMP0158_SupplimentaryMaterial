# 3.4.1 Surrogate-Guided Optimisation

Labels a preset bank with a frozen affect judge, distils that judge into a small
differentiable proxy, and optimises a mapping network against the proxy. This is
the loop that reward-hacked, and Section 4.1 is the result.

The section is worth reading as a negative result that was measured carefully
rather than a pipeline that works. Everything here runs, reproduces, and
produces a model that is not deployed. `inference/` is deliberately empty: see
[`inference/README.md`](inference/README.md).

## The run order

Every script takes its paths as flags and prints what it resolved. Paths below
are relative to `train/`.

| # | Command | What it does |
|---|---|---|
| 1 | `python label_dataset.py --data-dir ../data/chord10k` | scores a rendered preset bank with the frozen judge |
| 2 | `python train_cvae.py --data-dir ../data/chord10k` | the parameter-space inverse model |
| 3 | `python train_judge_proxy.py --data-dir ../data/chord10k --n-models 4` | distils the judge into a differentiable CNN ensemble |
| 4 | `python train_closed_loop.py --bank-dirs ../data/chord10k` | trains the mapper through the renderer against the proxy |
| 5 | `python evaluate_cycle_consistency.py --mode render --inverse mlp --data-dir ../data/cycle_mlp` | renders a VA grid from an inverse model |
| 6 | `python label_dataset.py --data-dir ../data/cycle_mlp` | scores those renders with the **real** judge |
| 7 | `python evaluate_cycle_consistency.py --mode report --data-dir ../data/cycle_mlp` | requested against read-back: MAE, bias, plot |
| 8 | `python dagger_render.py --n 1000 --out-dir ../data/dagger_r1` | a DAgger round, to close the proxy's distribution gap |

Steps 5–7 are a three-part cycle on purpose. The proxy is the thing being gamed,
so it cannot also be the referee; the real judge has to be run in between.

The supporting experiment, which asks whether the judge can hear harmony at all:

| # | Command | What it does |
|---|---|---|
| a | `python render_resolution_probe.py --out-dir ../data/resolution_probe` | 75 clips: minor, major, resolution, de-resolution, glide |
| b | `python label_dataset.py --data-dir ../data/resolution_probe` | scores them with the frozen judge |
| c | `python cross_domain_validity.py --data-dir ../data/resolution_probe` | scores the same clips with three different judges |
| d | `python cross_domain_indomain.py --n 150` | runs those judges on real audio in their home domains |

Step **d** is what makes a null result at **c** defensible: it separates "these
judges are blind here" from "these judges are broken".

## What runs without a dataset

Steps 5, 7 and a: the mapper, the CVAE, the judge heads and Section 3.3's timbre
prior all ship. `data/samples/` holds four renders and the commands that
reproduce them.

Steps 1, 6, b, c and d need the Hugging Face models, which download on first
use. Steps 1–4 need a rendered preset bank from Section 3.3.

## Datasets and pretrained models

| Resource | Used for | Licence | Where to get it |
|---|---|---|---|
| **DEAM** (MediaEval Database for Emotional Analysis in Music) | fitting the valence ridge head, 1,802 tracks; the in-domain control | research use, registration | <https://cvml.unige.ch/databases/DEAM/> |
| **Emo-Soundscapes** | fitting the arousal ridge head, 600 environmental recordings; the in-domain control | academic, non-commercial | <https://metacreation.net/emo-soundscapes/> |
| **MERT-v1-95M** | the embedding both ridge heads sit on | CC BY-NC 4.0 | <https://huggingface.co/m-a-p/MERT-v1-95M> |
| **audEERING wav2vec 2.0** (`wav2vec2-large-robust-12-ft-emotion-msp-dim`) | the third judge in the cross-domain comparison | CC BY-NC-SA 4.0 | <https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim> |

Set `$DRONE_DEAM` and `$DRONE_EMO`, or pass `--deam-root` and
`--emo-soundscapes-root`. Only `cross_domain_indomain.py` reads either dataset;
the ridge heads themselves ship fitted, so nothing else needs them.

## What ships in `weights/`

| File | What it is |
|---|---|
| `affect_ridges.npz` | the three frozen ridge heads, 20 KB |
| `inverse_cvae.pt` | the parameter-space CVAE |
| `judge_proxy.pt`, `judge_proxy_e0.pt` | the distilled judge proxy |
| `closed_loop_mapper.pt` | the closed-loop mapper, the model Section 4.1 evaluates |

**The frozen judge configuration is not what the original argument parser
defaulted to.** Valence is MERT to the DEAM ridge; arousal is the same MERT
embedding to the Emo-Soundscapes ridge. That is now the default here
(`--arousal-source emo`). The audEERING alternative is still reachable but was
not used for any reported result, because its arousal output is too compressed
on drone material. The report has this wrong in one place.

The ridge heads ship as one `.npz` of coefficients rather than as the original
pickled scikit-learn objects. A `Ridge` prediction is `x @ coef_ + intercept_`,
so the arithmetic is identical — verified equal to `Ridge.predict` to the last
bit — while the array file neither pins a scikit-learn version nor asks anyone
to unpickle a downloaded object. `RidgeHead.predict` reproduces sklearn's dtype
behaviour deliberately, float32 in and float32 out, because the published labels
were computed that way and float64 would shift every one of them by about 2e-7.

**Training scripts do not write to `weights/`.** Their `--out-dir` defaults to
this section's `data/`. `weights/` holds the artefacts that reproduce the
reported results, and a short or exploratory fit landing there would be
indistinguishable from them.

## Reference numbers

From a full run on this project's build host, for checking a port or a re-run.

| Check | Value |
|---|---|
| Cycle consistency, closed-loop mapper, 60 clips | valence MAE **0.337**, arousal MAE **0.536** |
| Euclidean mean error | **0.678** |
| Arousal bias, requested against read back | **+0.380** |
| Resolution probe, paired d(resolution − static_major) | +0.048, +0.051, +0.007 for the three judges — all flat |
| Inter-judge agreement on drones | +0.245, −0.128, −0.089 |
| In-domain correlation, home-domain heads, n=150 | DEAM **+0.72**, Emo-Soundscapes **+0.85** |

The last row is the point of the whole supporting experiment. The same heads
that are flat and mutually uncorrelated on synthetic drones track human valence
on real audio in their own domains, so the degeneracy belongs to the domain and
not to the models.

## Reproducibility

`render_resolution_probe.py` seeds torch as well as numpy. The original did not,
and the filtered noise floor inside the DDSP synthesiser draws from
`torch.randn`, so its 75 clips were never reproducible: re-rendering moved every
sample by up to 0.019 on a signal peaking near 0.88. Seeded, two runs are now
byte-identical. The judge scores are unchanged by the fix — condition means
agree with the original clips to 0.002 and inter-judge correlations to 0.004 —
so the reported conclusion is unaffected.

## Smoke test

```bash
python smoke_test.py     # exit 0, no dataset, no network, no checkpoint needed
```
