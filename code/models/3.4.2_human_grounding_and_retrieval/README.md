# 3.4.2 Human Grounding and Retrieval

Nine listeners rated 150 drone clips and agreed on almost nothing. This section
is what you can build anyway, and it is what the deployed system actually uses.

Section 3.4.1 optimised against an automated judge and produced a model that
was not shipped. This one takes the opposite route: measure the disagreement
honestly, propagate ONE rater's labels across the preset bank, and retrieve from
that bank rather than generating. `inference/` is where the conductor's engine
lives.

## The argument, in order

| # | Command (from `train/`) | What it establishes |
|---|---|---|
| 1 | `python rating_agreement.py` | Krippendorff's alpha = **0.096** across nine raters. No consensus label exists to be learned. |
| 2 | `python propagate_labels_krr.py --bank-index <bank>/labeled_index.csv` | LOOCV picks the propagation model *before* it is used. Winner: RBF kernel ridge, alpha 1.0, gamma 0.003, **rho = 0.337**, sign accuracy **69%**. |
| 3 | `python propagate_labels_krr.py --bank-index <bank>/labeled_index.csv --propagate` | writes `labeled_index_human.csv` beside each bank index, adding `valence_human` |
| 4 | `python fit_control_ridge.py` | the control: a ridge trained directly on the 150 raw ratings, so "propagation is the bottleneck" can be told apart from "the task is hard" |
| 5 | `python fit_guard.py` | fits the boundary guard and freezes it to `weights/boundary_guard.json` |
| 6 | `python train_attention_retrieval.py --bank-index <bank>/labeled_index.csv` | trains the learned-attention retrieval engine |

Step 1 is not a preamble. A low alpha is what forces everything after it: with
no shared ordering to recover, a consensus label would be an average of
disagreement, so the section commits to one rater's coherent judgement and says
so. That is the "individual curated prior, not universal consensus" framing the
report uses.

## Why the human axis matters

The frozen judge's valence labels top out at **+0.244** across the entire
20,000-preset bank. Half the valence plane is unreachable: ask for positive
valence and retrieval silently returns the least-negative preset it has.
Propagating one rater's 150 ternary judgements through theta-space kernel ridge
unrolls the axis across **[−1.0, +0.572]**, and the navigable manifold appears.

`data/samples/` has the same two requests rendered on both axes, which is the
shortest way to hear the difference.

## What reaches the runtime

Unlike Section 3.4.1, this section ships inference code — it is what the
conductor loads.

| Module | Role |
|---|---|
| `inference/retrieval.py` | the deployed engine: warmth-filtered retrieval, GP uncertainty, judge/human label spaces |
| `inference/decoupled_engine.py` | the same decoupling without the warmth filter or GP |
| `inference/retrieval_engines.py` | hard k-NN, soft k-NN, GP and learned-attention engines, for the comparison in Section 3.4.1's cycle evaluation |
| `inference/guard.py` | the fitted boundary guard at runtime |

**Nothing in `inference/` imports the path helper or resolves a dataset root.**
Bank index files are passed in by the caller. That is what lets the conductor
start with every dataset root missing, and the smoke test asserts it.

Two design points worth knowing before reading the code:

- **Timbre is retrieved, never averaged.** Two harmonic distributions that each
  sound like an instrument average into one that sounds like neither. Scalars
  interpolate cleanly; spectra do not. `retrieve()` goes further and takes
  scalars and harmonics from the same anchor, because blending the scalars
  regresses them toward the bank mean and then glues that averaged register
  onto a foreign timbre. Smoothness is handled in time by the caller's waypoint
  crossfade, not by averaging in VA space.
- **The guard is a fence, not an oracle.** It answers "how far is this from
  anywhere a listener has rated". It exposes no valence prediction, and it must
  not be promoted to one.

## The guard ships without any rating data

`weights/boundary_guard.json` holds the rated *coordinates*, the fitted kernel,
the inverted kernel matrix, tau, and the safe set. It holds **no rating
values**, because the posterior variance depends only on where clips were rated
and never on what anyone scored. The runtime recomputes the posterior in numpy
and needs no scikit-learn; `fit_guard.py` refuses to write a guard whose
posterior the runtime does not reproduce (it agrees to about 3e-15).

The derived trigger threshold is **tau = 0.590**.

## Human ratings

`human_ratings/` ships the 713-row rating log pseudonymised to `R01`…`R09`, the
150-clip pool metadata, and the MERT embeddings of the rated clips. The rated
audio itself is not distributed.

The map from real identities to `R01`…`R09` was recovered positionally against
the report's own pseudonymised copy, not guessed, and it is not in this
repository. Rater **R03** is the label source for propagation and for the guard:
one rater by design, per step 1.

Clip paths are repository-relative, and no absolute path or personal name
survives anywhere in the shipped files.

## Datasets and pretrained models

| Resource | Used for | Licence | Where to get it |
|---|---|---|---|
| **MERT-v1-95M** | embeddings for the control ridge and the propagation comparison | CC BY-NC 4.0 | <https://huggingface.co/m-a-p/MERT-v1-95M> |

Only `extract_pool_mert.py` needs MERT, and only to regenerate embeddings that
already ship. Everything else in this section runs from the repository plus a
labelled preset bank, which Sections 3.3 and 3.4.1 produce.

## What ships in `weights/`

| File | What it is |
|---|---|
| `boundary_guard.json` | the fitted guard, rating-free |
| `attn_retrieval.pt` | the learned-attention retrieval engine |
| `control_ridge_valence.npz` | the human-grounded control ridge, as coefficient arrays |

The control ridge ships as an `.npz` rather than a pickled pipeline, for the
same reasons as Section 3.4.1's judge heads: a StandardScaler-plus-Ridge is four
arrays, and reconstructing from them pins no library version and asks nobody to
unpickle a downloaded object. Verified bit-identical to the pipeline on the
shipped float64 features.

Training scripts default their `--out-dir` to this section's `data/`, never
`weights/`.

## Reference numbers

| Check | Value |
|---|---|
| Rating rows / effective ratings | 713 / 712 |
| Krippendorff's alpha, 9 raters, ordinal | **0.096** |
| Stratified alpha: consensus-extreme (n=59) / middle (n=91) | 0.366 / −0.135 |
| Unanimity on clips with ≥3 raters | 9/124 |
| theta-KRR LOOCV winner | RBF KRR, alpha 1.0, gamma 0.003 |
| theta-KRR LOOCV MAE / rho / sign accuracy | 0.566 / **0.337** / **0.69** |
| Predict-zero and predict-mean MAE baselines | 0.567 / 0.600 |
| Valence ceiling, judge axis / human axis | **+0.244** / **+0.572** |
| Boundary guard tau | **0.590** |

The stratified alpha is the one to read twice. If the raters shared nothing,
both strata would be flat; instead the consensus-extreme clips reach 0.366 while
the middle is negative. There is a narrow real signal at the poles, buried in
rater noise across the middle — range restriction, not absence of signal.

## Smoke test

```bash
python smoke_test.py     # exit 0, no dataset, no network, no bank needed
```
