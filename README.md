# COMP0158 Supplementary Material

Code, data and audio for *Emotion-Driven DDSP Drone Synthesis*.

Everything here is organised by the report section it belongs to, so a
directory name tells you which part of the thesis it implements. The pipeline
runs in report order, and every stage can be run on its own.

| | |
|---|---|
| Migration plan and file-by-file mapping | [`code/README.md`](code/README.md) |
| Environments | [`env/`](env/) |
| Rendered audio referenced by the report | [`audio/`](audio/) |
| Interactive figures | [`figures_3d/`](figures_3d/) |

`LICENSE` covers the code. `ATTRIBUTION.md` governs the third-party audio under
`code/conductor/assets/`, which carries different terms. See
[`code/README.md`](code/README.md) §0.3.

---

## 1. Build an environment

Two environment files, one per machine the project was developed on. Both pin
Python 3.12 and PyTorch 2.6.0.

```bash
# Linux with an NVIDIA GPU (CUDA 11.8). This is the one the results came from.
export ENV_PREFIX=$HOME/envs
micromamba create -f env/env_linux_nvidia.yml -p $ENV_PREFIX/env_linux_nvidia
micromamba activate $ENV_PREFIX/env_linux_nvidia

# Windows with an Intel Arc GPU. torch.xpu is built into PyTorch 2.5+,
# so no IPEX package is needed.
micromamba create -f env/env_windows_arc.yml -n env_thesis_arc
micromamba activate env_thesis_arc
```

CPU-only works for everything except training. Rendering, inference and the
conductor all run on CPU; `common/device.py` picks CUDA, then Intel XPU, then
CPU, and prints which it chose, so a silent CPU fallback is visible in the log.

Check the environment:

```bash
python -c "import torch, soundfile, scipy, pandas; \
           print(torch.__version__, torch.cuda.is_available())"
```

## 2. Point the scripts at your data

No dataset is redistributed here. Every script resolves each dataset root in
this order: the command-line flag, then the environment variable, then the path
on the cluster host the project was built on. That last one is recorded for
reproducibility and will not exist on your machine.

```bash
export DRONE_NSYNTH=/path/to/nsynth-train      # examples.json + audio/
export DRONE_DEAM=/path/to/deam
export DRONE_EMO=/path/to/emo_soundscapes
export DRONE_ESSEN=/path/to/essen
export DRONE_ECHOTHIEF=/path/to/echothief
export DRONE_DATA=/path/to/generated           # where this pipeline writes
```

Or per run: `python build_nsynth_prior.py --nsynth-root /path/to/nsynth-train`.

| Dataset | Section | Licence | Where to get it |
|---|---|---|---|
| NSynth | [3.3](code/models/3.3_drone_synthesis_and_nsynth_prior) | CC BY 4.0 | <https://magenta.tensorflow.org/datasets/nsynth> |
| DEAM | [3.4.1](code/models/3.4.1_surrogate_guided_optimisation) | research use, registration | <https://cvml.unige.ch/databases/DEAM/> |
| Emo-Soundscapes | [3.4.1](code/models/3.4.1_surrogate_guided_optimisation) | academic, non-commercial | <https://metacreation.net/emo-soundscapes/> |
| MERT-v1-95M | [3.4.1](code/models/3.4.1_surrogate_guided_optimisation) | CC BY-NC 4.0 | <https://huggingface.co/m-a-p/MERT-v1-95M> |
| audEERING wav2vec 2.0 | [3.4.1](code/models/3.4.1_surrogate_guided_optimisation) | CC BY-NC-SA 4.0 | <https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim> |
| Ambient transition corpus | [3.5](code/models/3.5_transition_dynamics_and_scheduling), [3.6](code/models/3.6_differentiable_reverberation) | commercial; **not redistributed**, measurements only | <https://archive.org/details/audio_music>, <https://soundcloud.com> |
| EchoThief impulse responses | [3.6](code/models/3.6_differentiable_reverberation) | free, credit requested | <http://www.echothief.com/> |
| Essen Folksong Collection | [3.7](code/models/3.7_melody_generation) | free for research | <http://www.esac-data.org/>, or `music21.corpus` |

No dataset is redistributed in this repository. The ambient corpus is the one
that cannot be: §3.5 and §3.6 ship fitted measurements, dwell distributions,
transition counts and decay estimates, with the audio referenced by hash only.

A missing dataset fails with its name and where to download it, not a
`FileNotFoundError`. Full table in [`code/common/paths.py`](code/common/paths.py).

## 3. Check it works before you commit to a long run

Each model directory has a `smoke_test.py` that needs no dataset and no
checkpoint. It builds synthetic inputs, runs the real code, and checks the
outputs. Exit code 0 means every check passed.

```bash
python code/models/3.3_drone_synthesis_and_nsynth_prior/smoke_test.py
```

---

## 4. What is where

Every model directory holds the same four subdirectories:

| | |
|---|---|
| `train/` | fitting scripts. Long-running, need a dataset. |
| `weights/` | fitted artefacts this section produced. |
| `inference/` | what the runtime imports. No dataset, no sampling, no training. |
| `human_ratings/` | scrubbed rating data this section used, with a README explaining the study. |

An empty `inference/` or `human_ratings/` is deliberate and its README says
why. Section 3.4.1 has no `inference/` because nothing from the failed
automated loop reached the running system, which is the finding of §4.1.

| Folder | Report | What it is | Train | Inference |
|---|---|---|---|---|
| [`common/`](code/common) | — | Shared imports only: the DDSP synth, device selection, dataset paths. | — | `import ddsp_synth` |
| [`models/3.3_drone_synthesis_and_nsynth_prior/`](code/models/3.3_drone_synthesis_and_nsynth_prior) | §3.3 | Measures an empirical timbre prior from NSynth bowed strings, then samples and renders the 20,000-preset bank. | `python train/build_nsynth_prior.py --metadata-only`<br>`python train/analyse_nsynth_timbre.py --audio-dir $DRONE_NSYNTH/audio`<br>`python train/generate_preset_bank.py --n 10000 --duration 10` | `python inference/theta_render.py --out demo.wav --f0 55 --duration 6` |
| [`models/3.4.1_surrogate_guided_optimisation/`](code/models/3.4.1_surrogate_guided_optimisation) | §3.4.1 | Labels the bank with frozen MERT and audEERING heads, distils a 250k-parameter CNN proxy, and optimises a mapper against it. This is the loop that reward-hacked. | `python train/label_dataset.py --data-dir <bank>`<br>`python train/train_judge_proxy.py --data-dir <bank>`<br>`python train/train_closed_loop.py --bank-dirs <bank>` | none by design, see §4.1 |
| [`models/3.4.2_human_grounding_and_retrieval/`](code/models/3.4.2_human_grounding_and_retrieval) | §3.4.2 | Propagates 150 human seed ratings across the bank with θ-KRR, then serves 1-NN retrieval. This is what the conductor actually navigates. | `python train/rating_agreement.py`<br>`python train/propagate_labels_krr.py --bank-index <bank>/labeled_index.csv`<br>`python train/fit_guard.py` | `python -c "import sys;sys.path.insert(0,'inference');import retrieval_engines as r;b=r.build_bank(['<bank>/labeled_index.csv']);print(r.make_engine('soft',b).blend(-0.3,0.4))"` |
| [`models/3.5_transition_dynamics_and_scheduling/`](code/models/3.5_transition_dynamics_and_scheduling) | §3.5 | Semi-Markov dwell scheduler over five transition archetypes, re-ranked by a Bradley–Terry preference GP. | `python train/fit_preference_gp.py` | `python inference/scheduler.py --preset overlay --trans-source corpus_dwell` |
| [`models/3.6_differentiable_reverberation/`](code/models/3.6_differentiable_reverberation) | §3.6 | Fits three scalars (decay, damping, wet gain) per impulse response by gradient descent under a multi-scale STFT loss. | `python train/fit_from_irs.py --echothief-root $DRONE_ECHOTHIEF` | `python inference/reverb_bank.py --list`<br>`python inference/reverb_bank.py --selftest` |
| [`models/3.7_melody_generation/`](code/models/3.7_melody_generation) | §3.7 | Causal micro-transformer over Essen folk phrases, plus an order-2 Markov baseline. Three trained sizes ship. | `python train/build_essen_dataset.py`<br>`python train/train_transformer.py --n-layers 6 --d-model 128` | `python inference/melody_transformer.py --demo out.wav --valence 0.2 --arousal -0.4` |
| [`models/3.9_human_evaluation_and_protocols/`](code/models/3.9_human_evaluation_and_protocols) | §3.9 | The listening studies: long-track recency and component preference. Merge and hygiene scripts, plus the rating data. | *not yet migrated; the rating data ships* | — |
| [`conductor/`](code/conductor) | §3.8 | The deployed runtime. Inference only: it starts with every dataset root missing. | — | *not yet migrated* |

Sections 3.1 and 3.2 ship no code. §3.1 is the architecture overview and §3.2
is the developmental explorations.

---

## 5. Rebuild the whole pipeline

Report order. Steps 1 to 3 need NSynth and a GPU; the rest are cheap.

```bash
cd code/models

# 3.3  timbre prior, then the preset bank
python 3.3_drone_synthesis_and_nsynth_prior/train/build_nsynth_prior.py --metadata-only
python 3.3_drone_synthesis_and_nsynth_prior/train/analyse_nsynth_timbre.py \
       --audio-dir $DRONE_NSYNTH/audio --out-dir 3.3_*/weights/timbre_prior
python 3.3_drone_synthesis_and_nsynth_prior/train/generate_preset_bank.py \
       --n 10000 --duration 10 --out-dir $DRONE_DATA/ddsp_chord10k

# 3.4.1  label it, distil the proxy, run the closed loop
# the frozen judge configuration is the default; no flags needed for it
python 3.4.1_surrogate_guided_optimisation/train/label_dataset.py \
       --data-dir $DRONE_DATA/ddsp_chord10k
python 3.4.1_surrogate_guided_optimisation/train/train_judge_proxy.py \
       --data-dir $DRONE_DATA/ddsp_chord10k
python 3.4.1_surrogate_guided_optimisation/train/train_closed_loop.py \
       --bank-dirs $DRONE_DATA/ddsp_chord10k

# 3.4.2  human grounding, which is what actually ships
python 3.4.2_human_grounding_and_retrieval/train/propagate_labels_krr.py \
       --bank-index $DRONE_DATA/ddsp_chord10k/labeled_index.csv --propagate
python 3.4.2_human_grounding_and_retrieval/train/fit_guard.py

# 3.5 to 3.7  scheduler, reverb, melody
python 3.5_transition_dynamics_and_scheduling/train/fit_preference_gp.py
python 3.6_differentiable_reverberation/train/fit_from_irs.py --echothief-root $DRONE_ECHOTHIEF
python 3.7_melody_generation/train/train_transformer.py --n-layers 6 --d-model 128
```

Nothing above is needed to run the conductor. Every fitted artefact it uses
already ships in `code/conductor/weights/`.

## 6. Run the conductor

```bash
cd code/conductor
pip install -r requirements.txt
python app.py                       # opens a Gradio UI on http://localhost:7860
```

It starts with no dataset present. That is the acceptance test for the whole
migration: the conductor must run with every dataset root pointing at
`/nonexistent`.

The interface is a 2D valence–arousal pad. Dragging the cursor retrieves the
nearest human-rated preset, updates the six-voice drone, and lets the
semi-Markov scheduler decide when chords advance. Melody, beds, crackle and
stereo distance are off by default.

Audio arrives in chunks rather than instantly. The synthesis core evaluates a
parameter block in about 3.4 ms, but the browser path buffers, so a change you
make takes a few seconds to become audible. Section 4.5.3 measures this.

## 7. Reproducibility notes

- **Seed both RNGs.** `--seed` covers theta sampling and the f0 wander. The
  filtered-noise floor is drawn by `torch.randn` inside the synth, which takes
  no generator, so `generate_preset_bank.py` also calls `torch.manual_seed`.
  Any other caller of `render_theta` must do the same or renders will differ
  audibly between runs. Found by the §3.3 smoke test.
- **The frozen judge configuration is not the script default.** Arousal comes
  from MERT to the Emo-Soundscapes ridge, selected with
  `--arousal-ridge emo --arousal-scale pm1`. The audEERING default in the
  argument parser was never used for any result in the report.
- **Rating data is pseudonymised.** Raters are `R01`…`R09` under a mapping kept
  outside this repository, so the copies here and in the report agree. See
  [`code/README.md`](code/README.md) §0.1 and §0.1b.
- **Two rating studies are fragmented** across web-app restarts, and merging
  them naively double-counts. Use the shipped merge scripts. The traps are
  documented in `code/README.md` §0.1b and in
  [`3.9_.../human_ratings/README.md`](code/models/3.9_human_evaluation_and_protocols/human_ratings/README.md).
