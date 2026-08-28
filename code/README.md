# COMP0158 Supplementary Code — Migration Plan

Working document. Plan first, progress log at the bottom. Nothing has been
migrated yet.

Source tree: `/cs/student/msc/dsml/2023/myeung/THESIS/Claude/ucl_dsml_thesis_claude`
(6.2 GB, 411 `.py`, 4,538 `.wav`, research layout — not shippable as-is).

The goal is two things that must not depend on each other:

- **`conductor/`** — runs on a clean machine with only Python packages and the
  files in `conductor/weights/`. No dataset roots, no `paths.py`, no imports
  reaching into `training/`.
- **`training/`** — reproduces every weight in `conductor/weights/`, reading
  datasets from a `--data-root` argument that defaults to this Linux box.

---

## 0. Blockers — resolve before anything is published

These are not migration tasks. They are decisions that change what gets copied.

### 0.1 Participant names are in the shipped payload

`FINAL_CONDUCTOR/app/` currently carries three rating CSVs with a `rater`
column holding what look like real first names and surname initials:

| file | rows | raters | dates |
|---|---|---|---|
| `al_pool/valence_ratings.csv` | 713 | 9 | 2026-07-03 → 07-06 |
| `al_pool/arc_ratings.csv` | 120 | 8 | 2026-07-05 → 07-08 |
| `al_pool/arc_valence_ratings.csv` | 199 | 4 | 2026-07-05 → 07-18 |

Values include `Kim Snoyman`, `Nick R`, `Olly T`, `Ross P`, `Stoyan`, `Tim`,
`Saad`, `Ramona`, plus per-response timestamps. Publishing these to a public
repo publishes identifiable human-subject data. Rater IDs also appear in code:
`s04a_report_figs.py:199` hardcodes `SENSITIVITY_EXCLUDE = {"sausagedog420"}`.

**Why they are there at all.** The conductor re-fits three models from raw
ratings on every boot rather than loading fitted parameters:

| consumer | reads | what it fits |
|---|---|---|
| `s06_boundary_guard.py:226` | `valence_ratings.csv`, `pool_meta.json` | `HybridBoundaryGuard` |
| `s09_arc_fusion.py:60-70` | `arc_ratings.csv`, `arc_valence_ratings.csv`, `texture_ratings.csv` | Bradley–Terry Laplace preference GP |
| `melodic_drone.py:288` | `pool_meta.json`, `valence_ratings.csv` | melody anchor lookup |
| `s10_overlay_explorer.py:84`, `s11_arrangement_scheduler.py:276` | `texture_ratings.csv` | dark-clip veto list for beds |

**Plan.** The conductor does not need the ratings, it needs what fitting them
produces. Pre-compute all four at build time into `conductor/weights/`:
`boundary_guard.json`, `preference_gp.npz` (kernel hyperparameters and the
Laplace posterior), `melody_anchor.json`, `bed_veto.json`. Then no rating CSV
ships at all, boot is faster, and the shipped conductor is frozen and
reproducible instead of silently changing if a CSV is edited.

The CSVs still travel with `training/`, so pseudonymise them there: replace
every rater value with a stable code (`R01`…`R09`) via a one-way mapping kept
**outside** this repo, and round timestamps to the day. The analysis only ever
groups by rater, never reads the name, so `s03_rating_irr.py` and the
intra-rater consistency filter keep working unchanged.

Also to reconcile: `appendix_datasets.tex` describes the valence seed set as
150 presets rated by the primary author (N=1), but `valence_ratings.csv` holds
713 rows from 9 raters. Probably the N=1 seed is a subset of this file, but the
appendix and the shipped data should agree before either is published.

### 0.2 Artist names are in the shipped payload

`app/SideProjects/reverb/logs/album_reverb_bank.json` is loaded at runtime and
its keys name five commercial artists, with local source paths:

```json
"sources": ["/…/data/SOTL", "/…/data/WillianBasinski", "/…/data/Loscil",
            "/…/data/Celer", "/…/data/KyleBobbyDunn"],
"category_summary": {"Celer": {…}, "KyleBobbyDunn": {…}, …}
```

Same names appear in comments in `s07a_corpus_stats.py`, `s07_arc_pool.py`,
`s07b_transition_typology.py`, `s13_corpus_hsmm_fit.py`,
`s11_arrangement_scheduler.py` and `reverb_bank.py`, and in
`conductor_common.py:72` which expands the tag `sotl` to a full album title.

`appendix_datasets.tex` states that artist metadata and source identifiers were
scrubbed from the project manifests. Shipping this alongside the thesis
disproves that claim with one `grep`.

**Plan:** re-key the reverb bank on the acoustic descriptors already used in the
report (`dark_long`, `bright_short`, …), rename the file to
`reverb_bank_measured.json`, drop `sources`, and sweep the comments. The fitted
numbers stay — they are measurements, which `ATTRIBUTION.md` already frames
correctly.

### 0.3 Redistribution rights for the audio

86 bed `.wav` files (ESC-50, CC BY-NC; Emo-Soundscapes, per-clip Freesound
terms) and 3 EchoThief IRs sit in the payload. `build_space.py` already refuses
to publish for this reason and says so in its docstring. A public GitHub repo
is the same decision.

**Decided (2026-08-28): commit the audio, with attribution.** The collections
carry academic/open terms that permit redistribution with credit, and
`ATTRIBUTION.md` already names each source correctly. No fetch script.

One consequence to get right. ESC-50 is CC BY-NC, so the repo must not carry a
single permissive licence that implicitly grants commercial use of audio we
cannot license. Split it: a `LICENSE` covering the **code only**, and
`ATTRIBUTION.md` governing `conductor/assets/`, with a line at the top of this
README saying which applies to what.

---

## 1. Target layout

Directories marked `+` need creating; the rest already exist.

```
code/
  README.md                     this file
  conductor/
    app.py                    + Gradio entry point
    fetch_assets.py           + downloads beds + IRs (see §0.3)
    requirements.txt          +
    engine/                   + synthesis and retrieval (no training imports)
    runtime/                  + scheduler, effects, melody playback
    UI/                         skins, faders, visuals, web audio, ring player
    weights/                    every fitted artefact (see §3)
    assets/                   + fetched audio, git-ignored
  training/
    DDSP/                       §3.3 synthesis engine + preset bank
    Transitions/                §3.4 — INFERENCE ONLY, see §2.4
    Reverb/                     §3.5
    Melody/                     §3.6
    SideProjects/               stereo, crackle, soundscape beds
    common/                   + paths.py, dataset loaders, shared DSP
```

Grouping follows the methodology chapter as it now stands: 3.3 Parametric Drone
Synthesis, 3.4 Transition Dynamics, 3.5 Reverberation Learning, 3.6 Melodic Lead
Generation, 3.7 Runtime Conductor.

---

## 2. What moves where

### 2.1 The authoritative inference set already exists

`Phase2/02_Conductor/s06_pack_demo.py` is a working packer. It owns the file
inventory, rewrites IR paths, writes `MANIFEST.json` with a sha256 per file, and
has a `verify()` that runs the packed stack with env vars pointed at
non-existent directories — the only thing that actually proves self-containment.
Its output, `FINAL_CONDUCTOR/app/`, is 143 files / 75 MB and runs standalone.

**Do not re-derive the inference set by hand.** Start from that manifest, then
apply the splits below. Keep `verify()` as the acceptance test for
`conductor/` — a migration that breaks it is not finished.

### 2.2 DDSP → `training/DDSP/` and `conductor/engine/`

| source | destination | new name |
|---|---|---|
| `ddsp_synth.py` | `conductor/engine/` | `ddsp_synth.py` |
| `Phase2/Gemini/step01_gp_softknn_engine.py` | `conductor/engine/` | `retrieval.py` |
| `Phase2/Gemini/step02_arranger.py` | `conductor/engine/` | `arranger.py` |
| `Phase2/01_MIDI_DDSP/s06_boundary_guard.py` | split, see §3.3 | `guard.py` / `fit_guard.py` |
| `s01_select_nsynth_strings.py`, `s03_generate_ddsp_dataset.py` | `training/DDSP/` | `build_nsynth_prior.py`, `generate_preset_bank.py` |
| `s04_relabel_bank.py`, `s14_control_ridge.py` | `training/DDSP/` | `propagate_labels_krr.py`, `fit_control_ridge.py` |
| `s01_gp_active_learning.py`, `s02_regen_pool.py` | `training/DDSP/` | `active_learning_pool.py`, `regenerate_pool.py` |
| `s03_rating_irr.py`, `s03_compute_irr.py` | `training/DDSP/` | `rating_agreement.py` |
| `data/ddsp_*10k/labeled_index.csv` | `conductor/weights/` | unchanged (17 MB, two files) |

`step02_arranger.py:25` does `import paths` and never uses it. Dead import —
delete it and `paths.py` leaves the conductor payload entirely.

### 2.3 Reverb → `training/Reverb/` and `conductor/runtime/`

| source | destination | new name |
|---|---|---|
| `SideProjects/reverb/learnable_reverb.py` | split | `conductor/runtime/reverb.py` (apply) + `training/Reverb/model.py` (fit) |
| `SideProjects/reverb/reverb_bank.py` | `conductor/runtime/` | `reverb_bank.py`, de-identified per §0.2 |
| `SideProjects/reverb/learn_from_irs.py` | `training/Reverb/` | `fit_from_irs.py` |
| `logs/ir_reverb_bank.json` | `conductor/weights/` | unchanged |
| `logs/album_reverb_bank.json` | `conductor/weights/` | `reverb_bank_measured.json`, re-keyed |
| `irs/*.wav` (3) | fetched, not committed | — |

`reverb_bank.py:149` imports `load_ir` from `learn_from_irs` — that one function
moves to the runtime side so the conductor stops importing the fitting script.

### 2.4 Transitions → `training/Transitions/` (inference only)

Per your note: no training files here, since the fit runs on the commercial
corpus. What ships:

| source | destination | new name |
|---|---|---|
| `Phase2/02_Conductor/s11_arrangement_scheduler.py` | `conductor/runtime/` | `scheduler.py` |
| `Phase2/02_Conductor/s17_coherence_reranker.py` | `conductor/runtime/` | `coherence_reranker.py` |
| `Phase2/01_MIDI_DDSP/figs/corpus_hsmm.json` | `conductor/weights/` | `hsmm_transitions.json` |
| `Phase2/02_Conductor/logs/arc_types.json` | `conductor/weights/` | unchanged |

`training/Transitions/` gets a `README.md` only: what the fit did
(`s13_corpus_hsmm_fit.py`, Foote novelty + K-means, K=5), what it consumed, why
the script is not included, and the sha256 of the resulting
`hsmm_transitions.json` so the shipped weight is traceable.

`s11_arrangement_scheduler.py:115` imports `s13_corpus_hsmm_fit` for
"corpus-fit dwell/trans, opt-in". The module reads nothing at import time — only
constants and loaders — so the fix is to lift those constants into
`scheduler.py` and drop the import.

**Decided (2026-08-28): the Bradley–Terry preference GP stays in the
conductor.** Chapter 3 no longer describes it, and that is deliberate — the
shipped system may do more than the thesis writes up. `s01_arc_policy.py` moves
to `conductor/runtime/arc_policy.py` and `s09_arc_fusion.py` splits: the fitted
posterior is pre-computed into `conductor/weights/preference_gp.npz` (§0.1) and
the fitting code goes to `training/Transitions/fit_preference_gp.py`, which is
the one training script that directory does keep, since it runs on the arc
ratings rather than the commercial corpus.

`s08_arc_rating_app.py` is a data-collection UI, not runtime. It goes to
`training/Transitions/` and leaves the conductor.

Do not let a later tidy-up "fix" the code/thesis mismatch by deleting this —
add a note in `conductor/README.md` saying the runtime intentionally carries
components the report does not cover.

### 2.5 Melody → `training/Melody/` and `conductor/runtime/`

| source | destination | new name |
|---|---|---|
| `SideProjects/melodic_drone/GEMINI/runtime_melodic_drone.py` | `conductor/runtime/` | `melody_transformer.py` |
| `SideProjects/melodic_drone/melodic_drone.py` | `conductor/runtime/` | `melody_markov.py` |
| `GEMINI/train_melodic_transformer.py` | `training/Melody/` | `train_transformer.py` |
| `GEMINI/dataset_builder.py` | `training/Melody/` | `build_essen_dataset.py` |
| `GEMINI/benchmark_and_render.py` | `training/Melody/` | `benchmark_configs.py` |
| `checkpoints/melodic_transformer_cpu.pt` + `.json` | `conductor/weights/` | `melodic_transformer_L6_d128.pt` — the default, see below |
| `checkpoints/melodic_transformer_L3_d128.pt` + `.json` | `conductor/weights/` | unchanged (2.4 MB), offered in the dropdown |
| Markov tables (in `melodic_drone.py`) | `conductor/weights/` | extract to `markov_order2.json` |

**Both checkpoints ship, and the filename is misleading.**
`melodic_transformer_cpu.pt` is not a CPU build of the compact model — its
sidecar reports `n_layers: 6, param_count: 1207203`. It is the L6/d128 model,
shipped under an alias, and `s02_conductor_app.py:1071` sets
`PREFERRED_CKPT = "L6_d128"`, so the conductor opens on it. That was a
deliberate porting decision: L6 sounded best and ran acceptably on CPU. Rename
the file to say what it is.

**This contradicts the thesis** — `melodic_drone_and_sequence_generation.tex:52`
says "By Occam's razor, the 3-layer configuration was selected for the live
system" and Table `tab:melodic_model_benchmark` bolds the 3-layer row as
deployed. The code is right and the report is wrong; fixing it is a report task,
logged here so it is not lost. See §7.

`train_melodic_transformer.py` is currently packed into the runtime but
`runtime_melodic_drone.py` does not import it. Verify, then drop it from the
conductor.

### 2.6 SideProjects → `training/SideProjects/` and `conductor/runtime/`

`stereo/stereo_pad.py`, `stereo/distance.py`, `crackle/crackle.py` and
`soundscape_synth.py` are pure runtime DSP with no training step. They move to
`conductor/runtime/` unchanged. `training/SideProjects/` keeps the bed-bank
builder (`s15_bed_bank.py`) and the rating apps, which are data-collection
tools, not runtime.

### 2.7 Conductor UI

`s20_skins.py`, `s21_faders.py`, `s22_visuals.py`, `s19_webaudio.py`,
`s18_ring_player.py`, `s10_overlay_explorer.py` → `conductor/UI/`, renamed to
`skins.py`, `faders.py`, `visuals.py`, `webaudio.py`, `ring_player.py`,
`overlay_mixer.py`. `s02_conductor_app.py` becomes `conductor/app.py`.

---

## 3. Breaking the training → inference dependencies

This is the real engineering work. The conductor currently imports eight
training-side modules. Each needs the same treatment: extract what the runtime
actually uses, leave the rest behind.

### 3.1 `s07_arc_pool.py`

Conductor uses `apply_chord`, `HOLD_S`, `RENDER_KW`, `aw_rms`, `AW_TARGET` —
constants and two pure functions out of a 21 KB pool builder. Extract into
`conductor/engine/render_params.py`. The pool builder stays in `training/DDSP/`.

### 3.2 `s09_arc_fusion.py` / `s08_arc_rating_app.py` / `s01_arc_policy.py`

The GP stays (§2.4), so the split is: `s01_arc_policy.py` is runtime and moves
as-is; `s09_arc_fusion.py` is a fit and moves to `training/Transitions/`, with
its posterior pre-computed into `conductor/weights/preference_gp.npz`;
`s08_arc_rating_app.py` is a rating UI and leaves the conductor entirely.

`s01_arc_policy.py` must be changed to load the pre-fit posterior instead of
calling into `s09`. That is the only code change of substance here.

### 3.3 `s06_boundary_guard.py`

`HybridBoundaryGuard` is **fitted at boot** from `valence_ratings.csv` and
`pool_meta.json`. That is why participant data is in the payload (§0.1).

Split into `training/DDSP/fit_guard.py`, which does the fit and writes
`conductor/weights/boundary_guard.json`, and `conductor/engine/guard.py`, which
loads that file. Removes 440 KB of rating CSVs from the conductor and the
privacy problem with them.

Same treatment for the melody anchor (`melodic_drone.py:288`) and the bed veto
list (`s10_overlay_explorer.py:84`), which are small aggregations over the same
CSVs. Together with §3.2 these four pre-fits are what let the conductor ship
with no ratings data at all.

### 3.4 `learnable_reverb.py` / `learn_from_irs.py`

Covered in §2.3. `apply_reverb_np` and `load_ir` are runtime; the fitting loop
and the STFT loss are training.

### 3.5 `s13_corpus_hsmm_fit.py`

Covered in §2.4. Lift the constants, drop the import.

### 3.6 Model classes living inside training scripts

Where a checkpoint is loaded by re-importing the script that trained it, move
the `nn.Module` definition into a `model.py` that both sides import. Applies to
the melodic transformer and the reverb model. Check for others while migrating.

---

## 4. Dataset paths

`training/common/paths.py` replaces the current `paths.py`, which hardcodes a
Windows laptop and this cluster:

```python
DEFAULTS = {
    "deam":            "/cs/student/msc/dsml/2023/myeung/THESIS_MYDIR/data/deam",
    "emo_soundscapes": "…/data/emo_soundscapes",
    "nsynth":          "…/data/nsynth-train",
    "essen":           "…/data/essen",          # via music21
    "echothief":       "…/data/echothief",
}
```

Every training script takes `--data-root` (or a per-dataset flag), falls back to
the env var, then to the default above. Each script fails with a message naming
the dataset and where to get it, rather than a `FileNotFoundError`.

`conductor/` imports none of this. The acceptance test is `s06_pack_demo.py`'s
`verify()`: run the conductor with `JAMAI_DATA` and friends pointed at
`/nonexistent` and it must still start and render.

---

## 5. Order of work

1. **De-identify** the rating CSVs and the reverb bank (§0.1, §0.2). Do this
   first — everything downstream copies these files.
2. **Scaffold** the target directories and `training/common/paths.py`.
3. **Write the four pre-fit scripts** (guard, preference GP, melody anchor, bed
   veto) and generate `conductor/weights/`. This is what lets the conductor
   ship with no ratings data (§3.2, §3.3).
4. **Migrate the conductor** from the existing manifest: copy, rename, rewire
   imports, point the four consumers at the pre-fit weights, break the
   remaining training dependencies (§3).
5. **Run `verify()`** with dataset roots pointed at `/nonexistent`. Not done
   until this passes.
6. **Migrate training**, section by section, in methodology order: DDSP →
   Reverb → Melody → SideProjects, plus `fit_preference_gp.py` under
   Transitions.
7. **Licence and attribution split** (§0.3), then confirm a clean clone plus
   `pip install -r requirements.txt` gives a working conductor.
8. **Packaging extras**: `.devcontainer/devcontainer.json` with
   `forwardPorts: [7860]` so it opens in Codespaces, a Dockerfile, and a
   GitHub Actions job that runs `verify()` on push.

Steps 1–5 make the conductor shippable. Steps 6–8 make the repo submittable.

---

## 6. Decisions taken (2026-08-28)

1. **Preference GP** — stays in the conductor. The shipped system may carry
   more than the thesis describes. (§2.4)
2. **Rater data** — the conductor ships none of it; four boot-time fits become
   pre-computed weights. The CSVs travel with `training/`, pseudonymised
   `R01`…`R09`. (§0.1, §3.2, §3.3)
3. **Bed audio** — committed, with attribution. Code licence and asset
   attribution kept separate. (§0.3)
4. **Checkpoints** — both ship. L6/d128 is the default by design; the file
   currently named `melodic_transformer_cpu.pt` gets renamed to match its
   architecture. (§2.5)

---

## 7. Report changes this migration implies

Findings from the code that Chapter 3 currently contradicts. Not code tasks —
logged here so they reach the report.

1. **Deployed melody model.** The conductor runs L6/d128 because it sounded
   best and ran acceptably on CPU. `melodic_drone_and_sequence_generation.tex:52`
   claims the 3-layer model was selected, on an Occam's razor argument. Replace
   with what actually happened: three configurations were trained, the two
   128-wide ones were within 0.0008 nats so validation loss could not separate
   them, and the choice was made by listening. Move the bold row in
   `tab:melodic_model_benchmark` to Deep-Transf. and check its size and latency
   figures against the shipped checkpoint.
2. **Rater count.** `appendix_datasets.tex` describes the valence seed set as
   N=1 (primary author); `valence_ratings.csv` holds 713 rows from 9 raters.
   Reconcile before either is published.
3. **Artist scrubbing.** `appendix_datasets.tex` claims artist metadata was
   scrubbed from the project manifests. True only once §0.2 is done.

---

## Progress log

| date | step | status |
|---|---|---|
| 2026-08-28 | Plan written | done |
| 2026-08-28 | Decisions 1–4 taken (§6) | done |
| | 1. De-identify | not started |
| | 2. Scaffold | not started |
| | 3. Four pre-fit scripts → `weights/` | not started |
| | 4. Migrate conductor | not started |
| | 5. `verify()` passes | not started |
| | 6. Migrate training | not started |
| | 7. Licence + attribution split | not started |
| | 8. Packaging extras | not started |
