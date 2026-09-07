# COMP0158 Supplementary Code — Migration Plan

Target: this directory,
`THESIS_MYDIR/COMP0158_SupplimentaryMaterial/code`.

Written 2026-08-28. **Layout and four decisions revised 2026-09-06** (see §6).
Working rule throughout: **propose, get approval, then apply.**

`LICENSE` covers the code. `ATTRIBUTION.md` governs `conductor/assets/`, which
carries third-party audio under different terms. See §0.3.

---

## Scope rule

**Ship only what the report declares, plus what those files need to run.**

The code tree holds 421 Python files across `experiment/`, `adhoc/`, `Phase3/`,
`SideProjects/`, `step1_conductor/`, `FINAL_CONDUCTOR/` and three copies of the
conductor. Most is exploration and does not belong in a submission.

The report names 24 `.py` files. That plus their imports is the whole job.
Two of the 24 **do not exist**: `train_cvae.py` and `s07_evaluate_cvae_human.py`
appear only in LaTeX `% Code ref:` comments, never in visible text, so they do
not affect the PDF. The real files are `s06_train_inverse_cvae.py` and
`s07_evaluate_cycle_consistency.py`.

---

## 0. Blockers — resolve before anything is published

These are not migration tasks. They change what gets copied.

### 0.1 Participant names are in the payload

Rating CSVs carry a `rater` column holding real first names and surname
initials. Confirmed 2026-09-06 across **eight duplicate copies** of the tree
(`Phase2/`, `Phase3/demo_snapshot*` x3, `FINAL_CONDUCTOR/{app,hf_space,space_build}`):

| file | rows | raters | values seen |
|---|---|---|---|
| `al_pool/valence_ratings.csv` | 713 | 9 | a full first-and-last name, four first-name-plus-initial forms, three handles |
| `al_pool/arc_ratings.csv` | 120 | 8 | three given names, five handles |
| `al_pool/arc_valence_ratings.csv` | 199 | 4 | handles only |
| `Phase2/02_Conductor/logs/texture_ratings.csv` | — | 1 | handle only |
| `al_pool_backup_20260703/valence_ratings.csv` | — | 3 | includes the author's own given name |

Rater IDs also appear in code: `s04a_report_figs.py:199` hardcodes
`SENSITIVITY_EXCLUDE = {"<handle>"}`.

**The mapping to `R01`…`R09` already exists.** The report's copy at
`results_and_discussions/data/03_rating_reliability/data/valence_ratings.csv`
is already pseudonymised. Verified 2026-09-06: both copies hold 713 rows in the
same order with identical per-rater counts
(150, 150, 100, 77, 68, 62, 52, 43, 11), so the map is recoverable positionally
and the two will stay consistent. **Recover it, apply it, and keep the mapping
file outside this repo.** Round timestamps to the day while you are there.

Nothing downstream breaks: the analysis only ever groups by rater, never reads
the name, so `s03_rating_irr.py` and the intra-rater consistency filter keep
working unchanged.

**Why the names are in the payload at all.** The conductor re-fits four models
from raw ratings on every boot instead of loading fitted parameters:

| consumer | reads | what it fits |
|---|---|---|
| `s06_boundary_guard.py:226` | `valence_ratings.csv`, `pool_meta.json` | `HybridBoundaryGuard` |
| `s09_arc_fusion.py:60-70` | `arc_ratings.csv`, `arc_valence_ratings.csv`, `texture_ratings.csv` | Bradley-Terry Laplace preference GP |
| `melodic_drone.py:288` | `pool_meta.json`, `valence_ratings.csv` | melody anchor lookup |
| `s10_overlay_explorer.py:84`, `s11_arrangement_scheduler.py:276` | `texture_ratings.csv` | dark-clip veto list for beds |

**Plan.** Pre-compute all four at build time into `conductor/weights/`:
`boundary_guard.json`, `preference_gp.npz`, `melody_anchor.json`,
`bed_veto.json`. The conductor then needs no rating data, boots faster, and is
frozen rather than silently changing if a CSV is edited. The scrubbed CSVs
still ship, but with the training code that uses them (§6.4), not with the
conductor.

### 0.1b Rating data: staging, fragments, merging and format

**Staging comes first.** The raw ratings are copied to
`COMP0158_SupplimentaryMaterial/_raw_ratings_DO_NOT_COMMIT/` **before** any
scrubbing, so the originals are never edited in place and a bad scrub can be
redone. That directory is in `.gitignore` and must stay there: it holds real
names. 35 files, 2.2 MB, staged 2026-09-06.

Scrubbed output goes to `models/<section>/human_ratings/`. The staging
directory is never the source for a shipped file without passing through the
scrub.

#### Which study each set belongs to

| staging folder | files | ships to |
|---|---|---|
| `seed_valence/` | `valence_ratings.csv` (713), `pool_meta.json`, one earlier backup (176) | `3.4.2_human_grounding_and_retrieval/human_ratings/` |
| `arc_pairwise/` | `arc_ratings.csv` (120), `arc_valence_ratings.csv` (199) | `3.5_transition_dynamics_and_scheduling/human_ratings/` |
| `reverb/` | `reverb_ratings.csv` (380) | `3.6_differentiable_reverberation/human_ratings/` |
| `texture/` | `texture_ratings.csv` (122) | `conductor/` bed veto pre-fit, not shipped as data |
| `longtrack/` | 12 session shards | `3.9_human_evaluation_and_protocols/human_ratings/` |
| `space_components/` | 14 session shards + 2 merged outputs | `3.9_human_evaluation_and_protocols/human_ratings/` |

#### Fragments, and the two traps in them

The rating web apps wrote one CSV per boot, so a session that survived a
restart is split across shards. Two families are affected, and naive
concatenation is wrong in both.

**Trap 1: `space_ratings_*` exists in two download folders and they are not
copies.** `SideProjects/rating_space/downloaded/` is an earlier pull;
`downloaded_fresh/` is a later one and a strict superset. Measured 2026-09-06:
`space_ratings_8c842679.csv` **grew from 156 to 312 rows** between pulls, and
three sessions (`211dc68d`, `e917df94`, `f9ce30a5`) appear only in the fresh
pull. Concatenating both folders double-counts the first 156 rows of
`8c842679`. **Use `downloaded_fresh/` only and discard `downloaded/`.**

**Trap 2: `longtrack_ratings_*` is stored twice.**
`longtrack_pool/huggingface_ratings/` and its own `logs/` subfolder hold the
same 11 shards, byte-identical (verified with `cmp`). The parent additionally
holds `longtrack_ratings_020fff1f.csv` (64 rows), which the `logs/` copy lacks.
**Use the parent folder**, which is the superset, and ignore `logs/`. Both also
carry a zero-byte `longtrack_ratings.csv`; drop it.

#### The merge already exists. Do not rewrite it.

`SideProjects/rating_space/analyze_space_ratings.py:39` `load()` is the merge,
and `SideProjects/results/build_results.py` is the cleaning pass on top. Between
them they already do everything this section would otherwise specify:

```python
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
# boot shards overlap when a Space restarts mid-session
df = df.drop_duplicates(subset=["rater", "test", "item_id", "timestamp"])
df = df[df.rater != "setup_check"]        # deploy smoke row
```

Verified end to end 2026-09-06: the 14 `downloaded_fresh` shards hold **709
rows across 15 raters**; `ratings_all_flagged.csv` holds **708 rows across 14
raters** and every one of its keys is present in the shards, with none the other
way. The single missing row and rater is the `setup_check` deploy smoke row that
line 48 drops. `ratings_clean.csv` (617 rows, 6 raters) is then exactly the
`clean == True` subset.

**Write your own merge and you will produce a different dataset from the one
the thesis analysed.** Ship these two scripts instead, and record their output
hashes.

#### Format normalisation, early runs to final

The shards and the merged file are two different schemas, and the merged one is
the target:

| | columns | shape |
|---|---|---|
| raw shard | 9 | `timestamp,rater,test,item_id,response,response2,elapsed_s,note,params` |
| merged | 28 | the 9 above, plus `ts`, the expanded `params` JSON (`score`, `use`, `effect`, `strength`, `reverb`, `treatment`, `style`, `level_db`, `bed_type`, `a_is_staged`, `a_is_reverb`, `pair_kind`, `has_melody`, `clip_id`, `drone`, `bed_id`), and three hygiene flags |

The normalisation is not a rename. `params` is a JSON blob whose keys differ by
test, `load()` expands it to one column per key, and `response` / `response2`
are mapped to numeric `score` and boolean `use`. Anything reading the raw shards
directly is reading a different, unusable shape.

The three hygiene flags are derived, and `build_results.py:27-33` documents the
derivation: `superseded_audio` marks ratings made before the tab's audio was
last re-rendered (filenames are content hashed, so a changed file set means the
rated audio no longer exists); `is_tester` marks six short smoke-test sessions
from the 26 July deploy; `clean` is the conjunction of neither.

`valence_ratings.csv` needs no merge. It and its one earlier backup share an
identical 9-column schema, and the backup is a superseded snapshot rather than a
shard, so it does not ship.

#### One name leak inside the merge script

`build_results.py:33` has `COMPLETERS = ["rater_22cd", "Antglb", "rater_7895"]`.
Two are already pseudonymous; the middle one is a real handle. The `TESTERS` set
two lines above is fully pseudonymous. Fix the one entry when this script moves
across, and apply the same `R01`…`R09` map used everywhere else (§0.1).
`build_results.py:16` also hardcodes the cluster path with the username (§0.4).

### 0.2 Artist names are in the payload

`SideProjects/reverb/logs/album_reverb_bank.json` is loaded at runtime and its
keys name five commercial artists, with local source paths:

```json
"sources": ["/…/data/SOTL", "/…/data/WillianBasinski", "/…/data/Loscil",
            "/…/data/Celer", "/…/data/KyleBobbyDunn"],
"category_summary": {"Celer": {…}, "KyleBobbyDunn": {…}, …}
```

The same names appear in comments in `s07a_corpus_stats.py`, `s07_arc_pool.py`,
`s07b_transition_typology.py`, `s13_corpus_hsmm_fit.py`,
`s11_arrangement_scheduler.py` and `reverb_bank.py`, and in
`conductor_common.py:72`, which expands the tag `sotl` to a full album title.

`appendix_datasets.tex` states that artist metadata and source identifiers were
scrubbed from the project manifests. Shipping this alongside the thesis
disproves that claim with one `grep`.

**Plan.** Re-key the reverb bank on the acoustic descriptors the report already
uses (`dark_long`, `bright_short`, …), rename the file to
`reverb_bank_measured.json`, drop `sources`, and sweep the comments. The fitted
numbers stay: they are measurements, which `ATTRIBUTION.md` frames correctly.

Five further files name artists but sit in directories the scope rule already
excludes (`step1_conductor/analyze_albums.py:54`,
`experiment/temporal_valence/*`, `experiment/AGY/analyze_valence_windows.py:11`,
`report/human/side_projects/generate_melodic_drone_figures.py:264`). The action
is to **confirm none is reachable from a shipped file**, not to edit them.
Checked 2026-09-06: the artist name in the melodic figure script does not reach
the thesis PDF; the only occurrence there is a bibliography entry for a cited
interview, which is legitimate.

### 0.3 Redistribution rights for the audio

86 bed `.wav` files (ESC-50, CC BY-NC; Emo-Soundscapes, per-clip Freesound
terms) and 3 EchoThief IRs. `build_space.py` refuses to publish for this reason
and says so in its docstring.

**Decided: the bed audio ships, with attribution** (2026-08-28, reaffirmed
2026-09-06). The collections carry academic or open terms permitting
redistribution with credit, and `ATTRIBUTION.md` names each source. No fetch
script.

One consequence to get right. ESC-50 is CC BY-NC, so the repo must not carry a
single permissive licence that implicitly grants commercial use of audio that
cannot be licensed that way. Split it: `LICENSE` for **code only**, and
`ATTRIBUTION.md` governing `conductor/assets/`, with a line at the top of this
README saying which applies to what.

### 0.4 Absolute paths carrying the username

`paths.py` and several scripts hardcode
`/cs/student/msc/dsml/2023/myeung/THESIS/Claude/ucl_dsml_thesis_claude/...`.
`paths.py` is a real dependency, so it must resolve from an environment
variable or a CLI argument with a documented default (§4).

**This also covers the result data**, which is easy to miss because it is not
code. Seven files under the report's `results_and_discussions/data/` carry
absolute paths, 1,670 lines in total:

| File | Lines | Leaks |
|---|---|---|
| `03_rating_reliability/data/valence_ratings.csv` | 713 | `/cs/student/projects3/COMP0158/grp_1/<user>/Claude/...` |
| `03_rating_reliability/data/pool_meta.json` | 450 | same |
| `02_affect_estimator_validity/data/cross_judge_indomain.csv` | 300 | `/cs/student/msc/dsml/2023/<user>/THESIS/data/DEAM/...` |
| `06_component_preferences/data/reverb_ratings.csv` | 182 | `/cs/student/msc/dsml/2023/<user>/THESIS_MYDIR/...` |
| `05_longtrack_recency/data/longtrack_meta.json` | 19 | `.../grp_1/<user>/Claude/...` |
| `06_component_preferences/data/stimulus_conditions.json` | 5 | same |
| `06_component_preferences/data/analysed/build_meta.json` | 1 | `/cs/student/project_msc/2025/dsml/<user>/COMP0158_Report/...` |

Three exposures: **two usernames**, which also links the two identities; the
**cluster layout**, including the group and module; and a path in
`cross_judge_indomain.csv` pointing at **DEAM audio** inside the personal tree,
which reads as redistribution of a research-licensed corpus.

**The fix is filename-only.** Every path points at a `.wav`, `.mp3` or `.json`
whose **basename is the join key** (`pool_013.wav`, `lt_up_c0_noarc.wav`,
`598.mp3`), and the analysis scripts join on the basename. Stripping the
directory breaks nothing.

### 0.5 Secrets

Grepped for tokens and API keys across `.py`, `.sh` and `.md`: **none found.**
Re-run on the assembled tree before handover, since `FINAL_CONDUCTOR/` carries
Hugging Face deploy scripts.

---

## 1. Target layout

**Revised 2026-09-06.** Model directories are named after the report section
that documents them: section number, underscore, human-readable name. Each
carries the same four subdirectories, so a reader who has just read §3.6 knows
exactly where to look and what they will find.

```
code/
  README.md                                  this file
  LICENSE                                    code only
  ATTRIBUTION.md                             governs conductor/assets/
  common/                                    the only shared import
      ddsp_synth.py
      soundscape_synth.py
      paths.py                               REWRITTEN, see §4
  models/
    3.3_drone_synthesis_and_nsynth_prior/
    3.4.1_surrogate_guided_optimisation/
    3.4.2_human_grounding_and_retrieval/
    3.5_transition_dynamics_and_scheduling/
    3.6_differentiable_reverberation/
    3.7_melody_generation/
    3.9_human_evaluation_and_protocols/
        README.md                            run order, with commands
        smoke_test.py                        no dataset needed; must exit 0
        train/                               fitting scripts
        weights/                             fitted artefacts that ship
        inference/                           what the runtime imports
        human_ratings/                       scrubbed rating data this section used
        data/                                generated output, git-ignored
  conductor/                                 §3.8, INFERENCE ONLY
      app.py                                 Gradio entry point
      requirements.txt
      engine/                                synthesis and retrieval glue
      runtime/                               scheduler, effects, melody playback
      UI/                                    skins, faders, visuals, web audio, ring player
      weights/                               pre-fits computed at build time
      assets/                                bed audio and IRs, per ATTRIBUTION.md
```

Section numbers are taken from the compiled thesis, not from memory:
3.3 Parametric Drone Synthesis and NSynth Prior; 3.4.1 Surrogate-Guided
Optimisation; 3.4.2 Human Grounding and Manifold Retrieval; 3.5 Transition
Dynamics and Scheduling; 3.6 Differentiable Reverberation and Spatial
Acoustics; 3.7 Melody Generation; 3.8 Runtime Conductor Orchestration.

§3.4 is split at subsection level because its two halves are the thesis's
central contrast: 3.4.1 is the automated loop that failed, 3.4.2 is the human
grounding that replaced it. Putting them in one directory would hide the
finding the report is built on.

§3.9 Human Evaluation Methodology and Listening Protocols gets a directory
because the long-track recency study and the component-preference study produce
data and merge scripts, and nothing else in the layout has a home for them.

§3.1 and §3.2 get no directory. §3.1 is the architecture overview and §3.2 is
the developmental explorations, neither of which ships code.

### Empty subdirectories are not a mistake

Three sections have no rating data (3.3, 3.4.1, 3.7) and one has no runtime
component (3.4.1, the loop that was abandoned). Their `human_ratings/` and
`inference/` directories keep a `README.md` saying so, and why. That is
deliberate: an empty `inference/` under "Surrogate-Guided Optimisation" is the
clearest possible statement that nothing from §4.1 reached the running system.

### Two rules that follow from this shape

- **`inference/` is the source of truth for runtime modules.** `conductor/`
  imports from `models/*/inference/` rather than holding copies, so there is one
  definition of the arranger, the scheduler and the reverb. `conductor/engine/`
  and `conductor/runtime/` shrink to the glue that is genuinely conductor-only.
- **Nothing imports across sibling model directories.** Shared code goes to
  `common/`. `ddsp_synth.py` lives there rather than in
  `3.3_drone_synthesis_and_nsynth_prior/inference/` because §3.3, §3.7 and the
  conductor all need it.

### Rules for every directory

- **No step-number prefixes on files.** `s09_train_closed_loop.py` becomes
  `train/train_closed_loop.py`. Run order lives in the README.
- **Every model directory gets a `README.md`** naming the report section it
  implements, the run order as a numbered list, and a working example command.
- **Dataset directories are arguments** with documented defaults (§4).
- Every file runs `python -m py_compile` clean after the rename pass.

---

## 1a. Conventions every migrated directory must meet

Raised across the 2026-09-06 session and binding on every section, not just the
one being worked on. A directory is not finished until all of these hold.

### Naming, and what must never appear

| Rule | Why |
|---|---|
| **No personal names anywhere.** Not `Matt`, `Matthew`, `matthew`, `maatt`, `matt4`, and not in comments, keys, docstrings or fixtures. | The report is anonymous-marked and the ratings are human-subject data. `s15_ladder_human_report.py:75` uses `by_rater["matthew"]` as a **lookup key**, so renaming it to `R01` must happen in the same commit as the rating scrub or the script looks up a rater that no longer exists. |
| **No `JAMAI` / `jamai` / `JamAI`.** The project codename is retired. Environment variables are `DRONE_DATA`, `DRONE_NSYNTH`, `DRONE_DEAM`, `DRONE_EMO`, `DRONE_ESSEN`, `DRONE_ECHOTHIEF`. | The codename appears in no report section, so a reader cannot connect it to anything. **`s06_pack_demo.py`'s `verify()` still sets `JAMAI_DATA`**: rename it when §3.8 migrates, or the acceptance test passes by reading nothing. |
| **No `Claude`, `Gemini` or other tool names in paths, filenames or defaults.** | They named working directories, not components. The generated-output root is `generated_audio`. |
| **No step-number prefixes on files.** `s09_train_closed_loop.py` becomes `train/train_closed_loop.py`. | Run order belongs in the README, where it can be explained. |
| **No absolute path carrying a username.** The one exception is `common/paths.py`'s `DEFAULTS`, which records the build host on purpose. | §0.4. |

Verification greps are in §5 step 1 and Wave D.

### Structure

- **Training and inference must be separate files.** Where the original
  conflates them, the refactor splits them: sampling and fitting to `train/`,
  anything the runtime imports to `inference/`. `inference/` must not import
  `paths`, resolve a dataset root, or sample anything.
- **Every script takes its dataset roots as arguments**, resolved flag, then
  environment variable, then the build-host default in `common/paths.py`. A
  missing dataset fails with its name and where to download it, never a
  `FileNotFoundError` or a traceback.
- **Generated output goes to the section's own `data/`**, which holds a
  `.gitignore` of `*` plus `!.gitignore`, so the directory is tracked and its
  contents are not. Nothing writes outside its own section by default.
- **Commit three or four sample outputs** under `data/samples/`, which the
  section `.gitignore` un-ignores with `!samples/` and `!samples/**`. Generate
  them with a fixed seed, keep them short, and give the directory a README with
  the exact commands that reproduce them and a table of what each one is.
  A reader can then hear or inspect a section without running anything. The
  §3.3 set is four 8-second clips, 1.1 MB total, and because that section's
  fitted prior ships, one command regenerates them with no dataset present.
- **`weights/` holds artefacts that ship, and nothing else.** A quick or
  truncated fit written there while testing will be mistaken for the real one.
  Delete it, or move it to `data/`. The §3.3 samples were made from a 300-note
  prior for speed, and that prior was removed from `weights/` afterwards.
- **Running instructions live in the script**, in the module docstring, as
  commands that can be pasted. Each names the previous and next step.

### Every directory ships a smoke test, and it must actually be run

`smoke_test.py` at the section root. No dataset, no checkpoint, no network. It
builds synthetic inputs, runs the real code, and exits non-zero on any failure.
Writing one is not enough: **run it and paste the result.** It must cover:

1. **The train/inference boundary**, by parsing imports with `ast` rather than
   grepping text. Docstrings legitimately mention the other module by name, and
   a grep-based check gives a false failure the moment one does. That happened.
2. **`--help` on every script**, asserting the dataset flags exist.
3. **A missing dataset**, asserting the message names the dataset and no
   traceback is printed.
4. **The inference path end to end**, checking shape, finiteness, normalisation
   and determinism.
5. **The training path end to end against a synthetic input**, checking the
   files and columns it claims to write.

### A dry run is not `--help`

Before a section is called done, run every script against the real data with
**no flags beyond the ones that shorten it** (`--limit`, `--n`). `--help`
passing proves nothing about defaults. Two defects in §3.3 were invisible to
the smoke test and only appeared on a real run:

- **The noise floor was unseeded.** `--seed` reproduced theta sampling and the
  f0 wander, but the filtered noise comes from `torch.randn` inside
  `ddsp_synth.py:76`, which takes no generator. Two renders of the same theta
  differed by 0.04 on a signal peaking at 0.9. Fixed by also calling
  `torch.manual_seed`. **Check this in every section that renders.**
- **The recommended path crashed.** `build_nsynth_prior.py --metadata-only`
  leaves no wavs in the output directory, and `analyse_nsynth_timbre.py`
  defaulted its audio directory to that same directory, so it read nothing and
  died in `np.concatenate` after all the setup. Fixed by defaulting
  `--audio-dir` to `<nsynth-root>/audio` and failing early with a message.

Record the numbers the dry run produced and check them against the report. The
§3.3 run returned **8,269** matching NSynth notes, which is what the thesis
states, so the filter survived the port.

---

## 2. What moves where

Destinations updated 2026-09-06 for the layout in §1. Paths below are relative
to `models/` unless they start with `conductor/` or `common/`.

### 2.1 The authoritative inference set already exists

`Phase2/02_Conductor/s06_pack_demo.py` is a working packer. It owns the file
inventory, rewrites IR paths, writes `MANIFEST.json` with a sha256 per file, and
has a `verify()` that runs the packed stack with env vars pointed at
non-existent directories. That is the only thing that actually proves
self-containment. Its output, `FINAL_CONDUCTOR/app/`, is 143 files / 75 MB and
runs standalone.

**Do not re-derive the inference set by hand.** Start from that manifest, apply
the splits below, and keep `verify()` as the acceptance test. A migration that
breaks it is not finished.

### 2.2 `3.3_drone_synthesis_and_nsynth_prior/`

| source | destination | new name |
|---|---|---|
| `s01_select_nsynth_strings.py` | `train/` | `build_nsynth_prior.py` |
| `s03_generate_ddsp_dataset.py` | `train/` | `generate_preset_bank.py` |
| `s01_gp_active_learning.py` | `train/` | `active_learning_pool.py` |
| `s02_regen_pool.py` | `train/` | `regenerate_pool.py` |
| `s07_arc_pool.py` | split, see §3.1 | `train/arc_pool.py` + `conductor/engine/render_params.py` |
| `data/ddsp_*10k/labeled_index.csv` | `weights/` | unchanged (17 MB, two files) |
| `ddsp_synth.py` | `common/` | unchanged, shared |
| — | `inference/` | README pointer to `common/ddsp_synth.py` |
| — | `human_ratings/` | README, one line: none used |

### 2.3 `3.4.1_surrogate_guided_optimisation/`

The estimators, the distilled proxy, the CVAE and the closed loop. The 28 August
plan had no home for any of these, and they are three of the six results
sections.

| source | destination | new name |
|---|---|---|
| `s04_label_ddsp_dataset.py` | `train/` | `label_dataset.py` |
| `experiment/cross_judge/cross_judge.py` | `train/` | `cross_domain_validity.py` |
| `experiment/cross_judge/cross_judge_indomain.py` | `train/` | `cross_domain_indomain.py` |
| `s06_train_inverse_cvae.py` | `train/` | `train_cvae.py` |
| `s07_evaluate_cycle_consistency.py` | `train/` | `evaluate_cycle_consistency.py` |
| `s08_train_judge_proxy.py` | `train/` | `train_judge_proxy.py` |
| `s09_train_closed_loop.py` | `train/` | `train_closed_loop.py` |
| `s10_dagger_render.py` | `train/` | `dagger_render.py` |
| — | `inference/` | **empty by design.** README: nothing here reached the runtime. |
| — | `human_ratings/` | README, one line: none used. |

`label_dataset.py` is the frozen judge. Its documented configuration is
`--arousal_ridge_path emo --arousal_scale pm1`: valence from MERT to the DEAM
ridge, arousal from MERT to the Emo-Soundscapes ridge. The audEERING default in
the argument parser was never used in any run. Say so in the directory README,
because the thesis has it wrong in one place (§7.2).

### 2.4 `3.4.2_human_grounding_and_retrieval/`

| source | destination | new name |
|---|---|---|
| `s04_relabel_bank.py` | `train/` | `propagate_labels_krr.py` |
| `s14_control_ridge.py` | `train/` | `fit_control_ridge.py` |
| `s03_rating_irr.py`, `s03_compute_irr.py` | `train/` | `rating_agreement.py` |
| `s06_boundary_guard.py` | split, see §3.3 | `train/fit_guard.py` + `inference/guard.py` |
| `retrieval_engines.py` | `inference/` | `retrieval_engines.py` |
| `Phase2/Gemini/step01_gp_softknn_engine.py` | `inference/` | `retrieval.py` |
| fitted guard, anchor | `weights/` | `boundary_guard.json`, `melody_anchor.json` |
| `al_pool/valence_ratings.csv`, `pool_meta.json` | `human_ratings/` | scrubbed to `R01`…`R09` per §0.1 |

### 2.5 `3.5_transition_dynamics_and_scheduling/`

| source | destination | new name |
|---|---|---|
| `s09_arc_fusion.py` | `train/` | `fit_preference_gp.py` |
| `s08_arc_rating_app.py` | `train/` | `arc_rating_app.py` (data collection, not runtime) |
| `s11_arrangement_scheduler.py` | `inference/` | `scheduler.py` |
| `s17_coherence_reranker.py` | `inference/` | `coherence_reranker.py` |
| `s01_arc_policy.py` | `inference/` | `arc_policy.py` |
| `figs/corpus_hsmm.json` | `weights/` | `hsmm_transitions.json` |
| `logs/arc_types.json` | `weights/` | unchanged |
| fitted GP posterior | `weights/` | `preference_gp.npz` |
| `al_pool/arc_ratings.csv`, `arc_valence_ratings.csv` | `human_ratings/` | scrubbed per §0.1 |
| `s13_corpus_hsmm_fit.py` | `train/` | `fit_corpus_hsmm.py` |
| `s07b_transition_typology.py` | `train/` | `extract_transition_typology.py` |

**Revised 09-07: the HSMM fitter DOES ship, generalised.** The method is not
the corpus. Both halves migrate and take `--audio-dir` as a list, so the
pipeline retrains on any audio: each directory becomes a group label and is
walked recursively. This also removes §0.2's artist-name blocker at the root --
`ARTIST_DIRS` was a hardcoded map of artist names to directories, and it is
replaced by whatever the caller passes.

The corpus itself still does not ship. `weights/SHA256SUMS` records the fitted
artefact so it stays traceable, and the directory README states which numbers
are corpus-fit and which are authored.

Verified: refitting from the original descriptor cache reproduces the shipped
`hsmm_transitions.json` with **zero** differing fields.

`s11_arrangement_scheduler.py:115` imports `s13_corpus_hsmm_fit` for
"corpus-fit dwell/trans, opt-in". The module reads nothing at import time, only
constants and loaders, so lift those constants into `scheduler.py` and drop the
import.

**The Bradley-Terry preference GP stays in the conductor** (decided
2026-08-28). Chapter 3 does not describe it, and that is deliberate: the shipped
system may do more than the thesis writes up. Do not let a later tidy-up "fix"
the mismatch by deleting it. Put a note in `conductor/README.md` saying the
runtime intentionally carries components the report does not cover.

### 2.6 `3.6_differentiable_reverberation/`

| source | destination | new name |
|---|---|---|
| `SideProjects/reverb/learnable_reverb.py` | split | `train/model.py` (fit) + `inference/reverb.py` (apply) |
| `SideProjects/reverb/learn_from_irs.py` | `train/` | `fit_from_irs.py` |
| `SideProjects/reverb/reverb_bank.py` | `inference/` | `reverb_bank.py`, de-identified per §0.2 |
| `logs/ir_reverb_bank.json` | `weights/` | unchanged |
| `logs/album_reverb_bank.json` | `weights/` | `reverb_bank_measured.json`, re-keyed |
| `reverb_ratings.csv` | `human_ratings/` | scrubbed per §0.1 |
| `irs/*.wav` (3) | `conductor/assets/` | committed, per §0.3 |

`reverb_bank.py:149` imports `load_ir` from `learn_from_irs`. That one function
moves to the inference side so the runtime stops importing a fitting script.

The fitted model is three `nn.Parameter` scalars per impulse response, not a
network. There is no checkpoint to ship: `weights/` holds the JSON banks.

### 2.7 `3.7_melody_generation/`

| source | destination | new name |
|---|---|---|
| `GEMINI/train_melodic_transformer.py` | `train/` | `train_transformer.py` |
| `GEMINI/dataset_builder.py` | `train/` | `build_essen_dataset.py` |
| `GEMINI/benchmark_and_render.py` | `train/` | `benchmark_configs.py` |
| `GEMINI/runtime_melodic_drone.py` | `inference/` | `melody_transformer.py` |
| `SideProjects/melodic_drone/melodic_drone.py` | `inference/` | `melody_markov.py` |
| three `.pt` checkpoints + sidecars | `weights/` | see below |
| Markov tables (inside `melodic_drone.py`) | `weights/` | extract to `markov_order2.json` |
| — | `human_ratings/` | README, one line: none used |

**All three transformer checkpoints ship** (decided 2026-09-06), because the UI
lets the user pick any of them as well as the Markov generator. Measured
2026-09-06 from `SideProjects/melodic_drone/GEMINI/checkpoints/`:

| file | size | parameters | thesis row |
|---|---|---|---|
| `melodic_transformer_L3_d128.pt` | 2.4 MB | 612,387 | Compact, 0.61M |
| `melodic_transformer_L6_d128.pt` | 4.7 MB | 1,207,203 | Deep, 1.20M |
| `melodic_transformer_L6_d256.pt` | 18.7 MB | 4,773,667 | Wide, 4.77M |

All three match the reported parameter counts exactly. Total 25.8 MB.

**`melodic_transformer_cpu.pt` is not a fourth model.** It is 4.7 MB with
1,207,203 parameters and `layers.0`…`layers.5`, so it is the L6/d128 weights
under a promoted-copy alias. `s02_conductor_app.py:1086` says so directly: "The
packed app and the Space ship the L6_d128 weights AS
`melodic_transformer_cpu.pt`". Ship the three real names and update the loader,
which currently matches by label to work around the alias.

**Only `GEMINI/checkpoints/` holds all three.** Every `FINAL_CONDUCTOR/` copy
carries just `L3_d128` and the `cpu` alias, so the deployed Space could offer
two of the three, not three. Confirm what the UI dropdown was actually populated
with before describing it in the report.

### 2.8 `conductor/` (§3.8)

`s02_conductor_app.py` becomes `conductor/app.py`. `s20_skins.py`,
`s21_faders.py`, `s22_visuals.py`, `s19_webaudio.py`, `s18_ring_player.py` and
`s10_overlay_explorer.py` become `conductor/UI/skins.py`, `faders.py`,
`visuals.py`, `webaudio.py`, `ring_player.py`, `overlay_mixer.py`.

`Phase2/Gemini/step02_arranger.py` becomes `conductor/engine/arranger.py`.
`stereo/stereo_pad.py`, `stereo/distance.py` and `crackle/crackle.py` become
`conductor/runtime/`. `soundscape_synth.py` goes to `common/`.

`conductor/weights/` holds the four build-time pre-fits (§3.3) plus the bank
indexes. Everything else it needs is imported from `models/*/inference/`.

`step02_arranger.py:25` does `import paths` and never uses it. Dead import.
Delete it and `paths.py` leaves the conductor payload entirely.

---

## 3. Breaking the training to inference dependencies

The real engineering work. The conductor imports eight training-side modules.
Each needs the same treatment: extract what the runtime uses, leave the rest.

### 3.1 `s07_arc_pool.py`

The conductor uses `apply_chord`, `HOLD_S`, `RENDER_KW`, `aw_rms` and
`AW_TARGET`: constants and two pure functions out of a 21 KB pool builder.
Extract into `conductor/engine/render_params.py`. The pool builder stays in
`3.3_drone_synthesis_and_nsynth_prior/train/`.

### 3.2 `s09_arc_fusion.py` / `s08_arc_rating_app.py` / `s01_arc_policy.py`

The GP stays (§2.5), so: `s01_arc_policy.py` is runtime and moves as-is;
`s09_arc_fusion.py` is a fit and moves to
`3.5_transition_dynamics_and_scheduling/train/fit_preference_gp.py`, with its
posterior pre-computed into `conductor/weights/preference_gp.npz`;
`s08_arc_rating_app.py` is a rating UI and leaves the conductor.

`s01_arc_policy.py` must load the pre-fit posterior instead of calling into
`s09`. That is the only code change of substance here.

### 3.3 `s06_boundary_guard.py`

`HybridBoundaryGuard` is **fitted at boot** from `valence_ratings.csv` and
`pool_meta.json`. That is why participant data is in the payload (§0.1).

Split into `3.4.2_human_grounding_and_retrieval/train/fit_guard.py`, which fits and writes
`conductor/weights/boundary_guard.json`, and
`3.4.2_human_grounding_and_retrieval/inference/guard.py`, which loads it. Removes 440 KB of rating CSVs from the conductor and the privacy
problem with them.

Same treatment for the melody anchor (`melodic_drone.py:288`) and the bed veto
list (`s10_overlay_explorer.py:84`), both small aggregations over the same CSVs.
With §3.2 these four pre-fits are what let the conductor ship with no ratings
data at all.

### 3.4 `learnable_reverb.py` / `learn_from_irs.py`

Covered in §2.4. `apply_reverb_np` and `load_ir` are runtime; the fitting loop
and the multi-scale STFT loss are training.

### 3.5 `s13_corpus_hsmm_fit.py`

Covered in §2.5. Lift the constants, drop the import.

### 3.6 Model classes living inside training scripts

Where a checkpoint is loaded by re-importing the script that trained it, move
the `nn.Module` definition into a `model.py` that both sides import. Applies to
the melodic transformer and the reverb model. Check for others while migrating.

---

## 4. Dataset paths

`common/paths.py` replaces the current `paths.py`, which hardcodes a Windows
laptop and this cluster:

```python
DEFAULTS = {
    "deam":            "…/data/deam",
    "emo_soundscapes": "…/data/emo_soundscapes",
    "nsynth":          "…/data/nsynth-train",
    "essen":           "…/data/essen",          # via music21
    "echothief":       "…/data/echothief",
}
```

Every training script takes `--data-root` (or a per-dataset flag), falls back to
the environment variable, then to the default. Each fails with a message naming
the dataset and where to obtain it, not a bare `FileNotFoundError`.

`conductor/` imports none of this. **Acceptance test:** `s06_pack_demo.py`'s
`verify()` with `DRONE_DATA` and friends pointed at `/nonexistent`. The
conductor must still start and render.

---

## 5. Order of work

1. **De-identify.** Ratings are already staged (§0.1b). Apply the `R01`…`R09`
   map to the staged copies, re-key the reverb bank (§0.2), and strip the
   absolute paths (§0.4). Do this before anything is copied onward: doing it
   later means scrubbing several copies.
2. **Scaffold** the six model directories with their four subdirectories each, plus `common/paths.py` and the `conductor/` skeleton.
3. **Write the four pre-fit scripts** (guard, preference GP, melody anchor, bed
   veto) and generate `conductor/weights/`. This is what lets the conductor ship
   with no ratings data (§3.2, §3.3).
4. **Migrate the conductor** from the existing manifest: copy, rename, rewire
   imports, point the four consumers at the pre-fit weights, break the remaining
   training dependencies (§3).
5. **Run `verify()`** with dataset roots at `/nonexistent`. Not done until this
   passes.
6. **Migrate the model directories** in report order: `3.3` → `3.4.1` →
   `3.4.2` → `3.5` → `3.6` → `3.7`. Each is done when its four subdirectories
   are populated or carry a one-line README saying why they are empty, and its
   own README lists a run order with a working command.
7. **Licence and attribution split** (§0.3), then confirm a clean clone plus
   `pip install -r requirements.txt` gives a working conductor.
8. **Packaging extras**: `.devcontainer/devcontainer.json` with
   `forwardPorts: [7860]` for Codespaces, a Dockerfile, and a GitHub Actions job
   running `verify()` on push.

Steps 1 to 5 make the conductor shippable. Steps 6 to 8 make the repo
submittable.

---

## 6. Decisions taken

### 2026-08-28

1. **Preference GP** stays in the conductor. The shipped system may carry more
   than the thesis describes. (§2.5)
2. **Rater data** does not ship with the conductor; four boot-time fits become
   pre-computed weights. (§0.1, §3.2, §3.3)
3. **Bed audio** is committed, with attribution. Code licence and asset
   attribution kept separate. (§0.3)

### 2026-09-06

4. **Layout.** `common/`, `models/` and `conductor/` at the top. Model
   directories are named `<report section>_<human readable>` and each carries
   `train/`, `weights/`, `inference/` and `human_ratings/`. §3.4 splits at
   subsection level, since 3.4.1 and 3.4.2 are the thesis's central contrast.
   `conductor/` keeps `engine/runtime/UI/weights/assets`, because that is what
   enforces the inference boundary the acceptance test checks. (§1)
5. **Bed audio ships.** Reaffirmed. (§0.3)
6. **Checkpoints: everything ships, runtime and training.** The melody set is
   **all three transformer checkpoints plus the Markov tables**, because the UI
   offers the user a choice between them (§2.7). The training checkpoints ship
   too, so §4.1 and §4.2 are reproducible without a retrain: `judge_proxy.pt`,
   `judge_proxy_e0.pt` and `closed_loop_mapper.pt` under
   `3.4.1_surrogate_guided_optimisation/weights/`, `inverse_cvae.pt` under the
   same, and `attn_retrieval.pt` under
   `3.4.2_human_grounding_and_retrieval/weights/`. Confirmed 2026-09-06 that
   none of these five is referenced anywhere in a conductor tree, so they are
   training artefacts only and must not be wired into `conductor/weights/`.
   Total 2.6 MB.
7. **Rating CSVs ship, scrubbed, in their section's `human_ratings/`.**
   `valence_ratings.csv` and `pool_meta.json` under `3.4.2`; `arc_ratings.csv`
   and `arc_valence_ratings.csv` under `3.5`; `reverb_ratings.csv` under `3.6`.
   Not with the conductor, which needs none of them (§3.3). Pseudonymise to
   `R01`…`R09` using the map already applied to the report copy (§0.1).

8. **Rating data is staged before it is scrubbed.** Raw copies go to
   `COMP0158_SupplimentaryMaterial/_raw_ratings_DO_NOT_COMMIT/`, which is in
   `.gitignore`. The originals in the code tree are never edited in place.
   Staged 2026-09-06: 35 files, 2.2 MB. (§0.1b)

9. **Fragmented studies are merged with the existing scripts, not new ones.**
   `analyze_space_ratings.load()` and `build_results.py` already do the
   concatenation, the boot-shard de-duplication and the format expansion, and
   they are what produced the numbers in the report. Writing a fresh merge would
   produce a different dataset. Both ship in
   `3.9_human_evaluation_and_protocols/train/`. (§0.1b)

10. **Generated output lives in the section's own `data/`.** Each holds a
    `.gitignore` of `*` and `!.gitignore`, so the directory is tracked and its
    contents are not. This replaced defaulting to a shared root under the build
    host, which made every default path print a cluster directory. Scripts now
    write inside their own section unless told otherwise. (§1a)

11. **Naming rules are binding across all sections**, not per directory: no
    personal names, no `JAMAI` codename, no tool names such as `Claude` in
    paths or defaults, no step-number prefixes. Environment variables are
    `DRONE_*`. (§1a)

12. **Fitted artefacts ship at full precision, unchanged.** The §3.3 timbre
    prior is 29 MB and was shipped as-is rather than cast to float16, which
    would have halved it. A shipped artefact should be bit-identical to the one
    that produced the reported results; sha256 it against the source after
    copying. Apply the same rule to every `weights/` entry.

---

## 7. Report changes this migration implies

Findings from the code that the report currently contradicts. Not code tasks.
Logged here so they reach the report.

1. **Deployed melody model. CONFIRMED 2026-09-06.**
   `melodic_drone_and_sequence_generation.tex:44` says the 3-layer
   configuration was selected because "larger models provided no perceptible
   musical benefit and only increased latency and memory footprint". The
   conductor loads `melodic_transformer_cpu.pt`, measured at **1,207,203
   parameters across six layers**, which is the thesis's own Deep L6/d128 row,
   and `s02_conductor_app.py:1071` sets `PREFERRED_CKPT = "L6_d128"`.
   The report's own figures make the parsimony argument thin anyway: 1.5202
   against 1.5194 nats, so the deeper model is marginally *better*. Replace
   with what happened: three configurations were trained, the two 128-wide ones
   were within 0.0008 nats so validation loss could not separate them, and the
   choice was made by listening. Move the bold row in
   `tab:melodic_model_benchmark` to the deep configuration and check its size
   and latency figures against the shipped checkpoint.

2. **Arousal estimator.** `closed_loop_optimization_and_reward_hacking.tex:20`
   says arousal comes from an audEERING wav2vec 2.0 model plus an
   Emo-Soundscapes ridge. Every invocation in the tree passes
   `--arousal_ridge_path emo --arousal_scale pm1`, so arousal is MERT to the
   Emo-Soundscapes ridge and audEERING was never used;
   `memory/project_overview.md:56` records why ("audEERING arousal too
   compressed"). Appendix C.4 and A.4 are already correct. the author skipped this
   on 2026-09-06 because it changes no result; it is logged, not open.

3. **Artist scrubbing.** `appendix_datasets.tex` claims artist metadata was
   scrubbed from the project manifests. True only once §0.2 is done.

4. **Not an error, do not "fix" it.** `appendix_datasets.tex:33` gives the
   active-learning ratings as `N=1` while `valence_ratings.csv` holds 713 rows
   from 9 raters. Both are correct. The `N=1` is the theta-KRR fitting set,
   which is R01 alone, and `human_grounding_and_retrieval.tex:19` already says
   "annotated by the author (R01)". The 9-rater figures are the separate
   reliability study. The 28 August note calling this a contradiction was
   wrong; checked 2026-09-06.

---

## Progress log

| date | step | status |
|---|---|---|
| 2026-08-28 | Plan written | done |
| 2026-08-28 | Decisions 1-3 (§6) | done |
| 2026-09-06 | Decisions 4-7 (§6), layout revised, §2 destinations re-pointed | done |
| 2026-09-06 | Checkpoint audit: 3 melody models measured, 5 training `.pt` excluded | done |
| 2026-09-06 | Participant-name scan across 8 tree copies; R01-R09 map confirmed recoverable | done |
| | 1. De-identify and strip paths | not started |
| 2026-09-06 | 2. Scaffold: 7 model dirs x 4 subdirs, `common/`, `conductor/` | done |
| 2026-09-06 | Rating audit: 2 fragment families, 2 double-count traps found | done |
| 2026-09-06 | Raw ratings staged to `_raw_ratings_DO_NOT_COMMIT/`, gitignored | done |
| 2026-09-06 | `human_ratings/README.md` written for all 7 model dirs | done |
| | 3. Four pre-fit scripts to `conductor/weights/` | not started |
| | 4. Migrate conductor | not started |
| | 5. `verify()` passes | not started |
| 2026-09-06 | Conventions fixed (§1a): naming, structure, smoke test, dry run | done |
| 2026-09-06 | 6a. **§3.3 migrated**: 3 train scripts, 1 inference module, smoke test (26 checks), README | done |
| 2026-09-06 | 6a. §3.3 dry-run end to end on real NSynth; 8,269 notes matches the report | done |
| | 6b. §3.4.1 surrogate-guided optimisation | not started |
| | 6c. §3.4.2 human grounding and retrieval | not started |
| | 6d. §3.5 transition dynamics | not started |
| | 6e. §3.6 differentiable reverberation | not started |
| | 6f. §3.7 melody generation | not started |
| | 6g. §3.9 human evaluation and protocols | not started |
| | 7. Licence and attribution split | not started |
| | 8. Packaging extras | not started |
