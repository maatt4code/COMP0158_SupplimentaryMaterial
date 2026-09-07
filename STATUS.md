# STATUS — supplementary code migration

Last updated **2026-09-07**. Written so a fresh session on
another machine can continue without re-scanning the code tree. Everything
marked *verified* was derived from source in that session: do not re-derive it.

**Where we stopped.** §3.3, §3.4.1, §3.4.2, §3.5 and §3.6 are complete and
dry-run. §3.3 is `c2890bd`, §3.4.1 is `2a95564`. **§3.4.2, §3.5 and §3.6 are
written but NOT yet committed.** De-identification is DONE for seed-valence,
arc-pairwise, texture and reverb; longtrack and space-components remain, with
§3.9. **Both §0.2 artist-name blockers are now closed** (§13, §14). Next: §3.7. The report side is separately
finished and committed, and its log is
`THESIS_MYDIR/COMP0158_Report/notes/CLAUDE.md`, session 2026-09-06 (evening).

---

## 0. Read these first, in this order

1. **`code/README.md`** — the migration plan. 850 lines. Its **§1a** holds the
   conventions every directory must meet, and **§2** the file-by-file mapping
   with rename targets. Read §1a before writing any code.
2. **`README.md`** (top level) — the reader-facing document: environments,
   dataset table with URLs and licences, the folder table, how to run
   everything.
3. This file, for what is done and what is next.

The report-side plan is a separate document:
`THESIS_MYDIR/COMP0158_Report/notes/REPORT_REVIEW.md`. Do not confuse them.
`notes/CODE_CLEANUP.md` in that repo is an **earlier, superseded** plan of mine;
`code/README.md` is the one of record.

## 1. Locations

| What | Where |
|---|---|
| This repo | `/cs/student/msc/dsml/2023/myeung/THESIS_MYDIR/COMP0158_SupplimentaryMaterial` |
| Source code tree, everything is copied *from* here | `/cs/student/msc/dsml/2023/myeung/THESIS/Claude/ucl_dsml_thesis_claude` |
| Thesis LaTeX | `/cs/student/msc/dsml/2023/myeung/THESIS_MYDIR/COMP0158_Report` |
| Datasets on this host | `/cs/student/msc/dsml/2023/myeung/THESIS_MYDIR/data` |
| Python with torch 2.6.0, soundfile, scipy | `/cs/student/projects3/COMP0158/grp_1/bin/env/env_linux_nvidia/bin/python` |

**Two paths reach the same directory.** `THESIS_MYDIR/...` and
`/cs/student/project_msc/2025/dsml/myeung/...` are the same place through a
symlink, so scripts print one prefix while you typed the other. Not a bug. The
system `python3` has no torch: use the env above for anything that runs.

## 2. Working rules

- **Propose before editing. Every time.** Show the exact before and after, wait
  for an explicit yes, then apply. "Move on to the next one" is not approval.
- **Check the source before proposing.** Every finding checked against the code
  first changed shape. Several turned out to be no error at all.
- **A dry run is not `--help`.** Run each script against the real data with no
  flags beyond `--limit` or `--n`. Both §3.3 defects were invisible to the
  smoke test and appeared only on a real run.
- **Write the smoke test and then actually run it.** Paste the result.
- Be concise. Plain words. No AI register.

## 3. Layout

```
code/
  README.md            reproduction guide, ships (decision 10)
  common/              ddsp_synth.py, device.py, paths.py
  models/<section>/    train/ weights/ inference/ human_ratings/ data/
                       plus README.md and smoke_test.py
  conductor/           app.py engine/ runtime/ UI/ weights/ assets/   (§3.8)
env/                   env_linux_nvidia.yml, env_windows_arc.yml
_raw_ratings_DO_NOT_COMMIT/    staged raw ratings, git-ignored, real names
```

Model directories are named `<report section>_<human readable>`, taken from the
compiled thesis: 3.3 drone synthesis and NSynth prior, 3.4.1 surrogate-guided
optimisation, 3.4.2 human grounding and retrieval, 3.5 transition dynamics and
scheduling, 3.6 differentiable reverberation, 3.7 melody generation, 3.9 human
evaluation and protocols. §3.1 and §3.2 ship no code.

## 4. Done

- **Scaffold**: 7 model directories x 5 subdirectories, `common/`, `conductor/`.
- **`common/`**: `paths.py` rewritten (flag, then env var, then build-host
  default; fails with the dataset name and a URL), `device.py` (CUDA, XPU, CPU),
  `ddsp_synth.py` copied unchanged.
- **§3.3 migrated in full.** Three train scripts, one inference module, a smoke
  test of 26 checks, a README, four committed sample clips, and the 29 MB
  measured timbre prior. The section runs end to end with no dataset at all:
  `generate_preset_bank.py` alone reproduces the samples.
- **Environments** copied and scrubbed of the username.
- **READMEs** for all seven sections with dataset URLs and licences, plus
  `human_ratings/README.md` for each.
- **Raw ratings staged** to `_raw_ratings_DO_NOT_COMMIT/`, 35 files, 2.2 MB.
- **§3.6 migrated in full** (09-07). Two scripts in `train/`, two modules in
  `inference/`, two re-keyed banks plus three IRs in `weights/`, pseudonymised
  and re-keyed ratings, a smoke test of 60 checks, the whole ladder rendered as
  samples. See §14.
- **§3.5 migrated in full** (09-07). Four scripts in `train/`, three modules in
  `inference/`, three artefacts in `weights/`, four pseudonymised data files in
  `human_ratings/`, a smoke test of 87 checks, two sample CSVs. See §13.
- **§3.4.2 migrated in full** (09-07). Six scripts in `train/`, four modules in
  `inference/` (this section DOES reach the runtime), three artefacts in
  `weights/`, pseudonymised ratings plus MERT features in `human_ratings/`, a
  smoke test of 71 checks, four samples. See §12.
- **§3.4.1 migrated in full** (09-07). Ten scripts in `train/`, an empty
  `inference/` with a README saying why, five artefacts in `weights/`, a smoke
  test of 63 checks, four committed samples, and a README carrying the
  reference numbers. Dry-run against real data; see §11.

### §3.3, file by file

| New | From | Notes |
|---|---|---|
| `train/build_nsynth_prior.py` | `s01_select_nsynth_strings.py` | |
| `train/analyse_nsynth_timbre.py` | `s02_analyze_nsynth_timbre.py` | not in the report's 24-file list, but the chain breaks without it |
| `train/generate_preset_bank.py` | `s03_generate_ddsp_dataset.py`, sampling half | |
| `inference/theta_render.py` | `s03_...`, render half | the split |
| `common/{paths,device,ddsp_synth}.py` | `paths.py`, `soundscape_synth.get_device`, `ddsp_synth.py` | |

Verify it still works:

```bash
E=/cs/student/projects3/COMP0158/grp_1/bin/env/env_linux_nvidia/bin/python
$E code/models/3.3_drone_synthesis_and_nsynth_prior/smoke_test.py   # expect exit 0
```

## 5. Decisions taken — do not re-open

| # | Date | Decision |
|---|---|---|
| 1 | 08-28 | The Bradley-Terry preference GP **stays in the conductor**, even though Chapter 3 does not describe it. The shipped system may do more than the report writes up. Do not let a tidy-up delete it. |
| 2 | 08-28 | The conductor ships **no rating data**. Four boot-time fits become pre-computed weights. |
| 3 | 08-28, 09-06 | **Bed audio ships**, with attribution. `LICENSE` covers code only; `ATTRIBUTION.md` governs `conductor/assets/`. |
| 4 | 09-06 | **Layout**: section-numbered model directories, four subdirectories each. `conductor/` keeps its internal split, because that is what enforces the inference boundary. |
| 5 | 09-06 | **All checkpoints ship**, runtime and training. Melody: all three transformers plus the Markov tables, because the UI offers a choice. Training: `judge_proxy.pt`, `judge_proxy_e0.pt`, `closed_loop_mapper.pt`, `inverse_cvae.pt`, `attn_retrieval.pt`, about 2.6 MB, so §4.1 and §4.2 are reproducible. None of those five is referenced in any conductor tree, so they must **not** be wired into `conductor/weights/`. |
| 6 | 09-06 | **Rating CSVs ship, scrubbed, in their section's `human_ratings/`.** Staged raw first, never edited in place. |
| 7 | 09-06 | **Generated output goes to the section's own `data/`**, git-ignored, with three or four sample outputs committed under `data/samples/`. |
| 8 | 09-06 | **Naming**: no personal names, no `JAMAI`, no tool names such as `Claude` in paths, no step-number prefixes. Environment variables are `DRONE_*`. |
| 9 | 09-06 | **The measured timbre prior ships**, unchanged. `3.3/weights/timbre_prior/frames.npz`, 29 MB, sha256-verified identical to the source. Shipped at full float32 rather than halved to float16, so the artefact is bit-identical to the one that produced the results. §3.3 then runs with no NSynth download. |
| 10 | 09-07 | **`code/README.md` ships.** It is rewritten from a migration plan into a reproduction guide addressed to a human grader and to a future Claude session: what each section does, how to run it, what it needs, what it produces. Consequence: it must **pass** the leak gate, so `--exclude=README.md` comes off, the five references to the private source tree go, and the banned strings it currently quotes as rules are re-expressed without naming them. |
| 12 | 09-07 | **`conductor/` is SELF-CONTAINED.** It carries its OWN copy of every weight, parameter file and asset it loads, rather than importing or symlinking from `models/<section>/`. The directory is meant to stand alone: someone should be able to take `conductor/` and run it. Consequence: artefacts are duplicated on purpose (reverb banks and IRs, `preference_gp.npz`, `boundary_guard.json`, `hsmm_transitions.json`, `arc_types.json`, the melody checkpoints), and duplication can drift — so packaging must VERIFY the copies are byte-identical to the section originals, by checksum, and fail if they are not. |
| 11 | 09-07 | **The melody-model discrepancy is not an error to fix.** All three transformers are offered in a UI drop-down, so deploying `L6_d128` does not contradict the parsimony argument for `L3_d128`. §10's live finding is closed; no report change. |

## 6. Verified — do not re-derive

- **Melody checkpoints.** `L3_d128` 612,387 params; `L6_d128` 1,207,203;
  `L6_d256` 4,773,667. All three match the thesis figures exactly.
  `melodic_transformer_cpu.pt` is **not a fourth model**: it is `L6_d128` under
  a promoted-copy alias, stated at `s02_conductor_app.py:1086`. Only
  `SideProjects/melodic_drone/GEMINI/checkpoints/` holds all three; every
  `FINAL_CONDUCTOR/` copy has just `L3_d128` and the alias.
- **The conductor loads exactly one `.pt`**, the melody transformer. The five
  training checkpoints appear nowhere in any conductor tree.
- **Participant names are real** in the rating CSVs across **eight** copies of
  the source tree. The `R01`…`R09` map already exists: the report's copy at
  `results_and_discussions/data/03_rating_reliability/data/valence_ratings.csv`
  is pseudonymised, has the same 713 rows in the same order, and identical
  per-rater counts (150, 150, 100, 77, 68, 62, 52, 43, 11), so the map is
  recoverable positionally.
- **Two rating studies are fragmented, and both have a double-count trap.**
  `space_ratings_*`: use `downloaded_fresh/` only, because `8c842679` grew from
  156 to 312 rows between pulls and three sessions exist only there.
  `longtrack_ratings_*`: use the parent `huggingface_ratings/`, not its `logs/`
  subfolder, which is missing `020fff1f.csv`.
- **The merge already exists.** `analyze_space_ratings.py:39` `load()` plus
  `build_results.py`. Verified: 14 shards hold 709 rows / 15 raters,
  `ratings_all_flagged.csv` holds 708 / 14, and the single difference is the
  `setup_check` smoke row that line 48 drops. `ratings_clean.csv` is the
  `clean == True` subset. **Writing a new merge produces a different dataset
  from the one the report analysed.**
- **§3.3 dry run** on real NSynth: 289,205 train notes scanned, **8,269**
  matched, which is exactly the thesis figure. 289,205 train + 12,678 valid +
  4,096 test = 305,979, which is the "complete NSynth" number in
  `appendix_datasets.tex:84`. Both correct, no discrepancy.

## 7. Two defects found in §3.3, both fixed, both worth re-checking elsewhere

1. **The noise floor was unseeded.** `--seed` reproduced theta sampling and the
   f0 wander, but the filtered noise comes from `torch.randn` inside
   `ddsp_synth.py:76`, which takes no generator. Two renders of the same theta
   differed by 0.04 on a signal peaking at 0.9, on CPU as well as GPU. Fixed by
   also calling `torch.manual_seed`. **Check this in every section that
   renders.**
2. **The recommended path crashed.** `build_nsynth_prior.py --metadata-only`
   leaves no wavs in its output directory, and `analyse_nsynth_timbre.py`
   defaulted its audio directory to that same directory, so it read nothing and
   died in `np.concatenate` after all the setup. Fixed by defaulting
   `--audio-dir` to `<nsynth-root>/audio` and failing early with a message.

## 8. Open questions for the user

None outstanding. Both former questions were answered 09-07; see decisions 10
and 11.

## 9. Next steps, in order

1. ~~**De-identify** the seed-valence set~~ **done 09-07** (§12); the tool is
   `_raw_ratings_DO_NOT_COMMIT/deidentify.py`, git-ignored. Still to do for the
   other five staged sets when their sections land.
   Original note: Apply the `R01`…`R09` map to the staged ratings, re-key the
   reverb bank off artist names (`code/README.md` §0.2), strip absolute paths
   (§0.4). Everything downstream copies these files, so this is first.
   Three name sites that must change together, because one is a lookup key:
   `s15_ladder_human_report.py:75` `by_rater["matthew"]`,
   `figs/ladder_human.json` `readback` field, `build_results.py:31` comment.
   `build_results.py:33` also has a real handle in `COMPLETERS`.
2. ~~Migrate §3.4.1~~ (`2a95564`), ~~3.4.2~~, ~~3.5~~, ~~3.6~~ **done 09-07,
   uncommitted.** Then 3.7, 3.9. Each: copy, drop step
   prefixes, split train from inference, dataset roots as arguments, run
   instructions in the docstring, README, smoke test, **dry run**, four sample
   outputs.
3. **Four pre-fit scripts** into `conductor/weights/`: boundary guard,
   preference GP, melody anchor, bed veto. This is what lets the conductor ship
   with no rating data.
4. **Migrate the conductor**, then run `s06_pack_demo.py`'s `verify()` with
   every dataset root pointed at `/nonexistent`. **Rename `JAMAI_DATA` in that
   script when you do**, or the test passes by reading nothing.
   Per decision 12 the conductor carries its own copies of every weight and
   asset, so packaging must also checksum them against the section originals
   and fail on drift. §3.6's three IRs and both reverb banks are the first
   case: `reverb_bank.py` resolves IR paths relative to its own `weights/`, so
   a copied tree works unchanged — that resolution rule exists precisely to
   make a self-contained copy possible.
5. Licence and attribution split, then packaging extras.
6. **Rewrite `code/README.md` as the shipped reproduction guide** (decision 10).
   Last, because it can only describe sections that exist. Then drop
   `--exclude=README.md` from the §9a leak gate and re-run it.

## 9a. Exactly how to resume

```bash
cd /cs/student/msc/dsml/2023/myeung/THESIS_MYDIR/COMP0158_SupplimentaryMaterial
E=/cs/student/projects3/COMP0158/grp_1/bin/env/env_linux_nvidia/bin/python

# 1. confirm nothing rotted
$E code/models/3.3_drone_synthesis_and_nsynth_prior/smoke_test.py    # expect exit 0

# 2. confirm the shipped prior alone reproduces the samples, no dataset needed
$E code/models/3.3_drone_synthesis_and_nsynth_prior/train/generate_preset_bank.py \
   --n 4 --duration 8 --seed 0 --out-dir /tmp/check

# 3. the leak gate. Must print nothing.
#    STATUS.md and code/README.md are excluded on purpose: they are the
#    documents that STATE the rules, so they quote every banned string.
#    code/README.md's exclusion is TEMPORARY: decision 10 ships that file, so
#    step 6 rewrites it and this --exclude must then come off.
grep -rIn --exclude=STATUS.md --exclude=README.md \
     -e JAMAI -e jamai -e Matthew -e matthew -e maatt -e Claude -e Gemini \
     code/ env/
grep -rIn -e JAMAI -e jamai -e Matthew -e matthew -e maatt README.md
```

One deliberate exception survives and is not caught by the above:
`code/common/paths.py`'s `DEFAULTS` records the build-host paths, which carry
the username. That is decision 8's stated exemption. Everything else must be
clean.

Then work `code/README.md` §2.3 for §3.4.1, following §1a's conventions.

## 14. §3.6, and the second artist-name blocker

**§0.2 is now fully closed.** §3.5 killed the hardcoded `ARTIST_DIRS` map; this
closes the other half, the reverb bank. Two ladder condition ids WERE artist
names. They are re-keyed onto the measurement -- `long_bright` (5.05 s,
2434 Hz) and `long_dark` (5.16 s, 438 Hz) -- which is what the experiment
actually manipulates. The re-key is DERIVED from each category's own measured
medians (tail and tone buckets), not hand-assigned, and the tool aborts if the
mapping is not one-to-one. Track names and paths are dropped entirely from the
measured bank; only the measurements survive. The ratings' `reverb` column was
re-keyed in the SAME operation, including compound ids, because the ratings
must join to their stimulus.

**A pandas bug the audit caught, affecting every earlier section.**
`df[col].dtype == object` NEVER FIRES on this pandas: text columns come back as
a dedicated `str` dtype, so the path-rewriting step was silently skipped in all
three scrubbers. Absolute paths had been surviving into shipped files. Fixed
with a `scrub_text_columns` helper that tests numeric-ness instead, and every
section re-scrubbed and re-verified against the report's copies. This is exactly
what the audit step exists for.

**A real defect in the study's loudness control, kept and documented.**
Conditions are RMS-matched to dry so a level difference cannot masquerade as a
reverb effect -- but the peak guard runs AFTER the match and pulls a clipping
condition back down, undoing it. On the smoke test's source it costs
`stairwell` **-4.69 dB**, the longest real space, which is where a level
confound matters most. NOT fixed: these are the stimuli the listeners actually
heard, and re-levelling now would make the shipped code render something they
did not hear. The fix for a future round is one common headroom scalar across
all conditions. The smoke test asserts the MECHANISM (any deviation must be
downward and coincide with the peak ceiling), because which conditions bind
depends on source level.

**Verified.** The three shipped IRs reproduce their banked RT60 and centroid
exactly (0.952/3036.9, 0.294/2653.8, 2.388/2684.7); IR-supervised fitting
recovers decay 0.25 s, damping 900 Hz and gain 0.60 from a planted target; the
tone-matched pair holds at 0.11 s tail difference and a 5.6x tone ratio.

**`weights/` holds three .wav files, and that is correct.** For the IR
conditions the WAV IS THE MODEL: the runtime convolves the actual measurement,
and the fitted params only choose which one represents a category. §1a's rule
is "artefacts that ship", not "only numbers". Putting them elsewhere would give
the section a runtime dependency outside `weights/`, and decision 12's
self-contained conductor copy would then need two source directories.

**Gap found and closed (user's question, 09-07):** §3.6 was shipping
third-party audio with NO attribution, while every other section carries a
licence table and §0.3 names those three EchoThief IRs explicitly. Added a
licence table to the section README and `weights/irs/ATTRIBUTION.md` beside the
files, so a copied `weights/` carries its own credit. The smoke test now
asserts the attribution exists, names the source and URL, and lists every
shipped IR.

**Report-side finding:** the thesis's own data directory still ships
`reverb_ratings.csv` with the original artist-derived condition ids, which
`appendix_datasets.tex`'s scrubbing claim covers. Report-side fix, not code.

## 13. §3.5, and the global rater map

**The report keeps ONE global rater numbering across studies**, and the tool now
matches it. `deidentify.py` derives a map from EVERY file the report also ships
pseudonymised, merges them, and aborts if two disagree. R03 is the primary rater
in seed-valence, both arc files and texture. New arc raters are R11..R26 -- the
report's own numbering, not invented here. Scrubbed arc files are byte-identical
to the report's copies (17/17 and 16/16 columns, rater included), and §3.4.2's
file is unchanged by the generalisation.

**Two files beyond §2.5's mapping, both required**: `texture_ratings.csv` (122
rows, feeds the GP's absolute term) and `arc_pool_meta.json` (230 arcs, the
feature vocabulary; carried 230 absolute paths, scrubbed).

**The preference GP is split like the guard.** `s01_arc_policy.py` imported the
training modules and refit the GP live, which breaks the runtime boundary.
`train/fit_preference_gp.py` now freezes the Laplace posterior to
`weights/preference_gp.npz` and `inference/arc_policy.py` loads it. Verified:
the frozen posterior reproduces `f` and `diag(Sigma)` at training rows to
4.9e-4 / 7.0e-4 (the original's own 1e-3 tolerance) and batch-vs-single to
3.6e-15.

**DECISION REVERSED 09-07, at the user's instruction: the HSMM fitter SHIPS,
generalised.** §2.5 had it excluded because it ran on a commercial corpus. The
method is not the corpus. BOTH halves now migrate and take `--audio-dir` as a
list, so the timing model retrains on any audio -- each directory becomes a
group label, walked recursively:

  `s07b_transition_typology.py` -> `train/extract_transition_typology.py`
  `s13_corpus_hsmm_fit.py`      -> `train/fit_corpus_hsmm.py`

This also kills **§0.2's artist-name blocker at the root**: `ARTIST_DIRS` was a
hardcoded artist-name-to-directory map, now replaced by whatever the caller
passes. `artist` became `group` throughout.

Verified two ways. (1) Refitting from the ORIGINAL descriptor cache reproduces
the shipped `hsmm_transitions.json` with **zero** differing fields (only
`n_artists` -> `n_groups`, the deliberate rename). (2) The generic path runs end
to end on three arbitrary directories: 6 tracks -> 256 transitions -> a full
k=3 fit with its own dwell windows and a return rate of 0.430, against 0.039 on
the shipped corpus.

The corpus audio still does not ship. `weights/SHA256SUMS` records the fitted
artefact's checksum. The s10 overlay-DSP dependency was dropped with the batch
renderer; the scheduler core needs no audio.

**Verified reproduction.** Tie rate 10.8%, Davidson nu 0.245, sign accuracy
65.4%, 219 absolute ratings over 51 scenes; 293 seed pairs (96/96/96 + 5
probes); rating target **125 of 711** possible pairs, both matching the numbers
in the original docstrings; corpus active dwell (2.0, 28.6)s floored to
(20.0, 28.6)s; live-vs-batch envelope equality to 1.2e-15, which is the
join-click test the original called out.

## 12. §3.4.2, de-identification, and a correction

**§9 step 1 is done for the seed-valence set.** The `R01`..`R09` map was
recovered positionally against the report's own pseudonymised copy (same 713
rows, same order, every non-rater column identical), verified one-to-one, and
written to `_raw_ratings_DO_NOT_COMMIT/rater_map.json`, which is git-ignored
along with the `deidentify.py` that produced it. Scrubbed: rater ids, absolute
clip paths, `bank_files` inside `pool_meta.json` (a LIST of paths -- walking
only top-level strings missed 3 per entry, 450 in all), and `wav_dir` in
`pool_mert_meta.json`. The scrub is **behaviour-preserving**: the migrated IRR
script on the pseudonymised CSV reproduces alpha = **0.096**, the stratified
0.366 / -0.135, and unanimity 9/124 exactly.

**CORRECTION to §6 and §10: the theta-KRR label source is `R03`, not `R01`.**
The claim that it is one rater (N=1) is right; only the pseudonym was wrong.
`"matthew"` never appears in the CSV -- it is synthesised by a hygiene rule that
remaps a session-test account's real-day rows, and that account is `R03`.
Checked, not assumed: propagating R03 gives MAE 0.566, rho **0.337**, sign
**0.69**, exactly the report's row; R01 (who also rated all 150) gives 0.603 /
0.211 / 0.64. `code/models/3.4.2.../human_ratings/README.md` has been corrected.

**Verified reproduction.** alpha 0.096; theta-KRR winner RBF KRR alpha=1.0
gamma=0.003 with MAE 0.566 / rho 0.337 / sign 0.69 over the 20,000-anchor bank,
all 150 clips verified against their anchor rows; guard tau **0.5898** against
the 0.590 recorded in the original; judge valence ceiling **+0.244** against the
human axis's **+0.572**, which are the report's "+0.24" and "[-1.0, +0.57]".

**Design decisions taken.**
1. The guard splits so that **no rating data ships**. Posterior sigma depends
   only on the rated COORDINATES and the kernel, never on the values, and the
   live app reads sigma/tau/safe_points only. `weights/boundary_guard.json`
   carries X, kernel, K_inv, tau and the safe set -- no ratings. The runtime
   recomputes the posterior in numpy with no sklearn, and `fit_guard.py`
   refuses to write a guard the runtime cannot reproduce (agrees to 3e-15).
2. `inference/` must not import `paths` or resolve a dataset root; bank indexes
   are passed in. The smoke test asserts it on all four modules. This is what
   keeps the conductor startable with every dataset missing.
3. The control ridge ships as `.npz`, matching §3.4.1's judge heads. Bit-exact
   on the shipped float64 features.

**Four files beyond §2.4's mapping**, all blocking something: `decoupled_engine.py`
(the bank loader, also in 3 conductor trees), `pool_mert.npz` + `s05_extract_mert.py`
(the control ridge is unrunnable without the features, and the pool audio is not
shipped), and `s11_train_attention_retrieval.py` (trains the shipped
`attn_retrieval.pt`). User approved shipping the npz + extractor, and the npz
form of the control ridge.

**Canonical copies.** `s06_boundary_guard.py` and `step01_gp_softknn_engine.py`
each exist in TWO versions: the `Phase2/` originals match all three
`FINAL_CONDUCTOR/` trees, while the `Phase3/jamai_demo*` copies are older and
shorter. §2.4's Phase2 mapping points at what shipped.

**Control-ridge result** (the amendment-5 disambiguation): MERT embeddings ->
human valence is at chance. Best LOOCV rho 0.056 linear / 0.106 RBF, against a
permutation null with median |rho| 0.091 and p95 0.216. theta-space propagation
reaches 0.337. So parameters carry the human valence signal and the semantic
embeddings do not -- which is what `methodology_stage.tex:65` argues.

## 11. §3.4.1, and what the dry run established

**Verified reproduction.** The port is not merely equivalent, it is bit-exact
where it can be:

- `label_dataset.py` re-labels the 60-clip calm probe to **byte-identical**
  values in all four columns, against the committed `labeled_index.csv`.
- The cycle chain reproduces exactly: valence MAE **0.337**, arousal MAE
  **0.536**, Euclidean **0.678**, arousal bias **+0.380**. All 60 rendered wavs
  are byte-identical to the originals'.
- `cross_domain_indomain.py --n 150` returns **+0.72 / +0.85** for the
  home-domain heads, which are the two numbers hard-coded into the original
  script's figure title.
- The proxy trainer's member-0 validation MSE matches the original run.

**Four defects found and fixed.**

1. **The float32 discovery.** `Ridge.predict` computes in **float32** when
   handed a float32 array, and MERT embeddings are float32, so the published
   labels were produced in float32. Reconstructing the heads in float64 shifted
   every label by ~2e-7 and broke bit-exactness. `RidgeHead.predict` now
   reproduces sklearn's dtype behaviour on purpose. **Check this anywhere a
   fitted sklearn model is reconstructed from arrays.**
2. **`common/paths.py` DEFAULTS were wrong for the build host** — `deam`,
   `emo_soundscapes` and `echothief` all pointed at directories that do not
   exist. Fixed to `DEAM`, `Emo-Soundscapes`, and
   `reverbs/EchoThiefImpulseResponseLibrary`. §3.3 never noticed because it only
   reads `nsynth`. **`essen` is still wrong/absent** — resolve it when §3.7
   lands; the comment says music21 supplies it.
3. **The §3.3 noise-seed defect recurred.** `resolution_experiment_v2.py` seeded
   numpy but not torch, so its 75 clips were never reproducible. Seeded, two
   runs are byte-identical. Regenerated audio differs from the shipped clips by
   up to 0.019 on a 0.88 peak, but condition means agree to 0.002 and
   inter-judge r to 0.004, so the reported conclusion is untouched.
4. **Training scripts defaulted their output into the committed `models/`
   directory.** All three now default `--out-dir` to the section's `data/`, and
   the smoke test asserts it. This is the §1a rule about `weights/` made
   mechanical.

**Two extra decisions, both taken 09-07 with the user.** The three frozen ridge
heads live OUTSIDE the migration source tree, at
`THESIS/code/soundscapes/models`, and nothing in §3.4.1 runs without them: they
now ship as one 20 KB `weights/affect_ridges.npz` rather than as pickles.
`cross_domain_validity.py` had an unmapped input, so
`resolution_experiment_v2.py` was migrated too, as
`train/render_resolution_probe.py` — one file beyond §2.3's mapping.

**An incident worth not repeating.** A dry run of the ORIGINAL `s08` was given
no `--out-dir`, and its default is the committed `models/` directory, so a
2-epoch toy fit overwrote the real `judge_proxy.pt` and `judge_proxy_e0.pt`.
Both were restored from `RERUNS/20260905_prior_init/models/judge_proxy.pt`
(byte-verified for `judge_proxy.pt`; `judge_proxy_e0.pt` reconstructed from it
by the `shutil.copyfile` at `s08:239`, which makes the two identical by
construction — sound, but no independent copy survives to prove it).
`closed_loop_mapper.pt` escaped only because s09 saves from step 100 and the run
was 20 steps. **Always pass `--out-dir` when dry-running the originals.**

## 10. Report-side findings this migration produced

Logged in `code/README.md` §7. The live one:

**The conductor deploys the six-layer melody model while
`melodic_drone_and_sequence_generation.tex:44` says the three-layer was chosen
on a parsimony argument.** Confirmed by measurement: 1,207,203 parameters,
`layers.0`…`layers.5`, and `PREFERRED_CKPT = "L6_d128"`. The report's own
figures make the argument thin anyway, 1.5202 against 1.5194 nats. This is a
factual error in Chapter 3 that the whole-report review missed.

Also logged there, and **not** an error despite an earlier note saying so:
`appendix_datasets.tex:33`'s `N=1` is the theta-KRR fitting set, R01 alone, and
`human_grounding_and_retrieval.tex:19` already says so. The 9-rater figures are
the separate reliability study.
