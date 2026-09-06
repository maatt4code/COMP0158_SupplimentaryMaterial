# STATUS — supplementary code migration

Last updated **2026-09-06, end of session**. Written so a fresh session on
another machine can continue without re-scanning the code tree. Everything
marked *verified* was derived from source in that session: do not re-derive it.

**Where we stopped.** §3.3 is complete and dry-run. Nothing is committed yet:
`git status` shows the whole migration as untracked. Next session starts at §9
step 1, de-identifying the staged ratings. The report side is separately
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
  README.md            migration plan, working document, probably not shipped
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

1. **Does `code/README.md` ship?** It is a working migration plan that quotes
   source-tree paths as evidence. The reader-facing document is the top-level
   `README.md`. My recommendation is that it stays out of the submission.

## 9. Next steps, in order

1. **De-identify.** Apply the `R01`…`R09` map to the staged ratings, re-key the
   reverb bank off artist names (`code/README.md` §0.2), strip absolute paths
   (§0.4). Everything downstream copies these files, so this is first.
   Three name sites that must change together, because one is a lookup key:
   `s15_ladder_human_report.py:75` `by_rater["matthew"]`,
   `figs/ladder_human.json` `readback` field, `build_results.py:31` comment.
   `build_results.py:33` also has a real handle in `COMPLETERS`.
2. **Migrate §3.4.1**, then 3.4.2, 3.5, 3.6, 3.7, 3.9. Each: copy, drop step
   prefixes, split train from inference, dataset roots as arguments, run
   instructions in the docstring, README, smoke test, **dry run**, four sample
   outputs.
3. **Four pre-fit scripts** into `conductor/weights/`: boundary guard,
   preference GP, melody anchor, bed veto. This is what lets the conductor ship
   with no rating data.
4. **Migrate the conductor**, then run `s06_pack_demo.py`'s `verify()` with
   every dataset root pointed at `/nonexistent`. **Rename `JAMAI_DATA` in that
   script when you do**, or the test passes by reading nothing.
5. Licence and attribution split, then packaging extras.

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
