# How this code is organised, and why

The reproduction guide — environments, dataset paths, and the command for
every stage — is the [top-level README](../README.md). This document is the
other half: the handful of rules the whole tree obeys, why each exists, and
where each is enforced. Read it if you are extending the code, reviewing it,
or trying to work out why something is split the way it is.

Each section's own README explains what that section *does*. Nothing here
repeats them.

---

## The shape

Seven model directories, one per report section, plus the runtime:

```
common/     shared by everything: the DDSP synth, device selection, dataset
            paths, the loudness protocol
models/3.x/ one directory per report section
conductor/  the live application (§3.8)
verify.py   the packaging check
```

Every model directory has the same five subdirectories, and the names are
promises rather than conveniences:

| | |
|---|---|
| `train/` | fitting. Long-running, usually needs a dataset. |
| `inference/` | what the runtime imports. No dataset, no fitting, no rating data. |
| `weights/` | the fitted artefacts this section produced. |
| `human_ratings/` | the pseudonymised ratings this section used, with a README describing the study. |
| `data/` | generated output. Git-ignored; a few committed examples live in `data/samples/`. |

**An empty directory is a finding, not an omission — so it stays, carrying a
README that says why.** Section 3.4.1 ships no inference module because
nothing from the failed automated loop reached the running system; that
absence *is* the result reported in §4.1. Section 3.9 ships neither an
inference module nor a weight, because it measures rather than synthesises.

Deleting those directories instead would be tidier and worse: a reader
scanning seven sections could not tell a deliberate absence from a packaging
slip, and the smoke tests assert both that they stay empty and that they stay
explained.

---

## Rule 1: training and inference are separated, and the separation is tested

**`inference/` may not import a dataset path, fit a model, or read rating
data.** It loads frozen artefacts and computes. That is the difference between
a runtime that behaves identically everywhere and one whose behaviour depends
on what happens to be on the machine.

This is not a convention anyone remembers to follow. It is checked:

* every section's `smoke_test.py` parses the module-scope imports of its
  `inference/` modules with `ast` and fails if `paths` appears;
* `conductor/smoke_test.py` goes further and imports the whole engine with the
  repository removed from `sys.path` entirely, so anything reaching back into
  the tree fails there rather than in a grader's clone.

Both were necessary. The melody runtime had been importing its model class
from the *training* script in order to rebuild an architecture before loading
a checkpoint, which the static check now forbids. And a deferred import inside
a function — invisible to the static check — was still bootstrapping the
original development tree; only the `sys.path` test caught it.

### What it costs, and why it is worth paying

Splitting a module is real work. `s`-prefixed research scripts tended to fit
and apply in the same file, so the boundary runs through them:

| Section | Fitting stays in `train/` | The runtime gets |
|---|---|---|
| 3.4.2 | the boundary guard's Laplace fit | `guard.py`, which loads the fitted JSON |
| 3.5 | the preference GP's Bradley–Terry fit | `arc_policy.py`, which loads the posterior |
| 3.6 | the gradient-descent IR fit | `reverb.py`, which convolves |
| 3.7 | the grammar and transformer training | `grammar.py`, `transformer_model.py` |

The payoff is that `conductor/` starts with every dataset root pointing at a
non-existent path, and carries no participant data at all.

---

## Rule 2: nothing is fitted at startup

Four things used to be fitted when the application booted. Each has become a
pre-computed artefact:

| Was fitted at boot | Now ships as |
|---|---|
| the Bradley–Terry preference GP over rated arcs | `preference_gp.npz` |
| the boundary guard, once per valence label space | `boundary_guard.json`, `boundary_guard_human.json` |
| the melody timbre anchor, chosen from the rated seed pool | `melody_anchor.json` |
| the background bed bank, from the curation log | `bed_bank.json` |

Two reasons, and the second is the one that bites.

**It removes the rating data from the runtime.** All four fits read
participant ratings, so shipping them meant shipping the ratings.

**A fit that cannot run does not fail — it returns something else.** The
melody anchor is the worked example. Choosing it needs §3.4.2's rated pool;
the conductor ships none, so the refit fell back and resolved a *different*
anchor from the one the report names — the fourth-ranked candidate, carrying
fifty times the swell of the intended one. The deployed melody wobbled in
exactly the way the selection filter exists to prevent, and nothing logged a
problem. A frozen artefact cannot do that.

---

## Rule 3: the conductor is self-contained, and its copies are checksummed

`conductor/` carries its **own copy** of every inference module and weight it
uses, rather than importing across the tree, so the directory can be lifted
out and run.

Duplication drifts, and drift here is silent: a fix made in a section fails to
reach the application, the application keeps working, and nobody notices. So
`conductor/engine/COPIED_FROM` pins every copy to its source by checksum and
`verify.py` fails when one stops matching. It separates the two cases, because
they need opposite fixes:

* **a copy edited in place** — the edit will be lost on the next re-copy;
* **a source edited without re-copying** — the application is running old code.

Never edit a file listed in `COPIED_FROM`. Edit the section original and
re-copy. The check is deliberately exercised: both failure modes were induced
on purpose and confirmed to fail the build, because a drift check that has
never failed is not known to work.

---

## Rule 4: datasets are located, never assumed

No script hardcodes a path. `common/paths.py` resolves each dataset in a fixed
order — an explicit flag, then an environment variable, then a recorded
default — and a miss raises a message naming the dataset and where to obtain
it rather than a bare `FileNotFoundError`.

| Dataset | Variable |
|---|---|
| NSynth | `DRONE_NSYNTH` |
| DEAM | `DRONE_DEAM` |
| Emo-Soundscapes | `DRONE_EMO` |
| Essen Folksong Collection | `DRONE_ESSEN` |
| EchoThief impulse responses | `DRONE_ECHOTHIEF` |
| (parent of all of the above) | `DRONE_DATA` |

`paths.py` is the one file allowed to record a build-host default, because
that is what lets the pipeline run unattended on the machine the results came
from. It is a stated exemption in `verify.py`, not an oversight — and nothing
under `inference/` or `conductor/` imports it.

**No corpus is redistributed.** Each is documented with its licence and a URL,
and the code fetches or rebuilds. The exceptions are deliberate and small: the
measured timbre prior (a measurement, not audio), three impulse responses, and
one background bed per type so the conductor is not silent out of the box —
each credited in `ATTRIBUTION.md`.

---

## Rule 5: the ratings are pseudonymous, and the raw data never enters the tree

Every rating file ships with participants replaced by stable pseudonyms under
one global map, so a rater who appears in several studies is recognisably the
same person across them without being identifiable.

Two further points matter for reading the report against the data:

**Raw ratings are staged outside the repository and are never edited in
place.** The de-identification tool reads the raw copies and writes new ones;
the originals are never touched, so a scrub can be re-run and audited.

**Two reverb conditions were named after the commercial albums they were
measured from, and have been re-keyed onto descriptors of the measurement.**
The ratings and the reverb bank were re-keyed in the same commit, because a
stimulus id that no longer joins its bank fails silently rather than loudly.
The two section READMEs that document this mapping are exempt from the
identifier sweep for that reason, and only for that reason: explaining a
rename requires naming what was renamed.

Reproducing a report table that uses the old ids means mapping them forward.
**No numeric value changed.**

### Reading a CSV in this tree

Read rating files as text — `dtype=str, keep_default_na=False`. Type inference
is not free here. One session identifier in the long-track study is valid
scientific notation, and a plain `read_csv` parses it as a float that
overflows to `inf`, destroying that session's identity and half of the merge's
de-duplication key. The same round trip silently dropped the last significant
digit of every float column.

---

## Verifying the package

```bash
python code/verify.py            # everything
python code/verify.py --quick    # skip the section smoke tests
```

It runs the eight smoke tests, sweeps every shipped text file for participant,
artist and build-host identifiers, checks that every command in the top-level
README resolves to a real script with real flags, verifies the shipped weights
against their checksums, and confirms the conductor's copies still match their
sources. No dataset, no network, no GPU.

The README audit exists because documentation that does not run is worse than
none: a reader copies a command, it fails, and they cannot tell whether the
instruction or the code is wrong.

---

## Where to look next

| | |
|---|---|
| Running any of it | [top-level README](../README.md) |
| What a section does, and its reference numbers | that section's `README.md` |
| The listening studies and their data | [`models/3.9_…/human_ratings/`](models/3.9_human_evaluation_and_protocols/human_ratings) |
| The live application | [`conductor/README.md`](conductor/README.md) |
| Licences and credits for everything not written here | [`ATTRIBUTION.md`](../ATTRIBUTION.md) |

Each section README carries a **reference numbers** table: the values that
section reproduces, so you can tell at a glance whether a run came out right.
`python code/verify.py` checks the ones that can be checked without a
dataset.
