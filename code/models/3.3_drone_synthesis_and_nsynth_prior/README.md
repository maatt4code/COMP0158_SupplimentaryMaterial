# 3.3 Parametric Drone Synthesis and NSynth Prior

Measures an empirical timbre prior from NSynth bowed strings, then samples and
renders the preset bank the rest of the system navigates.

## Datasets

| Dataset | Used for | Licence | Where to get it |
|---|---|---|---|
| **NSynth** (`nsynth-train.jsonwav`) | the empirical timbre prior: 8,269 sustained acoustic bowed-string notes | CC BY 4.0 | <https://magenta.tensorflow.org/datasets/nsynth> |

Point at it with `--nsynth-root`, or set `$DRONE_NSYNTH`. No NSynth audio is
redistributed here.

**You do not need NSynth to use this section.** The measured prior it produces
ships in [`weights/timbre_prior/`](weights/timbre_prior): `frames.npz`, 29 MB,
311,296 analysis frames from 8,269 notes, plus `notes_index.csv`. Steps 1 and 2
below only need running if you want to rebuild it.

`frames.npz` holds four parallel arrays, one row per frame:

| Array | Shape | What it is |
|---|---|---|
| `harm_dist` | (311296, 32) | the 32 harmonic amplitudes, each row summing to 1 |
| `loudness` | (311296,) | frame RMS, kept separate so timbre shape is loudness-invariant |
| `pitch` | (311296,) | MIDI pitch of the source note, 40 to 72 |
| `velocity` | (311296,) | NSynth's five velocity layers |

The sampler reads only `harm_dist` and `pitch`, the latter to restrict draws to
the bottom octave of the register: 114,722 of the 311,296 frames, 37%. Because
every row sums to 1, each is a point on the 31-simplex, which is what lets
log-amplitude interpolation between two of them stay a valid distribution.

## Run order

1. **`train/build_nsynth_prior.py`** selects the sustained, acoustic, bowed
   string subset and writes `selected_metadata.json`.

   ```bash
   python train/build_nsynth_prior.py                    # dry count, writes nothing
   python train/build_nsynth_prior.py --metadata-only    # index in place
   ```

   Prefer `--metadata-only`. `--copy` duplicates thousands of wavs, which runs
   into inode quotas for no benefit.

2. **`train/analyse_nsynth_timbre.py`** measures harmonic amplitudes per frame.
   f0 is known from the MIDI pitch label, so this is a direct projection onto
   complex exponentials rather than an estimate. Notes are grouped by pitch so
   each group shares one basis and analyses in a single batched matmul.

   ```bash
   python train/analyse_nsynth_timbre.py --limit 50 --out-dir weights/timbre_prior   # quick test
   python train/analyse_nsynth_timbre.py --audio-dir $DRONE_NSYNTH/audio \
          --out-dir weights/timbre_prior                                             # full run
   ```

   Writes `frames.npz` (harmonic distribution, loudness, pitch, velocity per
   frame) and `notes_index.csv`. Only the sustained portion is analysed, 0.25 s
   to 2.75 s, skipping the bow attack and stopping before the release.

3. **`train/generate_preset_bank.py`** samples theta and renders.

   ```bash
   python train/generate_preset_bank.py --n 20 --duration 10 --out-dir /tmp/bank
   python train/generate_preset_bank.py --n 10000 --duration 10 --out-dir $DRONE_DATA/ddsp_chord10k
   python train/generate_preset_bank.py --n 10000 --duration 10 --calm --out-dir $DRONE_DATA/ddsp_calm10k
   ```

   Writes one wav per clip plus `theta_index.csv`, one row per clip holding all
   41 theta values.

## Inference

`inference/theta_render.py` turns an existing theta into audio. It samples
nothing, trains nothing, and resolves no dataset root.

```bash
python inference/theta_render.py --out demo.wav --f0 55 --duration 6
```

Used as a library by the generator and by the conductor:

```python
from theta_render import render_theta, polish, SR
audio = render_theta(synth, theta, n_samples, device, rng)
```

## Sample output

Four 8-second clips are committed under [`data/samples/`](data/samples), with
their theta and a README giving the commands that reproduce them. Everything
else written to `data/` is git-ignored.

## Smoke test

```bash
python smoke_test.py     # 26 checks, no dataset needed, exit 0 on success
```

It builds a synthetic timbre prior in a temporary directory, so it exercises
the real generator end to end without NSynth.

## Notes

- **Theta is 41-dimensional**: 32 harmonic weights plus nine scalars. `tilt` is
  applied to the harmonic distribution at generation time rather than carried
  as a separate coordinate, which is why it is 41 and not 42. `theta_index.csv`
  still records the tilt that was used, for analysis.
- **Register matching matters.** The prior is measured at MIDI 40–72, about
  82–523 Hz, but drones render at 33–131 Hz. Transposing a bright string
  spectrum down one or two octaves packs 32 strong harmonics into a low dense
  band, which buzzes. Two mitigations: frames are drawn only from the bottom
  octave of the prior's register, and a sampled spectral tilt darkens them.
- **`--calm` changes ranges, not call order.** The number and order of rng
  calls is identical to the default path, so default datasets stay
  bit-reproducible when the flag is added.
- **Seed both RNGs.** `--seed` covers theta sampling and the f0 wander. The
  noise floor comes from `torch.randn` inside the synth, which takes no
  generator, so this script also calls `torch.manual_seed`. Any other caller of
  `render_theta` must do the same. Without it two renders of the same theta
  differ by about 0.04 on a signal peaking at 0.9.
- **The split.** `s03_generate_ddsp_dataset.py` originally held both sampling
  and rendering. Sampling stayed in `train/`, rendering moved to `inference/`.
  The smoke test asserts the boundary by parsing imports, so the two cannot
  quietly grow back together.
