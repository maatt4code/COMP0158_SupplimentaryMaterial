# Sample output

Four clips so the section can be heard without running anything. They are
illustrative, not the shipped preset bank: the thesis bank is 20,000 clips and
is not redistributed here.

| clip | f0 (Hz) | swell rate (Hz) | swell depth | noise level | noise cutoff (Hz) | chord voices |
|---|---|---|---|---|---|---|
| `ddsp_00000.wav` | 123.0 | 0.178 | 0.84 | 0.33 | 216 | root, fifth, octave |
| `ddsp_00001.wav` | 87.6 | 0.013 | 0.37 | 0.00 | 440 | root, minor third, fifth, octave |
| `ddsp_00002.wav` | 105.9 | 0.444 | 0.62 | 0.17 | 5629 | root, minor third, octave |
| `ddsp_00003.wav` | 184.0 | 0.030 | 0.62 | 0.09 | 1060 | root, major third, fifth |

8 seconds each, 16 kHz mono, peak-normalised to 0.9. `theta_index.csv` holds
the full 41-dimensional theta for all four, one row per clip: 32 harmonic
weights plus the nine scalars, and the tilt that shaped the harmonics.

Together they cover the range the sampler draws from. `ddsp_00001` is nearly
static, swelling once every 77 seconds with no noise floor at all;
`ddsp_00002` breathes about every 2 seconds and opens its noise cutoff to
5.6 kHz, which is the bright, airy end. Three of the four carry a third, so
they are chords rather than single drones.

## Reproducing them

One command. The measured timbre prior ships in `../../weights/timbre_prior/`,
so NSynth is not needed:

```bash
python train/generate_preset_bank.py --n 4 --duration 8 --seed 0 --out-dir data/samples
```

Both RNGs are seeded (numpy for theta and the f0 wander, torch for the noise
floor), so this reproduces the clips exactly on the same machine.

To rebuild the prior itself from NSynth instead, which takes a GPU pass over
8,269 notes:

```bash
python train/build_nsynth_prior.py --metadata-only
python train/analyse_nsynth_timbre.py
```
