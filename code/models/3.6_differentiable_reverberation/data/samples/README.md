# Sample outputs — the reverb ladder

The six ladder conditions applied to one plain harmonic source, plus the
late-field variant. Everything here regenerates from this repository alone: the
three impulse responses and both banks ship.

| file | tail | tone | applied by | length | level vs dry |
|---|---|---|---|---|---|
| `dry.wav` | — | — | — | 4.00 s | 0.00 dB |
| `nature.wav` | 0.29 s | 2654 Hz | measured IR | 4.36 s | +0.00 dB |
| `brutalism.wav` | 0.95 s | 3037 Hz | measured IR | 5.05 s | −0.00 dB |
| `stairwell.wav` | 2.39 s | 2685 Hz | measured IR | 6.27 s | **−3.51 dB** |
| `long_bright.wav` | 5.05 s | 2434 Hz | fitted decay | 9.02 s | −0.77 dB |
| `long_dark.wav` | 5.16 s | 438 Hz | fitted decay | 9.02 s | −0.30 dB |
| `stairwell_late.wav` | 2.39 s | 2685 Hz | late field only | 6.27 s | — |

`ladder.csv` holds these rows machine-readably, and
`stimulus_conditions.json` records the exact ladder — the constant wet level,
the source release, and every resolved condition — as a reproducibility
sidecar.

## What to listen for

The last two files are the point of the whole ladder. They sit at essentially
the same tail length, 5.05 s against 5.16 s, and differ by more than five times
in tone, 2434 Hz against 438 Hz. Any difference you hear between them is
BRIGHTNESS, not length. Without that pair, tail and tone are confounded, because
the long spaces in this material also tend to be dark.

`stairwell_late.wav` drops the direct sound and early reflections and keeps only
the diffuse tail. It is smooth where the plain version is grating: two
correlated copies of the same signal a few milliseconds apart comb-filter, and
removing the coherent direct sound removes the interference.

## The level column, which is a caveat not a curiosity

Conditions are deliberately RMS-matched to dry, so that "more reverb sounded
less aroused" cannot be an artefact of a quieter clip. Four conditions match to
within 0.01 dB on the smoke test's source. But the peak guard runs after the
match and pulls a clipping condition back down, which on this louder source
costs `stairwell` 3.51 dB.

That is a real limitation of the study's loudness control, documented in
`inference/reverb.py`'s `_match_level`, and it is preserved rather than fixed
because these are the stimuli the listeners actually heard.

## Reproducing

```bash
python ../../inference/reverb_bank.py --list       # the ladder in force
python ../../inference/reverb_bank.py --selftest   # its invariants
```
