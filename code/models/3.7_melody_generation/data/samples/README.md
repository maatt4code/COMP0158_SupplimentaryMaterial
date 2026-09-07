# Sample outputs — four melodies

Four 45-second melodies, rendered by the standalone voice in
`inference/melody_markov.py`. Everything regenerates from this repository: the
fitted grammar ships, and no corpus is needed.

| file | source | valence / arousal | scale | note | rhythm | rests |
|---|---|---|---|---|---|---|
| `phrase_bright.wav` | phrase bank | +0.6 / +0.3 | major pentatonic | 1.7 s | 2.0:1 | 11% |
| `phrase_calm.wav` | phrase bank | −0.4 / −0.7 | minor pentatonic | 2.9 s | 2.0:1 | 13% |
| `chain_bright.wav` | order-2 chain | +0.6 / +0.3 | major pentatonic | 1.7 s | 2.0:1 | 15% |
| `authored_calm.wav` | authored fallback | −0.4 / −0.7 | minor pentatonic | 2.9 s | 3.6:1 | 12% |

`melodies.csv` holds these rows machine-readably.

## What to listen for

**Valence picks the mode, arousal sets the pace.** That mapping is authored and
declared, not learned — folk songs carry no affect labels, so nothing here
claims to have learned an emotion. The two calm files sit at 2.9 s per note
against 1.7 s for the bright ones, and minor against major.

**The first two files are the reason the phrase bank exists.** They replay whole
contours a human wrote, so a figure is stated and then restated. `chain_bright`
generates note by note from the order-2 chain instead: locally plausible,
because it knows what tends to follow the last two symbols, but it never
restates anything, because that is all it knows. An order-2 chain reproduces the
average of its corpus, and averaging 22,119 phrases is generic however good the
corpus is.

**Rests are not silence to be trimmed.** Between 11% and 15% of each file is
rest, and that space is most of what makes the result read as ambient rather
than as a tune played slowly.

## Reproducing

```bash
python ../../inference/melody_markov.py --selftest    # the invariants
```

The samples themselves come from `generate_melody(...)` at the valence and
arousal in the table, `seed=3`, driven by `grammar.load_bank()` for the phrase
files and `grammar.load()` for the chain, then `render_standalone(notes, seed=3)`.

The rendered audio is a plain standalone voice, deliberately. Inside the
conductor the melody is voiced on a rated preset from Section 3.4.2's pool and
mixed against the drone; that path needs the arranger of Section 3.8.
