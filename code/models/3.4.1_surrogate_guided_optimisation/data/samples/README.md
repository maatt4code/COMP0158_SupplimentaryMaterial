# Sample outputs — the closed-loop mapper, asked for four coordinates

Four renders from the shipped `weights/closed_loop_mapper.pt`, at the corners of
the evaluation grid. They need **no dataset**: the mapper and Section 3.3's
renderer are both in the repository, so one command regenerates all sixty clips
this set was drawn from.

```bash
python ../../train/evaluate_cycle_consistency.py \
    --mode render --inverse mlp --data-dir ../cycle_mlp
```

The four kept here are the `s=0` draws at the grid corners, which is index
`v_i*15 + a_i*3` — 0, 12, 45 and 57.

| file | asked for | judged back as | what it shows |
|---|---|---|---|
| `cyc_0000.wav` | v −0.9, a −0.8 | v −0.22, a −0.09 | the dark, still corner is not reached on either axis |
| `cyc_0012.wav` | v −0.9, a +0.8 | v −0.47, a +0.47 | arousal moves the right way, valence stops half way |
| `cyc_0045.wav` | v 0.0, a −0.8 | v −0.81, a −0.00 | asked for neutral valence, returned the darkest clip in the set |
| `cyc_0057.wav` | v 0.0, a +0.8 | v −0.17, a +0.87 | arousal lands; valence never gets near zero |

`labeled_index.csv` holds these four rows with the full theta and both label
columns, so the numbers above can be checked without re-rendering.

## What to listen for

Arousal roughly tracks the request — the bright, agitated clips really are
brighter and more agitated. Valence does not. The mapper optimised a real
objective to a low loss against the judge proxy, and the audio still does not
carry the requested valence, because the proxy's valence axis does not measure
what a listener hears. Reading the whole grid at once, the arousal bias is
**+0.380**: the loop is systematically rewarded for audio the judge misreads.

That gap is Section 4.1, and it is why the shipped system uses the
human-grounded retrieval of Section 3.4.2 instead of this mapper.
