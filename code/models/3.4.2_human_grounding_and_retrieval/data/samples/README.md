# Sample outputs — the same request, on two valence axes

Four 8-second renders of what the deployed engine returns for two VA targets,
once navigating the frozen judge's valence and once navigating the propagated
human valence. Everything else is held fixed: same engine, same bank, same
renderer, same seed. The only difference is which column retrieval reads.

| file | axis | requested | retrieved anchor | anchor valence | f0 |
|---|---|---|---|---|---|
| `judge_vm0.8_am0.4.wav` | judge | v −0.8, a −0.4 | 15909 | −0.795 | 251.6 Hz |
| `human_vm0.8_am0.4.wav` | human | v −0.8, a −0.4 | 6797 | −0.805 | 49.8 Hz |
| `judge_vp0.3_ap0.2.wav` | judge | v +0.3, a +0.2 | 7929 | **+0.192** | 75.3 Hz |
| `human_vp0.3_ap0.2.wav` | human | v +0.3, a +0.2 | 19730 | **+0.297** | 135.9 Hz |

`retrieved_index.csv` holds these rows machine-readably.

## What to listen for

The second pair is the section's result. Asked for positive valence, the judge
axis cannot get there: its labels top out at **+0.244** across the whole
20,000-preset bank, so the nearest anchor sits at +0.192 and the request is
quietly clipped. The human axis reaches **+0.572**, so the same request lands at
+0.297 and returns a different preset entirely — an octave up, a different
timbre.

The first pair shows the axes agree about darkness. Both land near −0.8. The
disagreement is not a global offset; it is specifically that the automated axis
has no positive half to navigate, which is why half the VA plane was unreachable
before propagation.

## Reproducing them

These need the labelled preset bank, which is generated rather than shipped
(Section 3.3 renders it, Section 3.4.1 labels it, and
`../../train/propagate_labels_krr.py --propagate` adds the human column):

```bash
python ../../train/propagate_labels_krr.py \
    --bank-index <bank>/labeled_index.csv --propagate
```

Then retrieve on each axis with `inference/retrieval.py`'s
`GPSoftKNNEngine.set_label_space`, and render through Section 3.3's
`theta_render.render_theta` with `torch.manual_seed(0)`.

The boundary guard's stress walk, by contrast, needs nothing but this
repository:

```bash
python ../../train/fit_guard.py --self-test
```
