# Attribution — impulse responses

The three `.wav` files in this directory are measured impulse responses from
the **EchoThief Impulse Response Library**, by Christopher Warren.

    http://www.echothief.com/

They are third-party recordings redistributed here with credit, not output of
this project. They ship because Section 3.6's stimulus ladder convolves the
actual measurement rather than an approximation of it, so the ladder cannot be
rendered without them.

| file | space | RT60 | tail centroid |
|---|---|---|---|
| `Nature/FatMansMisery.wav` | open air | 0.294 s | 2654 Hz |
| `Brutalism/PepperCanyonHall.wav` | concrete room | 0.952 s | 3037 Hz |
| `Stairwells/CCRMAStairwell.wav` | stairwell | 2.388 s | 2685 Hz |

Each is the representative of its category in `../ir_reverb_bank.json`: the IR
whose measured RT60 is closest to that category's median. The full library
holds 115 spaces; only these three are needed to render the ladder.

The measurements above are this project's own, computed by
`../../train/fit_from_irs.py`. The recordings are not.

`LICENSE` at the repository root covers code only. This file governs the audio
beside it, and travels with it wherever `weights/` is copied.
