# 3.7 Melody Generation

A causal micro-transformer over folk phrases, plus an order-2 Markov baseline.
The conductor offers both, and all three trained transformer sizes.

## Datasets

| Resource | Used for | Licence | Where to get it |
|---|---|---|---|
| **Essen Folksong Collection** | 22,119 monophonic phrases from 8,514 folk melodies | free for research | <http://www.esac-data.org/> |
| same, via **music21** | easier route: `music21.corpus.search('essen')` | BSD (toolkit) | <https://www.music21.org/music21docs/> |

Symbolic data only. No audio is used or redistributed.

Set `$DRONE_ESSEN`, or pass `--essen-root`. The music21 route needs no
download.

## The two generators, and why there are two

**An order-2 Markov chain** over JOINT (interval, duration) symbols, fitted on a
notated folksong corpus. Joint is the point: in notated melody pitch and rhythm
are coupled — long notes at phrase ends, short notes in runs — and learning them
separately then multiplying gives back a bag of notes with better statistics.

**A phrase bank** of whole contours from the same corpus. It exists because the
chain reproduces the AVERAGE of its corpus: it knows what tends to follow the
last two symbols and nothing else, so it cannot know that a phrase has a shape
or is going anywhere. Sampling whole contours sidesteps that — the shape is
never regenerated, so it survives intact, and restating one is what a listener
hears as a tune.

**A causal micro-transformer**, conditioned on continuous valence and arousal.
Sized for CPU latency inside the live conductor rather than for the last
fraction of a nat.

## Why learn from notation at all

The authored generator produced a bag of notes, with two defects and only one of
them about pitch: every duration sat in a narrow band, so nothing read as
rhythm, and a first-order walk never restates anything, so there was no idea to
recognise. Better interval statistics alone would have fixed neither.

Rhythm is the thing to learn, and rhythm is **symbolic** — it cannot be
recovered from ambient audio, whose whole aesthetic is the absence of clear
onsets. A folksong corpus is monophonic by construction, so there is no
melody-extraction step and no skyline heuristic to defend, and it is notated, so
durations arrive as real note values.

**What is not learned, deliberately: the valence/arousal mapping.** Folk songs
carry no affect labels, so nothing here claims an emotion. VA remains an
authored, declared convention setting mode, pace, register and phrase length.
What transfers from the corpus is the GRAMMAR of how intervals and durations go
together, replayed slower in a different register and timbre.

## Run order

| # | Command | What it does |
|---|---|---|
| 1 | `python train/build_markov_grammar.py` | corpus → the fitted grammar (music21 route needs no download) |
| 2 | `python train/build_essen_dataset.py --essen` | grammar → the transformer's training set |
| 3 | `python train/train_transformer.py --n_layers 6 --d_model 128` | trains one size |
| 4 | `python train/benchmark_configs.py` | CPU latency across all three sizes |
| 5 | `python train/pick_melody_anchor.py --freeze` | freezes the timbre anchor |

Steps 1 and 5 reproduce shipped weights; 2–4 are the training path.

## Style names

Two of the four aesthetic registers were named after recording artists. They are
re-keyed onto the knobs that actually separate them — pace differs by about five
times — so the styles are now `folk`, `chant`, `pentatonic_fast` and
`pentatonic_slow`. No numeric value changed.

## Checkpoints

All three trained sizes ship in [`weights/`](weights), because the interface
lets the user pick between them. The architectures reproduce the reported
parameter counts exactly, and each checkpoint loads into the architecture the
smoke test rebuilds from scratch:

| file | parameters | validation loss |
|---|---|---|
| `melodic_transformer_L3_d128.pt` | 612,387 | 1.5202 nats |
| `melodic_transformer_L6_d128.pt` | 1,207,203 | 1.5194 nats |
| `melodic_transformer_L6_d256.pt` | 4,773,667 | 1.5391 nats |

The deployed default is **L6/d128**. The two 128-wide models are within
0.0008 nats, so validation loss could not separate them and the choice was made
by listening.

## What ships

| Path | What it is |
|---|---|
| `weights/markov_order2.json` | the fitted grammar: 8,514 tunes, 22,119 phrases, 4,643 chain states, 18,699 contours |
| `weights/melodic_transformer_*.pt` + `.json` | the three sizes with their sidecars |
| `weights/melody_anchor.json` | the frozen timbre anchor, so the runtime needs no rating data |
| `data/samples/` | four melodies contrasting the phrase bank, the chain and the authored fallback |

**No promoted-copy alias ships.** The deployed app carried the L6/d128 weights
under a second filename, which made it look like a fourth model; the three real
names ship instead and the smoke test asserts the alias is absent.

## The frozen melody anchor

The melody is voiced by borrowing a preset from Section 3.4.2's rated seed pool
— a real, human-rated timbre rather than an invented one. Choosing it needs that
pool's ratings, and the conductor ships none, so the choice is made once by
`train/pick_melody_anchor.py` and frozen.

Candidates must clear three bars at once: a positive human rating, a harmonic
centroid inside a band (too dark and the melody disappears under the drone, too
bright and it reads as a separate instrument), and enough survivors that the
pick is not simply the only one left. Four presets qualify and the choice is
deterministic. The BED anchor is not frozen: it depends on live valence and
arousal.

**Why the anchor lives here and not in 3.4.2**, whose rated pool it reads: the
selection logic and the centroid band are melody concerns, so they belong
beside the melody. The pool itself stays in 3.4.2 and is referenced rather
than copied.

## Reference numbers

| Check | Value |
|---|---|
| Corpus | 8,514 tunes → 22,119 phrases |
| Grammar | order 2, 4,643 chain states, 18,699 stored contours |
| Modes in the corpus | 6,633 major, 1,881 minor |
| Parameter counts | 612,387 / 1,207,203 / 4,773,667 |
| Melody anchor | preset 4174, from 4 qualifying candidates |
| Rebuild check | refitting from the corpus reproduces the shipped grammar exactly |

## Smoke test

```bash
python smoke_test.py     # exit 0, no corpus, no music21, no network
```
