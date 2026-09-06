# 3.7 Melody Generation

A causal micro-transformer over folk phrases, plus an order-2 Markov baseline.
The conductor offers both, and all three trained transformer sizes.

**Not yet migrated.** Sources and destinations are in the top-level
[`code/README.md`](../../README.md) §2.7.

## Datasets

| Resource | Used for | Licence | Where to get it |
|---|---|---|---|
| **Essen Folksong Collection** | 22,119 monophonic phrases from 8,514 folk melodies | free for research | <http://www.esac-data.org/> |
| same, via **music21** | easier route: `music21.corpus.search('essen')` | BSD (toolkit) | <https://www.music21.org/music21docs/> |

Symbolic data only. No audio is used or redistributed.

Set `$DRONE_ESSEN`, or pass `--essen-root`. The music21 route needs no
download.

## Checkpoints

All three trained sizes ship in [`weights/`](weights), because the interface
lets the user pick between them:

| file | parameters | validation loss |
|---|---|---|
| `melodic_transformer_L3_d128.pt` | 612,387 | 1.5202 nats |
| `melodic_transformer_L6_d128.pt` | 1,207,203 | 1.5194 nats |
| `melodic_transformer_L6_d256.pt` | 4,773,667 | 1.5391 nats |

The deployed default is **L6/d128**. The two 128-wide models are within
0.0008 nats, so validation loss could not separate them and the choice was made
by listening.
