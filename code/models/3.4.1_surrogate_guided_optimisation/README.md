# 3.4.1 Surrogate-Guided Optimisation

Labels the preset bank with frozen affect estimators, distils them into a
small CNN proxy, and optimises a mapping network against that proxy. This is
the loop that reward-hacked, and §4.1 is the result.

**Not yet migrated.** Sources and destinations are in the top-level
[`code/README.md`](../../README.md) §2.3.

## Datasets and pretrained models

| Resource | Used for | Licence | Where to get it |
|---|---|---|---|
| **DEAM** (MediaEval Database for Emotional Analysis in Music) | training the valence ridge head, 1,802 tracks | research use, registration | <https://cvml.unige.ch/databases/DEAM/> |
| **Emo-Soundscapes** | training the arousal ridge head, 600 environmental recordings | academic, non-commercial | <https://metacreation.net/emo-soundscapes/> |
| **MERT-v1-95M** | audio embeddings both ridge heads sit on | CC BY-NC 4.0 | <https://huggingface.co/m-a-p/MERT-v1-95M> |
| **audEERING wav2vec 2.0** (`wav2vec2-large-robust-12-ft-emotion-msp-dim`) | benchmarked as a cross-domain comparison in §4.1 | CC BY-NC-SA 4.0 | <https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim> |

Set `$DRONE_DEAM` and `$DRONE_EMO`, or pass `--deam-root` and
`--emo-soundscapes-root`. The two Hugging Face models download on first use.

**The frozen judge configuration is not the script default.** Valence is MERT
to the DEAM ridge; arousal is MERT to the Emo-Soundscapes ridge, selected with
`--arousal-ridge emo --arousal-scale pm1`. The audEERING default in the
argument parser was never used for any result in the report, because its
arousal output was too compressed.
