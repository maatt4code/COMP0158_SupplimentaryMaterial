# 3.6 Differentiable Reverberation and Spatial Acoustics

Fits three scalars per impulse response, decay time, damping cutoff and wet
gain, by gradient descent under a multi-scale STFT loss.

**Not yet migrated.** Sources and destinations are in the top-level
[`code/README.md`](../../README.md) §2.6.

## Datasets

| Resource | Used for | Licence | Where to get it |
|---|---|---|---|
| **EchoThief Impulse Response Library** | measured impulse responses the model is fitted to | free for any use, credit requested | <http://www.echothief.com/> |

Set `$DRONE_ECHOTHIEF`, or pass `--echothief-root`. Three impulse responses
ship with the conductor under [`conductor/assets/`](../../conductor/assets),
governed by `ATTRIBUTION.md`.

A second set of reverb characteristics was measured from commercial ambient
recordings rather than from impulse responses. Those are decay and damping
estimates only, re-keyed onto acoustic descriptors such as `dark_long` and
`bright_short`, with no artist or source identifiers. Same provenance as
[§3.5](../3.5_transition_dynamics_and_scheduling): archive.org and SoundCloud.

## Human ratings

`reverb_ratings.csv` ships in [`human_ratings/`](human_ratings), but read its
README first: 348 of 380 rows come from one rater, and no result in the report
rests on it.
