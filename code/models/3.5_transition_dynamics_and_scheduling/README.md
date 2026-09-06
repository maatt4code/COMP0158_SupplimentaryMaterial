# 3.5 Transition Dynamics and Scheduling

A semi-Markov dwell scheduler over five transition archetypes, with a
Bradley-Terry preference GP re-ranking its candidates.

**Not yet migrated.** Sources and destinations are in the top-level
[`code/README.md`](../../README.md) §2.5.

## Corpus

The transition statistics were measured from **95 commercially released ambient
tracks**, obtained as streamed or archived audio:

- <https://archive.org/details/audio_music>
- <https://soundcloud.com>

**No audio from this corpus is redistributed here, and the fitting script does
not ship.** What ships is the fitted result: dwell distributions, transition
counts and the five archetypes, in
[`weights/hsmm_transitions.json`](weights). Those are measurements, durations,
spectral deltas and decay estimates, not recordings.

The corpus is referenced by hash only. Artist names and source paths are
removed from every manifest, which is what `appendix_datasets.tex` claims and
what §0.2 of the top-level README enforces.

Because the fit cannot be re-run without the audio, `weights/` records the
sha256 of the shipped artefact so the number in the report is traceable to the
file.

## Human ratings

The preference GP is fitted on 120 pairwise comparisons collected for this
project. See [`human_ratings/README.md`](human_ratings/README.md), including
what that study did and did not test.
