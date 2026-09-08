# Bed audio — sources and licences

Every file in this directory is third-party audio, redistributed here
under its own licence. **Nothing in it was recorded for this project.**

One bed per type ships so the conductor's bed layer works out of the
box. The full curated bank is larger and is NOT redistributed: fetch
the two corpora and point `$DRONE_BEDS` at them, or rebuild the bank
with `models/3.5_transition_dynamics_and_scheduling/train/build_bed_bank.py`.

## Corpora

| Corpus | Licence | Where |
|---|---|---|
| ESC-50 | CC BY-NC 3.0 (the ESC-10 subset is CC BY 3.0) | <https://github.com/karolpiczak/ESC-50> |
| Emo-Soundscapes | Creative Commons, assembled from Freesound | <https://metatlas.github.io/> |

Both are non-commercial corpora. This material is academic
supplementary work and redistributes only the subset below.

## The shipped beds

| File | Type | Corpus | Licence | Credit | Original |
|---|---|---|---|---|---|
| `2-95567-A-23.wav` | breathing | ESC-50 | CC0 | Kuroseishin | [breathing_female.wav](http://www.freesound.org/people/Kuroseishin/sounds/95567/) |
| `r_0indicator68634_40561-hq.wav` | rain | Emo-Soundscapes | Creative Commons (per-clip, via Freesound) | — | [Freesound 68634](http://www.freesound.org/people/Rocktopus/sounds/68634/) |
| `4-182613-B-11.wav` | sea_waves | ESC-50 | CC0 | Carlvus | [CFX-20130331-UK-DorsetSeaBeach02.wav](http://www.freesound.org/people/Carlvus/sounds/182613/) |
| `r_0society49599_571325-hq.wav` | street-crowd | Emo-Soundscapes | Creative Commons (per-clip, via Freesound) | — | [Freesound 49599](http://www.freesound.org/people/BoilingSand/sounds/49599/) |
| `r_0quiet127242_116752-hq.wav` | thunder-storm | Emo-Soundscapes | Creative Commons (per-clip, via Freesound) | — | [Freesound 127242](http://www.freesound.org/people/Sempoo/sounds/127242/) |
| `4-161519-A-19.wav` | thunderstorm | ESC-50 | CC0 | daveincamas | [20120720Thunder.wav](http://www.freesound.org/people/daveincamas/sounds/161519/) |
| `1-51037-A-16.wav` | wind | ESC-50 | CC-BY | dobroide | [20080322.wind.strong.wav](http://www.freesound.org/people/dobroide/sounds/51037/) |

`va` is the corpus's own valence-arousal rating where it has one;
ESC-50 carries none, and those beds are eligible in any mood.
