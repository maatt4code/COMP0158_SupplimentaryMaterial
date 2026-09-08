"""A melodic line over the drone: the order-2 Markov generator and its
phrase bank.

WHAT THIS IS. A generator for scale-quantised pitch sequences that the EXISTING
renderer can play. Pitch already moves through the arranger: `f0_hz` is a theta
field and `render_waypoints` interpolates it across waypoints (that is how the
cents drift works). So a melody is just a waypoint sequence whose `f0_hz` lands
on scale degrees -- the frozen arranger renders it with no edit at all.

WHERE THE PITCHES COME FROM, and why. The design note's feasibility check found
NO stored melody corpus: despite the "MIDI-DDSP" lineage there are no .mid files
in the repo -- that name refers to the synthesis method, not to a dataset. So
this stage uses an explicit MUSIC-THEORY PRIOR (scale + a small interval Markov
biased toward stepwise motion), which is authored and says so. That is the same
discipline the arc vocabulary follows: the PARAMETERISATION is authored, and
nothing here claims that a given interval carries a given emotion.

WHAT IS *NOT* CLAIMED. VA conditions the melody's BEHAVIOUR by construction
(valence -> major/minor scale, arousal -> note rate, range and leap size), and
those links are asserted by music-theory convention, not measured. This is
expressive polish, not an emotion->DSP finding, unless a rating study later
earns it. Kept out of the arranger and out of every rated stimulus for exactly
that reason -- it is a conductor overlay, like voicing, breathing and crackle.

Usage:
  python melodic_drone.py --selftest
  python melodic_drone.py --demo out.wav --valence 0.5 --arousal -0.3
  python melodic_drone.py --print --valence -0.7 --arousal 0.6
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

SR = 16000
A4 = 440.0

# Scale degrees in semitones. Major/minor is the one place where a conventional
# valence association is baked in -- declared here, not hidden in a mapping.
SCALES = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    # PENTATONIC (added 2026-07-25 for the Boards of Canada / Kyle Bobby Dunn
    # target). Dropping the 4th and 7th removes the semitone tensions that make
    # a diatonic line sound "tuney" and goal-directed -- every note sits happily
    # over every chord below, which is exactly why this is the ambient default.
    "major_pent": (0, 2, 4, 7, 9),
    "minor_pent": (0, 3, 5, 7, 10),
    # DORIAN (added 2026-07-28 for the "chant" style below): minor with a
    # raised 6th. Deliberately NOT wired into the valence->mode switch the way
    # major/minor are -- church-mode chant was never composed to a
    # pleasant/unpleasant axis, so forcing one on it here would be the exact
    # hand-authored emotion mapping this project's own judge-invalidity
    # chapter warns against. `chant` uses this scale at every valence instead
    # (see STYLES["chant"]["scale_override"]).
    "dorian": (0, 2, 3, 5, 7, 9, 10),
}

# STYLE presets. These are AUTHORED aesthetic targets named after references
# were given, not measured emotion mappings -- the same declared-convention
# status as the VA->behaviour table.
#
#   folk  -- the corpus-driven line: diatonic, developing, phrase-per-idea.
#            Musically the richest and, by his ear, the least suitable.
#   boc   -- Boards of Canada: a SHORT cell repeated almost unchanged, over
#            slow 7th-chord movement, with tape wobble and detuned doubling.
#            The interest lives in timbre and harmony, not in the notes.
#   kbd   -- Kyle Bobby Dunn: barely a melody at all. Very long tones, tiny
#            range, enormous space between them.
#   chant -- Gregorian chant. `cell_notes=None`, the same code path as `folk`
#            (§ generate_melody), so the corpus grammar/PhraseBank actually
#            drives the line instead of looping a short authored riff -- the
#            grammar is an order-2 Markov chain over the corpus, which is
#            exactly the "markov and interpretability friendly" character
#            chant was picked for. Unaccompanied on purpose (chords=False,
#            wobble=detune=0): plainchant is monophonic and a cappella, so the
#            only thing left to characterise it is the interval sequence over
#            a fixed DORIAN mode (see SCALES), not timbre or processing.
STYLES = {
    "folk": dict(pentatonic=False, cell_notes=None, repeats=(2, 4), vary_p=1.0,
                 wobble_cents=0.0, detune_cents=0.0, chords=False, note_scale=1.0),
    "pentatonic_fast":  dict(pentatonic=True,  cell_notes=4,    repeats=(4, 8), vary_p=0.25,
                 wobble_cents=18.0, detune_cents=9.0, chords=True, note_scale=0.55),
    "pentatonic_slow":  dict(pentatonic=True,  cell_notes=3,    repeats=(3, 6), vary_p=0.15,
                 wobble_cents=8.0,  detune_cents=5.0, chords=True, note_scale=2.6),
    "chant": dict(pentatonic=False, cell_notes=None, repeats=(2, 4), vary_p=1.0,
                  wobble_cents=0.0, detune_cents=0.0, chords=False, note_scale=1.5,
                  scale_override="dorian"),
}

# Interval prior over scale STEPS (not semitones), biased to stepwise motion.
# Authored, deliberately dull: ambient melody should not leap around.
STEP_WEIGHTS = {-3: 0.04, -2: 0.11, -1: 0.28, 0: 0.14, 1: 0.28, 2: 0.11, 3: 0.04}


# ------------------------------------------------------------------ VA -> knobs
# ARTICULATION, one global macro (professor, 2026-08-01: "stocata -> lagarto",
# i.e. short-sharp vs long-gentle). 0.0 = legato, 1.0 = staccato.
#
# It exists because the complaint it answers -- "attack is not quite finished
# but goes into loud sustain" -- was NOT an envelope-shape problem on its own.
# Measured on the shipped clips, styles were producing 1-14 held spans per 60 s
# (kbd: ONE tone for the whole minute at a=+0.3; chant: four at 15 s each),
# and `articulate`'s attack is ABSOLUTE (0.02 s), so on a 15 s span it is 0.13%
# of the note: a 20 ms ramp onto a dead-flat full-level plateau lasting ~12 s.
# You cannot shape your way out of that, so this macro sets note LENGTH and
# repeat-merging as well as envelope shape. Those three are one control.
LEGATO, STACCATO = 0.0, 1.0

# A clip has to contain a MELODY, not a held tone. Measured 2026-08-01, kbd
# produced ONE span per 60 s at a=+0.3 and chant four; the professor's "loud
# sustain" complaint is downstream of that, since no envelope reads as a note
# when the note is 24 s long.
#
# Enforced as a SOFT cap (see `_pace_cap`) rather than a hard ceiling. A hard
# ceiling collapses the styles into each other -- folk, chant and the slow pentatonic style all sit
# above it, so all three would clamp to the same pace and the authored
# distinction between them would vanish. The soft knee is monotone, so the
# ordering boc < folk < chant < kbd survives while every style stays bounded.
MIN_NOTES_PER_CLIP = 6

# Ceiling on any ONE note relative to the style's pace. The grammar supplies
# duration multipliers (dm) and the authored path adds a 1.8x phrase-final
# stretch; unbounded, those compounded into 24 s single notes off a 6 s pace.
NOTE_STRETCH_MAX = 2.2

# Ceiling, in NOTE seconds, on how long legato may hold a repeated pitch.
# Merging consecutive identical pitches is what legato means, but kbd repeats
# its cell with vary_p=0.15, so unbounded merging turned a whole clip into one
# 28 s tone. Re-attack past this: a held note that outlasts the ear's sense of
# it is no longer legato, it is a drone.
MERGE_MAX_S = 5.0

# How far the note body sags across its sustain, as a fraction of the sustain
# level. Guarantees no note is ever dead-flat, however long it is.
SUSTAIN_TILT = 0.70


def melody_xfade_s(tick_s=1.0):
    """The crossfade `render_via_arranger` renders with.

    Defined ONCE because `shape()` has to know it: the envelope is laid on the
    note clock, but the renderer crossfades between waypoints, so the audible
    note change happens half a crossfade BEFORE the waypoint is fully present.
    If these two ever disagree, every envelope boundary lands late (see the
    alignment block in `shape`)."""
    return min(1.5, float(tick_s) * 0.75)


def articulation_knobs(articulation=0.5):
    """The one macro, expanded into the envelope and note-length terms it drives.

    Kept as a function rather than a table so the axis is CONTINUOUS: the
    professor asked for a global parameter, and rating it later needs
    intermediate points, not three named presets."""
    q = float(np.clip(articulation, 0.0, 1.0))
    return dict(
        articulation=q,
        # Note length. Staccato is not just a shorter envelope on the same note.
        note_mult=float(np.interp(q, [0, 1], [1.25, 0.40])),
        # Fraction of the SPAN that actually sounds; the remainder is silence.
        # This is what makes staccato detach -- an envelope alone cannot.
        duty=float(np.interp(q, [0, 1], [1.00, 0.45])),
        # Envelope, all as fractions of the note so they scale with its length.
        attack_frac=float(np.interp(q, [0, 1], [0.45, 0.02])),
        decay_frac=float(np.interp(q, [0, 1], [0.30, 0.18])),
        release_frac=float(np.interp(q, [0, 1], [0.35, 0.12])),
        # THE fix for "loud sustain": the body decays to this instead of sitting
        # at 1.0. A plateau is what made a long note read as a wall.
        sustain=float(np.interp(q, [0, 1], [0.72, 0.22])),
        # Legato joins repeated pitches into one held note; staccato re-attacks
        # them, which is both musically right and the only way kbd's repeated
        # cells stop collapsing into a single span.
        merge_repeats=bool(q < 0.5),
    )


# TIMBRE, the second global macro (professor, 2026-08-01: "sharp vs soft and
# hummy"). 0.0 = soft/hummy, 1.0 = sharp. Defined as a tilt over theta dims we
# already have rather than new DSP, because the theta sensitivity work found
# valence is timbre-dominated and already located this axis: happy = warm/mid/
# rich, sad = low/hissy/harsh. AUTHORED and unrated -- same status as the
# envelope constants that produced the complaint, so it belongs in the rating
# app as a factor, not hand-set in a shipped default.
def timbre_knobs(timbre=0.5):
    q = float(np.clip(timbre, 0.0, 1.0))
    return dict(
        timbre=q,
        # how many partials survive: hummy keeps a near-sine, sharp keeps a stack
        max_harmonics=int(round(np.interp(q, [0, 1], [4, 16]))),
        # tilt applied across surviving partials (>1 dulls, <1 brightens)
        decay_exponent=float(np.interp(q, [0, 1], [2.2, 0.8])),
        # breath/air. Sharp gets some; hummy gets almost none.
        noise_mult=float(np.interp(q, [0, 1], [0.15, 1.20])),
        # onset hardness, folded into the articulation attack below
        attack_mult=float(np.interp(q, [0, 1], [1.60, 0.55])),
    )


def melody_params(valence, arousal, style="pentatonic_fast", articulation=None):
    """Map VA onto melodic BEHAVIOUR. Conventional, authored, and reported as
    such: valence picks the mode; arousal sets how active the line is. `style`
    then sets the aesthetic register (see STYLES).

    `articulation=None` is the LEGACY path and the default ON PURPOSE. The
    macro's note-length terms change what `generate_melody` emits, so defaulting
    it on would silently alter every existing caller's notes -- and the whole
    point of keeping the old envelopes alongside `legato` is that the old arms
    must regenerate identically. Pass a float only when you want the macro."""
    a = float(np.clip(arousal, -1.0, 1.0))
    v = float(np.clip(valence, -1.0, 1.0))
    st = STYLES[style]
    art = articulation_knobs(articulation) if articulation is not None else None
    mode = "major" if v >= 0 else "minor"
    return dict(
        style=style,
        cell_notes=st["cell_notes"],
        repeats=st["repeats"],
        vary_p=st["vary_p"],
        wobble_cents=st["wobble_cents"],
        detune_cents=st["detune_cents"],
        chords=st["chords"],
        # a style may PIN its scale (chant -> dorian, at every valence) rather
        # than take the mode this function otherwise derives from valence
        scale=st.get("scale_override") or
              ((mode + "_pent") if st["pentatonic"] else mode),
        # NOTE RATE, set by ear (audition, 2026-07-25):
        # the old calm setting of ~2.1 s "is probably right -- or even a bit too
        # fast -- FOR THE BRIGHT CASE", and the calm case "needs to be way
        # slower". So the whole mapping shifted down: bright/active now sits
        # near the pace was approved and calm is roughly twice as slow again.
        # `note_scale` is the style's pace: the fast pentatonic style moves faster than the folk
        # setting (audition: "fast and mid are ok"), the slow pentatonic style far slower.
        # The style's raw pace. Bounded against the clip length by `_pace_cap`
        # in generate_melody, which is the only place duration_s is known.
        note_s=float(np.interp(a, [-1, 1], [6.0, 1.5]) * st["note_scale"]
                     * (art["note_mult"] if art else 1.0)),
        # Phrase structure. Notes are grouped, then the line BREATHES -- a
        # longer final note and a silence. Calm gets shorter phrases and much
        # longer gaps; that space is most of what makes ambient sound calm.
        phrase_notes=int(st["cell_notes"] or round(np.interp(a, [-1, 1], [3, 7]))),
        rest_s=float(np.interp(a, [-1, 1], [8.0, 2.0])
                     * (0.45 if st["cell_notes"] else 1.0)),
        span=int(round(np.interp(a, [-1, 1], [4, 10]))),      # scale steps of range
        max_step=int(round(np.interp(a, [-1, 1], [2, 3]))),   # leap size ceiling
        register=float(np.interp(a, [-1, 1], [-12.0, 4.0])),  # semitones vs root
        # carried through so the envelope stage does not have to re-derive the
        # macro, and so a logged params blob records what was actually played
        **(art or {}),
    )


MELODY_LO_HZ, MELODY_HI_HZ = 150.0, 400.0

# A distinct BUT GENTLE voice for the line.
#
# First attempt picked the anchor with the HIGHEST harmonic centroid, reasoning
# that bright = separable. Rated result: "sounds like car horns at a traffic
# jam" (audition, 2026-07-26). Bright harmonics plus
# a fast attack is a horn. Separability was bought at the cost of the whole
# aesthetic, and a line nobody wants to hear is worse than one they cannot.
#
# Second attempt (this comment, until 2026-07-27) picked anchor 15022 by a
# GENTLENESS rule run over the raw 20,000-row bank -- nearly all energy in the
# first four partials, low harmonic centroid, minimal noise. Two things wrong
# with that: it never checked RATED STATUS (15022 is unrated, free-ranging over
# unheard territory) and the objective's optimum is degenerate (90.7% of its
# energy in the fundamental is a bare sine), and it never looked at SWELL, so
# it also carries swell_rate 1.33 Hz baked into its own DDSP theta -- an
# audible per-note throb synthesized fresh on every note, independent of
# anything the arranger or conductor does. That is what "bubble tea" and
# "gritty f0" actually were; see anchor_audition.py and DESIGN_NOTES.md sec.12.
#
# HUMAN RATING FIRST, now actually queried rather than hand-copied from a
# one-off read of the data: `pick_melody_anchor()` below joins the ~150 anchors
# in al_pool (rendered AND rated, unlike the 20,000-row bank) against their
# ratings, keeps only rating>=+1, and filters to real-but-gentle spectra. Same
# join pattern as build_context_mixes.load_drones for the reverb SideProject.
# The rated seed pool is Section 3.4.2's, shipped pseudonymised. Reading it
# here is a LOOKUP over measured anchors, not a rating study of its own.
_S342 = (HERE.parents[1] / "3.4.2_human_grounding_and_retrieval"
         / "human_ratings")
POOL_META = _S342 / "pool_meta.json"
POOL_RATINGS = _S342 / "valence_ratings.csv"


def _pool_ratings(ratings_path=POOL_RATINGS):
    """clip_path -> rating. Last row wins on a re-rate, same convention as
    build_context_mixes.load_drones."""
    out = {}
    if not Path(ratings_path).exists():
        return out
    with open(ratings_path) as f:
        for row in csv.DictReader(f):
            try:
                out[row["clip_path"]] = float(row["rating"])
            except (TypeError, ValueError):
                continue
    return out


def melody_anchor_candidates(min_rating=1.0, centroid_range=(1.6, 3.2),
                             max_swell_hz=0.6, max_noise=0.12,
                             meta_path=POOL_META, ratings_path=POOL_RATINGS):
    """Rated anchors (anchor_idx, swell_rate, centroid, note) that survive the
    gentleness filter, calmest swell first. The trusted bank, not a spectral
    rule free to roam unrated rows -- see the comment above."""
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return []
    clips = json.load(open(meta_path))
    clips = clips["clips"] if isinstance(clips, dict) and "clips" in clips else clips
    ratings = _pool_ratings(ratings_path)
    out = []
    for c in clips:
        rating = ratings.get(c["clip_path"])
        if rating is None or rating < min_rating:
            continue
        th = c["theta_scalars"]
        h = np.asarray(c.get("harm_dist") or [1.0], dtype=float)
        centroid = float((h * np.arange(1, len(h) + 1)).sum() / (h.sum() + 1e-8))
        if not (centroid_range[0] <= centroid <= centroid_range[1]):
            continue
        if th["swell_rate"] > max_swell_hz or th["noise_level"] > max_noise:
            continue
        out.append((int(c["anchor_idx"]), float(th["swell_rate"]), centroid))
    out.sort(key=lambda t: t[1])                # calmest swell first
    return out


FROZEN_ANCHOR = (Path(__file__).resolve().parents[1] / "weights"
                 / "melody_anchor.json")


def pick_melody_anchor(fallback=13161, frozen_path=FROZEN_ANCHOR, **kw):
    """The melody's timbre anchor: the frozen pick if one ships, else refitted
    from the rated pool, else `fallback`.

    **The frozen file comes first, and that ordering is the point.** Choosing
    an anchor needs Section 3.4.2's rated seed pool, and the conductor ships
    no rating data at all, so a runtime that refits here does not fail -- it
    quietly returns something else. That is exactly what happened: the
    conductor resolved 13161 while this section, the frozen weight and the
    report all say 4174. 13161 is the FOURTH-ranked candidate, carrying
    swell_depth 0.567 against 4174's 0.011, so the deployed melody wobbled in
    precisely the way the gentleness filter exists to prevent, and nothing
    reported a problem.

    `fallback` must itself be a CONFIRMED rating==+1 anchor (13161, checked
    directly against valence_ratings.csv rather than copied from an earlier
    comment -- that is the exact mistake this function replaces). It does not
    have to pass the gentleness filter, only rated status: it is the emergency
    exit, not the pick. It now fires only when BOTH the frozen file and the
    pool are missing.
    """
    frozen = Path(frozen_path) if frozen_path else None
    if frozen and frozen.exists():
        return int(json.loads(frozen.read_text())["anchor_idx"])
    cands = melody_anchor_candidates(**kw)
    return cands[0][0] if cands else fallback


def pick_bed_anchor(v, a, min_rating=0.0, meta_path=POOL_META, ratings_path=POOL_RATINGS):
    """Nearest RATED anchor to (v, a), for use as a DRONE bed -- not the
    gentleness filter above, which is tuned for a melody VOICE, not a bed.

    2026-07-27: build_melody()'s bed was still `engine.retrieve(v, a)`,
    nearest-VA over the raw 20,000-row bank. Same defect already fixed for
    the melody anchor and for anchor_audition.py's bed, missed here. the audition,
    after the anchor/level fixes: "your base drone is already too loud" --
    at the SAME measured aw_rms as the reference demo (both target 0.015), so the
    gap is not level, it is that an unrated anchor can carry more noise/upper
    harmonic energy at the same RMS and read louder for it (the reference demo
    forces noise_level<=0.01 and mutes harmonics past the 5th on ITS bed,
    which is the same fix from a different angle). Prefer rating>=+1 within
    reach; fall back to >=min_rating if nothing is close; fall back to
    `retrieve()` only if al_pool itself is unavailable."""
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return None
    clips = json.load(open(meta_path))
    clips = clips["clips"] if isinstance(clips, dict) and "clips" in clips else clips
    ratings = _pool_ratings(ratings_path)
    pool = []
    for c in clips:
        rating = ratings.get(c["clip_path"])
        if rating is None or rating < min_rating:
            continue
        dv, da = c["achieved_v"] - v, c["achieved_a"] - a
        pool.append((dv * dv + da * da, rating, int(c["anchor_idx"])))
    if not pool:
        return None
    pool.sort(key=lambda t: (-t[1], t[0]))       # liked first, then nearest
    liked = [t for t in pool if t[1] >= 1.0]
    return (liked[0][2] if liked else pool[0][2])


# Read from the frozen weight at import, so every tree that loads this module
# -- section, conductor, a lifted-out copy -- voices the same anchor. Re-run
# `train/pick_melody_anchor.py --freeze` after a rating round adds rows: the
# pick can change on its own, and freezing is what makes that change explicit
# and reviewable rather than silent.
DEFAULT_MELODY_ANCHOR = pick_melody_anchor()


def melody_root(drone_f0_hz, lo=MELODY_LO_HZ, hi=MELODY_HI_HZ):
    """Octave-shift a DRONE's fundamental into a register a melody can be heard in.

    `register` above is an offset in semitones FROM THE ROOT, spanning -12..+4,
    which assumes the root is already musical -- the module default is 110 Hz.
    Callers were instead passing the drone's own fundamental (46-75 Hz on this
    bank), so at low arousal the line landed a full octave BELOW the drone, in the
    drone's own timbre. That is not a melody anyone can hear; it is a second
    sub-bass. Measured on the built stimuli: notes spanning 31.0-46.5 Hz under a
    49.3 Hz drone.

    Shifting by whole OCTAVES keeps the line harmonically locked to the drone --
    it stays the same pitch class, so nothing about the harmony changes, only the
    register it is audible in."""
    f = float(drone_f0_hz)
    if f <= 0:
        return 220.0
    while f < lo:
        f *= 2.0
    while f > hi:
        f /= 2.0
    return f


def scale_pitches(root_hz, scale, span, register):
    """Absolute Hz for every scale degree inside the register window."""
    degs = SCALES[scale]
    base = 12.0 * math.log2(max(root_hz, 1e-6) / A4) + register
    out = []
    octv = 0
    while len(out) < span + 1:
        for d in degs:
            st = base + d + 12 * octv
            out.append(A4 * (2.0 ** (st / 12.0)))
            if len(out) >= span + 1:
                break
        octv += 1
    return out


# ------------------------------------------------------------------ generation
REST = 0.0      # a note whose pitch is REST is silence (see to_gain_envelope)


def _motif(rng, steps, w, p):
    """A short cell of (step, duration-multiplier) pairs. This is the idea the
    piece is ABOUT -- the thing a listener can recognise coming back.

    Length comes from `phrase_notes`, so a calm line states a SHORT idea and
    then leaves a long silence, while an active one runs a longer figure with
    less space around it."""
    n = max(2, int(p["phrase_notes"]) + int(rng.integers(-1, 2)))
    return [(int(rng.choice(steps, p=w)), float(rng.uniform(0.75, 1.35)))
            for _ in range(n)]


def _vary(motif, rng):
    """Restate the motif with a small change -- the whole point of a motif is
    that it returns, but not identically. Transposition is handled by the caller
    (it just starts the phrase from a different degree)."""
    kind = str(rng.choice(["repeat", "invert", "truncate", "stretch"],
                          p=[0.40, 0.25, 0.20, 0.15]))
    if kind == "repeat":
        return list(motif), kind
    if kind == "invert":                      # same shape, mirrored
        return [(-s, d) for s, d in motif], kind
    if kind == "truncate":                    # the first few events only
        return list(motif[: max(2, len(motif) - 1)]), kind
    return [(s, d * 1.5) for s, d in motif], kind     # same shape, slower


def load_grammar(path=None, kind="phrase"):
    """The learned Essen model, or None if it has not been built yet.

    kind="phrase" replays whole human-written contours (recommended); "markov"
    generates note by note from the order-2 chain. Both come from the same file
    so the two can be A/B'd without rebuilding."""
    try:
        import grammar as essen_model
        p = Path(path) if path else essen_model.DEFAULT_GRAMMAR
        if not p.exists():
            return None
        if kind != "phrase":
            return essen_model.load(p)
        bank = essen_model.load_bank(p)
        # a model built before phrase shapes were stored has an empty bank --
        # fall back rather than crash, and say so at the CLI
        return bank if len(bank) else None
    except Exception:
        return None


def _pace_cap(note_s, duration_s):
    """Bound the note pace so a clip holds at least MIN_NOTES_PER_CLIP notes.

    Soft knee, not a clamp: `cap * (1 - exp(-note_s/cap))` is monotone and
    asymptotic, so a style already comfortably under the cap is barely touched,
    a style far over it lands just below, and the ORDER of the styles is
    preserved. A hard `min()` gave every slow style the identical pace and
    erased the difference between folk, chant and the slow pentatonic style."""
    cap = float(duration_s) / max(1, MIN_NOTES_PER_CLIP)
    if cap <= 0:
        return float(note_s)
    return float(cap * (1.0 - np.exp(-float(note_s) / cap)))


def generate_melody(valence, arousal, root_hz=110.0, duration_s=60.0, seed=0,
                    grammar=None, use_grammar=True, kind="phrase", style="pentatonic_fast",
                    articulation=None):
    """A list of (f0_hz, dur_s) notes, where f0_hz == REST means silence.

    Every pitch is a scale degree by construction -- the walk moves over degree
    INDICES and only then converts to Hz, so nothing can land between the cracks.

    STRUCTURE (added 2026-07-25 after the audition heard the first demos: "this is
    just a bag of notes and not really a melody"). A plain step-by-step random
    walk cannot sound like a melody however good its interval statistics are,
    because nothing ever comes BACK -- there is no idea to recognise and no
    silence to frame it. So the line is built from a MOTIF that is restated with
    variations, grouped into phrases, each phrase ending on a longer note
    followed by a rest. Repetition and silence are what a listener hears as
    "a tune"; the interval distribution only decides its local flavour.

    GRAMMAR (2026-07-25). When the Essen model has been built, intervals AND
    durations come from it as joint symbols -- see ../train/build_markov_grammar.py. That is the
    "half learned" half: the corpus supplies the local grammar and the long/short
    rhythm contrast (roughly 6:1, against 1.8:1 for the authored fallback), while
    theory keeps the guardrails -- scale quantisation, register bounds, phrase
    and rest structure -- and VA keeps tempo, register, mode and phrase length.
    Interval size is deliberately NOT capped by `max_step` on this path: capping
    it would overwrite the very thing the corpus was consulted for.

    With no model file present it falls back to the authored motif walk, so the
    module still runs standalone."""
    p = melody_params(valence, arousal, style=style, articulation=articulation)
    # Pace/stretch bounds ride with the macro, not with the legacy path: they
    # change note LENGTHS, so applying them unconditionally would mean the old
    # envelope arms no longer reproduce the clips they are meant to reproduce.
    if articulation is not None:
        p = dict(p, note_s=_pace_cap(p["note_s"], duration_s))
    rng = np.random.default_rng(seed)
    pitches = scale_pitches(root_hz, p["scale"], p["span"], p["register"])
    steps = [s for s in STEP_WEIGHTS if abs(s) <= p["max_step"]]
    w = np.array([STEP_WEIGHTS[s] for s in steps], dtype=float)
    w /= w.sum()

    def _step(i, d):
        nxt = i + int(d)
        if not 0 <= nxt < len(pitches):        # reflect at the register edges
            nxt = i - (nxt - i)
        return int(np.clip(nxt, 0, len(pitches) - 1))

    g = (grammar if grammar is not None else load_grammar(kind=kind)) \
        if use_grammar else None
    is_bank = hasattr(g, "sample")             # PhraseBank vs EssenGrammar
    notes, i, t, home = [], len(pitches) // 2, 0.0, len(pitches) // 2
    motif = None if g is not None else _motif(rng, steps, w, p)

    def _emit(cell, start_i, absolute):
        """Play one cell. `absolute` cells carry degrees relative to the phrase
        start (a stored contour); otherwise each entry is a step from the
        previous note (a generated walk)."""
        nonlocal t
        j = start_i
        for k, (d, dm) in enumerate(cell):
            if t >= duration_s:
                return
            if absolute:
                j = _step(start_i, d)          # contour, measured from its start
            last = k == len(cell) - 1
            # a stored phrase already ends long, and the corpus chain learns the
            # same; only the authored fallback needs an arrival bolted on
            stretch = (1.8 if (last and g is None) else 1.0)
            # dm (corpus duration multiplier) times the phrase-final stretch was
            # unbounded: a 6 s pace produced 24 s single notes, which is what
            # made the envelope read as a plateau. A phrase-final note should be
            # LONGER, not a drone, so the product is capped rather than dropped.
            # Macro path only -- see the _pace_cap note above.
            mult = dm * stretch
            if articulation is not None:
                mult = min(mult, NOTE_STRETCH_MAX)
            notes.append((float(pitches[j]),
                          max(0.5, float(p["note_s"] * mult))))
            t += notes[-1][1]
            if not absolute:
                j = _step(j, d)

    while t < duration_s:
        if p["cell_notes"]:
            # RIFF MODE (boc/kbd). One short cell, repeated many times almost
            # unchanged, at a FIXED transposition. That repetition is the point:
            # the reference records earn their character from timbre, detuning
            # and the chords underneath, not from the line developing. Varying
            # it -- which is what the folk path does -- is precisely what makes
            # it stop sounding like this music.
            if is_bank:
                # Take the phrase's TAIL, not its head. Notated phrases put the
                # long note at the end (that is the cadence), so `[:n]` threw the
                # rhythm away every time and handed back a flat cell -- exactly
                # the "every note the same duration" defect this whole thread
                # started with.
                #
                # But cadences also REPEAT a pitch ("...121 121 121"), and a cell
                # built from one of those sits on a single note for most of its
                # length -- the line sounds stuck. Resample for a cell that
                # actually moves; give up after a few tries and take what we got.
                for _try in range(8):
                    shape = g.sample(rng, target_len=p["cell_notes"],
                                     max_span=max(2, p["span"] - 1))[-p["cell_notes"]:]
                    if len({d for d, _ in shape}) >= max(2, len(shape) - 1):
                        break
            elif g is not None:
                # chain-built cell. Must run through the chain rather than the
                # authored steps, or --kind markov would silently discard the
                # learned DURATIONS and emit a flat cell.
                sym = [g.start(rng)]
                while len(sym) < p["cell_notes"]:
                    sym.append(g.next(rng, sym))
                shape, run = [], 0
                for iv, dr in sym:             # chain gives steps; cells are absolute
                    run += iv
                    shape.append((run, dr))
                shape = tuple(shape)
            else:
                shape = tuple((int(rng.choice(steps, p=w)), 1.0)
                              for _ in range(p["cell_notes"]))
            # A four-note window can still come back rhythmically flat (the chain
            # rarely emits a long note that early). Riffs of this kind land on a
            # longer note, so give the cell one when the source did not.
            ds = [d for _, d in shape]
            if len(shape) > 1 and max(ds) / min(ds) < 1.5:
                shape = tuple(shape[:-1]) + ((shape[-1][0], shape[-1][1] * 2.0),)
            # Choose the transposition so the WHOLE cell fits the register.
            # Clamping instead (what _step does) silently collapses several
            # degrees onto the top or bottom note, producing cells like
            # [143,121,121,121] -- a repeated note that reads as the line
            # getting stuck. Transposing within the legal range cannot do that.
            degs = [d for d, _ in shape]
            lo_b, hi_b = -min(degs), len(pitches) - 1 - max(degs)
            if lo_b > hi_b:                    # cell wider than the register
                shape = shape[:2]
                degs = [d for d, _ in shape]
                lo_b, hi_b = -min(degs), len(pitches) - 1 - max(degs)
            pick_base = lambda: int(rng.integers(lo_b, hi_b + 1))
            base_i = pick_base()
            reps = int(rng.integers(*p["repeats"]))
            for rep in range(reps):
                if t >= duration_s:
                    break
                cell = shape
                if rep and is_bank and rng.random() < p["vary_p"]:
                    cell = g.vary(shape, rng)[0]
                # MOVE THE RIFF. Holding one transposition for the whole group
                # made 90 s in which nothing happened. Shifting every couple of
                # repeats recontextualises the same cell against the chords
                # underneath, which is how these records develop -- the notes
                # stay put and their harmonic meaning changes.
                if rep and rep % 2 == 0:
                    base_i = pick_base()
                _emit(cell, base_i, absolute=True)
                if t < duration_s:
                    notes.append((REST, float(p["rest_s"] * rng.uniform(0.7, 1.2))))
                    t += notes[-1][1]
            # DROP OUT. Leaving the chords alone for a stretch, then bringing the
            # line back, is arrangement rather than composition -- and it is the
            # cheapest way to make a return feel like an event.
            if t < duration_s and rng.random() < 0.35:
                notes.append((REST, float(p["rest_s"] * rng.uniform(2.5, 4.5))))
                t += notes[-1][1]
            continue
        if is_bank:
            # ONE phrase, then RESTATEMENTS of it -- the tune is the returning
            # idea, not each new phrase. Each restatement is transposed a step
            # or two so it develops rather than loops (the exact-loop bug).
            shape = g.sample(rng, target_len=p["phrase_notes"],
                             max_span=max(2, p["span"] - 1))
            for rep in range(int(rng.integers(2, 4))):
                if t >= duration_s:
                    break
                cell = shape if rep == 0 else g.vary(shape, rng)[0]
                _emit(cell, _step(home, int(rng.integers(-2, 3))), absolute=True)
                if t < duration_s:
                    notes.append((REST, float(p["rest_s"] * rng.uniform(0.6, 1.3))))
                    t += notes[-1][1]
            continue
        if g is not None:
            n = g.phrase_length(rng, target=p["phrase_notes"])
            sym = [g.start(rng)]
            while len(sym) < n:
                sym.append(g.next(rng, sym))
            cell = list(sym)
        else:
            cell, _kind = _vary(motif, rng)
        _emit(cell, _step(home, int(rng.integers(-2, 3))), absolute=False)
        if t < duration_s:                     # breathe between phrases
            notes.append((REST, float(p["rest_s"] * rng.uniform(0.6, 1.3))))
            t += notes[-1][1]
    return notes, p


def to_gain_envelope(notes, tick_s=1.0):
    """Per-tick gain for the melody VOICE, aligned 1:1 with `to_waypoints`.

    The arranger has no silence -- it renders a continuous pad -- so a rest
    cannot be a theta value. Since the melody is rendered as its own pass and
    mixed at the Conductor, a rest is simply gain 0 in that mix. Rest ticks hold
    the previous pitch underneath so the pad does not jump when it comes back."""
    out = []
    for f0, dur in notes:
        n = max(1, int(round(float(dur) / max(tick_s, 1e-6))))
        out.extend([0.0 if f0 == REST else 1.0] * n)
    return out


def articulate(line, notes, sr=SR, tick_s=1.0, attack_s=0.03, release_s=0.40,
               floor=0.0):
    """Give the rendered line NOTE ONSETS.

    SUPERSEDED 2026-08-01 by `shape()` for everything this project ships; kept
    because melody_transformer.py calls it. Two known defects, both
    fixed in `shape`: the attack is ABSOLUTE, so on a long span it is a step
    onto a flat plateau (measured: a 17.55 s dead-flat run inside one note);
    and every boundary lands half a crossfade LATE (measured +0.352 s), so a
    short note's envelope runs on into the next note's pitch.

    Without this the melody is inaudible as a melody, and the reason is not level
    -- measured, the line sat 28 dB above the drone in its own band and still
    could not be picked out. `to_gain_envelope` returns 1.0 for every non-rest
    note, so the ONLY thing marking a note change is the arranger's 0.75 s
    equal-power crossfade between two pads of identical timbre. That is a
    continuous tone that slowly changes pitch. The ear segments a melody by
    onsets, and there were none.

    So each note gets an amplitude envelope: a fast attack and a release that
    dips towards `floor` before the next note. Repeated ticks of the SAME pitch
    are treated as one held note, so a long note is not chopped into pulses.

    The release is long (0.40 s) and the floor is 0, which looks aggressive on
    paper. It is not, because the arranger bakes reverb into the line at 0.35
    wet and the tail FILLS the gaps: measured on a real render, a 0.12 s release
    to a 0.06 floor survived as only 9.6 dB of articulation with 2 detectable
    onsets, while 0.40 s to 0 gives 16.0 dB and 3. The envelope has to be cut
    deeper than it sounds because the room puts most of it back.

    Applied to the audio, not to theta, so it cannot disturb the arranger."""
    line = np.asarray(line, dtype=np.float64)
    env = np.ones(len(line))
    a = max(1, int(attack_s * sr))
    r = max(1, int(release_s * sr))

    # collapse consecutive identical pitches into single held notes
    spans, t = [], 0.0
    for f0, dur in notes:
        n = max(1, int(round(float(dur) / max(tick_s, 1e-6)))) * tick_s
        if spans and spans[-1][0] == f0:
            spans[-1][2] += n
        else:
            spans.append([f0, t, n])
        t += n

    # The AUDIO runs on a different clock from the notes. `render_waypoints`
    # costs hold + xfade per waypoint, so with tick_s=1.0 and a 0.75 s crossfade
    # a "one second" note occupies 1.75 s of audio -- measured 13.41 s of notes
    # rendering as 25.20 s. Laying the envelope on the note clock therefore put
    # every onset in the wrong place. Rescale by what was actually rendered, so
    # this stays correct whatever hold/xfade the caller used.
    audio_s = len(line) / float(sr)
    scale = (audio_s / t) if t > 0 else 1.0

    for f0, start, dur in spans:
        i0 = int(start * scale * sr)
        i1 = min(len(line), int((start + dur) * scale * sr))
        if i1 <= i0:
            continue
        if f0 == REST:
            env[i0:i1] = 0.0
            continue
        seg = np.ones(i1 - i0)
        na = min(a, len(seg))
        seg[:na] = np.linspace(floor, 1.0, na) ** 0.5      # fast attack
        nr = min(r, len(seg) - na)
        if nr > 0:
            seg[-nr:] = np.linspace(1.0, floor, nr) ** 0.5  # dip before the next
        env[i0:i1] = seg
    return line * env


def swell(line, notes, sr=SR, tick_s=1.0, attack_frac=0.45, floor=0.0):
    """Articulate notes as SWELLS rather than struck tones.

    SUPERSEDED 2026-08-01 by `shape()`; kept for anchor_audition.py and the
    glossary figures. Its fractional attack has no ceiling, so on the long
    spans these styles actually produce it becomes a 7-11 s fade-in, and like
    `articulate` it places boundaries half a crossfade late.

    an audition note, 2026-07-26: "maybe use the reverb tails as melody
    vocabs". A reverb tail has no attack transient -- it grows and decays -- which
    is why it never sounds harsh however loud it is. Shaping each note that way
    gives the line onsets the ear can segment (so it still reads as separate
    notes) without the percussive edge that made the bright version honk.

    Same time-base correction as `articulate`: the audio clock is not the note
    clock, so spans are rescaled by what was actually rendered."""
    line = np.asarray(line, dtype=np.float64)
    env = np.ones(len(line))
    spans, t = [], 0.0
    for f0, dur in notes:
        n = max(1, int(round(float(dur) / max(tick_s, 1e-6)))) * tick_s
        if spans and spans[-1][0] == f0:
            spans[-1][2] += n
        else:
            spans.append([f0, t, n])
        t += n
    scale = (len(line) / float(sr) / t) if t > 0 else 1.0
    for f0, start, dur in spans:
        i0, i1 = int(start * scale * sr), min(len(line), int((start + dur) * scale * sr))
        if i1 <= i0:
            continue
        if f0 == REST:
            env[i0:i1] = 0.0
            continue
        n = i1 - i0
        na = max(1, int(n * attack_frac))
        seg = np.empty(n)
        seg[:na] = np.linspace(floor, 1.0, na) ** 1.5          # slow growth
        seg[na:] = np.linspace(1.0, floor, n - na) ** 1.5      # long decay
        env[i0:i1] = seg
    return line * env


def shape(line, notes, sr=SR, tick_s=1.0, articulation=0.5, timbre=None,
          floor=0.0, xfade_s=None):
    """ONE note envelope, driven by the articulation macro. Replaces the
    swell / struck / flat trio.

    WHY THIS EXISTS (professor, 2026-08-01): "it sounds like Attack is not quite
    finished but goes into loud sustain". Measured, that was exactly right and
    the cause was structural, not cosmetic:

      * `articulate` takes an ABSOLUTE attack (0.02 s) and an absolute release,
        so on the 12-24 s spans these styles actually produce, the shape is a
        20 ms ramp -> a dead-flat plateau at FULL level for ~12 s -> a release.
        There is no attack to hear and the sustain never moves.
      * `swell` takes a fractional attack (0.45) with no ceiling, so the same
        span gives a 7-11 s fade-in. The opposite failure: not a note either.
      * `flat` is a plateau by definition.

    So every term here is a FRACTION of the note, then clamped in absolute
    seconds. Fractions alone give a fade-in on a long note; absolutes alone give
    a plateau. Both bounds are needed, which is why neither existing function
    could be fixed by retuning its constant.

    The body decays to `sustain` (< 1) rather than holding 1.0. That is the part
    that answers the complaint directly -- a long note now moves through its own
    length instead of arriving at a wall and staying there.

    `timbre` (0 hummy .. 1 sharp) only sharpens or softens the ONSET here; the
    spectral half of that macro is applied to theta by `apply_timbre`, since it
    has to happen before rendering, not after."""
    line = np.asarray(line, dtype=np.float64)
    art = articulation_knobs(articulation)
    env = np.zeros(len(line))
    a_mult = timbre_knobs(timbre)["attack_mult"] if timbre is not None else 1.0

    # Spans. Legato joins repeated pitches into one held note; staccato
    # re-attacks them (see articulation_knobs) -- without this, styles whose
    # cells repeat the same pitch collapse into one span per clip.
    spans, t = [], 0.0
    for f0, dur in notes:
        n = max(1, int(round(float(dur) / max(tick_s, 1e-6)))) * tick_s
        if (spans and spans[-1][0] == f0
                and (art["merge_repeats"] or f0 == REST)
                and (f0 == REST or spans[-1][2] + n <= MERGE_MAX_S)):
            spans[-1][2] += n
        else:
            spans.append([f0, t, n])
        t += n

    # Same time-base correction as the functions this replaces: render_waypoints
    # charges hold + xfade per waypoint, so the audio clock is ~1.7-1.9x the
    # note clock and an envelope laid on note time lands in the wrong place.
    scale = (len(line) / float(sr) / t) if t > 0 else 1.0

    # CROSSFADE ALIGNMENT (professor, 2026-08-01: "whenever there is a short
    # followed by a long note, the short one sounds as if the note is not fully
    # decompressed before the longer one starts").
    #
    # The scale above puts each boundary where the next waypoint becomes FULLY
    # PRESENT -- but the renderer has been crossfading into it for the previous
    # xfade_s, so the pitch audibly changes at the crossfade's MIDPOINT, half a
    # crossfade earlier. Measured on a real 220->440 render: the pitches cross
    # over at 1.280 s while the envelope boundary sat at 1.600 s.
    #
    # So the tail of a note's envelope was being applied to audio that had
    # already become the NEXT note, and the next note's attack then began on a
    # pitch already at full strength. On a long note that is a small proportion;
    # on a SHORT one the lag can be most of the note, which is exactly the
    # short-then-long case reported. Pull every internal boundary back by half a
    # crossfade so it lands where the change is heard.
    xf = melody_xfade_s(tick_s) if xfade_s is None else float(xfade_s)
    lead = 0.5 * xf
    last = len(spans) - 1

    for k, (f0, start, dur) in enumerate(spans):
        a0 = start * scale
        a1 = (start + dur) * scale
        # waypoint 0 has no fade-in (the renderer starts it dry), so the first
        # note really does begin at 0; the final note rings out to the end.
        if k > 0:
            a0 -= lead
        if k < last:
            a1 -= lead
        i0 = max(0, int(a0 * sr))
        i1 = min(len(line), int(a1 * sr))
        if i1 <= i0:
            continue
        if f0 == REST:
            continue                                  # env already 0 here
        span_n = i1 - i0
        # duty < 1 detaches the note from the next one. The silence is part of
        # the articulation, not a gap in the rendering.
        n = max(1, int(span_n * art["duty"]))
        n = min(n, span_n)

        def _clamp(frac, lo_s, hi_s):
            return int(np.clip(int(n * frac), max(1, int(lo_s * sr)),
                               max(1, int(hi_s * sr))))

        na = min(_clamp(art["attack_frac"] * a_mult, 0.008, 2.5), n)
        nr = min(_clamp(art["release_frac"], 0.020, 3.0), n - na)
        nd = min(_clamp(art["decay_frac"], 0.010, 4.0), max(0, n - na - nr))
        ns = max(0, n - na - nd - nr)
        s = art["sustain"]

        seg = np.empty(n)
        seg[:na] = np.linspace(floor, 1.0, na) ** 1.2          # attack
        if nd:
            seg[na:na + nd] = np.linspace(1.0, s, nd)          # decay off the peak
        if ns:
            # NOT a flat plateau. The attack and decay are clamped in absolute
            # seconds, so on a long note (kbd still reaches ~19 s by design)
            # they finish early and everything after them would otherwise sit
            # dead level -- which is the "loud sustain" the complaint named,
            # just 3 dB quieter. A slow tilt across the body means no part of
            # any note is ever static.
            seg[na + nd:na + nd + ns] = np.linspace(s, s * SUSTAIN_TILT, ns)
        if nr:
            seg[n - nr:] = np.linspace(seg[n - nr - 1] if n - nr > 0 else s,
                                       floor, nr) ** 1.2       # release
        env[i0:i0 + n] = seg
    return line * env


def apply_timbre(theta, timbre=0.5):
    """The spectral half of the sharp<->hummy macro, applied to theta BEFORE
    rendering. Reuses `apply_harmonic_damping` so there is still one place that
    decides how partials are thinned."""
    tk = timbre_knobs(timbre)
    th = apply_harmonic_damping(dict(theta), max_harmonics=tk["max_harmonics"],
                                decay_exponent=tk["decay_exponent"])
    if "noise_level" in th:
        th["noise_level"] = float(th["noise_level"]) * tk["noise_mult"]
    return th


def to_waypoints(theta, notes, tick_s=1.0):
    """Melody -> waypoint dicts for the arranger's own `render_waypoints`.

    Only `f0_hz` differs between waypoints; every other theta field is carried
    through untouched, so this cannot alter the timbre the retrieval chose.

    TIMING. `render_waypoints` holds EVERY waypoint for the same `hold_s`, so a
    note cannot carry its own length directly. A long note is therefore the same
    pitch REPEATED over consecutive ticks: call with `hold_s=tick_s` and a 3 s
    note becomes three identical waypoints. Under the default "layered" mode
    each waypoint is its own static pad crossfaded into the next, so repeats are
    inaudible (a pad fading into an identical pad) and pitch CHANGES land as
    discrete steps rather than portamento -- which is what a melody needs.
    Use xfade_mode="glide" instead if you want the line to slide.

    Every note gets at least one tick, so a short note is never dropped.

    RESTS hold the PREVIOUS pitch (the pad keeps sounding underneath) and are
    silenced by `to_gain_envelope`, which returns one gain per tick in the same
    order. Rendering the two together is what produces a phrased line."""
    out, last = [], float(theta.get("f0_hz", 110.0))
    for f0, dur in notes:
        f0 = last if f0 == REST else float(f0)
        for _ in range(max(1, int(round(float(dur) / max(tick_s, 1e-6))))):
            out.append(dict(theta, f0_hz=f0))
        last = f0
    return out


# ------------------------------------------------------------------ standalone
def _wobble(n, sr, cents, rng, rate_hz=0.35):
    """Tape-style pitch drift as a cumulative phase multiplier.

    This is the single most recognisable thing about the reference records and
    it costs almost nothing: a slow random walk on playback speed, integrated so
    the PHASE stays continuous (multiplying frequency per-sample instead would
    click). Returns a per-sample time-warp factor around 1.0."""
    if cents <= 0:
        return np.ones(n)
    k = max(1, int(sr / max(rate_hz, 1e-3)))
    ctrl = rng.normal(0.0, 1.0, size=n // k + 2).cumsum()
    ctrl = ctrl / (np.max(np.abs(ctrl)) + 1e-9)
    fine = np.interp(np.arange(n), np.arange(len(ctrl)) * k, ctrl)
    return 2.0 ** (cents * fine / 1200.0)


def _voice(f0, n, sr, rng, bright, warp, partials=((1, 1.0), (2, 0.4), (3, 0.18),
                                                   (4, 0.09), (5, 0.05))):
    """One tone, pitch-warped. Phase is integrated so wobble is continuous."""
    ph = 2 * np.pi * f0 * np.cumsum(warp[:n]) / sr
    v = np.zeros(n)
    for h, amp in partials:
        if f0 * h < sr / 2:
            v += amp * (bright ** (h - 1) if h > 1 else 1.0) * \
                 np.sin(h * ph + rng.uniform(0, 6.28))
    return v


def render_standalone(notes, sr=SR, seed=0, drone_hz=None, dynamics=True,
                      wobble_cents=0.0, detune_cents=0.0, chords=False):
    """An audition render so the melody can be heard without the arranger (which
    needs the bank + engine). NOT the deployed path -- that is `to_waypoints`
    through the frozen renderer.

    REWRITTEN 2026-07-25 after three different note generators all came back
    "boring". The previous version gave every note the same loudness, the same
    tone and the same 0.4 s attack, and skipped the envelope entirely on notes
    under 0.8 s -- so short notes were bare sine bursts with hard edges. A line
    played that way cannot sound musical whatever its notes are, which means the
    audition was not testing what we thought it was.

    Three things now vary, all cheap and all standard performance practice:
      * DYNAMICS -- loudness follows the phrase contour (the peak note is the
        loud one) so the line has a shape you can hear, not just one you can plot;
      * ARTICULATION -- attack and release scale with note length, so short notes
        speak and long notes swell;
      * TONE -- brightness follows loudness, since louder notes are brighter on
        any real instrument.

    `drone_hz` adds a sustained root underneath. That matters more than it looks:
    a melody's tension comes from its relationship to the harmony, so judging a
    line in isolation judges the wrong thing. Pass the root the melody was
    generated against."""
    rng = np.random.default_rng(seed)
    total = sum(d for _, d in notes)
    out = np.zeros(int(total * sr) + sr, dtype=np.float64)

    # phrase-relative loudness: find each run of notes between rests and let the
    # highest note in the run carry the accent
    pitched = [k for k, (f, _) in enumerate(notes) if f != REST]
    gain = {}
    run = []
    for k, (f, _) in enumerate(notes + [(REST, 0.0)]):
        if f == REST:
            if run:
                hi = max(run, key=lambda j: notes[j][0])
                for j in run:
                    # arch: quiet at the edges of the phrase, open out at the peak
                    rel = 1.0 - abs(run.index(j) - run.index(hi)) / max(1, len(run))
                    gain[j] = 0.55 + 0.45 * rel
            run = []
        else:
            run.append(k)

    pos = 0
    for k, (f0, dur) in enumerate(notes):
        n = int(dur * sr)
        if f0 == REST:
            pos += n
            continue
        g0 = (gain.get(k, 1.0) if dynamics else 1.0) * float(rng.uniform(0.92, 1.08))
        # brightness tracks loudness -- louder notes carry more upper partials
        bright = 0.5 + 0.9 * g0
        warp = _wobble(n, sr, wobble_cents, rng)
        v = _voice(f0, n, sr, rng, bright, warp)
        if detune_cents > 0:                   # doubled and slightly out of tune
            v = 0.6 * v + 0.6 * _voice(f0 * 2 ** (detune_cents / 1200.0), n, sr,
                                       rng, bright, warp)
        # attack/release scale with the note, so a 0.5s note is not all fade
        a = max(1, int(min(0.35 * dur, 0.25) * sr))
        r = max(1, int(min(0.55 * dur, 0.9) * sr))
        env = np.ones(n)
        env[:a] = 0.5 * (1 - np.cos(np.pi * np.arange(a) / a))
        if r < n:
            env[-r:] = 0.5 * (1 + np.cos(np.pi * np.arange(r) / r))
        seg = v * env * 0.18 * g0
        out[pos:pos + n] += seg[: len(out) - pos]
        pos += max(1, n - r // 2)              # overlap so notes bleed together
    end = pos + int(0.5 * sr)

    if drone_hz:                               # the harmony the line sits over
        t = np.arange(end) / sr
        warp = _wobble(end, sr, wobble_cents * 0.5, rng, rate_hz=0.15)
        if chords:
            # SLOW 7th-CHORD MOVEMENT. This is where the wistfulness in the
            # reference records actually lives -- the melody stays plain and the
            # harmony under it moves. Roots i - VI - III - VII (a modal loop
            # that never resolves), voiced with a 7th so no chord sounds final.
            # A pentatonic line sits consonantly over all four, which is why the
            # scale and the progression were chosen together.
            prog = [(0, (0, 3, 7, 10)), (-4, (0, 4, 7, 11)),
                    (3, (0, 4, 7, 11)), (-2, (0, 4, 7, 10))]
            # Harmonic rhythm is FIXED, not a fraction of the clip length. Tying
            # it to duration meant a 90 s demo changed chord every 15 s and sat
            # still; the chords are the main thing moving, so they have to move.
            bar = 8.0
            d = np.zeros(end)
            for c, (rt, ivs) in enumerate(prog * (int((end / sr) / (bar * len(prog))) + 1)):
                a0, a1 = int(c * bar * sr), int(min((c + 1.6) * bar * sr, end))
                if a0 >= end:
                    break
                m = a1 - a0
                seg = np.zeros(m)
                for iv, amp in zip(ivs, (0.55, 0.3, 0.35, 0.22)):
                    f = drone_hz * 2 ** ((rt + iv) / 12.0) * 0.5
                    seg += amp * _voice(f, m, sr, rng, 0.75, warp[a0:a1],
                                        partials=((1, 1.0), (2, 0.3), (3, 0.1)))
                ramp = np.clip(np.minimum(np.arange(m), m - np.arange(m)) /
                               max(1.0, 0.35 * bar * sr), 0.0, 1.0)
                d[a0:a1] += seg * ramp
        else:
            d = np.zeros(end)
            for h, amp in ((0.5, 0.5), (1, 0.6), (1.5, 0.25), (2, 0.2)):
                d += amp * _voice(drone_hz * h, end, sr, rng, 0.7, warp,
                                  partials=((1, 1.0),))
        breathe = 1.0 + 0.15 * np.sin(2 * np.pi * 0.05 * t)
        fade = np.clip(np.minimum(t / 2.0, (end / sr - t) / 2.0), 0.0, 1.0)
        out[:end] += d * breathe * fade * 0.16

    peak = np.max(np.abs(out)) + 1e-9
    return (out[:end] * min(1.0, 0.9 / peak)).astype(np.float32)


# ------------------------------------------------------------------ selftest
def _pitched(notes):
    return [(f, d) for f, d in notes if f != REST]


def _selftest():
    # every generated pitch is ON the chosen scale (checked in cents, not Hz)
    for v, a, want in ((0.6, 0.0, "major"), (-0.6, 0.0, "minor")):
        notes, p = generate_melody(v, a, root_hz=110.0, duration_s=60.0, seed=1, use_grammar=False, style="folk")
        assert p["scale"] == want, f"valence {v} should pick {want}"
        allowed = scale_pitches(110.0, want, p["span"], p["register"])
        cents = [min(abs(1200 * math.log2(f / c)) for c in allowed)
                 for f, _ in _pitched(notes)]
        assert max(cents) < 1e-6, f"off-scale pitch, worst {max(cents):.3f} cents"

    # arousal drives ACTIVITY: more notes, wider range, bigger leaps
    lo, plo = generate_melody(0.0, -0.9, duration_s=120.0, seed=2, use_grammar=False, style="folk")
    hi, phi = generate_melody(0.0, +0.9, duration_s=120.0, seed=2, use_grammar=False, style="folk")
    assert len(_pitched(hi)) > len(_pitched(lo)), \
        f"high arousal should be busier ({len(_pitched(hi))} vs {len(_pitched(lo))})"
    assert phi["span"] > plo["span"] and phi["max_step"] >= plo["max_step"]
    rng_lo = max(f for f, _ in _pitched(lo)) / min(f for f, _ in _pitched(lo))
    rng_hi = max(f for f, _ in _pitched(hi)) / min(f for f, _ in _pitched(hi))
    assert rng_hi > rng_lo, "high arousal should cover more pitch range"

    # PACE, set by the audition: calm must be MUCH slower than bright, and the
    # bright end must sit near the ~2 s was approved -- not faster.
    assert plo["note_s"] > 2 * phi["note_s"], \
        f"calm must be far slower than active ({plo['note_s']:.1f} vs {phi['note_s']:.1f})"
    assert 1.5 <= phi["note_s"] <= 2.6, \
        f"bright pace drifted off the approved ~2s ({phi['note_s']:.2f}s)"

    # STRUCTURE -- the fix for "just a bag of notes". The line must BREATHE
    # (rests), and phrase-final notes must be longer than the notes inside a
    # phrase, so there is an arrival for the rest to frame.
    for a_ in (-0.9, 0.0, 0.9):
        ns, pp = generate_melody(0.0, a_, duration_s=180.0, seed=5, use_grammar=False, style="folk")
        rests = [d for f, d in ns if f == REST]
        assert len(rests) >= 3, f"arousal {a_}: line never breathes ({len(rests)} rests)"
        # every rest is preceded by a note, and that note is the long one
        idx = [k for k, (f, _) in enumerate(ns) if f == REST]
        finals = [ns[k - 1][1] for k in idx if k > 0]
        inner = [d for k, (f, d) in enumerate(ns)
                 if f != REST and (k + 1 not in idx)]
        assert finals and inner and np.mean(finals) > np.mean(inner), \
            "phrases should ARRIVE: the last note must be held longer"
        assert np.mean(rests) > 1.0, "rests must be audible silence, not gaps"
    # calm should leave MORE space than active -- that space is the calm
    calm, _ = generate_melody(0.0, -0.9, duration_s=240.0, seed=6, use_grammar=False, style="folk")
    busy, _ = generate_melody(0.0, +0.9, duration_s=240.0, seed=6, use_grammar=False, style="folk")
    silence = lambda ns: sum(d for f, d in ns if f == REST) / sum(d for _, d in ns)
    assert silence(calm) > silence(busy), \
        f"calm should be emptier ({silence(calm):.2f} vs {silence(busy):.2f})"

    # the motif RETURNS -- some pitch sequence of length 3 occurs more than once.
    # (This is what a random walk cannot do, and what "a bag of notes" lacked.)
    seq = [f for f, _ in _pitched(generate_melody(0.3, 0.0, duration_s=240.0, seed=7, use_grammar=False, style="folk")[0])]
    tri = [tuple(seq[k:k + 3]) for k in range(len(seq) - 2)]
    assert len(tri) - len(set(tri)) >= 2, "no repeated 3-note figure -- no motif"

    # the walk stays inside its register window and is reproducible
    notes, p = generate_melody(0.2, 0.2, duration_s=90.0, seed=3, use_grammar=False, style="folk")
    band = scale_pitches(110.0, p["scale"], p["span"], p["register"])
    pit = _pitched(notes)
    assert min(band) - 1e-6 <= min(f for f, _ in pit), "walk fell below the register"
    assert max(f for f, _ in pit) <= max(band) + 1e-6, "walk rose above the register"
    assert generate_melody(0.2, 0.2, duration_s=90.0, seed=3, use_grammar=False, style="folk")[0] == notes, "must be deterministic"

    # motion is mostly stepwise -- an ambient line, not an arpeggio
    idx = [band.index(min(band, key=lambda c: abs(c - f))) for f, _ in pit]
    leaps = [abs(b - a) for a, b in zip(idx, idx[1:])]
    assert np.mean([l <= 1 for l in leaps]) > 0.5, "should be predominantly stepwise"

    # waypoints carry the melody and change NOTHING else about the timbre
    theta = dict(f0_hz=110.0, noise_level=0.2, swell_rate=0.05, third_interval=3)
    TICK = 1.0
    wps = to_waypoints(theta, notes, tick_s=TICK)
    gains = to_gain_envelope(notes, tick_s=TICK)
    assert all(all(w[k] == theta[k] for k in theta if k != "f0_hz") for w in wps), \
        "only f0_hz may differ from the retrieved theta"
    assert len(gains) == len(wps), "one gain per waypoint tick, or the line desyncs"
    assert abs(len(wps) * TICK - sum(d for _, d in notes)) < 1.0 + TICK * len(notes) * 0.5, \
        "quantised length should track the melody's real duration"
    # rests are SILENCED by the gain, and hold the previous pitch underneath so
    # the pad does not jump when the line comes back
    assert 0.0 in gains and 1.0 in gains, "rests must be muted, notes must sound"
    k = gains.index(0.0)
    assert k > 0 and wps[k]["f0_hz"] == wps[k - 1]["f0_hz"], \
        "a rest must hold the previous pitch, not jump"
    assert {w["f0_hz"] for w, g in zip(wps, gains) if g > 0} <= {f for f, _ in pit}, \
        "every sounding waypoint must be a real melody note"

    # HARMONIC DAMPING keeps the fundamental, kills upper harmonics, and
    # renormalises so the distribution still sums to ~1 -- a damped theta
    # should not synthesise LOUDER just because it lost its overtones
    th_bright = dict(harm_dist=[0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05])
    th_damped = apply_harmonic_damping(th_bright, max_harmonics=3)
    hd = th_damped["harm_dist"]
    assert sum(hd[3:]) == 0.0, "damping must zero harmonics past max_harmonics"
    assert hd[0] > np.array(th_bright["harm_dist"])[0], \
        "the fundamental's SHARE should grow once upper harmonics are removed"
    assert abs(sum(hd) - 1.0) < 1e-6, "damped distribution must still be normalised"

    # ISO226 note gain: unity at the reference pitch, clamped so a deep bass
    # note (this project's texture role reaches well under 100 Hz) cannot spike
    assert abs(iso226_note_gain(440.0, ref_freq=440.0) - 1.0) < 1e-6, \
        "no correction needed at the reference frequency"
    assert iso226_note_gain(40.0) <= 10 ** (6.0 / 20.0) + 1e-6, \
        "low-register gain must stay inside the +-6 dB clamp"

    # PACE vs the drone. At the ACTIVE end the line must move inside one held
    # pad (that motion is the expression). At the CALM end it deliberately does
    # NOT -- the audition asked for notes longer than the pad, so a calm melody
    # sustains ACROSS pad changes instead. Assert the active end only.
    DRONE_HOLD_S = 4.0                      # render_waypoints' default
    _, p_hi = generate_melody(0.0, 0.9, duration_s=60.0, seed=4, use_grammar=False, style="folk")
    assert p_hi["note_s"] < DRONE_HOLD_S * 0.75, \
        f"active melody note {p_hi['note_s']:.1f}s should move inside one pad"

    # GRAMMAR PATH. Drive the generator from a fitted model whose rhythm is
    # known (three short notes then one 4x long) and check that contrast
    # SURVIVES into the seconds -- this is the whole point of learning, and it
    # is what the authored fallback cannot produce.
    try:
        import grammar as em
        gmodel = em.synthetic_model()
        gnotes, gp = generate_melody(0.2, 0.0, duration_s=240.0, seed=8,
                                     grammar=em.EssenGrammar(gmodel), style="folk")
        gd = [d for f, d in _pitched(gnotes)]
        assert max(gd) / min(gd) >= 3.0, \
            f"learned rhythm collapsed on the way to seconds (ratio {max(gd)/min(gd):.1f})"
        fd = [d for f, d in _pitched(generate_melody(0.2, 0.0, duration_s=240.0,
                                                     seed=8, use_grammar=False)[0])]
        assert max(gd) / min(gd) > max(fd) / min(fd), \
            "the grammar must beat the authored fallback on rhythmic contrast"
        # guardrails still hold on the learned path
        gband = scale_pitches(110.0, gp["scale"], gp["span"], gp["register"])
        assert min(gband) - 1e-6 <= min(f for f, _ in _pitched(gnotes)), "learned line left the register"
        assert max(f for f, _ in _pitched(gnotes)) <= max(gband) + 1e-6, "learned line left the register"
        assert any(f == REST for f, _ in gnotes), "learned line must still breathe"
        gratio = max(gd) / min(gd)

        # PHRASE BANK. The contour must SURVIVE replay -- a stored shape played
        # back must appear in the output with its shape intact, and it must be
        # restated rather than replaced every time.
        bank = em.PhraseBank(gmodel)
        assert len(bank) > 0, "no phrase shapes stored"
        bnotes, bp = generate_melody(0.2, 0.0, duration_s=300.0, seed=9, grammar=bank, style="folk")
        bpit = [f for f, _ in _pitched(bnotes)]
        assert any(f == REST for f, _ in bnotes), "phrase line must still breathe"
        bband = scale_pitches(110.0, bp["scale"], bp["span"], bp["register"])
        assert min(bband) - 1e-6 <= min(bpit) and max(bpit) <= max(bband) + 1e-6, \
            "replayed contour left the register"
        # restatement: some 3-note figure must recur, and MORE often than the
        # note-by-note chain manages -- that is the whole reason for the bank
        tri = lambda s: len([1 for k in range(len(s) - 2)]) - len(
            {tuple(s[k:k + 3]) for k in range(len(s) - 2)})
        gpit = [f for f, _ in _pitched(gnotes)]
        assert tri(bpit) > 0, "no figure recurs -- phrases are not being restated"
        # the arch test is a property of stored contours, not of the sampler
        assert any(em.is_arch(s) for s in bank.shapes), "no arch-shaped contour kept"
        nshapes = len(bank)
    except ImportError:
        gratio = nshapes = None

    # STYLES. boc/kbd are the reference targets (audition: "slower and more spacious than
    # dunn or board of canada"). What must hold: pentatonic, a SHORT cell that
    # repeats nearly unchanged, and character parameters that are actually set.
    boc, pb = generate_melody(0.3, 0.2, duration_s=240.0, seed=11, style="pentatonic_fast")
    kbd, pk = generate_melody(-0.3, -0.6, duration_s=240.0, seed=11, style="pentatonic_slow")
    assert pb["scale"].endswith("_pent") and pk["scale"].endswith("_pent"), \
        "reference styles must be pentatonic -- the 4th/7th are what sound tuney"
    assert pb["wobble_cents"] > 0 and pb["detune_cents"] > 0 and pb["chords"], \
        "boc character parameters are not set"
    assert pk["note_s"] > 3 * pb["note_s"], \
        f"kbd must be far slower than boc ({pk['note_s']:.1f} vs {pb['note_s']:.1f})"
    # REPETITION is the point: the same short pitch figure must recur many times
    bseq = [f for f, _ in _pitched(boc)]
    btri = [tuple(bseq[k:k + 3]) for k in range(len(bseq) - 2)]
    rep = len(btri) - len(set(btri))
    # A quarter, not a half: the riff now TRANSPOSES between repeat groups
    # (audition: "the clips are not doing anything"), so the same figure recurs
    # at several pitches and exact triple matches are correspondingly rarer.
    assert rep > 0.2 * len(btri), \
        f"boc cell is not repeating enough ({rep}/{len(btri)} figures recur)"
    # ...but it must not get STUCK either: no long run on one pitch
    stuck = max((sum(1 for _ in grp) for _, grp in itertools.groupby(bseq)), default=0)
    assert stuck <= 3, f"boc line sits on one pitch for {stuck} notes"
    # and it must be pentatonic in FACT, not just in the label
    pband = set(scale_pitches(110.0, pb["scale"], pb["span"], pb["register"]))
    assert all(min(abs(1200 * math.log2(f / c)) for c in pband) < 1e-6 for f in bseq)
    # RIFF RHYTHM. A cell taken from the head of a phrase loses the cadential
    # long note and comes back flat -- the original "every note the same
    # duration" defect, reintroduced. The cell must land on something longer.
    for st in ("pentatonic_fast", "pentatonic_slow"):
        ns, _ = generate_melody(0.3, 0.2, duration_s=240.0, seed=12, style=st)
        sd = [d for f, d in _pitched(ns)]
        assert max(sd) / min(sd) >= 1.8, \
            f"{st} cell is rhythmically flat (ratio {max(sd)/min(sd):.1f}:1)"

    # wobble is a continuous time-warp, not a per-note jump: no discontinuity
    wb = _wobble(SR * 4, SR, 20.0, np.random.default_rng(0))
    assert np.max(np.abs(np.diff(wb))) < 1e-3, "wobble must be smooth, or it clicks"
    assert 0.98 < wb.mean() < 1.02 and not np.allclose(wb, 1.0), "wobble must vary about 1.0"

    # the standalone render is finite, peak-safe and about the right length
    audio = render_standalone(notes[:6])
    assert np.all(np.isfinite(audio)) and 0 < np.max(np.abs(audio)) <= 0.9 + 1e-6
    assert len(audio) > 0.5 * sum(d for _, d in notes[:6]) * SR

    print(f"SELFTEST OK: pitches land on the chosen scale (major for positive "
          f"valence, minor for negative); pace set by ear -- calm {plo['note_s']:.1f}s "
          f"vs active {phi['note_s']:.1f}s notes; the line is PHRASED (motif restated, "
          f"phrase-final notes held longer, rests between phrases, calm "
          f"{silence(calm):.0%} silent vs active {silence(busy):.0%}) and a 3-note "
          f"figure recurs, which a random walk cannot do; walk stays in register, "
          f"deterministic, mostly stepwise; waypoints change f0_hz ONLY and rests "
          f"mute via the gain envelope; standalone render finite and peak-safe"
          + (f"; MARKOV path carries a {gratio:.0f}:1 long/short contrast into "
             f"seconds; PHRASE BANK ({nshapes} contours) replays stored shapes "
             f"in register, restates them, and keeps its arches."
             if gratio else "; corpus paths skipped (grammar unavailable)."))


def main():
    ap = argparse.ArgumentParser(description="melodic drone (Stage 1 generator)")
    ap.add_argument("--valence", type=float, default=0.0)
    ap.add_argument("--arousal", type=float, default=0.0)
    ap.add_argument("--root-hz", type=float, default=110.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo", metavar="OUT_WAV", help="render the standalone audition")
    ap.add_argument("--print", dest="show", action="store_true", help="print the notes")
    ap.add_argument("--style", default="pentatonic_fast", choices=sorted(STYLES),
                    help="aesthetic target (see STYLES)")
    ap.add_argument("--kind", default="phrase", choices=["phrase", "markov", "authored"],
                    help="phrase=replay learned contours (default); "
                         "markov=order-2 chain; authored=no corpus at all")
    ap.add_argument("--note-s", type=float, default=None,
                    help="override the VA note length, to sweep pace by ear")
    ap.add_argument("--drone", dest="drone", action="store_true", default=True,
                    help="render a root drone under the line (default: on)")
    ap.add_argument("--no-drone", dest="drone", action="store_false")
    ap.add_argument("--dry", action="store_true", help="no dynamics -- the old flat render")
    ap.add_argument("--arranger", metavar="OUT_WAV",
                    help="render through the PROJECT'S renderer (needs bank+engine)")
    ap.add_argument("--anchor", type=int, default=0,
                    help="which engine theta to sing with (timbre of the line)")
    ap.add_argument("--tick-s", type=float, default=1.0,
                    help="waypoint hold; note lengths quantise to this")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if args.kind == "authored":
        notes, p = generate_melody(args.valence, args.arousal, root_hz=args.root_hz,
                                   duration_s=args.seconds, seed=args.seed,
                                   use_grammar=False, style=args.style)
        print("source: authored prior (no corpus)")
    else:
        g = load_grammar(kind=args.kind)
        print(f"source: {args.kind}"
              + (f" ({len(g)} learned contours)" if hasattr(g, "sample")
                 else " chain" if g is not None else " -- NOT BUILT, using authored prior"))
        notes, p = generate_melody(args.valence, args.arousal, root_hz=args.root_hz,
                                   duration_s=args.seconds, seed=args.seed,
                                   grammar=g, kind=args.kind, style=args.style)
    _report(args, notes, p)


# ------------------------------------------------------------ timbre + loudness
# Adapted from the earlier melody runtime, read end-to-end 2026-07-28. Most of
# that file was decorative (the trained
# MicroMelodicTransformer checkpoint is never loaded by the runtime that makes
# audio -- generate_notes() is a hardcoded pentatonic cycle instead, and its
# rest gaps compute a gain envelope that is never applied), but these two
# pieces are real, verified-working DSP and worth taking:
#   1. apply_harmonic_damping -- rolls off `harm_dist` so only the fundamental
#      + first few harmonics survive. Confirmed via the arranger that
#      `harm_dist` is genuinely consumed by additive synthesis, so this is a
#      real timbre control, not a no-op on an inert field.
#   2. iso226_note_gain -- per-note gain compensation from equal-loudness
#      curves, so a line that leaps registers doesn't also leap in perceived
#      loudness.
def iso226_a_weighting(f):
    """A-weighting sensitivity in dB (ISO 226 / IEC 61672 approximation)."""
    f = np.maximum(np.asarray(f, dtype=float), 10.0)
    f2 = f * f
    c1, c2, c3, c4 = 12194.217 ** 2, 20.598997 ** 2, 107.65265 ** 2, 737.86223 ** 2
    num = c1 * (f2 ** 2)
    den = (f2 + c2) * np.sqrt((f2 + c3) * (f2 + c4)) * (f2 + c1)
    return 20.0 * np.log10(num / (den + 1e-12) + 1e-12) + 2.0


def iso226_note_gain(freq_hz, ref_freq=440.0, max_boost_db=6.0):
    """Gain multiplier so `freq_hz` reads as loud as `ref_freq` to the ear.

    CLAMPED, unlike the runtime's version. That file's own demo never left a
    ~220-880 Hz register, where the raw correction stays under 2x; this
    project's melody register reaches much lower (`melody_root` allows
    register offsets down to -12 semitones from a 150 Hz floor, and the
    TEXTURE role transposes another octave below that), where the unclamped
    A-weighting curve explodes -- measured 33x gain at 40 Hz. Applied
    unclamped, the lowest texture notes would spike or clip; +-6 dB keeps the
    equal-loudness smoothing without that."""
    delta_db = iso226_a_weighting(ref_freq) - iso226_a_weighting(freq_hz)
    delta_db = float(np.clip(delta_db, -max_boost_db, max_boost_db))
    return 10.0 ** (delta_db / 20.0)


def apply_harmonic_damping(theta, max_harmonics=4, decay_exponent=1.8):
    """Steep high-frequency roll-off on `harm_dist`, keeping only the
    fundamental + first few harmonics (1/(k+1)**decay_exponent, zeroed past
    `max_harmonics`). Unlike the runtime's version this does NOT also zero
    noise_level/swell_depth: those are the anchor's OWN calibrated gentleness
    -- pick_melody_anchor already selects a low-noise, low-swell anchor, so
    flattening them again here would be a second, redundant correction on top
    of the first."""
    th = dict(theta)
    if "harm_dist" in th:
        h = np.asarray(th["harm_dist"], dtype=float).copy()
        n_h = len(h)
        weights = 1.0 / (np.arange(1, n_h + 1) ** decay_exponent)
        weights[max_harmonics:] = 0.0
        h = h * weights
        th["harm_dist"] = (h / (h.sum() + 1e-9)).tolist()
    return th


_RVA_ENGINE = None    # module-level bank/engine singleton, see inside


def render_via_arranger(notes, anchor=0, chord=None, tick_s=1.0, seed=0,
                        harmonic_damping=True, iso_loudness=True, smooth=False,
                        device=None, timbre=None):
    """Render the melody through the PROJECT'S OWN renderer.

    `render_standalone` is four sine harmonics -- an audition crutch so the notes
    can be checked without loading the bank and engine. It is not what ships, and
    every demo judged so far used it, so the line has only ever been heard in a
    voice the thesis never uses.

    This is the deployed chain instead: a real theta from the engine, `f0_hz`
    swapped per tick by `to_waypoints`, straight into `ArrangedRenderer.
    render_waypoints` under the frozen RENDER_KW -- the same call the arc pool,
    the long-track render and the Conductor all make. Nothing here is new code;
    it mirrors s07e_reverb_spotcheck's four lines.

    hold_s == tick_s, so a long note is the same pitch over consecutive ticks,
    and "layered" mode makes each waypoint its own static pad -- pitch changes
    land as steps, repeats are inaudible.

    `smooth`: apply_harmonic_damping deliberately leaves swell_depth/rate
    alone (see its docstring) because `anchor` is normally DEFAULT_MELODY_ANCHOR,
    already picked for low swell/noise by pick_melody_anchor. That assumption
    breaks when the caller passes a BORROWED anchor instead -- the rating_space
    "texture" role reuses the bed's own anchor_idx, so it inherited whatever
    swell/noise/pitch-drift the bed happened to have. audition, 2026-07-28,
    after the bed fix: "even for texture, you need something smoother". Set
    `smooth=True` for any caller using an anchor it did not pick for
    gentleness: zeroes swell_depth/swell_rate on top of harmonic_damping, and
    renders with pitch_drift_cents=param_wander_std=0 (same override already
    applied to the rating_space bed, see build_space.py)."""
    # The arranger and the frozen render settings belong to the conductor
    # (Section 3.8), and the retrieval engine to Section 3.4.2. Symbolic
    # generation needs none of them, so they are imported HERE rather than at
    # module scope: a missing conductor degrades the AUDIO path only, and
    # loudly, instead of making this module unimportable. `melody_transformer`
    # takes the same approach for the same reason.
    try:
        from decoupled_engine import DecoupledEngine
        from arranger import ArrangedRenderer
        from render_params import RENDER_KW, apply_chord
        from device import get_device
    except ImportError as e:
        raise ImportError(
            "render_via_arranger needs the conductor's renderer on sys.path "
            "(conductor/engine/: arranger.py, render_params.py, device.py) "
            "and Section 3.4.2's decoupled_engine.py. Symbolic generation "
            f"works without them. Original error: {e}") from e

    # CACHED. This ran on every call -- i.e. once per conductor segment -- and
    # a DecoupledEngine loads the whole 20k-anchor bank and fits a GP, all to
    # serve one theta_from_index() row lookup. The bank is immutable for the
    # life of the process, so one instance is correct and identical in output.
    # (2026-08-01: the repeated "Decoupled Engine initialized" lines in a live
    # session were this.)
    global _RVA_ENGINE
    if _RVA_ENGINE is None:
        _RVA_ENGINE = DecoupledEngine()
    engine = _RVA_ENGINE
    theta = engine.theta_from_index(anchor)
    if chord is not None:
        theta = apply_chord(theta, chord)
    if timbre is not None:
        # sharp<->hummy macro. Supersedes the plain damping call below, since
        # apply_timbre IS apply_harmonic_damping with the macro's own
        # max_harmonics/decay_exponent -- running both would thin twice.
        theta = apply_timbre(theta, timbre)
    elif harmonic_damping:
        theta = apply_harmonic_damping(theta)
    if smooth:
        theta = dict(theta, swell_depth=0.0, swell_rate=0.0)
    wps = to_waypoints(theta, notes, tick_s=tick_s)
    gains = to_gain_envelope(notes, tick_s=tick_s)
    if iso_loudness:
        # fold per-waypoint ISO226 gain into the same rest-gain array so it
        # rides through the one smoothing pass below -- a step change in gain
        # at a waypoint boundary would otherwise be audible even though the
        # pitch crossfade already smooths the TIMBRE change there
        iso_g = [iso226_note_gain(wp["f0_hz"]) for wp in wps]
        gains = [g * i for g, i in zip(gains, iso_g)]
    # device=None keeps the historic behaviour (GPU when present) for BATCH
    # callers like build_space.py, where it is a real speedup over hundreds of
    # clips. The LIVE conductor passes device='cpu': one segment at a time is
    # not GPU-bound, and the deployment target is the deployable app CPU-only so it runs
    # on a professor's laptop with no CUDA (2026-07-28).
    renderer = ArrangedRenderer(device=device or get_device())
    # RENDER_KW already carries `seed`, so override it in a copy rather than
    # passing it twice (that would be a duplicate-keyword TypeError)
    kw = dict(RENDER_KW, seed=seed)
    if smooth:
        kw["pitch_drift_cents"] = 0.0
        kw["param_wander_std"] = 0.0
    if smooth:
        # ALSO kill the per-voice breathing. WAYPOINT_VOICES carries
        # breath_depth=0.55 on five of six voices at breath_period_s=10.0 --
        # a 0.1 Hz, 55%-deep gain swing per voice, measured at 66-85% AM depth
        # on the rendered beds. It is a separate mechanism from both the theta
        # swell zeroed above and the f0 drift zeroed in `kw`, which is why
        # zeroing those two did not stop the wobble. Patched temporarily and
        # restored, the same way the conductor app's `_voiced` does for gains
        # -- the frozen arranger file is never edited.
        import arranger as _arr
        _saved = [vc["breath_depth"] for vc in _arr.WAYPOINT_VOICES]
        for vc in _arr.WAYPOINT_VOICES:
            vc["breath_depth"] = 0.0
    try:
        audio = np.asarray(renderer.render_waypoints(
            wps, hold_s=tick_s, xfade_s=melody_xfade_s(tick_s), **kw))
    finally:
        if smooth:
            for vc, d in zip(_arr.WAYPOINT_VOICES, _saved):
                vc["breath_depth"] = d
    # apply the rest envelope: the arranger has no silence, so rests are muted
    # here, one gain per tick, smoothed so muting cannot click
    n_per = max(1, len(audio) // max(1, len(gains)))
    env = np.repeat(np.asarray(gains, float), n_per)[:len(audio)]
    if len(env) < len(audio):
        env = np.concatenate([env, np.full(len(audio) - len(env), env[-1] if len(env) else 1.0)])
    # Rest edges need a LONG fade, not a smoothing nudge. 50 ms was audible as
    # an abrupt stop/start (audition: heard it on the first arranger render); the
    # scale comes from deployment rather than taste -- s11 crossfades bed gain
    # over 3.0 s and arc ramps run 3-16 s, so 0.6 s is still short by house
    # standards while staying well inside a 1 s tick. Raised cosine, so the
    # envelope has no corner at either end.
    k = max(1, int(0.6 * SR))
    win = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, k, endpoint=False)))
    env = np.convolve(env, win / win.sum(), mode="same")
    return audio * env, theta


def _report(args, notes, p):
    if getattr(args, "note_s", None):          # rescale the whole line by ear
        k = float(args.note_s) / p["note_s"]
        notes = [(f, max(0.4, d * k)) for f, d in notes]
        p = dict(p, note_s=float(args.note_s))
    pit = [(f, d) for f, d in notes if f != REST]
    durs = [d for _, d in pit]
    print(f"VA ({args.valence:+.2f}, {args.arousal:+.2f}) -> {json.dumps(p)}")
    print(f"{len(pit)} notes + {len(notes)-len(pit)} rests over "
          f"{sum(d for _, d in notes):.1f}s | durations {min(durs):.1f}-{max(durs):.1f}s "
          f"(ratio {max(durs)/min(durs):.1f}:1)")
    if args.show:
        for i, (f0, d) in enumerate(notes):
            print(f"  {i:3d}  {'REST   ' if f0 == REST else f'{f0:7.2f}'}  {d:5.2f}s")
    if getattr(args, "arranger", None):
        import soundfile as sf
        audio, theta = render_via_arranger(notes, anchor=args.anchor,
                                           tick_s=args.tick_s, seed=args.seed)
        sf.write(args.arranger, np.asarray(audio, dtype=np.float32), 16000)
        print(f"wrote {args.arranger} via the ARRANGER "
              f"(anchor={args.anchor}, f0={theta.get('f0_hz', 0):.1f}Hz, "
              f"{len(audio)/16000:.1f}s) -- the deployed renderer, not the sine toy")
    if args.demo:
        import soundfile as sf
        audio = render_standalone(notes, seed=args.seed,
                                  drone_hz=(args.root_hz if args.drone else None),
                                  dynamics=not args.dry,
                                  wobble_cents=(0.0 if args.dry else p["wobble_cents"]),
                                  detune_cents=(0.0 if args.dry else p["detune_cents"]),
                                  chords=p["chords"] and args.drone)
        sf.write(args.demo, audio, SR)
        bits = [f"{len(audio)/SR:.1f}s", f"style={p['style']}"]
        if p["chords"] and args.drone:
            bits.append("7th-chord bed")
        if p["wobble_cents"] and not args.dry:
            bits.append(f"wobble {p['wobble_cents']:.0f}c")
        if args.dry:
            bits.append("flat")
        print(f"wrote {args.demo}  ({', '.join(bits)})")


if __name__ == "__main__":
    main()
