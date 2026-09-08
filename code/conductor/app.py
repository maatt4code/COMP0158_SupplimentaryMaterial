"""The Unfinished Conductor: the live application (Section 3.8).

This is where every other section becomes one thing you can listen to. A
session is an OU walk through valence-arousal space; at each step the walk is
kept inside the region the bank can actually voice, a preset is retrieved for
the current target, a transition into it is chosen, and about 15-30 seconds of
audio is rendered ahead of playback.

WHAT IT WIRES TOGETHER

  3.3   the measured timbre prior behind the preset bank
  3.4.2 retrieval over 20,000 labelled anchors, and the boundary guard
  3.5   the arc policy, the preference GP, and the bed scheduler
  3.6   the measured reverb spaces
  3.7   the melodic line, when it is switched on

It loads FROZEN weights and reads no rating data. That is the train/inference
boundary, and `smoke_test.py` enforces it by importing the engine with the
repo removed from `sys.path` entirely.

CONTINUOUS, HONESTLY DESCRIBED. Audio is rendered in segments ahead of
playback and auto-queued, so it *feels* continuous rather than being
sample-by-sample streaming synthesis. The timer is re-armed after every
segment to that segment's real duration. It first ran on a fixed 10 s period,
which was shorter than every real segment (14 s normal, 13-26 s triggered,
since ramp_s in {3, 6, 12, 16} plus two holds), so the player was handed a
replacement file mid-playback on every tick -- most often mid-crossfade.
Listeners heard stuttering every 8-10 seconds and transitions that never
resolved, because the period always cut in before the arc had finished.

HOW A TRANSITION IS CHOSEN. When the target's valence moves away from the
current state by more than RISE_THRESHOLD -- symmetric in both directions,
because the pool holds real preference data for darkening transitions too --
or when the listener forces one, the conductor asks the arc policy for the
best transition at the live context by BOTH available methods, and logs both
whichever one renders:

  * `lookup_best_arc`    the honest baseline: the best already-rated arc at
                         this context.
  * `best_by_gp_predict` the preference GP's posterior over a dense grid of
                         policy-controllable parameters (ramp x chord x glide
                         x texture, ~800 combinations scored in one batch),
                         ranked by LOWER CONFIDENCE BOUND rather than raw
                         mean. Argmax of the mean was observed picking
                         parameter values far outside anything ever rated --
                         extrapolation, not interpolation -- and the
                         uncertainty penalty is what keeps the policy from
                         chasing GP noise in unrated territory.

Which one renders is switchable live, so the baseline can be HEARD rather
than merely logged. Lookup mode replays the winning arc's transition RECIPE
on today's retrieved presets -- not that arc's pre-rendered audio.

WHY THERE IS A RE-RANKER. `lookup_best_arc` is memoryless: it picks the best
arc for the current context independently every time, so nothing models order
and it will happily emit two large brightenings back to back. In lookup mode
the near-tie band -- candidates the preference GP likes about equally -- is
re-ranked toward the corpus type that most coherently follows the last arc
played, using Section 3.5's corpus type-transition matrix.

**The discipline matters more than the mechanism: human ratings decide what is
GOOD, and the corpus chain only breaks TIES.** It never selects outside the
near-tie band, so a preference judgement is never overridden by a statistic.
It degrades to plain best-utility when the typology is missing.

THE SURFACE. One skin ships, the deck: five panels grouped by PROVENANCE, each
badged with where its behaviour came from, so a listener can see which knob is
backed by ratings and which is authored. The skin returns its components in a
dict keyed by name and `skins.validate` enforces that contract, so a surface
that forgets a control fails at construction instead of rendering buttons that
quietly do nothing.

Human overrides: valence/arousal sliders plus "Set target", which takes effect
on the NEXT tick rather than interrupting audio in flight, and "Force
transition now", which bypasses the automatic trigger.

Run:
  python app.py                    # the real engine, bank and arc pool
  python app.py --no-share         # local only
  python app.py --selftest         # control flow only: fake engine and
                                   # renderer, no server, no audio
"""

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ENGINE, UI_DIR, RUNTIME = HERE / "engine", HERE / "UI", HERE / "runtime"
WEIGHTS, ASSETS = HERE / "weights", HERE / "assets"
(HERE / "gradio_tmp").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(HERE / "gradio_tmp"))

# The conductor is self-contained (decision 12): everything it imports lives
# in one of its own three subdirectories, so this directory can be lifted out
# and run. It resolves NO dataset root and imports no `paths` module -- that
# is the train/inference boundary, and `smoke_test.py` asserts it by importing
# the engine with the repo removed from sys.path entirely.
for _p in (ENGINE, UI_DIR, RUNTIME):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# The VA pad background. It is a MEASUREMENT of the bank's reachable region,
# so there is one per label space: judge valence tops out at +0.24 and human
# reaches +0.57, leaving 38.5% vs 27.4% of the square unreachable. Showing the
# wrong one would invite the user to aim at targets the engine cannot hit.
PAD_IMG = ASSETS / "pad_judge.png"
PAD_HUMAN = ASSETS / "pad_human.png"

# The labelled retrieval banks, in ONE place. Every component that retrieves
# -- the main engine, and the melody layer's own lookup -- resolves them from
# here, because two components searching different banks would disagree about
# what a valence target even means. They used to be resolved inside the engine
# through a dataset-root helper; passing them in explicitly is what removed
# the last dataset dependency from the runtime.
BANKS = sorted((WEIGHTS / "banks").glob("*/labeled_index.csv"))

# ---- the engine: copies of the migrated inference modules -----------------
import arc_policy as policy                    # noqa: E402  arcs + preference GP
import scheduler as sched                      # noqa: E402  LiveBedScheduler
import coherence_reranker as reranker          # noqa: E402  sequential coherence
import bed_bank as bedbank                     # noqa: E402  bed bank + VA choice
from guard import BoundaryGuard                # noqa: E402
from render_params import apply_chord, HOLD_S, RENDER_KW   # noqa: E402
from loudness import aw_rms, AW_TARGET         # noqa: E402

# Optional layers. Each is a knob the user can turn on, so a missing module
# degrades that ONE control to off rather than taking the app down.
try:
    import reverb_bank as revbank              # noqa: E402  measured spaces
except ImportError:
    revbank = None
try:
    import melody_markov as meldrone           # noqa: E402  melodic line
except ImportError:
    meldrone = None

# ---- UI: the deck surface and its helpers ---------------------------------
import skins                                   # noqa: E402  the deck
import faders                                  # noqa: E402  drawn faders
import visuals as vis                          # noqa: E402  reactive visuals
import overlay_mixer as s10overlay             # noqa: E402  bed DSP: load/loop/mix
try:
    import webaudio                            # noqa: E402  gapless in-browser
except ImportError:
    webaudio = None
_WEBAUDIO = False                              # set by main() when --webaudio
try:
    import ring_player as ringplayer           # noqa: E402  continuous audio out
except ImportError:
    ringplayer = None
_RING = None                                   # set by main() when --audio-out

# ---- runtime overlays -----------------------------------------------------
try:
    import crackle                             # noqa: E402  surface noise
except ImportError:
    crackle = None
try:
    import distance as stereodist              # noqa: E402  near/far placement
except ImportError:
    stereodist = None

# Segment container for the BROWSER path. A 22 s segment is 704 KB as WAV and
# 96 KB as MP3 -- and over a public relay that 7.3x is the difference between
# a seamless join and a wait, because once render-ahead landed (2026-08-01)
# transport WAS the remaining gap. Lossy is safe here: the conductor never
# renders rated stimuli, so no measurement is affected. 'flac' is the lossless
# escape hatch at 1.6x.
SEG_FORMAT = "mp3"                             # mp3 | ogg | flac | wav

# THE APP'S DISPLAY NAME, in one place. It appeared three times in build_demo()
# -- browser tab title, the deck's folded intro accordion, and the surface's
# H1 -- so renaming it meant finding all three and getting them to agree.
# Every user-visible use reads THIS.
APP_NAME = "Unfinished Conductor"

# The transformer melody generator is held lazily: constructing it stands up
# its OWN retrieval engine and renderer, and paying that at startup for a
# generator that is off by default would double the bank load for nothing.
_MEL_TRANSFORMER = {}          # ckpt short name -> verified runtime
# Written at runtime, not shipped: session logs go under the conductor's own
# data/ so a checkout stays clean and nothing is written outside this tree.
CONDUCTOR_LOG = HERE / "data" / "conductor_log.csv"
CORPUS_HSMM = WEIGHTS / "hsmm_transitions.json"
BED_BANK_PATH = WEIGHTS / "bed_bank.json"
# Sequential-coherence re-ranker: the arc->corpus-type map plus the 5x5 corpus
# type-transition matrix, both fitted in Section 3.5. Loaded ONCE at import;
# either being absent makes reranker.rerank() degrade to plain best-utility,
# so the conductor runs unchanged without them.
# _selftest reassigns these globals to fake typologies, mirroring how it already
# redirects CONDUCTOR_LOG.
ARC_TYPES, COH_TRANS = reranker.load_coherence()
# ANNOTATION_LOG (2026-07-13): free-form rater reactions ("hate it", "VA just
# changed, need different settings", "same tone dragged on too long", ...),
# logged whenever the rater clicks, NOT tied to a triggered transition like
# CONDUCTOR_LOG -- a separate append-only file rather than a new CONDUCTOR_LOG
# column, since an annotation can happen at any step, not just on trigger.
# The two verdicts. Mutually exclusive within each pair, and deliberately
# short: this is an evaluation instrument now, not a tuning aid, so the
# rater answers "was it good" and "was it faithful" and nothing else. Any
# detail goes in the comment box, which is where improvement notes live.
# Two questions, each a plain Yes/No. The QUESTION carries the meaning and
# lives in the radio's label, so the options stay two short words -- which is
# what lets each group sit on its own row and leaves the width for the
# comment box, where the actionable detail actually goes.
LIKE_LABEL = "Like it?"
MATCH_LABEL = "Sounds like how it's supposed to feel?"
YESNO = ["Yes", "No"]
ANNOTATION_LOG = HERE / "logs" / "annotations.csv"
# SCHEMA CUTOVER 2026-08-08. The conductor has moved from collecting signal to
# IMPROVE it to collecting signal to EVALUATE it, so the free-form reason list
# is replaced by two Yes/No verdicts and the note box becomes an improvement
# comment. `reason_tag` is KEPT but no longer written -- it holds the old
# free-text reasons and stays readable in place. `liked` and
# `feels_like_sound` are new and empty on every row before this date; both are
# "Yes"/"No", and the column name is what says which question was asked. The header of an existing
# annotations.csv is migrated in place on first append (a .bak is written
# first), the same convention the texture-tuner log used.
ANNOTATION_FIELDS = ["timestamp", "session_id", "step", "context_v", "context_a",
                    "target_v", "target_a", "policy_mode", "liveliness",
                    "reason_tag", "liked", "feels_like_sound", "note"]
LOG_FIELDS = ["timestamp", "session_id", "step", "trigger", "context_v", "context_a",
             "fixed_dist", "fixed_glide_semitones", "policy_mode",
             "lookup_arc_id", "lookup_f", "lookup_std",
             "gp_ramp_s", "gp_chord_start", "gp_chord_end", "gp_is_glide",
             "gp_texture", "gp_f", "gp_std", "chosen_method",
             "chosen_ramp_s", "chosen_chord_start", "chosen_chord_end", "chosen_is_glide",
             "chosen_texture", "liveliness",
             "prev_type", "cand_type", "coherence", "rerank_reason", "rerank_arc_id"]
# CSV SCHEMA NOTE (2026-07-23, 5th cutover): the five coherence-re-ranker
# columns (board 11b). `prev_type` = the last rendered pool arc's corpus type
# (s16); `cand_type` = the type of the arc actually rendered; `coherence` =
# P(cand_type | prev_type) from s13's 5x5 corpus type-transition matrix;
# `rerank_reason` = s17's decision reason ("coherence_tiebreak"/"best_utility"/
# ...); `rerank_arc_id` = the pool arc that rendered (== `lookup_arc_id` unless
# the coherence tie-break moved it within the near-tie band). ALL blank on
# gp_predict-mode rows and every pre-cutover row (the re-ranker runs in LOOKUP
# mode only -- gp_predict scores synthesized grid combos that are never rendered
# and so cannot be audio-typed). Read a missing value as "not re-ranked".
# CSV SCHEMA NOTE (2026-07-13, 4th cutover): `gp_texture` and `chosen_texture`
# are the signed texture level the policy scored (gp_texture) and rendered
# (chosen_texture), added when the texture axis was un-pinned in s01_arc_policy
# (board 9d stage 2). Absent on every pre-2026-07-13 row (csv.DictReader returns
# None); those rows were rendered with texture pinned neutral, so read a missing
# value as 0.0 (none). DOMAIN WIDENED 2026-07-14: the level is now one of the
# 5-rung ladder {-1,-0.5,0,+0.5,+1} (policy.TEXTURE_CHOICES), not just {-1,0,+1}
# -- rows between 2026-07-13 and -14 only ever carry +-1/0.
# CSV SCHEMA NOTE (2026-07-12, 3rd cutover): `liveliness` is NEW -- a compact
# "thr=..,energy=..,wander=..,k=..,hold=..,xfade=.." string of the Tier A
# session settings active when the transition triggered (absent on all
# earlier rows). Defaults reproduce the pre-cutover constants, so old and
# new rows with liveliness at defaults are directly comparable.
# CSV SCHEMA NOTE (2026-07-09): the `trigger` column's vocabulary changed under
# the same append-only file. Rows logged before 2026-07-09 use "rise" (the
# original valence-rise-only trigger); rows from 2026-07-09 onward use
# "val_shift" for the same underlying condition, now bidirectional (see the
# module docstring's "CORRECTED 2026-07-09" note). "force" is unchanged in
# both eras. Treat "rise" and "val_shift" as the same trigger category when
# analyzing this log across the cutover.
# CSV SCHEMA NOTE (2026-07-09, 2nd cutover, SAME DAY): `policy_mode` and the
# four `chosen_*` columns are NEW as of the play-mode switch below -- absent
# in every row logged before this change (csv.DictReader returns None for
# them on old rows). Before this cutover `chosen_method` was hardcoded to
# "gp_predict" on every triggered row (gp_predict always rendered); read
# pre-cutover `chosen_method=="gp_predict"` rows as "the only mode that
# existed yet", not as a rater's deliberate choice.

CONTROL_HZ = 20
RISE_THRESHOLD = 0.15     # valence-target-shift trigger margin (bidirectional
                         # as of 2026-07-09 -- see module docstring)
STEP_SECONDS = 6.0        # OU-walk seconds advanced per tick (decoupled from
                         # rendered-audio duration, same as step03's waypoint
                         # thinning being decoupled from the 20Hz walk rate)
OU_DRIFT_K = 0.10
OU_NOISE_STD = 0.03
NORMAL_HOLD_S = 3.0
NORMAL_XFADE_S = 8.0


def voicing_multipliers(v, a, depth):
    """DARK VOICING (2026-07-12, the brief after the human-bank A/B: "the
    system tries too hard to have a key high frequency at all times,
    whereas for sad and dark times you can have just a single wandering
    sub-bass or a low key frequency"). Root cause: the arranger's
    WAYPOINT_VOICES stack is FIXED -- fifth (0.50) and air/octave (0.35)
    voices render on every pad regardless of VA, so darkness is
    structurally impossible, and the equal-loudness leveler then anchors
    segment loudness to that high content. This maps the CURRENT VA to
    per-voice gain multipliers: brightness = mean of (v+1)/2 and (a+1)/2;
    the air voice fades out below brightness 0.75 (gone by 0.35), the
    fifth below 0.6 (gone by 0.2), sub_bass gains up to +30% as it
    darkens. Root is never touched; the chord voices (min3/maj3) are
    never touched -- they carry the arc policy's LEARNED chord choice.
    depth in [0,1] blends between the frozen stack (0) and the full
    curve (1). Demo-expressive layer (Week-4 polish precedent): applied
    ONLY in the live conductor, never to rated stimuli; logged per
    transition in the liveliness column."""
    def ramp(x, lo, hi):
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
    b = ((v + 1.0) / 2.0 + (a + 1.0) / 2.0) / 2.0
    full = {"air": ramp(b, 0.35, 0.75), "pad_fifth": ramp(b, 0.20, 0.60),
            "sub_bass": 1.0 + 0.3 * (1.0 - b)}
    return {name: (1.0 - depth) + depth * m for name, m in full.items()}


def voicing_theta(theta, v, a, depth):
    """NOISE DUCK, second half of dark voicing (2026-07-12, feedback: after
    thinning fifth+air, the remaining "key high frequency at all times" is
    the ROOT voice's noise band -- and his own low-valence ratings favor
    hiss (factorial low profile: cutoff ~3.7 kHz), so retrieval at sad
    targets actively brings hissy anchors; "demo-dark" wants the opposite).
    Returns a COPY of theta with noise_level scaled toward 0.25x and
    noise_cutoff_hz pulled toward 0.4x (hiss -> rumble) as brightness
    falls, blended by the same depth slider. sess-stored thetas are never
    mutated -- voicing is a render-time-only view, same as the gain patch."""
    def ramp(x, lo, hi):
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
    b = ((v + 1.0) / 2.0 + (a + 1.0) / 2.0) / 2.0
    nl_full = 0.25 + 0.75 * ramp(b, 0.20, 0.75)
    nc_full = 0.40 + 0.60 * ramp(b, 0.20, 0.75)
    nl = (1.0 - depth) + depth * nl_full
    nc = (1.0 - depth) + depth * nc_full
    out = dict(theta)
    out["noise_level"] = theta["noise_level"] * nl
    out["noise_cutoff_hz"] = theta["noise_cutoff_hz"] * nc
    return out


from contextlib import nullcontext as _nullctx


class _voiced:
    """Temporarily scales WAYPOINT_VOICES gains around one render call and
    restores the frozen table afterwards -- the frozen arranger file is
    never edited (its sha stays valid for s03/s07's checks). The advance
    loop is single-threaded, so the patch window is race-free."""

    def __init__(self, mults):
        self.mults = mults or {}

    def __enter__(self):
        import arranger as arr
        self._arr, self._saved = arr, [vc["gain"] for vc in arr.WAYPOINT_VOICES]
        for vc in arr.WAYPOINT_VOICES:
            if vc["name"] in self.mults:
                vc["gain"] = vc["gain"] * self.mults[vc["name"]]

    def __exit__(self, *exc):
        for vc, g in zip(self._arr.WAYPOINT_VOICES, self._saved):
            vc["gain"] = g
        return False


def postprocess_segment(audio, arousal, sess, sr):
    """Loudness stage (2026-07-12, feedback: "the volume in the higher pitch
    is perceived too loud" -- correct: the live path had NO loudness
    handling beyond a flat peak guard, while every RATED stimulus was
    A-weighted-normalized to 0.015; equal raw RMS at ~200 Hz reads ~25 dB
    louder than at ~45 Hz, the exact gap s03 v2 measured on the long
    tracks). Two separated effects:

    1. EQUAL-LOUDNESS LEVELING (default ON): per-segment A-weighted
       normalization to the frozen 0.015 target -- removes the
       pitch-loudness artifact and RESTORES consistency with the rated
       protocol (the un-leveled live output was the deviation).
    2. AROUSAL -> LOUDNESS SPAN (explicit, logged): +-span_db across
       arousal [-1,1]. Leveling alone would also strip the loudness
       variation that legitimately carries arousal (the long-track judge
       traces went null when loudness was protocol-flattened) -- this
       knob reinstates it EXPLICITLY and controllably instead of as an
       uncontrolled side effect of retrieval. Demo-expressive layer under
       the Week-4 "polish, demo-only, no emotion->DSP claims" precedent;
       never applied to anything rated; settings logged per transition.

    2026-08-01: effect 2 MOVED to final_loudness(). It belongs at the end of
    the chain for two reasons -- it is the last word on how loud the segment
    is, and applied here it was a per-segment STEP of up to 2*span dB landing
    exactly on a join. There it is a ramp between consecutive segments, so the
    arousal gain is continuous by construction. This function is now purely the
    equal-loudness leveler; `arousal` is kept in the signature (unused) so the
    call site and the selftest stay unchanged.
    """
    if sess.get("level_loudness", True):
        audio = audio * (AW_TARGET / (aw_rms(audio, sr) + 1e-9))
    # NO peak guard here. It used to be a CONDITIONAL scale (if peak > 0.95),
    # which fires on peaky segments and not on quiet ones -- undoing the
    # levelling above for exactly half the material. Rare on measured material
    # (real segments level out at peak 0.2-0.5) but up to 7 dB when it does
    # fire, i.e. an occasional large unexplained drop. Peak safety is now owned
    # by final_loudness() at the END of the chain, where it is applied
    # consistently and with a slow release. Nothing clips in between: the
    # intermediate stages are float64.
    return audio


LOUD_CEILING = 0.95        # output peak ceiling
LOUD_MATCH_S = 5.0         # junction level-match window (correction decays over it)
LOUD_MEAS_S = 1.5          # window the edge levels are measured over
LOUD_MATCH_MAX_DB = 9.0    # most the junction match may correct by


def _soft_limit(audio, ceiling=LOUD_CEILING, knee=0.15):
    """Sign-preserving soft limiter. Linear below (ceiling - knee), tanh-
    compressed above, asymptotic at the ceiling. Only the transient tips are
    touched -- unlike a whole-segment rescale, which changes the loudness of
    material that was never in danger."""
    audio = np.asarray(audio, dtype=np.float64)
    t = max(ceiling - knee, 1e-6)
    a = np.abs(audio)
    over = a > t
    if not np.any(over):
        return audio
    out = audio.copy()
    out[over] = np.sign(audio[over]) * (
        t + (ceiling - t) * np.tanh((a[over] - t) / (ceiling - t)))
    return out


def final_loudness(audio, arousal, sess, sr):
    """FINAL loudness pass, after the whole chain (2026-08-01). Must stay last.

    Fixes the two remaining causes of the segment-to-segment loudness step,
    and owns peak safety for the chain.

    1. LEVEL WHAT IS HEARD. `postprocess_segment` levels the DRONE, but melody,
       bed, crackle and reverb are all added after it -- so the quantity held
       constant was never the one the listener hears, and it moved whenever a
       layer came or went. This measures the FINISHED mix.
    2. ONE CONTINUOUS GAIN ENVELOPE for everything that follows levelling --
       the arousal span and the headroom trim both. Each is a per-segment
       quantity, and a per-segment gain is a STEP at the join by definition:
       the arousal span alone put up to 2*span dB there, and the headroom trim
       (below) would have added its own. Instead the segment ramps from the
       gain the PREVIOUS segment ENDED on to the gain this one needs, so
       consecutive segments always meet at the same value. The joins are
       continuous by construction rather than by tuning a smoothing constant.

    3. JUNCTION MATCH. Levelling normalises a WHOLE segment, which says nothing
       about its EDGES -- and a segment has internal structure (hold, ramp,
       hold). Two segments can both sit at the target and still meet with the
       first ending on a bright pad and the second opening on a dark one.
       Measured on real pool renders that is the BIGGEST term, several dB at
       most joins, larger than the arousal step it was hiding behind. So the
       first LOUD_MATCH_S of each segment is nudged to meet the level the last
       one ended on, the correction decaying to nothing over that window
       (raised cosine, clamped to +-LOUD_MATCH_MAX_DB). Only the seam is
       touched; the segment's own dynamics are left alone.

    HEADROOM. A-weighted normalisation discounts low frequencies heavily, so
    levelling a dark segment can drive its raw peak past full scale (rare on
    measured material, but up to ~7 dB when it happens). The old per-stage
    `if peak > 0.95: scale` handled that by silently pulling that one segment
    down -- an unexplained drop. Here it is folded into the same envelope, and
    a soft limiter catches whatever is left.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        return audio.astype(np.float32)
    if not sess.get("level_loudness", True):
        return _soft_limit(audio).astype(np.float32)
    lev = audio * (AW_TARGET / (aw_rms(audio, sr) + 1e-9))
    peak_db = 20.0 * np.log10(LOUD_CEILING / (float(np.max(np.abs(lev))) + 1e-12))
    span = float(sess.get("arousal_gain_db", 5.0))
    end_db = min(span * float(np.clip(arousal, -1.0, 1.0)), peak_db)
    # start where the last segment left off -- but never above this segment's
    # own headroom, so the whole ramp stays under the ceiling
    start_db = min(float(sess.get("gain_end_db", end_db)), peak_db)
    sess["gain_end_db"] = end_db
    out = lev * (10.0 ** (np.linspace(start_db, end_db, len(lev)) / 20.0))
    out = _match_junction(out, sess, sr)
    out = _soft_limit(out)
    m = min(int(LOUD_MEAS_S * sr), len(out))
    sess["junction_aw"] = aw_rms(out[-m:], sr) if m > 1 else None
    return out.astype(np.float32)


def _match_junction(out, sess, sr):
    """Nudge the head of this segment to meet the level the last one ended on.

    The correction decays to nothing across LOUD_MATCH_S, so it fixes the seam
    without flattening the segment. Clamped, because a genuinely quiet opening
    is sometimes what the walk asked for -- this is meant to remove a join
    artefact, not to police dynamics."""
    n = len(out)
    w = min(int(LOUD_MATCH_S * sr), n)
    m = min(int(LOUD_MEAS_S * sr), n)
    prev = sess.get("junction_aw")
    if prev is None or w <= 1 or m <= 1:
        return out
    head = aw_rms(out[:m], sr)
    if head <= 0 or prev <= 0:
        return out
    corr_db = float(np.clip(20.0 * np.log10(prev / head),
                            -LOUD_MATCH_MAX_DB, LOUD_MATCH_MAX_DB))
    # never let the match boost the head into the limiter -- a seam fixed by
    # distorting the opening is not a fix
    head_room_db = 20.0 * np.log10(
        LOUD_CEILING / (float(np.max(np.abs(out[:w]))) + 1e-12))
    corr_db = min(corr_db, head_room_db)
    decay = 0.5 * (1.0 + np.cos(np.pi * np.arange(w) / w))       # 1 -> 0
    env = np.ones(n)
    env[:w] = 10.0 ** ((corr_db * decay) / 20.0)
    return out * env


# ------------------------------------------------------- soundscape bed (11a)
# Layer 5: a curated bed mixed under the drone, ARRANGED by the s11
# LiveBedScheduler as a causal function of absolute time (off / duck / bursts),
# selected from the s15 bed bank by the emotion (VA) gate. Default OFF -- every
# read is .get-defaulted so the selftest's hand-built sess and any pre-bed
# session render exactly as before.

_BED_AUDIO_CACHE = {}


def _resolve_bed(p):
    """Where a bed's audio actually is, given the bank stores a BASENAME.

    The bank keys beds by their original ESC-50 / Emo-Soundscapes filename,
    because the corpora are fetched rather than redistributed (decision 13):
    an absolute path would encode the build machine's layout and resolve
    nowhere else. Looked for first in the conductor's own `assets/beds/`,
    which is where the shipped demo subset lives, then under `$DRONE_BEDS` if
    the listener has fetched the full corpora.

    A bed that resolves nowhere is not an error here -- `_bed_audio` reports
    it and the layer carries on with the beds it has. Silently returning a
    path that does not exist is what gave the packaged app 83 beds that all
    failed to load."""
    q = Path(p)
    if q.is_absolute():
        return str(q)
    for root in (ASSETS / "beds", *( [Path(os.environ["DRONE_BEDS"])]
                                     if os.environ.get("DRONE_BEDS") else [] )):
        cand = root / q.name
        if cand.exists():
            return str(cand)
        hits = list(root.rglob(q.name)) if root.is_dir() else []
        if hits:
            return str(hits[0])
    return str(ASSETS / "beds" / q.name)          # reported, not silent
# how many beds warm_up() decodes ahead of time. The cache is unbounded and a
# bed is a whole wav in memory, so this is a deliberate ceiling rather than
# "all of them" -- the VA gate only ever reaches a handful per session.
BED_WARM_N = 12


def build_bed_scheduler(seed=0):
    """LiveBedScheduler over s11's 'overlay' preset: corpus_dwell (trained
    timing + hand-set arrangement, the recommended mode) when the s13 corpus
    fit is present, else hand-set. The causal live path, NOT sample_schedule."""
    corpus = None
    if CORPUS_HSMM.exists():
        try:
            corpus = json.load(open(CORPUS_HSMM))
        except (OSError, ValueError):
            corpus = None
    src = "corpus_dwell" if corpus else "handset"
    states, trans = sched.preset("overlay", trans_source=src, corpus=corpus)
    return sched.LiveBedScheduler(states, trans, seed=seed)


def load_bed_bank():
    """The bed bank, filtered to the beds whose AUDIO IS ACTUALLY HERE.

    The bank's own `resolved` flag records whether the corpus was present when
    the bank was BUILT, which is a different question from whether the audio
    is present now. Only a subset of the beds is redistributed (decision 13),
    so shipping the full bank unfiltered offered beds that could not be
    loaded: selection would pick one, `_bed_audio` would fail, and the bed
    layer would report itself broken for a bed the listener never had.

    Re-checking at load time makes the bank describe THIS installation. A
    listener who fetches the corpora and sets `$DRONE_BEDS` gets all of them
    back with no rebuild.
    """
    if not BED_BANK_PATH.exists():
        return []
    try:
        beds = json.load(open(BED_BANK_PATH)).get("beds", [])
    except (OSError, ValueError):
        return []
    here = []
    for b in beds:
        if b.get("path") and Path(_resolve_bed(b["path"])).exists():
            here.append(dict(b, resolved=True))
    if beds and not here:
        print(f"[beds] none of the {len(beds)} beds in the bank have audio "
              f"here; the bed layer will stay off. See "
              f"{ASSETS / 'beds' / 'ATTRIBUTION.md'}.")
    elif len(here) < len(beds):
        print(f"[beds] {len(here)} of {len(beds)} available "
              f"({len(beds) - len(here)} need the full corpora; set $DRONE_BEDS)")
    return here


def _bed_audio(path):
    if path not in _BED_AUDIO_CACHE:
        _BED_AUDIO_CACHE[path] = s10overlay.load_mono_16k(path)
    return _BED_AUDIO_CACHE[path]


# ---------------------------------------------------------------- distance
# NEAR / FAR PLACEMENT. Off by default; `distance_enabled` is the switch, and
# with it off this function returns the input array UNCHANGED, so every
# existing path is bit-for-bit what it was.
#
# WHY DISTANCE AND NOT WIDTH. The deck strip was drawn as STEREO|MONO, i.e. a
# width control, and width was already MEASURED as the wrong axis for this
# instrument (SideProjects/stereo/distance.py's own docstring): at the rated
# anchor sub_bass carries 60.5% of the energy and pad_root 31.9%, both centred
# because panning bass collapses a mix, while the only voice that actually
# moved was `air` at 7.6%. The other 92% sits at 63-126 Hz, where human azimuth
# localisation is poor whatever you do. Panning therefore moved 7.6% of the
# signal -- which is exactly why it came back "barely noticeable", and why that
# is structural rather than a tuning failure.
#   Distance is not: its cues act on the whole signal at any frequency, and the
# dominant one is the direct-to-reverberant RATIO, not level. That is why a
# far-away drone reads as far away rather than as someone turning it down.
#
# WHY IT RUNS LAST, AFTER final_loudness(). Two reasons, and both are the
# reason this is safe:
#   1. The chain above it stays MONO. `aw_rms` -- the primitive the frozen
#      loudness protocol runs on -- calls lfilter() with its default axis=-1,
#      which on an (N,2) array filters ACROSS THE TWO CHANNELS rather than
#      along time. It does not raise; it returns a number ~14 dB wrong, and
#      every segment would then be levelled to the frozen 0.015 target using
#      that. Keeping stereo strictly downstream of the last aw_rms call means
#      the trap is never sprung. (The same trap is already flagged in
#      SideProjects/rating_space/build_space.py.)
#   2. final_loudness() would otherwise CANCEL the effect: it levels every
#      segment to a fixed target, which is precisely the loudness cue distance
#      is trying to apply. Running afterwards leaves the D/R and air-absorption
#      cues -- the dominant ones -- intact.
#
# PEAK SAFETY still belongs to the chain, so the added reverberant energy is
# re-limited here with the same soft limiter.
DIST_XF_S = 6.0     # ease a distance change over this much of the segment head


def apply_distance(audio, sess, sr):
    """Place the finished mix near or far. Returns mono in, mono or (N,2) out.

    OFF unless sess['distance_enabled']; then the array is returned untouched.
    """
    if stereodist is None:
        return audio
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim != 1:                      # already stereo: nothing to place
        return audio
    d = float(np.clip(sess.get("distance", 0.35), 0.0, 1.0))
    cur = ("%.4f" % d) if sess.get("distance_enabled") else "off"
    prev = str(sess.get("dist_applied", "off"))
    if cur == "off" and prev == "off":
        return audio                     # the bit-identical guarantee, untouched
    # EASE EVERY CHANGE OF PLACEMENT (2026-08-09, feedback: "there was a
    # discontinuity just now when I changed the farness"). This stage runs
    # AFTER final_loudness, i.e. after the junction matching that makes every
    # other setting continuous at joins -- so a new distance was landing as a
    # STEP at the segment boundary, the one thing the rest of the chain goes
    # to lengths never to do (the arousal span is ramped for exactly this
    # reason). The segment where the setting changed therefore renders BOTH
    # placements and raised-cosine-fades between them across the head: at the
    # join it is 100% the OLD placement, which is what the previous segment
    # ended with, so the boundary is continuous by construction. Turning the
    # layer ON eases from dry, and OFF eases back to dry through one final
    # stereo segment before the pure-mono passthrough resumes.
    def _placed(mode):
        if mode == "off":
            return a, a                  # dry, duplicated: identical playback
        try:
            return stereodist.place(a, distance=float(mode))
        except Exception as exc:         # degrade LOUDLY, never take the app down
            print(f"[distance] disabled this step: {exc}")
            return None
    pair = _placed(cur)
    if pair is None:
        return audio
    l, r = pair
    if prev != cur:
        oldpair = _placed(prev)
        if oldpair is not None:
            ol, orr = oldpair
            n = len(a); k = max(1, min(n, int(DIST_XF_S * sr)))
            w = np.ones(n)
            w[:k] = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)
            l = ol * (1.0 - w) + np.asarray(l) * w
            r = orr * (1.0 - w) + np.asarray(r) * w
    sess["dist_applied"] = cur
    # MONO OUTPUT PATH (--audio-out). s18_ring_player is constructed with
    # channels=1, so pushing an (N,2) block would break it. Take the LEFT
    # channel rather than summing: L and R differ by a 7 ms reflection offset,
    # and summing them combs that into a notch around 140 Hz -- right in this
    # instrument's fundamental range. The left channel on its own is already a
    # complete distance rendering (air-absorbed direct + wet at the right
    # ratio); only the L/R decorrelation is lost, which a mono device could
    # not have reproduced anyway.
    if _RING is not None:
        return _soft_limit(np.asarray(l, dtype=np.float64))
    out = np.stack([l, r], axis=1)
    # WHOLE-MIX DRIFT (2026-08-09, feedback: on headphones the distance cues
    # read as space, not "stereo" -- measured L/R corr at the default 0.35 is
    # 0.76, i.e. subtle thickening). The rated verdict against panning was
    # about PER-VOICE panning, which moved 7.6% of the signal; a slow drift of
    # the WHOLE mix moves all of it, so it is the audible form of width that
    # measurement never ruled out. Equal-power balance, centre-unity, on
    # stereo_pad's own 45 s period -- a drift you notice over a phrase, not a
    # wobble. Phase is carried in the session so it is continuous across
    # segment joins (same pattern as the bed clock); the prerender copy
    # adopts it with the rest of the session.
    drift = float(sess.get("distance_drift", 0.0)) if cur != "off" else 0.0
    if drift > 0.0:
        n = out.shape[0]
        ph0 = float(sess.get("dist_pan_phase", 0.0))
        tt = ph0 + np.arange(n) / float(sr)
        sess["dist_pan_phase"] = ph0 + n / float(sr)
        pan = drift * np.sin(2.0 * np.pi * tt / 45.0)
        th = (pan + 1.0) * np.pi / 4.0
        c45 = np.cos(np.pi / 4.0)
        out[:, 0] *= np.cos(th) / c45
        out[:, 1] *= np.sin(th) / c45
    return _soft_limit(out)


def _stable_drone_env(audio, sess, sr):
    """Per-sample drone loudness envelope in [0,1] for the bed's duck/bursts
    shaping, normalized by a RUNNING reference (EMA of the segment 95th-pct
    RMS) instead of s10.loud_env's per-window min-max -- so ducking depth stays
    consistent across segment joins rather than jumping (the normalization gap
    flagged for this wiring in board 11a)."""
    n = len(audio)
    hop, win = int(0.05 * sr), int(0.25 * sr)
    pad = np.concatenate([audio, np.zeros(win)])
    nf = max(1, (n + hop - 1) // hop)
    rms = np.array([np.sqrt(np.mean(pad[i * hop:i * hop + win] ** 2))
                    for i in range(nf)])
    k = max(1, int(1.0 / 0.05))
    kern = np.ones(k)
    rms = np.convolve(rms, kern, "same") / np.convolve(np.ones_like(rms), kern,
                                                        "same")
    seg_ref = float(np.percentile(rms, 95)) + 1e-9
    prev = sess.get("env_ref")
    ref = seg_ref if prev is None else 0.85 * prev + 0.15 * seg_ref
    sess["env_ref"] = ref
    env_fr = np.clip(rms / ref, 0.0, 1.0)
    return np.interp(np.arange(n), np.arange(nf) * hop, env_fr)


def apply_bed(audio, sess, sr):
    """Mix the selected bed under the leveled drone segment, arranged by the
    LiveBedScheduler across ABSOLUTE time. Advances sess['t_abs'] every call
    (even when disabled) so the bed clock stays continuous across segments.
    No-op when the bed is off / unset / no scheduler -- pre-bed sessions and
    the selftest's hand-built dicts are untouched."""
    audio = np.asarray(audio, dtype=np.float64)
    t0 = float(sess.get("t_abs", 0.0))
    n = len(audio)
    sess["t_abs"] = t0 + n / float(sr)
    scheduler = sess.get("bed_scheduler")
    bed = sess.get("bed")
    if not sess.get("bed_enabled") or scheduler is None or not bed \
            or not bed.get("path"):
        return audio
    try:
        wav = _bed_audio(_resolve_bed(bed["path"]))
    except (OSError, RuntimeError, ValueError):
        return audio                                    # unreadable bed -> skip
    bedw = s10overlay.loop_to_len(wav, n)
    awd, awb = aw_rms(audio, sr), aw_rms(bedw, sr)
    gain_db = float(sess.get("bed_gain_db", bed.get("gain_db") or -12.0))
    if awb > 0:
        bedw = bedw * (awd * (10.0 ** (gain_db / 20.0)) / awb)
    env = _stable_drone_env(audio, sess, sr)
    shape = scheduler.bed_env(t0, n, env, sr=sr,
                              context=tuple(sess["curr_va"]),
                              duck_depth=float(sess.get("bed_duck_depth", 0.7)),
                              burst_s=float(sess.get("bed_burst_s", 3.0)))
    # no peak guard -- final_loudness() owns peak safety for the whole chain
    return audio + bedw * shape


def apply_crackle(audio, sess, sr):
    """Mix the SideProjects vinyl-crackle overlay under a finished segment.
    No-op when disabled / the module is unavailable, so pre-crackle sessions and
    the selftest render exactly as before. Seeded per step so it varies but stays
    reproducible for a given session."""
    audio = np.asarray(audio, dtype=np.float64)
    if not sess.get("crackle_enabled") or crackle is None:
        return audio.astype(np.float32)
    return crackle.mix_under(
        audio, sr=sr,
        gain_db=float(sess.get("crackle_gain_db", -18.0)),
        density=float(sess.get("crackle_density", 1.0)),
        intensity=float(sess.get("crackle_intensity", 0.5)),
        hiss=float(sess.get("crackle_hiss", 0.15)),
        seed=int(sess.get("step", 0)))


def melody_style_for(sess):
    """Which STYLE the line plays in.

    'auto' is an AUTHORED convention, not a measured mapping: low arousal picks
    the Kyle-Bobby-Dunn setting (very long tones, almost no movement), anything
    livelier picks the Boards-of-Canada one (a short cell repeated over slow
    chord movement). Named after the references the brief gave. It is declared
    here rather than buried so a rating round can overturn it -- the same status
    as the VA->behaviour table in melody_markov.py."""
    st = str(sess.get("melody_style", "auto"))
    if st != "auto":
        return st
    return "pentatonic_slow" if float(sess.get("curr_va", (0.0, 0.0))[1]) < -0.35 else "pentatonic_fast"


def apply_melody(audio, sess, sr, theta):
    """Mix a generated melodic line over a finished segment.

    GENERATED LIVE, never banked. The notes cost microseconds and the second
    render runs ~20x faster than realtime (measured), so the line is produced
    per segment against THIS segment's theta and VA. Pre-generating a bank would
    be cheaper and would also break the point: the melody could no longer
    respond to the live VA state, which is the closed loop the thesis claims.

    The line sings with the SEGMENT'S OWN THETA -- only `f0_hz` differs -- so its
    timbre is whatever retrieval chose and it cannot drift away from the drone.
    Placed BEFORE apply_bed/apply_reverb in the chain so the bed levels itself
    against the whole musical signal and the melody shares the drone's room
    (which is also what smooths the note edges).

    No-op when disabled or unavailable, so existing sessions and the selftest
    render exactly as before."""
    audio = np.asarray(audio, dtype=np.float64)
    if not sess.get("melody_enabled") or meldrone is None:
        return audio.astype(np.float32)
    try:
        v, a = (float(x) for x in sess.get("curr_va", (0.0, 0.0)))
        # octave-shift the drone's fundamental into a register the line can be
        # HEARD in. Passing the raw f0 put the melody at or below the drone's own
        # root (measured: 31-46 Hz notes under a 49 Hz drone), where it reads as a
        # second sub-bass rather than a tune. Whole octaves only, so the pitch
        # class -- and therefore the harmony -- is unchanged.
        root = meldrone.melody_root(float((theta or {}).get("f0_hz", 110.0) or 110.0))
        secs = len(audio) / float(sr)
        # NOTE SOURCE is the only thing this switch changes. Everything below --
        # the rated melody anchor, the swell envelope, A-weighted levelling, the
        # broadband and narrowband ducks -- runs identically for both, so the
        # two engines are comparable on the notes alone (see melody_transformer).
        env_name = str(sess.get("melody_envelope", "swell"))
        # legato is the one envelope that also changes the NOTES: its macro
        # bounds note lengths (pentatonic_slow was otherwise producing a single held tone
        # per minute). The other three are pure post-render amplitude curves,
        # so they share the legacy note generation untouched.
        _art = meldrone.LEGATO if env_name == "legato" else None
        if str(sess.get("melody_engine", "markov")) == "transformer":
            notes = transformer_notes(v, a, root, max(2.0, secs),
                                      ckpt=sess.get("melody_ckpt", "cpu"))
        else:
            notes, _p = meldrone.generate_melody(
                v, a, root_hz=root, duration_s=max(2.0, secs),
                seed=int(sess.get("step", 0)), style=melody_style_for(sess),
                articulation=_art)
        line, _th = meldrone.render_via_arranger(
            # `or`, NOT .get's default: new_session() sets melody_anchor=None
            # explicitly, so the KEY EXISTS and .get never reaches its default
            # -- int(None) raised on every segment, and apply_melody's blanket
            # handler turned that into "[melody] disabled this step". The layer
            # therefore never ran in a fresh session (2026-08-01).
            notes, anchor=int(sess.get("melody_anchor")
                              or meldrone.DEFAULT_MELODY_ANCHOR),
            tick_s=float(sess.get("melody_tick_s", 1.0)),
            seed=int(sess.get("step", 0)), device="cpu",
            # The retrieval banks, passed explicitly. The melody renderer
            # resolves no dataset root of its own, and it caches the engine it
            # builds, so this costs nothing after the first segment.
            banks=BANKS)
    except Exception as e:                        # never take the session down
        print(f"[melody] disabled this step: {e}")
        return audio.astype(np.float32)

    # NOTE ENVELOPE. All four are selectable here (the listening test carries
    # only swell + legato -- see build_space.MELODY_ENVELOPES for why), because
    # in the conductor an extra option costs a dropdown entry rather than a
    # rater's attention. The brief's audition, 2026-08-01:
    #   swell  -- smoothest. The shipped default. Slow attack, no percussive
    #             edge (a fast attack on this synth's bright anchors rated as
    #             "car horns").
    #   legato -- the survivor of the articulation experiment: bounded note
    #             lengths, envelope terms clamped in seconds as well as scaled,
    #             boundaries aligned to the crossfade.
    #   struck -- reproduces the professors' "attack not finished, then loud
    #             sustain" exactly. Kept as an audible reference, not a default.
    #   flat   -- no envelope; consecutive same-pitch notes stop being
    #             distinguishable. The control that justifies having any.
    _line = np.asarray(line, dtype=np.float64)
    _tick = float(sess.get("melody_tick_s", 1.0))
    if env_name == "legato":
        line = meldrone.shape(_line, notes, sr=sr, tick_s=_tick,
                              articulation=meldrone.LEGATO)
    elif env_name == "struck":
        line = meldrone.articulate(_line, notes, sr=sr, tick_s=_tick,
                                   attack_s=0.02, release_s=3.0, floor=0.05)
    elif env_name == "flat":
        line = _line
    else:
        line = meldrone.swell(_line, notes, sr=sr, tick_s=_tick)
    if line.size == 0:
        return audio.astype(np.float32)
    if len(line) < len(audio):                    # match the segment exactly
        line = np.pad(line, (0, len(audio) - len(line)))
    line = line[:len(audio)]
    # level the line RELATIVE to the drone in PERCEIVED loudness (A-weighted
    # RMS), not raw amplitude. Raw RMS treats every frequency equally; the ear
    # does not (equal-loudness contours) -- the same physical amplitude reads
    # far louder in the 2-5kHz band than at 40Hz, so a fixed level_db drifted
    # whenever a different note/anchor put its energy somewhere else in the
    # spectrum. This recomputes per segment, so it tracks whatever note the
    # generator actually picked.
    d_rms = aw_rms(audio, sr) + 1e-9
    l_rms = aw_rms(line, sr) + 1e-9
    gain = (d_rms / l_rms) * (10.0 ** (float(sess.get("melody_level_db", -14.0)) / 20.0))
    # DUCK the drone where the line sings -- a sparse line under a continuous
    # five-voice drone is masked however loud it is, level alone was never the
    # real lever (build_space.py's build_melody finding). Two effects stack:
    # a broadband gain dip (the audible fix) plus a gentle narrowband dip
    # centred on the melody's own pitch (the brief, 2026-07-26, re a third
    # suggestion -- baked in but kept shallow so it reads as headroom, not EQ
    # movement).
    voiced_drone = audio
    duck = np.ones(len(audio))
    duck_db = float(sess.get("melody_duck_db", -6.0))
    if duck_db < 0:
        env = np.abs(line)
        k = max(1, int(0.15 * sr))
        env = np.convolve(env, np.ones(k) / k, mode="same")
        env = env / (env.max() + 1e-9)
        duck = 1.0 + env * (10.0 ** (duck_db / 20.0) - 1.0)
        voiced_drone = _narrowband_duck(audio, sr, root, env, depth_db=2.5, q=1.2)
    # no peak guard -- final_loudness() owns peak safety for the whole chain
    return (voiced_drone * duck + line * gain).astype(np.float32)


def _narrowband_duck(drone, sr, center_hz, env, depth_db=2.5, q=1.2):
    """Dip the drone's spectrum in a narrow band AROUND the melody's own
    pitch, wherever the melody is active (`env` in [0,1]) -- an acoustic slot
    for the line, instead of turning the whole drone down. Kept shallow and
    moderate-Q on purpose: a deep or wide cut is audible as EQ movement in its
    own right. RBJ peaking-EQ biquad (audio EQ cookbook); mirrors
    SideProjects/rating_space/build_space.py's copy of the same function."""
    import scipy.signal as sps
    center_hz = float(np.clip(center_hz, 40.0, sr / 2 - 100.0))
    w0 = 2.0 * np.pi * center_hz / sr
    alpha = np.sin(w0) / (2.0 * q)
    A = 10.0 ** (-abs(depth_db) / 40.0)
    b0, b1, b2 = 1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A
    a0, a1, a2 = 1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    notched = sps.lfilter(b, a, np.asarray(drone, dtype=np.float64))
    env = np.clip(env, 0.0, 1.0)
    return drone * (1.0 - env) + notched * env


def melody_checkpoint_choices():
    """The trained transformer checkpoints, as (label, value) pairs.

    All three trained sizes ship, because the capacity comparison is a claim
    the report makes and a dropdown is what makes it AUDIBLE rather than only
    a table of validation losses.

    This used to be far more elaborate, and the complexity was entirely
    downstream of one bad artefact. The deployed app carried a fourth file,
    `melodic_transformer_cpu.pt`, which was a COPY of whichever size was
    currently preferred. It read as a fourth model in the dropdown, so the
    label had to be recovered by loading the checkpoint and counting layers,
    and then de-duplicated against the original by comparing tensors -- all to
    undo the confusion the copy created. The alias does not ship (§2.7): the
    three real names do, `weights/` is the only place they live, and the
    section's smoke test asserts the copy is absent. Reading the names off
    disk is then the whole job.
    """
    names = sorted(p.stem.replace("melodic_transformer_", "")
                   for p in WEIGHTS.glob("melodic_transformer_*.pt"))
    return [(n, n) for n in names]


def melody_checkpoint_names():
    """Just the checkpoint NAMES -- what every loader wants.

    melody_checkpoint_choices() returns (label, value) pairs so the dropdown can
    show "L6_d128" instead of the meaningless "cpu". Everything that
    LOADS a checkpoint needs the bare name, and iterating the pairs silently
    builds a filename out of a tuple repr:
        melodic_transformer_('L6_d128', 'cpu').pt
    which is exactly what the Space's warm-up did. One accessor per audience.
    """
    return [c[1] if isinstance(c, (tuple, list)) else c
            for c in melody_checkpoint_choices()]


# ---- FIRST-RUN DEFAULTS (2026-08-07) --------------------------------------
# So a visitor -- a phone, a professor, anyone who has not been told what the
# controls do -- can press START and hear the system at its best. Every value
# here is also reachable from the UI; none of them is a new capability.
#
# START_VA is used by new_session AND by both skins' target faders and the pad
# markers, so the number the walk starts from and the number the display shows
# cannot drift apart.
START_VA = (-0.45, 0.10)
# Human labels by default. NOTE FOR THE WRITE-UP, because this is not cosmetic:
# it swaps the retrieval INSTRUMENT (the two label spaces disagree about the
# nearest anchor at 99.8% of VA targets, README SS12) and it leaves the arc
# policy querying judge-space contexts -- Phase B in SS33.4, still open and
# still unverified. Fine for listening; do not let it become the configuration
# a recorded result is claimed from without saying so.
DEFAULT_LABEL_SPACE = "human labels"
PREFERRED_CKPT = "L6_d128"
PREFERRED_REVERB = "stairwell"


def default_ckpt():
    """The transformer checkpoint the demo opens on, chosen DEFENSIVELY.

    A hardcoded name that is not on disk is not merely a poor default, it is a
    CRASH: gradio's Dropdown.preprocess rejects any value outside `choices` on
    every submit that touches the component, which took down the whole UI on
    2026-08-07 ("Value: cpu is not in the list of choices"). So state the
    preference and fall back to whatever the checkpoints directory actually
    offers.

    MATCH BY LABEL FIRST, not by filename. The packed app and the Space ship
    the L6_d128 weights AS `melodic_transformer_cpu.pt` -- the promoted-copy
    alias -- and no file of that name (verified in the 2026-08-07 Space log:
    "Loaded ... (L=6, d=128) from melodic_transformer_cpu.pt"). Matching the
    VALUE alone therefore missed, fell through to "first available", and only
    arrived at the right architecture by luck of the sort order. The LABEL is
    read from the checkpoint's own state dict, so it says what the weights
    actually are wherever they happen to live."""
    choices = melody_checkpoint_choices()
    for lab, val in choices:
        if lab == PREFERRED_CKPT:
            return val
    for lab, val in choices:                    # named outright, if it exists
        if val == PREFERRED_CKPT:
            return val
    return choices[0][1] if choices else "cpu"


def default_pad():
    """The VA map to OPEN with, matched to DEFAULT_LABEL_SPACE.

    `swap_pad` keeps the picture honest from then on, but it only runs on the
    dropdown's `.change` event and nothing fires that at load -- so with the
    default flipped to human on 2026-08-07 the app came up retrieving on human
    labels while SHOWING the judge map (same day). That is precisely
    the misreport the pad exists to prevent: the two manifolds differ (judge
    reaches +0.244 with 38.5% of the square unreachable, human +0.572 / 27.4%),
    so the grey region would have been drawn in the wrong place."""
    want = (PAD_HUMAN if str(DEFAULT_LABEL_SPACE).startswith("human")
            else PAD_IMG)
    if want.exists():
        return want
    return PAD_IMG if PAD_IMG.exists() else want


def default_reverb():
    """The room the demo opens on.

    PICKED, not taken off the top of the list. This returned `rooms[0]` for one
    revision, which is "nature" -- a field-recording IR whose tail carries
    broadband surface noise. With reverb defaulting ON at wet 0.35 that was
    audible on every segment and read as the CRACKLE layer being on (the brief,
    2026-08-07; his session log says `crackle=off` throughout, so it was never
    crackle). Alphabetical order is not a musical judgement.
    `stairwell` is the room the listening round actually favoured on a held
    drone -- +0.40 against dry +0.17 (README SS21) -- which is the closest
    thing to evidence this choice has.
    Same defensive rule as default_ckpt otherwise: fall back through the list
    rather than name a room that might not be packed, because a value absent
    from `choices` is rejected by gradio's Dropdown on every submit that
    touches it. "off" is always present and is the honest answer when no IR
    bank travelled."""
    rooms = [r for r in reverb_choices()
             if str(r).lower() not in ("off", "none")]
    if PREFERRED_REVERB in rooms:
        return PREFERRED_REVERB
    return rooms[0] if rooms else "off"


def melody_transformer(ckpt="cpu"):
    """the causal micro-transformer, built once per checkpoint on first use.

    ONLY `generate_notes` is used from it -- never `render_melodic_line` or
    `mix_over_drone`, which is where this deviates from the integration the transformer work
    proposed. Those two re-retrieve theta with `engine.retrieve(v, a)`: nearest
    VA over the raw 20,000-row bank, the "retrieval is not rating" defect this
    project has already had to fix twice (pick_melody_anchor, pick_bed_anchor),
    and they then apply their own ducking, A-weighted levelling and peak guard
    on top of the ones `apply_melody` already applies against the SEGMENT'S own
    theta. Calling them would double every stage and reintroduce a known bug.

    Routing the transformer's notes through the existing chain instead is also
    the only way the comparison means anything: Markov and transformer then
    differ in HOW THE NOTES WERE CHOSEN and in nothing else -- same voice, same
    envelope, same level, same duck. Swapping in the whole runtime would compare
    two pipelines and then credit the difference to the generator.

    LOAD IS VERIFIED HERE, and that is not defensive padding -- it closes a
    measured silent-failure hole. `LiveMelodicDroneRuntime.__init__` assigns
    `self.model` BEFORE calling `load_state_dict`, so when the load raises (the
    architecture is read from a sidecar .json, and a missing or stale one gives
    the wrong n_layers) the exception is caught and printed but `self.model`
    stays bound to a RANDOMLY-INITIALISED network. `generate_notes` then finds a
    non-None model, runs inference on random weights, raises nothing, and
    returns perfectly well-formed notes. Reproduced 2026-07-28: an orphaned
    checkpoint yielded 5 notes off untrained weights with one easily-missed
    startup line. Rating a "transformer" that is actually noise would be a
    silent data-integrity failure, so a checkpoint that did not verifiably load
    raises instead -- apply_melody's own handler then degrades the segment to
    pass-through and says why."""
    ckpt = str(ckpt or "cpu")
    if ckpt not in _MEL_TRANSFORMER:
        import torch as _t
        from melody_transformer import LiveMelodicDroneRuntime
        import melody_transformer as _rtmod
        # SUPPRESS the runtime's own DDSP stack. Its __init__ builds a whole
        # second DecoupledEngine (20k anchors + a GP fit) and ArrangedRenderer
        # for its standalone --demo mode; `generate_notes` never touches either
        # -- it reads self.model and self.tok_to_pitch and nothing else. We
        # take the NOTES only and render them through the conductor's own chain
        # (that is the point of the design: it is what makes the two engines
        # comparable on the notes alone). Left alone, that construction runs
        # INSIDE apply_melody inside advance(), i.e. on the first melody
        # segment of a live session, and the app appears to hang while it
        # rebuilds an engine the conductor already has (2026-08-01, the brief:
        # "thing froze when loading transformer weights").
        # Done by making the constructors fail: the runtime already wraps them
        # in `except Exception: pass` and leaves both None, so this uses its own
        # documented degrade rather than reaching into its internals.
        class _Suppressed:
            def __init__(self, *a, **k):
                raise RuntimeError("suppressed by the conductor: notes only")
        _saved = {n: getattr(_rtmod, n, None)
                  for n in ("DecoupledEngine", "ArrangedRenderer")}
        for n in _saved:
            if _saved[n] is not None:
                setattr(_rtmod, n, _Suppressed)
        try:
            rt = LiveMelodicDroneRuntime(ckpt_name=ckpt, force_cpu=True)
        finally:
            for n, v in _saved.items():
                if v is not None:
                    setattr(_rtmod, n, v)
        path = WEIGHTS / f"melodic_transformer_{ckpt}.pt"
        if rt.model is None:
            raise RuntimeError(f"transformer {ckpt}: no model was constructed")
        want = _t.load(path, map_location="cpu")
        got = rt.model.state_dict()
        if set(want) != set(got):
            raise RuntimeError(
                f"transformer {ckpt}: checkpoint/architecture mismatch -- the "
                f"live model is UNTRAINED. Check "
                f"melodic_transformer_{ckpt}.json matches the .pt.")
        # keys matching proves the shape; compare one real tensor to prove the
        # WEIGHTS actually landed rather than the load having failed silently
        if not _t.equal(want["tok_emb.weight"], got["tok_emb.weight"]):
            raise RuntimeError(
                f"transformer {ckpt}: weights did not load -- model is UNTRAINED")
        _MEL_TRANSFORMER[ckpt] = rt
    return _MEL_TRANSFORMER[ckpt]


def transformer_notes(v, a, root_hz, duration_s, ckpt="cpu"):
    """`generate_notes` output translated into THIS project's note convention.

    Their rest is `(f0, rest_dur)` carrying the PRECEDING note's pitch, meant to
    be silenced by a gain envelope their runtime never actually applies. Ours is
    the `REST` sentinel (pitch 0.0), which `to_gain_envelope` and `swell` both
    understand. Untranslated, every rest would sound as a held note, the line
    would come out continuous instead of phrased, and the transformer would rate
    badly for a reason that has nothing to do with the transformer.

    Rests are recoverable because `rest_dur = max(1.5, 5 - 2.5a)` is strictly
    greater than `note_dur = max(1.0, 3.5 - 1.5a)` at every arousal in [-1, 1],
    so the duration identifies them exactly."""
    notes = melody_transformer(ckpt).generate_notes(
        v, a, root_hz=root_hz, duration_s=duration_s)
    rest_dur = max(1.5, 5.0 - 2.5 * float(a))
    return [(meldrone.REST if abs(float(d) - rest_dur) < 1e-6 else f, d)
            for f, d in notes]


def melody_engine_choices():
    """Note generators available to the UI. The transformer is offered only
    when a checkpoint is actually on disk, so a missing .pt shows up as an
    absent option rather than a runtime traceback mid-session.

    ANY checkpoint counts -- NOT `melodic_transformer_cpu.pt` specifically,
    which is what this tested until 2026-08-07. That file is only the
    promoted-copy ALIAS that existed only on the build machine; a checkout
    could carry L6_d128 and friends without it, and gating on the alias HID A
    WORKING ENGINE there. The alias no longer ships at all, so the test is
    simply whether any real checkpoint is present."""
    has = any(WEIGHTS.glob("melodic_transformer_*.pt"))
    return ["markov"] + (["transformer"] if has else [])


def default_engine():
    """The note generator the demo opens on. Defensive for the same reason as
    default_ckpt: naming an engine that is not in `choices` is a crash, not a
    fallback."""
    return ("transformer" if "transformer" in melody_engine_choices()
            else "markov")


def melody_style_choices():
    """Styles available to the UI ('auto' + the authored presets)."""
    if meldrone is None:
        return ["auto"]
    try:
        return ["auto"] + sorted(meldrone.STYLES)
    except Exception:
        return ["auto"]


def reverb_choices():
    """Reverb conditions available to the UI ('off' + the measured ladder)."""
    if revbank is None:
        return ["off"]
    try:
        return ["off"] + [c for c in revbank.condition_ids() if c != "dry"]
    except Exception:
        return ["off"]


def apply_reverb(audio, sess, sr):
    """Convolution reverb over a CONTINUOUS stream, chosen live from the
    measured bank (SideProjects/reverb).

    Streaming, not clip-wise: `reverb_bank.apply()` edge-fades and lets the tail
    ring past the input, which is right for a rated clip and wrong here -- it
    would dip and lengthen every segment. Instead we convolve this segment and
    CARRY the overhanging tail in the session, adding it to the head of the next
    one, so a 5 s tail crosses segment joins the way it would in a real room.
    Output length always equals input length.

    Measured basis (2026-07-25 study, n=92): reverb shifts judged AROUSAL down
    0.3-0.4 and leaves valence unmeasurably changed -- so this is exposed as a
    manual character control, NOT wired into the VA mapping."""
    audio = np.asarray(audio, dtype=np.float64)
    cond = sess.get("reverb_cond", "off")
    wet = float(sess.get("reverb_wet", 0.35))
    # TURNING REVERB OFF LETS THE ROOM RING OUT (2026-08-01, feedback: "I can
    # really hear the change when I change reverb type"). This used to drop
    # `reverb_tail` on the floor, which cuts a 5 s tail dead mid-decay -- a
    # hard edit, not a fade. Now the remaining tail is played out over the
    # following segments and simply not replenished, which is what happens
    # when you stop exciting a real room.
    if revbank is None or cond in (None, "off", "dry") or wet <= 0:
        tail = sess.get("reverb_tail")
        if tail is None or len(tail) == 0:
            sess["reverb_tail"] = None
            return audio.astype(np.float32)
        n = len(audio)
        t = np.asarray(tail, dtype=np.float64)
        out = audio.copy()
        m = min(len(t), n)
        out[:m] += t[:m] * wet if wet > 0 else t[:m]
        sess["reverb_tail"] = t[n:].astype(np.float32) if len(t) > n else None
        return out.astype(np.float32)
    import scipy.signal as sps
    key = (cond, sr)
    if sess.get("_rev_key") != key:                # cache the IR per condition
        sess["_rev_ir"] = revbank.impulse_response(cond, sr=sr)
        sess["_rev_key"] = key
        # KEEP the old tail. It is just audio still ringing; letting the
        # previous room decay while the new one starts is a crossfade between
        # spaces, and costs nothing. Nulling it here was the audible chop.
    ir = np.asarray(sess.get("_rev_ir"), dtype=np.float64)
    if ir.size == 0:
        return audio.astype(np.float32)
    n = len(audio)
    rev = sps.fftconvolve(audio, ir)
    tail = sess.get("reverb_tail")
    if tail is not None and len(tail):             # ring-in from the last segment
        m = min(len(tail), len(rev))
        rev[:m] += np.asarray(tail, dtype=np.float64)[:m]
        if len(tail) > len(rev):
            rev = np.concatenate([rev, np.asarray(tail, np.float64)[len(rev):]])
    sess["reverb_tail"] = rev[n:].astype(np.float32) if len(rev) > n else None
    rev = rev[:n]
    # wet level set by RMS-matching over this segment, then mixed
    rev *= ((np.sqrt(np.mean(audio ** 2)) + 1e-9)
            / (np.sqrt(np.mean(rev ** 2)) + 1e-9))
    # no peak guard -- final_loudness() owns peak safety for the whole chain
    return (audio * (1.0 - wet) + rev * wet).astype(np.float32)


def log_decision(row):
    CONDUCTOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not CONDUCTOR_LOG.exists()
    with open(CONDUCTOR_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def log_annotation(row):
    ANNOTATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not ANNOTATION_LOG.exists()
    if not new:
        _migrate_annotation_header()
    with open(ANNOTATION_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ANNOTATION_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def _migrate_annotation_header():
    """Widen an existing annotations.csv to ANNOTATION_FIELDS, once.

    Appending a row with a new column to a file whose header lacks it would
    silently misalign every reader, so the file is rewritten with the new
    header and blanks for the missing column. Old rows are preserved exactly;
    a .bak is left beside the file. No-op once the header already matches."""
    try:
        with open(ANNOTATION_LOG, newline="") as f:
            rows = list(csv.DictReader(f))
            hdr = rows and list(rows[0].keys()) or []
        if not hdr or set(hdr) >= set(ANNOTATION_FIELDS):
            return
        shutil.copyfile(ANNOTATION_LOG, str(ANNOTATION_LOG) + ".bak")
        with open(ANNOTATION_LOG, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ANNOTATION_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in ANNOTATION_FIELDS})
    except Exception as exc:            # never let logging kill a session
        print(f"[annotations] header migration skipped: {exc}")


def liveliness_str(sess):
    """Compact Tier A settings snapshot, shared by CONDUCTOR_LOG's per-
    transition `liveliness` column and ANNOTATION_LOG's per-reaction one, so
    the two logs stay directly comparable."""
    return (f"thr={sess.get('trigger_threshold', RISE_THRESHOLD):.2f},"
           f"energy={sess.get('walk_energy', 1.0):.1f},"
           f"wander={sess.get('wander_std', 0.0):.2f},"
           f"k={int(sess.get('variety_k', 8))},"
           f"hold={sess.get('hold_s', NORMAL_HOLD_S):.0f},"
           f"xfade={sess.get('xfade_s', NORMAL_XFADE_S):.0f},"
           f"level={int(bool(sess.get('level_loudness', True)))},"
           f"again={sess.get('arousal_gain_db', 5.0):.0f},"
           f"voicing={sess.get('dark_voicing', 0.0):.1f},"
           f"breath={sess.get('tex_breath_depth', 0.0):.1f}/"
           f"cents={sess.get('cents_breath_depth', 0.0):.0f}/"
           f"rev={sess.get('reverb_breath_depth', 0.0):.1f}@"
           f"{sess.get('tex_breath_period', 30.0):.0f}s,"
           f"revspace={sess.get('reverb_cond', 'off')}"
           f"@{float(sess.get('reverb_wet', 0.0)):.2f},"
           f"bed={(sess.get('bed') or {}).get('bed_type', '?') if sess.get('bed_enabled') else 'off'}"
           f"@{sess.get('bed_gain_db', -12.0):.0f}dB,"
           f"crackle={('%.0fx@%.0fdB' % (sess.get('crackle_density', 1.0), sess.get('crackle_gain_db', -18.0))) if sess.get('crackle_enabled') else 'off'},"
           f"mel={((('tf' if sess.get('melody_engine') == 'transformer' else melody_style_for(sess)) + '/%s' % sess.get('melody_envelope', 'swell') + '@%.0fdB' % sess.get('melody_level_db', -14.0)) if sess.get('melody_enabled') else 'off')},"
           f"dist={('%.2f/%.2f' % (sess.get('distance', 0.35), sess.get('distance_drift', 0.0))) if sess.get('distance_enabled') else 'off'}")


def _esc(x):
    """Escape a cell value. Most are formatted by us and safe, but `bed_type`
    comes from bed_bank.json -- i.e. from data -- and a value carrying a `<`
    would otherwise break the table open."""
    import html as _h
    return _h.escape(str(x))


def settings_rows(sess):
    """The committed settings, as an ordered [(label, value)] list.

    This is the NEXT column of the LED readout: what `sess` holds right now,
    which is what the next segment will be rendered with. The NOW column is
    this same list captured at the moment the segment currently PLAYING was
    rendered (stashed as `sess["played"]` by `advance`), so both sides are
    always the same length, in the same order, and directly comparable row by
    row.

    WHY THIS IS TRUTHFUL RATHER THAN A GUESS (professors, 2026-08-07: "since
    the changes are not applied immediately, show current vs next"). Every
    setter calls `invalidate_prerender`, so touching any control DISCARDS the
    queued segment and it is rendered again from the session as it now stands.
    That is what makes "NEXT" an observed fact rather than an inference -- the
    exact distinction SS28.3 records as the cause of two silent failures here.

    Deliberately NOT derived from `liveliness_str`: that is a LOG format, keyed
    so conductor_log.csv and annotations.csv can be joined, and reshaping it to
    suit a display would break that join.

    The row set is FIXED -- every row is emitted whether or not its layer is on
    -- because a table that grows and shrinks is what makes the LED resize, and
    not resizing is half of what was actually asked for. (Nine rows since
    2026-08-09: distance joined as a permanent row, "off" when disabled --
    same rule, one constant row taller, never a dynamic resize. The splitter
    below already balances an odd count as 5+4.)"""
    g = sess.get
    tv, ta = (list(g("target_va", (0.0, 0.0))) + [0.0, 0.0])[:2]

    if g("melody_enabled"):
        eng = str(g("melody_engine", "markov"))
        voice = "transformer" if eng == "transformer" else melody_style_for(sess)
        melody = f"{voice} @{g('melody_level_db', -14.0):.0f}dB"
    else:
        melody = "off"

    if g("bed_enabled"):
        bed = (f"{(g('bed') or {}).get('bed_type', '?')} "
               f"@{g('bed_gain_db', -12.0):.0f}dB"
               + ("  by mood" if g("bed_auto") else ""))
    else:
        bed = "off"

    crackle = (f"{g('crackle_density', 1.0):.0f}/s @{g('crackle_gain_db', -18.0):.0f}dB"
               if g("crackle_enabled") else "off")

    cond = str(g("reverb_cond", "off"))
    reverb = "off" if cond == "off" else f"{cond} @{float(g('reverb_wet', 0.0)):.2f}"

    if g("distance_enabled"):
        dist = f"far{g('distance', 0.35):.2f}"
        if float(g("distance_drift", 0.0)) > 0.0:
            dist += f" dr{g('distance_drift', 0.0):.2f}"
    else:
        dist = "off"

    # LABELS AND VALUES ARE BOTH KEPT SHORT ON PURPOSE. The readout is laid out
    # as two columns of four (see settings_panel), which halves its height so it
    # fits the LED without the box having to grow -- and that only works if a
    # row stays on one line. Anything wordier belongs in the message above.
    return [
        ("mood",    f"v{tv:+.2f} a{ta:+.2f}"),
        ("policy",  "baseline" if g("policy_mode") == "lookup" else "learned"),
        ("melody",  melody),
        ("bed",     bed),
        ("crackle", crackle),
        ("reverb",  reverb),
        ("distance", dist),
        ("texture", f"dk{g('dark_voicing', 0.0):.1f} wn{g('tex_breath_depth', 0.0):.1f}"),
        ("pacing",  f"{g('hold_s', NORMAL_HOLD_S):.0f}s/{g('xfade_s', NORMAL_XFADE_S):.0f}s"),
    ]


# Inline styles, NOT css classes. gradio's Markdown sanitizes the HTML it
# renders, and whether `class` survives that is precisely the sort of thing
# this directory's notes say to measure rather than assume -- inline `style`
# survives either way, so the table cannot come out unstyled on the Space and
# be discovered by a professor rather than by us.
# COMPACT ON PURPOSE. The LED takes whatever height the map group leaves in the
# middle column, and the deck is meant to land on one screen -- so eight rows
# have to cost ~130px, not ~200px. Hence the tight line-height and the small
# type rather than a roomier table that would push the strips off the fold.
_NX_TD = ("padding:0 8px 0 0;white-space:nowrap;"
          "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;")


def settings_panel(sess):
    """The NOW / NEXT table that sits under the LED's message.

    Rows whose NEXT differs from NOW are highlighted and the header says how
    many are queued, so "I pressed APPLY and nothing happened" has a visible
    answer for the ~11-40s until the next segment starts.
    """
    if not sess:
        return ""
    nxt = settings_rows(sess)
    now = dict(sess.get("played") or [])

    def cell(label, val):
        """One setting as `label  value`, or `label  now > next` when queued.

        FOLDED INTO ONE CELL rather than separate NOW and NEXT columns
        (2026-08-07). Three columns x eight rows did not fit the LED, and the
        LED cannot be made taller without growing the whole middle column --
        it is height:100% of whatever the map leaves. Showing the old value
        only while it still differs says the same thing in a quarter of the
        space, and stops the steady state (everything in sync) from spending
        room on a column of identical pairs.
        """
        cur = now.get(label, None)
        diff = cur is not None and cur != val
        # amber = queued but not yet audible, the same colour as the APPLY
        # pulse so the two read as one story
        col = "#f59e0b" if diff else "#6b7280"
        txt = (f"{_esc(cur)} &#9656; {_esc(val)}") if diff else _esc(val)
        return (f"<td style='{_NX_TD}color:#4b5563'>{_esc(label)}</td>"
                f"<td style='{_NX_TD}color:{col};padding-right:16px'>{txt}</td>")

    changed = sum(1 for l, v in nxt
                  if now.get(l) is not None and now.get(l) != v)
    # two columns of four: half the height, and every row still one line
    half = (len(nxt) + 1) // 2
    body = ["<tr>" + cell(*nxt[i]) + cell(*nxt[i + half]) + "</tr>"
            if i + half < len(nxt) else "<tr>" + cell(*nxt[i]) + "</tr>"
            for i in range(half)]
    head = ("&#9656; %d queued -- takes effect next segment" % changed if changed
            else "in sync -- nothing queued")
    return (
        # PINNED TO THE BOTTOM (2026-08-07), which is safe here for one
        # specific reason: the row count is FIXED. settings_rows always emits
        # eight entries whatever the layers are doing, so this box has a
        # constant height and the clearance below can be a constant too.
        # That is what makes an out-of-flow box acceptable -- it grows upward
        # from its anchor, so a table that could change height would climb into
        # the message, which is exactly what happened when it briefly carried
        # nine rows. Keep the row count fixed or move it back into flow.
        # `bottom:44px` clears the foot: #conductor-eq is 30px tall at bottom:6px.
        # `.jledwrap` is the positioned ancestor, the same one the spectrum and
        # the status line hang off.
        "<div style='position:absolute;left:12px;right:12px;bottom:44px;"
        "border-top:1px solid #2b2f37;padding-top:4px'>"
        # tightened 2026-08-07 to buy back the height the shorter LED gave up:
        # line-height 1.25 -> 1.15 over four rows plus a smaller caption is
        # ~8px, which is the difference between the table clearing the message
        # and touching it.
        f"<div style='font-size:.76em;letter-spacing:.06em;line-height:1.2;"
        f"color:{'#f59e0b' if changed else '#4b5563'};margin-bottom:1px'>{head}</div>"
        "<table style='border-collapse:collapse;font-size:.78em;line-height:1.15'>"
        + "".join(body) + "</table></div>")


def ou_step(curr, target, guard, n_steps, energy=1.0):
    """Advance the OU walk n_steps control-rate steps toward target, through
    the s06 hybrid guard. Same dynamics as step03_demo.run_simulation,
    factored out so both the real app and _selftest exercise identical
    logic. `energy` (liveliness Tier A, 2026-07-12) scales BOTH the pull
    and the noise -- >1 moves faster and roams wider, still guard-bounded."""
    for _ in range(n_steps):
        g, _info = guard.guard_drift(curr)
        drift = energy * OU_DRIFT_K * (target - curr) + g
        noise = energy * OU_NOISE_STD * np.random.normal(0, 1, size=2)
        curr = np.clip(curr + drift + noise, -1.0, 1.0)
    return curr


def audible_dist_and_semitones(engine, theta_a, theta_b):
    """Measure audible-feature distance + pitch distance between two theta
    dicts, in the SAME Week-1.1 vetted space s07 used to compute audible_dist
    for the rated pool -- so the live policy query is scored against a
    fixed_dist that means the same thing as the pool's own values."""
    from retrieval import harmonic_centroid
    za = engine._audible_features(
        theta_a["f0_hz"], harmonic_centroid(np.asarray(theta_a["harm_dist"])),
        theta_a["swell_rate"], theta_a["swell_depth"], theta_a["noise_level"])
    zb = engine._audible_features(
        theta_b["f0_hz"], harmonic_centroid(np.asarray(theta_b["harm_dist"])),
        theta_b["swell_rate"], theta_b["swell_depth"], theta_b["noise_level"])
    dist = float(np.sqrt(((za - zb) ** 2).sum()))
    semis = float(abs(12.0 * np.log2(max(theta_b["f0_hz"], 1e-6) /
                                     max(theta_a["f0_hz"], 1e-6))))
    return dist, semis


def texture_of(m):
    """Signed texture level of a POOL ARC -- s09's canonical detection (explicit
    texture_signed field, else legacy override reverse-map), so the conductor's
    lookup path and the GP's training feature agree. In the 5-level ladder
    [-1..+1]; see policy.TEXTURE_GAINS."""
    return policy.texture_signed(m)


def texture_overrides(texture):
    """Signed texture level -> the theta_end_overrides that render it, via
    s09's canonical TEXTURE_GAINS map (so a conductor-rendered level is the same
    voicing the GP learned its preference from). Applied to the END pad after
    apply_chord. Snaps to the nearest defined level so the live loop never
    KeyErrors -- callers pass discrete TEXTURE_CHOICES values regardless."""
    nearest = min(policy.TEXTURE_GAINS, key=lambda L: abs(L - float(texture)))
    return policy.texture_overrides(nearest)


# ── TEXTURE BREATHING (Tier B, demo-only, 2026-07-14) ─────────────────────────
# feedback: "the boredom is in long held drones -- we need texture to evolve
# WITHIN a held drone." The per-transition texture axis only sets a STATIC
# richness; this animates it. A slow OU process breathes the voice richness up
# and down over time so held drones' voices fade in and out, centred on whatever
# richness the drone is currently at (i.e. your trained preference). Built on the
# frozen arranger: a held segment is rendered as several same-/interpolated-theta
# pads whose overtone gains follow the breath curve, blended by the arranger's
# existing layered crossfade -- no arranger edit, never touches rated stimuli.
BREATH_PAD_S = 6.0     # ~ target musical duration per breath sub-pad
BREATH_MAX_PADS = 6
# Demo breathing pushes overtones PAST the rated ladder's 0.5 cap so the swing
# is actually audible (2026-07-15, feedback: "the texture right now is too
# subtle"). Rated/policy texture (s09/s07e) stays capped at 0.5 -- this ceiling
# is demo-render character only, never touches a rated stimulus.
BREATH_RICH_CEILING = 0.85


def _breath_richness_theta(theta, r, ceiling=0.5):
    """Continuous voice-richness modulation around the pad's OWN gains. r in
    [-1,+1]: 0 = unchanged; +1 pushes fifth+octave toward `ceiling` (thicker);
    -1 fades the overtones then the third out toward a bare root/sub. Distinct
    from the discrete texture_overrides (which SETS fixed target gains for the
    rated/policy axis) -- this reads the pad's current gains so it breathes
    around them. `ceiling` defaults to the rated 0.5 (keeps existing callers /
    tests unchanged); demo breathing passes BREATH_RICH_CEILING for a wider,
    more audible swing."""
    if r == 0.0:
        return theta
    th = dict(theta)
    b_fifth = float(th.get("fifth_gain", 0.0))
    b_oct = float(th.get("octave_gain", 0.0))
    b_third = float(th.get("third_gain", 0.0))
    if r > 0.0:
        th["fifth_gain"] = b_fifth + r * max(0.0, ceiling - b_fifth)
        th["octave_gain"] = b_oct + r * max(0.0, ceiling - b_oct)
    else:
        s = 1.0 + r                       # 1 at r=0 -> 0 at r=-1
        th["fifth_gain"] = b_fifth * s
        th["octave_gain"] = b_oct * s
        if r < -0.5:                      # fade the third only in the bottom half
            th["third_gain"] = b_third * max(0.0, (r + 1.0) / 0.5)
    return th


def _breath_theta(theta, r, cents_offset):
    """One breath sub-pad's theta: voice-richness swing (r) PLUS a slow pitch
    detune (cents_offset, in cents) that floats the whole pad up/down like tape
    wow. Both are encoded straight into theta (overtone gains + f0_hz are theta
    fields) so the frozen arranger renders them with no edit. Demo character
    only -- rated stimuli never pass through here."""
    th = _breath_richness_theta(theta, r, BREATH_RICH_CEILING)
    if cents_offset != 0.0 and th.get("f0_hz") is not None:
        th = dict(th)                     # th may be the caller's theta if r==0
        th["f0_hz"] = float(th["f0_hz"]) * (2.0 ** (cents_offset / 1200.0))
    return th


def _lerp_theta(a, b, frac):
    """Linear blend of two theta dicts (scalars + harm_dist array). Held drones
    have prev~=next so this is ~constant; a moving segment gets a smooth tween."""
    out = {}
    for k in set(a) | set(b):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            va = np.asarray(va if va is not None else vb, dtype=float)
            vb = np.asarray(vb if vb is not None else va, dtype=float)
            out[k] = (1 - frac) * va + frac * vb
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            out[k] = (1 - frac) * va + frac * vb
        else:
            out[k] = vb if vb is not None else va
    return out


def breathing_waypoints(prev_theta, next_theta, sess):
    """Subdivide a held segment into K breath sub-pads (theta interpolated
    prev->next) whose voice richness AND pitch detune each follow their own slow
    OU process, advanced in sess so they evolve continuously across segments +
    the matching hold/xfade that keeps the segment's total duration the same.
    Two independent OUs (same period, separate draws) so richness and cents
    don't move in lockstep. Returns (waypoints, hold_s, xfade_s)."""
    depth = float(sess.get("tex_breath_depth", 0.0))
    cents_depth = float(sess.get("cents_breath_depth", 0.0))
    period = max(float(sess.get("tex_breath_period", 30.0)), 1.0)
    total = 2 * sess.get("hold_s", NORMAL_HOLD_S) + sess.get("xfade_s", NORMAL_XFADE_S)
    K = int(min(BREATH_MAX_PADS, max(2, round(total / BREATH_PAD_S))))
    step_s = total / (2 * K - 1)          # equal hold=xfade -> total preserved
    dt = total / K                        # musical time each sub-pad occupies
    decay = np.exp(-dt / period)
    sd = np.sqrt(1 - decay * decay)
    wps = []
    for i in range(K):
        frac = i / (K - 1) if K > 1 else 0.0
        b = sess.get("tex_breath", 0.0) * decay + sd * np.random.randn()
        sess["tex_breath"] = float(b)
        r = depth * float(np.tanh(b))     # bounded to +-depth, organic
        c = sess.get("cents_breath", 0.0) * decay + sd * np.random.randn()
        sess["cents_breath"] = float(c)
        cents_offset = cents_depth * float(np.tanh(c))   # bounded to +-cents_depth
        base = _lerp_theta(prev_theta, next_theta, frac)
        wps.append(_breath_theta(base, r, cents_offset))
    return wps, step_s, step_s


def reverb_breathe(audio, sess):
    """Breathe the reverb of a held segment: mix a slow OU envelope between the
    already-rendered (dry-ish, 0.35-wet) segment and a fully-wet version of it,
    so the space opens and closes over the hold. Additive only -- the arranger's
    baked-in reverb is the floor, this can add wetness but never remove it (you
    can't un-bake a convolution), so the segment is never drier than the frozen
    baseline. Demo character only; the OU state persists in sess so it evolves
    continuously across held segments. depth 0 -> unchanged."""
    depth = float(sess.get("reverb_breath_depth", 0.0))
    audio = np.asarray(audio, dtype=np.float64)
    if depth <= 0.0 or audio.size == 0:
        return audio
    from arranger import SR, convolve_reverb
    n = len(audio)
    wet = np.asarray(convolve_reverb(audio, wet=1.0), dtype=np.float64)[:n]
    if len(wet) < n:                       # convolve_reverb clips to len(audio)
        wet = np.pad(wet, (0, n - len(wet)))
    period = max(float(sess.get("tex_breath_period", 30.0)), 1.0)
    ctrl_hz = 20.0
    n_ctrl = max(2, int(n / SR * ctrl_hz))
    dt = 1.0 / ctrl_hz
    decay = np.exp(-dt / period)
    sd = np.sqrt(1 - decay * decay)
    x = float(sess.get("reverb_breath", 0.0))
    curve = np.empty(n_ctrl)
    for i in range(n_ctrl):
        x = x * decay + sd * np.random.randn()
        curve[i] = x
    sess["reverb_breath"] = x
    env = depth * 0.5 * (1.0 + np.tanh(curve))          # [0, depth]
    env_audio = np.interp(np.linspace(0.0, 1.0, n),
                          np.linspace(0.0, 1.0, n_ctrl), env)
    # no peak guard -- final_loudness() owns peak safety for the whole chain
    return (audio * (1.0 - env_audio) + wet * env_audio).astype(np.float32)


def _lookup_choice(pool, base):
    """Turn lookup_best_arc's (arc_id, f_mean, f_std) into the same
    ramp_s/chord_start/chord_end/is_glide/texture shape best_by_gp_predict
    returns, so PLAY_MODE="lookup" can drive rendering exactly like gp_predict
    does -- reads the winning POOL ARC's own transition mechanics (not its
    audio; the live waypoints are still the CURRENTLY retrieved prev/next
    theta)."""
    arc_id, f_mean, f_std = base
    m = pool[arc_id]
    ov_render = m.get("render_overrides") or {}
    return dict(ramp_s=float(m["ramp_s"]), chord_start=m["chord_start"],
               chord_end=m["chord_end"],
               is_glide=(ov_render.get("xfade_mode") == "glide"),
               texture=texture_of(m), mean=f_mean, std=f_std)


def decide_and_render(sess, engine, renderer, fit, pool, contexts, trigger):
    """The one function both the real app and _selftest call. Returns
    (audio, new_prev_theta, decision_info dict).

    PLAY_MODE (sess["policy_mode"], "gp_predict" or "lookup") selects which
    of the two arc-selection methods actually renders -- both are ALWAYS
    computed and logged regardless of mode, so the log stays a fair
    side-by-side comparison; only `chosen_*`/`chosen_method` reflect what
    was actually played. Added 2026-07-09 (a design review: querying
    the GP "like a discrete lookup table" leaves no way to hear the honest
    baseline live; previously `lookup_best_arc` was computed only to be
    logged, gp_predict's choice ALWAYS rendered)."""
    curr_va = sess["curr_va"]
    # liveliness Tier A (2026-07-12, all .get-defaulted so old sessions and
    # the selftest's hand-built dicts keep today's frozen behavior):
    # variety_k widens retrieve_diverse's VA-nearest candidate set (the
    # audibly-farthest of k wins -- k=1 degenerates to plain 1-NN, the
    # "same anchor for minutes" boredom mode); hold/xfade set the pace.
    next_theta = engine.retrieve_diverse(curr_va[0], curr_va[1],
                                         prev_theta=sess["prev_theta"],
                                         k=int(sess.get("variety_k", 8)))
    if sess["prev_theta"] is None:
        sess["prev_theta"] = next_theta

    # dark voicing: 0.0 when the key is absent (selftest dicts, old
    # sessions) -> _voiced is a no-op and the frozen stack renders as-is
    vm = (voicing_multipliers(curr_va[0], curr_va[1], sess["dark_voicing"])
          if sess.get("dark_voicing", 0.0) > 0.0 else None)

    def _vt(theta):
        # render-time noise duck; sess keeps the canonical unvoiced theta
        return (voicing_theta(theta, curr_va[0], curr_va[1],
                              sess["dark_voicing"]) if vm else theta)

    if not trigger:
        # held-segment character wander (Tier B, demo-only): when any wander
        # knob is up the held drone evolves WITHIN the hold (the brief 2026-07-15:
        # "the boredom is in long held drones ... they should wander whilst you
        # are in a hold segment"). Three independent slow OUs: voice richness +
        # pitch detune are subdivided into breath sub-pads (both encode into
        # theta, no arranger edit); reverb breathes as a post-render send. At
        # all depths 0 (default / selftest dicts / old sessions) this is exactly
        # the original 2-waypoint render.
        breathe = (sess.get("tex_breath_depth", 0.0) > 0.0
                   or sess.get("cents_breath_depth", 0.0) > 0.0)
        if breathe:
            wps, h, x = breathing_waypoints(sess["prev_theta"], next_theta, sess)
            wps = [_vt(w) for w in wps]
        else:
            wps = [_vt(sess["prev_theta"]), _vt(next_theta)]
            h = sess.get("hold_s", NORMAL_HOLD_S)
            x = sess.get("xfade_s", NORMAL_XFADE_S)
        with _voiced(vm) if vm else _nullctx():
            audio = renderer.render_waypoints(wps, hold_s=h, xfade_s=x, **RENDER_KW)
        audio = reverb_breathe(audio, sess)
        return audio, next_theta, dict(trigger="none")

    context = policy.nearest_context(contexts, curr_va[0], curr_va[1])
    fixed_dist, fixed_semis = audible_dist_and_semitones(
        engine, sess["prev_theta"], next_theta)

    base = policy.lookup_best_arc(pool, fit, context)
    gp = policy.best_by_gp_predict(fit, context, fixed_dist, fixed_semis)

    policy_mode = sess.get("policy_mode", "gp_predict")
    rerank_info = dict(prev_type="", cand_type="", coherence="",
                       rerank_reason="", rerank_arc_id="")
    if policy_mode == "lookup" and base is not None:
        # sequential-coherence re-ranker (board 11b): among the pool arcs at
        # this context the preference GP likes about equally (utility within
        # `coherence_tie_margin` of the best), prefer the corpus type that most
        # coherently FOLLOWS the last rendered arc's type via s13's 5x5 matrix.
        # It NEVER selects outside the near-tie band, so a preference judgement
        # is never overridden -- it only reorders arcs the GP is indifferent
        # between. Degrades to plain lookup_best_arc when the corpus typology is
        # missing (ARC_TYPES/COH_TRANS None) or on the first transition of a
        # session (prev_type None). This is the ONLY place ORDER enters the
        # otherwise-memoryless policy.
        cands = policy.lookup_candidates(pool, fit, context)
        prev_type = sess.get("prev_type")
        aid, rr = reranker.rerank([(a, u) for a, u, _ in cands], prev_type,
                                  ARC_TYPES, COH_TRANS,
                                  tie_margin=sess.get("coherence_tie_margin", 0.1))
        winner = next((c for c in cands if c[0] == aid), base)
        chosen, chosen_method = _lookup_choice(pool, winner), "lookup"
        # remember the rendered arc's type for the next transition's coherence
        sess["prev_type"] = (ARC_TYPES or {}).get(aid)
        rerank_info = dict(
            prev_type=("" if prev_type is None else prev_type),
            cand_type=("" if rr.get("cand_type") is None else rr.get("cand_type")),
            coherence=rr.get("coherence", ""),
            rerank_reason=rr.get("reason", ""),
            rerank_arc_id=aid)
    else:
        # No rated pool arc shares this exact snapped context (base is None)
        # -- falls back to gp_predict rather than skip the transition. The
        # re-ranker fields stay blank: gp_predict combos are synthesized, never
        # a typed pool arc.
        chosen = gp
        chosen_method = "gp_predict" if policy_mode == "gp_predict" else "lookup_fallback_gp"

    info = dict(trigger=("force" if sess["force"] else "val_shift"),
               context_v=context[0], context_a=context[1],
               fixed_dist=fixed_dist, fixed_glide_semitones=fixed_semis,
               policy_mode=policy_mode,
               lookup_arc_id=base[0] if base else "", lookup_f=base[1] if base else "",
               lookup_std=base[2] if base else "",
               gp_ramp_s=gp["ramp_s"], gp_chord_start=gp["chord_start"],
               gp_chord_end=gp["chord_end"], gp_is_glide=gp["is_glide"],
               gp_texture=gp["texture"], gp_f=gp["mean"], gp_std=gp["std"],
               chosen_method=chosen_method,
               chosen_ramp_s=chosen["ramp_s"], chosen_chord_start=chosen["chord_start"],
               chosen_chord_end=chosen["chord_end"], chosen_is_glide=chosen["is_glide"],
               chosen_texture=chosen.get("texture", 0.0),
               liveliness=liveliness_str(sess), **rerank_info)

    theta_start = apply_chord(sess["prev_theta"], chosen["chord_start"])
    theta_end = apply_chord(next_theta, chosen["chord_end"])
    # texture (board 9d): thin/thicken the END pad's voice stack -- applied
    # AFTER apply_chord so a "thin" choice zeroes the chord voices too, exactly
    # like s07e's rated texture arcs (canonical sess theta is never mutated:
    # theta_end is already a fresh apply_chord copy).
    theta_end.update(texture_overrides(chosen.get("texture", 0.0)))
    kw = dict(RENDER_KW)
    kw["xfade_mode"] = "glide" if chosen["is_glide"] else "layered"
    with _voiced(vm) if vm else _nullctx():
        audio = renderer.render_waypoints([_vt(theta_start), _vt(theta_end)],
                                          hold_s=HOLD_S,
                                          xfade_s=chosen["ramp_s"], **kw)
    return audio, theta_end, info


def _selftest():
    """Verifies the CONTROL FLOW (trigger detection, force override,
    target-setting, dual-policy logging) with FAKE engine/renderer -- no
    real bank/audio needed, mirrors the project's existing convention of
    smoke-testing logic against synthetic stand-ins (s08/s08b's smoke_pool).
    Final audio-quality verification of the real server is the brief's, per
    project ground rules (he runs the shell/Gradio process himself)."""

    class FakeEngine:
        def retrieve_diverse(self, v, a, prev_theta=None, k=8):
            return dict(f0_hz=110.0 + 10 * v, swell_rate=0.05, swell_depth=0.02,
                       noise_level=0.01, noise_cutoff_hz=4000.0,
                       third_interval=3.0, third_gain=0.2, fifth_gain=0.3,
                       octave_gain=0.1, harm_dist=np.ones(32, dtype=np.float32) / 32)

        def _audible_features(self, f0, centroid, sr, sd, nl):
            return np.array([np.log(f0), centroid, sr, sd, nl])

    class FakeRenderer:
        def render_waypoints(self, waypoints, hold_s, xfade_s, **kw):
            return np.zeros(100, dtype=np.float32)

    # The REAL arc pool and the REAL frozen posterior. The selftest used to
    # build a synthetic pool, which meant the control flow was only ever
    # exercised against arcs that could not occur. Both ship in weights/ and
    # neither needs audio or rating data, so there is no reason to fake them.
    pool = policy.load_pool()
    fit = policy.PreferenceGP.load()
    contexts = policy.pool_contexts(pool)

    engine, renderer = FakeEngine(), FakeRenderer()
    global CONDUCTOR_LOG, ANNOTATION_LOG, ARC_TYPES, COH_TRANS
    CONDUCTOR_LOG = HERE / "data" / "selftest_conductor_log.csv"
    if CONDUCTOR_LOG.exists():
        CONDUCTOR_LOG.unlink()
    ANNOTATION_LOG = HERE / "data" / "selftest_annotations.csv"
    if ANNOTATION_LOG.exists():
        ANNOTATION_LOG.unlink()

    sess = dict(curr_va=np.array([-0.5, -0.3]), target_va=np.array([-0.5, -0.3]),
               prev_theta=None, force=False, step=0)

    # 1. No valence shift, no force -> normal segment, no log row
    audio, theta, info = decide_and_render(sess, engine, renderer, fit, pool,
                                           contexts, trigger=False)
    assert info["trigger"] == "none"
    sess["prev_theta"] = theta

    # 1b. Held-segment wander ON (Tier B demo animation: richness + cents +
    # reverb) -> held segment still renders finite audio and all three OU
    # states evolve; then reset to the frozen default so it doesn't affect the
    # transition tests below.
    sess.update(tex_breath_depth=0.6, cents_breath_depth=15.0,
                reverb_breath_depth=0.5, tex_breath_period=30.0,
                tex_breath=0.0, cents_breath=0.0, reverb_breath=0.0)
    audio, theta, info = decide_and_render(sess, engine, renderer, fit, pool,
                                           contexts, trigger=False)
    assert info["trigger"] == "none" and np.all(np.isfinite(audio))
    assert sess["tex_breath"] != 0.0, "richness-breathing OU did not evolve"
    assert sess["cents_breath"] != 0.0, "cents-wander OU did not evolve"
    assert sess["reverb_breath"] != 0.0, "reverb-wander OU did not evolve"
    sess.update(tex_breath_depth=0.0, cents_breath_depth=0.0,
                reverb_breath_depth=0.0, prev_theta=theta)

    # 2. Target valence shifts UPWARD past threshold -> triggered, both
    # methods computed and logged, gp_predict renders (default policy_mode).
    sess["target_va"] = np.array([0.6, -0.3])
    val_shift_up = abs(sess["target_va"][0] - sess["curr_va"][0]) > RISE_THRESHOLD
    assert val_shift_up, "test setup should trigger an upward valence shift"
    audio, theta, info = decide_and_render(sess, engine, renderer, fit, pool,
                                           contexts, trigger=True)
    assert info["trigger"] in ("val_shift", "force")
    assert "gp_ramp_s" in info and 3.0 <= info["gp_ramp_s"] <= 16.0
    assert info["chosen_method"] == "gp_predict"
    # texture axis (board 9d): policy scores + logs a texture level, rendered
    # value is one of the 5-level ladder policy.TEXTURE_CHOICES (was {-1,0,+1}
    # before the 2026-07-14 magnitude batch made the feature continuous).
    assert "gp_texture" in info and info["gp_texture"] in policy.TEXTURE_CHOICES
    assert info["chosen_texture"] in policy.TEXTURE_CHOICES
    log_decision(dict(timestamp="t", session_id="selftest", step=sess["step"], **info))
    sess["prev_theta"] = theta

    # 2b. Target valence shifts DOWNWARD past threshold -> also triggered.
    # Gap this closes (a design review): the trigger condition was made
    # symmetric (abs(target-curr) > threshold) on 2026-07-09, but until now
    # this selftest only ever exercised the UPWARD direction (test 2 above),
    # so a regression back to a one-directional trigger would have passed
    # silently. curr_va starts high, target drops well below it.
    sess["curr_va"] = np.array([0.5, -0.3])
    sess["target_va"] = np.array([-0.6, -0.3])
    val_shift_down = abs(sess["target_va"][0] - sess["curr_va"][0]) > RISE_THRESHOLD
    assert val_shift_down, "test setup should trigger a downward valence shift"
    audio, theta, info = decide_and_render(sess, engine, renderer, fit, pool,
                                           contexts, trigger=True)
    assert info["trigger"] in ("val_shift", "force")
    assert "gp_ramp_s" in info and 3.0 <= info["gp_ramp_s"] <= 16.0
    assert info["chosen_method"] == "gp_predict"
    log_decision(dict(timestamp="t", session_id="selftest", step=sess["step"], **info))
    sess["prev_theta"] = theta

    # 3. Force button, independent of valence shift
    sess["force"] = True
    sess["target_va"] = sess["curr_va"].copy()   # no shift this time
    audio, theta, info = decide_and_render(sess, engine, renderer, fit, pool,
                                           contexts, trigger=True)
    log_decision(dict(timestamp="t", session_id="selftest", step=sess["step"], **info))
    sess["force"] = False

    # 4. policy_mode="lookup" -- verify the baseline method actually RENDERS
    # when selected, not just logged (a design review: gp_predict's
    # choice always rendered before this, lookup_best_arc was computed only
    # to be logged as a comparison point). chosen_* must mirror the winning
    # POOL ARC's own transition mechanics, not gp_predict's pick. With the
    # corpus typology absent (ARC_TYPES/COH_TRANS None) the re-ranker degrades
    # to plain best-utility, so lookup renders exactly lookup_best_arc's pick --
    # this is the pure-baseline / graceful-degradation case.
    ARC_TYPES, COH_TRANS = None, None
    sess["policy_mode"] = "lookup"
    sess["force"] = True
    audio, theta, info = decide_and_render(sess, engine, renderer, fit, pool,
                                           contexts, trigger=True)
    assert info["policy_mode"] == "lookup"
    assert info["chosen_method"] == "lookup", (
        "lookup mode should render lookup_best_arc's own choice when a "
        "rated arc exists at this context (smoke pool always has one)")
    arc_meta = pool[info["lookup_arc_id"]]
    assert info["chosen_ramp_s"] == arc_meta["ramp_s"]
    assert info["chosen_chord_start"] == arc_meta["chord_start"]
    assert info["chosen_chord_end"] == arc_meta["chord_end"]
    assert info["chosen_texture"] == texture_of(arc_meta), (
        "lookup mode should render the winning pool arc's own texture level")
    assert info["rerank_reason"] in ("best_utility", ""), (
        "no corpus typology -> re-ranker must fall back to best-utility")
    assert info["rerank_arc_id"] == info["lookup_arc_id"], (
        "degraded re-ranker must render the baseline best-utility arc")
    log_decision(dict(timestamp="t", session_id="selftest", step=sess["step"], **info))
    sess["force"] = False

    # 4b. Sequential-coherence re-ranker (board 11b): with the corpus typology
    # present, lookup mode routes ORDER through s17.rerank. The re-rank LOGIC is
    # proven in s17's own selftest; here we verify the WIRING -- prev_type is
    # threaded in and out, the re-ranked winner drives what renders, the log
    # carries the five coherence columns, and (load-bearing) the chosen arc
    # NEVER leaves the near-tie band (preference is not overridden).
    ctx = policy.nearest_context(contexts, sess["curr_va"][0], sess["curr_va"][1])
    cands = policy.lookup_candidates(pool, fit, ctx)
    assert cands, "smoke pool must have candidates at the snapped context"
    k = 5
    fake_types = {aid: (i % k) for i, (aid, _, _) in enumerate(cands)}
    for aid in pool:                       # make classify total over the pool
        fake_types.setdefault(aid, 0)
    ARC_TYPES = fake_types
    COH_TRANS = np.full((k, k), 1.0 / k)   # uniform: coherence can't break ties
    sess["prev_type"] = 2
    sess["coherence_tie_margin"] = 0.1
    sess["force"] = True
    audio, theta, info = decide_and_render(sess, engine, renderer, fit, pool,
                                           contexts, trigger=True)
    assert info["prev_type"] == 2, "prev_type must be threaded into the log"
    assert info["rerank_arc_id"] != "", "lookup mode must record the rendered arc"
    tie_margin = 0.1
    winner_u = next(u for a, u, _ in cands if a == info["rerank_arc_id"])
    assert winner_u >= cands[0][1] - tie_margin - 1e-9, (
        "re-ranker must never select outside the near-tie band")
    assert sess["prev_type"] == fake_types[info["rerank_arc_id"]], (
        "prev_type must advance to the rendered arc's type for the next hop")
    assert info["cand_type"] == fake_types[info["rerank_arc_id"]]
    log_decision(dict(timestamp="t", session_id="selftest", step=sess["step"], **info))
    sess["force"] = False
    ARC_TYPES, COH_TRANS = None, None      # restore for any later cases

    assert CONDUCTOR_LOG.exists()
    rows = list(csv.DictReader(open(CONDUCTOR_LOG)))
    assert len(rows) == 5, f"expected 5 logged decisions, got {len(rows)}"

    # 5. Annotation logging (2026-07-13): independent of trigger state --
    # a reaction can be logged on a normal (untriggered) step too.
    log_annotation(dict(timestamp="t", session_id="selftest", step=sess["step"],
                        context_v=sess["curr_va"][0], context_a=sess["curr_va"][1],
                        target_v=sess["target_va"][0], target_a=sess["target_va"][1],
                        policy_mode=sess.get("policy_mode", "gp_predict"),
                        liveliness=liveliness_str(sess),
                        reason_tag="\U0001F44E hate it", note="too static"))
    assert ANNOTATION_LOG.exists()
    arows = list(csv.DictReader(open(ANNOTATION_LOG)))
    assert len(arows) == 1, f"expected 1 logged annotation, got {len(arows)}"
    assert arows[0]["note"] == "too static"

    # 6. Soundscape bed (layer 5, board 11a): apply_bed mixes a bed under the
    # drone via the LiveBedScheduler, advances the absolute clock, and is a
    # strict no-op (clock aside) when disabled.
    sr = 16000
    fake_path = "__selftest_bed__"
    # prime the cache under the RESOLVED key, so apply_bed still reaches this
    # synthetic bed after _resolve_bed rewrites relative paths against the
    # package root -- and so the test exercises that resolution instead of
    # quietly sidestepping it (2026-08-01)
    _BED_AUDIO_CACHE[_resolve_bed(fake_path)] = (
        0.1 * np.sin(2 * np.pi * 220 * np.arange(sr * 3) / sr))
    drone = 0.2 * np.sin(2 * np.pi * 60 * np.arange(int(sr * 11)) / sr)
    # a single always-on 'constant' bed state guarantees the bed is audible;
    # the real overlay preset legitimately holds drone_solo for a long settled
    # dwell (15.5-121.2 s), so it would show no bed in an 11 s cold-start window
    # two bed-on states (zero diagonal required -- self-transitions are the
    # dwell's job); huge dwell => effectively a permanent bed-on schedule
    on_sched = sched.LiveBedScheduler(
        [sched.SchedState("bed_on", {"bed": "constant"}, (1e9, 1e9)),
         sched.SchedState("bed_on2", {"bed": "constant"}, (1e9, 1e9))],
        [[0.0, 1.0], [1.0, 0.0]])
    bsess = dict(curr_va=np.array([-0.6, -0.4]), bed_enabled=True,
                 bed_auto=True, bed_scheduler=on_sched,
                 bed=dict(path=fake_path, gain_db=-12.0, bed_type="test"),
                 bed_gain_db=-12.0, bed_duck_depth=0.7, bed_burst_s=3.0,
                 t_abs=0.0, env_ref=None)
    out_on = apply_bed(drone.copy(), bsess, sr)
    assert len(out_on) == len(drone) and np.all(np.isfinite(out_on))
    # peak safety is final_loudness()'s job now (case 6b), not this layer's
    assert abs(bsess["t_abs"] - len(drone) / sr) < 1e-6, "clock must advance"
    assert not np.allclose(out_on, drone), "enabled bed-on state must mix audio"
    # the real corpus_dwell scheduler builds + starts in drone_solo (bed off):
    # apply_bed must be finite there too, just silent
    real_sess = dict(bsess, bed_scheduler=build_bed_scheduler(), t_abs=0.0,
                     env_ref=None)
    out_real = apply_bed(drone.copy(), real_sess, sr)
    assert np.all(np.isfinite(out_real))
    # disabled bed: pass-through, clock still advances
    dsess = dict(bsess, bed_enabled=False, t_abs=0.0, env_ref=None)
    out_off = apply_bed(drone.copy(), dsess, sr)
    assert np.allclose(out_off, drone), "disabled bed must be pass-through"
    assert abs(dsess["t_abs"] - len(drone) / sr) < 1e-6

    # 6b. FINAL loudness pass (2026-08-01, the segment-jump fix). What has to
    # hold: the finished mix lands on the frozen target, the output is peak-safe
    # whatever the chain handed it, the arousal span still works, and -- the
    # point of the exercise -- consecutive segments MEET at the same gain
    # instead of stepping, however far apart their arousal or headroom is.
    quiet = drone.astype(np.float64) * 0.1
    l_q = final_loudness(quiet, 0.0, dict(level_loudness=True,
                                          arousal_gain_db=0.0), sr)
    assert np.all(np.isfinite(l_q)) and np.max(np.abs(l_q)) <= LOUD_CEILING + 1e-6
    assert abs(aw_rms(l_q, sr) - AW_TARGET) / AW_TARGET < 0.05, \
        "finished mix must land on the frozen loudness target"
    # arousal span: settled (same arousal twice) the two ends are span*2 apart.
    # A 1 kHz tone, because A-weighting leaves it alone -- so the span is not
    # competing with the headroom clamp, which is tested separately below.
    tone = 0.2 * np.sin(2 * np.pi * 1000 * np.arange(int(sr * 11)) / sr)
    la = dict(level_loudness=True, arousal_gain_db=6.0)
    hs, ls = dict(la), dict(la)
    final_loudness(tone, 1.0, hs, sr)
    final_loudness(tone, -1.0, ls, sr)
    hi = final_loudness(tone, 1.0, hs, sr)       # 2nd call: ramp has settled
    lo = final_loudness(tone, -1.0, ls, sr)
    span_db = 20.0 * np.log10(aw_rms(hi, sr) / (aw_rms(lo, sr) + 1e-12))
    assert abs(span_db - 12.0) < 0.5, f"arousal span must survive ({span_db:.2f} dB)"
    # JOIN CONTINUITY: run two segments whose arousal is at opposite extremes
    # (the old code's worst case -- a 2*span dB step landing on the seam) and
    # compare the tail of one against the head of the next. A 30 Hz tone is
    # thrown in as the middle segment because A-weighting discounts it ~39 dB,
    # so levelling it drives the raw peak past full scale: the dark-drone case
    # the old conditional guard used to pull down on its own.
    hot = 0.9 * np.sin(2 * np.pi * 30 * np.arange(int(sr * 11)) / sr)
    jsess = dict(level_loudness=True, arousal_gain_db=6.0)
    w = int(1.0 * sr)
    segs = [final_loudness(s, a, jsess, sr) for s, a in
            ((tone, -1.0), (hot, 1.0), (tone, -1.0), (tone, 1.0))]
    for k in range(len(segs) - 1):
        step = 20.0 * np.log10((aw_rms(segs[k + 1][:w], sr) + 1e-12)
                               / (aw_rms(segs[k][-w:], sr) + 1e-12))
        # 1.5 dB, not tighter: the 30 Hz segment is pinned AT the ceiling, so
        # it physically cannot be raised to meet the one before it (measured
        # residual -1.3 dB). Real pool renders, which are nowhere near the
        # ceiling, come in under 0.7 dB at every join.
        assert abs(step) < 1.5, f"join {k} steps {step:+.2f} dB -- must be continuous"
    assert all(np.max(np.abs(s)) <= LOUD_CEILING + 1e-6 for s in segs), "peak-safe"
    # soft limiter: leaves quiet material untouched, never exceeds the ceiling
    assert np.allclose(_soft_limit(quiet), quiet)
    assert np.max(np.abs(_soft_limit(drone.astype(np.float64) * 40.0))) <= LOUD_CEILING

    # 7. Vinyl crackle overlay (SideProjects): enabled mixes crackle + stays
    # peak-safe; disabled is exact pass-through. Skipped only if the module is
    # unavailable (import guarded), so the conductor still runs without it.
    if crackle is not None:
        con = apply_crackle(drone.copy(), dict(crackle_enabled=True,
                            crackle_gain_db=-18.0, crackle_density=12.0, step=1), sr)
        assert con.shape == drone.shape and np.all(np.isfinite(con))
        assert np.max(np.abs(con)) <= 0.95 + 1e-6
        assert not np.allclose(con, drone.astype(np.float32)), "enabled crackle must mix"
        coff = apply_crackle(drone.copy(), dict(crackle_enabled=False), sr)
        assert np.allclose(coff, drone.astype(np.float32)), "disabled crackle must pass through"

    # 8. Reverb space (SideProjects/reverb): off is an exact pass-through; a
    # chosen space changes the audio, keeps the segment LENGTH (streaming, not
    # clip-wise), stays peak-safe, and carries its tail into the next segment.
    if revbank is not None:
        ids = [c for c in revbank.condition_ids() if c != "dry"]
        roff = apply_reverb(drone.copy(), dict(reverb_cond="off"), sr)
        assert np.allclose(roff, drone.astype(np.float32)), "reverb off must pass through"
        rs = dict(reverb_cond=ids[-1], reverb_wet=0.5)
        r1 = apply_reverb(drone.copy(), rs, sr)
        assert r1.shape == drone.shape, "streaming reverb must preserve segment length"
        assert np.all(np.isfinite(r1))
        assert not np.allclose(r1, drone.astype(np.float32)), "a chosen space must change it"
        assert rs.get("reverb_tail") is not None and len(rs["reverb_tail"]), \
            "tail must be carried for the next segment"
        r2 = apply_reverb(np.zeros_like(drone), rs, sr)
        assert np.max(np.abs(r2)) > 0, "carried tail must ring into the next segment"
        # switching OFF must RING OUT, not chop: the first segment after the
        # switch still carries the decaying tail, and only once that has drained
        # is the output an exact pass-through (2026-08-01)
        # ...on a FRESH tail: the zeros call above deliberately drains this
        # session, so switching off there would have nothing left to decay and
        # would prove nothing
        rs2 = dict(reverb_cond=ids[-1], reverb_wet=0.5)
        apply_reverb(drone.copy(), rs2, sr)       # excite the room
        assert rs2.get("reverb_tail") is not None and len(rs2["reverb_tail"])
        rs2["reverb_cond"] = "off"
        ring = apply_reverb(drone.copy(), rs2, sr)
        assert not np.allclose(ring, drone.astype(np.float32)), \
            "reverb OFF must let the previous room decay, not cut it dead"
        for _ in range(40):                       # drain it
            _t = rs2.get("reverb_tail")
            if _t is None or not len(_t):
                break
            apply_reverb(np.zeros_like(drone), rs2, sr)
        assert np.allclose(apply_reverb(drone.copy(), rs2, sr),
                           drone.astype(np.float32)), \
            "once drained, reverb OFF is an exact pass-through"

    # 9. Melodic line: OFF is an exact pass-through (so every pre-melody session
    # renders identically); the style AUTO rule follows arousal; a bad style or
    # a missing module degrades to pass-through rather than taking the session
    # down mid-render.
    moff = apply_melody(drone.copy(), dict(melody_enabled=False), sr, {"f0_hz": 110.0})
    assert np.allclose(moff, drone.astype(np.float32)), "melody off must pass through"
    assert melody_style_for(dict(melody_style="auto", curr_va=(0.0, -0.9))) == "pentatonic_slow", \
        "calm should pick the sparse style"
    assert melody_style_for(dict(melody_style="auto", curr_va=(0.0, 0.5))) == "pentatonic_fast", \
        "livelier should pick the repeating-figure style"
    assert melody_style_for(dict(melody_style="folk", curr_va=(0.0, -0.9))) == "folk", \
        "an explicit style must override auto"
    mbad = apply_melody(drone.copy(), dict(melody_enabled=True, melody_style="nope"),
                        sr, {"f0_hz": 110.0})
    assert mbad.shape == drone.shape and np.all(np.isfinite(mbad)), \
        "a melody failure must degrade to pass-through, not crash the session"
    if meldrone is not None:
        msess = dict(melody_enabled=True, melody_style="pentatonic_fast", melody_level_db=-14.0,
                     curr_va=(0.3, 0.2), step=1)
        assert "mel=pentatonic_fast/swell@-14dB" in liveliness_str(msess), \
            "melody settings must be logged so ratings capture what played"
        assert "mel=pentatonic_fast/legato@-14dB" in liveliness_str(
            dict(msess, melody_envelope="legato")), \
            "the ENVELOPE must be logged too -- four are selectable and they " \
            "are not interchangeable"
        # every envelope must actually render. legato is the one that also
        # changes note lengths, so a signature drift there would be silent.
        # ASSERT IT CHANGED THE AUDIO. "finite and same length" is also exactly
        # what the pass-through failure path returns, so the old form of this
        # check passed for two weeks while apply_melody raised int(None) on
        # every call and silently disabled itself (2026-08-01). The session
        # dicts below are built like new_session()'s -- melody_anchor=None
        # included -- so the .get-default trap is inside the test, not beside it.
        for _env in ("swell", "legato", "struck", "flat"):
            _out = apply_melody(drone.copy(),
                                dict(melody_enabled=True, melody_style="pentatonic_fast",
                                     melody_level_db=-14.0, melody_envelope=_env,
                                     melody_anchor=None, melody_tick_s=1.0,
                                     curr_va=(0.3, 0.2), step=1),
                                sr, {"f0_hz": 110.0})
            assert _out.shape == drone.shape and np.all(np.isfinite(_out)), \
                f"melody envelope {_env} must render a finite, same-length segment"
            assert not np.allclose(_out, drone.astype(np.float32)), \
                (f"melody envelope {_env} did NOT change the audio -- it fell "
                 f"through apply_melody's blanket except and disabled itself")
        assert "mel=off" in liveliness_str(dict(melody_enabled=False)), \
            "melody-off must be logged as off"
        assert "mel=tf" in liveliness_str(dict(melody_enabled=True,
                                               melody_engine="transformer",
                                               melody_level_db=-14.0,
                                               curr_va=(0.3, 0.2), step=1)), \
            "the transformer engine must be identifiable in the log, not read as a style"
        # transformer_notes' rest recovery depends ENTIRELY on rest_dur being
        # distinguishable from note_dur. Assert that across the whole arousal
        # range rather than trusting the algebra: if these ever collide, rests
        # silently become held notes and the line stops being phrased.
        for _a in (-1.0, -0.5, 0.0, 0.5, 1.0):
            _nd = max(1.0, 3.5 - 1.5 * _a)
            _rd = max(1.5, 5.0 - 2.5 * _a)
            assert abs(_nd - _rd) > 1e-3, \
                f"note/rest durations collide at arousal {_a} -- rests unrecoverable"

    print(f"SELFTEST OK: normal segment produced no log row; upward- and "
          f"downward-val_shift, force-triggered, lookup-policy-mode, and "
          f"coherence-re-ranked (board 11b) transitions all logged both policy "
          f"methods correctly, the re-ranker threading prev_type through and "
          f"never leaving the near-tie band "
          f"({len(rows)} rows in {CONDUCTOR_LOG.name}); reaction annotation "
          f"logged independently of trigger state ({len(arows)} row in "
          f"{ANNOTATION_LOG.name}); bed layer mixes + advances the clock when "
          f"on and is pass-through when off; final loudness lands the finished "
          f"mix on the frozen target, keeps the arousal span, stays under the "
          f"{LOUD_CEILING} ceiling and meets every join continuously; melodic "
          f"line passes through when off, picks its style from arousal on auto, "
          f"degrades to pass-through on error, and logs what played.")


# Shown at the top of the UI, so it is on screen from the moment the page
# loads -- before anyone presses Start, and while the session is still warming.
# The package redistributes third-party AUDIO (soundscape beds, reverb IRs), so
# attribution has to travel with it rather than sit in a file nobody opens.
# ATTRIBUTION.md in the package root carries the same text at length.
# CORRECTED 2026-08-02 (feedback: "the only third party audio are the
# soundscapes; everything else is learned"). He is right that the drone itself
# is synthesised from fitted models and contains no third-party recording --
# the old blanket "this demo plays third-party audio" overstated it. Two
# things, not one, are third-party AUDIO in this package though: the beds, and
# the three EchoThief impulse responses, which are recordings of real rooms
# convolved with the drone (verified: CCRMAStairwell / FatMansMisery /
# PepperCanyonHall ship as wavs). Everything below them is a fit.
def build_id():
    """Which build is this? Shown in the UI so the question is answerable at a
    glance rather than by comparing files.

    Read from the package's MANIFEST.json, which the packer stamps with a
    timestamp plus a digest over every shipped file's sha256 -- so it cannot
    drift from what is actually running, the way a hand-bumped constant does.
    Running straight from the repo there is no manifest, so fall back to a
    digest of the UI sources themselves and SAY it is a dev tree: a stamp that
    silently looked like a release would be worse than none.
    """
    mf = HERE.parent.parent / "MANIFEST.json"
    try:
        return "build " + json.loads(mf.read_text())["build_id"]
    except Exception:
        pass
    try:
        h = hashlib.sha256()
        for f in sorted(HERE.glob("s0[12]*.py")) + sorted(HERE.glob("s2*.py")):
            h.update(f.read_bytes())
        return f"dev tree - {h.hexdigest()[:7]}"
    except Exception:
        return "dev tree"


def build_chip():
    """The build stamp as a FIXED corner chip.

    It first went at the end of the credits banner and was effectively
    invisible -- the brief looked for it twice and could not find it. Fixed
    bottom-LEFT: never scrolls away, and opposite the WebAudio badge
    bottom-right so the two never collide. Monospace + tabular figures so two
    builds line up when you compare them."""
    return (
        "<div id='conductor-build' style='position:fixed;left:10px;bottom:10px;"
        "z-index:9999;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;"
        "font-variant-numeric:tabular-nums;padding:5px 9px;border-radius:5px;"
        "background:#15181d;color:#8b919b;border:1px solid #2b2f37;"
        "pointer-events:none;opacity:.92'>" + build_id() + "</div>")


ATTRIBUTION_AUDIO = [
    ("Soundscape beds",
     "Emo-Soundscapes and ESC-50, both assembled from Freesound recordings by "
     "their original contributors. Redistributed here under the terms those "
     "collections carry; per-clip credits are in the bed bank."),
    ("Reverb impulse responses",
     "EchoThief Impulse Response Library (Christopher Warren). Measured room "
     "responses, convolved with the drone."),
]
ATTRIBUTION_FITTED = [
    ("Timbre prior",
     "NSynth (Magenta). Used to fit the synthesis prior; no NSynth audio "
     "ships in this package."),
    ("Melodic note model",
     "A small transformer trained on MIDI corpora. Symbolic training data "
     "only -- no audio, and none of it is reproduced."),
    ("Arousal labels",
     "MERT embeddings with a ridge head trained on DEAM. Labels only -- no "
     "model weights ship here."),
    ("Transition and reverb-tone priors",
     "Measured from commercially released ambient recordings obtained via "
     "archive.org. MEASUREMENTS ONLY -- durations, spectral deltas and decay "
     "fits. No recording, and no audio derived from one, is included, "
     "reproduced or redistributed; nothing here is a licence claim over "
     "those recordings."),
]
ATTRIBUTION_SOURCES = ATTRIBUTION_AUDIO + ATTRIBUTION_FITTED
ATTRIBUTION_BANNER = (
    "<div style='background:#eef2ff;border:1px solid #c7d2fe;border-radius:8px;"
    "padding:9px 12px;margin-bottom:10px;font-size:.86em;color:#1e1b4b'>"
    "<b>Credits.</b> The soundscape you hear is SYNTHESISED. The only "
    "third-party audio in this package is the soundscape beds and the reverb "
    "impulse responses: "
    + " &nbsp;·&nbsp; ".join(f"<b>{k}:</b> {v.split('.')[0]}."
                             for k, v in ATTRIBUTION_AUDIO)
    + " &nbsp;Everything else is fitted from data, with no audio reproduced: "
    + ", ".join(k.lower() for k, _ in ATTRIBUTION_FITTED)
    + ". &nbsp;See ATTRIBUTION.md.</div>")


class GuardBySpace:
    """One boundary guard per valence label space, LOADED not fitted.

    The guard is fitted on the rated clips' COORDINATES, so it is
    space-specific: a fence fitted in judge coordinates keeps the walk out of
    exactly the region the human axis exists to reach. Measured -- asked for
    v=+0.4, the judge-fitted fence settles the walk at -0.31 and the
    human-fitted one at +0.30. Before the pair existed, the dropdown swapped
    the retrieval bank and left the fence behind, so choosing "human labels"
    changed which anchors were searched while still forbidding the half of
    the plane they live in.

    **Both are frozen weights** (decision 2). Fitting either one needs the
    rating data, and the conductor ships none; it also cost about 2.3 s of
    startup. Loading a JSON costs nothing, so both spaces are available
    immediately and neither can silently differ from the one the report
    measured. `tau` is carried in the file: 0.5898 judge, 0.5505 human.

    `get` returns the space ACTUALLY applied, so a failure is reported to the
    listener rather than silently degrading -- every optional layer in this
    app degrades quietly by design, which is how the melody layer once
    shipped dead for two weeks.
    """

    PATHS = {"judge": WEIGHTS / "boundary_guard.json",
             "human": WEIGHTS / "boundary_guard_human.json"}

    def __init__(self, **kw):
        self._kw = kw
        self._g = {"judge": BoundaryGuard.load(self.PATHS["judge"])}

    def get(self, space):
        want = "human" if str(space).lower().startswith("human") else "judge"
        if want not in self._g:
            try:
                self._g[want] = BoundaryGuard.load(self.PATHS[want])
                print(f"[guard] loaded {want} coordinates, "
                      f"tau={self._g[want].tau:.4f}")
            except Exception as e:                    # noqa: BLE001
                print(f"[guard] {want}-space guard FAILED to load ({e}); "
                      f"falling back to the judge fence")
                self._g[want] = None
        g = self._g.get(want)
        return (g, want) if g is not None else (self._g["judge"], "judge")


def build_demo(engine, renderer, guards, fit, pool, contexts):
    import gradio as gr

    bank = load_bed_bank()
    bed_by_label = {f"{b['dataset']}:{b['bed_type']}/{b['bed_file']}": b
                    for b in bank if b.get("resolved")}
    bed_labels = list(bed_by_label)

    def new_session(label_space="judge", layers=None):
        # `layers` = the layer controls as they READ ON SCREEN when START was
        # pressed, so a layer configured before the session existed is
        # honoured. See adopt_layers for why. None (the selftest, and any
        # caller that does not care) keeps the hardcoded defaults below.
        #
        # Applied HERE and nowhere else: swapping mid-walk would change the
        # instrument under the listener. At session start the walk is being
        # built anyway, so there is nothing to be inconsistent with.
        applied = engine.set_label_space(str(label_space).split()[0].lower()) \
            if hasattr(engine, "set_label_space") else "judge"
        # The FENCE has to follow the bank. It is fitted on the rated clips'
        # coordinates, so a judge-space guard over a human-space bank pulls
        # the walk back from precisely the territory the swap opened up
        # (FINAL_CONDUCTOR/README.md SS32). Keyed off `applied`, not off what
        # was requested, so a bank without valence_human cannot end up with a
        # human fence over a judge bank -- the same bug mirrored.
        _g, guard_space = guards.get(applied)
        default_bed = bedbank.select_bed(bank, va=START_VA) if bank else None
        sess = dict(curr_va=np.array(START_VA, dtype=float),
                   target_va=np.array(START_VA, dtype=float),
                   prev_theta=None, force=False, session_id=uuid.uuid4().hex[:8],
                   step=0, policy_mode="gp_predict",
                   # sequential-coherence re-ranker (board 11b): prev_type is
                   # the last rendered arc's corpus type (None until the first
                   # lookup-mode transition). tie_margin = the utility near-tie
                   # band the coherence chain is allowed to reorder within.
                   prev_type=None, coherence_tie_margin=0.1,
                   # liveliness Tier A defaults == the pre-2026-07-12 frozen
                   # constants: a fresh session sounds exactly like before
                   # until the sliders move (the brief tunes by ear -- "I can
                   # be the prior")
                   trigger_threshold=RISE_THRESHOLD, walk_energy=1.0,
                   wander_std=0.0, variety_k=8,
                   hold_s=NORMAL_HOLD_S, xfade_s=NORMAL_XFADE_S,
                   level_loudness=True, arousal_gain_db=5.0, dark_voicing=0.7,
                   # held-segment character wander (Tier B): all depths 0 = off
                   # = frozen behavior. richness + cents subdivide the hold into
                   # breath sub-pads; reverb breathes as a post-render send. One
                   # shared period, independent OU states.
                   tex_breath_depth=0.0, tex_breath_period=30.0, tex_breath=0.0,
                   cents_breath_depth=0.0, cents_breath=0.0,
                   reverb_breath_depth=0.0, reverb_breath=0.0,
                   # soundscape bed (layer 5, board 11a): default OFF so a fresh
                   # session is unchanged. Scheduler + bank built once here.
                   # near/far placement: OFF, so a fresh session is mono and
                   # unchanged. `distance` is only read when enabled.
                   distance_enabled=False, distance=0.35,
                   distance_drift=0.0, dist_pan_phase=0.0,
                   bed_enabled=False, bed_auto=True,
                   bed_scheduler=build_bed_scheduler(), bed_bank=bank,
                   bed=default_bed,
                   bed_gain_db=(float(default_bed["gain_db"])
                                if default_bed and default_bed.get("gain_db")
                                else -12.0),
                   bed_duck_depth=0.7, bed_burst_s=3.0, t_abs=0.0, env_ref=None,
                   # bed variety (2026-07-25): rotate among the k VA-nearest
                   # beds at each transition (current bed excluded in advance())
                   bed_variety_k=3, bed_rng=np.random.default_rng(),
                   # vinyl crackle overlay (SideProjects): default OFF; mixed on
                   # top of the finished segment, so a fresh session is unchanged.
                   reverb_cond="off", reverb_wet=0.35, reverb_tail=None,
                   crackle_enabled=False, crackle_gain_db=-18.0,
                   crackle_density=1.0, crackle_intensity=0.5, crackle_hiss=0.15,
                   # melodic line (SideProjects item 4): default OFF, generated
                   # live per segment against that segment's theta and VA.
                   melody_enabled=False, melody_engine="markov", melody_ckpt="cpu",
                   melody_style="auto", melody_envelope="swell",
                   melody_level_db=-14.0, melody_anchor=None, melody_tick_s=1.0)
        sess["label_space"] = applied
        sess["guard_space"] = guard_space
        adopted = adopt_layers(sess, layers)
        return sess, (f"Session started on the **{applied}** valence axis. "
                      f"Set a target and wait for the first segment."
                      + (f"  Adopted from the panel: {', '.join(adopted)}."
                         if adopted else "")
                      + ("" if applied == str(label_space).split()[0].lower()
                         else "  (human labels unavailable in this bank)")
                      + ("" if guard_space == applied
                         else f"  (boundary guard fell back to "
                              f"{guard_space} coordinates)"))

    def set_liveliness(sess, thresh, energy, wander, k, hold, xfade,
                       level, again_db, voicing, breath_depth, breath_period,
                       cents_depth, reverb_depth):
        if not sess:
            return sess, "Press Start session first."
        invalidate_prerender(sess)   # queued segment is now stale
        sess.update(trigger_threshold=float(thresh), walk_energy=float(energy),
                    wander_std=float(wander), variety_k=int(k),
                    hold_s=float(hold), xfade_s=float(xfade),
                    level_loudness=bool(level), arousal_gain_db=float(again_db),
                    dark_voicing=float(voicing),
                    tex_breath_depth=float(breath_depth),
                    tex_breath_period=float(breath_period),
                    cents_breath_depth=float(cents_depth),
                    reverb_breath_depth=float(reverb_depth))
        return sess, (f"Settings applied: sensitivity {thresh:.2f}, energy ×{energy:.1f}, "
                     f"drift {wander:.2f}, variety {int(k)}, hold {hold:.0f}s, "
                     f"crossfade {xfade:.0f}s, levelling {'on' if level else 'off'}, "
                     f"arousal range ±{again_db:.0f}dB, dark voicing {voicing:.1f}, "
                     f"movement richness {breath_depth:.1f} / pitch {cents_depth:.0f} / "
                     f"reverb {reverb_depth:.1f} @{breath_period:.0f}s — from the "
                     f"next segment.")

    def log_reaction(sess, tag, feels, note):
        if not sess:
            return sess, "Press Start session first.", note
        if not tag and not feels and not note.strip():
            return sess, "Pick a verdict or type a comment before logging.", note
        v, a = sess["curr_va"]
        tv, ta = sess["target_va"]
        log_annotation(dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                            session_id=sess["session_id"], step=sess["step"],
                            context_v=v, context_a=a, target_v=tv, target_a=ta,
                            policy_mode=sess.get("policy_mode", "gp_predict"),
                            liveliness=liveliness_str(sess),
                            reason_tag="",
                            liked=tag or "",
                            feels_like_sound=feels or "",
                            note=note.strip()))
        parts = []
        if tag:
            parts.append(f"like: {tag}")
        if feels:
            parts.append(f"matches: {feels}")
        shown = " / ".join(parts) if parts else "(comment only)"
        if note.strip():
            shown += f" -- \"{note.strip()}\""
        return sess, f"Logged: {shown}", ""

    def set_policy_mode(sess, mode_label):
        if not sess:
            return sess, "Press Start session first."
        invalidate_prerender(sess)   # queued segment is now stale
        sess["policy_mode"] = "lookup" if mode_label.startswith("Baseline") else "gp_predict"
        return sess, (f"Transition policy set to **{mode_label}** — applies to the "
                     f"next transition.")

    def set_bed(sess, enable, auto, label, gain):
        if not sess:
            return sess, "Press Start session first."
        invalidate_prerender(sess)   # queued segment is now stale
        sess["bed_enabled"] = bool(enable)
        sess["bed_auto"] = bool(auto)
        sess["bed_gain_db"] = float(gain)
        if label and label in bed_by_label:
            sess["bed"] = bed_by_label[label]
        name = sess["bed"]["bed_type"] if sess.get("bed") else "none"
        msg = (f"Bed {'on' if enable else 'off'} — {name} @{gain:.0f}dB, "
               f"auto-select {'on' if auto else 'off'} — from the next segment.")
        if not bank:
            msg += " (No beds available.)"
        return sess, msg

    def set_distance(sess, enable, amount, drift):
        """Near/far placement. OFF by default; see apply_distance's note for
        why this runs after final_loudness and why the axis is distance rather
        than the width the strip used to advertise."""
        if not sess:
            return sess, "Press Start session first."
        invalidate_prerender(sess)   # queued segment is now stale
        sess["distance_enabled"] = bool(enable)
        sess["distance"] = float(amount)
        sess["distance_drift"] = float(drift)
        if enable and stereodist is None:
            return sess, ("Distance unavailable: SideProjects/stereo/distance.py "
                          "did not import. Staying mono.")
        if not enable:
            return sess, "Distance off — mono, unchanged — from the next segment."
        where = ("close" if amount < 0.25 else
                 "mid" if amount < 0.6 else "far")
        extra = " (mono device: left channel only, no drift)" if _RING is not None else ""
        dr = f", drift {drift:.2f}" if drift > 0 else ""
        return sess, (f"Distance on — {where} ({amount:.2f}){dr}{extra} — "
                      f"from the next segment.")

    def set_crackle(sess, enable, gain, density):
        if not sess:
            return sess, "Press Start session first."
        invalidate_prerender(sess)   # queued segment is now stale
        sess["crackle_enabled"] = bool(enable)
        sess["crackle_gain_db"] = float(gain)
        sess["crackle_density"] = float(density)
        msg = (f"Crackle {'on' if enable else 'off'} — {density:.0f} clicks/s "
               f"@{gain:.0f}dB — from the next segment.")
        if crackle is None:
            msg += " (crackle module unavailable.)"
        return sess, msg

    def set_melody(sess, enabled, engine, ckpt, style, level_db,
                   envelope="swell"):
        if not sess:
            return sess, "Press Start session first."
        invalidate_prerender(sess)   # queued segment is now stale
        sess["melody_enabled"] = bool(enabled)
        sess["melody_engine"] = str(engine)
        sess["melody_ckpt"] = str(ckpt)
        sess["melody_style"] = str(style)
        sess["melody_level_db"] = float(level_db)
        sess["melody_envelope"] = str(envelope)
        if meldrone is None:
            return sess, "Melody unavailable (engine/melody_markov.py not importable)."
        if not enabled:
            return sess, "Melodic line off."
        v, a = sess.get("curr_va", (0.0, 0.0))
        if str(engine) == "transformer":
            # style is deliberately not reported: it does not reach this path
            return sess, (f"Melodic line: transformer [{ckpt}] "
                          f"— style ignored, pace from arousal {a:+.2f}, "
                          f"{level_db:.0f}dB under the drone — from the next segment.")
        st = melody_style_for(sess)
        p = meldrone.melody_params(v, a, style=st)
        return sess, (f"Melodic line: {st}"
                      + (f" (auto from arousal {a:+.2f})" if style == "auto" else "")
                      + f" — {p['scale']}, {p['note_s']:.1f}s notes, "
                      f"{level_db:.0f}dB under the drone — from the next segment.")

    def set_reverb(sess, cond, wet):
        if not sess:
            return sess, "Press Start session first."
        invalidate_prerender(sess)   # queued segment is now stale
        # the old tail is DELIBERATELY kept when the space changes: the
        # previous room decays while the new one starts, which is a crossfade.
        # Dropping it here was the other half of the audible chop.
        sess["reverb_cond"] = str(cond)
        sess["reverb_wet"] = float(wet)
        if cond == "off" or revbank is None:
            return sess, ("Reverb off." if revbank is not None
                          else "Reverb unavailable (SideProjects/reverb not importable).")
        c = revbank.conditions().get(cond, {})
        return sess, (f"Reverb: {cond} — {c.get('tail_s', 0):.2f}s tail, "
                      f"{wet:.0%} wet — from the next segment.")

    def adopt_layers(sess, vals):
        """Seed a fresh session from the layer controls as they read on screen.

        WHY THIS EXISTS (2026-08-04): "I normally turn on Melody and
        Reverb first, and APPLY, then I set VA, then I click START ... from the
        first clip generated, I know that neither of them are actually
        applied."  He was right, and nothing about that order was wrong.

        Every setter refuses before a session exists --
        `if not sess: return sess, "Press Start session first."` -- and
        new_session then built its dict from hardcoded defaults
        (melody_enabled=False, reverb_cond="off", ...). So configuring the
        layers first wrote NOWHERE, and START overwrote the lot. The first
        segment came out bare by construction.

        Adopting the visible state is the honest fix: what is on screen is what
        the session starts with. The alternative -- gating the strips until
        START -- makes the app truthful too, but forces you to sit through one
        22-42s default segment before anything you asked for lands.

        Reuses the SETTERS rather than assigning session keys directly, so
        every coercion, every clamp and the reverb tail rule stay defined in
        exactly one place. A second copy would drift, silently, the way the
        packaged skin drifted from the source one.

        Returns the names of the layers that came up ON, for the status line.
        """
        if not vals:
            return []
        on = []
        if "melody_on" in vals:
            sess, _ = set_melody(sess, vals["melody_on"], vals["melody_engine"],
                                 vals["melody_ckpt"], vals["melody_style"],
                                 vals["melody_level"], vals["melody_env"])
            if sess.get("melody_enabled"):
                on.append(f"melody ({sess.get('melody_engine')})")
        if "reverb_pick" in vals:
            # The deck splits the room from the switch (README SS27.3): the
            # dropdown says WHICH, the strip button says WHETHER. Fold them the
            # same way set_reverb_gated does. Classic has no reverb_enable and
            # keeps "off" in the dropdown, so the .get default is True.
            cond = vals["reverb_pick"] if vals.get("reverb_enable", True) else "off"
            sess, _ = set_reverb(sess, cond, vals["reverb_wet"])
            if str(sess.get("reverb_cond", "off")) != "off":
                on.append(f"reverb ({sess.get('reverb_cond')})")
        if "bed_enable" in vals:
            sess, _ = set_bed(sess, vals["bed_enable"], vals["bed_auto"],
                              vals["bed_choice"], vals["bed_gain"])
            if sess.get("bed_enabled"):
                on.append("soundscape")
        if "crackle_enable" in vals:
            sess, _ = set_crackle(sess, vals["crackle_enable"],
                                  vals["crackle_gain"], vals["crackle_density"])
            if sess.get("crackle_enabled"):
                on.append("crackle")
        return on

    def set_target(sess, v, a):
        if not sess:
            return sess, "Press Start session first."
        sess["target_va"] = np.array([float(v), float(a)])
        invalidate_prerender(sess)   # the queued segment used the OLD target
        return sess, (f"Target set to valence {v:.2f}, arousal {a:.2f} — applies "
                     f"on the next segment.")

    def start_or_stop(sess, label_space, layers=None):
        """START / STOP on one button (deck skin).

        STOP drains what is queued and stops the clock: the pre-rendered
        segment is discarded, the timer goes inactive so no further segments
        are produced, and the ring player (if streaming to the sound card) is
        stopped.

        The BUTTON LABEL is the running state. s21's transport lights read it
        to decide which control pulses, so there is no second flag that can
        fall out of step with the session -- the same reason the strip gate is
        emitted by the toggle that sets it.
        """
        # The valence axis is honoured at session start and nowhere else, so it
        # is LOCKED while a session runs and released by STOP (2026-08-02).
        # feedback: a live-looking dropdown "makes user think they can change
        # inflight, and you are not actually honouring it" -- the same class of
        # lie as a control that looks live and does nothing. Swapping mid-walk
        # would change the instrument under the listener (it picks a different
        # anchor at 99.8% of targets), so the honest fix is to grey it, not to
        # start honouring it.
        if sess and sess.get("running"):
            sess["running"] = False
            invalidate_prerender(sess)
            if _RING is not None:
                try:
                    _RING.stop()
                except Exception as e:          # a stopped stream is not fatal
                    print(f"[stop] ring player: {e}")
            return (sess, "Stopped. Press **START** to begin a new session — "
                    "the valence axis can be changed now.",
                    gr.Timer(active=False), gr.update(value="START"),
                    gr.update(interactive=True))
        sess, msg = new_session(label_space, layers)
        sess["running"] = True
        # A new session is a new delivery clock -- otherwise the lead is
        # measured from the PREVIOUS session's first segment and the first
        # tick of this one computes a huge stale lead and never fires.
        _WA_CLOCK.pop(sess.get("session_id"), None)
        # Ask for the first segment straight away in gapless mode: the old
        # ~10s wait was dead air AND it meant playback started with no cushion
        # at all (see _webaudio_tick).
        first = 1.0 if _WEBAUDIO else STEP_SECONDS + 4.0
        return (sess, msg, gr.Timer(value=first, active=True),
                gr.update(value="STOP"), gr.update(interactive=False))

    def apply_policy_now(sess, mode_label):
        """The policy panel's APPLY: commit the staged policy AND move now.

        CHANGE NOW became this button (2026-08-02). Committing a
        policy you cannot hear until the next spontaneous transition is not a
        commit you can judge, so applying it also forces the move.
        """
        sess, msg = set_policy_mode(sess, mode_label)
        if not sess:
            return sess, msg
        sess, _ = force_transition(sess)
        return sess, msg.replace("applies to the next transition",
                                 "moving now") if "applies to" in msg else msg

    def force_transition(sess):
        if not sess:
            return sess, gr.update()
        sess["force"] = True
        invalidate_prerender(sess)
        return sess, "Forcing a transition on the next segment."

    def render_step(sess):
        """Advance the walk by one step and render ONE segment.

        Everything that used to be the body of advance(). Split out so it can
        be run on a COPY of the session by the pre-render thread -- the copy is
        what makes discarding a stale segment possible, because the render
        mutates a dozen pieces of session state (the walk, the step counter,
        the reverb tail, the bed clock, the loudness junction) and rolling that
        back in place would be hopeless."""
        # auto-wander (liveliness Tier A): the TARGET itself takes a slow
        # random step each segment, so the piece keeps evolving -- and keeps
        # crossing the trigger threshold -- without console input. 0 = off
        # (the pre-2026-07-12 static-target behavior). The walk still goes
        # through the guard, so an off-manifold target just pulls, safely.
        wander = float(sess.get("wander_std", 0.0))
        if wander > 0.0:
            sess["target_va"] = np.clip(
                sess["target_va"] + np.random.normal(0.0, wander, size=2),
                -1.0, 1.0)
        n_steps = int(STEP_SECONDS * CONTROL_HZ)
        # Looked up per segment rather than captured: the guard is fitted per
        # label space and the space is a per-SESSION choice. Kept out of
        # `sess` deliberately -- sess is deep-copied for the render-ahead
        # snapshot, and a fitted GP is not something to copy every segment.
        _guard, _ = guards.get(sess.get("guard_space", "judge"))
        sess["curr_va"] = ou_step(sess["curr_va"], sess["target_va"], _guard,
                                  n_steps, energy=sess.get("walk_energy", 1.0))
        val_shift = (abs(sess["target_va"][0] - sess["curr_va"][0])
                     > sess.get("trigger_threshold", RISE_THRESHOLD))
        trigger = val_shift or sess["force"]

        audio, theta, info = decide_and_render(sess, engine, renderer, fit, pool,
                                               contexts, trigger)
        from arranger import SR as _SR
        audio = postprocess_segment(np.asarray(audio, dtype=np.float64),
                                    sess["curr_va"][1], sess, _SR)
        # layer 5: re-pick the mood-matched bed at a transition (VA has moved),
        # then mix it under the leveled drone; no-op unless the bed is enabled.
        # VARIETY (2026-07-25): exclude the CURRENT bed and sample among the
        # k VA-nearest, so each transition rotates to a different mood-matched
        # bed instead of the deterministic argmin replaying one bed per region.
        if sess.get("bed_enabled") and sess.get("bed_auto") and trigger:
            cur = (sess.get("bed") or {}).get("bed_file")
            nb = bedbank.select_bed(sess.get("bed_bank", []),
                                    va=tuple(sess["curr_va"]),
                                    exclude=({cur} if cur else frozenset()),
                                    k=int(sess.get("bed_variety_k", 3)),
                                    rng=sess.get("bed_rng"))
            if nb:
                sess["bed"] = nb
        audio = apply_melody(audio, sess, _SR, theta)  # melodic line (on/off + style)
        audio = apply_bed(audio, sess, _SR)
        audio = apply_crackle(audio, sess, _SR)   # vinyl crackle overlay (on/off)
        audio = apply_reverb(audio, sess, _SR)    # measured reverb space (on/off + choice)
        # FINAL loudness: levels what is actually HEARD (drone + every layer)
        # and owns peak safety for the whole chain. Must stay last.
        audio = final_loudness(audio, sess["curr_va"][1], sess, _SR)
        # AFTER final_loudness on purpose -- see apply_distance's note. No-op
        # unless the distance control is enabled, so the default path is
        # unchanged and still mono.
        audio = apply_distance(audio, sess, _SR)
        sess["prev_theta"] = theta
        sess["step"] += 1
        if trigger:
            log_decision(dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                              session_id=sess["session_id"], step=sess["step"], **info))
        sess["force"] = False

        v, a = sess["curr_va"]
        tv, ta = sess["target_va"]
        status = (f"**Step {sess['step']}** — mood valence {v:.2f}, arousal {a:.2f} "
                 f"(target {tv:.2f}, {ta:.2f})\n\n")
        if trigger:
            method_name = {"gp_predict": "learned model",
                           "lookup": "baseline",
                           "lookup_fallback_gp": "learned model (no baseline match)"
                           }.get(info["chosen_method"], info["chosen_method"])
            reason_name = "forced" if info["trigger"] == "force" else "mood shift"
            tex_name = policy.TEXTURE_NAMES.get(
                info.get("chosen_texture", 0.0), info.get("chosen_texture", 0.0))
            status += (f"Transition ({reason_name}, via {method_name}): "
                      f"ramp {info['chosen_ramp_s']}s, "
                      f"chord {info['chosen_chord_start']}→{info['chosen_chord_end']}, "
                      f"texture {tex_name}")
        else:
            status += "Holding (no transition)"

        # VISUALS (s22). The numbers THIS segment was rendered from, so the
        # picture and the audio describe the same moment -- the visual is a
        # readout, and a readout that leads or lags what is playing would be
        # the same class of lie as a status line that reports a segment the
        # browser never received (SS28). Chord and texture only move on a
        # transition, so they persist between them rather than resetting.
        if trigger:
            sess["vis_chord"] = info.get("chosen_chord_end",
                                         sess.get("vis_chord", "min"))
            sess["vis_tex"] = float(info.get("chosen_texture", 0.0) or 0.0)
            sess["vis_ramp"] = float(info.get("chosen_ramp_s", 0.0) or 0.0)
            # the GP's own uncertainty about this choice -- drawn as a halo, so
            # the picture shows where the model is guessing rather than only
            # what it decided. Absent on the lookup path, hence the .get chain.
            sess["vis_std"] = float(info.get("gp_std", 0.0) or 0.0)
        sess["vis"] = dict(v=float(v), a=float(a), tv=float(tv), ta=float(ta),
                           chord=sess.get("vis_chord", "min"),
                           tex=sess.get("vis_tex", 0.0),
                           ramp=sess.get("vis_ramp", 0.0),
                           std=sess.get("vis_std", 0.0),
                           # the arranger's own breath period, so the vortex can
                           # pulse at a rate that is already audible rather than
                           # at one the shader invented.
                           period=float(sess.get("tex_breath_period", 30.0)),
                           # `trig` is an EDGE, not a level: the browser draws a
                           # one-off event mark when the step number changes and
                           # this is set, so it must not persist between segments.
                           trig=1 if trigger else 0,
                           step=int(sess["step"]))

        from arranger import SR
        wav_path = None
        if _RING is None:   # browser paths (audio element OR webaudio) need a file
            # only the browser player needs a file on disk; streaming straight
            # to the card would otherwise litter one wav per segment forever
            ext = SEG_FORMAT.lower()
            wav_path = str(HERE / "gradio_tmp"
                           / f"seg_{sess['session_id']}_{sess['step']}.{ext}")
            import soundfile as sf
            if ext == "wav":
                sf.write(wav_path, audio, SR)
            else:
                sf.write(wav_path, audio, SR, format=ext.upper())

        # Re-arm the timer to fire just AFTER this segment actually finishes
        # playing, instead of a fixed guess -- segment length varies with the
        # chosen ramp_s (13-26s for a triggered arc, 14s for a normal
        # crossfade) while the old fixed STEP_SECONDS+4.0=10s period was
        # shorter than every one of those, so the player was being handed a
        # replacement audio file (and restarting from silence) 4-16s before
        # the current segment ended -- the reported "stuttering" and
        # transitions that never seem to resolve are both this same bug.
        # SNAPSHOT THE SETTINGS THIS SEGMENT WAS RENDERED WITH. Taken here, not
        # read back later, because by the time it is playing the session may
        # already carry a change the listener has not heard yet -- which is the
        # whole point of the NOW / NEXT readout. `advance` copies this onto
        # sess["played"] once the segment is actually served.
        return dict(path=wav_path, status=status, audio=audio,
                    settings=settings_rows(sess),
                    duration=len(audio) / SR)

    def advance(sess):
        """Serve a segment, then render the NEXT one while this one plays.

        THE GAP THIS FIXES (measured 2026-08-01 on a real session: median 4.3s,
        max 7.1s of silence at every join): the timer was armed to
        `duration + 0.5s`, so segment N+1 only STARTED rendering once N had
        finished playing. The listener heard the renderer.

        Now the render happens during playback, on a COPY of the session, and
        the finished segment waits in `_PRERENDER`. Two consequences worth
        being explicit about:

        * a control you touch mid-segment DISCARDS the pre-rendered one and
          renders fresh, so response stays one segment -- what it is today.
          Without that, at 22-42s segments a knob change would land up to 80s
          later, which is useless for something steered by ear.
        * the copy is why discarding is possible at all. The render mutates
          the walk, the step counter, the reverb tail, the bed clock and the
          loudness junction; a discarded in-place render would leave all of
          that advanced with nothing played.

        If anything goes wrong -- deepcopy refuses, the thread dies -- it falls
        back to rendering inline, i.e. exactly the old behaviour with the old
        gap. Never a broken session for the sake of a faster one.
        """
        if not sess:
            return sess, None, "Press Start session first.", gr.Timer(value=STEP_SECONDS + 4.0)
        sid = sess.get("session_id")
        slot = _PRERENDER.pop(sid, None)

        seg = None
        if slot is not None:
            th = slot.get("thread")
            if th is not None:
                th.join()                      # normally finished long ago
            if not slot.get("stale") and slot.get("seg") is not None:
                seg = slot["seg"]
                sess.clear()                   # adopt the copy the render used
                sess.update(slot["sess"])
        if seg is None:
            seg = render_step(sess)            # first segment, or discarded

        # What is about to be HEARD becomes the NOW column. Set after the adopt
        # above -- `sess.clear()` wipes this key, so writing it earlier would
        # lose it on every prerendered segment (i.e. almost all of them).
        sess["played"] = seg.get("settings") or []

        _prerender_start(sess, render_step)
        prune_segments(sess.get("session_id"))

        # STOP must stay stopped: NEXT shares this handler with the timer tick,
        # so without this a click on NEXT after STOP would re-arm the clock and
        # quietly resume generating. Classic sessions never set "running", so
        # they default to True and behave exactly as before.
        _running = bool(sess.get("running", True))

        if _RING is not None:
            # STREAMING: hand the samples to the sound card and leave the
            # browser player empty. Pacing is driven by how much audio is
            # BUFFERED, not by segment length -- at `duration + 0.5` we would
            # push `duration` of audio every `duration + 0.5` seconds and the
            # buffer would bleed 0.5s per segment until it underran.
            _RING.push(seg["audio"])
            if _RING.ready():
                _RING.start()
            buffered = _RING.buffered_s
            status = (seg["status"] + f"\n\n🔊 streaming to the sound card — "
                      f"{buffered:.0f}s buffered"
                      + (f", {_RING.underruns} underruns" if _RING.underruns else ""))
            return (sess, None, status + settings_panel(sess),
                    gr.Timer(value=max(1.0, buffered - 1.0), active=_running))

        if _WEBAUDIO:
            # GAPLESS IN-BROWSER. The page schedules this segment to begin at
            # the exact sample the previous one ends, so we can hand it over
            # EARLY -- the 0.5s hole in the branch below exists only because a
            # plain <audio> element truncates whatever is playing when you set
            # a new source. Firing at half a segment keeps the page roughly one
            # segment ahead without flooding it.
            # the panel goes to the LED only -- sink_html's copy is a compact
            # in-player overlay, and a settings table does not belong in it.
            return (sess, webaudio.sink_html(webaudio.file_url(seg["path"]),
                                             seg["status"]),
                    seg["status"] + settings_panel(sess),
                    gr.Timer(value=_webaudio_tick(sess, seg), active=_running))

        return (sess, seg["path"], seg["status"] + settings_panel(sess),
                gr.Timer(value=seg["duration"] + 0.5, active=_running))

    def _panel(fn, idx=1):
        """Wrap a status-returning handler so the LED keeps its NOW/NEXT table.

        Every setter writes to C["status"], and that REPLACES the whole
        readout -- so without this the table would disappear the moment you
        touched any control, which is precisely when it is being looked at.
        Wrapping also means the table refreshes on the same round-trip that
        stages the change, with no extra event and no second timer (a 1s poll
        is what flooded the share relay in s04 and returned HTML error pages).

        Applied at the WIRING sites rather than inside the handlers because
        several have many return paths -- set_melody alone has five -- and a
        missed one would leave the table STALE rather than absent, which is the
        silent-wrong-answer failure SS22.7 exists to prevent.

        `idx` is the position of the status string in the handler's output
        tuple; it is not always 1 (pad_click returns it 4th).
        """
        import functools

        @functools.wraps(fn)
        def wrapped(*a, **kw):
            out = fn(*a, **kw)
            tup = isinstance(out, tuple)
            vals = list(out) if tup else [out]
            if idx < len(vals) and isinstance(vals[idx], str):
                sess = vals[0] if vals and isinstance(vals[0], dict) else None
                vals[idx] = vals[idx] + settings_panel(sess)
            return tuple(vals) if tup else vals[0]
        return wrapped

    _INTRO = (
        "A continuously evolving emotional soundscape. Set a target mood "
        "on the valence and arousal sliders below, or let it drift on its "
        "own. When the target mood shifts, a learned preference model "
        "chooses how to move between soundscapes. **Force transition now** "
        "triggers a move immediately. The **transition policy** control "
        "selects whether the learned model or the baseline drives each move.")

    # delete_cache=(frequency_s, age_s). gradio COPIES every file it serves
    # into GRADIO_TEMP_DIR and never sweeps it on its own -- 299 MB locally,
    # and this app serves a fresh segment every ~22 s forever. `prune_segments`
    # only owns our own seg_* files, not gradio's copies of them.
    # Ages are deliberately generous: a segment is served, then FETCHED by the
    # browser a moment later, and deleting one in that window is exactly the
    # "no audio but the status says it's working" failure of SS28. 15 minutes
    # is ~40 segments of slack.
    with gr.Blocks(title=APP_NAME,
                   delete_cache=(300, 900)) as demo:
        gr.HTML(ATTRIBUTION_BANNER)
        gr.HTML(build_chip())
        # The deck is meant to fit ON ONE SCREEN (2026-08-02), and
        # a six-line paragraph of vertical space is the cheapest thing to
        # give up -- it is orientation for a first-time reader, not a
        # control. Folded, not deleted.
        with gr.Accordion(f"{APP_NAME} — what this is", open=False):
            gr.Markdown(_INTRO)
        # UI AS A SKIN (2026-08-02). The surface is built by a skin that
        # returns a component dict keyed BY NAME, so layout and behaviour stay
        # independent -- every handler below wires `C["name"]` and never a
        # layout position. `validate()` enforces the contract, so a skin that
        # forgets a control fails at construction rather than building a UI
        # whose buttons quietly do nothing.
        #
        # Only the DECK ships. A second "classic" surface existed and was
        # dropped on migration: it was 272 lines of duplicate component
        # construction that had to be kept in step with this one by hand, and
        # every drift between them was a bug that only appeared under one
        # skin. The contract stays because it is what keeps the app free of
        # layout assumptions, not because there is a choice to make.
        C = skins.build_deck(gr, dict(
            pad=(str(default_pad()) if default_pad().exists() else None),
            reverb_choices=reverb_choices(),
            bed_labels=sorted(bed_by_label),
            melody_engines=melody_engine_choices(),
            melody_engine=default_engine(),
            melody_ckpts=melody_checkpoint_choices(),
            # first-run defaults, passed IN so the skin never hardcodes
            # them and cannot drift from the values the engine starts at
            # (see START_VA)
            melody_ckpt=default_ckpt(),
            start_va=START_VA,
            label_space=DEFAULT_LABEL_SPACE,
            melody_styles=melody_style_choices(),
            yesno=YESNO, like_label=LIKE_LABEL, match_label=MATCH_LABEL,
            audio_factory=(lambda: gr.HTML(webaudio.sink_html(
                None, "gapless mode — press Start"))) if _WEBAUDIO
                else (lambda: gr.Audio(label="Live soundscape",
                                       type="filepath", autoplay=True)),
            timer_start=STEP_SECONDS + 4.0))
        # The contract check. It ran only against the classic surface
        # before, so the deck -- the one that shipped -- was never
        # actually validated against REQUIRED. It is now.
        skins.validate(C, "deck")

        def swap_pad(label_space):
            """The pad is a MEASUREMENT of the bank, so it has to follow the
            label space: judge valence tops out at +0.24 and human at +0.57,
            and 38.5% vs 27.4% of the square is unreachable. Showing the judge
            map while retrieving on human labels would misreport where the
            instrument can actually go -- the very thing the pad exists to
            prevent (README §22.2)."""
            want = PAD_HUMAN if str(label_space).startswith("human") else PAD_IMG
            if not want.exists():
                return gr.update()
            # visible=True as well as the value. The skin hides this component
            # when the asset is absent AT CONSTRUCTION, so an app started
            # before the pads were in place keeps a hidden pad for the life of
            # the process -- setting only the value updates something nobody
            # can see. Asserting visibility here lets the surface recover.
            return gr.update(value=str(want), visible=True)

        if C.get("va_pad") is not None:
            C["label_space"].change(swap_pad, inputs=[C["label_space"]],
                                    outputs=[C["va_pad"]])

        # START READS THESE (2026-08-04). A layer configured BEFORE the session
        # existed used to be thrown away -- see adopt_layers for the full
        # account -- so START now seeds the session from whatever these
        # controls read at the moment it is pressed.
        #
        # Built by filtering C, so the deck's extra `reverb_enable` is picked
        # up and classic, which has no such control, simply does not send it.
        # ONE list builds the `inputs` AND unpacks the values, so the wiring
        # order and the unpack order cannot drift apart -- the failure mode
        # that the "ORDER MUST MATCH set_melody's signature" comment below
        # exists to warn about.
        _ADOPT = [k for k in ("melody_on", "melody_engine", "melody_ckpt",
                              "melody_style", "melody_level", "melody_env",
                              "reverb_enable", "reverb_pick", "reverb_wet",
                              "bed_enable", "bed_auto", "bed_choice", "bed_gain",
                              "crackle_enable", "crackle_gain", "crackle_density")
                  if C.get(k) is not None]
        _adopt_in = [C[k] for k in _ADOPT]

        # One button, two jobs, and its LABEL is the state the transport
        # lights read (s21). Classic keeps a plain START.
        def _start_deck(sess, label_space, *vals):
            return start_or_stop(sess, label_space, dict(zip(_ADOPT, vals)))

        C["start_btn"].click(_panel(_start_deck),
                             inputs=[C["state"], C["label_space"]] + _adopt_in,
                             outputs=[C["state"], C["status"], C["timer"],
                                      C["start_btn"], C["label_space"]])
        def pad_click(sess, evt: gr.SelectData):
            """Pixel -> VA. make_pad_assets draws the axes with no margins over
            exactly [-1,1]^2, so this is a straight linear map; y is flipped
            because image rows run downward and arousal runs up.

            The click STAGES the target on the two controls; SET TARGET commits
            it (2026-08-02). It used to apply immediately, but the deck pulses
            that button while a selection is uncommitted -- and a confirm
            prompt for something already applied is a button that lies. Both
            routes into the target now behave the same way: move, then commit.
            """
            if C.get("va_pad") is None:
                return sess, gr.update(), gr.update(), "no pad asset"
            (px, py), (w, h) = evt.index, (512, 512)
            v = float(np.clip(2.0 * px / w - 1.0, -1, 1))
            a = float(np.clip(1.0 - 2.0 * py / h, -1, 1))
            if not sess:
                return sess, v, a, "Press Start session first."
            return sess, v, a, (f"Target staged: valence {v:+.2f}, arousal "
                                f"{a:+.2f} — press **SET TARGET** to apply.")

        if C["va_pad"] is not None:
            if C.get("va_dots") is not None:
                def pad_click_dots(sess, evt: gr.SelectData):
                    # The red dot follows the CLICK, not the commit: a staged
                    # target you cannot see is not a staged target. SET TARGET
                    # repaints it anyway, from the session rather than the
                    # click, so the two can never disagree.
                    out = pad_click(sess, evt)
                    return out + (skins.dot_style("tgt", float(out[1]),
                                                  float(out[2])),)

                C["va_pad"].select(_panel(pad_click_dots, 3), inputs=[C["state"]],
                                   outputs=[C["state"], C["v_slider"],
                                            C["a_slider"], C["status"],
                                            C["va_dot_tgt"]])
            else:
                C["va_pad"].select(_panel(pad_click, 3), inputs=[C["state"]],
                              outputs=[C["state"], C["v_slider"], C["a_slider"], C["status"]])

        if C.get("va_dots") is not None:
            def set_target_dots(sess, v, a):
                sess, msg = set_target(sess, v, a)
                return sess, msg, skins.dot_style("tgt", float(v), float(a))

            C["set_btn"].click(_panel(set_target_dots),
                               inputs=[C["state"], C["v_slider"], C["a_slider"]],
                               outputs=[C["state"], C["status"],
                                        C["va_dot_tgt"]])
        else:
            C["set_btn"].click(_panel(set_target),
                               inputs=[C["state"], C["v_slider"], C["a_slider"]],
                               outputs=[C["state"], C["status"]])
        # The policy panel is a strip like any other: the two square
        # buttons STAGE a choice into the hidden radio and pulse APPLY;
        # APPLY (formerly CHANGE NOW) commits it and moves the music so
        # the choice can actually be heard.
        C["force_btn"].click(_panel(apply_policy_now),
                             inputs=[C["state"], C["policy_radio"]],
                             outputs=[C["state"], C["status"]])
        _live_ins = [C["state"], C["live_thresh"], C["live_energy"], C["live_wander"],
                     C["live_k"], C["live_hold"], C["live_xfade"],
                     C["live_level"], C["live_again"], C["live_voicing"],
                     C["live_breath"], C["live_breath_p"],
                     C["live_cents"], C["live_reverb"]]
        if C.get("texture_on") is not None:
            # The deck's TEXTURE strip has an ON/OFF like every other strip.
            # The engine has no "texture off" state -- only depths that reach
            # zero -- so OFF is DEFINED here as flat: no dark voicing, no
            # wander. Variety is retrieval, not texture, so it is left alone.
            # Doing it at apply time means the fader positions are remembered
            # and come back when the strip is switched on again.
            def set_liveliness_gated(sess, on, thresh, energy, wander, k, hold,
                                     xfade, level, again, voicing, breath,
                                     breath_p, cents, reverb):
                if not on:
                    voicing, breath = 0.0, 0.0
                sess, msg = set_liveliness(sess, thresh, energy, wander, k, hold,
                                           xfade, level, again, voicing, breath,
                                           breath_p, cents, reverb)
                if sess and not on:
                    msg += "  Texture **off** — flat voicing, no wander."
                return sess, msg

            C["live_btn"].click(_panel(set_liveliness_gated),
                                inputs=[_live_ins[0], C["texture_on"]] + _live_ins[1:],
                                outputs=[C["state"], C["status"]])
        else:
            C["live_btn"].click(_panel(set_liveliness), inputs=_live_ins,
                                outputs=[C["state"], C["status"]])
        # Toggling a layer APPLIES IMMEDIATELY -- on the classic surface only.
        # The deck is apply-driven (2026-08-02): its strip buttons stage the
        # change, grey the strip in or out, and pulse that strip's APPLY, which
        # is what reaches the conductor. Wiring both would mean the deck sent a
        # half-configured layer on the toggle and the configured one on APPLY.
        C["bed_btn"].click(_panel(set_bed),
                      inputs=[C["state"], C["bed_enable"], C["bed_auto"], C["bed_choice"], C["bed_gain"]],
                      outputs=[C["state"], C["status"]])
        # OPTIONAL strip: the classic skin does not build it, so wire only if
        # the active skin provided it -- same contract as vis_data.
        if C.get("dist_btn") is not None:
            C["dist_btn"].click(_panel(set_distance),
                          inputs=[C["state"], C["dist_enable"], C["dist_amount"],
                                  C["dist_drift"]],
                          outputs=[C["state"], C["status"]])
        C["crackle_btn"].click(_panel(set_crackle),
                          inputs=[C["state"], C["crackle_enable"], C["crackle_gain"], C["crackle_density"]],
                          outputs=[C["state"], C["status"]])
        def melody_slot(engine):
            """One middle dropdown, whichever the engine actually reads.

            The brief, 2026-08-02 -- and he is right that they are half-dead as a
            pair: `style` is ignored on the transformer path (set_melody says so
            in its own return), and `ckpt` means nothing to the markov engine,
            which has no weights. Showing both invites you to set one that will
            be discarded. Swapping VISIBILITY rather than merging them into one
            component keeps both in the contract, so set_melody's signature and
            the classic surface are untouched.
            """
            tf = str(engine) == "transformer"
            return gr.update(visible=not tf), gr.update(visible=tf)

        C["melody_engine"].change(melody_slot, inputs=[C["melody_engine"]],
                                  outputs=[C["melody_style"], C["melody_ckpt"]])

        if C.get("reverb_enable") is not None:
            # The deck's REVERB strip has its own on/off button, so "off" left
            # the dropdown: the dropdown says WHICH room, the button says
            # WHETHER. Fold them back together here rather than teaching
            # set_reverb a second meaning -- its "off" contract, and the
            # classic skin that relies on it, stay exactly as they were.
            def set_reverb_gated(sess, on, cond, wet):
                return set_reverb(sess, cond if on else "off", wet)

            C["reverb_btn"].click(_panel(set_reverb_gated),
                                  inputs=[C["state"], C["reverb_enable"],
                                          C["reverb_pick"], C["reverb_wet"]],
                                  outputs=[C["state"], C["status"]])
        else:
            C["reverb_btn"].click(_panel(set_reverb),
                                  inputs=[C["state"], C["reverb_pick"], C["reverb_wet"]],
                                  outputs=[C["state"], C["status"]])
        C["melody_btn"].click(_panel(set_melody),
                         # ORDER MUST MATCH set_melody's signature:
                         # (sess, enabled, engine, ckpt, style, level_db, envelope)
                         inputs=[C["state"], C["melody_on"], C["melody_engine"], C["melody_ckpt"],
                                 C["melody_style"], C["melody_level"], C["melody_env"]],
                         outputs=[C["state"], C["status"]])
        C["log_btn"].click(_panel(log_reaction),
                      inputs=[C["state"], C["fb_like"], C["fb_match"],
                              C["note_box"]],
                      outputs=[C["state"], C["status"], C["note_box"]])
        _adv_out = [C["state"], C["audio"], C["status"], C["timer"]]
        _adv_fn = advance
        if C.get("va_dots") is not None:
            # Live VA markers (deck only): green where the walk is, red where
            # it is being sent. Computed from the session AFTER the segment, so
            # the dot never claims a position the audio has not reached.
            def _va(sess, key, default=(0.0, 0.0)):
                """curr_va/target_va are numpy ARRAYS, so `x or default` raises
                (truth value of an array is ambiguous). Test for None."""
                x = (sess or {}).get(key)
                if x is None:
                    return float(default[0]), float(default[1])
                return float(x[0]), float(x[1])

            def advance_dots(sess):
                # GREEN ONLY. The red target is a user intention -- it moves
                # when you click the map or press SET TARGET, and at no other
                # time. Repainting it here would drag it across the map on
                # every clip change, because the target auto-wanders.
                out = advance(sess)
                cv, ca = _va(out[0], "curr_va")
                return out + (skins.dot_style("cur", cv, ca),)

            _adv_fn = advance_dots
            _adv_out = _adv_out + [C["va_dot_cur"]]
        if C.get("vis_data") is not None:
            # Push the segment's own numbers to the visuals tab, on the SAME
            # round-trip that serves the audio -- one update per segment, not
            # per frame. The animation interpolates between these; a per-frame
            # server feed is what would flood the share relay (s04's 1s timer
            # returned HTML error pages).
            _vis_base = _adv_fn

            def advance_vis(sess):
                out = _vis_base(sess)
                return out + (vis.data_html((out[0] or {}).get("vis")),)

            _adv_fn = advance_vis
            _adv_out = _adv_out + [C["vis_data"]]
        C["timer"].tick(_adv_fn, inputs=[C["state"]], outputs=_adv_out)
        C["next_btn"].click(_adv_fn, inputs=[C["state"]], outputs=_adv_out)
    return demo


# ------------------------------------------------ render-ahead (2026-08-01)
# Kept OUT of the gr.State session dict on purpose: gradio may copy state
# between events, and a Thread is not copyable. Keyed by session_id.
_PRERENDER = {}
_RENDER_LOCK = threading.Lock()


def _prerender_start(sess, render_fn):
    """Render the NEXT segment in the background, on a COPY of the session."""
    sid = sess.get("session_id")
    if sid is None:
        return
    try:
        work = copy.deepcopy(sess)
    except Exception as e:                     # unpicklable state -> no ahead
        print(f"[render-ahead] disabled: {e}")
        return
    slot = {"thread": None, "seg": None, "sess": work, "stale": False}

    def _run():
        try:
            # one render at a time: `_voiced` patches the frozen arranger's
            # WAYPOINT_VOICES globally and restores them, which is only safe
            # because the render path is serialised.
            with _RENDER_LOCK:
                slot["seg"] = render_fn(work)
        except Exception as e:
            print(f"[render-ahead] failed, will render inline: {e}")
            slot["seg"] = None

    t = threading.Thread(target=_run, daemon=True)
    slot["thread"] = t
    _PRERENDER[sid] = slot
    t.start()


def invalidate_prerender(sess):
    """A control changed: the segment waiting in the wings is out of date.

    Marks rather than deletes, because the thread may still be running -- and
    joining a render on the UI thread is exactly the stall this whole thing
    exists to remove."""
    slot = _PRERENDER.get((sess or {}).get("session_id"))
    if slot is not None:
        slot["stale"] = True


DEFAULT_PORT = 7860


def _port_free(port, host="0.0.0.0"):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, int(port)))
            return True
        except OSError:
            return False


def other_conductors():
    """PIDs of other running conductors -- the usual reason the port is taken.

    Linux only (there is no /proc on Windows); an empty list there just means
    the message loses the PID, not that the check is skipped."""
    out = []
    if not os.path.isdir("/proc"):
        return out
    me, mine = os.getpid(), Path(__file__).name
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                argv = fh.read().decode("utf8", "replace").split("\0")
        except OSError:
            continue
        # An ARGV ENTRY must BE the script -- not merely a command line that
        # mentions it. Matching the raw string caught shells running `grep
        # s02_conductor_app` and reported them as stale servers.
        if any(a.rsplit("/", 1)[-1] == mine for a in argv if a):
            out.append((int(pid), " ".join(a for a in argv if a)))
    return out


# Segments requested back-to-back at session start, to build a playback
# cushion before the first one is consumed. See _webaudio_tick.
# How much audio the browser is kept ahead by. A FLOOR, not a target: the
# lead falls to this, one segment is handed over, and it decays again -- so it
# oscillates in [floor, floor+duration] instead of growing without limit. Well
# above the 1-2s lateness ever measured on the Space, and small enough that a
# knob change is heard within a segment or two.
WEBAUDIO_LEAD_FLOOR_S = 8.0
# session_id -> {"t0": monotonic at first delivery, "sched": seconds handed
# over}. Module-level ON PURPOSE -- see _webaudio_tick: state written to `sess`
# here is rolled back by the prerender adopt-the-copy step one tick later.
_WA_CLOCK = {}


def _webaudio_tick(sess, seg):
    """How soon to ask for the next segment: pace by LEAD, not by fraction.

    THE CUSHION IS THE POINT, BUT IT MUST BE BOUNDED. The previous rule
    asked again after half a segment, which hands over a whole segment
    every half segment -- so the browser's scheduled lead grew by
    duration/2 EVERY TICK, without limit. Simulated at 22s segments: 93s
    of lead after 17s of listening, 258s after 3 minutes, 478s after 7.
    That is why the gaps vanished, and it is also why the deck went deaf:
    WebAudio schedules each segment at an exact sample offset and nothing
    can recall it, so turning on crackle, reverb, texture or melody
    changed audio that would not be reached for MINUTES. Four layers
    "not working" was one pacing bug (2026-08-02).

    So: track how much audio has been handed over against the wall clock,
    and wait until the lead decays to `WEBAUDIO_LEAD_FLOOR_S` before
    sending the next one. The lead then oscillates between the floor and
    floor+duration instead of growing, the cushion stays far above the
    1-2s lateness ever measured, and a control change is heard within a
    segment or two.

    The clock lives in a MODULE dict, not in `sess`: `advance` deepcopies
    the session for the background render BEFORE this runs and adopts that
    copy on the next tick, so anything written to `sess` here is rolled
    back one tick later -- which would restore the unbounded growth while
    looking correct.
    """
    sid = sess.get("session_id")
    now = time.monotonic()
    clk = _WA_CLOCK.get(sid)
    if clk is None:
        clk = _WA_CLOCK[sid] = {"t0": now, "sched": 0.0}
    clk["sched"] += float(seg["duration"])
    lead = clk["sched"] - (now - clk["t0"])
    return max(1.0, lead - WEBAUDIO_LEAD_FLOOR_S)



SEG_KEEP = 6          # segments retained for THIS session, newest first
SEG_MAX_AGE_S = 300   # anything else must be this old before it may be deleted


def prune_segments(session_id=None, keep=SEG_KEEP, max_age_s=SEG_MAX_AGE_S):
    """Delete rendered segments that nothing can still be playing.

    NOTHING cleaned these up before (2026-08-02: 339 files / 38 MB over 7
    sessions). One file per segment for an installation meant to run for hours
    is unbounded growth on the machine a professor is running it on.

    CORRECTED the same day. The first version deleted every segment belonging
    to another session id, on the reasoning that "once a session ends nothing
    refers to them again". That reasoning is wrong in its premise: a different
    session id is NOT evidence that the session is dead. Sessions coexist
    whenever there are two browser tabs, a reloaded page whose old connection
    has not dropped yet, or -- the one that actually bit -- a RESTARTED APP
    whose previous process is still alive, since gradio then binds the next
    free port while the old one keeps serving on 7860 and both share this
    directory. Each live session then deleted the other's just-rendered
    segment in the window between the server handing over the URL and the
    browser fetching it: status text (server-side) said playing, audio (a
    fetch of a file now gone) was silent. Restarting "fixed" it only by
    eventually leaving one instance alive.

    So liveness is judged by AGE, which is observable, rather than by session
    identity, which is not:
      * within the CURRENT session keep the newest `keep`. Render-ahead means
        segment N+1 exists before N has played, and the browser may still be
        fetching one a moment after the next arrives, so the margin has to
        cover more than the one in flight. Six is ~2-4 minutes of audio.
      * anything else goes only once it is `max_age_s` old -- long past the
        point where a live session would still be fetching it, and far short
        of unbounded growth (a dead session leaves ~5 minutes of files).

    Never raises: a locked or vanished file just stays.
    """
    d = HERE / "gradio_tmp"
    if not d.is_dir():
        return 0
    now, mine, theirs = time.time(), [], []
    for q in d.glob("seg_*"):
        sid = q.name.split("_")[1] if q.name.count("_") >= 2 else ""
        (mine if session_id and sid == session_id else theirs).append(q)
    doomed = []
    for q in theirs:
        try:
            if now - q.stat().st_mtime > max_age_s:
                doomed.append(q)
        except OSError:
            pass                      # vanished under us; nothing to do
    if len(mine) > keep:
        doomed += sorted(mine, key=lambda q: q.stat().st_mtime)[:-keep]
    n = 0
    for q in doomed:
        try:
            q.unlink()
            n += 1
        except OSError:
            pass
    return n


def warm_up():
    """Build every lazily-constructed singleton BEFORE the server starts.

    2026-08-01, feedback: *"everything should really be loaded upfront and all
    objects created waiting to be used."* Right, and the symptom was concrete --
    ticking Melody on stalled the session while the transformer checkpoint and
    a whole DecoupledEngine were built inside `apply_melody`, inside
    `advance()`, i.e. on the audio thread in front of a listener. Startup is
    free; mid-session is not.

    Everything warmed here is held in a MODULE-LEVEL global, so nothing is
    collectable while the process lives -- `_MEL_TRANSFORMER`,
    `melody_markov._RVA_ENGINE`, `_BED_AUDIO_CACHE`, and reverb_bank's own
    lru_cache. Failures are reported and swallowed: a warm-up is an
    optimisation, and must never stop the app from starting.
    """
    print("warming up...")
    if meldrone is not None:
        try:                       # the melody layer's own bank/engine lookup
            from decoupled_engine import DecoupledEngine
            if getattr(meldrone, "_RVA_ENGINE", None) is None:
                meldrone._RVA_ENGINE = DecoupledEngine(BANKS)
            print("  melody render engine ready")
        except Exception as e:
            print(f"  melody render engine NOT warmed: {e}")
        for ck in melody_checkpoint_names():
            try:
                melody_transformer(ck)
                print(f"  transformer '{ck}' ready")
            except Exception as e:
                print(f"  transformer '{ck}' NOT warmed: {e}")
    if revbank is not None:        # IR load + resample, per condition
        try:
            from arranger import SR as _SR
            for cid in revbank.condition_ids():
                if cid != "dry":
                    revbank.impulse_response(cid, sr=_SR)
            print(f"  {len(revbank.condition_ids())} reverb IRs ready")
        except Exception as e:
            print(f"  reverb IRs NOT warmed: {e}")
    try:                           # bed wavs: decode once, not at first use
        bank = load_bed_bank()
        for b in bank[:BED_WARM_N]:
            if b.get("path"):
                _bed_audio(_resolve_bed(b["path"]))
        print(f"  {min(len(bank), BED_WARM_N)} of {len(bank)} beds decoded")
    except Exception as e:
        print(f"  beds NOT warmed: {e}")
    print("warm-up done")


def _announce_build():
    """Print the build FIRST, before anything slow.

    The log is where you look when a fix "did not work", and until now it said
    nothing about which build produced it -- so a stale process and a stale
    Space were indistinguishable from a broken fix. Banner-first, always."""
    line = build_id()
    print("=" * (len(line) + 8), flush=True)
    print(f"    {line}", flush=True)
    print("=" * (len(line) + 8), flush=True)


def make_app(webaudio_on=True, label_space="judge",
             audio_format=None, min_comparisons=40, audio_out=False,
             share=False, port=DEFAULT_PORT):
    """Build the demo AND its launch kwargs.

    Factored out of main() so the HuggingFace Space wrapper runs the SAME
    construction path as the CLI. The head/css assembly in particular is
    fiddly enough (see the note about css= below) that a second copy would
    drift, and a Space that quietly differs from the local app is the hardest
    kind of bug to chase.
    """
    global SEG_FORMAT, _RING, _WEBAUDIO
    _announce_build()
    from retrieval import GPSoftKNNEngine
    from arranger import ArrangedRenderer

    pool = policy.load_pool()

    # THE PREFERENCE GP IS LOADED, NOT FITTED (decision 2). This used to refit
    # a Laplace-approximated Bradley-Terry GP from the raw pairwise
    # comparisons at every boot -- which meant the conductor carried the
    # rating data, and that the policy a listener heard depended on a fit that
    # ran on their machine. The frozen posterior ships instead, so the arc
    # policy here is the one the report measured.
    fit = policy.PreferenceGP.load()
    contexts = policy.pool_contexts(pool)

    engine = GPSoftKNNEngine(BANKS)
    if label_space == "human":
        got = engine.set_label_space("human")
        print(f"label space: {got}" + ("" if got == "human" else
              "  (no valence_human column in this bank -- repack to get it)"))
    guards = GuardBySpace()
    renderer = ArrangedRenderer(device="cpu")

    SEG_FORMAT = (audio_format or SEG_FORMAT)
    print(f"segments served as .{SEG_FORMAT}")
    # RESET, don't just set. `_WEBAUDIO` is a module global and the branch
    # below only ever raised it, so a second make_app() in one process
    # inherited the first call's answer and `webaudio_on=False` silently kept
    # the gapless path. Invisible to the CLI, which builds one app -- but it
    # made a verification of the opt-out report the wrong thing, which is worse
    # than the bug.
    _WEBAUDIO = False
    if webaudio_on:
        if webaudio is None:
            print("playback: s19_webaudio missing, falling back to the "
                  "<audio> player (joins will be audible; no spectrum)")
        else:
            _WEBAUDIO = True
            print("playback: gapless in-browser scheduling ON")
    else:
        # SAY WHICH PATH IS LIVE, always. The <audio> fallback also silences
        # the spectrum and the delivery readout (both live in WEBAUDIO_HEAD),
        # and a deck that renders those two widgets regardless is exactly the
        # looks-live-does-nothing trap this project keeps re-learning.
        print("playback: plain <audio> player -- joins audible, "
              "spectrum and delivery readout inactive")

    if audio_out:
        from arranger import SR as _SR
        if ringplayer is None:
            print("--audio-out: s18_ring_player missing, using the browser player")
        elif not ringplayer.available():
            # a headless box, or sounddevice/portaudio not installed. Say so
            # and fall back rather than starting a silent session.
            print("--audio-out: no usable output device on this machine "
                  "(is this an SSH session?), using the browser player")
        else:
            _RING = ringplayer.RingPlayer(_SR)
            print(f"--audio-out: streaming to this machine's sound card at "
                  f"{_SR} Hz. NOTE the audio comes out HERE, not in the "
                  f"browser.")

    warm_up()

    # The VA pad is the PRIMARY CONTROL SURFACE, not an optional overlay: the
    # listener steers the session by dragging on it. The skin hides it when
    # the asset is missing at construction, which is a reasonable thing to do
    # with a broken image but a terrible thing to do silently -- the app comes
    # up looking deliberate, with no way to drive it and nothing in the log.
    # Every other optional layer degrades quietly by design; this one says so.
    missing = [q.name for q in (PAD_IMG, PAD_HUMAN) if not q.exists()]
    if missing:
        print(f"[pad] MISSING {', '.join(missing)} in {ASSETS} -- the VA pad "
              f"will not be shown and the session can only be steered with "
              f"the sliders. Regenerate with: python "
              f"../models/3.4.2_human_grounding_and_retrieval/train/"
              f"make_va_pads.py")

    demo = build_demo(engine, renderer, guards, fit, pool, contexts)
    # The ASSETS dir must be allowed too, not just gradio_tmp. A component's
    # initial value is handled at construction, but a path returned at RUNTIME
    # (the pad swapping when the label space changes) is checked against
    # allowed_paths and refused otherwise -- which is why the swap raised
    # InvalidPathError while the first load worked (2026-08-02).
    paths = []
    for q in (HERE / "gradio_tmp", PAD_IMG.parent):
        paths += [str(q), os.path.realpath(str(q))]
    paths = sorted(set(paths))
    launch_kw = dict(server_name="0.0.0.0", server_port=port,
                     share=not (not share), allowed_paths=paths)
    # THE SKIN'S CSS GOES THROUGH head=, NOT css= (2026-08-02). Measured:
    # a stylesheet passed as launch(css=...) reaches the browser inside the
    # config payload and is never applied -- the page has no <style> for
    # it -- while the SAME rules delivered in head= do apply, which is why
    # the drawn faders were styled (their CSS lives in FADER_HEAD) but
    # nothing from this file was. Symptom: the deck rendered with gradio's
    # default widths, so the Mood panel wrapped under the map and the
    # strips ran full width. This is the css= twin of the head=-on-Blocks
    # trap in README SS23 -- gradio accepts the argument and silently drops
    # the effect, so ALWAYS confirm a style rule actually landed.
    # s22's script MUST ride head= for a second, stronger reason than the
    # CSS above: gradio renders a component's HTML by writing innerHTML,
    # and a <script> inserted that way NEVER EXECUTES. Shipping the
    # animation loop as part of the tab's markup would have produced a
    # canvas that simply stays black, with nothing in the console.
    launch_kw["head"] = (launch_kw.get("head", "")
                         + f"<style>{skins.CSS}</style>"
                         + faders.FADER_HEAD
                         + vis.VIS_HEAD)
    if _WEBAUDIO and webaudio is not None:
        # head= on launch(), NEVER on Blocks() -- gradio 6 accepts it there and
        # silently drops it (the long-track Space lost a day to exactly this)
        launch_kw["head"] = launch_kw.get("head", "") + webaudio.WEBAUDIO_HEAD
    return demo, launch_kw


def main():
    global SEG_FORMAT, _RING, _WEBAUDIO         # all set from the CLI below; the
                                 # declaration must precede the first USE,
                                 # and --audio-format's default reads it
    ap = argparse.ArgumentParser(description="Arc-policy conductor (Phase 1 demo)")
    ap.add_argument("--selftest", action="store_true",
                    help="verify control-flow logic with fake engine/renderer, no server")
    ap.add_argument("--no-share", action="store_true")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help="port to serve on. Startup REFUSES if it is taken, "
                         "rather than quietly moving to the next one and "
                         "leaving your browser tab on the old instance.")
    ap.add_argument("--min-comparisons", type=int, default=40)
    ap.add_argument("--audio-format", default=SEG_FORMAT,
                    choices=["mp3", "ogg", "flac", "wav"],
                    help="container for segments sent to the browser. mp3 is "
                         "7.3x smaller than wav on real material, which is what "
                         "closes the join over a share tunnel; flac is the "
                         "lossless fallback at 1.6x.")
    ap.add_argument("--label-space", default="judge", choices=["judge", "human"],
                    help="which valence axis retrieval uses. SESSION-START "
                         "ONLY by design: the two disagree about the nearest "
                         "anchor at 99.8%% of VA targets, so this picks the "
                         "instrument, it does not tune it.")
    ap.add_argument("--webaudio", action="store_true",
                    help="(default, kept for compatibility) gapless in-browser "
                         "scheduling. Accepting this as a no-op keeps "
                         "space/app.py, the pixi `gapless` task and existing "
                         "command lines working unchanged.")
    ap.add_argument("--no-webaudio", dest="no_webaudio", action="store_true",
                    help="fall back to the plain <audio> player. Not gapless by "
                         "construction -- setting a new source truncates what "
                         "is playing, so the server must wait for the drain -- "
                         "and the spectrum and delivery readout go dark with "
                         "it. For debugging the fallback path only.")
    ap.add_argument("--audio-out", action="store_true",
                    help="stream continuously to THIS machine's sound card "
                         "instead of the browser player. Removes the segment "
                         "join entirely -- but the sound comes out HERE, so "
                         "over SSH with a share link you will hear nothing.")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    # REFUSE to start beside a still-running conductor rather than letting
    # gradio silently take the next port -- the browser goes back to :7860 out
    # of habit, which is the OLD process, and the two then share gradio_tmp.
    # CLI only: a Space owns its container and its port.
    if not _port_free(args.port):
        others = other_conductors()
        who = "".join(f"\n    pid {p}: {c[:90]}" for p, c in others)
        raise SystemExit(
            f"\nPort {args.port} is already in use"
            + (f", by a conductor that is still running:{who}" if others else ".")
            + "\n\nStarting anyway would put a second instance on another port "
              "while your\nbrowser tab keeps talking to the old one -- and both "
              "would write to the\nsame gradio_tmp. Either stop the old one:\n"
            + (f"\n    kill {' '.join(str(q) for q, _ in others)}\n"
               if others else "\n    (find it with:  ss -lptn 'sport = :%d' )\n"
               % args.port)
            + f"\nor run this one somewhere else:\n\n    --port {args.port + 1}\n")

    # --audio-out OWNS THE AUDIO when it is asked for: the ring player drives
    # this machine's sound card, so also injecting the browser scheduler would
    # schedule every segment twice and leave the delivery badge reporting a
    # graph nobody is listening to. `advance` already prefers _RING at runtime;
    # this stops the two being armed at once now that webaudio is the default.
    _wa = not args.no_webaudio and not args.audio_out
    demo, launch_kw = make_app(
        webaudio_on=_wa,
        label_space=args.label_space, audio_format=args.audio_format,
        min_comparisons=args.min_comparisons, audio_out=args.audio_out,
        share=not args.no_share, port=args.port)
    demo.launch(**launch_kw)


if __name__ == "__main__":
    main()


# --- Phase 2 design note (deferred, not implemented) ------------------------
# True real-time streaming synthesis needs:
#   1. An incremental/blockwise render path in ArrangedRenderer -- today's
#      render_waypoints builds a whole numpy buffer up front; streaming needs
#      a generator yielding fixed-size blocks (e.g. 20ms) while curr_va/
#      target_va/the arc policy can still change between blocks.
#   2. A persistent audio-output process (sounddevice OutputStream callback
#      is the default assumption -- lower risk than the alternative).
#   3. Local-vs-remote delivery is UNDECIDED (2026-07-07, feedback: decide
#      later). If remote/web delivery is wanted, this becomes a materially
#      harder problem (chunked HTTP audio streaming, latency/buffering) --
#      Gradio has no first-class support for this today. Flag before
#      starting Phase 2 work, don't assume.
# This is new infrastructure the original 6-week plan never budgeted --
# treat it as its own scoped follow-up once Phase 1 has been shown to the
# professors and validated, not as "more conductor integration work."
