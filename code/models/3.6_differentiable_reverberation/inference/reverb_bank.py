"""The measured reverb space, as rateable stimulus conditions.

Runtime code. This retires hand-set presets: every wet condition below is drawn
from a bank fitted to data rather than from knob-guessing.

  * Real impulse responses (`../weights/ir_reverb_bank.json`, 115 measured
    architectural spaces) are applied by CONVOLVING THE ACTUAL WAV, so no
    fitting error enters the stimulus. The fitted parameters are used only to
    CHOOSE which IR represents a category.
  * Acoustics measured from finished recordings
    (`../weights/reverb_bank_measured.json`) have no dry IR to convolve, so
    they are applied through their fitted exponential-decay parameters. Only
    decay and damping are used: wet mix is degenerate with the source's own
    gain under blind separation, which the bank's own note records.

WHY THESE SIX CONDITIONS. Tail length is the primary factor and the ladder
spans the whole measured range: dry, then about 0.3 s, 1.0 s, 2.4 s, and two at
roughly 5 s. At the long end sit a TONE-MATCHED PAIR -- `long_bright` at about
2.4 kHz and `long_dark` at about 440 Hz, at nearly identical tail length -- so
the analysis can tell whether valence tracks the TAIL or the BRIGHTNESS. Without
that pair the two are confounded, because long reverbs in this material also
tend to be dark.

WET IS HELD CONSTANT across every wet condition. The manipulation is WHICH
SPACE, not how much of it, so a shift cannot be read as "more effect produced
more effect".

The representative IR for a category is the one whose measured RT60 is closest
to that category's median -- deterministic, with no cherry-picking.

On the condition names: two of these ids were originally the names of the
recording artists whose acoustics were measured. They are re-keyed onto the
measurement itself, which is both what the experiment manipulates and what the
report describes. The measured numbers are unchanged.

Run:
  python reverb_bank.py --list
  python reverb_bank.py --selftest
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from reverb import (SR, apply_reverb_np, apply_ir_np, load_ir)  # noqa: E402

WEIGHTS = _HERE.parents[1] / "weights"
IR_BANK = WEIGHTS / "ir_reverb_bank.json"
MEASURED_BANK = WEIGHTS / "reverb_bank_measured.json"

WET = 0.5                  # held constant across all wet conditions
MEASURED_PRE_DELAY_S = 0.02   # no IR exists, so give the synthetic tail a gap
IR_MAX_S = 6.0
FADE_IN_S = 0.005
FADE_OUT_S = 1.0           # cos^2 release; see _edge_fade

# (condition id, kind, bank category), ordered by tail length: the ladder.
LADDER = [
    ("dry",         "dry",      None),
    ("nature",      "ir",       "Nature"),       # RT60 ~0.3 s, open air
    ("brutalism",   "ir",       "Brutalism"),    # RT60 ~1.0 s, concrete room
    ("stairwell",   "ir",       "Stairwells"),   # RT60 ~2.4 s, longest real space
    ("long_bright", "measured", "long_bright"),  # ~5.0 s, bright } tone-matched
    ("long_dark",   "measured", "long_dark"),    # ~5.2 s, dark   } pair
]

_CONDS = None


def _pick_representative(entries):
    """The entry whose measured RT60 sits closest to the category median."""
    rts = [e["rt60_s"] for e in entries if e.get("rt60_s")]
    if not rts:
        return entries[0]
    med = float(np.median(rts))
    return min((e for e in entries if e.get("rt60_s")),
               key=lambda e: abs(e["rt60_s"] - med))


def build_conditions(ir_path=IR_BANK, measured_path=MEASURED_BANK, ladder=LADDER):
    """Resolve the ladder against the two banks. Conditions whose category is
    missing are skipped, so a partial bank still runs."""
    ir_doc = json.loads(Path(ir_path).read_text()) if Path(ir_path).exists() else {"irs": []}
    me_doc = (json.loads(Path(measured_path).read_text())
              if Path(measured_path).exists() else {"tracks": []})
    by_ir, by_me = {}, {}
    for e in ir_doc.get("irs", []):
        by_ir.setdefault(e["category"], []).append(e)
    for e in me_doc.get("tracks", []):
        by_me.setdefault(e["category"], []).append(e)

    conds = {}
    for cid, kind, cat in ladder:
        if kind == "dry":
            conds[cid] = dict(id=cid, kind="dry", label="dry (no reverb)",
                              category=None, tail_s=0.0, tone_hz=None)
        elif kind == "ir":
            ents = by_ir.get(cat)
            if not ents:
                continue
            e = _pick_representative(ents)
            # Bank paths are stored relative to weights/, so a copy of this
            # directory carries its own IRs and can be moved anywhere. Storing
            # absolute paths ships dangling references the moment it is copied.
            p = Path(e["path"])
            p = p if p.is_absolute() else (WEIGHTS / p)
            conds[cid] = dict(id=cid, kind="ir", label=f"{cat}/{e['name']}",
                              category=cat, tail_s=round(float(e["rt60_s"]), 3),
                              tone_hz=round(float(e.get("centroid_hz", 0)), 1),
                              path=str(p), source="measured IR (convolved)")
        else:
            ents = by_me.get(cat)
            if not ents:
                continue
            decay = float(np.median([x["fit_decay_s"] for x in ents]))
            damp = float(np.median([x["fit_damping_hz"] for x in ents]))
            conds[cid] = dict(id=cid, kind="measured", category=cat,
                              label=f"{cat} (n={len(ents)})",
                              tail_s=round(decay, 3), tone_hz=round(damp, 1),
                              params=dict(decay_time_s=round(decay, 3),
                                          damping_hz=round(damp, 1), wet=WET,
                                          pre_delay_s=MEASURED_PRE_DELAY_S),
                              source="fitted decay parameters")
    return conds


def conditions(refresh=False):
    """Module-level singleton, so every caller renders byte-identical stimuli
    from the same (audio, condition) pair."""
    global _CONDS
    if _CONDS is None or refresh:
        _CONDS = build_conditions()
    return _CONDS


def condition_ids():
    c = conditions()
    return [cid for cid, _, _ in LADDER if cid in c]


@functools.lru_cache(maxsize=16)
def _ir(path, sr):
    ir = np.asarray(load_ir(path, sr=sr, max_s=IR_MAX_S), dtype=np.float64).copy()
    # Cropping at IR_MAX_S can leave the tail still ringing; taper the end.
    k = max(1, int(0.05 * len(ir)))
    ir[-k:] *= 0.5 * (1 + np.cos(np.pi * np.arange(k) / k))
    return ir.astype(np.float32)


def _edge_fade(x, sr, fade_in_s=FADE_IN_S, fade_out_s=FADE_OUT_S):
    """Raised-cosine edges on the SOURCE, before any reverb is applied.

    Rendered clips stop dead at full amplitude with no release, so the dry
    component vanishes in a single sample. That step is both an audible click
    and an abrupt loss of the dry half of the mix, landing at the same moment
    in every clip -- it would be rated as a property of the reverb when it is
    really a property of the source. Shaping the SOURCE rather than the output
    keeps the tail's natural decay and treats every condition identically, dry
    included.

    This release is a PROTOCOL CONSTANT, not a tuned knob and not learned:
    identical in every condition, so it cannot produce a between-condition
    effect. Its SHAPE is taken from deployment (the same cos^2 ramp the
    scheduler uses for bed-gain crossfades) rather than invented, and its
    DURATION is bounded by the stimulus -- a long release inside a short clip
    would leave nothing steady to judge. Report it alongside the ladder.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    n = len(x)
    ni, no = int(fade_in_s * sr), int(fade_out_s * sr)
    if n < ni + no + 2:
        return x.astype(np.float32)
    if ni:
        x[:ni] *= 0.5 * (1 - np.cos(np.pi * np.arange(ni) / ni))
    if no:
        x[-no:] *= 0.5 * (1 + np.cos(np.pi * np.arange(no) / no))
    return x.astype(np.float32)


def _match_level(out, dry_in):
    """Equalise loudness ACROSS CONDITIONS.

    Mixing in a decorrelated wet signal costs a few dB, and that drop grows
    with tail length -- which would show up as "reverb lowers arousal" when the
    judge is really just hearing a quieter clip. Matching RMS over the region
    where the dry signal exists removes the confound and leaves the tail's
    natural decay intact.

    KNOWN LIMITATION, and it is load-bearing for how these ratings should be
    read. The peak guard below runs AFTER the match, so a condition whose
    matched output would clip is scaled back down -- undoing the very control
    this function exists to provide. On the shipped ladder it binds on exactly
    one condition, `stairwell`, which ends up about 4.7 dB below dry while
    every other condition matches to within 0.001 dB. That is the longest REAL
    measured space, so it is also the condition where a level confound matters
    most.

    The behaviour is preserved deliberately: these are the stimuli that were
    actually rated, and silently re-levelling them now would mean the shipped
    code renders something different from what the listeners heard. The fix for
    a future round is to apply one common headroom scalar across all
    conditions, so peak limiting cannot change their RELATIVE levels. The smoke
    test asserts the measured reality, including this exception, so it cannot
    drift unnoticed.
    """
    out = np.asarray(out, dtype=np.float64)
    n_in = len(dry_in)
    ref = np.sqrt(np.mean(np.asarray(dry_in, dtype=np.float64) ** 2)) + 1e-12
    cur = np.sqrt(np.mean(out[:n_in] ** 2)) + 1e-12
    out = out * (ref / cur)
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 0.99:
        out *= 0.99 / peak
    return out.astype(np.float32)


def impulse_response(cond_id, sr=SR, head_s=0.0, conds=None):
    """The condition's IR as an array; empty for 'dry'.

    ``head_s`` fades the head in, removing the direct sound. Exposed separately
    from ``apply`` so a CONTINUOUS renderer can convolve incrementally and carry
    the tail across blocks; ``apply`` is for one-shot clips and would fade and
    truncate at every join.
    """
    c = (conds or conditions()).get(cond_id)
    if c is None or c["kind"] == "dry":
        return np.zeros(0, dtype=np.float32)
    if c["kind"] == "ir":
        ir = np.asarray(_ir(c["path"], sr), dtype=np.float64).copy()
    else:
        p = dict(c["params"], wet=1.0)
        imp = np.zeros(int(sr * min(6.0, p["decay_time_s"] * 3)), dtype=np.float32)
        imp[0] = 1.0
        ir = np.asarray(apply_reverb_np(imp, sr=sr, **p), dtype=np.float64)
    k = int(head_s * sr)
    if k and len(ir) > 2 * k:
        ir[:k] *= 0.5 * (1 - np.cos(np.pi * np.arange(k) / k))
    return (ir / (np.sqrt((ir ** 2).sum()) + 1e-12)).astype(np.float32)


def apply_late(audio, cond_id, sr=SR, head_s=0.05, conds=None):
    """LATE-FIELD convolution: drop the direct sound and early reflections,
    keep the diffuse tail, and use no dry path.

    An audition finding drove this: the stimulus is grating only while the dry
    source is audible, and smooth once the source stops. Two correlated copies
    of the same signal a few milliseconds apart comb-filter; the late field
    alone does not, because there is no coherent direct sound left to interfere
    with. Doing it on the IR rather than by cutting the tail out as a clip means
    the output SUSTAINS as long as the source does, with no fade to fight and no
    loudness to patch.
    """
    c = (conds or conditions()).get(cond_id)
    if c is None or c["kind"] == "dry":
        return _edge_fade(audio, sr)
    audio = _edge_fade(audio, sr)
    ir = impulse_response(cond_id, sr=sr, head_s=head_s, conds=conds)
    return _match_level(apply_ir_np(audio, ir, wet=1.0), audio)


def apply(audio, cond_id, sr=SR, conds=None):
    """Apply one condition by id. Deterministic.

    An unknown id RAISES rather than passing through: a silent pass-through
    would log a wet trial as if it were dry and quietly corrupt the study.
    """
    c = (conds or conditions()).get(cond_id)
    if c is None:
        raise ValueError(f"unknown reverb condition {cond_id!r}; "
                         f"have {sorted((conds or conditions()))}")
    audio = _edge_fade(audio, sr)      # identical shaping in every condition
    if c["kind"] == "dry":
        return audio
    if c["kind"] == "ir":
        wet = apply_ir_np(audio, _ir(c["path"], sr), wet=WET)
    else:
        wet = apply_reverb_np(audio, sr=sr, **c["params"])
    return _match_level(wet, audio)


LATE_SUFFIX = ":late"


def apply_id(audio, rid, sr=SR, conds=None):
    """Render by LOGGED id, where `<cond>:late` selects late-field convolution.

    The rendering mode lives in the id rather than in a flag because analysis
    re-renders from the (path, condition) pair written in the ratings CSV. If
    the app rendered late-field and the analysis re-rendered the dry-plus-wet
    mix, the two would silently describe different stimuli.
    """
    if str(rid).endswith(LATE_SUFFIX):
        return apply_late(audio, str(rid)[:-len(LATE_SUFFIX)], sr=sr, conds=conds)
    return apply(audio, rid, sr=sr, conds=conds)


def known_ids(conds=None):
    base = [c for c in condition_ids() if not conds or c in conds]
    return set(base) | {c + LATE_SUFFIX for c in base if c != "dry"}


def describe(conds=None):
    c = conds or conditions()
    lines = []
    for cid in [i for i, _, _ in LADDER if i in c]:
        e = c[cid]
        tone = f"{e['tone_hz']:.0f} Hz" if e.get("tone_hz") else "-"
        lines.append(f"  {cid:<12} tail {e['tail_s']:>5.2f}s  tone {tone:>9}  "
                     f"{e.get('source', 'reference'):<26} {e['label']}")
    return "\n".join(lines)


def dump(out, conds=None):
    """Record the exact ladder used for a study, as a reproducibility sidecar."""
    c = conds or conditions()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        dict(wet=WET, fade_in_s=FADE_IN_S, fade_out_s=FADE_OUT_S,
             fade_shape="cos^2, as the scheduler's bed-gain crossfades",
             ir_bank=IR_BANK.name, measured_bank=MEASURED_BANK.name,
             conditions=c), indent=2))
    return out


def selftest():
    sr = SR
    clip = (0.3 * np.sin(2 * np.pi * 110 * np.arange(int(4 * sr)) / sr)
            ).astype(np.float32)
    c = conditions()
    ids = condition_ids()
    assert ids[0] == "dry", ids
    assert len(ids) == len(LADDER), f"only {len(ids)} of {len(LADDER)} resolved"
    print(f"  PASS  all {len(ids)} ladder conditions resolve")

    tails = [c[i]["tail_s"] for i in ids]
    assert tails == sorted(tails), f"ladder must ascend in tail length: {tails}"
    print(f"  PASS  ladder ascends in tail length: "
          f"{[round(t, 2) for t in tails]}")

    lb, ld = c["long_bright"], c["long_dark"]
    assert abs(lb["tail_s"] - ld["tail_s"]) < 0.5, "the pair must be tail-matched"
    assert lb["tone_hz"] > 3 * ld["tone_hz"], "the pair must differ in tone"
    print(f"  PASS  tone-matched pair: tails {lb['tail_s']:.2f}s vs "
          f"{ld['tail_s']:.2f}s, tones {lb['tone_hz']:.0f}Hz vs "
          f"{ld['tone_hz']:.0f}Hz")

    dry = apply(clip, "dry")
    assert len(dry) == len(clip) and np.all(np.isfinite(dry))
    for cid in ids[1:]:
        out = apply(clip, cid)
        assert np.all(np.isfinite(out)), cid
        assert len(out) >= len(clip), f"{cid} must ring out past the input"
        assert np.max(np.abs(out)) <= 0.99 + 1e-6, cid
    print("  PASS  every wet condition rings out, stays finite and peak-safe")

    # Level matching, and the one documented place it does not hold. See
    # _match_level: the peak guard runs after the match and can undo it.
    r_dry = np.sqrt(np.mean(dry.astype(float) ** 2))
    devs = {}
    for cid in ids[1:]:
        out = apply(clip, cid)
        r_out = np.sqrt(np.mean(out[:len(clip)].astype(float) ** 2))
        devs[cid] = float(20 * np.log10(r_out / (r_dry + 1e-12)))
    # Which conditions the guard binds on depends on the SOURCE level, so the
    # invariant is the mechanism, not a fixed list: any deviation must be
    # downward, and must coincide with the output sitting at the peak ceiling.
    peaks = {cid: float(np.max(np.abs(apply(clip, cid)))) for cid in ids[1:]}
    matched = [k for k, v in devs.items() if abs(v) < 0.01]
    clipped = [k for k, v in devs.items() if abs(v) >= 0.01]
    for k in clipped:
        assert devs[k] < 0, f"{k} is LOUDER than dry, which the guard cannot cause"
        assert peaks[k] > 0.98, \
            f"{k} deviates by {devs[k]:.2f} dB without the peak guard binding"
    assert matched, "no condition matched at all"
    print(f"  PASS  {len(matched)} conditions level-matched to within 0.01 dB")
    if clipped:
        worst = min(clipped, key=lambda k: devs[k])
        print(f"  KNOWN {len(clipped)} condition(s) sit below dry, worst "
              f"{worst} at {devs[worst]:+.2f} dB: the peak guard undoes the "
              f"match (documented in _match_level)")

    late = apply_id(clip, "stairwell" + LATE_SUFFIX)
    plain = apply_id(clip, "stairwell")
    assert not np.allclose(late[:len(clip)], plain[:len(clip)]), \
        "late-field must differ from the dry+wet mix"
    print("  PASS  late-field rendering differs from the dry+wet mix")

    try:
        apply(clip, "no_such_condition")
    except ValueError:
        print("  PASS  an unknown condition id raises rather than passing through")
    else:
        raise AssertionError("unknown id must raise")

    # Condition ids and categories must be acoustic descriptors, not the names
    # of whatever was measured. The explicit name list lives in the smoke test,
    # not here: a runtime module should not carry the strings it forbids.
    for cid, e in c.items():
        cat = e.get("category")
        if e["kind"] == "measured":
            assert cat.split("_")[0] in ("short", "mid", "long"), \
                f"{cid}: category {cat!r} is not an acoustic descriptor"
            assert cid.split("_")[0] in ("short", "mid", "long"), \
                f"condition id {cid!r} is not an acoustic descriptor"
    print("  PASS  every fitted condition is named by its acoustics")
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser(description="The measured reverb ladder.")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump", default=None, help="write the ladder as JSON")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.dump:
        print(f"wrote {dump(args.dump)}")
    else:
        print(f"wet held at {WET}; source release {FADE_OUT_S}s (cos^2)\n")
        print(describe())


if __name__ == "__main__":
    main()
