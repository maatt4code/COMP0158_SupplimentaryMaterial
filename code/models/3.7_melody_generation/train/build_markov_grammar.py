"""Fit a melodic grammar from a notated corpus.

Step 1 of Section 3.7, and what produces `../weights/markov_order2.json`. The
sampling side lives in `../inference/grammar.py`; this is the fitting side, and
it is the only part that needs music21 or a corpus.

WHY LEARN FROM NOTATION. An authored generator produced "a bag of notes", with
two defects and only one of them about pitch: every duration sat in a narrow
band, so nothing read as rhythm -- there was no metrical grid, just jitter
around one value -- and a first-order random walk never restates anything, so
there is no idea to recognise. Better interval statistics alone would have
fixed neither. Rhythm is the thing to learn, and rhythm is SYMBOLIC: it cannot
be recovered from ambient audio, whose whole aesthetic is the absence of clear
onsets. So the grammar is learned from notated melody.

WHY A FOLKSONG CORPUS. It is monophonic by construction, so there is no
melody-extraction step and no skyline heuristic to defend, and it is notated, so
durations arrive as real note values -- a genuine metrical contrast inside a
single tune rather than a narrow band.

WHAT IS LEARNED: the local grammar -- which interval tends to follow which, and
crucially which DURATION goes with it, as an order-2 chain over JOINT
(interval, duration) symbols. Joint matters, because in notated melody pitch and
rhythm are coupled and learning them separately then multiplying gives back a
bag of notes. Also learned: how many notes a phrase runs before it breathes, and
the phrase contours themselves.

WHAT IS NOT LEARNED, deliberately: the valence/arousal mapping. Folk songs carry
no affect labels, so nothing here claims an emotion. VA remains an authored,
declared convention setting tempo, register, mode and phrase length. Folk songs
are also not ambient; what transfers is the GRAMMAR of how intervals and
durations go together, replayed several times slower in a different register and
timbre.

Two design choices worth stating:

  * Pitch becomes a DIATONIC SCALE DEGREE from the letter name, not the
    semitone. Folk sources are full of chromatic inflections, and letter-name
    degrees fold those onto the same rung instead of dropping the note or
    inventing a chromatic step the synthesiser cannot play.
  * Durations are normalised by each phrase's own modal length, so tempo does
    not leak into the grammar and the caller reapplies its own pace.

Both the chain and the phrase shapes are kept, because they answer different
questions -- see `../inference/grammar.py`.

Run:
  python build_markov_grammar.py                    # the whole corpus via music21
  python build_markov_grammar.py --limit-tunes 20   # a quick subset
  python build_markov_grammar.py --essen-root <dir> # a local corpus instead
  python build_markov_grammar.py --selftest         # synthetic corpus, no data

Next: ../inference/grammar.py loads what this writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_SECTION = _HERE.parents[1]
sys.path.insert(0, str(_SECTION / "inference"))
sys.path.insert(0, str(_HERE.parents[3] / "common"))
import paths                                              # noqa: E402
from grammar import (DUR_LADDER, MAX_INTERVAL, ORDER,     # noqa: E402
                     snap, sym_key, EssenGrammar, PhraseBank)

DEFAULT_OUT = _SECTION / "data" / "markov_order2.json"
REST = None               # a phrase boundary inside a tune


def tune_to_events(score):
    """One tune -> ([(degree, quarterLength)], mode), with REST marking a break."""
    from music21 import note

    key = score.analyze("key")
    tonic_step = key.tonic.diatonicNoteNum           # letter position, octave-aware
    out = []
    for e in score.flatten().notesAndRests:
        ql = float(e.duration.quarterLength)
        if ql <= 0:
            continue
        if isinstance(e, note.Note):
            out.append((int(e.pitch.diatonicNoteNum - tonic_step), ql))
        else:
            out.append((REST, ql))
    return out, key.mode


def to_phrases(events):
    """Split a tune at rests. Rests are where a melody breathes, and phrase
    length is one of the things worth learning."""
    phrases, cur = [], []
    for deg, ql in events:
        if deg is REST:
            if len(cur) >= 2:
                phrases.append(cur)
            cur = []
        else:
            cur.append((deg, ql))
    if len(cur) >= 2:
        phrases.append(cur)
    return phrases


def phrase_to_symbols(phrase):
    """A phrase -> [(interval, duration_ratio)] symbols, tempo removed."""
    quals = [ql for _, ql in phrase]
    base = Counter(quals).most_common(1)[0][0] or 1.0
    syms = []
    for k in range(1, len(phrase)):
        iv = int(np.clip(phrase[k][0] - phrase[k - 1][0],
                         -MAX_INTERVAL, MAX_INTERVAL))
        syms.append((iv, snap(phrase[k][1] / base)))
    return syms


def phrase_shape(phrase):
    """A phrase as (degree relative to its FIRST note, duration ratio).

    Relative degrees make transposition free, and normalising durations by the
    phrase's own modal length makes tempo free, so one stored shape replays in
    any register at any pace without touching its contour.
    """
    quals = [ql for _, ql in phrase]
    base = Counter(quals).most_common(1)[0][0] or 1.0
    d0 = phrase[0][0]
    return tuple((int(np.clip(d - d0, -MAX_INTERVAL * 2, MAX_INTERVAL * 2)),
                  snap(ql / base)) for d, ql in phrase)


def fit(tunes, order=ORDER, keep_phrases=40000):
    """Count order-N transitions over joint symbols, and keep the shapes."""
    trans = defaultdict(Counter)
    starts, lens, modes, shapes = Counter(), Counter(), Counter(), Counter()
    n_phrases = 0
    for events, mode in tunes:
        modes[mode] += 1
        for phrase in to_phrases(events):
            syms = phrase_to_symbols(phrase)
            if len(syms) < order + 1:
                continue
            n_phrases += 1
            lens[len(phrase)] += 1
            starts[syms[0]] += 1
            if len(shapes) < keep_phrases or phrase_shape(phrase) in shapes:
                shapes[phrase_shape(phrase)] += 1        # dedup by counting
            for k in range(order, len(syms)):
                trans[tuple(syms[k - order:k])][syms[k]] += 1
    return dict(
        order=order, n_tunes=len(tunes), n_phrases=n_phrases,
        modes=dict(modes),
        starts={sym_key(s): c for s, c in starts.items()},
        phrase_len={str(k): v for k, v in sorted(lens.items())},
        trans={"|".join(sym_key(s) for s in st): {sym_key(y): c for y, c in ys.items()}
               for st, ys in trans.items()},
        shapes={";".join(sym_key(s) for s in sh): c for sh, c in shapes.items()},
    )


def load_corpus(essen_root=None, limit_tunes=None, verbose=True):
    """Parse the corpus into (events, mode) pairs.

    Without ``essen_root`` the collection bundled with music21 is used, which
    needs no download. With it, every parseable file under that directory is
    read instead, so any notated monophonic corpus works.
    """
    from music21 import converter, corpus

    if essen_root:
        root = Path(essen_root)
        if not root.is_dir():
            raise SystemExit(f"\nnot a directory:\n    {root}\n")
        items = sorted(p for p in root.rglob("*")
                       if p.suffix.lower() in (".krn", ".abc", ".mid", ".midi",
                                               ".xml", ".musicxml", ".mxl"))
        if not items:
            raise SystemExit(f"\nno parseable scores under:\n    {root}\n")
        parse = converter.parse
    else:
        items = sorted(corpus.getComposer("essenFolksong"))
        if not items:
            raise SystemExit(
                "\nmusic21 reports no bundled folksong corpus.\n"
                "Install the corpus, or pass --essen-root with your own scores.\n")
        parse = corpus.parse

    if limit_tunes:
        items = items[:limit_tunes]
    tunes = []
    for i, p in enumerate(items):
        try:
            op = parse(p if not essen_root else str(p))
        except Exception as e:                    # a few sources are malformed
            if verbose:
                print(f"  skip {Path(str(p)).name}: {e}")
            continue
        for s in (getattr(op, "scores", None) or [op]):
            try:
                ev, mode = tune_to_events(s)
            except Exception:
                continue
            if len(ev) >= 8:
                tunes.append((ev, mode))
        if verbose:
            print(f"  [{i+1}/{len(items)}] {Path(str(p)).name:24} tunes={len(tunes)}")
    return tunes


def build(essen_root=None, limit_tunes=None, out=None, verbose=True):
    tunes = load_corpus(essen_root, limit_tunes, verbose)
    if not tunes:
        raise SystemExit("\nno tunes parsed; nothing to fit\n")
    model = fit(tunes)
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(model))
    print(f"\n{model['n_tunes']} tunes -> {model['n_phrases']} phrases, "
          f"{len(model['trans'])} chain states, {len(model['shapes'])} shapes")
    print(f"modes: {model['modes']}")
    if out:
        print(f"wrote {out}")
    return model


def selftest():
    """A tiny hand-made corpus with an unmistakable rhythm: short notes then one
    long one, rising to a peak and coming back. If the fit cannot reproduce
    that, it is not learning duration at all."""
    # (degree, quarterLength); the long note is 3x the short one.
    phrase = [(0, 1.0), (1, 1.0), (2, 1.0), (3, 3.0), (2, 1.0), (1, 1.0), (0, 3.0)]
    tunes = [(phrase + [(REST, 1.0)] + phrase, "major") for _ in range(40)]

    model = fit(tunes)
    assert model["n_tunes"] == 40 and model["n_phrases"] == 80, model["n_phrases"]
    print(f"  PASS  fit counts {model['n_phrases']} phrases from "
          f"{model['n_tunes']} tunes")

    durs = {d for st in model["trans"] for _, d in
            [tuple(x.split(",")) for x in st.split("|")]}
    assert len(durs) > 1, "duration must vary in the learned symbols"
    print(f"  PASS  the grammar carries more than one duration class: "
          f"{sorted(durs)}")

    g = EssenGrammar(model)
    rng = np.random.default_rng(0)
    seq = [g.start(rng)]
    for _ in range(30):
        seq.append(g.next(rng, seq))
    ratios = [d for _, d in seq]
    assert max(ratios) / min(ratios) >= 2.5, \
        f"sampled rhythm is flat: {min(ratios)}..{max(ratios)}"
    print(f"  PASS  sampled rhythm keeps a {max(ratios)/min(ratios):.0f}:1 "
          f"long/short contrast")

    # Backoff must never dead-end, even from a state the corpus never saw.
    unseen = [(MAX_INTERVAL, 4.0), (-MAX_INTERVAL, 0.25)]
    assert g.next(rng, unseen) is not None
    print("  PASS  an unseen state backs off instead of dead-ending")

    bank = PhraseBank(model)
    assert len(bank) >= 1
    shape = bank.sample(rng, target_len=len(phrase))
    assert len(shape) >= 3
    varied, kind = bank.vary(shape, rng)
    assert kind in ("repeat", "invert", "truncate", "stretch", "tail")
    print(f"  PASS  the phrase bank samples and restates ({kind})")

    # A stored shape must be transposition- and tempo-free: it starts at 0.
    assert shape[0][0] == 0, "a shape must be relative to its own first note"
    print("  PASS  shapes are relative to their first note, so transposing is free")
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser(
        description="Fit a melodic grammar from a notated corpus.")
    paths.add_arg(ap, "essen")
    ap.add_argument("--limit-tunes", dest="limit_tunes", type=int, default=None,
                    help="only the first N corpus files (a quick subset)")
    ap.add_argument("--out", default=None, help=f"default {DEFAULT_OUT}")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    # The bundled music21 corpus is the default route and needs no download, so
    # a missing local root is not an error here -- unlike every other dataset.
    root = args.essen_root
    if root and not Path(root).is_dir():
        raise SystemExit(
            f"\nessen corpus not found at:\n    {root}\n\n"
            f"Omit --essen-root to use the copy bundled with music21, or\n"
            f"obtain it from: {paths.WHERE_TO_GET['essen']}\n")
    build(root, args.limit_tunes, args.out or DEFAULT_OUT)


if __name__ == "__main__":
    main()
