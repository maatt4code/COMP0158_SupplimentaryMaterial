"""Sampling from the fitted melodic grammar. Loading only, no corpus, no fitting.

Runtime code. `../train/build_markov_grammar.py` fits the grammar from a
notated corpus and writes `../weights/markov_order2.json`; this module loads it
and samples from it. The split matters: the runtime must never need music21 or
a corpus on disk.

Two sampling strategies live here, and they answer different questions.

`EssenGrammar` is the order-2 Markov chain over JOINT (interval, duration)
symbols. Joint is the point: in notated melody pitch and rhythm are coupled --
long notes at phrase ends, short notes in runs -- and learning them separately
then multiplying gives back a bag of notes with better statistics. It backs off
to order 1 and then to the marginal, so it can never dead-end.

`PhraseBank` replays whole contours a human actually wrote. It exists because
an order-2 chain reproduces the AVERAGE of its corpus: it knows what tends to
follow the last two symbols and nothing else, so it cannot know that a phrase
has a shape or is going anywhere. Averaging tens of thousands of phrases yields
generic material however good the corpus is. Sampling whole shapes sidesteps
that -- the contour is never regenerated, so it survives intact, while the
caller still chooses WHICH shape and how it is transposed, stretched and
restated. The corpus supplies contour; theory and the VA mapping keep the
guardrails.

What is NOT learned, deliberately: the valence/arousal mapping. Folk songs
carry no affect labels, so nothing here claims an emotion. VA continues to set
tempo, register, mode and phrase length as declared conventions. What transfers
from the corpus is the GRAMMAR of how intervals and durations go together,
replayed slower in a different register and timbre.

Durations are RATIOS, not seconds. Each tune was normalised by its own most
common note length during fitting, so a fast tune and a slow one contribute the
same rhythmic shape, and the caller's own pace stays in charge.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
DEFAULT_GRAMMAR = WEIGHTS / "markov_order2.json"

# Duration ladder, as multiples of a tune's own modal note length. It covers
# what notation actually uses -- halves, dotted values, doubles and the long
# phrase-final note -- without a class so rare it could never be estimated.
DUR_LADDER = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
MAX_INTERVAL = 7          # scale degrees; wider leaps fold to the edge
ORDER = 2


def snap(x, ladder=DUR_LADDER):
    return min(ladder, key=lambda c: abs(c - x))


def sym_key(sym):
    """A (interval, duration) symbol as its JSON key."""
    return f"{sym[0]},{sym[1]}"


def sym_parse(s):
    a, b = s.split(",")
    return int(a), float(b)


def is_arch(shape):
    """True if the contour rises to a single peak and comes back down.

    One melodic peak per phrase is the traditional mark of a shapely line, and
    it is the cheapest available proxy for "this phrase goes somewhere" --
    exactly what an averaged Markov chain cannot know.
    """
    degs = [d for d, _ in shape]
    if len(degs) < 3:
        return False
    peak = max(range(len(degs)), key=lambda k: degs[k])
    if peak in (0, len(degs) - 1):
        return False
    up = all(b >= a for a, b in zip(degs[:peak], degs[1:peak + 1]))
    down = all(b <= a for a, b in zip(degs[peak:], degs[peak + 1:]))
    return up and down


class EssenGrammar:
    """The fitted chain, with a backoff so sampling never dead-ends."""

    def __init__(self, model):
        self.order = int(model["order"])
        self.n_tunes = int(model.get("n_tunes", 0))
        self.n_phrases = int(model.get("n_phrases", 0))
        self.modes = dict(model.get("modes", {}))
        self.trans = {tuple(sym_parse(p) for p in st.split("|")):
                      ([sym_parse(y) for y in ys],
                       np.array(list(ys.values()), float))
                      for st, ys in model["trans"].items()}
        self.starts = ([sym_parse(y) for y in model["starts"]],
                       np.array(list(model["starts"].values()), float))
        pl = model["phrase_len"]
        self.phrase_len = (np.array([int(k) for k in pl], int),
                           np.array(list(pl.values()), float))
        # Order-1 and order-0 backoffs, folded from the same counts.
        self.back1 = defaultdict(Counter)
        self.back0 = Counter()
        for st, (ys, ws) in self.trans.items():
            for y, w in zip(ys, ws):
                self.back1[st[-1:]][y] += w
                self.back0[y] += w

    def _pick(self, rng, ys, ws):
        return ys[int(rng.choice(len(ys), p=np.asarray(ws, float) / np.sum(ws)))]

    def start(self, rng):
        return self._pick(rng, *self.starts)

    def next(self, rng, state):
        """Order 2 where the state was seen, else order 1, else the marginal."""
        st = tuple(state[-self.order:])
        if st in self.trans:
            return self._pick(rng, *self.trans[st])
        b1 = self.back1.get(st[-1:])
        if b1:
            return self._pick(rng, list(b1), list(b1.values()))
        return self._pick(rng, list(self.back0), list(self.back0.values()))

    def phrase_length(self, rng, target=None):
        """Sample a phrase length. ``target`` biases toward the caller's own
        length, so the authored arousal mapping still governs density."""
        lens, ws = self.phrase_len
        if target is not None:
            ws = ws * np.exp(-0.5 * ((lens - target) / 2.0) ** 2)
            if ws.sum() <= 0:
                ws = self.phrase_len[1]
        return int(self._pick(rng, list(lens), list(ws)))


class PhraseBank:
    """Whole contours from the corpus, sampled and restated rather than
    regenerated."""

    def __init__(self, model):
        self.shapes, counts, lens, spans, arch = [], [], [], [], []
        for s, c in model.get("shapes", {}).items():
            sh = tuple(sym_parse(x) for x in s.split(";"))
            if len(sh) < 3:
                continue
            degs = [d for d, _ in sh]
            self.shapes.append(sh)
            counts.append(float(c))
            lens.append(len(sh))
            spans.append(max(degs) - min(degs))
            arch.append(is_arch(sh))
        self.counts = np.array(counts, float)
        self.lens = np.array(lens, int)
        self.spans = np.array(spans, int)
        self.arch = np.array(arch, bool)

    def __len__(self):
        return len(self.shapes)

    def sample(self, rng, target_len=8, max_span=8, arch_bias=3.0):
        """Pick a phrase: length near the target, range inside the register,
        arch-shaped contours favoured."""
        ok = self.spans <= max_span
        if not ok.any():
            ok = self.spans <= self.spans.min()
        w = self.counts * ok
        w = w * np.exp(-0.5 * ((self.lens - target_len) / 3.0) ** 2)
        w = w * np.where(self.arch, arch_bias, 1.0)
        if w.sum() <= 0:
            w = ok.astype(float)
        return self.shapes[int(rng.choice(len(self.shapes), p=w / w.sum()))]

    def vary(self, shape, rng):
        """Restate a phrase with ONE change.

        Restatement is what a listener hears as a tune. Changing everything on
        every repeat is why a pure chain never sounds like one.
        """
        kind = str(rng.choice(["repeat", "invert", "truncate", "stretch", "tail"],
                              p=[0.34, 0.18, 0.18, 0.15, 0.15]))
        if kind == "repeat":
            return shape, kind
        if kind == "invert":
            return tuple((-d, q) for d, q in shape), kind
        if kind == "truncate":
            return shape[: max(3, len(shape) - 2)], kind
        if kind == "stretch":
            return tuple((d, q * 1.5) for d, q in shape), kind
        return shape[-max(3, len(shape) // 2):], kind        # just the ending


def _read(path=None):
    p = Path(path) if path else DEFAULT_GRAMMAR
    if not p.exists():
        raise SystemExit(
            f"\nfitted grammar not found at:\n    {p}\n\n"
            "It ships with this section. Refit it with\n"
            "    python ../train/build_markov_grammar.py\n")
    return json.loads(p.read_text())


def load(path=None):
    """The order-2 chain."""
    return EssenGrammar(_read(path))


def load_bank(path=None):
    """The phrase bank."""
    return PhraseBank(_read(path))


def synthetic_model(repeats=40):
    """A hand-made model with an UNMISTAKABLE rhythm, for tests.

    Three short notes then one four-times-longer one, rising to a peak and
    coming back. If a generator driven by this cannot reproduce a 4:1 duration
    contrast, it is not carrying the learned rhythm through to seconds at all.

    Built directly in the fitted format rather than by fitting, so the runtime
    can test itself with no corpus, no music21 and no shipped weight.
    """
    from collections import Counter, defaultdict

    # (degree, quarterLength) -- the same shape the fitter would see.
    phrase = [(0, 1.0), (1, 1.0), (2, 1.0), (3, 4.0),
              (2, 1.0), (1, 1.0), (0, 4.0)]
    base = Counter(ql for _, ql in phrase).most_common(1)[0][0]
    syms = [(phrase[k][0] - phrase[k - 1][0], snap(phrase[k][1] / base))
            for k in range(1, len(phrase))]
    shape = tuple((d - phrase[0][0], snap(ql / base)) for d, ql in phrase)

    trans = defaultdict(Counter)
    for k in range(ORDER, len(syms)):
        trans[tuple(syms[k - ORDER:k])][syms[k]] += repeats
    return dict(
        order=ORDER, n_tunes=repeats, n_phrases=repeats,
        modes={"major": repeats},
        starts={sym_key(syms[0]): repeats},
        phrase_len={str(len(phrase)): repeats},
        trans={"|".join(sym_key(x) for x in st): {sym_key(y): c for y, c in ys.items()}
               for st, ys in trans.items()},
        shapes={";".join(sym_key(x) for x in shape): repeats},
    )
