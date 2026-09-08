"""Semi-Markov arrangement scheduler: WHEN the arrangement changes.

Runtime code. It reads only `../weights/hsmm_transitions.json` and needs no
dataset, no corpus and no rating data.

Why this exists. The texture-round comments showed that what broke coherence
was PACING, not parameters -- "the transition around 18s comes too soon after
the one from 12s", "transition too short, undecided about which direction to
go". A first-order Markov chain has geometric, memoryless dwell times and will
happily re-switch seconds after a switch, which is exactly that failure. So
each state carries an explicit dwell WINDOW instead: a semi-Markov process.

Why uniform dwell windows rather than a fitted shape: the bounded support IS
the point. The upper bound is the anti-deadness guarantee and the lower bound
the anti-overload one. Within those bounds the distribution's shape is
statistically unidentifiable from the handful of segments a listener actually
hears per track, and a lognormal either loses the upper bound -- unbounded
dwells being the deadness the bound exists to prevent -- or gets truncated,
recreating the same hard support edges it was meant to remove. Two
interpretable numbers win.

Naming honesty: this is NOT a hidden Markov model. Nothing is hidden and no
emissions are decoded. It would become one only if the states were later
INFERRED from ratings, which is stated future work alongside fitting the dwell
and transition parameters the way every other layer here is fitted.

The contract:

  * A state is SchedState(name, layers, dwell_s), where `layers` maps a layer
    name to a directive, e.g. {"bed": "duck"}. The core is layer-AGNOSTIC:
    drone-only, soundscape overlay and melodic arrangements are just different
    state sets. Renderers implement the layers they know and must WARN rather
    than crash on the rest, so a cut layer keeps its slot.
  * VA is an INPUT, never an output. The optional gate(context, s_from, s_to)
    hook lets valence and arousal modulate which arrangements are reachable,
    but the scheduler never moves the VA target -- doing so would invert the
    closed loop this whole system is built around.
  * The transition matrix has a ZERO diagonal. Staying put is the dwell's job.
  * Gate weights are hard constraints, so an all-veto means "the current
    arrangement is the only appropriate one under this context" and the state
    simply dwells again, merged into one segment. The scheduler warns after
    three consecutive all-vetoes so a misconfigured gate is visible; a gate
    should leave at least one exit reachable in every context.

Corpus-fitted timing. `--trans-source` selects where the overlay preset's dwell
windows and transition matrix come from. The corpus fit shipped in
`../weights/hsmm_transitions.json` was estimated on a commercial ambient corpus
(1027 state transitions, 95 tracks, 14 artists); the script that produced it
does not ship, because the corpus does not. Read carefully before citing it:
the per-state dwell windows and the active-to-settled return rate are
corpus-FIT, the duck-versus-bursts split is AUTHORED because the corpus has no
correlate for it, and the whole thing is a compositional prior, never a
listener-preference model.

  handset       hand-set numbers. The DEFAULT, never silently replaced.
  corpus_dwell  corpus-fit per-state dwell, hand-set transition matrix.
                RECOMMENDED: the corpus grounds TIMING, but it has no
                bed/no-bed distinction with which to ground ARRANGEMENT.
  corpus        corpus-fit dwell AND transition matrix.

The dwell floor is a listener-evidence DESIGN CONSTRAINT, not a data finding.
The corpus's active-regime holds start around 2 s, which would reintroduce
precisely the "re-switches too soon" failure this scheduler was built to fix,
so fitted windows are raised to the floor and both numbers are reported
separately so the two never blur together.

Run:
  python scheduler.py --selftest
  python scheduler.py --preset overlay --trans-source corpus_dwell
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
HSMM = WEIGHTS / "hsmm_transitions.json"

SR = 16000
XFADE_S = 3.0             # cos^2 crossfade at state boundaries, ambient-paced
BREATH_PERIOD_S = 22.0    # deliberately off the dwell grid
DWELL_FLOOR_S = 20.0      # deployment floor from the pacing complaints

BED_STATES = ["drone_solo", "bed_duck", "bed_bursts"]
# How drone_solo's exit mass splits between the two active states. AUTHORED:
# the corpus is silent on duck-versus-bursts, so this stays at the hand-set
# value rather than being invented from audio.
HANDSET_ACTIVE_SPLIT = {"bed_duck": 0.6, "bed_bursts": 0.4}


@dataclass
class SchedState:
    name: str
    layers: dict          # layer name -> directive string
    dwell_s: tuple        # (min_s, max_s) uniform dwell window


def load_corpus(path=HSMM):
    """The corpus semi-Markov fit, or None when it is absent."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def corpus_dwell_by_state(corpus, bed_states=BED_STATES):
    """Per-state (min_s, max_s) windows -- the state-CONDITIONAL durations
    that make this semi-Markov rather than a plain chain.

    drone_solo takes the settled group's window and the bed-on states share
    the active group's; the corpus cannot distinguish duck from bursts.
    """
    d = corpus["dwell_window_s_by_group"]
    return {b: tuple(d["settled" if b == "drone_solo" else "active"])
            for b in bed_states}


def corpus_overlay_trans(corpus, bed_states=BED_STATES,
                         active_split=HANDSET_ACTIVE_SPLIT):
    """3x3 transition matrix, zero diagonal, rows summing to 1.

    Two ingredients, kept distinguishable: the active-to-settled return rate
    is corpus-FIT; how drone_solo's exit mass splits, and which active state
    is "the other one", are AUTHORED.
    """
    p_ret = float(corpus["return_rate_active_to_settled"])
    i_solo = bed_states.index("drone_solo")
    actives = [b for b in bed_states if b != "drone_solo"]
    M = np.zeros((len(bed_states), len(bed_states)))
    tot = sum(active_split.get(b, 0.0) for b in actives) or 1.0
    for b in actives:
        M[i_solo, bed_states.index(b)] = active_split.get(b, 0.0) / tot
    for b in actives:
        i = bed_states.index(b)
        M[i, i_solo] = p_ret
        for o in actives:
            if o != b:
                M[i, bed_states.index(o)] = 1.0 - p_ret
    rs = M.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return (M / rs).tolist()


def _clamp_dwell(window, floor_s):
    """Raise a fitted window to the deployment floor, keeping min <= max.
    A window entirely below the floor collapses to (floor, floor) rather than
    inverting."""
    lo, hi = float(window[0]), float(window[1])
    return (max(lo, floor_s), max(hi, floor_s))


def preset(name, trans_source="handset", corpus=None, dwell_floor_s=DWELL_FLOOR_S):
    """(states, trans) for the three deployment arrangements.

    Transitions are biased back to the drone-solo anchor so the drone stays the
    protagonist. Hand-set is the default and only "overlay" is affected by
    trans_source.
    """
    if name == "drone_only":
        # Degenerate single state: the scheduler is a provable no-op.
        return [SchedState("drone_solo", {"bed": "off"}, (1e9, 1e9))], [[0.0]]

    if name == "overlay":
        states = [
            SchedState("drone_solo", {"bed": "off"},    (20.0, 50.0)),
            SchedState("bed_duck",   {"bed": "duck"},   (25.0, 60.0)),
            SchedState("bed_bursts", {"bed": "bursts"}, (20.0, 45.0)),
        ]
        trans = [[0.0, 0.6, 0.4],
                 [0.7, 0.0, 0.3],
                 [0.7, 0.3, 0.0]]
        if trans_source in ("corpus", "corpus_dwell"):
            if corpus is None:
                corpus = load_corpus()
            if corpus is None:
                raise SystemExit(
                    f"\ntrans_source={trans_source!r} needs the corpus fit at:\n"
                    f"    {HSMM}\n\nIt ships with this section.\n")
            dw = corpus_dwell_by_state(corpus)
            states = [SchedState(s.name, s.layers,
                                 _clamp_dwell(dw[s.name], dwell_floor_s))
                      for s in states]
            if trans_source == "corpus":
                trans = corpus_overlay_trans(corpus)
            # "corpus_dwell" keeps the hand-set matrix on purpose: the corpus
            # grounds timing, but has no bed/no-bed distinction to ground
            # arrangement.
        return states, trans

    if name == "overlay_melodic":
        # The melodic layer is scheduled but unrendered until that layer
        # exists; kept here to prove the contract is genuinely layer-agnostic.
        states, _ = preset("overlay")
        states = states + [SchedState("melody_lead",
                                      {"bed": "off", "melody": "lead"},
                                      (20.0, 45.0))]
        trans = [[0.0, 0.45, 0.30, 0.25],
                 [0.55, 0.0, 0.20, 0.25],
                 [0.60, 0.25, 0.0, 0.15],
                 [0.60, 0.25, 0.15, 0.0]]
        return states, trans

    raise ValueError(f"unknown preset {name!r}")


def example_va_gate(context, s_from, s_to):
    """Conditioning-hook demo: VA modulates reachability. Bursts were tagged
    on dark clips in the texture round, so veto them at high valence. Reads
    the context, never writes it."""
    if not context:
        return 1.0
    if s_to.name == "bed_bursts" and float(context.get("valence", 0.0)) > 0.5:
        return 0.0
    return 1.0


class SemiMarkovScheduler:
    """Batch scheduler: sample a whole schedule for a known duration."""

    def __init__(self, states, trans, seed=0, gate=None):
        self.states = list(states)
        self.trans = np.asarray(trans, dtype=float)
        n = len(self.states)
        assert self.trans.shape == (n, n), "trans must be n x n"
        assert np.allclose(np.diag(self.trans), 0.0), \
            "no self-transitions -- staying put is the dwell's job"
        assert np.all(self.trans >= 0)
        self.gate = gate
        self.rng = np.random.default_rng(seed)

    def sample_schedule(self, total_s, context=None, start=0):
        """Contiguous [0, total_s] cover as [{name, t0, t1, layers}, ...].
        Consecutive same-state stretches, where the gate vetoed every move,
        are merged into one segment."""
        segs, i, t, vetoes = [], start, 0.0, 0
        while t < total_s:
            lo, hi = self.states[i].dwell_s
            t1 = min(t + float(self.rng.uniform(lo, hi)), float(total_s))
            if segs and segs[-1]["name"] == self.states[i].name:
                segs[-1]["t1"] = t1
            else:
                segs.append(dict(name=self.states[i].name, t0=t, t1=t1,
                                 layers=dict(self.states[i].layers)))
            t = t1
            if t >= total_s:
                break
            w = np.array(self.trans[i])
            if self.gate is not None:
                w = w * np.array([float(self.gate(context, self.states[i], s2))
                                  for s2 in self.states])
                w[i] = 0.0
            if w.sum() <= 0:
                vetoes += 1
                if vetoes == 3:
                    print(f"note: the gate vetoed every exit from "
                          f"{self.states[i].name!r} three times running -- "
                          "staying is the only legal arrangement under this "
                          "context (gates should leave >=1 exit reachable)")
                continue
            vetoes = 0
            i = int(self.rng.choice(len(self.states), p=w / w.sum()))
        return segs


class LiveBedScheduler:
    """Open-ended, CAUSAL counterpart to `sample_schedule`.

    Why a second class rather than reusing the batch path: that method needs a
    known total duration and returns the whole segment list at once. The live
    conductor runs open-ended and renders short audio segments one at a time,
    so it needs to ask "what is the bed doing during [t0, t0+dur)?" without
    knowing when the piece ends. State changes are therefore decided LAZILY,
    the instant a dwell expires, and the VA gate is re-evaluated with FRESH
    context at each such decision -- which the batch path cannot do, because it
    samples the whole schedule up front.

    Fades are causal here: at a change at time tb the outgoing shape ramps down
    over [tb, tb+xfade) and the incoming ramps up over the same window. A batch
    renderer would instead CENTRE fades on boundaries, which needs lookahead --
    fine offline, impossible live.

    The envelope is a pure function of ABSOLUTE time, so it stays continuous
    across segment joins: the caller passes t0 and this class keeps no
    per-segment state beyond the schedule itself.
    """

    def __init__(self, states, trans, seed=0, gate=None, xfade_s=XFADE_S,
                 burst_period_s=BREATH_PERIOD_S, burst_phase_s=0.0):
        self.sched = SemiMarkovScheduler(states, trans, seed=seed, gate=gate)
        self.states = self.sched.states
        self.xfade_s = float(xfade_s)
        self.burst_period_s = float(burst_period_s)
        self.burst_phase_s = float(burst_phase_s)
        self.i = 0
        self.t_start = 0.0
        self.t_expire = None
        self._spans = []              # [(t0, t1, state_idx)] already decided

    def _draw_dwell(self, i):
        lo, hi = self.states[i].dwell_s
        return float(self.sched.rng.uniform(lo, hi))

    def _ensure(self, t0, t_end, context):
        """Extend the decided spans to cover t_end, then drop only spans that
        can no longer affect any sample at or after t0.

        The pruning bound MUST allow for the fade-out tail: a span ending just
        before t0 still ramps down across [b, b+xfade), which overlaps this
        segment. Pruning on t_end instead silently truncated that tail and
        produced an audible step at segment joins.
        """
        if self.t_expire is None:
            self.t_expire = self.t_start + self._draw_dwell(self.i)
            self._spans.append((self.t_start, self.t_expire, self.i))
        while self.t_expire < t_end:
            w = np.array(self.sched.trans[self.i], dtype=float)
            if self.sched.gate is not None:     # FRESH context, per the contract
                w = w * np.array([float(self.sched.gate(
                    context, self.states[self.i], s2)) for s2 in self.states])
                w[self.i] = 0.0
            if w.sum() <= 0:
                self.t_expire += self._draw_dwell(self.i)
                a, _, idx = self._spans[-1]
                self._spans[-1] = (a, self.t_expire, idx)
                continue
            self.i = int(self.sched.rng.choice(len(self.states), p=w / w.sum()))
            self.t_start = self.t_expire
            self.t_expire = self.t_start + self._draw_dwell(self.i)
            self._spans.append((self.t_start, self.t_expire, self.i))
        self._spans = [s for s in self._spans
                       if s[1] + self.xfade_s > t0] or self._spans[-1:]

    def _burst_shape(self, t, span_a, span_b, burst_s):
        """Hann bursts as a closed-form function of ABSOLUTE time.

        Centres sit on a fixed grid, phase-anchored to the span's own start, so
        the shape is continuous across segment joins by construction: a burst
        straddling a join simply continues, because nothing depends on where
        the segment happens to be cut.

        A dip-seeking placement -- ranking local minima of the drone envelope
        and accepting quietest-first -- sounds better offline but is a
        whole-window algorithm. It cannot be evaluated causally, and run per
        segment it picks different centres each time, which measures as a
        full-scale step at joins. Setting burst_period_s to the drone's breath
        period recovers dip alignment analytically when that period is known.
        """
        p = float(self.burst_period_s)
        out = np.zeros(len(t))
        k0 = int(np.floor((t[0] - span_a) / p)) - 1
        k1 = int(np.ceil((t[-1] - span_a) / p)) + 1
        for k in range(k0, k1 + 1):
            c = span_a + k * p + self.burst_phase_s
            if c + burst_s / 2.0 < span_a or c - burst_s / 2.0 >= span_b:
                continue                      # the burst must belong to a span
            m = np.abs(t - c) < burst_s / 2.0
            if m.any():
                out[m] = np.maximum(out[m],
                                    np.cos(np.pi * (t[m] - c) / burst_s) ** 2)
        return out

    def current_state(self, t):
        for a, b, i in self._spans:
            if a <= t < b:
                return self.states[i].name
        return self.states[self.i].name

    def bed_env(self, t0, n, drone_env, sr=SR, context=None,
                duck_depth=0.7, burst_s=3.0):
        """Per-sample bed gain over [t0, t0 + n/sr), as a function of absolute
        time. `drone_env` is that segment's loudness envelope, so duck and
        bursts couple to the drone."""
        drone_env = np.asarray(drone_env, dtype=float)
        assert len(drone_env) == n, (len(drone_env), n)
        t_end = t0 + n / float(sr)
        self._ensure(t0, t_end, context)
        t = t0 + np.arange(n) / float(sr)
        env = np.zeros(n)
        xf = self.xfade_s
        shapes = {}
        for a, b, i in self._spans:
            if b + xf <= t0 or a >= t_end:
                continue
            d = self.states[i].layers.get("bed", "off")
            if d == "off":
                continue
            # Duck and constant are span-independent, but bursts are
            # phase-anchored to their OWN span and must key on it -- keying
            # bursts by directive alone made a second bursts span in the same
            # call silently reuse the first span's grid.
            key = (d, a, b) if d == "bursts" else d
            if key not in shapes:
                if d == "constant":
                    shapes[key] = np.ones(n)
                elif d == "duck":
                    shapes[key] = 1.0 - float(duck_depth) * drone_env
                elif d == "bursts":
                    shapes[key] = self._burst_shape(t, a, b, float(burst_s))
                else:
                    print(f"note: unknown bed directive {d!r} -- treated as off")
                    shapes[key] = np.zeros(n)
            w = np.zeros(n)
            w[(t >= a) & (t < b)] = 1.0
            if a > 0.0:                       # causal ramp in at the boundary
                m = (t >= a) & (t < a + xf)
                w[m] = np.sin(0.5 * np.pi * (t[m] - a) / xf) ** 2
            m = (t >= b) & (t < b + xf)       # causal ramp out after it ends
            w[m] = np.cos(0.5 * np.pi * (t[m] - b) / xf) ** 2
            env += shapes[key] * w
        return np.clip(env, 0.0, 1.0)


def selftest():
    """Properties that must hold, checked without any audio."""
    ok = []

    # 1. drone_only is a provable no-op.
    st, tr = preset("drone_only")
    segs = SemiMarkovScheduler(st, tr, seed=0).sample_schedule(300.0)
    assert len(segs) == 1 and segs[0]["layers"]["bed"] == "off"
    ok.append("drone_only schedules exactly one silent-bed segment")

    # 2. Schedules cover [0, total] contiguously, with no gaps or overlaps.
    st, tr = preset("overlay")
    segs = SemiMarkovScheduler(st, tr, seed=1).sample_schedule(600.0)
    assert abs(segs[0]["t0"]) < 1e-9 and abs(segs[-1]["t1"] - 600.0) < 1e-9
    for a, b in zip(segs, segs[1:]):
        assert abs(a["t1"] - b["t0"]) < 1e-9, "schedule is not contiguous"
    ok.append(f"overlay covers 600 s contiguously in {len(segs)} segments")

    # 3. Every dwell respects its window, which is the semi-Markov property.
    win = {s.name: s.dwell_s for s in st}
    for s in segs[:-1]:
        lo, hi = win[s["name"]]
        assert lo - 1e-6 <= (s["t1"] - s["t0"]) + 1e-6, (s, lo)
    ok.append("every completed dwell respects its state's window")

    # 4. No segment ever repeats a state back to back (merging aside).
    assert all(a["name"] != b["name"] for a, b in zip(segs, segs[1:]))
    ok.append("no back-to-back repeats: staying put is the dwell's job")

    # 5. The gate is a hard constraint: bursts unreachable at high valence.
    st, tr = preset("overlay")
    segs = SemiMarkovScheduler(st, tr, seed=3, gate=example_va_gate
                               ).sample_schedule(900.0, context={"valence": 0.9})
    assert not any(s["name"] == "bed_bursts" for s in segs)
    ok.append("the VA gate makes bursts unreachable at high valence")

    # 6. The corpus fit loads and its dwell floor is applied.
    corpus = load_corpus()
    if corpus is not None:
        st_c, _ = preset("overlay", trans_source="corpus_dwell", corpus=corpus)
        raw = corpus_dwell_by_state(corpus)
        assert all(s.dwell_s[0] >= DWELL_FLOOR_S - 1e-9 for s in st_c)
        assert min(raw["bed_duck"]) < DWELL_FLOOR_S, \
            "the floor should actually bite on the corpus's active window"
        ok.append(f"corpus dwell {raw['bed_duck']} raised to "
                  f"{st_c[1].dwell_s} by the {DWELL_FLOOR_S:.0f}s floor")
        M = np.asarray(corpus_overlay_trans(corpus))
        assert np.allclose(np.diag(M), 0.0) and np.allclose(M.sum(axis=1), 1.0)
        ok.append("corpus transition matrix has a zero diagonal and unit rows")

    # 7. The live path equals the one-shot path when rendered in segments.
    #    This is what catches discontinuities at segment joins.
    st, tr = preset("overlay")
    n_total, sr = 40 * SR, SR
    drone = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n_total) / (12.0 * sr))
    live1 = LiveBedScheduler(st, tr, seed=7)
    one_shot = live1.bed_env(0.0, n_total, drone)
    live2 = LiveBedScheduler(st, tr, seed=7)
    chunks, step = [], 11 * sr
    for a in range(0, n_total, step):
        b = min(a + step, n_total)
        chunks.append(live2.bed_env(a / sr, b - a, drone[a:b]))
    segmented = np.concatenate(chunks)
    err = float(np.max(np.abs(one_shot - segmented)))
    assert err < 1e-9, f"segmented render differs from one-shot by {err:.3e}"
    ok.append(f"segmented render matches one-shot to {err:.1e} (no join clicks)")

    # 8. The envelope stays in range.
    assert one_shot.min() >= 0.0 and one_shot.max() <= 1.0
    ok.append("bed envelope stays within [0, 1]")

    for line in ok:
        print(f"  PASS  {line}")
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser(description="Semi-Markov arrangement scheduler.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--preset", default="overlay",
                    choices=["drone_only", "overlay", "overlay_melodic"])
    ap.add_argument("--trans-source", dest="trans_source", default="handset",
                    choices=["handset", "corpus_dwell", "corpus"])
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", action="store_true", help="apply the example VA gate")
    ap.add_argument("--valence", type=float, default=0.0)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    states, trans = preset(args.preset, trans_source=args.trans_source)
    print(f"preset={args.preset} trans_source={args.trans_source}")
    for s in states:
        print(f"  {s.name:12s} dwell {s.dwell_s[0]:6.1f}-{s.dwell_s[1]:6.1f}s  "
              f"layers {s.layers}")
    sched = SemiMarkovScheduler(states, trans, seed=args.seed,
                                gate=example_va_gate if args.gate else None)
    segs = sched.sample_schedule(args.seconds,
                                 context={"valence": args.valence})
    print(f"\n{len(segs)} segments over {args.seconds:.0f}s:")
    for s in segs:
        print(f"  {s['t0']:7.1f} - {s['t1']:7.1f}s  {s['name']}")


if __name__ == "__main__":
    main()
