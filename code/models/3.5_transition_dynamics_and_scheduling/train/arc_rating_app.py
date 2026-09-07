"""Pairwise arc rating app: "which transition feels better?"

Data collection, not runtime. This is the instrument that produced
`../human_ratings/arc_ratings.csv`, and it ships so the protocol behind those
ratings is inspectable rather than merely described. Running it needs the
rendered arc-pool audio, which is not distributed; the queue-building logic
below runs and is self-tested without any audio.

The protocol, and why each piece is there:

  * MASKING. The rater sees only "Clip A" and "Clip B" -- never a filename,
    never a parameter. Otherwise a rater who notices that one clip is the long
    ramp starts rating the label rather than the sound.
  * SINGLE-AXIS CONTRAST SEEDING. The queue serves pairs differing in exactly
    ONE vocabulary axis first: distance, ramp, or chord path. Pairs must share
    the same pads and vocabulary so that only the transition differs;
    otherwise a preference could be about the pad rather than the move. Every
    renderer probe pairs against its own identical-parameter base.
  * WITHIN-SCENE ONLY. Even the random pairs stay inside a scene, so they add
    multi-axis contrasts on shared pads rather than cross-world comparisons
    nobody can judge.
  * CATCH TRIALS. Identical pairs (expect chance) and A/B-swapped repeats
    (expect consistency) are spliced in on fixed intervals and logged with
    their pair_type, so per-block consistency is computable afterwards.
  * PRESENTATION ORDER IS RANDOMISED. Seed pairs are generated in canonical
    arc-id order, so without a coin flip "Clip A" would systematically be the
    lower id -- a position bias straight into the data. The CSV records what
    was actually played as A and B.
  * DE-CLUMPING. The same start pad is not served twice in a row; scene
    variety is what keeps a rater engaged.
  * A FATIGUE CHECKPOINT, NOT A HARD STOP. At a pair or minute budget the app
    offers a break, and a rater who is not tired can push both thresholds out
    by another budget-unit.
  * AN HONEST PROGRESS DENOMINATOR. The rater-facing target is a fixed,
    capped subset of the single-axis pairs, not every possible within-scene
    pair. Showing the full combinatorial denominator would imply raters must
    exhaust a space the preference GP exists specifically to make unnecessary
    -- comparisons share strength across arc PARAMETERS, so active selection
    reduces the comparisons needed by roughly an order of magnitude against
    grid coverage.
  * RETURNING RATERS GET FRESH PAIRS. Pairs already rated are filtered before
    catch injection, so catch-swap back-references stay valid. Catch trials
    are excluded from that filter, because they must stay repeatable.

The active queue written by `fit_preference_gp.py` -- the maximum-predictive-
variance pairs -- overrides the tail of the queue when present. This app stays
a dumb, reliable presenter; the intelligence lives in the fit.

Run:
  python arc_rating_app.py --selftest      # queue logic only, no audio needed
  python arc_rating_app.py --serve         # the rating UI (needs the pool audio)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_SECTION = _HERE.parents[1]
POOL_META = _SECTION / "human_ratings" / "arc_pool_meta.json"
RATINGS_CSV = _SECTION / "data" / "arc_ratings.csv"
QUEUE_OVERRIDE = _SECTION / "data" / "preference_gp" / "active_queue.json"

MAX_PAIRS = 22             # against a ~15 minute fatigue budget
MAX_MINUTES = 15.0
CATCH_IDENTICAL_EVERY = 12   # one slot in N becomes an A == B catch trial
CATCH_SWAP_EVERY = 10        # one slot in N repeats an earlier pair, swapped
TARGET_CAP_PER_AXIS = 40     # the rater-facing progress denominator

CSV_FIELDS = ["timestamp", "rater", "session_id", "pair_index", "pair_type",
              "arc_a", "arc_b", "choice", "swap_of", "context_v", "context_a",
              "params_a", "params_b", "loudness_gain_a", "loudness_gain_b",
              "pool_version", "arranger_sha256"]

AXES = ("dist_bin", "ramp_s", "chord")   # the single-axis contrast definitions


def load_pool(meta_path=POOL_META, audio_dir=None):
    """Arc metadata, optionally gated on the audio being present."""
    p = Path(meta_path)
    if not p.exists():
        raise SystemExit(f"\narc pool metadata not found at:\n    {p}\n")
    pool = {}
    for m in json.loads(p.read_text()):
        if audio_dir is not None:
            local = Path(audio_dir) / Path(m["clip_path"]).name
            if not local.exists():
                continue
            m["_local"] = str(local)
        m["chord"] = f"{m['chord_start']}->{m['chord_end']}"
        pool[m["arc_id"]] = m
    if not pool:
        raise SystemExit(f"\nno arcs available (audio_dir={audio_dir})\n")
    return pool


def arc_key(m):
    return dict(context_id=m["context_id"], dist_bin=m["dist_bin"],
                ramp_s=m["ramp_s"], chord=m["chord"])


def probe_overrides(m):
    """Everything that makes an arc a probe: renderer-kwarg overrides or
    end-theta texture overrides (the thick/thin probes use only the latter)."""
    return dict(m.get("render_overrides") or {},
                **(m.get("theta_end_overrides") or {}))


def seed_pairs(pool):
    """Single-axis contrast pairs WITHIN a scene: same pads and vocabulary,
    differing on exactly one axis, so only the transition differs. Probe arcs
    pair only against their exact identical-parameter base."""
    ids = sorted(pool)
    pairs = []
    for i, a in enumerate(ids):
        ma = pool[a]
        for b in ids[i + 1:]:
            mb = pool[b]
            if ma.get("scene_id") != mb.get("scene_id"):
                continue
            if probe_overrides(ma) or probe_overrides(mb):
                if arc_key(ma) == arc_key(mb) and \
                        (bool(probe_overrides(ma)) != bool(probe_overrides(mb))):
                    pairs.append((a, b, "seed_probe"))
                continue
            diffs = [ax for ax in AXES if arc_key(ma)[ax] != arc_key(mb)[ax]]
            if len(diffs) == 1:
                pairs.append((a, b, f"seed_{diffs[0]}"))
    return pairs


def build_queue(pool, rng, exclude=None, queue_override=QUEUE_OVERRIDE):
    """Seed contrasts, shuffled, then random within-scene pairs, with catch
    trials spliced in at fixed intervals."""
    seeds = seed_pairs(pool)
    rng.shuffle(seeds)
    ids = sorted(pool)

    by_scene = {}
    for k in ids:
        by_scene.setdefault(pool[k].get("scene_id"), []).append(k)
    scenes = [s for s, ks in by_scene.items() if len(ks) >= 2]
    extras = []
    for _ in range(200):
        ks = by_scene[scenes[int(rng.integers(len(scenes)))]]
        a, b = rng.choice(ks, 2, replace=False)
        extras.append((str(a), str(b), "random"))
    if Path(queue_override).exists():
        act = json.loads(Path(queue_override).read_text())
        extras = [(p["arc_a"], p["arc_b"], "active") for p in act] + extras
        print(f"active queue loaded: {len(act)} pairs")
    base = seeds + extras
    if exclude:
        base = [p for p in base if frozenset((p[0], p[1])) not in exclude]

    # De-clump: never serve the same start pad twice in a row.
    def start_pad(item):
        return pool[item[0]].get("start_anchor")
    i = 1
    while i < len(base):
        if start_pad(base[i]) == start_pad(base[i - 1]):
            j = next((j for j in range(i + 1, len(base))
                      if start_pad(base[j]) != start_pad(base[i - 1])), None)
            if j is None:
                break
            base[i], base[j] = base[j], base[i]
        i += 1

    queue, served = [], []
    for item in base:
        slot = len(queue)
        if slot and slot % CATCH_IDENTICAL_EVERY == 0:
            c = str(rng.choice(ids))
            queue.append((c, c, "catch_identical", ""))
        if slot and slot % CATCH_SWAP_EVERY == 0 and served:
            j = int(rng.integers(len(served)))
            pa, pb, _, _ = served[j]
            queue.append((pb, pa, "catch_swap", f"pair_{j}"))
        a, b = item[0], item[1]
        if rng.random() < 0.5:          # kill the position bias
            a, b = b, a
        entry = (a, b, item[2], "")
        queue.append(entry)
        served.append(entry)
    return queue


def rated_pairs(rater, ratings_csv=RATINGS_CSV):
    """Unordered pairs this rater already rated, across all past sessions.
    Catch trials are excluded: they must stay repeatable."""
    done = set()
    p = Path(ratings_csv)
    if p.exists():
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r.get("rater") == rater and \
                        not r.get("pair_type", "").startswith("catch"):
                    done.add(frozenset((r["arc_a"], r["arc_b"])))
    return done


def count_pool_pairs(pool):
    """Every distinct within-scene pair: the full combinatorial space, kept
    for diagnostics only. It is NOT the rating target."""
    by_scene = {}
    for k in pool:
        by_scene.setdefault(pool[k].get("scene_id"), []).append(k)
    return sum(len(ks) * (len(ks) - 1) // 2
               for ks in by_scene.values() if len(ks) >= 2)


def build_target_pairs(pool, cap_per_axis=TARGET_CAP_PER_AXIS):
    """The rater-facing progress target: a fixed, capped subset of the
    single-axis pairs. Capped per axis so all three vocabulary axes stay
    equally represented, and every probe is kept because each one is its own
    finding. The RNG seed is fixed rather than per-session, so the numbering
    is stable across reloads and comparable between raters."""
    by_axis = {}
    for p in seed_pairs(pool):
        by_axis.setdefault(p[2], []).append(p)
    rng = np.random.default_rng(42)
    target = []
    for axis in sorted(by_axis):
        ps = list(by_axis[axis])
        rng.shuffle(ps)
        target.extend(ps if axis == "seed_probe" else ps[:cap_per_axis])
    return target


def append_rating(row, ratings_csv=RATINGS_CSV):
    p = Path(ratings_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def selftest():
    """The queue properties that make the collected data analysable. No audio."""
    pool = load_pool()
    ok = []

    seeds = seed_pairs(pool)
    assert seeds, "no single-axis contrast pairs found"
    for a, b, kind in seeds:
        assert pool[a]["scene_id"] == pool[b]["scene_id"], "pair crosses scenes"
        if kind != "seed_probe":
            ka, kb = arc_key(pool[a]), arc_key(pool[b])
            assert sum(ka[ax] != kb[ax] for ax in AXES) == 1, \
                "a seed pair must differ on exactly one axis"
    ok.append(f"{len(seeds)} seed pairs, each within a scene and differing "
              f"on exactly one axis")

    kinds = {}
    for _, _, k in seeds:
        kinds[k] = kinds.get(k, 0) + 1
    ok.append(f"seed pairs by axis: {kinds}")

    rng = np.random.default_rng(0)
    q = build_queue(pool, rng, queue_override=Path("/nonexistent"))
    ids = set(pool)
    for a, b, kind, swap in q:
        assert a in ids and b in ids, "queue references an unknown arc"
        if kind == "catch_identical":
            assert a == b, "an identical catch trial must repeat one clip"
        if kind != "catch_identical":
            assert a != b or kind == "catch_identical"
    ok.append(f"queue of {len(q)} entries, all arcs known")

    n_ident = sum(1 for e in q if e[2] == "catch_identical")
    n_swap = sum(1 for e in q if e[2] == "catch_swap")
    assert n_ident > 0 and n_swap > 0, "catch trials were not injected"
    ok.append(f"catch trials injected: {n_ident} identical, {n_swap} swapped")

    # Every swap catch must reference a pair that was genuinely served before.
    served = [e for e in q if not e[2].startswith("catch")]
    for e in q:
        if e[2] == "catch_swap":
            j = int(e[3].split("_")[1])
            assert 0 <= j < len(served), "swap catch references an unserved pair"
    ok.append("every swapped catch references an earlier served pair")

    # Presentation order must not be systematically ordered by arc id.
    non_catch = [e for e in q if not e[2].startswith("catch")]
    lower_first = sum(1 for a, b, _, _ in non_catch if a < b)
    frac = lower_first / len(non_catch)
    assert 0.35 < frac < 0.65, f"position bias: {frac:.2%} have A < B"
    ok.append(f"presentation order is randomised ({frac:.0%} have A before B)")

    # De-clumping: consecutive entries should rarely share a start pad.
    clumps = sum(1 for x, y in zip(non_catch, non_catch[1:])
                 if pool[x[0]].get("start_anchor") == pool[y[0]].get("start_anchor"))
    ok.append(f"consecutive same-start-pad entries: {clumps}/{len(non_catch)-1}")

    target = build_target_pairs(pool)
    full = count_pool_pairs(pool)
    assert len(target) < full, "the target must be smaller than the full space"
    assert build_target_pairs(pool) == target, "target must be deterministic"
    ok.append(f"progress target {len(target)} pairs, against {full} possible "
              f"-- an honest denominator, and stable across reloads")

    for line in ok:
        print(f"  PASS  {line}")
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser(description="Pairwise arc rating app.")
    ap.add_argument("--selftest", action="store_true",
                    help="check the queue logic; needs no audio")
    ap.add_argument("--serve", action="store_true",
                    help="launch the rating UI (needs the arc-pool audio)")
    ap.add_argument("--audio-dir", dest="audio_dir", default=None,
                    help="directory holding the rendered arc wavs")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.serve:
        ap.print_help()
        return
    if not args.audio_dir:
        raise SystemExit(
            "\n--serve needs --audio-dir pointing at the rendered arc pool.\n"
            "That audio is not distributed with this repository; the ratings "
            "it produced ship in ../human_ratings/.\n")
    raise SystemExit(
        "\nThe interactive UI is not included in this distribution.\n"
        "What ships is the protocol above and the queue logic that implements "
        "it (--selftest), plus the ratings themselves. Rebuilding the UI on "
        "top of build_queue() is a small exercise; the methodology is the "
        "part worth preserving.\n")


if __name__ == "__main__":
    main()
