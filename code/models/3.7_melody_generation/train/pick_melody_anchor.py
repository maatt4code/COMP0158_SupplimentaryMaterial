"""Freeze the melody's timbre anchor, so the runtime needs no rating data.

One of the boot-time fits that the conductor must not perform for itself. The
melody is voiced by borrowing a preset from the rated seed pool of Section
3.4.2 -- a real, human-rated timbre rather than an invented one -- and choosing
it requires reading that pool's ratings. The conductor ships no rating data, so
the choice is made here, once, and written to `../weights/melody_anchor.json`.

HOW THE CHOICE IS MADE, and why it is not arbitrary. Candidates must clear
three bars at once:

  * a positive human rating, so the timbre is one a listener actually liked;
  * a harmonic centroid inside a band, because a melody voiced on a very dark
    preset disappears under the drone and a very bright one cuts through it as
    a separate instrument rather than reading as part of the same piece;
  * enough candidates left that the pick is not simply the only survivor.

The pick is deterministic: the same pool gives the same anchor every time.

The bed anchor is a different question and stays dynamic: it depends on the
live valence and arousal, so there is nothing to freeze.

Run:
  python pick_melody_anchor.py                 # report the choice
  python pick_melody_anchor.py --freeze        # write the weight

Previous: Section 3.4.2's rated pool
Next:     ../inference/melody_markov.py reads the frozen anchor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_SECTION = _HERE.parents[1]
sys.path.insert(0, str(_SECTION / "inference"))
import melody_markov as mm                               # noqa: E402

DEFAULT_OUT = _SECTION / "data" / "melody_anchor.json"


def choose(min_rating=1.0, centroid_range=(1.6, 3.2)):
    cands = mm.melody_anchor_candidates(min_rating=min_rating,
                                        centroid_range=centroid_range)
    anchor = mm.pick_melody_anchor(min_rating=min_rating,
                                   centroid_range=centroid_range)
    return anchor, cands


def main():
    ap = argparse.ArgumentParser(
        description="Freeze the melody timbre anchor for the runtime.")
    ap.add_argument("--min-rating", dest="min_rating", type=float, default=1.0,
                    help="lowest human rating a candidate preset may carry")
    ap.add_argument("--centroid-lo", dest="lo", type=float, default=1.6)
    ap.add_argument("--centroid-hi", dest="hi", type=float, default=3.2)
    ap.add_argument("--freeze", action="store_true", help="write the weight")
    ap.add_argument("--out", default=None, help=f"default {DEFAULT_OUT}")
    args = ap.parse_args()

    if not mm.POOL_RATINGS.exists():
        raise SystemExit(
            f"\nrated pool not found at:\n    {mm.POOL_RATINGS}\n\n"
            "It ships with Section 3.4.2.\n")

    anchor, cands = choose(args.min_rating, (args.lo, args.hi))
    print(f"pool      : {mm.POOL_RATINGS}")
    print(f"candidates: {len(cands)} presets clear rating >= {args.min_rating} "
          f"and centroid in [{args.lo}, {args.hi}]")
    print(f"anchor    : {anchor}")
    if len(cands) < 2:
        print("NOTE: fewer than two candidates, so this is close to a forced "
              "choice rather than a selection. Widen the bands to check.")

    if args.freeze:
        out = Path(args.out) if args.out else DEFAULT_OUT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "anchor_idx": int(anchor),
            "n_candidates": int(len(cands)),
            # candidates are (index, ...) records; keep the whole row so
            # the choice is auditable, not just its winner
            "candidates": [list(c) if isinstance(c, (tuple, list)) else c
                           for c in cands],
            "min_rating": float(args.min_rating),
            "centroid_range": [float(args.lo), float(args.hi)],
            "source": "Section 3.4.2 rated seed pool",
            "note": ("Chosen once so the runtime needs no rating data. "
                     "Deterministic given the pool. The BED anchor is not "
                     "frozen: it depends on live valence and arousal."),
        }, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
