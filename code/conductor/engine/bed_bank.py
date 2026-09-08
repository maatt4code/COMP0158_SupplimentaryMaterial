"""Choosing a background bed for the current mood: the RUNTIME half.

The bank itself is built on the training side by
`models/3.5_transition_dynamics_and_scheduling/train/build_bed_bank.py`, which
reads the curation log and writes `weights/bed_bank.json`. Only the selection
lives here, because that is all the conductor does with it: the curation
verdicts are rating data, and the conductor ships none.

TWO GATES condition a bed, and only one of them is applied here.

  * The EMOTION gate, applied below. A bed carries a valence-arousal rating
    when its source corpus provides one -- the Emo-Soundscapes beds do, the
    ESC-50 beds do not. Untagged beds are eligible in ANY mood and score at a
    fixed, beatable distance, so a well-matched tagged bed wins while an
    untagged one stays reachable rather than being excluded outright.

  * The DRONE gate is NOT applied here. Most beds were auditioned bed-only, so
    the bed-drone pairing is sparsely rated and the sparse pairings are
    validation cases rather than training data. It is resolved acoustically at
    mix time, against the drone actually playing -- and this module has no
    audio.
"""
from __future__ import annotations

import json
from pathlib import Path

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
DEFAULT_BANK = WEIGHTS / "bed_bank.json"

# Distance assigned to a bed with no VA rating. It has to be beatable by a
# genuinely close tagged bed but small enough that an untagged bed still wins
# when nothing is close, which is what keeps the ESC-50 half of the bank in
# play instead of decorative.
NEUTRAL_DISTANCE = 0.75


def load_bank(path=None):
    """The built bed bank, or an empty list if none ships.

    Returning empty rather than raising is deliberate: the bed layer is one
    optional voice among several, and a conductor with no beds should still
    play a drone."""
    p = Path(path) if path else DEFAULT_BANK
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return data["beds"] if isinstance(data, dict) and "beds" in data else data


def select_bed(bank, va=None, prefer_type=None, exclude=frozenset(),
               k=1, rng=None):
    """Pick a bed for the current valence-arousal point, or None.

    Deterministic by default (`k=1`, no RNG) so the conductor does not swap
    beds jitterily; the CALLER decides when to re-select, which keeps the
    choice stable across the ticks of a single segment.

    VARIETY. When the curated bank grew from 20 to 83 beds, the deterministic
    argmin kept replaying one bed per region of the mood space -- a larger
    bank made the output LESS varied, because more beds only meant more ties
    that the same bed kept winning. With `k > 1` and a numpy Generator,
    sample uniformly among the k nearest instead; callers typically also pass
    `exclude={current bed}` to guarantee rotation.
    """
    pool = [b for b in bank if b.get("resolved")
            and b["bed_file"] not in exclude
            and (prefer_type is None or b["bed_type"] == prefer_type)]
    if not pool:
        # Nothing of the preferred type: fall back to the whole bank rather
        # than returning None, so a type filter can never silence the layer.
        pool = [b for b in bank if b.get("resolved")
                and b["bed_file"] not in exclude]
    if not pool:
        return None
    if va is None:
        return pool[0]

    def dist(b):
        if not b.get("va"):
            return NEUTRAL_DISTANCE
        return ((b["va"][0] - va[0]) ** 2 + (b["va"][1] - va[1]) ** 2) ** 0.5

    if k <= 1 or rng is None:
        return min(pool, key=dist)
    cands = sorted(pool, key=dist)[: int(k)]
    return cands[int(rng.integers(len(cands)))]
