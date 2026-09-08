"""Sequential-coherence re-ranker: break ties toward what follows coherently.

Runtime code. `arc_policy.py` is MEMORYLESS -- it picks the best arc for the
current VA context independently every time -- so nothing models ORDER, and it
will happily emit two large brightenings back to back. This fixes that using
the 5x5 corpus type-transition matrix already fitted on the ambient corpus:
classify the candidate arcs into corpus types, then, among the candidates the
preference GP likes about equally, prefer the one whose type most coherently
FOLLOWS the previous transition's type.

The discipline is load-bearing and is what makes this safe:

    human ratings decide what is GOOD; the corpus chain only breaks TIES
    toward what follows coherently.

The re-ranker never selects outside a near-tie band around the best utility, so
it cannot override a preference judgement -- it only reorders arcs the GP is
indifferent between. That is why a corpus chain is acceptable here where a
per-axis corpus chain over the whole decision would not be: it sits on top of
the validated preference model rather than replacing it.

Both artefacts are optional. When either is missing, `rerank` degrades to plain
best-utility, so the conductor runs before the type assignment exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
ARC_TYPES = WEIGHTS / "arc_types.json"
HSMM = WEIGHTS / "hsmm_transitions.json"


def load_coherence(arc_types_path=ARC_TYPES, hsmm_path=HSMM):
    """(arc_type: {arc_id -> int}, trans: (k,k) array), or (None, None) if
    either artefact is missing or unreadable -- rerank then falls back."""
    if not Path(arc_types_path).exists() or not Path(hsmm_path).exists():
        return None, None
    try:
        types = json.loads(Path(arc_types_path).read_text()).get("arcs", {})
        tm = json.loads(Path(hsmm_path).read_text())["corpus_type_model"]["transition_matrix"]
    except (OSError, ValueError, KeyError):
        return None, None
    return {a: int(v["type"]) for a, v in types.items()}, np.asarray(tm, float)


def rerank(candidates, prev_type, arc_types, trans, tie_margin=0.1):
    """Pick an arc from ``candidates`` = [(arc_id, utility), ...].

    Among candidates within ``tie_margin`` of the best utility -- a near-tie the
    preference GP is effectively indifferent over; the default 0.1 is about the
    utility noise floor -- choose the one whose corpus type most coherently
    follows ``prev_type``. Falls back to best-utility when prev_type is None,
    the band is a singleton, or the coherence data is absent.

    Returns (arc_id, info).
    """
    if not candidates:
        return None, dict(reason="no candidates")
    best_u = max(u for _, u in candidates)
    band = [(a, u) for a, u in candidates if best_u - u <= tie_margin]

    can_cohere = (prev_type is not None and arc_types is not None
                  and trans is not None and len(band) > 1)
    if not can_cohere:
        aid, u = max(band, key=lambda x: x[1])
        return aid, dict(reason="best_utility", band_size=len(band),
                         prev_type=prev_type,
                         cand_type=(arc_types or {}).get(aid),
                         utility=round(float(u), 4))

    def coh(a):
        t = arc_types.get(a)
        return float(trans[prev_type][t]) if t is not None else -1.0

    # Coherence first, utility as the tie-break inside the band.
    aid, u = max(band, key=lambda x: (coh(x[0]), x[1]))
    plain = max(band, key=lambda x: x[1])[0]
    return aid, dict(reason="coherence_tiebreak", band_size=len(band),
                     prev_type=prev_type, cand_type=arc_types.get(aid),
                     coherence=round(coh(aid), 4), utility=round(float(u), 4),
                     changed_from_best_utility=(aid != plain))


def selftest():
    """Three candidates: A has the best utility but is incoherent, B is
    slightly lower and coherent, C is far lower (outside the band)."""
    trans = np.array([[0.0, 0.1, 0.9],       # from type 0, strongly to type 2
                      [0.5, 0.0, 0.5],
                      [0.5, 0.5, 0.0]])
    arc_types = {"A": 1, "B": 2, "C": 2}
    cands = [("A", 1.00), ("B", 0.95), ("C", 0.50)]

    aid, info = rerank(cands, 0, arc_types, trans, tie_margin=0.1)
    assert aid == "B", (aid, info)
    assert info["changed_from_best_utility"] and info["cand_type"] == 2

    aid2, info2 = rerank(cands, 0, arc_types, trans, tie_margin=0.0)
    assert aid2 == "A" and info2["reason"] == "best_utility"

    # Must NEVER pick C, outside the band, however coherent its type is.
    for tm in (0.05, 0.1, 0.2, 0.4):
        assert rerank(cands, 0, arc_types, trans, tie_margin=tm)[0] != "C"

    assert rerank(cands, None, arc_types, trans)[0] == "A"
    assert rerank(cands, 0, None, None)[1]["reason"] == "best_utility"
    assert rerank([], 0, arc_types, trans)[0] is None
    print("selftest OK: coherence breaks ties inside the band toward the type "
          "that follows prev_type; never selects outside the band, so a "
          "preference judgement is never overridden; degrades gracefully when "
          "prev_type, the coherence data or the candidates are absent.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Sequential-coherence re-ranker.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        at, tr = load_coherence()
        if at is None:
            print(f"no arc_types.json / hsmm_transitions.json under {WEIGHTS}")
        else:
            print(f"loaded {len(at)} arc types; transition matrix {tr.shape}")
            print("type distribution:",
                  {t: sum(1 for v in at.values() if v == t)
                   for t in sorted(set(at.values()))})
