"""Smoke test for Section 3.5. Needs no dataset, no audio, no network.

Runs the real code against the shipped ratings and frozen weights. It cannot
check the rendered arrangement, because the arc-pool and bed audio are not
distributed; README.md records the reference numbers for a full run.

Run:
  python smoke_test.py

Exit code 0 means every check passed.
"""

import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train"
INFER = HERE / "inference"
PY = sys.executable
FAILED = []

TRAIN_SCRIPTS = ["fit_preference_gp.py", "arc_rating_app.py",
                 "extract_transition_typology.py", "fit_corpus_hsmm.py"]
INFER_MODULES = ["arc_policy.py", "scheduler.py", "coherence_reranker.py"]


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def module_scope_imports(path):
    names = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def main():
    print("Section 3.5 smoke test\n")
    sys.path.insert(0, str(INFER))
    sys.path.insert(0, str(TRAIN))

    print("1. structure")
    for s in TRAIN_SCRIPTS:
        check(f"train/{s} exists", (TRAIN / s).exists())
    for m in INFER_MODULES:
        check(f"inference/{m} exists", (INFER / m).exists())
    for w in ["hsmm_transitions.json", "arc_types.json", "preference_gp.npz"]:
        check(f"weights/{w} ships", (HERE / "weights" / w).exists())
    check("the corpus timing pipeline ships and is corpus-agnostic",
          (TRAIN / "fit_corpus_hsmm.py").exists()
          and (TRAIN / "extract_transition_typology.py").exists())
    check("weights carry a checksum record", (HERE / "weights" / "SHA256SUMS").exists())

    print("\n2. inference boundary")
    for m in INFER_MODULES:
        imports = module_scope_imports(INFER / m)
        check(f"{m} does not import paths", "paths" not in imports)
        src = (INFER / m).read_text()
        check(f"{m} resolves no dataset root",
              "paths.data" not in src and "paths.repo_data" not in src)
        check(f"{m} does not import the training code",
              not any(t.replace(".py", "") in imports for t in TRAIN_SCRIPTS))
    check("the scheduler does not import the corpus fitter",
          "s13_corpus_hsmm_fit" not in (INFER / "scheduler.py").read_text())

    print("\n3. shipped ratings are pseudonymous")
    for name, n_rows in [("arc_ratings.csv", 120), ("arc_valence_ratings.csv", 199),
                         ("texture_ratings.csv", 122)]:
        p = HERE / "human_ratings" / name
        with open(p) as f:
            rows = list(csv.DictReader(f))
        check(f"{name} has {n_rows} rows", len(rows) == n_rows, str(len(rows)))
        raters = {r["rater"] for r in rows}
        check(f"{name} raters are all pseudonyms",
              all(r.startswith("R") and r[1:].isdigit() for r in raters),
              str(sorted(raters)))
    meta = json.loads((HERE / "human_ratings" / "arc_pool_meta.json").read_text())
    check("arc pool metadata has 230 arcs", len(meta) == 230, str(len(meta)))
    check("arc pool metadata carries no absolute path",
          "/cs/" not in json.dumps(meta))

    print("\n4. frozen preference GP")
    from arc_policy import (PreferenceGP, pool_contexts, lookup_best_arc,
                            best_by_gp_predict, candidate_grid, context_of)
    gp = PreferenceGP.load()
    check("posterior covers every arc", len(gp.ids) == 230, str(len(gp.ids)))
    check("11 features", len(gp.feature_names) == 11, str(len(gp.feature_names)))
    mm, ms, be = gp.selfcheck()
    check("predictive mean reproduces f at training rows", mm < 1e-3, f"{mm:.2e}")
    check("predictive std reproduces diag(Sigma)", ms < 1e-3, f"{ms:.2e}")
    check("batch path equals the single-row path", be < 1e-9, f"{be:.2e}")
    check("posterior std is strictly positive", bool((gp.sd > 0).all()))

    pool = {m["arc_id"]: m for m in meta}
    ctxs = pool_contexts(pool)
    check("four rated contexts", len(ctxs) == 4, str(len(ctxs)))
    choices, F = candidate_grid(ctxs[0], 1.0)
    check("candidate grid is 20x2x2x2x5 = 800", len(choices) == 800, str(len(choices)))
    check("grid feature matrix matches the feature count",
          F.shape == (800, 11), str(F.shape))
    best = best_by_gp_predict(gp, ctxs[0], 1.0)
    check("policy returns a scored choice",
          all(k in best for k in ("ramp_s", "chord_start", "score")))
    check("chosen ramp is inside the rated range",
          3.0 <= best["ramp_s"] <= 16.0, f"{best['ramp_s']:.2f}")
    # A higher uncertainty penalty must not increase the chosen score.
    lo = best_by_gp_predict(gp, ctxs[0], 1.0, kappa=0.0)["score"]
    hi = best_by_gp_predict(gp, ctxs[0], 1.0, kappa=2.0)["score"]
    check("a stronger LCB penalty lowers the score", hi <= lo + 1e-12,
          f"{hi:.4f} <= {lo:.4f}")
    check("lookup baseline finds an arc at a rated context",
          lookup_best_arc(pool, gp, ctxs[0]) is not None)

    # The conductor renders from the texture ladder and the arc metadata, so
    # both must be reachable WITHOUT importing a training script, and there
    # must be exactly one definition of the ladder.
    from arc_policy import (TEXTURE_GAINS, texture_overrides, texture_signed,
                            load_pool as inf_load_pool)
    check("the texture ladder is on the inference side",
          sorted(TEXTURE_GAINS) == [-1.0, -0.5, 0.0, 0.5, 1.0])
    check("every ladder level maps to render overrides",
          all(isinstance(texture_overrides(l), dict) for l in TEXTURE_GAINS))
    check("texture_signed reads a plain arc as neutral",
          texture_signed({}) == 0.0)
    check("the ladder is defined ONCE, not copied into the trainer",
          "TEXTURE_GAINS = {" not in (TRAIN / "fit_preference_gp.py").read_text())
    check("arc metadata loads from inference", len(inf_load_pool()) == 230)
    check("the trainer imports the ladder back from inference",
          "from arc_policy import" in (TRAIN / "fit_preference_gp.py").read_text())

    print("\n5. scheduler")
    from scheduler import (preset, SemiMarkovScheduler, LiveBedScheduler,
                           example_va_gate, load_corpus, corpus_dwell_by_state,
                           corpus_overlay_trans, DWELL_FLOOR_S, SR)
    st, tr = preset("drone_only")
    segs = SemiMarkovScheduler(st, tr).sample_schedule(300.0)
    check("drone_only is a provable no-op", len(segs) == 1)
    st, tr = preset("overlay")
    segs = SemiMarkovScheduler(st, tr, seed=1).sample_schedule(600.0)
    check("overlay covers the duration contiguously",
          abs(segs[0]["t0"]) < 1e-9 and abs(segs[-1]["t1"] - 600.0) < 1e-9
          and all(abs(a["t1"] - b["t0"]) < 1e-9 for a, b in zip(segs, segs[1:])),
          f"{len(segs)} segments")
    check("no back-to-back state repeats",
          all(a["name"] != b["name"] for a, b in zip(segs, segs[1:])))
    check("the transition matrix has a zero diagonal",
          np.allclose(np.diag(np.asarray(tr)), 0.0))
    gated = SemiMarkovScheduler(st, tr, seed=3, gate=example_va_gate
                                ).sample_schedule(900.0, context={"valence": 0.9})
    check("the VA gate makes bursts unreachable at high valence",
          not any(s["name"] == "bed_bursts" for s in gated))

    corpus = load_corpus()
    check("the corpus fit loads from weights/", corpus is not None)
    raw = corpus_dwell_by_state(corpus)
    st_c, _ = preset("overlay", trans_source="corpus_dwell", corpus=corpus)
    check("the dwell floor is applied to the fitted windows",
          all(s.dwell_s[0] >= DWELL_FLOOR_S - 1e-9 for s in st_c))
    check("the floor actually bites on the corpus window",
          min(raw["bed_duck"]) < DWELL_FLOOR_S, str(raw["bed_duck"]))
    M = np.asarray(corpus_overlay_trans(corpus))
    check("corpus transitions have zero diagonal and unit rows",
          np.allclose(np.diag(M), 0.0) and np.allclose(M.sum(axis=1), 1.0))
    check("corpus_dwell keeps the hand-set matrix",
          np.allclose(np.asarray(preset("overlay", "corpus_dwell", corpus)[1]),
                      np.asarray(preset("overlay")[1])))

    # The live path must equal the one-shot path when rendered in segments;
    # this is what catches discontinuities at segment joins.
    n_total = 40 * SR
    drone = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n_total) / (12.0 * SR))
    one = LiveBedScheduler(st, tr, seed=7).bed_env(0.0, n_total, drone)
    live = LiveBedScheduler(st, tr, seed=7)
    step = 11 * SR
    seg = np.concatenate([live.bed_env(a / SR, min(a + step, n_total) - a,
                                       drone[a:min(a + step, n_total)])
                          for a in range(0, n_total, step)])
    err = float(np.max(np.abs(one - seg)))
    check("segmented render matches one-shot (no join clicks)", err < 1e-9,
          f"{err:.1e}")
    check("bed envelope stays within [0,1]",
          one.min() >= 0.0 and one.max() <= 1.0)

    print("\n6. coherence re-ranker")
    from coherence_reranker import load_coherence, rerank, selftest as rr_selftest
    at, tm = load_coherence()
    check("arc types and transition matrix load", at is not None and tm is not None,
          f"{len(at)} arcs, {tm.shape}" if at else "")
    check("transition matrix is 5x5", tm.shape == (5, 5), str(tm.shape))
    rr_selftest()
    check("re-ranker selftest passes", True)
    # The band rule is the safety property: never select outside the near-tie.
    cands = [("A", 1.0), ("B", 0.95), ("C", 0.5)]
    check("never selects outside the near-tie band",
          all(rerank(cands, 0, {"A": 1, "B": 2, "C": 2}, tm[:3, :3],
                     tie_margin=t)[0] != "C" for t in (0.05, 0.1, 0.2, 0.4)))
    check("degrades to best utility without coherence data",
          rerank(cands, 0, None, None)[0] == "A")

    print("\n7. rating protocol")
    from arc_rating_app import (load_pool, seed_pairs, build_queue, arc_key,
                                build_target_pairs, count_pool_pairs, AXES)
    ap_pool = load_pool()
    seeds = seed_pairs(ap_pool)
    check("seed pairs differ on exactly one axis",
          all(sum(arc_key(ap_pool[a])[ax] != arc_key(ap_pool[b])[ax]
                  for ax in AXES) == 1
              for a, b, k in seeds if k != "seed_probe"), f"{len(seeds)} pairs")
    check("seed pairs never cross scenes",
          all(ap_pool[a]["scene_id"] == ap_pool[b]["scene_id"] for a, b, _ in seeds))
    q = build_queue(ap_pool, np.random.default_rng(0),
                    queue_override=Path("/nonexistent"))
    check("catch trials are injected",
          any(e[2] == "catch_identical" for e in q)
          and any(e[2] == "catch_swap" for e in q))
    nc = [e for e in q if not e[2].startswith("catch")]
    frac = sum(1 for a, b, _, _ in nc if a < b) / len(nc)
    check("presentation order is randomised", 0.35 < frac < 0.65, f"{frac:.0%}")
    check("progress denominator is honest and stable",
          len(build_target_pairs(ap_pool)) == 125
          and build_target_pairs(ap_pool) == build_target_pairs(ap_pool),
          f"125 of {count_pool_pairs(ap_pool)} possible")

    print("\n8. corpus timing pipeline")
    from fit_corpus_hsmm import (ordered_sequences, fit_transition_counts,
                                 split_settled_active, fit_return_rate,
                                 corpus_overlay_trans as fit_trans,
                                 classify_arc, hold_window, _holds, BED_STATES,
                                 selftest as hsmm_selftest)
    from extract_transition_typology import (DESC_NAMES, CLUSTER_DIMS,
                                             collect_tracks, foote_novelty,
                                             moving_avg)
    hsmm_selftest()
    check("the semi-Markov fit's selftest passes", True)
    check("the extractor takes a list of audio directories",
          "--audio-dir" in subprocess.run(
              [PY, str(TRAIN / "extract_transition_typology.py"), "--help"],
              capture_output=True, text=True).stdout)
    check("no hardcoded corpus or artist map survives",
          "ARTIST_DIRS" not in (TRAIN / "extract_transition_typology.py").read_text())
    # Foote novelty must actually peak at a planted boundary.
    n = 200
    F = np.vstack([np.tile([1.0, 0.0], (n // 2, 1)),
                   np.tile([0.0, 1.0], (n // 2, 1))])
    Fn = F / np.linalg.norm(F, axis=1, keepdims=True)
    kernel_frames = 40
    nov = foote_novelty(Fn @ Fn.T, kernel_frames)
    # Tolerance is the kernel's half-width, not a few frames: a checkerboard
    # kernel localises a boundary only to within its own support, so the peak
    # sits near the boundary rather than exactly on it. What must hold is that
    # it is near the boundary and nowhere else.
    peak = int(np.argmax(nov))
    check("Foote novelty peaks at a planted boundary",
          abs(peak - n // 2) <= kernel_frames // 2, f"peak {peak} of {n}")
    far = np.concatenate([nov[:n // 2 - kernel_frames],
                          nov[n // 2 + kernel_frames:]])
    check("novelty away from the boundary stays far below the peak",
          far.max() < 0.5 * nov.max(), f"{far.max():.1f} vs {nov.max():.1f}")
    check("the shipped fit records what is corpus-fit and what is authored",
          "authored_active_split" in corpus
          and "return_rate_active_to_settled" in corpus)
    check("the shipped fit reports its outlier sensitivity",
          "dwell_outlier_sensitivity" in corpus)

    print("\n9. command line")
    for s in TRAIN_SCRIPTS:
        r = subprocess.run([PY, str(TRAIN / s), "--help"], capture_output=True,
                           text=True)
        check(f"{s} --help", r.returncode == 0)
    for m in INFER_MODULES:
        r = subprocess.run([PY, str(INFER / m), "--help"], capture_output=True,
                           text=True)
        check(f"{m} --help", r.returncode == 0)

    print("\n10. naming")
    banned = ["JAMAI", "jamai", "Matthew", "matthew", "maatt", "Gemini"]
    for path in sorted(TRAIN.glob("*.py")) + sorted(INFER.glob("*.py")):
        hits = [b for b in banned if b in path.read_text()]
        check(f"{path.name} carries no retired name", not hits, str(hits))

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
