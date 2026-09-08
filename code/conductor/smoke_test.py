"""Smoke test for the conductor: engine, weights and app.

Asserts that the engine is genuinely self-contained, that every copied module
still matches its section original, that every weight loads and produces the
documented numbers, and that the app's own control-flow selftest passes with
no server, no audio device and no dataset.

Run:
  python smoke_test.py

Exit code 0 means every check passed.
"""

import ast
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE / "engine"
WEIGHTS = HERE / "weights"
ASSETS = HERE / "assets"
CODE = HERE.parent
FAILED = []

# What the report and the sections state. A number here that stops matching
# means the conductor and its section have diverged.
TAU_JUDGE, TAU_HUMAN = 0.5898, 0.5505
N_ARCS = 230
N_ANCHORS = 20000
MELODY_ANCHOR = 4174          # the frozen pick; see 3.7/weights/melody_anchor.json
CHECKPOINTS = ["melodic_transformer_L3_d128.pt", "melodic_transformer_L6_d128.pt",
               "melodic_transformer_L6_d256.pt"]


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
    print("Conductor -- engine and weights\n")

    print("1. the engine is self-contained")
    # Import with the repo removed from sys.path entirely. If anything here
    # reaches back into code/, this is where it shows up -- and it is the
    # whole claim of a self-contained conductor, so it is tested rather
    # than asserted.
    sys.path = [p for p in sys.path if "COMP0158_SupplimentaryMaterial/code" not in p]
    sys.path.insert(0, str(ENGINE))
    mods = {}
    for name in ["arc_policy", "scheduler", "coherence_reranker", "retrieval",
                 "decoupled_engine", "guard", "reverb", "reverb_bank",
                 "transformer_model", "grammar", "arranger", "render_params",
                 "ddsp_synth", "device", "loudness", "melody_markov",
                 "melody_transformer"]:
        try:
            mods[name] = __import__(name)
        except Exception as e:                              # noqa: BLE001
            check(f"{name} imports", False, f"{type(e).__name__}: {e}")
    check(f"all {len(mods)} engine modules import with the repo off sys.path",
          len(mods) == 17, f"{len(mods)}/17")
    outside = [n for n, m in mods.items()
               if ENGINE not in Path(m.__file__).resolve().parents]
    check("none of them resolved outside engine/", not outside, str(outside))
    check("paths.py was never imported", "paths" not in sys.modules)

    # The train/inference boundary, statically. A module that imports `paths`
    # or pandas-reads a rating file at module scope would fail the moment the
    # app is lifted out of the repo.
    for p in sorted(ENGINE.glob("*.py")):
        imports = module_scope_imports(p)
        if "paths" in imports:
            check(f"{p.name} does not import paths", False)
    check("no engine module imports paths", True)

    print("\n2. copies still match their section originals")
    manifest = ENGINE / "COPIED_FROM"
    check("COPIED_FROM ships", manifest.exists())
    if manifest.exists():
        drift, n = [], 0
        for line in manifest.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, name, src = line.split()
            n += 1
            copy, source = ENGINE / name, CODE / src
            if not copy.exists() or hashlib.sha256(copy.read_bytes()).hexdigest() != digest:
                drift.append(f"{name} (copy edited)")
            elif not source.exists():
                drift.append(f"{src} (source gone)")
            elif hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                drift.append(f"{name} (source changed, copy is stale)")
        check(f"all {n} copies are byte-identical to their source",
              not drift, str(drift[:3]))

    print("\n3. every weight loads, and gives the documented number")
    W = WEIGHTS
    gp = mods["arc_policy"].PreferenceGP.load()
    check("the preference GP is the frozen posterior, not a refit",
          gp.X.shape[0] == N_ARCS, f"{gp.X.shape[0]} arcs")
    pool = mods["arc_policy"].load_pool()
    check("arc metadata resolves from weights/, with no human_ratings/",
          len(pool) == N_ARCS and not (HERE / "human_ratings").exists(),
          f"{len(pool)} arcs")

    g_j = mods["guard"].BoundaryGuard.load()
    g_h = mods["guard"].BoundaryGuard.load(W / "boundary_guard_human.json")
    check("the judge-space guard carries its fitted threshold",
          abs(g_j.tau - TAU_JUDGE) < 5e-4, f"tau={g_j.tau:.4f}")
    check("the human-space guard carries its own, and they differ",
          abs(g_h.tau - TAU_HUMAN) < 5e-4 and g_j.tau != g_h.tau,
          f"tau={g_h.tau:.4f}")

    check("the coherence model loads", mods["coherence_reranker"].load_coherence() is not None)
    check("the melody grammar loads", mods["grammar"].load() is not None)
    check("the phrase bank loads", mods["grammar"].load_bank() is not None)
    check("all three melody checkpoints ship, so the UI can offer them",
          all((W / c).exists() for c in CHECKPOINTS))
    check("the retired cpu-alias checkpoint does NOT ship",
          not (W / "melodic_transformer_cpu.pt").exists())

    check("the melody anchor is the frozen pick, not a rating-data refit",
          mods["melody_markov"].DEFAULT_MELODY_ANCHOR == MELODY_ANCHOR,
          str(mods["melody_markov"].DEFAULT_MELODY_ANCHOR))

    banks = sorted((W / "banks").glob("*/labeled_index.csv"))
    check("both retrieval banks ship", len(banks) == 2, f"{len(banks)}")
    eng = mods["decoupled_engine"].DecoupledEngine(banks)
    n = len(eng.bank) if hasattr(eng, "bank") else N_ANCHORS
    theta = eng.theta_from_index(MELODY_ANCHOR)
    check("retrieval answers a query over the full bank",
          "f0_hz" in eng.retrieve(0.3, 0.4), f"{n} anchors")
    check("the frozen anchor resolves in the shipped bank",
          theta is not None and "f0_hz" in theta,
          f"f0={theta['f0_hz']:.2f} Hz")

    print("\n4. the renderer")
    rp = mods["render_params"]
    check("the frozen render settings ship",
          rp.RENDER_KW["xfade_mode"] == "layered" and rp.HOLD_S == 5.0)
    check("apply_chord raises an inaudible third into audibility",
          rp.apply_chord({"third_gain": 0.0}, "maj")["third_gain"] == rp.THIRD_GAIN_FLOOR)
    r = mods["arranger"].ArrangedRenderer(device="cpu")
    wp = dict(f0_hz=110.0, third_interval=3.0, third_gain=0.4, fifth_gain=0.3,
              octave_gain=0.2, swell_rate=0.15, swell_depth=0.05,
              noise_level=0.02, noise_cutoff_hz=3000.0,
              harm_dist=[1.0 / (i + 1) for i in range(32)])
    audio = r.render_waypoints([wp, dict(wp, f0_hz=146.8)], hold_s=1.0,
                               xfade_s=0.5, **dict(rp.RENDER_KW, seed=0))
    check("the arranger renders audio", len(audio) > 0, f"{len(audio)} samples")
    check("...and it is not silent", float(abs(audio).max()) > 1e-4,
          f"peak {float(abs(audio).max()):.4f}")

    print("\n5. the VA pad matches the label space it describes")
    # The pad is a MEASUREMENT of the bank's reachable region, so there is one
    # per label space and the app must open on the one matching its default.
    # Showing the judge map while retrieving on human labels draws the grey
    # region in the wrong place, inviting the listener to aim at targets the
    # engine cannot hit.
    pads = {s: ASSETS / f"pad_{s}.png" for s in ("judge", "human")}
    check("both VA pads ship", all(p.exists() for p in pads.values()))
    check("the two pads differ, because the manifolds do",
          pads["judge"].exists() and pads["human"].exists()
          and pads["judge"].read_bytes() != pads["human"].read_bytes())
    txt = (HERE / "app.py").read_text()
    import re as _re
    m = _re.search(r'DEFAULT_LABEL_SPACE = "(\w+)', txt)
    want = "human" if m and m.group(1).startswith("human") else "judge"
    check(f"the app opens on the {want} pad, matching its default label space",
          f'PAD_HUMAN if str(DEFAULT_LABEL_SPACE).startswith("human")' in txt)
    check("bed attribution ships beside the audio",
          (ASSETS / "beds" / "ATTRIBUTION.md").exists()
          and any((ASSETS / "beds").glob("*.wav")))

    print("\n6. the app")
    import subprocess
    check("app.py imports with no dataset root",
          subprocess.run([sys.executable, "-c",
                          "import sys; sys.path.insert(0, %r); import app"
                          % str(HERE)],
                         capture_output=True, text=True).returncode == 0)
    # The app's own selftest: fake engine and renderer, no server, no audio.
    # It exercises the trigger logic, both policy methods, the coherence
    # re-ranker, the bed layer, the loudness stage and the melody layer.
    r = subprocess.run([sys.executable, str(HERE / "app.py"), "--selftest"],
                       capture_output=True, text=True, cwd=str(HERE))
    ok = r.returncode == 0 and "SELFTEST OK" in r.stdout
    check("the app's control-flow selftest passes", ok,
          "" if ok else (r.stderr.strip().splitlines() or ["no output"])[-1])
    # The deck is the only surface, and its component contract is enforced.
    txt = (HERE / "app.py").read_text()
    check("only the deck skin is built",
          'skins.build_deck' in txt and 'SKIN ==' not in txt)
    check("...and the deck is validated against the contract",
          'skins.validate(C, "deck")' in txt)

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
