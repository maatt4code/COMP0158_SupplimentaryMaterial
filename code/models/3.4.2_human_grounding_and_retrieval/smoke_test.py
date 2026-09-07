"""Smoke test for Section 3.4.2. Needs no dataset, no network, no bank.

Runs the real code against the shipped ratings and fitted weights. What it
cannot check is the propagation numbers, which need the 20,000-preset bank;
README.md records those for a full run.

Run:
  python smoke_test.py

Exit code 0 means every check passed.
"""

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train"
INFER = HERE / "inference"
PY = sys.executable
FAILED = []

TRAIN_SCRIPTS = ["rating_agreement.py", "propagate_labels_krr.py",
                 "fit_control_ridge.py", "fit_guard.py",
                 "extract_pool_mert.py", "train_attention_retrieval.py"]
INFER_MODULES = ["decoupled_engine.py", "retrieval.py", "retrieval_engines.py",
                 "guard.py"]


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def module_scope_imports(path):
    """Only the imports that run at load time."""
    names = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def main():
    print("Section 3.4.2 smoke test\n")
    sys.path.insert(0, str(HERE.parent.parent / "common"))
    sys.path.insert(0, str(INFER))
    sys.path.insert(0, str(TRAIN))

    # 1. structure -------------------------------------------------------
    print("1. structure")
    for s in TRAIN_SCRIPTS:
        check(f"train/{s} exists", (TRAIN / s).exists())
    for m in INFER_MODULES:
        check(f"inference/{m} exists", (INFER / m).exists())
    for w in ["boundary_guard.json", "attn_retrieval.pt",
              "control_ridge_valence.npz"]:
        check(f"weights/{w} ships", (HERE / "weights" / w).exists())

    # 2. the inference boundary ------------------------------------------
    # This section DOES reach the runtime, so the rule bites: nothing in
    # inference/ may resolve a dataset root or import the path helper.
    print("\n2. inference boundary")
    for m in INFER_MODULES:
        imports = module_scope_imports(INFER / m)
        check(f"{m} does not import paths", "paths" not in imports,
              str(sorted(imports)) if "paths" in imports else "")
        src = (INFER / m).read_text()
        check(f"{m} resolves no dataset root",
              "paths.resolve" not in src and "paths.data" not in src
              and "paths.repo_data" not in src)
    check("guard.py needs no sklearn at runtime",
          "sklearn" not in module_scope_imports(INFER / "guard.py"))

    # 3. the shipped ratings are pseudonymous ----------------------------
    print("\n3. shipped ratings")
    import csv
    ratings = HERE / "human_ratings" / "valence_ratings.csv"
    with open(ratings) as f:
        rows = list(csv.DictReader(f))
    raters = sorted({r["rater"] for r in rows})
    check("713 rating rows", len(rows) == 713, str(len(rows)))
    check("raters are R01..R09", raters == [f"R{i:02d}" for i in range(1, 10)],
          str(raters))
    check("no clip path is absolute",
          not any(r["clip_path"].startswith("/") for r in rows))
    meta = json.loads((HERE / "human_ratings" / "pool_meta.json").read_text())
    check("pool metadata has 150 entries", len(meta) == 150, str(len(meta)))
    check("pool metadata carries no absolute path",
          "/cs/" not in json.dumps(meta))

    # 4. rating agreement reproduces the reported alpha -------------------
    print("\n4. inter-rater agreement")
    from rating_agreement import (load_ratings, krippendorff_alpha_ordinal,
                                  PRIMARY_RATER)
    by_rater = load_ratings(ratings)
    check("nine raters", len(by_rater) == 9, str(len(by_rater)))
    check("the primary rater has 150 clips",
          len(by_rater[PRIMARY_RATER]) == 150, str(len(by_rater[PRIMARY_RATER])))
    clips = sorted(set(c for d in by_rater.values() for c in d))
    rs = sorted(by_rater)
    data = np.full((len(rs), len(clips)), np.nan)
    for i, rt in enumerate(rs):
        for j, c in enumerate(clips):
            if c in by_rater[rt]:
                data[i, j] = by_rater[rt][c]
    alpha = krippendorff_alpha_ordinal(data)
    check("Krippendorff alpha is the reported 0.096", abs(alpha - 0.096) < 5e-4,
          f"{alpha:.4f}")

    # 5. the fitted guard --------------------------------------------------
    print("\n5. boundary guard")
    from guard import BoundaryGuard, ou_walk
    g = BoundaryGuard.load()
    check("guard loads from weights/", g.X.shape[1] == 2, str(g.X.shape))
    check("guard carries no rating values",
          "y" not in json.loads((HERE / "weights" / "boundary_guard.json").read_text()))
    check("tau matches the recorded 0.590", abs(g.tau - 0.590) < 1e-3,
          f"{g.tau:.4f}")
    s_in = float(g.sigma(np.array([-0.6, 0.0])))
    s_out = float(g.sigma(np.array([0.95, 0.95])))
    check("sigma is low inside rated territory", s_in < g.tau, f"{s_in:.3f}")
    check("sigma is high outside it", s_out > g.tau, f"{s_out:.3f}")
    drift, info = g.guard_drift(np.array([0.95, 0.95]))
    check("the hard term triggers out of distribution", info["hard_triggered"])
    check("the hard term is capped",
          info["hard_norm"] <= g.hard_step_cap + 1e-9, f"{info['hard_norm']:.4f}")
    drift_in, info_in = g.guard_drift(np.array([-0.6, 0.0]))
    check("the hard term is off inside", not info_in["hard_triggered"])
    guarded, sg = ou_walk(g, (-0.6, 0.0), (0.9, 0.9), steps=120, use_guard=True)
    free, sf = ou_walk(g, (-0.6, 0.0), (0.9, 0.9), steps=120, use_guard=False)
    check("the guard contains the walk", sg.max() < sf.max(),
          f"max sigma {sg.max():.3f} guarded vs {sf.max():.3f} free")

    # 6. the control ridge reconstructs its pipeline ----------------------
    print("\n6. control ridge")
    d = np.load(HERE / "weights" / "control_ridge_valence.npz")
    check("control ridge has scaler and coefficients",
          all(k in d for k in ("scaler_mean", "scaler_scale", "coef", "intercept")))
    check("768-dimensional", d["coef"].shape == (768,), str(d["coef"].shape))
    z = np.load(HERE / "human_ratings" / "pool_mert.npz")
    check("MERT features ship for 150 clips", len(z.files) == 150, str(len(z.files)))
    X = np.stack([z[f] for f in z.files])
    check("features are float64, as the ridge was fitted",
          X.dtype == np.float64, str(X.dtype))
    pred = ((X - d["scaler_mean"]) / d["scaler_scale"]) @ d["coef"] + d["intercept"]
    check("the ridge scores every clip finitely",
          bool(np.isfinite(pred).all()) and len(pred) == 150)

    # 7. retrieval engines on a synthetic bank ----------------------------
    print("\n7. retrieval engines")
    import pandas as pd
    import retrieval_engines as R
    rng = np.random.default_rng(0)
    n = 200
    fake = pd.DataFrame({
        "valence": rng.uniform(-1, 0.3, n), "arousal": rng.uniform(-1, 1, n),
        "f0_hz": rng.uniform(40, 200, n), "swell_rate": rng.uniform(0.01, 2, n),
        "swell_depth": rng.uniform(0.05, 0.9, n),
        "noise_level": rng.uniform(0, 0.4, n),
        "noise_cutoff_hz": rng.uniform(200, 6000, n),
    })
    harm = rng.dirichlet(np.ones(32), n)
    for j in range(32):
        fake[f"h{j+1:02d}"] = harm[:, j]
    with tempfile.TemporaryDirectory() as td:
        idx = Path(td) / "labeled_index.csv"
        fake.to_csv(idx, index=False)
        bank = R.build_bank([str(idx)])
        check("bank builds", bank["n"] == n, str(bank["n"]))
        for name in ("hard", "soft"):
            th = R.make_engine(name, bank).blend(-0.5, 0.2)
            check(f"{name} engine returns a simplex",
                  abs(float(th["harm_dist"].sum()) - 1.0) < 1e-5)
            check(f"{name} engine returns finite scalars",
                  all(np.isfinite(v) for k, v in th.items() if k != "harm_dist"))

        from decoupled_engine import DecoupledEngine
        e = DecoupledEngine([str(idx)])
        t1 = e.retrieve(-0.5, 0.2)
        check("whole-row retrieval reports its anchor", "anchor_idx" in t1)
        t2 = e.retrieve(-0.5, 0.2)
        check("retrieval is deterministic", t1["anchor_idx"] == t2["anchor_idx"])
        # A tiny move must not change the anchor: that is what hysteresis is for.
        t3 = e.retrieve(-0.5 + 1e-4, 0.2)
        check("hysteresis holds the anchor through a tiny move",
              t3["anchor_idx"] == t1["anchor_idx"])

        from retrieval import GPSoftKNNEngine, harmonic_centroid
        ge = GPSoftKNNEngine([str(idx)], gp_subsample=50)
        tb = ge.blend(-0.5, 0.2)
        check("warmth-filtered blend returns a simplex",
              abs(float(tb["harm_dist"].sum()) - 1.0) < 1e-5)
        check("the warmth filter picks a darker-than-median candidate",
              harmonic_centroid(tb["harm_dist"]) < np.median(ge.centroids),
              f"{harmonic_centroid(tb['harm_dist']):.2f} vs median "
              f"{np.median(ge.centroids):.2f}")
        check("uncertainty is higher off-manifold than on",
              ge.get_uncertainty(0.95, 0.95) > ge.get_uncertainty(-0.5, 0.0))
        check("a bank without valence_human disables the human space",
              ge.va_human is None)

    # 8. command line ------------------------------------------------------
    print("\n8. command line")
    for s in TRAIN_SCRIPTS:
        r = subprocess.run([PY, str(TRAIN / s), "--help"],
                           capture_output=True, text=True)
        check(f"{s} --help", r.returncode == 0,
              r.stderr.strip().splitlines()[-1] if r.returncode else "")

    print("\n9. missing input is reported, not crashed on")
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([PY, str(TRAIN / "propagate_labels_krr.py"),
                            "--bank-index", str(Path(td) / "nope.csv")],
                           capture_output=True, text=True)
        msg = r.stdout + r.stderr
        check("missing bank exits non-zero", r.returncode != 0)
        check("missing bank prints no traceback", "Traceback" not in msg)

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
