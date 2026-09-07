"""Smoke test for Section 3.4.1. Needs no dataset, no network, no checkpoint.

Builds synthetic inputs, runs the real code, and exits non-zero on any failure.
What it cannot check is numerical agreement with the reported results: that
needs a labelled preset bank and the two Hugging Face models. See README.md for
the reference numbers a full run reproduces.

Run:
  python smoke_test.py

Exit code 0 means every check passed.
"""

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train"
PY = sys.executable
FAILED = []

SCRIPTS = ["label_dataset.py", "train_cvae.py", "train_judge_proxy.py",
           "train_closed_loop.py", "evaluate_cycle_consistency.py",
           "dagger_render.py", "render_resolution_probe.py",
           "cross_domain_validity.py", "cross_domain_indomain.py"]


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def _names(nodes):
    names = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def imported(path):
    """Every import name bound anywhere in the file, deferred ones included.

    Parsed rather than grepped: docstrings here legitimately name sibling
    modules, and a text search misreads them as imports.
    """
    return _names(ast.walk(ast.parse(path.read_text())))


def imported_at_module_scope(path):
    """Only the imports that run when the module is loaded.

    An import inside a function is deliberately deferred, so it must not count
    against a check that an optional dependency stays optional.
    """
    return _names(ast.parse(path.read_text()).body)


def main():
    import numpy as np

    print("Section 3.4.1 smoke test\n")
    sys.path.insert(0, str(HERE.parent.parent / "common"))
    sys.path.insert(0, str(TRAIN))

    # 1. structure -------------------------------------------------------
    print("1. structure")
    check("inference/ ships no code, by design",
          not list((HERE / "inference").glob("*.py")))
    check("inference/README.md explains why", (HERE / "inference" / "README.md").exists())
    for s in SCRIPTS:
        check(f"train/{s} exists", (TRAIN / s).exists())
    check("weights/affect_ridges.npz ships",
          (HERE / "weights" / "affect_ridges.npz").exists())
    check("data/ is git-ignored but tracked",
          (HERE / "data" / ".gitignore").exists())

    # 2. the frozen judge ------------------------------------------------
    print("\n2. frozen judge heads")
    from frozen_judge import load_ridges, RidgeHead, HEAD_NAMES
    heads = load_ridges()
    check("all three heads load", sorted(heads) == sorted(HEAD_NAMES),
          str(sorted(heads)))
    check("each head is 768-dimensional",
          all(h.coef.shape == (768,) for h in heads.values()))

    # A ridge is a dot product plus a bias. Verify against an independent
    # computation, so a corrupted npz cannot pass silently.
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (8, 768))
    h = heads["deam_valence"]
    check("predict equals X @ coef + intercept",
          np.allclose(h.predict(X), X @ h.coef + h.intercept))

    # The published labels were produced in float32. Reproducing them requires
    # matching that, not improving on it.
    X32 = X.astype(np.float32)
    check("float32 in gives float32 out (matches sklearn)",
          h.predict(X32).dtype == np.float32, str(h.predict(X32).dtype))
    check("float64 in gives float64 out", h.predict(X).dtype == np.float64)

    toy = RidgeHead(np.ones(3), 0.5, "toy")
    check("wrong feature count is rejected",
          _raises(lambda: toy.predict(np.zeros((1, 4))), ValueError))

    # 3. model architectures round-trip ----------------------------------
    print("\n3. architectures")
    import torch
    from train_cvae import InverseCVAE, normalize_scalars, denormalize_scalars, SCALARS
    from train_closed_loop import Mapper, denorm_scalars_torch, BASE_SCALAR_RANGES
    from train_judge_proxy import JudgeProxy

    cvae = InverseCVAE(latent_dim=8, n_scalars=len(SCALARS))
    harm, scal = cvae.decode(torch.zeros(4, 8), torch.zeros(4, 2))
    check("CVAE harm head is a simplex",
          torch.allclose(harm.sum(dim=-1), torch.ones(4), atol=1e-5))
    check("CVAE scalar head is in [0,1]",
          bool((scal >= 0).all() and (scal <= 1).all()))

    mapper = Mapper(n_scalars=len(BASE_SCALAR_RANGES))
    mh, ms = mapper(torch.zeros(4, 2))
    check("mapper harm head is a simplex",
          torch.allclose(mh.sum(dim=-1), torch.ones(4), atol=1e-5))
    check("mapper scalar head is in [0,1]",
          bool((ms >= 0).all() and (ms <= 1).all()))

    proxy = JudgeProxy(size="small")
    out = proxy(torch.zeros(2, 16000 * 2))
    check("proxy maps audio to (B, 2)", tuple(out.shape) == (2, 2), str(tuple(out.shape)))
    n_params = sum(p.numel() for p in proxy.parameters())
    check("proxy is small enough to backprop through", n_params < 1_000_000,
          f"{n_params:,} params")

    # 4. normalisation round-trip ----------------------------------------
    print("\n4. theta normalisation")
    import pandas as pd
    row = {"f0_hz": 55.0, "tilt": 0.5, "swell_rate": 0.1, "swell_depth": 0.3,
           "noise_level": 0.2, "noise_cutoff_hz": 1500.0}
    df = pd.DataFrame([row])
    back = denormalize_scalars(normalize_scalars(df), SCALARS)
    check("normalise then denormalise is the identity",
          all(np.isclose(back[k].item(), v) for k, v in row.items()))

    scal01 = torch.full((1, len(BASE_SCALAR_RANGES)), 0.5)
    d = denorm_scalars_torch(scal01, BASE_SCALAR_RANGES)
    check("torch denormalisation matches the numpy ranges",
          np.isclose(d["f0_hz"].item(), np.exp((np.log(32.7) + np.log(261.6)) / 2)))
    check("log-scaled parameters stay inside their range",
          32.7 <= d["f0_hz"].item() <= 261.6)

    # 5. train/inference boundary ----------------------------------------
    print("\n5. boundaries")
    # Nothing here reaches the runtime, so the rule this section must obey is
    # the reverse of the usual one: no conductor code may import it, and the
    # renderer must come from 3.3 rather than being copied.
    ev = (TRAIN / "evaluate_cycle_consistency.py").read_text()
    check("evaluation imports 3.3's renderer rather than duplicating it",
          "from theta_render import" in ev)
    check("evaluation defines no renderer of its own",
          "def render_theta" not in ev and "def theta_to_synth_inputs" not in ev)
    cl = imported(TRAIN / "train_closed_loop.py")
    check("closed loop imports the proxy definition, not a copy",
          "train_judge_proxy" in cl)
    # 3.4.2 is a forward dependency and must stay optional.
    top_level = imported_at_module_scope(TRAIN / "evaluate_cycle_consistency.py")
    check("3.4.2's engines are not imported at module scope",
          "retrieval_engines" not in top_level, str(sorted(top_level)))
    check("3.4.2's engines are imported somewhere (deferred)",
          "retrieval_engines" in imported(TRAIN / "evaluate_cycle_consistency.py"))

    # 6. every script offers --help and names its flags -------------------
    print("\n6. command line")
    for s in SCRIPTS:
        r = subprocess.run([PY, str(TRAIN / s), "--help"],
                           capture_output=True, text=True)
        check(f"{s} --help", r.returncode == 0,
              r.stderr.strip().splitlines()[-1] if r.returncode else "")
    r = subprocess.run([PY, str(TRAIN / "cross_domain_indomain.py"), "--help"],
                       capture_output=True, text=True)
    check("in-domain script exposes --deam-root", "--deam-root" in r.stdout)
    check("in-domain script exposes --emo-soundscapes-root",
          "--emo-soundscapes-root" in r.stdout)

    # 7. a missing dataset fails cleanly ---------------------------------
    print("\n7. missing data is reported, not crashed on")
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([PY, str(TRAIN / "label_dataset.py"),
                            "--data-dir", str(Path(td) / "nope")],
                           capture_output=True, text=True)
        msg = r.stdout + r.stderr
        check("missing index exits non-zero", r.returncode != 0)
        check("missing index names the file", "theta_index.csv" in msg)
        check("missing index prints no traceback", "Traceback" not in msg)

        r = subprocess.run([PY, str(TRAIN / "cross_domain_indomain.py"),
                            "--deam-root", str(Path(td) / "nope"),
                            "--emo-soundscapes-root", str(Path(td) / "nope")],
                           capture_output=True, text=True)
        msg = r.stdout + r.stderr
        check("missing dataset names the dataset", "deam" in msg.lower())
        check("missing dataset gives a download URL", "http" in msg)
        check("missing dataset prints no traceback", "Traceback" not in msg)

    # 8. training writes outside weights/ by default ----------------------
    print("\n8. shipped weights are not a default output")
    for s in ["train_judge_proxy.py", "train_cvae.py", "train_closed_loop.py"]:
        r = subprocess.run([PY, str(TRAIN / s), "--help"],
                           capture_output=True, text=True)
        line = [ln for ln in r.stdout.splitlines() if "--out-dir" in ln]
        check(f"{s} defaults --out-dir away from weights/",
              "default ../data" in r.stdout,
              line[0].strip() if line else "")

    # 9. naming ----------------------------------------------------------
    print("\n9. naming")
    banned = ["JAMAI", "jamai", "Matthew", "matthew", "maatt", "Gemini"]
    # This file is excluded on purpose: it has to contain the strings it is
    # searching for. Same exemption the repository's leak gate makes for the
    # documents that state the naming rules.
    for path in sorted(TRAIN.glob("*.py")):
        text = path.read_text()
        hits = [b for b in banned if b in text]
        check(f"{path.name} carries no retired name", not hits, str(hits))

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


if __name__ == "__main__":
    sys.exit(main())
