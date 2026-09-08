"""Smoke test for Section 3.9. Needs no audio, no network, no dataset root.

Runs the real analysis code against the shipped pseudonymised ratings and
checks it reproduces the numbers the report states.

Run:
  python smoke_test.py

Exit code 0 means every check passed.
"""

import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train"
RATINGS = HERE / "human_ratings"
INFER = HERE / "inference"
WEIGHTS = HERE / "weights"
PY = sys.executable
FAILED = []

SCRIPTS = ["analyse_space_ratings.py", "build_results.py",
           "analyse_longtrack.py", "analyse_recency.py"]

# Every rater is either a study pseudonym or the anonymous handle the rating
# app minted for a walk-up listener. Anything else is a real identifier.
RATER_RE = re.compile(r"^(R\d{2}|rater_[0-9a-f]{4})$")

# The two reverb conditions that were named after commercial albums. They were
# re-keyed onto acoustic descriptors, and the old ids must not survive
# anywhere, because a stimulus id that no longer joins to its bank is worse
# than useless -- it silently mismatches.
RETIRED_IDS = ["sotl", "basinski"]

# What the report states. See ../../README.md for where each is quoted.
EXPECT_SPACE = dict(rows=760, clean=668, raters=17, tests=8)
EXPECT_LONGTRACK = dict(experimental=255, raters=8, usable=5,
                        h1=-0.431, h2=0.1708, h3=0.5903)
EXPECT_RECENCY = dict(tracks=50, raters=7, rho=[0.3588, 0.456867, 0.744332])


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def module_scope_imports(path):
    """Top-level imports only. A deferred import inside a function is a
    deliberate choice (statsmodels is optional); a module-scope one is a hard
    dependency of merely importing the file."""
    names = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def close(a, b, tol=5e-4):
    return abs(float(a) - float(b)) <= tol


def main():
    print("Section 3.9 -- human evaluation and protocols\n")

    print("1. the analysis scripts stand alone")
    for name in SCRIPTS:
        p = TRAIN / name
        check(f"{name} exists", p.exists())
        if not p.exists():
            continue
        imports = module_scope_imports(p)
        # This section reads ratings that ship beside it. It must never need a
        # dataset root, and it must never reach into another section.
        check(f"{name} does not import paths", "paths" not in imports)

    # An analysis section has no runtime half and fits nothing. Both
    # directories stay, carrying a README, so that a reader can tell a
    # deliberate absence from a packaging slip.
    check("inference/ ships no code, by design",
          not list(INFER.glob("*.py")))
    check("inference/README.md explains why", (INFER / "README.md").exists())
    check("weights/ ships no artefact, by design",
          not [p for p in WEIGHTS.iterdir() if p.name != "README.md"])
    check("weights/README.md explains why", (WEIGHTS / "README.md").exists())

    print("\n2. the shipped ratings are pseudonymous")
    shards = sorted(RATINGS.glob("*.csv"))
    check("rating files ship", len(shards) >= 26, f"{len(shards)} files")
    bad_raters, retired = set(), []
    for f in shards:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        if "rater" in df.columns:
            bad_raters |= {r for r in df.rater.unique() if not RATER_RE.match(r)}
        text = f.read_text()
        retired += [f"{f.name}:{i}" for i in RETIRED_IDS
                    if re.search(rf"(?<![A-Za-z0-9]){i}(?![A-Za-z0-9])", text)]
    check("every rater id is a pseudonym", not bad_raters,
          "" if not bad_raters else str(sorted(bad_raters)[:5]))
    check("no retired album-derived condition id survives", not retired,
          "" if not retired else str(retired[:5]))
    meta = (RATINGS / "longtrack_meta.json").read_text()
    check("the stimulus metadata carries no build-host path",
          "/cs/" not in meta and ":\\" not in meta)

    print("\n3. the component-preference merge")
    sys.path.insert(0, str(TRAIN))
    from analyse_space_ratings import load                    # noqa: PLC0415
    df = load(RATINGS)
    check("the merge reproduces the analysed row count",
          len(df) == EXPECT_SPACE["rows"],
          f"{len(df)} rows (expected {EXPECT_SPACE['rows']})")
    check("...and its rater count", df.rater.nunique() == EXPECT_SPACE["raters"],
          f"{df.rater.nunique()} raters")
    check("...and all eight tabs", df.test.nunique() == EXPECT_SPACE["tests"],
          f"{df.test.nunique()} tests")
    shipped = pd.read_csv(RATINGS / "ratings_all_flagged.csv")
    check("the shipped merged table has the same rows as a fresh merge",
          len(shipped) == len(df), f"{len(shipped)} vs {len(df)}")

    print("\n4. the hygiene pass and its outputs")
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([PY, str(TRAIN / "build_results.py"),
                            "--out-dir", tmp], capture_output=True, text=True)
        check("build_results.py runs", r.returncode == 0,
              r.stderr.strip().splitlines()[-1] if r.returncode else "")
        out = Path(tmp)
        csvs = sorted((out / "data").glob("*.csv"))
        figs = sorted((out / "figs").glob("*.png"))
        check("it writes eleven tables", len(csvs) == 11, f"{len(csvs)} CSVs")
        check("it writes five figures", len(figs) == 5, f"{len(figs)} PNGs")
        meta_p = out / "data" / "build_meta.json"
        if meta_p.exists():
            m = json.loads(meta_p.read_text())
            check("the clean subset is the size the report analysed",
                  m["rows_clean"] == EXPECT_SPACE["clean"],
                  f"{m['rows_clean']} rows")
            check("the recorded snapshot path is section-relative",
                  not Path(m["snapshot_dir"]).is_absolute(), m["snapshot_dir"])
            # The shipped clean table must be that same subset.
            fresh = pd.read_csv(out / "data" / "ratings_clean.csv")
            ship = pd.read_csv(RATINGS / "ratings_clean.csv")
            check("the shipped clean table matches a fresh build",
                  len(fresh) == len(ship), f"{len(fresh)} vs {len(ship)}")

    print("\n5. the long-track contrasts")
    r = subprocess.run([PY, str(TRAIN / "analyse_longtrack.py"), "--selftest"],
                       capture_output=True, text=True)
    check("the contrast machinery recovers planted effects", r.returncode == 0,
          r.stderr.strip().splitlines()[-1] if r.returncode else "")
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([PY, str(TRAIN / "analyse_longtrack.py"),
                            "--out", str(Path(tmp) / "lt.json")],
                           capture_output=True, text=True)
        check("analyse_longtrack.py runs on the shipped ratings",
              r.returncode == 0,
              r.stderr.strip().splitlines()[-1] if r.returncode else "")
        p = Path(tmp) / "lt.json"
        if p.exists():
            res = json.loads(p.read_text())
            m = res["meta"]
            check("it reproduces the analysed rating count",
                  m["n_experimental"] == EXPECT_LONGTRACK["experimental"],
                  f"{m['n_experimental']}")
            check("...and the usable-rater count",
                  m["n_usable_raters"] == EXPECT_LONGTRACK["usable"],
                  f"{m['n_usable_raters']} of {m['n_raters']}")
            for key, tag in [("H1_manipulation_up_vs_down", "h1"),
                             ("H2_transition_presence_arc_vs_noarc", "h2"),
                             ("H3_dissociation_majend_vs_minend", "h3")]:
                got = res[key]["mean_of_paired_diffs"]
                check(f"{tag.upper()} reproduces the reported difference",
                      close(got, EXPECT_LONGTRACK[tag]),
                      f"{got:+.4f} (report {EXPECT_LONGTRACK[tag]:+.4f})")

    print("\n6. the recency result")
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([PY, str(TRAIN / "analyse_recency.py"),
                            "--out", tmp], capture_output=True, text=True)
        check("analyse_recency.py runs", r.returncode == 0,
              r.stderr.strip().splitlines()[-1] if r.returncode else "")
        p = Path(tmp) / "data" / "recency_probe_vs_overall.csv"
        if p.exists():
            res = pd.read_csv(p)
            check("it uses the same 50 completed tracks",
                  int(res.n.iloc[0]) == EXPECT_RECENCY["tracks"],
                  f"n={int(res.n.iloc[0])}")
            ok = all(close(a, b) for a, b in zip(res.rho, EXPECT_RECENCY["rho"]))
            check("rho rises across the three probes to the reported +0.744",
                  ok, " ".join(f"{v:+.3f}" for v in res.rho))
            check("...and it is monotonic, which is the result itself",
                  list(res.rho) == sorted(res.rho))

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
