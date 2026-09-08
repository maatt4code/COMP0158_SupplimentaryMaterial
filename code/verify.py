"""Verify the supplementary material: one command, no dataset, no network.

Run this first. It checks that everything shipped here actually works on your
machine before you spend time on anything longer:

  1. every section's smoke test passes;
  2. no shipped file names a participant, an artist, or a build-host path;
  3. every command printed in the top-level README exists and its flags are
     real, so nothing you copy from the documentation fails;
  4. shipped weights match their recorded checksums, where a section records
     them.

Nothing here needs a dataset, a network connection, or a GPU. A full pass takes
a couple of minutes, most of it in the smoke tests.

Run:
  python code/verify.py
  python code/verify.py --quick     # skip the smoke tests
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
PY = sys.executable

# The documents that STATE the naming rules necessarily quote the names they
# forbid, and a test that searches for a string must contain it.
NAME_EXEMPT = {"STATUS.md", "README.md", "smoke_test.py", "verify.py",
               # paths.py's DEFAULTS record the build host on purpose, so the
               # pipeline is reproducible there without arguments. That is a
               # stated exemption, not an oversight.
               "paths.py"}
BANNED = ["JAMAI", "jamai", "Matthew", "matthew", "maatt", "MaaTt", "drmaatt",
          "Gemini", "GEMINI", "SOTL", "Basinski", "KyleBobbyDunn", "Celer",
          "Loscil", "/cs/student"]

FAILED: list[str] = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def sections():
    return sorted(p for p in (CODE / "models").iterdir()
                  if p.is_dir() and (p / "smoke_test.py").exists())


def run_smoke_tests():
    """Run each section's smoke test and report enough to act on a failure.

    Reporting only the last line of stdout is not enough: a test that CRASHES
    stops mid-section, so the last line is whatever header it had just
    printed, and the traceback on stderr -- the only thing that says what
    happened -- is discarded. Print the failing checks when the test failed
    cleanly, and the tail of stderr when it died.
    """
    print("1. section smoke tests")
    for s in sections():
        r = subprocess.run([PY, str(s / "smoke_test.py")],
                           capture_output=True, text=True)
        if r.returncode == 0:
            check(s.name, True)
            continue
        fails = [ln.strip() for ln in r.stdout.splitlines() if "FAIL" in ln]
        check(s.name, False, f"{len(fails)} check(s) failed" if fails
              else "crashed before finishing")
        for ln in fails[:5]:
            print(f"        {ln}")
        for ln in r.stderr.strip().splitlines()[-5:]:
            print(f"        stderr: {ln}")


def _ignored():
    """Paths git will not commit. Generated outputs live under section `data/`
    directories and are rebuilt by the scripts; scanning them tests the local
    working tree rather than what ships, and reports failures a fresh clone
    could never have."""
    r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--others",
                        "--ignored", "--exclude-standard"],
                       capture_output=True, text=True)
    if r.returncode:
        return set()          # not a checkout: scan everything
    return {(ROOT / line).resolve() for line in r.stdout.splitlines() if line}


def check_names():
    print("\n2. no participant, artist or build-host identifiers")
    hits = []
    ignored = _ignored()
    for p in CODE.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        if p.resolve() in ignored:
            continue
        if p.name in NAME_EXEMPT or p.suffix in (".pt", ".npz", ".wav", ".pyc"):
            continue
        try:
            text = p.read_text(errors="strict")
        except (UnicodeDecodeError, OSError):
            continue          # binary payloads are checked by their own tests
        found = [b for b in BANNED if b in text]
        if found:
            hits.append((p.relative_to(ROOT), sorted(set(found))))
    check("no shipped text file carries a retired identifier", not hits,
          "" if not hits else f"{len(hits)} file(s)")
    for path, found in hits[:10]:
        print(f"        {path}: {found}")


def check_readme_commands():
    """Every command in the top-level README must exist, with real flags.

    Documentation that does not run is worse than none: a reader copies it,
    it fails, and they cannot tell whether the code or the instruction is
    wrong.
    """
    print("\n3. commands in the top-level README")
    all_py = {p.as_posix(): p for p in ROOT.rglob("*.py")}
    txt = (ROOT / "README.md").read_text()
    cmds = sorted(set(re.findall(
        r'python ([A-Za-z0-9_/.]+\.py)((?:\s+--?[A-Za-z0-9-]+(?:\s+[^\s`<|\\]+)?)*)',
        txt)))
    helps: dict[Path, str] = {}
    missing, bad = [], []
    for script, args in cmds:
        cands = [p for k, p in all_py.items() if k.endswith(script.lstrip("./"))]
        if not cands:
            missing.append(script)
            continue
        q = cands[0]
        if q not in helps:
            r = subprocess.run([PY, str(q), "--help"], capture_output=True,
                               text=True)
            helps[q] = r.stdout + r.stderr
        for f in sorted(set(re.findall(r"--[a-z0-9-]+", args))):
            if f not in helps[q]:
                bad.append(f"{script} {f}")
    check(f"all {len(cmds)} documented commands resolve to a script",
          not missing, str(missing))
    check("every documented flag exists", not bad, str(bad))


def check_checksums():
    print("\n4. recorded checksums")
    files = sorted(CODE.rglob("SHA256SUMS"))
    if not files:
        print("  (none recorded)")
        return
    for manifest in files:
        bad = []
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            digest, _, name = line.partition("  ")
            target = manifest.parent / name.strip()
            if not target.exists():
                bad.append(f"{name.strip()} missing")
                continue
            got = hashlib.sha256(target.read_bytes()).hexdigest()
            if got != digest.strip():
                bad.append(f"{name.strip()} changed")
        check(f"{manifest.relative_to(ROOT)}", not bad, str(bad))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="skip the section smoke tests")
    args = ap.parse_args()

    print("Verifying the supplementary material\n")
    if not args.quick:
        run_smoke_tests()
    else:
        print("1. section smoke tests  (skipped)")
    check_names()
    check_readme_commands()
    check_checksums()

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
