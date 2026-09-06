"""Smoke test for Section 3.3. Needs no dataset.

Checks that the split between training and inference actually holds and that
the render path produces sane audio, using a synthetic timbre prior written to
a temporary directory. It does not check numerical agreement with the shipped
bank: that needs NSynth.

Run:
  python smoke_test.py

Exit code 0 means every check passed.
"""

import subprocess
import sys
import tempfile
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def main():
    import numpy as np

    print("Section 3.3 smoke test\n")
    sys.path.insert(0, str(HERE.parent.parent / "common"))
    sys.path.insert(0, str(HERE / "inference"))

    # 1. the render module must not import anything that samples or trains
    print("1. module boundaries")
    # Check imports, not mentions: the docstrings legitimately name each other.
    import ast
    def imported(path):
        names = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        return names

    ren = imported(HERE / "inference" / "theta_render.py")
    check("theta_render does not import the generator",
          "generate_preset_bank" not in ren, str(sorted(ren)))
    check("theta_render does not import paths", "paths" not in ren)
    src = (HERE / "inference" / "theta_render.py").read_text()
    check("theta_render resolves no dataset root", "paths.resolve" not in src)
    gen = (HERE / "train" / "generate_preset_bank.py").read_text()
    check("generator imports the renderer rather than duplicating it",
          "from theta_render import" in gen)
    check("generator defines no render function",
          "def render_theta" not in gen and "def theta_to_synth_inputs" not in gen)

    # 2. every script exposes --help and its dataset flags
    print("\n2. command-line surface")
    for script, flag in [("train/build_nsynth_prior.py", "--nsynth-root"),
                         ("train/analyse_nsynth_timbre.py", "--out-dir"),
                         ("train/generate_preset_bank.py", "--prior-path")]:
        r = subprocess.run([PY, str(HERE / script), "--help"],
                           capture_output=True, text=True)
        check(f"{script} --help", r.returncode == 0 and flag in r.stdout,
              "" if r.returncode == 0 else r.stderr.strip().splitlines()[-1:])

    # 3. missing data fails with a message, not a traceback
    print("\n3. missing data is reported, not crashed on")
    r = subprocess.run([PY, str(HERE / "train" / "build_nsynth_prior.py"),
                        "--nsynth-root", "/nonexistent"],
                       capture_output=True, text=True)
    check("names the dataset and where to get it",
          "nsynth dataset not found" in r.stderr and "magenta" in r.stderr)
    check("no traceback", "Traceback" not in r.stderr)

    # 4. render path produces sane audio
    print("\n4. render path")
    import torch
    from ddsp_synth import DifferentiableDDSPSynth
    from theta_render import render_theta, polish, chord_voices, SR, _demo_theta
    from device import get_device

    dev = get_device()
    synth = DifferentiableDDSPSynth(sample_rate=SR).to(dev)
    rng = np.random.default_rng(0)
    n = SR  # one second
    a = render_theta(synth, _demo_theta(55.0), n, dev, rng)
    check("length", a.shape == (n,), str(a.shape))
    check("finite", bool(np.isfinite(a).all()))
    check("peak normalised to 0.9", abs(float(np.abs(a).max()) - 0.9) < 1e-3,
          f"{float(np.abs(a).max()):.4f}")
    check("not silent", float(np.sqrt(np.mean(a ** 2))) > 1e-3)

    th = _demo_theta(55.0)
    check("single voice when chord gains are zero", len(chord_voices(th)) == 1)
    th.update(third_gain=0.5, fifth_gain=0.4, octave_gain=0.3)
    check("four voices when all gains are set", len(chord_voices(th)) == 4)

    p = polish(a)
    check("polish keeps length and peak guard",
          p.shape == a.shape and float(np.abs(p).max()) <= 0.95 + 1e-6)

    # Determinism. The rng argument covers the f0 wander only; the noise floor
    # comes from torch.randn inside the synth, so both RNGs must be seeded.
    torch.manual_seed(0)
    d1 = render_theta(synth, _demo_theta(55.0), n, dev, np.random.default_rng(0))
    torch.manual_seed(0)
    d2 = render_theta(synth, _demo_theta(55.0), n, dev, np.random.default_rng(0))
    check("seeding numpy and torch reproduces the render", bool(np.allclose(d1, d2)))
    d3 = render_theta(synth, _demo_theta(55.0), n, dev, np.random.default_rng(0))
    check("seeding numpy alone does not (known, documented)",
          not bool(np.allclose(d1, d3)))

    # 5. generator end to end against a synthetic prior
    print("\n5. generator end to end (synthetic prior, no NSynth needed)")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pr = td / "frames.npz"
        h = np.arange(1, 33, dtype=np.float32) ** -1.2
        np.savez(pr, harm_dist=np.tile(h / h.sum(), (64, 1)).astype(np.float32),
                 pitch=np.full(64, 40, dtype=np.int64))
        out = td / "bank"
        r = subprocess.run([PY, str(HERE / "train" / "generate_preset_bank.py"),
                            "--n", "3", "--duration", "1.0",
                            "--prior-path", str(pr), "--out-dir", str(out)],
                           capture_output=True, text=True)
        check("exit 0", r.returncode == 0,
              "" if r.returncode == 0 else r.stderr.strip().splitlines()[-1:])
        wavs = sorted(out.glob("*.wav"))
        check("wrote 3 wavs", len(wavs) == 3, f"{len(wavs)}")
        idx = out / "theta_index.csv"
        check("wrote theta_index.csv", idx.exists())
        if idx.exists():
            import csv
            rows = list(csv.DictReader(open(idx)))
            cols = set(rows[0])
            check("index has 3 rows", len(rows) == 3)
            check("32 harmonic columns",
                  sum(c.startswith("h") and c[1:].isdigit() for c in cols) == 32)
            for c in ("f0_hz", "swell_rate", "swell_depth", "noise_level",
                      "noise_cutoff_hz", "third_interval", "third_gain",
                      "fifth_gain", "octave_gain"):
                check(f"index has {c}", c in cols)
        if wavs:
            with wave.open(str(wavs[0])) as w:
                check("wav is 16 kHz mono 1.0 s",
                      w.getframerate() == SR and w.getnchannels() == 1
                      and abs(w.getnframes() / SR - 1.0) < 0.01,
                      f"{w.getframerate()} Hz, {w.getnchannels()} ch, "
                      f"{w.getnframes() / w.getframerate():.2f} s")

    print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'All checks passed.'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
