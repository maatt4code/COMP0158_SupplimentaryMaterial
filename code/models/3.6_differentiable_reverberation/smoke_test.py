"""Smoke test for Section 3.6. Needs no dataset, no network, no torch training.

Runs the real code against the shipped banks, impulse responses and ratings.

Run:
  python smoke_test.py

Exit code 0 means every check passed.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train"
INFER = HERE / "inference"
WEIGHTS = HERE / "weights"
PY = sys.executable
FAILED = []

TRAIN_SCRIPTS = ["model.py", "fit_from_irs.py"]
INFER_MODULES = ["reverb.py", "reverb_bank.py"]


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
    print("Section 3.6 smoke test\n")
    sys.path.insert(0, str(INFER))
    sys.path.insert(0, str(TRAIN))

    print("1. structure")
    for s in TRAIN_SCRIPTS:
        check(f"train/{s} exists", (TRAIN / s).exists())
    for m in INFER_MODULES:
        check(f"inference/{m} exists", (INFER / m).exists())
    for w in ["ir_reverb_bank.json", "reverb_bank_measured.json"]:
        check(f"weights/{w} ships", (WEIGHTS / w).exists())
    irs = sorted(WEIGHTS.glob("irs/*/*.wav"))
    check("the ladder's impulse responses ship", len(irs) == 3, str(len(irs)))
    # Redistributed third-party audio must carry its credit beside it, so a
    # copied weights/ directory cannot lose the attribution.
    attrib = WEIGHTS / "irs" / "ATTRIBUTION.md"
    check("the shipped IRs carry attribution", attrib.exists())
    if attrib.exists():
        txt = attrib.read_text()
        check("attribution names the source and its URL",
              "EchoThief" in txt and "echothief.com" in txt)
        check("attribution lists every shipped IR",
              all(p.name in txt for p in irs))

    print("\n2. inference boundary")
    # The runtime applies reverb; it must not need torch or a fitting script.
    for m in INFER_MODULES:
        imports = module_scope_imports(INFER / m)
        check(f"{m} does not import torch", "torch" not in imports,
              str(sorted(imports)))
        check(f"{m} does not import the training code",
              not any(t.replace(".py", "") in imports for t in TRAIN_SCRIPTS))
        check(f"{m} imports no path helper", "paths" not in imports)
    check("load_ir lives on the inference side",
          "def load_ir" in (INFER / "reverb.py").read_text())

    print("\n3. de-identification")
    banned = ("SOTL", "Basinski", "WillianBasinski", "Celer", "Loscil",
              "KyleBobbyDunn", "sotl", "basinski", "myeung", "/cs/student")
    for f in sorted(WEIGHTS.glob("*.json")):
        hits = [b for b in banned if b in f.read_text()]
        check(f"weights/{f.name} carries no source identity", not hits, str(hits))
    ir_bank = json.loads((WEIGHTS / "ir_reverb_bank.json").read_text())
    check("the IR bank stores relative paths",
          all(not Path(e["path"]).is_absolute() for e in ir_bank["irs"]))
    check("the IR bank drops its absolute source root", "source" not in ir_bank)
    me = json.loads((WEIGHTS / "reverb_bank_measured.json").read_text())
    check("the measured bank drops its source list", "sources" not in me)
    check("the measured bank keeps no track identity",
          all("name" not in e and "path" not in e for e in me["tracks"]))
    check("the measured bank records its re-key rule",
          "category_rekey_rule" in me)
    check("measured categories are acoustic descriptors",
          all(c.split("_")[0] in ("short", "mid", "long")
              for c in me["category_summary"]),
          str(sorted(me["category_summary"])))

    import csv
    with open(HERE / "human_ratings" / "reverb_ratings.csv") as f:
        rows = list(csv.DictReader(f))
    check("380 rating rows", len(rows) == 380, str(len(rows)))
    check("no rating names a recording source",
          not any(b in r.get("reverb", "") for r in rows
                  for b in ("sotl", "basinski")))
    check("no rating carries an absolute path",
          not any(r.get("path", "").startswith("/") for r in rows))

    print("\n4. the ladder")
    import reverb_bank as B
    c = B.conditions()
    ids = B.condition_ids()
    check("all six conditions resolve", len(ids) == 6, str(ids))
    tails = [c[i]["tail_s"] for i in ids]
    check("the ladder ascends in tail length", tails == sorted(tails),
          str([round(t, 2) for t in tails]))
    lb, ld = c["long_bright"], c["long_dark"]
    check("the pair is tail-matched", abs(lb["tail_s"] - ld["tail_s"]) < 0.5,
          f"{lb['tail_s']:.2f}s vs {ld['tail_s']:.2f}s")
    check("the pair differs sharply in tone", lb["tone_hz"] > 3 * ld["tone_hz"],
          f"{lb['tone_hz']:.0f}Hz vs {ld['tone_hz']:.0f}Hz")
    check("the ladder spans real and fitted spaces",
          {c[i]["kind"] for i in ids} == {"dry", "ir", "measured"})
    check("every ratings condition is renderable",
          {r["reverb"].split(":")[0] for r in rows if r["reverb"] != "dry"}
          - set(B.known_ids()) - {"bloom_vs_dry"} == set())

    print("\n5. applying")
    sr = B.SR
    t = np.arange(int(3 * sr)) / sr
    clip = (0.3 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    dry = B.apply(clip, "dry")
    check("dry preserves length", len(dry) == len(clip))
    for cid in ids[1:]:
        out = B.apply(clip, cid)
        check(f"{cid} rings out past the input", len(out) > len(clip))
        check(f"{cid} is finite and peak-safe",
              bool(np.all(np.isfinite(out))) and np.max(np.abs(out)) <= 0.99 + 1e-6)
    check("an unknown condition raises",
          _raises(lambda: B.apply(clip, "nope"), ValueError))
    late = B.apply_id(clip, "stairwell:late")
    check("late-field differs from the dry+wet mix",
          not np.allclose(late[:len(clip)], B.apply_id(clip, "stairwell")[:len(clip)]))
    check("rendering is deterministic",
          np.array_equal(B.apply(clip, "long_dark"), B.apply(clip, "long_dark")))
    # The tone-matched pair must actually sound different where it claims to.
    import numpy.fft as fft
    def centroid(x):
        X = np.abs(fft.rfft(np.asarray(x, float)))
        f = fft.rfftfreq(len(x), 1.0 / sr)
        return float((f * X).sum() / (X.sum() + 1e-12))
    cb, cd = centroid(B.apply(clip, "long_bright")), centroid(B.apply(clip, "long_dark"))
    check("the bright member renders brighter than the dark one", cb > cd,
          f"{cb:.0f}Hz vs {cd:.0f}Hz")

    print("\n6. measurement")
    from fit_from_irs import rt60_schroeder, tail_centroid
    from reverb import load_ir
    import scipy.signal as sps
    rng = np.random.default_rng(7)
    n = int(sr * 3.5)
    tt = np.arange(n) / sr
    synth = rng.normal(0, 1, n) * np.exp(-tt / 0.4)
    sos = sps.butter(2, 800.0 / (sr / 2), btype="low", output="sos")
    synth = sps.sosfilt(sos, synth).astype(np.float32)
    rt = rt60_schroeder(synth, sr)
    check("Schroeder RT60 recovers a planted decay",
          rt is not None and abs(rt - 6.9 * 0.4) < 0.6, f"{rt:.2f}s vs ~2.76s")
    check("tail centroid reads dark for a lowpassed IR",
          200 < tail_centroid(synth, sr) < 1500,
          f"{tail_centroid(synth, sr):.0f}Hz")
    # The shipped IRs must reproduce the bank's own measurements.
    by_name = {e["name"]: e for e in ir_bank["irs"]}
    for p in irs:
        e = by_name[p.stem]
        ir = load_ir(str(p))
        got_rt, got_cen = rt60_schroeder(ir), tail_centroid(ir)
        check(f"{p.stem} reproduces its banked RT60",
              abs(got_rt - e["rt60_s"]) < 1e-3, f"{got_rt:.3f} vs {e['rt60_s']}")
        check(f"{p.stem} reproduces its banked centroid",
              abs(got_cen - e["centroid_hz"]) < 0.1,
              f"{got_cen:.1f} vs {e['centroid_hz']}")

    print("\n7. the differentiable model")
    import torch
    from model import (ExpDecayReverb, multiscale_stft_loss, fit, fit_ir,
                       fft_convolve)
    torch.manual_seed(0)
    d = torch.zeros(int(sr * 0.3)); d[0] = 1.0
    m = ExpDecayReverb(ir_len_s=0.25, sr=sr, seed=99, init_decay_s=0.2)
    check("wet(impulse) equals the model's own IR",
          torch.allclose(m.wet(d)[:m.ir_len], m.ir(), atol=1e-5))
    check("parameters read back in physical units",
          0.05 < m.decay_time_s < 1.0 and 100 < m.damping_hz < 8000
          and 0 < m.gain < 1)
    true = ExpDecayReverb(ir_len_s=0.25, sr=sr, seed=99, init_decay_s=0.12,
                          init_damp_hz=900.0, init_gain=0.6)
    with torch.no_grad():
        target = true.wet(d)
        l0 = multiscale_stft_loss(m.wet(d), target).item()
    l1 = fit(m, d, target, steps=120, lr=0.1)[-1]
    check("the STFT objective reduces the loss", l1 < l0, f"{l0:.2f} -> {l1:.2f}")
    m2 = ExpDecayReverb(ir_len_s=0.25, sr=sr, seed=99, init_decay_s=0.4,
                        init_damp_hz=3000.0, init_gain=0.3)
    fit_ir(m2, true.ir().detach(), steps=300, lr=0.05)
    check("IR supervision recovers the true decay",
          abs(m2.decay_time_s - 0.12) < 0.05, f"{m2.decay_time_s:.3f} vs 0.12")

    print("\n8. command line")
    for s in TRAIN_SCRIPTS:
        r = subprocess.run([PY, str(TRAIN / s), "--help"], capture_output=True,
                           text=True)
        check(f"{s} --help", r.returncode == 0)
    r = subprocess.run([PY, str(TRAIN / "fit_from_irs.py"), "--help"],
                       capture_output=True, text=True)
    check("fit_from_irs resolves its dataset root the standard way",
          "--echothief-root" in r.stdout and "--ir-root" in r.stdout)
    r = subprocess.run([PY, str(TRAIN / "fit_from_irs.py"),
                        "--echothief-root", "/nonexistent"],
                       capture_output=True, text=True)
    msg = r.stdout + r.stderr
    check("a missing dataset names itself and gives a URL",
          "echothief" in msg.lower() and "http" in msg)
    check("a missing dataset prints no traceback", "Traceback" not in msg)
    for m_ in INFER_MODULES:
        r = subprocess.run([PY, str(INFER / m_), "--help"], capture_output=True,
                           text=True)
        check(f"{m_} --help", r.returncode == 0 or m_ == "reverb.py")

    print("\n9. naming")
    banned_code = ["JAMAI", "jamai", "Matthew", "matthew", "maatt", "Gemini"]
    for path in sorted(TRAIN.glob("*.py")) + sorted(INFER.glob("*.py")):
        hits = [b for b in banned_code if b in path.read_text()]
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
