"""Learn reverb parameters from measured impulse responses.

Step 1 of Section 3.6, and what produces `../weights/ir_reverb_bank.json`.
Point it at a directory tree of impulse-response wavs; the shipped bank was
built from a public library of measured architectural spaces.

For each impulse response:

  1. MEASURE ground truth directly from the recording. RT60 by Schroeder
     backward integration, fitting the -5 to -25 dB span and extrapolating to
     60 dB (a T20 estimate -- the full 60 dB is usually buried in noise), and
     the spectral centroid of the tail as a measure of tone, dark against
     bright.
  2. FIT the interpretable differentiable reverb to the IR by multi-scale STFT
     loss, initialised FROM the measurements above so optimisation starts near
     the answer rather than searching for it.
  3. Write everything to a labelled bank, grouped by the library's own
     category, which is the reverb control space the conductor navigates.

Measuring and fitting are kept separate on purpose. The measurements are
properties of the recording and need no model; the fit is a model's best
attempt to reproduce them. Where a real IR exists the runtime CONVOLVES IT
DIRECTLY, so no fitting error reaches the audio -- the fitted parameters are
used only to choose a representative and to place the space in the control
space.

The impulse-response root resolves the usual way: the flag, then
``$DRONE_ECHOTHIEF``, then the build-host default. A missing dataset fails with
its name and where to obtain it, never a bare traceback.

Run:
  python fit_from_irs.py                                # the resolved root
  python fit_from_irs.py --echothief-root /path/to/irs  # or point it anywhere
  python fit_from_irs.py --pilot                        # 3 IRs, quick check
  python fit_from_irs.py --selftest                     # synthetic IR, no data

Next: ../inference/reverb_bank.py resolves the ladder against the bank.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_SECTION = _HERE.parents[1]
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_SECTION / "inference"))
sys.path.insert(0, str(_HERE.parents[3] / "common"))
import paths                                        # noqa: E402
from reverb import SR, load_ir                      # noqa: E402

DEFAULT_OUT = _SECTION / "data" / "ir_reverb_bank.json"


def rt60_schroeder(ir, sr=SR):
    """RT60 from Schroeder backward integration.

    Fits the -5 to -25 dB span of the energy-decay curve and extrapolates to
    60 dB. Starting at -5 dB avoids the direct sound, and stopping at -25 dB
    stays above the noise floor of a real measurement; extrapolating is
    standard practice precisely because the full 60 dB is rarely visible.
    Returns seconds, or None when the decay is not measurable.
    """
    e = np.asarray(ir, dtype=np.float64) ** 2
    if e.sum() <= 0:
        return None
    edc = np.cumsum(e[::-1])[::-1]
    edc_db = 10.0 * np.log10(edc / edc[0] + 1e-12)
    t = np.arange(len(ir)) / sr

    def cross(db):
        idx = np.where(edc_db <= db)[0]
        return int(idx[0]) if len(idx) else None

    i5, i25 = cross(-5.0), cross(-25.0)
    if i5 is None or i25 is None or i25 <= i5:
        return None
    slope = np.polyfit(t[i5:i25 + 1], edc_db[i5:i25 + 1], 1)[0]   # dB per second
    if slope >= -1e-9:
        return None
    return float(-60.0 / slope)


def tail_centroid(ir, sr=SR):
    """Spectral centroid of the impulse response: the tone of the space."""
    X = np.abs(np.fft.rfft(np.asarray(ir, dtype=np.float64)))
    f = np.fft.rfftfreq(len(ir), 1.0 / sr)
    return float((f * X).sum() / (X.sum() + 1e-12))


def fit_ir_params(ir, sr=SR, steps=250, seed=1234):
    """Fit the interpretable reverb to a measured IR by multi-scale STFT loss.

    Excitation is an impulse, so the model's wet output IS its impulse
    response and the comparison is IR against IR. Initialised from the measured
    RT60 and centroid: RT60 is about 6.9 decay time constants, so dividing gives
    the exponential decay the model parameterises.
    """
    import torch
    from model import ExpDecayReverb, multiscale_stft_loss, fit

    rt60 = rt60_schroeder(ir, sr) or 1.0
    cen = tail_centroid(ir, sr)
    ir_len_s = min(len(ir) / sr, 4.0)
    model = ExpDecayReverb(ir_len_s=ir_len_s, sr=sr, seed=seed,
                           init_decay_s=float(np.clip(rt60 / 6.9, 0.02, 3.0)),
                           init_damp_hz=float(np.clip(cen, 100.0, 7000.0)),
                           init_gain=0.5)
    n = int(ir_len_s * sr)
    dry = torch.zeros(n)
    dry[0] = 1.0
    target = torch.tensor(np.asarray(ir[:n], dtype=np.float32))
    if len(target) < n:
        target = torch.nn.functional.pad(target, (0, n - len(target)))
    with torch.no_grad():
        loss0 = multiscale_stft_loss(model.wet(dry), target).item()
    hist = fit(model, dry, target, steps=steps, lr=0.05)
    return dict(fit_decay_s=round(model.decay_time_s, 3),
                fit_damping_hz=round(model.damping_hz, 1),
                fit_gain=round(model.gain, 3),
                fit_loss0=round(loss0, 3), fit_loss=round(hist[-1], 3))


def scan_irs(root, limit=None):
    """(category, name, path) for every wav under the tree. The category is the
    containing directory's name, so any library organised that way works."""
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"\nnot a directory:\n    {root}\n")
    items = [(p.parent.name, p.stem, str(p)) for p in sorted(root.rglob("*.wav"))]
    if not items:
        raise SystemExit(f"\nno .wav impulse responses under:\n    {root}\n")
    return items[:limit] if limit else items


def build_bank(root, limit=None, steps=250, measure_only=False, out=None,
               verbose=True):
    items = scan_irs(root, limit)
    entries = []
    t0 = time.time()
    for i, (cat, name, path) in enumerate(items):
        ir = load_ir(path)
        if ir.size < SR // 10:
            continue
        e = dict(name=name, category=cat, path=path,
                 rt60_s=(lambda r: round(r, 3) if r else None)(rt60_schroeder(ir)),
                 centroid_hz=round(tail_centroid(ir), 1),
                 ir_len_s=round(len(ir) / SR, 2))
        if not measure_only:
            e.update(fit_ir_params(ir, steps=steps))
        entries.append(e)
        if verbose:
            extra = ("" if measure_only else
                     f"  fit: decay {e['fit_decay_s']}s "
                     f"damp {e['fit_damping_hz']}Hz "
                     f"loss {e['fit_loss0']}->{e['fit_loss']}")
            print(f"[{i+1}/{len(items)}] {cat}/{name}: RT60 {e['rt60_s']}s "
                  f"centroid {e['centroid_hz']:.0f}Hz{extra}", flush=True)

    cats = {}
    for e in entries:
        cats.setdefault(e["category"], []).append(e)
    summary = {c: dict(n=len(v),
                       rt60_median=round(float(np.median(
                           [e["rt60_s"] for e in v if e["rt60_s"]])), 3),
                       centroid_median=round(float(np.median(
                           [e["centroid_hz"] for e in v])), 1))
               for c, v in sorted(cats.items())}
    doc = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), sr=SR,
               n_irs=len(entries), measure_only=measure_only,
               category_summary=summary, irs=entries,
               note=("RT60 is a Schroeder T20 extrapolated to 60 dB; fit_* are "
                     "the interpretable reverb fitted by multi-scale STFT loss. "
                     "Use fit_decay_s and fit_damping_hz as the control space, "
                     "and convolve the raw IR when one exists."))
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2))
    if verbose:
        print(f"\n{len(entries)} IRs in {time.time() - t0:.0f}s"
              + (f" -> {out}" if out else ""))
        for c, s in summary.items():
            print(f"  {c:14} n={s['n']:3d}  RT60~{s['rt60_median']}s  "
                  f"centroid~{s['centroid_median']:.0f}Hz")
    return doc


def selftest():
    """A synthetic IR with a known decay and tone. No dataset."""
    import scipy.signal as sps
    import soundfile as sf
    import tempfile, os

    sr = SR
    rng = np.random.default_rng(7)
    true_decay = 0.4                      # RT60 is about 6.9 * decay
    n = int(sr * 3.5)
    t = np.arange(n) / sr
    ir = rng.normal(0, 1, n) * np.exp(-t / true_decay)
    sos = sps.butter(2, 800.0 / (sr / 2), btype="low", output="sos")
    ir = sps.sosfilt(sos, ir).astype(np.float32)
    ir /= np.sqrt((ir.astype(np.float64) ** 2).sum())

    rt = rt60_schroeder(ir, sr)
    assert rt is not None and abs(rt - 6.9 * true_decay) < 0.6, rt
    print(f"  PASS  Schroeder RT60 {rt:.2f}s recovers the planted "
          f"~{6.9 * true_decay:.2f}s")

    cen = tail_centroid(ir, sr)
    assert 200 < cen < 1500, cen
    print(f"  PASS  tail centroid {cen:.0f}Hz reads dark for an 800Hz-lowpassed IR")

    fitp = fit_ir_params(ir, sr, steps=120)
    assert fitp["fit_loss"] < fitp["fit_loss0"]
    assert 0.15 <= fitp["fit_decay_s"] <= 0.9, fitp["fit_decay_s"]
    print(f"  PASS  fit recovers decay {fitp['fit_decay_s']}s (true "
          f"{true_decay}), loss {fitp['fit_loss0']} -> {fitp['fit_loss']}")

    d = tempfile.mkdtemp()
    p = os.path.join(d, "x.wav")
    sf.write(p, np.concatenate([np.zeros(1000, dtype=np.float32), ir]), sr)
    tr = load_ir(p)
    assert int(np.argmax(np.abs(tr))) < 10, "leading silence must be trimmed"
    assert abs(float((tr.astype(np.float64) ** 2).sum()) - 1.0) < 1e-4, \
        "the loaded IR must be unit energy"
    print("  PASS  loading trims to the direct sound and normalises energy")
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser(
        description="Measure and fit reverb parameters from impulse responses.")
    paths.add_arg(ap, "echothief")
    ap.add_argument("--ir-root", dest="ir_root", default=None, metavar="DIR",
                    help="alias for --echothief-root; any tree of IR wavs")
    ap.add_argument("--out", default=None, help=f"default {DEFAULT_OUT}")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--pilot", action="store_true", help="3 IRs, a quick check")
    ap.add_argument("--measure-only", dest="measure_only", action="store_true",
                    help="skip fitting; report RT60 and centroid only")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    root = args.ir_root or paths.require("echothief", args.echothief_root)
    print(f"Impulse responses: {root}")
    build_bank(root, limit=(3 if args.pilot else args.limit),
               steps=args.steps, measure_only=args.measure_only,
               out=args.out or DEFAULT_OUT)


if __name__ == "__main__":
    main()
