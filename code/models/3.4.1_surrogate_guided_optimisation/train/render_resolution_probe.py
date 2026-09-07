"""Render the harmonic-resolution probe: does the frozen judge hear tension?

Supporting experiment for Section 3.4.1. The static-drone ablations over harmony
and timbre both came back flat, which leaves one hypothesis by elimination: that
valence needs TIME -- movement, not a held chord. If that were true, the whole
approach would have to pivot from static theta to a theta(t) trajectory. This
probe tests it directly, and ``cross_domain_validity.py`` scores the result.

Five conditions, each ending on the same held endpoint as its static twin:

  static_minor   root plus minor third, held
  static_major   root plus major third, held        <- the known valence ceiling
  resolution     minor -> major, gain crossfade     <- a real harmonic release
  de_resolution  major -> minor, gain crossfade     <- directional control
  glide          minor -> major, third pitch glides 3 -> 4 semitones

The design is a paired test, and the pairing is what makes it worth running:

  - N variants per condition (default 15), because valence varies about 0.2 std
    from clip to clip, so a single example per condition cannot decide anything.
  - Within a variant every condition shares the SAME base voicing -- f0, timbre
    frame, swell -- and the SAME f0-wander realisation, reseeded per condition.
    The only thing that differs is the harmony and its trajectory.
  - The resolution is a gain crossfade between a fixed in-tune minor third and a
    fixed major third, so the tension comes from the one-semitone overlap and
    the release is to a clean major. That is a harmonic resolution, not a
    microtonal pitch slide.
  - de_resolution is the control that separates DIRECTION from mere movement. If
    resolution beats static_major and de_resolution does not, the judge hears
    direction; if both move together, it is only responding to modulation.
  - Rendering goes through Section 3.3's real render path, so every clip is
    in-distribution for the judge.

Output (in --out-dir):
  <condition>_<variant>.wav   75 clips at the default 15 variants
  theta_index.csv             filename, condition, variant, base voicing

Run:
  python render_resolution_probe.py --out-dir ../data/resolution_probe
  python label_dataset.py --data-dir ../data/resolution_probe
  python render_resolution_probe.py --mode analyze --out-dir ../data/resolution_probe

Next: cross_domain_validity.py, which scores these clips with three judges.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_SECTION = _HERE.parents[1]
_CODE = _HERE.parents[3]
_S33 = _CODE / "models" / "3.3_drone_synthesis_and_nsynth_prior"
sys.path.insert(0, str(_CODE / "common"))
sys.path.insert(0, str(_S33 / "inference"))

T0, T1 = 2.0, 6.0        # crossfade window: minor 0-2 s, move 2-6 s, hold 6-10 s
CLIP_SEC = 10.0
THIRD_GAIN = 0.8         # third-voice level, loud enough to be clearly harmonic
CONDITIONS = ["static_minor", "static_major", "resolution", "de_resolution", "glide"]


def render_all(out_dir: Path, n_variants: int, prior_path: Path, seed: int):
    import torch
    import soundfile as sf
    from device import get_device
    from ddsp_synth import DifferentiableDDSPSynth
    from theta_render import theta_to_synth_inputs, polish, SR

    if not prior_path.exists():
        raise SystemExit(
            f"\ntimbre prior not found at:\n    {prior_path}\n\n"
            "It ships with Section 3.3; pass --prior-path to point elsewhere.\n")

    device = get_device()
    synth = DifferentiableDDSPSynth(sample_rate=SR).to(device)
    n_samples = int(CLIP_SEC * SR)
    t = np.arange(n_samples) / SR
    prior_harm = np.load(prior_path)["harm_dist"]

    R_MIN = 2.0 ** (3.0 / 12.0)
    R_MAJ = 2.0 ** (4.0 / 12.0)
    up = np.interp(t, [T0, T1], [0.0, 1.0])
    down = 1.0 - up
    glide_ratio = 2.0 ** (np.interp(t, [T0, T1], [3.0, 4.0]) / 12.0)

    def voice(base, f0_mult, gain, with_noise, rng):
        """One voice. f0_mult and gain may be scalars or (n_samples,) arrays."""
        th = dict(base)
        th["f0_hz"] = base["f0_hz"] * f0_mult
        if not with_noise:
            th["noise_level"] = 0.0
        f0_t, amp_t, hd_t, noise_t = theta_to_synth_inputs(th, n_samples, device, rng)
        with torch.no_grad():
            a = synth(f0_t, amp_t, hd_t, noise_t).squeeze(0).cpu().numpy()
        return a * gain

    def finish(voices):
        total = np.sum(voices, axis=0)
        total = total / (np.max(np.abs(total)) + 1e-8) * 0.9
        return polish(total)

    def build(cond, base, wander_seed):
        # Reseeding per condition makes the root voice's wander identical across
        # conditions within a variant, removing it as a confound.
        def fresh():
            return np.random.default_rng(wander_seed)
        if cond == "static_minor":
            vs = [voice(base, 1.0, 1.0, True, fresh()),
                  voice(base, R_MIN, THIRD_GAIN, False, fresh())]
        elif cond == "static_major":
            vs = [voice(base, 1.0, 1.0, True, fresh()),
                  voice(base, R_MAJ, THIRD_GAIN, False, fresh())]
        elif cond == "resolution":
            vs = [voice(base, 1.0, 1.0, True, fresh()),
                  voice(base, R_MIN, THIRD_GAIN * down, False, fresh()),
                  voice(base, R_MAJ, THIRD_GAIN * up, False, fresh())]
        elif cond == "de_resolution":
            vs = [voice(base, 1.0, 1.0, True, fresh()),
                  voice(base, R_MAJ, THIRD_GAIN * down, False, fresh()),
                  voice(base, R_MIN, THIRD_GAIN * up, False, fresh())]
        elif cond == "glide":
            vs = [voice(base, 1.0, 1.0, True, fresh()),
                  voice(base, glide_ratio, THIRD_GAIN, False, fresh())]
        else:
            raise ValueError(cond)
        return finish(vs)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)              # the noise floor is drawn inside the synth
    samp = np.random.default_rng(seed)   # samples the per-variant voicings
    rows = []
    for v in range(n_variants):
        base = {
            "harm_dist": prior_harm[samp.integers(0, len(prior_harm))].astype(np.float32),
            "f0_hz": float(np.exp(samp.uniform(np.log(60.0), np.log(160.0)))),
            "swell_rate": float(samp.uniform(0.05, 0.30)),
            "swell_depth": float(samp.uniform(0.20, 0.50)),
            "noise_level": float(samp.uniform(0.03, 0.10)),
            "noise_cutoff_hz": float(samp.uniform(1000.0, 2500.0)),
        }
        wander_seed = 1000 + v           # shared across conditions in this variant
        for cond in CONDITIONS:
            audio = build(cond, base, wander_seed)
            fname = f"{cond}_{v:02d}.wav"
            sf.write(out_dir / fname, audio, SR)
            rows.append({"filename": fname, "condition": cond, "variant": v,
                         "f0_hz": base["f0_hz"], "swell_rate": base["swell_rate"]})
        print(f"variant {v+1}/{n_variants} (f0 {base['f0_hz']:.1f} Hz) "
              f"-> {len(CONDITIONS)} clips")

    idx = pd.DataFrame(rows)
    idx.to_csv(out_dir / "theta_index.csv", index=False)
    print(f"\nWrote {len(idx)} clips and theta_index.csv to {out_dir}")
    print(f"Next: python label_dataset.py --data-dir {out_dir}")


def analyze(out_dir: Path):
    path = out_dir / "labeled_index.csv"
    if not path.exists():
        raise SystemExit(
            f"\nlabelled index not found at:\n    {path}\n\n"
            f"Label the clips first:\n"
            f"    python label_dataset.py --data-dir {out_dir}\n")
    df = pd.read_csv(path)

    order = [c for c in CONDITIONS if c in df["condition"].unique()]
    g = df.groupby("condition")
    print(f"\n=== Per-condition valence and arousal "
          f"(n={g.size().iloc[0]} each) ===")
    for c in order:
        s = g.get_group(c)
        print(f"  {c:14s}  valence {s['valence'].mean():+.3f} ± {s['valence'].std():.3f}"
              f"   arousal {s['arousal'].mean():+.3f} ± {s['arousal'].std():.3f}")

    def paired(cond_a, cond_b):
        """Same variant id means the same base voicing, so this is paired."""
        a = df[df.condition == cond_a].set_index("variant")["valence"]
        b = df[df.condition == cond_b].set_index("variant")["valence"]
        d = (a - b).dropna()
        if len(d) < 2:
            return None
        mean, sd, n = d.mean(), d.std(ddof=1), len(d)
        return mean, sd, n, mean / (sd / np.sqrt(n) + 1e-12), float((d > 0).mean())

    print("\n=== Paired valence differences (same voicing per variant) ===")
    for a, b in [("resolution", "static_major"), ("resolution", "static_minor"),
                 ("glide", "static_major"), ("resolution", "de_resolution"),
                 ("de_resolution", "static_minor")]:
        if a in order and b in order:
            r = paired(a, b)
            if r:
                mean, sd, n, tstat, frac = r
                print(f"  {a:14s} - {b:14s}: dv {mean:+.3f} ± {sd:.3f}  "
                      f"(t≈{tstat:+.2f}, {frac*100:.0f}% of variants positive)")

    if "resolution" in order and "static_major" in order:
        res = g.get_group("resolution")["valence"].mean()
        maj = g.get_group("static_major")["valence"].mean()
        print("\n=== Read ===")
        if res - maj > 0.05:
            print(f"  Resolution beats the static-major ceiling by {res-maj:+.3f}.")
            print("  The frozen judge hears tension and release; a theta(t) "
                  "trajectory mapping is worth pursuing.")
            print("  Confirm the direction: resolution should also beat "
                  "de_resolution.")
        else:
            print(f"  Resolution does not beat static-major (delta {res-maj:+.3f}).")
            print("  The judge appears valence-blind to harmonic movement in "
                  "drones: it scores timbre, not harmony.")
            print("  Next: cross_domain_validity.py, to test whether any "
                  "available judge hears it.")
        print("  (0.05 is a heuristic; weigh it against the spreads above.)")


def main():
    ap = argparse.ArgumentParser(
        description="Render and analyse the harmonic-resolution probe.")
    ap.add_argument("--mode", choices=["render", "analyze"], default="render")
    ap.add_argument("--out-dir", dest="out_dir", default=None, metavar="DIR",
                    help="output directory (default ../data/resolution_probe)")
    ap.add_argument("--n-variants", dest="n_variants", type=int, default=15)
    ap.add_argument("--prior-path", dest="prior_path", default=None,
                    help="timbre prior frames.npz "
                         "(default: Section 3.3's weights/timbre_prior/frames.npz)")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    out_dir = (Path(args.out_dir) if args.out_dir
               else _SECTION / "data" / "resolution_probe")
    if args.mode == "render":
        prior_path = (Path(args.prior_path) if args.prior_path
                      else _S33 / "weights" / "timbre_prior" / "frames.npz")
        render_all(out_dir, args.n_variants, prior_path, args.seed)
    else:
        analyze(out_dir)


if __name__ == "__main__":
    main()
