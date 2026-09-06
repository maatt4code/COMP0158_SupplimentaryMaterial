"""Sample theta from the NSynth timbre prior and render the preset bank.

Step 3 of Section 3.3. Draws a static parameter set per clip, renders it, and
logs every parameter. This is the generation half of the old
``s03_generate_ddsp_dataset.py``; the render half now lives in
``../inference/theta_render.py`` and is imported here.

Per clip, all held constant for the whole clip:
  - harmonic distribution: one frame drawn from the empirical prior
  - f0: one drone pitch, log-uniform in [F0_LO, F0_HI]
  - amplitude: slow swell, rate and depth sampled
  - noise: filtered floor, level and lowpass cutoff sampled
  - chord voices: third interval and three gains, each zeroed with some
    probability so plain single-voice drones stay in the bank

Output:
  <out-dir>/ddsp_XXXXX.wav   16 kHz mono
  <out-dir>/theta_index.csv  one row per clip, filename plus every theta value

Run:
  # quick listen, 20 clips
  python generate_preset_bank.py --n 20 --duration 10 --out-dir /tmp/bank

  # one 10,000-preset subset
  python generate_preset_bank.py --n 10000 --duration 10 --out-dir ../data/chord10k

  # the calm-biased subset that fills the lower-left quadrant
  python generate_preset_bank.py --n 10000 --duration 10 --calm --out-dir ../data/calm10k

Previous: analyse_nsynth_timbre.py
Next:     ../../3.4.1_surrogate_guided_optimisation/train/label_dataset.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch and MKL both ship libiomp

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "common"))
sys.path.insert(0, str(_HERE.parents[1] / "inference"))
from device import get_device                       # noqa: E402
from ddsp_synth import DifferentiableDDSPSynth      # noqa: E402
from theta_render import render_theta, polish, SR, N_HARMONICS  # noqa: E402

# Drone pitch range, roughly C1 to C4. Widened upward after pilot labelling
# showed all-low-register drones cluster in the low-valence, low-arousal
# quadrant; higher registers are needed to reach positive valence.
F0_LO, F0_HI = 32.7, 261.6


def sample_theta(prior_harm: np.ndarray, prior_pitch: np.ndarray,
                 rng: np.random.Generator, calm: bool = False) -> dict:
    """Draw one static parameter set.

    Register matching. The prior was measured at MIDI 40-72, about 82-523 Hz,
    but drones render at 33-131 Hz. Transposing a bright string spectrum down
    one or two octaves packs 32 strong harmonics into a low dense band, which
    buzzes. So frames are drawn only from notes near the bottom of the prior's
    register, and a sampled spectral tilt n^(-tilt) darkens the distribution.

    Tilt is applied to the harmonic distribution here rather than carried as a
    separate coordinate, which is why theta is 41-dimensional and not 42.

    calm=True biases the arousal-driving knobs to their low ends, filling the
    lower-left quadrant that uniform sampling leaves sparse. Only the numeric
    ranges change: the rng call order and count are identical to the default
    path, so default datasets stay bit-reproducible.
    """
    f0_hi = 98.0 if calm else F0_HI                  # calm: C1..G2 only
    f0_hz = float(np.exp(rng.uniform(np.log(F0_LO), np.log(f0_hi))))

    # restrict to prior frames within an octave of the lowest register
    lo_mask = prior_pitch <= (prior_pitch.min() + 12)
    lo_idx = np.flatnonzero(lo_mask)
    harm = prior_harm[lo_idx[rng.integers(len(lo_idx))]].copy()

    # spectral tilt: attenuate harmonic n by n^(-tilt), then renormalise. The
    # range was widened down to 0.0, meaning no darkening, after pilot
    # labelling showed uniformly dark timbres pin valence negative.
    tilt = float(rng.uniform(0.6 if calm else 0.0, 1.5))
    n = np.arange(1, len(harm) + 1, dtype=np.float64)
    harm = harm * n ** (-tilt)
    harm = harm / (harm.sum() + 1e-12)

    # Chord voices, deliberately neutral: whether a major third reads as higher
    # valence is for the affect estimator to decide, not for the sampler to
    # assume. Gains are zeroed with some probability so plain drones remain.
    third_interval = float(rng.choice([3.0, 4.0]))          # minor or major
    third_gain = float(rng.uniform(0.0, 1.0)) if rng.random() > 0.3 else 0.0
    fifth_gain = float(rng.uniform(0.0, 0.8)) if rng.random() > 0.3 else 0.0
    octave_gain = float(rng.uniform(0.0, 0.6)) if rng.random() > 0.4 else 0.0

    return {
        "harm_dist": harm.astype(np.float32),
        "f0_hz": f0_hz,
        "tilt": tilt,
        # Swell, log-uniform from glacial pressure to audible tremolo. The main
        # arousal lever. calm: slow only, at most 0.25 Hz.
        "swell_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.25 if calm else 2.0)))),
        "swell_depth": float(rng.uniform(0.05, 0.6 if calm else 0.9)),
        "noise_level": float(rng.uniform(0.0, 0.12 if calm else 0.4)),
        "noise_cutoff_hz": float(rng.uniform(200.0, 1500.0 if calm else 6000.0)),
        "third_interval": third_interval,
        "third_gain": third_gain,
        "fifth_gain": fifth_gain,
        "octave_gain": octave_gain,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Sample theta from the timbre prior and render the preset bank.")
    ap.add_argument("--n", type=int, default=20, help="number of clips")
    ap.add_argument("--duration", type=float, default=10.0, help="clip seconds")
    ap.add_argument("--out-dir", dest="out_dir", type=str, default=None,
                    help="output directory (default: ../data/preset_bank, "
                         "which is git-ignored)")
    ap.add_argument("--prior-path", dest="prior_path", type=str, default=None,
                    help="frames.npz from analyse_nsynth_timbre.py "
                         "(default: ../weights/timbre_prior/frames.npz)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--polish", action="store_true",
                    help="apply the fixed production chain before saving")
    ap.add_argument("--calm", action="store_true",
                    help="bias theta sampling to the low-arousal end, filling the "
                         "under-sampled calm quadrant")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        _HERE.parents[1] / "data" / "preset_bank"
    prior_path = Path(args.prior_path) if args.prior_path else \
        _HERE.parents[1] / "weights" / "timbre_prior" / "frames.npz"
    if not prior_path.exists():
        raise SystemExit(
            f"\ntimbre prior not found at:\n    {prior_path}\n\n"
            f"Run analyse_nsynth_timbre.py first, or pass --prior-path.\n")

    if args.calm:
        print("CALM mode: arousal-driving knobs biased low")
    print(f"Prior      : {prior_path}")
    print(f"Output dir : {out_dir}")

    device = get_device()
    rng = np.random.default_rng(args.seed)
    # The filtered-noise floor is drawn with torch.randn inside
    # DifferentiableDDSPSynth.synthesize_noise, which takes no generator. Seed
    # the global torch RNG too, otherwise --seed reproduces the theta values
    # and the f0 wander but not the noise, and two runs differ audibly.
    torch.manual_seed(args.seed)
    os.makedirs(out_dir, exist_ok=True)

    prior = np.load(prior_path)
    prior_harm, prior_pitch = prior["harm_dist"], prior["pitch"]
    n_low = int((prior_pitch <= prior_pitch.min() + 12).sum())
    print(f"Prior frames: {len(prior_harm):,} (low-register subset: {n_low:,})")

    synth = DifferentiableDDSPSynth(sample_rate=SR).to(device)
    n_samples = int(args.duration * SR)

    rows = []
    for i in tqdm(range(args.n), desc="Rendering"):
        theta = sample_theta(prior_harm, prior_pitch, rng, calm=args.calm)
        audio = render_theta(synth, theta, n_samples, device, rng)
        if args.polish:
            audio = polish(audio)

        fname = f"ddsp_{i:05d}.wav"
        sf.write(os.path.join(out_dir, fname), audio.astype(np.float32), SR)

        row = {"filename": fname}
        row.update({k: theta[k] for k in
                    ("f0_hz", "tilt", "third_interval", "third_gain", "fifth_gain",
                     "octave_gain", "swell_rate", "swell_depth", "noise_level",
                     "noise_cutoff_hz")})
        for k in range(N_HARMONICS):
            row[f"h{k + 1:02d}"] = float(theta["harm_dist"][k])
        rows.append(row)

    index_path = os.path.join(out_dir, "theta_index.csv")
    pd.DataFrame(rows).to_csv(index_path, index=False)
    print(f"\nDone. {args.n} clips in {out_dir}")
    print(f"Theta index: {index_path}")


if __name__ == "__main__":
    main()
