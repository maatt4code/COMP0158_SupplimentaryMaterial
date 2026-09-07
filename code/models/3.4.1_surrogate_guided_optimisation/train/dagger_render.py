"""DAgger round: render the mapper's own outputs so the judge can label them.

Step 6 of Section 3.4.1, and the fix for a distribution gap that the cycle
evaluation exposes.

The proxy is trained on preset-bank clips, whose theta was drawn at random. At
mapper-training time it scores MAPPER renders, which occupy a quite different
region of theta space -- a distribution it has never seen. The measured cost of
that gap is a real-judge arousal bias of about +0.38: the mapper is systematically
rewarded for audio the proxy misreads.

The classic remedy is DAgger. Render the current mapper's outputs at random
targets, label those with the real judge, add them to the proxy's training set,
and retrain. The proxy then becomes accurate where it is actually queried.

Output (in --out-dir):
  dagger_XXXXX.wav   the mapper's renders at random reachable targets
  theta_index.csv    filename, the requested (v_target, a_target), and theta

Run the full round:
  python dagger_render.py --n 1000 --out-dir ../data/dagger_r1
  python label_dataset.py --data-dir ../data/dagger_r1
  python train_judge_proxy.py --data-dir ../data/chord10k ../data/mixed10k ../data/dagger_r1
  python train_closed_loop.py --bank-dirs ../data/chord10k

Previous: train_closed_loop.py
Next:     train_judge_proxy.py, with this directory added
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

_HERE = Path(__file__).resolve()
_SECTION = _HERE.parents[1]
_CODE = _HERE.parents[3]
sys.path.insert(0, str(_CODE / "common"))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_CODE / "models" / "3.3_drone_synthesis_and_nsynth_prior"
                       / "inference"))
from device import get_device                                   # noqa: E402
from ddsp_synth import DifferentiableDDSPSynth                   # noqa: E402
from theta_render import render_theta, polish, SR, N_HARMONICS   # noqa: E402
from evaluate_cycle_consistency import make_mlp_inverse          # noqa: E402
from train_closed_loop import V_RANGE, A_RANGE                   # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Render the closed-loop mapper's outputs for a DAgger round.")
    ap.add_argument("--n", type=int, default=1000, help="clips to render")
    ap.add_argument("--duration", type=float, default=10.0,
                    help="clip seconds; matches the preset bank, which the "
                         "judge and proxy both crop to 9 s")
    ap.add_argument("--out-dir", dest="out_dir", default=None, metavar="DIR",
                    help="output directory (default ../data/dagger_r1)")
    ap.add_argument("--ckpt", default=None,
                    help="mapper checkpoint (default ../weights/closed_loop_mapper.pt)")
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _SECTION / "data" / "dagger_r1"
    ckpt = args.ckpt or _SECTION / "weights" / "closed_loop_mapper.pt"

    device = get_device()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)   # the noise floor is drawn inside the synth
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same loading path as the cycle evaluation, sized from the checkpoint.
    inverse_fn = make_mlp_inverse(device, ckpt)
    synth = DifferentiableDDSPSynth(sample_rate=SR).to(device)
    n_samples = int(args.duration * SR)
    print(f"Mapper     : {ckpt}")
    print(f"Output     : {out_dir}")

    rows = []
    for i in range(args.n):
        # Random targets over the region train_closed_loop.py trains on.
        v = float(rng.uniform(*V_RANGE))
        a = float(rng.uniform(*A_RANGE))
        theta = inverse_fn(v, a, 0, rng)

        audio = polish(render_theta(synth, theta, n_samples, device, rng))
        fname = f"dagger_{i:05d}.wav"
        sf.write(out_dir / fname, audio.astype(np.float32), SR)

        row = {"filename": fname, "v_target": v, "a_target": a,
               **{k: theta[k] for k in theta if k != "harm_dist"},
               "tilt": theta.get("tilt", float("nan"))}
        for k in range(N_HARMONICS):
            row[f"h{k+1:02d}"] = float(theta["harm_dist"][k])
        rows.append(row)

        if (i + 1) % 100 == 0:
            print(f"  rendered {i+1}/{args.n}")

    pd.DataFrame(rows).to_csv(out_dir / "theta_index.csv", index=False)
    print(f"\nRendered {args.n} mapper clips to {out_dir}")
    print(f"Next: python label_dataset.py --data-dir {out_dir}")
    print("then retrain the proxy with this directory added to --data-dir.")


if __name__ == "__main__":
    main()
