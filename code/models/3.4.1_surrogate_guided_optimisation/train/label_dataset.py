"""Attach frozen valence/arousal labels to a rendered preset bank.

Step 1 of Section 3.4.1. Reads a directory written by Section 3.3's
``generate_preset_bank.py`` -- wavs plus ``theta_index.csv`` -- scores every
clip with the frozen judge, and writes ``labeled_index.csv``: the same rows plus
four columns. Those (VA, theta) pairs are the training data for everything
downstream, so this is the step that decides what "the objective" means for the
whole section.

The frozen configuration, which produced every number reported in Section 4.1:

  valence  MERT-v1-95M mean-pooled embedding -> DEAM ridge head, [0,1] -> [-1,1]
  arousal  the SAME MERT embedding           -> Emo-Soundscapes ridge, already [-1,1]

That is this script's default. The alternative arousal source, audEERING's
wav2vec2 regression head, is reachable with ``--arousal-source audeering`` but
was not used for any reported result: its arousal output is too compressed on
drone material. The original script defaulted the other way round, which is
worth knowing when reading the older logs.

Clipped and unclipped labels are both written. About 18% of drone clips drive
the arousal ridge past +1, which is extrapolation; clipping those to +1 censors
their ordering, so ``valence_raw`` / ``arousal_raw`` keep the uncensored value
for calibration work and the proxy targets in ``train_judge_proxy.py``.

Output:
  <data-dir>/labeled_index.csv   theta_index.csv + valence, arousal,
                                 valence_raw, arousal_raw

Run:
  # the frozen config: label a preset bank in place
  python label_dataset.py --data-dir ../data/chord10k

  # score the closed-loop mapper's own renders (the DAgger round)
  python label_dataset.py --data-dir ../data/dagger_r1

Previous: ../../3.3_drone_synthesis_and_nsynth_prior/train/generate_preset_bank.py
Next:     train_cvae.py, or train_judge_proxy.py
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
sys.path.insert(0, str(_HERE.parent))
from device import get_device                                    # noqa: E402
from frozen_judge import (load_ridges, load_mert, load_audeering,  # noqa: E402
                          mert_embedding, audeering_logits)


def main():
    ap = argparse.ArgumentParser(
        description="Label a rendered preset bank with the frozen affect judge.")
    ap.add_argument("--data-dir", dest="data_dir", required=True, metavar="DIR",
                    help="directory holding the wavs and theta_index.csv")
    ap.add_argument("--index-name", dest="index_name", default="theta_index.csv",
                    help="input index (default theta_index.csv)")
    ap.add_argument("--out-name", dest="out_name", default="labeled_index.csv",
                    help="output index (default labeled_index.csv)")
    ap.add_argument("--arousal-source", dest="arousal_source",
                    choices=["emo", "audeering"], default="emo",
                    help="arousal head. 'emo' is the frozen config used for every "
                         "reported result; 'audeering' is the unused alternative")
    ap.add_argument("--ridges", default=None, metavar="NPZ",
                    help="override ../weights/affect_ridges.npz")
    ap.add_argument("--limit", type=int, default=None,
                    help="label only the first N clips (for a quick check)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    index_path = data_dir / args.index_name
    if not index_path.exists():
        raise SystemExit(
            f"\nindex not found at:\n    {index_path}\n\n"
            "Render a preset bank first with Section 3.3's generate_preset_bank.py,\n"
            "or point --data-dir at a directory that has one.\n"
        )

    df = pd.read_csv(index_path)
    if args.limit:
        df = df.head(args.limit).copy()

    device = get_device()
    heads = load_ridges(args.ridges)
    ridge_valence = heads["deam_valence"]
    ridge_arousal = heads["emo_arousal"]

    print(f"Index      : {index_path}  ({len(df)} clips)")
    print(f"Valence    : MERT -> DEAM ridge, [0,1] rescaled to [-1,1]")
    print(f"Arousal    : " + ("MERT -> Emo-Soundscapes ridge, native [-1,1]"
                              if args.arousal_source == "emo"
                              else "audEERING wav2vec2 logits[0], [0,1] rescaled to [-1,1]"))

    mert_proc, mert = load_mert(device)
    aud_proc, aud = (load_audeering(device)
                     if args.arousal_source == "audeering" else (None, None))

    valences, arousals, valences_raw, arousals_raw = [], [], [], []
    for fname in tqdm(df["filename"], desc="Labelling"):
        audio_np, sr = sf.read(data_dir / fname, always_2d=True)
        waveform = torch.tensor(audio_np.mean(axis=1), dtype=torch.float32).unsqueeze(0)

        emb = mert_embedding(mert_proc, mert, waveform, sr, device)
        v01 = float(ridge_valence.predict(emb)[0])

        if args.arousal_source == "emo":
            # Native [-1,1]: the Emo-Soundscapes labels were already signed.
            a_raw = float(ridge_arousal.predict(emb)[0])
        else:
            a_raw = float(audeering_logits(aud_proc, aud, waveform, sr, device)[0]) * 2.0 - 1.0

        v_raw = v01 * 2.0 - 1.0
        valences_raw.append(v_raw)
        arousals_raw.append(a_raw)
        valences.append(float(np.clip(v_raw, -1.0, 1.0)))
        arousals.append(float(np.clip(a_raw, -1.0, 1.0)))

    df["valence"] = valences
    df["arousal"] = arousals
    df["valence_raw"] = valences_raw
    df["arousal_raw"] = arousals_raw

    out_path = data_dir / args.out_name
    df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Valence: mean {np.mean(valences):+.3f}, std {np.std(valences):.3f}, "
          f"range [{np.min(valences):+.3f}, {np.max(valences):+.3f}]")
    print(f"Arousal: mean {np.mean(arousals):+.3f}, std {np.std(arousals):.3f}, "
          f"range [{np.min(arousals):+.3f}, {np.max(arousals):+.3f}]")
    n_clipped = int(np.sum(np.abs(arousals_raw) > 1.0))
    if n_clipped:
        print(f"Note: {n_clipped}/{len(df)} clips ({100*n_clipped/len(df):.0f}%) had "
              f"|arousal| > 1 before clipping; *_raw keeps their ordering.")


if __name__ == "__main__":
    main()
