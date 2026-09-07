"""Three valence judges on the same clips: is the flatness the head, or the domain?

Supporting experiment for Section 3.4.1. The resolution probe showed the frozen
judge is flat across static_minor, static_major and resolution, while a human
listener prefers the moving clips. That leaves three possible explanations, and
this script separates them by scoring the SAME clips with three different
valence sources:

  v_deam   MERT-v1-95M embedding -> DEAM ridge head       (the frozen judge)
  v_emo    the SAME embedding    -> Emo-Soundscapes ridge (same features, new head)
  v_aud    audEERING wav2vec2 msp-dim NATIVE valence      (different model and domain)

Reading the result:

  - If a judge scores resolution above static_major and ranks the conditions the
    way a listener does, then drones CAN carry valence and the DEAM head simply
    cannot hear it. Retrain against the judge that can.
  - If all three are flat, no available judge hears harmonic resolution in
    drones. A human listening test becomes the only valid valence axis, and the
    multi-judge blindness is itself the evaluation-methodology contribution.

Inter-judge agreement is reported alongside, because three judges that disagree
with each other are not three independent measurements of one quantity.

The companion script ``cross_domain_indomain.py`` answers the obvious objection
to a null result -- that the drones are simply out of distribution -- by running
these same judges on real audio in their home domains.

Output (in --data-dir):
  cross_judge.csv   the input index plus v_deam, v_emo, v_aud, a_aud

Run:
  python render_resolution_probe.py --out-dir ../data/resolution_probe
  python cross_domain_validity.py --data-dir ../data/resolution_probe

Previous: render_resolution_probe.py
Next:     cross_domain_indomain.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve()
_SECTION = _HERE.parents[1]
sys.path.insert(0, str(_HERE.parents[3] / "common"))
sys.path.insert(0, str(_HERE.parent))
from device import get_device                                     # noqa: E402
from frozen_judge import (load_ridges, load_mert, load_audeering,  # noqa: E402
                          mert_embedding, audeering_logits)

CONDITIONS = ["static_minor", "static_major", "resolution", "de_resolution", "glide"]
JUDGES = ["v_deam", "v_emo", "v_aud"]


def label(data_dir: Path, ridges_path, limit=None) -> pd.DataFrame:
    """Score every clip in ``data_dir`` with all three valence judges."""
    src = data_dir / "labeled_index.csv"
    if not src.exists():
        src = data_dir / "theta_index.csv"
    if not src.exists():
        raise SystemExit(
            f"\nno index found in:\n    {data_dir}\n\n"
            "Render the probe first:\n"
            "    python render_resolution_probe.py --out-dir <dir>\n")
    df = pd.read_csv(src)
    if limit:
        df = df.head(limit).copy()
    print(f"Index      : {src.name}  ({len(df)} clips)")

    device = get_device()
    heads = load_ridges(ridges_path)
    ridge_deam, ridge_emo = heads["deam_valence"], heads["emo_valence"]
    mert_proc, mert = load_mert(device)
    aud_proc, aud = load_audeering(device)

    v_deam, v_emo, v_aud, a_aud = [], [], [], []
    for fname in tqdm(df["filename"], desc="Judging"):
        wav, sr = sf.read(data_dir / fname, always_2d=True)
        wav = torch.tensor(wav.mean(axis=1), dtype=torch.float32).unsqueeze(0)

        emb = mert_embedding(mert_proc, mert, wav, sr, device)
        v_deam.append(float(ridge_deam.predict(emb)[0]) * 2.0 - 1.0)  # [0,1] -> [-1,1]
        v_emo.append(float(ridge_emo.predict(emb)[0]))                # already [-1,1]

        logits = audeering_logits(aud_proc, aud, wav, sr, device)     # [a, d, v]
        v_aud.append(float(logits[2]) * 2.0 - 1.0)
        a_aud.append(float(logits[0]) * 2.0 - 1.0)

    df["v_deam"], df["v_emo"] = v_deam, v_emo
    df["v_aud"], df["a_aud"] = v_aud, a_aud
    out = data_dir / "cross_judge.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    return df


def paired(df, cond_a, cond_b, col):
    """Paired difference on ``col``; variant id pairs the same base voicing."""
    x = df[df.condition == cond_a].set_index("variant")[col]
    y = df[df.condition == cond_b].set_index("variant")[col]
    d = (x - y).dropna()
    m, sd, k = d.mean(), d.std(ddof=1), len(d)
    return m, m / (sd / np.sqrt(k) + 1e-12), float((d > 0).mean())


def analyze(df):
    order = [c for c in CONDITIONS if c in df["condition"].unique()]
    g = df.groupby("condition")
    n = g.size().iloc[0]

    print(f"\n=== Per-condition valence by judge (mean ± std, n={n}) ===")
    print(f"  {'condition':14s} " + "  ".join(f"{j:>16s}" for j in JUDGES))
    for c in order:
        s = g.get_group(c)
        print(f"  {c:14s} " +
              "  ".join(f"{s[j].mean():+6.3f} ± {s[j].std():.3f}" for j in JUDGES))

    print("\n=== Paired dvalence, resolution - static_major, per judge ===")
    hits = []
    for j in JUDGES:
        m, tstat, frac = paired(df, "resolution", "static_major", j)
        if m > 0.05 and abs(tstat) > 2:
            hits.append(j)
            verdict = "HEARS it"
        else:
            verdict = "flat" if abs(m) <= 0.05 else "weak/uncertain"
        print(f"  {j:8s}: d {m:+.3f}  (t≈{tstat:+.2f}, "
              f"{frac*100:.0f}% positive)  -> {verdict}")

    print("\n=== Paired dvalence, resolution - de_resolution (directional) ===")
    for j in JUDGES:
        m, tstat, frac = paired(df, "resolution", "de_resolution", j)
        print(f"  {j:8s}: d {m:+.3f}  (t≈{tstat:+.2f}, {frac*100:.0f}% positive)")

    agreement(df)

    print("\n=== Read ===")
    if hits:
        print(f"  {hits} hear resolution, so drones CAN express valence.")
        print("  Retraining against that judge is the next move.")
    else:
        print("  No judge separates resolution from static_major. No available")
        print("  model hears harmonic resolution in drones, so a human listening")
        print("  test is the only valid valence axis -- and that blindness, with")
        print("  the human-preference divergence, is the contribution.")
        print("  Confirm it is the domain and not the models: "
              "cross_domain_indomain.py")


def agreement(df):
    print(f"\n=== Inter-judge agreement (Pearson r over {len(df)} clips) ===")
    for a, b in itertools.combinations(JUDGES, 2):
        print(f"  r({a}, {b}) = {np.corrcoef(df[a], df[b])[0, 1]:+.3f}")


def main():
    ap = argparse.ArgumentParser(
        description="Score clips with three valence judges and compare them.")
    ap.add_argument("--data-dir", dest="data_dir", default=None, metavar="DIR",
                    help="clip directory holding theta_index.csv or "
                         "labeled_index.csv (default ../data/resolution_probe)")
    ap.add_argument("--mode", choices=["run", "analyze"], default="run",
                    help="run = judge the clips then analyse; "
                         "analyze = re-use a saved cross_judge.csv")
    ap.add_argument("--ridges", default=None, metavar="NPZ",
                    help="override ../weights/affect_ridges.npz")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    data_dir = (Path(args.data_dir) if args.data_dir
                else _SECTION / "data" / "resolution_probe")

    if args.mode == "analyze":
        saved = data_dir / "cross_judge.csv"
        if not saved.exists():
            raise SystemExit(f"\nno saved results at:\n    {saved}\n")
        df = pd.read_csv(saved)
        print(f"Loaded {len(df)} judged clips from {saved}")
    else:
        df = label(data_dir, args.ridges, args.limit)

    if "condition" in df.columns:
        analyze(df)          # the resolution probe
    else:
        agreement(df)        # any other clip set: agreement is still meaningful


if __name__ == "__main__":
    main()
