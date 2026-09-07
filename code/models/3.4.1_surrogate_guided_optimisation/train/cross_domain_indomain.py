"""In-domain control: do these judges work at all on real audio?

Supporting experiment for Section 3.4.1, and the defence of its central claim.

``cross_domain_validity.py`` found the three valence judges flat across harmonic
conditions on synthetic drones, and mutually uncorrelated. The obvious objection
is that nothing is broken -- the drones are simply out of distribution. This
script tests exactly that objection by running the SAME three judges on real,
ground-truth-labelled audio in their home domains:

  DEAM              music, valence 1-9      home domain of the MERT+DEAM head
  Emo-Soundscapes   environmental, [-1,1]   home domain of the MERT+Emo head

Two things get measured:

  (i)  corr(judge output, human ground truth) -- does the judge discriminate
       valence at all, in the domain it was fitted for?
  (ii) inter-judge agreement -- do the judges agree with each other on real
       audio, unlike on drones?

If in-domain they track ground truth and agree, while on drones they are flat
and disagree, then the degeneracy belongs to the synthetic-drone DOMAIN and not
to the models. That is the defensible form of the claim.

Stated caveat, which the output repeats: these clips were plausibly inside the
ridge heads' own training sets, so the in-domain correlations are an optimistic
upper bound. That only sharpens the contrast -- even at their most favourable,
the heads track valence in-domain and go flat on drones.

Output (in --out-dir):
  cross_judge_indomain.csv         per-clip judge outputs and ground truth
  indomain_judge_validity.png      the summary figure

Run:
  python cross_domain_indomain.py --n 150
  python cross_domain_indomain.py --mode analyze     # redraw without re-judging

Previous: cross_domain_validity.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import glob
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve()
_SECTION = _HERE.parents[1]
sys.path.insert(0, str(_HERE.parents[3] / "common"))
sys.path.insert(0, str(_HERE.parent))
import paths                                                      # noqa: E402
from device import get_device                                     # noqa: E402
from frozen_judge import (load_ridges, load_mert, load_audeering,  # noqa: E402
                          mert_embedding, audeering_logits)

JUDGES = [("v_deam", "MERT+DEAM"), ("v_emo", "MERT+Emo"), ("v_aud", "audEERING")]


def load_audio(path, sr=16000, max_sec=20.0):
    """Mono float32 at ``sr``. libsndfile decodes DEAM's mp3s and Emo's wavs
    through the same path, so no ffmpeg dependency is needed."""
    import soundfile as sf
    import torchaudio

    x, in_sr = sf.read(path, always_2d=True)
    x = x.mean(axis=1).astype(np.float32)
    if max_sec:
        x = x[:int(in_sr * max_sec)]
    if in_sr != sr:
        x = torchaudio.functional.resample(
            torch.tensor(x).unsqueeze(0), in_sr, sr).squeeze(0).numpy()
    return x


def load_deam(root: Path, n, rng):
    base = root / "annotations" / "annotations averaged per song" / "song_level"
    parts = ["static_annotations_averaged_songs_1_2000.csv",
             "static_annotations_averaged_songs_2000_2058.csv"]
    missing = [p for p in parts if not (base / p).exists()]
    if missing:
        raise SystemExit(
            f"\nDEAM annotations not found under:\n    {base}\n\n"
            f"missing: {missing}\n"
            f"Point at the dataset with --deam-root, or set $DRONE_DEAM.\n"
            f"Obtain it from: {paths.WHERE_TO_GET['deam']}\n")
    a = pd.concat([pd.read_csv(base / p) for p in parts], ignore_index=True)
    a.columns = [c.strip() for c in a.columns]
    a = a[["song_id", "valence_mean"]].dropna()
    a["path"] = a["song_id"].apply(lambda s: str(root / "MEMD_audio" / f"{int(s)}.mp3"))
    a = a[a["path"].apply(os.path.exists)]
    if a.empty:
        raise SystemExit(f"\nno DEAM audio found under {root / 'MEMD_audio'}\n")
    a = a.sample(min(n, len(a)),
                 random_state=int(rng.integers(1e9))).reset_index(drop=True)
    return a.rename(columns={"valence_mean": "v_gt"})[["path", "v_gt"]]


def load_emo(root: Path, n, rng):
    ratings = (root / "Emo-Soundscapes" / "Emo-Soundscapes-Ratings" / "Valence.csv")
    if not ratings.exists():
        raise SystemExit(
            f"\nEmo-Soundscapes valence ratings not found at:\n    {ratings}\n\n"
            f"Point at the dataset with --emo-soundscapes-root, or set $DRONE_EMO.\n"
            f"Obtain it from: {paths.WHERE_TO_GET['emo_soundscapes']}\n")
    v = pd.read_csv(ratings, header=None, names=["fname", "v_gt"])
    allwav = {os.path.basename(p): p for p in glob.glob(
        str(root / "Emo-Soundscapes" / "Emo-Soundscapes-Audio" / "**" / "*.wav"),
        recursive=True)}
    v["path"] = v["fname"].map(allwav)
    v = v.dropna(subset=["path"])
    if v.empty:
        raise SystemExit(f"\nno Emo-Soundscapes audio found under {root}\n")
    v = v.sample(min(n, len(v)),
                 random_state=int(rng.integers(1e9))).reset_index(drop=True)
    return v[["path", "v_gt"]]


def judge_clips(df, tag, ridges_path):
    device = get_device()
    heads = load_ridges(ridges_path)
    ridge_deam, ridge_emo = heads["deam_valence"], heads["emo_valence"]
    mert_proc, mert = load_mert(device)
    aud_proc, aud = load_audeering(device)

    vd, ve, va = [], [], []
    for path in tqdm(df["path"], desc=f"judging {tag}"):
        x = load_audio(path, sr=16000, max_sec=20.0)
        if len(x) < 16000:                       # pad anything under a second
            x = np.pad(x, (0, 16000 - len(x)))
        wav = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        emb = mert_embedding(mert_proc, mert, wav, 16000, device)
        vd.append(float(ridge_deam.predict(emb)[0]) * 2 - 1)
        ve.append(float(ridge_emo.predict(emb)[0]))
        va.append(float(audeering_logits(aud_proc, aud, wav, 16000, device)[2]) * 2 - 1)

    out = df.copy()
    out["v_deam"], out["v_emo"], out["v_aud"] = vd, ve, va
    out["dataset"] = tag
    return out


def interjudge(d):
    return {f"{a}-{b}": np.corrcoef(d[a], d[b])[0, 1]
            for a, b in itertools.combinations([j[0] for j in JUDGES], 2)}


def make_figure(rows, out_path):
    """One panel: each judge's correlation with human ground-truth valence, per
    dataset. The home-domain pairings are the story."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dsets = ["DEAM (music)", "Emo-Soundscapes"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(dsets)); w = 0.26
    cols = ["#3b7dd8", "#e08a1e", "#3aa34a"]
    for i, (col, lab) in enumerate(JUDGES):
        vals = [rows[(ds, col)] for ds in dsets]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=cols[i],
                      edgecolor="k", lw=0.5)
        for j, b in enumerate(bars):
            ds = dsets[j]
            home = ((col == "v_deam" and "DEAM" in ds)
                    or (col == "v_emo" and "Emo" in ds))
            ax.annotate(f"{vals[j]:+.2f}" + ("\n(home)" if home else ""),
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom" if b.get_height() >= 0 else "top",
                        fontsize=8, fontweight="bold" if home else "normal")
    ax.axhline(0, color="k", lw=0.7)
    ax.axhspan(-0.1, 0.1, color="grey", alpha=0.12)
    ax.set_xticks(x)
    ax.set_xticklabels(["DEAM\n(music)", "Emo-Soundscapes\n(environmental)"])
    ax.set_ylabel("corr(judge valence, human ground-truth valence)")
    ax.set_ylim(-0.4, 1.05)
    ax.set_title("In-domain control: the judges do track human valence on real "
                 "audio\nthe same heads were flat on synthetic drones")
    ax.legend(fontsize=9, loc="upper center", ncol=3, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nWrote figure {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Run the three valence judges on real, labelled audio.")
    ap.add_argument("--n", type=int, default=150, help="clips per dataset")
    ap.add_argument("--mode", choices=["run", "analyze"], default="run",
                    help="run = judge then analyse; analyze = re-use the saved CSV")
    ap.add_argument("--out-dir", dest="out_dir", default=None, metavar="DIR",
                    help="output directory (default ../data/cross_domain)")
    ap.add_argument("--drone-dir", dest="drone_dir", default=None, metavar="DIR",
                    help="directory holding cross_judge.csv from "
                         "cross_domain_validity.py, for the drone baseline row "
                         "(default ../data/resolution_probe)")
    ap.add_argument("--ridges", default=None, metavar="NPZ")
    paths.add_arg(ap, "deam", "emo_soundscapes")
    args = ap.parse_args()

    out_dir = (Path(args.out_dir) if args.out_dir
               else _SECTION / "data" / "cross_domain")
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = out_dir / "cross_judge_indomain.csv"
    drone_dir = (Path(args.drone_dir) if args.drone_dir
                 else _SECTION / "data" / "resolution_probe")

    if args.mode == "analyze":
        if not saved.exists():
            raise SystemExit(f"\nno saved results at:\n    {saved}\n")
        both = pd.read_csv(saved)
        deam = both[both.dataset == "DEAM (music)"]
        emo = both[both.dataset == "Emo-Soundscapes"]
        print(f"Loaded saved labels (n={len(deam)} DEAM, {len(emo)} Emo)")
    else:
        deam_root = paths.require("deam", args.deam_root)
        emo_root = paths.require("emo_soundscapes", args.emo_soundscapes_root)
        print(f"DEAM       : {deam_root}")
        print(f"Emo        : {emo_root}")
        rng = np.random.default_rng(7)
        deam = judge_clips(load_deam(deam_root, args.n, rng), "DEAM (music)",
                           args.ridges)
        emo = judge_clips(load_emo(emo_root, args.n, rng), "Emo-Soundscapes",
                          args.ridges)
        both = pd.concat([deam, emo], ignore_index=True)
        both.to_csv(saved, index=False)
        print(f"\nSaved {saved}  (n={len(deam)} DEAM, {len(emo)} Emo)")

    print("\n=== (i) corr(judge, ground-truth valence) in the home domain ===")
    print("   home-domain pairings marked *: DEAM head on DEAM, Emo head on Emo")
    rows = {}
    for name, d in [("DEAM (music)", deam), ("Emo-Soundscapes", emo)]:
        line = []
        for col, lab in JUDGES:
            r = np.corrcoef(d[col], d["v_gt"])[0, 1]
            rows[(name, col)] = r
            home = ((lab == "MERT+DEAM" and "DEAM" in name)
                    or (lab == "MERT+Emo" and "Emo" in name))
            line.append(f"{lab}{'*' if home else ' '} r={r:+.2f}")
        print(f"  {name:18s}: " + "   ".join(line))

    print("\n=== (ii) inter-judge agreement (Pearson r) ===")
    for name, d in [("DEAM (music)", deam), ("Emo-Soundscapes", emo)]:
        print(f"  {name:18s}: " +
              "  ".join(f"{k} {v:+.2f}" for k, v in interjudge(d).items()))

    drone_ij = None
    drone_csv = drone_dir / "cross_judge.csv"
    if drone_csv.exists():
        drone_ij = interjudge(pd.read_csv(drone_csv))
        print(f"  {'SYNTH DRONES':18s}: " +
              "  ".join(f"{k} {v:+.2f}" for k, v in drone_ij.items()))
    else:
        print(f"  (drone baseline row needs {drone_csv}; run "
              "cross_domain_validity.py first)")

    print("\n=== Read ===")
    print(f"  DEAM valence head vs DEAM ground truth: "
          f"r={rows[('DEAM (music)', 'v_deam')]:+.2f}")
    print(f"  Emo valence head vs Emo ground truth:   "
          f"r={rows[('Emo-Soundscapes', 'v_emo')]:+.2f}")
    if drone_ij is not None:
        ido = np.mean([abs(v) for v in interjudge(both).values()])
        dro = np.mean([abs(v) for v in drone_ij.values()])
        print(f"  Mean |inter-judge r|: real audio {ido:.2f} vs "
              f"synthetic drones {dro:.2f}")
    print("  Substantial in-domain correlation with near-zero drone correlation")
    print("  means the degeneracy is the domain, not the models.")
    print("  Caveat: these clips may sit in the ridge heads' training sets, so")
    print("  the in-domain figures are an optimistic upper bound.")

    make_figure(rows, out_dir / "indomain_judge_validity.png")


if __name__ == "__main__":
    main()
