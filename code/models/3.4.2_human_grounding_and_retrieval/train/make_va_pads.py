"""The conductor's VA pad background: a MEASUREMENT, not a decoration.

The pad the listener drags a target around is the interface's promise about
what the instrument can do, so it has to show where the bank ACTUALLY REACHES.
That is not a design choice. It is the bank's (valence, arousal) coverage, and
it differs sharply depending on which valence column is read:

  judge  `valence`        -1.000 .. +0.244   mean -0.676   38.5% unreachable
  human  `valence_human`  -1.000 .. +0.572   mean -0.034   27.4% unreachable

So there is one pad per label space, and the conductor swaps them with the
dropdown. Showing the judge map while retrieving on human labels would draw
the grey region in the wrong place -- inviting the listener to aim at targets
the engine cannot hit, or to avoid ones it can.

**Grey out the unreachable part; do not colour the reachable part.** Colouring
the reachable region reads as "this area is blocked", the exact opposite of
what it means. Greyed-out-equals-unavailable is the convention people already
have.

Precomputed so the runtime needs neither pandas, nor matplotlib, nor the 18 MB
bank indexes to draw its own background.

Run:
  python make_va_pads.py --banks ../../../conductor/weights/banks
  python make_va_pads.py --banks DIR --space judge --out DIR
"""

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
CODE = SECTION.parent.parent
# The shipped banks carry BOTH columns: the human index is the judge index
# plus `valence_human`, so one file per bank serves both label spaces.
DEFAULT_BANKS = CODE / "conductor" / "weights" / "banks"
DEFAULT_OUT = CODE / "conductor" / "assets"
PX = 512                      # background resolution
GRID = 192                    # distance-field resolution (upscaled to PX)
NEAR = 0.05                   # <= this from an anchor = solidly reachable
FAR = 0.25                    # >= this = effectively unreachable


def load_anchors(space, banks_dir):
    """(n, 2) array of (valence, arousal) for one label space.

    Reads the bank indexes directly rather than resolving a dataset root: this
    produces a conductor asset, and the conductor's own banks are the thing it
    must describe. A pad drawn from a different bank than the one being
    searched would be a picture of somewhere else.
    """
    import pandas as pd
    col = "valence_human" if space == "human" else "valence"
    frames = []
    for p in sorted(Path(banks_dir).glob("*/labeled_index.csv")):
        cols = pd.read_csv(p, nrows=0).columns
        if col not in cols:
            raise SystemExit(
                f"\n{p} has no '{col}' column.\n"
                f"The human label space needs a bank index carrying "
                f"valence_human; see 3.4.2/train/propagate_labels_krr.py.\n")
        frames.append(pd.read_csv(p, usecols=[col, "arousal"]))
    if not frames:
        raise SystemExit(f"\nno */labeled_index.csv under {banks_dir}\n")
    df = pd.concat(frames, ignore_index=True).dropna()
    return df[[col, "arousal"]].to_numpy(np.float32)


def distance_field(va, n=GRID):
    """Distance from each grid cell to the nearest anchor, over [-1,1]^2."""
    gv, ga = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    pts = np.column_stack([gv.ravel(), ga.ravel()]).astype(np.float32)
    out = np.empty(len(pts), np.float32)
    for i in range(0, len(pts), 4096):          # chunked: n*20000 blows up fast
        ch = pts[i:i + 4096]
        d2 = ((ch[:, None, :] - va[None, :, :]) ** 2).sum(-1)
        out[i:i + 4096] = np.sqrt(d2.min(1))
    return out.reshape(n, n)


def render(space, va, d, out_png):
    """Background image: reachable region bright, unreachable dimmed out."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(PX / 100, PX / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])             # no margins: pixel == VA, exactly
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    # GREY OUT the unreachable part, do not colour the reachable part. Colouring
    # the reachable region reads as "this area is blocked" -- the exact opposite
    # of what it means. Greyed-out = unavailable is the convention users already
    # have, so the mask is an alpha ramp of neutral grey over a clear pad.
    reach = np.clip((FAR - d) / (FAR - NEAR), 0.0, 1.0)
    mask = np.zeros(reach.shape + (4,), np.float32)
    mask[..., :3] = 0.58                        # neutral grey
    mask[..., 3] = (1.0 - reach) * 0.82         # fully clear where reachable
    ax.imshow(mask, origin="lower", extent=(-1, 1, -1, 1),
              interpolation="bilinear", zorder=3)

    g = np.linspace(-1, 1, d.shape[0])
    ax.contour(g, g, d, levels=[NEAR], colors="#2563eb",
               linewidths=1.4, alpha=0.9, zorder=4)

    ax.set_facecolor("#f8fafc")
    for gl in np.arange(-0.75, 1.0, 0.25):      # faint grid, reads as a pad
        ax.axhline(gl, color="#e2e8f0", lw=0.6, zorder=1)
        ax.axvline(gl, color="#e2e8f0", lw=0.6, zorder=1)
    ax.axhline(0, color="#94a3b8", lw=1.0, zorder=2)
    ax.axvline(0, color="#94a3b8", lw=1.0, zorder=2)
    for x, y, t in ((-0.55, 0.9, "tense"), (0.55, 0.9, "excited"),
                    (-0.55, -0.93, "sad"), (0.55, -0.93, "calm")):
        ax.text(x, y, t, ha="center", va="center", fontsize=10,
                color="#475569", style="italic", zorder=5)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space", choices=["judge", "human", "both"], default="both")
    ap.add_argument("--banks", type=Path, default=DEFAULT_BANKS,
                    help="directory of <bank>/labeled_index.csv")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="where pad_<space>.png is written")
    args = ap.parse_args()
    spaces = ["judge", "human"] if args.space == "both" else [args.space]

    args.out.mkdir(parents=True, exist_ok=True)
    for space in spaces:
        va = load_anchors(space, args.banks)
        d = distance_field(va)
        render(space, va, d, args.out / f"pad_{space}.png")
        pct = 100.0 * (d > 0.10).mean()
        print(f"{space:6s}  n={len(va):6d}  valence {va[:,0].min():+.3f}..{va[:,0].max():+.3f}"
              f"  unreachable {pct:.1f}%  -> {args.out.name}/pad_{space}.png")


if __name__ == "__main__":
    main()
