"""Decoupled parameter blending: VA -> theta over the labelled bank.

Runtime code. Nothing here reads a dataset root, resolves a path, or trains
anything: bank index files are passed in by the caller. That is what keeps the
conductor startable with every dataset root missing.

The engine decouples theta into two groups, which is the whole idea:

  continuous performance scalars  soft-blended over the k nearest anchors
  discrete timbre (32 harmonics)  retrieved whole from ONE anchor, never averaged

Averaging harmonic distributions across anchors smudges them: two different
spectra that each sound like an instrument average into one that sounds like
neither. Scalars interpolate cleanly, spectra do not.

``retrieve()`` goes further and takes scalars AND harmonics from the same
anchor. That is the deployable path. Soft-blending the scalars regresses them
toward the bank mean -- their spread across VA targets came out 5-8x smaller
than 1-NN -- and then glues that averaged register onto a foreign 1-NN timbre.
Whole-row retrieval keeps theta on the measured manifold, and smoothness is
handled in TIME by the caller's waypoint hold and crossfade, not by averaging
in VA space.

Hysteresis exists because 1-NN retrieval at a fast control rate clicks every
time the walk crosses a Voronoi boundary. The engine keeps the previous anchor
unless a new one beats it by a margin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_HARM = 32
BASE_SCALARS = ["f0_hz", "swell_rate", "swell_depth", "noise_level",
                "noise_cutoff_hz"]
CHORD_SCALARS = ["third_interval", "third_gain", "fifth_gain", "octave_gain"]

# Drone-era banks predate the chord voices; these are the values that make a
# chord-aware renderer reproduce a plain drone.
CHORD_DEFAULTS = {"third_interval": 3.0, "third_gain": 0.0,
                  "fifth_gain": 0.0, "octave_gain": 0.0}


class DecoupledEngine:
    """Blend or retrieve theta from one or more labelled bank indexes."""

    def __init__(self, bank_indexes, k=16, bandwidth=0.15,
                 hysteresis_threshold=0.01, valence_column="valence"):
        if isinstance(bank_indexes, (str, Path)):
            bank_indexes = [bank_indexes]
        paths_in = [Path(p) for p in bank_indexes]
        missing = [p for p in paths_in if not p.exists()]
        if missing:
            raise SystemExit(
                "\nbank index not found:\n"
                + "\n".join(f"    {p}" for p in missing)
                + "\n\nPass the labelled_index.csv file(s) that form the "
                  "retrieval bank.\n")

        # Deduplicate by the bank's directory name. A packaged copy and a bulk
        # copy of the same bank are content-identical, and loading both
        # duplicates every anchor, which silently halves the effective k.
        seen, unique = set(), []
        for p in paths_in:
            if p.parent.name not in seen:
                seen.add(p.parent.name)
                unique.append(p)
        self.bank_files = [str(p) for p in unique]

        self.k = k
        self.bandwidth = float(bandwidth)
        self.hysteresis_threshold = float(hysteresis_threshold)
        self.prev_best_idx = None

        frames = []
        for p in unique:
            df = pd.read_csv(p)
            for col, default in CHORD_DEFAULTS.items():
                if col not in df.columns:
                    df[col] = default
            # Columns can exist with individual NaN cells, which dropna would
            # discard silently. Fill them, but say so: genuine corruption must
            # stay visible.
            n_nan = int(df[list(CHORD_DEFAULTS)].isna().sum().sum())
            if n_nan:
                print(f"  note: filled {n_nan} missing chord cells in {p.name}")
                df = df.fillna(CHORD_DEFAULTS)
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)

        self.harm_cols = [f"h{j + 1:02d}" for j in range(N_HARM)]
        self.scalar_keys = BASE_SCALARS + CHORD_SCALARS

        if valence_column not in df.columns:
            raise SystemExit(
                f"\nvalence column {valence_column!r} not in the bank.\n"
                f"Available: {sorted(c for c in df.columns if 'valence' in c)}\n"
                "propagate_labels_krr.py writes labeled_index_human.csv with "
                "a valence_human column.\n")
        required = [valence_column, "arousal"] + self.scalar_keys + self.harm_cols
        df = df.dropna(subset=required)
        if df.empty:
            raise SystemExit("\nbank is empty after dropping rows with missing "
                             "required columns.\n")

        self.valence_column = valence_column
        self.va = df[[valence_column, "arousal"]].to_numpy(np.float64)
        self.harm = df[self.harm_cols].to_numpy(np.float64)
        self.scal = df[self.scalar_keys].to_numpy(np.float64)
        self.n = len(df)
        self.is_fallback_bank = not any(
            "chord10k" in f or "mixed10k" in f for f in self.bank_files)
        print(f"Bank: {self.n} anchors from "
              f"{[Path(f).parent.name for f in self.bank_files]} "
              f"on '{valence_column}'")

    def reset_hysteresis(self):
        self.prev_best_idx = None

    def theta_from_index(self, idx):
        """Render-ready theta from one anchor row, with provenance keys."""
        harm = self.harm[idx]
        harm = harm / (np.sum(harm) + 1e-8)
        theta = {"harm_dist": harm.astype(np.float32)}
        for j, key in enumerate(self.scalar_keys):
            theta[key] = float(self.scal[idx, j])
        theta["anchor_idx"] = int(idx)
        theta["anchor_v"] = float(self.va[idx, 0])
        theta["anchor_a"] = float(self.va[idx, 1])
        return theta

    def _nearest(self, v, a, use_hysteresis):
        d2 = (self.va[:, 0] - v) ** 2 + (self.va[:, 1] - a) ** 2
        closest = int(np.argmin(d2))
        if use_hysteresis:
            if self.prev_best_idx is not None and \
                    d2[self.prev_best_idx] - d2[closest] < self.hysteresis_threshold:
                closest = self.prev_best_idx
            self.prev_best_idx = closest
        return d2, closest

    def retrieve(self, v, a, use_hysteresis=True):
        """Whole-row 1-NN: scalars and harmonics from the SAME anchor."""
        _, closest = self._nearest(v, a, use_hysteresis)
        return self.theta_from_index(closest)

    def blend(self, v, a, use_hysteresis=True):
        """1-NN timbre, kernel-blended scalars. Kept for comparison against
        ``retrieve``; the reported system uses ``retrieve``."""
        d2 = (self.va[:, 0] - v) ** 2 + (self.va[:, 1] - a) ** 2
        k_eff = min(self.k, len(d2))
        part = np.argpartition(d2, k_eff - 1)[:k_eff]
        nearest = part[np.argsort(d2[part])]

        closest = nearest[0]
        if use_hysteresis:
            if self.prev_best_idx is not None and \
                    d2[self.prev_best_idx] - d2[closest] < self.hysteresis_threshold:
                closest = self.prev_best_idx
            self.prev_best_idx = closest

        harm = self.harm[closest]
        harm = harm / (np.sum(harm) + 1e-8)

        w = np.exp(-d2[nearest] / (2.0 * self.bandwidth ** 2))
        s = w.sum()
        if s <= 1e-12:
            w = np.zeros_like(w); w[0] = 1.0; s = 1.0
        w = w / s
        blended = w @ self.scal[nearest]

        theta = {"harm_dist": harm.astype(np.float32)}
        for i, key in enumerate(self.scalar_keys):
            theta[key] = float(blended[i])
        return theta
