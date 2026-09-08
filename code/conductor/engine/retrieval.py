"""The deployed retrieval engine: VA -> theta, with a warmth filter.

Runtime code, and the engine the conductor actually loads. Like
``decoupled_engine.py`` it takes its bank index files as arguments and resolves
no dataset root, so the conductor starts with every dataset missing.

Three ideas are combined here:

  1. Decoupled parameters. Continuous performance scalars are blendable;
     discrete timbre is retrieved, never averaged, because averaging two
     harmonic distributions smudges both.
  2. A Gaussian process over the bank supplies a predictive variance, which is
     a usable manifold-uncertainty read-out: high variance means the query sits
     where the bank is sparse.
  3. A warmth filter. Among the VA-nearest candidates, those with the highest
     harmonic centroid are discarded before blending, which removes the harsh,
     buzzy out-of-distribution outliers that otherwise get retrieved at the
     edges of the manifold.

Two label spaces. The bank carries the frozen judge's valence and, when
``propagate_labels_krr.py`` has been run, a ``valence_human`` column from the
theta-KRR propagation. ``set_label_space`` swaps which one retrieval navigates.
It is a SESSION-START operation: the two disagree about the nearest anchor at
almost every VA target, so switching mid-walk changes the instrument under the
listener.

``retrieve`` is the deployable path, not ``blend`` -- see decoupled_engine.py
for why whole-row retrieval beats blending scalars.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_HARM = 32
BASE_SCALARS = ["f0_hz", "swell_rate", "swell_depth", "noise_level",
                "noise_cutoff_hz"]
CHORD_SCALARS = ["third_interval", "third_gain", "fifth_gain", "octave_gain"]


def harmonic_centroid(harm_weights):
    """Spectral centre of mass of a harmonic distribution.
    Low is warm and dark; high is bright and buzzy."""
    indices = np.arange(1, len(harm_weights) + 1)
    return np.sum(harm_weights * indices) / (np.sum(harm_weights) + 1e-8)


class GPSoftKNNEngine:
    """Warmth-filtered retrieval with a GP uncertainty read-out."""

    def __init__(self, bank_indexes, bandwidth=0.15, k=16,
                 warmth_fraction=0.5, gp_subsample=1500, gp_length_scale=0.25,
                 seed=42, hysteresis_threshold=0.01, fit_gp=True):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (RBF, ConstantKernel,
                                                      WhiteKernel)

        if isinstance(bank_indexes, (str, Path)):
            bank_indexes = [bank_indexes]
        paths_in = [Path(p) for p in bank_indexes]
        present = [p for p in paths_in if p.exists()]
        if not present:
            raise SystemExit(
                "\nno bank index found. Tried:\n"
                + "\n".join(f"    {p}" for p in paths_in)
                + "\n\nPass the labelled_index.csv file(s) forming the bank.\n")

        # One file per bank directory: a packaged copy and a bulk copy of the
        # same bank are identical, and taking both loads every anchor twice.
        seen, unique = set(), []
        for p in present:
            if p.parent.name not in seen:
                seen.add(p.parent.name)
                unique.append(p)

        self.h = float(bandwidth)
        self.k = k
        self.warmth_fraction = warmth_fraction
        self.hysteresis_threshold = float(hysteresis_threshold)
        self.prev_best_idx = None
        self.bank_files = [str(p) for p in unique]

        df = pd.concat([pd.read_csv(p) for p in unique], ignore_index=True)
        self.harm_cols = [f"h{j + 1:02d}" for j in range(N_HARM)]
        self.scalar_keys = BASE_SCALARS + [k for k in CHORD_SCALARS
                                           if k in df.columns]
        required = ["valence", "arousal"] + self.scalar_keys + self.harm_cols
        df = df.dropna(subset=[c for c in required if c in df.columns])
        if df.empty:
            raise SystemExit("\nbank is empty after dropping incomplete rows.\n")

        self.va = df[["valence", "arousal"]].to_numpy(np.float64)
        self.va_judge = self.va.copy()
        self.va_human = None
        # The human axis costs one float column, not a second bank: the human
        # index IS the judge index plus valence_human. Retrieval reads only
        # self.va, so swapping the column is a complete swap.
        if "valence_human" in df.columns and df["valence_human"].notna().all():
            vh = self.va.copy()
            vh[:, 0] = df["valence_human"].to_numpy(np.float64)
            self.va_human = vh
        elif "valence_human" in df.columns:
            # A partial column would silently mix two label spaces. Refuse it.
            print(f"  note: valence_human present but "
                  f"{int(df['valence_human'].isna().sum())} rows lack it -- "
                  f"human label space DISABLED")
        self.label_space = "judge"

        self.harm = df[self.harm_cols].to_numpy(np.float64)
        self.scal = df[self.scalar_keys].to_numpy(np.float64)
        self.n = len(df)
        self.centroids = np.array([harmonic_centroid(h) for h in self.harm])

        self.gp = None
        if fit_gp:
            rng = np.random.default_rng(seed)
            sel = rng.choice(self.n, size=min(gp_subsample, self.n),
                             replace=False)
            gp_Y = self.scal[sel]
            self.gp_mean = gp_Y.mean(0)
            self.gp_std = gp_Y.std(0) + 1e-8
            kernel = (ConstantKernel(1.0) * RBF(length_scale=gp_length_scale)
                      + WhiteKernel(noise_level=1e-2))
            self.gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
            self.gp.fit(self.va[sel], (gp_Y - self.gp_mean) / self.gp_std)
        print(f"Engine: {self.n} anchors, "
              f"human label space {'available' if self.va_human is not None else 'unavailable'}")

    def set_label_space(self, space):
        """Swap the valence axis. Session start only. Returns what was applied."""
        if space == "human" and self.va_human is not None:
            self.va, self.label_space = self.va_human, "human"
        else:
            self.va, self.label_space = self.va_judge, "judge"
        self.prev_best_idx = None          # hysteresis state is now stale
        return self.label_space

    def get_uncertainty(self, v, a):
        """GP predictive variance at (v, a). High means off-manifold."""
        if self.gp is None:
            raise RuntimeError("engine built with fit_gp=False")
        _, std = self.gp.predict([[v, a]], return_std=True)
        return float(np.mean(std[0] ** 2))

    def theta_from_index(self, idx):
        harm = self.harm[idx]
        harm = harm / (np.sum(harm) + 1e-8)
        theta = {"harm_dist": harm.astype(np.float32)}
        for j, key in enumerate(self.scalar_keys):
            theta[key] = float(self.scal[idx, j])
        theta["anchor_idx"] = int(idx)
        theta["anchor_v"] = float(self.va[idx, 0])
        theta["anchor_a"] = float(self.va[idx, 1])
        return theta

    def retrieve(self, v, a, use_hysteresis=True):
        """Whole-row 1-NN retrieval. The deployable path."""
        d2 = (self.va[:, 0] - v) ** 2 + (self.va[:, 1] - a) ** 2
        closest = int(np.argmin(d2))
        if use_hysteresis:
            if self.prev_best_idx is not None and \
                    d2[self.prev_best_idx] - d2[closest] < self.hysteresis_threshold:
                closest = self.prev_best_idx
            self.prev_best_idx = closest
        return self.theta_from_index(closest)

    def blend(self, v, a):
        """Warmth-filtered retrieval of timbre, kernel blend of scalars."""
        d2 = (self.va[:, 0] - v) ** 2 + (self.va[:, 1] - a) ** 2
        nearest = np.argsort(d2)[:self.k]

        # Warmth filter: keep the darkest fraction of the candidates.
        warm_ranks = np.argsort(self.centroids[nearest])
        cutoff = max(1, int(np.round(self.k * self.warmth_fraction)))
        warm = nearest[warm_ranks[:cutoff]]

        harm = self.harm[warm[0]]
        harm = harm / (np.sum(harm) + 1e-8)

        w = np.exp(-d2[warm] / (2.0 * self.h ** 2))
        s = w.sum()
        if s <= 1e-12:
            w = np.zeros_like(w); w[0] = 1.0; s = 1.0
        w = w / s
        blended = w @ self.scal[warm]

        theta = {"harm_dist": harm.astype(np.float32)}
        for i, key in enumerate(self.scalar_keys):
            theta[key] = float(blended[i])
        return theta

    def _audible_features(self, f0, centroid, swell_rate, swell_depth,
                          noise_level):
        """The audible-diversity feature vector: log f0, harmonic centroid,
        swell rate, swell depth, noise level, z-scored against the bank.

        Raw z-scored theta distance is NOT audible distance: linear-Hz f0
        overweights high octaves, and noise_cutoff_hz moves the metric even
        when there is essentially no noise to shape. This is the same feature
        list the rated pool's max-min selection used.
        """
        if not hasattr(self, "_af_stats"):
            i = {key: j for j, key in enumerate(self.scalar_keys)}
            bank = np.column_stack([
                np.log(np.maximum(self.scal[:, i["f0_hz"]], 1e-6)),
                self.centroids,
                self.scal[:, i["swell_rate"]],
                self.scal[:, i["swell_depth"]],
                self.scal[:, i["noise_level"]],
            ])
            self._af_stats = (bank.mean(axis=0), bank.std(axis=0) + 1e-8)
            self._af_bank = (bank - self._af_stats[0]) / self._af_stats[1]
        vec = np.array([np.log(max(float(f0), 1e-6)), float(centroid),
                        float(swell_rate), float(swell_depth),
                        float(noise_level)])
        return (vec - self._af_stats[0]) / self._af_stats[1]

    def retrieve_diverse(self, v, a, prev_theta=None, k=8):
        """Waypoint retrieval that guarantees audible change.

        Among the k VA-nearest anchors, return the one furthest from
        ``prev_theta`` in audible feature space. VA position is only weakly
        informative about how a preset sounds, so a slow or guard-pinned walk
        retrieves the same anchor at consecutive waypoints and nothing audibly
        changes. This trades a little VA fidelity inside one neighbourhood for
        guaranteed timbral movement. Stateless: the caller's waypoint hold and
        crossfade already prevent clicking at this rate.
        """
        d2 = (self.va[:, 0] - v) ** 2 + (self.va[:, 1] - a) ** 2
        k_eff = min(k, self.n)
        cand = np.argpartition(d2, k_eff - 1)[:k_eff]
        if prev_theta is None:
            best = int(cand[np.argmin(d2[cand])])
        else:
            prev_z = self._audible_features(
                prev_theta["f0_hz"],
                harmonic_centroid(np.asarray(prev_theta["harm_dist"])),
                prev_theta["swell_rate"], prev_theta["swell_depth"],
                prev_theta["noise_level"])
            cand_z = self._af_bank[cand]
            best = int(cand[np.argmax(((cand_z - prev_z) ** 2).sum(axis=1))])
        return self.theta_from_index(best)
