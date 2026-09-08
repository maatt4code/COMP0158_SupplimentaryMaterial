"""Boundary guard: keep a wandering VA walk inside rated territory.

Runtime half of the guard. It loads a fitted JSON from ``../weights/`` and
needs no ratings, no sklearn fit, and no dataset. ``../train/fit_guard.py``
produces that JSON.

The problem: the conductor drives a random walk through VA space, and nothing
stops it wandering into coordinates no listener ever rated, where the retrieval
bank is sparse and the audio degenerates. The guard is a fence built from where
people actually rated.

A Gaussian process is fitted over the rated coordinates, and its posterior
sigma(v, a) measures distance from rated territory. Two drift terms come out of
it, both added to the walk:

  SOFT, always on:   -gamma * grad sigma^2(x)
      A smooth early correction that pushes the walk down the uncertainty
      gradient before it gets lost.
  HARD, gated on sigma > tau:   ramp(sigma) * beta * (safe_target(x) - x)
      A safety net for deep out-of-distribution, where the GP variance
      saturates, grad sigma^2 goes to zero, and the soft force alone stops
      working.

The hard term is deliberately smooth in three ways, each fixing something that
went wrong without it: ``safe_target`` is the inverse-distance-weighted mean of
the k nearest safe points, because a single nearest point made the field
Voronoi-discontinuous; the force fades in over a band above tau rather than
switching on at tau; and its per-step magnitude is capped, so containment comes
from a sustained bounded force instead of a one-step teleport back to safety.

tau is derived, not chosen: it is a quantile of the posterior sigma evaluated
at the rated points themselves.

**No rating values are needed at runtime, and none ship.** The posterior
variance depends only on the rated COORDINATES and the kernel, never on what
anyone scored -- so sigma, tau and the safe set are all computable without the
ratings. This module exposes no ``valence()``: the guard answers "how far is
this from anywhere a listener has rated", which is a density question, not a
prediction. It is a fence, and it must not be promoted to an oracle.

A guard is only valid for a walk navigating the SAME valence axis it was fitted
on. ``label_space`` in the JSON records which, and the caller must match it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "boundary_guard.json"


def _rbf(A, B, length_scale):
    d2 = ((np.atleast_2d(A)[:, None, :] - np.atleast_2d(B)[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / (length_scale ** 2))


class BoundaryGuard:
    """Fitted guard. Construct with :meth:`load`, then call ``guard_drift``."""

    def __init__(self, X, length_scale, noise_level, K_inv, tau, safe_points,
                 label_space="judge", gamma=0.08, beta=1.0, fd_step=0.02,
                 k_safe=5, ramp_band=0.05, hard_step_cap=0.08,
                 tau_quantile=0.90, y_scale=1.0):
        self.X = np.asarray(X, float)
        self.length_scale = float(length_scale)
        self.noise_level = float(noise_level)
        self.K_inv = np.asarray(K_inv, float)
        self.tau = float(tau)
        self.safe_points = np.asarray(safe_points, float)
        self.label_space = label_space
        self.gamma, self.beta = float(gamma), float(beta)
        self.fd_step = float(fd_step)
        self.k_safe = int(k_safe)
        self.ramp_band = float(ramp_band)
        self.hard_step_cap = float(hard_step_cap)
        self.tau_quantile = float(tau_quantile)
        self.y_scale = float(y_scale)
        self.log = []

    @classmethod
    def load(cls, path=None):
        p = Path(path) if path else DEFAULT_WEIGHTS
        if not p.exists():
            raise SystemExit(
                f"\nfitted guard not found at:\n    {p}\n\n"
                "It ships with this section. Refit it with "
                "../train/fit_guard.py if you need to.\n")
        d = json.loads(p.read_text())
        return cls(
            X=d["X"], length_scale=d["length_scale"],
            noise_level=d["noise_level"], K_inv=d["K_inv"], tau=d["tau"],
            safe_points=d["safe_points"], label_space=d.get("label_space", "judge"),
            gamma=d.get("gamma", 0.08), beta=d.get("beta", 1.0),
            fd_step=d.get("fd_step", 0.02), k_safe=d.get("k_safe", 5),
            ramp_band=d.get("ramp_band", 0.05),
            hard_step_cap=d.get("hard_step_cap", 0.08),
            tau_quantile=d.get("tau_quantile", 0.90),
            y_scale=d.get("y_scale", 1.0))

    # -- posterior ----------------------------------------------------------
    def sigma(self, x):
        """Posterior standard deviation. Depends on the rated COORDINATES and
        the kernel only, never on the rating values."""
        x = np.atleast_2d(np.asarray(x, float))
        Ks = _rbf(x, self.X, self.length_scale)                 # (m, n)
        var = (1.0 + self.noise_level
               - np.einsum("ij,jk,ik->i", Ks, self.K_inv, Ks))
        s = np.sqrt(np.maximum(var, 0.0)) * self.y_scale
        return s if len(s) > 1 else float(s[0])

    def grad_sigma2(self, x):
        """Central-difference gradient of sigma^2, one batched call."""
        h = self.fd_step
        pts = np.array([[x[0] + h, x[1]], [x[0] - h, x[1]],
                        [x[0], x[1] + h], [x[0], x[1] - h]])
        s2 = np.asarray(self.sigma(pts)) ** 2
        return np.array([(s2[0] - s2[1]) / (2 * h),
                         (s2[2] - s2[3]) / (2 * h)])

    # -- the guard ----------------------------------------------------------
    def guard_drift(self, x, step=None):
        """Drift correction at state x. Returns (drift (2,), info)."""
        x = np.asarray(x, float)
        sig = float(self.sigma(x))
        soft = -self.gamma * self.grad_sigma2(x)
        hard = np.zeros(2)
        triggered = bool(sig > self.tau)
        ramp = 0.0
        if triggered:
            ramp = min((sig - self.tau) / self.ramp_band, 1.0)
            d2 = ((self.safe_points - x) ** 2).sum(axis=1)
            k = min(self.k_safe, len(self.safe_points))
            idx = np.argpartition(d2, k - 1)[:k]
            w = 1.0 / (d2[idx] + 1e-8)
            w /= w.sum()
            safe_target = (self.safe_points[idx] * w[:, None]).sum(axis=0)
            hard = ramp * self.beta * (safe_target - x)
            n = float(np.linalg.norm(hard))
            if n > self.hard_step_cap:
                hard *= self.hard_step_cap / n
        info = {"step": step, "v": float(x[0]), "a": float(x[1]),
                "sigma": sig, "tau": self.tau, "ramp": float(ramp),
                "soft_norm": float(np.linalg.norm(soft)),
                "hard_norm": float(np.linalg.norm(hard)),
                "hard_triggered": triggered}
        self.log.append(info)
        return soft + hard, info

    def summary(self):
        n = len(self.log)
        return {"steps": n,
                "hard_triggers": sum(e["hard_triggered"] for e in self.log),
                "label_space": self.label_space,
                "tau": self.tau, "tau_quantile": self.tau_quantile,
                "gamma": self.gamma, "beta": self.beta,
                "max_sigma": max((e["sigma"] for e in self.log), default=None),
                "mean_soft_norm": (float(np.mean([e["soft_norm"] for e in self.log]))
                                   if n else None),
                "n_rated": int(len(self.X)),
                "n_safe_points": int(len(self.safe_points))}


def ou_walk(guard, start, target, steps=600, drift_k=0.10, noise_std=0.03,
            use_guard=True, seed=1):
    """Ornstein-Uhlenbeck walk toward ``target``, optionally guarded.

    The comparison this exists for: run it with and without the guard from the
    same seed and compare how far each strays into unrated territory.
    """
    rng = np.random.default_rng(seed)
    x = np.array(start, float)
    path, sigmas = [], []
    for t in range(steps):
        g = np.zeros(2)
        if use_guard:
            g, _ = guard.guard_drift(x, step=t)
        x = (x + drift_k * (np.asarray(target, float) - x) + g
             + rng.normal(0.0, noise_std, 2))
        x = np.clip(x, -1.0, 1.0)
        path.append(x.copy())
        sigmas.append(float(guard.sigma(x)))
    return np.array(path), np.array(sigmas)
