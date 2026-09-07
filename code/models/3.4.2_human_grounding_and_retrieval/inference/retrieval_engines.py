"""
Retrieval engines: VA -> theta over the measured manifold (the labeled bank).

All engines map a target (valence, arousal) to a synthesis parameter vector theta
by operating over the SAME bank used by the hard k-NN deploy: the labeled
synthetic clips (the chord-era and mixed preset banks), each row a real, judged theta.
None of them touch the albums (which carry no VA ground truth) -- the supervision,
where any exists, is the bank's (VA -> theta) pairs. This keeps every engine
comparable to the hard-kNN cycle-consistency floor under the frozen judge.

Engines:
  hard   -- nearest measured-VA neighbour (the deployed baseline; here for parity)
  soft   -- KERNEL / SOFT k-NN: Gaussian-weighted blend of the (k) nearest theta.
            One hyperparameter (bandwidth). No training: the bank is the model.
  gp     -- GAUSSIAN PROCESS regression VA -> theta. Fits a smooth surface through
            the bank with a few kernel hyperparameters; predicts a mean theta
            (and a usable predictive variance for an OOD / uncertainty read-out).
  attn   -- LEARNED ATTENTION over the bank. A small net learns a VA-similarity
            metric and a soft assignment over bank rows, trained self-supervised on
            the bank to reconstruct each row's theta from its VA with the row itself
            masked out (../train/train_attention_retrieval.py). Learns from the BANK.

Every engine exposes .blend(v, a, rng=None) -> theta dict with keys
  {"harm_dist": (N_HARM,) float32, <scalar_key>: float, ...}
matching the renderer's and the cycle evaluation's expectations. Harmonic weights are clipped
non-negative and renormalised to a simplex; scalars are blended in raw units.
"""
import numpy as np
import pandas as pd

N_HARM = 32
BASE_SCALARS = ["f0_hz", "swell_rate", "swell_depth", "noise_level", "noise_cutoff_hz"]
CHORD_SCALARS = ["third_interval", "third_gain", "fifth_gain", "octave_gain"]


# ----------------------------------------------------------------------------- #
# Bank
# ----------------------------------------------------------------------------- #
def build_bank(indexes):
    """Concatenate labeled_index.csv file(s) into a retrieval bank.

    Returns a dict with parallel arrays:
      va    (N, 2)            measured valence, arousal
      harm  (N, N_HARM)       harmonic distribution columns h01..hNN
      scal  (N, S)            scalar columns (BASE + any present CHORD scalars)
      harm_cols, scalar_keys  column name lists (scalar order == scal columns)
    """
    if isinstance(indexes, str):
        indexes = [indexes]
    df = pd.concat([pd.read_csv(p) for p in indexes], ignore_index=True)
    harm_cols = [f"h{j + 1:02d}" for j in range(N_HARM)]
    scalar_keys = list(BASE_SCALARS) + [k for k in CHORD_SCALARS if k in df.columns]
    return {
        "va": df[["valence", "arousal"]].to_numpy(np.float64),
        "harm": df[harm_cols].to_numpy(np.float64),
        "scal": df[scalar_keys].to_numpy(np.float64),
        "harm_cols": harm_cols,
        "scalar_keys": scalar_keys,
        "n": len(df),
    }


def _theta_from_vec(harm_vec, scal_vec, scalar_keys):
    """Pack a blended (harm, scalars) vector into the render-ready theta dict."""
    harm = np.clip(np.asarray(harm_vec, np.float64), 0.0, None)
    harm = harm / (harm.sum() + 1e-8)
    theta = {"harm_dist": harm.astype(np.float32)}
    for i, k in enumerate(scalar_keys):
        theta[k] = float(scal_vec[i])
    return theta


# ----------------------------------------------------------------------------- #
# hard / soft k-NN  (no training; the bank is the model)
# ----------------------------------------------------------------------------- #
class HardKNN:
    """Nearest measured-VA neighbour (k=1) or a random draw from the k nearest."""
    name = "hard"

    def __init__(self, bank, k=1):
        self.bank, self.k = bank, k

    def blend(self, v, a, rng=None):
        va = self.bank["va"]
        d2 = (va[:, 0] - v) ** 2 + (va[:, 1] - a) ** 2
        nn = np.argsort(d2)[: self.k]
        idx = nn[0] if (self.k == 1 or rng is None) else int(rng.choice(nn))
        return _theta_from_vec(self.bank["harm"][idx], self.bank["scal"][idx],
                               self.bank["scalar_keys"])


class SoftKNN:
    """Gaussian kernel-weighted blend of the (k) nearest theta in VA space.

    weight_i = exp(-||va_i - (v,a)||^2 / 2 bandwidth^2), normalised. bandwidth sets
    how local the blend is; k=None uses the whole bank (pure kernel regression),
    k>0 restricts to the k nearest for speed and locality. Deterministic -> smooth
    trajectories (no per-step flicker), unlike hard k-NN.
    """
    name = "soft"

    def __init__(self, bank, bandwidth=0.15, k=64):
        self.bank, self.h, self.k = bank, float(bandwidth), k

    def blend(self, v, a, rng=None):
        va = self.bank["va"]
        d2 = (va[:, 0] - v) ** 2 + (va[:, 1] - a) ** 2
        idx = np.argsort(d2)[: self.k] if self.k else np.arange(len(va))
        w = np.exp(-d2[idx] / (2.0 * self.h ** 2))
        s = w.sum()
        if s <= 1e-12:                       # query far from everything -> nearest
            w = np.zeros_like(w); w[int(np.argmin(d2[idx]))] = 1.0; s = 1.0
        w = w / s
        harm = w @ self.bank["harm"][idx]
        scal = w @ self.bank["scal"][idx]
        return _theta_from_vec(harm, scal, self.bank["scalar_keys"])


# ----------------------------------------------------------------------------- #
# Gaussian Process regression  (a few kernel hyperparameters, fit on the bank)
# ----------------------------------------------------------------------------- #
class GPEngine:
    """GP regression VA -> theta. Subsamples the bank (exact GP is O(n^3)), fits a
    shared-kernel multi-output GP in standardised theta space, predicts the mean.

    .last_std holds the predictive std of the most recent query (uncertainty /
    OOD read-out: high where VA space is sparsely covered by the bank).
    """
    name = "gp"

    def __init__(self, bank, subsample=1500, length_scale=0.25, noise=1e-2,
                 optimize=False, seed=0):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        rng = np.random.default_rng(seed)
        n = bank["n"]
        sel = rng.choice(n, size=min(subsample, n), replace=False)
        X = bank["va"][sel]
        Y = np.concatenate([bank["harm"][sel], bank["scal"][sel]], axis=1)
        self.y_mean = Y.mean(0)
        self.y_std = Y.std(0) + 1e-8
        Yn = (Y - self.y_mean) / self.y_std
        kernel = (ConstantKernel(1.0) * RBF(length_scale=length_scale)
                  + WhiteKernel(noise_level=noise))
        # optimize=False -> fixed hyperparameters (fast, deterministic); True ->
        # marginal-likelihood fit of length scale / noise on the bank subsample.
        self.gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=False,
            optimizer="fmin_l_bfgs_b" if optimize else None)
        self.gp.fit(X, Yn)
        self.nh = bank["harm"].shape[1]
        self.scalar_keys = bank["scalar_keys"]
        self.last_std = None

    def blend(self, v, a, rng=None):
        mu, std = self.gp.predict([[v, a]], return_std=True)
        self.last_std = float(np.mean(std))
        y = mu[0] * self.y_std + self.y_mean
        return _theta_from_vec(y[: self.nh], y[self.nh:], self.scalar_keys)


# ----------------------------------------------------------------------------- #
# Learned attention over the bank  (trained by ../train/train_attention_retrieval.py)
# ----------------------------------------------------------------------------- #
def _build_attn_net(d_model):
    import torch.nn as nn

    class AttentionRetriever(nn.Module):
        """Soft, learned assignment over the bank. Query/key MLPs map VA -> R^d;
        a temperature-scaled dot-product softmax over bank keys gives weights that
        blend the bank's (standardised) theta values. The metric and temperature
        are learned; the values are the bank itself."""

        def __init__(self, d=d_model):
            super().__init__()
            self.d = d
            self.q = nn.Sequential(nn.Linear(2, 128), nn.GELU(), nn.Linear(128, d))
            self.k = nn.Sequential(nn.Linear(2, 128), nn.GELU(), nn.Linear(128, d))
            self.log_temp = nn.Parameter(__import__("torch").zeros(()))

        def weights(self, query_va, bank_va, mask_idx=None):
            import torch
            Q = self.q(query_va)                              # (B, d)
            K = self.k(bank_va)                               # (N, d)
            scale = (self.d ** 0.5) * torch.exp(self.log_temp)
            scores = (Q @ K.t()) / scale                      # (B, N)
            if mask_idx is not None:                          # leave-self-out
                scores[torch.arange(len(mask_idx)), mask_idx] = float("-inf")
            return torch.softmax(scores, dim=-1)

        def forward(self, query_va, bank_va, bank_theta_norm, mask_idx=None):
            w = self.weights(query_va, bank_va, mask_idx)     # (B, N)
            return w @ bank_theta_norm                        # (B, theta_dim)

    return AttentionRetriever()


class AttnEngine:
    """Inference wrapper around a trained AttentionRetriever checkpoint."""
    name = "attn"

    def __init__(self, bank, ckpt_path, device="cpu"):
        import torch
        # weights_only=False: trusted local checkpoint, and it stores numpy
        # arrays (y_mean/y_std) that the PyTorch 2.6 default loader rejects.
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.net = _build_attn_net(ck["d"]).to(device)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.device = device
        self.nh = bank["harm"].shape[1]
        self.scalar_keys = bank["scalar_keys"]
        theta = np.concatenate([bank["harm"], bank["scal"]], axis=1)
        self.y_mean = ck["y_mean"]
        self.y_std = ck["y_std"]
        theta_norm = (theta - self.y_mean) / self.y_std
        self.bank_va = torch.tensor(bank["va"], dtype=torch.float32, device=device)
        self.bank_theta = torch.tensor(theta_norm, dtype=torch.float32, device=device)

    def blend(self, v, a, rng=None):
        import torch
        q = torch.tensor([[v, a]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            yn = self.net(q, self.bank_va, self.bank_theta)[0].cpu().numpy()
        y = yn * self.y_std + self.y_mean
        return _theta_from_vec(y[: self.nh], y[self.nh:], self.scalar_keys)


# ----------------------------------------------------------------------------- #
# Factory
# ----------------------------------------------------------------------------- #
def make_engine(name, bank, *, k=1, bandwidth=0.15, soft_k=64,
                gp_subsample=1500, gp_length_scale=0.25, gp_optimize=False,
                attn_ckpt=None, device="cpu", seed=0):
    """Construct an engine by name. See module docstring for semantics."""
    if name == "hard":
        return HardKNN(bank, k=k)
    if name == "soft":
        return SoftKNN(bank, bandwidth=bandwidth, k=soft_k)
    if name == "gp":
        return GPEngine(bank, subsample=gp_subsample, length_scale=gp_length_scale,
                        optimize=gp_optimize, seed=seed)
    if name == "attn":
        if not attn_ckpt:
            raise ValueError("attn engine needs attn_ckpt; train one with "
                             "../train/train_attention_retrieval.py, or use "
                             "the fitted ../weights/attn_retrieval.pt")
        return AttnEngine(bank, attn_ckpt, device=device)
    raise ValueError(f"unknown engine: {name}")
