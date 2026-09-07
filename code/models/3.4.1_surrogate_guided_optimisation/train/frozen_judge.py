"""The frozen affect judge: MERT embeddings plus fitted ridge heads.

Shared by every script in this section that scores audio. Section 3.4.1 never
trains these: they are the fixed measuring instrument the surrogate is distilled
from, and §4.1 is the story of what happens when you optimise against them.

Three heads ship in ``../weights/affect_ridges.npz``:

  ``deam_valence``   MERT -> DEAM ridge, predicts in [0,1]. The valence judge.
  ``emo_arousal``    MERT -> Emo-Soundscapes ridge, predicts in [-1,1] already.
  ``emo_valence``    MERT -> Emo-Soundscapes ridge. Used only by the two
                     cross-domain scripts, as a same-features/different-head
                     comparison against ``deam_valence``.

Each was fitted as ``sklearn.linear_model.Ridge(alpha=1.0)`` on 768-dimensional
mean-pooled MERT embeddings. Only ``coef_`` and ``intercept_`` are needed to
reproduce a prediction exactly, so they ship as one 20 KB ``.npz`` rather than
as pickles. That is deliberate: the originals were pickled under scikit-learn
1.7.2 and warn on load under newer versions, and a plain array file is both
version-proof and readable. The arithmetic is identical -- verified equal to
``Ridge.predict`` to the last bit on all three heads.

The audEERING model is downloaded from Hugging Face on first use; it is not
refitted here either. It supplies the arousal axis in the *unused* default
configuration, and its native valence in the cross-domain comparison.

Refitting the ridge heads is out of scope for this section: they came from the
exploratory pass over DEAM and Emo-Soundscapes described in the report. Nothing
in this repository needs them refitted to reproduce §4.1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

MERT_ID = "m-a-p/MERT-v1-95M"
AUDEERING_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

MERT_SR = 24000
AUDEERING_SR = 16000

RIDGES_PATH = Path(__file__).resolve().parents[1] / "weights" / "affect_ridges.npz"
HEAD_NAMES = ("deam_valence", "emo_arousal", "emo_valence")


class RidgeHead:
    """A fitted ridge head: ``predict(X) == X @ coef_ + intercept_``.

    Deliberately not an sklearn object. Restoring one from arrays sidesteps the
    version-pinning that pickled estimators impose, and the prediction is exact.
    """

    def __init__(self, coef: np.ndarray, intercept: float, name: str = ""):
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.name = name

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Match ``Ridge.predict`` exactly, including its dtype behaviour.

        sklearn preserves a float32 input and computes the dot product in
        float32. MERT embeddings arrive as float32, so the published labels were
        produced that way; computing in float64 instead shifts every prediction
        by ~2e-7 and the section stops reproducing bit-for-bit. Precision is not
        the point here -- reproducing the recorded artefact is.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.coef.shape[0]:
            raise ValueError(
                f"{self.name}: expected {self.coef.shape[0]} features, got {X.shape[1]}"
            )
        if X.dtype not in (np.float32, np.float64):
            X = X.astype(np.float64)
        coef = self.coef.astype(X.dtype, copy=False)
        return X @ coef + X.dtype.type(self.intercept)

    def __repr__(self) -> str:
        return f"RidgeHead({self.name!r}, n_features={self.coef.shape[0]})"


def load_ridges(path: str | Path | None = None) -> dict[str, RidgeHead]:
    """Load all three heads. Fails with the path rather than a KeyError."""
    p = Path(path) if path else RIDGES_PATH
    if not p.exists():
        raise SystemExit(
            f"\naffect ridge heads not found at:\n    {p}\n\n"
            "This file ships with the repository; pass --ridges to point elsewhere.\n"
        )
    z = np.load(p)
    heads = {}
    for name in HEAD_NAMES:
        ck, ik = f"{name}_coef", f"{name}_intercept"
        if ck not in z:
            raise SystemExit(f"\n{p} is missing head {name!r}; found {sorted(z.files)}\n")
        heads[name] = RidgeHead(z[ck], float(z[ik]), name)
    return heads


# --- audEERING regression head -----------------------------------------------
# Architecture must match the published checkpoint exactly or the weights will
# not load; this mirrors the model card's reference implementation.

class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = torch.tanh(self.dense(self.dropout(x)))
        return self.out_proj(self.dropout(x))


class EmotionModel(Wav2Vec2PreTrainedModel):
    """Outputs logits ordered [arousal, dominance, valence], each in [0,1]."""

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    @property
    def all_tied_weights_keys(self):
        # transformers >= 5 requires custom subclasses to expose this. This
        # model ties no weights.
        return {}

    def forward(self, input_values):
        hidden_states = self.wav2vec2(input_values)[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        return hidden_states, self.classifier(hidden_states)


# --- loaders and the embedding step ------------------------------------------

def load_mert(device):
    """Returns (feature_extractor, model). Downloads on first use."""
    proc = Wav2Vec2FeatureExtractor.from_pretrained(MERT_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MERT_ID, trust_remote_code=True).to(device).eval()
    return proc, model


def load_audeering(device):
    """Returns (processor, model). Downloads on first use."""
    proc = Wav2Vec2Processor.from_pretrained(AUDEERING_ID)
    model = EmotionModel.from_pretrained(AUDEERING_ID).to(device).eval()
    return proc, model


def mert_embedding(proc, model, waveform: torch.Tensor, sr: int, device) -> np.ndarray:
    """Mean-pooled MERT embedding, (768,), from a (1, n) waveform at ``sr``.

    Resamples to 24 kHz, which is what the heads were fitted against; feeding
    the model at any other rate silently shifts every prediction.
    """
    import torchaudio

    wav = (torchaudio.functional.resample(waveform, sr, MERT_SR)
           if sr != MERT_SR else waveform)
    with torch.no_grad():
        inputs = proc(wav.squeeze().numpy(), sampling_rate=MERT_SR, return_tensors="pt")
        out = model(inputs["input_values"].to(device))
        return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def audeering_logits(proc, model, waveform: torch.Tensor, sr: int, device) -> np.ndarray:
    """(3,) array [arousal, dominance, valence] in [0,1], from a (1, n) waveform."""
    import torchaudio

    wav = (torchaudio.functional.resample(waveform, sr, AUDEERING_SR)
           if sr != AUDEERING_SR else waveform)
    with torch.no_grad():
        inputs = proc(wav.squeeze().numpy(), sampling_rate=AUDEERING_SR, return_tensors="pt")
        _, logits = model(inputs["input_values"].to(device))
    return logits[0].cpu().numpy()
