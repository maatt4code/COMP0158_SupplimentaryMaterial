"""Extract MERT embeddings for a directory of clips.

Provenance for ``../human_ratings/pool_mert.npz``, which ships already
extracted. The 150 rated pool clips are not distributed, so this script cannot
be re-run against them from the repository alone -- it is here so the recipe
that produced those embeddings is on the record, and so the same protocol can
be applied to new audio.

Three constraints are locked, and each matters:

  1. LOUDNESS PARITY. Every clip gets the same in-memory A-weighted
     normalisation before it reaches MERT. The rated pool was loudness-
     normalised on disk while bank clips were not, so without this the model
     would learn amplitude rather than timbre. Files are never modified.
  2. CHUNK AND MEAN. Audio is split into 5-second chunks at 24 kHz, each is a
     separate forward pass, and the per-chunk time-means are combined weighted
     by chunk length. This keeps sequences inside MERT's training window.
  3. RECIPE PARITY with the frozen judge of Section 3.4.1: same model, mono
     mixdown, 24 kHz, last_hidden_state mean-pooled. Deviating would put these
     embeddings in a different space from the judge's, and the controlled
     contrast in fit_control_ridge.py -- same features, different labels --
     would no longer hold.

Resume-safe: with --resume, filenames already in --out are skipped and new ones
appended. Without it, an existing --out aborts rather than being overwritten.
A JSON sidecar records the full recipe.

Run:
  python extract_pool_mert.py --wav-dir <clips> --out ../data/pool_mert.npz

Next: fit_control_ridge.py consumes the npz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import bilinear, lfilter

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "common"))
from device import get_device                       # noqa: E402

MODEL_ID = "m-a-p/MERT-v1-95M"
MERT_SR = 24000
CHUNK_S = 5.0
MIN_TAIL_S = 0.5          # tail chunks shorter than this are dropped
AW_TARGET = 0.015         # frozen loudness protocol value -- do not change
PEAK_GUARD = 0.9


def a_weighting_ba(sr):
    """IEC 61672 A-weighting as a digital filter, via the bilinear transform.

    Inlined rather than imported: it is ten lines of standard filter design,
    and the loudness-normalisation script it came from is not otherwise needed
    here.
    """
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    A1000 = 1.9997
    nums = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20.0)), 0, 0, 0, 0]
    dens = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                      [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
    dens = np.polymul(np.polymul(dens, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2])
    return bilinear(nums, dens, sr)


def normalize_aweighted(audio, sr):
    """A-weighted normalisation to the frozen target. Never touches the file.
    Idempotent on already-normalised clips, where the gain comes out near 1."""
    b, a = a_weighting_ba(sr)
    aw = lfilter(b, a, audio)
    aw_rms = np.sqrt(np.mean(aw ** 2)) + 1e-8
    gain = AW_TARGET / aw_rms
    peak = np.max(np.abs(audio * gain))
    if peak > PEAK_GUARD:
        gain *= PEAK_GUARD / peak
    return audio * gain


def embed_clip(path, proc, model, device):
    """Load, mono, A-weight, resample to 24 kHz, chunk and mean."""
    audio, sr = sf.read(path, always_2d=True)
    audio = normalize_aweighted(audio.mean(axis=1), sr)

    wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    if sr != MERT_SR:
        wav = torchaudio.functional.resample(wav, sr, MERT_SR)
    wav = wav.squeeze(0)

    chunk_len = int(CHUNK_S * MERT_SR)
    min_tail = int(MIN_TAIL_S * MERT_SR)
    chunks = [wav[lo:lo + chunk_len] for lo in range(0, len(wav), chunk_len)]
    chunks = [c for c in chunks if len(c) >= min_tail] or [wav]

    means, weights = [], []
    for c in chunks:
        with torch.no_grad():
            inputs = proc(c.numpy(), sampling_rate=MERT_SR, return_tensors="pt")
            out = model(inputs["input_values"].to(device))
            means.append(out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy())
            weights.append(len(c))
    w = np.asarray(weights, float)
    return (np.stack(means) * (w / w.sum())[:, None]).sum(axis=0)


def main():
    ap = argparse.ArgumentParser(description="Extract MERT embeddings for a "
                                             "directory of wav clips.")
    ap.add_argument("--wav-dir", dest="wav_dir", required=True, metavar="DIR")
    ap.add_argument("--out", required=True, help=".npz keyed by filename")
    ap.add_argument("--resume", action="store_true",
                    help="extend an existing --out, skipping finished files")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N clips (a quick check)")
    args = ap.parse_args()

    wav_dir = Path(args.wav_dir)
    if not wav_dir.is_dir():
        raise SystemExit(f"\nnot a directory:\n    {wav_dir}\n")
    wavs = sorted(f for f in os.listdir(wav_dir) if f.endswith(".wav"))
    if args.limit:
        wavs = wavs[:args.limit]
    if not wavs:
        raise SystemExit(f"\nno .wav files in {wav_dir}\n")

    out = Path(args.out)
    done = {}
    if out.exists():
        if not args.resume:
            raise SystemExit(f"\n{out} exists. Pass --resume to extend it.\n")
        with np.load(out) as z:
            done = {k: z[k] for k in z.files}
        print(f"resuming: {len(done)} embeddings already in {out}")
    todo = [f for f in wavs if f not in done]
    print(f"{len(wavs)} wavs in {wav_dir}, {len(todo)} to extract")
    if not todo:
        return

    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    device = get_device()
    print(f"Loading {MODEL_ID}...")
    proc = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID,
                                                    trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID,
                                      trust_remote_code=True).to(device).eval()

    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, fname in enumerate(todo, 1):
        done[fname] = embed_clip(wav_dir / fname, proc, model, device)
        if i % 25 == 0 or i == len(todo):
            rate = i / (time.time() - t0)
            print(f"  {i}/{len(todo)} ({rate:.1f} clips/s, "
                  f"~{(len(todo) - i) / max(rate, 1e-6):.0f}s left)")
            np.savez(out, **done)                     # checkpoint

    np.savez(out, **done)
    side = out.with_suffix("").as_posix() + "_meta.json"
    Path(side).write_text(json.dumps({
        "model": MODEL_ID, "pooling": "last_hidden_state.mean(time)",
        "sample_rate": MERT_SR,
        "chunking": f"{CHUNK_S}s chunk-and-mean, length-weighted, "
                    f"tails < {MIN_TAIL_S}s dropped",
        "normalization": f"in-memory A-weighted (IEC 61672) to aw-RMS "
                         f"{AW_TARGET}, peak guard {PEAK_GUARD} "
                         f"(files untouched)",
        "embedding_dim": int(next(iter(done.values())).shape[0]),
        "n_clips": len(done), "wav_dir": wav_dir.name,
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print(f"written -> {out} ({len(done)} embeddings), sidecar -> {side}")


if __name__ == "__main__":
    main()
