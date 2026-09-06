"""
Extract an empirical DDSP timbre prior from the NSynth string subset (GPU).

For each note we know f0 exactly (MIDI pitch label), so harmonic analysis is
a direct measurement: project each analysis frame onto complex exponentials
at integer multiples of f0 to estimate harmonic amplitudes, plus frame RMS.

GPU strategy: the projection basis depends only on f0, so notes are grouped
by MIDI pitch and all frames of a pitch group are analyzed in one batched
matmul on the GPU (CUDA > XPU > CPU via get_device).

What we keep per frame:
  - harmonic_distribution: K amplitudes, normalized to sum to 1 (timbre shape)
  - loudness: frame RMS (kept separate -- timbre shape is loudness-invariant)
Only the sustained portion is analyzed (skip attack, stop before release);
near-silent frames are dropped.

Output: <out-dir>, by default ../weights/timbre_prior/
  - frames.npz        : harm_dist (N,K), loudness (N,), pitch (N,), velocity (N,)
  - notes_index.csv   : one row per note (file, instrument, pitch, velocity, n_frames)

Run:
  # quick test on 50 notes
  python analyse_nsynth_timbre.py --limit 50

  # full run, reading NSynth audio in place (pair with
  # build_nsynth_prior.py --metadata-only)
  python analyse_nsynth_timbre.py \
      --in-dir  $DRONE_DATA/nsynth_strings \
      --audio-dir $DRONE_NSYNTH/audio \
      --out-dir ../weights/timbre_prior

Previous: build_nsynth_prior.py
Next:     generate_preset_bank.py
"""

import os
import json
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import pandas as pd
import torch
from tqdm import tqdm

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from device import get_device  # noqa: E402
import paths                   # noqa: E402

SR          = 16000          # NSynth sample rate
N_HARMONICS = 32             # K: harmonics measured per frame
FRAME_LEN   = 2048           # 128 ms at 16 kHz
HOP         = 1024           # 50% overlap
SUSTAIN_START_SEC = 0.25     # skip the bow attack
SUSTAIN_END_SEC   = 2.75     # stop before the release at 3.0 s
SILENCE_RMS = 1e-3           # drop frames quieter than this
NOTES_PER_BATCH = 256        # notes loaded/projected per GPU batch
LOADER_THREADS  = 16         # concurrent wav reads (I/O-bound: sf.read releases the GIL)


def midi_to_hz(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def make_basis(f0_hz: float, device) -> tuple[torch.Tensor, int]:
    """Windowed complex projection basis at exact harmonic frequencies.
    Returns (basis (K_valid, FRAME_LEN) complex64, K_valid)."""
    t = torch.arange(FRAME_LEN, device=device, dtype=torch.float32) / SR
    window = torch.hann_window(FRAME_LEN, periodic=False, device=device)
    harm_freqs = f0_hz * torch.arange(1, N_HARMONICS + 1, device=device, dtype=torch.float32)
    valid = harm_freqs < (SR / 2) * 0.95
    k_valid = int(valid.sum())
    phase = -2j * torch.pi * harm_freqs[:k_valid, None].to(torch.complex64) * t[None, :].to(torch.complex64)
    basis = torch.exp(phase) * window[None, :].to(torch.complex64)
    return basis, k_valid


def frame_note(audio: np.ndarray) -> np.ndarray:
    """Slice the sustained portion into overlapping frames: (n_frames, FRAME_LEN)."""
    start = int(SUSTAIN_START_SEC * SR)
    end   = min(int(SUSTAIN_END_SEC * SR), len(audio))
    audio = audio[start:end]
    if len(audio) < FRAME_LEN:
        return np.empty((0, FRAME_LEN), dtype=np.float32)
    n = (len(audio) - FRAME_LEN) // HOP + 1
    idx = np.arange(FRAME_LEN)[None, :] + HOP * np.arange(n)[:, None]
    return audio[idx].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Measure an empirical DDSP timbre prior from the NSynth subset.")
    paths.add_arg(parser, "nsynth")
    parser.add_argument("--limit", type=int, default=None,
                        help="analyse only the first N notes (testing)")
    parser.add_argument("--in-dir", dest="in_dir", type=str, default=None,
                        help="directory holding selected_metadata.json "
                             "(default: ../data/nsynth_strings)")
    parser.add_argument("--out-dir", dest="out_dir", type=str, default=None,
                        help="where to write frames.npz and notes_index.csv "
                             "(default: ../weights/timbre_prior)")
    parser.add_argument("--audio-dir", dest="audio_dir", type=str, default=None,
                        help="read wavs from here (default: <nsynth-root>/audio, "
                             "which is what build_nsynth_prior.py --metadata-only "
                             "expects; pass --in-dir instead if you used --copy)")
    args = parser.parse_args()

    in_dir = args.in_dir or str(Path(__file__).resolve().parents[1] / "data" / "nsynth_strings")
    out_dir = args.out_dir or str(Path(__file__).resolve().parents[1] / "weights" / "timbre_prior")
    # build_nsynth_prior.py --metadata-only is the recommended path and leaves
    # no wavs in in_dir, so the NSynth audio directory is the right default.
    audio_dir = args.audio_dir or str(paths.resolve("nsynth", args.nsynth_root) / "audio")
    meta_path = os.path.join(in_dir, "selected_metadata.json")
    if not os.path.exists(meta_path):
        raise SystemExit(
            f"\nselected_metadata.json not found at:\n    {meta_path}\n\n"
            f"Run build_nsynth_prior.py --metadata-only first.\n")
    if not any(Path(audio_dir).glob("*.wav")):
        raise SystemExit(
            f"\nno .wav files under:\n    {audio_dir}\n\n"
            f"Point at the NSynth audio with --audio-dir, or use --in-dir if you "
            f"ran build_nsynth_prior.py --copy.\n")
    print(f"Metadata : {meta_path}")
    print(f"Audio    : {audio_dir}")
    print(f"Output   : {out_dir}")

    device = get_device()
    win_sum = float(np.hanning(FRAME_LEN).sum())

    with open(os.path.join(in_dir, "selected_metadata.json")) as f:
        meta = json.load(f)

    keys = sorted(meta.keys())
    if args.limit:
        keys = keys[: args.limit]

    # Group notes by MIDI pitch -> shared projection basis per group
    by_pitch = defaultdict(list)
    for key in keys:
        by_pitch[meta[key]["pitch"]].append(key)

    all_dists, all_louds, all_pitch, all_vel = [], [], [], []
    index_rows = []

    def load_and_frame(key):
        path = os.path.join(audio_dir, key + ".wav")
        if not os.path.exists(path):
            return None
        audio, _ = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        fr = frame_note(audio)
        return fr if len(fr) else None

    pbar = tqdm(total=len(keys), desc="Analyzing notes")
    pool = ThreadPoolExecutor(max_workers=LOADER_THREADS)
    for pitch, group in sorted(by_pitch.items()):
        basis, k_valid = make_basis(midi_to_hz(pitch), device)

        for b0 in range(0, len(group), NOTES_PER_BATCH):
            batch_keys = group[b0:b0 + NOTES_PER_BATCH]

            # Load and frame on CPU (threaded; map preserves key order)
            frames_list, owners = [], []
            for key, fr in zip(batch_keys, pool.map(load_and_frame, batch_keys)):
                if fr is not None:
                    frames_list.append(fr)
                    owners.append((key, len(fr)))

            pbar.update(len(batch_keys))
            if not frames_list:
                continue

            frames = torch.from_numpy(np.concatenate(frames_list)).to(device)  # (F, FRAME_LEN)

            # Batched projection: (F, FRAME_LEN) @ (FRAME_LEN, K) -> (F, K)
            amps = torch.abs(frames.to(torch.complex64) @ basis.T) * (2.0 / win_sum)
            rms  = torch.sqrt(torch.mean(frames ** 2, dim=1))

            # Pad to N_HARMONICS, normalize each row to sum 1
            full = torch.zeros(amps.shape[0], N_HARMONICS, device=device)
            full[:, :k_valid] = amps
            totals = full.sum(dim=1)
            keep = (rms >= SILENCE_RMS) & (totals >= 1e-8)
            full = full[keep] / totals[keep, None]
            rms  = rms[keep]

            keep_np = keep.cpu().numpy()
            dist_np = full.cpu().numpy().astype(np.float32)
            rms_np  = rms.cpu().numpy().astype(np.float32)

            # Reattach frames to their notes for the index
            cursor_all, cursor_kept = 0, 0
            for key, n_fr in owners:
                kept_here = int(keep_np[cursor_all:cursor_all + n_fr].sum())
                cursor_all += n_fr
                if kept_here == 0:
                    continue
                m = meta[key]
                all_dists.append(dist_np[cursor_kept:cursor_kept + kept_here])
                all_louds.append(rms_np[cursor_kept:cursor_kept + kept_here])
                all_pitch.append(np.full(kept_here, m["pitch"], dtype=np.int16))
                all_vel.append(np.full(kept_here, m["velocity"], dtype=np.int16))
                cursor_kept += kept_here
                index_rows.append({
                    "note_str": key,
                    "instrument": m["instrument_str"],
                    "pitch": m["pitch"],
                    "velocity": m["velocity"],
                    "n_frames": kept_here,
                })
    pbar.close()
    pool.shutdown()

    harm_dist = np.concatenate(all_dists)
    loudness  = np.concatenate(all_louds)
    pitch_arr = np.concatenate(all_pitch)
    velocity  = np.concatenate(all_vel)

    os.makedirs(out_dir, exist_ok=True)
    # float16 storage: ~3 significant digits is ample for a sampling prior,
    # and keeps multi-family priors under GitHub's 100 MB file limit
    # (s03 renormalizes each sampled frame on use).
    np.savez_compressed(
        os.path.join(out_dir, "frames.npz"),
        harm_dist=harm_dist.astype(np.float16), loudness=loudness.astype(np.float16),
        pitch=pitch_arr, velocity=velocity,
    )
    pd.DataFrame(index_rows).to_csv(os.path.join(out_dir, "notes_index.csv"), index=False)

    print(f"\nNotes analyzed: {len(index_rows):,}")
    print(f"Total timbre frames: {len(harm_dist):,}")
    print(f"Mean harmonic distribution (first 8): {harm_dist.mean(axis=0)[:8].round(4)}")
    print(f"Saved: {os.path.join(out_dir, 'frames.npz')}, notes_index.csv")


if __name__ == "__main__":
    main()
