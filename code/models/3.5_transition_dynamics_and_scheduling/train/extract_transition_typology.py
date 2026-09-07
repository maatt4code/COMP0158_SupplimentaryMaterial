"""Extract transition descriptors from a corpus of audio.

Step A of the scheduler's timing pipeline, and the front half of what produces
`../weights/hsmm_transitions.json`. Point it at one or more directories of
audio and it finds the structural transitions in every track, measures what
KIND of transition each one is, and caches the result. `fit_corpus_hsmm.py`
then clusters those descriptors and fits the semi-Markov model.

**Any corpus works.** Each `--audio-dir` becomes a group label, taken from the
directory name, and tracks are found recursively inside it. Nothing about the
method is specific to the corpus this project happened to use, and no corpus
ships with this repository -- the fitted result does. Point it at your own
audio and refit.

The method, which is descriptive and unsupervised throughout:

  1. Per track, frame features at about 4 Hz: log-RMS loudness, log spectral
     centroid, bandwidth, spectral flatness, low and high band energy share,
     chroma projected to a 6-D tonal centroid (tonnetz -- the direct evidence
     channel for harmonic movement), and three label-free TEXTURE proxies.
     True simultaneous-track counts are unrecoverable from a stereo mixdown,
     so texture is approximated by NMF layer counting: decompose the mel
     spectrogram into spectral objects and count, per frame, how many carry
     more than a threshold share of that frame's energy. Report these as
     active spectral objects, never as literal track counts.
  2. Segment boundaries by FOOTE NOVELTY -- a checkerboard kernel on the
     feature self-similarity matrix. This is the standard structure method for
     onset-free music, where a derivative threshold fails. Detection is
     MULTI-SCALE (4, 8, 16, 32 s kernels): a single 16 s kernel is blind below
     about 5 s, which is exactly the scale the arc ramps live at. Boundaries
     are deduplicated across scales and each event's extent is measured at the
     coarsest scale that detected it, because fine smoothing reads a 20 s
     swell as its 2 s steepest core and reports the wrong duration.
  3. Per transition, a descriptor vector of duration, signed feature deltas,
     tonnetz shift, total magnitude, and the holds either side.

Two independent noise floors, and the second one matters:

  * A magnitude floor in z-units drops events indistinguishable from
    measurement noise.
  * ABSOLUTE floors in raw units. Every other threshold here is RELATIVE to
    the track -- novelty peaks against that track's maximum, quantiles over
    that track's own distribution -- and the z-unit floor inherits the same
    hole: a nearly static track has a tiny standard deviation, which INFLATES
    z-units, so noise can clear a z-threshold. The absolute floors are in
    units a listener could verify, and an event must move at least one of
    them to be recorded. A genuinely static track therefore yields zero
    transitions, and keeps its long holds.

Sub-floor events are skipped WITHOUT resetting the hold clock, because a
sub-noise wiggle should not shorten the measured stillness around it.

Extraction is slow and cached. Re-running a subset of groups keeps every other
group's cached rows, so a one-directory run does not wipe the corpus.

Run:
  python extract_transition_typology.py --audio-dir /path/to/corpus/artist_a \
      /path/to/corpus/artist_b --out ../data/typology_transitions.json
  python extract_transition_typology.py --audio-dir <dir> --limit-tracks 2

Next: fit_corpus_hsmm.py, which fits the semi-Markov model on the cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_SECTION = _HERE.parents[1]
DEFAULT_CACHE = _SECTION / "data" / "typology_transitions.json"

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".aif", ".aiff", ".ogg")

SR = 22050
HOP = 5512                 # about 0.25 s frames; ample for this material
N_FFT = 8192               # about 2.7 Hz bins, which resolves 50-100 Hz drone
                           # fundamentals -- 4096 was too coarse and gave weak
                           # chroma and tonnetz on bass-heavy material
KERNEL_SCALES_S = (4.0, 8.0, 16.0, 32.0)
FINE_SMOOTH_S = 1.25       # extent and duration measurement resolution
MIN_TRANS_S = 1.0
MIN_MAGNITUDE = 0.5        # z-unit record floor

# Absolute perceptual floors, in raw units. See the module docstring for why
# a relative floor alone is not enough.
ABS_FLOOR_LOUD_DB = 1.5
ABS_FLOOR_LOG_CENT = 0.10  # about a 10% spectral-centroid shift
ABS_FLOOR_BAND_DB = 2.0
ABS_FLOOR_LAYERS = 0.5
ABS_FLOOR_TONNETZ = 0.15   # a purely harmonic move still qualifies

GROW_Q = 0.60              # grow a transition's extent while |dZ| exceeds this
BAND_EDGES_HZ = (200.0, 2000.0)   # the LOW band is register movement, which is
                                  # distinct from centroid brightness
N_NMF_COMP = 10
LAYER_SHARE_THR = 0.10

DESC_NAMES = ["duration_s", "d_loudness", "d_centroid", "d_bandwidth",
              "d_flatness", "d_low_band", "d_high_band",
              "d_pitch_classes", "d_peaks", "d_layers", "tonnetz_shift",
              "total_magnitude", "scale_s", "pre_hold_s", "post_hold_s"]

# Pitch-class and peak counts are RECORDED on every transition, for profiles
# and texture verdicts, but stay OUT of the clustering geometry and the
# outlier veto: including them collapsed the type structure into an up/down
# split on peak count and roughly inflated exclusions. The layer count -- the
# meaningful texture dimension -- stays in both.
CLUSTER_EXCLUDE = ("d_peaks", "d_pitch_classes")
CLUSTER_DIMS = [i for i, n in enumerate(DESC_NAMES) if n not in CLUSTER_EXCLUDE]

# Below this total magnitude an event is BREATHING micro-motion. Not "fake" --
# the absolute floors guarantee it is audible -- but it is the wander prior,
# not a state transition, and only state transitions feed the typology.
STATE_MAG_MIN = 1.5

MAD_THRESHOLD = 3.5
# Outliers are judged on transition-INTRINSIC dimensions only. Long holds are
# corpus SIGNAL -- some artists sit still for twenty minutes, and holds are the
# statistic that most needs to stay unbiased -- so they never trigger
# exclusion. Detection scale is a CATEGORY, not an anomaly: excluding events
# "because scale_s = 32" once killed every long morph and produced an
# artefactual "all transitions are under 16 s".
INTRINSIC_DIMS = [i for i, n in enumerate(DESC_NAMES)
                  if n not in ("pre_hold_s", "post_hold_s", "scale_s")
                  and n not in CLUSTER_EXCLUDE]
SCALE_DIM = DESC_NAMES.index("scale_s")
# Non-negative right-skewed dims get log1p before the MAD. On the linear scale
# their MAD is tiny and every substantial REAL event reads as an outlier -- an
# early full-corpus run excluded a third of all transitions, mostly "because
# tonnetz_shift", which is to say it was deleting the corpus's actual harmonic
# movement: the very evidence being collected.
LOG_DIMS = [DESC_NAMES.index(n)
            for n in ("duration_s", "tonnetz_shift", "total_magnitude")]
MAX_OUTLIER_PRINT = 15


def moving_avg(x, w):
    if len(x) < w or w < 2:
        return x
    k = np.ones(w) / w
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, x)


def foote_novelty(ssm, kernel_frames):
    """Checkerboard-kernel novelty over a self-similarity matrix (Foote 2000)."""
    L = max(kernel_frames // 2, 2)
    sign = np.sign(np.add.outer(np.arange(-L, L), np.arange(-L, L)) + 1e-9)
    gauss = np.exp(-0.5 * (np.arange(-L, L) / (L / 2.0)) ** 2)
    kernel = -sign * np.outer(gauss, gauss)
    nov = np.zeros(len(ssm))
    for i in range(L, len(ssm) - L):
        nov[i] = float((ssm[i - L:i + L, i - L:i + L] * kernel).sum())
    return np.maximum(nov, 0)


def frame_features(y, do_nmf=True):
    """Per-frame features, silence-trimmed.

    Returns (base[n,7], tonnetz[n,6], density[n,3], hop_s), where base is
    [loudness_dB, log centroid, bandwidth/1000, flatness, low band dB,
    high band dB, layer count]. do_nmf=False zeroes the layer count and skips
    the expensive decomposition.
    """
    import librosa
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    rms = librosa.feature.rms(S=S, frame_length=N_FFT, hop_length=HOP)[0]
    cent = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    bw = librosa.feature.spectral_bandwidth(S=S, sr=SR)[0]
    flat = librosa.feature.spectral_flatness(S=S)[0]
    chroma = librosa.feature.chroma_stft(S=S, sr=SR)
    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=SR)

    # Texture density proxies. Only the layer count joins the detection stack,
    # because it is smooth and lets a voice entering with little envelope
    # movement cut a boundary; pitch-class and peak counts are too jittery for
    # detection and spawned many spurious micro-boundaries.
    cn = chroma / (chroma.max(axis=0, keepdims=True) + 1e-8)
    pc_count = (cn > 0.4).sum(axis=0).astype(float)
    Sl = 20 * np.log10(S + 1e-10)
    peak_count = ((Sl[1:-1] > Sl[:-2]) & (Sl[1:-1] > Sl[2:])
                  & (Sl[1:-1] > Sl.max(axis=0) - 30.0)).sum(axis=0).astype(float)

    if do_nmf:
        from sklearn.decomposition import NMF
        mel = librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_mels=96)
        nmf = NMF(n_components=N_NMF_COMP, init="nndsvda", max_iter=250,
                  random_state=0)
        act = nmf.fit_transform(np.sqrt(mel).T)
        contrib = act * nmf.components_.sum(axis=1)
        share = contrib / (contrib.sum(axis=1, keepdims=True) + 1e-12)
        layer_count = (share > LAYER_SHARE_THR).sum(axis=1).astype(float)
    else:
        layer_count = np.zeros(len(rms))

    lo = (S[freqs < BAND_EDGES_HZ[0]] ** 2).sum(0)
    hi = (S[freqs > BAND_EDGES_HZ[1]] ** 2).sum(0)
    total = (S ** 2).sum(0) + 1e-12
    low_band = 10 * np.log10(lo / total + 1e-8)
    high_band = 10 * np.log10(hi / total + 1e-8)

    hop_s = HOP / SR
    loud = 20 * np.log10(np.maximum(rms, 1e-8))
    keep = loud > (loud.max() - 60)      # trim silence relative to track peak
    base = np.column_stack([loud, np.log(np.maximum(cent, 1.0)), bw / 1000.0,
                            flat, low_band, high_band, layer_count])
    dens = np.column_stack([pc_count, peak_count, layer_count])
    return base[keep], tonnetz.T[keep], dens[keep], hop_s


def extract_track(path, min_track_s=60.0):
    """Find and describe every structural transition in one track."""
    import librosa
    from scipy.signal import find_peaks

    y, _ = librosa.load(str(path), sr=SR, mono=True)
    if len(y) < min_track_s * SR:
        return None
    base, tz, dens, hop_s = frame_features(y, do_nmf=True)
    if len(base) < 240:
        return None
    Z = (base - base.mean(0)) / (base.std(0) + 1e-8)

    wf = max(int(FINE_SMOOTH_S / hop_s), 2)
    Zf, tzf = moving_avg(Z, wf), moving_avg(tz, wf)
    densf = moving_avg(dens, wf)
    rawf = moving_avg(base, wf)          # RAW units, for the absolute floors
    Ff = np.column_stack([Zf, tzf])
    dz = np.linalg.norm(np.diff(Ff, axis=0), axis=1)
    dz = moving_avg(dz[:, None], wf)[:, 0]
    thr = np.quantile(dz, GROW_Q)

    events, dz_by_scale, thr_by_scale = [], {}, {}
    for ks in KERNEL_SCALES_S:
        w = max(int(ks / 4 / hop_s), 2)
        F = np.column_stack([moving_avg(Z, w), moving_avg(tz, w)])
        dzk = np.linalg.norm(np.diff(F, axis=0), axis=1)
        dzk = moving_avg(dzk[:, None], w)[:, 0]
        dz_by_scale[ks] = dzk
        thr_by_scale[ks] = float(np.quantile(dzk, GROW_Q))
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
        nov = foote_novelty(Fn @ Fn.T, int(ks / hop_s))
        if nov.max() <= 0:
            continue
        peaks, _ = find_peaks(nov, prominence=0.25 * nov.max(),
                              distance=max(int(ks / 2 / hop_s), 2))
        tol = int(ks / 2 / hop_s)
        for p in peaks:
            hit = next((ev for ev in events if abs(ev["frame"] - p) <= tol), None)
            if hit:
                hit["scales"].append(ks)
            else:
                events.append(dict(frame=int(p), scales=[ks]))
    events.sort(key=lambda ev: ev["frame"])

    pad = int(2.0 / hop_s)
    transitions, last_end = [], 0
    for ev in events:
        ks = max(ev["scales"])            # the coarsest scale that saw it
        dzk, thrk = dz_by_scale[ks], thr_by_scale[ks]
        p = min(ev["frame"], len(dzk) - 1)
        s = e = p
        while s > 0 and dzk[s - 1] > thrk:
            s -= 1
        while e < len(dzk) - 1 and dzk[e + 1] > thrk:
            e += 1
        dur = (e - s) * hop_s
        sf_, ef_ = p, p                   # the steep core inside the morph
        while sf_ > 0 and dz[sf_ - 1] > thr:
            sf_ -= 1
        while ef_ < len(dz) - 1 and dz[ef_ + 1] > thr:
            ef_ += 1
        if dur < MIN_TRANS_S or s < last_end:
            continue
        pre_i, post_i = max(s - pad, 0), min(e + pad, len(Ff) - 1)
        d_z = Zf[post_i] - Zf[pre_i]
        tz_shift = float(np.linalg.norm(tzf[post_i] - tzf[pre_i]))
        d_dens = densf[post_i] - densf[pre_i]
        total_mag = float(np.linalg.norm(np.append(d_z, tz_shift)))
        d_raw = rawf[post_i] - rawf[pre_i]
        moved = (abs(d_raw[0]) >= ABS_FLOOR_LOUD_DB
                 or abs(d_raw[1]) >= ABS_FLOOR_LOG_CENT
                 or abs(d_raw[4]) >= ABS_FLOOR_BAND_DB
                 or abs(d_raw[5]) >= ABS_FLOOR_BAND_DB
                 or abs(d_dens[2]) >= ABS_FLOOR_LAYERS
                 or tz_shift >= ABS_FLOOR_TONNETZ)
        if total_mag < MIN_MAGNITUDE or not moved:
            # A noise-floor event: do not record it, and do not reset the hold
            # clock, because a sub-noise wiggle should not shorten the measured
            # stillness around it.
            continue
        pre_hold = (s - last_end) * hop_s if last_end else np.nan
        transitions.append(dict(
            t_start_s=float(s * hop_s), duration_s=float(dur),
            duration_fine_s=float((ef_ - sf_) * hop_s),
            d_loudness=float(d_z[0]), d_centroid=float(d_z[1]),
            d_bandwidth=float(d_z[2]), d_flatness=float(d_z[3]),
            d_low_band=float(d_z[4]), d_high_band=float(d_z[5]),
            d_pitch_classes=float(d_dens[0]), d_peaks=float(d_dens[1]),
            d_layers=float(d_dens[2]),
            layers_pre=float(densf[pre_i, 2]), layers_post=float(densf[post_i, 2]),
            tonnetz_shift=tz_shift, total_magnitude=total_mag,
            scale_s=float(ks), scales=sorted(ev["scales"]),
            pre_hold_s=float(pre_hold), post_hold_s=float("nan")))
        if transitions[:-1]:
            transitions[-2]["post_hold_s"] = float((s - last_end) * hop_s)
        last_end = e
    return dict(hop_s=hop_s, transitions=transitions,
                track_len_s=float(len(base) * hop_s))


def collect_tracks(audio_dir):
    """Every audio file under one directory, recursively, sorted."""
    d = Path(audio_dir)
    if not d.is_dir():
        raise SystemExit(f"\nnot a directory:\n    {d}\n")
    return sorted(p for p in d.rglob("*")
                  if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)


def flag_outliers(X, rows, out_json=None):
    """Robust median/MAD z-scores over intrinsic dimensions, computed WITHIN
    each detection-scale stratum.

    Stratifying matters: a 30 s duration is normal at the 32 s scale and
    anomalous at the 4 s scale, and pooled statistics let the micro-event
    majority veto legitimate section-scale morphs. The report names the
    offending dimensions and, where identifiable, a physical cause -- the
    explanation IS the statistic; there are no labels and no training here.
    """
    Xt = X.copy()
    Xt[:, LOG_DIMS] = np.log1p(np.maximum(Xt[:, LOG_DIMS], 0.0))
    rz = np.zeros_like(Xt)
    for ks in np.unique(Xt[:, SCALE_DIM]):
        g = Xt[:, SCALE_DIM] == ks
        ref = Xt[g] if g.sum() >= 10 else Xt      # tiny stratum: fall back
        med = np.median(ref, axis=0)
        mad = np.median(np.abs(ref - med), axis=0) * 1.4826
        # MAD floor. Near-constant dimensions -- spectral flatness on tonal
        # drones is near zero everywhere -- have a degenerate MAD and would
        # flag everything; flatness alone once drove most exclusions. Floor it
        # at a fraction of the dimension's own std, so hyper-peaked dims are
        # tempered while well-behaved ones are untouched.
        mad = np.maximum(mad, 0.3 * ref.std(axis=0)) + 1e-8
        rz[g] = (Xt[g] - med) / mad
    mask = np.abs(rz[:, INTRINSIC_DIMS]).max(axis=1) > MAD_THRESHOLD
    if mask.any():
        print(f"-- outliers excluded ({int(mask.sum())}/{len(X)}, "
              f"robust-z > {MAD_THRESHOLD}) --")
    records = []
    for i in np.where(mask)[0]:
        r = rows[i]
        offending = sorted(INTRINSIC_DIMS, key=lambda j: -abs(rz[i, j]))
        dims = ", ".join(f"{DESC_NAMES[j]}={X[i, j]:+.2f} ({rz[i, j]:+.1f} MADs)"
                         for j in offending[:3] if abs(rz[i, j]) > MAD_THRESHOLD)
        causes = []
        if r.get("d_loudness", 0) < -3:
            causes.append("loudness collapse, likely a track fade-out")
        elif r.get("d_loudness", 0) > 3:
            causes.append("loudness surge, likely a track fade-in")
        if r.get("t_start_s", 1e9) < 30:
            causes.append("within 30s of the track start")
        tl = r.get("track_len_s")
        if tl and r["t_start_s"] + r["duration_s"] > tl - 30:
            causes.append("within 30s of the track end")
        mm, ss = divmod(int(r["t_start_s"]), 60)
        records.append(dict(group=r.get("group"), track=r.get("track"),
                            at=f"{mm:02d}:{ss:02d}", because=dims,
                            cause="; ".join(causes) or
                            "no edge or fade heuristic matched -- inspect by ear"))
        if len(records) <= MAX_OUTLIER_PRINT:
            print(f"  {str(r.get('group'))[:9]:9s} {str(r.get('track'))[:44]:44s} "
                  f"@{mm:02d}:{ss:02d}\n    because: {dims}")
    if records and out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(records, indent=1))
    return ~mask


def run_extraction(audio_dirs, out_path, limit_tracks=None, min_track_s=60.0):
    """Extract every directory, merging with anything already cached.

    Merge semantics: re-extracting a subset of groups keeps every other
    group's cached rows, so a one-directory run does not wipe the corpus.
    """
    out_path = Path(out_path)
    groups = [Path(d).name for d in audio_dirs]
    rows = []
    if out_path.exists():
        old = json.loads(out_path.read_text())
        rows = [r for r in old if r.get("group") not in groups
                and "d_layers" in r]
        if rows:
            print(f"keeping {len(rows)} cached rows from other groups")

    for d in audio_dirs:
        group = Path(d).name
        tracks = collect_tracks(d)
        if limit_tracks:
            tracks = tracks[:limit_tracks]
        print(f"\n{group}: {len(tracks)} tracks under {d}")
        for t in tracks:
            try:
                r = extract_track(t, min_track_s=min_track_s)
            except Exception as e:
                print(f"  skip {t.name}: {e}")
                continue
            if r is None:
                continue
            for tr in r["transitions"]:
                rows.append(dict(tr, group=group, track=t.name,
                                 track_len_s=r.get("track_len_s")))
            print(f"  {t.name[:58]:58s} {len(r['transitions']):3d} transitions")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=1))
    n_tracks = len({(r["group"], r["track"]) for r in rows})
    n_groups = len({r["group"] for r in rows})
    print(f"\ncached {len(rows)} transitions from {n_tracks} tracks "
          f"across {n_groups} groups -> {out_path}")
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Extract transition descriptors from a corpus of audio.")
    ap.add_argument("--audio-dir", dest="audio_dir", nargs="+", required=True,
                    metavar="DIR",
                    help="one or more directories of audio; each becomes a "
                         "group label and is searched recursively")
    ap.add_argument("--out", default=None,
                    help=f"descriptor cache (default {DEFAULT_CACHE})")
    ap.add_argument("--limit-tracks", dest="limit_tracks", type=int, default=None,
                    help="only the first N tracks per directory (a quick check)")
    ap.add_argument("--min-track-s", dest="min_track_s", type=float, default=60.0,
                    help="skip tracks shorter than this (default 60 s)")
    args = ap.parse_args()
    run_extraction(args.audio_dir, args.out or DEFAULT_CACHE,
                   args.limit_tracks, args.min_track_s)


if __name__ == "__main__":
    main()
