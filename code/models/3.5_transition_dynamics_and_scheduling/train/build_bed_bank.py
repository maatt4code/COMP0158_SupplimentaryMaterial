"""Build the conductor's background-bed bank from the curation log.

The append-safe bridge between curation and deployment. It reads
`human_ratings/overlay_curation.csv`, keeps the LATEST verdict per bed file,
resolves each keeper to its audio path, valence-arousal rating, gain and mode,
and writes `bed_bank.json` for the conductor to load. Adding a bed is
therefore: audition it (the log appends) -> re-run this -> the conductor picks
it up next session, with no conductor code change.

WHY THIS IS ON THE TRAINING SIDE. It needs two corpora on disk and it reads
curation verdicts. The conductor must do neither, so only `select_bed` ships
there, in `conductor/engine/bed_bank.py`. This is the same split every other
boot-time fit got.

THE BEDS ARE NOT REDISTRIBUTED. They come from two public corpora, and the
bank keys them by their ORIGINAL filenames so a fetched copy resolves without
translation:

  ESC-50            https://github.com/karolpiczak/ESC-50   (CC BY-NC 3.0)
  Emo-Soundscapes   https://metatlas.github.io/            (Creative Commons)

Only the Emo-Soundscapes half carries valence-arousal ratings; ESC-50 beds are
recorded with `va: null` and are then eligible in any mood. See
`conductor/assets/ATTRIBUTION.md` for the demo subset that does ship.

Run:
  python build_bed_bank.py --bed-root /path/to/datasets
  python build_bed_bank.py --bed-root DIR --out ../../../conductor/weights/bed_bank.json
  python build_bed_bank.py --selftest         # synthetic rows, no corpora
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
CURATION = SECTION / "human_ratings" / "overlay_curation.csv"
OUT = SECTION / "data" / "bed_bank.json"

# ESC-50 categories that are usable as an ambient bed. The rest of the corpus
# is foreground material -- dog barks, sneezes, door knocks -- which reads as
# an event rather than an environment and will not sit under a drone.
AMBIENT_ESC = ["rain", "thunderstorm", "sea_waves", "wind", "crickets",
               "insects", "frog", "chirping_birds", "crackling_fire",
               "church_bells", "water_drops", "pouring_water", "clock_tick",
               "breathing", "footsteps", "engine", "train"]

# Emo-Soundscapes' own category labels are too coarse to select on, so beds
# are gathered by searching the free-text tags instead. Each key is the bed
# type the conductor sees; the values are the tag substrings that find it.
EMO_TYPES = {
    "birds": ["birds", "birdsong", "bird"],
    "rain": ["rain"],
    "thunder-storm": ["thunder", "storm"],
    "wind": ["wind"],
    "water-stream": ["water", "stream", "river", "sea", "ocean"],
    "night-insects": ["cricket", "insect", "frog"],
    "bells-church": ["bell", "church", "chapel"],
    "street-crowd": ["street", "crowd"],
    "dark-ambience": ["drone"],
}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _read_csv(p):
    """Tolerant CSV read. The Emo-Soundscapes metadata ships with mixed line
    endings, so normalise before parsing rather than trusting the dialect
    sniffer."""
    txt = Path(p).read_text(errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    return list(csv.DictReader(io.StringIO(txt)))


def read_keeps(path=CURATION):
    """The current keepers: latest verdict per bed file, filtered to 'keep'.

    The log is append-only, so the LAST row for a file is its standing
    verdict. A bed re-judged 'no' after an earlier 'keep' therefore drops out,
    and one re-judged 'keep' comes back -- which is what makes re-auditioning
    a bed a matter of appending a row rather than editing history.
    """
    if not Path(path).exists():
        return []
    latest = {}
    for r in _read_csv(path):
        if (r.get("verdict") or "").strip() and r.get("bed_file"):
            latest[r["bed_file"]] = r
    return [r for r in latest.values() if r["verdict"] == "keep"]


def candidate_index(bed_root):
    """{bed_file: {path, dataset, bed_type, emo_v, emo_a}} across both corpora.

    Returns what it can find: a missing corpus yields fewer candidates rather
    than an error, and the keepers it cannot resolve are reported as
    unresolved instead of being silently dropped.
    """
    root = Path(bed_root)
    esc, emo = root / "ESC-50-master", root / "Emo-Soundscapes" / "Emo-Soundscapes"
    idx = {}

    meta = esc / "meta" / "esc50.csv"
    if meta.exists():
        for r in _read_csv(meta):
            if r["category"] in AMBIENT_ESC:
                idx[r["filename"]] = dict(
                    dataset="esc50", bed_type=r["category"],
                    path=str(esc / "audio" / r["filename"]),
                    bed_file=r["filename"], emo_v="", emo_a="")

    if emo.exists():
        def ratings(name):
            f = emo / "Emo-Soundscapes-Ratings" / f"{name}.csv"
            out = {}
            if f.exists():
                for r in _read_csv(f):
                    k = list(r.keys())
                    try:
                        out[r[k[0]].strip()] = float(r[k[1]])
                    except (ValueError, TypeError):
                        pass
            return out
        val, aro = ratings("Valence"), ratings("Arousal")
        audio600 = emo / "Emo-Soundscapes-Audio" / "600_Sounds"
        rows = []
        for cat_csv in (emo / "Emo-Soundscapes-Metadata").glob("*.csv"):
            for r in _read_csv(cat_csv):
                r["_cat"] = cat_csv.stem
                rows.append(r)
        for typ, keys in EMO_TYPES.items():
            for r in rows:
                if not any(k in r.get("Tags", "").lower() for k in keys):
                    continue
                wav = r["FileName"].strip().replace(".mp3", ".wav")
                p = audio600 / r["_cat"] / wav
                if p.exists():
                    idx[wav] = dict(dataset="emo", bed_type=typ, path=str(p),
                                    bed_file=wav, emo_v=val.get(wav, ""),
                                    emo_a=aro.get(wav, ""))
    return idx


def build_bank(keeps, cand_index):
    """One entry per kept bed, with its path and VA resolved.

    Paths are stored as BASENAMES, not absolute paths: the corpora are fetched
    rather than redistributed, so an absolute path would encode the build
    machine's layout and break on every other machine. The conductor resolves
    a basename against its own assets directory.
    """
    beds = []
    for r in keeps:
        bf = r["bed_file"]
        cand = cand_index.get(bf, {})
        # Prefer the VA logged at curation time; fall back to the corpus's own
        # rating. They agree when both exist; the log wins because it is what
        # the curator actually saw.
        v = _f(r["emo_v"]) if (r.get("emo_v") or "").strip() else _f(cand.get("emo_v"))
        a = _f(r["emo_a"]) if (r.get("emo_a") or "").strip() else _f(cand.get("emo_a"))
        beds.append(dict(
            bed_file=bf,
            dataset=r.get("dataset") or cand.get("dataset"),
            bed_type=r.get("bed_type") or cand.get("bed_type"),
            path=Path(cand["path"]).name if cand.get("path") else None,
            va=[v, a] if (v is not None and a is not None) else None,
            gain_db=_f(r.get("gain_db")),
            mode=(r.get("mode") or "constant"),
            approved_drones=[],
            resolved=bool(cand.get("path"))))
    return beds


def attach_pairings(beds, path=CURATION):
    """Record which drone(s) each bed was approved OVER.

    Most beds were auditioned bed-only, so this pairing evidence is sparse. It
    is recorded as VALIDATION material for the conductor's acoustic drone-fit
    to check itself against -- not as something to gate on, which sparse data
    could not support.
    """
    by_file = defaultdict(set)
    if Path(path).exists():
        for r in _read_csv(path):
            if (r.get("verdict") == "keep" and r.get("bed_only") == "0"
                    and (r.get("drone_clip") or "").strip()):
                by_file[r["bed_file"]].add(r["drone_clip"])
    for b in beds:
        b["approved_drones"] = sorted(by_file.get(b["bed_file"], ()))
    return beds


def build(bed_root, curation=CURATION, out=OUT):
    keeps = read_keeps(curation)
    idx = candidate_index(bed_root) if bed_root else {}
    beds = attach_pairings(build_bank(keeps, idx), curation)
    n_res = sum(b["resolved"] for b in beds)
    n_va = sum(b["va"] is not None for b in beds)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        dict(beds=beds,
             note=("Bed audio is NOT redistributed: entries key the original "
                   "ESC-50 / Emo-Soundscapes filenames. See the module "
                   "docstring for the corpora and their licences."),
             n_beds=len(beds), n_resolved=n_res, n_with_va=n_va), indent=1))
    print(f"{len(beds)} kept beds, {n_res} resolved, {n_va} carry a VA rating")
    by_type = defaultdict(int)
    for b in beds:
        by_type[b["bed_type"]] += 1
    for t, n in sorted(by_type.items()):
        print(f"    {t:16s} {n}")
    print(f"wrote {out}")
    return beds


def _selftest():
    """Synthetic curation rows, no corpora on disk."""
    import tempfile
    rows = [dict(timestamp="2026-07-01T00:00:00", verdict="keep", dataset="emo",
                 bed_type="birds", bed_file="a.wav", emo_v="0.5", emo_a="0.2",
                 drone_clip="d1.wav", gain_db="-18", bed_only="0",
                 mode="constant", duck_depth="", burst_s="", render_seq="",
                 comment=""),
            # a later 'no' must supersede the earlier 'keep'
            dict(timestamp="2026-07-02T00:00:00", verdict="keep", dataset="esc50",
                 bed_type="rain", bed_file="b.wav", emo_v="", emo_a="",
                 drone_clip="", gain_db="-20", bed_only="1", mode="constant",
                 duck_depth="", burst_s="", render_seq="", comment=""),
            dict(timestamp="2026-07-03T00:00:00", verdict="no", dataset="esc50",
                 bed_type="rain", bed_file="b.wav", emo_v="", emo_a="",
                 drone_clip="", gain_db="", bed_only="1", mode="constant",
                 duck_depth="", burst_s="", render_seq="", comment="")]
    with tempfile.TemporaryDirectory() as tmp:
        c = Path(tmp) / "curation.csv"
        with open(c, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        keeps = read_keeps(c)
        assert [k["bed_file"] for k in keeps] == ["a.wav"], keeps
        idx = {"a.wav": dict(path="/somewhere/a.wav", dataset="emo",
                             bed_type="birds", emo_v="0.5", emo_a="0.2")}
        beds = attach_pairings(build_bank(keeps, idx), c)
        assert beds[0]["va"] == [0.5, 0.2], beds
        assert beds[0]["path"] == "a.wav", "paths must be stored as basenames"
        assert beds[0]["approved_drones"] == ["d1.wav"], beds
        assert beds[0]["resolved"] is True
        # an unresolvable keeper is reported, not dropped
        beds2 = build_bank(keeps, {})
        assert beds2[0]["resolved"] is False and beds2[0]["path"] is None
    print("selftest OK: latest-verdict-wins, VA carried, basenames stored, "
          "pairings attached, unresolved beds reported not dropped")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bed-root", type=Path,
                    help="directory holding ESC-50-master/ and "
                         "Emo-Soundscapes/; without it every bed is "
                         "unresolved and the bank records that")
    ap.add_argument("--curation", type=Path, default=CURATION)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    build(args.bed_root, args.curation, args.out)


if __name__ == "__main__":
    main()
