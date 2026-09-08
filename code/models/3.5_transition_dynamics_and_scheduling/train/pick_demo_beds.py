"""Choose the demo bed subset that ships with the conductor, and attribute it.

The bed corpora are NOT redistributed: ESC-50 is CC BY-NC and
Emo-Soundscapes is assembled from Creative Commons Freesound material, so the
supplementary material documents and fetches them the same way it does NSynth
and Essen. But a conductor whose bed layer is silent until someone downloads
two corpora demonstrates nothing, so ONE bed per type ships, with full per-clip
attribution, and the rest is fetched.

WHICH ONE, and why it is not a hand-pick. Selection is deterministic and
licence-first: CC0 before CC-BY before anything else, then a bed carrying a
valence-arousal rating before one without, then alphabetical. Preferring the
most permissive licence keeps the shipped subset as re-distributable as
possible; preferring a VA-rated bed means the emotion gate is demonstrable
rather than theoretical. ESC-50 beds carry no VA and are eligible in any mood,
so the subset ends up exercising both selection paths.

Attribution is READ FROM THE CORPORA, never written by hand: ESC-50 records
each clip's Freesound origin, contributor and licence in its own LICENSE file,
and Emo-Soundscapes records a Freesound id and URL per clip. A hand-copied
credit is a credit that goes stale.

Run:
  python pick_demo_beds.py --bed-root DIR --out ../../../conductor/assets/beds
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
BANK = SECTION.parent.parent / "conductor" / "weights" / "bed_bank.json"
OUT = SECTION.parent.parent / "conductor" / "assets" / "beds"

# Most permissive first. A bed under a licence not in this list still ships if
# it is the only one of its type, but it sorts last and its licence is stated.
LICENCE_RANK = ["CC0", "CC-BY", "CC-BY-SA", "CC-Sampling+"]


def _read_csv(p):
    txt = Path(p).read_text(errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    return list(csv.DictReader(io.StringIO(txt)))


def esc_attribution(bed_root):
    """{clip stem: {source, url, who, licence}} from ESC-50's own LICENSE."""
    f = Path(bed_root) / "ESC-50-master" / "LICENSE"
    if not f.exists():
        return {}
    pat = re.compile(r"\[([\w-]+)\.ogg\]: clip derived from (.+?) "
                     r"\((http\S+?)\) by (.+?) \[(.+?)\]")
    return {m.group(1): dict(source=m.group(2), url=m.group(3),
                             who=m.group(4), licence=m.group(5))
            for m in pat.finditer(f.read_text(errors="replace"))}


def emo_attribution(bed_root):
    """{filename: {FsID, FsUrl, Tags}} from the Emo-Soundscapes metadata."""
    root = (Path(bed_root) / "Emo-Soundscapes" / "Emo-Soundscapes"
            / "Emo-Soundscapes-Metadata")
    out = {}
    if root.is_dir():
        for c in root.glob("*.csv"):
            for r in _read_csv(c):
                out[r["FileName"].strip().replace(".mp3", ".wav")] = r
    return out


def _stem(bed_file):
    """ESC-50 keys its licence table by the clip id without the class suffix:
    `2-84965-A-23.wav` is recorded as `2-84965-A`."""
    return bed_file.rsplit(".", 1)[0].rsplit("-", 1)[0]


def choose(bank, esc):
    """One bed per type, licence-first then VA-first then alphabetical."""
    by_type = collections.defaultdict(list)
    for b in bank:
        if b.get("resolved"):
            by_type[b["bed_type"]].append(b)

    def rank(b):
        lic = esc.get(_stem(b["bed_file"]), {}).get("licence") \
            if b["dataset"] == "esc50" else None
        li = LICENCE_RANK.index(lic) if lic in LICENCE_RANK else len(LICENCE_RANK)
        return (li, b["va"] is None, b["bed_file"])

    return [sorted(v, key=rank)[0] for _, v in sorted(by_type.items())]


def attribute(picks, esc, emo):
    rows = []
    for b in picks:
        f = b["bed_file"]
        if b["dataset"] == "esc50":
            a = esc.get(_stem(f), {})
            rows.append(dict(bed_file=f, bed_type=b["bed_type"], corpus="ESC-50",
                             licence=a.get("licence", "see the ESC-50 LICENSE"),
                             credit=a.get("who", ""), url=a.get("url", ""),
                             source=a.get("source", ""), va=b["va"]))
        else:
            r = emo.get(f, {})
            rows.append(dict(bed_file=f, bed_type=b["bed_type"],
                             corpus="Emo-Soundscapes",
                             licence="Creative Commons (per-clip, via Freesound)",
                             credit="", url=r.get("FsUrl", ""),
                             source=f"Freesound {r.get('FsID', '?')}", va=b["va"]))
    return rows


def write_attribution(rows, out_dir):
    lines = [
        "# Bed audio — sources and licences",
        "",
        "Every file in this directory is third-party audio, redistributed here",
        "under its own licence. **Nothing in it was recorded for this project.**",
        "",
        "One bed per type ships so the conductor's bed layer works out of the",
        "box. The full curated bank is larger and is NOT redistributed: fetch",
        "the two corpora and point `$DRONE_BEDS` at them, or rebuild the bank",
        "with `models/3.5_transition_dynamics_and_scheduling/train/build_bed_bank.py`.",
        "",
        "## Corpora",
        "",
        "| Corpus | Licence | Where |",
        "|---|---|---|",
        "| ESC-50 | CC BY-NC 3.0 (the ESC-10 subset is CC BY 3.0) | <https://github.com/karolpiczak/ESC-50> |",
        "| Emo-Soundscapes | Creative Commons, assembled from Freesound | <https://metatlas.github.io/> |",
        "",
        "Both are non-commercial corpora. This material is academic",
        "supplementary work and redistributes only the subset below.",
        "",
        "## The shipped beds",
        "",
        "| File | Type | Corpus | Licence | Credit | Original |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        credit = r["credit"] or "—"
        url = f"[{r['source'] or 'source'}]({r['url']})" if r["url"] else "—"
        lines.append(f"| `{r['bed_file']}` | {r['bed_type']} | {r['corpus']} "
                     f"| {r['licence']} | {credit} | {url} |")
    lines += ["",
              "`va` is the corpus's own valence-arousal rating where it has one;",
              "ESC-50 carries none, and those beds are eligible in any mood.",
              ""]
    (Path(out_dir) / "ATTRIBUTION.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bed-root", type=Path, required=True,
                    help="directory holding ESC-50-master/ and Emo-Soundscapes/")
    ap.add_argument("--bank", type=Path, default=BANK)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    data = json.loads(args.bank.read_text())
    bank = data["beds"] if isinstance(data, dict) else data
    esc, emo = esc_attribution(args.bed_root), emo_attribution(args.bed_root)
    picks = choose(bank, esc)
    args.out.mkdir(parents=True, exist_ok=True)

    # Resolve each pick back to real audio and copy it in.
    root = Path(args.bed_root)
    copied, missing = 0, []
    for b in picks:
        hits = list(root.rglob(b["bed_file"]))
        if not hits:
            missing.append(b["bed_file"])
            continue
        shutil.copy2(hits[0], args.out / b["bed_file"])
        copied += 1
    rows = attribute(picks, esc, emo)
    write_attribution(rows, args.out)

    print(f"{copied} of {len(picks)} demo beds copied to {args.out}")
    for r in rows:
        print(f"    {r['bed_type']:15s} {r['bed_file']:34s} "
              f"{r['corpus']:16s} {r['licence']}")
    if missing:
        print("  NOT FOUND:", ", ".join(missing))
    print(f"wrote {args.out / 'ATTRIBUTION.md'}")


if __name__ == "__main__":
    main()
