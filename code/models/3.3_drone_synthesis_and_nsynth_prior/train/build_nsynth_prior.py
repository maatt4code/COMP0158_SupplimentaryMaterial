"""Select the sustained, bowed-string subset of NSynth used for the timbre prior.

Step 1 of Section 3.3. Reads NSynth's ``examples.json``, keeps notes that are
monophonic, sustained, harmonic and cleanly pitched, and writes the metadata
subset the next step analyses.

Filters:
  - instrument_family_str is in --families (default: string)
  - instrument_source_str is allowed for that family. Strings must be acoustic;
    NSynth organs are mostly drawbar electronic and its leads are synthetic, so
    demanding "acoustic" there would match nothing.
  - pitch within PITCH_LO..PITCH_HI, which avoids extreme registers
  - not percussive and not fast_decay, which excludes pizzicato and plucked

Run:
  # dry count, writes nothing
  python build_nsynth_prior.py

  # index the notes in place (preferred: no copying, so no inode-quota problem)
  python build_nsynth_prior.py --metadata-only

  # or extract the matching wavs
  python build_nsynth_prior.py --copy --out-dir /somewhere/nsynth_strings

  # against your own copy of NSynth
  python build_nsynth_prior.py --nsynth-root /path/to/nsynth-train --metadata-only

Next: analyse_nsynth_timbre.py
"""

import os
import json
import shutil
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import paths  # noqa: E402

PITCH_LO, PITCH_HI = 40, 72          # MIDI range, about E2 to C5
EXCLUDE_QUALITIES = {"percussive", "fast_decay"}

# Allowed family to sources mapping. See the note in the module docstring for
# why the allowed sources differ per family.
FAMILY_SOURCES = {
    "string":     {"acoustic"},
    "organ":      {"electronic", "acoustic"},
    "synth_lead": {"synthetic"},
}


def matches(meta: dict, families: set) -> bool:
    fam = meta.get("instrument_family_str")
    if fam not in families:
        return False
    if meta.get("instrument_source_str") not in FAMILY_SOURCES.get(fam, set()):
        return False
    if not (PITCH_LO <= meta.get("pitch", -1) <= PITCH_HI):
        return False
    if EXCLUDE_QUALITIES & set(meta.get("qualities_str", [])):
        return False
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Select the sustained bowed-string subset of NSynth.")
    paths.add_arg(ap, "nsynth")
    ap.add_argument("--copy", action="store_true",
                    help="copy matching wavs into the output dir (default: dry count only)")
    ap.add_argument("--metadata-only", dest="metadata_only", action="store_true",
                    help="write selected_metadata.json without copying any wav; "
                         "analyse_nsynth_timbre.py then reads the audio in place "
                         "via --audio-dir")
    ap.add_argument("--families", type=str, nargs="+", default=["string"],
                    choices=sorted(FAMILY_SOURCES),
                    help="NSynth instrument families to include")
    ap.add_argument("--out-dir", dest="out_dir", type=str, default=None,
                    help="output directory (default: ../data/nsynth_strings, "
                         "which is git-ignored)")
    args = ap.parse_args()

    nsynth_root = paths.require("nsynth", args.nsynth_root)
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(__file__).resolve().parents[1] / "data" / "nsynth_strings"
    print(f"NSynth root: {nsynth_root}")
    print(f"Output dir : {out_dir}")

    examples_path = os.path.join(nsynth_root, "examples.json")
    audio_dir = os.path.join(nsynth_root, "audio")
    if not os.path.exists(examples_path):
        raise SystemExit(f"examples.json not found under {nsynth_root}")

    print(f"\nLoading metadata: {examples_path}")
    with open(examples_path, "r") as f:
        examples = json.load(f)
    print(f"Total notes in train set: {len(examples):,}")

    families = set(args.families)
    selected = [k for k, v in examples.items() if matches(v, families)]
    print(f"\nMatching ({'/'.join(sorted(families))}, sustained, "
          f"pitch {PITCH_LO}-{PITCH_HI}): {len(selected):,}")

    fam_counts = Counter(examples[k]["instrument_family_str"] for k in selected)
    inst = Counter(examples[k]["instrument_str"] for k in selected)
    pitch = Counter(examples[k]["pitch"] for k in selected)
    vel = Counter(examples[k]["velocity"] for k in selected)
    print(f"  per family:           {dict(fam_counts)}")
    print(f"  distinct instruments: {len(inst)}")
    print(f"  velocities present:   {sorted(vel)}")
    print(f"  pitch span:           {min(pitch) if pitch else '-'}..{max(pitch) if pitch else '-'}")

    if not (args.copy or args.metadata_only):
        print("\nDry run. Re-run with --metadata-only to index these notes in "
              "place, or --copy to extract the wavs.")
        return

    os.makedirs(out_dir, exist_ok=True)
    copied = 0
    if args.copy:
        for k in selected:
            src = os.path.join(audio_dir, k + ".wav")
            dst = os.path.join(out_dir, k + ".wav")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied += 1

    meta_path = os.path.join(out_dir, "selected_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({k: examples[k] for k in selected}, f, indent=2)

    if args.copy:
        print(f"\nCopied {copied:,} wavs to {out_dir}")
    else:
        print(f"\nMetadata only: the wavs stay in {audio_dir}")
    print(f"Wrote metadata subset: {meta_path}")


if __name__ == "__main__":
    main()
