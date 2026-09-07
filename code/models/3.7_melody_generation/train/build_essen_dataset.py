"""Build the transformer's training set from the fitted grammar or from MIDI.

Step 2 of Section 3.7. The transformer trains on phrase contours, and there are
two ways to supply them: the phrase shapes already fitted by
build_markov_grammar.py, or a directory of MIDI stems.

Supports TWO data generation pipelines:
1. Essen Folksong Corpus (built via `python SideProjects/melodic_drone/essen_model.py --build`).
2. Custom MIDI Stems (parses a directory of .mid / .midi files from ambient compositions).

Usage:
  python dataset_builder.py --essen                     # Parses pre-bundled Essen corpus
  python dataset_builder.py --midi_dir ./my_midi_files   # Parses custom ambient MIDI directory
"""

import os
import sys
import json
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
DATA_DIR = SECTION / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DATASET_PATH = DATA_DIR / "ambient_melodic_dataset.json"


def build_from_essen():
    """Generates dataset from the pre-bundled Essen Folksong collection."""
    essen_script = HERE / "build_markov_grammar.py"
    essen_json = SECTION / "weights" / "markov_order2.json"
    
    print(f"[dataset_builder] Building Essen corpus using {essen_script}...")
    # Run build if json missing
    if not essen_json.exists():
        os.system(f"python {essen_script}")
        
    if not essen_json.exists():
        raise FileNotFoundError(f"Could not locate or build {essen_json}")
        
    with open(essen_json, 'r') as f:
        data = json.load(f)
        
    shapes = data.get("shapes", {})
    sequences = []
    for shape_str, count in shapes.items():
        steps = shape_str.split(";")
        seq = []
        for step in steps:
            parts = step.split(",")
            if len(parts) >= 2:
                try:
                    pitch_off = int(parts[0])
                    dur_ratio = float(parts[1])
                    seq.append({
                        "pitch_offset": pitch_off,
                        "duration_ratio": dur_ratio
                    })
                except ValueError:
                    continue
        if len(seq) >= 4:
            sequences.append(seq)
            
    dataset = {
        "source": "essen_folksong_collection",
        "n_phrases": len(sequences),
        "sequences": sequences
    }
    
    with open(OUTPUT_DATASET_PATH, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"[dataset_builder] Saved {len(sequences)} phrase sequences to {OUTPUT_DATASET_PATH}")
    return dataset


def build_from_midi_directory(midi_dir):
    """Parses custom ambient MIDI files into relative pitch & duration sequences."""
    midi_path = Path(midi_dir)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI directory not found: {midi_dir}")
        
    try:
        import music21
    except ImportError:
        raise ImportError("music21 is required for MIDI parsing. Install via `pip install music21`.")
        
    midi_files = list(midi_path.glob("*.mid")) + list(midi_path.glob("*.midi"))
    print(f"[dataset_builder] Found {len(midi_files)} MIDI files in {midi_dir}...")
    
    sequences = []
    for m_file in midi_files:
        try:
            score = music21.converter.parse(str(m_file))
            for part in score.parts:
                notes = part.flat.notes
                seq = []
                last_pitch = None
                for n in notes:
                    if isinstance(n, music21.note.Note):
                        pitch_midi = n.pitch.midi
                        ql = float(n.quarterLength)
                        offset = 0 if last_pitch is None else pitch_midi - last_pitch
                        last_pitch = pitch_midi
                        seq.append({"pitch_offset": offset, "duration_ratio": ql})
                if len(seq) >= 4:
                    sequences.append(seq)
        except Exception as e:
            print(f"[dataset_builder] Warning skipping {m_file.name}: {e}")
            
    dataset = {
        "source": str(midi_dir),
        "n_phrases": len(sequences),
        "sequences": sequences
    }
    
    with open(OUTPUT_DATASET_PATH, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"[dataset_builder] Saved {len(sequences)} MIDI phrase sequences to {OUTPUT_DATASET_PATH}")
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset Builder for Melodic Transformer")
    parser.add_argument("--essen", action="store_true", help="Build from Essen Folksong Corpus")
    parser.add_argument("--midi_dir", type=str, default=None, help="Directory of custom MIDI files")
    args = parser.parse_args()

    if args.midi_dir:
        build_from_midi_directory(args.midi_dir)
    else:
        build_from_essen()
