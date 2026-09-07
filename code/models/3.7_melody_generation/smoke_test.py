"""Smoke test for Section 3.7. Needs no corpus, no music21, no network.

Runs the real code against the shipped grammar and checkpoints.

Run:
  python smoke_test.py

Exit code 0 means every check passed.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train"
INFER = HERE / "inference"
WEIGHTS = HERE / "weights"
PY = sys.executable
FAILED = []

TRAIN_SCRIPTS = ["build_markov_grammar.py", "build_essen_dataset.py",
                 "train_transformer.py", "benchmark_configs.py",
                 "pick_melody_anchor.py"]
INFER_MODULES = ["grammar.py", "melody_markov.py", "transformer_model.py",
                 "melody_transformer.py"]
SIZES = [("L3_d128", 612_387), ("L6_d128", 1_207_203), ("L6_d256", 4_773_667)]


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def module_scope_imports(path):
    names = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def main():
    print("Section 3.7 smoke test\n")
    sys.path.insert(0, str(INFER))
    sys.path.insert(0, str(TRAIN))

    print("1. structure")
    for s in TRAIN_SCRIPTS:
        check(f"train/{s} exists", (TRAIN / s).exists())
    for m in INFER_MODULES:
        check(f"inference/{m} exists", (INFER / m).exists())
    check("the fitted grammar ships", (WEIGHTS / "markov_order2.json").exists())
    check("the melody anchor is frozen", (WEIGHTS / "melody_anchor.json").exists())
    for tag, _ in SIZES:
        check(f"checkpoint {tag} ships",
              (WEIGHTS / f"melodic_transformer_{tag}.pt").exists())
        check(f"checkpoint {tag} has its sidecar",
              (WEIGHTS / f"melodic_transformer_{tag}.json").exists())
    check("no promoted-copy alias ships",
          not (WEIGHTS / "melodic_transformer_cpu.pt").exists())

    print("\n2. inference boundary")
    for m in INFER_MODULES:
        imports = module_scope_imports(INFER / m)
        check(f"{m} does not import a training script",
              not any(t.replace(".py", "") in imports for t in TRAIN_SCRIPTS),
              str(sorted(imports)) if any(t.replace(".py", "") in imports
                                          for t in TRAIN_SCRIPTS) else "")
        check(f"{m} does not import music21", "music21" not in imports)
    # Check IMPORTS, not prose: the docstring legitimately names the library
    # it is explaining the absence of.
    check("the grammar loader imports no corpus library",
          not (module_scope_imports(INFER / "grammar.py") & {"music21"}))
    check("the trainer imports the architecture from inference",
          "transformer_model" in module_scope_imports(TRAIN / "train_transformer.py"))

    print("\n3. naming and identifiers")
    banned = ["JAMAI", "jamai", "Matthew", "matthew", "maatt", "Gemini", "GEMINI",
              "kyle bobby", "KyleBobbyDunn", '"boc"', '"kbd"']
    for path in sorted(TRAIN.glob("*.py")) + sorted(INFER.glob("*.py")):
        hits = [b for b in banned if b in path.read_text()]
        check(f"{path.name} carries no retired name", not hits, str(hits))
    import melody_markov as mm
    check("styles are named by their character, not by an artist",
          set(mm.STYLES) == {"folk", "chant", "pentatonic_fast", "pentatonic_slow"},
          str(sorted(mm.STYLES)))

    print("\n4. the fitted grammar")
    import grammar as G
    g = G.load()
    check("order 2", g.order == 2, str(g.order))
    check("the corpus size matches the reported figures",
          g.n_tunes == 8514 and g.n_phrases == 22119,
          f"{g.n_tunes} tunes, {g.n_phrases} phrases")
    check("the chain has many states", len(g.trans) > 1000, str(len(g.trans)))
    bank = G.load_bank()
    check("the phrase bank holds contours", len(bank) > 1000, str(len(bank)))
    check("some contours are arch-shaped", int(bank.arch.sum()) > 0,
          f"{int(bank.arch.sum())} of {len(bank)}")

    rng = np.random.default_rng(0)
    seq = [g.start(rng)]
    for _ in range(40):
        seq.append(g.next(rng, seq))
    check("sampling produces (interval, duration) pairs",
          all(isinstance(x, tuple) and len(x) == 2 for x in seq))
    durs = [d for _, d in seq]
    check("sampled rhythm is not flat", max(durs) / min(durs) >= 2.0,
          f"{max(durs)/min(durs):.1f}:1")
    # Backoff must never dead-end from an unseen state.
    check("an unseen state backs off rather than failing",
          g.next(rng, [(7, 4.0), (-7, 0.25)]) is not None)

    syn = G.synthetic_model()
    gs = G.EssenGrammar(syn)
    s2 = [gs.start(rng)]
    for _ in range(30):
        s2.append(gs.next(rng, s2))
    d2 = [d for _, d in s2]
    check("the synthetic fixture carries its planted 4:1 contrast",
          max(d2) / min(d2) >= 3.0, f"{max(d2)/min(d2):.1f}:1")

    print("\n5. melody generation")
    notes, p = mm.generate_melody(0.5, 0.0, root_hz=110.0, duration_s=60.0, seed=1)
    check("a melody is produced", len(notes) > 0, f"{len(notes)} events")
    check("positive valence selects a major scale", "major" in p["scale"],
          p["scale"])
    _, pn = mm.generate_melody(-0.5, 0.0, duration_s=60.0, seed=1)
    check("negative valence selects a minor scale", "minor" in pn["scale"],
          pn["scale"])
    _, pc = mm.generate_melody(0.0, -0.9, duration_s=60.0, seed=2)
    _, pa = mm.generate_melody(0.0, +0.9, duration_s=60.0, seed=2)
    check("higher arousal shortens the notes", pa["note_s"] < pc["note_s"],
          f"calm {pc['note_s']:.1f}s vs active {pa['note_s']:.1f}s")
    check("the line breathes", any(f == mm.REST for f, _ in notes))
    band = mm.scale_pitches(110.0, p["scale"], p["span"], p["register"])
    pitched = [f for f, _ in notes if f != mm.REST]
    check("the line stays in its register",
          min(band) - 1e-6 <= min(pitched) and max(pitched) <= max(band) + 1e-6)
    again, _ = mm.generate_melody(0.5, 0.0, root_hz=110.0, duration_s=60.0, seed=1)
    check("generation is deterministic given a seed", notes == again)

    print("\n6. the transformer")
    import torch
    from transformer_model import MicroMelodicTransformer, VOCAB_SIZE
    for tag, n_params in SIZES:
        L, d = (3, 128) if tag == "L3_d128" else ((6, 128) if tag == "L6_d128" else (6, 256))
        m = MicroMelodicTransformer(d_model=d, n_layers=L,
                                    n_heads=4 if d == 128 else 8,
                                    d_ffn=512 if d == 128 else 1024)
        got = sum(x.numel() for x in m.parameters())
        check(f"{tag} architecture matches the reported parameter count",
              got == n_params, f"{got:,} vs {n_params:,}")
        ck = torch.load(WEIGHTS / f"melodic_transformer_{tag}.pt",
                        map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict") or ck.get("state_dict") or ck
        check(f"{tag} checkpoint loads into that architecture",
              set(sd) == set(m.state_dict()))
    m = MicroMelodicTransformer(d_model=128, n_layers=3, n_heads=4, d_ffn=512)
    out = m(torch.zeros(2, 8, dtype=torch.long), torch.zeros(2, 2))
    check("forward returns logits over the vocabulary",
          tuple(out.shape) == (2, 8, VOCAB_SIZE), str(tuple(out.shape)))
    gen = m.generate(torch.tensor([0.3, -0.2]), max_len=12)
    check("generation returns a token sequence", len(gen) >= 1, str(len(gen)))

    print("\n7. the frozen melody anchor")
    anc = json.loads((WEIGHTS / "melody_anchor.json").read_text())
    check("the anchor records its index", isinstance(anc["anchor_idx"], int))
    check("the anchor records its candidates and bands",
          anc["n_candidates"] >= 1 and "centroid_range" in anc,
          f"{anc['n_candidates']} candidates")
    check("the frozen anchor is the one the picker chooses",
          mm.pick_melody_anchor() == anc["anchor_idx"], str(anc["anchor_idx"]))
    check("the anchor reads Section 3.4.2's shipped pool",
          mm.POOL_RATINGS.exists() and "3.4.2" in str(mm.POOL_RATINGS))

    print("\n8. command line")
    for s in TRAIN_SCRIPTS:
        r = subprocess.run([PY, str(TRAIN / s), "--help"], capture_output=True,
                           text=True)
        check(f"{s} --help", r.returncode == 0,
              r.stderr.strip().splitlines()[-1] if r.returncode else "")
    r = subprocess.run([PY, str(TRAIN / "build_markov_grammar.py"), "--help"],
                       capture_output=True, text=True)
    check("the grammar builder exposes --essen-root", "--essen-root" in r.stdout)
    r = subprocess.run([PY, str(TRAIN / "build_markov_grammar.py"), "--selftest"],
                       capture_output=True, text=True)
    check("the grammar builder's selftest passes", r.returncode == 0)
    r = subprocess.run([PY, str(INFER / "melody_markov.py"), "--selftest"],
                       capture_output=True, text=True)
    check("the generator's selftest passes", r.returncode == 0,
          r.stderr.strip().splitlines()[-1] if r.returncode else "")

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
