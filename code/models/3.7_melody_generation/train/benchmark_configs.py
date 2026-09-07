"""Benchmark the three transformer sizes: CPU latency, and audio to listen to.

Step 4 of Section 3.7, and how the size comparison in the report was produced.
It times symbolic generation on CPU for each checkpoint across the four affect
quadrants, because latency is the constraint that sized this model in the first
place -- it runs live inside the conductor.

Rendering audio needs the arranger, which belongs to Section 3.8. Without it
this still reports the timings, which is the part the size comparison rests on.

Run:
  python benchmark_configs.py --out-dir ../data/benchmark
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
sys.path.insert(0, str(SECTION / "inference"))

from melody_transformer import (LiveMelodicDroneRuntime,      # noqa: E402
                                aw_rms, AW_TARGET, _ARRANGER)
try:
    from melody_transformer import RENDER_KW
except ImportError:
    RENDER_KW = {}

def run_benchmark_and_render(out_dir=None):
    out_dir = Path(out_dir) if out_dir else (SECTION / "data" / "benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_names = ["L3_d128", "L6_d128", "L6_d256"]
    quadrants = [
        ("calm_peaceful", 0.6, -0.5),
        ("sad_melancholic", -0.6, -0.5),
        ("bright_energetic", 0.7, 0.6),
        ("dark_tense", -0.7, 0.6),
    ]
    
    print("=" * 70)
    print("PyTorch CPU Inference Benchmark & Audio Generation")
    print("=" * 70)
    
    bench_results = {}
    
    for tag in ckpt_names:
        ckpt_pt = HERE / "checkpoints" / f"melodic_transformer_{tag}.pt"
        if not ckpt_pt.exists():
            print(f"[*] Checkpoint {ckpt_pt.name} not found, skipping...")
            continue
            
        rt = LiveMelodicDroneRuntime(ckpt_name=tag, force_cpu=True)
        
        # Benchmark CPU inference timing over 50 phrase generations
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            _notes = rt.generate_notes(valence=0.5, arousal=0.2, duration_s=10.0)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms
            
        avg_lat = np.mean(latencies)
        p95_lat = np.percentile(latencies, 95)
        bench_results[tag] = {
            "mean_ms": avg_lat,
            "p95_ms": p95_lat,
            "layers": rt.model.n_layers if rt.model else 0,
            "d_model": rt.model.d_model if rt.model else 0,
        }
        
        print(f"[{tag}] Model L={rt.model.n_layers}, d={rt.model.d_model} | CPU Latency: {avg_lat:.2f} ms (p95: {p95_lat:.2f} ms)")
        
        # Render sample tracks for each quadrant
        for q_name, v, a in quadrants:
            fname = f"demo_{q_name}_60s_{tag}.wav"
            drone_theta = dict(rt.engine.retrieve(v, a))
            drone_theta["noise_level"] = min(0.01, float(drone_theta.get("noise_level", 0.01)))
            if "harm_dist" in drone_theta:
                h_dr = np.asarray(drone_theta["harm_dist"], float).copy()
                h_dr[5:] *= 0.05
                drone_theta["harm_dist"] = h_dr / (np.sum(h_dr) + 1e-9)
            drone_wps = [drone_theta] * 30
            drone_raw = np.asarray(rt.renderer.render_waypoints(drone_wps, hold_s=2.0, xfade_s=1.0, **dict(RENDER_KW, seed=0)))
            d_aw = aw_rms(drone_raw, 16000) + 1e-9
            drone_audio = (drone_raw * (AW_TARGET / d_aw)).astype(np.float64)
            
            notes = rt.generate_notes(v, a, duration_s=60.0)
            mixed = rt.mix_over_drone(drone_audio, notes, valence=v, arousal=a, level_db=3.0)
            sf.write(out_dir / fname, mixed, 16000)
            print(f"  -> Saved {fname}")

    print("=" * 70)
    print("Benchmark & Generation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Benchmark the transformer sizes on CPU.")
    ap.add_argument("--out-dir", dest="out_dir", default=None,
                    help="default ../data/benchmark")
    args = ap.parse_args()
    if not _ARRANGER:
        print("note: the arranger (Section 3.8) is absent, so this reports "
              "timings only and renders no audio.")
    run_benchmark_and_render(args.out_dir)
