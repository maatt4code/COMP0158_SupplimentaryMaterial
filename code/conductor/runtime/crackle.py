"""
crackle.py -- procedural vinyl crackle / surface-noise / record-scratch overlay.

A CONDUCTOR-LEVEL overlay (mixed on top of a finished segment, like apply_bed) --
deliberately NOT a synthesis voice, so it never touches the frozen arranger or
any rated stimulus. Two layers:
  * surface hiss -- continuous band-limited noise ("air" between the clicks)
  * crackle      -- sparse Poisson click impulses (randomised polarity + amplitude),
                    each a short decaying transient; plus rarer, louder "pops".
Deterministic given a seed. Caller sets the mix level (dB under the drone).

CLI:
  python crackle.py --selftest
  python crackle.py --demo out.wav --seconds 6 --density 12
"""

import argparse

import numpy as np

SR = 16000
_KERNEL_BANK = 8      # distinct tick shapes drawn per call, cycled at random


def _click_kernel(sr, decay_s=0.003, f=2500.0, rng=None):
    """A short decaying transient standing in for a dust tick.

    BROADBAND, not tonal (2026-08-01): this was a pure sine under an
    exponential decay, i.e. the synthesis recipe for a water drop.

    ACTUALLY FILTERED, and with an attack (2026-08-02, feedback: "crackle now
    seems to be creating some slapping noises"). Two faults in the first fix:
    `f` only mixed in a first difference, so f=700 did not make the pop DULL --
    it left raw white noise at a 3.9 kHz centroid, and 12 ms of full-band noise
    is a hand clap. And the envelope opened instantly at full scale, which is
    the "slap". Now `f` drives a real one-pole low-pass, so 700 Hz is a thump
    and 2500 Hz is a tick, and a ~0.3 ms raised-cosine attack takes the edge
    off without softening the transient enough to lose the click.
    """
    n = max(1, int(sr * decay_s))
    t = np.arange(n) / sr
    rng = np.random.default_rng(0) if rng is None else rng
    x = rng.normal(0, 1, n)
    # one-pole low-pass at f -- numpy only, and the kernels are tens of samples
    a = float(np.exp(-2.0 * np.pi * max(f, 1.0) / sr))
    y = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc = (1.0 - a) * x[i] + a * acc
        y[i] = acc
    env = np.exp(-t / (decay_s / 3.0))
    atk = max(2, int(sr * 0.0003))                  # ~0.3 ms
    if n > atk:
        env[:atk] *= 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, atk)))
    k = y * env
    return (k / (np.max(np.abs(k)) + 1e-9)).astype(np.float64)


def generate_crackle(n, sr=SR, density=8.0, intensity=0.5, hiss=0.15,
                     pop_rate=0.4, seed=0):
    """`n` samples of crackle.

    density   -- clicks per second (Poisson rate)
    intensity -- [0,1] scales click amplitude
    hiss      -- [0,1] continuous surface-noise level
    pop_rate  -- louder pops per second
    Returns float32, peak-safe (<= ~0.9)."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(n), dtype=np.float64)
    n = out.shape[0]
    if n == 0:
        return out.astype(np.float32)

    # surface hiss: white noise, +6 dB/oct tilt (first difference) so it reads airy
    if hiss > 0:
        noise = rng.normal(0, 1, n)
        noise = np.diff(noise, prepend=noise[0])
        out += hiss * 0.05 * noise

    # crackle clicks. A BANK of kernels, not one: the kernel is noise now, so
    # reusing a single draw would make every tick acoustically identical --
    # which reads as a machine, the same way one repeated pitch read as a drip.
    clicks = [_click_kernel(sr, rng=rng) for _ in range(_KERNEL_BANK)]
    kl = len(clicks[0])
    n_clicks = int(rng.poisson(max(0.0, density) * n / sr))
    for _ in range(n_clicks):
        pos = int(rng.integers(0, n))
        click = clicks[int(rng.integers(0, _KERNEL_BANK))]
        amp = intensity * (0.3 + 0.7 * rng.random()) * (1.0 if rng.random() < 0.5 else -1.0)
        end = min(n, pos + kl)
        out[pos:end] += amp * click[: end - pos]

    # rarer, louder pops (lower, longer transient)
    pops = [_click_kernel(sr, decay_s=0.006, f=400.0, rng=rng)
            for _ in range(_KERNEL_BANK)]
    pl = len(pops[0])
    n_pops = int(rng.poisson(max(0.0, pop_rate) * n / sr))
    for _ in range(n_pops):
        pos = int(rng.integers(0, n))
        pop = pops[int(rng.integers(0, _KERNEL_BANK))]
        amp = (0.5 + 0.5 * rng.random()) * (1.0 if rng.random() < 0.5 else -1.0)
        end = min(n, pos + pl)
        out[pos:end] += amp * pop[: end - pos]

    peak = np.max(np.abs(out)) + 1e-9
    if peak > 0.9:
        out *= 0.9 / peak
    return out.astype(np.float32)


def mix_under(drone, sr=SR, gain_db=-18.0, **kw):
    """Mix crackle under a drone segment at gain_db below the drone's RMS.
    Mirrors the bed-mixing convention in s02/s10 so it drops straight into the
    conductor. Returns float32 the same length as `drone`."""
    drone = np.asarray(drone, dtype=np.float64)
    crk = generate_crackle(len(drone), sr=sr, **kw).astype(np.float64)
    d_rms = np.sqrt(np.mean(drone ** 2)) + 1e-9
    c_rms = np.sqrt(np.mean(crk ** 2)) + 1e-9
    crk *= (d_rms * (10.0 ** (gain_db / 20.0))) / c_rms
    out = drone + crk
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 0.95:
        out *= 0.95 / peak
    return out.astype(np.float32)


def _selftest():
    sr = SR
    g = generate_crackle(sr * 2, sr=sr, density=10, seed=1)
    assert g.shape == (sr * 2,) and np.all(np.isfinite(g)), "shape/finite"
    assert np.max(np.abs(g)) <= 0.9 + 1e-6, "peak guard"

    # deterministic given the seed
    g2 = generate_crackle(sr * 2, sr=sr, density=10, seed=1)
    assert np.array_equal(g, g2), "not reproducible for a fixed seed"
    g3 = generate_crackle(sr * 2, sr=sr, density=10, seed=2)
    assert not np.array_equal(g, g3), "different seeds should differ"

    # density scales the amount of crackle energy
    lo = generate_crackle(sr * 4, sr=sr, density=2, hiss=0, pop_rate=0, seed=5)
    hi = generate_crackle(sr * 4, sr=sr, density=60, hiss=0, pop_rate=0, seed=5)
    assert np.sqrt((hi ** 2).mean()) > np.sqrt((lo ** 2).mean()), "density should add energy"

    # everything off -> silence
    z = generate_crackle(sr, sr=sr, density=0, hiss=0, pop_rate=0, seed=0)
    assert np.allclose(z, 0.0), "all-off should be silent"

    # mix_under keeps length + peak-safe and actually changes the drone
    drone = 0.2 * np.sin(2 * np.pi * 60 * np.arange(sr * 3) / sr)
    mixed = mix_under(drone, sr=sr, gain_db=-18.0, density=12, seed=3)
    assert mixed.shape == drone.shape and np.all(np.isfinite(mixed))
    assert np.max(np.abs(mixed)) <= 0.95 + 1e-6
    assert not np.allclose(mixed, drone.astype(np.float32)), "mix should add crackle"

    print("SELFTEST OK: crackle finite + peak-safe + reproducible; density scales "
          "energy; all-off is silent; mix_under is length-preserving and peak-safe.")


def main():
    ap = argparse.ArgumentParser(description="Procedural vinyl crackle overlay")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", metavar="OUT.wav", help="write a crackle demo wav")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--density", type=float, default=8.0)
    ap.add_argument("--intensity", type=float, default=0.5)
    ap.add_argument("--hiss", type=float, default=0.15)
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if args.demo:
        import soundfile as sf
        n = int(SR * args.seconds)
        crk = generate_crackle(n, density=args.density, intensity=args.intensity,
                               hiss=args.hiss, seed=0)
        sf.write(args.demo, crk, SR)
        print(f"Wrote {args.seconds:.0f}s crackle -> {args.demo} "
              f"(density {args.density}, intensity {args.intensity}, hiss {args.hiss})")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
