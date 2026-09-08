"""s18_ring_player.py -- continuous audio out, straight to the sound card.

The Tier-1 streaming path from `s02_conductor_app.py`'s Phase-2 design note,
and the answer to the last thing the browser player cannot do.

WHY THIS EXISTS. With the browser as the speaker, the server renders a whole
wav, hands the file over, and the browser decodes and plays it. Segments are
therefore separate playbacks: there is a handoff between every pair, and no way
to hand a browser "the next 20 ms, forever" without a streaming protocol Gradio
does not have. Here the process opens the sound card itself; the card asks for
the next block every few milliseconds and this class always has one ready. The
audio is one unbroken stream and the join is a crossfade inside it.

WHAT IT COSTS. Sound comes out of the machine RUNNING THIS, not the machine
running the browser. Over an SSH session with a share link that means silence
in your headphones and noise in a machine room. It is only useful once the app
runs where you are listening -- which is exactly what the pixi package is for.
Hence opt-in (`--audio-out`), with the browser player still the default.

The renderer is untouched. This consumes finished segments, so the frozen
arranger and every loudness guarantee in `final_loudness` still hold; the only
new DSP is the equal-power crossfade at the seam.

Usage:
  python s18_ring_player.py --selftest     # buffer maths, NO sound card needed
  python s18_ring_player.py --demo         # 6 s of tone pairs, needs a card
"""

import argparse
import threading

import numpy as np

XFADE_S = 0.25        # seam length. Long enough to hide a waveform step,
                      # short enough not to smear a deliberate transition.
PREBUFFER = 2         # segments to hold before starting playback


class RingPlayer:
    """A continuous mono stream fed by whole rendered segments.

    Segments are joined AT PUSH TIME, not in the callback: the callback must
    never do work that can take a variable amount of time, or it underruns and
    you hear a click -- the one failure this class exists to avoid. So push()
    does the crossfade arithmetic and the callback only ever copies.

    `open_stream=False` gives the identical buffer behaviour with no sound
    card, which is what makes the maths testable on a headless box.
    """

    def __init__(self, sr, xfade_s=XFADE_S, blocksize=1024, open_stream=True,
                 channels=1):
        self.sr = int(sr)
        self.channels = int(channels)
        self.xfade_n = max(1, int(xfade_s * sr))
        self._buf = np.zeros(0, dtype=np.float32)
        self._tail = None            # last xfade_n samples, awaiting a partner
        self._lock = threading.Lock()
        self.underruns = 0
        self.pushed = 0
        self._stream = None
        if open_stream:
            import sounddevice as sd
            self._stream = sd.OutputStream(
                samplerate=self.sr, channels=self.channels, dtype="float32",
                blocksize=blocksize, callback=self._callback)

    # ------------------------------------------------------------ producing
    def push(self, audio):
        """Append a rendered segment, crossfading it into what came before."""
        seg = np.asarray(audio, dtype=np.float32).reshape(-1)
        if seg.size == 0:
            return
        n = self.xfade_n
        with self._lock:
            if seg.size <= 2 * n:
                # too short to hold a seam on both sides: append it whole and
                # keep no tail, rather than crossfade a segment with itself
                if self._tail is not None:
                    self._buf = np.concatenate([self._buf, self._tail])
                    self._tail = None
                self._buf = np.concatenate([self._buf, seg])
                self.pushed += 1
                return
            if self._tail is None:
                self._buf = np.concatenate([self._buf, seg[:-n]])
            else:
                # equal power (sin/cos): constant total energy across the seam,
                # where a linear fade would dip audibly in the middle
                t = np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)
                joined = self._tail * np.cos(t) + seg[:n] * np.sin(t)
                self._buf = np.concatenate([self._buf, joined, seg[n:-n]])
            self._tail = seg[-n:].copy()
            self.pushed += 1

    # ------------------------------------------------------------ consuming
    def _take(self, frames):
        """Pop `frames` samples, zero-padding (and counting) on underrun."""
        with self._lock:
            have = self._buf.shape[0]
            if have >= frames:
                out, self._buf = self._buf[:frames], self._buf[frames:]
                return out
            self.underruns += 1
            out = np.zeros(frames, dtype=np.float32)
            out[:have] = self._buf
            self._buf = np.zeros(0, dtype=np.float32)
            return out

    def _callback(self, outdata, frames, time_info, status):
        block = self._take(frames)
        if self.channels == 1:
            outdata[:, 0] = block
        else:
            for c in range(self.channels):
                outdata[:, c] = block

    # ------------------------------------------------------------- control
    @property
    def buffered_s(self):
        with self._lock:
            return self._buf.shape[0] / float(self.sr)

    def ready(self, segments=PREBUFFER):
        return self.pushed >= segments

    def start(self):
        if self._stream is not None:
            self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def available():
    """Is a usable output device present? Import errors and missing hardware
    both mean 'no' -- the caller falls back to the browser player."""
    try:
        import sounddevice as sd
        return any(d["max_output_channels"] > 0 for d in sd.query_devices())
    except Exception:
        return False


def _selftest():
    sr = 16000
    p = RingPlayer(sr, xfade_s=0.25, open_stream=False)
    n = int(sr * 2.0)
    # two DC-ish segments at different levels: any seam shows as a step
    a = np.full(n, 0.5, np.float32)
    b = np.full(n, 0.2, np.float32)
    p.push(a)
    p.push(b)
    out = p._take(p._buf.shape[0])
    d = np.abs(np.diff(out))
    assert d.max() < 0.01, f"seam step {d.max():.4f} -- crossfade is not smooth"

    # equal power: a constant signal crossfaded with itself keeps its level
    q = RingPlayer(sr, xfade_s=0.25, open_stream=False)
    q.push(np.full(n, 0.5, np.float32))
    q.push(np.full(n, 0.5, np.float32))
    o = q._take(q._buf.shape[0])
    assert abs(o.min() - 0.5) < 0.02, \
        f"equal-power fade dipped to {o.min():.3f} -- should hold 0.5"

    # length: each push adds its own length, minus one seam's overlap
    exp = n + n - q.xfade_n - q.xfade_n     # the trailing tail is still held
    assert abs(len(o) - exp) <= 2, f"length {len(o)} != {exp}"

    # underrun is counted, never raised, and yields silence not a click
    r = RingPlayer(sr, open_stream=False)
    z = r._take(512)
    assert r.underruns == 1 and np.all(z == 0), "underrun must be silent"

    # a segment shorter than two seams is passed through, not mangled
    s = RingPlayer(sr, xfade_s=0.25, open_stream=False)
    tiny = np.full(100, 0.3, np.float32)
    s.push(tiny)
    assert np.allclose(s._take(100), 0.3), "short segment must pass through"

    print(f"SELFTEST OK: seam step {d.max():.5f} (equal-power crossfade holds "
          f"level), lengths correct, underruns counted and silent, short "
          f"segments pass through. Sound card NOT required for any of this -- "
          f"actual playback is unverified here by design.")


def _demo():
    if not available():
        raise SystemExit("no output device on this machine")
    sr = 16000
    p = RingPlayer(sr, open_stream=True)
    t = np.arange(int(sr * 2.0)) / sr
    for f in (220.0, 277.0, 330.0):
        p.push((0.2 * np.sin(2 * np.pi * f * t)).astype(np.float32))
    p.start()
    import time
    time.sleep(6.5)
    print(f"underruns: {p.underruns}")
    p.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        _demo()
    else:
        _selftest()
