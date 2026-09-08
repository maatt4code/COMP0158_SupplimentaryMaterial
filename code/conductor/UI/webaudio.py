"""s19_webaudio.py -- gapless playback IN THE BROWSER, via the WebAudio API.

The third playback path, and the only one that is both gapless AND plays on the
LISTENER's machine. Origin: 2026-08-01, adjudicated and adopted.

WHY THE OTHER TWO ARE NOT ENOUGH
  browser <audio>  -- what ships today. One finished file at a time, and the
      server can only hand over the next AFTER the current has drained, because
      setting a new source mid-playback truncates what is playing. That is a
      structural >=0.5 s hole at every join, on top of fetch and decode.
  sounddevice ring buffer (s18) -- truly gapless, but drives the sound card of
      the machine running PYTHON. Over SSH that is a room you are not in.

WebAudio removes both limits. `AudioBufferSourceNode.start(t)` schedules against
the AudioContext's own clock, so segment N+1 can be fetched, decoded and
scheduled WHILE N is still playing, to begin at the exact sample N ends. No
truncation, because scheduling is additive rather than a replacement. Audio
comes out of the browser, so a share link still works.

WHAT THE SERVER DOES: writes a segment, then pokes its URL into a hidden div.
That is a plain single-return Gradio update -- the ONLY delivery pattern the
long-track Space work found reliable through the relay (chained .then()
dispatches and generator streaming both lost data in real browsers). All the
scheduling lives in the page.

Sample rate: segments are 16 kHz; decodeAudioData resamples to the context rate
automatically, so no conversion is needed here.
"""

# The <script> must be passed to launch(head=...). Gradio 6 accepts `head` on
# the Blocks constructor and SILENTLY IGNORES it -- the trap that cost a day on
# the long-track Space. Every launch site must pass it explicitly.
WEBAUDIO_HEAD = """
<script>
(function () {
  const S = {
    ctx: null,          // AudioContext, created on the first user gesture
    next: 0,            // when the next segment should START, on ctx's clock
    seen: new Set(),    // urls already scheduled -- the observer can double-fire
    queued: 0,
    XF: 0.12            // seam fade, seconds. REVERTED from 0.30
                        // (2026-08-01: 0.30 was audible as a
                        // RELEASE on every segment -- a fade that
                        // long reads as a musical gesture, not a
                        // de-click). Drop to ~0.02 for a pure
                        // de-click if any release is still heard.
  };
  window.__conductor = S;

  // Browsers refuse to start an AudioContext without a user gesture. The
  // conductor's own Start button is that gesture; this also covers a stray
  // click anywhere, so the page can never end up mute with no way back.
  function ensureCtx() {
    if (!S.ctx) S.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (S.ctx.state === 'suspended') S.ctx.resume().then(diag, diag);
    return S.ctx;
  }
  document.addEventListener('click', ensureCtx, {capture: true});

  // SAY SO WHEN THE AUDIO IS NOT PLAYING. The status line is written by the
  // server, so it reports a segment as served whether or not the browser
  // could play it -- which is how a missing file and a suspended context both
  // read as "working" (2026-08-02, feedback: "no audio even tho the display
  // text says it's working"). This badge is client-side and reports what the
  // BROWSER actually did.
  // The readout's bottom line if the skin provides one (#conductor-statusbar,
  // rendered inside the LED box), otherwise a fixed corner chip. The brief wanted
  // it "like a permanent line on the bottom line like in tmux" -- a corner chip
  // floating over the page is easy to miss and easy to mistake for chrome.
  function badge() {
    const slot = document.getElementById('conductor-statusbar');
    if (slot) return slot;
    let b = document.getElementById('conductor-diag');
    if (!b) {
      b = document.createElement('div');
      b.id = 'conductor-diag';
      b.style.cssText = 'position:fixed;right:10px;bottom:10px;z-index:9999;' +
        'font:12px ui-monospace,monospace;padding:5px 9px;border-radius:5px;' +
        'pointer-events:none;opacity:.9';
      document.body.appendChild(b);
    }
    return b;
  }

  // ---- spectrum ---------------------------------------------------------
  // An AnalyserNode is a PASSIVE TAP: it reads the FFT the audio path already
  // produces and passes samples through untouched. It cannot slow fetching or
  // decoding, because it is not in either path -- it sits on the graph, and
  // the graph only runs while audio plays. Drawing is one canvas on
  // requestAnimationFrame, which the browser pauses when the tab is hidden.
  function analyser(ctx) {
    if (!S.an) {
      S.an = ctx.createAnalyser();
      // 2048 -> 1024 bins at ~23 Hz each (was 128 -> 64 bins of 375 Hz).
      // At 375 Hz a bin, 30-80 Hz and 80-250 Hz both fell inside bin 0, so
      // sub-bass could not be told from bass at all and the visuals had no
      // usable low-end detail -- see VISUALS.md 7.8. The FFT itself is cheap;
      // what made this a coupled change is that drawEq below reads the SAME
      // tap and had to gain log binning in the same edit, or the deck's EQ
      // would show only 0-750 Hz and look permanently bass-heavy.
      S.an.fftSize = 2048;
      S.an.smoothingTimeConstant = 0.75;
      S.an.connect(ctx.destination);
      // PUBLISHED for the visuals tab (s22). `S` is closure-scoped, so without
      // this the only way for another script to reach the FFT would be to
      // build a SECOND AnalyserNode on the same graph -- doubling the work and
      // giving two readings of the same audio that could disagree. One tap,
      // read by whoever wants it. Absent by design on the <audio> and ring
      // paths, where no graph exists; s22 falls back to params-only rather
      // than pretending the visual is audio-reactive.
      window.CONDUCTOR_AN = S.an;
    }
    return S.an;
  }
  function drawEq() {
    const c = document.getElementById('conductor-eq');
    if (!c || !S.an) { requestAnimationFrame(drawEq); return; }
    const g = c.getContext('2d');
    const w = c.width = c.clientWidth, h = c.height = c.clientHeight;
    const bins = S.an.frequencyBinCount;
    if (!S.buf || S.buf.length !== bins) S.buf = new Uint8Array(bins);
    S.an.getByteFrequencyData(S.buf);
    g.clearRect(0, 0, w, h);
    // LOG BINNING. `Math.min(bins,32)` took the first 32 bins linearly, which
    // at 1024 bins is 0-750 Hz -- three quarters of the display spent on the
    // drone's fundamental. Pitch is logarithmic, so the bars are too: 32
    // bands geometrically spaced over 40 Hz - 8 kHz, each the MAX of its
    // bins (max, not mean, so a narrow partial still lights its bar).
    const n = 32, bw = w / n, cell = Math.max(2, Math.floor(h / 8));
    const sr = (S.an.context && S.an.context.sampleRate) || 48000;
    const binHz = (sr * 0.5) / bins, F0 = 40, F1 = 8000;
    for (let i = 0; i < n; i++) {
      // quantise to whole cells so it reads as pixels rather than a smooth
      // curve -- matches the deck's look and hides FFT jitter
      const flo = F0 * Math.pow(F1 / F0, i / n),
            fhi = F0 * Math.pow(F1 / F0, (i + 1) / n);
      const lo = Math.max(0, Math.floor(flo / binHz)),
            hi = Math.min(bins, Math.max(lo + 1, Math.ceil(fhi / binHz)));
      let v = 0;
      for (let k2 = lo; k2 < hi; k2++) if (S.buf[k2] > v) v = S.buf[k2];
      const lit = Math.round((v / 255) * (h / cell));
      for (let k = 0; k < lit; k++) {
        const y = h - (k + 1) * cell;
        g.fillStyle = k > h / cell - 3 ? '#f59e0b' : (k > h / cell - 6 ? '#22c55e' : '#166534');
        g.fillRect(i * bw + 1, y + 1, bw - 2, cell - 2);
      }
    }
    requestAnimationFrame(drawEq);
  }
  requestAnimationFrame(drawEq);
  function say(msg, level) {
    const b = badge();
    b.textContent = msg;
    b.style.background = level === 'bad' ? '#dc2626'
                       : level === 'warn' ? '#f59e0b' : '#15181d';
    b.style.color = level ? '#08131f' : '#7c8390';
  }
  function diag() {
    const st = S.ctx ? S.ctx.state : 'none';
    if (st === 'suspended') { say('AUDIO BLOCKED - click anywhere', 'warn'); return; }
    if (S.failed) { say('SEGMENT MISSING x' + S.failed + ' - see console', 'bad'); return; }
    // LEAD is the diagnosis for a gap between clips: seconds of audio already
    // scheduled beyond now. Healthy and steady = delivery is keeping up and a
    // gap is a decode/seam problem. Shrinking towards 0 segment by segment =
    // the RENDERER is losing ground and no amount of scheduling will help;
    // lengthen the holds instead. Late arrivals are counted too, because one
    // late segment is a gap even if the next twenty are early.
    const lead = S.ctx ? Math.max(0, S.next - S.ctx.currentTime) : 0;
    say('audio ' + st + ' - lead ' + lead.toFixed(1) + 's'
        + (S.late ? ' - ' + S.late + ' late' : ''),
        lead < 2 ? 'warn' : null);
  }

  async function schedule(url) {
    if (!url || S.seen.has(url)) return;
    S.seen.add(url);
    const ctx = ensureCtx();
    let buf;
    try {
      const r = await fetch(url);
      // A 404 here is the segment file having been deleted between the server
      // handing over this url and the browser asking for it -- fetch() does
      // NOT throw on an http error, so without this check a missing segment
      // fails inside decodeAudioData with a far less obvious message.
      if (!r.ok) throw new Error('HTTP ' + r.status);
      buf = await ctx.decodeAudioData(await r.arrayBuffer());
    } catch (e) {
      S.failed = (S.failed || 0) + 1;
      S.seen.delete(url);                  // allow a retry if it reappears
      console.warn('[conductor] fetch/decode failed', url, e);
      diag();
      return;
    }

    const src = ctx.createBufferSource();
    src.buffer = buf;
    // Equal-power fade at each edge. The segments already meet in LEVEL
    // (final_loudness matches the junction server-side); this only hides the
    // waveform step, which no amount of levelling can remove.
    const g = ctx.createGain();
    src.connect(g).connect(analyser(ctx));

    // If we have fallen behind (a slow fetch), start slightly in the future
    // rather than in the past -- start(t) with t < currentTime plays instantly
    // and the overlap turns into a stutter.
    const now = ctx.currentTime;
    let t = S.next;
    if (t < now + 0.02) {
      // Arrived too late to butt against the previous segment: whatever gap
      // that leaves is now audible, so record it rather than hide it.
      if (S.next > 0) {
        S.late = (S.late || 0) + 1;
        S.lastGap = now - S.next;
        console.warn('[conductor] LATE by ' + (now - S.next).toFixed(2) + 's'
                     + ' -- audible gap; renderer or delivery is behind');
      }
      t = now + 0.02;
    }

    // EQUAL POWER, not exponential. Two exponential ramps crossing do not sum
    // to constant power -- they dip in the middle of the overlap, which is
    // heard as a release on every seam. sin/cos curves do sum to unity.
    const d = buf.duration, N = 32;
    const up = new Float32Array(N), dn = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const x = (i / (N - 1)) * (Math.PI / 2);
      up[i] = Math.sin(x); dn[i] = Math.cos(x);
    }
    g.gain.setValueAtTime(0.0, t);
    g.gain.setValueCurveAtTime(up, t, S.XF);
    g.gain.setValueAtTime(1.0, t + S.XF);
    g.gain.setValueCurveAtTime(dn, t + d - S.XF, S.XF);

    src.start(t);
    // overlap by the crossfade so the tail of one sits under the head of the
    // next, instead of leaving a XF-long dip between them
    S.next = t + d - S.XF;
    S.queued += 1;
    console.log('[conductor] scheduled', S.queued, 'at', t.toFixed(2),
                'dur', d.toFixed(1), 'lead', (t - now).toFixed(2) + 's',
                'ctx', ctx.state);
    diag();
  }

  // The server pokes a url into #conductor-sink's data-seg. A MutationObserver is
  // used rather than an event handler because the value arrives as ordinary
  // HTML in a normal component update -- no extra dispatch to lose.
  //
  // RESOLVE THE SINK ON EVERY READ, and observe the DOCUMENT (2026-08-02).
  // The first version captured the element once and observed that node. A
  // gradio HTML update does not edit the node, it REPLACES it -- so the
  // observer was left watching a detached div and `el.querySelector` kept
  // searching the dead one. It went deaf after the first re-render and the
  // page fell silent while the server happily went on rendering segments.
  // s21 already re-attaches on every repaint for exactly this reason; this
  // file never did. The interval is the belt to that braces: `schedule`
  // dedupes on the url, so re-reading costs nothing.
  function watch() {
    const read = () => {
      const el = document.getElementById('conductor-sink');
      const n = el && el.querySelector('[data-seg]');
      if (n) schedule(n.getAttribute('data-seg'));
    };
    new MutationObserver(read).observe(document.documentElement,
                                       {childList: true, subtree: true,
                                        attributes: true});
    setInterval(read, 400);
    read();
  }
  watch();
})();
</script>
"""


def sink_html(url=None, note=""):
    """The div the page watches. `url` is served by Gradio's own file route."""
    seg = f"<span data-seg='{url}'></span>" if url else ""
    return (f"<div id='conductor-sink' style='font-size:.8em;color:#666'>{seg}"
            f"{note}</div>")


def file_url(path):
    """Gradio's static file route for an allowed path."""
    return f"/gradio_api/file={path}"


def _selftest():
    a = sink_html("/gradio_api/file=/x/seg_1.mp3")
    assert "id='conductor-sink'" in a and "data-seg=" in a and "seg_1.mp3" in a
    b = sink_html(None, "waiting")
    assert "data-seg" not in b and "waiting" in b
    assert "decodeAudioData" in WEBAUDIO_HEAD and "src.start(t)" in WEBAUDIO_HEAD
    # the two failure modes worth pinning: scheduling in the past, and
    # double-scheduling the same segment
    assert "now + 0.02" in WEBAUDIO_HEAD, "must never start(t) in the past"
    assert "S.seen.has(url)" in WEBAUDIO_HEAD, "must dedupe repeated urls"
    assert file_url("/a/b.mp3") == "/gradio_api/file=/a/b.mp3"
    print("SELFTEST OK: sink markup, url route, dedupe guard and "
          "no-scheduling-in-the-past guard all present. Actual playback is "
          "browser-side and NOT verified here.")


if __name__ == "__main__":
    _selftest()
