"""s21_faders.py -- real vertical faders, drawn rather than rotated.

2026-08-02, feedback: *"you are just using those sliders and you flipped them 90
degrees, they are really short and the scroll bars make them impossible to
use."* Correct. `transform: rotate(-90deg)` keeps the element's ORIGINAL
box for layout and hit-testing, so a rotated slider has the throw of its old
width, a hit area in the wrong place, and it provokes scrollbars in the
column it sits in. It looks like a fader in a screenshot and is unusable
under a finger.

WHAT THIS DOES INSTEAD. Each fader is a purpose-drawn div: a slot, a cap, and
a value read-out, dragged with pointer events. It writes its value into a
HIDDEN Gradio slider by dispatching a native `input` event on that slider's
own `<input type=range>`, which is exactly what Gradio's frontend listens for.
So the fader is cosmetic and the Gradio component is still the state -- every
handler, every `.change`, every input list stays untouched.

WHAT GRADIO'S SLIDER ACTUALLY LISTENS FOR (read out of the bundled frontend,
gradio 6.19 `Index-Cw5cUAyv.js`, rather than assumed -- the same method that
finally cracked the s04 playhead bug):

    i("input", s, M)        // range input -> dispatch("input")
    i("pointerup", s, G)    // range input -> dispatch("release", value)
    E(() => value != S && (S = value, C()))   // ANY change -> dispatch("change")

Three consequences drive the code below:

1. `.change` fires on EVERY value change, so it fires on every pointer-move of
   a drag. Never wire a server handler to `.change` on a fader-driven slider --
   through the share relay that is a request per move. `.release` is the safe
   one, and it only fires on a native `pointerup` ON THE RANGE INPUT -- which a
   drawn fader never produces -- so the fader dispatches one synthetically when
   the drag ends. One event per gesture, matching native semantics.
2. Gradio assigns `m.value = ...` directly when the SERVER updates a slider,
   and an assignment fires no `input` event. A fader listening only for `input`
   therefore goes stale whenever a handler moves its component -- e.g. clicking
   the VA pad, which writes both target sliders. Hence `sync()`: cheap polling
   that repaints from the component whenever something else changed it.
3. The pulse/flash on a submit button is pure AFFORDANCE, not state, so it is
   done entirely client-side. Routing it through the server would buy nothing
   and would put a round-trip on a pointer-move.

WHY NOT A DIFFERENT FRAMEWORK. NiceGUI's Quasar sliders are genuinely vertical
and would look right out of the box, but the port is the entire event model:
gr.State sessions, the timer loop, the WebAudio sink, ~12 wirings. This app
ALREADY drives custom JS successfully -- s19_webaudio schedules playback from a
head script, through the share relay, which is the delivery path that has
failed here before. Reusing that proven route is a stylesheet-and-a-script;
changing framework is a rewrite with five weeks to submission.

Pointer events (not mouse) so it works on a trackpad, a touchscreen and a
mixer-shaped finger alike.
"""

FADER_HEAD = """
<style>
.jfx {display:flex; flex-direction:column; align-items:center; gap:6px;
      user-select:none; padding:4px 2px;}
.jfx .cap {font-size:.62em; letter-spacing:.12em; text-transform:uppercase;
           color:#7c8390;}
.jfx .val {font-size:.72em; color:#cbd2da; font-family:ui-monospace,monospace;}
.jfx .slot {position:relative; width:34px; height:var(--jfx-h,170px);
            background:linear-gradient(#0a0c0f,#14171c);
            border:1px solid #2b2f37; border-radius:6px; cursor:ns-resize;
            box-shadow:inset 0 0 8px #000; touch-action:none;}
.jfx .slot::before {content:""; position:absolute; left:50%; top:8px;
            bottom:8px; width:3px; margin-left:-1.5px; border-radius:2px;
            background:#05070a; box-shadow:inset 0 0 3px #000;}
.jfx .fill {position:absolute; left:50%; width:3px; margin-left:-1.5px;
            bottom:8px; border-radius:2px; background:var(--jfx-c,#22c55e);
            box-shadow:0 0 8px var(--jfx-c,#22c55e);}
.jfx .knob {position:absolute; left:2px; width:28px; height:16px;
            margin-bottom:-8px; border-radius:3px;
            background:linear-gradient(#4a515b,#22262c);
            border:1px solid #11141a; box-shadow:0 2px 4px #000;}
.jfx .knob::after {content:""; position:absolute; left:4px; right:4px; top:7px;
            height:2px; background:var(--jfx-c,#22c55e); border-radius:1px;
            box-shadow:0 0 6px var(--jfx-c,#22c55e);}

/* ---- submit affordance -------------------------------------------------
   A fader whose value has moved ARMS its submit button: it pulses until the
   value is committed, then lights and fades out. Entirely client-side -- see
   the module docstring, point 3. */
@keyframes jarmpulse {
  0%, 100% {box-shadow:0 0 0 rgba(245,158,11,0);      border-color:#3a3f47;}
  50%      {box-shadow:0 0 16px rgba(245,158,11,.85); border-color:#f59e0b;}
}
/* lit for 0.5s, then fades out over the remaining 0.75s (the brief's timing) */
@keyframes jflashfade {
  0%   {background:#f59e0b; color:#08131f; box-shadow:0 0 18px rgba(245,158,11,.9);}
  40%  {background:#f59e0b; color:#08131f; box-shadow:0 0 18px rgba(245,158,11,.9);}
  100% {background:#15181d; color:#8b919b; box-shadow:0 0 0 rgba(245,158,11,0);}
}
button.jarmed {animation: jarmpulse 1.15s ease-in-out infinite;
               color:#f0b45e !important;}
button.jflash {animation: jflashfade 1.25s ease-out 1 forwards;}
/* NOTHING STAGED, NOTHING TO COMMIT. The pulse says "there is a change
   waiting"; when there is none the button must not merely stop pulsing, it
   must stop WORKING -- with the layer off you could still press APPLY, send
   the params and watch the readout report a change that had not happened
   (2026-08-02). Armed state and clickability are now the same fact.
   `:not(.jflash)` keeps the light-then-fade fully lit on the way out. */
button.jarm:not(.jarmed):not(.jflash) {pointer-events:none !important;
        opacity:.4 !important;}
/* STOP announces itself, quietly. A running deck should show that stopping is
   an option without nagging (2026-08-02) -- so the same slow rate as
   NEXT, in red, and never at the same colour as an APPLY waiting on you. */
@keyframes jstoppulse {
  0%, 44%, 56%, 100% {box-shadow:0 0 0 rgba(239,68,68,0);   border-color:#3a3f47;}
  50%                {box-shadow:0 0 15px rgba(239,68,68,.85); border-color:#ef4444;}
}
button.jstop {animation: jstoppulse 6s ease-in-out infinite;
              color:#f08a8a !important;}
</style>
<script>
(function () {
  // Find the real <input type=range> Gradio rendered for a component id.
  function rangeOf(id) {
    const host = document.getElementById(id);
    return host ? host.querySelector('input[type=range]') : null;
  }
  // Gradio listens for a native `input` event on its own control, so writing
  // the value and dispatching one is a first-class update -- no private API.
  function push(inp, v) {
    const setter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype, 'value').set;
    setter.call(inp, String(v));
    inp.dispatchEvent(new Event('input', {bubbles: true}));
    dotSync();     // during a DRAG the 250ms poll is too coarse to look live
  }
  // End of a gesture. Gradio binds `release` to a native pointerup ON THE
  // RANGE INPUT, which a drawn fader never produces -- so synthesise one.
  // Exactly one per drag, unlike `change`, which fires on every move.
  function commit(inp) {
    inp.dispatchEvent(new Event('pointerup', {bubbles: true}));
  }
  function armBtn(id) {
    const btn = id && document.getElementById(id);
    if (!btn) return;
    btn.classList.remove('jflash');
    btn.classList.add('jarmed');
  }
  // Only the pulse comes off -- never `jflash`, which is a one-shot animation
  // reporting a commit that really happened.
  function disarmBtn(id) {
    const btn = id && document.getElementById(id);
    if (btn) btn.classList.remove('jarmed');
  }
  // Arm the submit button this fader belongs to (if any).
  function arm(el) { armBtn(el.dataset.arm); }

  // The red TARGET dot must follow the two VA faders while they move, not just
  // a pad click. It is written INLINE, which beats the server's <style>
  // carrier -- and it is always computed FROM the sliders, which is exactly
  // what a pad click and SET TARGET write into as well, so the three routes
  // can never disagree. Same linear map as skins.dot_style; y flips because
  // rows run down and arousal runs up. The green CURRENT dot is untouched:
  // that one is a measurement of the walk, not an intention.
  function dotSync() {
    const dot = document.getElementById('va_dot_tgt');
    if (!dot) return;
    const vi = rangeOf('sl_target_v'), ai = rangeOf('sl_target_a');
    if (!vi || !ai) return;
    const v = parseFloat(vi.value), a = parseFloat(ai.value);
    if (!isFinite(v) || !isFinite(a)) return;
    const x = ((v + 1) * 50).toFixed(2), y = ((1 - a) * 50).toFixed(2);
    const at = x + ',' + y;
    if (dot.dataset.jfxAt === at) return;    // idempotent: called every 250ms
    dot.dataset.jfxAt = at;
    dot.style.left = x + '%';
    dot.style.top = y + '%';
  }

  // Everything a strip's APPLY should react to, as a single string. Watching
  // VALUES rather than events is what makes DROPDOWNS work: a gradio Dropdown
  // is not a <select>, so picking an option fires no input/change inside the
  // strip -- and its listbox may be portaled outside the strip entirely, so
  // even a document-level listener would not see it as belonging here. Only
  // the faders worked, because they dispatch a native input event themselves.
  // `.jnoarm` controls are excluded: they belong to a different button.
  function stripValues(strip) {
    const out = [];
    // THE LAMP'S LABEL IS THE ON/OFF STATE, and it must be in the signature.
    // MEASURED in a real browser (2026-08-02): a gradio Checkbox with
    // visible=False is NOT RENDERED AT ALL -- the crackle strip contains four
    // inputs, all belonging to its two faders, and no checkbox. So the hidden
    // state component this panel's on/off buttons drive is invisible here, and
    // toggling a layer changed nothing in the signature: APPLY stayed dark
    // until you also moved a fader (2026-08-02). `_lit` keeps the
    // label as the state ("ON" / "TURN ON" since 2026-08-07 -- s20_skins
    // BTN_ON/BTN_OFF), which is exactly what to read. Only the INEQUALITY of
    // the two strings matters here, so the wording is free to change.
    // ...AND FOR A PICK GROUP THE LABEL IS CONSTANT. `_pick`'s two buttons say
    // LEARNED and BASELINE forever; what moves is WHICH ONE IS LIT, and the
    // lamp arrives as a <style> carrier naming that button's id -- there is no
    // class or attribute on the button itself to read. Its hidden Radio is
    // `visible=False`, so (same measurement as the checkbox above) it is not in
    // the DOM either. Signature was therefore constant and the TRANSITION strip
    // could never arm its APPLY (2026-08-03). Read the carriers.
    const lamps = Array.prototype.map.call(
        strip.querySelectorAll('.jstyle'), s => s.textContent || '').join('');
    strip.querySelectorAll('.jlit').forEach(b => {
      if (b.closest('.jnoarm')) return;
      // '#id{' , not a bare id: 'btn_bed' is a substring of 'btn_bed_apply'.
      const on = b.id && lamps.indexOf('#' + b.id + '{') >= 0;
      out.push((b.textContent || '').trim() + (on ? '*' : ''));
    });
    strip.querySelectorAll('input, textarea').forEach(el => {
      if (el.closest('.jnoarm')) return;
      // A RADIO's `value` is a constant attribute -- reading it would make
      // every option look identical and the TRANSITION strip would never
      // notice a change. What varies is which one is checked.
      out.push(el.type === 'checkbox' || el.type === 'radio'
               ? (el.checked ? '1' : '0') : el.value);
    });
    return out.join('\\u0001');
  }

  // A STRIP arms its own APPLY. Convention: a container `strip_<key>` owns the
  // button `btn_<key>_apply` (see s20_skins._apply). Delegated in the CAPTURE
  // phase so it still fires if a component stops propagation, and covering
  // `input`+`change` so dropdowns, checkboxes and faders all count. Clicking
  // the strip's own on/off button arms too -- toggling STAGES the layer for
  // this skin, it does not reach the conductor until APPLY.
  function attachStrip(strip) {
    if (strip.dataset.jfxStrip) return;
    const key = strip.id.slice(6);              // strip_<key>
    const applyId = 'btn_' + key + '_apply';
    if (!document.getElementById(applyId)) return;   // not painted yet; retry
    strip.dataset.jfxStrip = '1';
    // `.jnoarm` opts a control out: something that lives in the strip for
    // layout reasons but is read by a DIFFERENT button (the valence axis sits
    // in the Mood panel but is consumed by START). Arming here would promise a
    // commit that does not apply to it.
    const armIt = e => {
      const t = e && e.target;
      if (t && t.closest && t.closest('.jnoarm')) return;
      syncStripArm(strip);
    };
    strip.addEventListener('input', armIt, true);
    strip.addEventListener('change', armIt, true);
    strip.dataset.jfxCommitted = stripValues(strip);  // baseline: never arm on load
    strip.dataset.jfxSettle = String(Date.now() + 2000);
    strip.addEventListener('click', e => {
      // never re-arm from the APPLY click itself -- that click DISarms.
      // Toggling a layer arms OPTIMISTICALLY: the label only changes when the
      // server round-trip lands, and waiting ~300ms for the light to come on
      // reads as a dead button. The next poll compares properly and puts it
      // back out if the toggle happened to return to the committed state.
      if (e.target.closest('button.jlit')) armBtn(applyId);
    }, true);
  }
  // ARMING IS A COMPARISON, NOT AN EDGE. It used to be "something changed ->
  // pulse", which cannot be undone: pick a dropdown value, change your mind,
  // switch the layer back off, and APPLY kept pulsing over a dead strip with
  // nothing left to commit (2026-08-02). So the strip remembers what
  // was last COMMITTED and pulses only while it differs from that -- put every
  // control back, including the on/off button, and the prompt goes away by
  // itself. The on/off state is in the signature because it is a hidden
  // checkbox inside the strip.
  function syncStripArm(strip) {
    const applyId = 'btn_' + strip.id.slice(6) + '_apply';
    // SETTLE WINDOW. Gradio fills a dropdown's input AFTER the block first
    // paints, and a load-time `.change` can swap components around, so the
    // first moments of a strip's life look like edits that never happened --
    // which is why MELODY and REVERB flashed APPLY before anything was
    // touched. Re-baseline quietly instead of arming.
    if (Date.now() < Number(strip.dataset.jfxSettle || 0)) {
      strip.dataset.jfxCommitted = stripValues(strip);
      return;
    }
    if (stripValues(strip) === strip.dataset.jfxCommitted) disarmBtn(applyId);
    else armBtn(applyId);
  }

  // A submit button: clicking it disarms and plays the light-then-fade.
  function attachArmable(btn) {
    if (btn.dataset.jfxArmable) return;
    btn.dataset.jfxArmable = '1';
    btn.addEventListener('click', () => {
      // THIS is what "committed" means: whatever the strip reads right now is
      // the state the conductor is being given, so it becomes the baseline
      // every later comparison is made against.
      const strip = btn.closest('[id^="strip_"]');
      if (strip) {
        strip.dataset.jfxCommitted = stripValues(strip);
        // A HANDLER THAT WRITES BACK INTO ITS OWN STRIP would otherwise re-arm
        // the button it just disarmed. LOG is the case: s02's log_reaction
        // returns "" for note_box, so the note clears a round-trip AFTER this
        // click and the next poll reads it as a fresh edit -- LOG would light
        // up again having just been pressed, with nothing staged.
        // Same settle window the strip gets at load, and for the same reason:
        // a write the USER did not make must re-baseline, not arm.
        // Deliberately short. It is a blunt instrument -- an edit made inside
        // the window re-baselines too, so typing again within ~1s of pressing
        // LOG will not light it until the next keystroke after that. Widening
        // this to be safe would swallow real edits; a round-trip on localhost
        // is well under 300ms.
        strip.dataset.jfxSettle = String(Date.now() + 900);
      }
      btn.classList.remove('jarmed');
      btn.classList.remove('jflash');
      void btn.offsetWidth;             // restart the animation if re-clicked
      btn.classList.add('jflash');
    });
    btn.addEventListener('animationend', e => {
      if (e.animationName === 'jflashfade') btn.classList.remove('jflash');
    });
  }
  function attach(el) {
    if (el.dataset.jfxReady) return;
    const target = el.dataset.target;
    const inp = rangeOf(target);
    if (!inp) return;                       // gradio not painted yet; retry
    el.dataset.jfxReady = '1';
    const lo = parseFloat(inp.min), hi = parseFloat(inp.max);
    const step = parseFloat(inp.step) || 0.01;
    const slot = el.querySelector('.slot');
    const fill = el.querySelector('.fill');
    const knob = el.querySelector('.knob');
    const val = el.querySelector('.val');

    function paint(v) {
      const f = (v - lo) / (hi - lo || 1);
      const h = slot.clientHeight - 16;
      fill.style.height = (f * h) + 'px';
      knob.style.bottom = (8 + f * h) + 'px';
      val.textContent = (Math.abs(step) < 1) ? v.toFixed(2) : String(Math.round(v));
      el._jfxShown = v;
    }
    function fromY(clientY) {
      const r = slot.getBoundingClientRect();
      let f = 1 - (clientY - r.top - 8) / (r.height - 16);
      f = Math.max(0, Math.min(1, f));
      let v = lo + f * (hi - lo);
      v = Math.round(v / step) * step;
      return Math.max(lo, Math.min(hi, v));
    }
    let dragging = false;
    function move(e) {
      if (!dragging) return;
      const v = fromY(e.clientY);
      paint(v); push(inp, v);
      e.preventDefault();
    }
    function end() {
      if (!dragging) return;
      dragging = false;
      commit(inp);                          // one `release`, not one per move
      arm(el);                              // moved but not submitted yet
    }
    slot.addEventListener('pointerdown', e => {
      dragging = true; slot.setPointerCapture(e.pointerId); move(e);
    });
    slot.addEventListener('pointermove', move);
    slot.addEventListener('pointerup', end);
    slot.addEventListener('pointercancel', end);
    // follow the component if something else changes it (a preset, a reset)
    inp.addEventListener('input', () => {
      if (!dragging) paint(parseFloat(inp.value));
    });
    el._jfxInput = inp;
    el._jfxPaint = paint;
    el._jfxDragging = () => dragging;
    paint(parseFloat(inp.value));
  }
  // A SERVER-side update assigns .value directly and fires no event, so the
  // drawn fader would silently go stale -- e.g. clicking the VA pad, which
  // writes both target sliders. Cheap poll, no network, idempotent.
  function sync() {
    document.querySelectorAll('.jfx').forEach(el => {
      const inp = el._jfxInput;
      if (!inp || !el._jfxPaint || el._jfxDragging()) return;
      const v = parseFloat(inp.value);
      if (!isFinite(v) || v === el._jfxShown) return;
      el._jfxPaint(v);
      arm(el);           // something moved it; it still needs committing
    });
  }
  // ---- transport lights --------------------------------------------------
  // START pulses while stopped ("ready to be clicked"); NEXT pulses, more
  // slowly, only while running. Both are derived from START'S OWN LABEL, which
  // the server sets -- so the lights cannot disagree with the session. NEXT's
  // click lights it for 0.5s, fades out, and stays dark for 5s before
  // inviting again.
  const T = {until: 0};
  function attachNext(n) {
    if (n.dataset.jfxNext) return;
    n.dataset.jfxNext = '1';
    n.addEventListener('click', () => {
      n.classList.remove('jslow', 'jflash');
      void n.offsetWidth;                       // restart if clicked again
      n.classList.add('jflash');
      T.until = Date.now() + 5000;
    });
    n.addEventListener('animationend', e => {
      if (e.animationName === 'jflashfade') n.classList.remove('jflash');
    });
  }
  // START opens a session that knows NOTHING about the strips: the server
  // builds a fresh state with every layer off. So a layer the panel shows as
  // ON has not been committed to this session, however committed it was to the
  // last one -- re-arm exactly those and leave the rest quiet.
  function attachStart(s) {
    if (s.dataset.jfxStart) return;
    s.dataset.jfxStart = '1';
    s.addEventListener('click', () => {
      document.querySelectorAll('[id^="strip_"]').forEach(strip => {
        if (!strip.dataset.jfxStrip) return;
        const lamp = strip.querySelector('.jlit');
        // PINNED TO s20_skins.BTN_ON, and it must stay an equality test on the
        // ON label specifically. `.jlit` is also worn by _pick buttons
        // (LEARNED / BASELINE), which have no on/off meaning -- reading the
        // <style> lamp carrier instead would see the policy strip's
        // always-emitted default lamp and re-arm it on every START. The OFF
        // label is free to say anything; this one is not.
        if (lamp && (lamp.textContent || '').trim() === 'ON')
          strip.dataset.jfxCommitted = '\u0000new-session';
        else strip.dataset.jfxCommitted = stripValues(strip);
      });
    });
  }
  function transport() {
    const s = document.getElementById('btn_start');
    const n = document.getElementById('btn_next');
    if (!s || !n) return;
    attachNext(n);
    attachStart(s);
    const running = (s.textContent || '').trim().toUpperCase().indexOf('STOP') === 0;
    // Stopped: amber, "press me". Running: slow red, "you can stop". Never
    // both, or the one button would carry two grammars at once.
    s.classList.toggle('jarmed', !running);
    s.classList.toggle('jstop', running);
    if (!running) {                 // stopped: NEXT invites nothing
      n.classList.remove('jslow');
      T.until = 0;
    } else if (!n.classList.contains('jflash') && Date.now() >= T.until) {
      n.classList.add('jslow');
    }
  }

  // Catch every value change a strip cares about, however it was made --
  // dropdown pick, checkbox, fader drag, or a server-side update.
  // Catch every value change a strip cares about, however it was made --
  // dropdown pick, checkbox, fader drag, or a server-side update. Dropdowns
  // are why this poll exists at all: a gradio Dropdown is not a <select>, so
  // picking an option fires no event inside the strip.
  function pollStrips() {
    document.querySelectorAll('[id^="strip_"]').forEach(strip => {
      if (strip.dataset.jfxStrip) syncStripArm(strip);
    });
  }

  // START ADOPTS THE VISIBLE STATE (s02's new_session -> adopt_layers,
  // 2026-08-04), so at the instant START is pressed every strip IS committed.
  // Re-baseline them all instead of letting them re-arm.
  //
  // Without this the strips pulse again immediately after START, which is what
  // made the bug look purely cosmetic: gradio re-fills dropdown inputs on the
  // response, and a re-fill reads here as an edit. It is the SAME late-fill
  // that made MELODY and REVERB flash at load -- exactly the two strips with
  // dropdowns -- which is why attachStrip opens a settle window there. START
  // needs its own, because the load window is long expired by then, and a
  // wider one, because START does real work first (the human boundary guard is
  // fitted lazily on first use, ~2.3s).
  //
  // This is honest only BECAUSE the server now adopts these values. If START
  // ever stops reading them, delete this with it -- a strip that reports
  // "committed" over settings the conductor never received is the lie this
  // whole round was about.
  function rebaselineStrips(ms) {
    document.querySelectorAll('[id^="strip_"]').forEach(strip => {
      strip.dataset.jfxCommitted = stripValues(strip);
      strip.dataset.jfxSettle = String(Date.now() + (ms || 3000));
      disarmBtn('btn_' + strip.id.slice(6) + '_apply');
    });
  }
  function attachStart() {
    const s = document.getElementById('btn_start');
    if (!s || s.dataset.jfxStart) return;
    s.dataset.jfxStart = '1';
    // Capture phase, same reason as the strips: gradio components stop
    // propagation. STOP shares this button and re-baselining there is
    // harmless -- the next START re-reads the controls anyway.
    s.addEventListener('click', () => rebaselineStrips(3000), true);
  }

  function scan() {
    document.querySelectorAll('.jfx').forEach(attach);
    document.querySelectorAll('button.jarm').forEach(attachArmable);
    document.querySelectorAll('[id^="strip_"]').forEach(attachStrip);
    transport();
    attachStart();
    sync();
    dotSync();     // a pad click assigns .value server-side and fires no event
    pollStrips();
  }
  // A fader's slot can now be sized in vh/calc, so its pixel height changes
  // when the window does -- repaint, or the cap sits at the old position.
  window.addEventListener('resize', () => {
    document.querySelectorAll('.jfx').forEach(el => {
      if (el._jfxPaint && el._jfxShown !== undefined) el._jfxPaint(el._jfxShown);
    });
  });

  new MutationObserver(scan).observe(document.documentElement,
                                     {childList: true, subtree: true});
  setInterval(scan, 250);       // gradio repaints lazily; cheap and idempotent
  scan();
})();
</script>
"""


def fader_html(target_id, label, colour="#22c55e", height=170, arm=None):
    """The drawn fader. `target_id` is the elem_id of the hidden Gradio slider
    it drives -- that slider remains the state, this is only the handle.

    `arm` is the elem_id of a submit button to pulse when this fader moves
    (see the module docstring, point 3); None for faders that apply live.

    `height` is px when it is a number, or any CSS length when it is a string
    -- e.g. `calc(var(--pad) - 148px)` for a fader that has to fill a panel
    whose height is tied to something else."""
    a = f" data-arm='{arm}'" if arm else ""
    h = f"{height}px" if isinstance(height, (int, float)) else str(height)
    return (f"<div class='jfx' data-target='{target_id}'{a} "
            f"style='--jfx-c:{colour};--jfx-h:{h}'>"
            f"<div class='cap'>{label}</div>"
            f"<div class='slot'><div class='fill'></div><div class='knob'></div></div>"
            f"<div class='val'>-</div></div>")


def _selftest():
    h = fader_html("sl_bed_gain", "level", "#22c55e", 180)
    assert "data-target='sl_bed_gain'" in h and "--jfx-h:180px" in h
    assert "class='slot'" in h and "class='knob'" in h
    assert "data-arm" not in h, "a fader with no submit button must not arm one"
    a = fader_html("sl_target_v", "valence", "#22c55e", 180, arm="btn_set")
    assert "data-arm='btn_set'" in a
    # the two things that make it a real control rather than a rotated one
    assert "pointerdown" in FADER_HEAD and "setPointerCapture" in FADER_HEAD, \
        "must use pointer events, so a trackpad and a touchscreen both work"
    assert "rotate(" not in FADER_HEAD, "nothing here may be a rotated slider"
    # and the bit that keeps Gradio as the source of truth
    assert "dispatchEvent(new Event('input'" in FADER_HEAD, \
        "must write through a native input event -- that is what Gradio hears"
    assert "MutationObserver" in FADER_HEAD, "gradio repaints; must re-attach"
    # release fires only on a native pointerup on the range input, which a
    # drawn fader never produces -- so the drag end must synthesise one
    assert "new Event('pointerup'" in FADER_HEAD, \
        "drag end must commit, or `.release` can never fire for a drawn fader"
    # a server-side update assigns .value and fires nothing; without the poll
    # the fader silently desyncs from its own component
    assert "function sync()" in FADER_HEAD and "_jfxShown" in FADER_HEAD, \
        "must repaint when a handler (e.g. the VA pad) moves the component"
    # the pulse must never become a server round-trip: `change` fires per move
    assert "jarmpulse" in FADER_HEAD and "jflashfade" in FADER_HEAD
    # a strip arms its own APPLY: any param change inside it, and the strip's
    # on/off button, which only STAGES the layer on this skin
    assert "function attachStrip" in FADER_HEAD
    assert "'btn_' + key + '_apply'" in FADER_HEAD, \
        "strip_<key> -> btn_<key>_apply is a convention s20_skins._apply shares"
    for ev in ("'input', armIt, true", "'change', armIt, true"):
        assert ev in FADER_HEAD, \
            f"strip must arm on {ev} in the CAPTURE phase (components stop " \
            f"propagation), or dropdowns and checkboxes never pulse APPLY"
    assert "button.jlit" in FADER_HEAD, "toggling a layer must arm its APPLY"
    # a gradio Dropdown is not a <select>: picking an option fires no event in
    # the strip, so APPLY has to be armed by watching VALUES, not events
    assert "function stripValues" in FADER_HEAD and "function pollStrips" in FADER_HEAD, \
        "dropdown changes must arm APPLY -- events alone miss them"
    # transport lights are derived from START's label, never from a flag of
    # their own -- a second copy of "is it running" is a second thing to drift
    assert "function transport()" in FADER_HEAD
    assert "indexOf('STOP')" in FADER_HEAD, \
        "running state must be READ OFF the START/STOP button the server sets"
    assert "T.until = Date.now() + 5000" in FADER_HEAD, \
        "NEXT goes quiet for 5s after a click before inviting again"
    # START seeds the session from the visible controls (s02 adopt_layers), so
    # pressing it COMMITS every strip -- they must re-baseline, not re-arm on
    # the dropdown re-fill that comes back with the response
    assert "function rebaselineStrips" in FADER_HEAD \
        and "function attachStart" in FADER_HEAD, \
        "START must re-baseline the strips it just committed"
    assert "attachStart();" in FADER_HEAD, \
        "attachStart must be called from scan() or the listener is never bound"
    # the red dot is the target's READOUT, so it must follow BOTH routes into
    # the target -- the pad click AND the two faders (exercised for real in
    # smoke/s21_arming_test.js; this only catches deletion)
    assert "function dotSync" in FADER_HEAD and "va_dot_tgt" in FADER_HEAD, \
        "the VA faders must move the red target dot, not just the pad"
    print("SELFTEST OK: fader markup, pointer-event dragging, native input "
          "dispatch, synthetic release-on-commit, external-update sync, "
          "client-side arm/flash and strip-wide arming (params + the on/off "
          "button) all present. Nothing is rotated, and nothing in the pulse "
          "path touches the server. Actual dragging, and whether the gate "
          "greys what it should, are browser-side and NOT verified here.")


if __name__ == "__main__":
    _selftest()
