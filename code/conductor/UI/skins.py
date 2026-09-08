"""s20_skins.py -- the conductor's UI as a swappable SKIN.

2026-08-02, feedback: *"treat UI as a skin and as an arg to the run script."*

THE CONTRACT. A skin declares Gradio components and returns them in a dict
under the names below. `s02_conductor_app.build_demo` then wires every event
BY NAME. Layout and behaviour are therefore independent: a skin can put the
reverb picker anywhere, or hide it inside an accordion, without touching a
single handler.

`validate()` is not optional politeness. Every optional layer in this app
degrades quietly by design, which is exactly how the melody layer shipped dead
for two weeks -- so a skin that forgets a component must fail
LOUDLY at construction, not produce a UI whose buttons silently do nothing.

Skins available:
  classic -- the research surface: every knob, grouped by subsystem. Still
             declared inline in s02 (it is the known-good layout; moving it is
             a separate, riskier change).
  deck    -- this file. Five panels grouped by PROVENANCE -- what learned this
             control, and from what data -- with a badge on each. That
             regrouping is the substance; the deck styling is secondary.
"""

# every component build_demo wires an event to, or reads as an input
REQUIRED = {
    "state", "label_space", "start_btn", "va_pad", "v_slider", "a_slider", "set_btn",
    "policy_radio", "force_btn", "live_thresh", "live_energy", "live_wander",
    "live_k", "live_hold", "live_xfade", "live_level", "live_again",
    "live_voicing", "live_breath", "live_cents", "live_reverb",
    "live_breath_p", "live_btn", "bed_enable", "bed_auto", "bed_choice",
    "bed_gain", "bed_btn", "crackle_enable", "crackle_density", "crackle_gain",
    "crackle_btn", "reverb_pick", "reverb_wet", "reverb_btn", "melody_on",
    "melody_engine", "melody_ckpt", "melody_style", "melody_level",
    "melody_env", "melody_btn", "fb_like", "fb_match", "note_box", "log_btn",
    "status", "audio", "timer", "next_btn",
}

# SHORT tag, full caption ON HOVER (2026-08-02: the inline notes were "all over
# the place and eat into the panel description"). Everything explanatory lives
# in the title attribute, so the chassis stays clean and the provenance is one
# hover away rather than competing with the controls for space.
BADGES = {
    "learned":  ("LEARNED", "#16a34a", "Learned from the author's own listening "
                                      "ratings - a fitted model, not a rule."),
    "corpus":   ("CORPUS", "#7c3aed", "Measured from six ambient artists: a "
                                      "compositional prior, not a preference."),
    "judge":    ("JUDGE",  "#ca8a04", "Frozen MERT+ridge labels. Valid on "
                                      "arousal; valence is the axis this "
                                      "thesis shows to be unreliable."),
    "authored": ("AUTHORED", "#dc2626", "Hand-set by ear. NOT rated, and no "
                                      "emotion claim attaches to it."),
}

# ---------------------------------------------------------------- deck look
# the design brief (2026-08-02), from two DJ-controller photos: buttons are
# ON/OFF and LIT when on; a control that has both a level and a choice becomes
# a MULTI-CLICK button that cycles and changes colour; faders stand in for
# dropdowns he would never actually slide; the VA map sits centre-top like the
# screen on a controller.
#
# HOW THE LIGHTS WORK. Gradio cannot restyle a Button per click -- `variant`
# is the only handle and it has three values. So every lit button gets a
# stable elem_id and one hidden HTML component emits a <style> block for the
# whole panel at once. One server round-trip repaints every light, and it goes
# down the plain single-return channel that is the only one this project has
# found reliable through the share relay.
LIT = {                       # channel -> colour when lit
    "bed": "#22c55e", "melody": "#38bdf8", "crackle": "#f59e0b",
    "reverb": "#a855f7", "texture": "#ef4444", "policy": "#eab308",
    "distance": "#14b8a6",
}

CSS = """
/* The panel is now transport + map + mood + strips ACROSS, so the chassis
   gets wider rather than the map getting smaller (2026-08-02): the
   map is the readout the whole thing exists to show. 1280 was gradio's
   default column, not a measurement. */
/* THE CHASSIS SIZES ITSELF. 1720px left ~500px of dead space right of the
   strips; my arithmetic replacement for it (--midw + 3*--strip + slack) was
   ~30px short and CROPPED BED and TEXTURE, because it cannot see gradio's own
   container and row padding (2026-08-02).
   So stop computing it: `.jdeck` is the bordered box the brief calls the
   chassis, and `width:max-content` makes it exactly as wide as the row it
   contains -- flush against the strips by construction, and self-correcting if
   --pad or --strip ever move. The outer container stays generous and simply
   centres it; its background is the page colour, so its width is not visible.
   `flex-shrink:0` on the columns is the belt: if the viewport is narrower than
   the deck, it scrolls rather than crushing a strip. */
.gradio-container {background:#0b0d10; max-width: 1720px !important;}
.jdeck {width: max-content !important; margin-left:auto !important;
        margin-right:auto !important;}
.jdeck > * {flex-shrink:0 !important;}
.jdeck {background:linear-gradient(#1a1d23,#121418); border:1px solid #2b2f37;
        border-radius:12px; padding:8px; margin:4px 0;
        box-shadow:inset 0 1px 0 #33383f;}
/* HUMAN FEEDBACK: the same chassis, and dark like the rest of it. This row
   used to sit in gradio's default light styling under a dark deck, which is
   why it read as a different application rather than part of the instrument.
   Widget internals are targeted by TYPE (input, label) rather than by
   gradio's class names, which change between releases. */
.jfb {background:linear-gradient(#1a1d23,#121418); border:1px solid #2b2f37;
      border-radius:12px; padding:8px 12px 10px; margin:4px auto;
      box-shadow:inset 0 1px 0 #33383f;
      width:max-content !important; max-width:100% !important;}
.jfb .jhead {justify-content:flex-start; margin-bottom:8px;}
/* THE NOTE BOX FILLS THE PANEL, AND LOG SITS AT ITS TOP (2026-08-03).
   `align-items:center` was the whole problem: it gave every child its natural
   height and floated it in the middle of the row, so the note box was one line
   tall with dead space above and below it, and LOG hung halfway down beside it.
   `stretch` hands each child the row's full height -- which the radio pills set,
   since they wrap -- and only LOG opts back out. */
/* Human Feedback is now an EVALUATION instrument (2026-08-08): two mutually
   exclusive pairs stacked into a 2x2 block, plus a comment box for anything
   to improve. Tight gap so each pair reads as ONE question rather than two
   rows of unrelated controls. */
/* MIN-WIDTH IS WHY LOG HUNG OUT OF THE PANEL. Gradio writes an inline
   `min-width` on every Row child -- 320px for a Column, 160px for a
   component -- and .jfbrow is `flex-wrap:nowrap`, so three children whose
   minimums exceed the strip do not wrap, they OVERFLOW, and the last one
   (LOG) hangs past the edge. min_width=0 is passed in Python for all three;
   this backstops it in case a gradio default reappears. */
#strip_feedback > * {min-width:0 !important;}
/* The two questions are boxed and EXACTLY equal: the column is a flex
   column with the row's full height, and each question takes `flex:1 1 0`,
   which splits that height evenly regardless of how many lines each label
   wraps to. Sizing them by content would make them differ, because the two
   questions are different lengths. */
/* HEIGHT MUST BE EXPLICIT HERE. `height:100%` plus `flex:1 1 0` and
   `min-height:0` on the children collapsed the whole group to nothing
   (2026-08-08, feedback: "the questions and buttons have disappeared") --
   the percentage had no resolved parent height to work from, so it fell
   back to auto, and children that are allowed to shrink to zero duly did.
   A fixed height cannot do that, and it is also the only way the two boxes
   come out EQUAL: they split a known height rather than sizing to their
   labels, which are very different lengths. */
/* THE THREE NUMBERS ARE NOW DERIVED, NOT MATCHED BY HAND (2026-08-09).
   It used to be two hardcoded 186s -- one on .jfbgrid, one on .jnote -- kept
   equal by whoever remembered, because equal heights under
   `align-items:flex-start` is what lines the bottom of the lower question up
   with the bottom of the comment box. Any change to the gap silently broke
   that alignment. Now only two numbers are authored:
     --fbq   height of ONE question box (was 186/2 = 93)
     --fbgap the space between the two
   and --fbh falls out of them, so BOTH columns read the same derived value and
   the bottoms cannot drift apart. Widening the gap now makes the panel taller
   rather than squeezing the questions, which is the intended trade. */
.gradio-container {--fbq: 93px; --fbgap: 18px;
                   --fbh: calc(var(--fbq) * 2 + var(--fbgap));}
.jfbgrid {gap:var(--fbgap) !important; display:flex !important;
  flex-direction:column !important; height:var(--fbh) !important;
  flex:0 0 152px !important;}
/* ...AND THE GAP HAS TO LAND ON THE WRAPPER GRADIO ACTUALLY BUILDS. The two
   Radios are consecutive form components, so gradio wraps BOTH of them in ONE
   shared `<div class="form">`. That leaves .jfbgrid with a single child, so its
   `gap` had nothing to sit between and the two boxes rendered flush against
   each other -- the gap was authored and simply never applied.
   Same family of mistake as `.jfbrow :has(> .jnote)`: assuming a wrapper per
   component when gradio built one wrapper for the group. The fix is to style
   the real wrapper, and it must also carry the height, or the `flex:1 1 0` on
   the boxes has no resolved parent height to divide.
   (1,2,0) beats `.jfb .form {padding:0}` at (0,2,0) -- checked with
   check_css.py, not by eye. */
#strip_feedback .jfbgrid .form {display:flex !important;
  flex-direction:column !important; gap:var(--fbgap) !important;
  height:100% !important; min-height:0 !important; background:none !important;}
/* PREFIXED WITH #strip_feedback ON PURPOSE, and this is why the boxes did not
   appear for three rounds. `.jfb .block { border:none !important }` below
   strips gradio's frames from the whole panel; against that, a plain `.jfbq`
   is specificity (0,1,0) versus (0,2,0) and LOSES -- !important does not
   settle a tie between two !important rules, specificity does, and source
   order only breaks an exact tie. An id makes it (1,1,0), which cannot lose.
   The repo's own note says to check specificity before reaching for another
   !important; I added the !important and not the check.
   The box is on .jfbq -- the radio's OWN block -- so it also does not depend
   on how many wrappers gradio puts in between; `flex:1 1 0` inside a fixed
   height is what makes the two exactly equal despite different label lengths. */
#strip_feedback .jfbq {flex:1 1 0 !important; min-height:66px !important;
  display:flex !important; flex-direction:column !important;
  justify-content:center !important; overflow:visible !important;
  border:1px solid #3a4150 !important; border-radius:8px !important;
  background:#12151acc !important; padding:6px 10px !important;
  box-shadow:inset 0 1px 0 #ffffff08 !important;}
/* the question is the label; bold it, and beat `.jfb label, .jfb span` which
   sets the size for everything in the panel. */
#strip_feedback .jfbq span[data-testid='block-info'],
#strip_feedback .jfbq .block-info {font-weight:700 !important;
  font-size:11.5px !important; color:#dbe1ea !important; opacity:1 !important;
  white-space:normal !important; line-height:1.2 !important;
  margin-bottom:4px !important;}
/* Yes/No on ONE row under its question; the LABEL wraps instead, because
   every pixel the question takes is a pixel off the comment box. */
.jfbgrid .wrap {display:flex !important; flex-wrap:nowrap !important;
  gap:4px 14px !important;}
.jfbgrid span[data-testid='block-info'], .jfbgrid .block-info {
  white-space:normal !important; line-height:1.15 !important;
  font-size:11px !important; opacity:.85; margin-bottom:3px !important;}
.jfbgrid label {white-space:nowrap;}
/* LOG is the only child that does not stretch: fixed and narrow, so the
   comment box gets everything the two boxed questions do not. */
/* THE COMMENT BOX GETS A FLOOR, and the row is allowed to widen to honour it
   (feedback: "if the entire human feedback section needs to be wider to
   accommodate it, so be it"). Trimming the other two only ever bought a few
   dozen pixels; a hard min-width on the box itself is the only thing that
   actually doubles it, because it makes the ROW grow rather than the box
   shrink. Questions 152px + LOG 104px + box >=340px. */
/* ALL THREE ID-SCOPED, because the `#strip_feedback > *` backstop above is
   (1,0,0) and a bare `.jnote` is (0,1,0) -- the backstop WINS and pins the
   box to min-width:0, which is the opposite of what is wanted here. Caught
   by check_css.py, not by reading. `#strip_feedback .jnote` is (1,1,0) and
   clears it; the two id-only rules tie at (1,0,0) and are ordered after. */
#strip_feedback #btn_feedback_apply {flex:0 0 104px !important;
  min-width:104px !important;}
#strip_feedback .jnote {flex:1 1 auto !important; min-width:340px !important;
  height:var(--fbh) !important; min-height:var(--fbh) !important;}
/* and the textarea has to follow it, or the box is --fbh tall with a 90px
   control floating inside. This must out-specify `.jnote .wrap` (0,2,0). */
#strip_feedback .jnote .wrap, #strip_feedback .jnote > div,
#strip_feedback .jnote textarea {height:100% !important; min-height:0 !important;}
#strip_feedback {width:100% !important;}
/* flex-start, not stretch: with both columns now carrying the same explicit
   --fbh height, stretching would fight those numbers. Aligning the TOPS and
   giving both the same height is what lines the bottoms up -- which is why
   --fbh is derived once and read twice, rather than typed twice. */
.jfbrow {align-items:flex-start !important; gap:14px !important;
         flex-wrap:nowrap !important;}
/* SCOPE THIS TO .jnote AND NOTHING ABOVE IT. The first attempt reached upwards
   with `.jfbrow :has(> .jnote)`, copying the catch-all that works for .jstrips.
   It broke the panel (2026-08-03, feedback: "you fucked it"): gradio puts ONE
   SHARED wrapper around all three components in this Row, so `:has(> .jnote)`
   matched that wrapper and `flex-direction:column` stacked the radio, the note
   box and LOG on top of each other.
   `:has()` is only safe when the element it finds wraps the ONE thing being
   styled. Here it does not -- the strips case had a wrapper per group, this row
   has a wrapper for the row. The row must stay left-to-right: radio, then note
   box, then LOG.
   So: the row supplies the height via `align-items:stretch` above, and the
   chain below runs strictly DOWNWARD from .jnote to its <textarea>. The
   textarea is excluded from the flex-column rule (`> div`, not `> *`) since
   making the control itself a flex container is meaningless; `height` is
   !important because gradio's autoresize writes an inline height. */
   TWO DIFFERENT THINGS, and conflating them cost a round each way. Reaching
   UPWARD is required -- `height:100%` resolves against the parent, so every
   box between the row and the textarea needs a height or the percentage
   collapses to auto and the box falls back to one line. What broke the panel
   was not reaching up, it was changing DIRECTION up there. So ancestors get
   height and nothing else; only .jnote and below become a flex column. */
.jfbrow :has(.jnote) {height:100% !important; align-items:stretch !important;
        min-height:0 !important;}
.jnote, .jnote > div, .jnote label, .jnote .wrap, .jnote .container {
        display:flex !important; flex-direction:column !important;
        height:100% !important; min-height:0 !important;}
.jfb textarea {flex:1 1 auto !important; height:100% !important;
        min-height:0 !important; resize:none !important;
        box-sizing:border-box !important;}
/* LOG keeps its natural height and pins to the top edge the note box starts at.
   BY ID ONLY, for the reason above -- anything that selects the button's
   ancestor risks selecting the wrapper the whole row shares. */
#btn_feedback_apply {align-self:flex-start !important; flex:0 0 auto !important;}
/* strip gradio's own frames so the panel is one surface, not three boxes.
   MARGIN, NOT JUST PADDING (2026-08-09, feedback: "the huge margin gap at the
   bottom"). This reset zeroed `padding` and left `margin` alone, and gradio
   ships a margin-bottom on its block/form wrappers -- so every wrapper in the
   panel contributed dead space that no padding rule here could reach, and it
   pooled at the bottom of the chassis under the tallest column. Zeroed
   together, because a frame-strip that removes the border and keeps the
   spacing is only half a strip. Panel spacing is owned by .jfb's own padding
   and the two authored variables above, so there is nothing left for a
   gradio default to contribute. */
.jfb .block, .jfb .form, .jfb .wrap, .jfb .styler, .jfb .gr-group,
.jfb fieldset {background:transparent !important; border:none !important;
      box-shadow:none !important; padding:0 !important; margin:0 !important;}
.jfb label, .jfb span, .jfb .wrap label {color:#cbd2da !important;
      font-size:.82em !important;}
.jfb input[type="text"], .jfb textarea {background:#05070a !important;
      color:#cbd2da !important; border:1px solid #2b2f37 !important;
      border-radius:6px !important; padding:7px 9px !important;}
.jfb input[type="text"]::placeholder,
.jfb textarea::placeholder {color:#5b636e !important;}
/* the radio pills: dark chips, lit like a strip button when chosen */
.jfb .wrap {display:flex !important; flex-wrap:wrap !important;
      gap:6px !important;}
.jfb .wrap label {background:#14171c !important;
      border:1px solid #2b2f37 !important; border-radius:6px !important;
      padding:5px 10px !important; cursor:pointer;}
.jfb .wrap label:has(input:checked) {background:#2b3340 !important;
      border-color:#4b5563 !important; color:#e4e4e7 !important;}
.jfb input[type="radio"] {accent-color:#38bdf8;}
.jscreen {background:#05070a; border:2px solid #2b2f37; border-radius:8px;
          padding:6px;}
.jstrip {background:#15181d; border:1px solid #262a31; border-radius:8px;
         padding:5px 3px; margin:0 2px;}
.jstrip .jlabel {font-size:.7em; letter-spacing:.14em; text-transform:uppercase;
                 color:#7c8390; text-align:center; margin-bottom:4px;}
/* A strip's controls are DEAD until its top button turns the layer on
   (2026-08-02). The base rule gates; _lit's <style> block overrides
   it with id specificity when the layer goes on -- so the light and the gate
   are the same single round-trip, and the gate can never disagree with the
   lamp. `.jgate` marks a gated control; the on/off and APPLY buttons are
   deliberately NOT gated (you must be able to turn a layer off and commit it). */
.jstrip.gated .jgate, .jstrip.gated .jfx {opacity:.3; pointer-events:none;
        filter:grayscale(.85); user-select:none;}
/* panel header: name + short provenance tag, caption on hover. Without these
   the badges wrapped into the panel body -- "all over the place and eats into
   the panel description" (2026-08-02). */
/* WRAP, not nowrap: "FLANGE / PHASER" + its AUTHORED badge is wider than a
   138px strip, and with nowrap the heading overflowed BOTH edges -- the
   rendered deck read "ANGE / PHASER" with the badge sliced off (measured
   2026-08-03). Wrapping drops the badge to a second line instead. A heading
   must never be able to lose characters to make a layout fit. */
.jhead {display:flex; align-items:center; justify-content:center; gap:6px;
        flex-wrap:wrap; margin:0 0 6px 0; cursor:help;}
.jhead .jt {font-size:.74em; letter-spacing:.14em; text-transform:uppercase;
            color:#cbd2da; font-weight:700; white-space:nowrap;}
.jhead .jbadge {font-size:.52em; letter-spacing:.08em; color:#08131f;
                padding:1px 5px; border-radius:3px; font-weight:700;
                white-space:nowrap;}
/* the VA map, TIGHT to its canvas: gradio wraps an Image in padded frames
   that leave the picture floating in the middle of the screen panel. The
   frame SHRINK-WRAPS the picture (fit-content), so there is no dead space
   left and right of it, and both take their size from --pad so map and frame
   always agree. */
.jpad, .jpad > div, .jpad .image-container, .jpad .image-frame,
.jpad .wrap, .jpad .block {background:transparent !important;
        border:0 !important; padding:0 !important; margin:0 !important;
        box-shadow:none !important; min-height:0 !important;}
/* --pad is the ONE number that sizes this row: the map is square, so it is
   both the picture's side and the Mood panel's height. */
/* FIXED px, not vh: the deck should look the same whatever the window is
   doing (2026-08-02). vh made every panel breathe as the browser
   resized, which is exactly the aesthetic drift he wanted gone. --sq is the
   side of every square button. */
.gradio-container {--pad: 360px; --sq: 52px;}
/* Explicit size, NOT a percentage. `min(40vh, 100%)` inside a fit-content
   frame is circular -- the percentage resolves against a container that is
   itself sizing to the image -- and the picture collapsed. */
.jpad img {width:var(--pad) !important; height:var(--pad) !important;
        display:block; margin:0; object-fit:contain; border-radius:5px;}
.jscreencol {flex:0 0 auto !important; width:fit-content !important;
        min-width:0 !important;}
.jscreen {width:fit-content; margin:0; padding:5px; position:relative;}

/* LIVE VA MARKERS over the map: green = where the walk IS, red = where it is
   being sent. Both blink so they read as instruments rather than decoration,
   and they are offset by half their size so the dot's CENTRE lands on the
   coordinate. */
/* The gradio BLOCK around the markers must leave the flow too, or it sits
   below the picture as an ordinary div and widens the whole column -- which is
   what put the dots off the map and added margin beside the Mood strip. */
.jdotwrap {position:absolute !important; inset:5px !important;
        margin:0 !important; padding:0 !important; border:0 !important;
        background:transparent !important; pointer-events:none !important;
        min-width:0 !important; z-index:5;}
.jdotwrap > * {margin:0 !important; padding:0 !important;
        background:transparent !important; border:0 !important;}
.jdots {position:absolute; inset:0; pointer-events:none;}
.jdots span {position:absolute; width:13px; height:13px; margin:-6.5px 0 0 -6.5px;
        border-radius:50%; border:2px solid #05070a;}
/* SAME period, opposite phase: different periods made them drift in and out of
   step, which reads as flicker. One breathes up while the other breathes down,
   and the swing is shallow (1 -> .5) so they pulse rather than flash. The
   negative delay starts the target half a cycle in, so it is out of phase
   immediately instead of after one period. */
#va_dot_cur {background:#22c55e; box-shadow:0 0 10px #22c55e;
        animation:jblink 2.4s ease-in-out infinite;}
#va_dot_tgt {background:#ef4444; box-shadow:0 0 10px #ef4444;
        animation:jblink 2.4s ease-in-out infinite; animation-delay:-1.2s;}
@keyframes jblink {0%,100%{opacity:1;} 50%{opacity:.5;}}
/* THE MOOD CONTROLS SIT BESIDE THE MAP, NEVER UNDER IT. Gradio gives a Column
   a default min-width of 320px and lets its Row wrap, so the middle column was
   too narrow to hold map+mood on one line and the panel dropped underneath --
   which is exactly what it looked like: "still UNDER the VA PNG". Shrink the
   column to its content and forbid the wrap. */
.jmid {flex:0 0 auto !important; width:fit-content !important;
        min-width:0 !important; display:flex !important;
        flex-direction:column !important; align-self:stretch !important;}
/* the LED group takes whatever height the map group does not, so the bottom
   of this column lines up with the bottom of the strips */
.jmid > *:first-child {flex:1 1 auto !important; display:flex !important;
        flex-direction:column !important;}
.jmid > *:first-child .jled {flex:1 1 auto !important; height:100% !important;}
/* The readout's own foot: a pixel spectrum with the delivery numbers BESIDE
   it, not under it (2026-08-02). Both are absolutely positioned, so
   neither the bars nor a long status line can resize the group -- a loud
   passage cannot push the readout around, and a server-side markdown update,
   which replaces the readout's content, cannot wipe them. */
.jledwrap {position:relative;}
/* ONE BLACK BOX, NOT TWO. MEASURED 2026-08-09 (full subtree probe): gradio's
   Markdown renders `.block jled` (the component) wrapping `.prose jled` (the
   rendered markdown), and BOTH carry the elem_classes -- the same
   class-lands-on-several-nested-elements trap CLAUDE.md records for Groups.
   So every .jled rule applied twice: two backgrounds, two borders, and
   padding 10/42 counted TWICE, which pushed the inner box 10px PAST its own
   parent (block 264..483, prose 274..493) to paint black over the chassis
   padding -- the "bleed into the grey margin". The foot is positioned against
   the block, so the bars looked like they were escaping a box that was
   actually the inner one overflowing.
   The INNER copy is neutralised here: the component keeps the box, the prose
   is just text inside it. Height must be released too, or the 219px pin
   applies twice as well. */
.jledwrap .prose.jled {background:transparent !important; border:0 !important;
        padding:0 !important; min-height:0 !important; max-height:none !important;
        height:auto !important;}
.jfoot, .jfoot > * {padding:0 !important; margin:0 !important;
        background:transparent !important; border:0 !important;
        min-height:0 !important;}
#conductor-eq {position:absolute; left:10px; bottom:6px; width:58%; height:30px;
        display:block; opacity:.85; pointer-events:none;}
/* top of the foot, right of the bars: fixed box, clipped rather than growing */
#conductor-statusbar {position:absolute; right:12px; bottom:24px; left:62%;
        height:14px; line-height:14px; text-align:right;
        font:11px ui-monospace,SFMono-Regular,Menlo,monospace;
        color:#7c8390; pointer-events:none; white-space:nowrap;
        overflow:hidden;}
/* keep the server's text clear of the foot */
.jled, .jled p, .jled li {padding-bottom:0;}
.jledwrap .jled {padding-bottom:42px !important;}
/* NO BLANK LINE BETWEEN THE MESSAGE LINES (2026-08-07). The status is
   markdown and its "\n\n" makes each line its own <p>, which carries gradio's
   paragraph margin -- so a two-line message spent a third line on air. Killed
   here rather than by rewriting the message strings, because the same strings
   also go to the WebAudio player overlay, where the markdown is not rendered
   and a <br> would show up as literal text. */
.jledwrap .jled p {margin:0 !important;}
/* THE ONE NUMBER THAT SIZES THIS BOX, and the only one to touch.
   The NOW/NEXT table is pinned out of flow at bottom:44px (see settings_panel
   in s02), so it contributes NO height -- meaning this floor is what has to
   account for it. The budget, measured in the browser's terms rather than
   guessed at a fourth time:
       10px  padding-top
       28px  two lines of message (.74em x 1.4)
       ~66px the 4-row table + its caption and rule
       44px  clearance to the foot (#conductor-eq is 30px at bottom:6px)
     = ~148px
   Set to 150 for 2px of slack. The brief asked for ~2 lines shorter than the
   previous 178, and the table was tightened (line-height 1.25 -> 1.15) to pay
   for it -- without that this does not fit.
   It is a FLOOR, not a height: the rule ~30 lines above still gives this box
   `flex:1 1 auto; height:100%`, so on a tall window it simply fills what the
   map leaves and this never binds.
   IF THE TABLE TOUCHES THE GREEN TEXT, RAISE THIS. If dead space returns
   above the table, lower it. Nothing else needs to move. */
.jledwrap .jled {min-height:219px; max-height:219px !important;}
/* HEIGHT IS NOW PINNED, NOT STRETCHED. MEASURED 2026-08-09 (console probe):
   the box was 242px because `.jmid > *:first-child .jled {height:100%}`
   stretches it to whatever the middle column leaves -- so min-height was
   never the lever. Text flows from the TOP, the table is pinned to the
   BOTTOM (bottom:44px, inline in s02), so all the slack landed between
   them: 132px of green-text band for two lines of message.
   226 IS MEASURED, NOT ESTIMATED -- my 176 was arithmetic and it was wrong
   twice over: it left a 53px HOLE (the outer group is flex-stretched to the
   strips column and does not shrink with this box) and it starved the message
   band. From the probe: outer wrap 68..303 with 4px bottom padding, inner
   starts at 73, so 303-4-73 = 226 is exactly the height that reaches the
   bottom. That fills the hole AND returns ~3 lines to the message band.
   TRIMMED 226 -> 219 (2026-08-09): 226 filled the group exactly, which
   put the black box flush against the chassis padding and let the foot
   read as bleeding into the grey. 7px of breathing room fixes it.
   The statusbar's right inset went 10 -> 12px in the same pass, to
   match the table block's inset -- the trailing 's' of "lead 27.8s"
   was overhanging the table's right rule by exactly that 2px.
   THE ONE KNOB: this number. max-height must stay equal to min-height, or
   the flex rule above wins again.
   CAVEAT, because it is a pin and not a fill: it is tied to the deck's
   current height (set by the strips column). If the deck ever gets taller a
   hole returns below the console, and this number is what closes it -- run
   the probe in CLAUDE.md and use `wrap[0].bottom - 4 - wrap[1].top`. */   /* 150 + 40: MEASURED 2026-08-09 from
   a console probe -- outer .jledwrap 278..514, inner 283..474, i.e.
   the console border stopped 40px short of the group and the bars/status
   line sat in that dead strip. This box is what the wrap sizes to, so
   raising it by exactly that 40 grows the border DOWN to enclose them.
   The table anchors to the wrap bottom (bottom:44px, inline in s02's
   settings_panel) so it travels down with it and the 9th row -- distance,
   added the same day -- stops crowding the message. Committed geometry of
   the foot and table is otherwise untouched. */
/* NO min-height HERE, deliberately -- and this is the second time this box has
   taught the lesson. A `min-height:230px` floor was added with the NOW/NEXT
   table on 2026-08-07 and made the LED TALLER than the map beside it, growing
   the whole middle column and leaving a band of dead space (same
   day). The floor was never needed: the rule ~30 lines above already gives
   this box `flex:1 1 auto; height:100%`, so it ALWAYS fills whatever height
   the map group leaves -- it was big enough before the table existed and the
   dead space was already there.
   So the table is not stacked under the message, it is absolutely positioned
   into that existing dead space at the bottom of `.jledwrap` (see
   settings_panel in s02). It therefore costs ZERO height, which is what makes
   "the LED is big enough" and "no resizing" true at the same time, and it is
   literally what was asked for: the table at the BOTTOM.
   The pattern is the foot's own: #conductor-eq and #conductor-statusbar are both
   absolutely positioned in this same wrapper for the same reason -- so that
   nothing the server writes into the readout can resize the group. */
/* The LED box is PINNED to the width of the map+mood group below it, so a long
   line wraps instead of stretching the column and dragging the strips right.
   --pad (map) + the Mood strip + their margins and group padding. */
.jmid > * {width:var(--midw) !important; box-sizing:border-box;}
/* BOTH children of the middle column take this width, so the LED box and the
   map+mood panel share a right edge. = map + Mood strip + their frames; keep
   it in step if either changes. */
.gradio-container {--midw: calc(var(--pad) + 150px + 30px);}
/* THE RIGHT-HAND OVERHANG IS THE LEFT COLUMN. Counted five times as a strips
   problem and it never was; MEASURED 2026-08-03 it is an intrinsic-sizing
   mismatch two columns away from where it shows up.
   `.jdeck` is `width:max-content` (:90), so the deck reserves each child's
   INTRINSIC width. The left column lays out at 158px -- gradio's min_width
   floor, nothing more -- but its intrinsic width is ~215px, inflated by the
   long hover captions on Transition and Stereo. The deck reserves 215, the
   column uses 158, and the 57px difference falls out as dead air AFTER THE
   LAST COLUMN, which is why it read as the strips overhanging.
   The books balanced exactly: 215.5 + 544 + 434 + two 16px gaps = 1225.5,
   against a measured 1225.5px content box, with the columns themselves
   occupying only 1168.
   `max-width` is the fix and `width` is not: under a max-content constraint
   `width:fit-content` resolves BACK to max-content, so it cannot clamp the
   sizing pass -- max-width binds during intrinsic sizing and does.
   --leftw MUST TRACK the min_width on that column; it is set to the width the
   column already lays out at, so this clamps the reservation WITHOUT moving
   any control. Widen both together if anything in that column grows. */
.gradio-container {--leftw: 158px;}
.jleft {max-width:var(--leftw) !important;}
/* no slack to the right of the Mood strip -- it is the last thing in the row */
#strip_va {margin-right:0 !important;}
.jvarow > *:last-child {margin-right:0 !important;}
.jled {width:100% !important; box-sizing:border-box; overflow-wrap:anywhere;}
.jvarow {flex-wrap:nowrap !important; align-items:flex-start !important;
        width:fit-content !important;}
/* A group box hugs its contents, so fixed-width strips leave no dead space
   inside a stretched container. */
.jfit {width:fit-content !important;}
/* The Mood panel: TIGHT, and exactly as tall as the map beside it. Gradio
   columns grow to fill the row by default, which is what gave it all that
   space. */
/* min-height, NOT height+overflow:hidden -- that clipped the faders straight
   out of the panel when the chrome above them measured taller than my
   estimate. A layout guess must never be able to DELETE a control. */
#strip_va {flex:0 0 auto !important; width:150px !important;
        min-width:0 !important; min-height:var(--pad); box-sizing:border-box;}
/* STRIP WIDTHS ARE HARDCODED (2026-08-02, feedback: "hardcode their widths").
   min_width is only a FLOOR, and the columns were scale=1, which tells gradio
   to grow them -- so they ate every spare pixel in the row. scale=0 plus a
   fixed width is the only way to make a width mean what it says. --strip is
   the single dial for all of them. */
.gradio-container {--strip: 138px;}
.jstrips {flex-wrap:nowrap !important; justify-content:flex-start !important;
        align-items:stretch !important; flex:1 1 auto !important;}
.jstrips > * {flex:0 0 auto !important;}
/* The strips block matches the MAP + READOUT column beside it. Adding the EQ
   made the middle column taller, and the strips kept their old fixed heights,
   so the deck stopped squaring off at the bottom (2026-08-02). Every
   link in the chain has to pass the stretch down -- container, group, row,
   column -- or the last fixed height wins. */
.jdeck > *:last-child {display:flex !important; flex-direction:column !important;
        align-self:stretch !important;}
/* THE STRIPS COLUMN MUST HUG ITS GROUP, exactly as .jmid does.
   MEASURED 2026-08-03 in headless Chrome at 1900px: the strips group's right
   edge sat at x=1405 while the chassis ran to x=1487 -- 82px, of which only
   the 8px chassis padding is wanted. 74px of dead air on a 1258px chassis
   (5.9%), which is the right-hand overhang as reported.
   `.jfit` made the GROUP shrink-wrap, but the COLUMN around it was still free
   to be wider, and `.jdeck`'s max-content sizing then took the COLUMN as the
   contribution. Constrain the column too and the chassis closes on the
   strips. Same three-part treatment .jmid already gets; keep them in step. */
.jstripcol {flex:0 0 auto !important; width:fit-content !important;
        min-width:0 !important;}
/* MEASURED IN A REAL BROWSER (headless Chrome, 2026-08-02), after guessing at
   this three times: a gradio Group renders TWO NESTED `.gr-group` divs -- both
   carrying the elem_classes I asked for -- wrapped around a `.styler` div.
   `> .jgrp` therefore stretched only the OUTERMOST one (572px) while the inner
   group and the styler kept flex-grow:0 and collapsed to their content (366px).
   That 206px difference IS the dead space under the strips.
   So stretch EVERY ancestor of the strips row, not just the one I could see in
   the component tree. The `:has()` clause is the catch-all: any wrapper a
   future gradio inserts is, by definition, an element containing .jstrips. */
.jdeck > *:last-child .jgrp,
.jdeck > *:last-child .styler,
.jdeck > *:last-child :has(> .jstrips) {
        flex:1 1 auto !important; display:flex !important;
        flex-direction:column !important; min-height:0 !important;}
/* MEASURED 2026-08-03, and it settles the right-hand overhang for good: the
   strips chain is NOT the culprit and never was. In-browser, at the deck's
   own scale, .jstrips 855-1275 / .styler 855-1275 / inner .jgrp 855-1275 /
   outer .jgrp 850-1280 / .jstripcol 848-1282 -- 5px of group padding and 2px
   of column, nesting cleanly, no dead space anywhere in it.
   The gap is in .jdeck: children 158 + 544 + 434 plus two 16px gaps = 1168
   occupied, against a 1225.5px content box. 57.5px of air AFTER the last
   column. See the .jmid note below -- do not "fix" this by squeezing the
   strips again; four rounds of that measured no change, because the strips
   were already flush. */
/* height:100% AS WELL AS flex. flex:1 1 auto only stretches the BOX; the
   children inside it still had no definite height to resolve against, which
   is why the group grew and the strips in it did not (2026-08-02).
   Each link states its height explicitly so the next one down has something
   real to be a percentage of. */
.jstrips {height:100% !important; min-height:0 !important;}
/* A COLUMN OF STRIPS IS A GRID, for the same reason a strip's button row is
   (see .jfrow). Flex was the obvious answer and it did not deliver: `flex:1 1
   auto` grew the group but left the strips at their min-heights, twice
   (2026-08-02, feedback: "you have not increased the heights of the strips
   themselves - only the group box"). Grid does not negotiate -- two fr tracks
   divide whatever height the column has, and that is the end of it.
   The fr numbers are the OLD fixed heights used as a RATIO, so the top strips
   still match each other and the bottom strips match each other; both columns
   share the template, so the rows line up across the deck and MELODY finishes
   flush beside them.
   An fr track's automatic minimum is its content, so a strip can never be
   crushed -- if the deck is short the column grows instead. */
.jcol {display:grid !important; grid-template-rows:250fr 218fr !important;
        align-self:stretch !important; height:100% !important;
        gap:0 !important; min-height:0 !important;}
.jcol > * {min-height:0 !important; height:auto !important;}
#strip_melody {height:100% !important;}
#strip_melody, .jcol {width:var(--strip) !important; min-width:0 !important;
        flex:0 0 var(--strip) !important;}
/* MELODY spans the group's full height: stretch, rather than height:100% of a
   parent that is itself auto-height. */
#strip_melody {align-self:stretch !important; height:auto !important;}

/* The grid, per the brief (2026-08-02):
       col-0  MELODY   spanning both rows
       col-1  REVERB   over CRACKLE
       col-2  BED      over TEXTURE
   Heights pair by ROW, so the two top strips match, the two bottom strips
   match, and both columns therefore total the same -- which is what lets
   MELODY span them and finish flush. min-height, never height: a strip that
   outgrows the guess must get taller, not lose its contents. */
/* Floors for a short viewport only -- the grid above sets the real heights. */
#strip_reverb, #strip_bed {min-height:var(--rowa, 250px);}
#strip_crackle, #strip_texture {min-height:var(--rowb, 218px);}
/* "by mood" owns the soundscape dropdown: picking the bed by hand needs the
   automatic choice off. Two classes beat the one-class un-gate rule. */
.jstrip .jbedpick.jgate {opacity:.32; pointer-events:none; filter:grayscale(.85);}

/* ---- the LED readout -------------------------------------------------- */
/* The running commentary, moved up beside the transport (2026-08-02) and
   dressed as a console: phosphor green with a glow, amber for the bold bits
   markdown already emits, cyan for emphasis. Fixed height + scroll so a long
   line can never shove the deck down the page. */
/* NO fixed height, so nothing to scroll and nothing to crop: the box grows to
   its text. A fixed height gave a scrollbar first and a clipped last line
   after -- both are the same mistake, forcing content into a size I guessed.
   min-height keeps the console from twitching on short messages. */
.jled {background:#05070a !important; border:1px solid #2b2f37;
       border-radius:6px; padding:10px 12px !important;
       font-family:ui-monospace,SFMono-Regular,Menlo,monospace !important;
       height:auto !important; min-height:var(--led, 118px);
       max-height:none !important;}
.jled, .jled * {overflow:visible !important; max-height:none !important;}
.jled > *:last-child, .jled p:last-child {margin-bottom:0 !important;}

/* A GROUP is a chassis holding several strips: one border round the machine
   controls, one round the mood display, one round the layer strips. The
   strips keep their own frames inside it. */
.jgrp {background:#101317; border:1px solid #2b2f37; border-radius:10px;
       padding:4px; margin:0 2px 3px;}
.jgrp .jstrip {margin:0 2px 2px;}
.jgrp .jgrp {margin:0; padding:0; border:0; background:transparent;}

/* The style-carriers (_lit / _pick emit a <style> block through a gr.HTML)
   must take NO layout space -- an empty gradio block still occupies a row,
   which is the gap that looked like a spring under the Transition buttons.
   CSS inside a display:none element still applies. */
.jstyle {display:none !important;}
.jled, .jled p, .jled li {font-size:.74em !important; line-height:1.4 !important;
       color:#22c55e !important; text-shadow:0 0 6px rgba(34,197,94,.55);}
.jled strong {color:#f59e0b !important; text-shadow:0 0 8px rgba(245,158,11,.7);}
.jled em {color:#38bdf8 !important; font-style:normal;
       text-shadow:0 0 8px rgba(56,189,248,.6);}
.jled code {color:#e879f9 !important; background:transparent !important;
       text-shadow:0 0 8px rgba(232,121,249,.6);}
.jpol {font-size:.7em;}
.jpol label {font-size:.92em !important;}

/* ---- tighter strips: the whole deck should land on one screen --------- */
/* Shrink-wrap the drawn fader and centre it: as a full-width block its slot
   sat hard left with dead space beside it. A WIDER slot then fills what is
   left of the column. */
.jfx {width:fit-content; margin:0 auto;}
/* A ROW IS A GRID, not a flex line. Third attempt at BY MOOD, and the first
   that cannot drift: flex cells size partly from their CONTENT, so the row
   [fader, hidden slider, button] and the row [button, button] above it were
   never guaranteed to divide the strip the same way, however the hidden slider
   was neutralised. `grid-auto-flow:column` with equal auto-columns divides the
   strip by COUNT alone -- two in-flow children give two identical columns in
   both rows, so the button under APPLY lands under APPLY by construction.
   TEXTURE's three faders get three equal columns by the same rule. */
.jfrow {display:grid !important; grid-auto-flow:column !important;
        grid-auto-columns:minmax(0,1fr) !important; gap:2px !important;
        align-items:center !important; position:relative !important;}
.jfrow > * {min-width:0 !important; display:flex !important;
        align-items:center !important; justify-content:center !important;}
/* ...EXCEPT the invisible ones. `_lit` puts its <style> carrier in the row
   next to the button, and the rule above was overriding `.jstyle`'s
   display:none (same specificity, later in the sheet) -- so every strip built
   with _lit laid out THREE columns and pushed its ON and APPLY out to the
   edges. BED looked right because it is the one strip that builds its carrier
   outside the row, which is exactly the tell (2026-08-02). Two
   classes beat one class plus a universal. */
.jfrow > .jstyle {display:none !important;}
/* A square must stay square in ANY cell. Next to a 100px fader the cell is
   tall, and without this the button stretched to fill it -- which is why BY
   MOOD did not match APPLY, whose cell is only as tall as a button. */
/* Centre EVERY strip control. jsq buttons already centre themselves; this
   catches the rest -- a lone APPLY, a dropdown, a checkbox -- which were
   sitting hard left or hard right depending on gradio's own block. */
.jstrip .block, .jstrip .form, .jstrip button {margin-left:auto !important;
        margin-right:auto !important;}
.jstrip button:not(.jsq) {display:block !important; width:92% !important;}
button.jsq {flex:0 0 auto !important;
        min-width:var(--sq) !important; max-width:var(--sq) !important;
        min-height:var(--sq) !important; max-height:var(--sq) !important;}
.jstrip .jfx {gap:2px; padding:1px 0;}
.jstrip .jfx .slot {width:40px;}
.jstrip .jfx .knob {width:36px; left:1px;}
/* TEXTURE carries THREE faders where the others carry one or two, so it gets
   a narrower slot rather than every strip getting wider to suit one panel. */
#strip_texture .jfx .slot {width:30px;}
#strip_texture .jfx .knob {width:26px;}
#strip_texture .jfx {padding:0;}
.jstrip .jfx .cap {font-size:.55em; letter-spacing:.06em;}
.jstrip .jfx .val {font-size:.66em;}
.jstrip button.jbtn {padding:3px 5px !important; font-size:.62em !important;
        min-width:0 !important;}
/* Dropdowns keep their NATIVE font size. Shrinking it did not shrink gradio's
   own padding, so every field ended up ringed in dead border (the brief,
   2026-08-02). Width is what buys a tighter strip, not type size. */
.jstrip .jgate {width:100% !important; min-width:0 !important;}
/* Trim the WRAPPER's spacing only. Do NOT restyle the field itself: putting a
   border on the input left it ringed by gradio's border AND mine, which read
   as one fat one (2026-08-02). Native field, tight wrapper. */
.jstrip .block, .jstrip .form {padding:0 !important; margin:0 !important;
        min-width:0 !important;}
.jstrip .jhead {margin-bottom:3px;}

/* ---- square buttons, transport and policy ----------------------------- */
/* A real square: fixed side, centred in whatever column it lands in.
   width:100%+aspect-ratio gave a wide rectangle whenever the column was wider
   than the max-height cap. */
button.jsq {width:var(--sq) !important; height:var(--sq) !important;
        min-width:0 !important; padding:2px !important; margin:0 auto !important;
        display:block !important; white-space:normal !important;
        line-height:1.05 !important; font-size:.56em !important;}
/* NEXT's prompt is slower than an APPLY pulse: it is an invitation, not an
   uncommitted change waiting on you. The LIT window stays ~0.7s and only the
   gap grows (2026-08-02) -- at 2.6s it read as an alarm, "users will
   think WTF is going on". Hence the flat 44/56% shoulders rather than a longer
   ease over the whole period, which would have dimmed the flash as well as
   spacing it out. */
@keyframes jslowpulse {
  0%, 44%, 56%, 100% {box-shadow:0 0 0 rgba(56,189,248,0);    border-color:#3a3f47;}
  50%                {box-shadow:0 0 15px rgba(56,189,248,.8); border-color:#38bdf8;}
}
button.jslow {animation: jslowpulse 6s ease-in-out infinite;
              color:#8fd3f4 !important;}
/* the slider a fader DRIVES is hidden: s21 draws the handle, gradio keeps the
   state. Not display:none -- the frontend must still render the <input> for
   the fader to find and write to. */
/* Zero WIDTH as well as height. _fader puts two components in the row -- the
   drawn handle and this hidden slider -- so with width left alone every hidden
   slider still claimed a full flex cell: half of TEXTURE's row, a third of the
   Mood row. That is what pushed the visible faders off-centre. Still not
   display:none -- the frontend must render the <input> for the handle to find
   and write to. */
.jhidden {position:absolute !important; height:0 !important;
          width:0 !important; flex:0 0 0 !important;
          min-width:0 !important; overflow:hidden !important;
          opacity:0 !important; margin:0 !important; padding:0 !important;}
button.jbtn {font-weight:700 !important; letter-spacing:.08em !important;
             text-transform:uppercase !important; font-size:.72em !important;
             border:1px solid #2b2f37 !important; color:#8b919b !important;
             background:#15181d !important;
             /* ROUNDED CORNERS (professors, 2026-08-07). Set here rather than
                on a narrower selector so every deck button -- power switches,
                APPLY, transport, the disabled placeholders -- gets the same
                shape; gradio's own radius is otherwise inherited unevenly
                depending on which of its wrappers a button lands in. */
             border-radius:10px !important;}
/* The squares get a proportionally larger radius: 10px on a 52px box reads as
   a barely-softened square, and the professors asked for visibly rounded. */
button.jsq {border-radius:14px !important;}
/* A power switch reading "TURN ON" is two words on a 52px square, so it must
   be allowed to break AT THE SPACE and sit centred on two lines. jsq already
   sets white-space:normal, which is all that takes.
   NO `overflow-wrap:anywhere` HERE. It was added with the TURN ON label on
   2026-08-07 as belt-and-braces and immediately broke something else: `.jlit`
   is worn by _pick buttons too, and `anywhere` breaks mid-WORD, so BASELINE
   came out as "BASELIN" + "E" (same day). A defensive rule that
   changes behaviour for every wearer of the class is not defensive. */
button.jlit {hyphens:none !important;
             display:flex !important; align-items:center !important;
             justify-content:center !important; text-align:center !important;}
/* LEARNED / BASELINE are WORDS, not the one-or-two-glyph labels the square is
   sized for, and they must sit on ONE line. Widened rather than shrunk: the
   type is already .62em and the deck is read across a room in a demo.
   Only the POLICY pick pair moves -- the transport squares, the power
   switches and every APPLY keep --sq, so nothing else in the column reflows.

   NARROWED 68 -> 60px (2026-08-09, feedback: "too rectangular ... more square
   but without making the text multilined"). Against the 52px height, 68 is
   aspect 1.31; 60 is 1.15.
   WHAT WAS SPENT, AND WHY IT WAS NOT THE TYPE: `button.jbtn` sets
   letter-spacing .08em, and across the 8 characters of BASELINE that is ~0.6em
   of pure tracking -- roughly the width recovered here, bought for style
   rather than legibility. Zeroing it on THESE TWO BUTTONS ONLY pays for the
   narrowing at no cost to reading distance, which is what the 2026-08-02
   decision above was protecting. Shrinking font-size would reverse it.
   `white-space:nowrap` makes the one-line requirement STRUCTURAL rather than a
   consequence of the width happening to be enough -- it is scoped to this
   strip, so TURN ON elsewhere still breaks at its space (see the jlit note).
   BUDGET: 2 x 60 + the row gap + strip padding ~= 132px, further inside the
   left column's --leftw:158px clamp than before. NARROWING IS THE SAFE
   DIRECTION -- it only adds slack. Widening still requires moving that clamp,
   or the column overflows and the old right-hand dead-air bug comes back.
   TO GO FULLY SQUARE (60 -> --sq 52) the remaining ~8px can only come out of
   the type, so measure the rendered label first; do not just set the number. */
.gradio-container {--pickw: 60px;}
#strip_policy button.jsq {width:var(--pickw) !important;
        min-width:var(--pickw) !important; max-width:var(--pickw) !important;
        letter-spacing:0 !important; padding:2px 1px !important;
        white-space:nowrap !important;}
"""

def head(title, kind=None, note=""):
    """Panel header: name, short provenance tag, everything else on hover.

    `note` is folded INTO the tooltip rather than printed beside the title --
    laid out inline it collided with the panel body text."""
    if not kind:
        return f"<div class='jhead'><span class='jt'>{title}</span></div>"
    label, colour, tip = BADGES[kind]
    full = tip + (f"  -  {note}" if note else "")
    return (f"<div class='jhead' title=\"{full}\">"
            f"<span class='jt'>{title}</span>"
            f"<span class='jbadge' style='background:{colour}'>{label}</span>"
            f"</div>")


def validate(c, skin_name):
    missing = REQUIRED - set(c)
    if missing:
        raise RuntimeError(
            f"skin '{skin_name}' is missing {len(missing)} component(s) the "
            f"event wiring needs: {sorted(missing)}. Every one of them is an "
            f"input or output of a handler -- a UI without them would build "
            f"fine and then do nothing.")
    return c


def lightboard(states):
    """One <style> block lighting every button that is currently ON.

    `states` maps channel -> None (off) or an index into its colour ramp. A
    multi-click channel lights a DIFFERENT shade per selection, which is how a
    single button carries both "is it on" and "which one".
    """
    css = []
    for ch, idx in states.items():
        if idx is None:
            continue
        base = LIT.get(ch, "#22c55e")
        # successive selections step the lightness so the choice is readable
        shade = ["", "cc", "99"][min(int(idx), 2)]
        css.append(
            f"button#btn_{ch}{{background:{base}{shade} !important;"
            f"color:#08131f !important;border-color:{base} !important;"
            f"box-shadow:0 0 12px {base}88 !important;}}")
    return "<style>" + "".join(css) + "</style>"


def _strip(gr, c, key, label, toggle_kw, faders, cycle=None):
    """One mixer channel: a lit button on top, faders under it.

    `cycle` is the list a multi-click button steps through; the SELECTION is
    written into a hidden component so the existing event wiring -- which
    expects a Dropdown or Checkbox -- keeps working untouched. The visible
    button drives the hidden control; nothing downstream knows the difference.
    """
    with gr.Column(min_width=110, elem_classes="jstrip"):
        gr.HTML(f"<div class='jlabel'>{label}</div>")
        c[f"{key}_btn_ui"] = gr.Button(toggle_kw["off"], elem_id=f"btn_{key}",
                                       elem_classes="jbtn", size="sm")
        for fk, fkw in faders:
            with gr.Column(elem_classes="jfader"):
                c[fk] = gr.Slider(**fkw)


# The two states of a layer's power switch. ASYMMETRIC ON PURPOSE (professors,
# 2026-08-07: "you cannot tell whether it's currently ON or click to turn ON").
#
#   dark  + "TURN ON"  -> unambiguously an ACTION. This was the broken half:
#                         a dark button reading "OFF" could equally mean
#                         "the layer is off" or "press me to switch it off".
#   lit   + "ON"       -> unambiguously a STATE. A glowing button saying ON
#                         cannot be read as an instruction, and it stays
#                         glanceable across six strips at once, which a
#                         symmetric "TURN OFF" would have cost.
#
# BTN_ON MUST STAY EXACTLY "ON". s21_faders.py's attachStart reads
# `lamp.textContent === 'ON'` to tell a _lit power switch apart from a _pick
# button (LEARNED / BASELINE), which also carries `.jlit` but has no on/off
# meaning. Reading the <style> carrier instead would make the policy strip look
# permanently ON -- its default lamp is always emitted -- and every START would
# wrongly re-arm it. Change this string and that check must change with it.
BTN_ON = "ON"
BTN_OFF = "TURN ON"


def _lit(gr, c, key, colour, label, target, strip=None, on=False):
    """A mixer button: lit when its layer is ON, and the strip's power switch.

    `on` starts the layer LIT and its strip UNGATED -- for the first-run
    defaults (s02's START_VA block), so a visitor can press START and hear the
    system configured rather than bare. The hidden `target` checkbox must be
    created with the SAME value, since that is what START actually adopts.

    Drives a HIDDEN Gradio control (`target`) rather than replacing it, so the
    handler wiring in s02 -- which expects a Checkbox -- is untouched. The
    button is the affordance; the checkbox is still the state.

    Colour is repainted by returning a one-rule <style> block, because Gradio
    cannot restyle a Button per click: `variant` is the only handle and it has
    three values, none of them per-channel.

    `strip` is the elem_id of the strip this button governs. Turning the layer
    ON emits the un-gate rule alongside the lamp, so ONE round-trip does both
    and the two can never disagree; turning it off emits nothing and the base
    `.jstrip.gated` rule takes the controls back out. Note this only STAGES the
    change -- s02 does not apply on toggle for this skin, the strip's APPLY
    does, which is why the click also arms APPLY (client-side, s21).
    """
    def lamp(now):
        """The whole visual state of the layer, as one <style> block: the lit
        button AND the strip's un-gate. Built by ONE function used for both the
        initial render and every toggle, so a layer that starts ON cannot come
        up lit-but-gated (or gated-but-lit) -- the two are the same code path
        rather than two copies that have to be kept in step."""
        if not now:
            return ""                   # base CSS re-gates the strip
        css = (f"button#btn_{key}{{background:{colour} !important;"
               f"color:#08131f !important;border-color:{colour} !important;"
               f"box-shadow:0 0 14px {colour}99 !important;}}")
        if strip:
            css += (f"#{strip} .jgate,#{strip} .jfx{{opacity:1 !important;"
                    f"pointer-events:auto !important;filter:none !important;}}")
        return f"<style>{css}</style>"

    btn = gr.Button(BTN_ON if on else BTN_OFF, elem_id=f"btn_{key}",
                    elem_classes=["jbtn", "jsq", "jlit"], size="sm")
    style = gr.HTML(lamp(on), elem_classes="jstyle")

    def toggle(is_on):
        now = not bool(is_on)
        # The LABEL carries the state, like START/STOP -- nothing to read but
        # the button itself. See BTN_ON/BTN_OFF for why the two are worded
        # differently and why BTN_ON is pinned to the literal "ON".
        return now, lamp(now), gr.update(value=BTN_ON if now else BTN_OFF)

    btn.click(toggle, inputs=[target], outputs=[target, style, btn])
    return btn


def _pick(gr, key, options, target, colour, default=0):
    """Complementary square buttons: picking one drops the other.

    Same trick as `_lit` -- the buttons drive a HIDDEN component that stays the
    state, so s02 keeps reading a Radio and its handler signature is untouched.
    Only the selected button's lamp is emitted, and the base style is unlit, so
    "select this one" and "deselect the other" are one and the same <style>.

    The buttons carry `jlit`, so picking also arms the panel's APPLY (s21):
    the choice is STAGED, and APPLY commits it.
    """
    def lamp(idx):
        return (f"<style>button#btn_{key}_{idx}{{background:{colour} "
                f"!important;color:#08131f !important;border-color:{colour} "
                f"!important;box-shadow:0 0 14px {colour}99 !important;}}"
                f"</style>")

    # LIT FROM THE START for the default (2026-08-02). The lamps only
    # ever appeared on a click, so on load nothing was lit and the deck did not
    # say which policy was actually in force -- while the session had one from
    # the first segment.
    style = gr.HTML(lamp(default), elem_classes="jstyle")
    made = []
    # SIDE BY SIDE, like START/NEXT: `jsq` is width:100% + aspect-ratio, so
    # stacked in a column each one spans the full width and comes out a wide
    # rectangle. A row is what makes them square.
    with gr.Row(elem_classes="jfrow"):
        for i, opt in enumerate(options):
            made.append((gr.Button(opt["label"], elem_id=f"btn_{key}_{i}",
                                   elem_classes=["jbtn", "jsq", "jlit"],
                                   size="sm"),
                         opt["value"], i))

    def picker(value, idx):
        def pick():
            return value, lamp(idx)
        return pick

    for btn, value, i in made:
        btn.click(picker(value, i), outputs=[target, style])
    return [b for b, _, _ in made]


def _apply(gr, key, label="APPLY", square=False, **kw):
    """A strip's APPLY. Deliberately the SAME control as SET TARGET -- same
    class, same pulse, same light-then-fade (2026-08-02): one visual
    grammar for "you have changed something; commit it".

    The elem_id is a CONVENTION s21's script relies on: a strip `strip_<key>`
    arms the button `btn_<key>_apply`. Keep them in step.

    `**kw` goes straight to gr.Button -- LOG needs `scale` because it sits in a
    proportioned Row rather than a channel strip. Anything passed here is
    layout only; the id and the classes are the contract and are not
    overridable."""
    cls = ["jbtn", "jarm"] + (["jsq"] if square else [])
    return gr.Button(label, elem_id=f"btn_{key}_apply", elem_classes=cls,
                     size="sm", **kw)


def _fader(gr, key, colour, label, height=128, arm=None, **kw):
    """A REAL fader: a drawn handle over a hidden Gradio slider.

    The slider stays the state (every handler input list is unchanged); s21's
    script binds the handle to it by elem_id and writes through a native input
    event. Replaces `transform: rotate(-90deg)`, which kept the element's
    original box for layout and hit-testing -- short throw, wrong hit area,
    stray scrollbars.

    `arm` is the elem_id of a submit button to pulse while this fader's value
    is uncommitted -- for the VA target, which applies on a button rather than
    live. Client-side only: gradio dispatches `change` on every pointer-move,
    so a server-side pulse would be a round-trip per move.

    The default height is what a CHANNEL STRIP gets; the deck should land on
    one screen (2026-08-02), so anything taller has to ask."""
    import faders as fx
    eid = f"sl_{key}"
    gr.HTML(fx.fader_html(eid, label, colour, height, arm=arm))
    return gr.Slider(elem_id=eid, elem_classes="jhidden", show_label=False,
                     container=False, **kw)


def dots_markup():
    """The two markers. Positions are NOT inline: each dot is placed by its own
    <style> carrier so the two can be updated independently -- see dot_style."""
    return ("<div class='jdots'><span id='va_dot_cur'></span>"
            "<span id='va_dot_tgt'></span></div>")


def dot_style(which, v, a):
    """Place ONE dot. `which` is "cur" or "tgt".

    Separate carriers because the two dots answer to different things: the
    green one follows the walk and must repaint every segment, while the red
    one is a USER INTENTION and must not move on its own (the brief, 2026-08-02:
    the target auto-wanders, so a shared blob would have crept the red dot
    across the map on every clip change). One blob for both would mean every
    update rewrote both positions.

    VA runs [-1,1] and make_pad_assets draws the axes with no margins, so the
    mapping is linear; y is flipped because image rows run downward."""
    x, y = (float(v) + 1.0) * 50.0, (1.0 - float(a)) * 50.0
    return (f"<style>#va_dot_{which}{{left:{x:.2f}%;top:{y:.2f}%;}}</style>")


def build_deck(gr, ctx):
    """A mixer strip deck: screen at the top, channel strips underneath.

    the design brief (2026-08-02) from two DJ-controller photos: buttons are
    ON/OFF and LIT when on; faders stand in for values he would not slide
    anyway; the VA map sits centre-top like the screen on a controller.

    The provenance tag stays on each strip, short, with the full caption on
    hover -- the deck has to work both as something to perform with and as
    something a professor can read.
    """
    c = {}
    c["state"] = gr.State(None)

    # ------------------------------------------------- the main control panel
    # ONE panel, three groups (2026-08-02): the machine controls down
    # the left, the mood display in the middle with its readout beneath, and
    # the six layer strips on the right. Everything that steers a running
    # session is now on one screen.
    _SH = 100      # fader height in the stacked strips, so two fit a column
    # TABS WRAP ONLY THE DECK ROW (professors, 2026-08-07: flipping to the
    # graphics must leave the intro and the "Pacing, loudness and wander"
    # sections visible). That falls out for free rather than needing a
    # restructure: the intro accordion is built by s02 BEFORE build_deck is
    # called, and the pacing accordion plus Human Feedback are built after this
    # `with` block closes -- so all three are already siblings of the tab set,
    # not children of it. Keep it that way; moving either accordion inside a
    # Tab is what would break the requirement.
    import visuals as vis
    with gr.Tabs():
      with gr.Tab("MIXER DECK"):
        with gr.Row(elem_classes="jdeck"):

            # ---- LEFT: transport, transition, stereo, stacked ----------------
            # jleft: the CSS clamps this column's INTRINSIC width to --leftw. Without
            # it the deck reserves ~215px for the hover captions below and only uses
            # 158, and the surplus surfaces as dead air right of the strips.
            with gr.Column(scale=0, min_width=158, elem_classes="jleft"):
                with gr.Group(elem_classes="jgrp"):
                    # TRANSPORT. START's LABEL is the running state: s21 reads it to decide
                    # which light pulses, so there is no second flag to drift out of step.
                    with gr.Column(scale=0, min_width=140, elem_classes="jstrip"):
                        gr.HTML(head("Transport"))
                        with gr.Row(elem_classes="jfrow"):
                            # size="sm" like every other square: gradio's default
                            # is "lg", which gave START/NEXT different padding and
                            # type inside the same 52px box.
                            c["start_btn"] = gr.Button("START", variant="primary",
                                                       elem_id="btn_start", size="sm",
                                                       elem_classes=["jbtn", "jsq"])
                            c["next_btn"] = gr.Button("NEXT", elem_id="btn_next",
                                                      size="sm",
                                                      elem_classes=["jbtn", "jsq"])
                    # POLICY. Two complementary square buttons over a hidden Radio, and
                    # CHANGE NOW recast as this panel's APPLY (2026-08-02) -- so
                    # a policy choice is staged and committed like every other strip, and
                    # committing it also moves the music so you can hear the difference.
                    with gr.Column(scale=0, min_width=150, elem_id="strip_policy",
                                   elem_classes="jstrip"):
                        gr.HTML(head("Transition", "learned",
                                     "which model chooses HOW to move between soundscapes"))
                        c["policy_radio"] = gr.Radio(
                            ["Learned preference model", "Baseline - best-rated match"],
                            value="Learned preference model", visible=False)
                        _pick(gr, "policy",
                              [dict(label="LEARNED", value="Learned preference model"),
                               dict(label="BASELINE", value="Baseline - best-rated match")],
                              c["policy_radio"], LIT["policy"])
                        c["force_btn"] = _apply(gr, "policy")
                    # DISTANCE, live since 2026-08-09 -- and note the RENAME. This
                    # strip was drawn as STEREO|MONO and left disabled, because a
                    # control that looks live and does nothing is the failure this
                    # project keeps repeating (README SS22.7). What unblocked it was
                    # not finishing the wiring but reading the measurement: WIDTH is
                    # the wrong axis for this instrument -- 92% of the energy is
                    # centred bass below ~126 Hz where azimuth localisation is poor,
                    # and panning moved 7.6% of the signal, which is why it came back
                    # "barely noticeable". Distance acts on the whole signal, so the
                    # control is NEAR..FAR and the label had to change with it.
                    # Still badged `authored`: the near/far mapping is a scenographic
                    # choice, nothing about it is rated.
                    #   OPTIONAL in the skin contract (not in REQUIRED, same as
                    # vis_data) so the classic skin is untouched and s02 wires these
                    # only if present.
                    # CONVENTION AS CRACKLE, exactly (2026-08-09, feedback: the
                    # first cut had the fader lit while the layer was off, and
                    # APPLY never armed). Both were the same omission: the strip
                    # machinery is keyed on ids. `elem_id="strip_distance"` is
                    # what s21's syncStripArm derives `btn_distance_apply` from
                    # (strip.id.slice(6)), and the `gated` class + `strip=` in
                    # _lit are what dim the fader until the layer is ON.
                    with gr.Column(scale=0, min_width=150, elem_id="strip_distance",
                                   elem_classes=["jstrip", "gated"]):
                        gr.HTML(head("Distance", "authored",
                                     "places the whole mix near or far -- wetter and "
                                     "duller as it recedes, not merely quieter"))
                        c["dist_enable"] = gr.Checkbox(value=False, visible=False)
                        with gr.Row(elem_classes="jfrow"):
                            _lit(gr, c, "distance", LIT["distance"], "NEAR/FAR",
                                 c["dist_enable"], strip="strip_distance")
                            c["dist_btn"] = _apply(gr, "distance", square=True)
                        with gr.Row(elem_classes="jfrow"):
                            c["dist_amount"] = _fader(gr, "dist_amt", LIT["distance"],
                                                      "far", height=_SH, minimum=0.0,
                                                      maximum=1.0, value=0.35, step=0.05)
                            # "far" is depth (wetter/duller); "drift" is the
                            # audible-on-headphones width -- a 45 s equal-power
                            # sweep of the WHOLE mix, which is the form of
                            # panning the 7.6% measurement never ruled out.
                            c["dist_drift"] = _fader(gr, "dist_drift", LIT["distance"],
                                                     "drift", height=_SH, minimum=0.0,
                                                     maximum=1.0, value=0.0, step=0.05)

                    # FLANGE / PHASER -- on the board, NOT cut (2026-08-02).
                    # Present and disabled until the effect is in the render chain;
                    # it gets a rated probe or a tuner slider when it is, the same
                    # way reverb did, rather than a hand-wired "chorus = happy".
                    with gr.Column(elem_classes="jstrip"):
                        gr.HTML(head("Flange / Phaser", "authored",
                                     "on the board, not yet wired - this does nothing"))
                        gr.Button("OFF", elem_classes=["jbtn", "jsq"], size="sm",
                                  interactive=False)
            # ---- MIDDLE: the readout, then the mood display -----------------
            with gr.Column(scale=0, min_width=0, elem_classes="jmid"):
                with gr.Group(elem_classes=["jgrp", "jledwrap"]):
                    c["status"] = gr.Markdown("Press **START**.",
                                              elem_classes="jled")
                    # The readout's foot: a pixel spectrum and one permanent status
                    # line, tmux-style (2026-08-02) -- the delivery numbers
                    # were in a floating corner chip nobody looked at. Both are
                    # SEPARATE components, not markdown, so a status update cannot
                    # wipe them; s19_webaudio finds them by id and draws into them,
                    # and falls back to the corner chip when a skin omits them.
                    # The analyser is a passive TAP on the existing graph: it reads
                    # what is already playing and cannot delay a fetch or a decode.
                    gr.HTML("<canvas id='conductor-eq' width='512' height='68'></canvas>"
                            "<div id='conductor-statusbar'>idle</div>",
                            elem_classes="jfoot")
                with gr.Group(elem_classes="jgrp"):
                    with gr.Row(elem_classes="jvarow"):
                        with gr.Column(elem_classes="jscreencol"):
                            with gr.Column(elem_classes="jscreen"):
                                if ctx["pad"]:
                                    # buttons=[] drops gradio's download/share/fullscreen
                                    # overlay (default is all three) -- a readout, not a
                                    # download.
                                    c["va_pad"] = gr.Image(ctx["pad"], show_label=False,
                                                           interactive=False, container=False,
                                                           buttons=[], elem_classes="jpad")
                                else:
                                    c["va_pad"] = gr.Image(None, show_label=False, visible=False)
                            c["va_dots"] = gr.HTML(dots_markup(), elem_classes="jdotwrap")
                            # both markers start where the SESSION starts
                            # (s02.START_VA), or the display would open telling
                            # a different story from the walk
                            c["va_dot_cur"] = gr.HTML(dot_style("cur", *ctx["start_va"]),
                                                      elem_classes="jstyle")
                            c["va_dot_tgt"] = gr.HTML(dot_style("tgt", *ctx["start_va"]),
                                                      elem_classes="jstyle")

                        # MOOD: headed like every other strip. NOT gated -- there is no "VA
                        # off", the walk always has a target.
                        with gr.Column(scale=0, min_width=190, elem_id="strip_va",
                                       elem_classes="jstrip"):
                            gr.HTML(head("Mood", "judge", "grey on the map = no anchors in "
                                                          "the bank, so the walk cannot go there"))
                            c["set_btn"] = _apply(gr, "va", "SET TARGET")
                            # Session-start-only is real but the disclaimer cost more room
                            # than it earned; it survives as the hover caption.
                            # jnoarm: this one is read by START, not by SET TARGET, so it must
                            # NOT pulse SET TARGET -- that would promise a commit that does not
                            # apply to it.
                            c["label_space"] = gr.Dropdown(
                                ["judge labels", "human labels"],
                                value=ctx["label_space"],
                                label="VALENCE AXIS", container=False,
                                elem_classes=["jgate", "jnoarm"])
                            # Fill whatever is left of the panel, so the panel stays exactly
                            # as tall as the map on any screen instead of overflowing a short
                            # one. 148px is the chrome above and below the slot (head, SET
                            # TARGET, the dropdown, the fader's own cap and read-out).
                            _h = "calc(var(--pad) - 148px)"
                            with gr.Row(elem_classes="jfrow"):
                                c["v_slider"] = _fader(gr, "target_v", "#38bdf8", "valence",
                                                       height=_h, arm="btn_va_apply",
                                                       minimum=-1, maximum=1,
                                                       value=ctx["start_va"][0],
                                                       step=0.05)
                                c["a_slider"] = _fader(gr, "target_a", "#f472b6", "arousal",
                                                       height=_h, arm="btn_va_apply",
                                                       minimum=-1, maximum=1,
                                                       value=ctx["start_va"][1],
                                                       step=0.05)

            # ---- RIGHT: the six layer strips ---------------------------------
            with gr.Column(scale=0, min_width=0, elem_classes="jstripcol"):
                with gr.Group(elem_classes=["jgrp", "jfit"]):
                    # ---- the 2x3 strip grid, RIGHT OF THE MAP ------------------------
                    # Everything that steers a running session is now in one row, on one
                    # screen (2026-08-02): map, mood, and the six layers.
                    # MELODY runs down the first column spanning both rows -- it carries
                    # the most controls -- then BED over REVERB, CRACKLE over TEXTURE.
                    #
                    # Every strip whose layer can be OFF is `gated`: its controls are inert
                    # until the top button turns the layer on, and every change inside arms
                    # that strip's own APPLY rather than reaching the conductor directly.
                    with gr.Row(elem_classes="jstrips"):
                        # ---- column 1 -- MELODY, spanning both rows -----------------------
                        # Two tracks: the note source, and how it is voiced.
                        with gr.Column(scale=0, min_width=0, elem_id="strip_melody",
                                       elem_classes=["jstrip", "gated"]):
                            gr.HTML(head("MELODY", "learned",
                                         "the note source IS trained - but on MIDI corpora, NOT on emotion ratings like the other learned layers; and every style scored negative in the one listening round it had"))
                            # ON by default (s02 START_VA block). The checkbox is
                            # what START adopts, so it and _lit's `on=` must agree.
                            c["melody_on"] = gr.Checkbox(value=True, visible=False)
                            with gr.Row(elem_classes="jfrow"):
                                _lit(gr, c, "melody", LIT["melody"], "MELODY", c["melody_on"],
                                     strip="strip_melody", on=True)
                                c["melody_btn"] = _apply(gr, "melody", square=True)
                            c["melody_level"] = _fader(gr, "mel_lvl", LIT["melody"], "level",
                                                       minimum=-24, maximum=-4, value=-14, step=1)
                            # FIXED ORDER, so nothing jumps when the model changes
                            # (2026-08-02): envelope, then the model, then
                            # the model-SPECIFIC control last. Style and ckpt share
                            # that last slot -- exactly one is visible -- so the two
                            # rows above never move. Default is markov, so the slot
                            # starts populated with styles.
                            c["melody_env"] = gr.Dropdown(
                                choices=["swell", "legato", "struck", "flat"],
                                value="swell", label="envelope", container=False,
                                elem_classes="jgate")
                            c["melody_engine"] = gr.Dropdown(
                                choices=ctx["melody_engines"], value=ctx["melody_engine"],
                                label="track 1 - notes", container=False,
                                elem_classes="jgate")
                            # VISIBILITY FOLLOWS THE ENGINE, and the two dropdowns
                            # share one slot (see the comment above). The default
                            # is now `transformer`, which reads WEIGHTS and ignores
                            # style -- so the pair starts swapped relative to the
                            # markov default they were written for. melody_slot()
                            # in s02 keeps them in step from here on; this is only
                            # the initial state, and it has to match or the panel
                            # opens showing a control the engine never reads.
                            c["melody_style"] = gr.Dropdown(choices=ctx["melody_styles"],
                                                            value="auto",
                                                            label="track 2 - voice",
                                                            container=False, visible=False,
                                                            elem_classes="jgate")
                            # VISIBLE on load, because the default engine is now
                            # `transformer` and this is the control it reads.
                            # NOT a hardcoded name -- ctx["melody_ckpt"] comes from
                            # s02.default_ckpt(), which PREFERS L6_d128 and falls back
                            # to whatever is on disk. A name that is not in `choices`
                            # is not a bad default, it is a crash: gradio's
                            # Dropdown.preprocess rejects it on every submit that
                            # touches the component, which is what took the UI down on
                            # 2026-08-07 ("Value: cpu is not in the list of choices").
                            c["melody_ckpt"] = gr.Dropdown(choices=ctx["melody_ckpts"],
                                                           value=ctx["melody_ckpt"],
                                                           label="track 2 - weights",
                                                           container=False, visible=True,
                                                           elem_classes="jgate")
    
                        # ---- column 2 -- REVERB over CRACKLE ----------------------------
                        with gr.Column(scale=0, min_width=0, elem_classes="jcol"):
                            # REVERB -- the dropdown lost "off" (2026-08-02): the strip button
                            # now owns WHETHER, so the dropdown only says WHICH ROOM. s02 folds
                            # the two back together at apply time (cond if on else "off"), so
                            # set_reverb's signature and the classic skin are untouched.
                            with gr.Column(elem_id="strip_reverb",
                                           elem_classes=["jstrip", "gated"]):
                                gr.HTML(head("REVERB", "learned", "13:1 for reverb ON A "
                                                                  "TRANSITION, p=0.002"))
                                # ON by default (s02 START_VA block) -- and the
                                # best-evidenced layer in the system at that, 13:1
                                # on transitions. The checkbox is what START adopts,
                                # so it and _lit's `on=` must agree.
                                c["reverb_enable"] = gr.Checkbox(value=True, visible=False)
                                with gr.Row(elem_classes="jfrow"):
                                    _lit(gr, c, "reverb", LIT["reverb"], "REVERB",
                                         c["reverb_enable"], strip="strip_reverb", on=True)
                                    c["reverb_btn"] = _apply(gr, "reverb", square=True)
                                rooms = [r for r in ctx["reverb_choices"]
                                         if str(r).lower() not in ("off", "none")]
                                c["reverb_wet"] = _fader(gr, "rev_wet", LIT["reverb"], "wet",
                                                         height=_SH, minimum=0, maximum=1,
                                                         value=0.35, step=0.05)
                                # Full width at the BOTTOM: beside the fader the room names
                                # were too narrow to read.
                                c["reverb_pick"] = gr.Dropdown(
                                    choices=rooms, value=(rooms[0] if rooms else None),
                                    label="space", container=False, elem_classes="jgate")
    
                            with gr.Column(elem_id="strip_crackle",
                                           elem_classes=["jstrip", "gated"]):
                                gr.HTML(head("CRACKLE", "authored", "surface noise; broadband "
                                                                    "ticks, not pitched drips"))
                                c["crackle_enable"] = gr.Checkbox(value=False, visible=False)
                                with gr.Row(elem_classes="jfrow"):
                                    _lit(gr, c, "crackle", LIT["crackle"], "CRACKLE",
                                         c["crackle_enable"], strip="strip_crackle")
                                    c["crackle_btn"] = _apply(gr, "crackle", square=True)
                                with gr.Row(elem_classes="jfrow"):
                                    # DEFAULTS MATCH THE CLASSIC SURFACE (-18 dB,
                                    # 8 ticks/s). The deck had shipped -24 dB at 1
                                    # tick/s -- the quietest, sparsest corner of its
                                    # own range -- so turning the layer on did
                                    # almost nothing (2026-08-02). Minimum
                                    # is 1, not 0: at 0 the fader becomes a second
                                    # off switch, which is the "control pretending
                                    # to a state" trap the TEXTURE strip avoids.
                                    c["crackle_gain"] = _fader(gr, "crk_lvl", LIT["crackle"],
                                                               "level", height=_SH, minimum=-36,
                                                               maximum=-6, value=-18, step=1)
                                    c["crackle_density"] = _fader(gr, "crk_den", LIT["crackle"],
                                                                  "ticks/s", height=_SH,
                                                                  minimum=1, maximum=40,
                                                                  value=1, step=1)
    
                        # ---- column 3 -- BED over TEXTURE -------------------------------
                        with gr.Column(scale=0, min_width=0, elem_classes="jcol"):
                            with gr.Column(elem_id="strip_bed",
                                           elem_classes=["jstrip", "gated"]):
                                gr.HTML(head("BED", "corpus", "pacing = semi-Markov; "
                                                              "-24..-18 dB is where raters settled"))
                                # BED has TWO lamps that both bear on the same dropdown:
                                # the strip's ON/OFF, and "by mood", which owns the
                                # soundscape picker (choosing a bed by hand needs the
                                # automatic choice off). They therefore write ONE shared
                                # style block, computed from BOTH states -- with a block
                                # each, whichever was clicked second would carry a stale
                                # copy of the other's state and they would fight over the
                                # picker. This is why BED does not use _lit.
                                c["bed_enable"] = gr.Checkbox(value=False, visible=False)
                                c["bed_auto"] = gr.Checkbox(value=True, visible=False)

                                def _bed_css(on, auto):
                                    css = ""
                                    if on:
                                        css += (f"button#btn_bed{{background:{LIT['bed']} "
                                                f"!important;color:#08131f !important;"
                                                f"border-color:{LIT['bed']} !important;"
                                                f"box-shadow:0 0 14px {LIT['bed']}99 "
                                                f"!important;}}"
                                                "#strip_bed .jgate,#strip_bed .jfx{"
                                                "opacity:1 !important;pointer-events:auto "
                                                "!important;filter:none !important;}")
                                    # the picker is live only when the layer is on AND the
                                    # bed is being chosen by hand
                                    css += ("#strip_bed .jbedpick.jgate{"
                                            + ("opacity:1 !important;pointer-events:auto "
                                               "!important;filter:none !important;}"
                                               if (on and not auto) else
                                               "opacity:.32 !important;pointer-events:none "
                                               "!important;filter:grayscale(.85) !important;}"))
                                    # LIT ONLY WHEN THE LAYER IS ON *AND* BY MOOD
                                    # is selected (2026-08-02). Lit while
                                    # the bed was off claimed a live state the
                                    # layer does not have -- the auto choice only
                                    # means anything once there is a bed to choose.
                                    # BACKLIT, not filled (2026-08-02).
                                    # BED's ON lamp and BY MOOD were the same solid
                                    # green, which read as two of the same control.
                                    # Same colour -- they belong to the same layer
                                    # -- but the lamp is the lit one and BY MOOD
                                    # glows behind its own dark face, so the
                                    # hierarchy is visible at a glance.
                                    if on and auto:
                                        css += (f"button#btn_bed_auto{{background:"
                                                f"#15181d !important;color:{LIT['bed']} "
                                                f"!important;border-color:{LIT['bed']} "
                                                f"!important;box-shadow:0 0 12px "
                                                f"{LIT['bed']}66,inset 0 0 14px "
                                                f"{LIT['bed']}44 !important;}}")
                                    return f"<style>{css}</style>"

                                # Lit from the start: BY MOOD defaults ON, so its lamp and the
                                # dimmed picker are both already true on load.
                                _bed_style = gr.HTML(_bed_css(False, True), elem_classes="jstyle")

                                def _bed_on(on, auto):
                                    now = not bool(on)
                                    # BED rolls its own toggle (it has a second lamp
                                    # to paint), so it does NOT get _lit's labels for
                                    # free -- it has to use the same constants
                                    # explicitly or it becomes the one strip still
                                    # reading the ambiguous bare "OFF".
                                    return (now, _bed_css(now, bool(auto)),
                                            gr.update(value=BTN_ON if now else BTN_OFF))

                                def _bed_by_mood(auto, on):
                                    now = not bool(auto)
                                    return (now, _bed_css(bool(on), now),
                                            gr.update(value="BY MOOD" if now else "BY HAND"))

                                with gr.Row(elem_classes="jfrow"):
                                    _on_btn = gr.Button(BTN_OFF, elem_id="btn_bed",
                                                        elem_classes=["jbtn", "jsq", "jlit"],
                                                        size="sm")
                                    c["bed_btn"] = _apply(gr, "bed", square=True)
                                with gr.Row(elem_classes="jfrow"):
                                    c["bed_gain"] = _fader(gr, "bed_gain", LIT["bed"], "level",
                                                           height=_SH, minimum=-30, maximum=-2,
                                                           value=-18, step=1)
                                    _auto_btn = gr.Button("BY MOOD", elem_id="btn_bed_auto",
                                                          elem_classes=["jbtn", "jsq", "jlit"],
                                                          size="sm")
                                c["bed_choice"] = gr.Dropdown(
                                    choices=ctx["bed_labels"], value=None, label="bed",
                                    container=False, elem_classes=["jgate", "jbedpick"])
                                _on_btn.click(_bed_on,
                                              inputs=[c["bed_enable"], c["bed_auto"]],
                                              outputs=[c["bed_enable"], _bed_style, _on_btn])
                                _auto_btn.click(_bed_by_mood,
                                                inputs=[c["bed_auto"], c["bed_enable"]],
                                                outputs=[c["bed_auto"], _bed_style, _auto_btn])
    
                            # TEXTURE -- NOT gated: there is no "texture off" in the conductor,
                            # only depths that reach 0. Inventing a switch would be a control
                            # pretending to a state the engine does not have.
                            with gr.Column(elem_id="strip_texture",
                                           elem_classes=["jstrip", "gated"]):
                                gr.HTML(head("TEXTURE", "learned",
                                             "voice richness is rated; the wander depths "
                                             "are tuned by ear. OFF = flat: no dark "
                                             "voicing, no wander"))
                                # Symmetric with the other strips, and HONEST about
                                # what off means: the engine has no "texture off",
                                # so OFF zeroes the two character depths (dark and
                                # wander) at apply time. Variety is retrieval, not
                                # texture, so it keeps its value.
                                c["texture_on"] = gr.Checkbox(value=False, visible=False)
                                with gr.Row(elem_classes="jfrow"):
                                    _lit(gr, c, "texture", LIT["texture"], "TEXTURE",
                                         c["texture_on"], strip="strip_texture")
                                    c["live_btn"] = _apply(gr, "texture", square=True)
                                with gr.Row(elem_classes="jfrow"):
                                    c["live_voicing"] = _fader(gr, "tex_dark", LIT["texture"],
                                                               "dark", height=_SH, minimum=0,
                                                               maximum=1, value=0.7, step=0.05)
                                    c["live_breath"] = _fader(gr, "tex_wan", LIT["texture"],
                                                              "wander", height=_SH, minimum=0,
                                                              maximum=1, value=0.0, step=0.05)
                                    c["live_k"] = _fader(gr, "tex_var", LIT["texture"],
                                                         "variety", height=_SH, minimum=1,
                                                         maximum=32, value=8, step=1)

      # ---- TAB 2: the reactive visual (s22) ------------------------------
      # Sibling of the deck tab, so flipping between them leaves the two
      # accordions and Human Feedback below untouched. The canvas markup is
      # STATIC -- everything that moves is drawn client-side from
      # `vis_data`'s attributes, so the picture cannot be wiped by a status
      # update and costs no bandwidth per frame.
      with gr.Tab("VISUALS"):
        gr.HTML(vis.panel_html())
        # The parameter carrier. s02 refreshes it once per segment; it is
        # display:none and lives here only so the animation loop has
        # something to poll. Optional in the contract -- the classic skin
        # has no visuals tab and does not build one, so s02 reads it with
        # C.get() exactly like va_dots.
        c["vis_data"] = gr.HTML(vis.data_html(), elem_classes="jstyle")

                    # ------------------------------------------------ everything else, tucked
    with gr.Accordion("Pacing, loudness and wander", open=False):
        with gr.Row(elem_classes="jfrow"):
            c["live_thresh"] = gr.Slider(0.02, 0.5, value=0.15, step=0.01,
                                         label="trigger")
            c["live_energy"] = gr.Slider(0.25, 4.0, value=1.0, step=0.05,
                                         label="energy")
            c["live_wander"] = gr.Slider(0.0, 0.15, value=0.0, step=0.005,
                                         label="drift")
            c["live_hold"] = gr.Slider(1, 12, value=3, step=1, label="hold s")
            c["live_xfade"] = gr.Slider(1, 16, value=8, step=1, label="xfade s")
        with gr.Row(elem_classes="jfrow"):
            c["live_cents"] = gr.Slider(0, 30, value=0, step=1, label="cents")
            c["live_reverb"] = gr.Slider(0, 1, value=0.0, step=0.05,
                                         label="reverb wander")
            c["live_breath_p"] = gr.Slider(5, 120, value=30, step=1,
                                           label="wander period s")
            c["live_level"] = gr.Checkbox(value=True, label="equal loudness")
            c["live_again"] = gr.Slider(0, 12, value=5, step=0.5,
                                        label="arousal->loudness dB")

    # HUMAN FEEDBACK -- its own chassis, headed like every other panel.
    # `label=""` does NOT remove a gradio label, it falls back to the COMPONENT
    # TYPE, so the panel rendered as "Radio" and "Textbox" (the brief,
    # 2026-08-03) -- naming the widget instead of what it is for, and the one
    # place in the deck where the implementation leaked into the surface.
    # show_label=False is what actually suppresses it.
    with gr.Column(elem_classes="jfb"):
        gr.HTML(head("Human Feedback", None,
                     "what you thought, whenever you thought it -- logged "
                     "against the full system state of the segment playing"))
        # LOG IS AN APPLY (2026-08-03). It is the same act as every
        # other commit on the deck -- something is staged, you press the lit
        # button -- so it gets the same control, the same pulse and the same
        # light-then-fade instead of a plain button that looks identical
        # whether or not there is anything to send.
        # `strip_feedback` + `btn_feedback_apply` is s21's arming convention
        # (see _apply): the script reads every input/textarea in the strip, so
        # the radio and the note box both arm it with no extra wiring. Keep the
        # two names in step.
        # A bonus that falls out of `.jarm`: with nothing picked and nothing
        # typed the button is dim and unclickable, which is exactly the state
        # s02's log_reaction already refused ("Pick a reason or type a note
        # before logging") -- the guard is now visible instead of a surprise.
        with gr.Row(elem_classes="jfbrow", elem_id="strip_feedback"):
            # FOUR ROWS: each question is a label row plus a Yes/No row, and
            # the two stack. Putting the meaning in the LABEL rather than in
            # the option text is what keeps the options two words wide, which
            # is what frees the width for the comment box beside it -- the
            # box is where anything actionable ends up, so it gets the room.
            # gr.Radio is single-select, so Yes/No exclusivity is free; TWO
            # radios rather than one four-way keeps the questions independent
            # -- "disliked it, but it does match the sound" is a real answer
            # and the interesting one.
            with gr.Column(scale=2, elem_classes="jfbgrid", min_width=0):
                # elem_classes, NOT `.jfbgrid > div`. gradio wraps components in
                # a varying number of divs, so a child selector is a guess about
                # wrapper depth -- which is why the group boxes did not appear.
                # A class lands on the component's own block and cannot miss.
                c["fb_like"] = gr.Radio(ctx["yesno"], label=ctx["like_label"],
                                        elem_classes="jfbq")
                c["fb_match"] = gr.Radio(ctx["yesno"], label=ctx["match_label"],
                                         elem_classes="jfbq")
            # jnote: the CSS runs the height chain down to this <textarea> so
            # it fills the panel rather than sitting one line tall in the
            # middle of the row.
            # `lines` MUST BE > 1. MEASURED 2026-08-03: with lines=1 gradio
            # renders `<input type="text">`, not a `<textarea>` -- an input is
            # single-line by construction, so no height rule can ever give it a
            # second line, and every `textarea` rule in the CSS matched nothing
            # at all. The box stayed one line high through two rounds of
            # "fixing" the height chain, which was never the problem.
            # The number is only the FLOOR; the CSS stretches it to the bottom
            # of the panel, whatever the radio pills set that to.
            c["note_box"] = gr.Textbox(placeholder="anything to improve? e.g. 'that tone went on too long', 'too gritty', 'too loud'",
                                       label="", show_label=False, lines=6,
                                       max_lines=20, scale=6, min_width=0,
                                       elem_classes="jnote")
            c["log_btn"] = _apply(gr, "feedback", "LOG", scale=1, min_width=0)

    c["audio"] = ctx["audio_factory"]()
    c["timer"] = gr.Timer(value=ctx["timer_start"])
    return validate(c, "deck")
