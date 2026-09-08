"""s22_visuals.py -- the conductor's reactive visual: a lighting rig.

WHY THIS EXISTS (professors, 2026-08-07): a visual companion driven by the
sound AND by the numbers inside the conductor, added without disturbing the
mixer deck's layout. It ships as a TAB rather than a panel inside the deck --
the left column is CSS-clamped to 158px (`--leftw`, see s20_skins' overhang
note) and a ~150px visual would have been decorative rather than legible.

WHAT IT DRAWS, AND WHY IT IS NOT A TRAJECTORY (2026-08-08). The first
build traced the VA walk as a pen leaving marks. Rejected, correctly, on two
grounds: a moving point cannot FILL a frame, and following the walk is only
interesting to someone who already knows what the walk is. The brief instead:
stage-performance lighting -- criss-crossing light curtains, a vortex turning
in haze -- continuously morphing, with the parameter values glitching over the
top. Explicitly NOT particles or orbits, which are done to death.

So VA no longer positions anything. It drives GLOBAL properties of a full-frame
image: colour temperature, sweep speed, beam spread, haze density. Every pixel
is doing something at all times, which is what "fills the canvas" actually
requires.

  curtains-- three FOLDING sheets, |y - A*sin(kx + wt + fBm)|, offset vertically
             so they criss-cross. The fold is the point: a sinusoidal PLANE
             spans the frame where distance-from-a-line only ever gives a
             spotlight cone. They are dispersed like smoke rather than drawn as
             bands -- see the four-part note at the curtain loop.
  tunnel  -- a polar vortex with a 0.30/(r+0.075) depth term and a twist that
             grows with depth, so it reads as looking DOWN a funnel.
  haze    -- 3-octave fBm under a domain warp, advected by a WIND vector that
             the client integrates from valence, arousal and the live level.
             Curtains and tunnel are MULTIPLIED by it, which is the whole
             trick: light only shows where there is something to catch it, so
             the shafts break up and appear volumetric rather than drawn.
  flash   -- a decaying envelope fired on each transition, so a musical event
             is a lighting event.
  glitch  -- the live parameters in RGB-split type over the top, tearing with
             the audio level.

Nothing here is learned and nothing claims to be: an AUTHORED readout, the same
class as the VA pad. It must never be described as an emotion->image mapping --
that is the hand-authored mapping the Week-4 DSP decision and the design notes
rule out, and the rule does not stop applying because the output is pixels.

TWO CLOCKS, WHICH IS THE ARCHITECTURE
  * PARAMETERS arrive once per SEGMENT (22-42 s), pushed by the server into the
    data attributes of `#conductor-vis-data`. Data attributes, NOT a <script>
    carrier: gradio updates a component by writing innerHTML and a <script>
    inserted that way NEVER EXECUTES, which is why the deck's lamps are
    `<style>` carriers. The loop polls the dataset -- the same no-network
    technique as s21's fader `sync()`.
  * AUDIO is read every FRAME from `window.CONDUCTOR_AN`, the AnalyserNode
    s19_webaudio publishes. No graph exists on the plain <audio> path or when
    streaming to the sound card, so the visual then runs on parameters alone
    and SAYS SO rather than quietly looking dead (README SS22.7).

  Parameters are TWEENED between arrivals, frame-rate independently. Un-tweened
  they would step once per segment, putting a visible seam exactly where
  WebAudio has worked to remove the audible one.

SECOND SCREEN. `pop out` opens a bare window and the MAIN page renders into it;
it is not a second Gradio session. A second tab on the same URL would build its
own `gr.State` and start a rival conductor with its own audio. Here there is
one AudioContext, one session, one render loop, and the popup is blitted from
the same WebGL canvas -- which is why the context is created with
`preserveDrawingBuffer`.

The script rides `head=`, never a component's value, for the innerHTML reason
above; and never `Blocks(head=)`, which gradio 6 accepts and silently drops.
"""

VIS_HEAD = """
<style>
#conductor-vis-wrap {--vis-w:16; --vis-h:9;
  position:relative; background:#05070a; border:2px solid #2b2f37;
  border-radius:10px; overflow:hidden; width:100%;
  aspect-ratio:var(--vis-w) / var(--vis-h); min-height:320px;}
#conductor-vis-wrap:fullscreen {border-radius:0; border:0; aspect-ratio:auto;
  width:100vw; height:100vh; min-height:0;
  display:flex; align-items:center; justify-content:center;}
#conductor-vis-wrap:fullscreen #conductor-vis,
#conductor-vis-wrap:fullscreen #conductor-vis-ov {
  width:min(100vw, calc(100vh * var(--vis-w) / var(--vis-h)));
  height:min(100vh, calc(100vw * var(--vis-h) / var(--vis-w)));}
/* the overlay sits exactly on the gl canvas; both scale together */
#conductor-vis, #conductor-vis-ov {display:block; width:100%; height:100%;}
#conductor-vis-ov {position:absolute; left:50%; top:50%;
  transform:translate(-50%,-50%); pointer-events:none;}
#conductor-vis-ui {position:absolute; top:10px; right:10px; z-index:4;
  display:flex; gap:6px; align-items:center; flex-wrap:wrap;
  justify-content:flex-end; max-width:70%;}
#conductor-vis-ui button, #conductor-vis-ui label {
  background:#15181dcc; color:#8b919b; border:1px solid #2b2f37;
  border-radius:8px; padding:5px 9px; cursor:pointer; font:700 11px/1.1
  ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.08em;
  text-transform:uppercase;}
#conductor-vis-ui button:hover {color:#e5e7eb; border-color:#4b5563;}
#conductor-vis-ui label {display:flex; gap:6px; align-items:center; cursor:default;}
#conductor-vis-ui input[type=range] {width:66px; accent-color:#f59e0b;}
/* the mode select inherits the control vocabulary rather than the browser's
   default chrome, which is white and would be the brightest thing on screen
   in a design whose whole point is protected blacks. */
#conductor-vis-ui select {background:#0f1216; color:#c9ced6; border:1px solid #2b2f37;
  border-radius:5px; padding:2px 4px; cursor:pointer; font:700 11px/1.1
  ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.06em;}
#conductor-vis-ui select:focus {outline:1px solid #4b5563;}
#conductor-vis-src {position:absolute; left:12px; bottom:10px; z-index:4;
  font:11px ui-monospace,SFMono-Regular,Menlo,monospace; color:#4b5563;
  pointer-events:none; letter-spacing:.06em;}
</style>
<script>
(function () {
  if (window.__conductorVis) return;          // head can be injected more than once
  window.__conductorVis = 1;

  var NB = 12;                            // beams == fft bands fed to the rig
  // ---------------------------------------------------------------- BUDGET
  // 2026-08-09, feedback: "pretty much all of the visuals eat GPU to like 97%,
  // so i don't think it's just Tesla". Correct, and the cost is MODE-
  // INDEPENDENT -- it is pixels per second, not shader complexity. Measured
  // for a 1920x1080 fullscreen laptop:
  //   * devicePixelRatio 2 makes the GL buffer 3840x2160 = 8.3 Mpx A FRAME;
  //   * requestAnimationFrame runs at the DISPLAY rate, so a 120 Hz laptop
  //     panel was asking for 1.0 G fragment invocations/second;
  //   * and the overlay canvas was sized to MATCH the GL canvas, so a second
  //     8.3 Mpx surface was being cleared (and composited) every frame -- for
  //     text whose own design width is 1600.
  // Three caps below, and an adaptive scaler behind them.
  var FRAME_MS = 1000 / 36;   // ambient drone does not need 60, let alone 120
  var DPR_CAP  = 1.75;        // 2.0 costs 30% more pixels for a SOFT image
  var OV_MAX_W = 1600;        // the readout's own design width; see drawOverlay
  var SPEC_NB = 128, SPEC_NT = 128;       // spectrogram: bands x time slices
  var WAVE_N  = 256;                      // live waveform samples (tesla arcs)
  var SPEC_HZ = 8;                        // rows per second -> ~16 s of history
  var S = {
    t0: performance.now(), t: 0,
    v: 0, a: 0, tex: 0, chord: 'min', ramp: 0, std: 0, step: -1,
    lvl: 0, flash: 0, buf: null, bands: new Float32Array(NB),
    flux: 0, prev: null, period: 30,
    spec: null, specData: null, specRow: 0, specT: 0,
    glitch: 0, glitchAt: 6, params: true,
    vLo: 0, vHi: 0, vSeen: false, tintT: 0.5,
    // DEFAULT = MONOLITH, not curtains. a standing complaint about the
    // curtain scene is that it is "too busy ... not calming", and the brief is
    // to draw a listener INTO the sound rather than decorate it. The calm mode
    // should therefore be what you get on opening the tab; curtains is one
    // click away in the selector.
    mode: 3, progs: null, us: null,
    haze: 0.75, spread: 0.5, quality: 1.0,
    auto: 1.0, ivAvg: FRAME_MS, lastFrame: 0,   // adaptive render scale
    gate: 0,                                    // audio-present gate (sparks)
    wind: 0.6, disp: 0.75,                // sliders
    gust: 0, wdx: 0, wdy: 1, wspd: 0, woffx: 0, woffy: 0,
    pop: null, popCtx: null, gl: null, prog: null
  };
  window.__conductorVisState = S;

  function ds() {
    var e = document.getElementById('conductor-vis-data');
    return e ? e.dataset : {};
  }
  function num(d, k, dflt) {
    var x = parseFloat(d[k]);
    return isFinite(x) ? x : dflt;
  }

  // ---------------------------------------------------------------- shader
  var VS = 'attribute vec2 p;void main(){gl_Position=vec4(p,0.0,1.0);}';
  // THREE SEPARATE PROGRAMS, not one shader branching on a uMode uniform.
  // Register allocation is per-shader and sized for the worst path, so a
  // combined shader would make the cheap modes pay the curtain scene's
  // occupancy even while drawing a ring in fog -- which defeats the whole
  // reason for having them on a GPU that is already the binding constraint.
  // FS_HEAD + one SCENE_* + FS_TAIL are concatenated and linked separately.
  var FS_HEAD = [
  'precision highp float;',
  'uniform vec2  uRes;',
  'uniform float uTime, uV, uA, uLvl, uTex, uFlash, uHaze, uSpread;',
  'uniform vec2  uWindDir, uWindOff;',
  'uniform float uWindSpd, uDisp;',
  // uFlux = spectral flux, the ambient analogue of tempo (see the JS note).
  // uPeriod = the arranger's own breath period, so the swirl can pulse at a
  // rate the ear is already tracking rather than at an invented one.
  'uniform float uFlux, uPeriod;',
  // uTint is valence ALREADY normalised to the range the walk is actually
  // using, computed client-side -- see the auto-range note in the loop.
  'uniform float uTint;',
  // THE SPECTRUM AS A TEXTURE, not a uniform array: an array is capped (uB is
  // 12) and in WebGL1 can only be indexed by a loop counter, whereas a texture
  // holds the full 128-band history and is sampled anywhere. x = band,
  // y = time; uSpecRow is where the newest column was written.
  'uniform sampler2D uSpec;',
  'uniform float uSpecRow;',
  // THE LIVE WAVEFORM, time-domain, from the SAME AnalyserNode as the
  // spectrum. SS3's rule forbids a SECOND AnalyserNode -- it does not forbid a
  // second read off the one we have, which costs nothing extra.
  'uniform sampler2D uWave;',
  // uGate = "there is actually sound coming out right now", 0..1, computed
  // client-side where BOTH facts are known: whether an analyser exists at all,
  // and what the live level is. A shader cannot tell silence from a missing
  // audio graph -- the no-graph fallback synthesises a level of ~0.18, which
  // would have sparked merrily over nothing.
  'uniform float uGate;',
  'uniform float uB[12];',
  'float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}',
  // QUINTIC fade, not cubic (2026-08-08, feedback: "sharp-edged floating shapes"
  // when haze is raised). f*f*(3-2f) is only C1 -- its SECOND derivative jumps
  // at every cell corner, and value noise shows that as straight creases along
  // the sampling grid: the blockiness v4c's gradient noise was going to fix at
  // twice the cost. Perlin's quintic f*f*f*(f*(6f-15)+10) is C2, which removes
  // the creases for about two extra multiplies per sample.
  'float vnoise(vec2 p){vec2 i=floor(p),f=fract(p);',
  '  f=f*f*f*(f*(f*6.0-15.0)+10.0);',
  '  float a=hash(i),b=hash(i+vec2(1.0,0.0)),c=hash(i+vec2(0.0,1.0)),d=hash(i+vec2(1.0,1.0));',
  '  return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);}',
  // PER-OCTAVE ROTATION. Without it every octave's grid is axis-aligned with
  // every other one, so the creases REINFORCE instead of cancelling and the
  // whole fBm inherits one square lattice. Rotating each octave costs four
  // multiplies and is the other half of the blockiness cure. Declared inside
  // the function -- a global `const mat2` is legal GLSL ES 1.0 but some
  // drivers are fussy, and this cannot be compiled here to find out.
  'float fbm2(vec2 p){mat2 R=mat2(0.80,0.60,-0.60,0.80);',
  '  float s=0.0,m=0.5;for(int i=0;i<2;i++){s+=m*vnoise(p);p=R*p*2.07;m*=0.5;}return s;}',
  'float fbm3(vec2 p){mat2 R=mat2(0.80,0.60,-0.60,0.80);',
  '  float s=0.0,m=0.5;for(int i=0;i<3;i++){s+=m*vnoise(p);p=R*p*2.03;m*=0.5;}return s;}',
  // DUOTONE, shared by every mode. Deep slate/indigo at negative valence, warm
  // tungsten/amber at positive. Full-spectrum HSV cycling is what makes this
  // look like a 2000s music visualiser, so there is one palette and no hue.
  'vec3 tintc(){return mix(vec3(0.13,0.32,0.62),vec3(1.00,0.58,0.20),',
  '                        clamp(uTint,0.0,1.0));}',
  // TRIANGLE, NOT SINE. A sine is smooth everywhere, which is exactly why a
  // sine-summed filament reads as a wave rather than as a discharge. A
  // triangle is piecewise linear, so summing octaves of it gives CORNERS at
  // every scale -- the closed-form stand-in for the fractal midpoint
  // displacement that lightning renderers use, at four cheap ops.
  'float tri(float x){return abs(fract(x)*2.0-1.0)*2.0-1.0;}',
  ''].join('\\n');

  // ------------------------------------------------ mode 1: MONOLITH/APERTURE
  // Anthony McCall's solid light and James Turrell's apertures: ONE luminous
  // volume in deep fog and nothing else in frame. Removing everything else is
  // what stops the eye scanning and lets attention go to the timbre -- the
  // "immersive rather than reactive" pole (VISUALS.md 7.1). Cheapest of the
  // fog modes at ~5 octaves a pixel against the curtain scene's ~14.
  var SCENE_MONOLITH = [
  'vec3 scene(vec2 uv,float T){',
  '  vec3 tint=tintc();',
  '  float energy=0.35+(uA+1.0)*0.55;',
  '  vec2 W=uWindOff;',
  '  vec2 c=vec2(sin(T*0.021)*0.030,cos(T*0.017)*0.022);',
  '  vec2 pv=uv-c;',
  '  float r=length(pv);',
  '  float br=sin(T*6.2831/max(uPeriod,4.0));',
  // A CIRCLE: the radius is the SAME at every angle. No per-angle spectrum
  // displacement, no noise wobble -- both were tried to answer "the ring just
  // sits there" and both destroyed the one clean shape that makes this work.
  // Sound still moves it: the WHOLE circle expands and contracts together.
  '  float R=(0.24+0.16*uSpread)*(1.0+0.16*br+0.38*uLvl+uFlash*0.10);',
  '  float th=(0.012+0.030*uDisp+0.010*uTex)*(0.85+0.85*uLvl);',
  '  float de=abs(r-R);',
  // A BLACK HOLE HAS NO BRIGHT CENTRE (feedback: "like falling into a black
  // hole for eternity"). Light is strongest just inside the rim and falls to
  // NOTHING at the middle, so the eye is given a void to fall into rather than
  // a lamp to look at. The old fill was brightest dead centre -- the exact
  // opposite, and why the middle read as a light source.
  '  float di=smoothstep(0.0,R*0.92,r)*smoothstep(R*1.04,R*0.96,r)',
  '          *(0.34+0.34*uLvl);',
  '  float rim=exp(-de*de/(th*th));',
  '  rim=mix(rim,th*th/(th*th+de*de*6.0),0.55);',
  '  rim*=0.90+0.75*uLvl;',
  // SPIN WITHOUT DEFORMING (2026-08-08, feedback: "make the ring feel like it's
  // spinning but keep it a ring"). A circle is rotationally symmetric, so it
  // CANNOT be seen to rotate -- the only way is to break the symmetry without
  // breaking the circle. The radius stays uniform and the BRIGHTNESS travels
  // around the rim instead: a broad bright side, plus a tight comet head that
  // gives the rotation a direction and a readable rate. Geometry untouched.
  '  float aang=atan(pv.y,pv.x);',
  '  float sp=T*(0.10+0.55*uFlux)*energy;',
  '  float u1=aang+sp;',
  '  float arc=0.62+0.38*cos(u1);',
  '  float comet=pow(max(0.5+0.5*cos(u1-0.55),0.0),9.0);',
  '  rim*=arc+comet*1.10;',
  // DUST BETWEEN THE LIGHT AND THE VIEWER (2026-08-08, feedback: the spiral was
  // "a static swirl inside the ring ... looks like a latte"). It was: a
  // logarithmic spiral is a fixed curve, and rotating a fixed curve reads as
  // latte art, not as atmosphere. Concentric echo rings had the same problem
  // from the other side -- perfect circles inside a circle. Both are gone.
  //   What replaces them is volumetric: four layers of sparse motes at
  // different DEPTHS, each drifting its own way, hanging in the air between
  // the aperture and the viewer. Irregular by construction, and no two layers
  // agree on direction, so nothing in the frame is trackable or repeating.
  //   THE APPROACH IS THE POINT. Each layer's scale shrinks over its cycle, so
  // its motes expand outward from the centre and sweep past you -- the optical
  // flow of moving forward. It is deliberately slow enough not to be noticed
  // as motion: you should register that you are falling into the light without
  // being able to point at what is moving. Each layer fades in and out across
  // its cycle so there is no pop when it recycles.
  // DUST THAT OCCLUDES, NOT DUST THAT GLOWS (2026-08-08, feedback: "more should
  // block the light rather than reflect or generate"). Physically right, and it
  // is why the earlier version read as a snow globe however small the motes
  // got: ADDITIVE specks are snow; only SUBTRACTIVE ones are dust. The interior
  // light is now multiplied DOWN where a mote sits, so motes are dark
  // silhouettes against the lit aperture -- backlit particulate, which is what
  // you actually see in a haze-filled beam.
  //   ONE MOTE PER CELL, at a hashed position, so the primitive is a point.
  // Thresholded value noise -- the version before that -- can only ever make
  // axis-aligned rectangles.
  //   MOTION IS PURELY RADIAL. The lateral drift term that was here made the
  // motes meander, and a meander is trackable and therefore noticeable. Each
  // layer's scale only shrinks across its cycle, so every mote moves straight
  // outward from the centre and past the viewer -- the optical flow of moving
  // forward, and nothing else. ~80 s a cycle, slow enough not to register.
  '  float dust=0.0;',
  '  for(int L=0;L<4;L++){',
  '    float fl=float(L);',
  '    float pe2=fract(T*(0.0125+0.004*fl)*energy+fl*0.25);',
  '    float sc=mix(2.6,0.9,pe2);',
  '    vec2 dp=uv*sc*30.0+fl*37.0;',
  '    vec2 ip=floor(dp), fp=fract(dp);',
  '    vec2 sd=ip+fl*11.0;',
  '    vec2 jit=vec2(hash(sd),hash(sd+vec2(31.7,17.3)));',
  '    float dd=length(fp-jit);',
  '    float rn=hash(sd+vec2(5.1,9.7));',
  '    float on=step(0.90,rn);',
  '    float rad2=1700.0+2800.0*fract(rn*37.0);',
  '    dust+=exp(-dd*dd*rad2)*on*sin(pe2*3.1416);',
  '  }',
  // DUST IS LIT BY THE RING -- not self-luminous, and not an occluder either.
  // Occlusion was the right instinct and became the wrong mechanism the moment
  // the interior went dark: a silhouette has nothing to take away from black,
  // which is exactly why it could not be seen. The motes now CATCH the ring's
  // light -- brightest just inside the rim, fading to nothing at the dead
  // centre. Deliberately faint: the outward drift should be SENSED, never
  // watched, and nothing here should be pointable at.
  '  float lit0=di;',
  // occlusion capped so a mote DIMS the light rather than punching a hole in
  // it, and masked to r<R so nothing shows in the outer halo.
  '  float inmask=smoothstep(R,R*0.92,r);',
  '  float cone=di+dust*lit0*inmask*0.90;',
  // FOG REMOVED 2026-08-08 (feedback: "remove those annoying floating
  // shapes from all modes but the vortex"). The domain-warped fBm here
  // was the source: at this frequency its cells are a large fraction of
  // the frame, so it never reads as atmosphere, only as a few drifting
  // patches. The aperture does not need a medium -- it is a light in a
  // void, and the void should be empty. Cheaper too: this was 5 of the
  // mode's ~7 octaves. The vortex keeps its haze, where the multiply is
  // the only thing making its shafts volumetric.
  '  float lit=rim*1.7+cone;',
  '  vec3 col=tint*lit;',
  '  col*=1.0+uFlux*0.55*energy;',
  '  return col;',
  '}'].join('\\n');

  // ---------------------------------------------- mode 4: RING + COLOUR FIELD
  // the brief's own suggestion, 2026-08-08: "combining the monolithic ring with
  // the colour field". The ring stays EXACTLY the circle mode 1 draws -- no
  // deformation, nothing keyed to angle -- and the colour field replaces the
  // fbm fog behind it. The field gives the frame slow colour and motion; the
  // ring gives it the one hard object that the field alone lacks. The ring is
  // pushed toward white so it reads as a light SOURCE in front of a coloured
  // volume, rather than as one more coloured region competing with it.
  var SCENE_RINGFIELD = [
  'vec3 scene(vec2 uv,float T){',
  '  vec3 cool=vec3(0.13,0.32,0.62), warm=vec3(1.00,0.58,0.20);',
  '  float v01=clamp(uTint,0.0,1.0);',
  '  float energy=0.35+(uA+1.0)*0.55;',
  '  vec3 bg=vec3(0.0);',
  '  for(int i=0;i<3;i++){',
  '    float fi=float(i);',
  '    float rt=0.055+fi*0.026+fi*fi*0.018;',
  '    float ph=T*rt*energy;',
  '    vec2 cc=vec2(sin(ph*1.7+fi*2.1)*0.42,cos(ph*1.31+fi*1.1)*0.30);',
  '    vec2 sz=vec2(0.85+0.35*fi,0.55+0.20*fi)*(1.0+0.35*uSpread);',
  '    vec2 wv=vec2(sin(uv.y*2.3+T*0.050+fi*2.1),',
  '                 sin(uv.x*1.9-T*0.041+fi*3.3))*(0.10+0.16*uDisp);',
  '    vec2 dd=(uv-cc+wv)/sz;',
  '    float f=exp(-dot(dd,dd)*(1.05+1.5*(1.0-uDisp)));',
  '    float t=clamp(v01+(fi-1.0)*0.22+0.10*sin(ph*0.9),0.0,1.0);',
  '    float bb=0.5+0.5*sin(T*6.2831/(max(uPeriod,4.0)*(1.0+0.61*fi))+fi*2.09);',
  '    float drv=fi<0.5?uLvl:(fi<1.5?uFlux*0.8:0.34);',
  '    bg+=mix(cool,warm,t)*f*(0.42+0.45*bb)*(0.55+0.95*drv);',
  '  }',
  '  vec2 c=vec2(sin(T*0.021)*0.030,cos(T*0.017)*0.022);',
  '  vec2 pv=uv-c;',
  '  float r=length(pv);',
  '  float br=sin(T*6.2831/max(uPeriod,4.0));',
  // identical to mode 1: radius the same at every angle
  '  float R=(0.24+0.16*uSpread)*(1.0+0.16*br+0.38*uLvl+uFlash*0.10);',
  '  float th=(0.012+0.030*uDisp+0.010*uTex)*(0.85+0.85*uLvl);',
  '  float de=abs(r-R);',
  '  float rim=exp(-de*de/(th*th));',
  '  rim=mix(rim,th*th/(th*th+de*de*6.0),0.55);',
  '  rim*=0.90+0.75*uLvl;',
  // THE MONOLITH RING EXACTLY, minus its interior and its dust (the brief,
  // 2026-08-08). Same rotating brightness: the radius is uniform and only
  // the luminance travels around the rim -- a broad bright side for the
  // rotation, a tight comet head for its direction and rate. The interior
  // fill and the centre glow are dropped on purpose, because here the
  // colour field is what occupies the inside of the ring; filling it as
  // well would just wash the field out behind a disc of light.
  '  float aang=atan(pv.y,pv.x);',
  '  float sp=T*(0.10+0.55*uFlux)*energy;',
  '  float u1=aang+sp;',
  '  float arc=0.62+0.38*cos(u1);',
  '  float comet=pow(max(0.5+0.5*cos(u1-0.55),0.0),9.0);',
  '  rim*=arc+comet*1.10;',
  '  vec3 rc=mix(mix(cool,warm,v01),vec3(1.0),0.45);',
  '  vec3 col=bg*mix(1.0,0.72,uHaze)+rc*rim*1.5;',
  '  return col;',
  '}'].join('\\n');

  // ------------------------------------------------------- mode 4: SPECTRUM
  // The FFT as the OBJECT, not as a driver. Freeman's register is primitive
  // forms in environments that could not exist physically, plus a deliberate
  // choice of WHICH data drives what -- so this takes the whole 1024-bin
  // spectrum at its new resolution rather than twelve smoothed bands, and
  // shows it as what it is.
  //   A SPECTROGRAM: frequency across, TIME down. Every other mode shows only
  // the present instant; this is the only one where the piece's recent past is
  // visible, which for drone -- whose subject is slow evolution -- is arguably
  // the honest view. Vertical time made literal: the frame IS the last ~16 s.
  // Bands are LOG-SPACED across x, so a semitone is the same width everywhere.
  var SCENE_SPECTRUM = [
  'vec3 scene(vec2 uv,float T){',
  '  vec3 tint=tintc();',
  '  vec2 q=uv*vec2(0.55,0.5)+0.5;',
  '  if(q.x<0.0||q.x>1.0||q.y<0.0||q.y>1.0) return vec3(0.0);',
  '  float age=1.0-q.y;',
  '  float row=fract(uSpecRow-age);',
  '  float e=texture2D(uSpec,vec2(q.x,row)).x;',
  '  e=pow(clamp(e*(1.0+2.0*uSpread),0.0,1.0),1.0+2.2*(1.0-uDisp));',
  '  float steps=mix(64.0,6.0,uHaze);',
  '  e=mix(e,floor(e*steps)/steps,0.75);',
  '  float edge=smoothstep(0.10,0.0,age)*0.55;',
  '  vec3 col=tint*e*(1.6+edge*2.0);',
  '  float g1=smoothstep(0.985,1.0,fract(q.x*12.0));',
  '  col+=tint*g1*0.05;',
  '  col*=1.0+uFlash*0.8;',
  '  return col;',
  '}'].join('\\n');

  // ------------------------------------------- mode 5: MONOLITH (LIT)
  // The ring as it stood BEFORE the black-hole rework, kept on the brief's
  // request so the two can be compared rather than one replacing the other.
  // Same circle, same rotating rim; what differs is entirely INSIDE it:
  //   lit    - interior brightest toward the centre, a soft core glow, and
  //            dust that OCCLUDES that light (dark motes on a lit field)
  //   void   - interior dark at the centre and brightest just inside the
  //            rim, with dust CATCHING the ring's light instead
  // They are different pictures, not different tunings, which is why this
  // is a second scene rather than a slider. This one is the default.
  var SCENE_MONOLITH_LIT = [
  'vec3 scene(vec2 uv,float T){',
  '  vec3 tint=tintc();',
  '  float energy=0.35+(uA+1.0)*0.55;',
  '  vec2 W=uWindOff;',
  '  vec2 c=vec2(sin(T*0.021)*0.030,cos(T*0.017)*0.022);',
  '  vec2 pv=uv-c;',
  '  float r=length(pv);',
  '  float br=sin(T*6.2831/max(uPeriod,4.0));',
  // A CIRCLE: the radius is the SAME at every angle. No per-angle spectrum
  // displacement, no noise wobble -- both were tried to answer "the ring just
  // sits there" and both destroyed the one clean shape that makes this work.
  // Sound still moves it: the WHOLE circle expands and contracts together.
  '  float R=(0.24+0.16*uSpread)*(1.0+0.16*br+0.38*uLvl+uFlash*0.10);',
  '  float th=(0.012+0.030*uDisp+0.010*uTex)*(0.85+0.85*uLvl);',
  '  float de=abs(r-R);',
  '  float di=smoothstep(R*1.30,0.0,r)*(0.20+0.40*uLvl);',
  '  float rim=exp(-de*de/(th*th));',
  '  rim=mix(rim,th*th/(th*th+de*de*6.0),0.55);',
  '  rim*=0.90+0.75*uLvl;',
  // SPIN WITHOUT DEFORMING (2026-08-08, feedback: "make the ring feel like it's
  // spinning but keep it a ring"). A circle is rotationally symmetric, so it
  // CANNOT be seen to rotate -- the only way is to break the symmetry without
  // breaking the circle. The radius stays uniform and the BRIGHTNESS travels
  // around the rim instead: a broad bright side, plus a tight comet head that
  // gives the rotation a direction and a readable rate. Geometry untouched.
  '  float aang=atan(pv.y,pv.x);',
  '  float sp=T*(0.10+0.55*uFlux)*energy;',
  '  float u1=aang+sp;',
  '  float arc=0.62+0.38*cos(u1);',
  '  float comet=pow(max(0.5+0.5*cos(u1-0.55),0.0),9.0);',
  '  rim*=arc+comet*1.10;',
  // DUST BETWEEN THE LIGHT AND THE VIEWER (2026-08-08, feedback: the spiral was
  // "a static swirl inside the ring ... looks like a latte"). It was: a
  // logarithmic spiral is a fixed curve, and rotating a fixed curve reads as
  // latte art, not as atmosphere. Concentric echo rings had the same problem
  // from the other side -- perfect circles inside a circle. Both are gone.
  //   What replaces them is volumetric: four layers of sparse motes at
  // different DEPTHS, each drifting its own way, hanging in the air between
  // the aperture and the viewer. Irregular by construction, and no two layers
  // agree on direction, so nothing in the frame is trackable or repeating.
  //   THE APPROACH IS THE POINT. Each layer's scale shrinks over its cycle, so
  // its motes expand outward from the centre and sweep past you -- the optical
  // flow of moving forward. It is deliberately slow enough not to be noticed
  // as motion: you should register that you are falling into the light without
  // being able to point at what is moving. Each layer fades in and out across
  // its cycle so there is no pop when it recycles.
  // DUST THAT OCCLUDES, NOT DUST THAT GLOWS (2026-08-08, feedback: "more should
  // block the light rather than reflect or generate"). Physically right, and it
  // is why the earlier version read as a snow globe however small the motes
  // got: ADDITIVE specks are snow; only SUBTRACTIVE ones are dust. The interior
  // light is now multiplied DOWN where a mote sits, so motes are dark
  // silhouettes against the lit aperture -- backlit particulate, which is what
  // you actually see in a haze-filled beam.
  //   ONE MOTE PER CELL, at a hashed position, so the primitive is a point.
  // Thresholded value noise -- the version before that -- can only ever make
  // axis-aligned rectangles.
  //   MOTION IS PURELY RADIAL. The lateral drift term that was here made the
  // motes meander, and a meander is trackable and therefore noticeable. Each
  // layer's scale only shrinks across its cycle, so every mote moves straight
  // outward from the centre and past the viewer -- the optical flow of moving
  // forward, and nothing else. ~80 s a cycle, slow enough not to register.
  '  float dust=0.0;',
  '  for(int L=0;L<4;L++){',
  '    float fl=float(L);',
  '    float pe2=fract(T*(0.0125+0.004*fl)*energy+fl*0.25);',
  '    float sc=mix(2.6,0.9,pe2);',
  '    vec2 dp=uv*sc*30.0+fl*37.0;',
  '    vec2 ip=floor(dp), fp=fract(dp);',
  '    vec2 sd=ip+fl*11.0;',
  '    vec2 jit=vec2(hash(sd),hash(sd+vec2(31.7,17.3)));',
  '    float dd=length(fp-jit);',
  '    float rn=hash(sd+vec2(5.1,9.7));',
  '    float on=step(0.90,rn);',
  '    float rad2=1700.0+2800.0*fract(rn*37.0);',
  '    dust+=exp(-dd*dd*rad2)*on*sin(pe2*3.1416);',
  '  }',
  '  float core=exp(-r*r*26.0);',
  // occlusion capped so a mote DIMS the light rather than punching a hole in
  // it, and masked to r<R so nothing shows in the outer halo.
  '  float inmask=smoothstep(R,R*0.90,r);',
  '  float occ=clamp(dust*inmask*1.30,0.0,0.80);',
  '  float cone=(di+core*(0.10+0.25*uLvl))*(1.0-occ);',
  // FOG REMOVED 2026-08-08 (feedback: "remove those annoying floating
  // shapes from all modes but the vortex"). The domain-warped fBm here
  // was the source: at this frequency its cells are a large fraction of
  // the frame, so it never reads as atmosphere, only as a few drifting
  // patches. The aperture does not need a medium -- it is a light in a
  // void, and the void should be empty. Cheaper too: this was 5 of the
  // mode's ~7 octaves. The vortex keeps its haze, where the multiply is
  // the only thing making its shafts volumetric.
  '  float lit=rim*1.7+cone;',
  '  vec3 col=tint*lit;',
  '  col*=1.0+uFlux*0.55*energy;',
  '  return col;',
  '}'].join('\\n');

  // -------------------------------------------------------- mode 5: CORONA
  // Rebuilt AGAIN 2026-08-08 from the second reference (SDO/AIA 171A), which
  // is a far more contained picture than the first: pure black around it, a
  // BRIGHT THIN LIMB, a DARK mottled interior, and only small coronal tufts
  // just outside the edge. The brief's ratio -- "75% ring and 25% corona" --
  // is that image described in one line, and it makes the limb the subject
  // rather than the streamers. The previous version filled the frame with
  // rays and blazing red; this one is a ring that happens to be a star.
  //   THE SURFACE ROTATES, which is the fix for "not moving at all". The old
  // version translated its noise, and translation reads as SLIDING -- a
  // texture being dragged across a shape. Sampling in real longitude and
  // latitude instead means features cross the disc, compress as they approach
  // the limb and disappear around it, which is what makes a sphere look like
  // it is turning rather than smeared.
  var SCENE_CORONA = [
  'vec3 scene(vec2 uv,float T){',
  '  vec3 tint=tintc();',
  '  float energy=0.35+(uA+1.0)*0.55;',
  '  float rr=length(uv)+1e-4;',
  '  float th=atan(uv.y,uv.x);',
  // SEAM-FREE ANGULAR SAMPLING. atan returns [-pi,pi] and jumps by 2*pi
  // across the NEGATIVE X AXIS, so noise sampled on the raw angle is
  // discontinuous down the LEFT-HAND SIDE -- exactly the seam the brief saw,
  // in this mode and the vortex both. Wrapping the angle does not help: the
  // noise FIELD has to be periodic, not the coordinate.
  //   So stop using the angle as a coordinate and sample the 2D noise ON A
  // CIRCLE. `dir` is the unit vector, so one trip round returns to the same
  // point and continuity is structural rather than arranged. It is also
  // FREE -- dir = uv/rr needs no trig, where the old path paid for an atan.
  // Radius k gives ~2*pi*k features around, which is how the old angular
  // frequencies carry over: th*7 -> k 1.11, th*34 -> 5.40, th*90 -> 14.3.
  '  vec2 dir=uv/rr;',
  // ISOTROPY. Circular sampling cured the atan seam, but the radial and time
  // terms were then added as vec2(x) -- which is (x,x), a vector along 45
  // degrees. That offset is COLLINEAR with dir at exactly 45 and -135, so
  // there the two contributions stop decorrelating and the field degenerates
  // into a streak: one diagonal seam through the centre, which is what
  // the brief measured. ANY fixed offset direction does this somewhere.
  //   The cure is to offset along the TANGENT instead. dir*k + prp*a has
  // angle th + atan(a/k) and radius sqrt(k*k+a*a), so a radial or time term
  // becomes a rotation plus a slight scale -- both functions of radius alone,
  // so no direction is special. It also costs nothing: prp is a swizzle, not
  // a sin/cos pair.
  '  vec2 prp=vec2(-dir.y,dir.x);',
  '  float R=0.30+0.10*uSpread;',
  // SPHERE COORDINATES. asin gives the true angle, so the foreshortening near
  // the limb is correct rather than approximated -- and it costs two calls.
  // the rough limb is computed HERE, above the surface, because the disc's
  // own edge is cut at Rl -- a clean arc there would show straight through
  // the roughened rim drawn on top of it.
  '  float lw=vnoise(dir*1.11+prp*(T*0.10));',
  '  float lf=vnoise(dir*5.40+prp*(-T*0.25)+3.7);',
  '  float Rl=R*(1.0+0.011*(lw-0.5)+0.006*(lf-0.5));',
  '  vec2 sp=uv/R;',
  '  float zz=sqrt(max(1.0-dot(sp,sp),0.0));',
  '  float lat=asin(clamp(sp.y,-1.0,1.0));',
  '  float clat=max(cos(lat),0.10);',
  '  float lon=asin(clamp(sp.x/clat,-1.0,1.0))+T*0.017;',
  // two scales, and each also EVOLVES in place: the slow term drifts in
  // latitude, the fine one in longitude, so the pattern is never simply the
  // same field at a different offset.
  '  float g1=fbm3(vec2(lon*2.6,lat*2.6+T*0.006));',
  '  float g2=fbm2(vec2(lon*6.2+T*0.004,lat*6.2));',
  '  float surf=0.55*g1+0.38*g2;',
  '  surf*=1.0-0.45*smoothstep(0.55,0.28,g1);',
  '  float disc=smoothstep(Rl*1.005,Rl*0.988,rr);',
  // INTERIOR IS DARK. In the reference the disc is dimmer than its own edge;
  // making it bright is what turned the last version into a lamp.
  '  float body=disc*(0.08+0.30*surf)*(0.45+0.55*zz);',
  // ---- THE RING: a thin bright limb. This is the 75%.
  // A PERFECT CIRCLE IS WHY IT LOOKED SOLID (2026-08-08, feedback: "what's with
  // the solid ring in corona mode, that's not organic"). Constant radius,
  // constant thickness and constant brightness all the way round is a drawn
  // circle -- and unlike the monolith, where that uniformity IS the subject, a
  // star's limb is a rough chromosphere. Three things vary, all in angle:
  //   radius     +-1% -- enough to lose the compass edge, not enough to lose
  //              the circle
  //   thickness  from the fine field, so the edge thins and thickens
  //   brightness from the coarse field, so it has bright and dead stretches
  // Two noises, one slow and one at spicule scale, both scrolling round at
  // different rates so the pattern never sits still.
  '  float tw=(0.010+0.008*uDisp)*(0.90+0.50*uLvl)*(0.70+0.95*lf);',
  // BLACK SUN (BLCK SUN -- AMIANGELIKA x 1100). The limb as thousands of fine
  // radial hairs rather than a smooth glow. GATED ON uHaze, which no other line
  // in this scene reads, so haze 0 reproduces the previous corona EXACTLY and
  // the slider is a live A/B -- the same discipline `disperse` got for the
  // smoke work, rather than replacing a look outright and hoping.
  '  float bs=clamp(uHaze,0.0,1.0);',
  // the soft glow narrows as the fringe takes over: two full-strength rim
  // treatments stacked read as a halo, which is the thing being replaced.
  '  tw*=1.0-0.45*bs;',
  '  float de=abs(rr-Rl);',
  '  float rim=exp(-de*de/(tw*tw));',
  '  rim=mix(rim,tw*tw/(tw*tw+de*de*5.0),0.50);',
  '  rim*=0.45+1.15*lw;',
  // the smooth limb STEPS ASIDE rather than merely narrowing. In the reference
  // there is no glow at all -- the rim IS the filaments -- and a gaussian left
  // underneath them is what keeps reading as a halo with hairs on it.
  '  rim*=1.0-0.62*bs;',
  // SPICULES: the fine radial hairs standing just off the limb. This is what
  // actually sells a stellar edge -- a rough line still reads as a line, a
  // fringe reads as a surface seen edge-on.
  '  float spic=pow(max(vnoise(dir*14.3+prp*(rr*18.0-T*3.0))-0.45,0.0),1.5)*3.2;',
  '  spic*=smoothstep(R*0.995,R*1.012,rr)*smoothstep(R*1.10,R*1.006,rr);',
  '  rim+=spic*0.55*(1.0-0.60*bs);',
  // NO LINE INTEGRAL CONVOLUTION. The usual way to get filaments is to step
  // 20-40 times along a flow field per pixel, which is worse than the ~14
  // octaves that already pinned this GPU at 97% (SS6.12.1). It is unnecessary
  // here because the flow is RADIAL and known in closed form: sampling noise
  // that varies FAST around the circle and SLOWLY with radius stretches its
  // features into hairs for ONE sample. dir/prp keep it seam-free -- see the
  // isotropy note at the top of this scene, which both terms below obey.
  //   hk is the hair COUNT: dir*k gives ~2*pi*k strands round the limb, so at
  // corona's default disperse the fine octave is ~790 strands. Screen width per
  // strand is r_px/k, so raising disperse thins them TOWARD ONE PIXEL and they
  // will shimmer -- that is the ceiling on this effect and the reason these
  // coefficients sit below the reference's apparent density.
  '  float hk=38.0+52.0*uDisp;',
  // the radial coefficients are LOW on purpose: they set how far a strand
  // wanders sideways as it travels out, so a large one curls the hairs into
  // tufts. The reference's are near-straight spikes, which needs these small.
  '  float h1=vnoise(dir*hk+prp*(rr*3.2-T*0.22));',
  '  float h2=vnoise(dir*(hk*2.05)+prp*(rr*4.6+T*0.15)+11.3);',
  '  float hair=0.62*h1+0.38*h2;',
  // threshold-and-power SEPARATES the strands. Without it this is a smear: a
  // fringe needs gaps between hairs, not a rough gradient.
  '  hair=pow(max(hair-0.42,0.0)*1.75,1.6);',
  // reach further out, and TAPER rather than stop: a hard outer cut gives the
  // fringe a second circular edge, which is the one thing it must not have.
  '  float hEnv=smoothstep(Rl*0.988,Rl*1.004,rr)*smoothstep(Rl*1.34,Rl*1.010,rr);',
  '  hEnv=pow(hEnv,1.35);',
  // BEADS. In the reference the limb is not evenly lit -- a few points blaze
  // and the rest is faint, and that unevenness is what stops the fringe reading
  // as a uniform brush. Low angular frequency, thresholded hard, drifting.
  '  float bead=vnoise(dir*3.3+prp*(T*0.045));',
  // harder threshold and more gain: in the reference the beads are BLOWN OUT
  // white while the limb between them is nearly invisible, and that range is
  // most of why it reads as violent rather than decorative.
  '  bead=pow(max(bead-0.60,0.0)*2.6,2.0);',
  '  float fringe=hair*hEnv*(0.55+3.2*bead);',
  '  fringe*=0.75+1.10*uLvl;',
  '  rim+=fringe*3.2*bs;',
  '  rim*=1.0+bead*3.0*bs;',
  // ---- CORONA: short tufts hugging the limb, not rays crossing the frame.
  // The outer smoothstep is what keeps it CONTAINED -- the previous version
  // had none, so the streamers ran to the corners and dominated everything.
  '  float lr=log(rr+0.02);',
  // the radial term is kept SMALL on purpose: a ray has to stay near-constant
  // along r, and a large one would turn the streaks back into blobs.
  '  float s1=fbm2(dir*0.80+prp*(lr*0.30-T*0.05));',
  '  float tuft=pow(max(s1-0.30,0.0),1.4)*2.6;',
  '  float ang=fract(th/6.2831+0.5+T*0.004);',
  '  float sh=0.0,ws=0.0;',
  '  for(int b=0;b<12;b++){',
  '    float bx=(float(b)+0.5)/12.0;',
  '    float dd=abs(ang-bx); dd=min(dd,1.0-dd);',
  '    float gw=exp(-pow(dd*10.0,2.0));',
  '    sh+=uB[b]*gw; ws+=gw;',
  '  }',
  '  sh/=max(ws,1e-3);',
  '  tuft*=0.45+1.5*sh;',
  '  tuft*=smoothstep(R*0.99,R*1.04,rr)*smoothstep(R*1.85,R*1.03,rr);',
  // ---- ACTIVE REGIONS, OCCASIONAL. They were on constantly, which is what
  // made the flares read as decoration. Each is gated by its own slow cycle
  // -- mostly dark, briefly alight -- on incommensurate rates so they never
  // fire together, and uFlash overrides the gate so a real transition still
  // lights one.
  // ---- ERUPTING PROMINENCES (2026-08-08, feedback: "static ... should move and
  // curl and erupt and throttle like real solar flares but react to the
  // music"). A gaussian blob fading in and out cannot do any of that, so the
  // flare is now a PATH rather than a point.
  //   THE CURL IS THE WHOLE THING. A real prominence rises radially while
  //   shearing sideways, because the footpoints rotate at different rates --
  //   so the expected angle at height h is a0 + curl*h^2, and a pixel's
  //   distance from THAT is what makes the plume. The h^2 means the shear
  //   accelerates with height, which is what gives the hook at the top; linear
  //   shear just leans.
  //   ERUPTION IS FAST-RISE, SLOW-DECAY, and mostly dormant in between -- a
  //   flare that is always present is a decoration. Each has its own long
  //   period (17-30 s) so they never fire together, and uFlash erupts one on a
  //   real transition.
  //   THE MUSIC SETS HEIGHT AND CURL, not just brightness: a loud band throws
  //   its plume further out and twists it harder, so the shape answers the
  //   sound rather than the shape being lit by it.
  '  float act=0.0;',
  '  for(int k=0;k<3;k++){',
  '    float fk=float(k);',
  '    float ph2=fract((T+fk*7.9)/(17.0+fk*6.3));',
  '    float life=smoothstep(0.0,0.05,ph2)*smoothstep(0.60,0.12,ph2);',
  '    life=max(life,uFlash*0.9);',
  '    float a0=fk*2.4+T*0.013;',
  '    float bk=uB[k];',
  '    float H=0.40+0.75*bk;',
  '    float hh=(rr-R)/(R*H);',
  '    float curl=(0.8+2.0*bk)*(fk<1.5?1.0:-1.0);',
  '    float da=th-(a0+curl*hh*hh*0.65);',
  '    da-=6.2831*floor(da/6.2831+0.5);',
  '    float latd=da*rr;',
  '    float wd=R*(0.045+0.11*max(hh,0.0));',
  '    float core=exp(-latd*latd/(wd*wd));',
  // filaments THREADED ALONG the plume, and scrolling outward, so the body of
  // it churns instead of being a smooth tongue
  '    float fl2=0.55+0.75*vnoise(vec2(hh*5.0-T*0.30,da*9.0+fk*17.0));',
  // THROTTLE: a fast flutter that travels up the plume, gated on level so it
  // only shudders when there is something to shudder to
  '    fl2*=0.80+0.34*sin(T*(3.1+fk*0.7)+hh*9.0)*(0.3+uLvl);',
  '    float env=smoothstep(-0.04,0.12,hh)*smoothstep(1.20,0.30,hh);',
  '    act+=core*env*fl2*life*(0.30+1.5*bk);',
  // the footpoint: a small bright root sitting ON the limb where it anchors
  '    vec2 rp=vec2(cos(a0),sin(a0))*R;',
  '    vec2 rd=uv-rp;',
  '    act+=exp(-dot(rd,rd)/0.00055)*life*(0.35+1.2*bk);',
  '  }',
  // 75 / 25: the rim carries the frame, everything else is trim.
  // uHaze IS A MORPH FROM A SUN TO AN ECLIPSE, and this line is where that
  // happens. The brief, on the first fringe: "nowhere as nice as BLKSUN" -- the
  // reason is that corona's other 25% (a lit surface, tufts, prominences) is
  // exactly what the reference REMOVES. A rim treatment cannot win against the
  // rest of the mode, so at high haze the trim recedes and the interior goes
  // to void, which is what makes a thin blazing edge read at all. haze 0 is
  // still the untouched corona; the prominences never fully leave, because the
  // reference does keep a couple of bright spikes off the limb.
  '  float v=rim*1.30+body*0.50*(1.0-0.85*bs)',
  '         +tuft*0.26*(1.0-0.75*bs)+act*0.85*(1.0-0.55*bs);',
  '  v*=1.0+uFlux*0.35*energy;',
  // SAME PALETTE AS THE REST OF THE DECK, per the brief -- the duotone, not the
  // red heat ramp the first build invented. A single tint keeps the modes
  // recognisably one system.
  '  return tint*v;',
  '}'].join('\\n');

  // ------------------------------------------------------------- shared tail
  // Mode-independent: emphasis, tonemap, toe, dither. Everything above this
  // line returns pre-tonemap linear-ish colour.
  // ---------------------------------------------------- mode 5: TESLA COIL
  // 2026-08-09, feedback: "2 electrodes (as a glow) inside that also move
  // around to music. and the sparks are between those 2 electrodes and not
  // with the ring itself."
  //
  // REVISION of the first build, and the correction matters: v1 fired sparks
  // from a centre electrode OUT to the rim, which made the ring part of the
  // circuit. It is not -- the ring is the enclosure, the discharge is a thing
  // happening INSIDE it. Two orbiting electrodes with the arc strung between
  // them also give the frame something the single-electrode version could not:
  // the gap LENGTH changes as they orbit, so the discharge visibly stretches
  // and slackens without any parameter being animated to do that.
  //
  // WHY IT IS STILL BUILT ON THE MONOLITH. SS6.5 records the one thing that
  // provably breaks this mode: displacing the ring's radius with the spectrum
  // destroyed the single clean silhouette it exists for. Discharges are
  // ADDITIVE and transient -- they never touch the geometry.
  //
  // THE TRIGGER IS AN EVENT, NOT THE SPECTRUM (SS11.6). uFlash is fired by
  // `trig`, a real conductor transition, so the brightest strikes land on the
  // system's own decisions; uFlux/uLvl only set the idle rate.
  //
  // GEOMETRY: the arc is a point-on-segment plus a perpendicular offset that
  // is PINNED TO ZERO AT BOTH ENDS by a sin(pi*t) taper, so it always terminates
  // exactly on the electrodes however wildly it bows in between. Real branching
  // lightning is recursive and cannot be evaluated per pixel; this is the cheap
  // form that still reads as a filament -- one length() per arc, no segment
  // loop, no noise calls anywhere.
  //
  // CLAIM DISCIPLINE: sparks-on-transitions is an authored display convention,
  // the same class as colour-from-valence. Nothing here is learned.
  var SCENE_TESLA = [
  'vec3 scene(vec2 uv,float T){',
  '  vec3 tint=tintc();',
  '  float energy=0.35+(uA+1.0)*0.55;',
  '  vec2 c=vec2(sin(T*0.021)*0.030,cos(T*0.017)*0.022);',
  '  vec2 pv=uv-c;',
  '  float r=length(pv);',
  '  float aang=atan(pv.y,pv.x);',
  '  float br=sin(T*6.2831/max(uPeriod,4.0));',
  // the enclosure, line-for-line the monolith ring: uniform radius at every
  // angle, breathing and level scaling the whole circle together.
  '  float R=(0.24+0.16*uSpread)*(1.0+0.16*br+0.38*uLvl+uFlash*0.10);',
  '  float th=(0.012+0.030*uDisp+0.010*uTex)*(0.85+0.85*uLvl);',
  '  float de=abs(r-R);',
  '  float rim=exp(-de*de/(th*th));',
  '  rim=mix(rim,th*th/(th*th+de*de*6.0),0.55);',
  '  rim*=0.90+0.75*uLvl;',
  '  float sp=T*(0.10+0.55*uFlux)*energy;',
  '  float u1=aang+sp;',
  '  rim*=(0.62+0.38*cos(u1))+pow(max(0.5+0.5*cos(u1-0.55),0.0),9.0)*1.10;',
  // TWO ELECTRODES, ORBITING INSIDE THE RING, driven by the music: the orbit
  // RATE follows spectral flux, the orbit RADIUS follows level and breath.
  // They are deliberately NOT exactly opposite -- `sw` drifts the second one
  // off antipodal, so the gap between them opens and closes and the pair never
  // looks like a rigid dumbbell rotating.
  // FULLY INDEPENDENT ORBITS (2026-08-09, feedback: "should move independently
  // but never collide or get too close to each other"). B now has its OWN
  // rate and turns the OTHER WAY, so the two sweep past one another instead of
  // holding station like a rigid bar -- which is what "independent" has to
  // mean if it is to be visible at all. That makes real approaches possible,
  // so the separation below is a hard constraint rather than a side effect of
  // keeping them antipodal.
  // The ANGULAR GAP is itself a slow LFO, bounded away from zero, rather than
  // two free-running angles. MEASURED, not assumed: with independent rates the
  // two do eventually line up (gap 0.03 rad at T=678 in a 1.5M-sample sweep),
  // and at that point the separation push is purely RADIAL -- it threw one
  // electrode to 1.21R, the enclosure clamp hauled it back to 0.92R, and the
  // separation collapsed to 0.33R. Bounding the gap removes that degenerate
  // case at the source instead of patching it downstream.
  //   It still reads as independent motion: B's angular velocity is A's plus
  // the gap's own rate, so B visibly advances and retreats against A and the
  // gap breathes between ~115 and ~245 degrees. What it can never do is close.
  // RATES, MEASURED AND RAISED (2026-08-09, feedback: "the electrodes still look
  // too stationary relatively"). The first version's relative angular speed was
  // 1.15*0.019 = 0.022 rad/s -- 1.25 deg/s, i.e. 13 degrees over ten seconds,
  // which is frozen on any timescale a viewer watches. The absolute orbit was
  // no better at 100 s per revolution. What follows is ~6x faster relatively
  // and ~2x absolutely, with a SECOND incommensurate term on the gap so it
  // breathes irregularly rather than sweeping like a metronome.
  //   Deliberately still slow in absolute terms: SS7.1 gives macro motion to
  // the slow parameters, and electrodes that dart are a music video. Roughly
  // 45 s per revolution, with the gap opening and closing over ~15 s, is the
  // compromise -- readable as motion within a few seconds of watching, without
  // anything moving at the FFT's timescale.
  '  float orbA=T*(0.16+0.55*uFlux)*energy;',
  '  float gap=3.14159+0.95*sin(T*0.085+0.70)+0.30*sin(T*0.211+2.40);',
  '  float orbB=orbA+gap;',
  // EACH ELECTRODE HAS ITS OWN DEPTH (2026-08-09, feedback: "should move
  // towards the ring more also and not just at fixed depths / locations").
  // v2 put BOTH on one shared radius, so the pair swept as a rigid bar at a
  // near-constant depth. Independent radial LFOs on incommensurate rates take
  // each one from close to the centre out to ~0.9R -- right up against the
  // enclosure -- and level pushes them outward, so a loud passage drives them
  // apart toward the ring rather than merely brightening them.
  // RADIAL travel, likewise: 134 s and 190 s periods meant the DEPTH never
  // visibly changed either. Now ~42 s and ~58 s with a faster second term, so
  // each electrode is plainly diving toward the centre and climbing back out
  // to the rim -- and the two are on different, incommensurate schedules, so
  // the pair never settles into a pattern.
  '  float lfA=0.5+0.38*sin(T*0.150+1.10)+0.12*sin(T*0.372+0.40);',
  '  float lfB=0.5+0.38*sin(T*0.108+3.90)+0.12*sin(T*0.293+1.90);',
  // radius FLOOR raised too: two electrodes both parked near the centre are
  // close however wide the angle between them is.
  '  float erA=R*(0.30+0.62*lfA)*(0.78+0.22*uLvl);',
  '  float erB=R*(0.30+0.62*lfB)*(0.78+0.22*uLvl);',
  '  vec2 EA=c+erA*vec2(cos(orbA),sin(orbA));',
  '  vec2 EB=c+erB*vec2(cos(orbB),sin(orbB));',
  // MINIMUM SEPARATION, enforced on the RESULT rather than arranged for by the
  // inputs. Constraining the LFOs so they cannot coincide would have coupled
  // them -- i.e. removed the independence just asked for. Pushing the two
  // apart along their own separation axis, only when they are inside dmin,
  // leaves the motion free everywhere else and cannot be defeated by any
  // future change to the orbits. The arc length L is what this really
  // protects: as the gap closes the filament has less room for its wander,
  // and below about 0.6R it collapses into a bright blob between two dots.
  '  vec2 dv=EB-EA; float dl=length(dv);',
  '  float dmin=R*0.62;',
  '  if(dl<dmin){',
  '    vec2 un=dl>1e-4?dv/dl:vec2(1.0,0.0);',
  '    float pu=(dmin-dl)*0.5;',
  '    EA-=un*pu; EB+=un*pu;',
  '  }',
  // and the push must not shove either one through the enclosure
  '  vec2 ra=EA-c; float la=length(ra);',
  '  if(la>R*0.92) EA=c+ra*(R*0.92/la);',
  '  vec2 rb=EB-c; float lb=length(rb);',
  '  if(lb>R*0.92) EB=c+rb*(R*0.92/lb);',
  '  float dA=length(uv-EA), dB=length(uv-EB);',
  // a tight core inside a soft halo -- an electrode is a glow, not a disc.
  '  float el=exp(-dA*dA*1400.0)+exp(-dB*dB*1400.0);',
  '  el+=(exp(-dA*dA*160.0)+exp(-dB*dB*160.0))*0.22;',
  '  el*=0.55+0.60*uLvl;',
  // THE GAP. t is the projection onto the straight electrode-to-electrode
  // line, computed ONCE and shared by every arc.
  '  vec2 ab=EB-EA; float L=max(length(ab),1e-4);',
  '  vec2 dir=ab/L, nrm=vec2(-dir.y,dir.x);',
  '  vec2 rel=uv-EA;',
  '  float tt=clamp(dot(rel,dir)/L,0.0,1.0);',
  // Time is cut into slots; each slot hashes a fresh filament, so no two are
  // alike and none can be tracked. `life` gives instant strike and exponential
  // decay -- a spark hits and fades, it never eases in.
  // THE CHANNEL SHAPE (2026-08-09, feedback: "make the sparks look like real
  // electricity sparks and not like a sine wave? or you can put the current
  // sound wave as the sparks"). BOTH, because they do different jobs:
  //   * the WAVEFORM supplies the large-scale wander, and it is the real
  //     time-domain signal off the analyser -- so the filament is literally
  //     the sound, never repeats, and goes still when the drone goes still;
  //   * SUMMED TRIANGLE OCTAVES supply the small-scale kinks. This is the
  //     whole reason v2 read as a wave: two sines are smooth at every scale,
  //     and smoothness is the one thing a discharge never has.
  // Arc 0 is the main channel, electrode to electrode. Arcs 1-2 are FORKS --
  // same cost, but they start partway along, deviate harder and DIE IN MID
  // AIR, which is what stops three parallel filaments reading as a ribbon.
  '  float sparks=0.0;',
  '  for(int k=0;k<3;k++){',
  '    float fk=float(k);',
  '    float rate=0.40+0.80*uLvl+1.50*uFlux;',
  '    float ph=T*rate+fk*0.41;',
  '    float slot=floor(ph), life=fract(ph);',
  '    float sd=hash(vec2(slot,fk*13.7));',
  '    float sd2=hash(vec2(slot+7.0,fk*3.1+1.7));',
  '    float fork=step(0.5,fk);',
  '    float ts=mix(-0.5,0.10+0.34*sd,fork);',
  '    float te=mix(1.5,0.42+0.30*sd+0.26*sd2,fork);',
  '    float span=smoothstep(ts,ts+0.06,tt)*smoothstep(te,te-0.14,tt);',
  '    float wv=texture2D(uWave,vec2(clamp(tt*0.98+0.01,0.0,1.0),0.5)).x-0.5;',
  '    float jag=tri(tt*3.0+sd*7.0)*0.50+tri(tt*7.0-sd2*5.0)*0.26',
  '             +tri(tt*15.0+sd*3.0)*0.14+tri(tt*31.0-sd2*11.0)*0.07;',
  '    float amp=(0.05+0.20*uDisp)*L*mix(1.0,2.10,fork)*(sd2*1.4-0.2);',
  '    float wob=amp*(jag+wv*(1.6+2.2*uLvl));',
  '    wob*=sin(tt*3.14159);',
  '    vec2 pp=EA+dir*(tt*L)+nrm*wob;',
  '    float d=length(uv-pp);',
  '    float w=(0.0026+0.0075*sin(tt*3.14159))*mix(1.0,0.68,fork);',
  '    sparks+=exp(-d*d/(w*w))*span*exp(-life*7.0)*(0.40+0.60*sd2)',
  '            *mix(1.0,0.72,fork);',
  '  }',
  // GATED ON REAL AUDIO. Applied to the whole discharge term, uFlash
  // included: a transition that fires while nothing is playing must not
  // strike either -- an event with no sound behind it is not an event the
  // viewer can hear, and a spark for it would be a lie about the output.
  '  sparks*=(0.60+3.20*uFlash)*uGate;',
  // A DISCHARGE IS HOTTER than the enclosure, so it desaturates toward white
  // while the ring keeps the duotone. The only place in this file where a
  // colour leaves the palette, and it is temperature doing physical work
  // rather than hue variety -- SS7.2's no-rainbow rule stands.
  '  vec3 hot=mix(tint,vec3(1.0),0.62);',
  '  vec3 col=tint*rim*1.7+hot*(el*0.90+sparks*1.35);',
  '  col*=1.0+uFlux*0.55*energy;',
  '  return col;',
  '}'].join('\\n');

  // ------------------------------------------------------- mode 6: LATTICE
  // the brief, 2026-08-09, on the first build: the sharp lines "look a bit shit
  // in fullscreen" at the render resolution, "make them all disperse so I can
  // just see light moving and morphing without being able to see the
  // underlying pattern", and -- emphatically -- "do NOT fold in those annoying
  // strange curtain / block shapes".
  //
  // ALL THREE ARE ONE DEFECT AND ITS CURE, and the third names a bug this file
  // already documented. v1's dissolve phase used the value-noise haze at
  // uv*1.3, i.e. roughly ONE CELL ACROSS THE FRAME -- which is precisely the
  // condition SS6.7/SS11.4 record as showing value noise's square lattice as
  // large rectangular patches. The quintic fade and per-octave rotation fix
  // creases at ordinary frequencies and CANNOT help at that scale. So:
  //
  //   THERE IS NO VALUE NOISE IN THIS MODE AT ALL. No fbm, no vnoise, one
  //   hash call in the whole scene (the per-cycle seed). Nothing here can
  //   produce a block, because nothing here samples a grid.
  //
  // Everything organic is GRIDLESS SUMS OF SINES -- SS11.4 lesson 6 applied
  // exactly as written ("gridless sines for a slow luminance wash"). The warp
  // bends the lattice continuously so the fold's own regularity never resolves.
  //
  // AND THE LINES ARE GONE AS LINES. Two changes, both needed:
  //   1. the falloff is a pure LORENTZIAN, no gaussian core -- heavy tails,
  //      so every element is a glow rather than a stroke with an edge;
  //   2. the whole fold is evaluated at THREE offset taps and averaged, which
  //      is a real spatial blur rather than a wider kernel. That is what fixes
  //      fullscreen: aliasing comes from a thin feature landing between pixel
  //      centres, and there is no longer a thin feature to alias.
  // Widths are generous and OPEN FURTHER as the cycle dissolves, so the
  // structure melts into formless light instead of fading out as a diagram.
  //
  // `haze` is repurposed here as the DISPERSION knob (there is no fog to
  // control), so the slider still does the job its name implies: more haze,
  // less readable structure.
  //
  // WHAT IT IS FOR (SS9.8, both artists verified before building; both names as
  // sent were wrong): Makoto Inoue -- fractal structure organised around
  // IMPERMANENCE, form arising, holding and dissolving, with what returns not
  // being what left (each cycle re-seeds the fold); Monocolor / Marian Essl --
  // imagery oscillating between fluid organic textures and rigid structures,
  // which is this mode's clock. It sits in SS7.1's vertical time from the far
  // side: stasis as change that never arrives anywhere.
  //
  // COST: 3 taps x 5 folds = 15 iterations of pure arithmetic plus ~8 sines.
  // No hashes in the structure, no noise -- CHEAPER than v1 was despite the
  // blur, which matters on the GPU that forced the SS6.3 revert.
  var SCENE_LATTICE = [
  'vec3 scene(vec2 uv,float T){',
  '  vec3 tint=tintc();',
  '  float energy=0.35+(uA+1.0)*0.55;',
  '  float per=max(uPeriod*4.0,40.0);',
  '  float ph=fract(T/per), cyc=floor(T/per);',
  '  float sd=hash(vec2(cyc,7.3));',
  '  float form=smoothstep(0.0,0.30,ph)*smoothstep(1.0,0.70,ph);',
  // GRIDLESS WARP. Two octaves of sine sums on incommensurate rates: no cell,
  // no corner, nothing that can quantise into a block however low the frequency.
  '  float wt=T*0.05*energy;',
  '  vec2 wv=uv;',
  '  wv+=0.16*vec2(sin(uv.y*2.3+wt*1.1+sd*6.0)+0.5*sin(uv.y*5.1-wt*0.7),',
  '                sin(uv.x*2.7-wt*0.9+sd*4.0)+0.5*sin(uv.x*4.3+wt*1.3));',
  '  wv+=0.07*vec2(sin(uv.x*6.1+wt*1.7),cos(uv.y*5.7-wt*1.9));',
  '  vec2 p0=wv*(1.5+1.1*uSpread);',
  '  float rot=T*0.015*energy+sd*6.2831;',
  '  p0=mat2(cos(rot),-sin(rot),sin(rot),cos(rot))*p0;',
  // THREE-TAP BLUR around the fold. The offset widens as the cycle dissolves,
  // so dissolution is literally the structure going out of focus.
  '  float disp=0.55+0.90*uHaze;',
  '  float glow=0.0;',
  '  for(int s=0;s<3;s++){',
  '    float fs=float(s);',
  '    vec2 off=vec2(cos(fs*2.1),sin(fs*2.1))*((0.010+0.055*(1.0-form))*disp);',
  '    vec2 p=p0+off;',
  '    float sc=1.0,d=1e9;',
  '    for(int i=0;i<5;i++){',
  '      float fi=float(i);',
  '      p=abs(p)-vec2(0.40,0.28)*(0.85+0.30*sin(T*0.037+fi*1.3+sd*9.0));',
  '      float an=0.62+0.22*sin(T*0.011+fi*2.1+sd*5.0);',
  '      p=mat2(cos(an),-sin(an),sin(an),cos(an))*p;',
  '      p*=1.32; sc*=1.32;',
  '      d=min(d,min(abs(p.x),abs(p.y))/sc);',
  '    }',
  // PURE LORENTZIAN, no gaussian core: heavy tails, so this is a glow and
  // never a stroke. Width opens right up as `form` falls.
  '    float w=mix(0.110,0.030,form)*disp+0.012*uLvl;',
  '    glow+=w*w/(w*w+d*d);',
  '  }',
  '  glow*=0.3333;',
  '  vec3 col=tint*glow*(0.55+1.25*form);',
  '  col*=1.0+uFlux*0.45*energy;',
  '  return col;',
  '}'].join('\\n');

  // ------------------------------------------------- mode 7: INVERTED FIELD
  // BLCK SUN (AMIANGELIKA x 1100), the light-ground half of that body of work:
  // dark veins on near-white, mirror-symmetric about a vertical spine, creases
  // converging to nodes.
  //
  // THE ONLY MODE THAT DRAWS INK ON PAPER, and that inverts three assumptions
  // baked into the shared tail. It returns a value ABOVE 1 and lets the tonemap
  // clip it to white while the ink subtracts. Consequences, all deliberate:
  // FS_TAIL's `col *= 1+uFlash*2` does nothing on a white ground, so a
  // transition BLEACHES the ink here instead; the toe that protects the blacks
  // never engages, because nothing in frame is near black; and the duotone
  // enters as a faint wash on the paper rather than as the colour of the light.
  //
  // FBM CANNOT MAKE A VEIN. Plain fBm is smooth by construction -- it has no
  // lines in it at any amplitude, which is why raising contrast on fog gives
  // lumps rather than filaments. Ridged octaves do: 1-|2n-1| folds each octave
  // about its own mid-level, turning a smooth field into a CREASE along that
  // level set, and summing them makes the creases branch and meet at nodes.
  // Same lesson as SS11.4's "match the primitive to the thing" -- points for
  // motes, fBm for fog, folds for architecture, ridges for veins.
  //
  // CLAIM DISCIPLINE: authored, like every other mode. Nothing is learned.
  var SCENE_INVERTED = [
  'vec3 scene(vec2 uv,float T){',
  '  vec3 tint=tintc();',
  '  float energy=0.35+(uA+1.0)*0.55;',
  // NO FOLD, THEREFORE NO SEAM (feedback: "there should NOT be a seam in the
  // middle"). The previous build put THREE things on the centre line: abs(uv.x)
  // creased the noise there, an explicit exp(-x*x) spine drew a dark rule on
  // it, and -- worst -- the per-side noise offset flipped sign across x=0,
  // which is a genuine DISCONTINUITY in the sampled field rather than merely a
  // crease. A mirror cannot help but show its axis.
  //   The bilateral feel now comes from the FLOW leaning toward the centre
  // column instead: a smooth horizontal compression that is strongest at x=0
  // and dies away outward. Nothing is folded, so there is nothing to see on the
  // axis, and the left and right halves are genuinely different structures
  // rather than one structure and its reflection.
  '  float cx=uv.x;',
  // SMOOTH sign, continuous through zero. sign() would reintroduce exactly the
  // step this scene just removed.
  '  float sgn=cx/(abs(cx)+0.10);',
  // THE DOMAIN IS ISOTROPIC AND MUST STAY THAT WAY. This is where the "strange
  // blocky shapes" came from, and it was NOT the lattice: the previous build
  // biased structure toward the centre by COMPRESSING x there
  // (cx*(1-conv*exp(-3cx^2))), which scales the domain along one axis only. A
  // 0.225 factor at the centre makes features 4.4x wider horizontally in a band
  // down the middle -- horizontal smears, and right-angle blobs where those
  // cross vertical vein structure.
  //   MEASURED, by replicating this shader in numpy and rendering it: mean
  // |d/dx| over |d/dy| was 0.86 with the compression and 1.00 without, and
  // removing the domain warp entirely changed nothing (0.85). Two earlier
  // guesses -- missing per-octave rotation, too few noise cells -- were both
  // wrong, which is why fixing them did not help.
  //   RULE: bias by DENSITY, never by geometry. Scaling the domain along one
  // axis stretches every feature in it; weighting the ink cannot distort
  // anything. The centre bias now lives on `ink` below.
  '  vec2 q=vec2(cx,uv.y);',
  '  float wt=T*(0.020+0.045*uFlux)*energy;',
  '  vec2 w=vec2(fbm3(q*2.30+vec2(0.0,wt)),fbm3(q*2.30+vec2(5.2,-wt*0.83)));',
  '  vec2 p=q*(5.00+4.00*uSpread)+(w-0.5)*(0.35+1.10*uDisp);',
  // RIDGED OCTAVES, NOW WITH PER-OCTAVE ROTATION -- this is the blockiness fix
  // (feedback: "stop using those weird blocky things as haze"). Two causes, both
  // documented in SS6.7 and both of which this loop had:
  //   1. NO ROTATION. fbm2/fbm3 have rotated each octave since SS6.7 precisely
  //      because an unrotated stack leaves every octave's value-noise grid
  //      axis-aligned with every other, so the lattice creases REINFORCE
  //      instead of cancelling and the whole field inherits one square grid.
  //      This loop rolls its own octaves so it can fold each one, and in doing
  //      so quietly dropped the rotation that made the shared helpers safe.
  //   2. TOO FEW CELLS ON SCREEN. The base frequency put two or three noise
  //      cells across the frame, which is the exact condition SS6.7 describes
  //      as giving "a few large lumps with visible straight edges" rather than
  //      fog. Base frequency is up ~2x and the stack is five octaves, not four.
  // Ridging VALUE noise is unusually prone to both: 1-|2n-1| creases along the
  // 0.5 level set, and for value noise that set hugs the cell boundaries.
  '  mat2 RM=mat2(0.80,0.60,-0.60,0.80);',
  '  float rg=0.0,amp=0.55,nrm=0.0;',
  '  for(int i=0;i<5;i++){',
  '    float fi=float(i);',
  // each octave drifts on its OWN vector, so the creases reorganise relative to
  // each other instead of sliding as one rigid picture (SS6.9).
  '    vec2 dp=p+vec2(sin(T*(0.031+fi*0.013)+fi*2.1),',
  '                   cos(T*(0.027+fi*0.011)-fi*1.7))*0.35;',
  '    float n=vnoise(dp);',
  '    n=1.0-abs(n*2.0-1.0);',
  '    rg+=amp*n*n; nrm+=amp;',
  '    p=RM*p*2.07;',
  '    amp*=0.55;',
  '  }',
  '  rg/=max(nrm,1e-3);',
  // ONE PASS, THREE ACCUMULATORS. sh is the height-keyed spectrum (no centre
  // here to run angles around, so the band loop is keyed to y and reads bottom
  // to top). lo/hi split the same bands low against high so the two sides can
  // be fed different ends of the spectrum -- the only part of the asymmetry a
  // listener can connect to what they are hearing.
  '  float hy=clamp(uv.y*0.9+0.5,0.0,1.0);',
  '  float sh=0.0,ws=0.0,lo=0.0,hi=0.0;',
  '  for(int b=0;b<12;b++){',
  '    float bx=(float(b)+0.5)/12.0;',
  '    float dd=abs(hy-bx);',
  '    float gw=exp(-pow(dd*9.0,2.0));',
  '    sh+=uB[b]*gw; ws+=gw;',
  '    lo+=uB[b]*(1.0-bx); hi+=uB[b]*bx;',
  '  }',
  '  sh/=max(ws,1e-3); lo/=6.0; hi/=6.0;',
  // asymmetry rides the WIND slider, which this scene has no other use for, so
  // wind 0 feeds both sides the same spectrum as a live A/B. The blend uses the
  // SMOOTH sign, so the left-right difference fades through the middle instead
  // of switching there.
  '  float asym=0.35+0.65*clamp(uWindSpd/1.5,0.0,1.0);',
  '  float drive=mix(sh,mix(lo,hi,clamp(sgn*0.5+0.5,0.0,1.0)),asym);',
  // the sound DEEPENS the ink rather than brightening it: on a light ground
  // more ink is the legible direction, and a whiter white is not a change.
  '  float ink=rg*(0.55+0.95*drive)*(0.70+0.85*uLvl)*(0.55+0.90*uHaze);',
  // centre bias, as pure density -- the veins gather down the middle column
  // without a single feature being stretched to do it.
  '  ink*=0.72+0.60*exp(-cx*cx*2.5);',
  // the breath now rides the ink as a whole -- it used to modulate the spine,
  // which no longer exists.
  '  float br=0.5+0.5*sin(6.2831*T/max(uPeriod,1.0));',
  '  ink*=0.85+0.30*br;',
  // CONTAINMENT. Without it the ink runs to the frame edge and this stops
  // being an object on paper.
  '  ink*=smoothstep(1.05,0.55,length(uv*vec2(0.85,1.0)));',
  // 2.90 is chosen so the tail's col/(1+col*0.55) lands the paper just inside
  // clipping: lower greys the ground, higher wastes range the ink needs.
  '  vec3 paper=mix(vec3(1.0),tint,0.14)*2.90;',
  '  vec3 dark=mix(vec3(1.0),tint,0.45);',
  '  float e=clamp(ink*1.35,0.0,1.0);',
  '  e*=1.0-0.75*uFlash;',
  '  return paper*(1.0-e)+dark*e*0.12;',
  '}'].join('\\n');

  var FS_TAIL = [
  'void main(){',
  '  vec2 uv=(gl_FragCoord.xy-0.5*uRes)/uRes.y;',
  '  float T=uTime;',
  '  vec3 col=scene(uv,T);',
  '  col*=1.0+uFlash*2.0;',
  // NO FLAT AMBIENT TERM. There was one, and it is exactly what washed the
  // frame into a grey gradient -- the darkness between the lit things is what
  // makes light read as physical.
  // MID-TONE LIFT WITHOUT LOSING THE BLACKS (feedback: too dark; earlier note:
  // protect negative space). A SINGLE GAMMA CANNOT DO BOTH -- pow(col,1.25)
  // deepens the mids along with the floor, pow(col,0.82) lifts the floor along
  // with the mids. This is an S-curve: gamma < 1 raises the mid-tones, then a
  // luma-gated toe pushes the near-black back down to true black. Adjust the
  // 0.82 to taste; adjust the 0.085 toe if the background greys up.
  '  col=col/(1.0+col*0.55);',
  '  col=pow(clamp(col,0.0,1.0),vec3(0.82));',
  '  float lu=dot(col,vec3(0.299,0.587,0.114));',
  '  col*=smoothstep(0.0,0.085,lu);',
  '  col+=(hash(gl_FragCoord.xy+T)-0.5)*0.016;',
  '  gl_FragColor=vec4(col,1.0);',
  '}'].join('\\n');

  // colour field REMOVED 2026-08-08 (feedback: "on its own is really not doing
  // anything"). Two rounds of fixes -- de-synchronised drift rates, gridless
  // veil -- made it correct without making it interesting, which is the
  // clearest signal it was the wrong idea rather than a bad implementation.
  // It survives INSIDE ring + field, where it is a background for the ring
  // rather than the whole picture, and that is the only job it was doing.
  // CURTAINS + VORTEX REMOVED 2026-08-09 (feedback: "the vortex and curtains
  // now look too cartoony, so you can remove that"). It was the last survivor
  // of the folding-sheet misreading (SS6.8) and the most expensive scene in the
  // file -- ~14 octaves a pixel against monolith's ~5 -- so the tab also got
  // cheaper. The REASONING it produced is the durable part and stays in
  // VISUALS.md SS6.1-6.4 and SS6.8; the code is in git.
  // These three lists are INDEX-PARALLEL and the <option value=> in
  // panel_html() indexes straight into them, so an edit to one is an edit to
  // all four. A mismatch does not throw -- it renders the wrong scene under
  // the right label, which is invisible until someone notices the picture is
  // not what the menu says. There is a test for exactly that; run it.
  var MODE_NAMES = ['monolith (void)','ring + field','spectrum',
                    'monolith','corona','tesla','lattice','inverted field'];
  var SCENES = [SCENE_MONOLITH, SCENE_RINGFIELD,
                SCENE_SPECTRUM, SCENE_MONOLITH_LIT, SCENE_CORONA,
                SCENE_TESLA, SCENE_LATTICE, SCENE_INVERTED];
  // MODES BUILT BUT DISABLED. 2026-08-09, feedback: tesla was "eating up a
  // lot of GPUs and choking my laptop", and his instruction was to HIDE rather
  // than delete -- then, clarified: "by hide, I mean include in dropdown but
  // the selection disabled". So the option STAYS VISIBLE and greys out; it is
  // not removed. That is the more honest presentation anyway -- a menu that
  // silently drops entries hides that the modes exist at all, where a greyed
  // row says "built, not available", which is what is actually true.
  // The scene, its defaults and its index all stay exactly where they are, so
  // re-enabling is deleting one number here.
  //   This is ALSO the single source of truth: panel_html() still emits every
  // mode, and the loop strips the hidden ones from the <select> at startup.
  // Keeping the option list in Python and the hidden list in JS as two
  // independent edits is the exact drift that renders the wrong scene under
  // the right label, which is why it is done this way round.
  //   A hidden mode is NOT COMPILED either (see initGL): its slot holds null,
  // so it costs no shader memory and no startup link time. Indices stay
  // aligned because the slot is still pushed.
  var MODE_HIDDEN = [6];   // 6 = lattice (tesla re-enabled 2026-08-09)
  function modeHidden(i) { return MODE_HIDDEN.indexOf(i) >= 0; }
  // PER-MODE SLIDER DEFAULTS. The same slider does a different job in each
  // scene, so one global default cannot suit all three -- monolith wants the
  // fog almost off and the rim soft (the brief tuned exactly that by hand:
  // "I had to crank up the disperse and zero out the haze for it"), while
  // curtains needs heavy haze because the multiply is what makes the shafts
  // volumetric at all. Applied on mode change and once at init, so the tab
  // opens already looking right instead of needing the same two drags again.
  // INDEX-PARALLEL with MODE_NAMES/SCENES above and with panel_html()'s
  // <option value=>. `mix` went with the vortex -- it was the only consumer of
  // uMix, so the slider that drove it is gone too.
  var MODE_DEFAULTS = [
    {haze:0.10, disp:0.90, spread:0.50},   // 0 monolith (void)
    {haze:0.30, disp:0.80, spread:0.50},   // 1 ring + field
    {haze:0.25, disp:0.55, spread:0.55},   // 2 spectrum
    {haze:0.10, disp:0.90, spread:0.50},   // 3 monolith (lit)  <- default
    // corona: haze is the BLACK SUN amount here (0 = the smooth limb it had
    // before the fringe went in), and disp sets both rim thickness and hair
    // count -- push disp up and the strands thin toward a pixel and shimmer.
    {haze:0.55, disp:0.45, spread:0.50},   // 4 corona
    // tesla: disp is the filament WANDER here, so it runs high; the ring wants
    // the same near-off fog as monolith, since sparks need black to read.
    {haze:0.10, disp:0.85, spread:0.50},   // 5 tesla
    // lattice: disp is unused (no curtain), haze feeds the dissolve phase, so
    // it sits mid -- too low and the organic half of the cycle is empty frame.
    {haze:0.45, disp:0.50, spread:0.45},   // 6 lattice
    // inverted field: haze is INK DENSITY (this mode has no fog to thin),
    // spread the vein scale, disp the domain-warp amount.
    {haze:0.55, disp:0.55, spread:0.50}    // 7 inverted field
  ];
  function applyModeDefaults(mi) {
    var d = MODE_DEFAULTS[mi]; if (!d) return;
    S.haze = d.haze; S.disp = d.disp; S.spread = d.spread;
    var set = function (id, v) {
      var e = document.getElementById(id); if (e) e.value = v;
    };
    set('conductor-vis-haze', d.haze);   set('conductor-vis-disp', d.disp);
    set('conductor-vis-spread', d.spread);
  }

  function mkShader(gl, type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error('[conductor-vis]', gl.getShaderInfoLog(s)); return null;
    }
    return s;
  }
  function initGL(c) {
    // preserveDrawingBuffer so the popup can drawImage() this canvas -- without
    // it the buffer may already be cleared by the time we blit.
    var gl = c.getContext('webgl', {preserveDrawingBuffer: true, alpha: false,
                                    antialias: false, depth: false});
    if (!gl) return null;
    var vs = mkShader(gl, gl.VERTEX_SHADER, VS);
    if (!vs) return null;
    var UNIFORMS =
      ['uRes','uTime','uV','uA','uLvl','uTex','uFlash','uHaze','uSpread',
       'uWindDir','uWindOff','uWindSpd','uDisp','uFlux','uPeriod','uTint',
       'uSpec','uSpecRow','uWave','uGate','uB'];
    // ALL THREE compiled up front (a few ms) rather than lazily on the first
    // mode switch, which would hitch exactly when someone is watching.
    S.progs = []; S.us = [];
    for (var mi = 0; mi < SCENES.length; mi++) {
      // hidden modes: keep the SLOT so every index still lines up, but build
      // nothing. The mode-change handler already guards on S.progs[mv].
      if (modeHidden(mi)) { S.progs.push(null); S.us.push(null); continue; }
      var fs = mkShader(gl, gl.FRAGMENT_SHADER,
                        FS_HEAD + SCENES[mi] + '\\n' + FS_TAIL);
      if (!fs) return null;
      var p = gl.createProgram();
      gl.attachShader(p, vs); gl.attachShader(p, fs);
      // force attribute 0 in EVERY program, so the single vertexAttribPointer
      // below stays valid across program switches and we never rebind.
      gl.bindAttribLocation(p, 0, 'p');
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
        console.error('[conductor-vis]', MODE_NAMES[mi], gl.getProgramInfoLog(p));
        return null;
      }
      var um = {};
      for (var ui = 0; ui < UNIFORMS.length; ui++)
        um[UNIFORMS[ui]] = gl.getUniformLocation(p, UNIFORMS[ui]);
      S.progs.push(p); S.us.push(um);
    }
    // SPECTROGRAM RING BUFFER. NB x NT LUMINANCE texture: x = log-spaced
    // band, y = time. One ROW is rewritten per update and uSpecRow says
    // where -- so the image scrolls without ever moving any data, which is
    // the whole reason this is a texture and not an array of uniforms.
    // LINEAR filtering, CLAMP_TO_EDGE: WebGL1 allows non-power-of-two only
    // with clamp and no mipmaps, and both dimensions here are powers of two
    // anyway. UNPACK_ALIGNMENT 1 because a LUMINANCE row is NB bytes wide
    // and the default alignment of 4 would corrupt every row.
    S.spec = gl.createTexture();
    S.specData = new Uint8Array(SPEC_NB * SPEC_NT);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.bindTexture(gl.TEXTURE_2D, S.spec);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, SPEC_NB, SPEC_NT, 0,
                  gl.LUMINANCE, gl.UNSIGNED_BYTE, S.specData);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // WAVEFORM, texture unit 1. One row of WAVE_N samples of the live
    // time-domain signal, replaced every frame. Small enough that the upload
    // is free; a uniform array could not hold it and could not be indexed by
    // a computed coordinate in WebGL1 anyway -- the same argument that made
    // the spectrogram a texture.
    S.wave = gl.createTexture();
    S.waveTex = new Uint8Array(WAVE_N);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, S.wave);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, WAVE_N, 1, 0,
                  gl.LUMINANCE, gl.UNSIGNED_BYTE, S.waveTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.activeTexture(gl.TEXTURE0);
    var b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    S.gl = gl;
    // a uniform that a given mode does not declare gets location null, and
    // gl.uniform*(null, ...) is a defined no-op -- so the upload block stays
    // one unconditional list rather than three.
    S.prog = S.progs[S.mode]; S.u = S.us[S.mode];
    gl.useProgram(S.prog);
    // the DOM exists by now, so the opening mode's preset can reach the sliders
    applyModeDefaults(S.mode);
    return gl;
  }

  // ---------------------------------------------------------------- loop
  function draw(now) {
    requestAnimationFrame(draw);
    var c = document.getElementById('conductor-vis');
    if (!c) return;
    // DISABLE HIDDEN MODES IN THE MENU, once, the first frame the select
    // exists. They stay LISTED and grey out (the brief's clarification) rather
    // than being removed. Done here rather than in panel_html() so MODE_HIDDEN
    // is the ONLY place a mode is disabled -- the markup is emitted by Python
    // and the list lives in JS, and keeping both in step by hand is precisely
    // the drift that silently mismatches labels to scenes.
    if (!S.menuDone) {
      var sel0 = document.getElementById('conductor-vis-mode');
      if (sel0) {
        var firstOk = -1;
        for (var oi = 0; oi < sel0.options.length; oi++) {
          var ov = parseInt(sel0.options[oi].value, 10);
          if (modeHidden(ov)) sel0.options[oi].disabled = true;
          else if (firstOk < 0) firstOk = ov;
        }
        // if the DEFAULT itself were ever disabled, fall back to the first
        // enabled option rather than pointing at a program that was never
        // built (its slot is null) and rendering nothing at all.
        if (modeHidden(S.mode) && firstOk >= 0) {
          S.mode = firstOk;
          sel0.value = String(firstOk);
        }
        S.menuDone = 1;
      }
    }
    // Skip while hidden. rAF stops when the BROWSER tab is hidden, but a
    // deselected gradio Tab is only display:none and would otherwise burn GPU
    // beside the renderer. A live popup keeps us drawing regardless.
    var shown = c.offsetParent !== null || document.fullscreenElement;
    if (!shown && !(S.pop && !S.pop.closed)) return;

    // FRAME CAP. rAF is still re-queued above (that costs nothing); this
    // just declines to DRAW more often than the budget. On a 120 Hz panel
    // this alone is a 3x cut, and on an ambient drone it is invisible --
    // parameters are tweened in wall-clock time (SS4), so a lower frame rate
    // changes smoothness, not speed.
    var iv = now - S.lastFrame;
    if (S.lastFrame && iv < FRAME_MS) return;
    // ADAPTIVE RENDER SCALE. The rAF interval is the honest signal available
    // without EXT_disjoint_timer_query: if we ask for FRAME_MS and keep being
    // handed more, the GPU did not finish in time. Drop fast, restore slowly,
    // so a transient hitch does not permanently soften the image and a
    // recovering machine does not oscillate.
    if (S.lastFrame) {
      S.ivAvg += (Math.min(iv, 200) - S.ivAvg) * 0.05;
      if (S.ivAvg > FRAME_MS * 1.35) S.auto = Math.max(0.50, S.auto - 0.020);
      else if (S.ivAvg < FRAME_MS * 1.06) S.auto = Math.min(1.00, S.auto + 0.004);
    }
    S.lastFrame = now;

    var gl = S.gl || initGL(c);
    if (!gl) return;
    var dt = Math.min(0.05, (now - S.t0) / 1000); S.t0 = now; S.t += dt;

    // resolution: both axes, whole pixels. A fractional assignment is
    // truncated by the canvas but not by Math.round, so the two never agree
    // and the buffer is reallocated every frame.
    var w = c.clientWidth || 960, h = c.clientHeight || 540;
    var dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP)
              * (S.quality || 1) * S.auto;
    var cw = Math.max(2, Math.round(w * dpr)), ch = Math.max(2, Math.round(h * dpr));
    if (c.width !== cw || c.height !== ch) { c.width = cw; c.height = ch; }
    gl.viewport(0, 0, cw, ch);

    // ---- audio, per frame
    var an = window.CONDUCTOR_AN, n = 0;
    if (an) {
      n = an.frequencyBinCount;
      if (!S.tdBuf || S.tdBuf.length !== an.fftSize)
        S.tdBuf = new Uint8Array(an.fftSize);
      if (!S.buf || S.buf.length !== n) S.buf = new Uint8Array(n);
      an.getByteFrequencyData(S.buf);
      var tot = 0;
      // LOG-SPACED OVER 40 Hz - 8 kHz, not linear over the whole spectrum.
      // MEASURED 2026-08-08: s19 sets fftSize=128, so at 48 kHz each bin is
      // 375 Hz. Splitting 64 bins LINEARLY into 12 gave band 0 = 0-2 kHz --
      // i.e. the entire drone -- and bands 1..11 = 2-24 kHz, where ambient
      // material has essentially nothing. Eleven of the twelve bands were dead,
      // which is why the rig barely reacted. Pitch is logarithmic; the bands
      // have to be too.
      var sr = (an.context && an.context.sampleRate) || 48000;
      var binHz = (sr * 0.5) / n, F0 = 40, F1 = 8000;
      for (var i = 0; i < NB; i++) {
        var flo = F0 * Math.pow(F1 / F0, i / NB),
            fhi = F0 * Math.pow(F1 / F0, (i + 1) / NB);
        var lo = Math.max(0, Math.floor(flo / binHz)),
            hi = Math.min(n, Math.max(lo + 1, Math.ceil(fhi / binHz))), s = 0;
        for (var k2 = lo; k2 < hi; k2++) s += S.buf[k2];
        var e = (hi > lo) ? s / (hi - lo) / 255 : 0;
        // asymmetric smoothing: snap up on a transient, fall back slowly --
        // a rig hits instantly and decays, it does not ease into a hit.
        // Damped on purpose. At 0.55 attack the sheets bounced on every
        // spectral spike, which destroys the stasis drone music lives in.
        // Macro motion belongs to arousal; the FFT only shimmers.
        S.bands[i] += (e - S.bands[i]) * (e > S.bands[i] ? 0.30 : 0.05);
        tot += e;
      }
      S.lvl += ((tot / NB) - S.lvl) * 0.25;
      // SPARKS ONLY WHEN SOMETHING IS PLAYING (2026-08-09). The idle
      // discharge rate had a constant term, so an idle deck arced away in
      // silence. Fast attack so the first note strikes immediately, SLOW
      // release so a fading drone's arcs taper out instead of being cut off
      // mid-strike -- which would read as a bug rather than as silence.
      var gt = Math.min(1, Math.max(0, (S.lvl - 0.008) / 0.027));
      S.gate += (gt - S.gate) * (gt > S.gate ? 0.35 : 0.03);
      // SPECTRAL FLUX = the sum of POSITIVE bin-to-bin changes between frames,
      // i.e. how much the spectrum is actually MOVING. Ambient drone has no
      // beat, so there is no tempo to extract -- but flux is the honest
      // analogue: it rises when the music is going somewhere and falls when it
      // settles. Only rises count; falling energy is a decay, not an event.
      // This is what the swirl follows, so the vortex is driven by the sound
      // rather than by the clock.
      if (!S.prev || S.prev.length !== n) S.prev = new Uint8Array(n);
      var fx = 0;
      for (var q2 = 0; q2 < n; q2++) {
        var dv = S.buf[q2] - S.prev[q2];
        if (dv > 0) fx += dv;
        S.prev[q2] = S.buf[q2];
      }
      fx = Math.min(1.5, (fx / n) / 255 * 24.0);
      S.flux += (fx - S.flux) * (fx > S.flux ? 0.35 : 0.025);
    } else {
      // NO AUDIO GRAPH: the bands below are synthetic, so the level they
      // produce is not evidence of anything. The gate stays shut -- tesla
      // shows a ring and two electrodes and no discharge, which is the honest
      // picture when we cannot hear the output at all.
      S.gate += (0 - S.gate) * 0.03;
      for (var j = 0; j < NB; j++)
        S.bands[j] = 0.16 + 0.10 * Math.sin(S.t * (0.4 + j * 0.11) + j);
      S.lvl += (0.18 - S.lvl) * 0.05;
      S.flux += (0.10 - S.flux) * 0.02;
    }

    // ---- params, tweened (frame-rate independent)
    var d = ds(), kk = 1 - Math.exp(-dt * 0.7);
    S.v += (num(d, 'v', S.v) - S.v) * kk;
    S.a += (num(d, 'a', S.a) - S.a) * kk;
    S.tex += (num(d, 'tex', S.tex) - S.tex) * kk;
    S.chord = d.chord || 'min';
    S.ramp = num(d, 'ramp', 0); S.std = num(d, 'std', 0);
    S.period += (num(d, 'period', S.period) - S.period) * kk;
    // ---- COLOUR AUTO-RANGE. Measured over 326 logged segments, the walk's
    // valence spans -0.90..-0.61 -- so a duotone keyed to the nominal [-1,+1]
    // uses 15% of its range and never leaves slate blue. That is why the
    // palette read as doing nothing (2026-08-08). A fixed remap
    // cannot fix it either: no constant gain rescues a variable that barely
    // varies without overfitting to one session's logs.
    //   So the range is LEARNED from the signal at runtime: expand instantly
    // to admit a new extreme, contract very slowly, and map the current value
    // into whatever span has been seen. The span is floored so that at startup
    // -- and during any long stretch of near-constant valence -- the tint does
    // not become hypersensitive to noise around a single point.
    //   AUTHORED DISPLAY CONVENTION, and a normalisation on top of one. It
    // says nothing about valence and must never be written up as though the
    // colour were measuring anything. It is also NOT the silent [-1,1] stretch
    // the thesis figure rule forbids: that rule governs plotted label units,
    // this is a display gain on a picture, and it is documented in VISUALS.md.
    if (!S.vSeen) { S.vLo = S.vHi = S.v; S.vSeen = true; }
    S.vLo = Math.min(S.v, S.vLo + (S.v - S.vLo) * 0.0004);
    S.vHi = Math.max(S.v, S.vHi + (S.v - S.vHi) * 0.0004);
    var vSpan = Math.max(S.vHi - S.vLo, 0.30), vMid = (S.vHi + S.vLo) * 0.5;
    var tgt = Math.min(1, Math.max(0, 0.5 + (S.v - vMid) / vSpan));
    S.tintT += (tgt - S.tintT) * kk;
    var step = Math.round(num(d, 'step', -1));
    if (step !== S.step) {                       // edge, not level
      if (S.step >= 0 && d.trig === '1') S.flash = 1.0;
      if (S.step >= 0) S.glitch = 1.0;           // a SEGMENT CHANGED
      S.step = step;
    }
    // GLITCH IS AN EVENT, NOT A TEXTURE (2026-08-08, feedback: the constant
    // tungsten flicker "is a bit annoying now"). Anything continuous stops
    // being read as an effect within about a minute and just becomes noise
    // you are trying to see through. So the readout is STILL by default and
    // tears only when something happens: a segment change above, or a random
    // interval here. Random rather than periodic, because a fixed period is
    // itself something the eye learns and then anticipates.
    if (S.t > S.glitchAt) {
      S.glitch = 1.0;
      S.glitchAt = S.t + 7.0 + Math.random() * 16.0;
    }
    S.glitch *= Math.exp(-dt * 3.4);             // short: a tear, not a mode
    S.flash *= Math.exp(-dt * 1.6);

    // ---- WIND, and the reason the frame is never still. VA arrives once a
    // SEGMENT, so binding the wind to VA alone would freeze the smoke between
    // transitions -- which is the opposite of what was asked for. Three terms,
    // on three different clocks:
    //   valence  -> which way the breeze leans        (per segment, tweened)
    //   arousal  -> how hard it lifts. Smoke rises even in still air, so this
    //               biases upward and never reaches zero  (per segment)
    //   loudness -> gusts, attack-fast / release-slow like the band envelopes,
    //               so a swell physically pushes the smoke   (per frame)
    // plus a MEANDER on two incommensurate periods (~23 s and ~9 s), which is
    // what keeps it evolving when the walk is parked and the level is steady.
    // AUTHORED CONVENTION, exactly like colour-from-valence -- nothing here is
    // learned and it must never be written up as an emotion->image mapping.
    S.gust += (S.lvl - S.gust) * (S.lvl > S.gust ? 0.10 : 0.015);
    var wx = S.v * 0.85, wy = 0.42 + (S.a + 1) * 0.34;
    var mdr = 0.40 * Math.sin(S.t * 0.043) + 0.22 * Math.sin(S.t * 0.107 + 1.7);
    var ang = Math.atan2(wy, wx) + mdr;
    var spd = Math.min(1.2, Math.hypot(wx, wy) * (0.55 + 1.05 * S.gust)) * S.wind;
    S.wdx = Math.cos(ang); S.wdy = Math.sin(ang);
    // integrate, never speed*t -- see the shader's advection note
    S.woffx += S.wdx * spd * dt; S.woffy += S.wdy * spd * dt;
    S.wspd = spd;

    gl.uniform2f(S.u.uRes, cw, ch);
    gl.uniform1f(S.u.uTime, S.t);
    gl.uniform1f(S.u.uV, S.v);      gl.uniform1f(S.u.uA, S.a);
    gl.uniform1f(S.u.uLvl, S.lvl);  gl.uniform1f(S.u.uTex, S.tex);
    gl.uniform1f(S.u.uFlash, S.flash);
    gl.uniform1f(S.u.uGate, S.gate);
    gl.uniform1f(S.u.uHaze, S.haze); gl.uniform1f(S.u.uSpread, S.spread);
    gl.uniform2f(S.u.uWindDir, S.wdx, S.wdy);
    gl.uniform2f(S.u.uWindOff, S.woffx, S.woffy);
    gl.uniform1f(S.u.uWindSpd, S.wspd);
    gl.uniform1f(S.u.uDisp, S.disp);
    gl.uniform1f(S.u.uFlux, S.flux);
    gl.uniform1f(S.u.uPeriod, S.period);
    gl.uniform1f(S.u.uTint, S.tintT);
    // one spectrogram row every 1/SPEC_HZ s, NOT every frame: the history
    // window must be a fixed span of TIME, not of frames, or the image
    // scrolls at whatever rate the GPU happens to manage.
    if (S.spec && S.buf && S.t - S.specT > 1.0 / SPEC_HZ) {
      S.specT = S.t;
      var nb = S.buf.length, bhz = ((an && an.context ? an.context.sampleRate
                                     : 48000) * 0.5) / nb;
      var row = new Uint8Array(SPEC_NB);
      for (var sb = 0; sb < SPEC_NB; sb++) {
        var f0 = 40 * Math.pow(8000 / 40, sb / SPEC_NB),
            f1 = 40 * Math.pow(8000 / 40, (sb + 1) / SPEC_NB);
        var l0 = Math.max(0, Math.floor(f0 / bhz)),
            l1 = Math.min(nb, Math.max(l0 + 1, Math.ceil(f1 / bhz))), mx = 0;
        // MAX, not mean: a single strong partial inside a wide high band
        // would be averaged away to nothing, and partials are the content.
        for (var k3 = l0; k3 < l1; k3++) if (S.buf[k3] > mx) mx = S.buf[k3];
        row[sb] = mx;
      }
      var y = S.specRow % SPEC_NT;
      gl.bindTexture(gl.TEXTURE_2D, S.spec);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, y, SPEC_NB, 1,
                       gl.LUMINANCE, gl.UNSIGNED_BYTE, row);
      S.specRow = (S.specRow + 1) % SPEC_NT;
    }
    if (S.spec) {
      gl.bindTexture(gl.TEXTURE_2D, S.spec);
      gl.uniform1i(S.u.uSpec, 0);
      gl.uniform1f(S.u.uSpecRow, S.specRow / SPEC_NT);
    }
    // WAVEFORM UPLOAD. Strided across the whole analyser window rather than
    // taking a contiguous slice: at 48 kHz, 256 consecutive samples is ~5 ms,
    // less than one cycle of a 60 Hz drone, so a contiguous read would be a
    // smooth arc. Striding covers ~43 ms -- several cycles -- which is what
    // gives the filament something to follow.
    if (S.wave) {
      if (an && S.tdBuf) {
        an.getByteTimeDomainData(S.tdBuf);
        var st = Math.max(1, Math.floor(S.tdBuf.length / WAVE_N));
        for (var wi = 0; wi < WAVE_N; wi++) S.waveTex[wi] = S.tdBuf[wi * st];
      } else {
        // no audio graph: a slow synthetic wander, so the arcs still move
        // rather than snapping to a dead straight line (README SS22.7).
        for (var wj = 0; wj < WAVE_N; wj++)
          S.waveTex[wj] = 128 + Math.round(40 * Math.sin(wj * 0.21 + S.t * 1.7)
                                         + 22 * Math.sin(wj * 0.57 - S.t * 2.3));
      }
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, S.wave);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, WAVE_N, 1,
                       gl.LUMINANCE, gl.UNSIGNED_BYTE, S.waveTex);
      gl.uniform1i(S.u.uWave, 1);
      gl.activeTexture(gl.TEXTURE0);
    }
    gl.uniform1fv(S.u.uB, S.bands);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // ---- glitch overlay (2D canvas stacked on the gl one)
    var ov = document.getElementById('conductor-vis-ov');
    if (ov) {
      // THE READOUT DOES NOT NEED THE ART'S RESOLUTION. It was sized to match
      // the GL canvas, so fullscreen at dpr 2 meant a SECOND 8.3 Mpx surface
      // cleared and composited every frame -- for text that drawOverlay then
      // scaled from a 1600-wide design anyway. Capped at that design width and
      // stretched by the existing CSS; 5.8x fewer pixels at fullscreen, and
      // the glitch/tear rects scale with it because they are drawn in the same
      // coordinate space.
      var ow = Math.min(cw, OV_MAX_W);
      var oh = Math.max(2, Math.round(ow * ch / Math.max(cw, 1)));
      if (ov.width !== ow || ov.height !== oh) { ov.width = ow; ov.height = oh; }
      drawOverlay(ov.getContext('2d'), ow, oh, step);
    }
    // ---- second screen: blit gl + overlay into the popup
    if (S.pop && !S.pop.closed && S.popCtx) {
      var pc = S.popCtx.canvas;
      var pw = S.pop.innerWidth, ph = S.pop.innerHeight;
      if (pc.width !== pw || pc.height !== ph) { pc.width = pw; pc.height = ph; }
      S.popCtx.fillStyle = '#05070a'; S.popCtx.fillRect(0, 0, pw, ph);
      // letterbox to the same aspect as the inline view
      var ar = cw / ch, sw = pw, sh = pw / ar;
      if (sh > ph) { sh = ph; sw = ph * ar; }
      var ox = (pw - sw) / 2, oy = (ph - sh) / 2;
      S.popCtx.drawImage(c, ox, oy, sw, sh);
      if (ov) S.popCtx.drawImage(ov, ox, oy, sw, sh);
    }

    // The caption is now shown ONLY when something is wrong or unusual
    // (2026-08-08, feedback: remove it). In the normal case -- audio graph
    // present, no popup -- it says nothing, because a label confirming that
    // things are working is just clutter on the picture.
    //   It is NOT deleted, because VISUALS.md 2 requires that an optional
    // layer never degrade silently: with no audio graph the visual is driven
    // by parameters alone and must SAY so rather than merely looking dead.
    // That is the one case it still appears.
    var src = document.getElementById('conductor-vis-src');
    if (src) {
      var want = (an ? '' : 'parameters only (no audio graph)');
      // SAY WHEN THE SCALER HAS BACKED OFF. An optional layer must not degrade
      // silently (README SS22.7) -- and without this line a soft picture on a
      // slow machine looks like a design choice rather than a budget being
      // enforced, which is exactly the confusion that wasted a session on the
      // "mode X is heavy" theory.
      if (S.auto < 0.95)
        want += (want ? '   |   ' : '') + 'render ' + Math.round(S.auto * 100) + '%';
      if (S.pop && !S.pop.closed) want += (want ? '   |   ' : '') + '2nd screen live';
      if (src.textContent !== want) src.textContent = want;
    }
  }

  function drawOverlay(g, cw, ch, step) {
    // With params OFF this canvas is already blank, and clearing a full-screen
    // surface every frame for nothing is exactly the kind of cost that does
    // not show up in any one mode. Clear once on the way off, then leave it.
    if (!S.params && S.ovClear) return;
    g.clearRect(0, 0, cw, ch);
    S.ovClear = !S.params;
    // OFF means gone, not dimmed: the canvas is already cleared above, so
    // returning here leaves the frame with nothing over it at all. The
    // popup blit reads this same canvas, so the second screen follows.
    if (!S.params) return;
    var sc = cw / 1600;
    g.save(); g.scale(sc, sc);
    // TWO COLUMNS, not one padded string. The right-hand values used to sit
    // wherever the left field's text happened to end, so they stepped in and
    // out by a few characters between rows; drawing the right column at its
    // own fixed x is the only way it actually lines up.
    var L = ['v ' + S.v.toFixed(3),
             'chord ' + S.chord,
             'ramp ' + S.ramp.toFixed(1) + 's',
             'step ' + (step < 0 ? '--' : step)];
    var Rv = ['a ' + S.a.toFixed(3),
              'texture ' + S.tex.toFixed(2),
              'sigma ' + S.std.toFixed(3),
              ''];
    // A READOUT IS NOT THE SUBJECT (2026-08-08, feedback: bold amber type was
    // "taking too much attention from the visual"). So: unbolded, neutral
    // grey, low alpha, and moved out of the top-left corner -- the eye lands
    // top-left first, which is the worst place for the least important thing
    // in the frame. Bottom-left is where a slate belongs.
    //   Grey rather than amber because the tint changes per mode and per
    //   valence; a neutral reads as sitting BEHIND whatever is in front of it
    //   instead of competing as one more coloured element.
    g.font = '400 17px ui-monospace,SFMono-Regular,Menlo,monospace';
    var G = S.glitch;
    var H = ch / sc, x0 = 26, colw = 168;
    for (var i = 0; i < L.length; i++) {
      // bottom-up: the last line sits on the baseline and earlier ones stack
      // above it, so the block is anchored to the bottom edge
      var y = H - 26 - (L.length - 1 - i) * 24;
      var tear = 0.0;
      if (G > 0.02) {
        if (Math.sin(S.t * 90.0 + i * 3.7) > 0.35)
          tear = (Math.random() - 0.5) * 30.0 * G;
      }
      var x = x0 + tear;
      // MONOCHROME split. A red/cyan fringe put colour back into the one
      // element that is meant to stay out of the way; light/dark tears read as
      // signal loss just as well and stay neutral.
      if (G > 0.02) {
        g.globalCompositeOperation = 'source-over';
        g.fillStyle = 'rgba(255,255,255,' + (0.22 * G).toFixed(3) + ')';
        g.fillText(L[i], x - 2.4 * G, y);
        if (Rv[i]) g.fillText(Rv[i], x + colw - 2.4 * G, y);
        g.fillStyle = 'rgba(0,0,0,' + (0.30 * G).toFixed(3) + ')';
        g.fillText(L[i], x + 2.4 * G, y);
        if (Rv[i]) g.fillText(Rv[i], x + colw + 2.4 * G, y);
      }
      // a dark halo rather than a glow: it has to stay legible over the
      // spectrum mode's bright field as well as over black, and a bloom would
      // make it the brightest thing in frame again.
      g.globalCompositeOperation = 'source-over';
      g.shadowColor = 'rgba(0,0,0,.60)'; g.shadowBlur = 3;
      g.fillStyle = 'rgba(196,202,210,' + (0.52 - 0.14 * G).toFixed(3) + ')';
      g.fillText(L[i], x, y);
      if (Rv[i]) g.fillText(Rv[i], x + colw, y);
      g.shadowBlur = 0;
      if (G > 0.45 && Math.random() < 0.5) {
        g.globalCompositeOperation = 'destination-out';
        g.fillStyle = 'rgba(0,0,0,1)';
        g.fillRect(0, y - 11 + Math.random() * 15, cw, 1 + Math.random() * 2);
      }
    }
    g.globalCompositeOperation = 'source-over';
    g.restore();
  }
  requestAnimationFrame(draw);

  // ---------------------------------------------------------------- controls
  // Entirely client-side. gradio dispatches an event per pointer-move, which
  // through the share relay is a request per move (the flooding that returned
  // HTML error pages in s04).
  // EVERY LINE THAT BUILDS THE POPUP IS A WRITE INTO ANOTHER WINDOW'S DOM, and
  // that is exactly what an embedding page can deny. MEASURED 2026-08-09 on
  // `huggingface.co/spaces/...`: the app runs in HF's iframe there, the popup it
  // opens is not reachable from here, and all four writes throw. The popup's
  // <body> was left empty and unstyled (checked in its own devtools; an ad
  // blocker was ruled out by disabling it, and the same click works on the
  // direct `*.hf.space` URL).
  //
  // The old failure mode was the bad one: S.pop/S.popCtx are assigned LAST, so a
  // throw left them unset, draw()'s blit stayed gated off for the rest of the
  // session, and the window just sat there blank with nothing said anywhere.
  // An optional layer must not degrade silently (README SS22.7) -- so catch it,
  // close the dead window, and name the fix.
  //
  // NOT worth "fixing" by writing less: a popup that renders itself from a
  // blob: URL and takes frames over postMessage would avoid the denied
  // operation entirely, but it may inherit the same sandbox, and fullscreen
  // already covers everything except a genuine second display.
  function popOut() {
    if (S.pop && !S.pop.closed) { S.pop.focus(); return; }
    var w = window.open('', 'conductor-visuals', 'width=1280,height=720');
    if (!w) { alert('Pop-out blocked -- allow popups for this page.'); return; }
    try {
      var d = w.document;
      if (!d || !d.body) throw new Error('popup document unreachable');
      d.title = 'the Unfinished Conductor - visuals';
      d.body.style.cssText = 'margin:0;background:#05070a;overflow:hidden';
      var cv = d.createElement('canvas');
      cv.style.cssText = 'display:block;width:100vw;height:100vh';
      d.body.appendChild(cv);
      // dblclick anywhere = fullscreen on that screen; esc leaves, as usual
      d.body.addEventListener('dblclick', function () {
        if (d.fullscreenElement) d.exitFullscreen();
        else cv.requestFullscreen && cv.requestFullscreen();
      });
      S.pop = w; S.popCtx = cv.getContext('2d');
    } catch (err) {
      try { w.close(); } catch (e2) {}
      S.pop = null; S.popCtx = null;
      console.error('[conductor-vis] pop-out denied:', err);
      // Reading window.top THROWS when the parent is cross-origin, which is
      // itself the answer -- so a throw here means "embedded", same as an
      // inequality. location.* is the IFRAME's own url, i.e. precisely the
      // direct address the user needs, so this stays correct on any Space.
      var framed;
      try { framed = window.top !== window.self; } catch (e3) { framed = true; }
      alert(framed
        ? 'Pop-out will not work on this page: it is an embedded copy of the '
          + 'app, and the browser blocks writing into the window it opens.\\n\\n'
          + 'Open the app directly instead:\\n' + location.origin
          + location.pathname + '\\n\\nfullscreen works here either way.'
        : 'Pop-out failed: ' + err + '\\n\\nUse fullscreen instead.');
    }
  }
  window.addEventListener('beforeunload', function () {
    if (S.pop && !S.pop.closed) S.pop.close();
  });

  document.addEventListener('click', function (e) {
    var t = e.target;
    if (!t || !t.closest) return;
    var pb = t.closest('#conductor-vis-params');
    if (pb) {
      S.params = !S.params;
      pb.textContent = S.params ? 'params on' : 'params off';
      return;
    }
    if (t.closest('#conductor-vis-pop')) { popOut(); return; }
    if (t.closest('#conductor-vis-fs')) {
      var wrap = document.getElementById('conductor-vis-wrap');
      if (!wrap) return;
      if (document.fullscreenElement) document.exitFullscreen &&
        document.exitFullscreen();
      else if (wrap.requestFullscreen) wrap.requestFullscreen().catch(function () {});
    }
  });
  document.addEventListener('input', function (e) {
    var t = e.target;
    if (!t || !t.id) return;
    if (t.id === 'conductor-vis-mode') {
      var mv = parseInt(t.value, 10) || 0;
      if (modeHidden(mv)) return;  // disabled: no program was ever built
      if (S.progs && S.progs[mv]) {
        S.mode = mv; S.prog = S.progs[mv]; S.u = S.us[mv];
        if (S.gl) S.gl.useProgram(S.prog);
      } else { S.mode = mv; }        // pre-initGL: picked up when it builds
      applyModeDefaults(mv);
    }
    if (t.id === 'conductor-vis-haze') S.haze = parseFloat(t.value);
    if (t.id === 'conductor-vis-spread') S.spread = parseFloat(t.value);
    // disperse 0 == the pre-2026-08-08 solid sheets, kept as the A/B reference
    if (t.id === 'conductor-vis-disp') S.disp = parseFloat(t.value);
    if (t.id === 'conductor-vis-wind') S.wind = parseFloat(t.value);
    // render scale, for phones: a full-screen fBm shader will thermally
    // throttle a handset, and at this much haze the softness is invisible.
    if (t.id === 'conductor-vis-quality') S.quality = parseFloat(t.value);
  });
  document.addEventListener('fullscreenchange', function () {
    var b = document.getElementById('conductor-vis-fs');
    if (b) b.textContent = document.fullscreenElement ? 'exit (esc)' : 'fullscreen';
  });
})();
</script>
"""


def panel_html():
    """The visual's markup: the GL canvas, a 2D overlay stacked on it, and the
    controls.

    STATIC -- everything that moves is drawn by VIS_HEAD's loop, so this is
    never re-sent and cannot be wiped by a status update. The selects, sliders
    and buttons are plain DOM read by that loop; they deliberately do not
    round-trip to the server.

    NOTE this function body was rebuilt from __pycache__ on 2026-08-08 after a
    bad edit truncated the module (see VISUALS.md 11.4). The returned markup is
    byte-identical to what was running; only the formatting of the literal
    differs from the original source."""
    return (
        "<div id='conductor-vis-wrap'><canvas id='conductor-vis'></canvas><canvas id='conductor-vis-ov'></canvas><div id='conductor-vis-ui'><label>mode<select id='conductor-vis-mode'><option value='3' selected>monolith</option><option value='0'>monolith (void)</option><option value='4'>corona</option><option value='1'>ring + field</option><option value='2'>spectrum</option><option value='5'>tesla</option><option value='6'>lattice</option><option value='7'>inverted field</option></select></label><label>haze<input id='conductor-vis-haze' type='range' min='0' max='1' step='0.02' value='0.75'></label><label>spread<input id='conductor-vis-spread' type='range' min='0' max='1' step='0.02' value='0.5'></label><label>disperse<input id='conductor-vis-disp' type='range' min='0' max='1' step='0.02' value='0.75'></label><label>wind<input id='conductor-vis-wind' type='range' min='0' max='1.5' step='0.02' value='0.6'></label><label>res<input id='conductor-vis-quality' type='range' min='0.4' max='1' step='0.05' value='1'></label><button type='button' id='conductor-vis-params'>params on</button><button type='button' id='conductor-vis-pop'>pop out</button><button type='button' id='conductor-vis-fs'>fullscreen</button></div><div id='conductor-vis-src'>waiting for a session</div></div>"
    )


def data_html(d=None):
    """The parameter carrier: one empty div whose DATA ATTRIBUTES the animation
    loop polls.

    Data attributes rather than a <script> block because gradio applies a
    component update by writing innerHTML, and scripts inserted that way do not
    execute -- the same constraint that makes every lamp on this deck a
    `<style>` carrier. Values are formatted here so the browser only ever
    parses floats.
    """
    d = d or {}

    def g(k, dflt=0.0):
        try:
            return float(d.get(k, dflt))
        except (TypeError, ValueError):
            return dflt

    chord = "maj" if str(d.get("chord", "min")) == "maj" else "min"
    return ("<div id='conductor-vis-data' style='display:none'"
            f" data-v='{g('v'):.4f}' data-a='{g('a'):.4f}'"
            f" data-tv='{g('tv'):.4f}' data-ta='{g('ta'):.4f}'"
            f" data-tex='{g('tex'):.3f}' data-chord='{chord}'"
            f" data-ramp='{g('ramp'):.2f}' data-std='{g('std'):.4f}'"
            f" data-period='{g('period') or 30.0:.2f}'"
            f" data-trig='{int(g('trig'))}'"
            f" data-step='{int(g('step'))}'></div>")
