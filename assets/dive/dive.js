/* The dive. Everything on screen is a function of one number, t.

   t runs along the stops: stop i holds from t = i to i + 0.5, then the camera travels from i + 0.5 to
   i + 1.  Scrolling only drives t, through a table measured from the page, so a long text can have a
   long hold without changing any of the keyframes below.

   The camera is real scale. R is the length of the on-screen bar, half the short side of the view, so
   the field F = 2R is in millimetres from the first frame (R = 100 mm: ten centimetres) to the last
   (R = 0.0005 mm: the bar is exactly one 0.5 µm pixel of the scan wide). */
(() => {
  'use strict';
  const root = document.documentElement;
  const params = new URLSearchParams(location.search);
  const STILL = params.has('still');

  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
  const smoother = (t) => t * t * t * (t * (t * 6 - 15) + 10);
  const $ = (s) => document.querySelector(s);

  const stops = [...document.querySelectorAll('.stop')].map((el, i) => ({
    el, i, id: el.id,
    copy: el.querySelector('.copy'),
    plates: [...el.querySelectorAll('.plate')],
    lens: el.querySelector('.lens[data-mm]'),
    R: parseFloat(el.dataset.r),
    label: el.dataset.label,
    plateau: parseFloat(el.dataset.plateau) || 1,
    transit: parseFloat(el.dataset.transit) || 0,
    images: [...el.querySelectorAll('img[data-src]')],
    shown: false, opacity: -1,
  }));
  const N = stops.length;
  const LOG_TOP = Math.log10(100), LOG_END = Math.log10(0.0005);
  const year = $('#year'); if (year) year.textContent = new Date().getFullYear();

  function showImages(s) {
    // the stills are the plain page's stand-ins for the stage: the dive never needs them
    const diving = root.classList.contains('dive');
    s.images = s.images.filter((img) => {
      if (diving && img.classList.contains('flat-only')) return true;
      img.src = img.dataset.src; img.removeAttribute('data-src');
      return false;
    });
  }

  // ---------- readouts ----------
  const trim = (x) => String(+x.toPrecision(3));
  function stopLength(mm) {
    if (mm >= 10) return trim(mm / 10) + ' cm';
    if (mm >= 1) return trim(mm) + ' mm';
    return trim(mm * 1000) + ' µm';
  }
  function liveLength(mm) {
    if (mm >= 10) return (mm / 10).toFixed(1) + ' cm';
    if (mm >= 1) return mm.toFixed(2) + ' mm';
    if (mm >= 0.01) return (mm * 1000).toFixed(0) + ' µm';
    return (mm * 1000).toFixed(mm < 0.001 ? 2 : 1) + ' µm';
  }

  // ---------- the ruler ----------
  const rail = $('#rail'), hudValue = $('#hud-value'), hudLabel = $('#hud-label'), hudBar = $('#hud-bar');
  const list = document.createElement('ol');
  const SVGNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(SVGNS, 'svg');
  svg.setAttribute('aria-hidden', 'true');
  const marker = document.createElement('i'); marker.className = 'marker';
  stops.forEach((s) => {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = '#' + s.id;
    a.innerHTML = `<b>${s.label}</b><span>${stopLength(s.R)}</span>`;
    a.setAttribute('aria-label', `${s.label}, ${stopLength(s.R)}`);
    li.appendChild(a); list.appendChild(li); s.link = a; s.item = li;
  });
  rail.append(svg, list, marker);
  let railH = 0, narrow = false;
  const railY = (mm) => (LOG_TOP - Math.log10(mm)) / (LOG_TOP - LOG_END) * railH;

  function drawRail() {
    const box = rail.getBoundingClientRect();
    railH = box.height;
    svg.textContent = '';
    if (narrow) { stops.forEach((s) => { s.item.style.top = ''; }); return; }
    svg.setAttribute('width', box.width); svg.setAttribute('height', railH + 2);
    svg.setAttribute('viewBox', `0 -1 ${box.width} ${railH + 2}`);
    const axis = box.width - 46;
    const line = (x1, y1, x2, y2, cls) => {
      const l = document.createElementNS(SVGNS, 'line');
      for (const [k, v] of Object.entries({ x1, y1, x2, y2 })) l.setAttribute(k, v.toFixed(1));
      l.setAttribute('class', cls); svg.appendChild(l);
    };
    line(axis, 0, axis, railH, 'axis');
    for (let e = 2; e >= -4; e--) {
      for (let m = 9; m >= 1; m--) {
        const mm = m * Math.pow(10, e);
        if (mm > 100.01 || mm < 0.00049) continue;
        const y = railY(mm), major = m === 1;
        line(axis, y, axis + (major ? 12 : 5), y, major ? 'tick major' : 'tick');
        if (major) {
          const t = document.createElementNS(SVGNS, 'text');
          t.textContent = stopLength(mm);
          t.setAttribute('x', axis + 15); t.setAttribute('y', (y + 3).toFixed(1));
          svg.appendChild(t);
        }
      }
    }
    const endY = railY(0.0005);
    line(axis, endY, axis + 12, endY, 'tick major');
    const end = document.createElementNS(SVGNS, 'text');
    end.textContent = '0.5 µm'; end.setAttribute('x', axis + 15); end.setAttribute('y', (endY + 3).toFixed(1));
    svg.appendChild(end);
    // labels are evenly spaced; a leader takes each one to its true place on the log axis
    const step = railH / (N - 1);
    stops.forEach((s, i) => {
      const y = i * step;
      s.item.style.top = y.toFixed(1) + 'px';
      s.leader = document.createElementNS(SVGNS, 'polyline');
      s.leader.setAttribute('points', `${axis - 34},${y.toFixed(1)} ${axis - 22},${y.toFixed(1)} ${axis - 8},${railY(s.R).toFixed(1)} ${axis},${railY(s.R).toFixed(1)}`);
      s.leader.setAttribute('class', 'leader');
      svg.appendChild(s.leader);
    });
    const hit = document.createElementNS(SVGNS, 'rect');
    hit.setAttribute('x', axis - 6); hit.setAttribute('y', 0);
    hit.setAttribute('width', 46); hit.setAttribute('height', railH);
    hit.setAttribute('class', 'hit');
    svg.appendChild(hit);
    if (activeStop >= 0) stops[activeStop].leader.classList.add('on');
    hit.addEventListener('click', (e) => {
      const y = e.clientY - rail.getBoundingClientRect().top;
      if (onAxisClick) onAxisClick(Math.pow(10, LOG_TOP - y / railH * (LOG_TOP - LOG_END)));
    });
  }
  let onAxisClick = null;

  let activeStop = -1;
  function setActive(i) {
    if (i === activeStop) return;
    if (activeStop >= 0) { stops[activeStop].link.classList.remove('on'); stops[activeStop].link.removeAttribute('aria-current'); if (stops[activeStop].leader) stops[activeStop].leader.classList.remove('on'); }
    activeStop = i;
    const s = stops[i];
    s.link.classList.add('on'); s.link.setAttribute('aria-current', 'location');
    if (s.leader) s.leader.classList.add('on');
    hudLabel.textContent = s.label;
  }
  let lastRead = '';
  function setReadout(R, x) {
    const text = liveLength(R);
    if (text !== lastRead) { hudValue.textContent = text; lastRead = text; }
    marker.style.transform = narrow ? `translate3d(${x.toFixed(1)}px,0,0)` : `translate3d(0,${railY(R).toFixed(1)}px,0)`;
  }

  function readViewport() {
    narrow = innerWidth < 760 || innerWidth / innerHeight < 0.95;
    root.classList.toggle('narrow', narrow);
    hudBar.style.width = (Math.min(innerWidth, innerHeight) / 2).toFixed(0) + 'px';
  }

  // ---------- the plain page: no WebGL, reduced motion, or a failure on the way ----------
  function plainPage() {
    root.classList.remove('dive', 'paper');
    readViewport(); drawRail();
    document.getElementById('track').style.height = '0';
    for (const s of stops) {
      s.el.style.cssText = ''; if (s.copy) s.copy.style.transform = '';
      for (const p of s.plates) p.style.transform = '';
      for (const img of s.images) img.loading = 'lazy';
      showImages(s);
    }
    setActive(0); setReadout(stops[0].R, 0);
    const seen = new IntersectionObserver((es) => {
      for (const e of es) if (e.isIntersecting) {
        const s = stops.find((x) => x.el === e.target);
        setActive(s.i);
        setReadout(s.R, (s.i + 0.5) / N * innerWidth);
      }
    }, { rootMargin: '-45% 0px -45% 0px' });
    stops.forEach((s) => seen.observe(s.el));
    addEventListener('resize', () => { readViewport(); drawRail(); setReadout(stops[Math.max(0, activeStop)].R, (activeStop + 0.5) / N * innerWidth); });
  }

  if (!root.classList.contains('dive')) { plainPage(); return; }

  let dead = false, needDraw = true;
  const stage = window.createStage($('#stage'), {
    assets: 'assets/',
    onReady: () => { root.classList.add('ready'); needDraw = true; },
    onChange: () => { needDraw = true; },
    onLost: () => { dead = true; plainPage(); },
    onHistology: () => { printTarget(); },
    onFine: (mm) => { tagH.textContent = `Nissl histology, ${Math.round(mm * 1000)} µm`; tagM.textContent = 'T1 MRI, 0.29 × 0.29 × 0.5 mm'; },
    mri: 'assets/volumes/dharani_24gw_mri.bin.gz',
  });
  if (!stage) { plainPage(); return; }

  // ---------- keyframes ----------
  const track = (keys) => (t) => {
    if (t <= keys[0][0]) return keys[0][1];
    for (let i = 0; i < keys.length - 1; i++) {
      if (t < keys[i + 1][0]) return lerp(keys[i][1], keys[i + 1][1], smooth(keys[i][0], keys[i + 1][0], t));
    }
    return keys[keys.length - 1][1];
  };
  // stops: 0 hero 1 about 2 fetal 3 whereismyslice 4 registration 5 one section 6 eminence 7 migrating
  //        8 cytoarchitecture 9 myelin 10 cells 11 playground 12 publications 13 contact
  const K = {
    az: track([[0, -38], [0.5, -30], [1, -8], [1.5, -2], [2, -14], [2.5, -24], [3, -32], [3.55, -32], [3.95, 52], [4.22, 52], [4.3, 22], [4.5, 8], [4.72, 0]]),
    el: track([[0, 16], [0.5, 15], [1, 22], [2, 14], [3, 16], [3.55, 16], [3.95, 20], [4.22, 20], [4.3, 12], [4.5, 5], [4.72, 0]]),
    cut: track([[0.5, 1], [0.9, 0]]),
    hd: track([[0.5, 1], [0.9, 0]]),
    clip: track([[4.46, 0], [4.5, 1]]),
    onPlane: track([[4.5, 0], [4.72, 1]]),          // look-at point: middle of the box -> middle of the section
    // position on the chain BigBrain 24 22 21 17 14 17 21 22 24
    v: track([[1.56, 0], [1.8, 1], [1.84, 1], [1.87, 2], [1.9, 3], [1.93, 4], [1.97, 5], [2.04, 5], [2.12, 6], [2.22, 7], [2.32, 8], [2.44, 9]]),
    fold: track([[1.5, 0], [1.6, 1], [2.44, 1], [2.62, 0]]),
    volW: track([[4.82, 1], [4.9, 0]]),
    faceW: track([[4.76, 0], [4.84, 1], [4.9, 1], [4.98, 0]]),
    nisW: track([[4.88, 0], [4.97, 1]]),
    roll: track([[5.5, 0], [6, 1]]),                // the section turns upright on the way in
    glide: track([[5.5, 0], [6.5, 1]]),             // the target cell drifts to the clear side of the screen
    side: track([[2.55, 1], [2.85, -0.12], [4.47, -0.12], [4.5, 1], [11.5, 1], [12, 0]]),
    benchY: track([[2.6, 0], [2.9, 0.055], [4.47, 0.055], [4.5, 0]]),
    sideN: track([[2.55, 1], [2.85, 0.22], [4.47, 0.22], [4.5, 1], [11.5, 1], [12, 0]]),
    scrim: track([[0.6, 0], [1, 0.5], [2.5, 0.5], [2.8, 0], [4.45, 0], [4.8, 0.8], [11.4, 0.85], [12, 0]]),
    dim: track([[5, 0], [5.15, 0.45], [5.5, 0.45], [5.75, 0], [7.8, 0], [8, 0.6], [8.5, 0.6], [8.6, 0.3], [8.84, 0.3], [9.02, 1], [9.56, 1], [9.66, 0.3], [9.76, 0.3], [9.85, 0]]),
    stretch: track([[11.55, 0], [11.98, 1]]),
    // the bridge: fibres grow out of the cells in the stain's violet, turn to direction colours, then draw back in
    fibres: track([[8.54, 0], [8.6, 1], [9.66, 1], [9.8, 0]]),
    fibreGrow: track([[8.58, 0], [8.94, 1], [9.56, 1], [9.76, 0]]),
    fibreTint: track([[8.74, 1], [9.0, 0], [9.56, 0], [9.7, 1]]),
    bridge: track([[8.62, 0], [8.67, 1], [8.78, 1], [8.83, 0]]),
    // the bench: t 2.7 .. 4.5
    histX: track([[2.72, 0], [3, -41], [3.55, -41], [3.85, 0], [4.26, 0], [4.31, -41], [4.355, -41], [4.39, 0]]),
    mriX: track([[2.72, 34], [3, 41], [3.55, 41], [3.85, 150], [4.24, 150], [4.31, 41], [4.355, 41], [4.39, 0]]),
    mriOn: track([[2.72, 0], [2.92, 1], [3.6, 1], [3.8, 0], [4.25, 0], [4.29, 1], [4.468, 1], [4.484, 0]]),
    play: track([[2.92, 0], [3.05, 1], [3.5, 1], [3.7, 0]]),
    wmsCut: track([[2.9, 0], [3.08, 1], [3.52, 1], [3.72, 0]]),
    regCut: track([[4.385, 0], [4.405, 1]]),
    glow: track([[2.95, 0], [3.1, 1], [3.5, 1], [3.68, 0]]),
    blockFace: track([[3.62, 1], [3.9, 0]]),        // 1 stained, 0 as the frozen block
    blade: track([[4.06, 0], [4.16, 1]]),
    knife: track([[4.04, 0], [4.06, 1], [4.16, 1], [4.172, 0]]),
    gap: track([[4.06, 2.3], [4.2, 2.3], [4.245, 0]]),
    jitter: track([[4.06, 1], [4.245, 1], [4.275, 0]]),
    wave: track([[4.14, 0], [4.21, 1]]),
    checker: track([[4.395, 0], [4.415, 1], [4.468, 1], [4.484, 0]]),
    mis: track([[4.262, 0], [4.285, 1], [4.42, 1], [4.455, 0]]),
  };
  // what the registration scene is doing, in words, where it starts
  const STEPS = [
    [3.9, 'Block face', 'The whole brain in its block. Illustrated: BFI images are not in this demo, so the histology volume is recoloured.'],
    [4.05, 'Sectioning', 'The brain is cut into sections. Each one is mounted on its own, so each one shifts a little. Shown here as 24 slabs.'],
    [4.145, 'Staining', 'Every section turns the violet of the Nissl stain.'],
    [4.21, 'Stacking', 'The sections go back together, still out of line, then slice-to-slice alignment pulls them straight.'],
    [4.285, 'Meet the MRI', 'The same brain as an MRI scan, in its native space: the fixed reference the sections are matched to.'],
    [4.39, 'Overlay', 'A 3D checkerboard: alternate cells are MRI. Out of line, the folds break at every cell edge. This starting error is added to the histology for the demo.'],
    [4.43, 'Registration', 'The MRI stays fixed in its native space. The histology moves onto it.'],
    [4.456, 'Registered', 'Folds and ventricles run straight across the cells: histology and MRI share one space. Mask Dice 0.96, measured on these volumes.'],
  ];
  const ARM = 0.62;

  function barLength(t) {                           // R(t), millimetres
    const i = Math.min(N - 1, Math.floor(t)), f = t - i;
    const here = stops[i].R, creep = i < N - 2 ? 0.97 : 1;
    if (f <= 0.5 || i === N - 1) return here * lerp(1, creep, Math.min(1, f * 2));
    const e = smoother((f - 0.5) * 2);
    return Math.pow(10, lerp(Math.log10(here * creep), Math.log10(stops[i + 1].R), e));
  }
  const END_T = N - 1 + 0.5;

  // ---------- scroll <-> t ----------
  let W = 1, H = 1, S = 1, sideVec = [0, 0];
  let total = 0;
  const trackEl = $('#track');

  function measure() {
    W = innerWidth; H = innerHeight; S = Math.min(W, H);
    readViewport();
    sideVec = narrow ? [0, -0.2 * H / S] : [0.15 * W / S, 0];
    stage.resize(W, H);
    drawRail();
    const padT = narrow ? 56 : 44, padB = narrow ? 112 : 124;
    const avail = H - padT - padB;
    let y = 0;
    stops.forEach((s, i) => {
      s.copyH = s.copy ? s.copy.offsetHeight : 0;
      s.padT = padT; s.padB = padB; s.avail = avail;
      s.y0 = y;
      s.P = H * s.plateau + (s.copy && s.copy.classList.contains('top') ? 0 : Math.max(0, s.copyH - avail) * 1.4);
      y += s.P;
      if (i < N - 1) {
        const decades = Math.abs(Math.log10(s.R / stops[i + 1].R));
        s.T = H * (0.7 + 1.0 * decades + s.transit);
        y += s.T;
      } else s.T = 0;
      s.el.style.transformOrigin = `${(W / 2 + sideVec[0] * S).toFixed(0)}px ${(H / 2 + sideVec[1] * S).toFixed(0)}px`;
    });
    total = y;
    trackEl.style.height = (total + H) + 'px';
  }

  function tOfScroll(y) {
    y = clamp(y, 0, total);
    for (let i = N - 1; i >= 0; i--) {
      const s = stops[i];
      if (y >= s.y0) {
        const d = y - s.y0;
        if (d <= s.P) return i + 0.5 * d / s.P;
        return i + 0.5 + 0.5 * (d - s.P) / s.T;
      }
    }
    return 0;
  }
  function scrollOfT(t) {
    const i = Math.min(N - 1, Math.floor(t)), f = t - i, s = stops[i];
    if (f <= 0.5) return s.y0 + s.P * f * 2;
    return s.y0 + s.P + s.T * (f - 0.5) * 2;
  }
  function tOfBar(R) {                              // bisection: R(t) falls monotonically
    let a = 0, b = END_T;
    for (let k = 0; k < 40; k++) { const m = (a + b) / 2; if (barLength(m) > R) a = m; else b = m; }
    return b;
  }

  // ---------- jumps ----------
  let flight = null;
  function fly(t1, dest) {
    const t0 = tOfScroll(scrollY);
    const span = Math.abs(t1 - t0);
    if (span < 0.01) return;
    flight = { t0, t1, dest, start: performance.now(), dur: clamp(900 + 420 * span, 1100, 4600) };
  }
  const land = () => { flight = null; };
  for (const ev of ['wheel', 'touchstart']) addEventListener(ev, land, { passive: true });
  addEventListener('keydown', (e) => { if (['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '].includes(e.key)) land(); });
  document.addEventListener('click', (e) => {
    const a = e.target.closest && e.target.closest('a[href^="#"]');
    if (!a || dead) return;
    const s = stops.find((x) => '#' + x.id === a.getAttribute('href'));
    if (!s) return;
    e.preventDefault();
    fly(s.i, s.i);
  });
  onAxisClick = (R) => fly(tOfBar(R), -1);

  // ---------- what is on screen at t ----------
  function placeCopy(s, hold) {
    if (!s.copy) return;
    const h = s.copyH, bottom = H - s.padB;
    let y;
    if (s.copy.classList.contains('top')) y = s.padT;
    else if (h <= s.avail) y = narrow ? bottom - h : s.padT + (s.avail - h) / 2;
    else y = s.padT - (h - s.avail) * clamp((hold - 0.06) / 0.86);
    s.copy.style.transform = `translate3d(0,${y.toFixed(1)}px,0)`;
  }

  const gwValue = $('#gw-value'), gwItems = [...document.querySelectorAll('.gw li')], CHAIN_WEEKS = [24, 24, 22, 21, 17, 14, 17, 21, 22, 24];
  let lastGw = '', lastStep = null;
  const regTitle = $('#reg-title'), regText = $('#reg-text');
  const fibres = window.createFibres($('#fibres'), 'assets/img/myelin/streamlines_section.webp');
  const bridgeEl = $('#bridge');

  function updatePage(t, F) {
    const IN = 0.2, OUT = 0.2;
    for (const s of stops) {
      const i = s.i;
      if (Math.abs(t - i) < 2.2) showImages(s);
      const ein = i === 0 ? 1 : smooth(i - IN, i, t);
      const eout = i === N - 1 ? 0 : smooth(i + 0.5, i + 0.5 + OUT, t);
      let op = ein * (1 - eout);
      if (flight && flight.dest !== i) op = 0;
      if (op < 0.004) {
        if (s.shown) { s.el.style.visibility = 'hidden'; s.shown = false; s.opacity = -1; }
        continue;
      }
      if (!s.shown) { s.el.style.visibility = 'visible'; s.shown = true; }
      // the page moves with the camera: it arrives from the distance and passes the lens
      const scale = lerp(0.7, 1, ein) * lerp(1, 2.1, eout * eout);
      s.el.style.transform = `scale(${scale.toFixed(4)})`;
      if (Math.abs(op - s.opacity) > 0.002) { s.el.style.opacity = op.toFixed(3); s.opacity = op; }
      s.el.classList.toggle('live', op > 0.6);
      const hold = clamp((t - i) * 2);
      placeCopy(s, hold);
      for (const p of s.plates) p.style.transform = `scale(${(1 + 0.07 * hold).toFixed(4)})`;
      if (s.lens) {
        // same magnification as the view behind it
        const glass = s.lens.querySelector('.lens-glass');
        const k = Math.max(1, (parseFloat(s.lens.dataset.mm) / F * S) / (glass.clientWidth || 1));
        s.lens.querySelector('img').style.transform = `scale(${k.toFixed(4)})`;
      }
    }
    const v = K.v(t);
    const gw = CHAIN_WEEKS[clamp(Math.round(v), 0, 9)];
    if (String(gw) !== lastGw) {
      lastGw = String(gw); gwValue.textContent = lastGw;
      gwItems.forEach((li) => li.classList.toggle('on', +li.dataset.gw <= gw));
    }
    let step = STEPS[0];
    for (const x of STEPS) if (t >= x[0]) step = x;
    if (step !== lastStep) { lastStep = step; regTitle.textContent = step[1]; regText.textContent = step[2]; }
    const hasFibres = K.fibres(t);
    fibres.set(hasFibres, t, F / 2, K.fibreGrow(t), K.fibreTint(t));
    const bridge = flight ? 0 : K.bridge(t);
    bridgeEl.style.opacity = bridge.toFixed(3);
    bridgeEl.style.visibility = bridge > 0.004 ? 'visible' : 'hidden';
    setActive(clamp(Math.floor(t + 0.25), 0, N - 1));
    const whole = Math.floor(t), part = t - whole;
    const stepped = whole + smooth(0.5, 1, part);
    root.classList.toggle('paper', K.stretch(t) > 0.5);
    root.classList.toggle('underway', t > 0.12);
    setReadout(barLength(t), (stepped + 0.5) / N * W);
  }

  const rotate = (a, x, y) => [Math.cos(a) * x - Math.sin(a) * y, Math.sin(a) * x + Math.cos(a) * y];

  function camera(t, now) {
    const R = barLength(t), F = 2 * R;
    const side = narrow ? K.sideN(t) : K.side(t);
    const off = [sideVec[0] * side, sideVec[1] * side];
    const roll = K.roll(t) * stage.theta();
    const centre = stage.sectionCentre(), goal = stage.cellTarget();
    let view;
    if (t <= 5) {
      view = [centre[0] - off[0] * F, centre[1] - off[1] * F];
    } else {
      // from stop 4 on, the camera is defined by where the target cell sits on screen; that keeps the zoom
      // pure (one fixed point) and lands on the final pixel exactly
      const F4 = 2 * barLength(5);
      const from = [sideVec[0] + (goal[0] - centre[0]) / F4, sideVec[1] + (goal[1] - centre[1]) / F4];
      const g = K.glide(t);
      const onScreen = [lerp(from[0], off[0], g), lerp(from[1], off[1], g)];
      const r = rotate(roll, onScreen[0] * F, onScreen[1] * F);
      view = [goal[0] - r[0], goal[1] - r[1]];
    }
    const stretch = K.stretch(t);
    const alive = !STILL;
    let driftAz = 0;
    if (alive) {
      const clip = K.clip(t);
      driftAz = Math.sin(now * 0.00012) * 6 * (1 - clip);
      if (t > 4.98) {                               // under the lens the slide never sits perfectly still
        const a = now * 0.00021, k = F * 0.0035 * (1 - stretch);
        view = [view[0] + Math.cos(a) * k, view[1] + Math.sin(a * 1.3) * k];
      }
    }
    const on = K.onPlane(t);
    const play = K.play(t);
    return {
      F, roll, view, narrow, stretch,
      offX: off[0], offY: off[1] + (narrow ? 0 : K.benchY(t)),
      az: K.az(t) + driftAz + orbit.az * play, el: clamp(K.el(t) + orbit.el * play, -70, 70),
      fold: K.fold(t),
      bench: t > 2.7 && t < 4.5 ? benchState(t) : null,
      target: [stage.planeX() * on, -centre[1] * on, centre[0] * on],
      cut: K.cut(t), hd: K.hd(t), clip: K.clip(t), v: K.v(t),
      volW: K.volW(t), faceW: K.faceW(t), nisW: K.nisW(t),
      dim: K.dim(t), scrim: K.scrim(t),
    };
  }

  // ---------- the blade ----------
  const rad = (d) => d * Math.PI / 180, cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const orbit = { az: 0, el: 0 };
  const blade = { rot: -90, tilt: 0, pos: -6 };
  const TARGET = { rot: -68, tilt: 14, pos: 13 };        // an oblique cut through the eminence, not an easy one
  function axes(pl) {
    const r = rad(pl.rot), a = rad(pl.tilt);
    const n = [Math.sin(r) * Math.cos(a), Math.sin(a), Math.cos(r) * Math.cos(a)];
    const u = [Math.cos(r), 0, -Math.sin(r)];
    return { n, u, v: cross(n, u), d: pl.pos };
  }
  // mean distance between the two planes across the brain
  function gapMm(a, b) {
    const A = axes(a), B = axes(b);
    let nb = B.n, db = B.d;
    if (dot(A.n, nb) < 0) { nb = nb.map((x) => -x); db = -db; }
    const c = A.n.map((x) => x * A.d);
    let sum = Math.abs(dot(nb, c) - db), n = 1;
    for (let ring = 1; ring <= 3; ring++) for (let k = 0; k < 8; k++) {
      const r = 30 * ring / 3, ph = (k + ring * 0.5) * Math.PI / 4;
      const p = c.map((x, i) => x + A.u[i] * r * Math.cos(ph) + A.v[i] * r * Math.sin(ph));
      sum += Math.abs(dot(nb, p) - db); n++;
    }
    return sum / n;
  }
  const inputs = { pos: $('#rPos'), tilt: $('#rTilt'), rot: $('#rTurn') };
  const pctEl = $('#pct'), gapEl = $('#gap');
  let found = false;
  function readBlade() {
    const g = gapMm(blade, TARGET), pct = Math.round(Math.max(0, 100 * (1 - g / 34)));
    pctEl.textContent = pct;
    const hit = g < 2.5;
    if (hit !== found) { found = hit; $('#bench-panel').classList.toggle('found', hit); }
    gapEl.textContent = hit ? `Found it: ${g.toFixed(1)} mm from the true plane, and the MRI shows the same anatomy.` : `${g.toFixed(1)} mm away. Same blade, both volumes.`;
    needDraw = true;
  }
  const syncInputs = () => { inputs.pos.value = blade.pos; inputs.tilt.value = blade.tilt; inputs.rot.value = blade.rot; };
  for (const k of ['pos', 'tilt', 'rot']) inputs[k].addEventListener('input', () => { blade[k] = +inputs[k].value; show = null; readBlade(); });
  function printTarget() {
    const A = axes(TARGET);
    stage.printSection({ n: A.n, u: A.u, v: A.v, d: A.d }, $('#target'));
  }
  let show = null;
  $('#showme').addEventListener('click', () => { show = { from: { ...blade }, start: performance.now() }; });
  function stepShow(now) {
    if (!show) return;
    const p = clamp((now - show.start) / 1400), e = smoother(p);
    for (const k of ['pos', 'tilt', 'rot']) blade[k] = lerp(show.from[k], TARGET[k], e);
    syncInputs(); readBlade();
    if (p >= 1) show = null;
  }
  // on a brain the pointer moves the blade along its normal; off it, the pointer turns both brains
  const hitArea = $('#bench-hit');
  let drag = null, lastCam = null;
  hitArea.addEventListener('pointerdown', (e) => {
    if (!lastCam) return;
    const c = stage.aim(lastCam), A = axes(blade);
    const nx = dot(A.n, c.right), ny = dot(A.n, c.up);
    const pxPerMm = S / lastCam.F;
    const cx = W / 2 + lastCam.offX * S, cy = H / 2 + lastCam.offY * S;
    const reach = 40 * pxPerMm, sep = 41 * pxPerMm;
    const dist = narrow ? Math.min(Math.hypot(e.clientX - cx, e.clientY - cy - sep), Math.hypot(e.clientX - cx, e.clientY - cy + sep))
      : Math.min(Math.hypot(e.clientX - cx - sep, e.clientY - cy), Math.hypot(e.clientX - cx + sep, e.clientY - cy));
    drag = { x: e.clientX, y: e.clientY, onBrain: dist < reach, nx, ny, pos: blade.pos, az: orbit.az, el: orbit.el, pxPerMm };
    hitArea.setPointerCapture(e.pointerId); show = null;
    hitArea.classList.add('dragging');
  });
  hitArea.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (drag.onBrain) {
      const m = drag.nx * drag.nx + drag.ny * drag.ny;
      const along = m > 0.05 ? (dx * drag.nx - dy * drag.ny) / m : -dy;
      blade.pos = clamp(drag.pos + along / drag.pxPerMm * (m > 0.05 ? 1 : 0.5), -38, 38);
      syncInputs(); readBlade();
    } else { orbit.az = drag.az + dx * 0.35; orbit.el = drag.el + dy * 0.25; needDraw = true; }
  });
  const endDrag = () => { drag = null; hitArea.classList.remove('dragging'); };
  hitArea.addEventListener('pointerup', endDrag); hitArea.addEventListener('pointercancel', endDrag);

  const tagH = $('#tag-hist'), tagM = $('#tag-mri'), regStop = $('#registration');
  function placeTags(cam) {
    const px = S / cam.F, cx = W / 2 + cam.offX * S, cy = H / 2 + cam.offY * S, sep = (narrow ? 37 : 41) * px, top = 44 * px;
    const at = (el, x, y) => { el.style.transform = `translate3d(${x.toFixed(0)}px,${y.toFixed(0)}px,0) translateX(-50%)`; };
    if (narrow) { at(tagH, 62, cy - sep - top * 0.8); at(tagM, 62, cy + sep - top * 0.8); }
    else { at(tagH, cx - sep, cy - top); at(tagM, cx + sep, cy - top); }
  }

  function benchState(t) {
    const wms = K.wmsCut(t), reg = K.regCut(t), half = stage.halfMm();
    const A = wms > 0.001 ? axes(blade) : { n: [1, 0, 0], u: [0, 1, 0], v: [0, 0, 1], d: stage.geom.clipx * half[0] };
    const gone = K.blade(t) > 0;
    return {
      histX: K.histX(t) * (narrow ? 0.9 : 1), mriX: K.mriX(t) * (narrow ? 0.9 : 1), mriOn: K.mriOn(t),
      cut: { ...A, amount: wms > 0.001 ? wms : reg, glow: K.glow(t) },
      blade: K.blade(t), gap: K.gap(t), jitter: gone ? K.jitter(t) : 0,
      wave: t < 4.14 ? K.blockFace(t) : K.wave(t),
      knife: K.knife(t), checker: K.checker(t), mis: K.mis(t),
    };
  }

  // ---------- loop ----------
  let t = 0, lastNow = 0, lastDrawn = 0, slow = 0, frameEma = 16, quality = 1, settling = false;

  function frame(now) {
    if (dead) return;
    requestAnimationFrame(frame);
    if (document.hidden) { lastNow = now; return; }
    const dt = Math.min(64, now - lastNow || 16);
    lastNow = now;

    if (flight) {
      const p = clamp((now - flight.start) / flight.dur);
      const e = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
      scrollTo(0, scrollOfT(lerp(flight.t0, flight.t1, e)));
      if (p >= 1) flight = null;
    }
    const goal = tOfScroll(scrollY);
    const moving = Math.abs(goal - t) > 0.00008;
    t = moving ? t + (goal - t) * (1 - Math.exp(-dt / 110)) : goal;

    stepShow(now);
    if (fibres.running()) needDraw = true;
    const idle = !STILL && t < N - 2;
    if (!moving && !needDraw && !settling && !idle) return;
    if (!moving && !needDraw && !settling && now - lastDrawn < 45) return;    // about 20 fps at rest
    needDraw = false; lastDrawn = now;

    const cam = camera(t, now);
    lastCam = cam;
    updatePage(t, cam.F);
    fibres.draw(now);
    placeTags(cam);
    regStop.classList.toggle('compact', t > 4.06);
    settling = stage.draw(cam, now) || false;

    // keep it smooth on slow machines, measured on the real frame rhythm while moving
    if (moving) {
      frameEma = frameEma * 0.9 + dt * 0.1;
      if (frameEma > 34 && cam.volW > 0.01 && quality > 0.5) {
        if (++slow > 24) { quality *= 0.85; slow = 0; stage.setQuality(quality); }
      } else slow = 0;
    }
  }

  function relayout() {
    const keep = tOfScroll(scrollY);
    measure();
    scrollTo(0, scrollOfT(keep));
    t = keep; needDraw = true;
  }

  let resizeTimer = 0, lastW = innerWidth, lastH = innerHeight;
  addEventListener('resize', () => {
    // a phone's collapsing address bar changes the height by a little: not worth moving the page for
    const small = innerWidth === lastW && Math.abs(innerHeight - lastH) < 130;
    if (small && matchMedia('(pointer: coarse)').matches) return;
    lastW = innerWidth; lastH = innerHeight;
    clearTimeout(resizeTimer); resizeTimer = setTimeout(relayout, 120);
  });

  measure();
  const start = stops.find((s) => '#' + s.id === location.hash);
  if (start) scrollTo(0, scrollOfT(start.i));
  t = tOfScroll(scrollY);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(relayout);
  addEventListener('load', relayout);
  readBlade(); syncInputs();
  const saveData = navigator.connection && navigator.connection.saveData;
  stage.load(narrow || saveData ? null : 'assets/volumes/hist24/').catch((e) => { console.warn('stage:', e); dead = true; plainPage(); });
  requestAnimationFrame(frame);

  // for the browser checks: __dive.go(4), __dive.t(3.85), __dive.state()
  window.__dive = {
    go: (i) => fly(i, i),
    t: (x) => { flight = null; scrollTo(0, scrollOfT(x)); },
    state: () => ({ t, R: barLength(t), quality, total, blade: { ...blade }, pct: pctEl.textContent }),
    geom: stage.geom,
  };
})();
