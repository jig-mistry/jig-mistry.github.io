/* Scroll pieces that are not the background brain:
   - .zoomer: a tall block whose sticky canvas zooms through nested image levels (whole section -> cells)
   - .scrubber: a tall block whose sticky canvas steps through a pre-rendered frame sequence
   - the gestational-age ruler of the fetal chapter
   Each zoom level is an image placed at a rectangle given in [0, 1] coordinates of the coarsest level. */
(() => {
  'use strict';
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');

  // progress of a tall block through the viewport: 0 when its top reaches the top, 1 when its bottom leaves
  function progress(block) {
    const r = block.getBoundingClientRect();
    const span = r.height - innerHeight;
    return span > 0 ? Math.min(1, Math.max(0, -r.top / span)) : 0;
  }
  const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

  function fitCanvas(canvas) {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    return [w, h];
  }

  // ---------- zoomers ----------
  function niceLength(mm) {           // a round scale-bar length near mm
    const p = Math.pow(10, Math.floor(Math.log10(mm)));
    for (const m of [5, 2, 1]) if (m * p <= mm) return m * p;
    return p;
  }

  async function setupZoomer(fig) {
    const spec = await (await fetch(fig.dataset.zoom)).json();
    const canvas = fig.querySelector('canvas');
    const ctx = canvas.getContext('2d');
    const widthMm = parseFloat(fig.dataset.mm) || 0;
    const bar = fig.querySelector('.scalebar');
    const levels = spec.levels.map((l) => ({ ...l, img: null }));
    levels.forEach((l, i) => {
      const img = new Image();
      img.decoding = 'async';
      img.onload = () => { l.img = img; draw(); };
      // the coarse level first, the rest once the block is near
      if (i === 0) img.src = l.src; else l.pending = img;
    });
    const near = new IntersectionObserver((es) => {
      if (es.some((e) => e.isIntersecting)) {
        levels.forEach((l) => { if (l.pending) { l.pending.src = l.src; l.pending = null; } });
        near.disconnect();
      }
    }, { rootMargin: '150% 0px' });
    near.observe(fig);

    const A = spec.aspect;                       // width / height of the coarsest level
    const last = levels[levels.length - 1].rect;
    const c1 = [(last[0] + last[2]) / 2, (last[1] + last[3]) / 2];
    const w1 = last[2] - last[0];

    function draw() {
      const [W, H] = fitCanvas(canvas);
      const p = reduced.matches ? 0 : smooth(0.08, 0.92, progress(fig.closest('.zoom-block')));
      // work in level-0 width units (the level-0 image is 1 wide and 1/A tall);
      // the view width shrinks geometrically and the centre glides towards the last level with it
      const w0 = Math.max(1, W / (H * A));            // the whole section fits the stage
      const vw = w0 * Math.pow(w1 / w0, p);
      const k = (w0 - vw) / Math.max(1e-6, w0 - w1);
      const cx = 0.5 + (c1[0] - 0.5) * k, cy = (0.5 + (c1[1] - 0.5) * k) / A;
      const vh = vw * H / W;
      const s = W / vw;                               // canvas px per unit
      const x0 = cx - vw / 2, y0 = cy - vh / 2;
      ctx.clearRect(0, 0, W, H);                     // tissue only: the glass is transparent in the images
      // the frame around the view fades in as the zoom starts, so the zoomed view reads as a window
      const frame = smooth(0.15, 0.45, p);
      fig.style.setProperty('--frame', frame.toFixed(3));
      fig.classList.toggle('framed', frame > 0.5);
      ctx.imageSmoothingQuality = 'high';
      for (const l of levels) {
        if (!l.img) continue;
        const [a, b, c, d] = l.rect;
        const cover = (c - a) / vw;                   // fraction of the view this level spans
        const alpha = l === levels[0] ? 1 : smooth(0.12, 0.45, cover);
        if (alpha <= 0.01) continue;
        ctx.globalAlpha = alpha;
        ctx.drawImage(l.img, (a - x0) * s, (b / A - y0) * s, (c - a) * s, (d - b) / A * s);
      }
      ctx.globalAlpha = 1;
      if (bar && widthMm) {
        const mmPerPx = widthMm * vw / canvas.clientWidth;     // CSS pixels
        const len = niceLength(mmPerPx * canvas.clientWidth * 0.18);
        bar.querySelector('i').style.width = (len / mmPerPx).toFixed(1) + 'px';
        bar.querySelector('b').textContent = len >= 1 ? `${len} mm` : `${Math.round(len * 1000)} µm`;
      }
    }
    fig._draw = draw;
    draw();
  }

  // ---------- frame scrubbers ----------
  async function setupScrubber(fig) {
    const url = fig.dataset.frames;
    let spec;
    try { spec = await (await fetch(url)).json(); } catch (e) { fig.closest('.scrub-block').hidden = true; return; }
    const base = url.slice(0, url.lastIndexOf('/') + 1);
    const canvas = fig.querySelector('canvas');
    const ctx = canvas.getContext('2d');
    const frames = new Array(spec.frames).fill(null);
    canvas.style.aspectRatio = `${spec.width} / ${spec.height}`;
    const name = (i) => spec.pattern.replace('%03d', String(i).padStart(3, '0'));
    let loading = false;
    function loadAll() {
      if (loading) return; loading = true;
      // every 8th frame first so a coarse scrub works at once, then the rest
      const order = [...Array(spec.frames).keys()].sort((a, b) => (a % 8) - (b % 8) || a - b);
      for (const i of order) {
        const img = new Image();
        img.onload = () => { frames[i] = img; draw(); };
        img.src = base + name(i);
      }
    }
    new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) loadAll(); }, { rootMargin: '150% 0px' }).observe(fig);
    function draw() {
      const [W, H] = fitCanvas(canvas);
      const p = reduced.matches ? 0 : smooth(0.05, 0.95, progress(fig.closest('.scrub-block')));
      let i = Math.round(p * (spec.frames - 1));
      while (i > 0 && !frames[i]) i--;                  // nearest loaded frame at or before
      ctx.clearRect(0, 0, W, H);
      if (frames[i]) ctx.drawImage(frames[i], 0, 0, W, H);
    }
    fig._draw = draw;
  }

  // ---------- gestational-age ruler ----------
  const gwSteps = [...document.querySelectorAll('#fetal .step[data-gw]')];
  const gwValue = document.querySelector('.gw-value');
  const gwItems = [...document.querySelectorAll('.gw-ruler li')];
  function updateGw() {
    if (!gwSteps.length) return;
    let gw = gwSteps[0].dataset.gw;
    for (const s of gwSteps) if (s.getBoundingClientRect().top < innerHeight * 0.55) gw = s.dataset.gw;
    if (gwValue.textContent !== gw) gwValue.textContent = gw;
    gwItems.forEach((li) => li.classList.toggle('on', +li.dataset.gw <= +gw));
  }

  const figs = [...document.querySelectorAll('.zoomer')].map((f) => (setupZoomer(f), f))
    .concat([...document.querySelectorAll('.scrubber')].map((f) => (setupScrubber(f), f)));
  let queued = false;
  function frame() {
    queued = false;
    for (const f of figs) {
      const r = f.getBoundingClientRect();
      if (f._draw && r.bottom > -innerHeight && r.top < innerHeight * 2) f._draw();
    }
    updateGw();
  }
  const ask = () => { if (!queued) { queued = true; requestAnimationFrame(frame); } };
  addEventListener('scroll', ask, { passive: true });
  addEventListener('resize', ask);
  ask();
})();
