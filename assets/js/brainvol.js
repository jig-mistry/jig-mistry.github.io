/* Scroll-driven volume rendering of whole brains behind the page.

   Each element carrying data-scene="{...}" is a keyframe.  As the page scrolls, the scene is
   interpolated between the keyframe whose anchor was passed last and the next one, and the
   brain is ray-marched in WebGL2 from a 3D texture.

   Volumes (assets/volumes/volumes.json) are uint8 pairs per voxel: R = stain density,
   G = depth below the outer surface (so the outside can be drawn as a grey shell and the
   inside in the stain colour).  Axes: x left->right, y inferior->superior, z posterior->anterior.

   Scene keys (all optional, missing ones carry over from the previous keyframe):
     vol    volume id                 az, el   camera azimuth / elevation, degrees
     dist   camera distance (x 100 mm) sx, sy   where the brain centre sits on screen (NDC)
     cut    0..1 octant cut-away       clip     0..1 fraction of the clip slab removed
     clipx  clip plane position along x, as a fraction of the half width (-1..1)
     tx, ty, tz  look-at point, as a fraction of the half size     fade  0..1 opacity
*/
(() => {
  'use strict';
  const canvas = document.getElementById('brainvol');
  if (!canvas) return;
  const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false });
  const root = document.documentElement;
  if (!gl) { root.classList.add('no-webgl'); return; }

  const BASE = canvas.dataset.volumes || 'assets/volumes/';
  const MM = 0.01;                                   // world units per mm
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const still = new URLSearchParams(location.search).has('still');     // no idle drift (screenshots)
  const keys = [...document.querySelectorAll('[data-scene]')];
  const DEFAULTS = { vol: 'bigbrain', az: 30, el: 15, dist: 3.4, sx: 0.4, sy: 0, cut: 0, clip: 0, clipx: -1,
                     tx: 0, ty: 0, tz: 0, fade: 1 };
  const NUM = Object.keys(DEFAULTS).filter((k) => k !== 'vol');

  // resolve keyframes: every key carries the full state
  const frames = [];
  let prev = DEFAULTS;
  for (const el of keys) {
    let s = {};
    try { s = JSON.parse(el.dataset.scene); } catch (e) { console.warn('bad data-scene', el, e); }
    prev = Object.assign({}, prev, s);
    frames.push({ el, s: prev, anchor: 0 });
  }
  if (!frames.length) return;

  function measure() {
    let last = -Infinity;
    for (const f of frames) {
      const r = f.el.getBoundingClientRect();
      // a keyframe is reached when the element's top passes 40% of the viewport height;
      // a hidden element (no box) shares the previous anchor so the order never breaks
      const shown = r.width > 0 || r.height > 0;
      f.anchor = Math.max(shown ? r.top + scrollY - innerHeight * 0.4 : last, last);
      last = f.anchor;
    }
  }

  const ease = (t) => t * t * (3 - 2 * t);
  function sceneAt(y) {
    if (y <= frames[0].anchor) return { a: frames[0].s, b: frames[0].s, t: 0 };
    for (let i = 0; i < frames.length - 1; i++) {
      const f0 = frames[i], f1 = frames[i + 1];
      if (y < f1.anchor) {
        const raw = (y - f0.anchor) / Math.max(1, f1.anchor - f0.anchor);
        // hold each keyframe for a while so the brain is still while its text is read
        const t = ease(Math.min(1, Math.max(0, (raw - 0.35) / 0.6)));
        return { a: f0.s, b: f1.s, t };
      }
    }
    const last = frames[frames.length - 1].s;
    return { a: last, b: last, t: 0 };
  }

  // ---------------- shaders ----------------
  const VS = `#version 300 es
in vec2 aPos; out vec2 vNdc;
void main() { vNdc = aPos; gl_Position = vec4(aPos, 0.0, 1.0); }`;

  const FS = `#version 300 es
precision highp float;
precision highp sampler3D;
uniform sampler3D uVolA, uVolB;
uniform vec3 uHalfA, uHalfB;        // hs size of each volume, world units
uniform float uMix;                  // 0 = A only, 1 = B only
uniform int uPalA, uPalB;            // 0 = teal (BigBrain), 1 = Nissl violet
uniform vec3 uEye, uRight, uUp, uFwd;
uniform vec2 uTan;                   // tan(fov / 2) * aspect, tan(fov / 2)
uniform vec2 uShift;
uniform float uCut;                  // octant cut-away, 0..1
uniform vec3 uCutSign;
uniform float uClip;                 // 0..1 how much of the side facing -x is removed
uniform float uClipX;                // clip plane x, world units
uniform float uStep, uFade;
uniform vec3 uLight;
uniform float uDark;                 // 1 on a dark page
in vec2 vNdc; out vec4 outColor;

float rnd(vec2 p) { return fract(sin(dot(p, vec2(41.31, 289.17))) * 15731.743); }

vec2 fetch(sampler3D v, vec3 hs, vec3 p) {
  vec3 tc = p / (2.0 * hs) + 0.5;
  if (any(lessThan(tc, vec3(0.0))) || any(greaterThan(tc, vec3(1.0)))) return vec2(0.0);
  return texture(v, tc).rg;
}

vec2 sampleAt(vec3 p) {
  vec2 a = uMix < 0.999 ? fetch(uVolA, uHalfA, p) : vec2(0.0);
  if (uMix < 0.001) return a;
  return mix(a, fetch(uVolB, uHalfB, p), uMix);
}

vec3 palette(int pal, float d) {
  if (pal == 0) {          // teal: pale cyan white matter, deep teal cell-rich grey matter
    vec3 lo = vec3(0.78, 0.98, 0.98), mid = vec3(0.22, 0.80, 0.82), hi = vec3(0.02, 0.33, 0.38);
    return d < 0.5 ? mix(lo, mid, d * 2.0) : mix(mid, hi, d * 2.0 - 1.0);
  }
  vec3 lo = vec3(0.93, 0.87, 0.97), mid = vec3(0.66, 0.50, 0.86), hi = vec3(0.30, 0.12, 0.50);
  return d < 0.5 ? mix(lo, mid, d * 2.0) : mix(mid, hi, d * 2.0 - 1.0);
}

bool removed(vec3 p, vec3 hs) {
  if (uCut > 0.001 && all(greaterThan(p * uCutSign, hs * (1.0 - uCut) - vec3(1e-4)))) return true;
  if (uClip > 0.001 && p.x < mix(-hs.x * 1.05, uClipX, uClip)) return true;
  return false;
}

void main() {
  vec2 s = vNdc - uShift;
  vec3 dir = normalize(uFwd + s.x * uTan.x * uRight + s.y * uTan.y * uUp);
  vec3 hs = max(uHalfA * (1.0 - step(0.999, uMix)), uHalfB * step(0.001, uMix));
  vec3 inv = 1.0 / dir;
  vec3 t0 = (-hs - uEye) * inv, t1 = (hs - uEye) * inv;
  vec3 tn3 = min(t0, t1), tf3 = max(t0, t1);
  float tn = max(max(tn3.x, tn3.y), tn3.z), tf = min(min(tf3.x, tf3.y), tf3.z);
  if (tf <= max(tn, 0.0)) { outColor = vec4(0.0); return; }
  float t = max(tn, 0.0) + uStep * rnd(gl_FragCoord.xy);
  vec4 acc = vec4(0.0);
  vec3 shellCol = mix(vec3(0.60, 0.61, 0.63), vec3(0.70, 0.70, 0.72), uDark);
  for (int i = 0; i < 900; i++) {
    if (t > tf || acc.a > 0.97) break;
    vec3 p = uEye + dir * t;
    if (!removed(p, hs)) {
      vec2 v = sampleAt(p);
      float d = v.r;
      if (d > 0.01) {
        float depth = v.g;
        float lit = 1.0;
        if (depth < 0.45) {
          // outer surface normal from the depth field (it grows inwards); only needed near the skin
          float e = 0.006;
          vec3 g = vec3(sampleAt(p + vec3(e, 0, 0)).g, sampleAt(p + vec3(0, e, 0)).g, sampleAt(p + vec3(0, 0, e)).g) - depth;
          float g2 = dot(g, g);
          if (g2 > 1e-6) {
            vec3 n = -g * inversesqrt(g2);
            lit = 0.55 + 0.45 * max(dot(n, uLight), 0.0) + 0.25 * pow(1.0 - abs(dot(n, dir)), 3.0);
          }
        }
        vec3 stain = mix(palette(uPalA, d), palette(uPalB, d), uMix);
        // a thin grey skin outside, the stain colour wherever the brain is cut open
        float skin = 1.0 - smoothstep(0.05, 0.35, depth);
        vec3 col = mix(stain, shellCol, skin * 0.85) * lit;
        float dens = smoothstep(0.01, 0.25, d) * mix(180.0, 420.0, skin);
        float alpha = 1.0 - exp(-dens * uStep);
        acc.rgb += (1.0 - acc.a) * alpha * col;
        acc.a += (1.0 - acc.a) * alpha;
      }
    }
    t += uStep;
  }
  outColor = acc * uFade;
}`;

  function compile(type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    return sh;
  }
  let prog;
  try {
    prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  } catch (e) { console.warn('brainvol:', e); root.classList.add('no-webgl'); return; }
  gl.useProgram(prog);
  const U = {};
  for (let i = 0, n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS); i < n; i++) {
    const name = gl.getActiveUniform(prog, i).name;
    U[name] = gl.getUniformLocation(prog, name);
  }
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, 'aPos');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  gl.uniform1i(U.uVolA, 0); gl.uniform1i(U.uVolB, 1);

  // ---------------- volumes ----------------
  const vols = {};          // id -> {meta, half, tex}
  let empty = null;

  async function gunzip(url) {
    const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
    if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;     // server already decoded it
    const out = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Uint8Array(await new Response(out).arrayBuffer());
  }
  function texture(dims, data) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG8, dims[0], dims[1], dims[2], 0, gl.RG, gl.UNSIGNED_BYTE, data);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
                          [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE],
                          [gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_3D, k, v);
    return tex;
  }

  async function load() {
    empty = texture([1, 1, 1], new Uint8Array(2));
    const index = await (await fetch(BASE + 'volumes.json')).json();
    for (const m of index.volumes) {
      vols[m.id] = { meta: m, half: m.size_mm.map((x) => x * MM / 2), tex: null, pal: m.id === 'bigbrain' ? 0 : 1 };
    }
    // the first scene's brain first, then the rest in page order
    const order = [...new Set(frames.map((f) => f.s.vol))].filter((id) => vols[id]);
    for (const id of order) {
      try {
        vols[id].tex = texture(vols[id].meta.dims, await gunzip(BASE + vols[id].meta.file));
        if (id === order[0]) root.classList.add('brainvol-ready');
        dirty = true;
      } catch (e) { console.warn('brainvol: could not load', id, e); }
    }
  }

  // ---------------- drawing ----------------
  let scale = new URLSearchParams(location.search).has('hq') ? 1.0 : 0.62;                 // render resolution relative to CSS pixels
  let dirty = true, lastY = -1, slow = 0;
  const dark = matchMedia('(prefers-color-scheme: dark)');

  function resize() {
    const w = Math.max(1, Math.round(innerWidth * scale)), h = Math.max(1, Math.round(innerHeight * scale));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; dirty = true; }
  }

  function lerp(a, b, t) { return a + (b - a) * t; }

  function draw(now) {
    const { a, b, t } = sceneAt(scrollY);
    const S = {};
    for (const k of NUM) S[k] = lerp(a[k], b[k], t);
    // idle drift so the hero does not look frozen
    const drift = reduced.matches || still ? 0 : Math.sin(now * 0.00012) * 6;
    let va = vols[a.vol], vb = vols[b.vol];
    let mix = a.vol === b.vol ? 0 : t;
    if (!va || !va.tex) { va = vb; mix = 1; }
    if (!vb || !vb.tex) { vb = va; mix = 0; }
    if (!va || !va.tex) return;

    const fade = S.fade;
    canvas.style.opacity = fade.toFixed(3);
    if (fade < 0.01) return;

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, va.tex);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_3D, vb.tex);
    gl.uniform3fv(U.uHalfA, va.half); gl.uniform3fv(U.uHalfB, vb.half);
    gl.uniform1i(U.uPalA, va.pal); gl.uniform1i(U.uPalB, vb.pal);
    gl.uniform1f(U.uMix, mix);

    const half = va.half.map((h, i) => lerp(h, vb.half[i], mix));
    const target = [S.tx * half[0], S.ty * half[1], S.tz * half[2]];
    const az = (S.az + drift) * Math.PI / 180, el = S.el * Math.PI / 180;
    // camera on a sphere around the target; azimuth 0 looks at the left side of the brain
    const off = [-Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az)];
    const eye = target.map((c, i) => c + off[i] * S.dist);
    const fwd = off.map((x) => -x);
    let right = [fwd[1] * 0 - fwd[2] * 1, fwd[2] * 0 - fwd[0] * 0, fwd[0] * 1 - fwd[1] * 0];   // fwd x up(0,1,0)
    const rl = Math.hypot(...right) || 1; right = right.map((x) => x / rl);
    const up = [right[1] * fwd[2] - right[2] * fwd[1], right[2] * fwd[0] - right[0] * fwd[2], right[0] * fwd[1] - right[1] * fwd[0]];
    const fov = 30 * Math.PI / 180, asp = canvas.width / canvas.height;
    gl.uniform3fv(U.uEye, eye); gl.uniform3fv(U.uRight, right); gl.uniform3fv(U.uUp, up); gl.uniform3fv(U.uFwd, fwd);
    gl.uniform2f(U.uTan, Math.tan(fov / 2) * asp, Math.tan(fov / 2));
    // on narrow screens the brain sits behind the text, centred
    const narrow = innerWidth < 760;
    gl.uniform2f(U.uShift, narrow ? 0 : S.sx, narrow ? S.sy * 0.5 : S.sy);
    gl.uniform1f(U.uCut, S.cut);
    gl.uniform3f(U.uCutSign, Math.sign(eye[0] - target[0]) || 1, Math.sign(eye[1] - target[1]) || 1, Math.sign(eye[2] - target[2]) || 1);
    gl.uniform1f(U.uClip, S.clip); gl.uniform1f(U.uClipX, S.clipx * half[0]);
    const step = Math.max(half[0], half[1], half[2]) * 2 / 300;
    gl.uniform1f(U.uStep, step);
    gl.uniform1f(U.uFade, 1);
    const L = [-0.45, 0.75, 0.5], ll = Math.hypot(...L);
    gl.uniform3f(U.uLight, L[0] / ll, L[1] / ll, L[2] / ll);
    gl.uniform1f(U.uDark, dark.matches ? 1 : 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  let lastFrame = 0, frameMs = 16;
  function tick(now) {
    requestAnimationFrame(tick);
    if (document.hidden) return;
    const idle = !reduced.matches && !still;
    if (!dirty && !idle && scrollY === lastY) return;
    if (now - lastFrame < (scrollY === lastY ? 50 : 16)) return;     // ~20 fps while idle
    const t0 = performance.now();
    lastY = scrollY; dirty = false; lastFrame = now;
    draw(now);
    frameMs = frameMs * 0.9 + (performance.now() - t0) * 0.1;
    // keep it smooth on slow machines: drop resolution when frames get expensive
    if (frameMs > 30 && scale > 0.32) { if (++slow > 20) { scale *= 0.85; slow = 0; resize(); } }
    else slow = 0;
  }

  addEventListener('resize', () => { resize(); measure(); dirty = true; });
  addEventListener('load', () => { measure(); dirty = true; });
  if (window.ResizeObserver) new ResizeObserver(() => { measure(); dirty = true; }).observe(document.body);
  if (document.fonts) document.fonts.ready.then(() => { measure(); dirty = true; });
  resize(); measure();
  load().catch((e) => { console.warn('brainvol:', e); root.classList.add('no-webgl'); });
  requestAnimationFrame(tick);
  window.BrainVol = { remeasure: () => { measure(); dirty = true; } };
})();
