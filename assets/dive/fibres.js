/* Fibres behind the myelin stop. Hue in the published streamline figure encodes fibre direction, so the
   direction can be read back from it (orientation = hue / 2), smoothed in double-angle form where 0 and 180
   degrees agree, and integrated into streamlines. Drawn additively in their own small WebGL2 context, built
   only when the page first gets near the stop. Spikes are simulated; the pointer lights nearby fibres.
   The canvas is transparent, so the fibres can grow out of the Nissl field in its violet, each from its seed
   point, take on their direction colours at the myelin stop, and draw back into the cells on the way out. */
(() => {
  'use strict';
  const VERT = `#version 300 es
layout(location=0) in vec3 aPos;     // x, y, side
layout(location=1) in vec4 aInfo;    // tangent x, y, arc, depth
layout(location=2) in float aLine;   // 0 at the seed, 1 at the far end of the line
uniform vec2 scale, centre, pointer;
uniform float time, zoom, fade, grow, tint;
out vec3 vCol; out float vSide, vGrown;
vec3 hueRgb(float h) { return clamp(abs(fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, 0.0, 1.0); }
void main() {
  float z = 1.0 + aInfo.w * 0.5;
  vec2 p = (aPos.xy * z + centre) * zoom;
  vec2 nrm = vec2(-aInfo.y, aInfo.x);
  vec2 q = p * scale + nrm * aPos.z * 0.0022 * (1.2 - aInfo.w) * scale.x / scale.y * vec2(scale.y / scale.x, 1.0);
  float wave = pow(0.5 + 0.5 * sin(aInfo.z * 7.0 - time * 1.5 + aInfo.w * 9.0), 24.0);
  float near = exp(-dot(p - pointer, p - pointer) / 0.045);
  float hue = fract(atan(-aInfo.y, aInfo.x + 1e-5) / 3.14159265);
  vec3 c = mix(hueRgb(hue), vec3(0.62, 0.50, 1.0), tint);
  vCol = (mix(c, vec3(1.0, 0.97, 0.9), wave * 0.7) * (0.20 + 0.5 * tint + wave * 1.1 + near * 0.55)) * (1.0 - 0.6 * aInfo.w) * fade;
  vSide = aPos.z;
  vGrown = clamp((grow * 1.4 - aInfo.w * 0.35 - aLine) / 0.05, 0.0, 1.0);
  gl_Position = vec4(q, 0.0, 1.0);
}`;
  const FRAG = `#version 300 es
precision highp float;
in vec3 vCol; in float vSide, vGrown; out vec4 o;
void main() {
  vec3 c = vCol * vGrown * (1.0 - smoothstep(0.2, 1.0, abs(vSide)));
  o = vec4(c, max(c.r, max(c.g, c.b)));
}`;

  function createFibres(canvas, src) {
    let gl = null, prog = null, count = 0, u = {}, wanted = false, amount = 0, ready = false, W = 1, H = 1;
    const ptr = [9, 9];

    function trace(img) {
      const w = 220, h = Math.round(w * img.naturalHeight / img.naturalWidth);
      const c = document.createElement('canvas'); c.width = w; c.height = h;
      const g = c.getContext('2d', { willReadFrequently: true });
      g.drawImage(img, 0, 0, w, h);
      const d = g.getImageData(0, 0, w, h).data, n = w * h;
      let fx = new Float32Array(n), fy = new Float32Array(n);
      const mask = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        const r = d[i * 4], gg = d[i * 4 + 1], b = d[i * 4 + 2], mx = Math.max(r, gg, b), del = mx - Math.min(r, gg, b);
        if (mx < 80 || del < 40) continue;
        let hue = mx === r ? ((gg - b) / del + 6) % 6 : mx === gg ? (b - r) / del + 2 : (r - gg) / del + 4;
        hue *= Math.PI / 3;
        fx[i] = Math.cos(hue); fy[i] = Math.sin(hue); mask[i] = 1;
      }
      const blur = (s) => { const o = new Float32Array(n); for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) { let a = 0; for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) a += s[(y + j) * w + x + i]; o[y * w + x] = a / 9; } return o; };
      fx = blur(blur(fx)); fy = blur(blur(fy));
      const walk = (x, y, sign) => {
        const pts = []; let px = 0, py = 0;
        for (let st = 0; st < 60; st++) {
          const ix = x | 0, iy = y | 0;
          if (ix < 1 || iy < 1 || ix >= w - 1 || iy >= h - 1) break;
          const id = iy * w + ix;
          if (!mask[id]) break;
          const a = 0.5 * Math.atan2(fy[id], fx[id]);
          let dx = Math.cos(a), dy = Math.sin(a);
          if (st === 0) { dx *= sign; dy *= sign; } else if (dx * px + dy * py < 0) { dx = -dx; dy = -dy; }
          px = dx; py = dy; x += dx * 1.1; y += dy * 1.1;
          if (st % 2) pts.push([x, y]);
        }
        return pts;
      };
      let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
      const P = [], I = [], A = [], ix = [];
      let verts = 0, made = 0, tries = 0;
      const sc = 2.0 / h;
      while (made < 2200 && tries < 60000 && verts < 180000) {
        tries++;
        const sx = 1 + rnd() * (w - 2), sy = 1 + rnd() * (h - 2);
        if (!mask[(sy | 0) * w + (sx | 0)]) continue;
        const back = walk(sx, sy, -1).reverse(), line = back.concat([[sx, sy]], walk(sx, sy, 1));
        if (line.length < 7) continue;
        const reach = Math.max(back.length, line.length - 1 - back.length);
        const depth = rnd(), base = verts; let arc = rnd() * 6;
        line.forEach((p, i) => {
          const a = line[Math.max(0, i - 1)], b = line[Math.min(line.length - 1, i + 1)];
          let tx = b[0] - a[0], ty = -(b[1] - a[1]); const L = Math.hypot(tx, ty) || 1; tx /= L; ty /= L;
          if (i) arc += Math.hypot(p[0] - line[i - 1][0], p[1] - line[i - 1][1]) * sc;
          const X = (p[0] - w / 2) * sc, Y = -(p[1] - h / 2) * sc;
          for (const side of [-1, 1]) { P.push(X, Y, side); I.push(tx, ty, arc, depth); A.push(Math.abs(i - back.length) / reach); }
          if (i < line.length - 1) { const v = base + i * 2; ix.push(v, v + 1, v + 2, v + 1, v + 3, v + 2); }
        });
        verts += line.length * 2; made++;
      }
      const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
      [[P, 3], [I, 4], [A, 1]].forEach(([arr, k], loc) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, k, gl.FLOAT, false, 0, 0);
      });
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(ix), gl.STATIC_DRAW);
      count = ix.length; ready = true;
    }

    function start() {
      gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: true, depth: false, stencil: false });
      if (!gl) return;
      const sh = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x); if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)); return x; };
      try {
        prog = gl.createProgram();
        gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG)); gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
      } catch (e) { console.warn('fibres:', e); gl = null; return; }
      for (const n of ['scale', 'centre', 'pointer', 'time', 'zoom', 'fade', 'grow', 'tint']) u[n] = gl.getUniformLocation(prog, n);
      const img = new Image(); img.onload = () => trace(img); img.src = src;
    }

    addEventListener('pointermove', (e) => {
      if (!wanted) return;
      const a = W / H;
      ptr[0] = (e.clientX / W * 2 - 1) * a; ptr[1] = 1 - e.clientY / H * 2;
    }, { passive: true });

    return {
      set(a, t, R, grow, tint) {
        const near = t > 7.5 && t < 10.2;
        if (near && !gl && !prog) start();
        amount = a; wanted = a > 0.002;
        canvas.style.opacity = a.toFixed(3);
        canvas.style.visibility = wanted ? 'visible' : 'hidden';
        this.R = R; this.grow = grow; this.tint = tint;
      },
      running: () => wanted && ready,
      draw(now) {
        if (!wanted || !ready || !gl) return;
        const w = innerWidth, h = innerHeight, k = Math.min(devicePixelRatio || 1, 1.5);
        if (w !== W || h !== H || canvas.width !== Math.round(w * k)) {
          W = w; H = h; canvas.width = Math.round(w * k); canvas.height = Math.round(h * k);
          canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
        }
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(prog); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
        gl.uniform2f(u.scale, H / W, 1);
        gl.uniform2f(u.centre, -0.1, 0.0);
        gl.uniform2f(u.pointer, ptr[0], ptr[1]);
        gl.uniform1f(u.time, now * 0.001);
        gl.uniform1f(u.zoom, (W > H ? 2.35 : 1.75) * Math.pow(0.1 / this.R, 0.6));   // the fibres move with the camera
        gl.uniform1f(u.fade, 1);
        gl.uniform1f(u.grow, this.grow);
        gl.uniform1f(u.tint, this.tint);
        gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_INT, 0);
      },
    };
  }
  window.createFibres = createFibres;
})();
