/* Stage: one WebGL2 context, two passes.

   Pass 1 ray-marches the brains into a small offscreen buffer. The march is adapted from the site's earlier
   volume renderer (same volume layout, same cut-away and clip logic, same real sections
   on the cut faces); only the colour is different.

   Pass 2 runs at screen resolution and composites that buffer with the flat histology: the section
   drawn on the clip plane (face_24gw) and the four nested Nissl levels.  Every source goes through
   the same density -> colour ramp, which is what lets the volume, the section and the cells
   dissolve into each other without a visible change of material.

   Each flat layer is reached through an affine map, screen -> layer uv, computed in doubles on the
   CPU.  At the bottom of the dive one screen pixel is about a nanometre, far below float32 in world
   millimetres, so the shader never sees world coordinates. */
(() => {
  'use strict';

  // ---- geometry shared with the camera in dive.js (all millimetres, y down on the section plane) ----
  const GEOM = {
    nisslWidth: 72.2,                     // data-mm of the Nissl zoomer on the current site
    nisslAspect: 1.1168989802939289,      // nissl_zoom.json
    faceZ: 68.4, faceY: 75.6,             // dharani_24gw extent in z and y: the face image covers the box
    clipx: -0.42,                         // where face_24gw sits, fraction of the half width
    // Similarity that lays the Nissl section over the clip-plane section. Both are the 24-week brain but not
    // the same plane, so this is matched by eye on the band of the eminence arching over the striatum.
    // nissl point n0 (mm from the image's top-left) lands on plane point a0, turned by theta.
    align: { n0: [32.1, 25.45], a0: [5.67, -5.86], theta: 19.6 * Math.PI / 180 },
  };
  GEOM.nisslHeight = GEOM.nisslWidth / GEOM.nisslAspect;

  const rot = (a, x, y) => [Math.cos(a) * x - Math.sin(a) * y, Math.sin(a) * x + Math.cos(a) * y];
  GEOM.nisslToPlane = (nx, ny) => {
    const { n0, a0, theta } = GEOM.align;
    const r = rot(theta, nx - n0[0], ny - n0[1]);
    return [a0[0] + r[0], a0[1] + r[1]];
  };

  const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
  const lerp = (a, b, t) => a + (b - a) * t;

  // density 0..1 -> colour, shared by both passes
  const RAMP = `
vec3 faceRamp(float d) {
  vec3 lo = vec3(0.93, 0.90, 1.00), mid = vec3(0.50, 0.37, 0.95), hi = vec3(0.11, 0.05, 0.40);
  return d < 0.5 ? mix(lo, mid, d * 2.0) : mix(mid, hi, d * 2.0 - 1.0);
}
vec3 ramp(float d) {
  d = clamp(d, 0.0, 1.0);
  vec3 c0 = vec3(0.030, 0.026, 0.060), c1 = vec3(0.150, 0.090, 0.400), c2 = vec3(0.440, 0.290, 0.930);
  vec3 c3 = vec3(0.800, 0.720, 1.000), c4 = vec3(0.970, 0.950, 1.000);
  if (d < 0.30) return mix(c0, c1, d / 0.30);
  if (d < 0.62) return mix(c1, c2, (d - 0.30) / 0.32);
  if (d < 0.85) return mix(c2, c3, (d - 0.62) / 0.23);
  return mix(c3, c4, (d - 0.85) / 0.15);
}
`;

  const VERT = `#version 300 es
out vec2 screenPos;
void main() {
  screenPos = vec2(gl_VertexID & 1, gl_VertexID >> 1) * 2.0 - 1.0;
  gl_Position = vec4(screenPos, 0.0, 1.0);
}`;

  const VOL_FRAG = `#version 300 es
precision highp float;
precision highp sampler3D;
uniform sampler3D brainFrom, brainTo;
uniform vec3 box;                       // half size of the interpolated volume box
uniform float morph;                    // 0 = first brain only, 1 = second brain only
uniform vec2 depthGain;                 // per brain: depth value -> 0..1 of its typical inner depth
uniform float foldShade, foldReach, faceTone;
uniform vec3 camPos, camX, camY, camZ;
uniform vec2 lensSpread;
uniform vec2 screenOffset;
uniform float wedge;
uniform vec3 wedgeSide;
uniform float sliceAmount;
uniform float slicePlane;
uniform float marchStep;
uniform vec3 lightDir;
uniform sampler2D planeImage;
uniform float planeWeight, planeLod;
uniform sampler2D wedgeImageX, wedgeImageY, wedgeImageZ;
uniform float wedgeWeight;
uniform vec3 wedgeTexel;
uniform float pixelFootprint;
in vec2 screenPos;
out vec4 fragColour;
${RAMP}
float rnd(vec2 p) { return fract(sin(dot(p, vec2(41.31, 289.17))) * 15731.743); }

// x = density, y = depth below the surface, z = how much tissue is here. Both brains are read at the same place
// relative to one box that changes size, so shape and size move together. Tissue found in only one of them goes
// away from the surface inwards, or arrives from the inside out: the folds fill in, they do not dissolve.
vec3 sampleAt(vec3 p) {
  vec3 tc = p / (2.0 * box) + 0.5;
  if (any(lessThan(tc, vec3(0.0))) || any(greaterThan(tc, vec3(1.0)))) return vec3(0.0);
  vec2 a = texture(brainFrom, tc).rg;
  float inA = smoothstep(0.004, 0.02, a.r);
  if (morph < 0.001) return vec3(a, inA);
  vec2 b = texture(brainTo, tc).rg;
  float inB = smoothstep(0.004, 0.02, b.r);
  float uA = clamp(a.g * depthGain.x, 0.0, 1.0), uB = clamp(b.g * depthGain.y, 0.0, 1.0);
  float edgeA = mix(-0.1, 1.1, morph), edgeB = mix(1.1, -0.1, morph);
  float keepA = smoothstep(edgeA - 0.1, edgeA + 0.1, uA);
  float keepB = smoothstep(edgeB - 0.1, edgeB + 0.1, uB);
  float amount = max(inA * inB, max(inA * keepA, inB * keepB));
  float wa = inA * (1.0 - morph), wb = inB * morph, total = wa + wb + 1e-4;
  return vec3((wa * a.r + wb * b.r) / total, (wa * a.g + wb * b.g) / total, amount);
}
float surfaceField(vec3 p) { vec3 v = sampleAt(p); return v.y + 0.25 * v.z; }

bool removed(vec3 p, vec3 hs) {
  if (wedge > 0.001 && all(greaterThan(p * wedgeSide, hs * (1.0 - wedge) - vec3(1e-4)))) return true;
  if (sliceAmount > 0.001 && p.x < mix(-hs.x * 1.05, slicePlane, sliceAmount)) return true;
  return false;
}

void layer(inout vec4 dst, vec3 rgb, float a) {
  float room = 1.0 - dst.a;
  dst += vec4(room * a * rgb, room * a);
}

void main() {
  vec2 aim = screenPos - screenOffset;
  vec3 dir = normalize(camZ + aim.x * lensSpread.x * camX + aim.y * lensSpread.y * camY);
  vec3 hs = box;
  vec3 perUnit = 1.0 / dir;
  vec3 t0 = (-hs - camPos) * perUnit, t1 = (hs - camPos) * perUnit;
  vec3 tn3 = min(t0, t1), tf3 = max(t0, t1);
  float tn = max(max(tn3.x, tn3.y), tn3.z), tf = min(min(tf3.x, tf3.y), tf3.z);
  float tBox = max(tn, 0.0);
  if (tBox >= tf) { fragColour = vec4(0.0); return; }
  float planeX = mix(-hs.x * 1.05, slicePlane, sliceAmount);
  // nothing lives in front of the clip plane, so a ray heading into it can start there
  float tMarch = tBox;
  if (sliceAmount > 0.001 && dir.x > 1e-4) tMarch = max(tMarch, (planeX - camPos.x) / dir.x);
  float t = tMarch + marchStep * rnd(gl_FragCoord.xy);
  vec4 pixel = vec4(0.0);
  vec3 shellCol = vec3(0.52, 0.50, 0.62);
  if (planeWeight > 0.001 && dir.x > 1e-4) {
    float tp = (planeX - camPos.x) / dir.x;
    if (tp >= tBox && tp < tf) {
      vec3 p = camPos + dir * tp;
      vec3 tc = p / (2.0 * hs) + 0.5;
      vec4 f = textureLod(planeImage, tc.zy, planeLod);
      vec3 v = sampleAt(p + dir * 1e-4);
      float a = f.a * planeWeight * smoothstep(0.0, 0.006, v.x);
      float skin = 1.0 - smoothstep(0.05, 0.35, v.y);
      pixel = vec4(mix(ramp(f.r), shellCol, skin * 0.85) * a, a);
    }
  }
  float tq = -1.0;
  int qa = -1;
  if (wedgeWeight > 0.001) {
    vec3 c = hs * (1.0 - wedge);
    float lo = -1e9, hi = 1e9;
    for (int i = 0; i < 3; i++) {
      float d = dir[i] * wedgeSide[i], e = camPos[i] * wedgeSide[i];
      if (abs(d) < 1e-6) { if (e < c[i]) hi = -1e9; continue; }
      float ti = (c[i] - e) / d;
      if (d > 0.0) lo = max(lo, ti);
      else if (ti < hi) { hi = ti; qa = i; }
    }
    if (lo < hi && hi > tBox && hi < tf) tq = hi; else qa = -1;
  }
  for (int i = 0; i < 900; i++) {
    if (qa >= 0 && t >= tq) {
      vec3 p = camPos + dir * tq;
      vec3 tc = p / (2.0 * hs) + 0.5;
      float lod = max(0.0, log2(tq * pixelFootprint / max(abs(dir[qa]), 0.25) / wedgeTexel[qa]) - 0.6);
      vec4 f = qa == 0 ? textureLod(wedgeImageX, tc.zy, lod) : qa == 1 ? textureLod(wedgeImageY, tc.xz, lod) : textureLod(wedgeImageZ, tc.xy, lod);
      layer(pixel, faceRamp(f.r), f.a * wedgeWeight);
      qa = -1;
    }
    if (t > tf || pixel.a > 0.97) break;
    vec3 p = camPos + dir * t;
    if (!removed(p, hs)) {
      vec3 v = sampleAt(p);
      float d = v.x;
      if (d > 0.01 && v.z > 0.02) {
        float depth = v.y;
        float lit = 1.0, hollow = 0.0;
        if (depth < 0.45 || morph > 0.001) {
          float e = 0.006, here = depth + 0.25 * v.z;
          vec3 g = vec3(surfaceField(p + vec3(e, 0, 0)), surfaceField(p + vec3(0, e, 0)), surfaceField(p + vec3(0, 0, e))) - here;
          float g2 = dot(g, g);
          if (g2 > 1e-6) {
            vec3 n = -g * inversesqrt(g2);
            lit = 0.55 + 0.45 * max(dot(n, lightDir), 0.0) + 0.25 * pow(1.0 - abs(dot(n, dir)), 3.0);
            // step out along the normal: on a crown that is air, in a sulcus it is the opposite bank
            if (foldShade > 0.001) hollow = foldShade * (0.55 * sampleAt(p + n * foldReach).z + 0.45 * sampleAt(p + n * (foldReach * 2.2)).z);
          }
        }
        float skin = morph > 0.001 ? 1.0 : 1.0 - smoothstep(0.05, 0.35, depth);
        vec3 shell = mix(shellCol, vec3(0.13, 0.09, 0.40), hollow * 0.9);
        vec3 col = mix(mix(ramp(d), faceRamp(d), faceTone), shell, skin * 0.85) * lit * (1.0 - 0.22 * hollow);
        float dens = smoothstep(0.01, 0.25, d) * mix(180.0, 420.0, skin) * v.z;
        layer(pixel, col, 1.0 - exp(-dens * marchStep));
      }
    }
    t += marchStep;
  }
  fragColour = pixel;
}`;

  /* The bench: one camera, two volumes of the same 24-week brain on the same grid, Nissl histology and T1 MRI.
     The blade cuts both. The histology can also be taken apart into slabs along the real sectioning axis, each
     slab with its own small error, and the two volumes can interleave as a 3D checkerboard: where the cells of
     the board meet, a fold either carries on or it does not, which is what registration is for. */
  const BENCH_FRAG = `#version 300 es
precision highp float;
precision highp sampler3D;
uniform sampler3D histVol, mriVol, fineVol;
uniform float fineOn;
uniform vec3 box, camPos, camX, camY, camZ, lightDir;
uniform vec2 lensSpread, screenOffset;
uniform float marchStep;
uniform vec3 histPos, mriPos;
uniform float mriOn;
uniform vec4 cutPlane;
uniform float cutOn, cutSign, cutGlow;
uniform vec3 cutU, cutV;
uniform float slabN, slabT, slabB, slabG, slabShift, jitter, wave;
uniform float knifeOn, knifeX;
uniform float checker, tile;
uniform mat3 misRot;
uniform vec3 misOff;
uniform float misWarp;
in vec2 screenPos;
out vec4 fragColour;
${RAMP}
const vec3 LIME = vec3(0.839, 1.0, 0.247);
float rnd(vec2 p) { return fract(sin(dot(p, vec2(41.31, 289.17))) * 15731.743); }
float h1(float n) { return fract(sin(n * 12.9898 + 4.1) * 43758.5453); }
float h3(vec3 c) { return fract(sin(dot(c, vec3(12.99, 78.23, 37.72))) * 43758.5453); }

vec3 mriRamp(float d) {
  d = clamp(d, 0.0, 1.0);
  vec3 a = vec3(0.030, 0.028, 0.050), b = vec3(0.300, 0.300, 0.350), c = vec3(0.937, 0.918, 0.875);
  return d < 0.5 ? mix(a, b, d * 2.0) : mix(b, c, d * 2.0 - 1.0);
}
// a face of the frozen block: unstained, so grey matter is the darker tissue
vec3 blockRamp(float d) { return mix(vec3(0.78, 0.74, 0.66), vec3(0.20, 0.17, 0.19), smoothstep(0.05, 0.75, d)); }

vec2 texAt(sampler3D v, vec3 p) {
  vec3 tc = p / (2.0 * box) + 0.5;
  if (any(lessThan(tc, vec3(0.0))) || any(greaterThan(tc, vec3(1.0)))) return vec2(0.0);
  return texture(v, tc).rg;
}

// the pose error the demo starts from, on the histology only
vec3 misMap(vec3 p) {
  return misRot * p + misOff + misWarp * vec3(sin(p.y * 6.0), sin(p.z * 5.0 + 1.0), sin(p.x * 5.5 + 2.0));
}
vec3 histQ;     // where histAt last read the histology
vec2 histAt(vec3 p, out float slab) {
  float x = p.x - slabShift;
  vec3 q = vec3(x, p.yz);
  slab = floor((x + box.x) / slabT);
  if (slabG < 1e-5 && jitter < 1e-4) { histQ = misMap(q); return texAt(histVol, histQ); }
  float fb = floor(slabB);
  float solid = slabB >= slabN ? 1e9 : -box.x + (fb + 1.0) * slabT;
  if (x < solid) {
    float pitch = slabT + slabG, u = x + box.x + slabG * slabB;
    float i = floor(u / pitch), r = u - i * pitch;
    if (r > slabT || i < 0.0 || i > min(fb, slabN - 1.0)) return vec2(0.0);
    slab = i; q.x = -box.x + i * slabT + r;
  }
  float w = jitter * step(slab, slabB);
  float a = (h1(slab) - 0.5) * 0.34 * w;
  vec2 o = (vec2(h1(slab + 7.0), h1(slab + 13.0)) - 0.5) * box.y * 0.2 * w;
  float c = cos(a), s = sin(a);
  vec2 yz = q.yz - o;
  q.yz = vec2(c * yz.x - s * yz.y, s * yz.x + c * yz.y);
  histQ = misMap(q);
  return texAt(histVol, histQ);
}
// the MRI is the fixed reference: native shape, read as it is, never moved
vec2 mriAt(vec3 p) { return texAt(mriVol, p); }
float depthAt(bool isMri, vec3 p) { float s; return isMri ? mriAt(p).g : histAt(p, s).g; }

bool cellGone(vec3 w) {
  if (checker < 0.001) return false;
  vec3 c = floor(w / tile);
  return mod(c.x + c.y + c.z, 2.0) > 0.5 && checker > h3(c) * 0.98 + 0.01;
}

void layer(inout vec4 dst, vec3 rgb, float a) {
  float room = 1.0 - dst.a;
  dst += vec4(room * a * rgb, room * a);
}

vec4 marchObj(bool isMri, vec3 ro, vec3 dir, vec3 pos, out float hit) {
  hit = 1e9;
  vec3 lo = -box, hi = box;
  if (!isMri) {
    lo.x += slabShift - slabG * slabB; hi.x += slabShift;
    lo.yz *= 1.0 + 0.16 * jitter; hi.yz *= 1.0 + 0.16 * jitter;
    lo *= 1.0 + 9.0 * misWarp; hi *= 1.0 + 9.0 * misWarp;
  }
  vec3 inv = 1.0 / dir, t0 = (lo - ro) * inv, t1 = (hi - ro) * inv;
  vec3 tn3 = min(t0, t1), tf3 = max(t0, t1);
  float tS = max(max(max(tn3.x, tn3.y), tn3.z), 0.0), tf = min(min(tf3.x, tf3.y), tf3.z);
  if (tS >= tf) return vec4(0.0);
  bool face = false;
  if (cutOn > 0.5) {
    float sd = cutSign * (dot(ro, cutPlane.xyz) - cutPlane.w), sv = cutSign * dot(dir, cutPlane.xyz);
    if (abs(sv) > 1e-6) {
      float tp = -sd / sv;
      if (sv < 0.0) { if (tp > tS) { tS = tp; face = true; } }
      else tf = min(tf, tp);
    } else if (sd > 0.0) return vec4(0.0);
    if (tS >= tf) return vec4(0.0);
  }
  float t = tS + (face ? 1e-4 : marchStep * rnd(gl_FragCoord.xy));
  vec4 px = vec4(0.0);
  vec3 shellH = vec3(0.52, 0.50, 0.62), shellM = vec3(0.62, 0.61, 0.60);
  for (int i = 0; i < 640; i++) {
    if (t > tf || px.a > 0.97) break;
    vec3 p = ro + dir * t;
    if (isMri || !cellGone(p + pos)) {
      float slab = 0.0;
      vec2 v = isMri ? mriAt(p) : histAt(p, slab);
      float d = v.r;
      if (d > 0.01) {
        // the cut face reads the fine stack, stored section by section: the volume only says where tissue is
        if (face && !isMri && fineOn > 0.001) d = mix(d, texture(fineVol, (histQ / (2.0 * box) + 0.5).zyx).r, fineOn);
        float stain = isMri ? 1.0 : smoothstep(0.0, 2.0, wave - slab);
        vec3 inner = isMri ? mriRamp(d) : mix(blockRamp(d), ramp(d), stain);
        if (face) {
          layer(px, inner, 1.0);
          hit = min(hit, t);
        } else {
          float depth = v.g, lit = 1.0;
          if (depth < 0.45) {
            float e = 0.006;
            vec3 g = vec3(depthAt(isMri, p + vec3(e, 0, 0)), depthAt(isMri, p + vec3(0, e, 0)), depthAt(isMri, p + vec3(0, 0, e))) - depth;
            float g2 = dot(g, g);
            if (g2 > 1e-6) {
              vec3 n = -g * inversesqrt(g2);
              lit = 0.55 + 0.45 * max(dot(n, lightDir), 0.0) + 0.25 * pow(1.0 - abs(dot(n, dir)), 3.0);
            }
          }
          float skin = 1.0 - smoothstep(0.05, 0.35, depth);
          vec3 shell = isMri ? shellM : mix(vec3(0.74, 0.70, 0.62), shellH, stain);
          vec3 col = mix(inner, shell, skin * 0.85) * lit;
          float dens = smoothstep(0.01, 0.25, d) * mix(isMri ? 300.0 : 180.0, 420.0, skin);
          layer(px, col, 1.0 - exp(-dens * marchStep));
          if (px.a > 0.5) hit = min(hit, t);
        }
      }
    }
    face = false;
    t += marchStep;
  }
  return px;
}

// the lime outline of a blade: a square on a plane, drawn over anything behind it
vec4 bladeQuad(vec3 ro, vec3 dir, vec3 n, float d, vec3 u, vec3 v, float size, float glow, float maxT) {
  float dn = dot(dir, n);
  if (abs(dn) < 1e-5) return vec4(0.0);
  float t = (d - dot(ro, n)) / dn;
  if (t <= 0.0 || t > maxT) return vec4(0.0);
  vec3 p = ro + dir * t;
  float a = abs(dot(p, u)), b = abs(dot(p, v)), m = max(a, b);
  if (m > size) return vec4(0.0);
  float edge = smoothstep(size - 0.0032, size - 0.0016, m);
  float al = glow * (0.035 + 0.7 * edge);
  return vec4(LIME * al, al);
}

void main() {
  vec2 aim = screenPos - screenOffset;
  vec3 dir = normalize(camZ + aim.x * lensSpread.x * camX + aim.y * lensSpread.y * camY);
  float hH, hM = 1e9;
  vec4 H = marchObj(false, camPos - histPos, dir, histPos, hH);
  vec4 M = vec4(0.0);
  if (mriOn > 0.002) M = marchObj(true, camPos - mriPos, dir, mriPos, hM) * mriOn;
  vec4 px = hM < hH - 0.012 ? M + H * (1.0 - M.a) : H + M * (1.0 - H.a);
  float size = max(box.y, box.z) * 0.98;
  if (cutOn > 0.5 && cutGlow > 0.001) {
    vec4 q = bladeQuad(camPos - histPos, dir, cutPlane.xyz, cutPlane.w, cutU, cutV, size, cutGlow, 1e9);
    px = q + px * (1.0 - q.a);
    if (mriOn > 0.002) { q = bladeQuad(camPos - mriPos, dir, cutPlane.xyz, cutPlane.w, cutU, cutV, size, cutGlow * mriOn, 1e9); px = q + px * (1.0 - q.a); }
  }
  if (knifeOn > 0.001) {
    vec4 q = bladeQuad(camPos - histPos, dir, vec3(1, 0, 0), knifeX + slabShift, vec3(0, 1, 0), vec3(0, 0, 1), size, knifeOn, 1e9);
    px = q + px * (1.0 - q.a);
  }
  fragColour = px;
}`;

  const COMP_FRAG = `#version 300 es
precision highp float;
uniform sampler2D volTex, layFace, layA, layB, layC, layD;
uniform vec2 canvasSize;
uniform float shortPx;
uniform float volWeight, dimAmount, scrimAmount, scrimMode;
uniform vec4 mapA[5];
uniform vec2 mapB[5];
uniform float layWeight[5];
uniform vec3 densCurve;                // bias, gain, gamma: Nissl brightness -> density
uniform vec2 cellSize;
uniform float cellMag, pixAmount, stretchAmount, deepCurve;
uniform vec2 stretchRange;
out vec4 fragColour;
${RAMP}
float rnd(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

vec2 place(int i, vec2 s) {
  vec4 m = mapA[i];
  return vec2(m.x * s.x + m.y * s.y, m.z * s.x + m.w * s.y) + mapB[i];
}
float inside(vec2 uv) {
  vec2 a = smoothstep(vec2(0.0), vec2(0.012), uv) * smoothstep(vec2(0.0), vec2(0.012), 1.0 - uv);
  return a.x * a.y;
}
float nisslDensity(float lum) {
  return pow(clamp((1.0 - lum - densCurve.x) * densCurve.y, 0.0, 1.0), densCurve.z);
}
// under the lens: the curve above calls every nucleus simply "stained" and neighbours merge into one white mass,
// so spread the darkest part of the stain instead, where the chromatin is
float cellDensity(float lum) {
  return mix(nisslDensity(lum), 0.9 * pow(clamp((0.85 - lum) / 0.85, 0.0, 1.0), 2.6), deepCurve);   // below white: it would go grey under the scrim
}
void over(inout vec3 col, sampler2D tex, vec2 uv, float w) {
  vec4 c = texture(tex, uv);
  col = mix(col, ramp(cellDensity(c.r)), c.a * inside(uv) * w);
}
// magnification: Catmull-Rom passes through the scan's own pixels, so it adds no blur of its own
vec4 cubicTap(sampler2D tex, vec2 uv, vec2 size) {
  vec2 st = uv * size - 0.5;
  vec2 i = floor(st), f = st - i;
  vec2 wx[4] = vec2[4](f * (-0.5 + f * (1.0 - 0.5 * f)), 1.0 + f * f * (-2.5 + 1.5 * f),
                       f * (0.5 + f * (2.0 - 1.5 * f)), f * f * (-0.5 + 0.5 * f));
  ivec2 at = ivec2(i) - 1, top = ivec2(size) - 1;
  vec4 c = vec4(0.0);
  for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++)
    c += texelFetch(tex, clamp(at + ivec2(x, y), ivec2(0), top), 0) * wx[x].x * wx[y].y;
  return clamp(c, 0.0, 1.0);
}

void main() {
  vec2 fc = gl_FragCoord.xy;
  vec2 s = vec2(fc.x - canvasSize.x * 0.5, canvasSize.y * 0.5 - fc.y) / shortPx;
  vec3 col = vec3(0.030, 0.026, 0.060);
  if (volWeight > 0.001) {
    vec4 v = texture(volTex, fc / canvasSize);
    col = col * (1.0 - v.a * volWeight) + v.rgb * volWeight;
  }
  if (layWeight[0] > 0.001) {
    vec2 uv = place(0, s);
    vec4 f = texture(layFace, uv);
    col = mix(col, ramp(f.r), f.a * inside(uv) * layWeight[0]);
  }
  if (layWeight[1] > 0.001) over(col, layA, place(1, s), layWeight[1]);
  if (layWeight[2] > 0.001) over(col, layB, place(2, s), layWeight[2]);
  if (layWeight[3] > 0.001) over(col, layC, place(3, s), layWeight[3]);
  if (layWeight[4] > 0.001) {
    vec2 uv = place(4, s);
    vec4 c = cellMag > 1.3 ? cubicTap(layD, uv, cellSize) : texture(layD, uv);
    vec2 texel = uv * cellSize;
    if (pixAmount > 0.001) {
      ivec2 ip = clamp(ivec2(floor(texel)), ivec2(0), ivec2(cellSize) - 1);
      c = mix(c, texelFetch(layD, ip, 0), pixAmount);
    }
    float d = cellDensity(c.r);
    float a = c.a * inside(uv) * layWeight[4];
    if (stretchAmount > 0.001) {
      // inside the nucleus the pixels differ only slightly: spread them so they read as separate squares
      float raw = (1.0 - c.r - stretchRange.x) / max(stretchRange.y - stretchRange.x, 0.02);
      d = mix(d, mix(0.74, 1.0, clamp(raw, 0.0, 1.0)), stretchAmount);
      a = mix(a, 1.0, stretchAmount);
    }
    vec3 lit = ramp(d);
    if (pixAmount > 0.001) {
      vec2 e = abs(fract(texel) - 0.5);
      float edge = step(0.5 - 1.0 / max(cellMag, 1.0), max(e.x, e.y));
      lit *= 1.0 - 0.10 * edge * pixAmount;
    }
    col = mix(col, lit, a);
  }
  vec2 q = fc / canvasSize;
  float m = scrimMode < 0.5 ? 1.0 - smoothstep(0.28, 0.60, q.x) : 1.0 - smoothstep(0.30, 0.62, q.y);
  col *= 1.0 - scrimAmount * m;
  col *= 1.0 - dimAmount;
  vec2 k = q - 0.5;
  col *= 1.0 - (1.0 - stretchAmount) * 0.55 * dot(k, k);
  col += (rnd(fc) - 0.5) * 0.016 * (1.0 - stretchAmount);
  fragColour = vec4(col, 1.0);
}`;

  function createStage(canvas, opts) {
    const gl = canvas.getContext('webgl2', { alpha: false, antialias: false, depth: false, stencil: false, powerPreference: 'high-performance' });
    if (!gl) return null;

    const compile = (type, src) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src); gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
      return sh;
    };
    const link = (frag) => {
      const p = gl.createProgram();
      gl.attachShader(p, compile(gl.VERTEX_SHADER, VERT));
      gl.attachShader(p, compile(gl.FRAGMENT_SHADER, frag));
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      return p;
    };
    const slots = (p) => new Proxy({}, { get: (seen, n) => (n in seen ? seen[n] : (seen[n] = gl.getUniformLocation(p, n))) });

    let volProg, compProg, benchProg;
    try { volProg = link(VOL_FRAG); compProg = link(COMP_FRAG); benchProg = link(BENCH_FRAG); }
    catch (e) { console.warn('stage:', e); return null; }
    const V = slots(volProg), C = slots(compProg), B = slots(benchProg);

    // texture units: 0,1 volumes; 2 clip-plane section; 3-5 cut faces; 6 march buffer; 7-11 flat layers; 12 fine histology; 15 scratch
    const SCRATCH = 15;
    gl.useProgram(volProg);
    gl.uniform1i(V.brainFrom, 0); gl.uniform1i(V.brainTo, 1); gl.uniform1i(V.planeImage, 2);
    gl.uniform1i(V.wedgeImageX, 3); gl.uniform1i(V.wedgeImageY, 4); gl.uniform1i(V.wedgeImageZ, 5);
    gl.useProgram(benchProg);
    gl.uniform1i(B.histVol, 0); gl.uniform1i(B.mriVol, 1); gl.uniform1i(B.fineVol, 12);
    gl.useProgram(compProg);
    gl.uniform1i(C.volTex, 6); gl.uniform1i(C.layFace, 7); gl.uniform1i(C.layA, 8);
    gl.uniform1i(C.layB, 9); gl.uniform1i(C.layC, 10); gl.uniform1i(C.layD, 11);

    gl.activeTexture(gl.TEXTURE0 + SCRATCH);
    const blank = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, blank);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    function volumeTexture(dims, data) {
      const tex = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + SCRATCH);
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG8, dims[0], dims[1], dims[2], 0, gl.RG, gl.UNSIGNED_BYTE, data);
      for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
                            [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE],
                            [gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_3D, k, v);
      return tex;
    }
    const emptyVol = volumeTexture([1, 1, 1], new Uint8Array(2));

    // grey + alpha is all any section holds (for an RGB image the luminance channel takes red, which is
    // where a violet stain has its contrast), at half the memory of RGBA
    function greyImage(src) {
      return new Promise((done, fail) => {
        const img = new Image();
        img.onload = async () => {
          try { await img.decode(); } catch (e) { /* already decoded or undecodable: the upload will tell */ }
          const tex = gl.createTexture();
          gl.activeTexture(gl.TEXTURE0 + SCRATCH);
          gl.bindTexture(gl.TEXTURE_2D, tex);
          gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
          gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
          gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE_ALPHA, gl.LUMINANCE_ALPHA, gl.UNSIGNED_BYTE, img);
          gl.generateMipmap(gl.TEXTURE_2D);
          for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
                                [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
          done({ tex, img, w: img.naturalWidth, h: img.naturalHeight });
        };
        img.onerror = () => fail(new Error('could not load ' + src));
        img.src = src;
      });
    }

    async function inflate(url) {
      const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
      if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;     // the server already decoded it
      const out = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
      return new Uint8Array(await new Response(out).arrayBuffer());
    }

    // ---- content ----
    const ASSETS = opts.assets;
    // the story of the brains: BigBrain shrinks to 24 weeks, on down to 14, then grows back to 24
    const CHAIN = ['bigbrain', 'dharani_24gw', 'dharani_22gw', 'dharani_21gw', 'dharani_17gw', 'dharani_14gw',
                   'dharani_17gw', 'dharani_21gw', 'dharani_22gw', 'dharani_24gw'];
    const MM = 0.01;
    const vols = {};
    const cuts = { x: null, y: null, z: null };
    let face = null;
    // fallback rectangles until nissl_zoom.json arrives
    let levels = [{ rect: [0, 0, 1, 1] }, { rect: [0.1807, 0.3425, 0.3191, 0.4971] }, { rect: [0.2291, 0.3966, 0.2706, 0.4430] }, { rect: [0.2443, 0.4136, 0.2554, 0.4260] }];
    // the final pixel, in the coordinates of the deepest image (0..1), and its local contrast range
    const cell = { u: 0.5, v: 0.5, w: 1600, h: 1600, lo: 0.75, hi: 0.95 };

    // darkest cluster near the given nucleus (or the middle of the cell image); then the darkest pixel's neighbourhood
    function pickCell(img, w, h, at) {
      try {
        const R = 130, side = 2 * R;
        const c = document.createElement('canvas'); c.width = c.height = side;
        const x = c.getContext('2d', { willReadFrequently: true });
        const x0 = Math.round((at ? at[0] : 0.5) * w - R), y0 = Math.round((at ? at[1] : 0.5) * h - R);
        const edge = at ? R - 14 : 12;
        x.drawImage(img, -x0, -y0);
        const px = x.getImageData(0, 0, side, side).data;
        const dark = (i, j) => { const o = (j * side + i) * 4; return px[o + 3] > 200 ? 255 - px[o] : 0; };
        let best = -1, bi = R, bj = R;
        for (let j = edge; j < side - edge; j += 2) for (let i = edge; i < side - edge; i += 2) {
          let sum = 0;
          for (let b = -4; b <= 4; b++) for (let a = -4; a <= 4; a++) sum += dark(i + a, j + b);
          if (sum > best) { best = sum; bi = i; bj = j; }
        }
        let lo = 1, hi = 0;
        for (let b = -4; b <= 4; b++) for (let a = -7; a <= 7; a++) {
          const d = dark(bi + a, bj + b) / 255;
          if (d < lo) lo = d;
          if (d > hi) hi = d;
        }
        cell.u = (x0 + bi + 0.5) / w; cell.v = (y0 + bj + 0.5) / h; cell.lo = lo; cell.hi = hi;
      } catch (e) { console.warn('stage: keeping the default cell', e); }
    }

    function depthScale(data) {
      const hist = new Uint32Array(256);
      let n = 0;
      for (let i = 0; i + 1 < data.length; i += 194) if (data[i] > 3) { hist[data[i + 1]]++; n++; }
      let seen = 0;
      for (let v = 0; v < 256; v++) { seen += hist[v]; if (seen >= n * 0.8) return 255 / Math.max(v, 8); }
      return 2;
    }

    const nisslPoint = (u, v) => [u * GEOM.nisslWidth, v * GEOM.nisslHeight];
    const sectionCentre = () => GEOM.nisslToPlane(...nisslPoint(0.5, 0.5));
    function cellTarget() {
      const r = levels[3].rect;
      return GEOM.nisslToPlane(...nisslPoint(r[0] + (r[2] - r[0]) * cell.u, r[1] + (r[3] - r[1]) * cell.v));
    }

    // the histology at section resolution, for the blade's cut face only: sheets of sections, one 3D texture.
    // Layers run along the sectioning axis, so the shader reads it as .zyx
    async function loadFine(dir) {
      const m = await fetch(dir + 'index.json').then((r) => r.json());
      const [nu, nv, nw] = m.grid;
      const tex = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + SCRATCH);
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.texStorage3D(gl.TEXTURE_3D, 1, gl.R8, nw, nv, nu);
      if (gl.getError() !== gl.NO_ERROR) throw new Error('no room for the fine histology');
      for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
                            [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE],
                            [gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_3D, k, v);
      const c = document.createElement('canvas');
      c.width = nw * m.cols; c.height = nv * m.per / m.cols;
      const x = c.getContext('2d', { willReadFrequently: true });
      const sheet = (k) => { const i = new Image(); i.src = dir + m.files[k]; return i.decode().then(() => i); };
      const section = new Uint8Array(nw * nv);
      let next = sheet(0);
      for (let k = 0; k < m.files.length; k++) {
        const img = await next;
        if (k + 1 < m.files.length) next = sheet(k + 1);
        x.drawImage(img, 0, 0);
        const px = x.getImageData(0, 0, c.width, c.height).data;
        gl.activeTexture(gl.TEXTURE0 + SCRATCH);
        gl.bindTexture(gl.TEXTURE_3D, tex);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        for (let j = 0; j < m.per && k * m.per + j < nu; j++) {
          const ox = (j % m.cols) * nw, oy = Math.floor(j / m.cols) * nv;
          for (let r = 0, q = 0; r < nv; r++) {
            let o = ((oy + r) * c.width + ox) * 4;
            for (let i = 0; i < nw; i++, o += 4) section[q++] = px[o];
          }
          gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, k * m.per + j, nw, nv, 1, gl.RED, gl.UNSIGNED_BYTE, section);
        }
      }
      bench.fine = { tex, t0: performance.now() };
      if (opts.onFine) opts.onFine(m.voxel_mm);
      opts.onChange();
    }

    async function load(fine) {
      const base = ASSETS + 'volumes/';
      const manifest = await fetch(base + 'brains.json').then((r) => r.json());
      for (const m of manifest.brains) vols[m.id] = { meta: m, half: m.extent_mm.map((x) => x * MM / 2), tex: null };
      const fetchVol = async (id) => {
        try {
          const data = await inflate(base + vols[id].meta.data);
          vols[id].gain = depthScale(data);
          vols[id].tex = volumeTexture(vols[id].meta.grid, data);
          if (id === 'dharani_24gw') { bench.hist = data; if (opts.onHistology) opts.onHistology(); }
          opts.onChange();
        }
        catch (e) { console.warn('stage:', id, e); }
      };
      await fetchVol('bigbrain');
      if (!vols.bigbrain.tex) throw new Error('no volume');
      opts.onReady();
      await Promise.all(['x', 'y', 'z'].map(async (k) => {
        try { const r = await greyImage(`${ASSETS}img/cyto/bb_cut_${k}.webp`); cuts[k] = { tex: r.tex, rows: r.h, t0: performance.now() }; opts.onChange(); }
        catch (e) { console.warn('stage:', e); }
      }));
      // the flat chain before the fetal volumes: a jump down the ruler must land on something
      try {
        const spec = await fetch(ASSETS + 'img/ge/nissl_zoom.json').then((r) => r.json());
        GEOM.nisslAspect = spec.aspect; GEOM.nisslHeight = GEOM.nisslWidth / spec.aspect;
        levels = spec.levels.map((l) => ({ rect: l.rect, src: l.src.replace(/^\.?\/?/, ''), tex: null }));
        for (let i = 0; i < levels.length; i++) {
          const r = await greyImage(levels[i].src);
          levels[i].tex = r.tex; levels[i].w = r.w; levels[i].h = r.h;
          if (i === levels.length - 1) { pickCell(r.img, r.w, r.h, spec.cell); cell.w = r.w; cell.h = r.h; }
          opts.onChange();
        }
      } catch (e) { console.warn('stage:', e); }
      try { const r = await greyImage(ASSETS + 'img/ge/face_24gw.webp'); face = { tex: r.tex, rows: r.h }; opts.onChange(); }
      catch (e) { console.warn('stage:', e); }
      await fetchVol('dharani_24gw');
      try {
        bench.mri = volumeTexture(vols.dharani_24gw.meta.grid, await inflate(opts.mri));
        opts.onChange();
      } catch (e) { console.warn('stage: no MRI', e); }
      for (const id of ['dharani_22gw', 'dharani_21gw', 'dharani_17gw', 'dharani_14gw']) await fetchVol(id);
      if (fine) {
        try { await loadFine(fine); }
        catch (e) { console.warn('stage: keeping the coarse histology', e); }
      }
    }

    function nearestVol(i) {
      for (let k = 0; k < CHAIN.length; k++) {
        for (const j of [i - k, i + k]) { const v = vols[CHAIN[j]]; if (v && v.tex) return v; }
      }
      return null;
    }

    // ---- sizes ----
    let W = 1, H = 1, S = 1, dpr = 1, quality = 1;
    const march = { tex: gl.createTexture(), fb: gl.createFramebuffer(), w: 0, h: 0 };
    gl.activeTexture(gl.TEXTURE0 + SCRATCH);
    gl.bindTexture(gl.TEXTURE_2D, march.tex);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
                          [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);

    function fitMarch(scale) {
      const w = Math.max(2, Math.round(W * scale)), h = Math.max(2, Math.round(H * scale));
      if (w === march.w && h === march.h) return;
      march.w = w; march.h = h;
      gl.activeTexture(gl.TEXTURE0 + SCRATCH);
      gl.bindTexture(gl.TEXTURE_2D, march.tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, march.fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, march.tex, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    function resize(w, h) {
      W = w; H = h; S = Math.min(W, H);
      // the composite is cheap but fills every pixel: cap it at 1.5x and about 2.6 megapixels
      dpr = Math.min(devicePixelRatio || 1, 2, Math.sqrt(3.6e6 / (W * H)));
      canvas.width = Math.max(1, Math.round(W * dpr));
      canvas.height = Math.max(1, Math.round(H * dpr));
      canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
    }

    // ---- pass 1 ----
    const FOV = 30 * Math.PI / 180, TAN = Math.tan(FOV / 2);

    function aim(cam) {
      const target = cam.target.map((x) => x * MM);
      const dist = cam.F * MM / (2 * TAN);
      const az = cam.az * Math.PI / 180, el = cam.el * Math.PI / 180;
      const off = [-Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az)];
      const eye = target.map((c, k) => c + off[k] * dist);
      const fwd = off.map((x) => -x);
      let right = [-fwd[2], 0, fwd[0]];                                 // fwd x (0,1,0)
      const rl = Math.hypot(...right) || 1; right = right.map((x) => x / rl);
      const up = [right[1] * fwd[2] - right[2] * fwd[1], right[2] * fwd[0] - right[0] * fwd[2], right[0] * fwd[1] - right[1] * fwd[0]];
      return { eye, right, up, fwd, off };
    }

    // ---- the bench: histology and MRI of the same brain, one camera ----
    const bench = { hist: null, mri: null, fine: null };
    function benchPass(cam, now) {
      const h = vols.dharani_24gw;
      if (!h || !h.tex) return false;
      const k = cam.bench, half = h.half;
      // the fine cut face is worth every pixel of the screen; everything else keeps the cheaper buffer
      fitMarch(bench.fine && k.cut.amount > 0.5 ? Math.min(dpr, 1.5) * quality : Math.min(1, 0.85 * quality));
      gl.useProgram(benchProg);
      gl.bindFramebuffer(gl.FRAMEBUFFER, march.fb);
      gl.viewport(0, 0, march.w, march.h);
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, h.tex);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_3D, bench.mri || emptyVol);
      gl.activeTexture(gl.TEXTURE0 + 12); gl.bindTexture(gl.TEXTURE_3D, bench.fine ? bench.fine.tex : emptyVol);
      const arrive = bench.fine ? Math.min(1, (now - bench.fine.t0) / 800) : 0;
      gl.uniform1f(B.fineOn, smooth(0, 1, arrive));
      const { eye, right, up, fwd } = aim(cam);
      const axis = cam.narrow ? up.map((x) => -x) : right;
      gl.uniform3fv(B.camPos, eye); gl.uniform3fv(B.camX, right); gl.uniform3fv(B.camY, up); gl.uniform3fv(B.camZ, fwd);
      gl.uniform2f(B.lensSpread, TAN * W / S, TAN * H / S);
      gl.uniform2f(B.screenOffset, cam.offX * 2 * S / W, -cam.offY * 2 * S / H);
      gl.uniform3fv(B.box, half);
      gl.uniform1f(B.marchStep, Math.max(...half) * 2 / 300);
      const L = [-0.45, 0.75, 0.5], ll = Math.hypot(...L);
      gl.uniform3f(B.lightDir, L[0] / ll, L[1] / ll, L[2] / ll);
      gl.uniform3fv(B.histPos, axis.map((x) => x * k.histX * MM));
      gl.uniform3fv(B.mriPos, axis.map((x) => x * k.mriX * MM));
      gl.uniform1f(B.mriOn, bench.mri ? k.mriOn : 0);
      // the blade, pushed clear of the brain while it is put away
      const c = k.cut, sign = Math.sign(c.n[0] * eye[0] + c.n[1] * eye[1] + c.n[2] * eye[2] - c.d * MM) || 1;
      gl.uniform4f(B.cutPlane, c.n[0], c.n[1], c.n[2], c.d * MM + sign * (1 - c.amount) * 0.62);
      gl.uniform1f(B.cutOn, c.amount > 0.001 ? 1 : 0);
      gl.uniform1f(B.cutSign, sign);
      gl.uniform3fv(B.cutU, c.u); gl.uniform3fv(B.cutV, c.v);
      gl.uniform1f(B.cutGlow, c.glow);
      const N = 24, T = 2 * half[0] / N, G = k.gap * MM;
      gl.uniform1f(B.slabN, N); gl.uniform1f(B.slabT, T); gl.uniform1f(B.slabB, k.blade * N); gl.uniform1f(B.slabG, G);
      gl.uniform1f(B.slabShift, G * k.blade * N / 2);
      gl.uniform1f(B.jitter, k.jitter); gl.uniform1f(B.wave, k.wave * (N + 2));
      gl.uniform1f(B.knifeOn, k.knife);
      gl.uniform1f(B.knifeX, -half[0] + k.blade * N * T);
      gl.uniform1f(B.checker, k.checker); gl.uniform1f(B.tile, 0.072);
      // a deliberately wrong starting pose for the histology, which registration then removes
      const m = k.mis, a1 = 0.05 * m, a2 = -0.035 * m, s1 = 1 + 0.02 * m;   // about 3 and 2 degrees, 2%: the size of a real residual
      const ca = Math.cos(a1), sa = Math.sin(a1), cb = Math.cos(a2), sb = Math.sin(a2);
      gl.uniformMatrix3fv(B.misRot, false, new Float32Array([ca * s1, sa * s1, 0, -sa * cb * s1, ca * cb * s1, sb * s1, sa * sb * s1, -ca * sb * s1, cb * s1]));
      gl.uniform3f(B.misOff, 0.012 * m, -0.016 * m, 0.012 * m);
      gl.uniform1f(B.misWarp, 0.008 * m);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return { arriving: arrive > 0 && arrive < 1 };
    }

    function marchPass(cam, now) {
      if (cam.bench) return benchPass(cam, now);
      const i = Math.min(CHAIN.length - 1, Math.floor(cam.v));
      const frac = cam.v - i;
      const va = nearestVol(i), vb = frac < 0.0005 ? va : nearestVol(Math.min(CHAIN.length - 1, i + 1));   // resting on one brain: only that brain
      if (!va || !vb) return false;
      const mixv = va === vb ? 0 : frac;
      const half = va.half.map((h, k) => lerp(h, vb.half[k], mixv));
      const is24 = va === vols.dharani_24gw && vb === va;
      const isBig = va === vols.bigbrain && vb === va;

      // real sections take over as the cut planes arrive, as on the current site
      let faceOn = 0;
      if (face && is24 && cam.clip > 0.001) {
        const at = lerp(-half[0] * 1.05, GEOM.clipx * half[0], cam.clip);
        faceOn = 1 - smooth(0, 1, Math.abs(at - GEOM.clipx * half[0]) / (3 * MM));
      }
      let cutOn = 0, arriving = false;
      if (isBig && cuts.x && cuts.y && cuts.z && cam.cut > 0.001 && cam.hd > 0.001) {
        const arrive = Math.min(1, (now - Math.max(cuts.x.t0, cuts.y.t0, cuts.z.t0)) / 800);
        arriving = arrive < 1;
        cutOn = cam.hd * smooth(0, 1, arrive) * (1 - smooth(0, 1, Math.max(...half) * (1 - cam.cut) / (3 * MM)));
      }
      // while a section fills the cut, rays stop at it: screen resolution is affordable and needed
      const sharp = faceOn > 0.5 || cutOn > 0.5;
      // cut faces are drawn at the canvas's own resolution: rays stop at them, so it is affordable
      fitMarch(sharp ? Math.min(canvas.width / W, Math.sqrt(3.4e6 / (W * H))) * Math.max(quality, 0.8) : 0.62 * quality);

      gl.useProgram(volProg);
      gl.bindFramebuffer(gl.FRAMEBUFFER, march.fb);
      gl.viewport(0, 0, march.w, march.h);
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, va.tex || emptyVol);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_3D, vb.tex || emptyVol);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, face ? face.tex : blank);
      gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, cuts.x ? cuts.x.tex : blank);
      gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, cuts.y ? cuts.y.tex : blank);
      gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, cuts.z ? cuts.z.tex : blank);
      gl.uniform3fv(V.box, half);
      gl.uniform1f(V.morph, mixv);
      gl.uniform2f(V.depthGain, va.gain || 2, vb.gain || 2);
      gl.uniform1f(V.foldShade, cam.fold || 0);
      gl.uniform1f(V.foldReach, 0.012);
      gl.uniform1f(V.faceTone, 0);

      const { eye, right, up, fwd, off } = aim(cam);
      gl.uniform3fv(V.camPos, eye); gl.uniform3fv(V.camX, right); gl.uniform3fv(V.camY, up); gl.uniform3fv(V.camZ, fwd);
      gl.uniform2f(V.lensSpread, TAN * W / S, TAN * H / S);             // the field is measured on the short side
      gl.uniform2f(V.screenOffset, cam.offX * 2 * S / W, -cam.offY * 2 * S / H);
      gl.uniform1f(V.wedge, cam.cut);
      gl.uniform3f(V.wedgeSide, Math.sign(off[0]) || 1, Math.sign(off[1]) || 1, Math.sign(off[2]) || 1);
      gl.uniform1f(V.sliceAmount, cam.clip); gl.uniform1f(V.slicePlane, GEOM.clipx * half[0]);
      gl.uniform1f(V.marchStep, Math.max(half[0], half[1], half[2]) * 2 / 300);
      const L = [-0.45, 0.75, 0.5], ll = Math.hypot(...L);
      gl.uniform3f(V.lightDir, L[0] / ll, L[1] / ll, L[2] / ll);
      gl.uniform1f(V.planeWeight, faceOn);
      if (faceOn > 0) {
        const px = cam.F * MM / Math.min(march.w, march.h), texel = 2 * half[1] / face.rows;
        gl.uniform1f(V.planeLod, Math.max(0, Math.log2(px / texel)));
      } else gl.uniform1f(V.planeLod, 0);
      gl.uniform1f(V.wedgeWeight, cutOn);
      if (cutOn > 0) {
        gl.uniform1f(V.pixelFootprint, 2 * TAN * (H / S) / march.h);
        gl.uniform3f(V.wedgeTexel, 2 * half[1] / cuts.x.rows, 2 * half[2] / cuts.y.rows, 2 * half[1] / cuts.z.rows);
      }
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return { arriving };
    }

    // ---- pass 2 ----
    const mapA = new Float32Array(20), mapB = new Float32Array(10), layW = new Float32Array(5);

    // layer maps. p(s) = view + F * rot(roll) * s on the plane, s in short-side units, y down.
    function setMaps(cam) {
      const { view, F, roll } = cam, c = Math.cos(roll) * F, n = Math.sin(roll) * F;
      // face: rows are y of the volume (up), columns z
      const fz = 1 / GEOM.faceZ, fy = -1 / GEOM.faceY;
      mapA.set([fz * c, -fz * n, fy * n, fy * c], 0);
      mapB.set([view[0] * fz + 0.5, view[1] * fy + 0.5], 0);
      // nissl: n(s) = n0 + rot(-theta)(p - a0)
      const { n0, a0, theta } = GEOM.align;
      const b = rot(-theta, view[0] - a0[0], view[1] - a0[1]);
      const bx = n0[0] + b[0], by = n0[1] + b[1];
      const k = roll - theta, mc = Math.cos(k) * F, ms = Math.sin(k) * F;
      const viewWide = F * W / S;
      levels.forEach((l, i) => {
        const [r0, r1, r2, r3] = l.rect;
        const sx = 1 / (GEOM.nisslWidth * (r2 - r0)), sy = 1 / (GEOM.nisslHeight * (r3 - r1));
        mapA.set([sx * mc, -sx * ms, sy * ms, sy * mc], (i + 1) * 4);
        mapB.set([sx * bx - r0 / (r2 - r0), sy * by - r1 / (r3 - r1)], (i + 1) * 2);
        const cover = (r2 - r0) * GEOM.nisslWidth / viewWide;
        let w = i === 0 ? 1 : smooth(0.12, 0.45, cover);
        const next = levels[i + 1];
        if (next && (next.rect[2] - next.rect[0]) * GEOM.nisslWidth / viewWide > 2.2) w = 0;   // fully covered
        layW[i + 1] = l.tex ? w * cam.nisW : 0;
      });
      layW[0] = face ? cam.faceW : 0;
    }

    function draw(cam, now) {
      let busy = false;
      let marched = false;
      if (cam.volW > 0.001) {
        const r = marchPass(cam, now);
        if (r) { marched = true; busy = r.arriving; }
      }
      setMaps(cam);
      gl.useProgram(compProg);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, marched ? march.tex : blank);
      gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, face ? face.tex : blank);
      for (let i = 0; i < 4; i++) {
        gl.activeTexture(gl.TEXTURE8 + i);
        gl.bindTexture(gl.TEXTURE_2D, levels[i] && levels[i].tex ? levels[i].tex : blank);
      }
      const srcMm = (levels[3].rect[2] - levels[3].rect[0]) * GEOM.nisslWidth / cell.w;   // mm per source pixel
      const shortPx = S * canvas.width / W;
      const mag = shortPx / (cam.F / srcMm);
      gl.uniform2f(C.canvasSize, canvas.width, canvas.height);
      gl.uniform1f(C.shortPx, shortPx);
      gl.uniform1f(C.volWeight, marched ? cam.volW : 0);
      gl.uniform1f(C.dimAmount, cam.dim);
      gl.uniform1f(C.scrimAmount, cam.scrim);
      gl.uniform1f(C.scrimMode, cam.narrow ? 1 : 0);
      gl.uniform4fv(C['mapA[0]'], mapA);
      gl.uniform2fv(C['mapB[0]'], mapB);
      gl.uniform1fv(C['layWeight[0]'], layW);
      gl.uniform3f(C.densCurve, 0.08, 1.4, 0.8);
      gl.uniform2f(C.cellSize, cell.w, cell.h);
      gl.uniform1f(C.cellMag, mag);
      gl.uniform1f(C.deepCurve, 1 - smooth(750, 2250, cam.F / srcMm));     // by field of view, so a phone matches a laptop
      gl.uniform1f(C.pixAmount, smooth(14, 30, mag));
      gl.uniform1f(C.stretchAmount, cam.stretch);
      gl.uniform2f(C.stretchRange, cell.lo, cell.hi);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      return busy;
    }

    // a section of the 24-week histology, printed once into a small 2D canvas: the target to find
    function printSection(plane, into) {
      const data = bench.hist, h = vols.dharani_24gw;
      if (!data || !h) return false;
      const [gu, gv, gw] = h.meta.grid, sp = h.meta.voxel_mm, n = into.width, g = into.getContext('2d'), img = g.createImageData(n, n);
      const span = 86;
      const stops = [[0, [8, 7, 15]], [0.3, [38, 23, 102]], [0.62, [112, 74, 237]], [0.85, [204, 184, 255]], [1, [247, 242, 255]]];
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const a = (i / n - 0.5) * span, b = (0.5 - j / n) * span;
        const p = [0, 1, 2].map((k) => plane.n[k] * plane.d + plane.u[k] * a + plane.v[k] * b);
        const x = Math.round(p[0] / sp + gu / 2), y = Math.round(p[1] / sp + gv / 2), z = Math.round(p[2] / sp + gw / 2);
        let d = 0;
        if (x >= 0 && y >= 0 && z >= 0 && x < gu && y < gv && z < gw) d = data[((z * gv + y) * gu + x) * 2] / 255;
        let c = stops[0][1];
        for (let q = 0; q < 4; q++) if (d >= stops[q][0]) { const f = (d - stops[q][0]) / (stops[q + 1][0] - stops[q][0]); c = stops[q][1].map((v, r) => v + (stops[q + 1][1][r] - v) * Math.min(1, f)); }
        const o = (j * n + i) * 4;
        img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
      }
      g.putImageData(img, 0, 0);
      return true;
    }

    canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); opts.onLost(); });

    return {
      load, resize, draw, sectionCentre, cellTarget, aim, printSection,
      halfMm: () => (vols.dharani_24gw ? vols.dharani_24gw.meta.extent_mm.map((x) => x / 2) : [30.6, 37.8, 34.2]),
      setQuality: (q) => { quality = q; },
      planeX: () => GEOM.clipx * (vols.dharani_24gw ? vols.dharani_24gw.meta.extent_mm[0] / 2 : 30.6),
      theta: () => GEOM.align.theta,
      geom: GEOM,
    };
  }

  window.createStage = createStage;
})();
