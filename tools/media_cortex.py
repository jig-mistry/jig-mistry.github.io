"""The cytoarchitecture stop's cortex: zoom levels of the cortical wall on the same Nissl section as the eminence
(24 GW, DHARANI FB85, SE 1138), the three-colour immunostained wall (another fetal brain) laid on it, and its zones.

The site is where the wall's pial surface runs straight (y 63000-68000 of the full scan, 12 um off a line): the pia,
its direction and the wall's normal are fitted there, and the zones are read off the Nissl density along the normal
(averaged 1 mm either way along the pia), each boundary where the density crosses half way between its neighbours.
The camera's fixed point is just inside the plate's inner edge, so all three zones are in view. The immunostained wall (cyto/ihc_layers.webp, 2.5 um per
pixel, pia on its right) is turned so its pia lies along this one, at its own scale, and written in the section's frame
with its tissue as alpha. Everything goes to img/cyto/cortex_zoom.json in the section's 0..1 coordinates, like
img/ge/nissl_zoom.json.

usage: python media_cortex.py SITE_DIR
"""
import os, sys, json, subprocess, tempfile
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
Image.MAX_IMAGE_PIXELS = None
SITE = sys.argv[1]
OPJ = "/home/users/jignesh/miniconda3/envs/nisslprop/bin/opj_decompress"
J85 = "/home/users/jignesh/analytics/244/NISL/B_244_FB85-SL_380-ST_NISL-SE_1138_compressed.jp2"
W85, H85, UM = 144472, 129351, 0.5                 # full scan, um per pixel
TMP = tempfile.mkdtemp()

def jp2(r, box):
    out = os.path.join(TMP, "c.png")
    subprocess.run([OPJ, "-i", J85, "-o", out, "-r", str(r), "-d", ",".join(map(str, box)), "-threads", "8"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return Image.open(out).convert("RGB")
def save(im, name, q=80):
    f = f"{SITE}/assets/img/cyto/{name}.webp"; im.save(f, quality=q, method=6)
    print(name, im.size, os.path.getsize(f) // 1024, "KB", flush=True)
    return f"assets/img/cyto/{name}.webp"

# ---- the pia and the zones, on a 4 um view of the wall ----
X0, Y0 = 0, 50000
od = -np.log(np.clip(np.asarray(jp2(3, (X0, Y0, X0 + 16000, Y0 + 20000)), np.float32) / 255, 1 / 255, 1)).sum(-1)
tissue = ndi.binary_opening(ndi.gaussian_filter(od, 4) > 0.42, iterations=3)          # glass ~0.3, marginal zone ~0.5
rows = np.arange(tissue.shape[0]); ys = Y0 + rows * 8.0
pia_x = X0 + np.array([np.argmax(r) for r in tissue]) * 8.0
k = (ys > 63000) & (ys < 68000)
a, b = np.polyfit(ys[k], pia_x[k], 1)
print("pia off its line, um", round(float((pia_x[k] - (a * ys[k] + b)).std() * UM), 1))
YS = 65500.0; P = np.array([a * YS + b, YS])                      # the site on the pia, full pixels
d = np.array([a, 1.0]) / np.hypot(a, 1.0); n = np.array([d[1], -d[0]])     # along the pia (down), into the wall
sm = ndi.gaussian_filter(od, 2)
dist = np.arange(-400, 9000, 20.0); along = np.arange(-2000, 2001, 100.0)
prof = np.array([ndi.map_coordinates(sm, [(P[1] + n[1] * s + d[1] * along - Y0) / 8, (P[0] + n[0] * s + d[0] * along - X0) / 8], order=1).mean() for s in dist])
mm = dist * UM / 1000
def level(lo_mm, hi_mm): return float(np.median(prof[(mm >= lo_mm) & (mm < hi_mm)]))
mz, cp, sp = level(0.1, 0.3), level(0.5, 1.2), level(2.4, 3.4)
inner = (mm > 0.05)
mz_cp = float(mm[inner & (prof > (mz + cp) / 2)][0])                            # first rise into the plate
low = level(1.4, 1.8)                                                            # the plate's lower, paler part
cp_sp = float(mm[(mm > mz_cp + 0.3) & (prof < (low + sp) / 2)][0])
print(f"zones, mm from the pia: marginal 0-{mz_cp:.2f}, plate {mz_cp:.2f}-{cp_sp:.2f}, subplate {cp_sp:.2f}-")
T = P + n * (cp_sp - 0.1) * 1000 / UM                                            # the plate's deep edge

# ---- zoom levels around it: whole field of the crossing's last stretch, the stop, and the cells ----
levels = []
for name, half_mm, r, size in (("nissl_cortex", 12, 3, 2000), ("nissl_cortex_mid", 4, 1, 2000), ("nissl_cortex_cells", 1, 0, 2000)):
    h = half_mm / 2 * 1000 / UM
    x0, y0 = int(max(0, T[0] - h)), int(T[1] - h); x1, y1 = int(x0 + 2 * h), int(y0 + 2 * h)
    im = jp2(r, (x0, y0, x1, y1)).resize((size, size), Image.Resampling.LANCZOS)
    levels.append(dict(src=save(im, name), rect=[x0 / W85, y0 / H85, x1 / W85, y1 / H85], px=[size, size]))

# ---- the immunostained wall, turned onto this pia ----
ihc = np.asarray(Image.open(f"{SITE}/assets/img/cyto/ihc_layers.webp").convert("RGB"), np.float32)
S_IHC = 2.5 / UM                                                                  # full-scan pixels per ihc pixel
sig = ndi.gaussian_filter(ihc.max(-1), 3)
t_ihc = sig > 45; lab, m = ndi.label(t_ihc); t_ihc = ndi.binary_fill_holes(lab == 1 + int(np.argmax(ndi.sum(t_ihc, lab, range(1, m + 1)))))
r_ihc = np.arange(t_ihc.shape[0]); edge = np.array([np.nonzero(r)[0].max() if r.any() else -1 for r in t_ihc], float)
ok = edge > 0; ai, bi = np.polyfit(r_ihc[ok], edge[ok], 1)
di = np.array([ai, 1.0]) / np.hypot(ai, 1.0)                                     # its pia, downwards; tissue to its left
# turn so that -di runs along d (both keep tissue on the same hand), then put its pia's middle on the site's
ang = np.arctan2(d[1], d[0]) - np.arctan2(-di[1], -di[0])
Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
mid = np.array([ai * t_ihc.shape[0] / 2 + bi, t_ihc.shape[0] / 2])                # ihc pixel (x, y) on its pia
along_off = (T - P) @ d                                                          # centre it on the stop's fixed point
anchor = P + d * along_off
def to_scan(q): return anchor + (Rm @ ((q - mid) * S_IHC).T).T                   # ihc (x, y) -> full-scan (x, y)
hh, ww = t_ihc.shape
corners = to_scan(np.array([[0, 0], [ww, 0], [0, hh], [ww, hh]], float))
x0, y0 = corners.min(0); x1, y1 = corners.max(0)
NX, NY = int(np.ceil((x1 - x0) / S_IHC)), int(np.ceil((y1 - y0) / S_IHC))
gx, gy = np.meshgrid(x0 + (np.arange(NX) + 0.5) * S_IHC, y0 + (np.arange(NY) + 0.5) * S_IHC)
q = (Rm.T @ (np.stack([gx.ravel(), gy.ravel()]) - anchor[:, None])) / S_IHC + mid[:, None]     # back into the ihc
# the tissue runs on past the crop's edges: fade over 0.3 mm there, so only the pia is a hard edge
fade = np.clip(ndi.distance_transform_edt(np.pad(np.ones((hh - 2, ww - 2)), 1)) / (300 / 2.5), 0, 1)
alpha = ndi.map_coordinates(ndi.gaussian_filter(t_ihc.astype(np.float32), 1.5) * fade ** 1.5, [q[1], q[0]], order=1, cval=0).reshape(NY, NX)
rgb = np.stack([ndi.map_coordinates(ihc[..., c], [q[1], q[0]], order=1, cval=0).reshape(NY, NX) for c in range(3)], -1)
ov = Image.fromarray(np.dstack([np.clip(rgb, 0, 255), np.clip(alpha * 255, 0, 255)]).astype(np.uint8), "RGBA")
f = f"{SITE}/assets/img/cyto/ihc_on_wall.webp"; ov.save(f, quality=84, alpha_quality=70, method=6)
print("ihc_on_wall", ov.size, os.path.getsize(f) // 1024, "KB", "turned", round(float(np.degrees(ang)), 1), "deg", flush=True)

u = lambda p: [round(float(p[0] / W85), 6), round(float(p[1] / H85), 6)]
spec = dict(levels=levels,
            overlay=dict(src="assets/img/cyto/ihc_on_wall.webp", rect=[x0 / W85, y0 / H85, x1 / W85, y1 / H85], px=[NX, NY]),
            target=u(T), pia=u(anchor), along=[round(float(d[0]), 5), round(float(d[1]), 5)], normal=[round(float(n[0]), 5), round(float(n[1]), 5)],
            zones=[["marginal zone", 0, round(mz_cp, 3)], ["cortical plate", round(mz_cp, 3), round(cp_sp, 3)], ["subplate", round(cp_sp, 3), round(cp_sp + 0.7, 3)]])
json.dump(spec, open(f"{SITE}/assets/img/cyto/cortex_zoom.json", "w"), indent=1)
print(json.dumps({k: v for k, v in spec.items() if k != "levels"}))
