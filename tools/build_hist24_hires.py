"""The 24-week Nissl on the web grid at high resolution, for the blade's cut face (assets/dive/stage.js).

Takes the same 64 um stack that dharani_24gw comes from (histo_stacked.mnc, through prep_dharani.py and
build_volumes.py), repeats that chain's crop, pose and scale exactly, and samples the stack on a finer grid of the
same box. Written as WebP sheets of sections, one sheet per `per` sections along the sectioning axis u:
sheet k, tile j = section k * per + j; tile rows = v (inferior first), tile columns = w (posterior first).

usage: python build_hist24_hires.py DHARANI_DIR SITE_ASSETS CACHE.npy [VOXEL_MM] [QUALITY]
"""
import os, sys, json, gzip
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from PIL import Image

STACK = "/home/users/jignesh/workspace/application/tirl/mcmadras_reg/madras_run/results/B_plus4_nl3D/sub-005/align_nissl_1u/optical/histo_stacked.mnc"
D, A, CACHE = sys.argv[1:4]
HV = float(sys.argv[4]) if len(sys.argv) > 4 else 0.128
Q = int(sys.argv[5]) if len(sys.argv) > 5 else 82
SP = 0.36
PER, COLS = 16, 4

def largest(m):
    lab, n = ndi.label(m)
    return lab == 1 + np.argmax(ndi.sum(m, lab, range(1, n + 1)))

def rotate_plane(a, ang):
    ny, nx = a.shape[1:]; R = max(ny, nx)
    yy, xx = np.mgrid[0:R, 0:R].astype(np.float32) - R / 2
    t = np.deg2rad(ang)
    ys = ny / 2 + (np.cos(t) * yy - np.sin(t) * xx)
    xs = nx / 2 + (np.sin(t) * yy + np.cos(t) * xx)
    return np.stack([ndi.map_coordinates(sl, [ys, xs], order=1, cval=0) for sl in a])

# ---- the web volume's chain at 0.32 / 0.36 mm: its crop box and density scale ----
d32 = np.load(f"{D}/sub-005_032.npy"); n32 = np.array(d32.shape)
ang = json.load(open(f"{D}/orient.json"))["sub-005"]["angle"]
assert not json.load(open(f"{D}/orient.json"))["sub-005"]["flip"]
rot = rotate_plane(d32, ang); nrot = np.array(rot.shape)
d = ndi.zoom(rot, 0.32 / SP, order=1); nz = np.array(d.shape)
pos = d[d > 0]; lo, hi = np.percentile(pos, [2, 99.7])
m = ndi.binary_opening(d > np.percentile(pos, 12), iterations=2)
m = ndi.binary_closing(largest(m), iterations=2) & (d > 0)
m = ndi.binary_fill_holes(m)
idx = np.argwhere(m); clo = np.maximum(idx.min(0) - 3, 0); chi = np.minimum(idx.max(0) + 4, m.shape)
cu, cy, cx = chi - clo
print("box", clo, chi, "grid (u, v, w)", cu, cy, cx, "density scale", lo, hi, flush=True)

meta = [b for b in json.load(open(f"{A}/volumes/brains.json"))["brains"] if b["id"] == "dharani_24gw"][0]
assert meta["grid"] == [int(cu), int(cy), int(cx)], meta["grid"]
web = np.frombuffer(gzip.open(f"{A}/volumes/dharani_24gw.bin.gz").read(), np.uint8).reshape(cx, cy, cu, 2)[..., 0]   # (w, v, u)

nu, nv, nw = [int(round(n * SP / HV)) for n in (cu, cy, cx)]
print("high-resolution grid (u, v, w)", nu, nv, nw, round(nu * nv * nw / 1e6), "M voxels", flush=True)

def stack_coords(u, v, w, nst):
    """web-grid voxel coordinates -> voxel coordinates of the 64 um stack"""
    a0, yy, xx = clo[0] + u, clo[1] + (cy - 1 - v), clo[2] + (cx - 1 - w)
    k = (nrot - 1) / (nz - 1)                                     # ndi.zoom: output i reads input i * k
    a0, yy, xx = a0 * k[0], yy * k[1] - nrot[1] / 2, xx * k[2] - nrot[2] / 2
    t = np.deg2rad(ang)
    ys = n32[1] / 2 + (np.cos(t) * yy - np.sin(t) * xx)
    xs = n32[2] / 2 + (np.sin(t) * yy + np.cos(t) * xx)
    return [(c + 0.5) * nst[i] / n32[i] - 0.5 for i, c in enumerate((a0, ys, xs))]    # area resampling: centres

if os.path.exists(CACHE):
    vol = np.load(CACHE); assert vol.shape == (nu, nv, nw), vol.shape
else:
    im = nib.load(STACK)
    a = np.asarray(im.dataobj, dtype=np.float32); print("stack", a.shape, flush=True)
    nst = np.array(a.shape)
    s = (HV / 0.064 - 1) * 0.8                                     # low-pass to the new voxel before sampling
    if s > 0: a = ndi.gaussian_filter(a, s)
    print("filtered", flush=True)
    keep = ndi.binary_dilation(web > 0, iterations=1).astype(np.float32)
    vs = (np.arange(nv) + 0.5) * cy / nv - 0.5; ws = (np.arange(nw) + 0.5) * cx / nw - 0.5
    V, W = np.meshgrid(vs, ws, indexing="ij")
    vol = np.zeros((nu, nv, nw), np.uint8)
    for i in range(nu):
        u = (i + 0.5) * cu / nu - 0.5
        c = stack_coords(np.full_like(V, u), V, W, nst)
        x = ndi.map_coordinates(a, c, order=1, mode="constant")
        k = ndi.map_coordinates(keep, [W, V, np.full_like(V, u)], order=1, mode="constant") > 0.5
        vol[i] = np.where(k, np.clip((x - lo) / (hi - lo), 0, 1) ** 0.85 * 254 + 1, 0).astype(np.uint8)
        if i % 60 == 0: print(i, flush=True)
    np.save(CACHE, vol)

vs = (np.arange(nv) + 0.5) * cy / nv - 0.5; ws = (np.arange(nw) + 0.5) * cx / nw - 0.5
V, W = np.meshgrid(vs, ws, indexing="ij")
webf = web.astype(np.float32)
coarse = np.stack([ndi.map_coordinates(webf, [W, V, np.full_like(V, (i + 0.5) * cu / nu - 0.5)], order=1, mode="constant") for i in range(nu)]).astype(np.uint8)

# ---- the stack has empty and part-empty slots (lost sections), which the web volume averages over and an oblique cut
# would show as dark lines: in sections missing much of their tissue, fill the holes from the nearest whole sections ----
tissue = coarse > 20
hole = (vol < 0.5 * coarse) & tissue
frac = hole.reshape(nu, -1).sum(1) / np.maximum(tissue.reshape(nu, -1).sum(1), 1)
flag = frac > max(0.04, 4 * np.median(frac))
good = np.where(~flag)[0]
print("hole fraction, median", round(float(np.median(frac)), 4), "sections repaired", int(flag.sum()), "of", nu, flush=True)
for i in np.where(flag)[0]:
    a, b = good[good < i], good[good > i]
    if not len(a) or not len(b): continue
    a, b = a[-1], b[0]; t = (i - a) / (b - a)
    fill = (1 - t) * vol[a].astype(np.float32) + t * vol[b]
    vol[i] = np.where(hole[i], fill, vol[i]).astype(np.uint8)

# ---- stain varies from section to section (0.76x to 1.27x here), which the web volume averages away and an oblique
# cut would show as stripes: take the slow brightness from the web volume, keep only the fine detail of each section ----
SIG = 0.77 / HV
even = np.zeros_like(vol)
for i in range(nu):
    f = vol[i].astype(np.float32); c = coarse[i].astype(np.float32)
    t = ((f > 3) & (c > 3)).astype(np.float32)
    w = ndi.gaussian_filter(t, SIG) + 1e-3
    gain = np.clip((ndi.gaussian_filter(c * t, SIG) / w + 1) / (ndi.gaussian_filter(f * t, SIG) / w + 1), 0.4, 2.5)
    even[i] = np.where(f > 0, np.clip(f * gain, 1, 255), 0).astype(np.uint8)
vol = even

# ---- check: back on the web grid it must be the web volume ----
small = ndi.zoom(ndi.gaussian_filter(vol.astype(np.float32), 1.2), (cu / nu, cy / nv, cx / nw), order=1).transpose(2, 1, 0)   # (w, v, u)
q = web > 3
print("correlation with the web volume", round(float(np.corrcoef(small[q], web[q].astype(np.float32))[0, 1]), 4), flush=True)

# ---- sheets ----
out = f"{A}/volumes/hist24"; os.makedirs(out, exist_ok=True)
for f in os.listdir(out): os.remove(f"{out}/{f}")
rows = PER // COLS; total = 0; files = []
for k in range(0, nu, PER):
    sheet = np.zeros((rows * nv, COLS * nw), np.uint8)
    for j in range(min(PER, nu - k)):
        r, c = divmod(j, COLS)
        sheet[r * nv:(r + 1) * nv, c * nw:(c + 1) * nw] = vol[k + j]
    name = f"s{k // PER:03d}.webp"
    Image.fromarray(sheet).save(f"{out}/{name}", quality=Q, method=6)
    total += os.path.getsize(f"{out}/{name}"); files.append(name)
json.dump(dict(grid=[nu, nv, nw], voxel_mm=HV, per=PER, cols=COLS, files=files), open(f"{out}/index.json", "w"), indent=1)
print(len(files), "sheets", round(total / 1048576, 1), "MB at quality", Q, flush=True)
