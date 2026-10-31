"""Write web volumes: uint8 RG pairs (R = stain density, G = depth below the outer surface), gzip, axes
u: left->right, v: inferior->superior, w: posterior->anterior, stored C order [w][v][u]."""
import os, sys, json, gzip
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, nibabel as nib
from scipy import ndimage as ndi

SB = "/home/users/jignesh/.local/tmp/claude-1119/-home-users-jignesh/316a72fa-d44c-48eb-876a-e114ab9dce9a/scratchpad"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

def largest(m, keep=1):
    lab, n = ndi.label(m)
    if n == 0: return m
    sz = ndi.sum(m, lab, range(1, n + 1)); order = np.argsort(sz)[::-1][:keep]
    return np.isin(lab, order + 1)

def finish(dens, mask, spacing, shell_mm, pad=3, smooth=1.6, smooth_axis0=True):
    """crop to the mask, pad, quantise density and depth"""
    idx = np.argwhere(mask); lo = np.maximum(idx.min(0) - pad, 0); hi = np.minimum(idx.max(0) + pad + 1, mask.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    dens, mask = dens[sl], mask[sl]
    # smooth the outline so section-to-section steps do not show as terraces on the surface
    mask = ndi.gaussian_filter(mask.astype(np.float32), smooth) > 0.5
    dens = ndi.gaussian_filter(dens, (smooth * 0.6, smooth * 0.3, smooth * 0.3) if smooth_axis0 else smooth * 0.3)
    depth = ndi.gaussian_filter(ndi.distance_transform_edt(mask) * spacing, 1.0)
    r = np.where(mask, np.clip(dens, 0, 1) * 254 + 1, 0).astype(np.uint8)
    g = (np.clip(depth / shell_mm, 0, 1) * 255).astype(np.uint8)
    return np.stack([r, g], -1)

def write(name, vol, spacing, extra):
    w, v, u = vol.shape[:3]
    raw = np.ascontiguousarray(vol).tobytes()
    with gzip.open(f"{OUT}/{name}.bin.gz", "wb", compresslevel=9) as f: f.write(raw)
    meta = dict(id=name, file=f"{name}.bin.gz", dims=[u, v, w], spacing_mm=spacing,
                size_mm=[round(u * spacing, 2), round(v * spacing, 2), round(w * spacing, 2)], **extra)
    print(name, meta["dims"], meta["size_mm"], os.path.getsize(f"{OUT}/{name}.bin.gz") // 1024, "KB", flush=True)
    return meta

def rotate_plane(a, ang, flip):
    """same pose mapping as orient_search.py: output (y, x) samples input at a rotation of the canonical frame"""
    ny, nx = a.shape[1:]
    R = max(ny, nx)
    yy, xx = np.mgrid[0:R, 0:R].astype(np.float32) - R / 2
    s = -1 if flip else 1
    cy, cx = ny / 2, nx / 2
    t = np.deg2rad(ang)
    ys = cy + (np.cos(t) * yy - np.sin(t) * xx * s)
    xs = cx + (np.sin(t) * yy + np.cos(t) * xx * s)
    return np.stack([ndi.map_coordinates(sl, [ys, xs], order=1, cval=0) for sl in a])

metas = []
# ---- DHARANI (mcmadras pipeline stacks, already resampled to 0.32 mm iso) ----
orient = json.load(open(f"{SB}/site_build/dharani/orient.json"))
info = json.load(open(f"{SB}/site_build/dharani/info.json"))
SP = 0.36
for sub in ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005"]:
    d = np.load(f"{SB}/site_build/dharani/{sub}_032.npy")
    if sub == "sub-004":
        d = d.transpose(2, 1, 0)            # coronal stack -> (x, y, z): axis 0 = left-right, plane (y, z)
    o = orient[sub]
    d = rotate_plane(d, o["angle"], o["flip"])
    if o["flip"]: d = d[::-1]               # keep the handedness proper
    d = ndi.zoom(d, 0.32 / SP, order=1)
    pos = d[d > 0]
    lo, hi = np.percentile(pos, [2, 99.7])
    dens = (d - lo) / (hi - lo)
    mask = d > np.percentile(pos, 12)
    mask = ndi.binary_opening(mask, iterations=2)
    mask = largest(mask)
    mask = ndi.binary_closing(mask, iterations=2) & (d > 0)
    mask = ndi.binary_fill_holes(mask)
    # rows: y down = inferior, cols: x right = posterior  ->  v = -y, w = -x, u = axis 0
    vol = finish(np.clip(dens, 0, 1) ** 0.85, mask, SP, 1.6)          # (u, y, x, 2)
    vol = vol[:, ::-1, ::-1].transpose(2, 1, 0, 3)                   # (w, v, u, 2)
    metas.append(write(f"dharani_{info[sub]['gw']:02d}gw", vol, SP,
                       dict(gw=info[sub]["gw"], subject=sub, brain=info[sub]["brain"])))

# ---- BigBrain 200 um -> 0.6 mm, RAS -> (w=A, v=S, u=R) ----
bb = f"{SB}/bigbrain_dl"
d = np.asarray(nib.load(f"{bb}/full8_200um_optbal.nii.gz").dataobj).astype(np.float32)
c = np.asarray(nib.load(f"{bb}/full_cls_200um.nii.gz").dataobj)
tissue = np.isin(c, [2, 3, 5, 6, 8, 9])
dens = np.where(tissue, (65535 - d) / 65535.0, 0)
f = 0.2 / 0.6
dens = ndi.zoom(dens, f, order=1); m = ndi.zoom(tissue.astype(np.float32), f, order=1) > 0.5
m = ndi.binary_fill_holes(largest(m))
p = dens[m]; lo, hi = np.percentile(p, [1, 99.8])
vol = finish((dens - lo) / (hi - lo), m, 0.6, 3.0, smooth=0.8, smooth_axis0=False)                  # (R, A, S, 2)
vol = vol.transpose(1, 2, 0, 3)                                     # (A, S, R, 2) = (w, v, u)
metas.append(write("bigbrain", vol, 0.6, dict(name="BigBrain")))

json.dump(dict(axes="u: left->right, v: inferior->superior, w: posterior->anterior; uint8 pairs (density, depth), C order [w][v][u][2], gzip",
               credit="DHARANI fetal brains (Verma et al., J Comp Neurol 2025, CC BY 4.0), stacked with the mcmadras pipeline; BigBrain (Amunts et al., Science 2013, CC BY-NC-SA 4.0). Downsampled for the web.",
               volumes=metas), open(f"{OUT}/volumes.json", "w"), indent=1)
