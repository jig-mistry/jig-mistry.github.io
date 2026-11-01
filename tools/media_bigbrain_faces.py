"""High-resolution sections for BigBrain's octant cut (the cytoarchitecture chapter).

At cut = 1 the three cut planes meet in the middle of the web volume's box.  This takes the three planes through
that point from the 100 um release (full8_100um_optbal + full_cls_100um, same histological space as the 200 um
files the web volume is made from), recomputes the box exactly as tools/build_volumes.py does, and writes each
plane in the box frame as grey density (on the volume's own scale) + tissue alpha:
  bb_cut_x.webp  sagittal, rows = v (inferior first), columns = w (posterior first)
  bb_cut_y.webp  axial,    rows = w,                   columns = u (left first)
  bb_cut_z.webp  coronal,  rows = v,                   columns = u

usage: python media_bigbrain_faces.py BIGBRAIN_DIR SITE_DIR
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from PIL import Image

BB, SITE = sys.argv[1], sys.argv[2]
TISSUE = [2, 3, 5, 6, 8, 9]
PX = 0.1            # mm per face pixel

def largest(m):
    lab, n = ndi.label(m)
    if n == 0: return m
    return lab == 1 + np.argmax(ndi.sum(m, lab, range(1, n + 1)))

# ---- the web volume's box and density scale, as in build_volumes.py ----
im2 = nib.load(f"{BB}/full8_200um_optbal.nii.gz")
d = np.asarray(im2.dataobj).astype(np.float32)
tissue = np.isin(np.asarray(nib.load(f"{BB}/full_cls_200um.nii.gz").dataobj), TISSUE)
dens = np.where(tissue, (65535 - d) / 65535.0, 0)
f = 0.2 / 0.6
dens6 = ndi.zoom(dens, f, order=1); m = ndi.zoom(tissue.astype(np.float32), f, order=1) > 0.5
m = ndi.binary_fill_holes(largest(m))
lo, hi = np.percentile(dens6[m], [1, 99.8])
idx = np.argwhere(m)
blo = np.maximum(idx.min(0) - 3, 0); bhi = np.minimum(idx.max(0) + 3 + 1, m.shape)
n = bhi - blo                                                   # (R, A, S) = (u, w, v) voxels of the web volume
k = (np.array(d.shape) - 1) / (np.array(m.shape) - 1)          # ndi.zoom maps output i to input i * k
print("box", blo, bhi, "dims (u, v, w)", n[0], n[2], n[1], "density scale", lo, hi, flush=True)
del d, dens, dens6, tissue

# ---- 100 um data, kept as raw uint8 ----
im1 = nib.load(f"{BB}/full8_100um_optbal.nii.gz")
raw = np.asarray(im1.dataobj.get_unscaled())
slope, inter = im1.dataobj.slope, im1.dataobj.inter
cls = np.asarray(nib.load(f"{BB}/full_cls_100um.nii.gz").dataobj)
to1 = np.linalg.inv(im1.affine) @ im2.affine                    # 200 um voxel -> 100 um voxel
print("100 um", raw.shape, raw.dtype, "slope", slope, "inter", inter, flush=True)

def plane(normal, rows, cols):
    """sample the plane through the box centre normal to array axis `normal`; rows/cols are array axes"""
    nr, nc = int(round(n[rows] * 0.6 / PX)), int(round(n[cols] * 0.6 / PX))
    g6 = np.zeros((3, nr, nc), np.float64)
    g6[normal] = blo[normal] - 0.5 + n[normal] / 2
    g6[rows] = (blo[rows] - 0.5 + (np.arange(nr) + 0.5) / nr * n[rows])[:, None]
    g6[cols] = (blo[cols] - 0.5 + (np.arange(nc) + 0.5) / nc * n[cols])[None, :]
    g2 = g6 * k[:, None, None]
    g1 = np.einsum("ij,jrc->irc", to1[:3, :3], g2) + to1[:3, 3, None, None]
    val = ndi.map_coordinates(raw, g1, order=1, mode="nearest").astype(np.float32) * slope + inter
    t = ndi.map_coordinates(np.isin(cls, TISSUE).view(np.uint8), g1, order=1, mode="constant").astype(np.float32)
    # nearest class decides which pixels count as tissue at full resolution
    tn = np.isin(ndi.map_coordinates(cls, g1, order=0, mode="constant"), TISSUE)
    dens = np.clip(((65535 - val) / 65535.0 - lo) / (hi - lo), 0, 1)
    alpha = np.clip(t * 1.5 - 0.25, 0, 1) * ndi.binary_opening(tn, iterations=1)
    alpha = ndi.gaussian_filter(alpha, 0.6)
    # spread tissue values into the transparent parts so texture filtering never pulls in white
    mask = alpha > 0.5
    near = ndi.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    dens = dens[near[0], near[1]]
    return dens, alpha

os.makedirs(f"{SITE}/assets/img/cyto", exist_ok=True)
for name, (normal, rows, cols) in {"x": (0, 2, 1), "y": (2, 1, 0), "z": (1, 2, 0)}.items():
    dn, al = plane(normal, rows, cols)
    d8 = (dn * 255).round().astype(np.uint8)
    out = f"{SITE}/assets/img/cyto/bb_cut_{name}.webp"
    Image.fromarray(np.dstack([d8, d8, d8, (al * 255).round().astype(np.uint8)]), "RGBA").save(out, quality=82, alpha_quality=70, method=6)
    np.save(f"bb_cut_{name}_dens.npy", dn.astype(np.float16))
    print(name, d8.shape, "tissue", round(float((al > 0.5).mean()), 3), os.path.getsize(out) // 1024, "KB", flush=True)
