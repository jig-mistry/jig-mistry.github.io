"""The stroke brain (HB02, sub-001 of the stroke dataset) for the web, in its ex vivo MRI space.

  hb02.bin.gz        RG: coarse T1 density (the shape: faces come from the sheets), depth below the surface, same
                     layout as the other brains (0.6 mm)
  hb02_meta.bin.gz   RGBA, stored one channel after another (0.8 mm): block id (1-5 along left-right, 6 cerebellum and stem), the stroke (the lab's
                     segmentation, smooth: the page draws its outline at one half), the cerebellum and stem (smooth, so
                     they come away along a surface rather than voxels), and the lesion's brightness on diffusion MRI
  hb02_mri/          MRI for the cut faces, WebP sheets of sections along u, as hist24 (0.5 mm)

The one face the stop looks straight at comes from build_hb02_faces.py and fit_hb02_blockface.py.

All of it is already registered to one space: BFI_in_MRI_2 (block face in MRI), the Nissl reconstruction and the
in vivo diffusion scan. The stroke is the lab's own segmentation of the diffusion lesion, 11 days before death, already
moved into ex vivo space. Headers are true RAS here, so u = +x, v = +z, w = +y.

usage: python build_hb02.py SITE_ASSETS
"""
import os, sys, json, gzip
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from PIL import Image
A = sys.argv[1]
B = "/home/users/jignesh/workspace/application/bfi_homography"
E = "/home/users/jignesh/engshare/BIDS_Format_STROKE/derivatives"
SP, MP, FP = 0.6, 0.8, 0.5
PER, COLS, Q = 16, 4, 82
T1 = nib.load(f"{E}/registrations/sub-001/ses-exvivo/anat/sub-001_ses-exvivo_T1w.nii.gz")
BFI = nib.load(f"{B}/out/hb02/whole4/BFI_in_MRI_2.nii.gz")
NIS = nib.load(f"{E}/histology-recon/sub-001/ses-histology/hist/sub-001_ses-histology_space-exvivo_desc-recon_Nissl.nii.gz")
BLK = nib.load(f"{B}/seg_lab_transport/out/exvivo/labels_clipped_space-exvivo_blockid.nii.gz")
INV = "/home/users/jignesh/engshare/stroke-atlas/invivo_regTo_exvivo_MRI/11dpm"
SEG = nib.load(f"{INV}/11dpm_dwiseg_moved_cropped.nii.gz")
DIL = nib.load(f"{INV}/11dpm_dwiseg_moved_dilated_cropped.nii.gz")
DWI = nib.load(f"{INV}/11dpm_DWI_regto_exvivo_cropped.nii.gz")

def data(im):
    a = np.asanyarray(im.dataobj).astype(np.float32)
    return a[..., -1] if a.ndim > 3 else a
def sample(im, a, lo, n, sp, order=1):
    """resample onto the grid with corner lo, n voxels of sp mm, axes (x, y, z)"""
    ax = [lo[i] + (np.arange(n[i]) + 0.5) * sp for i in range(3)]
    inv = np.linalg.inv(im.affine); out = np.zeros(n, np.float32)
    Y, Z = np.meshgrid(ax[1], ax[2], indexing="ij")
    for i, x in enumerate(ax[0]):
        w = np.stack([np.full_like(Y, x), Y, Z, np.ones_like(Y)], -1) @ inv.T
        out[i] = ndi.map_coordinates(a, [w[..., 0], w[..., 1], w[..., 2]], order=order, mode="constant")
    return out
def largest(m):
    lab, n = ndi.label(m)
    return lab == 1 + np.argmax(ndi.sum(m, lab, range(1, n + 1)))

# ---- the brain, from the block face and the T1, on a box round it ----
bfi, t1, nis, blk = data(BFI), data(T1), data(NIS), data(BLK)
wide = np.array([-6.0, -256.0, 20.0]); nn = np.ceil(np.array([190.0, 262.0, 212.0]) / MP).astype(int)
b8, t8 = sample(BFI, bfi, wide, nn, MP), sample(T1, t1, wide, nn, MP)
thr = np.exp(np.log(t1[t1 > 20]).mean())
brain = largest(ndi.binary_opening((b8 > 0.05) & (t8 > thr * 0.25), iterations=2))
idx = np.argwhere(brain); lo = wide + (idx.min(0) - 4) * MP; hi = wide + (idx.max(0) + 5) * MP
print("box", lo.round(1), hi.round(1), "mm", (hi - lo).round(1), "cc", round(brain.sum() * MP ** 3 / 1000), flush=True)
del b8, t8

# ---- shell ----
n6 = np.round((hi - lo) / SP).astype(int); ext = n6 * SP
t = sample(T1, t1, lo, n6, SP); m = sample(BFI, bfi, lo, n6, SP) > 0.05
p = t[m]; a, b = np.percentile(p, [2, 99.5])
m = largest(ndi.binary_opening(m, iterations=1))                           # the block face's outline: the T1's 0.7 mm slices
m = ndi.binary_fill_holes(ndi.binary_closing(m, iterations=2))             # would show on the surface as hatching
m = ndi.gaussian_filter(m.astype(np.float32), 2.2) > 0.5
dens = np.clip((t - a) / (b - a), 0, 1)
depth = ndi.gaussian_filter(ndi.distance_transform_edt(m) * SP, 2.0)
coarse = np.round(ndi.gaussian_filter(dens, 0.8) * 7) / 7                 # eight levels: most of this file was MRI noise
# a binary mask read trilinearly has its surface on the voxels' faces, which shows as terraces: fade the density out
# over 1.2 mm of a smooth signed distance instead, so the surface falls between voxels
sd = ndi.gaussian_filter((ndi.distance_transform_edt(m) - ndi.distance_transform_edt(~m)).astype(np.float32), 1.0) * SP
edge = np.clip(0.5 + sd / 1.2, 0, 1)
r8 = np.where(edge > 0, (0.4 + 0.6 * coarse) * edge * 254 + 1, 0).astype(np.uint8); g8 = (np.clip(depth / 3.0, 0, 1) * 255).astype(np.uint8)
vol = np.stack([r8, g8], -1).transpose(2, 1, 0, 3)[:, :, :, :]           # (x, y, z) -> (z, y, x); then swap to (w, v, u)
vol = np.ascontiguousarray(vol.transpose(1, 0, 2, 3))                     # (w = y, v = z, u = x, 2)
with gzip.open(f"{A}/volumes/hb02.bin.gz", "wb", compresslevel=9) as f: f.write(vol.tobytes())
print("hb02", vol.shape, os.path.getsize(f"{A}/volumes/hb02.bin.gz") // 1024, "KB", flush=True)

# ---- blocks and stroke ----
n8 = np.round((hi - lo) / MP).astype(int); e8 = n8 * MP
blocks = sample(BLK, blk, lo, n8, MP, order=0).astype(np.uint8)
mm = ndi.zoom(m.astype(np.float32), n8 / n6, order=1) > 0.5
ax = [lo[i] + (np.arange(n8[i]) + 0.5) * MP for i in range(3)]
X, Y, Z = np.meshgrid(*ax, indexing="ij")
order = sorted(range(1, 6), key=lambda k: np.median(X[blocks == k]))              # left to right
planes = []
for p_, q_ in zip(order, order[1:]):
    A_, C_ = blocks == p_, blocks == q_
    xa = np.where(A_.any(0), A_.shape[0] - 1 - np.argmax(A_[::-1], 0), -1); xb = np.where(C_.any(0), np.argmax(C_, 0), -1)
    ok = (xa >= 0) & (xb >= 0); yi, zi = np.nonzero(ok)
    mid = lo[0] + ((xa[ok] + xb[ok]) / 2 + 0.5) * MP
    P = np.c_[ax[1][yi], ax[2][zi], np.ones(len(yi))]
    c, *_ = np.linalg.lstsq(P, mid, rcond=None); planes.append(c)                 # x = c0 y + c1 z + c2
rest = mm & (blocks == 0); lab, k = ndi.label(rest); sizes = ndi.sum(rest, lab, range(1, k + 1))
stem = lab == 1 + np.argmax(sizes)
ids = np.zeros(n8, np.uint8)
slab = np.zeros(n8, np.uint8)
for c in planes: slab += (X > c[0] * Y + c[1] * Z + c[2]).astype(np.uint8)
ids[mm] = slab[mm] + 1; ids[stem] = 6
les = sample(SEG, data(SEG), lo, n8, MP, order=1) > 0.5
print("stroke cc", round(float(les.sum() * MP ** 3 / 1000), 1), flush=True)
AX = (2.2 / MP * 1.1, 2.0 / MP, 2.0 / MP)                                 # its slices (2.2 mm in x; thicker in life) must not show
soft = ndi.gaussian_filter(les.astype(np.float32), AX)
# how bright the lesion is on diffusion MRI, inside the lab's dilated outline, above the healthy tissue round it
dil = sample(DIL, data(DIL), lo, n8, MP, order=1) > 0.5
dw = sample(DWI, data(DWI), lo, n8, MP, order=1)
base = float(np.median(dw[mm & ~dil & (dw > 0)])); top = float(np.percentile(dw[les], 95))
glow = ndi.gaussian_filter(np.clip((dw - base) / (top - base), 0, 1) * dil, AX)
glow = np.clip(glow / np.percentile(glow[les], 90), 0, 1)
print("diffusion: healthy", round(base), "lesion p95", round(top), "glow outside the outline", round(float(glow[~les & mm].mean() / glow[les].mean()), 3), flush=True)
cereb = ndi.gaussian_filter((ids == 6).astype(np.float32), 1.0)
q32 = lambda x: np.round(np.clip(x, 0, 1) * 31) * (255 / 31)               # 32 levels: read trilinearly they do not show
meta = np.stack([ids, q32(soft), q32(cereb), q32(glow)], 0).astype(np.uint8)
meta = np.ascontiguousarray(meta.transpose(0, 2, 3, 1))                   # planar, it packs better: (c, x, y, z) -> (c, w = y, v = z, u = x)
with gzip.open(f"{A}/volumes/hb02_meta.bin.gz", "wb", compresslevel=9) as f: f.write(meta.tobytes())
print("meta", meta.shape, os.path.getsize(f"{A}/volumes/hb02_meta.bin.gz") // 1024, "KB", flush=True)

# ---- cut faces ----
nf = np.round((hi - lo) / FP).astype(int)
mf = ndi.zoom(m.astype(np.float32), nf / n6, order=1) > 0.5
def sheets(name, a):
    out = f"{A}/volumes/{name}"; os.makedirs(out, exist_ok=True)
    for f in os.listdir(out): os.remove(f"{out}/{f}")
    nu, nv, nw = nf[0], nf[2], nf[1]; rows = PER // COLS; files = []; total = 0
    for k in range(0, nu, PER):
        s = np.zeros((rows * nv, COLS * nw), np.uint8)
        for j in range(min(PER, nu - k)):
            r, c = divmod(j, COLS); s[r * nv:(r + 1) * nv, c * nw:(c + 1) * nw] = a[k + j].T     # rows = v, columns = w
        fn = f"s{k // PER:03d}.webp"; Image.fromarray(s).save(f"{out}/{fn}", quality=Q, method=6)
        total += os.path.getsize(f"{out}/{fn}"); files.append(fn)
    json.dump(dict(grid=[int(nu), int(nv), int(nw)], voxel_mm=FP, per=PER, cols=COLS, files=files), open(f"{out}/index.json", "w"), indent=1)
    print(name, len(files), "sheets", round(total / 1048576, 1), "MB", flush=True)
f = sample(T1, t1, lo, nf, FP); q = f[mf]; a, b = np.percentile(q, [2, 99.5])
sheets("hb02_mri", np.where(mf, np.clip((f - a) / (b - a), 0, 1) * 254 + 1, 0).astype(np.uint8))

# ---- index: positions in the volume's centred frame, mm ----
c0 = (lo + hi) / 2
idx = json.load(open(f"{A}/volumes/brains.json"))
idx["brains"] = [x for x in idx["brains"] if x["id"] != "hb02"]
idx["brains"].append(dict(id="hb02", data="hb02.bin.gz", grid=[int(n6[0]), int(n6[2]), int(n6[1])], voxel_mm=SP,
    extent_mm=[round(float(ext[0]), 2), round(float(ext[2]), 2), round(float(ext[1]), 2)], name="HB02", meta="hb02_meta.bin.gz",
    meta_grid=[int(n8[0]), int(n8[2]), int(n8[1])], meta_layout="planar",
    planes=[dict(y=round(float(c[0]), 4), z=round(float(c[1]), 4), x=round(float(c[0] * c0[1] + c[1] * c0[2] + c[2] - c0[0]), 3)) for c in planes]))
json.dump(idx, open(f"{A}/volumes/brains.json", "w"), indent=1)
print("planes", [round(float(c[0] * c0[1] + c[1] * c0[2] + c[2] - c0[0]), 1) for c in planes], flush=True)
