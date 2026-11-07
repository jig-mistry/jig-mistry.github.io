"""The face the stroke stop looks straight at: one real Nissl section, with MRI and block face on the same surface.

After the blocks lift, the page shows a single face: the left side of block 3 (biosample 313), where the sagittal cut
crosses the stroke at its widest. So this is three flat images rather than volumes. The Nissl is one whole section at
its 60 um canvas resolution, read through block 3's transform from ex vivo space (seg_lab_transport/common.py), with no
blending between sections. A section is not exactly flat in ex vivo space, so each point of the face is moved along x
to where that section really lies, and the MRI is read there. The block-face photo of the same section is then fitted
to it by fit_hb02_blockface.py, which also writes the images.

Images are indexed by the page's own frame: columns w (posterior first), rows v (inferior first), over the whole box.

usage: python build_hb02_faces.py SITE_ASSETS OUT_DIR
"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
sys.path.insert(0, "/home/users/jignesh/workspace/application/bfi_homography/seg_lab_transport")
import numpy as np, nibabel as nib, SimpleITK as sitk, gzip
from scipy import ndimage as ndi
from PIL import Image
import common as C
Image.MAX_IMAGE_PIXELS = None
A, OUT = sys.argv[1], sys.argv[2]
LO, HI = np.array([3.6, -203.2, 90.4]), np.array([160.4, -23.2, 187.2])     # the box of build_hb02.py, RAS mm
PX = 0.06                                                                     # mm per pixel: the canvases
INWARD = 2.5                                                                  # mm into the block, past its ragged edge
BLOCK, PLANE = 3, 1
meta = [b for b in json.load(open(f"{A}/volumes/brains.json"))["brains"] if b["id"] == "hb02"][0]
c0 = (LO + HI) / 2; p = meta["planes"][PLANE]
W, H = int(round((HI[1] - LO[1]) / PX)), int(round((HI[2] - LO[2]) / PX))
ys = LO[1] + (np.arange(W) + 0.5) * PX; zs = LO[2] + (np.arange(H) + 0.5) * PX
Yc, Zc = np.meshgrid(ys[::24], zs[::24]); hs, ws = Yc.shape                  # the transform is smooth: evaluate it coarsely
Xc = c0[0] + p["y"] * (Yc - c0[1]) + p["z"] * (Zc - c0[2]) + p["x"] + INWARD

hist = sitk.ReadImage(C.histology_path(BLOCK), sitk.sitkFloat32)
T = sitk.ReadTransform(C.composite_transform_path(BLOCK))
def to_block(X, Y, Z):
    out = np.zeros(X.shape + (3,))
    for idx in np.ndindex(X.shape):
        q = T.TransformPoint((float(-X[idx]), float(-Y[idx]), float(Z[idx])))    # RAS -> LPS
        out[idx] = hist.TransformPhysicalPointToContinuousIndex(q)
    return out
ijk = to_block(Xc, Yc, Zc); print("slice range on the face", np.percentile(ijk[..., 2], [1, 50, 99]).round(1), flush=True)
dk = (to_block(Xc + 1.0, Yc, Zc)[..., 2] - ijk[..., 2]); dk = float(np.median(dk)); print("slices per mm along x", round(dk, 3), flush=True)

# ---- which section ----
Hv = sitk.GetArrayFromImage(hist); files = C.canvas_files(C.BLOCKS[BLOCK])
small = np.stack([np.asarray(Image.open(f).convert("L").resize((400, 400), Image.BOX), np.float32) for f in files]).transpose(0, 2, 1)
def z(a):
    a = a.reshape(a.shape[0], -1); a = a - a.mean(1, keepdims=True); return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
S = z(Hv) @ z(small).T; match, score = S.argmax(1), S.max(1)
k_mid = int(round(float(np.median(ijk[..., 2]))))
near = [k for k in range(max(0, k_mid - 5), min(Hv.shape[0], k_mid + 6)) if score[k] > 0.8]
tis = [float((Hv[k] > 20).mean()) for k in near]; k0 = near[int(np.argmax(np.array(tis) - 0.004 * np.abs(np.array(near) - k_mid)))]
print("slice", k0, "of", Hv.shape[0], "tissue", round(float((Hv[k0] > 20).mean()), 3), "corr", round(float(score[k0]), 3), files[match[k0]].split("/")[-1], flush=True)

# ---- the surface of that section, and the section itself ----
Xs = Xc.copy()
for _ in range(3):
    b = to_block(Xs, Yc, Zc); Xs = Xs + (k0 - b[..., 2]) / dk
b = to_block(Xs, Yc, Zc); print("left over, slices", round(float(np.abs(b[..., 2] - k0).max()), 3), "surface x spread mm", round(float(Xs.max() - Xs.min()), 1), flush=True)
def fine(a): return ndi.zoom(a, (H / a.shape[0], W / a.shape[1]), order=1)
I, J, X = fine(b[..., 0]), fine(b[..., 1]), fine(Xs)
canvas = np.asarray(Image.open(files[match[k0]]).convert("L"), np.float32)     # 3200 px; block pixel i = canvas row 8 i + 3.5
nissl = ndi.map_coordinates(canvas, [I * 8 + 3.5, J * 8 + 3.5], order=3, mode="constant")
Y, Z = np.meshgrid(ys, zs)

def read(path):
    im = nib.load(path); a = np.asanyarray(im.dataobj).astype(np.float32)
    w = np.stack([X, Y, Z, np.ones_like(X)], -1) @ np.linalg.inv(im.affine).T
    return ndi.map_coordinates(a, [w[..., 0], w[..., 1], w[..., 2]], order=3, mode="constant")
mri = read(C.EXVIVO_T1)
# the in vivo diffusion scan 11 days before death and the lab's segmentation of its lesion, already in ex vivo space
INV = "/home/users/jignesh/engshare/stroke-atlas/invivo_regTo_exvivo_MRI/11dpm"
dwi = read(f"{INV}/11dpm_DWI_regto_exvivo_cropped.nii.gz")
seg = read(f"{INV}/11dpm_dwiseg_moved_cropped.nii.gz")
# ---- tissue, from the shell ----
g = meta["grid"]; shell = np.frombuffer(gzip.open(f"{A}/volumes/hb02.bin.gz").read(), np.uint8).reshape(g[2], g[1], g[0], 2)[..., 0]
sp = meta["voxel_mm"]
coords = [(Y - LO[1]) / sp - 0.5, (Z - LO[2]) / sp - 0.5, (X - LO[0]) / sp - 0.5]
tissue = ndi.map_coordinates((shell > 0).astype(np.float32), coords, order=1) > 0.5
np.savez_compressed(f"{OUT}/face.npz", nissl=nissl.astype(np.float16), mri=mri.astype(np.float32), tissue=tissue,
                    dwi=dwi.astype(np.float32), seg=seg.astype(np.float16), x=X.astype(np.float32))
print("done", W, H, "tissue", round(float(tissue.mean()), 3), flush=True)
