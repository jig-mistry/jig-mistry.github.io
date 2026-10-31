"""Tissue-only zoom levels (transparent glass) for the two section zooms of the ganglionic eminence chapter.

Nissl, 24 GW (DHARANI FB85 = mcmadras sub-005, SE 1138): the tissue mask is the mcmadras nnU-Net tissue model
(Dataset510, tight masks) predicted on the 16 um MINC of this section and mapped onto the JP2 frame (MINC = JP2
turned 90 deg and mirrored; correlation 0.92).  Pseudo-fluorescence: glass is black, so the mask is the blurred
signal at 16 um.

For both: keep the brain and tissue pieces near it, fill every hole except the ventricles (holes > HOLE_MM2), then
draw the alpha from a smoothed signed distance to that outline, resampled at each zoom level, so the edge is a
smooth anti-aliased curve at every scale instead of a jagged threshold.

usage: python media_alpha.py SITE_DIR NNUNET_MASK.npy

NNUNET_MASK.npy (bool, JP2 frame at 16 um) was made with the mcmadras tissue model:
  mask_nnunet/scripts/mnc_nii.py to_nii IN s005_ madras_run/sub-005/histo16um/sub-005_nissl_16um_1138.mnc
  nnUNetv2_predict -d 510 -c 2d -f 0 -tr nnUNetTrainer_100epochs -p nnUNetPlans -chk checkpoint_final.pth -i IN -o OUT
  then the predicted (x, y) NIfTI transposed to (y, x), turned np.rot90(k=1) and flipped up-down onto the JP2 r5 frame.
"""
import os, sys, json, subprocess, tempfile
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import tifffile, zarr
from PIL import Image
from scipy import ndimage as ndi
from skimage import morphology
Image.MAX_IMAGE_PIXELS = None
SITE, NISSL_MASK = sys.argv[1], sys.argv[2]
OPJ = "/home/users/jignesh/miniconda3/envs/nisslprop/bin/opj_decompress"
TMP = tempfile.mkdtemp()
PX16 = 16.0                    # um per pixel of the coarse masks
HOLE_MM2 = 1.5                 # holes larger than this stay open (ventricles)
NEAR_MM = 1.5                  # pieces farther than this from the brain are dropped (glass specks, labels)

def clean(m, hole_mm2=HOLE_MM2, open_r=1):
    """brain + nearby pieces, holes filled except the large ones"""
    px_mm2 = (PX16 / 1000) ** 2
    m = morphology.remove_small_objects(m, max_size=int(0.3 / px_mm2))
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        brain = lab == 1 + int(np.argmax(sizes))
        near = ndi.distance_transform_edt(~brain) * PX16 / 1000 < NEAR_MM
        keep = np.unique(lab[near & m]); m = np.isin(lab, keep[keep > 0])
    # cut thin strands (loose fibres, threads) before filling holes, since a strand that loops back to the brain
    # would enclose a "hole" and be filled solid; the outline within 3 px of the opened shape is given back
    m = m & ndi.binary_dilation(morphology.opening(m, morphology.disk(open_r)), morphology.disk(3)) if open_r > 1 \
        else morphology.opening(m, morphology.disk(open_r))
    m = morphology.remove_small_objects(m, max_size=int(0.3 / px_mm2))
    return morphology.remove_small_holes(m, max_size=int(hole_mm2 / px_mm2))

def sdf(m):
    """signed distance in um (positive inside), smoothed so the 16 um staircase disappears"""
    d = ndi.distance_transform_edt(m) - ndi.distance_transform_edt(~m)
    return ndi.gaussian_filter(d.astype(np.float32), 1.2) * PX16

def alpha_for(sd, box16, size_wh, um_px):
    """alpha for an output image of size_wh covering box16 (x0, y0, x1, y1 in coarse-mask pixels)"""
    w, h = size_wh
    x0, y0, x1, y1 = box16
    xs = x0 + (np.arange(w) + 0.5) * (x1 - x0) / w - 0.5
    ys = y0 + (np.arange(h) + 0.5) * (y1 - y0) / h - 0.5
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    d = ndi.map_coordinates(sd, [yy, xx], order=3, mode="nearest")
    edge = max(1.0, 1.1 * um_px)                     # ~1 output pixel of anti-aliasing
    return np.clip(0.5 + d / edge, 0, 1)

def save_rgba(rgb, a, path, q=76):
    im = Image.fromarray(np.dstack([rgb, (a * 255).round()]).astype(np.uint8), "RGBA")
    im.save(path, quality=q, alpha_quality=80, method=6)
    print(os.path.basename(path), im.size, os.path.getsize(path) // 1024, "KB", flush=True)
    return im

# ---------------- Nissl, FB85 SE 1138 ----------------
J85 = "/home/users/jignesh/analytics/244/NISL/B_244_FB85-SL_380-ST_NISL-SE_1138_compressed.jp2"
W85, H85 = 144472, 129351
def jp2(r, box):
    out = os.path.join(TMP, "c.png")
    subprocess.run([OPJ, "-i", J85, "-o", out, "-r", str(r), "-d", ",".join(map(str, box)), "-threads", "8"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return Image.open(out).convert("RGB")
m = clean(np.load(NISSL_MASK))                       # JP2 frame at r5 = 16 um
sd = sdf(m)
cx, cy = 36100, 54300
levels = []
for name, r, box, size in [("nissl_section", 5, (0, 0, W85, H85), 2400),
                           ("nissl_ge", 3, (cx - 10000, cy - 10000, cx + 10000, cy + 10000), 1700),
                           ("nissl_ge_mid", 1, (cx - 3000, cy - 3000, cx + 3000, cy + 3000), 1600),
                           ("nissl_cells", 0, (cx - 800, cy - 800, cx + 800, cy + 800), 1600)]:
    im = jp2(r, box); im.thumbnail((size, size), Image.Resampling.LANCZOS)
    um_px = (box[2] - box[0]) * 0.5 / im.width
    a = alpha_for(sd, [v / 32 for v in box], im.size, um_px)
    save_rgba(np.asarray(im), a, f"{SITE}/assets/img/ge/{name}.webp")
    levels.append(dict(src=f"assets/img/ge/{name}.webp", rect=[box[0] / W85, box[1] / H85, box[2] / W85, box[3] / H85],
                       px=list(im.size)))
json.dump(dict(aspect=W85 / H85, levels=levels), open(f"{SITE}/assets/img/ge/nissl_zoom.json", "w"), indent=1)

# ---------------- pseudo-fluorescence (stored frame; turned 90 deg clockwise for display) ----------------
F = "/home/users/jignesh/neuroglancer/ihc_pair_registered/pseudofluor_DCXgreen_S100bred_0p5um.tif"
t = tifffile.TiffFile(F)
H0, W0 = t.pages[0].shape[:2]
def region(level, x0, y0, x1, y1):
    z = zarr.open(t.aszarr(level=level), mode="r"); f = 2 ** level
    return np.asarray(z[y0 // f:y1 // f, x0 // f:x1 // f])
coarse = region(5, 0, 0, W0, H0).astype(np.float32)          # 16 um
sig = ndi.gaussian_filter(coarse.max(-1), 2.0)
# dim bands of the cortex sit just above the black glass, and the ventricle here is ~36 mm2: a lower threshold,
# holes up to 10 mm2 filled, and a wider opening that removes the loose fibre trailing off the section
thr = float(os.environ.get("PF_THR", 8))
mp = clean(sig > thr, hole_mm2=10.0, open_r=10)
sdp = sdf(mp)
print("pseudofluor tissue fraction", round(mp.mean(), 3), "threshold", thr)
levels = []
gain = {"pf_section": 1.55, "pf_ge": 1.25}
for name, lvl, box, size in [("pf_section", 5, (0, 0, W0, H0), 2400),
                             ("pf_ge", 3, (32000, 34000, 56000, 58000), 1800),
                             ("pf_ge_mid", 1, (40416, 35712, 48416, 43712), 1800),
                             ("pf_cells", 0, (43416, 38712, 45416, 40712), 2000)]:
    im = Image.fromarray(region(lvl, *box)); im.thumbnail((size, size), Image.Resampling.LANCZOS)
    rgb = np.clip(np.asarray(im).astype(np.float32) * gain.get(name, 1.0), 0, 255)
    um_px = (box[2] - box[0]) * 0.5 / im.width
    a = alpha_for(sdp, [v / 32 for v in box], im.size, um_px)
    out = save_rgba(rgb, a, f"{SITE}/assets/img/ge/{name}.webp")
    out.transpose(Image.Transpose.ROTATE_270).save(f"{SITE}/assets/img/ge/{name}.webp", quality=76, alpha_quality=80, method=6)
    # display frame (clockwise 90): x' = H - y, y' = x
    levels.append(dict(src=f"assets/img/ge/{name}.webp",
                       rect=[(H0 - box[3]) / H0, box[0] / W0, (H0 - box[1]) / H0, box[2] / W0], px=[im.height, im.width]))
json.dump(dict(aspect=H0 / W0, levels=levels), open(f"{SITE}/assets/img/ge/pf_zoom.json", "w"), indent=1)
