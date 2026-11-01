"""High-resolution cut face for the ganglionic eminence chapter (24 GW, DHARANI FB85 = mcmadras sub-005).

The scene clips the 0.36 mm web volume along a sagittal plane at clipx = -0.42.  That plane is slice 270 of the
mcmadras stack (B_plus4_nl3D/sub-005/align_nissl_1u/optical/histo_stacked.mnc, 64 um), which is section 1924
(stack slices run from the highest section number down; slice 532 = section 1138, the section of the Nissl zoom).
Section 1924 at 16 um is registered onto its stack slice (affine + demons, correlation 0.91), then sampled in the
web volume's (v, w) frame through the same geometry as tools/build_volumes.py (rotation, zoom, crop, flips), and
written as grey density + tissue alpha (the stack's own tissue mask, which has no glass) at 2048 rows (37 um).

usage: python media_ge_face.py SITE_DIR      (works in the current directory; needs nibabel, SimpleITK, scikit-image)
"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, nibabel as nib, SimpleITK as sitk
from scipy import ndimage as ndi
from skimage import morphology
from PIL import Image
SITE = sys.argv[1]
MR = "/home/users/jignesh/workspace/application/tirl/mcmadras_reg/madras_run"
OPT = f"{MR}/results/B_plus4_nl3D/sub-005/align_nissl_1u/optical"
SECTION = 1924
np.save("nl1924.npy", np.asarray(nib.load(f"{OPT}/nl{SECTION}.mnc").dataobj)[0].astype(np.float32))
np.save("sec1924_16.npy", np.asarray(nib.load(f"{MR}/sub-005/histo16um/sub-005_nissl_16um_{SECTION}.mnc").dataobj).squeeze().astype(np.float32))

# ---------------- geometry of the web volume (same steps as build_volumes.py) ----------------
SB = "/home/users/jignesh/.local/tmp/claude-1119/-home-users-jignesh/316a72fa-d44c-48eb-876a-e114ab9dce9a/scratchpad"
SP = 0.36
def largest(m):
    lab, n = ndi.label(m); sz = ndi.sum(m, lab, range(1, n + 1)); return lab == 1 + int(np.argmax(sz))
def rotate_plane(a, ang, flip):
    ny, nx = a.shape[1:]; R = max(ny, nx)
    yy, xx = np.mgrid[0:R, 0:R].astype(np.float32) - R / 2
    s = -1 if flip else 1; cy, cx = ny / 2, nx / 2; t = np.deg2rad(ang)
    ys = cy + (np.cos(t) * yy - np.sin(t) * xx * s); xs = cx + (np.sin(t) * yy + np.cos(t) * xx * s)
    return np.stack([ndi.map_coordinates(sl, [ys, xs], order=1, cval=0) for sl in a])
d0 = np.load(f"{SB}/site_build/dharani/sub-005_032.npy")
o = json.load(open(f"{SB}/site_build/dharani/orient.json"))["sub-005"]
d = rotate_plane(d0, o["angle"], o["flip"])
R = d.shape[1]
d = ndi.zoom(d, 0.32 / SP, order=1)
pos = d[d > 0]; lo, hi = np.percentile(pos, [2, 99.7])
mask = d > np.percentile(pos, 12)
mask = ndi.binary_opening(mask, iterations=2); mask = largest(mask)
mask = ndi.binary_closing(mask, iterations=2) & (d > 0); mask = ndi.binary_fill_holes(mask)
idx = np.argwhere(mask); clo = np.maximum(idx.min(0) - 3, 0); chi = np.minimum(idx.max(0) + 3 + 1, mask.shape)
info = dict(shape032=list(d0.shape), R=R, zoomed=list(d.shape), crop_lo=clo.tolist(), crop_hi=chi.tolist(), lo=float(lo), hi=float(hi),
            thr=float(np.percentile(pos, 12)), angle=o["angle"], flip=o["flip"])
print(info)
json.dump(info, open("geom.json", "w"), indent=1)
# where the section's stack slice lands in the web volume, as the scene's clipx (expect -0.42)
SLICE = 270
n64, n32 = 933, d0.shape[0]
u = ((SLICE + 0.5) * n32 / n64 - 0.5) * (d.shape[0] - 1) / (n32 - 1) - clo[0]
print("section", SECTION, "slice", SLICE, "clipx", round(((u + 0.5) / (chi[0] - clo[0]) - 0.5) * 2, 4))

# ---------------- register the 16 um section onto its stack slice ----------------
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(8)
fix = np.load("nl1924.npy")                      # 64 um, glass masked to 0
mov16 = np.load("sec1924_16.npy")                # 16 um, glass ~500
# moving at 64 um by area averaging; glass below the stack's tissue threshold set to 0 so both look alike
H, W = (mov16.shape[0] // 4) * 4, (mov16.shape[1] // 4) * 4
mov = mov16[:H, :W].reshape(H // 4, 4, W // 4, 4).mean((1, 3))
tissue = ndi.binary_opening(mov > 1500, iterations=2)
tissue = ndi.binary_fill_holes(ndi.binary_closing(tissue, iterations=3))
lab, n = ndi.label(tissue); tissue = lab == 1 + int(np.argmax(ndi.sum(tissue, lab, range(1, n + 1))))
movm = np.where(tissue, mov, 0).astype(np.float32)
def img(a, sp):
    im = sitk.GetImageFromArray(a.astype(np.float32)); im.SetSpacing((sp, sp)); return im
F, M = img(fix, 0.064), img(movm, 0.064)
init = sitk.CenteredTransformInitializer(F, M, sitk.AffineTransform(2), sitk.CenteredTransformInitializerFilter.MOMENTS)
reg = sitk.ImageRegistrationMethod()
reg.SetMetricAsCorrelation(); reg.SetMetricSamplingStrategy(reg.RANDOM); reg.SetMetricSamplingPercentage(0.2, seed=7)
reg.SetInterpolator(sitk.sitkLinear)
reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-5, 400, relaxationFactor=0.6)
reg.SetOptimizerScalesFromPhysicalShift()
reg.SetShrinkFactorsPerLevel([8, 4, 2, 1]); reg.SetSmoothingSigmasPerLevel([4, 2, 1, 0]); reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
reg.SetInitialTransform(init, inPlace=False)
aff = reg.Execute(F, M)
print("affine metric", reg.GetMetricValue(), reg.GetOptimizerStopConditionDescription())
Ma = sitk.Resample(M, F, aff, sitk.sitkLinear, 0.0)
# demons on top of the affine (both tissue-masked)
dem = sitk.FastSymmetricForcesDemonsRegistrationFilter()
dem.SetNumberOfIterations(120); dem.SetStandardDeviations(2.0)
fa, ma = sitk.GetArrayFromImage(F), sitk.GetArrayFromImage(Ma)
s = np.percentile(fa[fa > 0], 99); fa_n = np.clip(fa / s, 0, 1.2); ma_n = np.clip(ma / np.percentile(ma[ma > 0], 99), 0, 1.2)
field = dem.Execute(img(fa_n, 0.064), img(ma_n, 0.064))
dt = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))
comp = sitk.CompositeTransform(2); comp.AddTransform(aff); comp.AddTransform(dt)
# composite applies the last added first: point in fixed -> demons -> affine -> moving
Mf = sitk.Resample(M, F, comp, sitk.sitkLinear, 0.0)
def corr(a, b):
    m = (a > 0) & (b > 0); return float(np.corrcoef(a[m], b[m])[0, 1])
print("corr affine", round(corr(fa, ma), 4), "affine+demons", round(corr(fa, sitk.GetArrayFromImage(Mf)), 4))
disp = sitk.TransformToDisplacementField(comp, sitk.sitkVectorFloat64, F.GetSize(), F.GetOrigin(), F.GetSpacing(), F.GetDirection())
np.save("disp_mm.npy", sitk.GetArrayFromImage(disp).astype(np.float32))     # (y, x, 2) as (dx, dy) in mm
# intensity map moving -> stack units
mf = sitk.GetArrayFromImage(Mf); m = (fa > 0) & (mf > 0)
A = np.polyfit(mf[m], fa[m], 1); print("intensity fit", A); np.save("ifit.npy", A)

# ---------------- face texture ----------------
g = json.load(open("geom.json"))
FV = 2048                                   # rows along v (WebGL2 guarantees 2048 texels)
NV, NW = g["crop_hi"][1] - g["crop_lo"][1], g["crop_hi"][2] - g["crop_lo"][2]       # 210, 190
FW = round(FV * NW / NV)
v = (np.arange(FV) + 0.5) / FV * NV - 0.5
w = (np.arange(FW) + 0.5) / FW * NW - 0.5
vv, ww = np.meshgrid(v, w, indexing="ij")
# web volume (v, w) -> cropped (y', x') -> zoomed -> rotated 0.32 mm grid -> 64 um stack slice
yz = (NV - 1 - vv) + g["crop_lo"][1]; xz = (NW - 1 - ww) + g["crop_lo"][2]
R = g["R"]; f = (R - 1) / (g["zoomed"][1] - 1)
yr, xr = yz * f - R / 2, xz * f - R / 2
t = np.deg2rad(g["angle"]); ny, nx = g["shape032"][1:]
ys = ny / 2 + np.cos(t) * yr - np.sin(t) * xr
xs = nx / 2 + np.sin(t) * yr + np.cos(t) * xr
y64, x64 = 5 * ys + 2, 5 * xs + 2
# stack slice -> moving section (displacement field in mm on the 64 um grid, components (dx, dy))
disp = np.load("disp_mm.npy")
dx = ndi.map_coordinates(disp[..., 0], [y64, x64], order=1, mode="nearest")
dy = ndi.map_coordinates(disp[..., 1], [y64, x64], order=1, mode="nearest")
my, mx = (y64 * 0.064 + dy) / 0.064, (x64 * 0.064 + dx) / 0.064     # moving 64 um index
i16y, i16x = 4 * my + 1.5, 4 * mx + 1.5
sec = np.load("sec1924_16.npy")
H, W = (sec.shape[0] // 2) * 2, (sec.shape[1] // 2) * 2
s32 = sec[:H, :W].reshape(H // 2, 2, W // 2, 2).mean((1, 3))           # 32 um, near the 37 um texel
qy, qx = (i16y - 0.5) / 2, (i16x - 0.5) / 2
val = ndi.map_coordinates(s32, [qy, qx], order=3, mode="constant", cval=0)
# tissue: the stack's own tissue mask (no glass) sampled in the same frame, minus the ventricle lumen (dark, large)
fix = np.load("nl1924.npy")
inside = ndi.map_coordinates(ndi.gaussian_filter((fix > 0).astype(np.float32), 0.7), [y64, x64], order=1, cval=0) > 0.5
sm = ndi.gaussian_filter(val, 1.5)
lumen = ndi.binary_opening(inside & (sm < float(os.environ.get("LUMEN", 1500))), iterations=3)
lab, n = ndi.label(lumen)
if n:
    sz = ndi.sum(lumen, lab, range(1, n + 1)); lumen = np.isin(lab, 1 + np.flatnonzero(sz * 0.037 ** 2 > 0.5))
m = inside & ~lumen
lab, n = ndi.label(m); sz = ndi.sum(m, lab, range(1, n + 1)); m = lab == 1 + int(np.argmax(sz))
m = morphology.remove_small_holes(m, max_size=int(0.1 / 0.037 ** 2))
print("lumen mm2", round(float(lumen.sum() * 0.037 ** 2), 1))
d = ndi.distance_transform_edt(m) - ndi.distance_transform_edt(~m)
alpha = np.clip(0.5 + ndi.gaussian_filter(d.astype(np.float32), 1.0), 0, 1)
A = np.load("ifit.npy")
stack_units = A[0] * val + A[1]
hi = max(g["hi"], float(np.percentile(stack_units[m], 99.7)))
dens = np.clip((stack_units - g["lo"]) / (hi - g["lo"]), 0, 1) ** 0.85
# spread tissue values into the transparent parts so texture filtering never pulls in black
idx = ndi.distance_transform_edt(~m, return_distances=False, return_indices=True)
dens = np.where(m, dens, dens[idx[0], idx[1]])
d8 = (dens * 255).round().astype(np.uint8)
out = f"{SITE}/assets/img/ge/face_24gw.webp"
Image.fromarray(np.dstack([d8, d8, d8, (alpha * 255).round().astype(np.uint8)]), "RGBA").save(out, quality=82, alpha_quality=70, method=6)
print("face", FW, "x", FV, os.path.getsize(out) // 1024, "KB", "hi", round(hi), "tissue", round(float(m.mean()), 3))
