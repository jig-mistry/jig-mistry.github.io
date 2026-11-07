"""The block-face photo of the same section, fitted to the Nissl face.

The photo is of the block just as this section came off it, so it is the same plane: both are taken upright, the tissue
from colour (the embedding medium is blue), a similarity fit on the outlines, then demons on signed distance maps to
take out the section's stretch. Grey matter is dark in the photo and bright in the inverted Nissl, so a good fit has a
strongly negative correlation. Run after build_hb02_faces.py, in the same OUT_DIR; it leaves the photo, in colour on the
face, as bfi.npz, and fit_hb02_stains.py writes the images.

usage: python fit_hb02_blockface.py SITE_ASSETS OUT_DIR
"""
import os, sys, json, gzip
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, SimpleITK as sitk
from scipy import ndimage as ndi
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(8)
A, OUT = sys.argv[1], sys.argv[2]
P = "/home/users/jignesh/analytics/313/BFIW/B_313-ST_BFIW-SE_1516.jpg"
d = np.load(f'{OUT}/face.npz'); nissl = d['nissl'].astype(np.float32)[::-1]; tissue = d['tissue'][::-1]     # upright: superior first
im = Image.open(P).convert('RGB'); K, F = 8, 8
sm = np.asarray(im.resize((im.width // K, im.height // K), Image.BOX), np.float32) / 255
rb = sm[..., 0] - sm[..., 2]
from skimage.filters import threshold_otsu
print('red minus blue pct', np.percentile(rb, [1, 25, 50, 75, 99]).round(3), 'otsu', round(float(threshold_otsu(rb)), 3), flush=True)
m = rb > threshold_otsu(rb)
m = ndi.binary_opening(m, iterations=2); lab, n = ndi.label(m); sz = ndi.sum(m, lab, range(1, n + 1)); m = lab == 1 + np.argmax(sz); m = ndi.binary_fill_holes(ndi.binary_closing(m, iterations=3))
print('photo tissue', round(float(m.mean()), 3), 'area mm2', round(float(m.sum() * (0.053 * K) ** 2)), flush=True)
fg = ndi.gaussian_filter(nissl, 4)[::F, ::F]
fm = (fg > 12) & tissue[::F, ::F]
fm = ndi.binary_fill_holes(ndi.binary_opening(fm, iterations=1)); lab, n = ndi.label(fm); fm = lab == 1 + np.argmax(ndi.sum(fm, lab, range(1, n + 1)))
print('section area mm2', round(float(fm.sum() * (0.06 * F) ** 2)), flush=True)
def sdm(a): return ndi.distance_transform_edt(a).astype(np.float32) - ndi.distance_transform_edt(~a).astype(np.float32)
fixed = sitk.GetImageFromArray(sdm(fm)); fixed.SetSpacing((0.06 * F, 0.06 * F))
mv = sitk.GetImageFromArray(sdm(m)); s = 0.053 * K; mv.SetSpacing((s, s))
grey = sm @ np.array([0.3, 0.59, 0.11], np.float32); gi = sitk.GetImageFromArray(grey); gi.SetSpacing((s, s))
best = None
for a0 in np.radians([-20, -10, 0, 10, 20]):
    init = sitk.Similarity2DTransform(sitk.CenteredTransformInitializer(fixed, mv, sitk.Similarity2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)); init.SetAngle(float(a0))
    r = sitk.ImageRegistrationMethod(); r.SetMetricAsCorrelation(); r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-3, 200); r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([4, 2, 1]); r.SetSmoothingSigmasPerLevel([2, 1, 0]); r.SetInitialTransform(init, inPlace=False)
    t = r.Execute(fixed, mv)
    w = sitk.GetArrayFromImage(sitk.Resample(mv, fixed, t, sitk.sitkLinear, -50.0)) > 0
    wg = sitk.GetArrayFromImage(sitk.Resample(gi, fixed, t, sitk.sitkLinear, 0.0)); both = w & fm
    c = float(np.corrcoef(ndi.gaussian_filter(wg, 2)[both], ndi.gaussian_filter(fg, 2)[both])[0, 1]); dice = float(2 * both.sum() / (w.sum() + fm.sum()))
    print('start', round(float(np.degrees(a0))), 'dice', round(dice, 3), 'anatomy corr', round(c, 3), flush=True)
    if best is None or dice > best[0]: best = (dice, t, c)
dice, t, c = best; print('similarity dice', round(dice, 3), 'corr', round(c, 3), flush=True)
moved = sitk.Resample(mv, fixed, t, sitk.sitkLinear, -50.0)
dem = sitk.DemonsRegistrationFilter(); dem.SetNumberOfIterations(300); dem.SetStandardDeviations(6.0); dem.SetSmoothDisplacementField(True)
field = dem.Execute(fixed, moved)
jac = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(field)); print('jacobian', round(float(jac.min()), 3), round(float(jac.max()), 3), flush=True)
disp = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64)); comp = sitk.CompositeTransform([t, disp])
w = sitk.GetArrayFromImage(sitk.Resample(mv, fixed, comp, sitk.sitkLinear, -50.0)) > 0
wg = sitk.GetArrayFromImage(sitk.Resample(gi, fixed, comp, sitk.sitkLinear, 0.0)); both = w & fm
print('demons dice', round(float(2 * both.sum() / (w.sum() + fm.sum())), 4), 'anatomy corr', round(float(np.corrcoef(ndi.gaussian_filter(wg, 2)[both], ndi.gaussian_filter(fg, 2)[both])[0, 1]), 3), flush=True)
rgb = np.asarray(im, np.float32) / 255
photo = sitk.GetImageFromArray(rgb, isVector=True); photo.SetSpacing((0.053, 0.053)); off = -(K - 1) / 2 * 0.053; photo.SetOrigin((off, off))
ref = sitk.Image(nissl.shape[1], nissl.shape[0], sitk.sitkFloat32); ref.SetSpacing((0.06, 0.06))
out = sitk.GetArrayFromImage(sitk.Resample(photo, ref, comp, sitk.sitkBSpline, 0.0))
mfull = sitk.GetImageFromArray(m.astype(np.float32)); mfull.SetSpacing((s, s))
tm = sitk.GetArrayFromImage(sitk.Resample(mfull, ref, comp, sitk.sitkLinear, 0.0)) > 0.5
np.savez_compressed(f"{OUT}/bfi.npz", rgb=np.clip(out, 0, 1).astype(np.float16), seen=tm)     # upright, as the face
print("photo on the face", out.shape, "seen", round(float(tm.mean()), 3), flush=True)
