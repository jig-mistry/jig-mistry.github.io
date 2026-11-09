"""Where the one Nissl section of the 24-week brain (SE 1138, the zoom's section) lies in the web volume, for stage.js GEOM.

The stage clips the volume at the eminence's face (clipx, section 1924), then carries the plane medially to this section
and lays the Nissl image over the volume's own cut. Section 1138 is stack slice 532, about 17 mm medial of 1924; this
finds the sagittal plane of the 0.36 mm web volume that matches the image best and the rigid map of the image onto it
(scale is left at one: a similarity fit returns 1.00). The page's plane frame is (z, -y) in mm, centred; the Nissl
image's is mm from its top-left. Prints GEOM.nissx and GEOM.align.

usage: python fit_nissl_plane.py SITE_DIR
"""
import sys, gzip, json
import numpy as np, SimpleITK as sitk
from PIL import Image
from scipy import ndimage as ndi
SITE = sys.argv[1]
NW = 72.2                                  # GEOM.nisslWidth: the section image's width, mm
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(8)

b = [x for x in json.load(open(f"{SITE}/assets/volumes/brains.json"))["brains"] if x["id"] == "dharani_24gw"][0]
gu, gv, gw = b["grid"]; sp = b["voxel_mm"]; hx, hy, hz = gu * sp / 2, gv * sp / 2, gw * sp / 2
raw = np.frombuffer(gzip.open(f"{SITE}/assets/volumes/{b['data']}").read(), np.uint8)
D = raw[0::2].reshape(gw, gv, gu).astype(np.float32) / 255             # density [z, y, x]

im = Image.open(f"{SITE}/assets/img/ge/nissl_section.webp")
a = np.asarray(im.convert("RGBA"), np.float32) / 255
dens = (1 - a[..., 0]) * a[..., 3]                                     # stain on light glass; alpha clears the glass
px = NW / im.size[0]; k = round(sp / px)
H, W = (dens.shape[0] // k) * k, (dens.shape[1] // k) * k
mov = sitk.GetImageFromArray(dens[:H, :W].reshape(H // k, k, W // k, k).mean((1, 3)).astype(np.float32))
mov.SetSpacing((px * k,) * 2); mov.SetOrigin((px * k / 2,) * 2)

def plane(xmm):
    """the volume's sagittal plane at x, in the page's plane frame: columns z, rows -y"""
    i = (xmm + hx) / sp - 0.5
    zz, yy = np.meshgrid(np.arange(gw), np.arange(gv), indexing="ij")
    P = ndi.map_coordinates(D, [zz, yy, np.full(zz.shape, i)], order=1).T[::-1]
    f = sitk.GetImageFromArray(P.astype(np.float32)); f.SetSpacing((sp, sp)); f.SetOrigin((-hz + sp / 2, -hy + sp / 2))
    return f

def register(fix, angle):
    fm, mm = sitk.Cast(fix > 0.02, sitk.sitkFloat32), sitk.Cast(mov > 0.05, sitk.sitkFloat32)
    init = sitk.CenteredTransformInitializer(fm, mm, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    init.SetAngle(angle)
    r = sitk.ImageRegistrationMethod(); r.SetMetricAsCorrelation(); r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 300); r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([4, 2, 1]); r.SetSmoothingSigmasPerLevel([2, 1, 0]); r.SetInitialTransform(init, inPlace=False)
    t = r.Execute(fix, mov)
    w, f = sitk.GetArrayFromImage(sitk.Resample(mov, fix, t, sitk.sitkLinear, 0.0)), sitk.GetArrayFromImage(fix)
    m = (w > 0.02) | (f > 0.02)
    return float(np.corrcoef(w[m], f[m])[0, 1]), sitk.Euler2DTransform(t.GetNthTransform(0) if hasattr(t, "GetNthTransform") else t)

best = None
for x in np.arange(-4.0, 12.01, 0.18):                                 # half a voxel apart across the middle of the brain
    fix = plane(x)
    for deg in (0, 20, 45, 67, 90):
        try: c, t = register(fix, float(np.radians(-deg)))
        except RuntimeError: continue
        if best is None or c > best[0]: best = (c, t, float(x))
c, t, x = best
# the page's form: plane point p = a0 + rot(theta)(n - n0), n in the image; n0 its centre
n0 = (NW / 2, NW * im.size[1] / im.size[0] / 2); a0 = t.GetInverse().TransformPoint(n0); theta = -t.GetAngle()
print(f"plane x {x:.2f} mm, correlation {c:.3f}")
print(f"nissx: {x / hx:.4f},")
print(f"align: {{ n0: [{n0[0]:.2f}, {n0[1]:.3f}], a0: [{a0[0]:.3f}, {a0[1]:.3f}], theta: {np.degrees(theta):.2f} * Math.PI / 180 }},")
