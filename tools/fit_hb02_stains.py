"""The neighbouring sections in other stains on the stroke stop's face, the lab's outline of the injury, and the images.

The Nissl face (build_hb02_faces.py) is one section of block 313, SE 1516. Its neighbours were stained otherwise: H&E
two sections away and two immunostains, 25 and 41 sections (0.5 and 0.8 mm) away. On the Nissl section 27 away the lab
outlined neuronal loss by hand. An immunostained section with little stain is nearly the colour of the glass, so each
slide's outline is taken from the lab's canvas of it (TRS: inverted, contrast stretched, tissue clear), which is fitted
onto the face as the block-face photo is (fit_hb02_blockface.py): a similarity from a few start angles, then demons on
signed distance maps, the pose checked on the anatomy. Two canvases still hold their slide's glass, and on one a smear
along the top of the section is as bright as tissue; the lab's canvases are aligned to one another, so those two take
the fit of the H&E canvas, whose outline is clean. The slide's own colours then come through the lab's placement of
the slide in its canvas (trs csv). Optical density is stretched to one range per stain.

Then every image of the face is written, in the page's frame (columns w, posterior first; rows v, inferior first):
face_mri (grey + tissue alpha), face_dwi, face_bfi, face_nissl, face_he, face_ihc1, face_ihc2, each also as *_s at half
size for phones, and face_lines_s (lossless, half size: red the diffusion lesion, green the lab's outline of neuronal loss,
as signed distance so the page can draw them at any size). Run after build_hb02_faces.py and fit_hb02_blockface.py, in the same OUT_DIR.

usage: python fit_hb02_stains.py SITE_ASSETS OUT_DIR
"""
import os, sys, glob, json, csv
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, SimpleITK as sitk
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu
Image.MAX_IMAGE_PIXELS = None
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(8)
A, OUT = sys.argv[1], sys.argv[2]
AN = "/home/users/jignesh/analytics/313"
UM16 = 0.016                       # mm per pixel of the slides' _downsampled.tif (full scan / 32)
CV, F = 0.06, 8                    # the lab's canvases (TRS) at 60 um, where each slide's tissue is clear; fitted at 0.48 mm
SE = {"he": 1514, "ihc1": 1541, "ihc2": 1475, "outlined": 1543}
ON_GLASS = {"ihc1": "he", "outlined": "he"}     # canvases that keep their glass take this one's fit
TRS = "/home/users/jignesh/engshare/stroke-atlas/TRS/313/All_files_renamed"
PLACE = "/home/users/jignesh/engshare/stroke-data/trs/313_trs_new_proc_all.csv"
OUTLINE = "/home/users/jignesh/engshare/stroke-data/manual_annotations/313/313_Nissl/313_Nissl_1543.json"

d = np.load(f"{OUT}/face.npz")
nissl = d["nissl"].astype(np.float32)[::-1]; tissue = d["tissue"][::-1]           # upright: superior first
fg = ndi.gaussian_filter(nissl, 4)[::F, ::F]
fm = (fg > 12) & tissue[::F, ::F]
fm = ndi.binary_fill_holes(ndi.binary_opening(fm, iterations=1)); lab, n = ndi.label(fm); fm = lab == 1 + np.argmax(ndi.sum(fm, lab, range(1, n + 1)))
def sdm(a): return ndi.distance_transform_edt(a).astype(np.float32) - ndi.distance_transform_edt(~a).astype(np.float32)
fixed = sitk.GetImageFromArray(sdm(fm)); fixed.SetSpacing((0.06 * F, 0.06 * F))
fixed_m = sitk.GetImageFromArray(fm.astype(np.float32)); fixed_m.SetSpacing((0.06 * F, 0.06 * F))
ref = sitk.Image(nissl.shape[1], nissl.shape[0], sitk.sitkFloat32); ref.SetSpacing((0.06, 0.06))

def density(rgb):                  # optical density summed over channels: the amount of stain, bright where stained
    return -np.log(np.clip(rgb, 1 / 255, 1)).sum(-1)
def clean(m):
    m = ndi.binary_opening(m, iterations=2); lab, n = ndi.label(m); sz = ndi.sum(m, lab, range(1, n + 1))
    m = np.isin(lab, 1 + np.nonzero(sz > 0.05 * sz.max())[0])
    return ndi.binary_fill_holes(ndi.binary_closing(m, iterations=3))
def section_mask(g):
    """tissue in a canvas (inverted grey, tissue bright on black). Where the slide's glass is still in the canvas the
    first mask is that rectangle, and the tissue is what is brighter than its rim"""
    m = clean(g > threshold_otsu(g)); r, c = np.nonzero(m)
    if m.sum() / ((np.ptp(r) + 1) * (np.ptp(c) + 1)) > 0.85:
        rim = m & ~ndi.binary_erosion(m, iterations=6) & ndi.binary_erosion(m, iterations=2)
        m = clean(m & (g > np.percentile(g[rim], 25) + 6))
    return m
def canvas_mask(cv):
    sm = np.asarray(Image.fromarray(cv.astype(np.uint8)).resize((cv.shape[1] // F, cv.shape[0] // F), Image.BOX), np.float32)
    od = ndi.gaussian_filter(sm, 1); return od, section_mask(od)

rows = {int(r["sectionno"]): r for r in csv.DictReader(open(PLACE))}
def placement(se, sign=1):
    """the lab's placement of slide se in its canvas, as a map from canvas mm to slide mm. A slide's pixels are scaled
    to 60 um (in mm a scale of exactly one), turned about their top-left corner and that corner moved to translation"""
    r = rows[se]; th = sign * float(r["rotation"]); tr = np.array([float(x) for x in r["translation"].strip("[] ").split()]) * CV
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    a = sitk.AffineTransform(2); a.SetMatrix(R.T.ravel().tolist()); a.SetTranslation((-R.T @ tr).tolist())
    return a

def fit(name, od, m):
    """canvas grey and tissue (at 0.48 mm) -> transform from the face's upright frame into the canvas, mm"""
    print(name, "section mm2", round(float(m.sum() * (CV * F) ** 2)), "face mm2", round(float(fm.sum() * (0.06 * F) ** 2)), flush=True)
    best = None
    for flip in (False,):              # the canvases keep the slides' handedness, as the Nissl ones the face is made of
        mm_, oo = (m[:, ::-1], od[:, ::-1]) if flip else (m, od)
        mv = sitk.GetImageFromArray(sdm(mm_)); mv.SetSpacing((CV * F,) * 2)
        mv_m = sitk.GetImageFromArray(mm_.astype(np.float32)); mv_m.SetSpacing((CV * F,) * 2)
        gi = sitk.GetImageFromArray(oo.astype(np.float32)); gi.SetSpacing((CV * F,) * 2)
        # the canvases are upright, and an outline alone fits as well turned over: a barrier-leak stain landed so
        for a0 in np.radians([-20, -10, 0, 10, 20]):
            # centres of mass of the masks: a distance map is negative outside, which is no mass at all
            init = sitk.Similarity2DTransform(sitk.CenteredTransformInitializer(fixed_m, mv_m, sitk.Similarity2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)); init.SetAngle(float(a0))
            r = sitk.ImageRegistrationMethod(); r.SetMetricAsCorrelation(); r.SetInterpolator(sitk.sitkLinear)
            r.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-3, 200); r.SetOptimizerScalesFromPhysicalShift()
            r.SetShrinkFactorsPerLevel([4, 2, 1]); r.SetSmoothingSigmasPerLevel([2, 1, 0]); r.SetInitialTransform(init, inPlace=False)
            t = r.Execute(fixed, mv)
            w = sitk.GetArrayFromImage(sitk.Resample(mv, fixed, t, sitk.sitkLinear, -50.0)) > 0
            wg = sitk.GetArrayFromImage(sitk.Resample(gi, fixed, t, sitk.sitkLinear, 0.0)); both = w & fm
            if both.sum() < 100: continue                                    # the optimiser left the image
            c = float(np.corrcoef(ndi.gaussian_filter(wg, 2)[both], ndi.gaussian_filter(fg, 2)[both])[0, 1]); dice = float(2 * both.sum() / (w.sum() + fm.sum()))
            print(f"  flip {flip} start {round(float(np.degrees(a0)))} dice {dice:.3f} anatomy corr {c:.3f}", flush=True)
            if best is None or dice + 0.5 * c > best[0]: best = (dice + 0.5 * c, t, flip, mv, gi, dice, c)
    _, t, flip, mv, gi, dice, c = best; print(name, "similarity: flip", flip, "dice", round(dice, 3), "corr", round(c, 3), flush=True)
    moved = sitk.Resample(mv, fixed, t, sitk.sitkLinear, -50.0)
    dem = sitk.DemonsRegistrationFilter(); dem.SetNumberOfIterations(300); dem.SetStandardDeviations(6.0); dem.SetSmoothDisplacementField(True)
    field = dem.Execute(fixed, moved)
    disp = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64)); comp = sitk.CompositeTransform([t, disp])
    w = sitk.GetArrayFromImage(sitk.Resample(mv, fixed, comp, sitk.sitkLinear, -50.0)) > 0
    wg = sitk.GetArrayFromImage(sitk.Resample(gi, fixed, comp, sitk.sitkLinear, 0.0)); both = w & fm
    print(name, "demons dice", round(float(2 * both.sum() / (w.sum() + fm.sum())), 4), "anatomy corr",
          round(float(np.corrcoef(ndi.gaussian_filter(wg, 2)[both], ndi.gaussian_filter(fg, 2)[both])[0, 1]), 3), flush=True)
    return t, disp

def onto_face(a, chain, spacing, order=sitk.sitkLinear):
    """an array (rows, cols[, channels]) on a slide or canvas of this spacing onto the face, upright"""
    im = sitk.GetImageFromArray(np.ascontiguousarray(a, np.float32), isVector=a.ndim == 3); im.SetSpacing((spacing, spacing))
    return sitk.GetArrayFromImage(sitk.Resample(im, ref, sitk.CompositeTransform(chain), order, 0.0))

out, fits, masks = {}, {}, {}
for name, se in SE.items():
    cv = np.asarray(Image.open(f"{TRS}/SE_{se}.tif").convert("L"), np.float32)     # tissue bright on black
    od_, m_ = canvas_mask(cv); masks[name] = m_
    if name in ON_GLASS:
        o = masks[ON_GLASS[name]]; t, disp = fits[ON_GLASS[name]]
        print(name, "on glass: the fit of", ON_GLASS[name], "; the two canvases' tissue, dice", round(float(2 * (m_ & o).sum() / (m_.sum() + o.sum())), 3), flush=True)
    else:
        t, disp = fit(name, od_, m_)
    fits[name] = (t, disp)
    f = rows[se]["dsPath"].replace("/analyticsdata/", "/home/users/jignesh/analytics/")
    rgb = np.asarray(Image.open(f).convert("RGB"), np.float32) / 255
    od = ndi.gaussian_filter(density(rgb), 1.5)
    # which way the lab's rotation turns: the slide must land on its own canvas
    small = cv[::4, ::4]; probe = sitk.Image(small.shape[1], small.shape[0], sitk.sitkFloat32); probe.SetSpacing((CV * 4,) * 2)
    odi = sitk.GetImageFromArray(od.astype(np.float32)); odi.SetSpacing((UM16, UM16))
    score = {}
    for sign in (1, -1):
        w = sitk.GetArrayFromImage(sitk.Resample(odi, probe, placement(se, sign), sitk.sitkLinear, 0.0))
        score[sign] = float(np.corrcoef(w.ravel(), small.ravel())[0, 1])
    sign = max(score, key=score.get); print(name, "slide on its canvas, corr by rotation sign", {k: round(v, 3) for k, v in score.items()}, flush=True)
    chain = [placement(se, sign), t, disp]
    wod = onto_face(od, chain, UM16); k = tissue & (wod > 0)
    print(name, "slide on the face, anatomy corr", round(float(np.corrcoef(ndi.gaussian_filter(wod, 8)[k], ndi.gaussian_filter(nissl, 8)[k])[0, 1]), 3), flush=True)
    seen = onto_face(m_.astype(np.float32), [t, disp], CV * F) > 0.5
    if name == "outlined":
        js = json.load(open(OUTLINE)); mask = Image.new("L", (rgb.shape[1], rgb.shape[0]), 0); dr = ImageDraw.Draw(mask)
        for ft in js["features"]:
            g = ft["geometry"]; polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            for poly in polys:
                ring = np.asarray(poly[0], np.float64)                      # full-scan pixels, x = column, y = -row
                dr.polygon([(x / 32 - 0.5, -y / 32 - 0.5) for x, y in ring], fill=255)
        m = np.asarray(mask, np.float32) / 255; print("outlined neuronal loss on the slide mm2", round(float(m.sum() * UM16 ** 2)), flush=True)
        out["outline"] = onto_face(m, chain, UM16); out["outlined_od"] = wod
        continue
    sm = np.stack([ndi.gaussian_filter(rgb[..., i], 1.5) for i in range(3)], -1)         # 16 um to 60 um without aliasing
    out[name] = np.clip(onto_face(sm, chain, UM16), 0, 1); out[name + "_seen"] = seen & (onto_face(np.ones(rgb.shape[:2], np.float32), chain, UM16) > 0.5)
    del rgb, sm, od
np.savez_compressed(f"{OUT}/stains.npz", **{k: (v.astype(np.float16) if v.dtype != bool else v) for k, v in out.items()})

# ---- the images: columns w, rows v, inferior first ----
up = lambda a: a[::-1]
tissue = up(tissue); nissl = up(nissl); mri = d["mri"].astype(np.float32); dwi = d["dwi"].astype(np.float32)
bf = np.load(f"{OUT}/bfi.npz"); bfi = up(bf["rgb"].astype(np.float32)); seen_b = up(bf["seen"])
def scale(x, k, lo=1, hi=99.5):
    a_, b_ = np.percentile(x[k], [lo, hi]); return np.clip((x - a_) / (b_ - a_), 0, 1)
alpha = ndi.gaussian_filter(tissue.astype(np.float32), 1.5)
def stretch(rgb, k, top=1.5):       # one optical density range per stain, hue kept
    od = -np.log(np.clip(rgb, 1 / 255, 1)); s = od.sum(-1)
    return np.exp(-od * (top / np.percentile(s[k], 99.5)))
grey = {"mri": scale(mri, tissue), "dwi": scale(dwi, tissue & (dwi > 0), 1, 99.7), "nissl": scale(nissl, tissue & (nissl > 12), 2, 99.7) ** 0.85}
colour = {}
# the photo: the camera's own colour, with local contrast brought up (the block face is lit flat)
from skimage import color as skc, exposure
L = skc.rgb2lab(np.clip(bfi, 0, 1)); l0 = L[..., 0] / 100
L[..., 0] = exposure.equalize_adapthist(np.clip(l0, 0, 1), kernel_size=256, clip_limit=0.012) * 100
colour["bfi"] = (np.clip(skc.lab2rgb(L), 0, 1), seen_b & tissue)
for k in ("he", "ihc1", "ihc2"):
    seen = up(out[k + "_seen"]) & tissue
    colour[k] = (stretch(up(out[k].astype(np.float32)), seen), seen)

os.makedirs(f"{A}/img/hb02", exist_ok=True)
def save(name, rgb, a, lossless=False, sizes=(("", 1), ("_s", 2))):
    for suffix, s in sizes:
        img = np.dstack([(np.clip(rgb, 0, 1) * 255).astype(np.uint8), (np.clip(a, 0, 1) * 255).astype(np.uint8)])
        im = Image.fromarray(img, "RGBA")
        if s > 1: im = im.resize((im.width // s, im.height // s), Image.LANCZOS)
        f = f"{A}/img/hb02/face_{name}{suffix}.webp"
        im.save(f, lossless=True, method=6) if lossless else im.save(f, quality=86, method=6, alpha_quality=80)
        print(name + suffix, im.size, os.path.getsize(f) // 1024, "KB", flush=True)
for k, v in grey.items():
    save(k, np.repeat(v[..., None], 3, -1), alpha * (dwi > 0) if k == "dwi" else alpha)
for k, (rgb, seen) in colour.items():
    a = alpha * ndi.gaussian_filter(seen.astype(np.float32), 1.0)
    save(k, np.where(seen[..., None], rgb, 1.0), a)
# the two outlines as signed distance, 0.5 on the line and 0.25 a millimetre: the page draws both where they cross one
# half, at one width, and a ramp read between texels stays a ramp
def edge(m):
    sd = (ndi.distance_transform_edt(m) - ndi.distance_transform_edt(~m)) * 0.06
    return np.clip(0.5 + sd / 4, 0, 1)
seg = ndi.gaussian_filter(d["seg"].astype(np.float32), 33) > 0.5    # smoothed 2 mm: the in vivo scan's thick slices show as steps
hist = ndi.gaussian_filter(up(out["outline"].astype(np.float32)), 4) > 0.5
z = np.zeros(seg.shape, np.float32)
save("lines", np.dstack([edge(seg), edge(hist), z]), np.ones_like(z), lossless=True, sizes=(("_s", 2),))   # smooth: half size is enough
on = tissue & (alpha > 0.5); a_dwi = float((seg & on).sum() * 0.06 ** 2); a_his = float((hist & on).sum() * 0.06 ** 2)
both = float((seg & hist & on).sum() * 0.06 ** 2)
print(f"on the face: diffusion lesion {a_dwi:.0f} mm2, outlined neuronal loss {a_his:.0f} mm2, both {both:.0f} mm2", flush=True)
json.dump(dict(dwi_mm2=round(a_dwi), histology_mm2=round(a_his), both_mm2=round(both)), open(f"{OUT}/outline_areas.json", "w"))
# where the stop looks on this face: the middle of the largest piece of the diffusion lesion, in the volume's centred
# frame (mm, v up and w forward), and its half size
LO, HI = np.array([3.6, -203.2, 90.4]), np.array([160.4, -23.2, 187.2])             # the box of build_hb02.py
lab, n = ndi.label(seg & on); big = lab == 1 + np.argmax(ndi.sum(on, lab, range(1, n + 1)))
r, c = np.nonzero(big); r0, r1, c0, c1 = r.min(), r.max(), c.min(), c.max()
v = LO[2] + ((r0 + r1) / 2 + 0.5) * 0.06 - (LO[2] + HI[2]) / 2; w = LO[1] + ((c0 + c1) / 2 + 0.5) * 0.06 - (LO[1] + HI[1]) / 2
half = max(r1 - r0, c1 - c0) / 2 * 0.06
idx = json.load(open(f"{A}/volumes/brains.json"))
for b in idx["brains"]:
    if b["id"] == "hb02": b["face_focus"] = [round(float(v), 1), round(float(w), 1), round(float(half), 1)]
json.dump(idx, open(f"{A}/volumes/brains.json", "w"), indent=1)
print("face focus v, w, half mm", round(float(v), 1), round(float(w), 1), round(float(half), 1), flush=True)

# a still for the plain page: the face in each modality, two by two, in the stage's colours, superior first; the last
# panel is the Nissl again with both outlines
def ramp(d_, stops, cols):
    return np.stack([np.interp(d_, stops, np.array(cols)[:, i]) for i in range(3)], -1)
vio = ramp(grey["nissl"], [0, .3, .62, .85, 1], [[.03, .026, .06], [.15, .09, .4], [.44, .29, .93], [.8, .72, 1], [.97, .95, 1]])
gre = lambda m8: ramp(m8, [0, .5, 1], [[.03, .028, .05], [.3, .3, .35], [.937, .918, .875]])
ring = lambda f: f ^ ndi.binary_erosion(f, iterations=6)
x0, x1 = np.nonzero(tissue.any(0))[0][[0, -1]]; y0, y1 = np.nonzero(tissue.any(1))[0][[0, -1]]
tiles = []
for k, c in enumerate((gre(grey["mri"]), gre(grey["dwi"]), colour["bfi"][0], vio, colour["he"][0], colour["ihc1"][0], colour["ihc2"][0], vio)):
    c = np.where(ring(seg)[..., None], np.array([0.839, 1.0, 0.247]), c)
    if k == 7: c = np.where(ring(hist)[..., None], np.array([0.98, 0.95, 0.88]), c)
    c = np.where(tissue[..., None], c, 0.03)[y0:y1 + 1, x0:x1 + 1][::-1]
    im = Image.fromarray((np.clip(c, 0, 1) * 255).astype(np.uint8)); tiles.append(im.resize((600, round(600 * im.height / im.width)), Image.LANCZOS))
th = tiles[0].height; still = Image.new("RGB", (1200, 4 * th), (8, 7, 15))
for k, im in enumerate(tiles): still.paste(im, ((k % 2) * 600, (k // 2) * th))
still.save(f"{A}/img/hb02/stroke_faces.webp", quality=80, method=6)
print("still", still.size, os.path.getsize(f"{A}/img/hb02/stroke_faces.webp") // 1024, "KB", flush=True)
