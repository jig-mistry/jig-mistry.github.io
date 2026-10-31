"""Three-colour immunostained crop of the fetal cortical wall with the glass made transparent.

Right of the blue pial band is glass with sparse coloured specks.  Tissue = blurred signal above threshold, largest
piece, holes filled; the alpha comes from a smoothed signed distance to that outline (smooth, anti-aliased edge).
Window: level 2 (2 um/px) x 15600-17200, y 6800-8800 of the registered 7-marker canvas (markers not named on the site).
usage: python media_ihc_alpha.py SITE_DIR
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, tifffile, zarr
from PIL import Image
from scipy import ndimage as ndi
from skimage import morphology
SITE = sys.argv[1]
P1 = "/home/users/jignesh/neuroglancer/ihc7_panels_registered/panel1_NeuN-green_CTIP2-red_Reelin-blue_0p5um.tif"
t = tifffile.TiffFile(P1)
a = np.asarray(zarr.open(t.aszarr(level=2), mode="r")[6800:8800, 15600:17200])        # (2000, 1600, 3), 2 um/px
sig = ndi.gaussian_filter(a.astype(np.float32).max(-1), 4.0)                         # ~8 um
thr = float(os.environ.get("THR", 45))      # high enough that the dim halo and specks on the pia fall outside
m = sig > thr
lab, n = ndi.label(m)
m = lab == 1 + int(np.argmax(ndi.sum(m, lab, range(1, n + 1))))                     # the wall; specks are separate
# the tissue runs on past the top, bottom and left of the crop: pad those sides as tissue while cleaning,
# so holes and the opening do not nick the crop border
P = 24
m = np.pad(m, ((P, P), (P, 0)), constant_values=True)
m = ndi.binary_fill_holes(m)
m = morphology.opening(m, morphology.disk(16))                                       # ~30 um: no specks or threads on the pia
d = ndi.distance_transform_edt(m) - ndi.distance_transform_edt(~m)
d = ndi.gaussian_filter(d.astype(np.float32), 5.0)[P:-P, P:]
m = m[P:-P, P:]
alpha = np.clip(0.5 + d / 1.1, 0, 1)
im = Image.fromarray(np.dstack([a, (alpha * 255).round()]).astype(np.uint8), "RGBA")
im.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
im.save(f"{SITE}/assets/img/cyto/ihc_layers.webp", quality=82, alpha_quality=80, method=6)
print(im.size, "tissue", round(m.mean(), 3), "threshold", thr, os.path.getsize(f"{SITE}/assets/img/cyto/ihc_layers.webp") // 1024, "KB")
