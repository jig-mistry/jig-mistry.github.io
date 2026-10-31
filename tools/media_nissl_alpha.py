"""Nissl zoom levels of the 24 GW brain (DHARANI FB85, SE 1138) with the glass made transparent, so only tissue
is drawn.  The tissue mask comes from each level's own image blurred to the same physical scale (~20 um), so the
outline agrees from the whole section down to single cells."""
import os, sys, json, subprocess, tempfile
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import morphology, segmentation
Image.MAX_IMAGE_PIXELS = None
SITE = sys.argv[1]
OPJ = "/home/users/jignesh/miniconda3/envs/nisslprop/bin/opj_decompress"
J85 = "/home/users/jignesh/analytics/244/NISL/B_244_FB85-SL_380-ST_NISL-SE_1138_compressed.jp2"
W85, H85 = 144472, 129351
UM = 0.5                      # level-0 pixel size, um
cx, cy = 36100, 54300
TMP = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(__file__)))
THR = float(os.environ.get("THR", 0.055))    # optical density after blurring: glass ~0.025 (p95 0.05), pale white matter >= 0.067

def jp2(r, box):
    out = os.path.join(TMP, "c.png")
    subprocess.run([OPJ, "-i", J85, "-o", out, "-r", str(r), "-d", ",".join(map(str, box)), "-threads", "8"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return Image.open(out).convert("RGB")

boxes = [("nissl_section", 5, (0, 0, W85, H85), 2400),
         ("nissl_ge", 3, (cx - 10000, cy - 10000, cx + 10000, cy + 10000), 1700),
         ("nissl_ge_mid", 1, (cx - 3000, cy - 3000, cx + 3000, cy + 3000), 1600),
         ("nissl_cells", 0, (cx - 800, cy - 800, cx + 800, cy + 800), 1600)]
levels = []
for name, r, box, size in boxes:
    im = jp2(r, box)
    im.thumbnail((size, size), Image.Resampling.LANCZOS)
    a = np.asarray(im).astype(np.float32)
    um_px = (box[2] - box[0]) * UM / im.width
    od = -np.log10(np.clip(a.min(-1), 1, 255) / 255.0)              # darkest channel: violet stain
    od = ndi.gaussian_filter(od, max(1.0, 30.0 / um_px))
    m = od > THR
    m = morphology.remove_small_objects(m, max_size=max(4, int(0.2e6 / um_px ** 2)))     # debris < 0.2 mm2
    if r == 5:                                 # whole section: the slide edge is not tissue
        m = segmentation.clear_border(m)
    m = morphology.remove_small_holes(m, max_size=max(4, int(0.5e6 / um_px ** 2)))       # pale gaps < 0.5 mm2; ventricles stay open
    alpha = ndi.gaussian_filter(m.astype(np.float32), max(0.7, 4.0 / um_px))        # soft ~4 um edge
    rgba = np.dstack([a, np.clip(alpha, 0, 1) * 255]).astype(np.uint8)
    path = f"{SITE}/assets/img/ge/{name}.webp"
    Image.fromarray(rgba, "RGBA").save(path, quality=74, alpha_quality=60, method=6)
    print(name, im.size, f"{um_px:.2f} um/px", f"tissue {m.mean():.2f}", os.path.getsize(path) // 1024, "KB", flush=True)
    levels.append(dict(src=f"assets/img/ge/{name}.webp", rect=[box[0] / W85, box[1] / H85, box[2] / W85, box[3] / H85],
                       px=[im.width, im.height]))
json.dump(dict(aspect=W85 / H85, levels=levels), open(f"{SITE}/assets/img/ge/nissl_zoom.json", "w"), indent=1)
