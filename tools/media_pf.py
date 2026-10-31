"""GE zoom levels from the two-colour pseudo-fluorescence section, turned upright (90 deg clockwise)."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, tifffile, zarr
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
F = "/home/users/jignesh/neuroglancer/ihc_pair_registered/pseudofluor_DCXgreen_S100bred_0p5um.tif"
OUT = sys.argv[1]
t = tifffile.TiffFile(F)
H0, W0 = t.pages[0].shape[:2]                      # stored frame, level 0
def region(level, x0, y0, x1, y1):
    """stored-frame level-0 box -> RGB array at that pyramid level"""
    z = zarr.open(t.aszarr(level=level), mode="r")
    f = 2 ** level
    return np.asarray(z[y0 // f:y1 // f, x0 // f:x1 // f])
def to_display_box(x0, y0, x1, y1):
    """stored box -> display box (clockwise 90: x' = H-1-y, y' = x), normalised to the display frame"""
    W, H = H0, W0
    return [(H0 - y1) / W, x0 / H, (H0 - y0) / W, x1 / H]
def save(arr, name, size):
    im = Image.fromarray(arr).transpose(Image.Transpose.ROTATE_270)   # 90 deg clockwise
    if size: im.thumbnail((size, size), Image.Resampling.LANCZOS)
    im.save(f"{OUT}/{name}.webp", quality=84, method=6)
    return im.size
levels = []
boxes = {
    "pf_section": (5, (0, 0, W0, H0), 2400),
    "pf_ge":      (3, (32000, 34000, 56000, 58000), 2400),
    "pf_ge_mid":  (1, (40416, 35712, 48416, 43712), 2400),
    "pf_cells":   (0, (43416, 38712, 45416, 40712), 2000),
}
for name, (lvl, box, size) in boxes.items():
    a = region(lvl, *box)
    sz = save(a, name, size)
    levels.append(dict(src=f"assets/img/ge/{name}.webp", rect=[round(v, 6) for v in to_display_box(*box)], px=list(sz)))
    print(name, a.shape, "->", sz, os.path.getsize(f"{OUT}/{name}.webp") // 1024, "KB", flush=True)
json.dump(dict(aspect=H0 / W0, levels=levels), open(f"{OUT}/pf_zoom.json", "w"), indent=1)
