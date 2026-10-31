"""Find the in-plane rotation (+mirror) putting each brain's lateral silhouette in sub-003's pose (dorsal up, anterior left)."""
import numpy as np, json
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
D = "dharani"
def sil(s):
    d = np.load(f"{D}/{s}_032.npy"); thr = np.percentile(d[d > 0], 20)
    m = ndi.binary_opening(d > thr, iterations=1)
    lab, n = ndi.label(m); sz = ndi.sum(m, lab, range(1, n + 1)); m = lab == (1 + np.argmax(sz))
    if s == "sub-004":   # coronal: sagittal view is (z, y); show as rows=y? keep (y, z) so rows = superior-inferior
        p = m.any(axis=2).T.astype(float)      # (y, z)
    else:
        p = m.any(axis=0).astype(float)        # (y, x)
    return p
def canon(p, out=160):
    cy, cx = ndi.center_of_mass(p); r = np.sqrt(p.sum() / np.pi)
    sc = 50.0 / r
    yy, xx = np.mgrid[0:out, 0:out] - out / 2
    return lambda ang, flip: ndi.map_coordinates(p, [cy + (np.cos(ang) * yy - np.sin(ang) * xx * (-1 if flip else 1)) / sc,
                                                    cx + (np.sin(ang) * yy + np.cos(ang) * xx * (-1 if flip else 1)) / sc], order=1) > 0.5
ref = canon(sil("sub-003"))(0.0, False)
res = {}
tiles = []
for s in ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005"]:
    f = canon(sil(s)); best = (-1, 0, False)
    for flip in (False, True):
        for a in range(0, 360, 3):
            m = f(np.deg2rad(a), flip); iou = (m & ref).sum() / (m | ref).sum()
            if iou > best[0]: best = (iou, a, flip)
    res[s] = dict(iou=round(float(best[0]), 3), angle=best[1], flip=best[2])
    tiles.append((s, f(np.deg2rad(best[1]), best[2])))
    print(s, res[s], flush=True)
json.dump(res, open(f"{D}/orient.json", "w"), indent=1)
img = Image.new("RGB", (160 * 5, 170)); dr = ImageDraw.Draw(img)
for k, (s, m) in enumerate(tiles):
    rgb = np.zeros(m.shape + (3,), np.uint8); rgb[..., 0] = m * 255; rgb[..., 1] = ref * 255
    img.paste(Image.fromarray(rgb), (160 * k, 10)); dr.text((160 * k + 4, 0), s, fill=(255, 255, 255))
img.save(f"{D}/orient_fit.png")
