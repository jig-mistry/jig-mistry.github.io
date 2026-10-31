"""HB63L streamlines as individual lines on black: a sparse whole-section view and a denser close-up."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
HB = "/home/users/jignesh/workspace/application/myelin_orientation/HB63L"
sys.path.insert(0, HB)
import numpy as np, cv2
import sections
from st_tracks import render_fast
OUT = os.path.abspath(sys.argv[1]); sec = sys.argv[2]
os.chdir(HB)
d = os.path.join("out_st", sec)
mask = np.load(os.path.join(d, "mask5.npy")) > 0
shape_raw = mask.shape
mask = sections.rot_img(mask, sec)
z = np.load(os.path.join(d, "streamlines.npz"))
lines = [sections.rot_points(z[k].astype(np.float64), sec, shape_raw) for k in z.files]
lines = [l for l in lines if len(l) > 12]
rng = np.random.default_rng(1)
def draw(S, keep, crop=None, outline=True):
    h, w = mask.shape
    base = np.zeros((round(h * S), round(w * S), 3), np.uint8)
    if outline:
        m = cv2.resize(mask.astype(np.uint8), (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(base, [c for c in cs if cv2.contourArea(c) > 2000 * S * S], -1, (60, 60, 60), max(1, round(2 * S)), cv2.LINE_AA)
    sel = [lines[i] for i in rng.permutation(len(lines))[:int(len(lines) * keep)]]
    img = render_fast(base, sel, scale=S, thickness=1)
    if crop:
        x0, y0, x1, y1 = (int(c * s) for c, s in zip(crop, (img.shape[1], img.shape[0]) * 2))
        img = img[y0:y1, x0:x1]
    return img
print(len(lines), "lines")
cv2.imwrite(f"{OUT}/{sec}_lines_full.png", draw(0.5, 0.12))
cv2.imwrite(f"{OUT}/{sec}_lines_zoom.png", draw(1.0, 0.25, crop=(0.28, 0.42, 0.62, 0.70), outline=False))
