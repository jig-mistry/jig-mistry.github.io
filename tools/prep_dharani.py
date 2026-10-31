"""Downsample the five DHARANI mcmadras stacks to ~0.32 mm and write mid-plane montages for an orientation check."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import sys, json
import numpy as np, nibabel as nib, torch
from PIL import Image

R = "/home/users/jignesh/workspace/application/tirl/mcmadras_reg/madras_run/results"
SUBS = {  # subject: (recipe, GW, brain)
    "sub-001": ("B_plus4_nl3D", 14, "FB40"),
    "sub-002": ("E_plus4_mixed", 17, "FB74"),
    "sub-003": ("B_plus4_nl3D", 21, "FB34"),
    "sub-004": ("B_plus4_nl3D", 22, "FB62"),
    "sub-005": ("B_plus4_nl3D", 24, "FB85"),
}
OUT = sys.argv[1]
TARGET = 0.32  # mm
os.makedirs(OUT, exist_ok=True)
info = {}
for sub, (rec, gw, fb) in SUBS.items():
    f = f"{R}/{rec}/{sub}/align_nissl_1u/optical/histo_stacked.mnc"
    im = nib.load(f)
    zooms = [float(z) for z in im.header.get_zooms()]
    a = np.asarray(im.dataobj, dtype=np.float32)
    t = torch.from_numpy(a).cuda()[None, None]
    size = [max(1, round(s * z / TARGET)) for s, z in zip(a.shape, zooms)]
    d = torch.nn.functional.interpolate(t, size=size, mode="area")[0, 0].cpu().numpy()
    p = np.percentile(d[d > 0], [1, 50, 99]) if (d > 0).any() else None
    print(sub, fb, gw, a.shape, zooms, "->", d.shape, "min/max", float(a.min()), float(a.max()), "pct", p, flush=True)
    np.save(f"{OUT}/{sub}_032.npy", d.astype(np.float32))
    info[sub] = dict(recipe=rec, gw=gw, brain=fb, src=f, shape=list(d.shape), spacing=TARGET)
    # mid planes, each normalised
    def norm(x):
        lo, hi = np.percentile(x, [1, 99.5]); return (np.clip((x - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
    planes = [d[d.shape[0] // 2], d[:, d.shape[1] // 2], d[:, :, d.shape[2] // 2]]
    ims = [Image.fromarray(norm(pl)) for pl in planes]
    H = max(i.height for i in ims); W = sum(i.width for i in ims) + 20
    m = Image.new("L", (W, H), 128); x = 0
    for i in ims: m.paste(i, (x, 0)); x += i.width + 10
    m.save(f"{OUT}/{sub}_planes.png")
    del a, t
json.dump(info, open(f"{OUT}/info.json", "w"), indent=1)
