"""Replace the 14 GW brain (sub-001 = FB40) with the lab's VALIS reconstruction, which stacks far more cleanly
than the mcmadras run.  Writes dharani/sub-001_032.npy in the same frame as prep_dharani.py
(axis 0 = section, then the section plane) and sets its pose in orient.json / info.json."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, nibabel as nib
from scipy import ndimage as ndi

SRC = "/home/users/jignesh/workspace/application/tirl/valis_ritesh/stain/out/141_nisl_v2/B_141_nisl_recon_60um.nii.gz"
D = sys.argv[1]            # .../site_build/dharani
TARGET = 0.32              # mm
BG = 5.0                   # density (0..255) below this is bare slide

im = nib.load(SRC)
d = np.asarray(im.dataobj, dtype=np.float32)            # (x, y, section), white background, 0 outside the slides
zooms = np.array(im.header.get_zooms()[:3], dtype=np.float64)
dens = np.where(d > 0, 255 - d, 0).transpose(2, 0, 1)   # stain density, (section, x, y)
f = zooms[[2, 0, 1]] / TARGET
dens = ndi.zoom(ndi.gaussian_filter(dens, (1 / f) / 2.5), f, order=1)
dens[dens < BG] = 0
if os.path.exists(f"{D}/sub-001_032.npy") and not os.path.exists(f"{D}/sub-001_032_mcmadras.npy"):
    os.rename(f"{D}/sub-001_032.npy", f"{D}/sub-001_032_mcmadras.npy")
np.save(f"{D}/sub-001_032.npy", dens.astype(np.float32))
print("sub-001", d.shape, zooms, "->", dens.shape)

# pose from the same silhouette fit as orient_search.py (IoU 0.86 against sub-003): upside down, no mirror
orient = json.load(open(f"{D}/orient.json"))
orient["sub-001"] = dict(iou=0.865, angle=174, flip=False)
json.dump(orient, open(f"{D}/orient.json", "w"), indent=1)
info = json.load(open(f"{D}/info.json"))
info["sub-001"].update(recipe="valis_141_nisl_v2", src=SRC, shape=list(dens.shape))
json.dump(info, open(f"{D}/info.json", "w"), indent=1)
