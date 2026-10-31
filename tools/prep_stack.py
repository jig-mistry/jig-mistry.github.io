"""Prepare the FB74 histology stack (histo_stacked.mnc) for the zoom render.

Reads the MINC once (nibabel + h5py), crops to the tissue bounding box, and writes to renders/work/:
  stack_crop.npy      uint16 full-resolution crop (z, y, x)
  stack_mask2.npy     float16 smoothed tissue mask at 2x down-sampling (normals)
  stack_occ4.npy      uint8 dilated tissue occupancy at 4x down-sampling (empty-space skip)
  stack_meta.json     crop offsets, spacing, tissue percentiles
"""
import os, json, time
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "8"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
os.makedirs(WORK, exist_ok=True)
SRC = "/home/users/jignesh/workspace/application/tirl/mcmadras_reg/madras_run/sub-002_v5/histo_stacked.mnc"
dev = torch.device("cuda")
torch.set_num_threads(8)

t0 = time.time()
v = np.asarray(nib.load(SRC).dataobj)
v = np.rint(v).astype(np.uint16)
print("loaded", v.shape, round(time.time() - t0, 1), "s")
m = v > 0
# drop isolated specks: keep components larger than 0.1% of the largest
m4 = m[::4, ::4, ::4]
lab, n = ndi.label(m4)
sz = np.bincount(lab.ravel()); sz[0] = 0
print("ds4 components:", n, "largest sizes", np.sort(sz)[::-1][:6].tolist())
keep4 = np.isin(lab, np.nonzero(sz > 0.001 * sz.max())[0])
keep4 = ndi.binary_dilation(keep4, iterations=2)
keep = np.repeat(np.repeat(np.repeat(keep4, 4, 0), 4, 1), 4, 2)[:v.shape[0], :v.shape[1], :v.shape[2]]
m &= keep
v = np.where(m, v, 0).astype(np.uint16)
idx = [np.nonzero(m.any(axis=tuple(a for a in range(3) if a != ax)))[0] for ax in range(3)]
lo = [max(0, int(ix.min()) - 6) for ix in idx]
hi = [min(s, int(ix.max()) + 7) for ix, s in zip(idx, v.shape)]
crop = v[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
mc = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
print("crop", list(zip(lo, hi)), crop.shape)
tis = crop[mc]
pct = {str(p): float(np.percentile(tis[::7], p)) for p in [0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.9]}
print("tissue percentiles", pct)
np.save(os.path.join(WORK, "stack_crop.npy"), crop)

with torch.no_grad():
    mt = torch.from_numpy(mc.astype(np.float32)).to(dev)[None, None]
    # pad to multiples of 4
    pad = [(-s) % 4 for s in mc.shape]
    mt = F.pad(mt, (0, pad[2], 0, pad[1], 0, pad[0]))
    m2 = F.avg_pool3d(mt, 2)
    # gaussian sigma 1.2 (ds2 voxels)
    r = 4
    x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / 1.2) ** 2); k = k / k.sum()
    y = m2
    y = F.conv3d(F.pad(y, (0, 0, 0, 0, r, r)), k.view(1, 1, -1, 1, 1))
    y = F.conv3d(F.pad(y, (0, 0, r, r, 0, 0)), k.view(1, 1, 1, -1, 1))
    y = F.conv3d(F.pad(y, (r, r, 0, 0, 0, 0)), k.view(1, 1, 1, 1, -1))
    np.save(os.path.join(WORK, "stack_mask2.npy"), y[0, 0].half().cpu().numpy())
    o4 = F.max_pool3d(mt, 4)
    o4 = F.max_pool3d(o4, 3, stride=1, padding=1)
    np.save(os.path.join(WORK, "stack_occ4.npy"), o4[0, 0].byte().cpu().numpy())
meta = dict(src=SRC, full_shape=list(v.shape), crop_lo=lo, crop_hi=hi, crop_shape=list(crop.shape),
            spacing_zyx_mm=[0.0632, 0.064, 0.064], percentiles=pct)
json.dump(meta, open(os.path.join(WORK, "stack_meta.json"), "w"), indent=1)
print("done", round(time.time() - t0, 1), "s")
