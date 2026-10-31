"""Prepare smoothed brain-envelope and GE fields at ~0.12 mm for the GE turntable render.

Writes renders/work/ge_fields_<brain>.npz with float16 fields (1 inside, 0 outside, smooth
0.5 iso-surface), spacing in mm, and a few stats. Never touches the source volumes.
"""
import os, sys, time
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
D = "/home/users/jignesh/workspace/application/dharani_seg/ge_seg/volumes/"
dev = torch.device("cuda")
CLOSE_SIG = 6.0
torch.set_num_threads(8)


def gauss_kernel(sigma):
    r = int(np.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=dev)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum(), r


def blur(v, sigma):
    """Separable 3-D Gaussian on a (Ni,Nj,Nk) float tensor, zero padding. sigma: scalar or 3-tuple."""
    if np.isscalar(sigma):
        sigma = (sigma, sigma, sigma)
    x = v[None, None]
    k, r = gauss_kernel(sigma[0])
    x = F.conv3d(F.pad(x, (0, 0, 0, 0, r, r)), k.view(1, 1, -1, 1, 1))
    k, r = gauss_kernel(sigma[1])
    x = F.conv3d(F.pad(x, (0, 0, r, r, 0, 0)), k.view(1, 1, 1, -1, 1))
    k, r = gauss_kernel(sigma[2])
    x = F.conv3d(F.pad(x, (r, r, 0, 0, 0, 0)), k.view(1, 1, 1, 1, -1))
    return x[0, 0]


def fill_holes(m):
    """3-D hole fill: background = components of ~m that touch the volume border."""
    lab, n = ndi.label(~m)
    border = np.unique(np.concatenate([
        lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
        lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    bg = np.isin(lab, border[border > 0])
    return ~bg


def largest_cc(m):
    lab, n = ndi.label(m)
    sz = np.bincount(lab.ravel())
    sz[0] = 0
    order = np.argsort(sz)[::-1]
    print("   CC sizes (top 5, voxels):", sz[order[:5]].tolist())
    return lab == order[0]


def run(brain, dz_um):
    t0 = time.time()
    nis = np.asarray(nib.load(D + f"nissl_{brain}_stain_60um.nii.gz").dataobj)
    ge = np.asarray(nib.load(D + f"ge_{brain}_CLEAN.nii.gz").dataobj)
    print(brain, nis.shape, "loaded", round(time.time() - t0, 1), "s")
    # crop to even sizes then 2x block-average on GPU
    sh = [s - s % 2 for s in nis.shape]
    nis = nis[:sh[0], :sh[1], :sh[2]]
    ge = ge[:sh[0], :sh[1], :sh[2]]
    with torch.no_grad():
        tis = torch.from_numpy(nis).to(dev)
        tis = (tis < 235).half()[None, None]
        tis = F.avg_pool3d(tis.float(), 2)[0, 0]
        gt = torch.from_numpy(ge).to(dev).float()[None, None]
        gt = F.avg_pool3d(gt, 2)[0, 0]
    del nis, ge
    spacing = (0.12, 0.12, 2 * dz_um / 1000.0)
    print("   ds2 shape", tuple(tis.shape), "spacing", spacing)

    # --- brain envelope ---
    PADZ = 12
    tis = F.pad(tis, (PADZ, PADZ))
    gt = F.pad(gt, (PADZ, PADZ))
    m = (tis > 0.5).cpu().numpy()
    m = largest_cc(m)
    for kk in range(m.shape[2]):  # 2-D hole fill inside every sagittal section
        if m[:, :, kk].any():
            m[:, :, kk] = ndi.binary_fill_holes(m[:, :, kk])
    m = fill_holes(m)
    # closing to seal ventricle openings (blur-threshold dilate/erode, r ~ 0.6 mm)
    with torch.no_grad():
        mt = torch.from_numpy(m).to(dev).float()
        dil = (blur(mt, CLOSE_SIG) > 0.08)
    dil = fill_holes(dil.cpu().numpy())
    with torch.no_grad():
        ero = blur(torch.from_numpy(dil).to(dev).float(), CLOSE_SIG) > 0.92
        env = ero | mt.bool()
    env = fill_holes(env.cpu().numpy())
    env = largest_cc(env)
    with torch.no_grad():
        e = torch.from_numpy(env).to(dev).float()
        e = (blur(e, 2.0) > 0.5).float()
        brain_field = blur(e, (2.2, 2.2, 4.0))
        # --- GE ---
        g = (gt > 0.5).float()
        g = g * (blur(e, 1.0) > 0.3).float()  # keep GE inside the envelope
        g = (blur(g, 1.75) > 0.5).float()
        ge_field = blur(g, (1.0, 1.0, 2.0))
        ge_ao = blur(g, 6.0)  # broad blur used for a cheap ambient-occlusion term
    vox_mm3 = spacing[0] * spacing[1] * spacing[2]
    print("   brain vol mm3 %.0f  GE vol mm3 %.0f" % (e.sum().item() * vox_mm3, g.sum().item() * vox_mm3))
    # bounding box of the envelope (index space) for tight crop
    nz = torch.nonzero(brain_field > 0.02)
    lo = (nz.min(0).values - 4).clamp(min=0).cpu().numpy()
    hi = (nz.max(0).values + 5).cpu().numpy()
    hi = np.minimum(hi, np.array(brain_field.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    out = dict(brain=brain_field[sl].half().cpu().numpy(), ge=ge_field[sl].half().cpu().numpy(),
               ge_ao=ge_ao[sl].half().cpu().numpy(), spacing=np.array(spacing),
               crop_lo=lo, shape_ds2=np.array(brain_field.shape))
    np.savez(os.path.join(WORK, f"ge_fields_{brain}.npz"), **out)
    print("   saved, crop", [(int(a), int(b)) for a, b in zip(lo, hi)], round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    which = sys.argv[1:] or ["142", "222"]
    for b in which:
        run(b, 66.0 if b == "142" else 60.0)
