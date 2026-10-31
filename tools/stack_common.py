"""Shared setup for the histology-stack render: data on GPU, anatomical frame, sampler, colour ramp."""
import os, json, math
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
dev = torch.device("cuda")

# in-plane orientation (decided from mid-sections; see README in report)
THETA = 90.0     # superior direction in the (y, x) section plane, degrees from +y towards +x
ANT = 1          # anterior = ANT * (-sin, cos) rotated from superior
RAMP = ["#EDDDF7", "#A880DB", "#4D1F80"]


def hex2rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def frame(theta=THETA, ant=ANT):
    """Columns = world X (lateral), Y (superior), Z (anterior) in physical (z, y, x) mm axes."""
    th = math.radians(theta)
    S = np.array([0.0, math.cos(th), math.sin(th)])
    A = ant * np.array([0.0, -math.sin(th), math.cos(th)])
    L = np.cross(S, A)
    return np.stack([L, S, A], 1)


class Stack:
    def __init__(self, theta=THETA, ant=ANT, full=True):
        meta = json.load(open(os.path.join(WORK, "stack_meta.json")))
        self.meta = meta
        self.sp = torch.tensor(meta["spacing_zyx_mm"], dtype=torch.float32, device=dev)
        pc = meta["percentiles"]
        self.p_lo, self.p_hi = pc["2"], pc["99.5"]
        if full:
            v = np.load(os.path.join(WORK, "stack_crop.npy"), mmap_mode="r")
            self.vol = torch.empty(v.shape, dtype=torch.float16, device=dev)
            for a in range(0, v.shape[0], 64):  # chunked upload, raw/65535 as float16
                self.vol[a:a + 64] = torch.from_numpy(np.asarray(v[a:a + 64]).astype(np.float32) / 65535.0).to(dev).half()
            self.shape = torch.tensor(v.shape, device=dev)
        m2 = np.load(os.path.join(WORK, "stack_mask2.npy")).astype(np.float32)
        self.mask2 = torch.from_numpy(m2).to(dev)[None, None]
        o4 = np.load(os.path.join(WORK, "stack_occ4.npy")).astype(np.float32)
        self.occ4 = torch.from_numpy(o4).to(dev)[None, None]
        cs = np.array(meta["crop_shape"], dtype=np.float32)
        self.cs = torch.tensor(cs, device=dev)
        # world origin = centre of the crop box; anatomical frame
        M = torch.tensor(frame(theta, ant), dtype=torch.float32, device=dev)
        self.M = M
        self.c = (self.cs - 1) / 2
        self.A = M / self.sp[:, None]  # idx = c + A @ world  (full-res crop index)

    def to_idx(self, p):
        return p @ self.A.T + self.c

    def box_hit(self, o, d):
        oi = self.to_idx(o)
        di = d @ self.A.T
        inv = 1.0 / torch.where(di.abs() < 1e-9, torch.full_like(di, 1e-9), di)
        t0 = (0 - oi) * inv
        t1 = (self.cs - 1 - oi) * inv
        return torch.minimum(t0, t1).amax(-1).clamp_min(0), torch.maximum(t0, t1).amin(-1)

    def sample_full(self, p, with_occ=False):
        """Exact trilinear sample of the float16 full-res volume at world points p (..., 3).
        with_occ: also return the trilinear tissue occupancy, and normalise the value by it so the
        zero background does not darken/lighten the tissue edge (normalised convolution)."""
        sh = p.shape[:-1]
        idx = self.to_idx(p.reshape(-1, 3))
        N0, N1, N2 = self.vol.shape
        inside = ((idx >= 0) & (idx <= (self.cs - 1))).all(-1)
        i0 = idx.floor().clamp_min(0)
        i0 = torch.minimum(i0, self.cs - 2)
        f = (idx - i0).clamp(0, 1)
        i0 = i0.long()
        base = (i0[:, 0] * N1 + i0[:, 1]) * N2 + i0[:, 2]
        flat = self.vol.view(-1)
        s1, s0 = N2, N1 * N2
        fz, fy, fx = f[:, 0], f[:, 1], f[:, 2]
        val = torch.zeros_like(fx)
        occ = torch.zeros_like(fx)
        for dz, wz in ((0, 1 - fz), (1, fz)):
            for dy, wy in ((0, 1 - fy), (1, fy)):
                for dx, wx in ((0, 1 - fx), (1, fx)):
                    v = flat[base + (dz * s0 + dy * s1 + dx)].float()
                    w = wz * wy * wx
                    val += w * v
                    if with_occ:
                        occ += w * (v > 0).float()
        val = val * inside
        if not with_occ:
            return val.view(sh)
        occ = occ * inside
        nv = val / occ.clamp_min(1e-4)
        return nv.view(sh), occ.view(sh)

    def _grid(self, p, shape_ds):
        """normalized grid coords for a down-sampled volume whose voxel n covers crop voxels
        [n*f, (n+1)*f) (avg/max pool): centre at idx (n + 0.5) * f - 0.5."""
        idx = self.to_idx(p.reshape(-1, 3))
        f = self.cs.new_tensor([1.0, 1.0, 1.0])
        return idx

    def sample_small(self, vol, factor, p):
        sh = p.shape[:-1]
        idx = self.to_idx(p.reshape(-1, 3))
        n = (idx + 0.5) / factor - 0.5  # index in the pooled volume
        D, H, W = vol.shape[2:]
        sz = torch.tensor([D, H, W], device=dev, dtype=torch.float32)
        g = 2 * n / (sz - 1) - 1
        g = g[:, [2, 1, 0]]
        out = F.grid_sample(vol, g.view(1, 1, 1, -1, 3), mode="bilinear", padding_mode="zeros", align_corners=True)
        return out.view(-1).view(sh)

    def density(self, raw):
        """raw (0..1 of uint16 range) -> normalised Nissl density 0..1."""
        lo, hi = self.p_lo / 65535.0, self.p_hi / 65535.0
        return ((raw - lo) / (hi - lo)).clamp(0, 1)


def ramp(d, gamma=0.85):
    """density 0..1 -> RGB through the three-stop Nissl violet ramp."""
    c = torch.tensor([hex2rgb(h) for h in RAMP], device=d.device)
    t = d.clamp(0, 1) ** gamma
    a = (t * 2).clamp(0, 1)
    b = (t * 2 - 1).clamp(0, 1)
    lowmid = c[0] * (1 - a[..., None]) + c[1] * a[..., None]
    return torch.where((t < 0.5)[..., None], lowmid, c[1] * (1 - b[..., None]) + c[2] * b[..., None])
