"""Small GPU ray-marcher utilities (torch).

World frame: X = screen-right at azimuth 0 (lateral), Y = up (dorsal), Z = towards the viewer at
azimuth 0. A Volume maps world mm -> array index through an orthonormal anatomical frame given in the
array's physical (i, j, k) axes, so data are never resampled to change orientation.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


def normalize(v, eps=1e-12):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


class Volume:
    """data: tensor (C, Ni, Nj, Nk). spacing: (si, sj, sk) mm.
    frame: 3x3, columns = world X, Y, Z axes expressed in physical (i, j, k) mm coordinates.
    center_idx: array index that sits at the world origin."""

    def __init__(self, data, spacing, frame, center_idx):
        self.data = data[None]  # (1, C, Ni, Nj, Nk)
        self.dev = data.device
        self.shape = torch.tensor(data.shape[1:], dtype=torch.float32, device=self.dev)
        sp = torch.tensor(spacing, dtype=torch.float32, device=self.dev)
        M = torch.tensor(np.asarray(frame, dtype=np.float32), device=self.dev)
        # index = c + A @ world
        self.A = M / sp[:, None]
        self.c = torch.tensor(np.asarray(center_idx, dtype=np.float32), device=self.dev)
        # normalized grid coords in grid_sample order (k, j, i): g = G @ world + g0
        scale = 2.0 / (self.shape - 1)
        G = self.A * scale[:, None]
        g0 = self.c * scale - 1
        perm = [2, 1, 0]
        self.G = G[perm].contiguous()
        self.g0 = g0[perm].contiguous()

    def to_idx(self, p):
        return p @ self.A.T + self.c

    def box_hit(self, o, d):
        """Slab test against [0, N-1] in index space. o, d: (N, 3) world. returns tnear, tfar."""
        oi = o @ self.A.T + self.c
        di = d @ self.A.T
        inv = 1.0 / torch.where(di.abs() < 1e-9, torch.full_like(di, 1e-9), di)
        t0 = (0 - oi) * inv
        t1 = (self.shape - 1 - oi) * inv
        tn = torch.minimum(t0, t1).amax(-1)
        tf = torch.maximum(t0, t1).amin(-1)
        return tn.clamp_min(0), tf

    def sample(self, p):
        """p: (..., 3) world mm -> (..., C)."""
        sh = p.shape[:-1]
        g = (p.reshape(-1, 3) @ self.G.T + self.g0).to(self.data.dtype)
        out = F.grid_sample(self.data, g.view(1, 1, 1, -1, 3), mode="bilinear",
                            padding_mode="zeros", align_corners=True)
        return out.view(self.data.shape[1], -1).T.reshape(*sh, self.data.shape[1]).float()


def look_at(eye, target, up=(0, 1, 0)):
    eye = torch.as_tensor(eye, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    up = torch.as_tensor(up, dtype=torch.float32)
    f = normalize(target - eye)
    r = normalize(torch.linalg.cross(f, up))
    u = torch.linalg.cross(r, f)
    return eye, f, r, u


def camera_rays(eye, target, fov_deg, W, H, device, up=(0, 1, 0), ss=1):
    """Perspective pinhole. Returns origins (N,3), dirs (N,3), basis (f, r, u). N = (H*ss)*(W*ss),
    row-major from the top-left of the supersampled image."""
    eye, f, r, u = look_at(eye, target, up)
    Wp, Hp = W * ss, H * ss
    tan = math.tan(math.radians(fov_deg) / 2)
    aspect = W / H
    xs = ((torch.arange(Wp, device=device) + 0.5) / Wp * 2 - 1) * tan * aspect
    ys = (1 - (torch.arange(Hp, device=device) + 0.5) / Hp * 2) * tan
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    f, r, u, eye = f.to(device), r.to(device), u.to(device), eye.to(device)
    d = f + xx[..., None] * r + yy[..., None] * u
    d = normalize(d.reshape(-1, 3))
    o = eye.expand_as(d)
    return o, d, (f, r, u)


def downsample_premult(rgb_pm, alpha, H, W, ss):
    """Box-filter supersampled premultiplied colour + alpha to (H, W)."""
    x = torch.cat([rgb_pm, alpha[:, None]], 1).view(H * ss, W * ss, 4).permute(2, 0, 1)[None]
    x = F.avg_pool2d(x, ss)[0].permute(1, 2, 0)
    return x


def to_rgba8(x, gamma=None):
    """x: (H, W, 4) premultiplied -> un-premultiplied uint8 RGBA numpy."""
    a = x[..., 3:4].clamp(0, 1)
    rgb = torch.where(a > 1e-4, x[..., :3] / a.clamp_min(1e-4), torch.zeros_like(x[..., :3]))
    rgb = rgb.clamp(0, 1)
    out = torch.cat([rgb, a], -1)
    return (out * 255 + 0.5).clamp(0, 255).byte().cpu().numpy()


def hex2rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
