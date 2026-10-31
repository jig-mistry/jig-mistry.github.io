"""RENDER 2: FB74 serial-section Nissl stack, cut-away corner, continuous zoom into the cut face.

Volume ray-marcher on the full-resolution float16 stack (exact trilinear sampling with float32 index
maths), empty-space skipping on a dilated 4x occupancy grid, front-to-back compositing of a
density->violet colour ramp, deferred Lambert shading at the first opaque depth (cut faces use the
plane normal, the outer surface and ventricle walls use the gradient of a smoothed tissue mask).
"""
import os, sys, json, math, time, argparse
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "8"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rm  # noqa: E402
import stack_common as SC  # noqa: E402

dev = torch.device("cuda")
torch.set_num_threads(8)

# ---- scene constants (world mm; X lateral, Y superior, Z anterior; origin = crop-box centre)
L0 = 3.0          # sagittal cut plane X = L0 (parallel to the sections)
A0 = 5.5          # coronal cut plane Z = A0 (perpendicular to the sections)
TARGET1 = (6.0, 4.5, A0)   # zoom target on the coronal face: GE / striatum boundary
FOV = 24.0
W, H = 1200, 800
NFR = 48
LIGHT = rm.normalize(torch.tensor([0.35, 0.80, 0.50]))
SIGMA = 50.0      # extinction per mm for tissue (opaque within ~50 um)


def wedge_sd(p):
    """>0 inside the removed corner {X > L0 and Z > A0}."""
    return torch.minimum(p[..., 0] - L0, p[..., 2] - A0)


def render(S, eye, target, fov, Wd, Hd, ss, dt, seed=0, batch=1_000_000, Kc=24, Kf=12):
    o, d, (f, r, u) = rm.camera_rays(eye, target, fov, Wd, Hd, dev, ss=ss)
    tn, tf = S.box_hit(o, d)
    N = o.shape[0]
    Cout = torch.zeros(N, 3, device=dev)
    Aout = torch.zeros(N, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    dtc = 0.1
    Lw = LIGHT.to(dev)
    wcut = 0.5 * dt
    valid = torch.nonzero(tf > tn).squeeze(1)
    for bs in torch.split(valid, batch):
        n = bs.numel()
        ob, db, tfb = o[bs], d[bs], tf[bs]
        t = tn[bs] + torch.rand(n, device=dev, generator=gen) * dtc
        T = torch.ones(n, device=dev)
        C = torch.zeros(n, 3, device=dev)
        ts_hit = torch.full((n,), -1.0, device=dev)
        mode = torch.zeros(n, dtype=torch.bool, device=dev)  # False = coarse, True = fine
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        it = 0
        while alive.any():
            it += 1
            # ---------------- coarse phase
            ca = torch.nonzero(alive & ~mode).squeeze(1)
            if ca.numel():
                stp = torch.arange(Kc, device=dev, dtype=torch.float32) * dtc
                tt = t[ca][:, None] + stp[None]
                P = ob[ca][:, None] + db[ca][:, None] * tt[..., None]
                occ = S.sample_small(S.occ4, 4, P)
                occ = occ * (wedge_sd(P) < 0.3)
                occ = occ * (tt <= tfb[ca][:, None])
                hitm = occ > 0.01
                anyh = hitm.any(1)
                first = torch.argmax(hitm.float(), 1)
                th = tt.gather(1, first[:, None]).squeeze(1)
                hi = ca[anyh]
                t[hi] = torch.maximum(th[anyh] - 0.25, t[hi])  # never step backwards
                mode[hi] = True
                mi = ca[~anyh]
                t[mi] = t[mi] + Kc * dtc
                alive[mi[t[mi] > tfb[mi]]] = False
            # ---------------- fine phase
            fa = torch.nonzero(alive & mode).squeeze(1)
            if fa.numel():
                stp = torch.arange(Kf, device=dev, dtype=torch.float32) * dt
                tt = t[fa][:, None] + stp[None]
                P = ob[fa][:, None] + db[fa][:, None] * tt[..., None]
                raw, occ = S.sample_full(P, with_occ=True)
                dens = S.density(raw)
                keep = (0.5 - wedge_sd(P) / wcut).clamp(0, 1)
                inr = (tt <= tfb[fa][:, None]).float()
                a = 1 - torch.exp(-SIGMA * dt * (0.3 + 0.7 * dens) * occ * keep * inr)
                col = SC.ramp(dens)
                Tc = torch.cumprod(1 - a, 1)
                Tex = torch.cat([torch.ones(fa.numel(), 1, device=dev), Tc[:, :-1]], 1)
                Tf = T[fa]
                C[fa] += Tf[:, None] * (Tex[..., None] * a[..., None] * col).sum(1)
                # first depth where accumulated opacity reaches 0.5
                acc = 1 - Tf[:, None] * Tc
                need = ts_hit[fa] < 0
                crossed = acc >= 0.25
                has = crossed.any(1) & need
                fi = torch.argmax(crossed.float(), 1)
                tsel = tt.gather(1, fi[:, None]).squeeze(1)
                ts_hit[fa[has]] = tsel[has]
                T[fa] = Tf * Tc[:, -1]
                t[fa] = t[fa] + Kf * dt
                done = (T[fa] < 4e-3) | (t[fa] > tfb[fa])
                alive[fa[done]] = False
                # back to coarse skipping if we are in empty space
                still = fa[~done]
                if still.numel():
                    pe = ob[still] + db[still] * t[still][:, None]
                    oc = S.sample_small(S.occ4, 4, pe) * (wedge_sd(pe) < 0.3)
                    back = still[oc <= 0.01]
                    mode[back] = False
        # ---------------- deferred shading
        alpha = 1 - T
        shade = torch.ones(n, device=dev)
        spec = torch.zeros(n, device=dev)
        hs = torch.nonzero(ts_hit >= 0).squeeze(1)
        if hs.numel():
            p = ob[hs] + db[hs] * ts_hit[hs][:, None]
            V = -db[hs]
            eps = 2.5 * dt + 0.01
            on_cor = ((p[:, 2] - A0).abs() < eps) & (p[:, 0] > L0 - eps)
            on_sag = ((p[:, 0] - L0).abs() < eps) & (p[:, 2] > A0 - eps) & ~on_cor
            # mask-gradient normal (ds2 smoothed mask), outward
            h = 0.09
            ex = torch.eye(3, device=dev) * h
            q = torch.cat([p + ex[0], p - ex[0], p + ex[1], p - ex[1], p + ex[2], p - ex[2]], 0)
            sv = S.sample_small(S.mask2, 2, q).view(6, -1)
            g = torch.stack([sv[0] - sv[1], sv[2] - sv[3], sv[4] - sv[5]], -1)
            gn = g.norm(dim=-1)
            nrm = -g / gn.clamp_min(1e-6)[:, None]
            weak = gn < 0.02
            nrm = torch.where(weak[:, None], V, nrm)
            # make normals face the viewer (ventricle walls seen from inside etc.)
            flip = (nrm * V).sum(-1) < 0
            nrm = torch.where(flip[:, None], -nrm, nrm)
            ndl = (nrm @ Lw).clamp(0, 1)
            sh = 0.50 + 0.58 * ndl
            Hh = rm.normalize(Lw[None] + V)
            sp = ((nrm * Hh).sum(-1).clamp(0, 1) ** 28) * 0.10
            sh = torch.where(on_cor, torch.full_like(sh, 1.0), sh)
            sh = torch.where(on_sag, torch.full_like(sh, 0.90), sh)
            sp = torch.where(on_cor | on_sag, torch.zeros_like(sp), sp)
            shade[hs] = sh
            spec[hs] = sp
        Cpm = C * shade[:, None] + (spec * alpha)[:, None]
        Cout[bs] = Cpm.clamp(0, 1)
        Aout[bs] = alpha
    return rm.downsample_premult(Cout, Aout, Hd, Wd, ss)


def smootherstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def dir_from(az, el):
    az, el = math.radians(az), math.radians(el)
    return torch.tensor([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])


def slerp(a, b, w):
    a, b = rm.normalize(a), rm.normalize(b)
    om = math.acos(float((a * b).sum().clamp(-1, 1)))
    if om < 1e-6:
        return a
    return (math.sin((1 - w) * om) * a + math.sin(w * om) * b) / math.sin(om)


def camera_path(c0, d0, d1, az0, el0, az1, el1):
    """Per-frame (eye, target, dist). Frames 0-10 hold the overview; 10-47 ease-in-out exponential zoom."""
    out = []
    c0 = torch.tensor(c0, dtype=torch.float32)
    p1 = torch.tensor(TARGET1, dtype=torch.float32)
    D1 = dir_from(az1, el1)
    for i in range(NFR):
        s = 0.0 if i <= 10 else smoothstep((i - 10) / (NFR - 1 - 10))  # ease-in-out on log-distance
        if i <= 10:  # gentle 8-degree drift into the overview pose, ending at rest on frame 10
            D0 = dir_from(az0 - 8.0 * (1 - smootherstep(i / 10)), el0)
        else:
            D0 = dir_from(az0, el0)
        dist = d0 * (d1 / d0) ** s
        w = (d0 - dist) / (d0 - d1)
        tgt = c0 + (p1 - c0) * w
        dr = slerp(D0, D1, w)
        out.append((tgt + dist * dr, tgt, dist))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(HERE, "stack_frames_png"))
    ap.add_argument("--d0", type=float, default=100.0)
    ap.add_argument("--az0", type=float, default=35.0)
    ap.add_argument("--el0", type=float, default=22.0)
    ap.add_argument("--az1", type=float, default=18.0)
    ap.add_argument("--el1", type=float, default=8.0)
    args = ap.parse_args()
    AR = args
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    S = SC.Stack()
    print("loaded stack", tuple(S.vol.shape), "%.1fs" % (time.time() - t0), flush=True)
    C0 = (0.0, 3.0, 0.5)
    path = camera_path(C0, d0=AR.d0, d1=7.2, az0=AR.az0, el0=AR.el0, az1=AR.az1, el1=AR.el1)
    Wd, Hd = int(W * args.scale), int(H * args.scale)
    frames = range(NFR) if not args.only else [int(x) for x in args.only.split(",")]
    for i in frames:
        eye, tgt, dist = path[i]
        pix = 2 * dist * math.tan(math.radians(FOV / 2)) / Hd
        dt = float(min(max(0.6 * pix, 0.004), 0.016))
        t1 = time.time()
        img = render(S, eye, tgt, FOV, Wd, Hd, args.ss, dt, seed=i)
        a = rm.to_rgba8(img)
        Image.fromarray(a, "RGBA").save(os.path.join(args.out, f"frame_{i:03d}.png"))
        print("frame %d dist %.1f mm  view height %.2f mm  dt %.4f  %.1fs" % (
            i, dist, pix * Hd, dt, time.time() - t1), flush=True)
