"""RENDER 1: turntable of the fetal ganglionic eminence inside a glass brain envelope.

Surface ray-marcher: fields are smooth (0..1); the 0.5 iso-crossings along each ray become
compositing events. Brain envelope crossings (front and back) -> translucent fresnel glass.
First GE entry -> opaque lit orange surface. Front-to-back premultiplied compositing.
"""
import os, sys, json, math, time, argparse
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "8"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rm  # noqa: E402

dev = torch.device("cuda")
torch.set_num_threads(8)

GE_RGB = rm.hex2rgb("#c2410c")
GLASS_RGB = rm.hex2rgb("#9a8f86")


def build_frame(theta_deg, ant_sign, lat_sign):
    """Anatomical frame in physical (i, j, k): superior in the section (i, j) plane at angle theta
    from +i; anterior perpendicular in-plane; lateral = cross(sup, ant) (proper rotation)."""
    th = math.radians(theta_deg)
    S = np.array([math.cos(th), math.sin(th), 0.0])
    A = ant_sign * np.array([-math.sin(th), math.cos(th), 0.0])
    L = np.cross(S, A)
    # world X = lateral, Y = superior, Z = anterior; X = Y x Z holds by construction
    if lat_sign < 0:  # spin the brain 180 deg about the vertical axis instead of mirroring
        L, A = -L, -A
    return np.stack([L, S, A], 1)


def load(brain, theta, ant_sign, lat_sign):
    z = np.load(os.path.join(HERE, "work", f"ge_fields_{brain}.npz"))
    sp = tuple(float(s) for s in z["spacing"])
    data = torch.from_numpy(np.stack([z["brain"], z["ge"], z["ge_ao"]]).astype(np.float32)).to(dev)
    # centre on the envelope bounding box (in index space)
    b = data[0] > 0.5
    nz = torch.nonzero(b).float()
    frame = build_frame(theta, ant_sign, lat_sign)
    # world coords of envelope voxels relative to an provisional centre -> world bbox centre
    c0 = nz.mean(0)
    sub = nz[:: max(1, nz.shape[0] // 400000)]
    phys = (sub - c0) * torch.tensor(sp, device=dev)
    Mt = torch.tensor(frame, dtype=torch.float32, device=dev)
    w = phys @ Mt  # world = M^T phys
    wc = (w.amax(0) + w.amin(0)) / 2
    centre_idx = c0 + (Mt @ wc) / torch.tensor(sp, device=dev)
    vol = rm.Volume(data, sp, frame, centre_idx.cpu().numpy())
    pts = w - wc  # envelope points in world, centred
    return vol, pts, sp


def normals(vol, p, ch, h):
    ex = torch.eye(3, device=p.device) * h
    q = torch.cat([p + ex[0], p - ex[0], p + ex[1], p - ex[1], p + ex[2], p - ex[2]], 0)
    s = vol.sample(q)[:, ch].view(6, -1)
    g = torch.stack([s[0] - s[1], s[2] - s[3], s[4] - s[5]], -1)
    return rm.normalize(-g)  # outward (field is 1 inside)


def render(vol, eye, target, fov, W, H, ss, dt, K=48, batch=600000, look=None, seed=0):
    look = look or {}
    o, d, (f, r, u) = rm.camera_rays(eye, target, fov, W, H, dev, ss=ss)
    tn, tf = vol.box_hit(o, d)
    N = o.shape[0]
    Cacc = torch.zeros(N, 3, device=dev)
    Tacc = torch.ones(N, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    # lights (camera-attached): key upper-left-front, fill right, back rim
    V0 = -f
    Lk = rm.normalize(-0.55 * r + 0.75 * u + 0.55 * V0)
    Lf = rm.normalize(0.75 * r - 0.15 * u + 0.45 * V0)
    ge_rgb = torch.tensor(GE_RGB, device=dev)
    gl_rgb = torch.tensor(GLASS_RGB, device=dev)
    h_gl = look.get("h_glass", 0.7)
    h_ge = look.get("h_ge", 0.14)
    a0, a1, ap = look.get("glass_a0", 0.08), look.get("glass_a1", 0.62), look.get("glass_p", 2.4)
    back = look.get("glass_back", 0.5)
    valid = torch.nonzero(tf > tn).squeeze(1)
    for bs in torch.split(valid, batch):
        ob, db, tfb = o[bs], d[bs], tf[bs]
        t = tn[bs] + torch.rand(bs.numel(), device=dev, generator=gen) * dt
        prev = vol.sample(ob + db * t[:, None])[:, :2]
        n = bs.numel()
        Cb = torch.zeros(n, 3, device=dev)
        Tb = torch.ones(n, device=dev)
        act = torch.arange(n, device=dev)
        steps = torch.arange(1, K + 1, device=dev, dtype=torch.float32) * dt
        while act.numel() > 0:
            oa, da, ta = ob[act], db[act], t[act]
            ts = ta[:, None] + steps[None]
            P = oa[:, None] + da[:, None] * ts[..., None]
            v = vol.sample(P)[..., :2]
            vv = torch.cat([prev[act][:, None], v], 1)
            b0, b1 = vv[:, :-1, 0], vv[:, 1:, 0]
            g0, g1 = vv[:, :-1, 1], vv[:, 1:, 1]
            inr = ts <= tfb[act][:, None]
            eb = ((b0 - 0.5) * (b1 - 0.5) < 0) & inr
            eg = (g0 < 0.5) & (g1 >= 0.5) & inr
            tb = ts - dt + dt * ((0.5 - b0) / (b1 - b0 + 1e-8)).clamp(0, 1)
            tg = ts - dt + dt * ((0.5 - g0) / (g1 - g0 + 1e-8)).clamp(0, 1)
            eb = eb & ~(eg & (tg < tb))
            m = act.numel()
            alpha = torch.zeros(m, K, 2, device=dev)
            col = torch.zeros(m, K, 2, 3, device=dev)
            # ---- glass events
            ri, si = torch.nonzero(eb, as_tuple=True)
            if ri.numel():
                dd = da[ri]
                pb = oa[ri] + dd * tb[ri, si][:, None]
                nb = normals(vol, pb, 0, h_gl)
                V = -dd
                ndv = (nb * V).sum(-1)
                front = ndv > 0
                nf = torch.where(front[:, None], nb, -nb)
                cosv = ndv.abs().clamp(0, 1)
                fres = (1 - cosv) ** ap
                a = a0 + a1 * fres
                a = torch.where(front, a, a * back)
                dif = ((nf @ Lk) * 0.5 + 0.5) ** 1.5
                Hk = rm.normalize(Lk[None] + V)
                spec = ((nf * Hk).sum(-1).clamp(0, 1) ** 70) * torch.where(front, 1.0, 0.35)
                c = gl_rgb[None] * (0.56 + 0.44 * dif[:, None])
                c = c + (1 - c) * (0.55 * spec)[:, None]
                a = (a + 0.35 * spec).clamp(0, 1)
                alpha[ri, si, 0] = a
                col[ri, si, 0] = c.clamp(0, 1)
            # ---- GE events
            ri, si = torch.nonzero(eg, as_tuple=True)
            if ri.numel():
                dd = da[ri]
                pg = oa[ri] + dd * tg[ri, si][:, None]
                ng = normals(vol, pg, 1, h_ge)
                V = -dd
                ndv = (ng * V).sum(-1).clamp(0, 1)
                wrap = 0.25
                dk = (((ng @ Lk) + wrap) / (1 + wrap)).clamp(0, 1)
                dfl = (ng @ Lf).clamp(0, 1)
                Hk = rm.normalize(Lk[None] + V)
                spec = (ng * Hk).sum(-1).clamp(0, 1) ** 36
                aoc = vol.sample(pg)[:, 2]
                ao = (1.32 - 1.0 * aoc).clamp(0.58, 1.0)
                rim = (1 - ndv) ** 2.5
                c = ge_rgb[None] * (0.30 + 0.80 * dk[:, None] + 0.20 * dfl[:, None]) * ao[:, None]
                c = c + torch.tensor([1.0, 0.86, 0.72], device=dev)[None] * (0.32 * spec * ao)[:, None]
                c = c + torch.tensor([1.0, 0.62, 0.40], device=dev)[None] * (0.10 * rim)[:, None]
                alpha[ri, si, 1] = 1.0
                col[ri, si, 1] = c.clamp(0, 1)
            a = alpha.view(m, 2 * K)
            c = col.view(m, 2 * K, 3)
            Tc = torch.cumprod(1 - a, 1)
            Tex = torch.cat([torch.ones(m, 1, device=dev), Tc[:, :-1]], 1)
            Cb[act] += Tb[act][:, None] * (Tex[..., None] * a[..., None] * c).sum(1)
            Tb[act] = Tb[act] * Tc[:, -1]
            prev[act] = v[:, -1]
            t[act] = ta + K * dt
            keep = (Tb[act] > 2e-3) & (t[act] < tfb[act])
            act = act[keep]
        Cacc[bs] = Cb
        Tacc[bs] = Tb
    img = rm.downsample_premult(Cacc, 1 - Tacc, H, W, ss)
    return img


def fit_distance(pts, target, fov, az_list, el, fill=0.86):
    """Smallest camera distance so the envelope stays inside `fill` of the frame for all azimuths."""
    tan = math.tan(math.radians(fov) / 2)
    p = pts.cpu()
    lo, hi = 1.0, 2000.0
    for _ in range(40):
        dist = (lo + hi) / 2
        ok = True
        for az in az_list:
            eye = orbit(target, dist, az, el)
            e, f, r, u = rm.look_at(eye, target)
            q = p - e
            z = q @ f
            x = (q @ r) / (z * tan)
            y = (q @ u) / (z * tan)
            if (z.min() <= 0) or x.abs().max() > fill or y.abs().max() > fill:
                ok = False
                break
        if ok:
            hi = dist
        else:
            lo = dist
    return hi


def orbit(target, dist, az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    t = torch.as_tensor(target, dtype=torch.float32)
    return t + dist * torch.tensor([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])


vol_spacing = (0.12, 0.12, 0.12)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="142")
    ap.add_argument("--theta", type=float, default=90.0)
    ap.add_argument("--ant", type=int, default=-1)
    ap.add_argument("--lat", type=int, default=1)
    ap.add_argument("--mode", default="views")
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--az0", type=float, default=40.0)
    ap.add_argument("--el", type=float, default=15.0)
    ap.add_argument("--fov", type=float, default=28.0)
    ap.add_argument("--out", default=os.path.join(HERE, "test"))
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    vol, pts, sp = load(args.brain, args.theta, args.ant, args.lat)
    vol_spacing = sp
    target = (0.0, 0.0, 0.0)
    dt = 0.35 * min(sp)
    if args.mode == "views":
        # lateral (from +X), front (+Z), top, and a 3/4 view, small size for orientation checks
        views = {"lat+X": (90, 0), "lat-X": (-90, 0), "front+Z": (0, 0), "back-Z": (180, 0),
                 "top": (0, 89), "q34": (args.az0, args.el)}
        tiles = []
        for name, (az, el) in views.items():
            dist = fit_distance(pts, target, args.fov, [az], el)
            eye = orbit(target, dist, az, el)
            t0 = time.time()
            img = render(vol, eye, target, args.fov, args.size, args.size, args.ss, dt)
            torch.cuda.synchronize()
            print(name, "%.2fs" % (time.time() - t0), flush=True)
            a = rm.to_rgba8(img)
            im = Image.fromarray(a, "RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            bg.alpha_composite(im)
            tiles.append(bg.convert("RGB"))
        Wt = args.size
        sheet = Image.new("RGB", (Wt * 3, Wt * 2), "white")
        for i, tl in enumerate(tiles):
            sheet.paste(tl, ((i % 3) * Wt, (i // 3) * Wt))
        sheet.save(os.path.join(args.out, f"views_{args.brain}_th{int(args.theta)}_a{args.ant}.png"))
    elif args.mode == "turntable":
        n = args.frames
        azs = [args.az0 + 360.0 * i / n for i in range(n)]
        dist = fit_distance(pts, target, args.fov, azs, args.el)
        print("camera distance %.1f mm" % dist)
        fr_dir = os.path.join(args.out, "frames")
        os.makedirs(fr_dir, exist_ok=True)
        idx = range(n) if not args.only else [int(s) for s in args.only.split(",")]
        for i in idx:
            t0 = time.time()
            eye = orbit(target, dist, azs[i], args.el)
            img = render(vol, eye, target, args.fov, args.size, args.size, args.ss, dt, seed=i)
            a = rm.to_rgba8(img)
            Image.fromarray(a, "RGBA").save(os.path.join(fr_dir, f"frame_{i:03d}.png"))
            print("frame", i, "%.1fs" % (time.time() - t0), flush=True)
