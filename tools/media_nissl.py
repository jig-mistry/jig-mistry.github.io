"""Nissl zoom levels (24 GW, DHARANI FB85) for the GE chapter; cortical wall crops (22 GW, FB62); three-colour IHC crop."""
import os, sys, json, subprocess, tempfile
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, tifffile, zarr
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
SITE = "/home/users/jignesh/workspace/jig-mistry.github.io"
OPJ = "/home/users/jignesh/miniconda3/envs/nisslprop/bin/opj_decompress"
TMP = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(__file__)))

def jp2(path, r, box):
    out = os.path.join(TMP, "c.png")
    subprocess.run([OPJ, "-i", path, "-o", out, "-r", str(r), "-d", ",".join(map(str, box)), "-threads", "8"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return Image.open(out).convert("RGB")

def save(im, path, size=None, q=80):
    if size: im.thumbnail((size, size), Image.Resampling.LANCZOS)
    im.save(path, quality=q, method=6)
    print(os.path.basename(path), im.size, os.path.getsize(path) // 1024, "KB", flush=True)
    return im.size

# ---- FB85 (24 GW), sagittal, SE 1138: whole section -> GE -> cells ----
J85 = "/home/users/jignesh/analytics/244/NISL/B_244_FB85-SL_380-ST_NISL-SE_1138_compressed.jp2"
W85, H85 = 144472, 129351
cx, cy = 36100, 54300
boxes = [("nissl_section", 5, (0, 0, W85, H85), 2400),
         ("nissl_ge", 3, (cx - 10000, cy - 10000, cx + 10000, cy + 10000), 2000),
         ("nissl_ge_mid", 1, (cx - 3000, cy - 3000, cx + 3000, cy + 3000), 1800),
         ("nissl_cells", 0, (cx - 800, cy - 800, cx + 800, cy + 800), 1600)]
levels = []
for name, r, b, size in boxes:
    im = jp2(J85, r, b)
    sz = save(im, f"{SITE}/assets/img/ge/{name}.webp", size)
    levels.append(dict(src=f"assets/img/ge/{name}.webp", rect=[b[0] / W85, b[1] / H85, b[2] / W85, b[3] / H85], px=list(sz)))
json.dump(dict(aspect=W85 / H85, levels=levels), open(f"{SITE}/assets/img/ge/nissl_zoom.json", "w"), indent=1)

# ---- FB62 (22 GW), coronal, SE 1189: the whole cortical wall, pia (left) to ventricle (right) ----
J62 = "/home/users/jignesh/analytics/213/NISL/B_213_FB62-SL_397-ST_NISL-SE_1189_compressed.jp2"
strip = jp2(J62, 2, (14133, 2800, 17133, 18800)).rotate(90, expand=True)
strip = strip.crop((int(strip.width * 0.12), 0, strip.width, strip.height))      # drop most of the empty glass
save(strip, f"{SITE}/assets/img/cyto/wall_strip.webp", 3200)
save(jp2(J62, 0, (14933, 5900, 16333, 7300)).rotate(90, expand=True), f"{SITE}/assets/img/cyto/cp_cells.webp", 1400)

# ---- three-colour immunofluorescence-style panel (markers deliberately not named) ----
P1 = "/home/users/jignesh/neuroglancer/ihc7_panels_registered/panel1_NeuN-green_CTIP2-red_Reelin-blue_0p5um.tif"
t = tifffile.TiffFile(P1)
z = zarr.open(t.aszarr(level=2), mode="r")
a = np.asarray(z[6800:8800, 15600:17200])
save(Image.fromarray(a), f"{SITE}/assets/img/cyto/ihc_layers.webp", 1600, q=82)
z5 = zarr.open(t.aszarr(level=5), mode="r")
save(Image.fromarray(np.asarray(z5[:])), f"{SITE}/assets/img/cyto/ihc_section.webp", 1600, q=80)
