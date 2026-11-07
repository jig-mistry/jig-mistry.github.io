"""The deepest Nissl level from the lossless scan of the section, at the scan's own 0.5 um per pixel.

media_nissl.py took every level from the _compressed.jp2 (2.3 GB). That is fine for the wider levels, but the deepest
one is shown up to 1000x and the compression had removed the edges of the nuclei. Same box as there, tissue alpha
kept from the current nissl_cells.webp (media_alpha.py). Two files: nissl_cells.webp stays in colour for the plain
page; nissl_cells_r.webp is one grey channel, lossless (WebP keeps colour at half resolution), for the stage.

The stage reads red, where a violet stain has its contrast, but red and green are clipped to 0 in about 10% of the
tissue, the inside of the nuclei. Blue never clips and holds the chromatin. So the grey is red, and below T (where the
stage's section-level curve is already flat, so nothing changes at low zoom) it is blue, rescaled into 0..T.
CELL is the target of the dive: a separate, compact nucleus about 9 um across, in pixels of this crop.

usage: python media_nissl_cells.py SITE_DIR [DECODED.png] [QUALITY]
"""
import os, sys, json, subprocess, tempfile
import numpy as np
from PIL import Image
SITE = sys.argv[1]
Q = int(sys.argv[3]) if len(sys.argv) > 3 else 90
J = "/home/users/jignesh/analytics/244/NISL/B_244_FB85-SL_380-ST_NISL-SE_1138_lossless.jp2"
OPJ = "/home/users/jignesh/miniconda3/envs/nisslprop/bin/opj_decompress"
cx, cy, r = 36100, 54300, 800
T = 51
CELL = (799, 439)
png = sys.argv[2] if len(sys.argv) > 2 else os.path.join(tempfile.mkdtemp(), "c.png")
if not os.path.exists(png):
    subprocess.run([OPJ, "-i", J, "-o", png, "-r", "0", "-d", f"{cx - r},{cy - r},{cx + r},{cy + r}", "-threads", "8"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
out = f"{SITE}/assets/img/ge/nissl_cells.webp"
alpha = np.array(Image.open(out).convert("RGBA"))[..., 3]
rgb = np.array(Image.open(png).convert("RGB"))
assert rgb.shape[:2] == alpha.shape
Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(out, quality=Q, method=6, alpha_quality=90)
red = out.replace("nissl_cells", "nissl_cells_r")
r = rgb[..., 0].astype(np.float32); b = rgb[..., 2].astype(np.float32)
g = np.where(r >= T, r, T * np.clip((b - 60) / 110, 0, 1)).astype(np.uint8)
Image.fromarray(np.dstack([g, g, g, alpha]), "RGBA").save(red, lossless=True, method=6)
print(os.path.getsize(out) // 1024, "KB colour,", os.path.getsize(red) // 1024, "KB red", flush=True)
z = f"{SITE}/assets/img/ge/nissl_zoom.json"
spec = json.load(open(z)); spec["levels"][3]["src"] = "assets/img/ge/nissl_cells_r.webp"
spec["cell"] = [(CELL[0] + 0.5) / rgb.shape[1], (CELL[1] + 0.5) / rgb.shape[0]]
json.dump(spec, open(z, "w"), indent=1)
