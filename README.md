# jig-mistry.github.io

Personal website of Jignesh Mistry. Live at https://jig-mistry.github.io

One continuous dive at true scale, from a whole brain at 10 cm to one 0.5 µm pixel of a scan. Edit `index.html` for
the content and `assets/dive/style.css` for the styling.

## How the page works

- `assets/dive/dive.js`: everything on screen is a function of one number, t. Each `.stop` in `index.html` is a stop
  on the ruler (`data-r` is the half field in mm); the keyframes, the blade and the registration timeline are here.
- `assets/dive/stage.js`: WebGL2 ray marching of the volumes, the two-volume bench (histology and MRI) and the
  composite with the flat Nissl levels. `?still` stops the idle motion, `?flat` forces the plain page, which is also
  what visitors without WebGL2 or with reduced motion get.
- `assets/dive/fibres.js`: the fibre background of the myelin stop, traced from the published streamline figure. The
  fibres grow out of the Nissl field and draw back into it, which bridges the cells and the myelin stop.

## Data

- `assets/volumes/`: BigBrain (200 µm release, downsampled to 0.6 mm) and the five DHARANI fetal brains (0.36 mm),
  as gzip'd uint8 pairs (density, depth below the surface). See `brains.json`. `dharani_24gw_mri.bin.gz` is the T1 of
  the 24-week brain on the same grid: the native scan, placed rigidly, no scale or warp (`tools/build_mri24_native.py`).
- `assets/volumes/hist24/`: the 24-week Nissl at 128 µm for the blade's cut face, as WebP sheets of sections, from the
  same 64 µm stack as `dharani_24gw` (`tools/build_hist24_hires.py`). Lost sections are filled and stain is evened out
  against the web volume. Loaded last and only on wide screens; the rest of the bench stays on the 0.36 mm volume.
- `assets/volumes/hb02*`: the stroke brain in its ex vivo MRI space (`tools/build_hb02.py`). `hb02_meta` holds the block
  ids, the lab's segmentation of the lesion on diffusion MRI, the cerebellum, and the lesion's brightness on diffusion MRI.
- `assets/img/hb02/`: the one face the stroke stop looks straight at, in every modality on the same surface: the MRI and the
  diffusion scan (`tools/build_hb02_faces.py`), a whole 60 µm Nissl section, the block-face photo of that section
  (`tools/fit_hb02_blockface.py`), and the neighbouring H&E and immunostained sections with the lab's hand outline of
  neuronal loss (`tools/fit_hb02_stains.py`). `*_s` are the half-size images phones load.
- `assets/img/ge/nissl_cells_r.webp`: the deepest Nissl level, lossless at the scan's 0.5 µm per pixel, from the
  `_lossless.jp2` of the section, with blue carrying the nuclei where red clips (`tools/media_nissl_cells.py`).
- `assets/img/cyto/cortex_zoom.json`, `nissl_cortex*.webp`, `ihc_on_wall.webp`: the cytoarchitecture stop's cortical wall
  on the same Nissl section as the eminence, its zones read off the stain, and another fetal brain's three-colour
  immunostained wall turned onto its pia at its own scale (`tools/media_cortex.py`).
- `assets/img/`: section images, zoom levels and renders. Immunostained and myelin images are shown without marker names.
- `tools/`: the scripts that made these files (paths are local to the machines they ran on).

Credits: BigBrain (Amunts et al., Science 2013; CC BY-NC-SA 4.0); DHARANI (Verma et al., J Comp Neurol 2025; CC BY 4.0); stroke brain from the C-Stroke atlas (Nanjappa et al., preprint; http://c-stroke.humanbrain.in/).
