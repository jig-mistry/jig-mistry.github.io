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
- `assets/dive/fibres.js`: the fibre background of the myelin stop, traced from the published streamline figure.

## Data

- `assets/volumes/`: BigBrain (200 µm release, downsampled to 0.6 mm) and the five DHARANI fetal brains (0.36 mm),
  as gzip'd uint8 pairs (density, depth below the surface). See `brains.json`. `dharani_24gw_mri.bin.gz` is the T1 of
  the 24-week brain on the same grid: the native scan, placed rigidly, no scale or warp (`tools/build_mri24_native.py`).
- `assets/img/`: section images, zoom levels and renders. Immunostained and myelin images are shown without marker names.
- `tools/`: the scripts that made these files (paths are local to the machines they ran on).

Credits: BigBrain (Amunts et al., Science 2013; CC BY-NC-SA 4.0); DHARANI (Verma et al., J Comp Neurol 2025; CC BY 4.0).
