# jig-mistry.github.io

Personal website of Jignesh Mistry. Live at https://jig-mistry.github.io

Edit `index.html` (content) and `assets/style.css` (styling). Put your CV at `assets/cv.pdf` and photo at `assets/photo.jpg`.

## How the page works

- `assets/neuro.js` draws the animated neuronal network in the background.
- `assets/js/brainvol.js` volume-renders whole brains (WebGL2 ray marching) on a fixed canvas behind the page.
  Every element with a `data-scene='{...}'` attribute is a keyframe (volume, camera angle, distance, cut-away,
  clip plane, fade); the scene is interpolated between keyframes as the page scrolls. Add `?still` to the URL to
  stop the idle drift (useful for screenshots).
- `assets/js/story.js` drives the scroll zooms (`.zoomer`, nested image levels described by a JSON file), the frame
  scrubber (`.scrubber`) and the gestational-age ruler.

## Data

- `assets/volumes/`: BigBrain (200 µm release, downsampled to 0.6 mm) and the five DHARANI fetal brains
  (mcmadras stacks, 0.36 mm), as gzip'd uint8 pairs (density, depth below the surface). See `volumes.json`.
- `assets/img/`: section images, zoom levels and renders. Immunostained and myelin images are shown without marker names.
- `tools/`: the scripts that made these files on the lab workstation (paths are local to that machine).

Credits: BigBrain (Amunts et al., Science 2013; CC BY-NC-SA 4.0); DHARANI (Verma et al., J Comp Neurol 2025; CC BY 4.0).
