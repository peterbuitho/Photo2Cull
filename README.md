# Photo2Cull

A desktop app for quickly culling a folder of photos down to your keepers -- point it at a folder, it scores every photo, and you decide what survives.

![Photo2Cull screenshot](docs/screenshot.png)

## What it does

- **Scans one or more folders** (subfolders included) of RAW (via `rawler`) or standard photos (PNG, JPG, TIFF, BMP, WebP) and scores each one for focus sharpness (variance-of-Laplacian).
- **Classifies each photo** as Landscape, Portrait, Animal, or Object, either automatically (ML face detection decides Portrait, ML animal detection -- birds, cats, dogs, horses, cows and similar COCO animals -- decides Animal, and a heuristic handles the rest) or manually per photo, and rescoring adapts to the region that matters for that mode (e.g. the detected face or animal).
- **Offers two ways to measure sharpness** when no face/animal is found: a fixed crop per photo type, or the sharpest tiles anywhere in the frame (better for macro and off-centre subjects).
- **Zooms on hover**: rest the pointer on a thumbnail and a circular magnifier lens follows it, showing that part of the photo at a set zoom (Hover zoom pane: zoom percent, delay in seconds, and lens size). Double-click still opens the full-size viewer.
- **Ranks photos with an Overall score** blending six factors -- sharpness, exposure, contrast, color, composition (rule-of-thirds), and subject prominence (for portraits and animals) -- with adjustable weights that re-rank instantly.
- **Groups duplicates and bursts** using perceptual hashing, and flags the best-of-group pick by Overall score.
- **Runs a full pipeline**: technical sharpness cull -> dedupe -> rank by Overall -> top-N shortlist, in one click.
- **Lets you override disqualification by hand**: photos at or below the threshold are marked automatically after a scan, and each photo has a "Disqualified" checkbox to mark or unmark it manually (manual choices survive threshold changes and count in the move and the ranking pipeline).
- **Selects like Explorer**: click a thumbnail to select it, Ctrl+click to toggle, Shift+click for a range, Ctrl+A for everything shown, Esc to deselect. The Disqualified checkbox (or the Disqualify/Keep buttons) then applies to the whole selection.
- **Moves disqualified photos** out of the way (into a `disqualified` subfolder of whichever chosen folder each photo came from, keeping its subfolder structure) instead of deleting them, so culling stays reversible.

## Getting started

Download a build from the [Releases](https://github.com/peterbuitho/Photo2Cull/releases) page (Windows, macOS Apple Silicon, or Linux x86_64), or build from source:

```
cargo run --release
```

Add one or more folders, click Scan, then sort/filter/group as needed.

## Hardware notes

Scanning is CPU-parallelized (more cores = faster) and every photo's thumbnail stays resident as a GPU texture for the session. As a rough guide for a few thousand photos: 4+ cores, 8GB RAM, an SSD (RAW files mean a lot of reading), and any GPU from the last decade (no discrete GPU required -- expect roughly 200KB of VRAM per photo for thumbnails). Lower-spec machines will work, just more slowly, and very low RAM may see swapping on large RAW batches.
