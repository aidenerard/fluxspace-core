# Explanation of `visualize_3d_heatmap.py`

This document explains the 3D pipeline script that **visualizes** the voxel volume (`volume.npz`) with PyVista: volume rendering, orthogonal slices, isosurfaces, and optional mesh overlay. It can save a screenshot PNG and optionally run an interactive viewer.

---

## Overview

**`visualize_3d_heatmap.py`** loads `volume.npz` (from **`mag_world_to_voxel_volume`**), builds a PyVista `ImageData` grid, and adds:

- **Volume rendering** with a linear opacity transfer and `coolwarm` colormap.
- **Orthogonal slice** through the grid center (axis chosen by `--slice-axis`).
- **Isosurface** at the volume median (or `--isosurface-value`).
- **Optional mesh overlay** (PLY/OBJ/GLB) for scene context.

Outputs go to `--out-dir` (default: volume’s parent): e.g. **`heatmap_3d_screenshot.png`** when `--screenshot` is used.

**Requires:** `pyvista`. Install with `pip install pyvista` (or use `./tools/3d/setup_pi.sh` on the Pi).

---

## What it does

1. **Load** `volume.npz`: read `volume` (3D array), `origin`, `voxel_size`; expand scalar `origin` to `[x,y,z]` if needed.
2. **Build PyVista grid:** `pv.ImageData` with dimensions, spacing, origin; attach `value` as point data (Fortran‑order ravel for PyVista).
3. **Volume rendering:** `add_volume` with `opacity='linear'`, `cmap='coolwarm'`, scalar range from data min/max. If all values are equal, add a simple mesh instead.
4. **Slice:** `grid.slice` with normal `x` / `y` / `z` (via `--slice-axis`) through grid center; add to plotter with `coolwarm` and opacity 0.7.
5. **Isosurface:** `grid.contour` at median or `--isosurface-value`; add orange mesh at 0.5 opacity.
6. **Mesh overlay:** If `--mesh` is set, `pv.read` the file and add as light gray, opacity 0.3. Ignore if load fails.
7. **Camera / output:** Set `camera_position = "yz"`, reset camera. If `--screenshot`, save **`heatmap_3d_screenshot.png`** in `--out-dir`. If not `--no-show` and no screenshot, call `pl.show()`.

---

## CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--in` | (required) | Path to `volume.npz`. Use `--volume` as alias. |
| `--out-dir` | (volume’s parent) | Directory for screenshot (and optional HTML). |
| `--mesh` | — | Optional mesh overlay (PLY/OBJ/GLB). |
| `--screenshot` | — | Save `heatmap_3d_screenshot.png` in `--out-dir`. |
| `--no-show` | — | Do not open interactive window (use with `--screenshot` for headless). |
| `--slice-axis` | `z` | Primary slice axis: `x`, `y`, or `z`. |
| `--isosurface-value` | (median) | Isosurface threshold; default = median of volume. |

---

## Example usage

```bash
# Interactive view + screenshot
python3 pipelines/3d/visualize_3d_heatmap.py \
  --in "$RUN_DIR/exports/volume.npz" \
  --out-dir "$RUN_DIR/exports" \
  --screenshot

# Headless screenshot only (e.g. on server/Pi)
python3 pipelines/3d/visualize_3d_heatmap.py \
  --in "$RUN_DIR/exports/volume.npz" \
  --out-dir "$RUN_DIR/exports" \
  --screenshot \
  --no-show

# Slices only (PNGs, no GUI)
python3 pipelines/3d/visualize_3d_heatmap.py \
  --in "$RUN_DIR/exports/volume.npz" \
  --out-dir "$RUN_DIR/exports" \
  --show-slices --save --no-show

# Custom isosurface, slice along y, optional mesh
python3 pipelines/3d/visualize_3d_heatmap.py \
  --in "$RUN_DIR/exports/volume.npz" \
  --out-dir "$RUN_DIR/exports" \
  --mesh "$RUN_DIR/raw/mesh.ply" \
  --slice-axis y \
  --isosurface-value 0.5 \
  --screenshot
```

---

## Relation to other 3D scripts

- **Input** `volume.npz` is produced by **`mag_world_to_voxel_volume`**.
- **Upstream:** **`fuse_mag_with_trajectory`** → **`mag_world_to_voxel_volume`** → **`visualize_3d_heatmap`**.

See [PIPELINE_3D.md](PIPELINE_3D.md) for the full 3D runbook.
