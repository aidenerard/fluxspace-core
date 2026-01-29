# Explanation of `mag_world_to_voxel_volume.py`

This document explains the 3D pipeline script that builds a **voxel volume** from `mag_world.csv` (world‑frame magnetic samples) and writes `volume.npz` for visualization and analysis.

---

## Overview

**`mag_world_to_voxel_volume.py`** takes scattered `(x, y, z, value)` points from `mag_world.csv`, defines a 3D grid that fits the data (with an optional margin), and **interpolates** values onto that grid. The result is a regular 3D array plus metadata (origin, voxel size, axis arrays) saved as **`volume.npz`**.

**Input:** `mag_world.csv` with columns `x`, `y`, `z`, `value` (e.g. from `fuse_mag_with_trajectory`).

**Output:** `volume.npz` containing `volume` (3D array), `origin`, `voxel_size`, `nx`, `ny`, `nz`, and `axes_x`, `axes_y`, `axes_z`.

---

## What it does

1. **Load** `mag_world.csv`, drop rows with NaN in `x`/`y`/`z`/`value`.
2. **Bounds:** Compute min/max in x, y, z; add `--margin` (meters) on all sides.
3. **Grid:** Construct a regular grid with spacing `--voxel-size`. Grid resolution `(nx, ny, nz)` is chosen so the grid covers the expanded bounds.
4. **Interpolation:**
   - **`--method idw`** (default): k‑nearest inverse‑distance weighting (IDW). Uses `scipy.spatial.cKDTree` for nearest neighbors; `--k` and `--power` tune behavior.
   - **`--method griddata`:** `scipy.interpolate.griddata` with `method='linear'`; NaN fill uses `nanmean` of values.
5. **Save** `volume.npz` (compressed). Downstream **`visualize_3d_heatmap`** reads this for slices, isosurfaces, and screenshots.

---

## CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--in` | (required) | Path to `mag_world.csv`. |
| `--out` | (auto) | Output `volume.npz` path. Default: `<input_dir>/../exports/volume.npz` or `<input_dir>/volume.npz`. |
| `--voxel-size` | 0.02 | Voxel edge length (meters). |
| `--margin` | 0.1 | Margin (meters) added to data bounds. |
| `--method` | `idw` | Interpolation: `idw` or `griddata`. |
| `--k` | 8 | Number of nearest neighbors for IDW. |
| `--power` | 2.0 | IDW power exponent. |

---

## Example usage

```bash
# Typical run
python3 pipelines/3d/mag_world_to_voxel_volume.py \
  --in "$RUN_DIR/processed/mag_world.csv" \
  --out "$RUN_DIR/exports/volume.npz" \
  --voxel-size 0.02 \
  --margin 0.1

# Finer grid, IDW with more neighbors
python3 pipelines/3d/mag_world_to_voxel_volume.py \
  --in "$RUN_DIR/processed/mag_world.csv" \
  --out "$RUN_DIR/exports/volume.npz" \
  --voxel-size 0.01 \
  --margin 0.15 \
  --k 12 \
  --power 2.5
```

---

## volume.npz contents

- **`volume`:** 3D float array `(nx, ny, nz)`.
- **`origin`:** `[xmin, ymin, zmin]` (after margin).
- **`voxel_size`:** float.
- **`nx`, `ny`, `nz`:** grid dimensions.
- **`axes_x`, `axes_y`, `axes_z`:** 1D arrays of cell coordinates.

---

## Relation to other 3D scripts

- **Before:** **`fuse_mag_with_trajectory`** produces `mag_world.csv`.
- **After:** **`visualize_3d_heatmap`** loads `volume.npz` for PyVista visualization (slices, isosurfaces, screenshots).

See [PIPELINE_3D.md](PIPELINE_3D.md) for the full 3D runbook.
