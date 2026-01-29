# Explanation of `visualize_3d_heatmapV2_gpr.py`

This document explains the **lightweight 3D GPR volume viewer** that displays a thresholded voxel scatter and optional orthogonal slices for `.npz` volumes produced by **`mag_world_to_voxel_volumeV2_gpr`**.

---

## Overview

**`visualize_3d_heatmapV2_gpr.py`** reads an `.npz` file containing `volume` and metadata (`origin`, `voxel_size`, `dims`, etc.), thresholds voxels by value (percentile or absolute), and either:

- Plots a **3D scatter** of voxels above the threshold (optionally subsampled for performance), and/or  
- Shows **XY, XZ, YZ slices** at configurable indices (default: grid center).

It uses **matplotlib** only (no PyVista), so it runs in environments where PyVista is not available. It is designed to work with the **GPR** `.npz` layout (e.g. `*_gpr_mean.npz`, `*_gpr_std.npz`, `*_gpr_grad.npz`).

---

## What it does

1. **Load:** `np.load` the `.npz`; read `volume` (shape `(nz, ny, nx)`), `origin`, `voxel_size`, `dims`.
2. **Threshold:** By default, use `--percentile` (e.g. 99) so only voxels ≥ that percentile are shown. If `--abs-thresh` is set, use that value instead.
3. **Mask:** `mask = volume >= thresh`; collect indices and values for voxels above the threshold.
4. **World coords:** Convert linear indices to `(x, y, z)` using `origin` and `voxel_size`.
5. **Subsample:** If the number of points exceeds `--max-points`, randomly subsample (seed `--seed`) for plotting.
6. **3D scatter:** `matplotlib` 3D axes, `scatter(x, y, z, c=values)`, colorbar, title with threshold.
7. **Slices (optional):** If `--show-slices`, plot XY slice at `--slice-z`, XZ at `--slice-y`, YZ at `--slice-x` (defaults: middle of each axis). Each in a separate figure.

---

## CLI arguments (summary)

| Argument | Default | Description |
|----------|---------|-------------|
| `--in` | (required) | Input `.npz` path (e.g. `*_gpr_mean.npz`). |
| `--percentile` | 99.0 | Show voxels ≥ this percentile. |
| `--abs-thresh` | — | Override with absolute threshold. |
| `--max-points` | 300000 | Cap points in 3D scatter (subsample if larger). |
| `--seed` | 7 | Random seed for subsampling. |
| `--show-slices` | — | Show XY/XZ/YZ slice figures. |
| `--slice-z`, `--slice-y`, `--slice-x` | (middle) | Slice indices for XY, XZ, YZ. |

---

## Example usage

```bash
# 3D scatter only (voxels ≥ 99th percentile)
python3 pipelines/3d/visualize_3d_heatmapV2_gpr.py \
  --in "$RUN_DIR/exports/mag_world_gpr_mean.npz"

# With orthogonal slices at default (middle) indices
python3 pipelines/3d/visualize_3d_heatmapV2_gpr.py \
  --in "$RUN_DIR/exports/mag_world_gpr_mean.npz" \
  --show-slices

# Custom percentile and slice indices
python3 pipelines/3d/visualize_3d_heatmapV2_gpr.py \
  --in "$RUN_DIR/exports/mag_world_gpr_mean.npz" \
  --percentile 95 \
  --show-slices \
  --slice-z 10 --slice-y 15 --slice-x 20
```

---

## Relation to other 3D scripts

- **Input:** `.npz` from **`mag_world_to_voxel_volumeV2_gpr`** (e.g. `*_gpr_mean.npz`, `*_gpr_std.npz`, `*_gpr_grad.npz`).
- **Alternative:** **`visualize_3d_heatmap`** uses PyVista for `volume.npz` produced by **`mag_world_to_voxel_volume`** (IDW/griddata); that script expects a different output layout and offers volume rendering, isosurfaces, and screenshots.

See [PIPELINE_3D.md](PIPELINE_3D.md), [mag_world_to_voxel_volumeV2_gpr_explanation.md](mag_world_to_voxel_volumeV2_gpr_explanation.md), and [visualize_3d_heatmap_explanation.md](visualize_3d_heatmap_explanation.md) (PyVista-based viewer for IDW/griddata volumes).
