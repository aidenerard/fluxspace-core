# Explanation of `mag_world_to_voxel_volumeV2_gpr.py`

This document explains the **3D GPR voxel volume** script that fits a Gaussian Process to `(x, y, z, value)` samples and rasterizes mean, std, and gradient-magnitude volumes to `.npz` files.

---

## Overview

**`mag_world_to_voxel_volumeV2_gpr.py`** is the 3D analogue of the 2D GPR heatmap:

- Fits a GPR to sparse `(x, y, z) → value` samples (default value column: `local_anomaly`).
- Evaluates on a 3D voxel grid (bounds from data + `--pad`, spacing `--voxel`).
- Writes **mean**, **std**, and **gradient-magnitude** volumes as separate `.npz` files with shared metadata (origin, voxel_size, dims, axis_order).

**Input:** CSV with `x`, `y`, `z`, and a value column (e.g. `mag_world.csv` from `fuse_mag_with_trajectory` or similar).  
**Output:** `*_gpr_mean.npz`, `*_gpr_std.npz`, `*_gpr_grad.npz` in `--out-dir`.  
**Dependency:** `scikit-learn`. GPR is O(N³); use `--max-points` to subsample. 3D grids grow quickly—start with `--voxel 0.02` or `0.03` to validate.

---

## What it does

1. **Load:** Read CSV, extract `(x, y, z, value)`; drop NaN. Optionally **subsample** to `--max-points` (seed `--seed`).
2. **Normalize:** Standardize values for kernel stability; undo for outputs.
3. **Kernel:** `ConstantKernel * RBF(length_scale) + WhiteKernel(noise)`. `--length-scale`, `--signal`, `--noise`, `--alpha`; `--no-optimize` fixes hyperparameters.
4. **Grid:** Build 3D voxel grid from data bounds + `--pad`, spacing `--voxel`.
5. **Predict:** Evaluate GPR mean and std on the grid in **chunks** (`--chunk`) to limit memory. Store volume as `(nz, ny, nx)`.
6. **Gradient:** Compute `|∇(mean)|` via `np.gradient` with voxel spacing.
7. **Save:** Write `*_gpr_mean.npz`, `*_gpr_std.npz`, `*_gpr_grad.npz` with `volume`, `origin`, `voxel_size`, `dims`, `axis_order`, `value_col`.

---

## CLI arguments (summary)

| Argument | Default | Description |
|----------|---------|-------------|
| `--in` | (required) | Input CSV path. |
| `--value-col` | `local_anomaly` | Value column to model. |
| `--x-col`, `--y-col`, `--z-col` | `x`, `y`, `z` | Coordinate columns. |
| `--out-dir` | (input folder) | Output directory. |
| `--name` | (input stem) | Output prefix. |
| `--voxel` | 0.02 | Voxel size (m). |
| `--pad` | 0 | Padding (m) around bounds. |
| `--max-points` | 2000 | Max training points. |
| `--length-scale` | 0.05 | RBF length-scale (m). |
| `--signal`, `--noise`, `--alpha` | 1.0, 0.001, 1e-6 | Kernel params. |
| `--no-optimize` | — | Disable hyperparameter optimization. |
| `--seed` | 7 | Random seed for subsampling. |
| `--chunk` | 200000 | Points per prediction batch. |

---

## Example usage

```bash
# From mag_world.csv (after fuse_mag_with_trajectory)
python3 pipelines/3d/mag_world_to_voxel_volumeV2_gpr.py \
  --in "$RUN_DIR/processed/mag_world.csv" \
  --value-col value \
  --out-dir "$RUN_DIR/exports" \
  --voxel 0.03 \
  --max-points 1500
```

---

## Relation to other 3D scripts

- **Input:** Typically `mag_world.csv` from **`fuse_mag_with_trajectory`** (columns `x`, `y`, `z`, `value`).
- **Output:** Consumed by **`visualize_3d_heatmapV2_gpr`** (thresholded scatter, slices). The standard **`visualize_3d_heatmap`** expects `volume.npz` from **`mag_world_to_voxel_volume`** (IDW/griddata); the GPR `.npz` layout matches the GPR visualizer.

See [PIPELINE_3D.md](PIPELINE_3D.md), [mag_world_to_voxel_volume_explanation.md](mag_world_to_voxel_volume_explanation.md), and [visualize_3d_heatmapV2_gpr_explanation.md](visualize_3d_heatmapV2_gpr_explanation.md).
