# Explanation of `interpolate_to_heatmapV3_gpr.py`

This document explains the **Gaussian Process Regression (GPR)** 2D heatmap script that produces a smooth mean map, a **gradient-magnitude** (edge‑aware) map, and an **uncertainty** (std) map from scattered `(x, y, value)` samples.

---

## Overview

**`interpolate_to_heatmapV3_gpr.py`** fits a GPR model to input CSV columns `x`, `y`, and a value column (default `local_anomaly`), evaluates it on a dense 2D grid, and writes:

- **`*_gpr_mean.png`** — GPR mean heatmap  
- **`*_gpr_grad.png`** — gradient magnitude of the mean (sharpens edges)  
- **`*_gpr_std.png`** — GPR standard deviation (uncertainty)  
- **`*_gpr_grid.csv`** — dense grid with `x`, `y`, `*_mean`, `*_std`, `*_gradmag`

**Input:** CSV with at least `x`, `y`, and a value column (e.g. `local_anomaly` or `B_total`).  
**Output:** Written to `--out-dir` (default: same folder as input).  
**Dependency:** `scikit-learn` (RBF + WhiteKernel + ConstantKernel). GPR is O(N³); use `--max-points` to subsample when you have many points.

---

## What it does

1. **Load:** Read CSV, extract `(x, y, value)`; drop NaN. Optionally **subsample** to `--max-points` (random seed `--seed`).
2. **Normalize:** Standardize the value column (mean 0, std 1) for kernel scaling; transform back for outputs.
3. **Kernel:** `ConstantKernel * RBF(length_scale) + WhiteKernel(noise)`. `--length-scale` controls smoothness (smaller = sharper). `--no-optimize` fixes hyperparameters.
4. **Fit:** `GaussianProcessRegressor` fit on `(X, y)`.
5. **Predict:** Evaluate mean and std on a regular grid (`--grid-nx`, `--grid-ny`, `--pad`). Compute **gradient magnitude** of the mean via finite differences.
6. **Save:** Write mean, grad, and std heatmaps (PNG) and a long-form grid CSV.

---

## CLI arguments (summary)

| Argument | Default | Description |
|----------|---------|-------------|
| `--in` | (required) | Input CSV path. |
| `--value-col` | `local_anomaly` | Value column to model. |
| `--x-col`, `--y-col` | `x`, `y` | Coordinate columns. |
| `--out-dir` | (input folder) | Output directory. |
| `--name` | (input stem) | Output file prefix. |
| `--grid-nx`, `--grid-ny` | 200 | Grid resolution. |
| `--pad` | 0 | Padding (m) around bounds. |
| `--max-points` | 3000 | Max training points (subsample if larger). |
| `--length-scale` | 0.03 | RBF length-scale (m). |
| `--signal`, `--noise`, `--alpha` | 1.0, 0.001, 1e-6 | Kernel params. |
| `--no-optimize` | — | Disable hyperparameter optimization. |
| `--seed` | 7 | Random seed for subsampling. |
| `--fixed-vmin`, `--fixed-vmax` | — | Fix color scale for mean heatmap. |
| `--grad-vmax` | — | Fix vmax for gradient heatmap. |

---

## Example usage

```bash
# After compute_local_anomaly_v2
python3 pipelines/2d/interpolate_to_heatmapV3_gpr.py \
  --in data/processed/mag_data_anomaly.csv \
  --value-col local_anomaly \
  --out-dir data/exports

# Coarser grid, smaller length-scale (sharper)
python3 pipelines/2d/interpolate_to_heatmapV3_gpr.py \
  --in data/processed/mag_data_anomaly.csv \
  --value-col local_anomaly \
  --grid-nx 150 --grid-ny 150 \
  --length-scale 0.02 \
  --max-points 2000
```

---

## Relation to other 2D scripts

- **Input:** Typically `*_anomaly.csv` from `compute_local_anomaly_v2`, or any CSV with `x`, `y`, and a value column (e.g. `B_total`).
- **Alternatives:** `interpolate_to_heatmapV1` / `V2` use IDW; `interpolate_to_Btotal_heatmap` targets B_total. GPR adds smooth mean, uncertainty, and gradient-based edge emphasis.

See [PIPELINE_2D.md](PIPELINE_2D.md) and [interpolate_to_heatmap_explanation.md](interpolate_to_heatmap_explanation.md).
