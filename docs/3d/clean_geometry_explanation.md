# Explanation of `clean_geometry.py`

This document explains the automatic geometry cleaning script that removes noise, crops to the scan region, repairs meshes, and produces clean point clouds and meshes ready for heatmap overlay.

---

## Overview

Raw 3D reconstructions from TSDF integration (`open3d_reconstruct.py`) typically contain floating noise points, outlier clusters, and mesh artefacts. **`clean_geometry.py`** applies a multi-stage cleaning pipeline to produce publication-quality geometry.

**Inputs:**
- Raw point cloud (`open3d_pcd_raw.ply`)
- Raw mesh (`open3d_mesh_raw.ply`, optional)
- Trajectory CSV (optional, for trajectory-based cropping)

**Outputs:**
- Cleaned point cloud (`open3d_pcd_clean.ply`)
- Cleaned mesh (`open3d_mesh_clean.ply`)
- Preview point cloud (`open3d_pcd_preview.ply`, downsampled)
- Cleaning report (`cleaning_report.json`)

---

## What it does

The pipeline runs these stages in order:

1. **Load raw point cloud** from `open3d_pcd_raw.ply`.
2. **Remove NaN/inf points** — explicitly filters invalid coordinates.
3. **Trajectory crop** (optional) — loads `trajectory.csv`, computes a bounding box from the camera path (with margin), and crops the point cloud to that region. Robust fallbacks:
   - Skips if fewer than 2 finite trajectory points.
   - Skips if computed bounds contain NaN/inf or are degenerate (lo >= hi).
   - Falls back to uncropped data if cropping leaves fewer than 100 points.
4. **Manual AABB crop** (optional) — alternative to trajectory crop using explicit `--crop-aabb` coordinates.
5. **Voxel downsample** — reduces point density to a uniform grid spacing (`--downsample`, default 0.005 m).
6. **Statistical Outlier Removal (SOR)** — removes points whose average distance to neighbours is abnormally large (`--sor-nb-neighbors`, `--sor-std-ratio`).
7. **Radius Outlier Removal (ROR)** — removes points that have too few neighbours within a given radius (`--ror-radius`, `--ror-min-points`). Skipped with `--no-ror`.
8. **DBSCAN cluster filtering** — clusters remaining points and keeps only the largest cluster(s), removing isolated noise blobs.
9. **Plane removal** (optional, `--remove-plane`) — uses RANSAC to detect the dominant plane (e.g. ground/floor) and removes it.
10. **Estimate normals** — computes and orients surface normals if not already present.
11. **Write cleaned point cloud** and preview (more aggressively downsampled).
12. **Mesh cleaning / generation:**
    - **Primary path:** If a raw mesh (`open3d_mesh_raw.ply`) is provided, it is cleaned: degenerate/duplicated triangles removed, non-manifold edges fixed, optionally simplified, then cropped to the cleaned point cloud's bounding box.
    - **Fallback 1:** Ball Pivoting Algorithm (BPA) from the cleaned point cloud.
    - **Fallback 2:** Poisson surface reconstruction with density-based artefact trimming.
13. **Write cleaning report** — `cleaning_report.json` records point counts at each stage, parameters, mesh method, warnings, and elapsed time.

---

## CLI arguments

### Input / Output

| Argument | Description |
|----------|-------------|
| `--run-dir DIR` | Run directory — derives default input/output paths. |
| `--in-pcd PATH` | Input point cloud (default: `RUN_DIR/processed/open3d_pcd_raw.ply`). |
| `--in-mesh PATH` | Input raw mesh for cleaning (default: `RUN_DIR/processed/open3d_mesh_raw.ply`). |
| `--out-pcd PATH` | Output cleaned point cloud. |
| `--out-mesh PATH` | Output cleaned mesh. |
| `--out-dir DIR` | Override output directory. |

### Downsample + Outlier Removal

| Argument | Default | Description |
|----------|---------|-------------|
| `--downsample` | 0.005 | Voxel downsample size (metres). |
| `--sor-nb-neighbors` | 30 | SOR: number of neighbours to consider. |
| `--sor-std-ratio` | 2.0 | SOR: standard deviation multiplier threshold. |
| `--ror-radius` | 0.02 | ROR: search radius (metres). |
| `--ror-min-points` | 16 | ROR: minimum neighbours within radius. |
| `--no-ror` | — | Skip radius outlier removal. |

### Cluster Filtering

| Argument | Default | Description |
|----------|---------|-------------|
| `--min-cluster-points` | 5000 | Minimum points for a cluster to be kept. |
| `--keep-top-k-clusters` | 1 | Number of largest clusters to retain. |

### Trajectory Crop

| Argument | Default | Description |
|----------|---------|-------------|
| `--trajectory PATH` | auto | Path to trajectory CSV. |
| `--crop-from-trajectory` | false | Enable trajectory-based cropping. |
| `--no-traj-crop` | false | Force skip trajectory crop. |
| `--crop-margin` | 0.25 | Margin around trajectory bounding box (metres). |
| `--crop-z-min`, `--crop-z-max` | — | Override Z bounds for crop. |

### Manual Crop

| Argument | Description |
|----------|-------------|
| `--crop-aabb XMIN YMIN ZMIN XMAX YMAX ZMAX` | Explicit axis-aligned bounding box. |

### Mesh Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--mesh-target-triangles` | 0 | Target triangle count for simplification (0 = no simplification). |
| `--save-glb` | — | Export cleaned mesh as `.glb`. |
| `--preview-downsample` | 0.015 | Voxel size for preview point cloud. |

### Plane Removal

| Argument | Default | Description |
|----------|---------|-------------|
| `--remove-plane` | false | Remove dominant plane (RANSAC). |
| `--plane-dist-thresh` | 0.01 | RANSAC distance threshold. |
| `--plane-ransac-n` | 3 | RANSAC minimum sample size. |
| `--plane-num-iter` | 1000 | RANSAC iterations. |

---

## Example usage

```bash
# Using run directory (recommended — derives all paths automatically)
python3 pipelines/3d/clean_geometry.py \
  --run-dir "$RUN_DIR" \
  --crop-from-trajectory \
  --remove-plane

# Explicit paths
python3 pipelines/3d/clean_geometry.py \
  --in-pcd "$RUN_DIR/processed/open3d_pcd_raw.ply" \
  --in-mesh "$RUN_DIR/processed/open3d_mesh_raw.ply" \
  --out-pcd "$RUN_DIR/processed/open3d_pcd_clean.ply" \
  --out-mesh "$RUN_DIR/processed/open3d_mesh_clean.ply" \
  --trajectory "$RUN_DIR/processed/trajectory.csv" \
  --crop-from-trajectory

# High-quality cleaning
python3 pipelines/3d/clean_geometry.py \
  --run-dir "$RUN_DIR" \
  --downsample 0.003 \
  --sor-nb-neighbors 40 \
  --sor-std-ratio 1.8 \
  --crop-from-trajectory \
  --remove-plane
```

---

## Quality presets (via `run_all_3d.sh`)

The pipeline runner `run_all_3d.sh` provides `--quality` presets that translate to `clean_geometry.py` parameters:

| Preset | `--downsample` | `--sor-nb-neighbors` | `--sor-std-ratio` |
|--------|----------------|----------------------|-------------------|
| `fast` | 0.01 | 20 | 2.5 |
| `balanced` (default) | 0.005 | 30 | 2.0 |
| `high` | 0.003 | 40 | 1.8 |

---

## Output: `cleaning_report.json`

```json
{
  "parameters": {
    "voxel_size": 0.005,
    "sor_nb_neighbors": 30,
    "sor_std_ratio": 2.0,
    "ror_radius": 0.02,
    "ror_min_points": 16,
    "min_cluster_points": 5000,
    "keep_top_k_clusters": 1,
    "remove_plane": true,
    "crop_margin": 0.25
  },
  "input_pcd_points": 245000,
  "after_nan_removal": 244800,
  "after_crop": 180000,
  "after_downsample": 95000,
  "after_sor": 88000,
  "after_ror": 85000,
  "after_cluster": 82000,
  "after_plane_removal": 70000,
  "final_pcd_points": 70000,
  "final_mesh_vertices": 68000,
  "final_mesh_triangles": 135000,
  "mesh_method": "cleaned_raw",
  "elapsed_seconds": 12.3,
  "warnings": []
}
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `open3d` | Point cloud / mesh I/O, filtering (SOR, ROR, DBSCAN), mesh operations. |
| `numpy` | Array operations, NaN handling, bounding box math. |

---

## Relation to other 3D scripts

- **Before:** `open3d_reconstruct.py` produces `open3d_pcd_raw.ply` and `open3d_mesh_raw.ply`.
- **After:** `view_scan_toggle.py` prefers cleaned geometry (`_clean`) over raw outputs.
- **Pipeline integration:** Called automatically by `run_all_3d.sh` (Step 2) unless `--skip-clean` is passed.

See [PIPELINE_3D.md](PIPELINE_3D.md) for the full 3D runbook.
