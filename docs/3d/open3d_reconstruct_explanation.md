# Explanation of `open3d_reconstruct.py`

This document explains the 3D pipeline script that takes the RGB + depth frames captured by `capture_oak_rgbd.py` and produces **two outputs**: a `trajectory.csv` (camera poses over time) and an `open3d_mesh.ply` (coloured triangle mesh), using Open3D's TSDF volume integration and frame-to-frame RGB-D odometry.

---

## Overview

After capturing an RGB-D sequence with the OAK-D Lite (see [capture_oak_rgbd_explanation.md](capture_oak_rgbd_explanation.md)), you have a folder of colour images, aligned 16-bit depth images, timestamps, and camera intrinsics. **`open3d_reconstruct.py`** loads these frames, estimates camera motion with frame-to-frame odometry, integrates every frame into a TSDF volume, extracts a mesh, and — critically — **exports `trajectory.csv`** in the same format used by the rest of the 3D pipeline (`t_rel_s, x, y, z, qx, qy, qz, qw`).

This means the output feeds directly into `fuse_mag_with_trajectory.py`, making the OAK-D workflow a drop-in replacement for the Polycam / RTAB-Map trajectory extraction step.

**Input:** Directory containing `color/color_*.jpg`, `depth/depth_*.png`, `timestamps.csv`, and optionally `intrinsics.json` (produced by `capture_oak_rgbd.py`).

**Outputs:**

- `trajectory.csv` — camera pose per frame, same columns as `polycam_raw_to_trajectory.py` output.
- `open3d_mesh.ply` — coloured triangle mesh viewable in Open3D, MeshLab, Blender, or any PLY viewer.

---

## What it does

1. **Parse arguments:** `--in` (input dir), `--out` (trajectory CSV path), `--mesh` (mesh PLY path), `--voxel-size`, `--no-viz`.
2. **Load frames:** Reads all `color_*.jpg` and `depth_*.png` files from `<in>/color/` and `<in>/depth/`, sorted by filename.
3. **Load intrinsics:** Reads `intrinsics.json` from the input directory if present (saved by `capture_oak_rgbd.py`). Falls back to approximate values if missing.
4. **Load timestamps:** Reads `timestamps.csv` to map frame indices to device timestamps. Uses these to compute `t_rel_s` for each pose. Falls back to frame index / 30 fps if the file is missing.
5. **TSDF volume:** Creates a `ScalableTSDFVolume` with configurable voxel size (default 1 cm) and appropriate truncation distance.
6. **Frame-to-frame odometry loop:**
   - For each consecutive pair of RGB-D images, computes the relative transform using `compute_rgbd_odometry` with the hybrid Jacobian (combines intensity and depth terms).
   - Chains the relative transforms to maintain a running camera pose (4x4 matrix).
   - If odometry fails for a frame (e.g. too much motion blur), the previous pose is kept (graceful degradation).
7. **TSDF integration:** Each frame is integrated into the volume at its estimated pose.
8. **Mesh extraction:** After all frames, extracts a triangle mesh from the TSDF volume and computes vertex normals.
9. **Write trajectory.csv:** Converts each 4x4 pose matrix to position `(x, y, z)` + quaternion `(qx, qy, qz, qw)`, paired with `t_rel_s` from the timestamps. Writes the CSV.
10. **Write mesh:** Saves `open3d_mesh.ply`.
11. **Visualise:** Opens an Open3D viewer unless `--no-viz` is set.

---

## CLI arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--in` | No | Input directory with `color/`, `depth/`, `timestamps.csv`, `intrinsics.json`. Default: `oak_capture`. |
| `--out` | No | Output `trajectory.csv` path. Default: auto-detect — `$RUN_DIR/processed/trajectory.csv` if input is under `$RUN_DIR/raw/`, otherwise `<in>/trajectory.csv`. |
| `--mesh` | No | Output mesh PLY path. Default: auto-detect — `$RUN_DIR/exports/open3d_mesh.ply` or `<in>/open3d_mesh.ply`. |
| `--voxel-size` | No | TSDF voxel size in metres. Default: 0.01 (1 cm). |
| `--no-viz` | No | Skip the Open3D interactive viewer (headless / CI). |

---

## Example usage

```bash
# Standalone (reads oak_capture/, writes to oak_capture/)
python3 pipelines/3d/open3d_reconstruct.py

# Pipeline-integrated (auto-detects processed/ and exports/ from input path)
python3 pipelines/3d/open3d_reconstruct.py \
  --in "$RUN_DIR/raw/oak_rgbd" \
  --no-viz
#   -> $RUN_DIR/processed/trajectory.csv
#   -> $RUN_DIR/exports/open3d_mesh.ply

# Explicit output paths
python3 pipelines/3d/open3d_reconstruct.py \
  --in "$RUN_DIR/raw/oak_rgbd" \
  --out "$RUN_DIR/processed/trajectory.csv" \
  --mesh "$RUN_DIR/exports/open3d_mesh.ply" \
  --voxel-size 0.005 \
  --no-viz
```

---

## Output format: `trajectory.csv`

| Column | Type | Description |
|--------|------|-------------|
| `t_rel_s` | float | Seconds since first frame (from device timestamps). |
| `x` | float | Camera X position in metres. |
| `y` | float | Camera Y position in metres. |
| `z` | float | Camera Z position in metres. |
| `qx` | float | Quaternion X component. |
| `qy` | float | Quaternion Y component. |
| `qz` | float | Quaternion Z component. |
| `qw` | float | Quaternion W component. |

This is the **same schema** as the output of `polycam_raw_to_trajectory.py` and `rtabmap_poses_to_trajectory.py`. It is consumed directly by `fuse_mag_with_trajectory.py`.

---

## Key parameters (in-script constants / CLI flags)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--voxel-size` | 0.01 (1 cm) | TSDF voxel size. Increase to 0.02–0.05 if slow. |
| SDF truncation | 4x voxel size | Auto-computed, minimum 0.04 m. |
| `depth_scale` | 1000.0 | Depth PNGs are mm; divides by 1000 to get metres. |
| `depth_trunc` | 3.0 m | Ignore depth values beyond 3 metres. |
| Timestamp fallback | 30 fps | Used if `timestamps.csv` is missing. |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `open3d` | RGB-D odometry, TSDF integration, mesh extraction, visualisation. |
| `numpy` | Pose matrix operations, quaternion conversion. |

Install (in the same virtualenv as the capture script):

```bash
pip install open3d numpy
```

---

## Improving quality

### 1. Real camera intrinsics (automatic)

`capture_oak_rgbd.py` now saves `intrinsics.json` with the real OAK-D calibration. `open3d_reconstruct.py` loads it automatically — no manual step needed.

### 2. Use a stronger SLAM backend

Frame-to-frame odometry accumulates drift over time. For larger scenes or longer captures, use a SLAM system like **RTAB-Map** or **ORB-SLAM3** to compute more accurate poses, then export as `trajectory.csv` and feed into `fuse_mag_with_trajectory.py`.

### 3. Tune TSDF parameters

- **Smaller voxels** (`--voxel-size 0.005`) give finer detail but use more memory and time.
- **Larger truncation** helps with noisy depth but can smear thin structures.
- **Lower `depth_trunc`** filters out unreliable far-range depth (edit the constant in the script).

---

## How it fits in the pipeline

```
capture_oak_rgbd.py
  └─ oak_rgbd/ (color, depth, timestamps, intrinsics)
       │
       v
open3d_reconstruct.py
  ├─ trajectory.csv  ─────> fuse_mag_with_trajectory.py ─> mag_world.csv
  └─ open3d_mesh.ply           │
                               v
                         mag_world_to_voxel_volume.py ─> volume.npz
                               │
                               v
                         visualize_3d_heatmap.py ─> screenshots, slices
```

Everything after `trajectory.csv` is **identical** to the Polycam / RTAB-Map path.

---

## Relation to other 3D scripts

- **Before:** Run **`capture_oak_rgbd.py`** to record the RGB-D frames. See [capture_oak_rgbd_explanation.md](capture_oak_rgbd_explanation.md).
- **After:** Feed `trajectory.csv` into **`fuse_mag_with_trajectory.py`** (with `mag_run.csv` + `extrinsics.json`) to produce `mag_world.csv`.
- **Alternative workflows:** Instead of OAK-D + Open3D, you can use Polycam or RTAB-Map for scanning and trajectory extraction (`polycam_raw_to_trajectory.py`, `rtabmap_poses_to_trajectory.py`).

See [PIPELINE_3D.md](PIPELINE_3D.md) for the full 3D runbook.
