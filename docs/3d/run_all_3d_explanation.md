# Explanation of `run_all_3d.sh`

This document explains the one-command pipeline runner that orchestrates the entire FluxSpace 3D processing workflow — from reconstruction through cleaning to visualisation.

---

## Overview

**`tools/3d/run_all_3d.sh`** is a bash script that runs the full 3D pipeline in sequence with a single command. It handles run directory selection, input validation, dependency checking, error recovery, and final summary output.

**Supported workflows:**
- **Full mode** (camera + magnetometer): reconstruct → clean → mag fusion → voxelise → viewer
- **Camera-only mode**: reconstruct → clean → viewer (no mag files needed)

---

## What it does

### 1. Dependency check

Verifies that `python3` is available and that required Python packages (`numpy`, `open3d`) are importable. In full mode, also checks for `pandas` and `scipy`. Prints hints about activating a virtualenv if imports fail.

### 2. Quality presets

Translates `--quality` presets into cleaning parameters:

| Preset | `--downsample` | `--sor-nb-neighbors` | `--sor-std-ratio` |
|--------|----------------|----------------------|-------------------|
| `fast` | 0.01 | 20 | 2.5 |
| `balanced` (default) | 0.005 | 30 | 2.0 |
| `high` | 0.003 | 40 | 1.8 |

### 3. Run directory selection

Three modes for choosing the working directory:

| Flag | Behaviour |
|------|-----------|
| `--run DIR` | Use an explicit existing directory. |
| `--latest` | Find the newest `data/runs/run_*` directory. |
| `--new` | Create `data/runs/run_YYYYMMDD_HHMM`. |

### 4. Camera-only auto-detection

If `--camera-only` is not specified but `raw/mag_run.csv` is missing, the script automatically switches to camera-only mode and prints a clear banner.

### 5. Input validation

Checks for required files and prints colour-coded pass/fail messages:
- **Always required:** `raw/oak_rgbd/color/`, `raw/oak_rgbd/depth/`, `raw/oak_rgbd/timestamps.csv`
- **Full mode only:** `raw/mag_run.csv` (required), `calibration.json` (warn if missing), `extrinsics.json` (warn if missing)
- Verifies colour/depth frame counts match.

### 6. Pipeline steps

Each step runs inside a `run_step` wrapper that:
- Prints a numbered banner.
- Captures output to a temp file (unless `--verbose`).
- Shows the last 8 lines on success.
- Prints full output and exits on failure.

| Step | Script | Description |
|------|--------|-------------|
| **1** | `open3d_reconstruct.py` | TSDF reconstruction → mesh + point cloud + trajectory. Detects `trajectory_device.csv` for VIO poses. |
| **2** | `clean_geometry.py` | Denoise, crop, cluster, mesh repair. If cleaning fails, continues with raw outputs. |
| **3** | `fuse_mag_with_trajectory.py` | Fuse magnetometer data with camera trajectory. *Skipped in camera-only mode.* |
| **4** | `mag_world_to_voxel_volume.py` | Build 3D voxel heatmap from fused mag data. *Skipped in camera-only mode.* |
| **5** | `view_scan_toggle.py` | Open interactive viewer. *Skipped with `--no-viewer`.* |

### 7. VIO pose-source detection

Before Step 1, the script checks for `raw/oak_rgbd/trajectory_device.csv`. If found (and `--pose-source` is `auto`), it informs the user that device poses will be used instead of RGB-D odometry.

### 8. Final summary

Prints a colour-coded summary block listing:
- Run directory, pipeline mode, pose source
- Paths to geometry, trajectory, reports
- Magnetometer and volume paths (full mode only)
- Command to re-open the viewer

---

## CLI arguments

### Mode (required — pick one)

| Flag | Description |
|------|-------------|
| `--run DIR` | Use an existing run directory. |
| `--latest` | Use the newest `data/runs/run_*`. |
| `--new` | Create a fresh run directory. |

### Pipeline mode

| Flag | Description |
|------|-------------|
| `--camera-only` | Skip mag fusion + voxelisation. Auto-detected if `mag_run.csv` is absent. |

### Reconstruction options

| Flag | Default | Description |
|------|---------|-------------|
| `--pose-source {auto,odom,device}` | auto | Pose source for reconstruction. |
| `--use-device-pose` | — | Alias for `--pose-source device`. |
| `--every-n N` | 1 | Use every Nth frame. |
| `--max-frames N` | 0 (all) | Stop reconstruction after N frames. |
| `--depth-trunc F` | 3.0 | Max depth in metres. |
| `--recon-voxel F` | 0.01 | TSDF voxel size. |
| `--odometry METHOD` | hybrid | `hybrid` or `color`. |
| `--save-glb` | — | Export meshes as `.glb`. |

### Cleaning options

| Flag | Default | Description |
|------|---------|-------------|
| `--quality PRESET` | balanced | `fast`, `balanced`, or `high`. |
| `--skip-clean` | — | Skip geometry cleaning entirely. |
| `--clean-voxel F` | (preset) | Override downsample voxel size. |
| `--clean-sor-nb N` | (preset) | Override SOR neighbour count. |
| `--clean-sor-std F` | (preset) | Override SOR std ratio. |

### Voxelisation options (ignored in camera-only mode)

| Flag | Default | Description |
|------|---------|-------------|
| `--voxel-size F` | 0.02 | Mag volume voxel edge length. |
| `--max-dim N` | 256 | Max voxels per axis. |

### Input overrides

| Flag | Description |
|------|-------------|
| `--mag PATH` | Override magnetometer CSV path. |
| `--oak PATH` | Override OAK capture directory. |
| `--extrinsics PATH` | Override extrinsics JSON path. |
| `--default-extrinsics STR` | Mount offset shorthand (e.g. `"behind_cm=2,down_cm=10"`). |

### General

| Flag | Description |
|------|-------------|
| `--repo-root PATH` | Repository root (default: auto-detect from git). |
| `--no-viewer` | Skip the viewer at the end. |
| `--verbose` | Print full Python output (not just last 8 lines). |

---

## Example usage

```bash
# Process the latest run (full pipeline)
./tools/3d/run_all_3d.sh --latest

# Camera-only, high quality
./tools/3d/run_all_3d.sh --latest --camera-only --quality high

# Specific run with frame subsampling
./tools/3d/run_all_3d.sh --run data/runs/run_20260210_1430 --every-n 2 --depth-trunc 1.5

# Headless processing (no viewer)
./tools/3d/run_all_3d.sh --latest --no-viewer

# Force device pose source
./tools/3d/run_all_3d.sh --latest --camera-only --use-device-pose

# Create and process a new run
./tools/3d/run_all_3d.sh --new
```

---

## Error handling

- **Missing inputs:** Fails early with clear messages about which files are missing and where they should be.
- **Step failure:** If any Python step fails, the full output is printed and the script exits. Exception: cleaning failure is treated as a warning (continues with raw geometry).
- **Missing dependencies:** Prints the package name and install command.
- **`set -euo pipefail`:** Strict bash mode — unset variables and pipe failures are caught.

---

## Dependencies

- **bash** (4.0+)
- **python3** with packages: `numpy`, `open3d` (always); `pandas`, `scipy` (full mode)
- Standard unix tools: `ls`, `sort`, `tail`, `wc`, `date`, `mktemp`

---

## Relation to other 3D scripts

This script is the primary entry point for running the pipeline. It calls:
- `pipelines/3d/open3d_reconstruct.py`
- `pipelines/3d/clean_geometry.py`
- `pipelines/3d/fuse_mag_with_trajectory.py` (full mode)
- `pipelines/3d/mag_world_to_voxel_volume.py` (full mode)
- `pipelines/3d/view_scan_toggle.py`

For validation after the fact, use `pipelines/3d/validate_run.py`.

See [PIPELINE_3D.md](PIPELINE_3D.md) for the full 3D runbook.
