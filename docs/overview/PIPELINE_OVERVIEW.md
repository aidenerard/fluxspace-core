# Pipeline Overview

Fluxspace Core has two pipelines: **2D** (grid survey + anomaly heatmaps) and **3D** (scan fusion + voxel heatmap). Both are invoked via **`pipelines/2d/`**, **`pipelines/3d/`**, and **`tools/3d/`** (stable entrypoints).

---

## New structure

- **Preferred:** Run from repo root: `python3 pipelines/2d/mag_to_csv.py`, `python3 pipelines/3d/fuse_mag_with_trajectory.py`, `./tools/3d/new_3d_scan.sh`. These are the stable paths to use in docs and automation.
- **`pipelines/2d/`** and **`pipelines/3d/`** = implementation and preferred Python entrypoints. **`tools/3d/`** = 3D shell scripts (new_3d_scan.sh, backup_usb_3d.sh).
- **Legacy:** Wrappers under `scripts/2d/` and `scripts/3d/` still work; preferred usage is `pipelines/...` and `tools/3d/`.

---

## When to use which

| Use case | Pipeline | Entrypoints | Doc |
|----------|----------|-------------|-----|
| 2D grid survey, anomaly maps, B_total heatmaps | **2D** | `pipelines/2d/mag_to_csv.py`, `validate_and_diagnosticsV1.py`, `compute_local_anomaly_v2.py`, `interpolate_to_heatmapV1.py`, `interpolate_to_Btotal_heatmap.py`; `./tools/2d/new_run.sh` | [PIPELINE_2D.md](../2d/PIPELINE_2D.md) |
| 3D scan (Polycam/RTAB-Map) + mag fusion, voxel heatmap | **3D** | `pipelines/3d/mag_calibrate_zero_logger.py` or `pipelines/2d/mag_to_csv_v2.py`, `polycam_raw_to_trajectory.py`, `rtabmap_poses_to_trajectory.py`, `fuse_mag_with_trajectory.py`, `mag_world_to_voxel_volume.py`, `visualize_3d_heatmap.py`; `./tools/3d/new_3d_scan.sh`, `./tools/3d/backup_usb_3d.sh` | [PIPELINE_3D.md](../3d/PIPELINE_3D.md) |
| 3D scan (OAK-D Lite) + Open3D reconstruction | **3D** | `pipelines/3d/capture_oak_rgbd.py`, `pipelines/3d/open3d_reconstruct.py` | [PIPELINE_3D.md](../3d/PIPELINE_3D.md) (see "OAK-D Lite" section) |

---

## Quick links

- **2D pipeline (concise):** [PIPELINE_2D.md](../2d/PIPELINE_2D.md) — grid collect → validate → anomaly → heatmaps → run snapshot.
- **3D pipeline (full runbook):** [PIPELINE_3D.md](../3d/PIPELINE_3D.md) — capture → trajectory → fuse → voxel → visualize; 3D scan storage (`data/scans/*__3d`).
- **Pi setup + testing:** [raspberry_pi_setup.md](../raspberry_pi_setup.md) — hardware, env, 2D and 3D commands.
- **Smoke tests:** [SMOKE_TESTS.md](../tests/SMOKE_TESTS.md) — quick checks that entrypoints and pipelines run.

**Legacy:** Wrappers under `scripts/2d/` and `scripts/3d/`, and shell scripts under `scripts/3d/`, still work. Preferred usage is **`pipelines/2d/`**, **`pipelines/3d/`**, and **`tools/3d/`**.
