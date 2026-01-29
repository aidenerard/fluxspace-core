# Pipeline Overview

Fluxspace Core has two pipelines: **2D** (grid survey + anomaly heatmaps) and **3D** (scan fusion + voxel heatmap). Both are invoked from the **`scripts/`** folder (stable entrypoints).

---

## New structure

- **`scripts/`** = **entrypoints**. Run commands from repo root, e.g. `python3 scripts/mag_to_csv.py` or `./scripts/new_3d_scan.sh`. These are the stable paths to use in docs and automation.
- **`pipelines/2d/`** and **`pipelines/3d/`** = **implementation**. The real Python code lives here; files in `scripts/` are thin wrappers that call them.
- **Legacy:** Existing commands (e.g. `python3 scripts/fuse_mag_with_trajectory.py --help`) still work; wrappers pass through to the implementation.

---

## When to use which

| Use case | Pipeline | Entrypoints | Doc |
|----------|----------|-------------|-----|
| 2D grid survey, anomaly maps, B_total heatmaps | **2D** | `scripts/mag_to_csv.py`, `validate_and_diagnosticsV1.py`, `compute_local_anomaly_v2.py`, `interpolate_to_heatmapV1.py`, `interpolate_to_Btotal_heatmap.py`; `./tools/new_run.sh` | [PIPELINE_2D.md](../2d/PIPELINE_2D.md) |
| 3D scan (Polycam/RTAB-Map) + mag fusion, voxel heatmap | **3D** | `scripts/mag_calibrate_zero_logger.py` or `mag_to_csv_v2.py`, `polycam_raw_to_trajectory.py`, `rtabmap_poses_to_trajectory.py`, `fuse_mag_with_trajectory.py`, `mag_world_to_voxel_volume.py`, `visualize_3d_heatmap.py`; `./scripts/new_3d_scan.sh`, `./scripts/backup_usb_3d.sh` | [PIPELINE_3D.md](../3d/PIPELINE_3D.md) |

---

## Quick links

- **2D pipeline (concise):** [PIPELINE_2D.md](../2d/PIPELINE_2D.md) — grid collect → validate → anomaly → heatmaps → run snapshot.
- **3D pipeline (full runbook):** [PIPELINE_3D.md](../3d/PIPELINE_3D.md) — capture → trajectory → fuse → voxel → visualize; 3D scan storage (`data/scans/*__3d`).
- **Pi setup + testing:** [raspberry_pi_setup.md](../raspberry_pi_setup.md) — hardware, env, 2D and 3D commands.
- **Smoke tests:** [SMOKE_TESTS.md](../tests/SMOKE_TESTS.md) — quick checks that entrypoints and pipelines run.
