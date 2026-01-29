# Deduplication Report

**Date:** 2025-01-23  
**Goal:** Clean up duplicated/unused scripts and docs while keeping both 2D and 3D pipelines intact. Single source of truth: `pipelines/2d` and `pipelines/3d`.

---

## 1. What was found (inventory)

### Scripts vs pipelines

- **scripts/2d/** and **scripts/3d/** (Python files): Every file is a **thin wrapper** only. Each wrapper:
  - Sets `_REPO_ROOT` (one level up from script dir: `scripts/2d` or `scripts/3d` → repo root).
  - Sets `_REAL_SCRIPT` to the corresponding path under `pipelines/2d/` or `pipelines/3d/`.
  - Runs `runpy.run_path(_REAL_SCRIPT, run_name="__main__")` and preserves `sys.argv`.
- **No duplicated algorithm logic** was found in scripts. All implementation lives in `pipelines/2d/` and `pipelines/3d/`.

### File pairs (entrypoint → implementation)

| Entrypoint (scripts/) | Implementation (pipelines/) |
|------------------------|-----------------------------|
| scripts/2d/calibrate_magnetometerV1.py | pipelines/2d/calibrate_magnetometerV1.py |
| scripts/2d/compute_local_anomaly_v1.py | pipelines/2d/compute_local_anomaly_v1.py |
| scripts/2d/compute_local_anomaly_v2.py | pipelines/2d/compute_local_anomaly_v2.py |
| scripts/2d/interpolate_to_Btotal_heatmap.py | pipelines/2d/interpolate_to_Btotal_heatmap.py |
| scripts/2d/interpolate_to_heatmapV1.py | pipelines/2d/interpolate_to_heatmapV1.py |
| scripts/2d/interpolate_to_heatmapV2.py | pipelines/2d/interpolate_to_heatmapV2.py |
| scripts/2d/mag_to_csv.py | pipelines/2d/mag_to_csv.py |
| scripts/2d/mag_to_csv_v2.py | pipelines/2d/mag_to_csv_v2.py |
| scripts/2d/validate_and_diagnosticsV1.py | pipelines/2d/validate_and_diagnosticsV1.py |
| scripts/3d/fuse_mag_with_trajectory.py | pipelines/3d/fuse_mag_with_trajectory.py |
| scripts/3d/mag_calibrate_zero_logger.py | pipelines/3d/mag_calibrate_zero_logger.py |
| scripts/3d/mag_world_to_voxel_volume.py | pipelines/3d/mag_world_to_voxel_volume.py |
| scripts/3d/polycam_raw_to_trajectory.py | pipelines/3d/polycam_raw_to_trajectory.py |
| scripts/3d/rtabmap_poses_to_trajectory.py | pipelines/3d/rtabmap_poses_to_trajectory.py |
| scripts/3d/visualize_3d_heatmap.py | pipelines/3d/visualize_3d_heatmap.py |

Shell scripts (scripts/3d/backup_scans_to_usb.sh, backup_usb_3d.sh, new_3d_scan.sh) have no pipeline counterpart; they are standalone entrypoints.

---

## 2. What was removed

- **scripts/overview/.gitkeep** — Removed. `scripts/overview/` is not intentionally empty (it contains `run_metadataV1.py`). Per rule: keep `.gitkeep` only in intentionally empty dirs (e.g. `scripts/tests/`).

No other files were deleted. There were no duplicate implementations (scripts already only contained runpy wrappers; pipelines hold the only implementation code).

---

## 3. What was refactored into wrappers

- **Nothing.** All scripts/2d and scripts/3d Python files were already thin wrappers calling pipelines via `runpy.run_path`. No algorithm code was moved or refactored.

---

## 4. Renamed paths

- None. Entrypoints remain `scripts/2d/<name>.py` and `scripts/3d/<name>.py`; implementations remain `pipelines/2d/<name>.py` and `pipelines/3d/<name>.py`.

---

## 5. Docs changes

- **docs/overview/OVERVIEW.md**
  - **calibrate_magnetometerV1.py:** Updated from “Placeholder - functionality to be implemented” to implemented script with pointer to `scripts/2d/calibrate_magnetometerV1.py` and [calibrate_magnetometer_explanation.md](../2d/calibrate_magnetometer_explanation.md).
  - **run_metadataV1.py:** Clarified as “Placeholder (overview/shared). Not yet implemented.”
  - **compute_local_anomaly_explanation.md:** Fixed link from `./compute_local_anomaly_explanation.md` to `../2d/compute_local_anomaly_explanation.md` (file lives in docs/2d/).

---

## 6. Sanity-check commands

Run from **repo root**:

```bash
# 2D pipeline
python3 scripts/2d/mag_to_csv.py --help

# 3D pipeline
python3 scripts/3d/fuse_mag_with_trajectory.py --help
python3 scripts/3d/visualize_3d_heatmap.py --help

# Shell
./scripts/3d/new_3d_scan.sh --help
```

All of the above should print usage/help; no broken imports or missing modules.  
**Note:** `visualize_3d_heatmap.py` may exit with an error if `pyvista` is not installed (dependency); the wrapper itself is correct and invokes the pipeline.

---

## 7. Verify docs don’t reference deleted files

```bash
rg "scripts/" docs -n
```

All `scripts/` references should point to existing paths under `scripts/2d/` or `scripts/3d/` (or `scripts/plot_examples/`, `scripts/overview/`). No references to removed files.

---

## 8. Summary

| Item | Status |
|------|--------|
| pipelines/2d, pipelines/3d | Single source of truth (unchanged) |
| scripts/2d, scripts/3d | Thin wrappers only (runpy); no duplicated logic |
| Files removed | scripts/overview/.gitkeep only |
| Docs | OVERVIEW.md placeholder/link fixes; no broken commands |
| .gitkeep | Only in scripts/tests/ (intentionally empty) |

Both pipelines remain intact and usable from the command line as before.
