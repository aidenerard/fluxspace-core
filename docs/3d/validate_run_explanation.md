# Explanation of `validate_run.py`

This document explains the smoke-test script that validates the integrity and completeness of a FluxSpace 3D run directory.

---

## Overview

After running the 3D pipeline, **`validate_run.py`** checks that all expected files exist, are non-empty, and contain valid data. It supports both full (camera + magnetometer) runs and camera-only runs.

This is useful for:
- Verifying a pipeline run completed successfully before moving on.
- Diagnosing which step failed if results look wrong.
- Automated CI/CD validation of pipeline outputs.

---

## What it checks

### Raw inputs

| Check | Full mode | Camera-only mode |
|-------|-----------|------------------|
| `raw/oak_rgbd/color/` — contains `.jpg` files | Required | Required |
| `raw/oak_rgbd/depth/` — contains `.png` files | Required | Required |
| `raw/oak_rgbd/timestamps.csv` — exists, non-empty | Required | Required |
| `raw/oak_rgbd/intrinsics.json` — exists | Warn if missing | Warn if missing |
| `raw/mag_run.csv` — exists, non-empty | Required | Skipped ("not required") |
| `raw/calibration.json` — exists | Warn if missing | Skipped |
| `raw/extrinsics.json` — exists | Warn if missing | Skipped |

### Processed outputs

| Check | Behaviour |
|-------|-----------|
| `processed/trajectory.csv` | Required. Checks row count, validates that x/y/z values are finite. Reports NaN count. |
| `processed/open3d_pcd_raw.ply` | Warn if missing. Checks file size > 1000 bytes. |
| `processed/open3d_mesh_raw.ply` or `open3d_mesh.ply` | Warn if missing. Accepts legacy name. |
| `processed/open3d_pcd_clean.ply` | Warn if missing (fail if `--require-clean`). |
| `processed/open3d_mesh_clean.ply` | Warn if missing (fail if `--require-clean`). |
| `processed/reconstruction_report.json` | Warn if missing. Parses JSON and reports internal warnings. |
| `processed/cleaning_report.json` | Warn if missing. Parses JSON and reports internal warnings. |

### Mag fusion + exports (full mode only)

| Check | Behaviour |
|-------|-----------|
| `processed/mag_world.csv` or `mag_world_m.csv` | Warn if missing. |
| `exports/volume.npz` | Warn if missing. Loads with numpy and reports volume shape. |

In camera-only mode these checks are skipped entirely with "not required" messages.

---

## CLI arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--run DIR` | Yes | Path to the run directory. |
| `--require-clean` | No | Fail (not just warn) if cleaned geometry outputs are missing. |
| `--camera-only` | No | Camera-only mode: skip mag/volume requirements. |

---

## Example usage

```bash
# Validate a full run (camera + magnetometer)
python3 pipelines/3d/validate_run.py --run data/runs/run_20260210_1430

# Validate a camera-only run
python3 pipelines/3d/validate_run.py --run data/runs/run_cam_only_20260210 --camera-only

# Strict validation: require clean geometry outputs
python3 pipelines/3d/validate_run.py --run data/runs/run_20260210_1430 --require-clean
```

---

## Output format

The script prints a structured report with status markers:

```
Validating: /path/to/run_20260210_1430
Mode: full (camera + mag)

[Raw inputs]
  OK   color/ — 450 files
  OK   depth/ — 450 files
  OK   timestamps.csv — 12345 bytes
  OK   intrinsics.json — present
  OK   mag_run.csv — 45678 bytes
  WARN calibration.json — missing (optional)
  WARN extrinsics.json — missing (optional)

[Processed outputs]
  OK   trajectory.csv — 448 rows, all valid
  OK   open3d_pcd_raw.ply — 2,345,678 bytes
  OK   open3d_mesh_raw.ply — 5,678,901 bytes
  OK   open3d_pcd_clean.ply — 1,234,567 bytes
  OK   open3d_mesh_clean.ply — 3,456,789 bytes
  OK   reconstruction_report.json — valid, no warnings
  OK   cleaning_report.json — valid, no warnings

PASSED — 2 warning(s)
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | All required checks passed (warnings are OK). |
| 1 | One or more required checks failed. |
| 2 | Invalid arguments (e.g. directory not found). |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Loading `volume.npz`, checking for NaN/inf in trajectory values. |

Standard library only beyond numpy — no Open3D required.

---

## Relation to other 3D scripts

- **After:** Run `validate_run.py` after `run_all_3d.sh` completes, or after individual pipeline steps.
- **Pipeline runner:** `run_all_3d.sh` does its own inline validation during execution; `validate_run.py` provides a standalone post-hoc check.
- **Camera-only:** Supports the same `--camera-only` flag as `run_all_3d.sh`.

See [PIPELINE_3D.md](PIPELINE_3D.md) for the full 3D runbook.
