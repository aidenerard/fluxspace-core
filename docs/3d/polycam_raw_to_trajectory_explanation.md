# Explanation of `polycam_raw_to_trajectory.py`

This document explains the 3D pipeline script that converts a **Polycam Raw Data** export folder into **`trajectory.csv`** (timestamped camera poses: position + quaternion) for use in magnetometer fusion.

---

## Overview

When you export **Raw Data** from the Polycam app (LiDAR or other 3D scan), you get a folder with camera poses—typically in **`cameras.json`** or **`corrected_cameras.json`**. **`polycam_raw_to_trajectory.py`** reads that export, extracts per‑frame position and orientation, and writes a **`trajectory.csv`** with columns `t_rel_s`, `x`, `y`, `z`, `qx`, `qy`, `qz`, `qw`.

**Input:** Path to the Polycam Raw Data export folder (containing `cameras.json` or `corrected_cameras.json`).

**Output:** `trajectory.csv` with normalized `t_rel_s` (relative time, starting at 0) and pose columns. Default output path: `processed/trajectory.csv` when the input folder is under `raw/`, otherwise `trajectory.csv` inside the input folder.

---

## What it does

1. **Locate JSON:** Looks for `cameras.json`, `corrected_cameras.json`, or `CorrectedCameras.json` in the given folder or one level down.
2. **Load frames:** Parses the JSON structure (array of frames, or object with `cameras` / `frames` / `poses` / `corrected_cameras`).
3. **Per frame:** Extracts position (`position`, `translation`, `pos`, `t`) and orientation (`rotation`, `orientation`, `quaternion`, `quat`, `q`). Supports both list and dict forms. Uses identity quaternion if rotation is missing.
4. **Timestamps:** Uses frame `timestamp` / `time` / `t` / `timestamp_sec` / `timestamp_ns` when present; otherwise derives relative time from frame index (assumes 30 fps) and prints a warning.
5. **Normalize `t_rel_s`:** Subtracts the minimum so time starts at 0.
6. **Write** `trajectory.csv` and print pose count and `t_rel_s` range.

---

## CLI arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--in` | Yes | Path to Polycam Raw Data export folder. |
| `--out` | No | Output `trajectory.csv` path. Default: `processed/trajectory.csv` next to `raw/`, or `<input_dir>/trajectory.csv`. |

---

## Example usage

```bash
# Input = Polycam export folder (e.g. in run raw/)
python3 pipelines/3d/polycam_raw_to_trajectory.py \
  --in "$RUN_DIR/raw/PolycamRawExport" \
  --out "$RUN_DIR/processed/trajectory.csv"
```

---

## Relation to other 3D scripts

- **Output** `trajectory.csv` is used by **`fuse_mag_with_trajectory`** together with the mag log and **`extrinsics.json`**.
- **Alternatives** that produce the same CSV shape:
  - **`rtabmap_poses_to_trajectory`** — from RTAB‑Map exported poses (TUM or KITTI).
  - **`open3d_reconstruct`** — from OAK-D Lite RGB-D capture + Open3D odometry.

See [PIPELINE_3D.md](PIPELINE_3D.md) for the full 3D runbook.
