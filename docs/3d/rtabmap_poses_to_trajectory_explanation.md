# Explanation of `rtabmap_poses_to_trajectory.py`

This document explains the 3D pipeline script that converts **RTAB‑Map exported poses** (TUM or KITTI format) into **`trajectory.csv`** (timestamped camera poses) for use in magnetometer fusion.

---

## Overview

RTAB‑Map can export camera poses as a text file in **TUM** or **KITTI** format. **`rtabmap_poses_to_trajectory.py`** reads that file, converts poses to a unified format (position + quaternion), normalizes time to `t_rel_s` starting at 0, and writes **`trajectory.csv`** with columns `t_rel_s`, `x`, `y`, `z`, `qx`, `qy`, `qz`, `qw`.

**Input:** Path to the RTAB‑Map poses file (TUM or KITTI).

**Output:** `trajectory.csv`. Default path: `processed/trajectory.csv` when the input file is under `raw/`, otherwise `trajectory.csv` in the input file’s directory.

---

## What it does

1. **Format detection:** With `--format auto` (default), inspects the first non‑empty, non‑comment lines to guess **TUM** vs **KITTI**. You can force `--format TUM` or `--format KITTI`.
2. **TUM:** Lines like `timestamp tx ty tz qx qy qz qw` (eight numbers). Comments (`#`) and short lines are skipped. `t_rel_s` is taken from the timestamp; output is normalized so min timestamp → 0.
3. **KITTI:** Four lines per pose (3×4 matrix rows, then blank). Translation from last column; 3×3 rotation converted to quaternion via standard matrix‑to‑quaternion formula. No timestamp in file → synthetic `t_rel_s` from frame index (1/30 s step).
4. **Normalize `t_rel_s`:** Subtract minimum so time starts at 0.
5. **Write** `trajectory.csv` and print pose count and `t_rel_s` range.

---

## CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--in` | (required) | Path to RTAB‑Map poses file. |
| `--out` | (auto) | Output `trajectory.csv` path. |
| `--format` | `auto` | `TUM`, `KITTI`, or `auto` (detect from content). |

---

## Example usage

```bash
# Auto-detect format
python3 pipelines/3d/rtabmap_poses_to_trajectory.py \
  --in "$RUN_DIR/raw/rtabmap_poses.txt" \
  --out "$RUN_DIR/processed/trajectory.csv"

# Force KITTI
python3 pipelines/3d/rtabmap_poses_to_trajectory.py \
  --in "$RUN_DIR/raw/rtabmap_poses.txt" \
  --out "$RUN_DIR/processed/trajectory.csv" \
  --format KITTI
```

---

## Relation to other 3D scripts

- **Output** `trajectory.csv` is used by **`fuse_mag_with_trajectory`** together with the mag log and **`extrinsics.json`**.
- **Alternative:** **`polycam_raw_to_trajectory`** produces the same CSV shape from Polycam Raw Data export.

See [PIPELINE_3D.md](PIPELINE_3D.md) for the full 3D runbook.
