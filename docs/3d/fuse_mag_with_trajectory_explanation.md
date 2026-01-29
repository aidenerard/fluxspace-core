# Explanation of `fuse_mag_with_trajectory.py`

This document explains the 3D pipeline script that fuses a timestamped magnetometer log with a camera trajectory to produce **world-frame** magnetic samples (`mag_world.csv`).

---

## Overview

During a 3D scan, the magnetometer is rigidly mounted relative to the phone (or camera rig). The phone provides a trajectory (position + orientation over time); the magnetometer provides field samples with timestamps. **`fuse_mag_with_trajectory.py`** aligns these by time, applies the magnetometer‑to‑camera extrinsics (translation and optional rotation), and writes each mag sample with its **world-frame** position `(x, y, z)` and a chosen value (e.g. baseline‑subtracted magnitude).

**Inputs:** `trajectory.csv` (from Polycam or RTAB‑Map), `mag_run.csv` (from `mag_to_csv_v2` or `mag_calibrate_zero_logger`), `extrinsics.json` (translation and optionally rotation from camera to magnetometer).

**Output:** `mag_world.csv` with columns `t_rel_s`, `x`, `y`, `z`, `value`, `value_type`.

---

## What it does

1. **Load inputs:** trajectory (pose per timestamp), mag log (samples with `t_rel_s`), extrinsics (translation `[x,y,z]` in meters; optional quaternion `[x,y,z,w]`).
2. **Time alignment:** For each mag sample timestamp, find the corresponding camera pose:
   - **Nearest‑neighbor** (default): use the closest trajectory pose by time.
   - **`--interpolate`:** linearly interpolate position between the two bracketing poses; optionally use the earlier pose’s quaternion to rotate the extrinsics vector.
3. **Apply extrinsics:** `world_mag_position = pose_position + pose_rotation @ translation_m` (or `+ translation_m` if no rotation). The translation is from camera frame to magnetometer frame; the script resolves the mag position in world frame.
4. **Choose value:** `--value-type` selects which scalar to write:
   - **`zero_mag`** (default): baseline‑subtracted magnitude. Prefers `zero_mag` from `mag_calibrate_zero_logger`; otherwise subtracts median of `b_total` / `B_total`.
   - **`raw`** / **`b_total`**: raw magnitude.
   - **`corr`**: corrected magnitude (e.g. after calibration, before baseline subtract).
5. **Write `mag_world.csv`:** one row per mag sample with `t_rel_s`, `x`, `y`, `z`, `value`, `value_type`.

---

## CLI arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--trajectory` | Yes | Path to `trajectory.csv` (`t_rel_s`, `x`, `y`, `z`, `qx`, `qy`, `qz`, `qw`). |
| `--mag` | Yes | Path to mag CSV from `mag_to_csv_v2` or `mag_calibrate_zero_logger`. |
| `--extrinsics` | Yes | Path to `extrinsics.json` with `translation_m` `[x,y,z]`; optional `rotation_quat_xyzw` `[x,y,z,w]`. |
| `--out` | No | Output path. Default: `<trajectory_dir>/mag_world.csv`. |
| `--value-type` | No | `zero_mag` (default), `raw`, `corr`, or `b_total`. |
| `--interpolate` | No | Linearly interpolate pose at mag timestamps instead of nearest‑neighbor. |

---

## Example usage

```bash
# Basic run (nearest‑neighbor, zero_mag)
python3 pipelines/3d/fuse_mag_with_trajectory.py \
  --trajectory "$RUN_DIR/processed/trajectory.csv" \
  --mag "$RUN_DIR/raw/mag_run.csv" \
  --extrinsics "$RUN_DIR/raw/extrinsics.json" \
  --out "$RUN_DIR/processed/mag_world.csv"

# Interpolate pose at mag timestamps
python3 pipelines/3d/fuse_mag_with_trajectory.py \
  --trajectory "$RUN_DIR/processed/trajectory.csv" \
  --mag "$RUN_DIR/raw/mag_run.csv" \
  --extrinsics "$RUN_DIR/raw/extrinsics.json" \
  --out "$RUN_DIR/processed/mag_world.csv" \
  --interpolate

# Use raw magnitude instead of baseline‑subtracted
python3 pipelines/3d/fuse_mag_with_trajectory.py \
  --trajectory "$RUN_DIR/processed/trajectory.csv" \
  --mag "$RUN_DIR/raw/mag_run.csv" \
  --extrinsics "$RUN_DIR/raw/extrinsics.json" \
  --value-type raw
```

---

## extrinsics.json

- **`translation_m`** (required): `[x, y, z]` in meters, from camera/origin to magnetometer in camera frame.
- **`rotation_quat_xyzw`** (optional): `[x, y, z, w]` quaternion for additional rotation from camera to mag frame. If omitted, only translation is applied.

---

## Relation to other 3D scripts

- **Before:** `polycam_raw_to_trajectory` or `rtabmap_poses_to_trajectory` → `trajectory.csv`; `mag_to_csv_v2` or `mag_calibrate_zero_logger` → `mag_run.csv`.
- **After:** `mag_world_to_voxel_volume` reads `mag_world.csv` to build the 3D voxel volume.

See [PIPELINE_3D.md](PIPELINE_3D.md) for the full 3D runbook.
