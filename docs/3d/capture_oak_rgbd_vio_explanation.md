# Explanation of `capture_oak_rgbd_vio.py`

This document explains the VIO (Visual-Inertial Odometry) capture script that records **RGB + aligned depth + IMU-derived trajectory** from a **Luxonis OAK-D Lite**, producing a stable device-side trajectory alongside the frame data.

---

## Overview

Standard frame-to-frame RGB-D odometry (used by `open3d_reconstruct.py` in `odom` mode) frequently produces "confetti" meshes on textureless surfaces because it cannot find enough visual features to track. **`capture_oak_rgbd_vio.py`** solves this by computing camera poses during capture using the OAK-D Lite's onboard IMU (BMI270 gyroscope + accelerometer) for rotation, and depth-based optical flow for translation.

The resulting `trajectory_device.csv` is automatically detected by `open3d_reconstruct.py` and used instead of odometry, producing far more coherent 3D reconstructions.

**Hardware required:** OAK-D Lite with IMU (BMI270), USB 3 cable, Mac or Raspberry Pi. Kickstarter-era OAK-D Lite units without an IMU will fall back to depth-only mode with a clear warning.

**Output directory** (`--out`):

```
$RUN_DIR/raw/oak_rgbd/
├── color/
│   ├── color_000000.jpg
│   └── ...
├── depth/
│   ├── depth_000000.png       # 16-bit PNG, depth in millimetres
│   └── ...
├── timestamps.csv              # idx, t_wall_s, t_rgb_dev_ms, t_depth_dev_ms
├── intrinsics.json             # fx, fy, cx, cy, width, height, source
├── trajectory_device.csv       # t_rel_s, x, y, z, qx, qy, qz, qw
└── imu_raw.csv                 # (optional, with --save-imu)
```

---

## What it does

1. **Parse arguments:** `--out` (required), `--fps`, `--no-preview`, `--save-imu`, `--imu-alpha`.
2. **Build DepthAI pipeline:** Same RGB + stereo depth setup as `capture_oak_rgbd.py`, plus an **IMU node** streaming `ACCELEROMETER_RAW` and `GYROSCOPE_RAW` at 200 Hz.
3. **IMU fallback:** If the pipeline with IMU fails (e.g. no IMU hardware), the script rebuilds the pipeline without IMU and continues in depth-only mode.
4. **Save intrinsics:** Reads camera calibration from the device and writes `intrinsics.json`.
5. **Main capture loop:**
   - **Drain IMU queue:** Processes all pending IMU packets. Each packet updates the `IMUTracker` complementary filter, which integrates gyroscope angular velocity and corrects pitch/roll drift using the accelerometer's gravity vector.
   - **Get frame pair:** Grabs the next synchronised RGB + depth frame.
   - **Save images:** Writes colour JPEG and 16-bit depth PNG, plus a row to `timestamps.csv`.
   - **Estimate translation:** The `TranslationEstimator` tracks features between consecutive grayscale frames using Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`), unprojects matched features to 3D using the depth map, and solves for translation given the known rotation from the IMU. A robust median estimator filters outliers.
   - **Accumulate pose:** Position is updated by adding the world-frame translation delta. Orientation comes directly from the IMU tracker.
   - **Record pose:** Each frame's `(t_rel_s, x, y, z, qx, qy, qz, qw)` is buffered.
   - **Preview:** Shows live RGB and depth windows unless `--no-preview`.
6. **Clean shutdown:** Handles `Ctrl+C` (SIGINT) gracefully — always flushes `timestamps.csv` and writes `trajectory_device.csv` before exiting.
7. **Write trajectory:** Filters out any NaN/inf poses and writes `trajectory_device.csv`.

---

## How the VIO works

### Orientation (IMU complementary filter)

The `IMUTracker` class fuses gyroscope and accelerometer data:

1. **Initialisation:** On the first IMU sample, the gravity reference direction is recorded from the accelerometer.
2. **Gyroscope integration:** Angular velocity is integrated over `dt` to produce a rotation delta quaternion, which is composed with the current orientation.
3. **Accelerometer correction:** The predicted gravity direction (from the gyro-integrated orientation) is compared to the measured gravity direction (from the accelerometer). A small correction quaternion is computed and blended in using spherical linear interpolation (SLERP), weighted by `alpha` (default 0.98 = 98% trust in gyro, 2% correction from accelerometer).

This corrects pitch and roll drift while preserving smooth gyroscope tracking. Yaw drifts slowly (~0.5°/min for BMI270) but is acceptable for 30–60 second scans.

### Translation (depth optical flow)

The `TranslationEstimator` class computes frame-to-frame displacement:

1. **Feature detection:** `cv2.goodFeaturesToTrack` finds up to 500 corner features.
2. **Optical flow tracking:** `cv2.calcOpticalFlowPyrLK` tracks features from the previous frame to the current frame.
3. **3D unprojection:** Matched 2D features are lifted to 3D using the depth map and camera intrinsics: `X = (u - cx) * d / fx`, `Y = (v - cy) * d / fy`, `Z = d`.
4. **Translation solve:** Given the known rotation from the IMU (`R_prev`, `R_curr`), the world-frame translation delta is: `dt = R_prev @ P_cam_prev - R_curr @ P_cam_curr` for each correspondence. The median across all correspondences provides a robust estimate.
5. **Outlier rejection:** If the standard deviation of translation samples exceeds 0.5 m, the translation is zeroed (considered unreliable).

### Quaternion convention

All quaternions use the `[x, y, z, w]` (Hamilton) convention, matching the `trajectory.csv` format used throughout the pipeline.

---

## CLI arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--out` | Yes | Output directory (e.g. `$RUN_DIR/raw/oak_rgbd`). |
| `--fps` | No | Camera FPS (default: 15). |
| `--no-preview` | No | Disable OpenCV preview windows (headless / Pi). |
| `--save-imu` | No | Write raw IMU readings to `imu_raw.csv` (for debugging / offline reprocessing). |
| `--imu-alpha` | No | Complementary filter gyro trust factor, 0–1 (default: 0.98). Lower values trust the accelerometer more (noisier short-term, less drift long-term). |

---

## Example usage

```bash
# Standard capture (VIO trajectory + frames)
python3 pipelines/3d/capture_oak_rgbd_vio.py --out "$RUN_DIR/raw/oak_rgbd"

# Headless capture on Raspberry Pi
python3 pipelines/3d/capture_oak_rgbd_vio.py --out "$RUN_DIR/raw/oak_rgbd" --no-preview

# Save raw IMU data for debugging
python3 pipelines/3d/capture_oak_rgbd_vio.py --out "$RUN_DIR/raw/oak_rgbd" --save-imu

# Then reconstruct (auto-detects trajectory_device.csv)
./tools/3d/run_all_3d.sh --latest --camera-only
```

---

## Output format details

### `trajectory_device.csv`

| Column | Type | Description |
|--------|------|-------------|
| `t_rel_s` | float | Time relative to first frame (seconds). |
| `x`, `y`, `z` | float | Camera position in world frame (metres). |
| `qx`, `qy`, `qz`, `qw` | float | Camera orientation quaternion (Hamilton, `[x,y,z,w]`). |

One row per captured frame (same order as `color_NNNNNN.jpg`). NaN/inf poses are filtered out.

### `imu_raw.csv` (optional, `--save-imu`)

| Column | Type | Description |
|--------|------|-------------|
| `t_dev_s` | float | OAK device timestamp (seconds, monotonic). |
| `ax`, `ay`, `az` | float | Accelerometer readings (m/s²). |
| `gx`, `gy`, `gz` | float | Gyroscope readings (rad/s). |

Recorded at ~200 Hz. Useful for offline IMU analysis or reprocessing with different filter parameters.

---

## Key parameters (in-script constants)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Capture resolution | 640 × 400 | Colour and depth, aligned. |
| IMU rate | 200 Hz | Accelerometer + gyroscope. |
| Depth range | 0.1–3.0 m | Points outside this range are ignored for translation estimation. |
| Optical flow corners | 500 max | `cv2.goodFeaturesToTrack` parameter. |
| Min correspondences | 5 | Below this, translation estimation is skipped for the frame. |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `depthai` | Luxonis OAK-D SDK — builds and runs the on-device pipeline + IMU. |
| `opencv-python` | Optical flow (`calcOpticalFlowPyrLK`), feature detection, image I/O. |
| `numpy` | Quaternion math, 3D geometry, array operations. |

Install (in a virtualenv):

```bash
pip install depthai opencv-python numpy
```

---

## When to use this vs `capture_oak_rgbd.py`

| Scenario | Use |
|----------|-----|
| Textureless surfaces (concrete, drywall) | `capture_oak_rgbd_vio.py` |
| Short captures with good texture | Either works; VIO is still better |
| OAK-D Lite without IMU (Kickstarter units) | `capture_oak_rgbd.py` |
| Maximum capture speed (minimal processing) | `capture_oak_rgbd.py` |

---

## Relation to other 3D scripts

- **After capture:** Run `open3d_reconstruct.py` with `--pose-source auto` (default) — it automatically detects `trajectory_device.csv` and uses the device poses instead of computing odometry.
- **One-command pipeline:** `./tools/3d/run_all_3d.sh --latest --camera-only` handles everything automatically.
- **Fallback:** If VIO was not used (no `trajectory_device.csv`), reconstruction falls back to frame-to-frame RGB-D odometry.

See [PIPELINE_3D.md](PIPELINE_3D.md) for the full 3D runbook.
