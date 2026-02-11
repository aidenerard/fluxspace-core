# Explanation of `open3d_reconstruct.py`

This document explains the 3D pipeline script that takes the RGB + depth frames captured by `capture_oak_rgbd.py` and reconstructs a **3D triangle mesh** using Open3D's TSDF volume integration and frame-to-frame RGB-D odometry.

---

## Overview

After capturing an RGB-D sequence with the OAK-D Lite (see [capture_oak_rgbd_explanation.md](capture_oak_rgbd_explanation.md)), you have a folder of colour images and aligned 16-bit depth images. **`open3d_reconstruct.py`** loads these frames, estimates camera motion with frame-to-frame odometry, integrates every frame into a TSDF (Truncated Signed Distance Function) volume, and extracts a coloured triangle mesh saved as `open3d_mesh.ply`.

**Input:** `oak_capture/` directory containing `color/color_*.jpg` and `depth/depth_*.png` (produced by `capture_oak_rgbd.py`).

**Output:** `oak_capture/open3d_mesh.ply` — a coloured triangle mesh viewable in Open3D, MeshLab, Blender, or any PLY viewer.

---

## What it does

1. **Load frames:** Reads all `color_*.jpg` and `depth_*.png` files from `oak_capture/`, sorted by filename (frame order).
2. **Camera intrinsics:** Uses a rough pinhole model (fx = fy = 600, cx = 320, cy = 200, image size 640 x 400). These are **approximate** — see "Improving quality" below for how to use real OAK-D calibration.
3. **TSDF volume:** Creates a `ScalableTSDFVolume` with 1 cm voxels and 4 cm truncation distance, using RGB8 colour.
4. **Frame-to-frame odometry loop:**
   - For each consecutive pair of RGB-D images, computes the relative transform using `compute_rgbd_odometry` with the hybrid Jacobian (combines intensity and depth terms).
   - Chains the relative transforms to maintain a running camera pose.
   - If odometry fails for a frame (e.g. too much motion blur), the previous pose is kept (graceful degradation).
5. **TSDF integration:** Each frame is integrated into the volume at its estimated pose.
6. **Mesh extraction:** After all frames, extracts a triangle mesh from the TSDF volume, computes vertex normals, and saves to `open3d_mesh.ply`.
7. **Visualisation:** Opens an Open3D viewer to display the resulting mesh.

---

## Key parameters (in-script constants)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Image size | 640 x 400 | Must match `capture_oak_rgbd.py` output depth size. |
| `fx`, `fy` | 600.0 | Approximate focal length in pixels. Not calibrated. |
| `cx`, `cy` | 320.0, 200.0 | Principal point (image centre). |
| `voxel_length` | 0.01 (1 cm) | TSDF voxel size. Increase to 0.02–0.05 if processing is slow. |
| `sdf_trunc` | 0.04 (4 cm) | Truncation distance. Typically 3–5x voxel size. |
| `depth_scale` | 1000.0 | Depth PNG values are in mm; divides by 1000 to get metres. |
| `depth_trunc` | 3.0 m | Ignore depth values beyond 3 metres. |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `open3d` | RGB-D odometry, TSDF integration, mesh extraction, visualisation. |
| `numpy` | Pose matrix operations. |

Install (in the same virtualenv as the capture script):

```bash
pip install open3d numpy
```

---

## Example usage

```bash
# After running capture_oak_rgbd.py and collecting frames in oak_capture/:
python3 pipelines/3d/open3d_reconstruct.py

# Progress prints every 30 frames; viewer opens when done.
# Mesh saved to oak_capture/open3d_mesh.ply
```

---

## Understanding the output

- **`open3d_mesh.ply`** is a standard PLY mesh file with vertex positions, normals, and colours.
- Open it in **MeshLab**, **Blender**, **CloudCompare**, or any PLY viewer.
- The mesh quality depends on: camera intrinsics accuracy, depth quality, capture speed, and scene texture.
- For a first run, expect a **rough but recognisable** reconstruction. Quality improves significantly with real calibration and slower capture.

---

## Improving quality

### 1. Use real camera intrinsics (biggest improvement)

The script currently uses approximate values. To get the real OAK-D calibration:

```python
import depthai as dai

with dai.Device() as device:
    calib = device.readCalibration()
    intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, 640, 400)
    # intrinsics is a 3x3 matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    print(intrinsics)
```

Then replace `fx`, `fy`, `cx`, `cy` in the script with the real values.

### 2. Use a stronger SLAM backend

Frame-to-frame odometry accumulates drift over time. For larger scenes or longer captures, use a SLAM system like **RTAB-Map** or **ORB-SLAM3** to compute more accurate poses, then feed the trajectory into the TSDF integration (or into `fuse_mag_with_trajectory.py`).

### 3. Tune TSDF parameters

- **Smaller voxels** (e.g. 0.005) give finer detail but use more memory and time.
- **Larger truncation** helps with noisy depth but can smear thin structures.
- **Lower `depth_trunc`** filters out unreliable far-range depth.

### 4. Loop closure / global optimisation

Open3D supports pose-graph optimisation and multi-scale registration. Adding these on top of the basic odometry loop would reduce drift for longer sequences.

---

## Relation to other 3D scripts

- **Before:** Run **`capture_oak_rgbd.py`** to record the RGB-D frames. See [capture_oak_rgbd_explanation.md](capture_oak_rgbd_explanation.md).
- **After (future):** The camera poses computed by this script can be exported as a `trajectory.csv` and fed into **`fuse_mag_with_trajectory.py`** to fuse magnetometer data with the 3D geometry.
- **Alternative workflows:** Instead of OAK-D + Open3D, you can use Polycam or RTAB-Map for scanning and trajectory extraction (`polycam_raw_to_trajectory.py`, `rtabmap_poses_to_trajectory.py`).

See [PIPELINE_3D.md](PIPELINE_3D.md) for the full 3D runbook.
