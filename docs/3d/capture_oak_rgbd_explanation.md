# Explanation of `capture_oak_rgbd.py`

This document explains the 3D pipeline script that records **RGB + aligned depth** frames from a **Luxonis OAK-D Lite** (or any OAK-D variant) using the DepthAI SDK, saving them in the format expected by `open3d_reconstruct.py`.

---

## Overview

The OAK-D Lite does **not** present itself like a normal webcam + depth sensor. Instead, the **DepthAI** SDK (`depthai`) builds an on-device pipeline: stereo mono cameras produce a depth map which is aligned to the RGB camera, and both streams are sent to the host over USB.

**`capture_oak_rgbd.py`** configures this pipeline, streams synchronised colour and depth frames, and saves them to disk in a layout Open3D can consume directly.

**Hardware required:** OAK-D Lite (or OAK-D / OAK-D Pro), USB 3 cable (USB-C to USB-A or USB-C to USB-C), Mac or Raspberry Pi.

**Output directory:** `oak_capture/` (created next to the script), containing:

```
oak_capture/
├── color/
│   ├── color_000000.jpg
│   ├── color_000001.jpg
│   └── ...
├── depth/
│   ├── depth_000000.png   # 16-bit PNG, depth in millimetres
│   ├── depth_000001.png
│   └── ...
└── timestamps.csv          # idx, t_wall_s, t_device_ms
```

---

## What it does

1. **Create output folders:** `oak_capture/color/` and `oak_capture/depth/` (created if they don't exist).
2. **Build DepthAI pipeline:**
   - **Color camera** — 1080p, BGR, non-interleaved.
   - **Stereo pair** — left + right mono cameras at 400p.
   - **StereoDepth node** — `HIGH_DENSITY` preset, **depth aligned to RGB** so colour and depth pixels correspond, output resized to 640 x 400.
3. **Stream loop:** Grabs the next RGB and depth frame from the device queues. When both are available:
   - Saves the colour frame as `color_NNNNNN.jpg`.
   - Saves the depth frame as `depth_NNNNNN.png` (16-bit, values in **millimetres**).
   - Writes a row to `timestamps.csv` with the frame index, wall-clock time (`time.time()`), and the device-side timestamp in milliseconds.
   - Shows a live preview of both streams (press **q** to stop).
4. **Finish:** Prints a summary line with the output path.

---

## Key parameters (in-script constants)

| Parameter | Value | Notes |
|-----------|-------|-------|
| RGB resolution | 1080p | Full colour; downscale later if needed. |
| Mono resolution | 400p | Stereo input; affects depth quality. |
| Depth output size | 640 x 400 | Kept small for first-run speed; increase for higher quality. |
| Stereo preset | `HIGH_DENSITY` | Denser disparity; alternatives: `HIGH_ACCURACY`. |
| Depth alignment | RGB socket | Depth pixels map 1-to-1 to colour pixels. |
| Depth format | 16-bit uint16 PNG | Values in millimetres; Open3D reads with `depth_scale=1000.0`. |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `depthai` | Luxonis OAK-D SDK — builds and runs the on-device pipeline. |
| `opencv-python` | Frame display (`imshow`) and image I/O (`imwrite`). |
| `numpy` | Depth normalisation for preview window. |

Install (in a virtualenv):

```bash
pip install depthai opencv-python numpy
```

---

## Example usage

```bash
# Plug in OAK-D Lite via USB 3, then:
python3 pipelines/3d/capture_oak_rgbd.py

# Walk slowly around a small object for 30–60 seconds, then press 'q'.
# Frames are saved to oak_capture/
```

---

## Capture tips

- **USB 3 required.** Use a direct USB 3 port on your Mac — avoid hubs for the first test.
- **Move slowly.** Fast movement causes motion blur and depth holes; aim for a steady walking pace.
- **30–60 seconds** around a small object (box, chair, concrete block) is enough for a first reconstruction.
- **Lighting matters.** Stereo depth works best in well-lit environments with textured surfaces.
- **Frame rate.** The script captures as fast as the device delivers (typically 25–30 fps at these resolutions). For a slower capture, add a small `time.sleep()` in the loop or reduce queue sizes.

---

## Output format details

### `timestamps.csv`

| Column | Type | Description |
|--------|------|-------------|
| `idx` | int | Zero-based frame index. |
| `t_wall_s` | float | Wall-clock time (`time.time()`) at save. |
| `t_device_ms` | int | OAK device-side timestamp in milliseconds (monotonic). |

### Colour images

- **Format:** JPEG (`.jpg`), BGR uint8, 1080p (or whatever the colour camera outputs).
- **Naming:** `color_000000.jpg`, `color_000001.jpg`, ...

### Depth images

- **Format:** 16-bit PNG (`.png`), uint16, values in **millimetres**.
- **Size:** 640 x 400 (matches the `setOutputSize` call).
- **Naming:** `depth_000000.png`, `depth_000001.png`, ...
- **Zero values** mean no depth data (occluded, too far, or stereo mismatch).

---

## Relation to other 3D scripts

- **After capture:** Run **`open3d_reconstruct.py`** on the `oak_capture/` folder to build a 3D mesh via TSDF volume integration. See [open3d_reconstruct_explanation.md](open3d_reconstruct_explanation.md).
- **Future integration:** The trajectory (poses) produced by Open3D odometry or a stronger SLAM backend (RTAB-Map, ORB-SLAM3) can be exported as `trajectory.csv` and fed into **`fuse_mag_with_trajectory.py`** for magnetic fusion.
- **Alternative trajectory sources:** If you use Polycam or RTAB-Map for scanning instead of the OAK-D, use `polycam_raw_to_trajectory.py` or `rtabmap_poses_to_trajectory.py`.

See [PIPELINE_3D.md](PIPELINE_3D.md) for the full 3D runbook.
