# 3D Magnetic Fusion Pipeline — Runbook

End-to-end guide to produce: **(1)** 3D geometry (mesh/point cloud), **(2)** timestamped camera trajectory, **(3)** timestamped magnetometer log (calibrated + zero), **(4)** fused `mag_world.csv`, and **(5)** 3D magnetic anomaly heatmap (voxel volume + PyVista visualization).

---

## New structure (2D vs 3D)

- **Preferred:** Run Python from repo root: `python3 pipelines/2d/...` and `python3 pipelines/3d/...`. Run 3D shell scripts from `./tools/3d/` (e.g. `new_3d_scan.sh`, `backup_usb_3d.sh`).
- **`pipelines/2d/`** and **`pipelines/3d/`** = implementation (and preferred entrypoints for Python).
- **Legacy:** Wrappers under **`scripts/2d/`** and **`scripts/3d/`**, and shell scripts under **`scripts/3d/`**, still work; preferred usage is **`pipelines/2d/`**, **`pipelines/3d/`**, and **`tools/3d/`** (see [Legacy commands](#legacy-commands) at end).

---

## Pi setup for 3D (magnetometer + optional viz)

If you run the **magnetometer logger** (or full pipeline) on a Raspberry Pi, install dependencies with the 3D setup script (same venv as 2D, plus PyVista for visualization):

```bash
cd ~/fluxspace-core
chmod +x tools/3d/setup_pi.sh
./tools/3d/setup_pi.sh
```

**What it installs:** System packages (I2C, python3-venv, pip), enables I2C, creates/reuses `~/fluxenv`, and installs Python deps: numpy, pandas, matplotlib, scipy, scikit-learn, sparkfun-qwiic-mmc5983ma, **pyvista**. Reboot after first run if I2C was just enabled. See [raspberry_pi_setup.md](../raspberry_pi_setup.md) for full Pi setup and 3D usage.

**OAK-D Lite + Open3D (optional):** If you plan to use the OAK-D Lite for RGB-D capture and Open3D for reconstruction, add the `--with-oakd` flag:

```bash
./tools/3d/setup_pi.sh --with-oakd
```

This additionally installs `depthai`, `opencv-python`, `open3d`, and the `libusb` system package needed by DepthAI on Linux. On **Mac**, install these manually: `pip install depthai opencv-python open3d`.

---

## Quickstart (minimal first run)

```bash
# 1. Create run folder
export RUN_DIR="data/runs/run_$(date +%Y%m%d_%H%M)"
mkdir -p "$RUN_DIR"/{raw,processed,exports}

# 2. Capture: start magnetometer logger (on Pi), then scan with Polycam/RTAB-Map
#    Use mag_to_csv_v2.py (or mag_calibrate_zero_logger.py if you have it)
python3 pipelines/2d/mag_to_csv_v2.py --out "$RUN_DIR/raw/mag_run.csv" --hz 80 --units uT --samples 1

# 3. Export trajectory from your scan app (see Capture Day below), put files in $RUN_DIR/raw/

# 4. Processing (on Mac or Pi)
python3 pipelines/3d/polycam_raw_to_trajectory.py --in "$RUN_DIR/raw/PolycamRawExport" --out "$RUN_DIR/processed/trajectory.csv"
# OR: python3 pipelines/3d/rtabmap_poses_to_trajectory.py --in "$RUN_DIR/raw/rtabmap_poses.txt" --out "$RUN_DIR/processed/trajectory.csv"

python3 pipelines/3d/fuse_mag_with_trajectory.py \
  --trajectory "$RUN_DIR/processed/trajectory.csv" \
  --mag "$RUN_DIR/raw/mag_run.csv" \
  --extrinsics "$RUN_DIR/raw/extrinsics.json" \
  --out "$RUN_DIR/processed/mag_world.csv" \
  --value-type zero_mag

python3 pipelines/3d/mag_world_to_voxel_volume.py \
  --in "$RUN_DIR/processed/mag_world.csv" \
  --out "$RUN_DIR/exports/volume.npz" \
  --voxel-size 0.02 \
  --margin 0.1

python3 pipelines/3d/visualize_3d_heatmap.py \
  --in "$RUN_DIR/exports/volume.npz" \
  --out-dir "$RUN_DIR/exports" \
  --screenshot
```

(Use `--volume` as an alias for `--in` if you prefer.) Expected outputs: `trajectory.csv`, `mag_world.csv`, `volume.npz`, and `exports/heatmap_3d_screenshot.png`.

---

## Overview (text diagram)

```
[CAPTURE DAY]
  Option A: Phone (Polycam / RTAB-Map)          Option B: OAK-D Lite (DepthAI)
  Pi + MMC5983MA (rigid ruler ~30 cm)            + optional Pi + MMC5983MA
       |                                              |
       v                                              v
  3D scan app                                    capture_oak_rgbd.py
  (export Raw Data / .db)                         → oak_capture/ (color, depth, timestamps)
       |                                              |
       v                                              v
  mag_to_csv_v2.py (or mag_calibrate_zero_logger.py) for mag capture (both options)

[PROCESSING DAY]
  Option A:                                       Option B:
  polycam_raw_to_trajectory.py                    open3d_reconstruct.py
    OR rtabmap_poses_to_trajectory.py               → open3d_mesh.ply (3D mesh)
       |                                              (trajectory from odometry poses
       v                                               can be exported for mag fusion)
  trajectory.csv (t_rel_s, x, y, z, qx, qy, qz, qw)
       |
       +--------+ extrinsics.json (ruler: phone/camera -> mag frame)
       |
       v
  fuse_mag_with_trajectory.py  -->  mag_world.csv (t_rel_s, x, y, z, value, value_type)
       |
       v
  mag_world_to_voxel_volume.py  -->  volume.npz (3D grid + metadata)
       |
       v
  visualize_3d_heatmap.py  -->  slices, isosurfaces, screenshot PNG (optional HTML)
```

---

## Alternative storage: 3D scans (data/scans/*__3d)

You can keep 3D scan snapshots in a **separate tree** so they don’t mix with 2D runs:

- **2D pipeline:** Still uses `data/runs/` and `./tools/2d/new_run.sh`, `./tools/2d/backup_runs_to_usb.sh` (unchanged).
- **3D pipeline:** Use `data/scans/<RUN_ID>__3d/` (or `data/scans/<RUN_ID>__3d__<label>/`).

**Create a 3D scan snapshot** (copies current `data/raw`, `data/processed`, `data/exports` into a new scan folder):

```bash
./tools/3d/new_3d_scan.sh
./tools/3d/new_3d_scan.sh --label block01
```

**Examples:** `data/scans/01-29-2026_13-57__3d/`, `data/scans/01-29-2026_13-57__3d__block01/`

**Back up 3D scans to USB only:**

```bash
./tools/3d/backup_usb_3d.sh
```

Backs up `data/scans/` → `/media/fluxspace/FLUXSPACE/fluxspace_scans_backup/`. Mount/unmount USB as in the main runbook (see docs). The 2D runs backup is separate (`./tools/2d/backup_runs_to_usb.sh`).

---

## Folder structure per run

Use one folder per capture session. All commands below assume either:

- **Run root (generic):** `data/runs/run_YYYYMMDD_HHMM/` or a 3D scan folder `data/scans/<RUN_ID>__3d/`
- **Inputs (raw):** magnetometer CSV, trajectory source (Polycam folder or RTAB-Map export), `extrinsics.json`
- **Outputs:** `processed/` (trajectory.csv, mag_world.csv), `exports/` (volume.npz, screenshots, optional mesh)

```
data/runs/run_20250123_1430/
├── raw/
│   ├── mag_run.csv              # From mag_to_csv_v2.py or mag_calibrate_zero_logger.py
│   ├── extrinsics.json           # Ruler rig: translation (and optional rotation) phone -> mag
│   ├── PolycamRawExport/         # OR: Polycam Raw Data export folder
│   │   └── (cameras.json or corrected_cameras, etc.)
│   └── rtabmap_poses.txt        # OR: RTAB-Map "Export poses" (TUM format)
├── processed/
│   ├── trajectory.csv            # t_rel_s, x, y, z, qx, qy, qz, qw
│   └── mag_world.csv             # t_rel_s, x, y, z, value, value_type
└── exports/
    ├── volume.npz                # 3D voxel grid + origin, voxel_size, axes
    ├── heatmap_3d_screenshot.png
    └── (optional) mesh.ply, heatmap_3d.html
```

---

## Capture day

### 1. Setup

- **Rig:** Magnetometer (MMC5983MA) mounted ~30 cm (1 foot) from phone/camera on a rigid ruler; orientation fixed relative to phone or OAK-D.
- **Option A — Phone:** Polycam LiDAR (Developer Mode → Raw Data export) **or** RTAB-Map iOS; ensure you can export poses/trajectory.
- **Option B — OAK-D Lite:** Luxonis OAK-D Lite connected via USB 3 to Mac (or Pi 5). No phone needed for geometry — the OAK-D provides RGB + stereo depth.
- **Pi:** I2C enabled, `qwiic_mmc5983ma` installed; same venv as other fluxspace scripts.

### 2. Start magnetometer logger

**Recommended:** Use `mag_calibrate_zero_logger.py` so you get **calibration + zero (baseline) + logging** in one run. Run it *before* you start the 3D scan; it will prompt you to move the sensor for calibration, then hold still for baseline, then log continuously.

On the Pi (or Mac if sensor connected):

```bash
cd ~/fluxspace-core
source ~/fluxenv/bin/activate   # if using venv

export RUN_DIR="data/runs/run_$(date +%Y%m%d_%H%M)"
mkdir -p "$RUN_DIR/raw"

# Option A: Calibrate + zero + log (recommended)
python3 pipelines/3d/mag_calibrate_zero_logger.py \
  --out "$RUN_DIR/raw/mag_run.csv" \
  --hz 80 \
  --units uT \
  --calib-seconds 20 \
  --zero-seconds 3 \
  --save-cal "$RUN_DIR/raw/calibration.json"
```

**Option B:** Log only (no calibration/zero), e.g. for quick tests:

```bash
python3 pipelines/2d/mag_to_csv_v2.py \
  --out "$RUN_DIR/raw/mag_run.csv" \
  --hz 80 \
  --units uT \
  --samples 1
```

- Press **Enter** (or type `start` + Enter) just before starting the 3D scan.
- Optional: short distinctive motion (e.g. small shake) for alignment.
- At end: press Enter again (or type `end`), then **Ctrl+C** or `q` + Enter to stop.

If using **mag_calibrate_zero_logger.py** (capture-time calibration + zero): run that script instead; output path and CSV columns should match what `fuse_mag_with_trajectory.py` expects (see script `--help`).

### 3. Record 3D scan

- **Polycam:** Start LiDAR scan after the `start` marker; stop scan; use Developer Mode → **Export Raw Data** to a folder (e.g. copy to `$RUN_DIR/raw/PolycamRawExport`).
- **RTAB-Map:** Record session; export database to Mac; open with `rtabmap-databaseViewer` and **Export poses** (TUM format or as documented); save to `$RUN_DIR/raw/rtabmap_poses.txt` (or similar).
- **OAK-D Lite:** Run `capture_oak_rgbd.py` (see "OAK-D Lite capture + Open3D reconstruction" below). Walk slowly around the object for 30–60 seconds, then press **q** to stop. Frames are saved to `oak_capture/`.

### 4. Extrinsics and run folder

- Copy/create **extrinsics.json** into `$RUN_DIR/raw/` (see “Extrinsics (ruler rig)” below).
- Ensure `mag_run.csv`, trajectory source (Polycam folder or RTAB-Map file), and `extrinsics.json` are all under `$RUN_DIR/raw/`.

**Create extrinsics template (translation-only, 30 cm along +X):**  
`printf '%s\n' '{ "translation_m": [0.30, 0.0, 0.0], "rotation_quat_xyzw": null }' > "$RUN_DIR/raw/extrinsics.json"`  
Adjust `translation_m` to your measured ruler offset (meters).

---

## OAK-D Lite capture + Open3D reconstruction (Option B)

This is an alternative to the Polycam / RTAB-Map workflow above. Instead of a phone, you use a **Luxonis OAK-D Lite** for both RGB and depth, and **Open3D** for offline 3D reconstruction.

### Prerequisites

```bash
# In your virtualenv:
pip install depthai opencv-python numpy open3d
```

Plug the OAK-D Lite into your Mac with a **USB 3** cable (direct port, avoid hubs for the first test).

### Verify the camera

```bash
python -m depthai_demo
```

You should see a UI with colour and depth previews. If that works, the camera and cable are good.

### Step 1 — Capture RGB + depth frames

```bash
python3 pipelines/3d/capture_oak_rgbd.py
```

- Walk slowly around a small object (box, chair, concrete block) for **30–60 seconds**.
- Press **q** to stop.
- Frames are saved to `oak_capture/color/`, `oak_capture/depth/`, and `oak_capture/timestamps.csv`.
- Depth images are **16-bit PNG** with values in millimetres.

See [capture_oak_rgbd_explanation.md](capture_oak_rgbd_explanation.md) for full details.

### Step 2 — Reconstruct a 3D mesh

```bash
python3 pipelines/3d/open3d_reconstruct.py
```

- Reads `oak_capture/color/` and `oak_capture/depth/`.
- Runs frame-to-frame RGB-D odometry to estimate camera poses.
- Integrates all frames into a TSDF volume (1 cm voxels).
- Extracts a coloured triangle mesh → `oak_capture/open3d_mesh.ply`.
- Opens an Open3D viewer to display the result.

See [open3d_reconstruct_explanation.md](open3d_reconstruct_explanation.md) for full details.

### What to do next

1. **Use real camera intrinsics** from OAK-D calibration (major quality improvement) — see the explanation doc for how to extract them.
2. **Use a stronger SLAM backend** (RTAB-Map or ORB-SLAM3) for reduced drift on longer captures.
3. **Export the trajectory** (camera poses) from the odometry loop → save as `trajectory.csv` → feed into `fuse_mag_with_trajectory.py` to fuse magnetometer data with the 3D geometry.

### Platform notes

- **Pi 5:** Can run capture + lightweight odometry on-device.
- **Pi 4:** Recommended to record frames on the Pi, then transfer `oak_capture/` to a Mac for Open3D reconstruction.

---

## Processing day

All commands use the same `RUN_DIR`. Run from repo root.

### Step 1: Trajectory CSV

**Option A — Polycam Raw Data**

```bash
python3 pipelines/3d/polycam_raw_to_trajectory.py \
  --in "$RUN_DIR/raw/PolycamRawExport" \
  --out "$RUN_DIR/processed/trajectory.csv"
```

If timestamps are missing, script uses frame order and prints a warning; output columns: `t_rel_s, x, y, z, qx, qy, qz, qw`.

**Option B — RTAB-Map poses**

```bash
python3 pipelines/3d/rtabmap_poses_to_trajectory.py \
  --in "$RUN_DIR/raw/rtabmap_poses.txt" \
  --out "$RUN_DIR/processed/trajectory.csv" \
  --format TUM
```

Output: same columns, `t_rel_s` normalized from first pose.

### Step 2: Fuse magnetometer with trajectory

```bash
python3 pipelines/3d/fuse_mag_with_trajectory.py \
  --trajectory "$RUN_DIR/processed/trajectory.csv" \
  --mag "$RUN_DIR/raw/mag_run.csv" \
  --extrinsics "$RUN_DIR/raw/extrinsics.json" \
  --out "$RUN_DIR/processed/mag_world.csv" \
  --value-type zero_mag \
  --interpolate
```

- **Time alignment:** Nearest-neighbor by default; `--interpolate` uses linear interpolation of pose at each mag timestamp.
- **Extrinsics:** Translation (and optional quaternion rotation) from phone/camera frame to magnetometer frame; applied to pose position (and optionally field vector if rotation present).
- **value_type:** `zero_mag` (baseline-subtracted magnitude), or `raw` / `corr` if your mag CSV has those columns.

Expected output: `mag_world.csv` with columns `t_rel_s, x, y, z, value, value_type`.

### Step 3: Voxel volume

**Option A — IDW (original):**

```bash
python3 pipelines/3d/mag_world_to_voxel_volume.py \
  --in "$RUN_DIR/processed/mag_world.csv" \
  --out "$RUN_DIR/exports/volume.npz" \
  --voxel-size 0.02 \
  --margin 0.1 \
  --method idw
```

**Option B — IDW or GPR + gradient (single volume.npz):**

```bash
python3 pipelines/3d/mag_world_to_voxel_volumeV2_gpr.py \
  --in "$RUN_DIR/processed/mag_world.csv" \
  --out "$RUN_DIR/exports/volume.npz" \
  --method idw
# or --method gpr  (slower; adds std uncertainty)
```

- Bounds from data + margin/pad. IDW is fast; GPR is O(N³), use `--max-points` if needed. Both write `volume` and `grad`; GPR also writes `std`.

### Step 4: 3D visualization

```bash
python3 pipelines/3d/visualize_3d_heatmap.py \
  --in "$RUN_DIR/exports/volume.npz" \
  --out-dir "$RUN_DIR/exports" \
  --screenshot
```

- Use `--volume` as alias for `--in` if you prefer. `--mode value | std | grad` to view main volume, uncertainty, or gradient. Headless: add `--no-show`. Slices: `--show-slices --save --no-show` to write slice PNGs to `--out-dir`.

---

## Timestamps and sync

- **Magnetometer:** `t_rel_s` = seconds since logger start; `t_unix_ns` = wall clock (for optional wall-clock sync).
- **Trajectory:** Scripts normalize to `t_rel_s` from first pose so both streams are “relative to session start.”
- **Sync strategy:**
  1. **Markers:** Use MARK rows (`start` / `end`) to trim and align segments.
  2. **Wall clock:** If phone and Pi are NTP-synced, matching `t_unix_ns` to camera timestamps can refine alignment.
  3. **Motion cue:** A short distinctive motion (e.g. shake) visible in both trajectory and mag can be used to nudge alignment manually or in a future script.

---

## Extrinsics (ruler rig)

**extrinsics.json** describes the magnetometer frame in the phone/camera frame (or vice versa, as documented in the script).

**Translation-only (V1):**

```json
{
  "comment": "Magnetometer position in phone frame; ruler ~30 cm along phone +X",
  "translation_m": [0.30, 0.0, 0.0],
  "rotation_quat_xyzw": null
}
```

**With rotation (optional):**

```json
{
  "translation_m": [0.30, 0.0, 0.0],
  "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]
}
```

- **Measuring:** With phone and ruler fixed, define phone frame (e.g. +X forward). Measure from phone origin to magnetometer center in meters; that is `translation_m`. If the mag axes are not aligned with the phone, measure or estimate orientation and set `rotation_quat_xyzw` (x, y, z, w).
- **Convention:** Scripts assume **phone/camera frame → magnetometer frame**: apply translation (and rotation if given) to put the mag reading into the same frame as the trajectory (or vice versa as coded); see `fuse_mag_with_trajectory.py` docstring.

---

## Recommended capture procedure

1. Create run folder; start mag logger; add MARK `start`; start 3D scan.
2. Move steadily; avoid occlusions (LiDAR) and motion blur.
3. Optional: 1–2 s distinctive motion for alignment.
4. MARK `end`; stop scan; stop logger.
5. Export Raw Data (Polycam) or Export poses (RTAB-Map) into `$RUN_DIR/raw/`.
6. Copy `extrinsics.json` into `$RUN_DIR/raw/`.

---

## Recommended export formats

- **Polycam:** Developer Mode → Export Raw Data → folder containing `cameras.json` or `corrected_cameras` (or equivalent); script will look for known keys.
- **RTAB-Map:** Export poses in **TUM** format: `timestamp tx ty tz qx qy qz qw` (one line per pose); or KITTI if supported by script.

---

## Dependencies (Python 3.10+)

- **Required (all workflows):** `numpy`, `pandas`, `matplotlib`, `scipy` (interpolation), `scikit-learn` (GPR), `pyvista` (3D viz / screenshot).
- **OAK-D capture:** `depthai`, `opencv-python` (for `capture_oak_rgbd.py`).
- **Open3D reconstruction:** `open3d` (for `open3d_reconstruct.py`; also used for PLY/OBJ loading in other scripts).

**Mac / Linux:**

```bash
# Core pipeline:
pip install -U numpy pandas matplotlib scipy scikit-learn pyvista

# OAK-D capture + Open3D reconstruction (optional, for OAK-D workflow):
pip install depthai opencv-python open3d
```

**Pi:** Run `./tools/3d/setup_pi.sh` (installs the core deps plus I2C/sensor libs). For OAK-D on Pi, also install `depthai` and `opencv-python` in the same venv. See [raspberry_pi_setup.md](../raspberry_pi_setup.md).

If a script needs a missing dependency, it will print a clear error and the package name.

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| **Drift (mag vs trajectory)** | Use MARK rows to trim to same segment; check NTP on Pi and phone; consider motion-cue alignment. |
| **Phone interference** | Keep mag on rigid ruler away from phone; avoid covering LiDAR; post-process with baseline (zero_mag) or calibration. |
| **Wrong cropping** | Ensure trajectory and mag CSV cover the same time span; use `--t-min` / `--t-max` in fuse script if added. |
| **Slow voxel / fusion** | Reduce `--voxel-size` or use `--method griddata`; for fuse, disable `--interpolate` for faster nearest-neighbor only. |
| **Polycam: “No cameras found”** | Confirm export folder contains `cameras.json` or `corrected_cameras`; check script for supported keys. |
| **RTAB-Map: wrong columns** | Use `--format TUM` or KITTI; ensure file has timestamp + 7 numeric fields per line. |
| **PyVista / viz errors** | Install `pyvista`; on headless server run with `--screenshot` only (no GUI). |
| **Permission denied (shell scripts)** | `chmod +x tools/2d/*.sh tools/3d/*.sh` (or the specific script). |
| **scipy missing** | `pip install scipy` (required for IDW and griddata). |
| **scikit-learn missing** | `pip install scikit-learn` (required for GPR method). |
| **pyvista add_volume / scalar_range error** | Fixed in this repo: script uses `clim` (not `scalar_range`). Update the script or re-pull. |
| **Headless / no GUI** | Use `--screenshot --no-show` for 3D screenshot only; use `--show-slices --save --no-show` for slice PNGs. |
| **OAK-D not detected** | Ensure USB 3 cable is plugged directly into Mac (not a hub). Run `python -c "import depthai; print(depthai.Device.getAllAvailableDevices())"` to check. |
| **OAK-D capture slow / laggy** | Reduce colour resolution or depth output size in `capture_oak_rgbd.py`. Ensure USB 3 (not USB 2). |
| **Open3D mesh is empty or garbled** | Check that intrinsics roughly match the camera. Use real OAK-D calibration (see explanation doc). Ensure depth images are 16-bit PNG in mm. |
| **Open3D odometry drift** | Expected for long captures with frame-to-frame only. Use a SLAM backend for better poses. |
| **`depthai` install fails** | See [Luxonis docs](https://docs.luxonis.com/software/depthai/manual-install/). On macOS, `pip install depthai` should work. On Pi, may need `libusb` dev packages. |

---

## Expected outputs summary

| Step | Output | Columns / content |
|------|--------|-------------------|
| Trajectory | `processed/trajectory.csv` | `t_rel_s, x, y, z, qx, qy, qz, qw` |
| Fuse | `processed/mag_world.csv` | `t_rel_s, x, y, z, value, value_type` |
| Voxel | `exports/volume.npz` | 3D array + `origin`, `voxel_size`, axis arrays |
| Viz | `exports/heatmap_3d_screenshot.png` | Slice + isosurface view; optional HTML |

All scripts use **argparse**, **clear prints** of inputs/outputs, and write into the **run folder**; defaults are set for reproducible runs.

---

## Script explanations

Detailed docs for each 3D pipeline script:

| Script | Explanation |
|--------|-------------|
| **OAK-D capture + Open3D** | |
| `capture_oak_rgbd` | [capture_oak_rgbd_explanation.md](capture_oak_rgbd_explanation.md) |
| `open3d_reconstruct` | [open3d_reconstruct_explanation.md](open3d_reconstruct_explanation.md) |
| **Trajectory extraction** | |
| `polycam_raw_to_trajectory` | [polycam_raw_to_trajectory_explanation.md](polycam_raw_to_trajectory_explanation.md) |
| `rtabmap_poses_to_trajectory` | [rtabmap_poses_to_trajectory_explanation.md](rtabmap_poses_to_trajectory_explanation.md) |
| **Mag fusion + voxel** | |
| `fuse_mag_with_trajectory` | [fuse_mag_with_trajectory_explanation.md](fuse_mag_with_trajectory_explanation.md) |
| `mag_world_to_voxel_volume` | [mag_world_to_voxel_volume_explanation.md](mag_world_to_voxel_volume_explanation.md) |
| `visualize_3d_heatmap` | [visualize_3d_heatmap_explanation.md](visualize_3d_heatmap_explanation.md) |
| `mag_calibrate_zero_logger` | [mag_calibrate_zero_logger_explanation.md](mag_calibrate_zero_logger_explanation.md) |
| `mag_to_csv_v2` (2D, used for 3D capture) | [mag_to_csv_v2_explanation.md](mag_to_csv_v2_explanation.md) |
| **GPR (optional)** | |
| `mag_world_to_voxel_volumeV2_gpr` | [mag_world_to_voxel_volumeV2_gpr_explanation.md](mag_world_to_voxel_volumeV2_gpr_explanation.md) |
| `visualize_3d_heatmapV2_gpr` | [visualize_3d_heatmapV2_gpr_explanation.md](visualize_3d_heatmapV2_gpr_explanation.md) |

See also: [new_3d_scan_explanation.md](new_3d_scan_explanation.md), [backup_scans_to_usb_explanation.md](backup_scans_to_usb_explanation.md), [backup_usb_3d_explanation.md](backup_usb_3d_explanation.md).

---

## Legacy commands

- **Python:** Wrappers under **`scripts/2d/`** and **`scripts/3d/`** still work; preferred usage is **`python3 pipelines/2d/<script>.py`** and **`python3 pipelines/3d/<script>.py`**.
- **Shell:** **`./scripts/3d/new_3d_scan.sh`** and **`./scripts/3d/backup_usb_3d.sh`** remain for backward compatibility; preferred usage is **`./tools/3d/new_3d_scan.sh`** and **`./tools/3d/backup_usb_3d.sh`**.
