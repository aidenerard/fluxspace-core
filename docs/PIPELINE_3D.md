# 3D Magnetic Fusion Pipeline — Runbook

End-to-end guide to produce: **(1)** 3D geometry (mesh/point cloud), **(2)** timestamped camera trajectory, **(3)** timestamped magnetometer log (calibrated + zero), **(4)** fused `mag_world.csv`, and **(5)** 3D magnetic anomaly heatmap (voxel volume + PyVista visualization).

---

## Quickstart (minimal first run)

```bash
# 1. Create run folder
export RUN_DIR="data/runs/run_$(date +%Y%m%d_%H%M)"
mkdir -p "$RUN_DIR"/{raw,processed,exports}

# 2. Capture: start magnetometer logger (on Pi), then scan with Polycam/RTAB-Map
#    Use mag_to_csv_v2.py (or mag_calibrate_zero_logger.py if you have it)
python3 scripts/mag_to_csv_v2.py --out "$RUN_DIR/raw/mag_run.csv" --hz 80 --units uT --samples 1

# 3. Export trajectory from your scan app (see Capture Day below), put files in $RUN_DIR/raw/

# 4. Processing (on Mac or Pi)
python3 scripts/polycam_raw_to_trajectory.py --in "$RUN_DIR/raw/PolycamRawExport" --out "$RUN_DIR/processed/trajectory.csv"
# OR: python3 scripts/rtabmap_poses_to_trajectory.py --in "$RUN_DIR/raw/rtabmap_poses.txt" --out "$RUN_DIR/processed/trajectory.csv"

python3 scripts/fuse_mag_with_trajectory.py \
  --trajectory "$RUN_DIR/processed/trajectory.csv" \
  --mag "$RUN_DIR/raw/mag_run.csv" \
  --extrinsics "$RUN_DIR/raw/extrinsics.json" \
  --out "$RUN_DIR/processed/mag_world.csv" \
  --value-type zero_mag

python3 scripts/mag_world_to_voxel_volume.py \
  --in "$RUN_DIR/processed/mag_world.csv" \
  --out "$RUN_DIR/exports/volume.npz" \
  --voxel-size 0.02 \
  --margin 0.1

python3 scripts/visualize_3d_heatmap.py \
  --volume "$RUN_DIR/exports/volume.npz" \
  --out-dir "$RUN_DIR/exports" \
  --screenshot
```

Expected outputs: `trajectory.csv`, `mag_world.csv`, `volume.npz`, and `exports/heatmap_3d_screenshot.png`.

---

## Overview (text diagram)

```
[CAPTURE DAY]
  Phone (Polycam LiDAR Raw / RTAB-Map)     Pi + MMC5983MA (rigid ruler ~30 cm)
       |                                              |
       v                                              v
  3D scan app                                    mag_to_csv_v2.py
  (export Raw Data / .db)                         (or mag_calibrate_zero_logger.py)
       |                                              |
       v                                              v
  Polycam folder / rtabmap export                 mag_run.csv
  (cameras, poses)                                (t_rel_s, bx, by, bz, b_total, row_type, note)

[PROCESSING DAY]
  polycam_raw_to_trajectory.py    OR    rtabmap_poses_to_trajectory.py
       |                                        |
       v                                        v
  trajectory.csv (t_rel_s, x, y, z, qx, qy, qz, qw)
       |
       +--------+ extrinsics.json (ruler: phone -> mag frame)
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

- **2D pipeline:** Still uses `data/runs/` and `./tools/new_run.sh`, `./tools/backup_runs_to_usb.sh` (unchanged).
- **3D pipeline:** Use `data/scans/<RUN_ID>__3d/` (or `data/scans/<RUN_ID>__3d__<label>/`).

**Create a 3D scan snapshot** (copies current `data/raw`, `data/processed`, `data/exports` into a new scan folder):

```bash
./scripts/new_3d_scan.sh
./scripts/new_3d_scan.sh --label block01
```

**Examples:** `data/scans/01-29-2026_13-57__3d/`, `data/scans/01-29-2026_13-57__3d__block01/`

**Back up 3D scans to USB only:**

```bash
./scripts/backup_usb_3d.sh
```

Backs up `data/scans/` → `/media/fluxspace/FLUXSPACE/fluxspace_scans_backup/`. Mount/unmount USB as in the main runbook (see docs). The 2D runs backup is separate (`./tools/backup_runs_to_usb.sh`).

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

- **Rig:** Magnetometer (MMC5983MA) mounted ~30 cm (1 foot) from phone on a rigid ruler; orientation fixed relative to phone.
- **Phone:** Polycam LiDAR (Developer Mode → Raw Data export) **or** RTAB-Map iOS; ensure you can export poses/trajectory.
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
python3 scripts/mag_calibrate_zero_logger.py \
  --out "$RUN_DIR/raw/mag_run.csv" \
  --hz 80 \
  --units uT \
  --calib-seconds 20 \
  --zero-seconds 3 \
  --save-cal "$RUN_DIR/raw/calibration.json"
```

**Option B:** Log only (no calibration/zero), e.g. for quick tests:

```bash
python3 scripts/mag_to_csv_v2.py \
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

### 4. Extrinsics and run folder

- Copy/create **extrinsics.json** into `$RUN_DIR/raw/` (see “Extrinsics (ruler rig)” below).
- Ensure `mag_run.csv`, trajectory source (Polycam folder or RTAB-Map file), and `extrinsics.json` are all under `$RUN_DIR/raw/`.

**Create extrinsics template (translation-only, 30 cm along +X):**  
`printf '%s\n' '{ "translation_m": [0.30, 0.0, 0.0], "rotation_quat_xyzw": null }' > "$RUN_DIR/raw/extrinsics.json"`  
Adjust `translation_m` to your measured ruler offset (meters).

---

## Processing day

All commands use the same `RUN_DIR`. Run from repo root.

### Step 1: Trajectory CSV

**Option A — Polycam Raw Data**

```bash
python3 scripts/polycam_raw_to_trajectory.py \
  --in "$RUN_DIR/raw/PolycamRawExport" \
  --out "$RUN_DIR/processed/trajectory.csv"
```

If timestamps are missing, script uses frame order and prints a warning; output columns: `t_rel_s, x, y, z, qx, qy, qz, qw`.

**Option B — RTAB-Map poses**

```bash
python3 scripts/rtabmap_poses_to_trajectory.py \
  --in "$RUN_DIR/raw/rtabmap_poses.txt" \
  --out "$RUN_DIR/processed/trajectory.csv" \
  --format TUM
```

Output: same columns, `t_rel_s` normalized from first pose.

### Step 2: Fuse magnetometer with trajectory

```bash
python3 scripts/fuse_mag_with_trajectory.py \
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

```bash
python3 scripts/mag_world_to_voxel_volume.py \
  --in "$RUN_DIR/processed/mag_world.csv" \
  --out "$RUN_DIR/exports/volume.npz" \
  --voxel-size 0.02 \
  --margin 0.1 \
  --method idw
```

- Bounds are computed from data + margin.
- Interpolation: IDW (k-nearest) or `griddata`; `idw` is default and robust for ~50k points.

### Step 4: 3D visualization

```bash
python3 scripts/visualize_3d_heatmap.py \
  --volume "$RUN_DIR/exports/volume.npz" \
  --out-dir "$RUN_DIR/exports" \
  --screenshot
```

- Optional: `--mesh path/to/mesh.ply` to overlay geometry.
- Produces orthogonal slice planes, isosurface threshold (basic slider in GUI), volume rendering; saves `heatmap_3d_screenshot.png` under `--out-dir`. Optional HTML export if supported.

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

- **Required:** `numpy`, `pandas`, `scipy` (interpolation), `pyvista` (3D viz).
- **Optional:** `open3d` (PLY/OBJ loading for overlay).

Install example:

```bash
pip install numpy pandas scipy pyvista
pip install open3d   # optional, for mesh loading
```

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

---

## Expected outputs summary

| Step | Output | Columns / content |
|------|--------|-------------------|
| Trajectory | `processed/trajectory.csv` | `t_rel_s, x, y, z, qx, qy, qz, qw` |
| Fuse | `processed/mag_world.csv` | `t_rel_s, x, y, z, value, value_type` |
| Voxel | `exports/volume.npz` | 3D array + `origin`, `voxel_size`, axis arrays |
| Viz | `exports/heatmap_3d_screenshot.png` | Slice + isosurface view; optional HTML |

All scripts use **argparse**, **clear prints** of inputs/outputs, and write into the **run folder**; defaults are set for reproducible runs.
