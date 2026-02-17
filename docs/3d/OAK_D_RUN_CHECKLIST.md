# OAK-D Lite + Magnetometer — Every Run Checklist

Quick reference for repeated captures. Full details: [PIPELINE_3D.md](PIPELINE_3D.md).

---

## EVERY RUN CHECKLIST

- [ ] OAK-D Lite plugged into Mac via **USB 3** (direct port, no hub)
- [ ] Pi powered on, SSH accessible, magnetometer connected (I2C)
- [ ] Decide on a `RUN_DIR` string — **same value on both Mac and Pi**
- [ ] Create run folders on Mac (or Pi, wherever data lives)
- [ ] Start mag logger on Pi **first**
- [ ] Start OAK-D capture on Mac **second**
- [ ] Mark `start` on the mag logger (press Enter or type `start` + Enter)
- [ ] Walk slowly around the object for **30–60 seconds**
- [ ] Mark `end` on the mag logger
- [ ] Press **q** in the OAK preview window to stop capture
- [ ] Stop mag logger (**Ctrl+C** or type `q` + Enter)
- [ ] Verify files exist (see "WHEN DONE" below)

---

## MAC COMMANDS

Run from the repo root (`~/Desktop/fluxspace/fluxspace-core`).

```bash
cd ~/Desktop/fluxspace/fluxspace-core
source .venv/bin/activate

# ---- 1. Set RUN_DIR (copy this EXACT string to the Pi too) ----
export RUN_DIR="data/runs/run_$(date +%Y%m%d_%H%M)"
echo "RUN_DIR=$RUN_DIR"
# ^ Copy that printed line and paste it into your Pi terminal

# ---- 2. Create folders ----
mkdir -p "$RUN_DIR"/{raw,processed,exports}

# ---- 3. Quick device check (optional, first time or if unsure) ----
python - <<'PY'
import depthai as dai
devs = dai.Device.getAllAvailableDevices()
print("found", len(devs), "device(s)")
for d in devs: print(" ", d)
PY

# ---- 4. Start capture (after mag logger is running on Pi) ----
python3 pipelines/3d/capture_oak_rgbd.py \
  --out "$RUN_DIR/raw/oak_rgbd"

# Press q in the preview window when done scanning.
```

---

## PI COMMANDS

SSH into the Pi, then:

```bash
ssh fluxspace-pi    # or ssh fluxspace@<pi-ip>
cd ~/fluxspace-core
source ~/fluxenv/bin/activate

# ---- 1. Paste the SAME RUN_DIR from your Mac ----
export RUN_DIR="data/runs/run_YYYYMMDD_HHMM"   # <-- replace with actual value from Mac
mkdir -p "$RUN_DIR/raw"

# ---- 2. Start mag logger (BEFORE starting OAK capture) ----
python3 pipelines/3d/mag_calibrate_zero_logger.py \
  --out "$RUN_DIR/raw/mag_run.csv" \
  --hz 80 \
  --units uT \
  --calib-seconds 20 \
  --zero-seconds 3 \
  --save-cal "$RUN_DIR/raw/calibration.json"

# Follow the prompts:
#   Phase 1: Rotate sensor through many orientations (20s)
#   Phase 2: Hold still for baseline (3s)
#   Phase 3: Logging — press Enter to mark "start" / "end"
#   Quit: Ctrl+C or type q + Enter
```

**Simpler alternative** (no calibration, just log):

```bash
python3 pipelines/2d/mag_to_csv_v2.py \
  --out "$RUN_DIR/raw/mag_run.csv" \
  --hz 80 --units uT --samples 1
```

---

## WHEN DONE

### Stop everything

1. Press **q** in the OAK preview window on Mac (or Ctrl+C if `--no-preview`)
2. Type `end` + Enter on the Pi mag logger, then **Ctrl+C**

### Verify outputs (Mac)

```bash
# Check OAK frames
echo "--- OAK-D output ---"
ls "$RUN_DIR/raw/oak_rgbd/color/" | head -3
ls "$RUN_DIR/raw/oak_rgbd/depth/" | head -3
wc -l "$RUN_DIR/raw/oak_rgbd/timestamps.csv"
cat "$RUN_DIR/raw/oak_rgbd/intrinsics.json"

# Check mag CSV (if captured on same machine, or after copying from Pi)
echo "--- Mag output ---"
wc -l "$RUN_DIR/raw/mag_run.csv"
head -2 "$RUN_DIR/raw/mag_run.csv"
```

**Expected:**

| File | What to see |
|------|-------------|
| `color/color_*.jpg` | Hundreds of JPEGs (30–60s at ~15 fps = ~450–900 frames) |
| `depth/depth_*.png` | Same count of 16-bit PNGs |
| `timestamps.csv` | Line count = frame count + 1 (header) |
| `intrinsics.json` | `"source": "oak_calibration"` (not `"approximate"`) |
| `mag_run.csv` | Thousands of rows; header includes `t_rel_s, bx, by, bz` |

### Create extrinsics (optional — skips with warning if missing)

```bash
# Option A: JSON file (reuse across runs)
printf '%s\n' '{ "translation_m": [0.30, 0.0, 0.0], "quaternion_xyzw": [0, 0, 0, 1] }' \
  > "$RUN_DIR/raw/extrinsics.json"
# ^ Adjust 0.30 to your actual ruler offset in metres

# Option B: Inline CLI shorthand (no file needed)
# Use --default-extrinsics "behind_cm=2,down_cm=10" in the fuse command
# (camera frame: +x right, +y down, +z forward)
```

### Process (when ready)

```bash
# Reconstruct -> processed/trajectory.csv + processed/open3d_mesh.ply
python3 pipelines/3d/open3d_reconstruct.py \
  --in "$RUN_DIR/raw/oak_rgbd" --no-viz

# Fuse mag with trajectory (--run auto-derives all paths)
# Extrinsics: uses raw/extrinsics.json if present, else identity (warns)
python3 pipelines/3d/fuse_mag_with_trajectory.py \
  --run "$RUN_DIR" \
  --value-type zero_mag
# Or for a known mount offset without extrinsics.json:
#   --default-extrinsics "behind_cm=2,down_cm=10"

# Voxel volume (auto-scales mm->m, clamps grid to max 256 voxels/axis)
python3 pipelines/3d/mag_world_to_voxel_volume.py \
  --in "$RUN_DIR/processed/mag_world.csv" \
  --out "$RUN_DIR/exports/volume.npz" \
  --voxel-size 0.02

# Visualise (interactive viewer with toggles)
python3 pipelines/3d/view_scan_toggle.py --run "$RUN_DIR"
```

---

## TROUBLESHOOTING

### RUN_DIR not set / points to wrong place

```bash
# Check it's set
echo "$RUN_DIR"
# Should print something like: data/runs/run_20260210_1430

# Check the folder exists
ls -d "$RUN_DIR"
# If "No such file or directory": you forgot mkdir or the variable is empty.

# Fix: re-export and re-create
export RUN_DIR="data/runs/run_$(date +%Y%m%d_%H%M)"
mkdir -p "$RUN_DIR"/{raw,processed,exports}
```

- On the Pi, make sure you pasted the **exact same string** (not re-ran `date`, which gives a different minute).
- If the Pi stores data locally, you'll need to `scp` or `rsync` the mag CSV to the Mac's `$RUN_DIR/raw/` before processing.

### OAK-D not detected / device count is 0

```bash
# Run the device check
python - <<'PY'
import depthai as dai
devs = dai.Device.getAllAvailableDevices()
print("found", len(devs), "device(s)")
PY
```

If 0 devices:

- **Unplug and replug** the USB cable (wait 3 seconds between).
- Use a **USB 3** port directly on the Mac — no hubs, no adapters.
- Try a different USB-C/USB-A cable. Some cables are charge-only.
- Check System Information > USB and confirm "OAK" or "Luxonis" appears.
- On macOS, no driver install is needed. If it still fails: `pip install -U depthai` to update.

### Outputs exist but folders are empty / no frames written

```bash
ls "$RUN_DIR/raw/oak_rgbd/color/" | wc -l
# If 0:
```

- The script likely crashed immediately. Scroll up in terminal for the error.
- **"RuntimeError: No device found"** — see above (OAK not detected).
- **"Cannot open display"** — if running headless, add `--no-preview`.
- **Permissions** — check you can write to the path: `touch "$RUN_DIR/raw/oak_rgbd/test" && rm "$RUN_DIR/raw/oak_rgbd/test"`.
- **Disk space** — 1 minute at 15 fps ~ 900 frames ~ 200–400 MB. Check: `df -h .`
- If timestamps.csv exists but has only the header (1 line), frames arrived but both RGB and depth never synced. Try lowering FPS: `--fps 10`.
