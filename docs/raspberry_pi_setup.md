# Fluxspace Core — Raspberry Pi Setup + Testing Runbook (MMC5983MA)

Complete guide for setting up your Raspberry Pi and running the magnetic field measurement pipeline.

**Preferred:** Run Python from repo root: **`python3 pipelines/2d/<script>.py`** and **`python3 pipelines/3d/<script>.py`**. Run 2D tools: **`./tools/2d/new_run.sh`**, **`./tools/2d/backup_runs_to_usb.sh`**. Run 3D tools: **`./tools/3d/new_3d_scan.sh`**, **`./tools/3d/backup_usb_3d.sh`**. Legacy wrappers under `scripts/2d/` and `scripts/3d/` still work.

---

## Part 1 — Initial Setup (One-time)

### 1.1. Hardware Setup
1. Insert microSD card with Raspberry Pi OS
2. Plug in Ethernet (or configure Wi‑Fi)
3. Plug in power (Pi turns on automatically)

### 1.2. SSH into the Pi

From your Mac, SSH into the Pi:

```bash
ssh fluxspace-pi
```

If you don't know the hostname or IP:
- Check your router's "connected devices" list
- Or on Mac, find the Pi's IP:
```bash
arp -a
```

Then SSH using the IP:
```bash
ssh fluxspace@<pi-ip-address>
```

### 1.3. Clone the Repository

On the Pi:
```bash
cd ~
git clone <YOUR_GITHUB_REPO_URL_HERE> fluxspace-core
cd ~/fluxspace-core
```

If the repo already exists:
```bash
cd ~/fluxspace-core
git pull
```

**Note:** Make sure `tools/2d/setup_pi.sh` exists in the repo. If it doesn't, create it first (the script should already be in your repo from the setup).

---

## Part 2 — Automated Setup (Run Once)

Once you've SSH'd into the Pi and cloned the repo, run the automated setup script. This installs everything you need in one go.

### 2.1. Run the Setup Script

```bash
cd ~/fluxspace-core
chmod +x tools/2d/setup_pi.sh
./tools/2d/setup_pi.sh
```

**What it does:**
- Installs system packages (git, python3-venv, python3-pip, i2c-tools, python3-smbus)
- Enables I2C non-interactively
- Creates Python virtual environment at `~/fluxenv`
- Installs all Python dependencies (numpy, pandas, matplotlib, sparkfun-qwiic, sparkfun-qwiic-mmc5983ma)

### 2.2. Reboot (Required After First-Time I2C Enable)

After the script completes, reboot the Pi:

```bash
sudo reboot
```

Wait for the Pi to restart (~30 seconds), then SSH back in.

### 2.3. Verify Setup

After rebooting and reconnecting via SSH:

```bash
cd ~/fluxspace-core
source ~/fluxenv/bin/activate
i2cdetect -y 1
python -c "import qwiic_mmc5983ma; print('qwiic_mmc5983ma OK')"
```

**Expected results:**
- `i2cdetect -y 1` should show `30` in the table (when sensor is plugged in)
- Python import should print `qwiic_mmc5983ma OK`

If both work, you're ready to use the pipeline!

### 2.4. Optional: Install tmux (Recommended for Persistent Sessions)

tmux allows you to keep terminal sessions running even if your SSH connection drops. This is especially useful for long-running operations or when working over unstable connections (like hotspots).

Install tmux:
```bash
sudo apt update
sudo apt install -y tmux
```

**Why use tmux?**
- Sessions survive SSH disconnects
- Can detach and reattach later
- Useful for long pipeline runs or when stepping away

---

## Part 3 — Wiring the Sensor

**Important:** Always power off the Pi before connecting/disconnecting hardware.

1. **Power OFF** the Pi (unplug USB‑C power)
2. Plug the Qwiic/STEMMA adapter into the Pi GPIO header (if using one)
3. Plug Qwiic cable into adapter + into the MMC5983MA board
4. **Power ON** the Pi (plug USB‑C power back in)

**Note:** It generally doesn't matter which port you use on a Qwiic adapter (they're usually parallel), but don't hot‑plug—power‑off is safest.

---

## Part 4 — Every Session: Connect and Activate

### 4.1. SSH into Pi
```bash
ssh fluxspace-pi
```

### 4.2. Enter Repo and Activate Environment
```bash
cd ~/fluxspace-core
source ~/fluxenv/bin/activate
```

### 4.3. Verify Sensor is Detected (Optional)
```bash
i2cdetect -y 1
```

You should see `30` in the table when the sensor is connected.

### 4.4. Optional: Run Smoke Test
```bash
python3 tools/2d/mmc5983ma_smoketest.py
```

Should print: `✅ MMC5983MA responding on I2C at address 0x30`

### 4.5. Optional: Use tmux for Persistent Sessions

**Start a new tmux session:**
```bash
tmux new -s flux
```

You're now "inside" tmux. All commands you run here will continue even if your SSH connection drops.

**Detach from tmux (keeps session running):**
Press: **Ctrl + b**, then **d**

Your session continues running in the background.

**Reattach to existing session:**
```bash
tmux attach -t flux
```

**List all sessions:**
```bash
tmux ls
```

**Why sessions might disappear:**
- Pi rebooted/crashed/lost power (tmux sessions live in RAM)
- You exited the session (instead of detaching)
- tmux wasn't installed

**Safe workflow for long operations:**
1. Start tmux: `tmux new -s flux`
2. Run your pipeline commands inside tmux
3. Detach when stepping away (Ctrl+b, then d)
4. Reattach later: `tmux attach -t flux`

**Note:** tmux helps with SSH drops, but sessions are lost on reboot. For hotspot work or unstable connections, always use tmux for long-running operations.

---

## Part 5 — Running the Pipeline

### 5.0. Ensure Folders Exist
```bash
cd ~/fluxspace-core
mkdir -p data/raw data/processed data/exports data/runs
```

### 5.1. Collect Data (mag_to_csv / mag_to_csv_v2)

You now have **two ways to collect magnetometer data**, depending on your experiment:

- **Grid survey** (structured 2D board scans) → `mag_to_csv.py`
- **Continuous logger** (free motion, 3D-fusion-ready) → `mag_to_csv_v2.py`

#### 5.1.1. Grid Survey (mag_to_csv.py)

This script uses **auto‑grid mode**:
- You move the magnetometer to each grid point
- Keep **height + orientation constant**
- Press **Enter** at each point to capture
- Repeats until "Grid complete"

```bash
python3 pipelines/2d/mag_to_csv.py --out data/raw/mag_data.csv
```

**Expected output:**
- `data/raw/mag_data.csv`

Use this when you want a **regular 2D grid** for anomaly maps and B_total heatmaps.

#### 5.1.2. Continuous Logger for 3D Fusion (mag_to_csv_v2.py)

This script logs a **continuous stream** of magnetometer data (no 2D grid), designed to be fused with a 3D trajectory (Polycam Raw Data, RTAB-Map, etc.).

**Basic run (field strength stream while scanning):**

```bash
python3 pipelines/2d/mag_to_csv_v2.py --out data/raw/mag_run01.csv --hz 80 --units uT --samples 1
```

While it’s running:
- Press **Enter** to add a `MARK` row.
- Or type a label (e.g. `start`, `end`) then Enter.
- Quit with **Ctrl+C** (or type `q` then Enter).

**If reads are noisy, average a bit:**

```bash
python3 pipelines/2d/mag_to_csv_v2.py \
  --out data/raw/mag_run01.csv \
  --hz 50 \
  --units uT \
  --samples 5 \
  --sample-delay 0.002
```

Columns in `mag_run01.csv`:

```text
t_unix_ns, t_utc_iso, t_rel_s, bx, by, bz, b_total, units, row_type, note
```

- `row_type = SAMPLE` for normal rows
- `row_type = MARK` for your Enter/label markers
- `row_type = INFO` at start (and on sensor read errors)

Use this when you are **moving freely in 3D** and plan to fuse the magnetometer stream with a separate pose/trajectory export.

### 5.2. Validate and Clean Data

```bash
python3 pipelines/2d/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv
```

To drop flagged outliers:
```bash
python3 pipelines/2d/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv --drop-outliers
```

**Expected outputs:**
- `data/processed/mag_data_clean.csv`
- `data/processed/mag_data_report.txt`
- `data/processed/mag_data_*.png` (diagnostic plots)

### 5.3. Compute Local Anomaly

For a 25cm × 25cm board using ~5cm spacing, use **radius = 0.10m** (≈ two grid steps):

```bash
python3 pipelines/2d/compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --radius 0.10 --plot
```

To drop rows where `_flag_any` is true (if present):
```bash
python3 pipelines/2d/compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --radius 0.10 --drop-flag-any --plot
```

**Expected output:**
- `data/processed/mag_data_anomaly.csv`

### 5.4. Generate Heatmap

The pipeline provides **two different heatmap scripts** for different visualization purposes:

#### 5.4.1. B_total Heatmap (Field Strength Visualization)

**Purpose:** Visualize the absolute magnetic field strength (B_total) across the area.

**Use when:**
- You want to see the actual field strength distribution
- You're doing magnetic detection or field mapping
- You need unit conversion (gauss ↔ microtesla)

**Input:** `data/processed/mag_data_clean.csv` (from step 5.2)

**Basic usage (gauss units):**
```bash
python3 pipelines/2d/interpolate_to_Btotal_heatmap.py --in data/processed/mag_data_clean.csv
```

**With microtesla units:**
```bash
python3 pipelines/2d/interpolate_to_Btotal_heatmap.py --in data/processed/mag_data_clean.csv --units uT
```

**With custom grid spacing:**
```bash
python3 pipelines/2d/interpolate_to_Btotal_heatmap.py --in data/processed/mag_data_clean.csv --units uT --grid-step 0.01
```

**Expected outputs:**
- `data/processed/mag_detection_grid.csv`
- `data/processed/mag_detection_heatmap.png`

#### 5.4.2. Anomaly Heatmap (Anomaly Detection Visualization)

**Purpose:** Visualize local anomalies (deviations from neighborhood).

**Use when:**
- You want to see where the field differs from nearby areas
- You're looking for magnetic anomalies (hot spots, cold spots)
- You've already run step 5.3 (compute_local_anomaly_v2.py)

**IMPORTANT:** Use `mag_data_anomaly.csv` (from step 5.3, not `mag_data_clean.csv`).

**If your anomaly CSV contains `local_anomaly_norm`:**
```bash
python3 pipelines/2d/interpolate_to_heatmapV2.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly_norm
```

**If it only contains `local_anomaly`:**
```bash
python3 pipelines/2d/interpolate_to_heatmapV2.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly
```

**With custom grid spacing:**
```bash
python3 pipelines/2d/interpolate_to_heatmapV2.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly --grid-step 0.01
```

**Expected outputs:**
- `data/exports/<stem>_grid.csv`
- `data/exports/<stem>_heatmap.png`

**Summary:**
- **B_total heatmap**: Shows absolute field strength (use `mag_data_clean.csv`)
- **Anomaly heatmap**: Shows local deviations (use `mag_data_anomaly.csv`)
- Both scripts use IDW interpolation but serve different analysis purposes

### 5.5. Organize Run Data

**Before starting a new test session**, create a new run folder:

```bash
cd ~/fluxspace-core
./tools/2d/new_run.sh
```

This creates a timestamped folder in `data/runs/` (e.g., `data/runs/12-28-2024_14-30`). Copy or note this path—you'll use it for this session's outputs.

**After completing a full pipeline run (5.1 → 5.4)**, all outputs are organized in that run folder.

**Verify outputs:**
```bash
ls data/runs
ls data/runs/<latest-run-folder>
```

### 5.6. Backup Runs to USB

After completing your test session, backup all runs to USB.

#### 5.6.1. Plug in and mount the USB drive

1. **Plug in the FLUXSPACE USB stick.**
2. **Create the mount point (once):**
   ```bash
   sudo mkdir -p /media/fluxspace/FLUXSPACE
   ```
3. **Mount the USB (FAT32) read–write for your user:**
   ```bash
   sudo mount -t vfat -o rw,uid=$(id -u),gid=$(id -g),umask=022 /dev/sda1 /media/fluxspace/FLUXSPACE
   ```

   - `/dev/sda1` is the **most common** device for a single USB stick on the Pi.
   - If this fails, run `lsblk` or `sudo fdisk -l` to confirm the correct device, then substitute it for `/dev/sda1`.

**Quick check (verify USB is mounted):**
```bash
ls /media/fluxspace
```

You should see `FLUXSPACE` directory.

#### 5.6.2. Run the backup script

```bash
cd ~/fluxspace-core
./tools/2d/backup_runs_to_usb.sh
```

This syncs all runs from `data/runs/` to your USB drive at `/media/fluxspace/FLUXSPACE/fluxspace_runs_backup/`.

**What it does:**
- Checks that USB is mounted
- Uses `rsync` to sync all run folders
- `--delete` flag keeps USB in sync (removes files on USB that no longer exist on Pi)
- Creates backup directory if it doesn't exist

**Note:** If USB is not mounted, the script will tell you. Plug in the USB drive and ensure it's mounted at `/media/fluxspace/FLUXSPACE` before running the backup.

### 5.7. Safely Unmount USB Drive

**Always unmount the USB drive before unplugging it** to prevent filesystem corruption. If you don't, FAT32 can get marked as "dirty" from an unsafe removal, causing the Pi to remount it read-only next time (`errors=remount-ro`).

**Safe unmount procedure:**

1. **Flush all pending writes:**
   ```bash
   sync
   ```

2. **Wait 2–5 seconds** after `sync` completes (or after rsync finishes) to ensure all data is written.

3. **Unmount the drive:**
   ```bash
   sudo umount /media/fluxspace/FLUXSPACE
   ```

4. **Verify unmount succeeded:** Wait until `umount` returns to the prompt with **no error** before unplugging.

**If `umount` says "device is busy":**

The USB is still in use by a process. Find what's using it:

```bash
lsof +D /media/fluxspace/FLUXSPACE | head
```

Stop the process (close terminals/files using the USB, or kill the process), then try `umount` again.

**Why this matters:**

- Prevents FAT32 filesystem corruption from unsafe removal
- Avoids read-only remounts on next insertion
- Ensures all data is fully written before disconnection
- Reduces need for `fsck.vfat` repairs

**Note:** If the USB stick itself is flaky or already has filesystem corruption, you may occasionally need `fsck.vfat` to repair it. But following this unmount procedure every time will prevent corruption from happening during normal use.

### 5.8. Alternative storage: 3D scans (data/scans/*__3d)

The **2D pipeline is unchanged** and still uses `data/runs/` and the existing scripts (`./tools/2d/new_run.sh`, `./tools/2d/backup_runs_to_usb.sh`).

For the **3D pipeline** (Polycam/RTAB-Map + magnetometer fusion), you can use a separate storage path so 3D scan snapshots stay distinct:

- **Location:** `data/scans/<RUN_ID>__3d/` or `data/scans/<RUN_ID>__3d__<label>/`
- **RUN_ID:** Same format as 2D: `MM-DD-YYYY_HH-MM` (America/New_York).

**Create a 3D scan snapshot** (after capture/processing, snapshot current `data/raw`, `data/processed`, `data/exports` into a new scan folder):

```bash
cd ~/fluxspace-core
./tools/3d/new_3d_scan.sh
```

With an optional label (e.g. block name):

```bash
./tools/3d/new_3d_scan.sh --label block01
```

**Examples:**
- `data/scans/01-29-2026_13-57__3d/`
- `data/scans/01-29-2026_13-57__3d__block01/`

Each scan folder contains `raw/`, `processed/`, `exports/`, and `config/` (for extrinsics.json, etc.).

**Back up 3D scans to USB** (separate from 2D runs backup):

```bash
./tools/3d/backup_usb_3d.sh
```

This backs up only `data/scans/` to `/media/fluxspace/FLUXSPACE/fluxspace_scans_backup/`. Mount/unmount USB the same way as for 2D (see 5.6.1 and 5.7). The 2D runs backup script is not modified; use `./tools/2d/backup_runs_to_usb.sh` for 2D runs as before.

---

## Part 6 — Quick "It Worked" Checklist

After completing the pipeline (5.1 → 5.4), verify all outputs exist:

```bash
ls -lh data/raw data/processed data/exports
```

You should see:
- A fresh raw CSV in `data/raw/`
- A `*_clean.csv` and `*_anomaly.csv` in `data/processed/`
- A `*_heatmap.png` and `*_grid.csv` in `data/exports/`

---

## Part 7 — Standard Test Plan

Do TWO runs to establish baseline and detect anomalies:

### 7.1. Baseline Run
1. Create new run folder: `./tools/2d/new_run.sh`
2. Cardboard only, no nearby metal
3. Run the full pipeline (5.1 → 5.4)
4. This establishes your "normal" magnetic field
5. Backup to USB: `./tools/2d/backup_runs_to_usb.sh`

### 7.2. Stimulus Run (Introduce Metal)
1. Create new run folder: `./tools/2d/new_run.sh`
2. Place steel/rebar near/under board at known location
3. Rerun the same grid pattern
4. Run the full pipeline (5.1 → 5.4)
5. Compare heatmaps + anomaly CSV
6. Backup to USB: `./tools/2d/backup_runs_to_usb.sh`

The difference between baseline and stimulus shows the metal's effect.

### 7.3. Tuning Parameters

After comparing runs, tune these parameters:

- **Grid spacing (DX/DY)**: Smaller spacing = higher resolution but more points
- **Samples per point (N)**: More samples = better averaging but slower
- **Anomaly radius**: Start with ~0.10m for 25cm board, adjust based on results

---

## Part 8 — One-Command Full Pipeline

Complete pipeline from start to finish (copy/paste):

```bash
cd ~/fluxspace-core
source ~/fluxenv/bin/activate
mkdir -p data/raw data/processed data/exports data/runs

# Create new run folder for this session
./tools/2d/new_run.sh

# Optional: verify sensor is detected
i2cdetect -y 1

# Step 1: Collect data
python3 pipelines/2d/mag_to_csv.py --out data/raw/mag_data.csv

# Step 2: Validate and clean
python3 pipelines/2d/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv

# Step 3: Compute anomalies
python3 pipelines/2d/compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --radius 0.10 --plot

# Step 4: Generate heatmap
python3 pipelines/2d/interpolate_to_heatmapV1.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly

# Step 5: Backup to USB (optional, but recommended)
./tools/2d/backup_runs_to_usb.sh

ls data/runs
```

---

## Part 9 — Troubleshooting

### 9.1. "Cannot write to data/raw/mag_data.csv: No such file or directory"
Fix:
```bash
cd ~/fluxspace-core
mkdir -p data/raw data/processed data/exports data/runs
```

### 9.2. "Module not found" (numpy/pandas/matplotlib/qwiic_mmc5983ma)
You forgot to activate the venv or packages aren't installed:
```bash
source ~/fluxenv/bin/activate
pip install numpy pandas matplotlib
pip install sparkfun-qwiic sparkfun-qwiic-mmc5983ma
```

**Note:** If you get `ModuleNotFoundError: No module named 'qwiic_mmc5983ma'`, make sure you installed **both** packages:
- `sparkfun-qwiic` (base Qwiic library)
- `sparkfun-qwiic-mmc5983ma` (MMC5983MA-specific driver)

The automated setup script (`tools/2d/setup_pi.sh`) installs both automatically.

### 9.3. "File not found" when a script reads input
Double-check your filenames:
```bash
ls data/raw
ls data/processed
```
Make sure you're using the correct output filename from the previous step.

### 9.4. Heatmap says "input file not found"
Use the correct filename from anomaly step:
- **Correct:** `data/processed/mag_data_anomaly.csv`
- **Wrong:** `data/processed/mag_data_clean.csv` (this is from step 5.2, not 5.3)

### 9.5. Sensor not detected on I2C
1. Check wiring (power off, reconnect, power on)
2. Verify I2C is enabled: `sudo raspi-config` → Interface Options → I2C → Enable
3. Reboot after enabling I2C
4. Run `i2cdetect -y 1` again

### 9.6. Import errors in `mag_to_csv.py` (duplicate imports, wrong order)
If you see import-related errors, check that:
1. `from __future__ import annotations` is the **very first line** (before any other imports)
2. No duplicate imports (argparse, sys, numpy, pandas, matplotlib, etc. should appear only once)
3. Imports are in correct order:
   - `from __future__` first
   - Standard library imports
   - Third-party imports (numpy, pandas, matplotlib)
   - Local/project imports (qwiic_mmc5983ma)

**Correct import order example:**
```python
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import qwiic_mmc5983ma
```

### 9.7. Hostname/IP issues when reconnecting
On some networks, hostnames won't resolve. Use IP instead:
```bash
# Find Pi IP on your network
sudo nmap -sn 192.168.1.0/24
# Then SSH using the IP
ssh fluxspace@<pi-ip>
```

### 9.8. tmux session disappeared ("no sessions")
If you see "no sessions" when trying to attach, one of these happened:
- Pi rebooted/crashed/lost power (tmux sessions live in RAM, not persistent across reboots)
- You exited the session instead of detaching
- tmux wasn't installed

**Fix:**
1. Install tmux if needed: `sudo apt install -y tmux`
2. Start a new session: `tmux new -s flux`
3. For hotspot/unstable connections, always use tmux for long operations

**Remember:** Detach with Ctrl+b, then d (don't just close the terminal or type `exit`).

---

## Part 10 — Additional Notes

### 10.1. Git Sync (Optional)

To sync changes between Pi and Mac:

**On Pi:**
```bash
cd ~/fluxspace-core
git add -A
git commit -m "Describe your change"
git push
```

**On Mac:**
```bash
cd ~/fluxspace-core
git pull
```

**Note:** Pi and Mac are separate clones. Changes on one don't automatically appear on the other—you need to commit/push and pull.

### 10.2. Configure Git Identity (if committing from Pi)
```bash
cd ~/fluxspace-core
git config user.name "Your Name"
git config user.email "you@example.com"
```

### 10.3. Power Off Cleanly
When done:
```bash
sudo shutdown -h now
```

Wait ~20 seconds, then unplug power.

---

## Summary

**One-time setup:**
1. SSH into Pi
2. Clone repo
3. Run `./tools/2d/setup_pi.sh`
4. Reboot
5. Verify with `i2cdetect` and Python import

**Every session:**
1. SSH into Pi
2. (Optional) Start/attach tmux: `tmux new -s flux` or `tmux attach -t flux`
3. `cd ~/fluxspace-core && source ~/fluxenv/bin/activate`
4. Create new run folder: `./tools/2d/new_run.sh`
5. Run pipeline (5.1 → 5.4)
6. Backup to USB: `./tools/2d/backup_runs_to_usb.sh`
7. (If using tmux) Detach before disconnecting: Ctrl+b, then d

**That's it!** The automated setup script handles all the complexity. Use tmux for persistent sessions, especially on unstable connections. Always backup your runs to USB after each test session.
