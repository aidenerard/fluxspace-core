# Fluxspace Core — Raspberry Pi Setup + Testing Runbook (MMC5983MA)

Complete guide for setting up your Raspberry Pi and running the magnetic field measurement pipeline.

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

**Note:** Make sure `tools/setup_pi.sh` exists in the repo. If it doesn't, create it first (the script should already be in your repo from the setup).

---

## Part 2 — Automated Setup (Run Once)

Once you've SSH'd into the Pi and cloned the repo, run the automated setup script. This installs everything you need in one go.

### 2.1. Run the Setup Script

```bash
cd ~/fluxspace-core
chmod +x tools/setup_pi.sh
./tools/setup_pi.sh
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
python3 tools/mmc5983ma_smoketest.py
```

Should print: `✅ MMC5983MA responding on I2C at address 0x30`

---

## Part 5 — Running the Pipeline

### 5.0. Ensure Folders Exist
```bash
cd ~/fluxspace-core
mkdir -p data/raw data/processed data/exports data/runs
```

### 5.1. Collect Data (mag_to_csv)

This script uses **auto‑grid mode**:
- You move the magnetometer to each grid point
- Keep **height + orientation constant**
- Press **Enter** at each point to capture
- Repeats until "Grid complete"

```bash
python3 scripts/mag_to_csv.py --out data/raw/mag_data.csv
```

**Expected output:**
- `data/raw/mag_data.csv`

### 5.2. Validate and Clean Data

```bash
python3 scripts/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv
```

To drop flagged outliers:
```bash
python3 scripts/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv --drop-outliers
```

**Expected outputs:**
- `data/processed/mag_data_clean.csv`
- `data/processed/mag_data_report.txt`
- `data/processed/mag_data_*.png` (diagnostic plots)

### 5.3. Compute Local Anomaly

For a 25cm × 25cm board using ~5cm spacing, use **radius = 0.10m** (≈ two grid steps):

```bash
python3 scripts/compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --radius 0.10 --plot
```

To drop rows where `_flag_any` is true (if present):
```bash
python3 scripts/compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --radius 0.10 --drop-flag-any --plot
```

**Expected output:**
- `data/processed/mag_data_anomaly.csv`

### 5.4. Generate Heatmap

**IMPORTANT:** Use `mag_data_anomaly.csv` (not `mag_data_clean.csv`).

If your anomaly CSV contains `local_anomaly_norm`:
```bash
python3 scripts/interpolate_to_heatmapV1.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly_norm
```

If it only contains `local_anomaly`:
```bash
python3 scripts/interpolate_to_heatmapV1.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly
```

With custom grid spacing:
```bash
python3 scripts/interpolate_to_heatmapV1.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly --grid-step 0.01
```

**Expected outputs:**
- `data/exports/<stem>_grid.csv`
- `data/exports/<stem>_heatmap.png`

### 5.5. Organize Run Data

After completing a full pipeline run (5.1 → 5.4), archive outputs:

```bash
./tools/new_run.sh
```

This creates a timestamped folder in `data/runs/` and copies all current outputs there.

**Verify outputs:**
```bash
ls data/raw
ls data/processed
ls data/exports
ls data/runs
```

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
- Cardboard only
- No nearby metal
- Run the full pipeline (5.1 → 5.4)
- This establishes your "normal" magnetic field

### 7.2. Stimulus Run (Introduce Metal)
- Place steel/rebar near/under board
- Rerun the same grid pattern
- Compare heatmaps + anomaly CSV
- The difference between baseline and stimulus shows the metal's effect

---

## Part 8 — One-Command Full Pipeline

Complete pipeline from start to finish (copy/paste):

```bash
cd ~/fluxspace-core
source ~/fluxenv/bin/activate
mkdir -p data/raw data/processed data/exports data/runs

# Optional: verify sensor is detected
i2cdetect -y 1

# Step 1: Collect data
python3 scripts/mag_to_csv.py --out data/raw/mag_data.csv

# Step 2: Validate and clean
python3 scripts/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv

# Step 3: Compute anomalies
python3 scripts/compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --radius 0.10 --plot

# Step 4: Generate heatmap
python3 scripts/interpolate_to_heatmapV1.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly

# Step 5: Organize run data
./tools/new_run.sh

ls data/raw
ls data/processed
ls data/exports
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

The automated setup script (`tools/setup_pi.sh`) installs both automatically.

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
3. Run `./tools/setup_pi.sh`
4. Reboot
5. Verify with `i2cdetect` and Python import

**Every session:**
1. SSH into Pi
2. `cd ~/fluxspace-core && source ~/fluxenv/bin/activate`
3. Run pipeline (5.1 → 5.4)

**That's it!** The automated setup script handles all the complexity.
