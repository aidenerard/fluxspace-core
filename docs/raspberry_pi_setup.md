# Fluxspace Core — Raspberry Pi Setup + Testing Runbook (MMC5983MA)

Complete guide covering:
- **One-time setup**: Hardware, software, and environment configuration
- **Test runs**: Step-by-step pipeline execution and verification

Single copy/paste runbook you can reuse forever.

---

## Part A — One-time Raspberry Pi setup (fresh Pi)

### A1. Hardware + boot
1) Insert microSD (Raspberry Pi OS)  
2) Plug in Ethernet (or Wi‑Fi)  
3) Plug in power (Pi turns on automatically)

---

### A2. SSH into the Pi (from Mac)

If you already know the IP:
```bash
ssh fluxspace-pi
```

If you don’t know the IP:
- Check your router “connected devices”
- Or on Mac:
```bash
arp -a
```

---

### A3. Update packages
On the Pi:
```bash
sudo apt update
sudo apt -y upgrade
```

---

### A4. Enable I2C (required for the magnetometer)
On the Pi:
```bash
sudo raspi-config
```
Go to:
- **Interface Options** → **I2C** → **Enable**
Then reboot:
```bash
sudo reboot
```

SSH back in after reboot.

---

### A5. Install I2C tools + git
On the Pi:
```bash
sudo apt install -y i2c-tools git
```

---

### A6. Confirm I2C bus exists
On the Pi:
```bash
ls /dev/i2c-*
```
Expected: something like `/dev/i2c-1`

---

### A7. Clone your repo on the Pi
Pick a location (home directory is fine):
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

---

### A8. Create Python venv on the Pi + install dependencies
On the Pi:
```bash
sudo apt install -y python3-venv python3-pip
python3 -m venv ~/fluxenv
source ~/fluxenv/bin/activate
python -m pip install --upgrade pip
```

Install your Python stack:
```bash
pip install numpy pandas matplotlib smbus2
```

Install SparkFun Qwiic (for MMC5983MA):
```bash
pip install sparkfun-qwiic
```

Quick verify:
```bash
python -c "import numpy, pandas, matplotlib, smbus2; print('Python stack OK')"
python -c "import qwiic_mmc5983ma; print('qwiic_mmc5983ma OK')"
```

---

### A9. Make sure repo data folders exist (on the Pi)
Even if they exist on your Mac, the Pi clone needs them too:
```bash
cd ~/fluxspace-core
mkdir -p data/raw data/processed data/exports data/runs tools
```

---

### A10. Optional: configure git identity (only needed to commit on the Pi)
If you want to commit from the Pi:
```bash
cd ~/fluxspace-core
git config user.name "Your Name"
git config user.email "you@example.com"
```

---

## Part B — Wiring checklist (Qwiic + adapter)

Typical order (safe + repeatable):
1) Power OFF Pi (unplug USB‑C power)
2) Plug the Qwiic/STEMMA adapter into the Pi GPIO header (if using one)
3) Plug Qwiic cable into adapter + into the MMC5983MA board
4) Power ON Pi (plug USB‑C power back in)

It generally does **not** matter which port you use on a Qwiic adapter (they’re usually parallel), but don’t hot‑plug unless you’re sure your adapter supports it—power‑off is safest.

---

## Part C — Every session: connect + activate environment

### C1. SSH into Pi
```bash
ssh fluxspace@192.168.1.213
```

### C2. Enter repo + activate venv
```bash
cd ~/fluxspace-core
source ~/fluxenv/bin/activate
```

(Optional sanity check to verify environment):
```bash
python -c "import numpy,pandas,matplotlib; print('env ok')"
```

---

## Part D — Sensor bring‑up / sanity checks

### D1. Confirm sensor appears on I2C (should show `30`)
```bash
i2cdetect -y 1
```

Expected: you see `30` in the table.

### D2. Optional: run your smoke test (if present)
```bash
python3 tools/mmc5983ma_smoketest.py
```

---

## Part E — Logging & testing pipeline (the exact run order)

### E0. Ensure folders exist (safe to run every time)
```bash
cd ~/fluxspace-core
mkdir -p data/raw data/processed data/exports data/runs
```

---

### E1. Collect data (mag_to_csv)

This script is **auto‑grid + press Enter**:
- Yes: you move the magnetometer to each grid point
- Keep **height + orientation constant**
- Press **Enter** at each point to capture
- It repeats until "Grid complete"

Basic run:
```bash
python3 scripts/mag_to_csv.py --out data/raw/mag_data.csv
```

If your acquisition script supports grid + averaging options:
```bash
# Fixed grid size (e.g., 9×9)
python3 scripts/mag_to_csv.py --out data/raw/mag_data.csv --grid-n 9 --samples 100

# Fixed grid spacing (e.g., 5cm spacing)
python3 scripts/mag_to_csv.py --out data/raw/mag_data.csv --grid-step 0.05 --samples 100
```

Expected output:
- `data/raw/mag_data.csv`

---

### E2. Validate + diagnostics (clean + plots + report)
Run:
```bash
python3 scripts/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv
```

If you want to explicitly drop flagged outliers:
```bash
python3 scripts/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv --drop-outliers
```

Expected outputs:
- `data/processed/mag_data_clean.csv`
- `data/processed/mag_data_report.txt`
- `data/processed/mag_data_*.png` (plots)

---

### E3. Compute local anomaly (recommended radius)
For a 25cm × 25cm board using ~5cm spacing (`DX = DY = 0.05m`), use:
- **radius = 0.10m** (≈ two grid steps)

Run:
```bash
python3 scripts/compute_local_anomaly_v2.py   --in data/processed/mag_data_clean.csv   --radius 0.10   --plot
```

If you want to drop rows where `_flag_any` is true (if present):
```bash
python3 scripts/compute_local_anomaly_v2.py   --in data/processed/mag_data_clean.csv   --radius 0.10   --drop-flag-any   --plot
```

Expected output:
- `data/processed/mag_data_anomaly.csv`

---

### E4. Heatmap interpolation
IMPORTANT: anomaly output is **mag_data_anomaly.csv** (not "clean_anomaly").

If your anomaly CSV contains `local_anomaly_norm`:
```bash
python3 scripts/interpolate_to_heatmapV1.py   --in data/processed/mag_data_anomaly.csv   --value-col local_anomaly_norm
```

If it only contains `local_anomaly`:
```bash
python3 scripts/interpolate_to_heatmapV1.py   --in data/processed/mag_data_anomaly.csv   --value-col local_anomaly
```

With custom grid spacing:
```bash
python3 scripts/interpolate_to_heatmapV1.py   --in data/processed/mag_data_anomaly.csv   --value-col local_anomaly   --grid-step 0.01
```

**Note:** Output files now default to `data/exports/` automatically (no need for `--out-dir` when input is in `data/processed/`).

Expected outputs:
- `data/exports/<stem>_grid.csv`
- `data/exports/<stem>_heatmap.png`

Confirm what you got:
```bash
ls data/raw
ls data/processed
ls data/exports
```

---

### E5. Organize run data (move to data/runs)
After completing a full pipeline run (E1 → E4), organize all outputs into a timestamped run folder:

Run:
```bash
./tools/new_run.sh
```

This script will:
- Create a new timestamped folder in `data/runs/` (e.g., `data/runs/2024-01-15_14-30-00`)
- Copy/move all current pipeline outputs from `data/raw/`, `data/processed/`, and `data/exports/` into the run folder
- Keep your data organized for comparison across multiple runs

Expected result:
- All pipeline outputs organized in `data/runs/<timestamp>/`

---

## Part E6 — Quick "it worked" checklist

After completing the pipeline (E1 → E4), verify all outputs exist:

```bash
ls -lh data/raw data/processed data/exports
```

You should see:
- A fresh raw CSV in `data/raw/`
- A `*_clean.csv` and `*_anomaly.csv` in `data/processed/`
- A `*_heatmap.png` and `*_grid.csv` in `data/exports/`

---

## Part F — Standard test plan (proves it works)

Do TWO runs to establish baseline and detect anomalies:

### F1. Baseline run
- Cardboard only
- No nearby metal
- Run the full pipeline (E1 → E4)
- This establishes your "normal" magnetic field

### F2. Stimulus run (introduce metal)
- Place steel/rebar near/under board
- Rerun the same grid pattern
- Compare heatmaps + anomaly CSV
- The difference between baseline and stimulus shows the metal's effect

### F3. Suggested 2D bench test protocol

For a controlled test environment:

1. **Setup:**
   - Place the sensor at a fixed height above the surface (consistent Z)
   - Use a tape measure and mark a small grid (e.g., 9×9)
   - Ensure consistent sensor orientation throughout

2. **Baseline measurement:**
   - Collect data **without metal** first (baseline)
   - Run full pipeline (E1 → E4)

3. **Stimulus measurement:**
   - Add a known magnet/metal object at a specific location
   - Repeat the same grid pattern
   - Run full pipeline (E1 → E4)

4. **Compare results:**
   - Check clean stats + flags (should show differences)
   - Compare anomaly maps (should highlight metal location)
   - Compare heatmap PNGs (visual difference should be clear)

---

## Part G — Common errors & fixes

### G1. "Cannot write to data/raw/mag_data.csv: No such file or directory"
Fix:
```bash
cd ~/fluxspace-core
mkdir -p data/raw data/processed data/exports data/runs
```

### G2. "Module not found" (numpy/pandas/matplotlib)
You forgot to activate/install in the venv:
```bash
source ~/fluxenv/bin/activate
pip install numpy pandas matplotlib
```

### G3. "File not found" when a script reads input
Double-check your filenames:
```bash
ls data/raw
ls data/processed
```
Make sure you're using the correct output filename from the previous step.

### G4. Heatmap says "input file not found"
Use the correct filename from anomaly step:
- Correct: `data/processed/mag_data_anomaly.csv`
- Wrong: `data/processed/mag_data_clean.csv` (this is from step E2, not E3)

### G5. `.local` / hostname issues when reconnecting
On some networks, `pi.local` won't resolve. Use IP instead:
```bash
# Find Pi IP on your network
sudo nmap -sn 192.168.1.0/24
# Then SSH using the IP
ssh fluxspace@<pi-ip>
```

### G6. Do I need to push to GitHub before running?
No.
- You can run anything that exists on the Pi immediately.
- GitHub is only for syncing between machines.

### G7. If you add files on the Pi, will they appear on your Mac automatically?
No.
- Pi + Mac are separate clones.
- To sync: commit/push from one machine, then pull on the other.

---

## Part H — Git save/sync workflow (optional)

### H1. See what changed
```bash
git status
```

### H2. Add + commit + push
```bash
git add -A
git commit -m "Describe your change"
git push
```

If commit fails with “Author identity unknown”, set:
```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

---

## Part I — Power off cleanly

When done:
```bash
sudo shutdown -h now
```

Wait ~20 seconds, then unplug power.

---

## Part J — One-command "full pipeline" (copy/paste)

Complete pipeline from start to finish:

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
python3 scripts/compute_local_anomaly_v2.py   --in data/processed/mag_data_clean.csv   --radius 0.10   --plot

# Step 4: Generate heatmap (outputs to data/exports/ automatically)
python3 scripts/interpolate_to_heatmapV1.py   --in data/processed/mag_data_anomaly.csv   --value-col local_anomaly

# Step 5: Organize run data
./tools/new_run.sh

ls data/raw
ls data/processed
ls data/exports
ls data/runs
```