# Explanation of `new_run.sh`

This document explains the run organization script that archives pipeline outputs.

---

## Overview

This bash script organizes completed pipeline runs by creating a timestamped folder and copying all current outputs from the pipeline into it. This helps keep data organized and allows you to compare results across multiple measurement sessions.

**What it does:**
- Creates a new timestamped folder in `data/runs/` (e.g., `12-28-2024_14-30`)
- Copies all files from `data/raw/`, `data/processed/`, and `data/exports/` into the run folder
- Preserves the directory structure (raw, processed, exports subfolders)
- Provides feedback about what was copied

**Typical usage:**
```bash
./tools/new_run.sh
```

**Expected output:**
- A new folder created at `data/runs/<timestamp>/`
- All current pipeline outputs copied into that folder
- Console messages confirming the operation

---

## Code Explanation

### Script Header (Lines 1-2)

```bash
#!/usr/bin/env bash
set -euo pipefail
```

**What it does:**
- `#!/usr/bin/env bash`: Shebang that makes the script executable and specifies bash
- `set -euo pipefail`: Bash safety flags
  - `-e`: Exit immediately if any command fails
  - `-u`: Treat unset variables as errors
  - `-o pipefail`: Return exit code of the rightmost failing command in a pipeline

### Timestamp Generation (Lines 4-6)

```bash
RUN_ID=$(TZ="America/New_York" date +"%m-%d-%Y_%H-%M")
RUN_DIR="data/runs/$RUN_ID"
```

**What it does:**
- Creates a timestamp in format: `MM-DD-YYYY_HH-MM` (e.g., `12-28-2024_14-30`)
- Uses Eastern Time (America/New_York timezone)
- Constructs the full path to the run directory: `data/runs/<timestamp>/`

**Why this format?**
- Human-readable date and time
- Sorts chronologically when listed alphabetically
- No spaces or special characters that could cause issues

### Directory Creation (Line 8)

```bash
mkdir -p "$RUN_DIR"/{raw,processed,exports}
```

**What it does:**
- Creates the run directory and three subdirectories: `raw/`, `processed/`, `exports/`
- `-p`: Creates parent directories if they don't exist (no error if they do)
- Uses brace expansion to create all three subdirectories at once

### File Copying (Lines 12-15)

```bash
cp -a data/raw/.       "$RUN_DIR/raw/"       || true
cp -a data/processed/. "$RUN_DIR/processed/" || true
cp -a data/exports/.   "$RUN_DIR/exports/"   || true
```

**What it does:**
- Copies all contents from each source directory to the corresponding run subdirectory
- `-a`: Archive mode (preserves permissions, timestamps, etc.)
- `data/raw/.`: The `.` means "contents of this directory" (not the directory itself)
- `|| true`: Continues even if a directory is empty (prevents script failure)

**What gets copied:**
- From `data/raw/`: Original sensor data CSV files
- From `data/processed/`: Cleaned data, anomaly files, diagnostic reports, plots
- From `data/exports/`: Grid CSVs, heatmap PNGs, final outputs

---

## When to Use This Script

Use this script:
- After completing a full pipeline run (E1 → E4 in the runbook)
- When you want to archive current outputs before starting a new measurement
- To organize multiple measurement sessions for comparison
- Before clearing or overwriting files in the main data directories

**Typical workflow:**
1. Run the full pipeline: `mag_to_csv.py` → `validate_and_diagnosticsV1.py` → `compute_local_anomaly_v2.py` → heatmap visualization
   - **B_total heatmap:** `interpolate_to_Btotal_heatmap.py` (field strength visualization)
   - **Anomaly heatmap:** `interpolate_to_heatmapV1.py` (anomaly detection visualization)
2. Run `./tools/new_run.sh` to archive all outputs
3. Start a new measurement session (files in `data/raw/`, etc. can be overwritten)

---

## Integration with Pipeline

This script is referenced in the Raspberry Pi setup runbook (Part E5) as the final step after completing a full pipeline run.

**Complete pipeline workflow:**
```bash
# Step 1: Collect data
python3 scripts/mag_to_csv.py

# Step 2: Validate and clean
python3 scripts/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv

# Step 3: Compute anomalies
python3 scripts/compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --radius 0.10 --plot

# Step 4a: Create B_total heatmap (field strength)
python3 scripts/interpolate_to_Btotal_heatmap.py --in data/processed/mag_data_clean.csv --units uT --grid-step 0.01

# Step 4b: Create anomaly heatmap (anomaly detection)
python3 scripts/interpolate_to_heatmapV1.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly --grid-step 0.01

# Step 5: Organize run data
./tools/new_run.sh
```

---

## Output Structure

After running the script, your `data/runs/` directory will look like:

```
data/runs/
├── 12-28-2024_14-30/
│   ├── raw/
│   │   └── mag_data.csv
│   ├── processed/
│   │   ├── mag_data_clean.csv
│   │   ├── mag_data_anomaly.csv
│   │   ├── mag_data_report.txt
│   │   └── mag_data_*.png
│   └── exports/
│       ├── mag_data_grid.csv
│       └── mag_data_heatmap.png
└── 12-28-2024_15-45/
    └── ...
```

This structure allows you to:
- Compare results across different measurement sessions
- Keep a history of all your runs
- Easily identify when each measurement was taken

---

## Notes

- The script copies files (doesn't move them), so original files remain in `data/raw/`, etc.
- If a source directory is empty, the copy command will fail silently (due to `|| true`)
- The timestamp uses Eastern Time - adjust `TZ` in the script if you need a different timezone
- Make sure the script is executable: `chmod +x tools/new_run.sh`

---

## Alternative: 3D scan storage (data/scans/*__3d)

The **2D pipeline** and `./tools/new_run.sh` are unchanged. For the **3D pipeline** (Polycam/RTAB-Map + magnetometer fusion), a separate script creates scan snapshots under `data/scans/` so 3D runs stay visually distinct:

- **Script:** `./scripts/new_3d_scan.sh` (does not call or modify `tools/new_run.sh`)
- **Folder:** `data/scans/<RUN_ID>__3d/` or `data/scans/<RUN_ID>__3d__<label>/`
- **RUN_ID:** Same format: `MM-DD-YYYY_HH-MM` (America/New_York)

**Create a 3D scan snapshot:**

```bash
./scripts/new_3d_scan.sh
./scripts/new_3d_scan.sh --label block01
```

**Examples:** `data/scans/01-29-2026_13-57__3d/`, `data/scans/01-29-2026_13-57__3d__block01/`

**Back up 3D scans to USB:** Use `./scripts/backup_usb_3d.sh` (backs up only `data/scans/` to a separate USB folder). The 2D runs backup (`./tools/backup_runs_to_usb.sh`) is unchanged.

