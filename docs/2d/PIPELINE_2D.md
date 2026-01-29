# 2D Pipeline — Quick Runbook

Grid survey: collect magnetometer data on a 2D grid, validate, compute anomalies, and generate heatmaps. All commands are run from the **repo root**; **`scripts/`** is the stable entrypoint folder.

---

## New structure

- **`scripts/`** = **entrypoints** (stable). Run e.g. `python3 scripts/mag_to_csv.py`. Implementations live in **`pipelines/2d/`**; scripts in `scripts/` are thin wrappers.
- **Legacy:** Commands like `python3 scripts/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv` still work unchanged.

---

## Quick start (copy/paste)

```bash
cd ~/fluxspace-core
mkdir -p data/raw data/processed data/exports data/runs

# 1. Collect data (grid survey)
python3 scripts/mag_to_csv.py --out data/raw/mag_data.csv

# 2. Validate and clean
python3 scripts/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv --drop-outliers

# 3. Compute local anomaly
python3 scripts/compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --radius 0.10 --plot

# 4a. B_total heatmap
python3 scripts/interpolate_to_Btotal_heatmap.py --in data/processed/mag_data_clean.csv --units uT

# 4b. Anomaly heatmap
python3 scripts/interpolate_to_heatmapV1.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly

# 5. Organize run (2D)
./tools/new_run.sh
```

Expected: `data/raw/mag_data.csv` → `data/processed/mag_data_clean.csv` → `data/processed/mag_data_anomaly.csv` → heatmaps in `data/exports/` (or processed); run snapshot in `data/runs/<timestamp>/`.

---

## Pipeline steps (summary)

| Step | Script (entrypoint) | Input | Output |
|------|----------------------|--------|--------|
| 1. Collect | `scripts/mag_to_csv.py` | — | `data/raw/mag_data.csv` |
| 2. Validate | `scripts/validate_and_diagnosticsV1.py` | raw CSV | `data/processed/*_clean.csv`, report, plots |
| 3. Anomaly | `scripts/compute_local_anomaly_v2.py` | clean CSV | `data/processed/*_anomaly.csv` |
| 4a. B_total heatmap | `scripts/interpolate_to_Btotal_heatmap.py` | clean CSV | grid CSV + heatmap PNG |
| 4b. Anomaly heatmap | `scripts/interpolate_to_heatmapV1.py` | anomaly CSV | grid CSV + heatmap PNG |
| 5. Snapshot run | `./tools/new_run.sh` | current data/ | `data/runs/<timestamp>/` |

---

## Optional

- **Calibration (offline):** `python3 scripts/calibrate_magnetometerV1.py --in data/raw/mag_cal.csv --method ellipsoid --earth-field-ut 52`
- **Backup runs to USB:** `./tools/backup_runs_to_usb.sh` (see main runbook for mount/unmount)

For full setup and 3D pipeline, see [raspberry_pi_setup.md](../raspberry_pi_setup.md) and [PIPELINE_3D.md](../3d/PIPELINE_3D.md).
