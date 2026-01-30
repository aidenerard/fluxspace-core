# 2D Pipeline — Quick Runbook

Grid survey: collect magnetometer data on a 2D grid, validate, compute anomalies, and generate heatmaps. All commands are run from the **repo root**; **`pipelines/2d/`** is the preferred entrypoint for Python (run from repo root).

---

## New structure

- **Preferred:** Run from repo root: `python3 pipelines/2d/mag_to_csv.py`, etc.
- **Legacy:** Wrappers under `scripts/2d/` still work; see [Legacy commands](#legacy-commands) below.

---

## Quick start (copy/paste)

```bash
cd ~/fluxspace-core
mkdir -p data/raw data/processed data/exports data/runs

# 1. Collect data (grid survey)
python3 pipelines/2d/mag_to_csv.py --out data/raw/mag_data.csv

# 2. Validate and clean
python3 pipelines/2d/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv --drop-outliers

# 3. Compute local anomaly
python3 pipelines/2d/compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --radius 0.10 --plot

# 4a. B_total heatmap
python3 pipelines/2d/interpolate_to_Btotal_heatmap.py --in data/processed/mag_data_clean.csv --units uT

# 4b. Anomaly heatmap
python3 pipelines/2d/interpolate_to_heatmapV1.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly

# 5. Organize run (2D)
./tools/2d/new_run.sh
```

Expected: `data/raw/mag_data.csv` → `data/processed/mag_data_clean.csv` → `data/processed/mag_data_anomaly.csv` → heatmaps in `data/exports/` (or processed); run snapshot in `data/runs/<timestamp>/`.

---

## Pipeline steps (summary)

| Step | Script (entrypoint) | Input | Output |
|------|----------------------|--------|--------|
| 1. Collect | `pipelines/2d/mag_to_csv.py` | — | `data/raw/mag_data.csv` |
| 2. Validate | `pipelines/2d/validate_and_diagnosticsV1.py` | raw CSV | `data/processed/*_clean.csv`, report, plots |
| 3. Anomaly | `pipelines/2d/compute_local_anomaly_v2.py` | clean CSV | `data/processed/*_anomaly.csv` |
| 4a. B_total heatmap | `pipelines/2d/interpolate_to_Btotal_heatmap.py` | clean CSV | grid CSV + heatmap PNG |
| 4b. Anomaly heatmap | `pipelines/2d/interpolate_to_heatmapV1.py` | anomaly CSV | grid CSV + heatmap PNG |
| 5. Snapshot run | `./tools/2d/new_run.sh` | current data/ | `data/runs/<timestamp>/` |

---

## Optional

- **Calibration (offline):** `python3 pipelines/2d/calibrate_magnetometerV1.py --in data/raw/mag_cal.csv --method ellipsoid --earth-field-ut 52`
- **Backup runs to USB:** `./tools/2d/backup_runs_to_usb.sh` (see main runbook for mount/unmount)
- **V3 heatmap (optional):** `python3 pipelines/2d/interpolate_to_heatmapV3_gpr.py --in data/processed/mag_data_anomaly.csv --out-dir data/exports --method gpr --also-grad` → standard + fixed-scale + gradient heatmaps. Use `--method idw` for V2-style IDW; use `--fixed-vmin` / `--fixed-vmax` for fixed scale. Requires `scikit-learn` for GPR. See [interpolate_to_heatmapV3_gpr_explanation.md](interpolate_to_heatmapV3_gpr_explanation.md).

For full setup and 3D pipeline, see [raspberry_pi_setup.md](../raspberry_pi_setup.md) and [PIPELINE_3D.md](../3d/PIPELINE_3D.md).

---

## Script and tool explanations

**Pipeline scripts:** [mag_to_csv_explanation.md](mag_to_csv_explanation.md), [validate_and_diagnostics_explanation.md](validate_and_diagnostics_explanation.md), [compute_local_anomaly_v2_explanation.md](compute_local_anomaly_v2_explanation.md), [interpolate_to_Btotal_heatmap_explanation.md](interpolate_to_Btotal_heatmap_explanation.md), [interpolate_to_heatmap_explanation.md](interpolate_to_heatmap_explanation.md), [calibrate_magnetometer_explanation.md](calibrate_magnetometer_explanation.md). **GPR (optional):** [interpolate_to_heatmapV3_gpr_explanation.md](interpolate_to_heatmapV3_gpr_explanation.md).

**2D tools:**

| Tool | Description |
|------|-------------|
| [new_run_explanation.md](new_run_explanation.md) | `./tools/2d/new_run.sh` — snapshot run to `data/runs/<timestamp>/` |
| [backup_runs_to_usb_explanation.md](backup_runs_to_usb_explanation.md) | `./tools/2d/backup_runs_to_usb.sh` — back up `data/runs/` to USB |
| [setup_pi_explanation.md](setup_pi_explanation.md) | `./tools/2d/setup_pi.sh` — Pi system + Python setup for 2D |
| [mmc5983ma_smoketest_explanation.md](mmc5983ma_smoketest_explanation.md) | `python3 tools/2d/mmc5983ma_smoketest.py` — I2C smoketest for MMC5983MA |

---

## Legacy commands

Wrappers under **`scripts/2d/`** remain for backward compatibility. Preferred usage is **`python3 pipelines/2d/<script>.py`** (see above).
