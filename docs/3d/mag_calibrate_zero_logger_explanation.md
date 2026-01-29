# Explanation of `mag_calibrate_zero_logger.py`

This document explains the 3D pipeline script that performs **live calibration**, **baseline (zero) measurement**, and **continuous magnetometer logging** for 3D fusion. It is an alternative to `mag_to_csv_v2` when you want calibration and zeroing **before** each run.

---

## Overview

**`mag_calibrate_zero_logger.py`** runs in three phases:

1. **CALIBRATE:** You move/rotate the sensor (and rigid rig) through many orientations for a set duration. The script fits hard‑iron offset and optional soft‑iron (diagonal) scaling from min/max ranges.
2. **ZERO:** You hold the sensor still in the intended **start pose**. The script averages the corrected field over a few seconds and uses that as the baseline. During logging, baseline‑subtracted values are written as `zero_mag`.
3. **LOG:** Continuous timestamped logging until you stop (Ctrl+C or `q` + Enter). Press Enter to insert MARK rows (e.g. `start` / `end`) for alignment with the trajectory.

**Output CSV** includes `raw_bx/by/bz`, `corr_bx/by/bz`, `zero_bx/by/bz`, and magnitudes `raw_mag`, `corr_mag`, `zero_mag`, plus `t_rel_s`, `row_type` (SAMPLE / MARK / INFO), and `note`. Optional **`--save-cal`** writes a JSON with calibration parameters and baseline.

---

## What it does

- **Sensor:** MMC5983MA over I2C (same hardware as 2D). Uses `qwiic_mmc5983ma`.
- **Calibration:** Min/max per axis → hard‑iron offset (box center) and, unless `--no-softiron`, diagonal soft‑iron scales so axis ranges match.
- **Zeroing:** Mean of corrected vectors during the zero phase → baseline. Logged values use `zero_mag` = magnitude of `(corr - baseline)`.
- **Optional rescaling:** `--expected-field-ut` rescales corrected magnitude to a target (e.g. local Earth field); only when `--units uT`.
- **Threading:** A background thread reads stdin for MARKs (Enter = MARK, `q` + Enter = quit) so you can log and mark without stopping.

---

## CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--out` | `data/raw/mag_run.csv` | Output CSV path. |
| `--hz` | 80 | Target logging rate (Hz) during LOG phase. |
| `--units` | `uT` | Output units: `uT` or `gauss`. |
| `--samples` | 1 | Sensor reads to average per logged sample. |
| `--sample-delay` | 0 | Delay (s) between averaged reads. |
| `--calib-seconds` | 20 | Calibration motion duration (s). |
| `--zero-seconds` | 3 | Baseline averaging duration (s). |
| `--no-softiron` | — | Disable soft‑iron; use only hard‑iron offset. |
| `--save-cal` | — | Optional path to save calibration+baseline JSON. |
| `--expected-field-ut` | 0 | Optional rescale corrected \|B\| to this (uT); 0 = off. |

---

## Example usage

```bash
# Basic run (calibrate 20s, zero 3s, then log until Ctrl+C)
python3 pipelines/3d/mag_calibrate_zero_logger.py \
  --out "$RUN_DIR/raw/mag_run.csv" \
  --units uT

# Save calibration JSON, rescale to ~52 µT
python3 pipelines/3d/mag_calibrate_zero_logger.py \
  --out "$RUN_DIR/raw/mag_run.csv" \
  --save-cal "$RUN_DIR/raw/calibration.json" \
  --expected-field-ut 52 \
  --calib-seconds 25 \
  --zero-seconds 5
```

During LOG, press **Enter** to add a MARK (optionally type a label first, e.g. `start` or `end`). Type **`q`** + Enter to stop.

---

## When to use vs `mag_to_csv_v2`

- **`mag_calibrate_zero_logger`:** Use when you want **per‑run** calibration and zeroing before logging. Produces `zero_mag`, `corr_mag`, `raw_mag`; `fuse_mag_with_trajectory` can use `zero_mag` with `--value-type zero_mag`.
- **`mag_to_csv_v2`:** Simpler continuous logger only. No calibration/zero; you can baseline‑subtract later (e.g. median) when fusing.

---

## Relation to other 3D scripts

- **Output** is used as the `--mag` input to **`fuse_mag_with_trajectory`**.
- **Extrinsics** (`extrinsics.json`) and **trajectory** (`trajectory.csv`) are still required for fusion.

See [PIPELINE_3D.md](PIPELINE_3D.md) for the full 3D runbook.
