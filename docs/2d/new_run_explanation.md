# Explanation of `tools/2d/new_run.sh`

This document explains the 2D pipeline tool that creates a **timestamped run folder** and snapshots the current `data/raw`, `data/processed`, and `data/exports` contents into it.

---

## Overview

**`./tools/2d/new_run.sh`** creates a new folder under `data/runs/<RUN_ID>/` and copies the latest pipeline outputs into it. The run ID uses **America/New_York** time in `MM-DD-YYYY_HH-MM` format. The script always runs from the **repo root** (it `cd`s there from `tools/2d/`), so you can run it from any directory.

**Input:** Current contents of `data/raw/`, `data/processed/`, and `data/exports/` (by copy, not move).

**Output:** `data/runs/<RUN_ID>/` with `raw/`, `processed/`, and `exports/` subfolders containing those snapshots.

---

## What it does

1. **Resolve paths:** Sets `SCRIPT_DIR` to the directory of the script (`tools/2d/`), then `REPO_ROOT` to the repository root (`SCRIPT_DIR/../..`). Changes into `REPO_ROOT`.
2. **Run ID:** `RUN_ID=$(TZ="America/New_York" date +"%m-%d-%Y_%H-%M")` (e.g. `01-29-2026_16-15`).
3. **Create folder:** `mkdir -p data/runs/$RUN_ID/{raw,processed,exports}`.
4. **Snapshot:** `cp -a data/raw/. data/runs/$RUN_ID/raw/` (and similarly for `processed/`, `exports/`). Uses `|| true` so missing or empty source folders don’t fail the script.
5. **Print:** Echoes the run folder path and that copies finished.

---

## Example usage

```bash
cd ~/fluxspace-core
./tools/2d/new_run.sh
```

Typical flow: run the 2D pipeline (collect → validate → anomaly → heatmaps), then run `./tools/2d/new_run.sh` to archive that run.

---

## Relation to 2D pipeline

- **Step 5** in the 2D runbook: after heatmaps, run `./tools/2d/new_run.sh` to snapshot the run.
- **Backup:** Use `./tools/2d/backup_runs_to_usb.sh` to back up `data/runs/` to USB. See [backup_runs_to_usb_explanation.md](backup_runs_to_usb_explanation.md) and [raspberry_pi_setup.md](../raspberry_pi_setup.md).

---

## 3D alternative

For 3D scans, use `./tools/3d/new_3d_scan.sh` instead; it writes to `data/scans/<RUN_ID>__3d/` and does not modify `tools/2d/new_run.sh`. See [PIPELINE_3D.md](../3d/PIPELINE_3D.md) and [new_3d_scan_explanation.md](../3d/new_3d_scan_explanation.md).
