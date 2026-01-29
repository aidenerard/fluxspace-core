# Explanation of `new_3d_scan.sh`

This document explains the 3D scan snapshot script that creates timestamped folders under `data/scans/` and copies current pipeline outputs into them. It does **not** modify the 2D pipeline or `tools/2d/new_run.sh`.

---

## Overview

**Purpose:** Create a new “3D scan run” folder under `data/scans/` using the same timestamp pattern as the 2D run script, but in a separate tree so 3D workflows stay distinct.

**What it does:**
- Generates a run ID in **America/New_York** time: `MM-DD-YYYY_HH-MM`
- Creates a folder: `data/scans/<RUN_ID>__3d/` or, with a label, `data/scans/<RUN_ID>__3d__<label>/`
- Creates subfolders: `raw/`, `processed/`, `exports/`, `config/`
- **Snapshot copies** current working data into the scan folder:
  - `data/raw/` → `data/scans/<...>/raw/`
  - `data/processed/` → `data/scans/<...>/processed/`
  - `data/exports/` → `data/scans/<...>/exports/`
- Prints the created folder path

**Typical usage:**
```bash
cd ~/fluxspace-core
./tools/3d/new_3d_scan.sh
./tools/3d/new_3d_scan.sh --label block01
```

**Expected output:**
- A new folder, e.g. `data/scans/01-29-2026_13-57__3d/` or `data/scans/01-29-2026_13-57__3d__block01/`
- Subfolders `raw/`, `processed/`, `exports/`, `config/` with current data copied into the first three

---

## CLI

| Option | Description |
|--------|-------------|
| `--label <string>` | Optional suffix after `__3d` (e.g. `block01`). Folder becomes `<RUN_ID>__3d__<label>`. |
| `-h`, `--help` | Print usage and exit. |

---

## Behavior details

- **Timestamp:** `RUN_ID=$(TZ="America/New_York" date +"%m-%d-%Y_%H-%M")` — same convention as the 2D run script.
- **Copy:** Uses `cp -a` (or equivalent) with safe quoting; copies fail gracefully (`|| true`) if a source directory is missing or empty.
- **Repo root:** Script expects to be run from the repo root (uses `REPO_ROOT` if set, else `$HOME/fluxspace-core`).
- **No 2D changes:** Does not call or edit `tools/2d/new_run.sh` or any 2D pipeline script.

---

## When to use

- After a 3D capture/processing session (Polycam/RTAB-Map + magnetometer fusion).
- When you want to archive the current `data/raw`, `data/processed`, and `data/exports` into a timestamped 3D scan folder.
- When you want 3D snapshots separate from 2D runs (e.g. `data/scans/` vs `data/runs/`).

---

## Relation to 2D pipeline

- **2D:** `./tools/2d/new_run.sh` → `data/runs/<RUN_ID>/` (unchanged).
- **3D:** `./tools/3d/new_3d_scan.sh` → `data/scans/<RUN_ID>__3d/` or `data/scans/<RUN_ID>__3d__<label>/`.

Backup is also separate: 2D runs → `./tools/2d/backup_runs_to_usb.sh`; 3D scans → `./tools/3d/backup_usb_3d.sh`.
