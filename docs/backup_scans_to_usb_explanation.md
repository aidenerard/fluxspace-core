# Explanation of `backup_scans_to_usb.sh`

This document explains the script that backs up **only 3D scans** (`data/scans/`) to a USB drive in a separate directory. It does **not** modify `tools/backup_runs_to_usb.sh` or the 2D runs backup.

---

## Overview

**Purpose:** Sync 3D scan folders to USB so they are stored separately from 2D run backups.

**What it does:**
- **Source:** `$HOME/fluxspace-core/data/scans/` (or `$FLUXSPACE_CORE/data/scans/` if set)
- **Destination:** `/media/fluxspace/FLUXSPACE/fluxspace_scans_backup/`
- Performs the same class of checks as the existing runs backup script:
  - USB mount point exists and is actually mounted
  - Mount point is writable
  - Source directory exists
- Creates the destination directory if needed
- Runs **rsync -av --delete** to mirror `data/scans/` to the USB folder
- Prints a clear summary: “Backed up scans: <src> -> <dest>”

**Typical usage:**
```bash
cd ~/fluxspace-core
# Mount USB first (see main runbook: mount at /media/fluxspace/FLUXSPACE)
./scripts/backup_scans_to_usb.sh
```

---

## Prerequisites

- USB drive mounted at `/media/fluxspace/FLUXSPACE` (same as for 2D runs backup).
- Mount point writable; see the main runbook for mount/unmount and safe eject procedure.

---

## Behavior details

- **Checks:** Same style as the existing backup script: mount point present, `mountpoint -q`, writable, source exists, destination writable after `mkdir -p`.
- **rsync:** Uses `-av --delete` for consistency with the 2D runs backup (archive, verbose; delete files on USB that no longer exist under `data/scans/`).
- **Exit:** Exits non-zero on failure (e.g. USB not mounted, rsync error) and prints guidance.

---

## Relation to 2D backup

- **2D runs:** `./tools/backup_runs_to_usb.sh` → backs up `data/runs/` to `fluxspace_runs_backup/` (unchanged).
- **3D scans:** `./scripts/backup_scans_to_usb.sh` → backs up `data/scans/` to `fluxspace_scans_backup/`.

Both use the same USB mount path; only the source and destination subfolders differ.
