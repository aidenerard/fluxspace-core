# Explanation of `tools/2d/backup_runs_to_usb.sh`

This document explains the 2D pipeline tool that backs up **`data/runs/`** (2D run folders) to a USB drive.

---

## Overview

**`./tools/2d/backup_runs_to_usb.sh`** rsyncs `data/runs/` to a fixed USB destination, with checks that the USB is mounted and writable before copying. It does **not** touch 3D scan data (`data/scans/`); for that, use `./tools/3d/backup_usb_3d.sh`.

**Source:** `data/runs/` (or `$FLUXSPACE_CORE/data/runs/` if set).  
**Destination:** `/media/fluxspace/FLUXSPACE/fluxspace_runs_backup/`  
**Mount point:** `/media/fluxspace/FLUXSPACE`

---

## What it does

1. **Mount checks:** Verifies `/media/fluxspace/FLUXSPACE` exists, is a mount point (`mountpoint -q`), and is writable.
2. **Source check:** Verifies the `data/runs/` directory exists (using `FLUXSPACE_CORE` or `$HOME/fluxspace-core` for the repo root).
3. **Destination:** `mkdir -p` for the backup directory, then checks it is writable.
4. **Backup:** `rsync -av --delete` from source to destination. `--delete` removes files on the USB that are no longer in `data/runs/`.
5. **Errors:** On failure, prints common causes (USB read-only, full disk, permissions, etc.) and exits non‑zero.

---

## Example usage

```bash
# Ensure USB is mounted at /media/fluxspace/FLUXSPACE, then:
./tools/2d/backup_runs_to_usb.sh
```

Mount/unmount details: see [raspberry_pi_setup.md](../raspberry_pi_setup.md) (e.g. § 5.6.1, § 5.7).

---

## Relation to 2D pipeline

- Use after `./tools/2d/new_run.sh` to archive runs to USB.
- **3D scans:** Back up `data/scans/` with `./tools/3d/backup_usb_3d.sh`; see [backup_usb_3d_explanation.md](../3d/backup_usb_3d_explanation.md).
