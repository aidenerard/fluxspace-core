# Explanation of `backup_usb_3d.sh`

This document explains the wrapper script that provides a single, obvious command to back up **3D scans only** to USB.

---

## Overview

**Purpose:** One command to back up 3D scan data to USB without touching the 2D runs backup.

**What it does:**
- Prints: “Backing up 3D scans to USB…”
- Calls `tools/3d/backup_scans_to_usb.sh` (from the same repo, using the script’s directory to resolve the path)
- On success: prints “Backup complete: 3D scans”
- On failure: exits with the same non-zero exit code as `backup_scans_to_usb.sh`

**Typical usage:**
```bash
cd ~/fluxspace-core
./tools/3d/backup_usb_3d.sh
```

---

## Why a wrapper

- **Clarity:** “backup_usb_3d” makes it obvious this is the 3D backup command.
- **Consistency:** Same pattern as having a single entry point for 3D backup; actual logic lives in `backup_scans_to_usb.sh`.

---

## Relation to other scripts

- **`tools/3d/backup_scans_to_usb.sh`** — Performs the actual rsync and checks; this script only invokes it.
- **`tools/2d/backup_runs_to_usb.sh`** — Unchanged; use it for 2D runs backup. Do not use `backup_usb_3d.sh` for 2D runs.
