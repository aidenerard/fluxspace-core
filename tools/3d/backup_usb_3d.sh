#!/usr/bin/env bash
set -euo pipefail

# Wrapper: back up 3D scans to USB only.
# One obvious command for 3D scan backup.

echo "Backing up 3D scans to USB…"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if "$SCRIPT_DIR/backup_scans_to_usb.sh"; then
  echo "Backup complete: 3D scans"
else
  exit 1
fi
