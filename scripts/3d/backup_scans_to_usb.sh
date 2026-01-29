#!/usr/bin/env bash
set -euo pipefail

# Back up ONLY 3D scans (data/scans/) to USB in a separate directory.
# Does NOT modify tools/backup_runs_to_usb.sh or the 2D runs backup.

SRC="${FLUXSPACE_CORE:-$HOME/fluxspace-core}/data/scans/"
DST="/media/fluxspace/FLUXSPACE/fluxspace_scans_backup/"
MOUNT_POINT="/media/fluxspace/FLUXSPACE"

echo "🔍 Checking USB mount..."

if [[ ! -d "$MOUNT_POINT" ]]; then
  echo "❌ USB not mounted at $MOUNT_POINT"
  echo "   Plug it in and confirm with: ls /media/fluxspace"
  exit 1
fi

if ! mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
  echo "⚠️  Warning: $MOUNT_POINT exists but doesn't appear to be a mount point"
  echo "   Try: sudo mount /dev/sda1 $MOUNT_POINT (adjust device as needed)"
  exit 1
fi

if [[ ! -w "$MOUNT_POINT" ]]; then
  echo "❌ USB mount point is not writable: $MOUNT_POINT"
  echo "   Check permissions: ls -ld $MOUNT_POINT"
  exit 1
fi

if [[ ! -d "$SRC" ]]; then
  echo "❌ Source directory does not exist: $SRC"
  exit 1
fi

echo "✅ USB mounted and writable"
echo "📁 Creating destination directory if needed..."
mkdir -p "$DST"

if [[ ! -w "$DST" ]]; then
  echo "❌ Cannot write to destination: $DST"
  echo "   Check permissions: ls -ld $DST"
  exit 1
fi

echo "🔄 Starting rsync backup (3D scans)..."
if rsync -av --delete "$SRC" "$DST"; then
  echo ""
  echo "✅ Backed up scans: $SRC -> $DST"
else
  echo ""
  echo "❌ rsync failed with exit code $?"
  echo "   Common causes: USB filesystem errors, read-only mount, disk space, permissions."
  exit 1
fi
