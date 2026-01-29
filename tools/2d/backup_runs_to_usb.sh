#!/usr/bin/env bash
set -euo pipefail

SRC="${FLUXSPACE_CORE:-$HOME/fluxspace-core}/data/runs/"
DST="/media/fluxspace/FLUXSPACE/fluxspace_runs_backup/"
MOUNT_POINT="/media/fluxspace/FLUXSPACE"

# sanity checks
echo "🔍 Checking USB mount..."

if [ ! -d "$MOUNT_POINT" ]; then
  echo "❌ USB not mounted at $MOUNT_POINT"
  echo "   Plug it in and confirm with: ls /media/fluxspace"
  exit 1
fi

# Check if mount point is actually a mount (not just a directory)
if ! mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
  echo "⚠️  Warning: $MOUNT_POINT exists but doesn't appear to be a mount point"
  echo "   This might indicate the USB wasn't mounted correctly"
  echo "   Try: sudo mount /dev/sda1 $MOUNT_POINT (adjust device as needed)"
  exit 1
fi

# Check if mount point is writable
if [ ! -w "$MOUNT_POINT" ]; then
  echo "❌ USB mount point is not writable: $MOUNT_POINT"
  echo "   Check permissions: ls -ld $MOUNT_POINT"
  echo "   You may need to: sudo chmod 755 $MOUNT_POINT"
  exit 1
fi

# Check if source directory exists
if [ ! -d "$SRC" ]; then
  echo "❌ Source directory does not exist: $SRC"
  exit 1
fi

echo "✅ USB mounted and writable"
echo "📁 Creating destination directory if needed..."
mkdir -p "$DST"

# Verify destination directory was created and is writable
if [ ! -w "$DST" ]; then
  echo "❌ Cannot write to destination: $DST"
  echo "   Check permissions: ls -ld $DST"
  exit 1
fi

echo "🔄 Starting rsync backup..."
if rsync -av --delete "$SRC" "$DST"; then
  echo ""
  echo "✅ Backup complete:"
  echo "   $SRC  ->  $DST"
else
  echo ""
  echo "❌ rsync failed with exit code $?"
  echo "   Common causes:"
  echo "   - USB filesystem errors (try: sudo fsck /dev/sda1)"
  echo "   - USB is read-only (check: mount | grep $MOUNT_POINT)"
  echo "   - Insufficient disk space (check: df -h $MOUNT_POINT)"
  echo "   - Permission issues (check: ls -ld $MOUNT_POINT)"
  exit 1
fi
