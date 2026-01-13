#!/usr/bin/env bash
set -euo pipefail

SRC="$HOME/fluxspace-core/data/runs/"
DST="/media/fluxspace/FLUXSPACE/fluxspace_runs_backup/"

# sanity checks
if [ ! -d "/media/fluxspace/FLUXSPACE" ]; then
  echo "❌ USB not mounted at /media/fluxspace/FLUXSPACE"
  echo "Plug it in and confirm with: ls /media/fluxspace"
  exit 1
fi

mkdir -p "$DST"
rsync -av --delete "$SRC" "$DST"

echo "✅ Backup complete:"
echo "   $SRC  ->  $DST"
