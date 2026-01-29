#!/usr/bin/env bash
set -euo pipefail

# Create a new 3D scan run folder under data/scans/ with snapshot of current data.
# Naming: data/scans/<RUN_ID>__3d or data/scans/<RUN_ID>__3d__<label>
# RUN_ID = MM-DD-YYYY_HH-MM (America/New_York)
# Does NOT modify the 2D pipeline or tools/new_run.sh.

usage() {
  echo "Usage: $0 [--label <string>] [-h|--help]"
  echo "  Creates data/scans/<RUN_ID>__3d/ (or __3d__<label>) and snapshots data/raw, data/processed, data/exports."
  echo "  RUN_ID = MM-DD-YYYY_HH-MM (America/New_York)."
  echo "  --label   Optional suffix after __3d (e.g. block01)."
  echo "  -h, --help  Show this message."
}

LABEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --label requires a value" >&2
        exit 1
      fi
      LABEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Run from repo root (same as 2D script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID=$(TZ="America/New_York" date +"%m-%d-%Y_%H-%M")
if [[ -n "$LABEL" ]]; then
  SCAN_DIR="data/scans/${RUN_ID}__3d__${LABEL}"
else
  SCAN_DIR="data/scans/${RUN_ID}__3d"
fi

mkdir -p "$SCAN_DIR"/{raw,processed,exports,config}

# Snapshot current working folders (same pattern as 2D run script)
cp -a data/raw/.       "$SCAN_DIR/raw/"      2>/dev/null || true
cp -a data/processed/. "$SCAN_DIR/processed/" 2>/dev/null || true
cp -a data/exports/.   "$SCAN_DIR/exports/"   2>/dev/null || true

echo "Scan folder: $SCAN_DIR"
echo "Copied latest outputs into $SCAN_DIR"
