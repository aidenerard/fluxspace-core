#!/usr/bin/env bash
set -euo pipefail

# New York time, readable folder name
RUN_ID=$(TZ="America/New_York" date +"%m-%d-%Y_%H-%M")
RUN_DIR="data/runs/$RUN_ID"

mkdir -p "$RUN_DIR"/{raw,processed,exports}

echo "Run folder: $RUN_DIR"

# Snapshot whatever is currently in the latest folders
cp -a data/raw/.       "$RUN_DIR/raw/"       || true
cp -a data/processed/. "$RUN_DIR/processed/" || true
cp -a data/exports/.   "$RUN_DIR/exports/"   || true

echo "Copied latest outputs into $RUN_DIR"
