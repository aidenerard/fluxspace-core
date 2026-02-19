# Explanation of `run_paths.py`

This document explains the shared helper module that provides consistent path resolution for all FluxSpace 3D pipeline scripts.

---

## Overview

**`run_paths.py`** is a small utility module (not a standalone script) imported by other pipeline scripts. It enforces the canonical run-folder layout and provides functions to resolve, create, and navigate run directories.

By centralising path logic in one place, all scripts agree on where raw inputs, processed outputs, and exports live — regardless of whether the run is on a local disk, a USB drive, or a custom path.

---

## Canonical run-folder layout

```
RUN_DIR/
├── raw/            # Sensor data: oak_rgbd/, mag_run.csv, extrinsics.json, calibration.json
├── processed/      # Derived outputs: trajectory.csv, meshes, point clouds, mag_world.csv
└── exports/        # Final outputs: volume.npz, screenshots
```

---

## Functions

### `resolve_run_dir(cli_arg: str | None = None) -> Path`

Resolves a run directory from one of two sources (in order):

1. **`cli_arg`** — a CLI argument like `--run data/runs/run_20260210_1430`.
2. **`$RUN_DIR` environment variable** — used when the CLI argument is not provided.

Returns the resolved, absolute `Path`. Raises `ValueError` if neither source is available, with a message instructing the user to pass `--run` or set `$RUN_DIR`.

### `raw_dir(run: Path) -> Path`

Returns `run / "raw"`.

### `processed_dir(run: Path) -> Path`

Returns `run / "processed"`.

### `exports_dir(run: Path) -> Path`

Returns `run / "exports"`.

### `ensure_dirs(run: Path) -> None`

Creates `raw/`, `processed/`, and `exports/` under the given run directory if they don't already exist. Uses `mkdir(parents=True, exist_ok=True)`.

### `infer_run_dir_from_path(p: Path) -> Path | None`

Walks up to 6 parent directories from `p` looking for a directory that contains a `raw/` or `processed/` subdirectory. Returns the inferred run directory, or `None` if not found.

This is used when a script receives only an input file path (e.g. `--in $RUN_DIR/raw/oak_rgbd`) and needs to find the run root to determine output paths.

---

## Example usage (inside other scripts)

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_paths import resolve_run_dir, ensure_dirs, infer_run_dir_from_path

# From CLI --run argument
run = resolve_run_dir(args.run)
ensure_dirs(run)

# From an input path
run = infer_run_dir_from_path(Path(args.input_dir))
if run is None:
    print("Could not determine run directory")
```

---

## Why `sys.path.insert` is needed

The `pipelines/3d/` directory name contains a digit (`3d`), which is not a valid Python package identifier. This means you cannot write `from pipelines.3d.run_paths import ...`. Instead, scripts add the directory to `sys.path` and import directly:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_paths import resolve_run_dir
```

---

## Scripts that use `run_paths.py`

| Script | Functions used |
|--------|---------------|
| `open3d_reconstruct.py` | `infer_run_dir_from_path` |
| `fuse_mag_with_trajectory.py` | `resolve_run_dir`, `raw_dir`, `processed_dir` |
| `view_scan_toggle.py` | `resolve_run_dir` |

---

## Dependencies

Standard library only (`os`, `pathlib`). No external packages required.

---

## Relation to other 3D scripts

- **Used by:** Most pipeline scripts import this module for consistent path handling.
- **Shell counterpart:** `run_all_3d.sh` implements the same path logic in bash (directory creation, input resolution) and passes explicit paths to each Python script.

See [PIPELINE_3D.md](PIPELINE_3D.md) for the full 3D runbook.
