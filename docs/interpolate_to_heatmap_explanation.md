# Complete Explanation of `interpolate_to_heatmapV1.py`

This document explains every part of the interpolation and heatmap generation script, step by step.

---

## ⚠️ Recent Changes

**Key updates to the script:**

1. **Output Directory Default Changed** ⚠️ **IMPORTANT**
   - **Previous behavior**: Output files were written to the same directory as the input file
   - **New behavior**: Output files default to `data/exports/` when input is in a path containing `data`
   - **Example**: Input `data/processed/file.csv` → Output `data/exports/file_grid.csv` (automatically)
   - You can still override with `--out-dir` if needed

2. **Argument name change**
   - `--outdir` → `--out-dir` (now uses hyphen for consistency)

3. **New `--no-plot` option**
   - Skip PNG heatmap generation if you only need the CSV grid data

4. **Improved code organization**
   - New `resolve_outdir()` function for output directory logic
   - Better error messages and code structure

---

## Overview

This script takes scattered measurement points (x, y, value) from a CSV file and interpolates them onto a regular grid using **IDW (Inverse Distance Weighting)**. It then exports both the grid data as CSV and a visual heatmap as PNG. This is useful for creating smooth, continuous visualizations from discrete measurement points.

**What it does:**
- Reads scattered (x, y, value) points from CSV
- Creates a regular grid covering the data extent
- Interpolates values onto the grid using IDW
- Exports grid data as CSV (long-form: x, y, value)
- Generates a heatmap PNG visualization
- Optionally filters out flagged data points

**Typical usage:**
```bash
# Basic usage (outputs to data/exports/ by default)
python3 scripts/interpolate_to_heatmapV1.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly

# Custom grid spacing
python3 scripts/interpolate_to_heatmapV1.py --in data/processed/mag_data_anomaly.csv --grid-step 0.05 --power 2.5
```

**Outputs:** ⚠️ **CHANGED: Default output location**
- `<stem>_grid.csv` - Regular grid with interpolated values (x, y, value format)
  - **Default location**: `data/exports/` (when input is in `data/processed/`)
- `<stem>_heatmap.png` - Visual heatmap image (optional with `--no-plot`)
  - **Default location**: `data/exports/` (when input is in `data/processed/`)

---

## Section 1: Imports and Setup (Lines 1-38)

```python
#!/usr/bin/env python3
"""
interpolate_to_heatmapv1.py

Takes scattered points (x, y, value) and interpolates them onto a regular grid,
then exports:
  - <stem>_grid.csv          (x,y,value on a grid, long-form table)
  - <stem>_heatmap.png       (quick preview)

Designed for your pipeline:
  validate_and_diagnostics.py  -> *_clean.csv
  compute_local_anomaly_v2.py  -> *_anomaly.csv
  interpolate_to_heatmapv1.py  -> grid + heatmap

Default interpolation: IDW (Inverse Distance Weighting) with a tunable power.

Example:
  python3 interpolate_to_heatmapv1.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly

If your x/y are in meters and you want 0.05 m grid spacing:
  python3 interpolate_to_heatmapv1.py --in ... --grid-step 0.05

Notes:
- Lightweight interpolator (no SciPy required).
- For dense grids over large datasets, naive IDW can be slow (O(N * grid_points)).
  For your 9x9 or 20x20 tests, it's totally fine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

**What it does:**

1. **`#!/usr/bin/env python3`** (Shebang)
   - Makes the script executable directly from the command line

2. **Module docstring** (Lines 2-27)
   - Documents the script's purpose and pipeline integration
   - Explains IDW interpolation method
   - Shows example usage
   - Notes performance characteristics

3. **`from __future__ import annotations`**
   - Enables postponed evaluation of type annotations
   - Allows cleaner type hints without quotes

4. **Standard library imports:**
   - `argparse`: Parses command-line arguments
   - `sys`: System-specific parameters (for error output with `sys.stderr`)
   - `pathlib.Path`: Modern path handling
   - `typing`: Type hints (`Optional`, `Tuple`)

5. **External library imports:**
   - `numpy`: Numerical operations and arrays
   - `pandas`: Data manipulation and CSV handling
   - `matplotlib.pyplot`: Plotting and visualization

---

## Section 2: Command-Line Arguments (Lines 44-105)

### Function: `parse_args()` (Lines 44-105)

```python
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interpolate scattered x,y,value points to a grid and save grid CSV + heatmap."
    )
    p.add_argument(
        "--in",
        dest="infile",
        required=True,
        help="Input CSV (e.g., data/processed/mag_data_anomaly.csv)",
    )
    p.add_argument(
        "--value-col",
        default="local_anomaly",
        help="Column to grid (default: local_anomaly)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. If omitted, defaults to data/exports when possible.",
    )
    p.add_argument(
        "--grid-step",
        type=float,
        default=None,
        help="Grid spacing in same units as x/y (e.g., 0.05). If omitted, uses --grid-n.",
    )
    p.add_argument(
        "--grid-n",
        type=int,
        default=200,
        help="Grid resolution per axis if --grid-step not given (default: 200).",
    )
    p.add_argument(
        "--power",
        type=float,
        default=2.0,
        help="IDW power parameter (default: 2.0). Higher = more local influence.",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Small epsilon to avoid divide-by-zero (default: 1e-12).",
    )
    p.add_argument(
        "--clip-percentile",
        type=float,
        default=99.0,
        help="Clip colormap to [100-clip, clip] percentiles for nicer plots (default: 99). "
        "Use 100 to disable clipping.",
    )
    p.add_argument(
        "--drop-flag-any",
        action="store_true",
        help="If set, drop rows where _flag_any is True (if present).",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="If set, do not generate the PNG heatmap (still writes the grid CSV).",
    )
    return p.parse_args()
```

**What it does:**

This function defines all command-line arguments the script accepts:

1. **`--in`** (required)
   - Input CSV file path
   - Example: `--in data/processed/mag_data_anomaly.csv`

2. **`--value-col`** (default: `"local_anomaly"`)
   - Which column to interpolate onto the grid
   - Could be `B_total`, `local_anomaly`, or any numeric column

3. **`--out-dir`** ⚠️ **CHANGED: Default behavior updated**
   - **NEW DEFAULT**: If omitted, automatically uses `data/exports` when the input file path contains a `data` directory
   - **Priority order:**
     1. If `--out-dir` is explicitly provided, use that directory
     2. If input path contains `data`, use `<...>/data/exports`
     3. Otherwise, fallback to `<input_dir>/exports`
   - **Example**: Input `data/processed/file.csv` → Output goes to `data/exports/` by default
   - **Example**: `--out-dir custom/path` → Uses `custom/path` instead

4. **`--grid-step`** (optional)
   - Grid spacing in same units as x/y coordinates
   - If provided, creates grid with this spacing
   - Example: `--grid-step 0.05` (5 cm spacing)
   - If omitted, uses `--grid-n` instead

5. **`--grid-n`** (default: 200)
   - Number of grid points per axis when `--grid-step` is not provided
   - Creates a 200×200 grid by default
   - Example: `--grid-n 100` creates 100×100 grid

6. **`--power`** (default: 2.0)
   - IDW power parameter (controls how quickly influence decays with distance)
   - Higher power = closer points have more influence
   - Typical values: 1.0 to 3.0

7. **`--eps`** (default: 1e-12)
   - Small epsilon value to prevent divide-by-zero errors
   - Added to distances before division
   - Very small number (0.000000000001)

8. **`--clip-percentile`** (default: 99.0)
   - Clips color scale for visualization
   - Uses 99th percentile to avoid extreme outliers dominating the color scale
   - Set to 100 to disable clipping

9. **`--drop-flag-any`** (flag, no value)
   - If present, drops rows where `_flag_any` column is True
   - Useful for filtering out problematic data points

10. **`--no-plot`** (flag, no value) ⚠️ **NEW**
    - If present, skips generating the PNG heatmap
    - Still writes the grid CSV file
    - Useful when you only need the data, not the visualization

**Example usage:**
```bash
# Basic usage (outputs to data/exports by default)
python3 scripts/interpolate_to_heatmapV1.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly

# Custom grid spacing
python3 scripts/interpolate_to_heatmapV1.py --in data/processed/file.csv --grid-step 0.05

# Custom power and explicit output directory
python3 scripts/interpolate_to_heatmapV1.py --in data/processed/file.csv --power 2.5 --out-dir data/exports

# Filter flagged data
python3 scripts/interpolate_to_heatmapV1.py --in data/processed/file.csv --drop-flag-any

# Generate only CSV, no PNG
python3 scripts/interpolate_to_heatmapV1.py --in data/processed/file.csv --no-plot
```

---

## Section 3: Output Directory Resolution (Lines 134-151)

### Function: `resolve_outdir()` (Lines 134-151)

```python
def resolve_outdir(infile: Path, out_dir_arg: Optional[str]) -> Path:
    """
    Resolve output directory.
    Priority:
      1) --out-dir (explicit)
      2) If path contains a 'data' directory, use <...>/data/exports
      3) Otherwise fallback to <input_dir>/exports
    """
    if out_dir_arg:
        return Path(out_dir_arg)

    parts = infile.resolve().parts
    if "data" in parts:
        data_root = Path(*parts[: parts.index("data") + 1])  # .../data
        return data_root / "exports"

    # fallback if not in a project layout
    return infile.parent / "exports"
```

**What it does:**

This function determines where output files should be saved based on the input file path and command-line arguments.

**Priority order:**

1. **Explicit `--out-dir` argument** (highest priority)
   - If the user provides `--out-dir`, use that directory exactly as specified
   - Example: `--out-dir custom/path` → uses `custom/path`

2. **Automatic `data/exports` detection** ⚠️ **NEW BEHAVIOR**
   - If the input file path contains a `data` directory, automatically uses `<...>/data/exports`
   - This is the new default behavior for files in the standard project structure
   - Example: Input `data/processed/file.csv` → Output `data/exports/file_grid.csv`
   - Example: Input `/project/data/raw/file.csv` → Output `/project/data/exports/file_grid.csv`

3. **Fallback to `<input_dir>/exports`**
   - If the input file is not in a `data` directory, creates an `exports` subdirectory in the input file's directory
   - Example: Input `other/path/file.csv` → Output `other/path/exports/file_grid.csv`

**Why this change?**

Previously, output files were written to the same directory as the input file (e.g., `data/processed/`). The new behavior automatically separates processed data from exported results:
- **Processed data** stays in `data/processed/`
- **Exported results** go to `data/exports/`

This keeps the directory structure cleaner and makes it easier to find final output files.

**Examples:**

```python
# Input: data/processed/mag_data_anomaly.csv
# Output: data/exports/mag_data_anomaly_grid.csv

# Input: /home/user/project/data/raw/file.csv
# Output: /home/user/project/data/exports/file_grid.csv

# Input: other/path/file.csv (no 'data' in path)
# Output: other/path/exports/file_grid.csv

# Input: data/processed/file.csv with --out-dir custom/path
# Output: custom/path/file_grid.csv (explicit override)
```

---

## Section 4: Grid Axis Generation (Lines 111-131)

### Function: `make_grid_axes()` (Lines 111-131)

```python
def make_grid_axes(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    grid_step: Optional[float],
    grid_n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns gx, gy 1D arrays defining the target grid axes.
    - If grid_step is provided, uses np.arange with a small inclusive buffer.
    - Otherwise uses np.linspace with grid_n points per axis.
    """
    if grid_step is not None and grid_step > 0:
        # Include the max edge (with a tiny buffer so endpoint isn't missed due to float rounding)
        gx = np.arange(xmin, xmax + grid_step * 0.5, grid_step)
        gy = np.arange(ymin, ymax + grid_step * 0.5, grid_step)
    else:
        gx = np.linspace(xmin, xmax, grid_n)
        gy = np.linspace(ymin, ymax, grid_n)
    return gx, gy
```

**What it does:**

This function creates the grid axes (x and y coordinate arrays) that define where interpolation will occur.

**Parameters:**
- `xmin, xmax`: Minimum and maximum x-coordinates of data
- `ymin, ymax`: Minimum and maximum y-coordinates of data
- `grid_step`: Optional spacing between grid points (if provided)
- `grid_n`: Number of grid points per axis (if `grid_step` not provided)

**Returns:**
- `gx, gy`: Arrays of grid coordinates

**Two modes:**

1. **Fixed spacing mode** (if `grid_step` is provided):
   ```python
   gx = np.arange(xmin, xmax + grid_step * 0.5, grid_step)
   ```
   - Creates grid with specified spacing
   - Example: `xmin=0.0, xmax=1.0, grid_step=0.2` → `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`
   - The `+ grid_step * 0.5` ensures the last point is included

2. **Fixed count mode** (if `grid_step` is None):
   ```python
   gx = np.linspace(xmin, xmax, grid_n)
   ```
   - Creates grid with specified number of points
   - Example: `xmin=0.0, xmax=1.0, grid_n=5` → `[0.0, 0.25, 0.5, 0.75, 1.0]`
   - Points are evenly spaced

**Example:**

```python
# Data extent: x from 0.0 to 2.0, y from 0.0 to 1.0

# Mode 1: Fixed spacing
gx, gy = make_grid_axes(0.0, 2.0, 0.0, 1.0, grid_step=0.5, grid_n=200)
# gx = [0.0, 0.5, 1.0, 1.5, 2.0]
# gy = [0.0, 0.5, 1.0]

# Mode 2: Fixed count
gx, gy = make_grid_axes(0.0, 2.0, 0.0, 1.0, grid_step=None, grid_n=5)
# gx = [0.0, 0.5, 1.0, 1.5, 2.0]  (5 points)
# gy = [0.0, 0.25, 0.5, 0.75, 1.0]  (5 points)
```

---

## Section 5: IDW Interpolation Function (Lines 157-194)

### Function: `idw_grid()` (Lines 157-194)

```python
def idw_grid(
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    power: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Inverse Distance Weighting interpolation.
    Returns grid Z with shape (len(gy), len(gx)), where rows correspond to gy (y-axis).
    """
    Xg, Yg = np.meshgrid(gx, gy)  # (ny, nx)

    tx = Xg.ravel()
    ty = Yg.ravel()

    Z = np.empty(tx.size, dtype=float)

    # Note: d2 is squared distance; using power/2 makes it equivalent to 1/d^power.
    for i in range(tx.size):
        dx = tx[i] - x
        dy = ty[i] - y
        d2 = dx * dx + dy * dy

        # If target is (numerically) on top of a source point, copy that exact value.
        j0 = int(np.argmin(d2))
        if d2[j0] <= eps:
            Z[i] = v[j0]
            continue

        # weights: 1 / (d^power) but computed via squared distances:
        # d^power == (d^2)^(power/2) == d2^(power/2)
        w = 1.0 / (d2 ** (power / 2.0) + eps)
        Z[i] = float(np.sum(w * v) / np.sum(w))

    return Z.reshape(Yg.shape)
```

**What it does:**

This function performs **Inverse Distance Weighting (IDW)** interpolation. IDW assigns values to grid points based on nearby measurement points, with closer points having more influence.

**Parameters:**
- `x, y`: Arrays of source point coordinates (measurement locations)
- `v`: Array of values at source points (what we're interpolating)
- `gx, gy`: Arrays defining the grid axes (where to interpolate)
- `power`: IDW power parameter (default 2.0)
- `eps`: Small epsilon to prevent divide-by-zero (default 1e-12)

**Returns:**
- `Z`: 2D array of interpolated values, shape `(len(gy), len(gx))`

**How IDW works:**

1. **For each grid point:**
   - Calculate distance to all source points
   - Compute weights: `weight = 1 / (distance^power)`
   - Closer points get higher weights
   - Interpolated value = weighted average of source values

2. **Weight formula:**
   ```
   weight[i] = 1 / (distance[i]^power + eps)
   interpolated_value = sum(weight[i] * value[i]) / sum(weight[i])
   ```

3. **Power parameter:**
   - `power = 1.0`: Linear decay (influence decreases linearly with distance)
   - `power = 2.0`: Quadratic decay (default, smoother results)
   - `power = 3.0`: Cubic decay (very local influence)

**Step-by-step execution:**

1. **Create grid mesh** (Line 73):
   ```python
   Xg, Yg = np.meshgrid(gx, gy)
   ```
   - Creates 2D arrays of all grid point coordinates
   - `Xg[i,j]` = x-coordinate of grid point (i,j)
   - `Yg[i,j]` = y-coordinate of grid point (i,j)

2. **Flatten to 1D** (Lines 76-77):
   ```python
   tx = Xg.ravel()  # All x-coordinates of grid points
   ty = Yg.ravel()  # All y-coordinates of grid points
   ```
   - Converts 2D grid to 1D arrays for easier iteration

3. **For each grid point** (Line 82):
   - Calculate distances to all source points (Lines 83-85)
   - Check if grid point exactly matches a source point (Lines 88-91)
   - If exact match, use source value directly
   - Otherwise, compute IDW weights (Line 93)
   - Calculate weighted average (Line 94)

4. **Reshape back to 2D** (Line 96):
   - Converts 1D result back to 2D grid shape

**Summary**
   - Filling in the points that were not exactly measured
   - Allows for a heatmap to be created determined by what the points should be based off of the data in the rest of the grid 

**Example:**

```python
# Source points
x = np.array([0.0, 1.0, 2.0])  # x-coordinates
y = np.array([0.0, 0.0, 0.0])  # y-coordinates
v = np.array([10.0, 20.0, 30.0])  # values

# Grid
gx = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
gy = np.array([0.0])

# Interpolate
Z = idw_grid(x, y, v, gx, gy, power=2.0)

# Result: Z[0] = [10.0, 15.0, 20.0, 25.0, 30.0]
# At x=0.5 (halfway between 0.0 and 1.0), value is 15.0 (average)
```

**Visual representation:**

```
Source points:          Grid points:
● (0, 0) = 10          ●───●───●───●───●
● (1, 0) = 20          0.0 0.5 1.0 1.5 2.0
● (2, 0) = 30

Interpolated values:
10.0  15.0  20.0  25.0  30.0
```

---

---

## Section 5: Main Function (Lines 200-323)

### Function: `main()` (Lines 110-209)

The main function orchestrates the entire interpolation and visualization process.

#### Part 1: Argument Parsing and Input Validation (Lines 200-222)

```python
def main() -> int:
    args = parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        print(f"ERROR: input file not found: {infile}", file=sys.stderr)
        return 2

    try:
        df = pd.read_csv(infile)
    except Exception as e:
        print(f"ERROR: could not read CSV: {e}", file=sys.stderr)
        return 2

    # Required columns
    needed = ["x", "y", args.value_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(
            f"ERROR: missing required column(s): {missing}. Columns found: {list(df.columns)}",
            file=sys.stderr,
        )
        return 2
```

**What it does:**

1. **Parse arguments** (Line 111)
   - Gets all command-line arguments

2. **Check input file exists** (Lines 112-115)
   - Verifies the input CSV file exists
   - Returns exit code 2 if not found

3. **Read CSV** (Lines 117-121)
   - Attempts to read the CSV file
   - Handles errors gracefully (file locked, corrupted, etc.)
   - Returns exit code 2 on error

4. **Validate required columns** (Lines 123-126)
   - Checks that `x`, `y`, and the value column exist
   - Provides helpful error message listing available columns
   - Returns exit code 2 if missing

#### Part 2: Data Filtering and Cleaning (Lines 224-245)

```python
    # Optional drop flagged rows
    if args.drop_flag_any and "_flag_any" in df.columns:
        before = len(df)
        # Make sure it's boolean-ish, then invert mask to keep unflagged
        mask = ~df["_flag_any"].astype(bool)
        df = df.loc[mask].copy()
        print(f"Note: dropped {before - len(df)} rows where _flag_any == True.")

    # Coerce numeric columns; bad values -> NaN
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df[args.value_col] = pd.to_numeric(df[args.value_col], errors="coerce")

    # Drop rows missing x/y/value
    df = df.dropna(subset=["x", "y", args.value_col]).copy()
    if len(df) == 0:
        print("ERROR: no valid rows after dropping NaNs in x/y/value.", file=sys.stderr)
        return 2

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    v = df[args.value_col].to_numpy(dtype=float)
```

**What it does:**

1. **Optional flag filtering** (Lines 129-133)
   - If `--drop-flag-any` is set and `_flag_any` column exists:
     - Filters out rows where `_flag_any` is True
     - Reports how many rows were dropped
   - Uses `~` (NOT operator) to invert boolean mask

2. **Convert to numeric** (Lines 135-137)
   - Converts x, y, and value columns to numeric types
   - `errors="coerce"` converts invalid values to NaN

3. **Drop invalid rows** (Line 138)
   - Removes rows with NaN in x, y, or value column
   - Ensures all data is valid for interpolation

4. **Check for empty data** (Lines 139-141)
   - Verifies we still have data after cleaning
   - Returns exit code 2 if all rows were invalid

5. **Extract arrays** (Lines 143-145)
   - Converts DataFrame columns to NumPy arrays
   - These arrays are used for interpolation

**Example:**

```python
# Before filtering:
#   x    y    local_anomaly  _flag_any
# 1.0  2.0       0.5         False
# 1.5  2.5       1.2         True   <- will be dropped
# 2.0  3.0       0.8         False

# After filtering (if --drop-flag-any):
#   x    y    local_anomaly
# 1.0  2.0       0.5
# 2.0  3.0       0.8
```

#### Part 3: Grid Setup (Lines 247-259)

```python
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    # Output directory (exports by default)
    outdir = resolve_outdir(infile, args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    stem = infile.stem
    grid_csv = outdir / f"{stem}_grid.csv"
    heatmap_png = outdir / f"{stem}_heatmap.png"

    # Build grid axes
    gx, gy = make_grid_axes(xmin, xmax, ymin, ymax, args.grid_step, args.grid_n)
```

**What it does:**

1. **Find data extent** (Lines 247-248)
   - Calculates min/max x and y coordinates
   - Defines the bounding box for the grid

2. **Set output directory** ⚠️ **CHANGED** (Lines 250-252)
   - **NEW**: Uses `resolve_outdir()` function to determine output directory
   - **Default behavior**: If input is in `data/processed/`, output goes to `data/exports/`
   - **Explicit override**: If `--out-dir` is provided, uses that directory
   - Creates directory if it doesn't exist (including parent directories)

3. **Define output filenames** (Lines 254-256)
   - `grid_csv`: Grid data CSV file
   - `heatmap_png`: Heatmap image file
   - Uses input file's stem (filename without extension)

4. **Create grid axes** (Line 259)
   - Generates grid coordinate arrays using `make_grid_axes()`

**Example:**

```python
# Data extent: x from 0.0 to 2.0, y from 0.0 to 1.0
# Input file: data/processed/mag_data_anomaly.csv

# NEW DEFAULT OUTPUTS:
# - data/exports/mag_data_anomaly_grid.csv  ⚠️ Changed from data/processed/
# - data/exports/mag_data_anomaly_heatmap.png  ⚠️ Changed from data/processed/

# Grid: 200×200 points (if using default --grid-n)
```

#### Part 4: Interpolation (Lines 261-267)

```python
    print(
        f"Interpolating '{args.value_col}' onto grid: nx={len(gx)}, ny={len(gy)} "
        f"(IDW power={args.power})"
    )

    # Compute interpolated grid
    Z = idw_grid(x, y, v, gx, gy, power=args.power, eps=args.eps)
```

**What it does:**

1. **Print progress message** (Lines 261-264)
   - Shows what's being interpolated and grid size

2. **Perform interpolation** (Line 267)
   - Calls `idw_grid()` to interpolate values onto grid
   - Returns 2D array `Z` with interpolated values

**Example output:**
```
Interpolating 'local_anomaly' onto grid: nx=200, ny=200 (IDW power=2.0)
```

#### Part 5: Export Grid CSV (Lines 269-284)

```python
    # Export grid CSV as long-form table (x,y,value)
    # (This rebuilds Xg/Yg only for labeling the grid with coordinates.)
    Xg, Yg = np.meshgrid(gx, gy)
    out_df = pd.DataFrame(
        {
            "x": Xg.ravel(),
            "y": Yg.ravel(),
            args.value_col: Z.ravel(),
        }
    )

    try:
        out_df.to_csv(grid_csv, index=False)
    except Exception as e:
        print(f"ERROR: could not write grid CSV: {e}", file=sys.stderr)
        return 3
```

**What it does:**

1. **Create grid mesh** (Line 164)
   - Generates 2D coordinate arrays for all grid points

2. **Create DataFrame** (Lines 165-169)
   - Converts 2D grid to long-form table (one row per grid point)
   - Columns: `x`, `y`, and the value column name
   - Uses `.ravel()` to flatten 2D arrays to 1D

3. **Write CSV** (Lines 170-173)
   - Saves grid data to CSV file
   - Handles write errors (permissions, disk full, etc.)
   - Returns exit code 3 on error

**CSV format:**

```csv
x,y,local_anomaly
0.0,0.0,0.523
0.0,0.1,0.541
0.0,0.2,0.558
0.1,0.0,0.536
...
```

This format is easy to import into GIS software or other analysis tools.

#### Part 6: Generate Heatmap (Lines 286-316) ⚠️ **NEW: Optional with --no-plot**

```python
    # Heatmap plot (optional)
    if not args.no_plot:
        clip = float(args.clip_percentile)
        if 0 < clip < 100:
            lo = np.nanpercentile(Z, 100 - clip)
            hi = np.nanpercentile(Z, clip)
            vmin, vmax = float(lo), float(hi)
        else:
            vmin, vmax = None, None

        plt.figure()
        im = plt.imshow(
            Z,
            origin="lower",
            extent=[float(gx.min()), float(gx.max()), float(gy.min()), float(gy.max())],
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(im, label=args.value_col)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Heatmap (IDW) of {args.value_col}")
        plt.tight_layout()

        try:
            plt.savefig(heatmap_png, dpi=160)
            plt.close()
        except Exception as e:
            print(f"ERROR: could not save heatmap PNG: {e}", file=sys.stderr)
            return 3
```

**What it does:**

1. **Clip color scale** (Lines 177-183)
   - Calculates percentile-based color limits
   - Example: `clip=99.0` uses 1st and 99th percentiles
   - Prevents extreme outliers from dominating the color scale
   - Makes visualization more readable

2. **Create figure** (Line 185)
   - Initializes matplotlib figure

3. **Display heatmap** (Lines 186-193)
   - `plt.imshow()`: Displays 2D array as image
   - `origin="lower"`: Puts y=0 at bottom (standard plot orientation)
   - `extent`: Sets axis limits to match data coordinates
   - `aspect="equal"`: Makes x and y scales equal (no distortion)
   - `vmin, vmax`: Color scale limits (from percentile clipping)

4. **Add colorbar** (Line 194)
   - Adds color scale legend
   - Labeled with the value column name

5. **Add labels and title** (Lines 195-197)
   - Sets axis labels and plot title

6. **Adjust layout** (Line 198)
   - `tight_layout()` prevents label cutoff

7. **Save image** (Lines 200-205)
   - Saves PNG file at 160 DPI (good quality)
   - Closes figure to free memory
   - Handles save errors (permissions, disk full, etc.)
   - Returns exit code 3 on error

**Visual result:**

The heatmap shows interpolated values as colors:
- **Red/hot colors**: High values
- **Blue/cold colors**: Low values
- **Smooth gradients**: IDW creates smooth transitions between points

#### Part 7: Success Message (Lines 318-323)

```python
    print(f"Wrote grid CSV: {grid_csv}")
    if not args.no_plot:
        print(f"Wrote heatmap:  {heatmap_png}")
    else:
        print("Note: --no-plot set, so no PNG was generated.")
    return 0
```

**What it does:**

- Prints success messages with output file paths
- Conditionally prints heatmap message based on `--no-plot` flag
- Returns exit code 0 (success)

---

## Section 6: Script Entry Point (Lines 326-331)

```python
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
```

**What it does:**

1. **`if __name__ == "__main__":`**
   - Only runs when script is executed directly (not imported)

2. **Run main function** (Lines 213-214)
   - Calls `main()` and exits with its return code
   - `raise SystemExit()` properly propagates exit codes

3. **Handle Ctrl+C** (Lines 215-217)
   - Catches `KeyboardInterrupt` (user presses Ctrl+C)
   - Prints message and exits with code 130 (standard for interrupted programs)

---

## Key Concepts

### Inverse Distance Weighting (IDW)

**What it is:**
- A method to estimate values at unknown locations based on nearby known values
- Closer points have more influence than distant points
- Creates smooth, continuous surfaces from discrete points

**Formula:**
```
value = sum(weight[i] * value[i]) / sum(weight[i])
where weight[i] = 1 / (distance[i]^power)
```

**Power parameter:**
- **Low power (1.0)**: Smooth, gradual influence decay
- **Medium power (2.0)**: Balanced (default)
- **High power (3.0+)**: Very local influence, sharper transitions

**Advantages:**
- Simple and intuitive
- No external dependencies (no SciPy needed)
- Works well for scattered data

**Limitations:**
- Can be slow for large datasets (O(N × grid_points))
- May create "bullseye" artifacts around isolated points
- Doesn't extrapolate beyond data extent

### Grid vs Scattered Data

**Scattered data:**
- Points at irregular locations
- Example: Field measurements at various (x, y) positions
- Hard to visualize directly

**Grid data:**
- Points at regular, evenly-spaced locations
- Example: 200×200 grid covering the area
- Easy to visualize as heatmap
- Can be used in GIS software

**Interpolation converts scattered → grid**

### Color Scale Clipping

**Problem:**
- Extreme outliers can dominate the color scale
- Most data appears in one color
- Hard to see variations in normal data

**Solution:**
- Clip color scale to percentiles (e.g., 1st and 99th)
- Extreme values are still in the data, just not shown in color
- Makes visualization more informative

**Example:**
```
Data range: -100 to +100
99th percentile: +5
1st percentile: -3

Without clipping: Most points appear blue (near 0)
With clipping: Can see variations between -3 and +5
```

---

## Tips for Tuning

### Grid Resolution

**Fine grid (`--grid-step 0.01` or `--grid-n 500`):**
- Higher detail, smoother appearance
- Slower computation
- Larger output files
- Use for: Final presentations, detailed analysis

**Coarse grid (`--grid-step 0.20` or `--grid-n 50`):**
- Faster computation
- Smaller files
- Less detail
- Use for: Quick previews, large datasets

### IDW Power

**Low power (1.0-1.5):**
- Smoother, more gradual transitions
- Less sensitive to individual points
- Use for: Noisy data, general trends

**Medium power (2.0-2.5):**
- Balanced (default 2.0 is good starting point)
- Use for: Most cases

**High power (3.0+):**
- Very local influence
- Sharp transitions
- Use for: Precise point features, high-quality data

### Color Scale Clipping

**No clipping (`--clip-percentile 100`):**
- Shows full data range
- May hide variations if outliers exist

**Moderate clipping (`--clip-percentile 95`):**
- Good balance
- Shows most of the data clearly

**Aggressive clipping (`--clip-percentile 90`):**
- Emphasizes central data
- May hide important extremes

---

## Integration with Pipeline

This script is typically the **final step** in the Fluxspace Core pipeline:

```
1. mag_to_csv.py
   → data/raw/mag_data.csv

2. validate_and_diagnosticsV1.py
   → data/processed/mag_data_clean.csv

3. compute_local_anomaly_v2.py
   → data/processed/mag_data_anomaly.csv

4. interpolate_to_heatmapV1.py  ← YOU ARE HERE
   → data/exports/mag_data_anomaly_grid.csv  ⚠️ Now defaults to exports/
   → data/exports/mag_data_anomaly_heatmap.png  ⚠️ Now defaults to exports/
```

**Typical workflow:**

```bash
# Step 1: Collect data
python3 scripts/mag_to_csv.py

# Step 2: Validate and clean
python3 scripts/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv

# Step 3: Compute anomalies
python3 scripts/compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --radius 0.30

# Step 4: Create heatmap (--out-dir is now optional, defaults to data/exports)
python3 scripts/interpolate_to_heatmapV1.py \
    --in data/processed/mag_data_anomaly.csv \
    --value-col local_anomaly \
    --grid-step 0.05
    # Outputs automatically go to data/exports/ ⚠️ No need for --out-dir anymore!
```

---

## Exit Codes

The script uses specific exit codes to indicate different outcomes:

- **0**: Success - Grid and heatmap created successfully
- **2**: Input error - File not found, invalid CSV, missing columns, or no valid data
- **3**: Output error - Could not write grid CSV or heatmap PNG
- **130**: Interrupted - User pressed Ctrl+C

These codes are useful for automation and error handling in scripts that call this program.

---

## Common Issues and Solutions

### Issue: "ERROR: input file not found"
**Solution:** Check the file path. Use absolute paths or paths relative to where you run the script.

### Issue: "ERROR: missing required column"
**Solution:** Check that your CSV has `x`, `y`, and the value column (default: `local_anomaly`). Use `--value-col` to specify a different column.

### Issue: Heatmap looks all one color
**Solution:** Try adjusting `--clip-percentile` (lower value like 95) or check if your data has very little variation.

### Issue: Interpolation is very slow
**Solution:** Use a coarser grid (`--grid-step 0.10` instead of `0.01`) or fewer points (`--grid-n 100` instead of 200).

### Issue: "bullseye" patterns around points
**Solution:** Lower the IDW power (`--power 1.5` instead of 2.0) for smoother results.

---

## Summary

This script takes scattered measurement points and creates a smooth, continuous visualization by:

1. **Reading** scattered (x, y, value) data from CSV
2. **Creating** a regular grid covering the data extent
3. **Interpolating** values onto the grid using IDW
4. **Exporting** grid data as CSV (for GIS/analysis)
5. **Generating** a heatmap visualization as PNG

It's designed to be the final step in the Fluxspace Core pipeline, creating publication-ready visualizations from processed anomaly data.
