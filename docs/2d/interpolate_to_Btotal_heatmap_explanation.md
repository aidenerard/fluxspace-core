# Complete Explanation of `interpolate_to_Btotal_heatmap.py`

This document explains every part of the B_total heatmap generation script, step by step.

---

## Overview

This script takes scattered measurement points (x, y, B_total) from a CSV file and interpolates them onto a regular grid using **IDW (Inverse Distance Weighting)**. It then exports both the grid data as CSV and a visual heatmap as PNG. This script is specifically designed for visualizing **B_total (total magnetic field strength)** and includes unit conversion support (gauss to microtesla).

**What it does:**
- Reads scattered (x, y, B_total) points from CSV
- Creates a regular grid covering the data extent
- Interpolates B_total values onto the grid using IDW
- Exports grid data as CSV (long-form: x, y, B_total)
- Generates a heatmap PNG visualization
- Supports unit conversion (gauss ↔ microtesla)

**Typical usage:**
```bash
# Basic usage (default: gauss units)
python3 pipelines/2d/interpolate_to_Btotal_heatmap.py --in data/processed/mag_data_clean.csv

# With microtesla units
python3 pipelines/2d/interpolate_to_Btotal_heatmap.py --in data/processed/mag_data_clean.csv --units uT

# Custom grid spacing
python3 pipelines/2d/interpolate_to_Btotal_heatmap.py --in data/processed/mag_data_clean.csv --grid-step 0.01 --units uT
```

**Outputs:**
- `<prefix>_grid.csv` - Regular grid with interpolated B_total values (x, y, B_total format)
- `<prefix>_heatmap.png` - Visual heatmap image showing B_total distribution

**Key difference from `interpolate_to_heatmapV1.py`:**
- **This script**: Designed specifically for **B_total visualization** (magnetic field strength)
- **V1 script**: Designed for **anomaly detection** visualization (uses `local_anomaly` column)
- Both use IDW interpolation, but serve different purposes in the pipeline

---

## Section 1: Imports and Setup (Lines 1-11)

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

**What it does:**

1. **`#!/usr/bin/env python3`** (Shebang)
   - Makes the script executable directly from the command line

2. **`from __future__ import annotations`**
   - Enables postponed evaluation of type annotations
   - Allows cleaner type hints without quotes

3. **Standard library imports:**
   - `argparse`: Parses command-line arguments
   - `pathlib.Path`: Modern path handling
   - `typing`: Type hints (`Tuple`)

4. **External library imports:**
   - `numpy`: Numerical operations and arrays
   - `pandas`: Data manipulation and CSV handling
   - `matplotlib.pyplot`: Plotting and visualization

---

## Section 2: Command-Line Arguments (Lines 13-38)

### Function: `parse_args()` (Lines 13-38)

```python
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interpolate B_total onto a grid and save a magnetic-detection heatmap."
    )
    p.add_argument("--in", dest="inp", required=True, help="Input CSV (typically data/processed/*_clean.csv)")
    p.add_argument("--value-col", default="B_total", help="Column to map (default: B_total)")
    p.add_argument("--grid-step", type=float, default=0.01, help="Grid spacing in meters (default: 0.01)")
    p.add_argument("--power", type=float, default=2.0, help="IDW power parameter (default: 2.0)")
    p.add_argument("--eps", type=float, default=1e-12, help="Small epsilon to avoid div-by-zero (default: 1e-12)")
    p.add_argument("--out-dir", default="", help="Output directory. If empty, uses the input file's directory.")
    p.add_argument("--out-prefix", default="mag_detection", help="Prefix for outputs (default: mag_detection)")
    p.add_argument("--units", choices=["gauss", "uT"], default="gauss", help="Display units. If uT, values are converted from gauss to microtesla (1 G = 100 uT).")
    return p.parse_args()
```

**What it does:**

This function defines all command-line arguments the script accepts:

1. **`--in`** (required)
   - Input CSV file path
   - Example: `--in data/processed/mag_data_clean.csv`

2. **`--value-col`** (default: `"B_total"`)
   - Which column to interpolate onto the grid
   - Defaults to `B_total` (the script's primary purpose)
   - Can be changed if your CSV uses a different column name

3. **`--grid-step`** (default: 0.01)
   - Grid spacing in meters
   - Example: `--grid-step 0.01` (1 cm spacing)
   - Example: `--grid-step 0.05` (5 cm spacing)

4. **`--power`** (default: 2.0)
   - IDW power parameter (controls how quickly influence decays with distance)
   - Higher power = closer points have more influence
   - Typical values: 1.0 to 3.0

5. **`--eps`** (default: 1e-12)
   - Small epsilon value to prevent divide-by-zero errors
   - Added to distances before division
   - Very small number (0.000000000001)

6. **`--out-dir`** (default: empty string)
   - Output directory for results
   - If empty, uses the input file's directory
   - Example: `--out-dir data/exports`

7. **`--out-prefix`** (default: "mag_detection")
   - Prefix for output filenames
   - Example: With prefix "mag_detection", outputs are:
     - `mag_detection_grid.csv`
     - `mag_detection_heatmap.png`

8. **`--units`** (default: "gauss", choices: ["gauss", "uT"])
   - Display units for the heatmap
   - **"gauss"**: Uses values as-is (assumes input is in gauss)
   - **"uT"**: Converts from gauss to microtesla (multiplies by 100)
   - Conversion: 1 gauss = 100 microtesla (µT)

**Example usage:**
```bash
# Basic usage (gauss units)
python3 pipelines/2d/interpolate_to_Btotal_heatmap.py --in data/processed/mag_data_clean.csv

# Microtesla units with custom grid spacing
python3 pipelines/2d/interpolate_to_Btotal_heatmap.py --in data/processed/mag_data_clean.csv --units uT --grid-step 0.05

# Custom output location and prefix
python3 pipelines/2d/interpolate_to_Btotal_heatmap.py --in data/processed/mag_data_clean.csv --out-dir data/exports --out-prefix Btotal_map
```

---

## Section 3: IDW Interpolation Function (Lines 41-82)

### Function: `idw_interpolate()` (Lines 41-82)

```python
def idw_interpolate(
    xs: np.ndarray,
    ys: np.ndarray,
    vs: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    power: float,
    eps: float,
) -> np.ndarray:
    """
    IDW interpolation onto a meshgrid defined by gx, gy (both 2D arrays).
    """
    # Flatten grid for vectorized compute
    gxf = gx.ravel()
    gyf = gy.ravel()

    # Distances from each grid point to each sample: shape (G, N)
    dx = gxf[:, None] - xs[None, :]
    dy = gyf[:, None] - ys[None, :]
    d2 = dx * dx + dy * dy

    # If any grid point exactly matches a sample point, take that sample value directly
    exact = d2 <= eps
    out = np.empty(gxf.shape[0], dtype=float)

    if np.any(exact):
        # For each grid point, if it matches one or more samples, pick the first match
        match_idx = np.argmax(exact, axis=1)  # index of first True
        has_match = np.any(exact, axis=1)
        out[has_match] = vs[match_idx[has_match]]

        # For non-matching points, do IDW
        idx_nomatch = np.where(~has_match)[0]
        if idx_nomatch.size > 0:
            d2_nm = d2[idx_nomatch, :]
            w = 1.0 / np.power(d2_nm + eps, power / 2.0)
            out[idx_nomatch] = (w @ vs) / np.sum(w, axis=1)
    else:
        w = 1.0 / np.power(d2 + eps, power / 2.0)
        out = (w @ vs) / np.sum(w, axis=1)

    return out.reshape(gx.shape)
```

**What it does:**

This function performs **Inverse Distance Weighting (IDW)** interpolation. IDW assigns values to grid points based on nearby measurement points, with closer points having more influence.

**Parameters:**
- `xs, ys`: Arrays of source point coordinates (measurement locations)
- `vs`: Array of values at source points (B_total values)
- `gx, gy`: 2D arrays defining the grid (from `np.meshgrid`)
- `power`: IDW power parameter (default 2.0)
- `eps`: Small epsilon to prevent divide-by-zero (default 1e-12)

**Returns:**
- `out`: 2D array of interpolated values, shape matching `gx.shape`

**How IDW works:**

1. **Flatten grid** (Lines 54-55):
   - Converts 2D grid arrays to 1D for vectorized computation
   - `gxf` = all x-coordinates of grid points (flattened)
   - `gyf` = all y-coordinates of grid points (flattened)

2. **Calculate distances** (Lines 57-60):
   - Computes squared distances from each grid point to each sample point
   - Uses broadcasting: `gxf[:, None] - xs[None, :]` creates a (G, N) matrix
   - `d2` = squared distances, shape (G, N) where G = grid points, N = samples

3. **Handle exact matches** (Lines 62-70):
   - If a grid point exactly matches a sample point (distance ≤ eps), use that sample value directly
   - Avoids division by zero and preserves exact measurements

4. **Compute IDW weights** (Lines 72-77 or 79-80):
   - For non-matching points, compute weights: `w = 1 / (distance^power + eps)`
   - Closer points get higher weights
   - Interpolated value = weighted average: `sum(weight[i] * value[i]) / sum(weight[i])`
   - Uses matrix multiplication (`@`) for efficiency

5. **Reshape to 2D** (Line 82):
   - Converts 1D result back to 2D grid shape

**Key differences from V1 script:**
- This version uses **vectorized computation** with matrix operations
- More efficient for large grids (uses `@` operator instead of loops)
- Handles exact matches explicitly to avoid numerical issues

**Example:**
```python
# Source points
xs = np.array([0.0, 1.0, 2.0])  # x-coordinates
ys = np.array([0.0, 0.0, 0.0])  # y-coordinates
vs = np.array([50.0, 52.0, 51.0])  # B_total values (gauss)

# Grid
gx, gy = np.meshgrid([0.0, 0.5, 1.0, 1.5, 2.0], [0.0])

# Interpolate
Z = idw_interpolate(xs, ys, vs, gx, gy, power=2.0, eps=1e-12)

# Result: Z[0] ≈ [50.0, 51.0, 52.0, 51.5, 51.0]
# At x=0.5 (halfway between 0.0 and 1.0), value is interpolated between 50.0 and 52.0
```

---

## Section 4: Main Function (Lines 85-165)

### Function: `main()` (Lines 85-165)

The main function orchestrates the entire interpolation and visualization process.

#### Part 1: Argument Parsing and Input Validation (Lines 86-100)

```python
def main() -> int:
    args = parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    out_dir = Path(args.out_dir) if args.out_dir else inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)

    # Required columns
    for col in ("x", "y", args.value_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {inp.name}. Columns: {list(df.columns)}")

    df = df.dropna(subset=["x", "y", args.value_col]).copy()
```

**What it does:**

1. **Parse arguments** (Line 87)
   - Gets all command-line arguments

2. **Check input file exists** (Lines 89-91)
   - Verifies the input CSV file exists
   - Raises `FileNotFoundError` if not found

3. **Set output directory** (Lines 93-94)
   - If `--out-dir` is provided, uses that directory
   - Otherwise, uses the input file's parent directory
   - Creates directory if it doesn't exist

4. **Read CSV** (Line 96)
   - Loads data into a pandas DataFrame

5. **Validate required columns** (Lines 98-101)
   - Checks that `x`, `y`, and the value column (default: `B_total`) exist
   - Provides helpful error message listing available columns
   - Raises `ValueError` if missing

6. **Drop invalid rows** (Line 103)
   - Removes rows with NaN in x, y, or value column
   - Ensures all data is valid for interpolation

#### Part 2: Extract Data and Convert Units (Lines 104-112)

```python
    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)
    vs = df[args.value_col].to_numpy(dtype=float)

    # Convert units if requested (your logger stores gauss)
    label_units = "gauss"
    if args.units == "uT":
        vs = vs * 100.0  # 1 gauss = 100 microtesla
        label_units = "µT"
```

**What it does:**

1. **Extract arrays** (Lines 104-106):
   - Converts DataFrame columns to NumPy arrays
   - These arrays are used for interpolation

2. **Unit conversion** (Lines 108-112):
   - **Default (gauss)**: Uses values as-is
   - **If `--units uT`**: Converts from gauss to microtesla
   - Conversion factor: 1 gauss = 100 microtesla
   - Updates `label_units` for display in plots

**Why unit conversion matters:**
- Sensor data is typically stored in gauss
- Scientific publications often use microtesla (µT)
- This script handles the conversion automatically

#### Part 3: Grid Setup (Lines 114-124)

```python
    # Grid bounds
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    step = float(args.grid_step)
    if step <= 0:
        raise ValueError("--grid-step must be > 0")

    xi = np.arange(xmin, xmax + step * 0.5, step)
    yi = np.arange(ymin, ymax + step * 0.5, step)
    gx, gy = np.meshgrid(xi, yi)
```

**What it does:**

1. **Find data extent** (Lines 115-116):
   - Calculates min/max x and y coordinates
   - Defines the bounding box for the grid

2. **Validate grid step** (Lines 118-120):
   - Ensures grid step is positive
   - Raises `ValueError` if invalid

3. **Create grid axes** (Lines 122-124):
   - Creates evenly-spaced grid points using `np.arange`
   - The `+ step * 0.5` ensures the last point is included
   - `np.meshgrid` creates 2D coordinate arrays

**Example:**
```python
# Data extent: x from 0.0 to 2.0, y from 0.0 to 1.0
# Grid step: 0.5

# xi = [0.0, 0.5, 1.0, 1.5, 2.0]
# yi = [0.0, 0.5, 1.0]
# gx, gy = 2D arrays covering all combinations
```

#### Part 4: Interpolation (Lines 126-131)

```python
    print(
        f"Interpolating '{args.value_col}' onto grid: nx={len(xi)}, ny={len(yi)} "
        f"(IDW power={args.power}, step={step})..."
    )

    grid = idw_interpolate(xs, ys, vs, gx, gy, power=float(args.power), eps=float(args.eps))
```

**What it does:**

1. **Print progress message** (Lines 126-129):
   - Shows what's being interpolated and grid size
   - Displays IDW power and grid step

2. **Perform interpolation** (Line 131):
   - Calls `idw_interpolate()` to interpolate values onto grid
   - Returns 2D array `grid` with interpolated values

**Example output:**
```
Interpolating 'B_total' onto grid: nx=201, ny=101 (IDW power=2.0, step=0.01)...
```

#### Part 5: Export Grid CSV (Lines 133-143)

```python
    # Save grid CSV
    grid_csv = out_dir / f"{args.out_prefix}_grid.csv"
    out_df = pd.DataFrame(
        {
            "x": gx.ravel(),
            "y": gy.ravel(),
            args.value_col: grid.ravel(),
            "units": label_units,
        }
    )
    out_df.to_csv(grid_csv, index=False)
```

**What it does:**

1. **Define output filename** (Line 134):
   - Uses `--out-prefix` (default: "mag_detection")
   - Example: `mag_detection_grid.csv`

2. **Create DataFrame** (Lines 135-141):
   - Converts 2D grid to long-form table (one row per grid point)
   - Columns: `x`, `y`, value column name, and `units`
   - Uses `.ravel()` to flatten 2D arrays to 1D

3. **Write CSV** (Line 143):
   - Saves grid data to CSV file
   - `index=False` prevents writing row numbers

**CSV format:**
```csv
x,y,B_total,units
0.0,0.0,50.234,gauss
0.0,0.01,50.241,gauss
0.0,0.02,50.248,gauss
...
```

This format is easy to import into GIS software or other analysis tools.

#### Part 6: Generate Heatmap (Lines 145-161)

```python
    # Save heatmap PNG
    heatmap_png = out_dir / f"{args.out_prefix}_heatmap.png"
    plt.figure()
    im = plt.imshow(
        grid,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="auto",
    )
    plt.title(f"Heatmap (IDW) of {args.value_col}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    cbar = plt.colorbar(im)
    cbar.set_label(f"{args.value_col} ({label_units})")
    plt.tight_layout()
    plt.savefig(heatmap_png, dpi=200)
    plt.close()
```

**What it does:**

1. **Define output filename** (Line 146):
   - Uses `--out-prefix` (default: "mag_detection")
   - Example: `mag_detection_heatmap.png`

2. **Create figure** (Line 147):
   - Initializes matplotlib figure

3. **Display heatmap** (Lines 148-153):
   - `plt.imshow()`: Displays 2D array as image
   - `origin="lower"`: Puts y=0 at bottom (standard plot orientation)
   - `extent`: Sets axis limits to match data coordinates
   - `aspect="auto"`: Allows automatic aspect ratio

4. **Add labels and title** (Lines 154-156):
   - Sets axis labels (x, y in meters)
   - Sets plot title

5. **Add colorbar** (Lines 157-158):
   - Adds color scale legend
   - Labeled with value column name and units (e.g., "B_total (gauss)" or "B_total (µT)")

6. **Adjust layout** (Line 159):
   - `tight_layout()` prevents label cutoff

7. **Save image** (Lines 160-161):
   - Saves PNG file at 200 DPI (high quality)
   - Closes figure to free memory

**Visual result:**

The heatmap shows interpolated B_total values as colors:
- **Red/hot colors**: High B_total (stronger magnetic field)
- **Blue/cold colors**: Low B_total (weaker magnetic field)
- **Smooth gradients**: IDW creates smooth transitions between points

#### Part 7: Success Message (Lines 163-165)

```python
    print(f"Wrote grid CSV:  {grid_csv}")
    print(f"Wrote heatmap:   {heatmap_png}")
    return 0
```

**What it does:**

- Prints success messages with output file paths
- Returns exit code 0 (success)

---

## Section 5: Script Entry Point (Lines 168-169)

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

**What it does:**

1. **`if __name__ == "__main__":`**
   - Only runs when script is executed directly (not imported)

2. **Run main function**:
   - Calls `main()` and exits with its return code
   - `raise SystemExit()` properly propagates exit codes

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
- Vectorized implementation is efficient

**Limitations:**
- May create "bullseye" artifacts around isolated points
- Doesn't extrapolate beyond data extent
- Can be slow for very large datasets

### Unit Conversion: Gauss to Microtesla

**Why it matters:**
- Sensor data is typically stored in **gauss** (G)
- Scientific publications often use **microtesla** (µT)
- Conversion: **1 gauss = 100 microtesla**

**Example:**
- Input: 50.0 gauss
- With `--units uT`: 5000 µT (50.0 × 100)

**When to use each:**
- **Gauss**: Raw sensor data, internal processing
- **Microtesla**: Scientific publications, comparisons with literature

### Grid vs Scattered Data

**Scattered data:**
- Points at irregular locations
- Example: Field measurements at various (x, y) positions
- Hard to visualize directly

**Grid data:**
- Points at regular, evenly-spaced locations
- Example: 0.01 m spacing covering the area
- Easy to visualize as heatmap
- Can be used in GIS software

**Interpolation converts scattered → grid**

---

## Comparison: B_total Heatmap vs Anomaly Heatmap

The Fluxspace Core pipeline has **two different heatmap scripts** for different purposes:

### `interpolate_to_Btotal_heatmap.py` (This script)
**Purpose:** Visualize **magnetic field strength (B_total)**

**Use when:**
- You want to see the **absolute magnetic field strength** across the area
- You're looking for **strong/weak field regions**
- You need **unit conversion** (gauss ↔ microtesla)
- You're doing **magnetic detection** or **field mapping**

**Input:**
- `data/processed/mag_data_clean.csv` (or any CSV with x, y, B_total)

**Output:**
- Shows B_total values directly (e.g., 48-52 gauss)

**Example:**
```bash
python3 pipelines/2d/interpolate_to_Btotal_heatmap.py --in data/processed/mag_data_clean.csv --units uT
```

### `interpolate_to_heatmapV1.py`
**Purpose:** Visualize **anomaly detection results**

**Use when:**
- You want to see **local anomalies** (deviations from neighborhood)
- You're looking for **magnetic anomalies** (hot spots, cold spots)
- You've already run `compute_local_anomaly_v2.py`
- You need **percentile clipping** for better visualization

**Input:**
- `data/processed/mag_data_anomaly.csv` (with `local_anomaly` column)

**Output:**
- Shows anomaly values (e.g., -2 to +2 gauss deviation from local mean)

**Example:**
```bash
python3 pipelines/2d/interpolate_to_heatmapV1.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly
```

**Typical workflow:**
```
1. mag_to_csv.py → raw data
2. validate_and_diagnosticsV1.py → clean data
3. compute_local_anomaly_v2.py → anomaly data
4a. interpolate_to_Btotal_heatmap.py → B_total visualization (field strength)
4b. interpolate_to_heatmapV1.py → anomaly visualization (anomaly detection)
```

Both scripts use IDW interpolation, but serve different analysis purposes.

---

## Tips for Tuning

### Grid Resolution

**Fine grid (`--grid-step 0.01`):**
- Higher detail, smoother appearance
- Slower computation
- Larger output files
- Use for: Final presentations, detailed analysis

**Coarse grid (`--grid-step 0.05` or `0.10`):**
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

### Unit Selection

**Gauss (`--units gauss`):**
- Matches raw sensor data
- Good for internal processing
- Typical range: 45-55 gauss (Earth's field)

**Microtesla (`--units uT`):**
- Standard scientific unit
- Good for publications
- Typical range: 4500-5500 µT (Earth's field)

---

## Integration with Pipeline

This script is typically used **after validation** in the Fluxspace Core pipeline:

```
1. mag_to_csv.py
   → data/raw/mag_data.csv

2. validate_and_diagnosticsV1.py
   → data/processed/mag_data_clean.csv

3. interpolate_to_Btotal_heatmap.py  ← YOU ARE HERE
   → data/processed/mag_detection_grid.csv
   → data/processed/mag_detection_heatmap.png
```

**Typical workflow:**

```bash
# Step 1: Collect data
python3 pipelines/2d/mag_to_csv.py

# Step 2: Validate and clean
python3 pipelines/2d/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv

# Step 3: Create B_total heatmap
python3 pipelines/2d/interpolate_to_Btotal_heatmap.py \
    --in data/processed/mag_data_clean.csv \
    --units uT \
    --grid-step 0.01
```

**Alternative workflow (with anomaly detection):**

```bash
# Steps 1-2: Same as above

# Step 3: Compute anomalies
python3 pipelines/2d/compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --radius 0.30

# Step 4a: B_total visualization (field strength)
python3 pipelines/2d/interpolate_to_Btotal_heatmap.py --in data/processed/mag_data_clean.csv --units uT

# Step 4b: Anomaly visualization (anomaly detection)
python3 pipelines/2d/interpolate_to_heatmapV1.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly
```

---

## Exit Codes

The script uses specific exit codes to indicate different outcomes:

- **0**: Success - Grid and heatmap created successfully
- **Non-zero**: Error - File not found, invalid CSV, missing columns, or other errors

These codes are useful for automation and error handling in scripts that call this program.

---

## Common Issues and Solutions

### Issue: "Input file not found"
**Solution:** Check the file path. Use absolute paths or paths relative to where you run the script.

### Issue: "Missing required column"
**Solution:** Check that your CSV has `x`, `y`, and `B_total` columns. The script expects `B_total` by default, but you can use `--value-col` to specify a different column.

### Issue: Heatmap looks all one color
**Solution:** Check if your data has very little variation. Try adjusting `--grid-step` for finer resolution, or check your data range.

### Issue: Interpolation is very slow
**Solution:** Use a coarser grid (`--grid-step 0.05` instead of `0.01`) or reduce the data extent.

### Issue: "bullseye" patterns around points
**Solution:** Lower the IDW power (`--power 1.5` instead of 2.0) for smoother results.

---

## Summary

This script takes scattered B_total measurement points and creates a smooth, continuous visualization by:

1. **Reading** scattered (x, y, B_total) data from CSV
2. **Converting units** if requested (gauss ↔ microtesla)
3. **Creating** a regular grid covering the data extent
4. **Interpolating** values onto the grid using IDW
5. **Exporting** grid data as CSV (for GIS/analysis)
6. **Generating** a heatmap visualization as PNG

It's designed specifically for **magnetic field strength visualization** and complements the anomaly detection workflow. Use this script when you want to see the absolute field strength, and use `interpolate_to_heatmapV1.py` when you want to see local anomalies.
