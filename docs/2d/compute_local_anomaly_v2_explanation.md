# Complete Explanation of `compute_local_anomaly_v2.py`

This document explains every part of the local anomaly computation script (version 2), step by step.

---

## Overview

This script detects **local anomalies** in magnetic field data. It's an enhanced version of `compute_local_anomaly_v1.py` with command-line interface, quality flag filtering, and optional plotting. Instead of comparing each point to the global average, it compares each point to its **local neighborhood** - nearby points within a certain radius. This helps identify small-scale variations that might be hidden by global trends.

**Key improvements over v1:**
- Command-line arguments for flexibility
- Respects quality flags from `validate_and_diagnosticsV1.py`
- Computes absolute and normalized anomaly values
- Optional visualization with scatter plots
- Better error handling and user feedback

**Typical usage:**
```bash
python3 compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --radius 0.30 --plot
```

---

## Section 1: Imports and Setup (Lines 1-39)

```python
#!/usr/bin/env python3
"""
compute_local_anomaly_v2.py

Pipeline-ready local anomaly computation.

Reads a CSV containing at least:
  - x, y
  - B_total   (or Bx, By, Bz to compute B_total)

Optionally respects quality flags from validate_and_diagnostics.py:
  - _flag_spike / _flag_any (will drop spikes by default)

Writes:
  - <input_stem>_anomaly.csv   (same folder by default)

Adds columns:
  - local_anomaly
  - local_anomaly_abs
  - local_anomaly_norm  (normalized by max absolute anomaly in the file)

Also optionally plots a quick scatter map.

Example:
  python3 compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --radius 0.30 --plot
"""

from __future__ import annotations

import argparse
import sys
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

**What it does:**

1. **`#!/usr/bin/env python3`** (Shebang)
   - Makes the script executable directly from the command line

2. **Module docstring** (Lines 2-26)
   - Documents the script's purpose and usage
   - Lists required and optional columns
   - Shows example command-line usage
   - Explains output files and added columns

3. **`from __future__ import annotations`**
   - Enables postponed evaluation of type annotations
   - Allows using types without quotes in older Python versions

4. **Standard library imports:**
   - `argparse`: Parses command-line arguments
   - `sys`: System-specific parameters (for error output with `sys.stderr`)
   - `math`: Mathematical operations (though not heavily used here)
   - `pathlib.Path`: Modern path handling
   - `typing`: Type hints for function signatures

5. **External library imports:**
   - `numpy`: Numerical operations and arrays
   - `pandas`: Data manipulation and CSV handling
   - `matplotlib.pyplot`: Plotting and visualization

---

## Section 2: Helper Functions

### Function 1: `compute_btotal_if_missing()` (Lines 41-52)

```python
def compute_btotal_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "B_total" in df.columns:
        df["B_total"] = pd.to_numeric(df["B_total"], errors="coerce")
        return df
    if all(c in df.columns for c in ["Bx", "By", "Bz"]):
        bx = pd.to_numeric(df["Bx"], errors="coerce")
        by = pd.to_numeric(df["By"], errors="coerce")
        bz = pd.to_numeric(df["Bz"], errors="coerce")
        df["B_total"] = np.sqrt(bx.to_numpy() ** 2 + by.to_numpy() ** 2 + bz.to_numpy() ** 2)
        return df
    raise ValueError("Missing B_total and cannot compute it (need Bx, By, Bz).")
```

**What it does:**

Ensures `B_total` exists in the DataFrame. If it's missing, computes it from `Bx`, `By`, `Bz` components using the 3D Pythagorean theorem.

**Line-by-line breakdown:**

1. **`def compute_btotal_if_missing(df: pd.DataFrame) -> pd.DataFrame:`**
   - Takes a DataFrame, returns DataFrame with B_total column

2. **`df = df.copy()`**
   - Creates copy to avoid modifying original DataFrame

3. **`if "B_total" in df.columns:`**
   - Checks if B_total already exists

4. **`df["B_total"] = pd.to_numeric(df["B_total"], errors="coerce")`**
   - Converts to numeric, invalid values become NaN
   - Returns DataFrame with validated B_total

5. **`if all(c in df.columns for c in ["Bx", "By", "Bz"]):`**
   - Checks if all three components exist

6. **`bx = pd.to_numeric(df["Bx"], errors="coerce")`**
   - Converts each component to numeric

7. **`df["B_total"] = np.sqrt(bx.to_numpy() ** 2 + by.to_numpy() ** 2 + bz.to_numpy() ** 2)`**
   - Computes: B_total = √(Bx² + By² + Bz²)
   - Uses NumPy for efficient vectorized calculation

8. **`raise ValueError(...)`**
   - Raises error if B_total is missing and can't be computed

**Example:**
- If B_total exists: Returns DataFrame with validated B_total
- If Bx=45, By=23, Bz=12: Computes B_total ≈ 51.94
- If components missing: Raises ValueError

---

### Function 2: `local_anomaly()` (Lines 55-81)

```python
def local_anomaly(coords: np.ndarray, B: np.ndarray, radius: float) -> np.ndarray:
    """
    For each point i:
      baseline_i = mean(B[j] for j within 'radius' of i, excluding i)
      anomaly_i  = B[i] - baseline_i

    If a point has no neighbors within radius, anomaly is set to 0.0.
    """
    N = coords.shape[0]
    out = np.zeros(N, dtype=float)

    # O(N^2) simple approach (fine for small grids like 9x9, 20x20, etc.)
    for i in range(N):
        dx = coords[:, 0] - coords[i, 0]
        dy = coords[:, 1] - coords[i, 1]
        dist = np.sqrt(dx * dx + dy * dy)

        # neighbors within radius, excluding self (dist > 0)
        mask = (dist <= radius) & (dist > 0)

        if np.any(mask):
            baseline = float(np.mean(B[mask]))
            out[i] = float(B[i] - baseline)
        else:
            out[i] = 0.0

    return out
```

**What it does:**

Computes local anomalies for each point by comparing it to its neighborhood. This is the **core algorithm** of the script.

**Line-by-line breakdown:**

1. **`def local_anomaly(coords: np.ndarray, B: np.ndarray, radius: float) -> np.ndarray:`**
   - **`coords`**: NumPy array of shape (N, 2) with x, y coordinates
   - **`B`**: NumPy array of length N with B_total values
   - **`radius`**: Distance threshold for finding neighbors
   - **Returns**: NumPy array of length N with anomaly values

2. **`N = coords.shape[0]`**
   - Gets number of points from first dimension of coordinates array
   - `shape[0]` = number of rows (points)

3. **`out = np.zeros(N, dtype=float)`**
   - Creates output array filled with zeros
   - Will store anomaly values

4. **`for i in range(N):`**
   - Loops through each point
   - **O(N²) complexity**: For each point, checks all other points
   - Fine for small grids (9×9=81, 20×20=400 points)

5. **`dx = coords[:, 0] - coords[i, 0]`**
   - **`coords[:, 0]`**: All x-coordinates (entire first column)
   - **`coords[i, 0]`**: X-coordinate of point i
   - **`dx`**: Array of x-differences: `[x0-xi, x1-xi, x2-xi, ..., xN-xi]`

6. **`dy = coords[:, 1] - coords[i, 1]`**
   - **`coords[:, 1]`**: All y-coordinates (entire second column)
   - **`coords[i, 1]`**: Y-coordinate of point i
   - **`dy`**: Array of y-differences: `[y0-yi, y1-yi, y2-yi, ..., yN-yi]`

7. **`dist = np.sqrt(dx * dx + dy * dy)`**
   - Computes distances using 2D Pythagorean theorem
   - **`dx * dx`**: Element-wise multiplication (squares each element)
   - **`dy * dy`**: Element-wise multiplication
   - **`np.sqrt(...)`**: Element-wise square root
   - **Result**: Array of distances from point i to all other points

8. **`mask = (dist <= radius) & (dist > 0)`**
   - Creates boolean mask for neighbors
   - **`dist <= radius`**: True for points within radius
   - **`dist > 0`**: True for points that are NOT point i (exclude self)
   - **`&`**: Element-wise AND (both conditions must be True)
   - **Result**: Boolean array marking which points are neighbors

9. **`if np.any(mask):`**
   - Checks if there are ANY neighbors
   - **`np.any(mask)`**: Returns True if any element in mask is True

10. **`baseline = float(np.mean(B[mask]))`**
    - **`B[mask]`**: Selects B values only for neighbors (boolean indexing)
    - **`np.mean(...)`**: Calculates mean of neighbor B values
    - **`float(...)`**: Converts to Python float
    - **Result**: Local baseline (average of neighbors)

11. **`out[i] = float(B[i] - baseline)`**
    - Computes anomaly: point's value minus local baseline
    - **Positive anomaly**: Point is higher than neighbors (hot spot)
    - **Negative anomaly**: Point is lower than neighbors (cold spot)

12. **`else:`** and **`out[i] = 0.0`**
    - If no neighbors found, set anomaly to 0.0
    - Prevents division by zero or undefined behavior

13. **`return out`**
    - Returns array of anomaly values

**Visual Example:**

```
Point locations and B values:
    45.0    45.5    46.0
    45.2  [46.5]   45.8    ← Point i with B=46.5
    45.1    45.3    45.4

If RADIUS = 0.3 includes the 8 surrounding points:

Step 1: Calculate distances from point i to all points
Step 2: Find neighbors (within radius, exclude self)
        Neighbors: 8 surrounding points
Step 3: Calculate baseline
        baseline = (45.0 + 45.5 + 46.0 + 45.2 + 45.8 + 45.1 + 45.3 + 45.4) / 8
                 = 45.4
Step 4: Compute anomaly
        anomaly[i] = 46.5 - 45.4 = +1.1  (positive anomaly - hot spot!)
```

**Distance Calculation Example:**

For point i at (1.2, 2.3) and point j at (1.5, 2.4):
```
dx = 1.5 - 1.2 = 0.3
dy = 2.4 - 2.3 = 0.1
dist = √(0.3² + 0.1²) = √(0.09 + 0.01) = √0.1 ≈ 0.316
```

If `radius = 0.3`:
- `dist <= radius`: 0.316 <= 0.3 → False (not a neighbor)
- Point j is just outside the radius

If `radius = 0.4`:
- `dist <= radius`: 0.316 <= 0.4 → True (is a neighbor)

**Algorithm Complexity:**

- **Time complexity: O(N²)**
  - For each of N points, check distance to all N points
  - Total operations: N × N = N²
- **Why acceptable?**
  - Small grids: 9×9 = 81 points → 6,561 operations (fast)
  - Medium grids: 20×20 = 400 points → 160,000 operations (still fast)
  - Large grids: 100×100 = 10,000 points → 100,000,000 operations (slower)
- **For very large datasets**, consider spatial indexing (KD-tree) for O(N log N)

---

## Section 3: Command-Line Arguments (Lines 84-94)

### Function: `parse_args()` (Lines 84-94)

```python
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute local magnetic anomaly and write *_anomaly.csv.")
    p.add_argument("--in", dest="infile", required=True, help="Input CSV path (e.g., data/processed/mag_data_clean.csv)")
    p.add_argument("--radius", type=float, default=0.30, help="Neighborhood radius in same units as x/y (default: 0.30)")
    p.add_argument("--out", default=None, help="Optional output CSV path. Default: <input_stem>_anomaly.csv next to input.")
    p.add_argument("--keep-spikes", action="store_true", help="If set, do not drop rows where _flag_spike is True.")
    p.add_argument("--drop-flag-any", action="store_true", help="If set, drop rows where _flag_any is True (stronger than spikes).")
    p.add_argument("--plot", action="store_true", help="If set, show a quick scatter plot colored by local_anomaly_norm.")
    p.add_argument("--no-show", action="store_true", help="If set with --plot, save plot PNG instead of displaying it.")
    p.add_argument("--plot-out", default=None, help="Output PNG path if using --plot --no-show. Default: <out_stem>_plot.png")
    return p.parse_args()
```

**What it does:**

Parses command-line arguments using argparse. Provides flexible configuration options.

**Arguments explained:**

1. **`--in`** (required)
   - Input CSV file path
   - Example: `--in data/processed/mag_data_clean.csv`

2. **`--radius`** (optional, default: 0.30)
   - Neighborhood radius in same units as x, y coordinates
   - Example: `--radius 0.20` (20 cm radius)
   - **Small radius** (0.1-0.2): Detects very local anomalies
   - **Medium radius** (0.3-0.5): Balanced detection
   - **Large radius** (0.8-1.0): Detects broader regional anomalies

3. **`--out`** (optional)
   - Custom output CSV path
   - Default: `<input_stem>_anomaly.csv` next to input file
   - Example: `--out results/anomalies.csv`

4. **`--keep-spikes`** (flag)
   - By default, drops rows where `_flag_spike == True`
   - Use this flag to keep spike-flagged rows
   - Example: `--keep-spikes`

5. **`--drop-flag-any`** (flag)
   - Drops rows where `_flag_any == True` (stronger filtering)
   - Includes spikes, outliers, and any other flags
   - Example: `--drop-flag-any`

6. **`--plot`** (flag)
   - Generates scatter plot visualization
   - Example: `--plot`

7. **`--no-show`** (flag)
   - When used with `--plot`, saves PNG instead of displaying
   - Example: `--plot --no-show`

8. **`--plot-out`** (optional)
   - Custom plot output path
   - Only used with `--plot --no-show`
   - Default: `<out_stem>_plot.png`

**Example usage:**

```bash
# Basic usage
python3 compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv

# With custom radius
python3 compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --radius 0.25

# With plot
python3 compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --plot

# Save plot to file
python3 compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --plot --no-show

# Keep spike-flagged data
python3 compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --keep-spikes

# Strong quality filtering
python3 compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --drop-flag-any
```

---

## Section 4: Main Function (Lines 97-200)

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

    # Basic schema checks
    for c in ["x", "y"]:
        if c not in df.columns:
            print(f"ERROR: missing required column '{c}'. Columns found: {list(df.columns)}", file=sys.stderr)
            return 2

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    try:
        df = compute_btotal_if_missing(df)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Drop rows missing core numeric values
    before = len(df)
    df = df.dropna(subset=["x", "y", "B_total"]).copy()
    if len(df) == 0:
        print("ERROR: no valid rows after dropping NaNs in x/y/B_total.", file=sys.stderr)
        return 2
    if len(df) != before:
        print(f"Note: dropped {before - len(df)} rows due to NaNs in x/y/B_total.")

    # Optional quality-flag filtering
    if args.drop_flag_any and "_flag_any" in df.columns:
        before = len(df)
        df = df.loc[~df["_flag_any"].astype(bool)].copy()
        print(f"Note: dropped {before - len(df)} rows where _flag_any == True.")
    elif (not args.keep_spikes) and "_flag_spike" in df.columns:
        before = len(df)
        df = df.loc[~df["_flag_spike"].astype(bool)].copy()
        print(f"Note: dropped {before - len(df)} rows where _flag_spike == True.")

    coords = df[["x", "y"]].to_numpy(dtype=float)
    B = df["B_total"].to_numpy(dtype=float)

    # Compute anomalies
    anomalies = local_anomaly(coords, B, radius=float(args.radius))
    df["local_anomaly"] = anomalies
    df["local_anomaly_abs"] = np.abs(anomalies)

    max_abs = float(np.max(np.abs(anomalies))) if len(anomalies) else 0.0
    if max_abs > 0:
        df["local_anomaly_norm"] = anomalies / max_abs
    else:
        df["local_anomaly_norm"] = 0.0

    # Write output
    if args.out is None:
        outpath = infile.with_name(infile.stem.replace("_clean", "") + "_anomaly.csv")
    else:
        outpath = Path(args.out)

    try:
        df.to_csv(outpath, index=False)
    except Exception as e:
        print(f"ERROR: could not write output CSV: {e}", file=sys.stderr)
        return 3

    print(f"Wrote anomaly CSV: {outpath}")

    # Plot (optional)
    if args.plot:
        plt.figure()
        sc = plt.scatter(
            df["x"], df["y"],
            c=df["local_anomaly_norm"],
            s=90,
            edgecolor="k"
        )
        plt.colorbar(sc, label="Normalized local anomaly")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Local Anomaly (radius={args.radius})")
        plt.gca().set_aspect("equal", "box")
        plt.tight_layout()

        if args.no_show:
            if args.plot_out is None:
                plot_out = outpath.with_suffix("").with_name(outpath.stem + "_plot.png")
            else:
                plot_out = Path(args.plot_out)
            plot_out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_out, dpi=160)
            plt.close()
            print(f"Wrote plot PNG: {plot_out}")
        else:
            plt.show()

    return 0
```

**What it does:**

Main entry point that orchestrates the anomaly computation pipeline. Handles file I/O, data validation, anomaly computation, and optional visualization.

**Line-by-line breakdown:**

**Initialization and File Loading (Lines 98-108):**

1. **`args = parse_args()`**
   - Parses command-line arguments

2. **`infile = Path(args.infile)`**
   - Converts input path to Path object

3. **`if not infile.exists():`**
   - Checks if input file exists

4. **`print(f"ERROR: input file not found: {infile}", file=sys.stderr)`** and **`return 2`**
   - Prints error to stderr and returns exit code 2

5. **`try:`** and **`df = pd.read_csv(infile)`**
   - Loads CSV into DataFrame with error handling

6. **`except Exception as e:`** and **`return 2`**
   - Catches CSV read errors and returns exit code 2

**Schema Validation (Lines 110-117):**

7. **`for c in ["x", "y"]:`**
   - Loops through required columns

8. **`if c not in df.columns:`**
   - Checks if required column exists

9. **`print(...)`** and **`return 2`**
   - Error message and exit if column missing

10. **`df["x"] = pd.to_numeric(df["x"], errors="coerce")`**
    - Converts coordinates to numeric
    - Invalid values become NaN

**B_total Handling (Lines 119-123):**

11. **`try:`** and **`df = compute_btotal_if_missing(df)`**
    - Ensures B_total exists (computes if needed)

12. **`except Exception as e:`** and **`return 2`**
    - Error if B_total can't be computed

**Data Cleaning (Lines 125-142):**

13. **`before = len(df)`**
    - Records initial row count

14. **`df = df.dropna(subset=["x", "y", "B_total"]).copy()`**
    - Removes rows with missing core values
    - **`subset=["x", "y", "B_total"]`**: Only checks these columns
    - Other columns can have NaN

15. **`if len(df) == 0:`**
    - Checks if all rows were dropped

16. **`return 2`**
    - Exit if no valid data

17. **`if len(df) != before:`**
    - If rows were dropped, print note

**Quality Flag Filtering (Lines 134-142):**

18. **`if args.drop_flag_any and "_flag_any" in df.columns:`**
    - If `--drop-flag-any` flag set and column exists

19. **`df = df.loc[~df["_flag_any"].astype(bool)].copy()`**
    - **`df["_flag_any"]`**: Gets flag column
    - **`.astype(bool)`**: Converts to boolean
    - **`~`**: Negates (True becomes False, False becomes True)
    - **`df.loc[~...]`**: Selects rows where flag is False
    - **Result**: Drops all flagged rows (spikes, outliers, etc.)

20. **`elif (not args.keep_spikes) and "_flag_spike" in df.columns:`**
    - Default behavior: drop spikes unless `--keep-spikes` set

21. **`df = df.loc[~df["_flag_spike"].astype(bool)].copy()`**
    - Drops only spike-flagged rows (less aggressive than `_flag_any`)

**Anomaly Computation (Lines 144-156):**

22. **`coords = df[["x", "y"]].to_numpy(dtype=float)`**
    - Extracts coordinates as NumPy array
    - Shape: (N, 2) where N is number of points

23. **`B = df["B_total"].to_numpy(dtype=float)`**
    - Extracts B_total as NumPy array
    - Shape: (N,) - 1D array

24. **`anomalies = local_anomaly(coords, B, radius=float(args.radius))`**
    - Calls core anomaly computation function
    - Returns array of anomaly values

25. **`df["local_anomaly"] = anomalies`**
    - Adds anomaly column (can be positive or negative)

26. **`df["local_anomaly_abs"] = np.abs(anomalies)`**
    - Adds absolute value column (magnitude only)
    - Useful for magnitude-based analysis

27. **`max_abs = float(np.max(np.abs(anomalies)))`**
    - Finds maximum absolute anomaly value
    - Used for normalization

28. **`if max_abs > 0:`**
    - Checks if there are any non-zero anomalies

29. **`df["local_anomaly_norm"] = anomalies / max_abs`**
    - Normalizes anomalies to range [-1, 1]
    - **Formula**: `normalized = anomaly / max_abs`
    - **Result**: 
      - Most positive anomaly → 1.0
      - Most negative anomaly → -1.0
      - Zero anomaly → 0.0

30. **`else:`** and **`df["local_anomaly_norm"] = 0.0`**
    - If all anomalies are zero, set normalized to 0.0

**Example:** Normalizing local magnetic anomalies

Assume the local anomaly calculation produces these values (gauss):

local_anomaly:
- `[-0.003, 0.001, 0.020, -0.002, 0.004]`

**Step 1: Find the maximum absolute anomaly**

Absolute values:
- `[0.003, 0.001, 0.020, 0.002, 0.004]`

`max_abs = 0.020 gauss`

**Step 2: Normalize anomalies**

local_anomaly_norm = local_anomaly / max_abs

**Result:**
- `[-0.15, 0.05, 1.00, -0.10, 0.20]`

**Interpretation:**
- The strongest positive anomaly becomes +1.0
- The strongest negative anomaly would be -1.0 (if present)
- All values are now constrained to the range [-1, +1]

**Why this is useful:**
- Heatmaps use a consistent color scale regardless of absolute field strength
- Different test runs can be compared visually
- Thresholds can be expressed in relative terms (e.g., |norm| > 0.6)

**Edge case:**
If all anomalies were zero:
- `local_anomaly = [0.0, 0.0, 0.0]`
then:
- `max_abs = 0`
and the script safely sets:
- `local_anomaly_norm = 0.0`
to avoid division by zero.

**Output File Handling (Lines 158-168):**

31. **`if args.out is None:`**
    - If no custom output path specified

32. **`outpath = infile.with_name(infile.stem.replace("_clean", "") + "_anomaly.csv")`**
    - **`infile.stem`**: Filename without extension (e.g., "mag_data_clean")
    - **`.replace("_clean", "")`**: Removes "_clean" if present (e.g., "mag_data")
    - **`+ "_anomaly.csv"`**: Adds suffix (e.g., "mag_data_anomaly.csv")
    - **`infile.with_name(...)`**: Creates path in same directory as input

33. **`df.to_csv(outpath, index=False)`**
    - Saves DataFrame to CSV
    - **`index=False`**: Don't save row numbers

34. **`print(f"Wrote anomaly CSV: {outpath}")`**
    - Success message

**Plotting (Lines 172-198):**

35. **`if args.plot:`**
    - If plotting requested

36. **`plt.figure()`**
    - Creates new figure

37. **`sc = plt.scatter(...)`**
    - Creates scatter plot
    - **`df["x"], df["y"]`**: Point locations
    - **`c=df["local_anomaly_norm"]`**: Color by normalized anomaly
    - **`s=90`**: Point size
    - **`edgecolor="k"`**: Black border around points

38. **`plt.colorbar(sc, label="Normalized local anomaly")`**
    - Adds color scale legend

39. **`plt.gca().set_aspect("equal", "box")`**
    - Sets equal aspect ratio (prevents distortion)

40. **`if args.no_show:`**
    - If saving plot instead of displaying

41. **`plt.savefig(plot_out, dpi=160)`**
    - Saves plot as PNG
    - **`dpi=160`**: Resolution (dots per inch)

42. **`plt.show()`**
    - Displays plot interactively

43. **`return 0`**
    - Success exit code

---

## Section 5: Script Entry Point (Lines 203-208)

```python
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
```

**What it does:**

Runs the main function when script is executed directly. Handles Ctrl+C gracefully.

**Line-by-line breakdown:**

1. **`if __name__ == "__main__":`**
   - Only runs if script is executed directly (not imported)

2. **`try:`** and **`raise SystemExit(main())`**
   - Calls main and propagates exit code
   - **Exit codes:**
     - `0`: Success
     - `2`: File/input error
     - `3`: Output write error

3. **`except KeyboardInterrupt:`**
   - Catches Ctrl+C

4. **`print("\nStopped.", file=sys.stderr)`** and **`raise SystemExit(130)`**
   - Prints message to stderr
   - Exits with code 130 (standard for SIGINT)

---

## Key Concepts

### Local vs Global Anomaly

**Global anomaly:**
- Compares each point to the overall average
- Example: If global mean = 50, point with B=52 has +2 global anomaly
- **Problem:** Misses local variations, affected by regional trends

**Local anomaly:**
- Compares each point to its nearby neighbors
- Example: If neighbors average 45, point with B=52 has +7 local anomaly
- **Advantage:** Detects small-scale features that global analysis misses

**Visual comparison:**

```
Data with regional trend:
Point A: B=50 (in high region)
Point B: B=45 (in low region)

Global anomaly:
- Global mean = 47.5
- Point A: +2.5 (seems normal)
- Point B: -2.5 (seems normal)

Local anomaly (radius=0.3):
- Point A neighbors: [49, 50, 51] → mean = 50
- Point A: 50 - 50 = 0 (normal for region)
- Point B neighbors: [44, 45, 46] → mean = 45
- Point B: 45 - 45 = 0 (normal for region)

Now add a local hot spot:
Point C: B=55 (in high region, but also local hot spot)
- Neighbors: [50, 51, 52] → mean = 51
- Local anomaly: 55 - 51 = +4 (detected!)
- Global anomaly: 55 - 47.5 = +7.5 (but doesn't distinguish local vs regional)
```

### Radius Parameter

**How radius affects results:**

**Small radius (0.1-0.2 m):**
- Few neighbors per point
- Detects very local, sharp anomalies
- More sensitive to noise
- Good for: Small objects, sharp boundaries

**Medium radius (0.3-0.5 m):**
- Balanced number of neighbors
- Detects moderate-scale anomalies
- Good balance of sensitivity and stability
- Good for: General anomaly detection (default: 0.3)

**Large radius (0.8-1.0 m):**
- Many neighbors per point
- Detects broader regional anomalies
- More stable, less sensitive to noise
- Good for: Regional trends, large features

**Visual example:**

```
Grid with spacing 0.2 m:
  A   B   C
  D  [E]  F    ← Point E at center
  G   H   I

Radius = 0.2 m:
- Neighbors: B, D, F, H (4 points, adjacent only)
- Very local comparison

Radius = 0.3 m:
- Neighbors: A, B, C, D, F, G, H, I (8 points, all surrounding)
- Standard neighborhood

Radius = 0.5 m:
- Neighbors: All 8 surrounding + points further out
- Broader comparison
```

### Quality Flag Filtering

**Flag columns from `validate_and_diagnosticsV1.py`:**

- **`_flag_spike`**: Sudden changes in B_total (sensor glitches, movement)
- **`_flag_outlier`**: Values far from distribution (robust z-score)
- **`_flag_any`**: Any flag is True (spike OR outlier)

**Filtering behavior:**

**Default (no flags):**
- Drops rows where `_flag_spike == True`
- Keeps outliers (they might be real anomalies!)

**`--keep-spikes`:**
- Keeps all spike-flagged rows
- Useful if spikes are real features, not errors

**`--drop-flag-any`:**
- Drops rows with ANY flag (spikes, outliers, etc.)
- Most aggressive filtering
- Use when you want only highest quality data

**Example:**

```python
# Input data
Row 1: x=0.0, y=0.0, B_total=52.0, _flag_spike=False, _flag_any=False  ✓ Kept
Row 2: x=0.2, y=0.0, B_total=52.5, _flag_spike=True,  _flag_any=True  ✗ Dropped (default)
Row 3: x=0.4, y=0.0, B_total=53.0, _flag_spike=False, _flag_outlier=True, _flag_any=True  ✓ Kept (default)
                                                                        ✗ Dropped (--drop-flag-any)
```

### Anomaly Value Types

**Three anomaly columns:**

1. **`local_anomaly`**: Raw anomaly value
   - Can be positive or negative
   - Units: Same as B_total (gauss)
   - Example: `+1.5` (hot spot), `-0.8` (cold spot)

2. **`local_anomaly_abs`**: Absolute value
   - Magnitude only (always positive)
   - Useful for: Finding strongest anomalies regardless of direction
   - Example: `1.5` (from +1.5 or -1.5)

3. **`local_anomaly_norm`**: Normalized value
   - Range: [-1, 1]
   - Most positive anomaly → 1.0
   - Most negative anomaly → -1.0
   - Zero → 0.0
   - Useful for: Visualization, comparing across datasets

**Normalization formula:**

```
max_abs = max(|anomaly_1|, |anomaly_2|, ..., |anomaly_N|)
normalized_i = anomaly_i / max_abs
```

**Example:**
- Anomalies: `[-2.0, -1.0, 0.0, +1.0, +3.0]`
- `max_abs = 3.0`
- Normalized: `[-0.667, -0.333, 0.0, +0.333, +1.0]`

---

## Visual Examples and Diagrams

### Example 1: Simple 3×3 Grid

**Input data:**
```
x    y    B_total
0.0  0.0  50.0
0.2  0.0  51.0
0.4  0.0  50.5
0.0  0.2  49.5
0.2  0.2  52.5  ← Hot spot
0.4  0.2  50.0
0.0  0.4  50.0
0.2  0.4  50.5
0.4  0.4  49.8
```

**With radius = 0.3 m:**

**Grid visualization:**
```
Y
↑
0.4 ┌─────┬─────┬─────┐
    │ 50.0│ 50.5│ 49.8│
0.2 ├─────┼─────┼─────┤
    │ 49.5│ 52.5│ 50.0│  ← Point (0.2, 0.2) with B=52.5
0.0 |─────|─────|─────|
    | 50.0| 51.0| 50.5|
    |_____|_____|_____|
    0.0   0.2   0.4    → X
```

**Anomaly calculation for point (0.2, 0.2) with B=52.5:**

1. **Find neighbors within radius:**
   - (0.0, 0.2): distance = 0.2 ✓
   - (0.2, 0.0): distance = 0.2 ✓
   - (0.4, 0.2): distance = 0.2 ✓
   - (0.2, 0.4): distance = 0.2 ✓
   - (0.0, 0.0): distance = 0.283 ✓
   - (0.4, 0.0): distance = 0.283 ✓
   - (0.0, 0.4): distance = 0.283 ✓
   - (0.4, 0.4): distance = 0.283 ✓
   - Total: 8 neighbors

2. **Calculate baseline:**
   ```
   baseline = (49.5 + 51.0 + 50.0 + 50.5 + 50.0 + 50.5 + 50.0 + 49.8) / 8
            = 50.1
   ```

3. **Compute anomaly:**
   ```
   anomaly = 52.5 - 50.1 = +2.4
   ```

**Result:**
- Point (0.2, 0.2) has strong positive anomaly (+2.4)
- This is a hot spot compared to its neighbors

### Example 2: Anomaly Map Visualization

**Input:** 9×9 grid with one hot spot

**B_total values (gauss):**
```
    51.0  51.1  51.2  51.3  51.4  51.5  51.6  51.7  51.8
    51.1  51.2  51.3  51.4  51.5  51.6  51.7  51.8  51.9
    51.2  51.3  51.4  51.5  51.6  51.7  51.8  51.9  52.0
    51.3  51.4  51.5  51.6 [54.0] 51.8  51.9  52.0  52.1  ← Hot spot
    51.4  51.5  51.6  51.7  51.8  51.9  52.0  52.1  52.2
    51.5  51.6  51.7  51.8  51.9  52.0  52.1  52.2  52.3
    51.6  51.7  51.8  51.9  52.0  52.1  52.2  52.3  52.4
    51.7  51.8  51.9  52.0  52.1  52.2  52.3  52.4  52.5
    51.8  51.9  52.0  52.1  52.2  52.3  52.4  52.5  52.6
```

**With radius = 0.3 m:**

**Anomaly values (local_anomaly):**
```
    0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    0.0   0.0   0.0   0.0  [+2.4] 0.0   0.0   0.0   0.0  ← Detected!
    0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
```

**Visualization (scatter plot colored by anomaly):**
- Most points: Blue/white (anomaly ≈ 0)
- Hot spot: Red (anomaly = +2.4, normalized = 1.0)

### Example 3: Distance Calculation Diagram

**Point locations:**
```
Y
↑
0.4 ┌─────┬─────┬─────┐
    │  A  │  B  │  C  │
0.2 ├─────┼─────┼─────┤
    │  D  │ [E] │  F  │  ← Point E at (0.2, 0.2)
0.0 |─────|─────|─────|
    |  G  |  H  |  I  |
    |_____|_____|_____|
   0.0   0.2   0.4    → X
```

**For point E at (0.2, 0.2) with radius = 0.3:**

**Distance calculations:**
- To A (0.0, 0.4): √[(0.0-0.2)² + (0.4-0.2)²] = √[0.04 + 0.04] = √0.08 ≈ 0.283 ✓
- To B (0.2, 0.4): √[(0.2-0.2)² + (0.4-0.2)²] = √[0 + 0.04] = 0.2 ✓
- To C (0.4, 0.4): √[(0.4-0.2)² + (0.4-0.2)²] = √[0.04 + 0.04] = √0.08 ≈ 0.283 ✓
- To D (0.0, 0.2): √[(0.0-0.2)² + (0.2-0.2)²] = √[0.04 + 0] = 0.2 ✓
- To E (0.2, 0.2): distance = 0.0 ✗ (excluded - self)
- To F (0.4, 0.2): √[(0.4-0.2)² + (0.2-0.2)²] = √[0.04 + 0] = 0.2 ✓
- To G (0.0, 0.0): √[(0.0-0.2)² + (0.0-0.2)²] = √[0.04 + 0.04] = √0.08 ≈ 0.283 ✓
- To H (0.2, 0.0): √[(0.2-0.2)² + (0.0-0.2)²] = √[0 + 0.04] = 0.2 ✓
- To I (0.4, 0.0): √[(0.4-0.2)² + (0.0-0.2)²] = √[0.04 + 0.04] = √0.08 ≈ 0.283 ✓

**Neighbors:** A, B, C, D, F, G, H, I (8 points, all within 0.3 m)

---

## Summary: The Complete Workflow

1. **Parse arguments** → Get input file, radius, options
2. **Load CSV** → Read data into DataFrame
3. **Validate schema** → Check for x, y columns
4. **Ensure B_total** → Compute if missing from Bx, By, Bz
5. **Clean data** → Drop rows with missing x, y, B_total
6. **Filter quality flags** → Optionally drop spike/outlier flagged rows
7. **Extract arrays** → Convert to NumPy arrays for computation
8. **Compute anomalies** → For each point, compare to neighbors
9. **Add columns** → local_anomaly, local_anomaly_abs, local_anomaly_norm
10. **Save output** → Write CSV with anomaly columns
11. **Optional plot** → Generate scatter plot visualization
12. **Return** → Exit with success code

---

## Tips for Usage

1. **Radius selection:**
   - Start with default (0.30 m)
   - Adjust based on your grid spacing
   - Rule of thumb: radius ≈ 1.5× grid spacing

2. **Quality filtering:**
   - Use default (drop spikes) for clean data
   - Use `--keep-spikes` if spikes might be real features
   - Use `--drop-flag-any` for highest quality only

3. **Visualization:**
   - Use `--plot` to quickly check results
   - Use `--plot --no-show` to save plots for reports

4. **Output interpretation:**
   - **Positive anomaly**: Hot spot (higher than neighbors)
   - **Negative anomaly**: Cold spot (lower than neighbors)
   - **Near zero**: Normal (similar to neighbors)
   - **Large absolute value**: Strong anomaly

---

## Integration with Other Scripts

**Typical workflow:**
```
mag_to_csv.py 
  → mag_data.csv 
  → validate_and_diagnosticsV1.py 
  → mag_data_clean.csv 
  → compute_local_anomaly_v2.py 
  → mag_data_anomaly.csv
  → interpolate_to_heatmapV1.py (anomaly visualization)
```

**Data flow:**
- `mag_to_csv.py`: Collects raw measurements
- `validate_and_diagnosticsV1.py`: Validates and flags issues
- `compute_local_anomaly_v2.py`: Computes local anomalies
- `interpolate_to_heatmapV1.py`: Creates anomaly heatmap visualization
- Output can be used in: QGIS, further analysis, visualization

**Note:** The pipeline also provides `interpolate_to_Btotal_heatmap.py` for visualizing absolute field strength (B_total). Use that script with `mag_data_clean.csv` when you want to see the field strength distribution, and use `interpolate_to_heatmapV1.py` with `mag_data_anomaly.csv` when you want to see local anomalies.

---

## Differences from v1

**v1 (simple script):**
- Hardcoded parameters
- No quality flag filtering
- Only computes local_anomaly
- Always creates plot
- Fixed output location

**v2 (pipeline-ready):**
- Command-line arguments
- Quality flag filtering
- Computes local_anomaly, local_anomaly_abs, local_anomaly_norm
- Optional plotting
- Flexible output paths
- Better error handling
- Exit codes for automation

**When to use which:**
- **v1**: Quick one-off analysis, simple use cases
- **v2**: Production pipelines, automation, flexible workflows

