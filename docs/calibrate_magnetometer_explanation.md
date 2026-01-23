# Complete Explanation of `calibrate_magnetometerV1.py`

This document explains every part of the magnetometer calibration script, step by step.

---

## Overview

This script fits calibration parameters to correct raw magnetometer measurements for sensor biases and distortions. Magnetometers in real-world applications (especially when mounted on drones or near electronic equipment) suffer from two types of errors:

1. **Hard-iron offset**: A constant bias added to all measurements (caused by permanent magnets, ferrous materials, or DC magnetic fields)
2. **Soft-iron distortion**: A matrix distortion that warps the measurement space (caused by materials that concentrate or redirect magnetic field lines)

The script takes a CSV file containing raw magnetometer measurements (Bx, By, Bz) collected while rotating the sensor through many orientations, fits calibration parameters, and outputs a calibration JSON file that can be used to correct future measurements.

**What it does:**
- Loads raw magnetometer data from CSV (Bx, By, Bz in microtesla)
- Fits hard-iron offset and soft-iron correction matrix using one of two methods:
  - **minmax**: Simple, robust method (per-axis offset and scaling)
  - **ellipsoid**: Advanced method (full 3x3 matrix correction)
- Optionally rescales calibrated magnitudes to match expected local Earth field strength
- Outputs calibration JSON file with all parameters
- Optionally outputs calibrated CSV with corrected measurements
- Can generate visualization plots showing before/after calibration

**Typical usage:**
```bash
# Basic calibration fit
python3 scripts/calibrate_magnetometerV1.py --in data/raw/mag_cal.csv --method ellipsoid --earth-field-ut 52

# Fit calibration + write calibrated CSV + save plots
python3 scripts/calibrate_magnetometerV1.py --in data/raw/mag_cal.csv --method minmax --earth-field-ut 50 \
    --write-calibrated --plot --no-show
```

**Inputs:**
- CSV file with raw magnetometer measurements (must contain Bx, By, Bz columns in microtesla)
- Calibration data should be collected by rotating the sensor through as many orientations as possible
- At least 200 samples recommended (default minimum)

**Outputs:**
- `<input_stem>_calibration.json` - Calibration parameters (offset, matrix, gain, metadata)
- `<input_stem>_calibrated.csv` (optional, if `--write-calibrated` is used) - Original data + calibrated columns
- `<input_stem>_calibration_plot.png` (optional, if `--plot --no-show` is used) - Visualization plots

**Context in the Fluxspace Project:**

Magnetometer calibration is a critical preprocessing step for accurate magnetic field measurements. In the Fluxspace Core pipeline:

- **Before calibration**: Raw sensor data may contain systematic biases from the sensor itself, mounting hardware, or nearby electronics (ESCs, motors, batteries, flight controllers)
- **After calibration**: Corrected measurements better reflect the true magnetic field, which is essential for:
  - Accurate anomaly detection (subtle local variations can be obscured by calibration errors)
  - Consistent measurements across different sensor orientations
  - Reliable comparisons between different measurement sessions

Calibration is typically performed once per sensor installation (or when hardware changes), and the resulting JSON file can be applied to correct measurements in subsequent data collection runs. The calibration data should be collected with the sensor in its final mounting configuration (on the drone frame, with all electronics powered on as they would be during normal operation) to capture the actual interference environment.

---

## Section 1: Imports and Dependencies (Lines 42-56)

```python
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

**What it does:**

1. **`from __future__ import annotations`**
   - Enables postponed evaluation of type annotations
   - Allows using types without quotes in function signatures
   - Improves forward compatibility

2. **Standard library imports:**
   - `argparse`: Parses command-line arguments (--in, --method, --earth-field-ut, etc.)
   - `json`: Reads and writes calibration JSON files
   - `math`: Mathematical operations (though NumPy is used for most math)
   - `sys`: System-specific parameters (for error output via `sys.stderr`)
   - `dataclasses.dataclass`: Creates structured data classes (used for `Calibration` class)
   - `datetime`, `timezone`: Generates UTC timestamps for calibration metadata
   - `pathlib.Path`: Modern, cross-platform path handling (better than string paths)
   - `typing`: Type hints (Optional, Tuple) for function signatures

3. **External library imports:**
   - `numpy`: Numerical operations, arrays, linear algebra (matrix operations, SVD)
   - `pandas`: CSV reading/writing, data manipulation (DataFrame operations)
   - `matplotlib.pyplot`: Plotting and visualization (scatter plots, histograms)

---

## Section 2: Platform Configuration (Lines 60-72)

```python
DEFAULT_PLATFORM_PARTS = {
    "frame": "ZMR250 / Readytosky 250 mm carbon frame",
    "motors": "Emax ECO II motors",
    "escs": "JHEMCU AM32A60 60A 4-in-1 ESC (3–6S) with current sensor",
    "propellers": "Gemfan 5 inch props",
    "battery": "Zeee 3S LiPo 2200mAh",
    "flight_controller": "Pixhawk 6c",
    "magnetometer": "QMC5883P",
    "raspberry_pi": "Vilros Raspberry Pi 4 4GB starter kit",
    "rx_tx": "Flysky FS-iA6B receiver",
    "gps": "u-blox NEO-M8N",
    "pdb": "QWinOut power distribution board",
}
```

**What it does:**

Stores a dictionary describing the hardware platform configuration. This metadata is included in the calibration JSON file as context, since different hardware components (especially motors, ESCs, batteries, and flight controllers) are common sources of magnetic interference.

**Line-by-line breakdown:**

This is a single dictionary assignment containing key-value pairs describing each hardware component. The values are stored in the calibration JSON's metadata so you can track which hardware configuration was used when the calibration was performed. This is helpful for:
- Debugging calibration issues
- Understanding why recalibration might be needed after hardware changes
- Documenting the measurement setup

**Example:**
When calibration JSON is written, this dictionary appears in the `notes.platform_parts` field, allowing you to later identify that a calibration was performed with "Emax ECO II motors" and "JHEMCU AM32A60 4-in-1 ESC", for example.

---

## Section 3: Helper Functions (Lines 80-111)

### Function 1: `_utc_now_iso()` (Lines 80-81)

```python
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
```

**What it does:**

Generates a UTC timestamp in ISO 8601 format (e.g., `"2024-01-15T14:30:00+00:00"`). Used to timestamp calibration JSON files.

**Line-by-line breakdown:**

1. **`def _utc_now_iso() -> str:`**
   - Function that takes no arguments and returns a string
   - `_` prefix indicates internal helper function

2. **`return datetime.now(timezone.utc).replace(microsecond=0).isoformat()`**
   - `datetime.now(timezone.utc)`: Gets current time in UTC timezone
   - `.replace(microsecond=0)`: Removes microseconds (rounds to seconds)
   - `.isoformat()`: Converts to ISO 8601 string format

**Example:**
```python
>>> _utc_now_iso()
'2024-01-15T14:30:45+00:00'
```

---

### Function 2: `_robust_z()` (Lines 84-94)

```python
def _robust_z(x: np.ndarray) -> np.ndarray:
    """
    Robust z-score using MAD (median absolute deviation).
    Returns z values; if MAD==0 -> zeros.
    """
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad <= 0:
        return np.zeros_like(x, dtype=float)
    return 0.6745 * (x - med) / mad
```

**What it does:**

Computes robust z-scores using the median absolute deviation (MAD) instead of standard deviation. This is more resistant to outliers than traditional z-scores. Used to detect and filter out outlier measurements before calibration fitting.

**Line-by-line breakdown:**

1. **`def _robust_z(x: np.ndarray) -> np.ndarray:`**
   - Takes a NumPy array and returns an array of z-scores
   - Z-score indicates how many "robust standard deviations" a value is from the median

2. **`x = np.asarray(x, dtype=float)`**
   - Converts input to NumPy array of floats (handles lists, pandas Series, etc.)

3. **`med = np.nanmedian(x)`**
   - Computes median, ignoring NaN values
   - Example: `[1, 2, 3, 4, 100]` → median = 3 (robust to the outlier 100)

4. **`mad = np.nanmedian(np.abs(x - med))`**
   - Median Absolute Deviation: median of absolute deviations from the median
   - Example: For `[1, 2, 3, 4, 100]` with median=3:
     - Deviations: `[-2, -1, 0, 1, 97]`
     - Absolute: `[2, 1, 0, 1, 97]`
     - MAD = median of `[2, 1, 0, 1, 97]` = 1

5. **`if not np.isfinite(mad) or mad <= 0:`**
   - Checks if MAD is invalid (NaN, infinite, or zero)
   - Returns zeros if data has no variation

6. **`return 0.6745 * (x - med) / mad`**
   - Converts MAD to equivalent standard deviation (factor 0.6745 makes MAD comparable to std for normal distributions)
   - Returns z-scores: values typically in range [-3, 3] for normal data, larger for outliers

**Example:**
```python
>>> data = np.array([50.0, 51.0, 49.0, 52.0, 200.0])  # 200 is an outlier
>>> z_scores = _robust_z(data)
>>> z_scores
array([-0.6745,  0.    , -1.349 ,  0.6745, 101.7055])  # Last value has huge z-score
```

---

### Function 3: `_ensure_cols()` (Lines 97-106)

```python
def _ensure_cols(df: pd.DataFrame, bx: str, by: str, bz: str) -> pd.DataFrame:
    missing = [c for c in (bx, by, bz) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Columns found: {list(df.columns)}")

    out = df.copy()
    out[bx] = pd.to_numeric(out[bx], errors="coerce")
    out[by] = pd.to_numeric(out[by], errors="coerce")
    out[bz] = pd.to_numeric(out[bz], errors="coerce")
    return out
```

**What it does:**

Validates that required columns (Bx, By, Bz) exist in the DataFrame and converts them to numeric type. Raises an error if columns are missing.

**Line-by-line breakdown:**

1. **`def _ensure_cols(df: pd.DataFrame, bx: str, by: str, bz: str) -> pd.DataFrame:`**
   - Takes a DataFrame and three column names, returns a DataFrame with validated/coerced columns

2. **`missing = [c for c in (bx, by, bz) if c not in df.columns]`**
   - List comprehension finds which required columns are missing
   - Example: If `bx="Bx"` and `df.columns = ["x", "y", "Bz"]`, then `missing = ["Bx", "By"]`

3. **`if missing:`**
   - Checks if any columns are missing (empty list is falsy)

4. **`raise ValueError(...)`**
   - Raises error with helpful message showing what's missing and what columns were found

5. **`out = df.copy()`**
   - Creates a copy to avoid modifying the original DataFrame

6. **`out[bx] = pd.to_numeric(out[bx], errors="coerce")`**
   - Converts column to numeric type
   - `errors="coerce"` converts non-numeric values to NaN instead of raising an error
   - Same for `by` and `bz` columns

7. **`return out`**
   - Returns DataFrame with validated numeric columns

**Example:**
```python
>>> df = pd.DataFrame({"Bx": ["1.5", "2.0", "3.0"], "By": [10, 20, 30], "Bz": [100, 200, 300]})
>>> df_validated = _ensure_cols(df, "Bx", "By", "Bz")
>>> df_validated["Bx"].dtype
dtype('float64')  # Converted from string to float
```

---

### Function 4: `_vector_magnitude()` (Lines 109-110)

```python
def _vector_magnitude(B: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(B * B, axis=1))
```

**What it does:**

Computes the magnitude (length) of each 3D vector in an array of vectors. Used to compute `|B| = sqrt(Bx² + By² + Bz²)` for each measurement.

**Line-by-line breakdown:**

1. **`def _vector_magnitude(B: np.ndarray) -> np.ndarray:`**
   - Takes array of shape `(N, 3)` where each row is a [Bx, By, Bz] vector
   - Returns array of shape `(N,)` with magnitude of each vector

2. **`return np.sqrt(np.sum(B * B, axis=1))`**
   - `B * B`: Element-wise squaring (Bx², By², Bz² for each row)
   - `np.sum(..., axis=1)`: Sums along axis 1 (across columns), giving Bx² + By² + Bz² for each row
   - `np.sqrt(...)`: Takes square root to get magnitude

**Example:**
```python
>>> B = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
>>> _vector_magnitude(B)
array([5.0, 5.0])  # sqrt(3²+4²+0²)=5, sqrt(0²+0²+5²)=5
```

---

## Section 4: Calibration Data Structure (Lines 113-120)

```python
@dataclass
class Calibration:
    method: str
    offset_ut: np.ndarray             # shape (3,)
    softiron_matrix: np.ndarray       # shape (3,3), applied as: B_cal = M @ (B_raw - offset)
    earth_field_ut: float
    gain: float                       # additional scalar gain applied after M (already folded into M if desired)
```

**What it does:**

Defines a data structure to store all calibration parameters. Uses `@dataclass` decorator to automatically generate `__init__`, `__repr__`, and other methods.

**Line-by-line breakdown:**

1. **`@dataclass`**
   - Decorator that automatically generates special methods for the class
   - Makes it easy to create instances: `Calibration(method="ellipsoid", offset_ut=offset, ...)`

2. **`method: str`**
   - Stores the calibration method used ("minmax" or "ellipsoid")

3. **`offset_ut: np.ndarray`**
   - Hard-iron offset vector, shape (3,), in microtesla
   - Example: `[10.5, -5.2, 3.1]` means Bx has +10.5 uT bias, By has -5.2 uT bias, etc.

4. **`softiron_matrix: np.ndarray`**
   - 3x3 correction matrix, applied as: `B_calibrated = matrix @ (B_raw - offset)`
   - Corrects for soft-iron distortions and per-axis scaling
   - Example: Identity matrix `[[1,0,0],[0,1,0],[0,0,1]]` means no correction

5. **`earth_field_ut: float`**
   - Expected Earth field magnitude in microtesla (used for scaling)
   - Example: 50.0 uT (typical for mid-latitudes)

6. **`gain: float`**
   - Additional scalar gain factor applied after matrix correction
   - Used to scale calibrated magnitudes to match `earth_field_ut`
   - Example: 1.02 means multiply all calibrated values by 1.02

**Example:**
```python
cal = Calibration(
    method="ellipsoid",
    offset_ut=np.array([10.0, -5.0, 2.0]),
    softiron_matrix=np.eye(3),
    earth_field_ut=50.0,
    gain=1.0
)
```

---

## Section 5: Calibration Fitting Methods (Lines 122-216)

### Method 1: `fit_minmax()` (Lines 122-139)

```python
def fit_minmax(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple, robust v1 fit:
      offset = (max + min)/2
      scale  = diag( avg_radius / axis_radius )
    This corrects axis biases and per-axis scaling, but NOT cross-axis coupling.
    """
    mins = np.nanmin(B, axis=0)
    maxs = np.nanmax(B, axis=0)
    offset = 0.5 * (maxs + mins)

    radii = 0.5 * (maxs - mins)
    if np.any(~np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("Invalid min/max radii; need variation on all axes.")

    avg_r = float(np.mean(radii))
    scale = np.diag(avg_r / radii)  # maps ellipsoid-ish ranges to similar radii
    return offset, scale
```

**What it does:**

Fits a simple calibration model using min/max statistics. This method:
- Estimates hard-iron offset as the center of the data range on each axis
- Estimates per-axis scaling to normalize the ranges
- Does NOT correct for cross-axis coupling (soft-iron effects that mix X, Y, Z components)

This method is robust and fast but less accurate than the ellipsoid method when significant soft-iron distortion exists.

**Line-by-line breakdown:**

1. **`def fit_minmax(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:`**
   - Takes array of shape `(N, 3)` with raw measurements
   - Returns tuple: `(offset, matrix)` where offset is shape (3,) and matrix is shape (3,3)

2. **`mins = np.nanmin(B, axis=0)`**
   - Finds minimum value along axis 0 (across rows), giving min for each column
   - Result shape (3,): `[min_Bx, min_By, min_Bz]`
   - Ignores NaN values
   - Example: If B has 100 rows, `mins` might be `[-45, -40, -35]` uT

3. **`maxs = np.nanmax(B, axis=0)`**
   - Finds maximum value along axis 0
   - Result shape (3,): `[max_Bx, max_By, max_Bz]`
   - Example: `[55, 50, 45]` uT

4. **`offset = 0.5 * (maxs + mins)`**
   - Computes offset as the midpoint (center) of the range on each axis
   - This assumes the true magnetic field (without interference) would be centered at zero after rotation
   - Example: `offset = 0.5 * ([55, 50, 45] + [-45, -40, -35]) = [5, 5, 5]` uT

5. **`radii = 0.5 * (maxs - mins)`**
   - Computes "radius" (half the range) on each axis
   - This is the spread of values on each axis
   - Example: `radii = 0.5 * ([55, 50, 45] - [-45, -40, -35]) = [50, 45, 40]` uT

6. **`if np.any(~np.isfinite(radii)) or np.any(radii <= 0):`**
   - Checks for invalid radii (NaN, infinite, or non-positive)
   - If any axis has no variation (min == max), calibration fails

7. **`raise ValueError(...)`**
   - Raises error if radii are invalid
   - Common cause: insufficient rotation data (sensor not moved enough)

8. **`avg_r = float(np.mean(radii))`**
   - Computes average radius across all three axes
   - Example: `avg_r = (50 + 45 + 40) / 3 = 45.0` uT

9. **`scale = np.diag(avg_r / radii)`**
   - Creates diagonal scaling matrix
   - `np.diag([a, b, c])` creates `[[a,0,0],[0,b,0],[0,0,c]]`
   - `avg_r / radii` gives scaling factors: `[45/50, 45/45, 45/40] = [0.9, 1.0, 1.125]`
   - Scales each axis so they all have similar "radius" (normalizes the ellipsoid to be more spherical)
   - Example result:
     ```
     [[0.9, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.125]]
     ```

10. **`return offset, scale`**
    - Returns the offset vector and scaling matrix

**Example:**
If raw measurements have ranges:
- Bx: -45 to +55 uT (center at 5, radius 50)
- By: -40 to +50 uT (center at 5, radius 45)
- Bz: -35 to +45 uT (center at 5, radius 40)

The minmax method produces:
- `offset = [5, 5, 5]` uT
- `scale = [[0.9, 0, 0], [0, 1.0, 0], [0, 0, 1.125]]`

After calibration: `B_cal = scale @ (B_raw - offset)`, the calibrated data should have more uniform spread across axes.

---

### Method 2: `fit_ellipsoid()` (Lines 142-216)

```python
def fit_ellipsoid(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ellipsoid fit (least squares) to estimate hard-iron offset + full 3x3 soft-iron correction.

    Fits quadratic form:
        x^T A x + b^T x + c = 0
    Derives center and shape matrix, then returns:
        offset (center) and M such that || M @ (x - offset) || ≈ 1.
    """
    x = B[:, 0]
    y = B[:, 1]
    z = B[:, 2]

    # Build design matrix for: [x^2, y^2, z^2, xy, xz, yz, x, y, z, 1]
    D = np.column_stack([
        x * x,
        y * y,
        z * z,
        x * y,
        x * z,
        y * z,
        x,
        y,
        z,
        np.ones_like(x),
    ])

    # Solve D @ v = 0 subject to ||v||=1.
    # Use SVD; solution is right singular vector corresponding to smallest singular value.
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    v = Vt[-1, :]

    # Unpack parameters
    # A is symmetric:
    # [a  d/2 e/2]
    # [d/2 b  f/2]
    # [e/2 f/2 c]
    a, b, c, d, e, f, g, h, i, j = v
    A = np.array([
        [a, d / 2.0, e / 2.0],
        [d / 2.0, b, f / 2.0],
        [e / 2.0, f / 2.0, c],
    ], dtype=float)
    bb = np.array([g, h, i], dtype=float)
    cc = float(j)

    if np.linalg.cond(A) > 1e12:
        raise ValueError("Ellipsoid fit ill-conditioned (A matrix). Try more diverse orientations or use --method minmax.")

    center = -0.5 * np.linalg.solve(A, bb)

    # Translate: x' = x - center.
    # Compute constant term for translated quadric:
    # k = cc + center^T A center + bb^T center
    k = cc + float(center.T @ A @ center) + float(bb.T @ center)

    if not np.isfinite(k) or abs(k) < 1e-12:
        raise ValueError("Ellipsoid fit failed (degenerate constant term).")

    # For an ellipsoid, we expect (x')^T A (x') = -k  (positive rhs)
    # Normalize shape matrix:
    Mshape = A / (-k)

    # Ensure positive definite
    evals, evecs = np.linalg.eigh(Mshape)
    if np.any(evals <= 0) or np.any(~np.isfinite(evals)):
        raise ValueError("Ellipsoid fit produced non-positive-definite shape matrix. Try --method minmax.")

    # We want transform T such that ||T x'||^2 = x'^T Mshape x' and thus ||T x'|| ≈ 1
    # If Mshape = R diag(l) R^T, then T = R diag(sqrt(l)) R^T
    T = (evecs @ np.diag(np.sqrt(evals)) @ evecs.T).astype(float)

    return center.astype(float), T
```

**What it does:**

Fits a full ellipsoid model to the data using least-squares. This method:
- Estimates both hard-iron offset (center of ellipsoid) and full 3x3 soft-iron correction matrix
- Can correct for cross-axis coupling (soft-iron effects)
- More accurate than minmax but requires more diverse data and can fail if data is insufficient

The algorithm fits a quadratic surface (ellipsoid) to the raw measurements, then derives a transformation matrix that maps the ellipsoid to a unit sphere.

**Line-by-line breakdown:**

1. **`def fit_ellipsoid(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:`**
   - Takes array of shape `(N, 3)` with raw measurements
   - Returns tuple: `(offset, matrix)` where offset is shape (3,) and matrix is shape (3,3)

2. **`x = B[:, 0]`, `y = B[:, 1]`, `z = B[:, 2]`**
   - Extracts X, Y, Z components as separate 1D arrays
   - Each has shape (N,)

3. **Design matrix construction (Lines 158-169):**
   - Builds a design matrix `D` where each row represents a measurement and columns are polynomial terms
   - The quadratic form `x^T A x + b^T x + c = 0` can be rewritten as a linear system
   - Columns: `[x², y², z², xy, xz, yz, x, y, z, 1]` (10 columns total)
   - `np.column_stack([...])` stacks arrays as columns
   - Example: If one measurement is `[10, 20, 30]`, the row is `[100, 400, 900, 200, 300, 600, 10, 20, 30, 1]`

4. **SVD solution (Lines 172-174):**
   - `np.linalg.svd(D, full_matrices=False)`: Performs Singular Value Decomposition
   - Returns `U, S, Vt` where `Vt` contains right singular vectors
   - `v = Vt[-1, :]`: Takes the last row (corresponding to smallest singular value)
   - This solves the homogeneous system `D @ v = 0` subject to `||v|| = 1`

5. **Parameter unpacking (Lines 181-188):**
   - Unpacks the 10-element vector `v` into parameters `a, b, c, d, e, f, g, h, i, j`
   - Constructs symmetric matrix `A` (3x3) and vector `bb` (3x1) and scalar `cc`
   - `A` represents the quadratic form coefficients
   - `bb` represents linear term coefficients
   - `cc` represents constant term

6. **Condition check (Lines 190-191):**
   - `np.linalg.cond(A)`: Computes condition number of matrix `A`
   - Condition number > 1e12 indicates the matrix is nearly singular (ill-conditioned)
   - This happens when data lacks diversity (e.g., all measurements along a line)
   - Raises error suggesting to use minmax method or collect more diverse data

7. **Center computation (Line 193):**
   - `center = -0.5 * np.linalg.solve(A, bb)`
   - Solves `A @ center = -0.5 * bb` to find the ellipsoid center
   - This is the hard-iron offset

8. **Constant term computation (Lines 198-199):**
   - Computes `k` after translating coordinate system to center
   - Used to normalize the shape matrix

9. **Shape matrix normalization (Lines 204-205):**
   - `Mshape = A / (-k)`: Normalizes the shape matrix
   - For a valid ellipsoid, `k` should be negative, so `-k` is positive

10. **Eigenvalue decomposition (Lines 207-210):**
    - `np.linalg.eigh(Mshape)`: Computes eigenvalues and eigenvectors
    - Eigenvalues should all be positive for a valid ellipsoid
    - Raises error if eigenvalues are non-positive (degenerate case)

11. **Transform matrix construction (Line 214):**
    - Constructs transformation matrix `T` that maps the ellipsoid to a unit sphere
    - Uses eigenvalue decomposition: `Mshape = R @ diag(λ) @ R^T`, so `T = R @ diag(√λ) @ R^T`
    - After transformation: `||T @ (x - center)|| ≈ 1` for all points

12. **Return (Line 216):**
    - Returns center (offset) and transformation matrix

**Example:**
If raw measurements form an ellipsoid (distorted sphere) due to soft-iron effects, the ellipsoid fit finds the center and the matrix that "straightens" the ellipsoid back into a sphere. The offset corrects hard-iron bias, and the matrix corrects soft-iron distortion.

---

## Section 6: Calibration Application and Scaling (Lines 219-231)

### Function: `apply_calibration()` (Lines 219-221)

```python
def apply_calibration(B: np.ndarray, cal: Calibration) -> np.ndarray:
    # B_cal = gain * (M @ (B - offset))
    return (cal.gain * (cal.softiron_matrix @ (B - cal.offset_ut).T)).T
```

**What it does:**

Applies calibration to raw measurements. The formula is: `B_calibrated = gain × (matrix @ (B_raw - offset))`

**Line-by-line breakdown:**

1. **`def apply_calibration(B: np.ndarray, cal: Calibration) -> np.ndarray:`**
   - Takes raw measurements (shape `(N, 3)`) and a `Calibration` object
   - Returns calibrated measurements (shape `(N, 3)`)

2. **`return (cal.gain * (cal.softiron_matrix @ (B - cal.offset_ut).T)).T`**
   - `(B - cal.offset_ut)`: Subtracts offset from each measurement (broadcasting)
   - `.T`: Transposes to shape `(3, N)` for matrix multiplication
   - `cal.softiron_matrix @ ...`: Matrix multiplication (3x3 matrix @ 3xN = 3xN result)
   - `cal.gain * ...`: Multiplies by scalar gain
   - `.T`: Transposes back to shape `(N, 3)`

**Example:**
```python
B_raw = np.array([[60.0, 55.0, 45.0]])  # One measurement
cal = Calibration(
    offset_ut=np.array([5.0, 5.0, 5.0]),
    softiron_matrix=np.eye(3),  # Identity matrix
    gain=1.0,
    ...
)
B_cal = apply_calibration(B_raw, cal)
# Result: [[55.0, 50.0, 40.0]]  # After subtracting offset
```

---

### Function: `compute_gain_to_match_earth()` (Lines 224-231)

```python
def compute_gain_to_match_earth(B_cal: np.ndarray, earth_field_ut: float) -> float:
    if earth_field_ut <= 0:
        return 1.0
    mag = _vector_magnitude(B_cal)
    med = float(np.nanmedian(mag))
    if not np.isfinite(med) or med <= 0:
        return 1.0
    return float(earth_field_ut / med)
```

**What it does:**

Computes a gain factor to scale calibrated magnitudes to match the expected Earth field magnitude. This ensures the calibrated sensor reads the correct absolute field strength (not just relative values).

**Line-by-line breakdown:**

1. **`def compute_gain_to_match_earth(B_cal: np.ndarray, earth_field_ut: float) -> float:`**
   - Takes calibrated measurements and target Earth field magnitude
   - Returns scalar gain factor

2. **`if earth_field_ut <= 0:`**
   - If Earth field scaling is disabled (0 or negative), return gain of 1.0 (no scaling)

3. **`mag = _vector_magnitude(B_cal)`**
   - Computes magnitude of each calibrated measurement
   - Result shape `(N,)`: `[|B₁|, |B₂|, ..., |Bₙ|]`

4. **`med = float(np.nanmedian(mag))`**
   - Computes median magnitude
   - Uses median (robust to outliers) instead of mean
   - Example: If magnitudes are `[48, 49, 50, 51, 52]` uT, median is `50.0` uT

5. **`if not np.isfinite(med) or med <= 0:`**
   - Checks if median is invalid (NaN, infinite, or non-positive)
   - Returns gain of 1.0 if invalid

6. **`return float(earth_field_ut / med)`**
   - Computes gain as target magnitude divided by current median
   - Example: If `earth_field_ut = 52.0` and `med = 50.0`, then `gain = 52.0 / 50.0 = 1.04`
   - After applying this gain, median magnitude should be approximately `earth_field_ut`

**Example:**
If calibrated measurements have median magnitude of 48 uT, but the local Earth field is 52 uT, the gain is `52/48 = 1.083`. Applying this gain scales all measurements so the median matches the expected Earth field.

---

## Section 7: JSON I/O Functions (Lines 234-255)

### Function: `write_calibration_json()` (Lines 234-245)

```python
def write_calibration_json(out_json: Path, cal: Calibration, meta: dict) -> None:
    payload = {
        "created_utc": _utc_now_iso(),
        "method": cal.method,
        "earth_field_ut": float(cal.earth_field_ut),
        "offset_ut": [float(x) for x in cal.offset_ut.tolist()],
        "softiron_matrix": [[float(x) for x in row] for row in cal.softiron_matrix.tolist()],
        "gain": float(cal.gain),
        "notes": meta,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
```

**What it does:**

Writes calibration parameters to a JSON file. The JSON can be loaded later to apply the calibration to new measurements.

**Line-by-line breakdown:**

1. **`def write_calibration_json(out_json: Path, cal: Calibration, meta: dict) -> None:`**
   - Takes output path, calibration object, and metadata dictionary
   - Writes JSON file (no return value)

2. **`payload = {...}`**
   - Creates dictionary with all calibration parameters
   - `"created_utc"`: Timestamp using `_utc_now_iso()`
   - `"method"`: Calibration method string ("minmax" or "ellipsoid")
   - `"earth_field_ut"`: Target Earth field magnitude (float)
   - `"offset_ut"`: Offset vector as list `[x, y, z]` (converted from NumPy array)
   - `"softiron_matrix"`: 3x3 matrix as nested list `[[...], [...], [...]]`
   - `"gain"`: Scalar gain factor
   - `"notes"`: Metadata dictionary (includes platform parts, interference analysis, etc.)

3. **`out_json.parent.mkdir(parents=True, exist_ok=True)`**
   - Creates parent directories if they don't exist
   - `parents=True`: Creates intermediate directories
   - `exist_ok=True`: Doesn't raise error if directory already exists

4. **`out_json.write_text(json.dumps(payload, indent=2) + "\n")`**
   - `json.dumps(payload, indent=2)`: Converts dictionary to JSON string with 2-space indentation
   - `+ "\n"`: Adds trailing newline
   - `write_text(...)`: Writes to file

**Example JSON output:**
```json
{
  "created_utc": "2024-01-15T14:30:45+00:00",
  "method": "ellipsoid",
  "earth_field_ut": 50.0,
  "offset_ut": [5.2, -3.1, 1.8],
  "softiron_matrix": [
    [0.98, 0.01, -0.02],
    [0.01, 1.02, 0.01],
    [-0.02, 0.01, 0.99]
  ],
  "gain": 1.04,
  "notes": {
    "platform_parts": {...}
  }
}
```

---

### Function: `load_calibration_json()` (Lines 248-255)

```python
def load_calibration_json(path: Path) -> Calibration:
    obj = json.loads(path.read_text())
    offset = np.array(obj["offset_ut"], dtype=float).reshape(3)
    M = np.array(obj["softiron_matrix"], dtype=float).reshape(3, 3)
    earth = float(obj.get("earth_field_ut", 0.0))
    gain = float(obj.get("gain", 1.0))
    method = str(obj.get("method", "unknown"))
    return Calibration(method=method, offset_ut=offset, softiron_matrix=M, earth_field_ut=earth, gain=gain)
```

**What it does:**

Loads calibration parameters from a JSON file and creates a `Calibration` object. This function allows you to reuse a calibration from a previous run.

**Line-by-line breakdown:**

1. **`def load_calibration_json(path: Path) -> Calibration:`**
   - Takes path to JSON file
   - Returns `Calibration` object

2. **`obj = json.loads(path.read_text())`**
   - Reads file text and parses JSON into dictionary

3. **`offset = np.array(obj["offset_ut"], dtype=float).reshape(3)`**
   - Converts list `[x, y, z]` to NumPy array of shape (3,)
   - `.reshape(3)` ensures correct shape

4. **`M = np.array(obj["softiron_matrix"], dtype=float).reshape(3, 3)`**
   - Converts nested list to 3x3 NumPy array

5. **`earth = float(obj.get("earth_field_ut", 0.0))`**
   - Gets Earth field magnitude, defaulting to 0.0 if missing
   - `.get(key, default)` safely handles missing keys

6. **`gain = float(obj.get("gain", 1.0))`**
   - Gets gain factor, defaulting to 1.0 if missing

7. **`method = str(obj.get("method", "unknown"))`**
   - Gets method string, defaulting to "unknown" if missing

8. **`return Calibration(...)`**
   - Creates and returns `Calibration` object with all parameters

---

## Section 8: Plotting Function (Lines 262-305)

### Function: `plot_projections()` (Lines 262-305)

```python
def plot_projections(B_raw: np.ndarray, B_cal: Optional[np.ndarray], out_png: Optional[Path], show: bool) -> None:
    """
    2D projections (xy, xz, yz) + magnitude histograms before/after.
    """
    fig = plt.figure(figsize=(10, 8))

    def _scatter(ax, X, Y, title):
        ax.scatter(X, Y, s=6)
        ax.set_title(title)
        ax.set_xlabel("uT")
        ax.set_ylabel("uT")
        ax.set_aspect("equal", "box")

    ax1 = fig.add_subplot(2, 2, 1)
    _scatter(ax1, B_raw[:, 0], B_raw[:, 1], "Raw XY")

    ax2 = fig.add_subplot(2, 2, 2)
    _scatter(ax2, B_raw[:, 0], B_raw[:, 2], "Raw XZ")

    ax3 = fig.add_subplot(2, 2, 3)
    _scatter(ax3, B_raw[:, 1], B_raw[:, 2], "Raw YZ")

    ax4 = fig.add_subplot(2, 2, 4)
    mag_raw = _vector_magnitude(B_raw)
    ax4.hist(mag_raw[np.isfinite(mag_raw)], bins=40, alpha=0.7, label="raw")
    if B_cal is not None:
        mag_cal = _vector_magnitude(B_cal)
        ax4.hist(mag_cal[np.isfinite(mag_cal)], bins=40, alpha=0.7, label="cal")
    ax4.set_title("|B| histogram")
    ax4.set_xlabel("uT")
    ax4.set_ylabel("count")
    ax4.legend()

    fig.tight_layout()

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=160)
        plt.close(fig)
        print(f"Wrote plot PNG: {out_png}")
    elif show:
        plt.show()
    else:
        plt.close(fig)
```

**What it does:**

Generates visualization plots showing raw and calibrated measurements. Creates a 2x2 grid of subplots:
- Three 2D scatter plots (XY, XZ, YZ projections)
- One histogram comparing raw and calibrated magnitudes

**Line-by-line breakdown:**

1. **`def plot_projections(...) -> None:`**
   - Takes raw measurements, optional calibrated measurements, optional output path, and show flag
   - Generates plots (no return value)

2. **`fig = plt.figure(figsize=(10, 8))`**
   - Creates figure with size 10x8 inches

3. **`def _scatter(ax, X, Y, title):` (Lines 268-273)**
   - Helper function to create scatter plot on given axes
   - `ax.scatter(X, Y, s=6)`: Scatter plot with small markers (size 6)
   - `ax.set_title(title)`: Sets subplot title
   - `ax.set_xlabel("uT")`, `ax.set_ylabel("uT")`: Labels axes
   - `ax.set_aspect("equal", "box")`: Makes axes equal aspect ratio (circles appear circular)

4. **Subplot 1: Raw XY (Lines 275-276)**
   - `fig.add_subplot(2, 2, 1)`: Creates subplot at position (1,1) in 2x2 grid
   - Plots Bx vs By from raw data

5. **Subplot 2: Raw XZ (Lines 278-279)**
   - Position (1,2) in grid
   - Plots Bx vs Bz from raw data

6. **Subplot 3: Raw YZ (Lines 281-282)**
   - Position (2,1) in grid
   - Plots By vs Bz from raw data

7. **Subplot 4: Magnitude histogram (Lines 284-293)**
   - `fig.add_subplot(2, 2, 4)`: Position (2,2) in grid
   - `mag_raw = _vector_magnitude(B_raw)`: Computes raw magnitudes
   - `ax4.hist(mag_raw[np.isfinite(mag_raw)], bins=40, alpha=0.7, label="raw")`: Histogram of raw magnitudes
     - `np.isfinite(...)`: Filters out NaN/infinite values
     - `bins=40`: 40 bins in histogram
     - `alpha=0.7`: 70% opacity (for overlay)
     - `label="raw"`: Legend label
   - `if B_cal is not None:`: If calibrated data provided, add its histogram
   - `ax4.legend()`: Shows legend (raw vs cal)

8. **`fig.tight_layout()` (Line 295)**
   - Adjusts subplot spacing to prevent overlap

9. **Output handling (Lines 297-305):**
   - `if out_png is not None:`: Save to file
     - Creates parent directories
     - Saves PNG with 160 DPI resolution
     - Closes figure to free memory
     - Prints confirmation message
   - `elif show:`: Display interactively
     - `plt.show()`: Opens plot window
   - `else:`: Close without showing/saving
     - Frees memory

**Example:**
The plots help visualize calibration quality:
- **Raw XY/XZ/YZ plots**: Should show an ellipsoid (distorted sphere) before calibration, and a more spherical shape after calibration
- **Magnitude histogram**: Raw magnitudes should be spread out; calibrated magnitudes should cluster around the target Earth field value

---

## Section 9: Command-Line Interface (Lines 312-339)

### Function: `parse_args()` (Lines 312-339)

```python
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit/apply magnetometer calibration and write calibration JSON (+ optional calibrated CSV).")
    p.add_argument("--in", dest="infile", required=True, help="Input CSV path containing Bx/By/Bz (uT).")
    p.add_argument("--bx-col", default="Bx", help="Column name for Bx (default: Bx)")
    p.add_argument("--by-col", default="By", help="Column name for By (default: By)")
    p.add_argument("--bz-col", default="Bz", help="Column name for Bz (default: Bz)")

    p.add_argument("--method", choices=["minmax", "ellipsoid"], default="minmax", help="Calibration method (default: minmax).")
    p.add_argument("--earth-field-ut", type=float, default=50.0, help="Expected Earth field magnitude in uT. Set <=0 to disable scaling.")
    p.add_argument("--no-earth-scaling", action="store_true", help="Disable Earth-field scaling regardless of earth_field_ut value.")

    p.add_argument("--clip-mag-z", type=float, default=6.0, help="Drop rows whose |B| robust-z exceeds this threshold (default: 6.0).")
    p.add_argument("--min-samples", type=int, default=200, help="Minimum number of samples required after cleaning (default: 200).")

    p.add_argument("--out-json", default=None, help="Output calibration JSON path. Default: <input_stem>_calibration.json next to input.")
    p.add_argument("--write-calibrated", action="store_true", help="If set, also write a calibrated CSV.")
    p.add_argument("--out-csv", default=None, help="Calibrated CSV path if --write-calibrated. Default: <input_stem>_calibrated.csv")

    # Optional: quick interference delta reporting if you labeled data
    p.add_argument("--segment-col", default=None, help="Optional column to segment data (e.g., 'power_state' or 'throttle').")
    p.add_argument("--segment-a", default=None, help="Segment label A (e.g., 'OFF').")
    p.add_argument("--segment-b", default=None, help="Segment label B (e.g., 'ON').")

    p.add_argument("--plot", action="store_true", help="If set, show/save quick projection plots + magnitude hist.")
    p.add_argument("--no-show", action="store_true", help="If set with --plot, save plot PNG instead of displaying it.")
    p.add_argument("--plot-out", default=None, help="Output PNG path if using --plot --no-show. Default: <out_json_stem>_plot.png")

    return p.parse_args()
```

**What it does:**

Defines and parses all command-line arguments. The script uses argparse to handle flexible CLI options.

**Line-by-line breakdown:**

1. **`def parse_args() -> argparse.Namespace:`**
   - Parses command-line arguments
   - Returns namespace object with argument values

2. **Input file arguments (Lines 314-317):**
   - `--in`: Required input CSV path
   - `--bx-col`, `--by-col`, `--bz-col`: Column names (default to "Bx", "By", "Bz")

3. **Calibration method arguments (Lines 319-321):**
   - `--method`: Choice between "minmax" or "ellipsoid" (default: "minmax")
   - `--earth-field-ut`: Target Earth field magnitude in microtesla (default: 50.0)
   - `--no-earth-scaling`: Flag to disable Earth-field scaling

4. **Data cleaning arguments (Lines 323-324):**
   - `--clip-mag-z`: Robust z-score threshold for outlier filtering (default: 6.0)
   - `--min-samples`: Minimum samples required after cleaning (default: 200)

5. **Output arguments (Lines 326-328):**
   - `--out-json`: Custom output JSON path (default: auto-generated)
   - `--write-calibrated`: Flag to also write calibrated CSV
   - `--out-csv`: Custom calibrated CSV path (default: auto-generated)

6. **Interference analysis arguments (Lines 331-333):**
   - `--segment-col`: Column name for segmenting data (e.g., "power_state")
   - `--segment-a`, `--segment-b`: Labels for two segments to compare

7. **Plotting arguments (Lines 335-337):**
   - `--plot`: Flag to generate plots
   - `--no-show`: Flag to save plot instead of displaying
   - `--plot-out`: Custom plot output path

**Example usage:**
```bash
python3 scripts/calibrate_magnetometerV1.py \
    --in data/raw/mag_cal.csv \
    --method ellipsoid \
    --earth-field-ut 52.0 \
    --write-calibrated \
    --plot \
    --no-show
```

---

## Section 10: Main Function (Lines 342-476)

The main function orchestrates the entire calibration process. Let's break it down section by section:

### Part 1: Argument Parsing and Input Loading (Lines 343-359)

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

    try:
        df = _ensure_cols(df, args.bx_col, args.by_col, args.bz_col)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
```

**What it does:**

1. **`args = parse_args()`**: Parses command-line arguments
2. **`infile = Path(args.infile)`**: Converts input path to Path object
3. **File existence check**: Returns error code 2 if file doesn't exist
4. **CSV reading**: Reads CSV with pandas, catches exceptions
5. **Column validation**: Ensures required columns exist and are numeric

**Exit codes:**
- `2`: Input file error (not found, unreadable, or missing columns)
- `3`: Output file error (can't write JSON/CSV)
- `0`: Success

---

### Part 2: Data Cleaning (Lines 361-384)

```python
    # Drop NaNs in components
    before = len(df)
    df = df.dropna(subset=[args.bx_col, args.by_col, args.bz_col]).copy()
    if len(df) == 0:
        print("ERROR: no valid rows after dropping NaNs in Bx/By/Bz.", file=sys.stderr)
        return 2
    if len(df) != before:
        print(f"Note: dropped {before - len(df)} rows due to NaNs in Bx/By/Bz.")

    B_raw = df[[args.bx_col, args.by_col, args.bz_col]].to_numpy(dtype=float)

    # Optional magnitude clipping to remove obvious spikes / saturations
    mag = _vector_magnitude(B_raw)
    z = _robust_z(mag)
    keep = np.abs(z) <= float(args.clip_mag_z)
    if np.any(~keep):
        dropped = int(np.sum(~keep))
        df = df.loc[keep].copy()
        B_raw = df[[args.bx_col, args.by_col, args.bz_col]].to_numpy(dtype=float)
        print(f"Note: dropped {dropped} samples where |B| robust-z > {args.clip_mag_z}.")

    if len(df) < int(args.min_samples):
        print(f"ERROR: only {len(df)} samples after cleaning; need at least {args.min_samples}.", file=sys.stderr)
        return 2
```

**What it does:**

1. **NaN removal (Lines 362-368):**
   - Drops rows with missing values in Bx/By/Bz
   - Reports how many rows were dropped
   - Returns error if no valid data remains

2. **Convert to NumPy array (Line 370):**
   - Extracts Bx/By/Bz columns and converts to NumPy array of shape `(N, 3)`

3. **Outlier filtering (Lines 372-380):**
   - Computes magnitude of each measurement
   - Computes robust z-scores
   - Filters out measurements with `|z| > clip_mag_z` (default 6.0)
   - Removes obvious spikes/saturations that would corrupt calibration

4. **Minimum samples check (Lines 382-384):**
   - Ensures at least `min_samples` (default 200) remain after cleaning
   - Calibration requires sufficient data for reliable fit

**Example:**
If input has 500 rows, but 50 have NaN values and 10 are outliers, 440 rows remain. If `min_samples=200`, this passes. If only 150 rows remain, calibration fails.

---

### Part 3: Calibration Fitting (Lines 386-405)

```python
    # Fit calibration
    try:
        if args.method == "ellipsoid":
            offset, M = fit_ellipsoid(B_raw)
        else:
            offset, M = fit_minmax(B_raw)
    except Exception as e:
        print(f"ERROR: calibration fit failed: {e}", file=sys.stderr)
        return 2

    earth_field_ut = float(args.earth_field_ut)
    if args.no_earth_scaling:
        earth_field_ut = 0.0

    cal0 = Calibration(method=args.method, offset_ut=offset, softiron_matrix=M, earth_field_ut=earth_field_ut, gain=1.0)
    B_cal0 = apply_calibration(B_raw, cal0)

    gain = compute_gain_to_match_earth(B_cal0, earth_field_ut=earth_field_ut)
    cal = Calibration(method=args.method, offset_ut=offset, softiron_matrix=M, earth_field_ut=earth_field_ut, gain=gain)
    B_cal = apply_calibration(B_raw, cal)
```

**What it does:**

1. **Fit calibration (Lines 387-393):**
   - Calls `fit_ellipsoid()` or `fit_minmax()` based on `--method` argument
   - Catches exceptions and returns error code 2 if fit fails

2. **Earth field scaling setup (Lines 395-397):**
   - Gets Earth field target from arguments
   - If `--no-earth-scaling` flag is set, disables scaling (sets to 0.0)

3. **Two-stage calibration (Lines 399-405):**
   - Creates initial calibration with `gain=1.0`
   - Applies calibration to compute intermediate result `B_cal0`
   - Computes gain factor to match Earth field magnitude
   - Creates final calibration with computed gain
   - Applies final calibration to get `B_cal`

**Why two-stage?**
The gain depends on the calibrated magnitudes, so we first calibrate without gain, compute the gain, then apply it. This ensures the final calibrated magnitudes match the target Earth field.

---

### Part 4: Statistics and Interference Analysis (Lines 407-432)

```python
    # Basic stats
    mag_raw = _vector_magnitude(B_raw)
    mag_cal = _vector_magnitude(B_cal)
    print(f"Raw |B| median: {float(np.median(mag_raw)):.3f} uT   (min={float(np.min(mag_raw)):.3f}, max={float(np.max(mag_raw)):.3f})")
    print(f"Cal |B| median: {float(np.median(mag_cal)):.3f} uT   (min={float(np.min(mag_cal)):.3f}, max={float(np.max(mag_cal)):.3f})")
    if earth_field_ut > 0:
        print(f"Earth scaling target: {earth_field_ut:.3f} uT   -> applied gain: {gain:.6f}")

    # Optional interference delta report
    meta: dict = {"platform_parts": DEFAULT_PLATFORM_PARTS}
    if args.segment_col and (args.segment_col in df.columns) and (args.segment_a is not None) and (args.segment_b is not None):
        seg = df[args.segment_col].astype(str)
        mask_a = seg == str(args.segment_a)
        mask_b = seg == str(args.segment_b)
        if np.any(mask_a) and np.any(mask_b):
            a_med = np.median(_vector_magnitude(B_cal[mask_a.to_numpy()]))
            b_med = np.median(_vector_magnitude(B_cal[mask_b.to_numpy()]))
            meta["segment_col"] = args.segment_col
            meta["segment_a"] = str(args.segment_a)
            meta["segment_b"] = str(args.segment_b)
            meta["segment_a_median_B_ut"] = float(a_med)
            meta["segment_b_median_B_ut"] = float(b_med)
            meta["segment_delta_median_B_ut"] = float(b_med - a_med)
            print(f"Interference check ({args.segment_col}): median |B| {args.segment_a}={a_med:.3f} uT, {args.segment_b}={b_med:.3f} uT (delta={b_med-a_med:+.3f} uT)")
        else:
            print("Note: segment labels not found in data for interference check; skipping.", file=sys.stderr)
```

**What it does:**

1. **Statistics printing (Lines 408-412):**
   - Computes raw and calibrated magnitudes
   - Prints median, min, max for both
   - Prints Earth field target and applied gain if scaling is enabled

2. **Interference analysis (Lines 414-431):**
   - Initializes metadata dictionary with platform parts
   - If segmentation arguments provided, compares two segments (e.g., power ON vs OFF)
   - Computes median magnitude for each segment
   - Stores results in metadata
   - Prints comparison

**Example output:**
```
Raw |B| median: 48.234 uT   (min=30.123, max=65.789)
Cal |B| median: 50.001 uT   (min=45.234, max=54.567)
Earth scaling target: 50.000 uT   -> applied gain: 1.036621
Interference check (power_state): median |B| OFF=50.001 uT, ON=51.234 uT (delta=+1.233 uT)
```

This shows that when power is ON, the magnetic field is 1.233 uT higher (interference from electronics).

---

### Part 5: Output Writing (Lines 434-463)

```python
    # Write calibration JSON
    if args.out_json is None:
        out_json = infile.with_name(infile.stem + "_calibration.json")
    else:
        out_json = Path(args.out_json)

    try:
        write_calibration_json(out_json, cal=cal, meta=meta)
    except Exception as e:
        print(f"ERROR: could not write calibration JSON: {e}", file=sys.stderr)
        return 3
    print(f"Wrote calibration JSON: {out_json}")

    # Optional calibrated CSV
    if args.write_calibrated:
        out_csv = Path(args.out_csv) if args.out_csv is not None else infile.with_name(infile.stem + "_calibrated.csv")

        df_out = df.copy()
        df_out["Bx_cal"] = B_cal[:, 0]
        df_out["Bx_cal"] = B_cal[:, 1]
        df_out["Bz_cal"] = B_cal[:, 2]
        df_out["B_total_cal"] = mag_cal

        try:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(out_csv, index=False)
        except Exception as e:
            print(f"ERROR: could not write calibrated CSV: {e}", file=sys.stderr)
            return 3
        print(f"Wrote calibrated CSV: {out_csv}")
```

**What it does:**

1. **JSON output path (Lines 435-438):**
   - If `--out-json` not provided, auto-generates path: `<input_stem>_calibration.json`
   - Example: `data/raw/mag_cal.csv` → `data/raw/mag_cal_calibration.json`

2. **Write calibration JSON (Lines 440-445):**
   - Calls `write_calibration_json()` to write file
   - Catches exceptions and returns error code 3 if write fails
   - Prints confirmation message

3. **Calibrated CSV (Lines 447-463):**
   - Only writes if `--write-calibrated` flag is set
   - Auto-generates path if `--out-csv` not provided: `<input_stem>_calibrated.csv`
   - Adds calibrated columns to DataFrame:
     - `Bx_cal`, `By_cal`, `Bz_cal`: Calibrated components
     - `B_total_cal`: Calibrated magnitude
   - Writes CSV (index=False means no row numbers)
   - Prints confirmation

---

### Part 6: Plotting (Lines 465-474)

```python
    # Plot (optional)
    if args.plot:
        if args.no_show:
            if args.plot_out is None:
                plot_out = out_json.with_suffix("").with_name(out_json.stem + "_plot.png")
            else:
                plot_out = Path(args.plot_out)
            plot_projections(B_raw=B_raw, B_cal=B_cal, out_png=plot_out, show=False)
        else:
            plot_projections(B_raw=B_raw, B_cal=B_cal, out_png=None, show=True)

    return 0
```

**What it does:**

1. **Plot generation (Lines 466-474):**
   - Only plots if `--plot` flag is set
   - If `--no-show` is set:
     - Auto-generates plot path if `--plot-out` not provided: `<out_json_stem>_plot.png`
     - Calls `plot_projections()` with `out_png` path, `show=False`
   - Otherwise:
     - Calls `plot_projections()` with `out_png=None`, `show=True` (displays interactively)

2. **Return success (Line 476):**
   - Returns 0 to indicate successful completion

---

## Section 11: Script Entry Point (Lines 479-484)

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
   - Allows script to be used as both standalone program and importable module

2. **`raise SystemExit(main())`**
   - Calls `main()` and exits with its return code (0=success, 2=error, 3=output error)
   - `raise SystemExit()` is equivalent to `sys.exit()` but cleaner

3. **`except KeyboardInterrupt:`**
   - Catches Ctrl+C interruption
   - Prints "Stopped." message
   - Exits with code 130 (standard exit code for keyboard interrupt)

---

## How to Use the Script

### Basic Usage

**Step 1: Collect calibration data**

Before running calibration, you need to collect data by rotating the magnetometer through many orientations. This can be done manually or with a data collection script. The data should be saved as a CSV with columns `Bx`, `By`, `Bz` (in microtesla).

**Example calibration data collection:**
- Mount the sensor in its final configuration (on drone frame, with all electronics)
- Slowly rotate the sensor through as many orientations as possible (tip, rotate, flip)
- Collect at least 200 samples (preferably 500+)
- Save to CSV: `data/raw/mag_cal.csv`

**Step 2: Run calibration**

```bash
# Simple calibration with default settings
python3 scripts/calibrate_magnetometerV1.py --in data/raw/mag_cal.csv

# Use ellipsoid method with Earth field scaling
python3 scripts/calibrate_magnetometerV1.py \
    --in data/raw/mag_cal.csv \
    --method ellipsoid \
    --earth-field-ut 52.0

# Full workflow: calibrate + write calibrated CSV + save plots
python3 scripts/calibrate_magnetometerV1.py \
    --in data/raw/mag_cal.csv \
    --method ellipsoid \
    --earth-field-ut 50.0 \
    --write-calibrated \
    --plot \
    --no-show
```

### Command-Line Arguments Summary

**Required:**
- `--in PATH`: Input CSV file with Bx/By/Bz columns

**Optional (calibration method):**
- `--method {minmax,ellipsoid}`: Calibration method (default: minmax)
- `--earth-field-ut FLOAT`: Target Earth field magnitude in uT (default: 50.0)
- `--no-earth-scaling`: Disable Earth-field scaling

**Optional (data cleaning):**
- `--clip-mag-z FLOAT`: Robust z-score threshold for outlier filtering (default: 6.0)
- `--min-samples INT`: Minimum samples required after cleaning (default: 200)
- `--bx-col NAME`: Column name for Bx (default: Bx)
- `--by-col NAME`: Column name for By (default: By)
- `--bz-col NAME`: Column name for Bz (default: Bz)

**Optional (outputs):**
- `--out-json PATH`: Output calibration JSON path (default: auto-generated)
- `--write-calibrated`: Also write calibrated CSV
- `--out-csv PATH`: Calibrated CSV path (default: auto-generated)

**Optional (interference analysis):**
- `--segment-col NAME`: Column name for segmenting data
- `--segment-a LABEL`: Label for segment A (e.g., "OFF")
- `--segment-b LABEL`: Label for segment B (e.g., "ON")

**Optional (plotting):**
- `--plot`: Generate visualization plots
- `--no-show`: Save plot PNG instead of displaying
- `--plot-out PATH`: Plot output path (default: auto-generated)

### Output Files

1. **Calibration JSON** (`<input_stem>_calibration.json`):
   - Contains all calibration parameters
   - Can be loaded and applied to new measurements
   - Includes metadata (timestamp, method, platform parts, interference analysis)

2. **Calibrated CSV** (if `--write-calibrated` is used):
   - Original data plus calibrated columns: `Bx_cal`, `By_cal`, `Bz_cal`, `B_total_cal`
   - Useful for validating calibration quality

3. **Plot PNG** (if `--plot --no-show` is used):
   - 2x2 grid showing XY/XZ/YZ projections and magnitude histogram
   - Helps visualize calibration effectiveness

### Interpreting Results

**Good calibration indicators:**
- Calibrated magnitudes cluster around the target Earth field value
- Magnitude histogram is narrow (low spread)
- 2D projection plots show more circular/spherical shape after calibration

**Troubleshooting:**
- **"calibration fit failed"**: Try `--method minmax` (more robust) or collect more diverse data
- **"only N samples after cleaning"**: Reduce `--clip-mag-z` threshold or collect more data
- **Large magnitude spread after calibration**: May indicate soft-iron effects not fully corrected (try ellipsoid method)
- **Calibrated magnitudes don't match Earth field**: Check `--earth-field-ut` value (varies by location: ~25-65 uT globally)

---

## Context in the Fluxspace Pipeline

### When to Use Calibration

Magnetometer calibration is typically performed:

1. **Initial setup**: When first installing the sensor
2. **After hardware changes**: When modifying the drone frame, motors, ESCs, or other magnetic components
3. **Periodic recalibration**: If measurements seem inconsistent over time

### Integration with Other Scripts

Currently, `calibrate_magnetometerV1.py` is a standalone script. In a future workflow, it could be integrated as:

1. **Preprocessing step** before data collection:
   - Calibrate sensor → save JSON
   - Apply calibration during data collection (in `mag_to_csv.py` or similar)

2. **Post-processing step**:
   - Collect raw data
   - Calibrate → save JSON
   - Apply calibration to existing datasets

The calibration JSON file format allows the calibration to be reused across multiple measurement sessions, as long as the hardware configuration remains the same.

### Why Calibration Matters

In the Fluxspace Core project, accurate magnetic field measurements are essential for:

- **Anomaly detection**: Small local anomalies (e.g., buried metal objects) can be obscured by calibration errors
- **Spatial mapping**: Calibration errors cause systematic biases that distort spatial patterns
- **Comparative analysis**: Measurements from different sessions or sensors need consistent calibration for valid comparisons

Without calibration, sensor biases and distortions can introduce errors larger than the anomalies you're trying to detect. For example, a 10 uT hard-iron offset could mask a 5 uT local anomaly.

---

## Key Concepts Explained

### Hard-Iron Offset

**What it is:** A constant bias added to all measurements, represented as a 3D vector offset.

**Causes:**
- Permanent magnets in the sensor or nearby hardware
- Ferrous materials (steel screws, frames) that are permanently magnetized
- DC magnetic fields from electronics

**Effect:** Shifts the center of the measurement sphere/ellipsoid away from the origin.

**Correction:** Subtract the offset vector from raw measurements: `B_corrected = B_raw - offset`

**Example:** If offset is `[10, -5, 2]` uT, a raw measurement of `[50, 45, 40]` uT becomes `[40, 50, 38]` uT after hard-iron correction.

---

### Soft-Iron Distortion

**What it is:** A matrix transformation that warps the measurement space, represented as a 3x3 matrix.

**Causes:**
- Materials that concentrate or redirect magnetic field lines (soft ferrous materials)
- Asymmetric sensor mounting
- Cross-axis coupling in the sensor electronics

**Effect:** Distorts the measurement sphere into an ellipsoid (stretched/compressed/rotated).

**Correction:** Multiply by correction matrix: `B_corrected = matrix @ (B_raw - offset)`

**Example:** If the matrix is:
```
[[1.1, 0.0, 0.0],
 [0.0, 0.9, 0.0],
 [0.0, 0.0, 1.0]]
```
This stretches X by 10% and compresses Y by 10%, correcting for soft-iron effects that caused opposite distortion.

---

### Earth Field Scaling

**What it is:** A scalar gain factor that scales calibrated magnitudes to match the expected local Earth field strength.

**Why it's needed:** Calibration corrects the *shape* of measurements (offset and distortion), but the absolute magnitude may still be off. Earth field scaling ensures the sensor reads the correct absolute field strength.

**How it works:**
1. After calibration (offset + matrix), compute median magnitude
2. Compare to expected Earth field magnitude for your location
3. Compute gain: `gain = expected / median`
4. Apply gain: `B_final = gain × B_calibrated`

**Example:** If calibrated median is 48 uT but local Earth field is 50 uT, gain is `50/48 = 1.042`. All measurements are multiplied by 1.042 to match the expected field.

**Finding your local Earth field:**
- Use online calculators (e.g., NOAA Magnetic Field Calculator)
- Typical values: 25-65 uT globally, ~45-55 uT for mid-latitudes
- Can be measured with a calibrated reference magnetometer

---

## Examples

### Example 1: Basic Calibration

**Input CSV** (`data/raw/mag_cal.csv`):
```csv
Bx,By,Bz
45.2,50.1,48.3
46.1,49.8,47.9
...
```

**Command:**
```bash
python3 scripts/calibrate_magnetometerV1.py --in data/raw/mag_cal.csv
```

**Output:**
- `data/raw/mag_cal_calibration.json`: Calibration parameters
- Console output:
  ```
  Raw |B| median: 48.234 uT   (min=30.123, max=65.789)
  Cal |B| median: 50.001 uT   (min=45.234, max=54.567)
  Earth scaling target: 50.000 uT   -> applied gain: 1.036621
  Wrote calibration JSON: data/raw/mag_cal_calibration.json
  ```

---

### Example 2: Full Workflow with Plots

**Command:**
```bash
python3 scripts/calibrate_magnetometerV1.py \
    --in data/raw/mag_cal.csv \
    --method ellipsoid \
    --earth-field-ut 52.0 \
    --write-calibrated \
    --plot \
    --no-show
```

**Outputs:**
- `data/raw/mag_cal_calibration.json`: Calibration JSON
- `data/raw/mag_cal_calibrated.csv`: Original data + calibrated columns
- `data/raw/mag_cal_calibration_plot.png`: Visualization plots

---

### Example 3: Interference Analysis

If your CSV has a `power_state` column with values "OFF" and "ON":

**Command:**
```bash
python3 scripts/calibrate_magnetometerV1.py \
    --in data/raw/mag_cal.csv \
    --segment-col power_state \
    --segment-a OFF \
    --segment-b ON
```

**Output:**
- Calibration JSON includes interference analysis in metadata
- Console shows: `Interference check (power_state): median |B| OFF=50.001 uT, ON=51.234 uT (delta=+1.233 uT)`

This helps quantify how much interference the electronics add when powered on.

---

## Summary

`calibrate_magnetometerV1.py` is a comprehensive tool for magnetometer calibration that:

1. **Fits calibration parameters** using robust mathematical methods (minmax or ellipsoid)
2. **Corrects for sensor errors** (hard-iron offset and soft-iron distortion)
3. **Scales to Earth field** to ensure correct absolute magnitudes
4. **Outputs reusable calibration files** (JSON format)
5. **Provides visualization and analysis tools** (plots, interference analysis)

The script is designed to integrate into the Fluxspace Core pipeline, ensuring that magnetic field measurements are accurate and consistent, which is essential for reliable anomaly detection and spatial mapping.
