# Complete Explanation of `validate_and_diagnosticsV1.py`

This document explains every part of the validation and diagnostics script, step by step.

---

## Overview

This script reads a magnetometer CSV file (like the one produced by `mag_to_csv.py`), validates it, cleans obvious issues, and generates quick diagnostics including plots and a text report. It helps ensure data quality before further analysis by detecting outliers, spikes, and other data quality issues.

**What it does:**
- Validates CSV structure and required columns
- Cleans missing/invalid data
- Detects outliers using robust statistics
- Detects spikes (sudden changes) in measurements
- Generates diagnostic plots
- Creates a text report with statistics

**Typical usage:**
```bash
python3 scripts/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv
```

**Outputs (in `data/processed/` by default):**
- `<stem>_clean.csv` - Cleaned data with flag columns
- `<stem>_report.txt` - Text report with statistics
- `<stem>_Btotal_vs_time.png` - Time series plot
- `<stem>_Btotal_hist.png` - Histogram of B_total
- `<stem>_scatter_xy_colored.png` - Spatial scatter plot
- `<stem>_spike_deltas.png` - Spike detection plot

---

## Section 1: Imports and Setup (Lines 1-30)

```python
#!/usr/bin/env python3
"""
validate_and_diagnostics.py

Reads a magnetometer CSV (like the one produced by mag_to_csv.py), validates it,
cleans obvious issues, and generates quick diagnostics (plots + a text report).

Typical usage:
  python3 scripts/validate_and_diagnostics.py --in data/raw/mag_data.csv

Outputs (by default) in data/processed/:
  - <stem>_clean.csv
  - <stem>_report.txt
  - <stem>_Btotal_vs_time.png
  - <stem>_Btotal_hist.png
  - <stem>_scatter_xy_colored.png
  - <stem>_spike_deltas.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

**What it does:**

1. **`#!/usr/bin/env python3`** (Shebang)
   - Makes the script executable directly from the command line

2. **Module docstring** (Lines 2-18)
   - Documents the script's purpose and usage
   - Shows example command-line usage
   - Lists output files

3. **`from __future__ import annotations`**
   - Enables postponed evaluation of type annotations
   - Allows using types without quotes in older Python versions
   - Allows calls from other scrips without endless looping issues

4. **Standard library imports:**
   - `argparse`: Allows use of command-line arguments
        - Defines options like --in, --outdir, --z-thresh
   - `sys`: System-specific parameters (for error output)
   - `pathlib.Path`: A safer, cleaner way to handle file paths than raw strings (Mac, Linux, Windows)
   - `typing`: Type hints for function signatures (autocomplete, catch mistakes, clarity)

5. **External library imports:**
   - `numpy`: Numerical operations and arrays
   - `pandas`: Data manipulation and CSV handling
   - `matplotlib.pyplot`: Plotting and visualization

---

## Section 2: Helper Functions (Lines 32-100)

### Function 1: `_normalize_columns()` (Lines 36-39)

```python
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df
```

**What it does:**

Normalizes column names by converting them to strings and stripping whitespace. This handles cases where column names might have extra spaces or be in different formats.

**Line-by-line breakdown:**

1. **`def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:`**
   - Function takes a DataFrame and returns a normalized DataFrame
   - `_` prefix indicates this is an internal helper function

2. **`df = df.copy()`**
   - Creates a copy to avoid modifying the original DataFrame

3. **`df.columns = [str(c).strip() for c in df.columns]`**
   - Converts each column name to string and strips whitespace
   - List comprehension processes all columns

4. **`return df`**
   - Returns the normalized DataFrame

**Example:**
- Input columns: `[" x ", "y ", " B_total"]`
- Output columns: `["x", "y", "B_total"]`

---

### Function 2: `_find_time_column()` (Lines 41-47)

```python
def _find_time_column(cols: List[str]) -> Optional[str]:
    # Common names you might use
    candidates = ["time", "timestamp", "t", "datetime", "date_time"]
    for c in candidates:
        if c in cols:
            return c
    return None
```

**What it does:**

Searches for a time column by checking common column name variations. Returns the first match found, or `None` if no time column is found.

**Line-by-line breakdown:**

1. **`def _find_time_column(cols: List[str]) -> Optional[str]:`**
   - Takes a list of column names
   - Returns the time column name if found, `None` otherwise

2. **`candidates = ["time", "timestamp", "t", "datetime", "date_time"]`**
   - List of common time column name variations

3. **`for c in candidates:`**
   - Loops through each candidate name

4. **`if c in cols:`**
   - Checks if candidate is in the column list (case-sensitive)

5. **`return c`**
   - Returns the first matching column name

6. **`return None`**
   - Returns `None` if no time column found

**Example:**
- Input: `["x", "y", "time", "B_total"]` → Returns: `"time"`
- Input: `["x", "y", "timestamp", "B_total"]` → Returns: `"timestamp"`
- Input: `["x", "y", "B_total"]` → Returns: `None`

---

### Function 3: `_coerce_time_series()` (Lines 49-70)

```python
def _coerce_time_series(s: pd.Series) -> Tuple[Optional[pd.Series], str]:
    """
    Return (time_series, note).
    Accepts:
      - ISO timestamps (strings)
      - unix seconds (float/int)
    """
    if s is None:
        return None, "No time column"
    # Try numeric unix time first
    if pd.api.types.is_numeric_dtype(s):
        # Treat as seconds since epoch
        t = pd.to_datetime(s, unit="s", errors="coerce", utc=True)
        ok = t.notna().mean()
        if ok > 0.8:
            return t, "Parsed numeric unix seconds as UTC"
    # Try general datetime parsing
    t = pd.to_datetime(s, errors="coerce", utc=True)
    ok = t.notna().mean()
    if ok > 0.8:
        return t, "Parsed as datetime (UTC)"
    return None, "Could not parse time column reliably"
```

**What it does:**

Attempts to parse a time column into pandas datetime format. Handles both numeric Unix timestamps and ISO datetime strings. Returns the parsed series and a note describing what was done. This function is robust and tries multiple parsing strategies to handle different time formats.

**Detailed Line-by-line breakdown:**

1. **`def _coerce_time_series(s: pd.Series) -> Tuple[Optional[pd.Series], str]:`**
   - **`s: pd.Series`**: Input pandas Series containing time data (could be strings, numbers, etc.)
   - **`-> Tuple[Optional[pd.Series], str]`**: Returns a tuple with:
     - First element: Parsed datetime Series if successful, `None` if failed
     - Second element: String note describing what happened
   - **Why Optional?** Function might fail to parse, so returns `None` instead of invalid data

2. **`if s is None:`**
   - Checks if no time column was provided (Series is `None`)
   - This happens when `_find_time_column()` returns `None`

3. **`return None, "No time column"`**
   - Early return if no time data provided
   - Returns `None` for the series and a descriptive note

4. **`if pd.api.types.is_numeric_dtype(s):`**
   - **`pd.api.types.is_numeric_dtype(s)`**: Checks if Series contains numeric data types
   - Returns `True` for: int, float, int64, float64, etc.
   - Returns `False` for: strings, objects, etc.
   - **Why check first?** Numeric data is likely Unix timestamps (seconds since epoch)
   - **Strategy**: Try numeric parsing first because it's faster and more specific

5. **`t = pd.to_datetime(s, unit="s", errors="coerce", utc=True)`**
   - **`pd.to_datetime()`**: Pandas function to convert to datetime
   - **`unit="s"`**: Specifies input is in seconds since Unix epoch (Jan 1, 1970 00:00:00 UTC)
   - **`errors="coerce"`**: 
     - Invalid values become `NaT` (Not a Time) instead of raising error
     - Example: `[1609459200, "invalid", 1609632000]` → `[2021-01-01, NaT, 2021-01-03]`
   - **`utc=True`**: 
     - Treats input as UTC timezone
     - Output datetime is timezone-aware (UTC)
     - Important for consistent time handling across timezones
   - **Result**: `t` is a DatetimeIndex with UTC timezone

6. **`ok = t.notna().mean()`**
   - **`t.notna()`**: Returns boolean Series: `True` for valid dates, `False` for NaT
   - **`.mean()`**: Calculates fraction of `True` values (success rate)
   - **Example**: If 90 out of 100 values parsed successfully, `ok = 0.9` (90%)
   - **Why check success rate?** Need to ensure parsing worked correctly

7. **`if ok > 0.8:`**
   - **Threshold: 80% success rate**
   - If >80% of values parsed successfully, accept this format
   - **Why 80%?** Allows for some invalid/missing values while still being confident about format
   - If <80% success, try alternative parsing method

8. **`return t, "Parsed numeric unix seconds as UTC"`**
   - Returns successfully parsed datetime Series
   - Returns descriptive note for logging/reporting

9. **`t = pd.to_datetime(s, errors="coerce", utc=True)`**
   - **Fallback parsing**: If numeric parsing failed or wasn't applicable
   - **No `unit` parameter**: Pandas tries to infer format automatically
   - **Handles formats like:**
     - ISO 8601: `"2024-03-15T14:23:45Z"`
     - Common formats: `"2024-03-15 14:23:45"`
     - Various date formats: `"03/15/2024"`, `"15-Mar-2024"`, etc.
   - **`errors="coerce"`**: Invalid values become NaT
   - **`utc=True`**: Output is UTC timezone

10. **`ok = t.notna().mean()`**
    - Calculates success rate for general parsing
    - Same logic as step 6

11. **`if ok > 0.8:`**
    - If >80% parsed successfully, accept this format

12. **`return t, "Parsed as datetime (UTC)"`**
    - Returns parsed datetime Series
    - Returns note indicating general datetime parsing was used

13. **`return None, "Could not parse time column reliably"`**
    - **Fallback**: If both parsing attempts failed or had <80% success
    - Returns `None` to indicate parsing failed
    - Returns descriptive error note

**Why This Two-Step Approach?**

1. **Numeric first**: Unix timestamps are common and unambiguous
2. **General second**: Handles various string formats as fallback
3. **Success threshold**: 80% ensures format is correct while allowing some bad data
4. **UTC timezone**: Ensures consistent time handling regardless of local timezone

**Detailed Examples:**

**Example 1: Unix Timestamps (Numeric)**
```python
# Input Series
s = pd.Series([1609459200, 1609545600, 1609632000, 1609718400])

# Step 1: Check if numeric
pd.api.types.is_numeric_dtype(s)  # Returns: True

# Step 2: Parse as Unix seconds
t = pd.to_datetime(s, unit="s", errors="coerce", utc=True)
# Result: DatetimeIndex(['2021-01-01 00:00:00+00:00', 
#                        '2021-01-02 00:00:00+00:00',
#                        '2021-01-03 00:00:00+00:00',
#                        '2021-01-04 00:00:00+00:00'], dtype='datetime64[ns, UTC]')

# Step 3: Check success rate
ok = t.notna().mean()  # Returns: 1.0 (100% success)

# Step 4: Return result
return t, "Parsed numeric unix seconds as UTC"
```

**Example 2: ISO 8601 Strings**
```python
# Input Series
s = pd.Series(["2024-03-15T14:23:45Z", 
               "2024-03-15T14:23:46Z", 
               "2024-03-15T14:23:47Z"])

# Step 1: Check if numeric
pd.api.types.is_numeric_dtype(s)  # Returns: False (strings)

# Step 2: Skip numeric parsing, try general parsing
t = pd.to_datetime(s, errors="coerce", utc=True)
# Result: DatetimeIndex(['2024-03-15 14:23:45+00:00',
#                        '2024-03-15 14:23:46+00:00',
#                        '2024-03-15 14:23:47+00:00'], dtype='datetime64[ns, UTC]')

# Step 3: Check success rate
ok = t.notna().mean()  # Returns: 1.0 (100% success)

# Step 4: Return result
return t, "Parsed as datetime (UTC)"
```

**Example 3: Mixed Valid/Invalid Data**
```python
# Input Series with some invalid values
s = pd.Series([1609459200, 1609545600, "invalid", 1609718400, None])

# Step 1: Check if numeric
pd.api.types.is_numeric_dtype(s)  # Returns: False (has strings/None)

# Step 2: Try general parsing
t = pd.to_datetime(s, errors="coerce", utc=True)
# Result: DatetimeIndex(['2021-01-01 00:00:00+00:00',
#                        '2021-01-02 00:00:00+00:00',
#                        NaT,  # invalid string
#                        '2021-01-04 00:00:00+00:00',
#                        NaT], dtype='datetime64[ns, UTC]')  # None

# Step 3: Check success rate
ok = t.notna().mean()  # Returns: 0.6 (60% success - 3 out of 5)

# Step 4: 60% < 80%, so parsing fails
return None, "Could not parse time column reliably"
```

**Example 4: Unix Timestamps with Some Invalid**
```python
# Input Series
s = pd.Series([1609459200, 1609545600, -1, 1609718400, 999999999999])

# Step 1: Check if numeric
pd.api.types.is_numeric_dtype(s)  # Returns: True

# Step 2: Parse as Unix seconds
t = pd.to_datetime(s, unit="s", errors="coerce", utc=True)
# Result: DatetimeIndex(['2021-01-01 00:00:00+00:00',
#                        '2021-01-02 00:00:00+00:00',
#                        NaT,  # -1 is invalid
#                        '2021-01-04 00:00:00+00:00',
#                        '2001-09-09 01:46:39+00:00'], dtype='datetime64[ns, UTC]')

# Step 3: Check success rate
ok = t.notna().mean()  # Returns: 0.8 (80% success - 4 out of 5)

# Step 4: 80% >= 80%, so parsing succeeds
return t, "Parsed numeric unix seconds as UTC"
```

**Understanding the Parameters:**

**`unit="s"` (seconds):**
- Unix epoch: January 1, 1970 00:00:00 UTC
- `1609459200` seconds = January 1, 2021 00:00:00 UTC
- Other units available: `"ms"` (milliseconds), `"us"` (microseconds), `"ns"` (nanoseconds)

**`errors="coerce"`:**
- **Alternative: `errors="raise"`**: Would raise exception on first invalid value
- **Alternative: `errors="ignore"`**: Would return original Series unchanged
- **`"coerce"`**: Best for data cleaning - converts invalid to NaT, keeps valid values

**`utc=True`:**
- **Without `utc=True`**: Datetimes are timezone-naive (ambiguous)
- **With `utc=True`**: Datetimes are timezone-aware (UTC)
- **Why UTC?** 
  - Consistent across different local timezones
  - Standard for scientific data
  - Avoids daylight saving time issues

**Success Rate Threshold (80%):**

**Why 80%?**
- **Too low (e.g., 50%)**: Might accept wrong format with many errors
- **Too high (e.g., 95%)**: Might reject correct format with a few bad values
- **80%**: Good balance - confident about format while allowing some data quality issues

**Real-world scenarios:**
- **100% valid**: Perfect parsing
- **90% valid**: One bad value in 10 - still confident
- **80% valid**: Two bad values in 10 - acceptable threshold
- **70% valid**: Too many errors - format might be wrong

**Return Value Usage:**

The function returns a tuple that's unpacked like this:
```python
t_series, tnote = _coerce_time_series(df[time_col] if time_col else None)

# t_series: Parsed datetime Series or None
# tnote: Descriptive string like "Parsed numeric unix seconds as UTC"
```

**The note is used for:**
- Diagnostic reporting (shows in the report file)
- Understanding what parsing method worked
- Debugging if parsing fails

**Common Time Formats Handled:**

1. **Unix timestamps (numeric):**
   - `1609459200` (seconds)
   - `1609459200000` (milliseconds, would need `unit="ms"`)

2. **ISO 8601 strings:**
   - `"2024-03-15T14:23:45Z"` (with Z for UTC)
   - `"2024-03-15T14:23:45+00:00"` (with timezone offset)
   - `"2024-03-15T14:23:45"` (without timezone, assumed UTC)

3. **Common date formats:**
   - `"2024-03-15 14:23:45"`
   - `"03/15/2024"`
   - `"15-Mar-2024"`

**Edge Cases Handled:**

1. **None input**: Returns early with "No time column" note
2. **All invalid values**: Success rate < 80%, returns None
3. **Mixed types**: General parsing handles mixed formats
4. **Timezone issues**: All output is UTC, avoiding timezone confusion
5. **Large numbers**: Handles both small (seconds) and large (milliseconds) Unix timestamps

**Why This Function is Important:**

- **Flexibility**: Handles multiple time formats automatically
- **Robustness**: Doesn't crash on invalid data, uses success threshold
- **Consistency**: Always outputs UTC timezone-aware datetimes
- **Diagnostics**: Returns notes explaining what happened
- **Data quality**: 80% threshold ensures format is correct

---

### Function 4: `_compute_btotal_if_missing()` (Lines 72-84)

```python
def _compute_btotal_if_missing(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    if "B_total" in df.columns:
        return df, "B_total present"
    # Compute from vector if possible
    if all(c in df.columns for c in ["Bx", "By", "Bz"]):
        bx = pd.to_numeric(df["Bx"], errors="coerce")
        by = pd.to_numeric(df["By"], errors="coerce")
        bz = pd.to_numeric(df["Bz"], errors="coerce")
        df["Bx"], df["By"], df["Bz"] = bx, by, bz
        df["B_total"] = np.sqrt(bx.to_numpy()**2 + by.to_numpy()**2 + bz.to_numpy()**2)
        return df, "Computed B_total = sqrt(Bx^2 + By^2 + Bz^2)"
    return df, "Missing B_total and cannot compute (need Bx,By,Bz)"
```

**What it does:**

Checks if `B_total` exists. If not, computes it from `Bx`, `By`, `Bz` components using the 3D Pythagorean theorem.

**Line-by-line breakdown:**

1. **`def _compute_btotal_if_missing(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:`**
   - Takes DataFrame, returns DataFrame with B_total and a note

2. **`df = df.copy()`**
   - Creates copy to avoid modifying original

3. **`if "B_total" in df.columns:`**
   - Checks if B_total already exists

4. **`return df, "B_total present"`**
   - Returns unchanged DataFrame if B_total exists

5. **`if all(c in df.columns for c in ["Bx", "By", "Bz"]):`**
   - Checks if all three components exist

6. **`bx = pd.to_numeric(df["Bx"], errors="coerce")`**
   - Converts to numeric, invalid values become NaN

7. **`df["B_total"] = np.sqrt(bx.to_numpy()**2 + by.to_numpy()**2 + bz.to_numpy()**2)`**
   - Computes: B_total = √(Bx² + By² + Bz²)
   - Uses NumPy for efficient vectorized calculation

8. **`return df, "Computed B_total = sqrt(Bx^2 + By^2 + Bz^2)"`**
   - Returns DataFrame with computed B_total

9. **`return df, "Missing B_total and cannot compute (need Bx,By,Bz)"`**
   - Returns unchanged DataFrame if can't compute

**Example:**
- If B_total exists: Returns DataFrame unchanged
- If Bx=45, By=23, Bz=12: Computes B_total ≈ 51.94
- If components missing: Returns error note

---

### Function 5: `_robust_z()` (Lines 86-93)

```python
def _robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score using MAD. Returns zeros if degenerate."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad
```

**What it does:**

Computes robust z-scores using Median Absolute Deviation (MAD) instead of standard deviation. This is more resistant to outliers than regular z-scores.

**Line-by-line breakdown:**

1. **`def _robust_z(x: np.ndarray) -> np.ndarray:`**
   - Takes NumPy array, returns robust z-scores

2. **`x = np.asarray(x, dtype=float)`**
   - Converts to NumPy float array

3. **`med = np.nanmedian(x)`**
   - Computes median (ignoring NaN values)

4. **`mad = np.nanmedian(np.abs(x - med))`**
   - Computes MAD: median of absolute deviations from median
   - More robust than standard deviation

5. **`if not np.isfinite(mad) or mad == 0:`**
   - Checks if MAD is invalid (NaN, inf, or zero)

6. **`return np.zeros_like(x)`**
   - Returns zeros if MAD is invalid (prevents division by zero)

7. **`return 0.6745 * (x - med) / mad`**
   - Computes robust z-score
   - Factor 0.6745 scales MAD to match standard deviation for normal distributions
   - Formula: z = 0.6745 × (x - median) / MAD

**Why robust z-score?**
- Regular z-score uses mean and std, which are sensitive to outliers
- Robust z-score uses median and MAD, which ignore outliers
- Better for detecting outliers in data that already has outliers

**Example:**
- Data: `[50, 51, 52, 53, 100]` (100 is an outlier)
- Regular z-score for 100: Very high (outlier affects mean/std)
- Robust z-score for 100: Still high, but median/MAD ignore it in calculation

---

### Function 6: `_save_plot()` (Lines 95-99)

```python
def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
```

**What it does:**

Saves the current matplotlib figure to a file and closes it. Creates output directory if needed.

**Line-by-line breakdown:**

1. **`def _save_plot(path: Path) -> None:`**
   - Takes a Path object for output file

2. **`path.parent.mkdir(parents=True, exist_ok=True)`**
   - Creates parent directory if it doesn't exist
   - `parents=True`: Creates all parent directories
   - `exist_ok=True`: Doesn't error if directory exists

3. **`plt.tight_layout()`**
   - Adjusts plot layout to prevent label cutoff

4. **`plt.savefig(path, dpi=160)`**
   - Saves figure to file
   - `dpi=160`: Resolution (dots per inch) for good quality

5. **`plt.close()`**
   - Closes figure to free memory

---

## Section 3: Main Pipeline Function (Lines 106-217)

### Function: `validate_and_clean()` (Lines 106-217)

This is the core function that validates, cleans, and flags issues in the data.

```python
def validate_and_clean(
    infile: Path,
    outdir: Path,
    drop_outliers: bool,
    z_thresh: float,
    delta_thresh: Optional[float],
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, float]]:
```

**Parameters:**
- `infile`: Path to input CSV file
- `outdir`: Directory for output files
- `drop_outliers`: If True, removes flagged rows from output
- `z_thresh`: Threshold for robust z-score outlier detection (default: 6.0)
- `delta_thresh`: Optional threshold for spike detection (absolute change in B_total)

**Returns:**
- `df_clean`: Cleaned DataFrame with flag columns
- `notes`: Dictionary of diagnostic notes
- `stats`: Dictionary of statistics

**Line-by-line breakdown:**

**Initialization (Lines 113-114):**
```python
notes: Dict[str, str] = {}
stats: Dict[str, float] = {}
```
- Creates empty dictionaries to store notes and statistics

**Loading and Normalization (Lines 116-117):**
```python
df = pd.read_csv(infile)
df = _normalize_columns(df)
```
- Loads CSV into DataFrame
- Normalizes column names (strips whitespace)

**Required Columns Check (Lines 119-121):**
```python
if not all(c in df.columns for c in ["x", "y"]):
    raise ValueError(f"CSV must contain columns x and y. Found: {list(df.columns)}")
```
- Ensures required spatial columns exist
- Raises error if missing

**Coordinate Conversion (Lines 123-125):**
```python
df["x"] = pd.to_numeric(df["x"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
```
- Converts x, y to numeric
- Invalid values become NaN

**B_total Handling (Lines 127-133):**
```python
df, bnote = _compute_btotal_if_missing(df)
notes["B_total"] = bnote
if "B_total" not in df.columns:
    raise ValueError("B_total is missing and could not be computed from Bx,By,Bz.")
df["B_total"] = pd.to_numeric(df["B_total"], errors="coerce")
```
- Computes B_total if missing
- Records note about B_total
- Ensures B_total exists and is numeric

**Time Parsing (Lines 135-143):**
```python
time_col = _find_time_column(list(df.columns))
notes["time_col"] = time_col if time_col else "None"
t_series, tnote = _coerce_time_series(df[time_col] if time_col else None)
notes["time_parse"] = tnote
if t_series is not None:
    df["_time_utc"] = t_series
else:
    df["_time_utc"] = pd.NaT
```
- Finds time column
- Parses time series
- Stores as `_time_utc` column (internal use)

**Drop Missing Values (Lines 145-151):**
```python
before = len(df)
df_clean = df.dropna(subset=["x", "y", "B_total"]).copy()
dropped_na = before - len(df_clean)
stats["rows_total"] = float(before)
stats["rows_dropped_nan"] = float(dropped_na)
stats["rows_after_nan_drop"] = float(len(df_clean))
```
- Removes rows with missing x, y, or B_total
- Records statistics about dropped rows

**Basic Statistics (Lines 153-161):**
```python
stats["x_min"] = float(df_clean["x"].min())
stats["x_max"] = float(df_clean["x"].max())
stats["y_min"] = float(df_clean["y"].min())
stats["y_max"] = float(df_clean["y"].max())
stats["B_total_min"] = float(df_clean["B_total"].min())
stats["B_total_max"] = float(df_clean["B_total"].max())
stats["B_total_mean"] = float(df_clean["B_total"].mean())
stats["B_total_std"] = float(df_clean["B_total"].std(ddof=1)) if len(df_clean) > 1 else float("nan")
```
- Computes min, max, mean, std for coordinates and B_total
- Records in stats dictionary

**Sample Rate Estimation (Lines 163-178):**
```python
if df_clean["_time_utc"].notna().sum() > 5:
    t = df_clean["_time_utc"].sort_values()
    dt = t.diff().dt.total_seconds().dropna()
    dt = dt[dt > 0]
    if len(dt) > 3:
        stats["dt_median_s"] = float(dt.median())
        stats["dt_mean_s"] = float(dt.mean())
        stats["sample_rate_hz_est"] = float(1.0 / dt.median())
        stats["dt_cv"] = float(dt.std(ddof=1) / dt.mean()) if dt.mean() > 0 else float("nan")
```
- If time column exists, estimates sampling rate
- Computes time differences between measurements
- Calculates median/mean interval and sample rate
- Computes coefficient of variation (jitter measure)

**Example:** Sampling-rate estimation from timestamps (expected values)

Assume the logger records UTC timestamps while averaging 100 samples per grid point
with a loop delay of ~0.01 s (≈100 Hz).

Example parsed timestamps:
  `2025-12-16T15:00:00.000Z`
  `2025-12-16T15:00:00.010Z`
  `2025-12-16T15:00:00.020Z`
  `2025-12-16T15:00:00.030Z`
  `2025-12-16T15:00:00.041Z`

Computed time differences (dt, seconds):
  `[0.010, 0.010, 0.010, 0.011]`

Median time step:
  `dt_median_s ≈ 0.010 s`

Estimated sampling rate:
  `sample_rate_hz_est = 1 / dt_median_s ≈ 100 Hz`

Mean time step:
  `dt_mean_s ≈ 0.01025 s`

Timing jitter (coefficient of variation):
  `dt_cv ≈ 0.03 – 0.06`

**Interpretation:**
- Sampling rate is close to the intended 100 Hz
- Low dt_cv indicates stable timing with minor OS-level jitter
- Values in this range indicate healthy data acquisition

If fewer than ~80% of timestamps are valid or fewer than ~6 samples exist,
sampling-rate estimation is skipped to avoid misleading results.

**Outlier Detection (Lines 180-182):**
```python
z = _robust_z(df_clean["B_total"].to_numpy())
outlier_mask = np.abs(z) > z_thresh
```
- Computes robust z-scores for B_total
- Flags points where |z| > threshold (default: 6.0)

**Example:** 
Robust outlier detection using median-based z-scores

Assume a grid test produces the following B_total values (gauss)
after averaging 100 samples per grid point:

B_total:
- `[0.987, 0.986, 0.988, 0.987, 0.989, 1.042, 0.986, 0.987]`

Most points cluster near ~0.987 gauss, but one point (1.042 gauss)
is unusually large.

Step 1: Compute robust z-scores

The script computes a robust z-score for each value using:
- median instead of mean
- MAD (median absolute deviation) instead of standard deviation

Resulting robust z-scores (approximate):

z:
- `[-0.2, -0.5,  0.3, -0.2,  0.7,  9.1, -0.5, -0.2]`

Step 2: Apply outlier threshold

Using a threshold of:
- `z_thresh = 6.0`

The outlier mask becomes:

outlier_mask:
- `[False, False, False, False, False, True, False, False]`

Only the point with z ≈ 9.1 is flagged as an outlier.

Interpretation:
- Normal grid points fall within ±1 robust z-unit
- The extreme value is >6 robust z-units from the median
- This point is flagged for inspection as a potential sensor glitch
  or unusually strong magnetic disturbance

Important:
These outliers are flagged, not automatically removed.
They can later be excluded, re-measured, or analyzed as possible
true magnetic anomalies depending on spatial context.

**Spike Detection (Lines 184-195):**
```python
if delta_thresh is not None:
    if df_clean["_time_utc"].notna().sum() > 5:
        df_ord = df_clean.sort_values("_time_utc").copy()
    else:
        df_ord = df_clean.copy()
    d = np.abs(np.diff(df_ord["B_total"].to_numpy(), prepend=df_ord["B_total"].iloc[0]))
    spike_mask_ord = d > delta_thresh
    spike_mask = pd.Series(spike_mask_ord, index=df_ord.index).reindex(df_clean.index).fillna(False).to_numpy(bool)
else:
    spike_mask = np.zeros(len(df_clean), dtype=bool)
```
**Example:** 
Spike detection using per-sample |ΔB_total| (delta threshold)

Goal:
Flag sudden jumps between consecutive measurements that are likely glitches
(e.g., I2C hiccup, electrical interference, dropped/duplicated sample).

Assume B_total values (gauss) ordered by time:

B_total (time-ordered):
- `[0.9870, 0.9872, 0.9869, 1.0200, 0.9871, 0.9870]`

Step 1: Compute per-sample absolute differences

The code computes:
- `d[i] = |B_total[i] - B_total[i-1]|`

Using prepend (first element uses itself):
d:
- `[0.0000, 0.0002, 0.0003, 0.0331, 0.0329, 0.0001]`

Step 2: Apply spike threshold

Choose a threshold (example):
- `delta_thresh = 0.010 gauss`

Then:
- `spike_mask_ord = d > delta_thresh`

spike_mask_ord:
- `[False, False, False, True, True, False]`

Interpretation:
- The jump from 0.9869 → 1.0200 creates a very large |ΔB_total| (0.0331),
  so the 1.0200 point is flagged as a spike.
- The next point (1.0200 → 0.9871) also creates a large |ΔB_total| (0.0329),
  so that next point is flagged too.
This is expected: a single “glitch spike” often causes two large deltas
(one going up, one coming back down).

Time ordering vs row ordering:
- If enough valid timestamps exist, the script sorts by _time_utc before
  computing deltas (so “consecutive” means consecutive in time).
- If timestamps are missing/unusable, it uses the existing row order.

Mapping back to original rows:
- spike_mask_ord is computed in sorted order.
- It is then reindexed back to the original df_clean row order so the flags
  line up with the original dataset rows.

If delta_thresh is not set:
- spike_mask is all False (no spike detection is applied).

**Combine Flags (Lines 197-203):**
```python
flagged = outlier_mask | spike_mask
stats["rows_flagged_outlier_or_spike"] = float(flagged.sum())
stats["rows_flagged_pct"] = float(100.0 * flagged.mean()) if len(flagged) else 0.0
df_clean["_flag_outlier"] = outlier_mask
df_clean["_flag_spike"] = spike_mask
df_clean["_flag_any"] = flagged
```
- Combines outlier and spike flags
- Records statistics
- Adds flag columns to DataFrame

**Optionally Drop Flagged Rows (Lines 205-213):**
```python
if drop_outliers and len(df_clean) > 0:
    before2 = len(df_clean)
    df_clean = df_clean.loc[~df_clean["_flag_any"]].copy()
    stats["rows_dropped_flagged"] = float(before2 - len(df_clean))
    stats["rows_after_flag_drop"] = float(len(df_clean))
else:
    stats["rows_dropped_flagged"] = 0.0
    stats["rows_after_flag_drop"] = float(len(df_clean))
```
- If `drop_outliers` flag set, removes flagged rows
- Records statistics about dropped rows

**Return (Lines 216-217):**
```python
outdir.mkdir(parents=True, exist_ok=True)
return df_clean, notes, stats
```
- Creates output directory
- Returns cleaned DataFrame, notes, and statistics

**validate_and_clean(infile, outdir, drop_outliers, z_thresh, delta_thresh)**

SUMMARY (what this function does)
1) Load CSV into a DataFrame (df) and normalize column names.
2) Verify required spatial columns exist: x and y.
3) Ensure B_total exists:
   - If B_total is present, keep it.
   - Else if Bx, By, Bz exist, compute:
       B_total = sqrt(Bx^2 + By^2 + Bz^2)
   - Else raise an error (cannot continue without B_total).
4) Try to detect and parse a time column (timestamp/time/etc.) into _time_utc:
   - If numeric: attempt unix-seconds parsing.
   - Else: attempt general datetime parsing.
   - If < ~80% timestamps parse, disable time-based diagnostics (store NaT).
5) Coerce x, y, B_total into numeric and drop rows with NaNs in any of these.
6) Compute summary stats for reporting:
   - row counts (total, dropped NaNs, remaining)
   - x/y min/max
   - B_total min/max/mean/std
   - (if time usable) dt_median, dt_mean, estimated sample rate (Hz), dt jitter (CV)
7) Flag suspicious rows (DOES NOT necessarily delete them):
   A. Outlier flag (global):
      - Compute robust z-scores for B_total using median + MAD
      - _flag_outlier = |z| > z_thresh
   B. Spike flag (time-adjacent jump):
      - If delta_thresh set: compute d[i] = |B_total[i] - B_total[i-1]|
        using time order if available (otherwise row order)
      - _flag_spike = d > delta_thresh
   C. Combined flag:
      - _flag_any = _flag_outlier OR _flag_spike
   - Store counts + percentage flagged in stats.
8) If drop_outliers=True:
   - Remove rows where _flag_any is True.
   - Otherwise keep flagged rows and only label them.
9) Ensure outdir exists, then return:
   - df_clean (cleaned dataframe + flag columns)
   - notes (human-readable parsing notes)
   - stats (numeric summary report values)


**BIG EXAMPLE** (what it looks like on a real run)
Input file: data/raw/mag_data.csv
Columns: time, x, y, Bx, By, Bz, units   (B_total missing)

Example rows (simplified):
  time                         x    y     Bx     By     Bz
  2025-12-16T15:00:00.000Z     0.0  0.0   0.12  -0.03  0.98
  2025-12-16T15:00:00.010Z     0.2  0.0   0.11  -0.02  0.97
  2025-12-16T15:00:00.020Z     0.4  0.0   NaN    0.01  0.96   <-- bad row (NaN)
  2025-12-16T15:00:00.030Z     0.6  0.0   0.10  -0.04  1.15   <-- possible spike/outlier

What validate_and_clean does:
- Computes B_total for each row:
    row1 B_total = sqrt(0.12^2 + (-0.03)^2 + 0.98^2) ≈ 0.9877
    row2 B_total ≈ 0.9763
    row3 B_total = NaN (because Bx is NaN)
    row4 B_total ≈ 1.1550
- Drops row3 because x/y/B_total contains NaN
- Parses time into _time_utc and estimates dt:
    dt values ≈ [0.010, 0.010, 0.010] seconds
    sample_rate_hz_est ≈ 100 Hz
- Flags suspicious data:
    _flag_outlier: robust z-score detects row4 is far from median -> True
    _flag_spike: if delta_thresh=0.05 gauss, the jump into row4 may be >0.05 -> True
    _flag_any = True if either flag is True
- Writes output-ready dataframe with added columns:
    time, x, y, Bx, By, Bz, B_total, units, _flag_outlier, _flag_spike, _flag_any
- Returns (df_clean, notes, stats) so the caller can:
    - save df_clean as <stem>_clean.csv
    - save stats/notes as <stem>_report.txt
    - produce diagnostic plots

---

## Section 4: Plotting Function (Lines 220-268)

### Function: `make_plots()` (Lines 220-268)

Generates diagnostic plots for quick visual inspection of data quality.

```python
def make_plots(df: pd.DataFrame, outbase: Path) -> None:
    """
    Generate a small set of plots for fast sanity-checking.
    outbase is like: outdir/<stem>
    """
```

**Plot 1: B_total vs Time (Lines 225-233)**
```python
if df["_time_utc"].notna().sum() > 5:
    df_t = df.sort_values("_time_utc")
    plt.figure()
    plt.plot(df_t["_time_utc"], df_t["B_total"])
    plt.xlabel("time (UTC)")
    plt.ylabel("B_total")
    plt.title("B_total over time")
    _save_plot(outbase.with_name(outbase.name + "_Btotal_vs_time.png"))
```
- Creates time series plot if time column exists
- Shows B_total values over time
- Helps identify trends, drifts, or sudden changes

**Example Plot 1: B_total vs Time**
```
B_total (gauss)
    |
0.99|     ●
    |   ●   ●
0.98|  ●       ●
    |             ●
0.97|                ●
    |                   ●
0.96|                      ●
    |________________________
    15:00:00  15:00:05  15:00:10
            time (UTC)

Example data points:
- 15:00:00.000 → B_total = 0.9872 gauss
- 15:00:00.500 → B_total = 0.9885 gauss
- 15:00:01.000 → B_total = 0.9878 gauss
- 15:00:01.500 → B_total = 0.9869 gauss
- 15:00:02.000 → B_total = 0.9871 gauss
- 15:00:02.500 → B_total = 0.9875 gauss
```

**Plot 2: Histogram of B_total (Lines 235-241)**
```python
plt.figure()
plt.hist(df["B_total"].to_numpy(), bins=60)
plt.xlabel("B_total")
plt.ylabel("count")
plt.title("Histogram: B_total")
_save_plot(outbase.with_name(outbase.name + "_Btotal_hist.png"))
```
- Creates histogram of B_total distribution
- Shows data distribution and potential outliers
- 60 bins for good resolution

**Example Plot 2: Histogram of B_total**
```
count
  |
25|     █
  |     █
20|     █
  |     █
15|     █  █
  |     █  █
10|  █  █  █
  |  █  █  █
 5|  █  █  █
  |  █  █  █        █
 0|__█__█__█________█________█_
   0.95 0.96 0.97 0.98 0.99 1.00
            B_total (gauss)

Example distribution:
- Most values cluster around 0.987 gauss (peak at ~22 counts)
- Few values below 0.96 gauss (outliers)
- Few values above 0.99 gauss (outliers)
- Mean ≈ 0.987 gauss, Std ≈ 0.003 gauss
```

**Plot 3: XY Scatter Colored by B_total (Lines 243-250)**
```python
plt.figure()
sc = plt.scatter(df["x"], df["y"], c=df["B_total"], s=14)
plt.xlabel("x")
plt.ylabel("y")
plt.title("XY scatter colored by B_total")
plt.colorbar(sc, label="B_total")
_save_plot(outbase.with_name(outbase.name + "_scatter_xy_colored.png"))
```
- Creates spatial scatter plot
- Color represents B_total value
- Helps visualize spatial patterns

**Example Plot 3: XY Scatter Colored by B_total**
```
y (meters)
  |
2.0|  ●(0.987)  ●(0.988)  ●(0.987)
    |
1.5|  ●(0.986)  ●(0.987)  ●(0.988)
    |
1.0|  ●(0.987)  ●(0.987)  ●(0.987)
    |
0.5|  ●(0.988)  ●(0.986)  ●(0.987)
    |
0.0|_____________________________
   0.0  0.5  1.0  1.5  2.0
            x (meters)

Color scale (colorbar):
0.985 gauss (blue/cool)  →  0.987 gauss (green)  →  0.989 gauss (red/warm)

Example data points:
- (0.0, 0.0) → B_total = 0.9872 gauss
- (0.5, 0.0) → B_total = 0.9868 gauss
- (1.0, 0.0) → B_total = 0.9875 gauss
- (0.0, 0.5) → B_total = 0.9881 gauss
- (1.0, 1.0) → B_total = 0.9873 gauss
```

**Plot 4: Spike Deltas (Lines 252-267)**
```python
if df["_time_utc"].notna().sum() > 5:
    df_t = df.sort_values("_time_utc")
    series = df_t["B_total"].to_numpy()
    xaxis = df_t["_time_utc"]
else:
    series = df["B_total"].to_numpy()
    xaxis = np.arange(len(df))

deltas = np.abs(np.diff(series, prepend=series[0]))
plt.figure()
plt.plot(xaxis, deltas)
plt.xlabel("time (UTC)" if df["_time_utc"].notna().sum() > 5 else "row index")
plt.ylabel("|Δ B_total|")
plt.title("Per-sample |ΔB_total| (spike check)")
_save_plot(outbase.with_name(outbase.name + "_spike_deltas.png"))
```
- Plots absolute change in B_total between consecutive measurements
- Helps identify spikes and sudden changes
- Uses time axis if available, otherwise row index

**Example Plot 4: Spike Deltas**
```
|Δ B_total| (gauss)
    |
0.04|                    ●
    |                    |
0.03|                    |
    |                    |
0.02|                    |
    |                    |
0.01|  ●   ●   ●   ●   ●
    |  ●   ●   ●   ●   ●
0.00|________________________
    15:00:00  15:00:05  15:00:10
            time (UTC)

Example data:
Time          B_total    |Δ B_total|
15:00:00.000  0.9872     0.0000 (first point)
15:00:00.500  0.9885     0.0013
15:00:01.000  0.9878     0.0007
15:00:01.500  0.9869     0.0009
15:00:02.000  0.9871     0.0002
15:00:02.500  1.0200     0.0329 ← SPIKE! (exceeds threshold)
15:00:03.000  0.9875     0.0325 ← large delta (returning to normal)

Spike threshold example: delta_thresh = 0.010 gauss
- Points with |Δ B_total| > 0.010 would be flagged as spikes
- In this example, the spike at 15:00:02.500 is clearly visible
```

---

## Section 5: Report Writing Function (Lines 270-287)

### Function: `write_report()` (Lines 270-287)

Writes a text report with notes and statistics.

```python
def write_report(notes: Dict[str, str], stats: Dict[str, float], report_path: Path, infile: Path) -> None:
    lines = []
    lines.append(f"Validate + Diagnostics Report")
    lines.append(f"Input file: {infile}")
    lines.append("")
    lines.append("NOTES")
    for k, v in notes.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("STATS")
    for k in sorted(stats.keys()):
        v = stats[k]
        if isinstance(v, float) and np.isfinite(v):
            lines.append(f"  - {k}: {v:.6g}")
        else:
            lines.append(f"  - {k}: {v}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
```

**What it does:**

1. Creates list of report lines
2. Adds header with input file name
3. Adds NOTES section with diagnostic notes
4. Adds STATS section with sorted statistics
5. Formats numbers to 6 significant digits
6. Creates output directory if needed
7. Writes report to file

**Example report:**
```
Validate + Diagnostics Report
Input file: data/raw/mag_data.csv

NOTES
  - B_total: B_total present
  - time_col: time
  - time_parse: Parsed as datetime (UTC)
  - sample_rate: 1.0 Hz

STATS
  - B_total_max: 52.3456
  - B_total_mean: 51.9876
  - B_total_min: 51.1234
  - rows_after_nan_drop: 81
  - rows_total: 81
  ...
```

---

## Section 6: Command-Line Arguments (Lines 290-302)

### Function: `parse_args()` (Lines 290-302)

Parses command-line arguments using argparse.

```python
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate, clean, and generate diagnostics for magnetometer CSV.")
    p.add_argument("--in", dest="infile", required=True, help="Input CSV path (e.g., data/raw/mag_data.csv)")
    p.add_argument("--outdir", default="data/processed", help="Output directory (default: data/processed)")
    p.add_argument("--drop-outliers", action="store_true", help="If set, drop rows flagged as outlier/spike")
    p.add_argument("--z-thresh", type=float, default=6.0, help="Robust z-score threshold for B_total outliers (default: 6.0)")
    p.add_argument(
        "--delta-thresh",
        type=float,
        default=None,
        help="Optional absolute per-sample |ΔB_total| threshold to flag spikes (units same as B_total).",
    )
    return p.parse_args()
```

**Arguments:**
- `--in`: Required input CSV file path
- `--outdir`: Output directory (default: `data/processed`)
- `--drop-outliers`: Flag to remove outliers from output
- `--z-thresh`: Z-score threshold for outliers (default: 6.0)
- `--delta-thresh`: Optional threshold for spike detection

**Example usage:**
```bash
python3 validate_and_diagnosticsV1.py --in data/raw/mag_data.csv
python3 validate_and_diagnosticsV1.py --in data/raw/mag_data.csv --drop-outliers --z-thresh 5.0
python3 validate_and_diagnosticsV1.py --in data/raw/mag_data.csv --delta-thresh 2.0
```

---

## Section 7: Main Function (Lines 305-343)

### Function: `main()` (Lines 305-343)

Main entry point that orchestrates the validation pipeline.

```python
def main() -> int:
    args = parse_args()
    infile = Path(args.infile)
    outdir = Path(args.outdir)

    if not infile.exists():
        print(f"ERROR: input file not found: {infile}", file=sys.stderr)
        return 2

    stem = infile.stem
    clean_path = outdir / f"{stem}_clean.csv"
    report_path = outdir / f"{stem}_report.txt"
    outbase = outdir / stem  # used for naming plots

    try:
        df_clean, notes, stats = validate_and_clean(
            infile=infile,
            outdir=outdir,
            drop_outliers=args.drop_outliers,
            z_thresh=args.z_thresh,
            delta_thresh=args.delta_thresh,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Save cleaned
    df_clean.to_csv(clean_path, index=False)

    # Plots
    make_plots(df_clean, outbase)

    # Report
    write_report(notes, stats, report_path, infile)

    print(f"Wrote cleaned CSV: {clean_path}")
    print(f"Wrote report:      {report_path}")
    print(f"Wrote plots with prefix: {outbase}_*.png")
    return 0
```

**Line-by-line breakdown:**

1. **`args = parse_args()`**
   - Parses command-line arguments

2. **`infile = Path(args.infile)`** and **`outdir = Path(args.outdir)`**
   - Converts to Path objects

3. **`if not infile.exists():`**
   - Checks if input file exists

4. **`return 2`**
   - Returns error code if file not found

5. **`stem = infile.stem`**
   - Gets filename without extension (e.g., "mag_data" from "mag_data.csv")

6. **`clean_path = outdir / f"{stem}_clean.csv"`**
   - Constructs output file paths

7. **`try:`** block
   - Calls `validate_and_clean()` with error handling

8. **`df_clean.to_csv(clean_path, index=False)`**
   - Saves cleaned CSV

9. **`make_plots(df_clean, outbase)`**
   - Generates diagnostic plots

10. **`write_report(notes, stats, report_path, infile)`**
    - Writes text report

11. **Print success messages**
    - Shows what files were created

12. **`return 0`**
    - Returns success code

---

## Section 8: Script Entry Point (Lines 346-347)

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

**What it does:**

Runs the main function when script is executed directly. `SystemExit` ensures proper exit code handling.

---

## Key Concepts

### Outlier Detection

**Robust z-score:**
- Uses median and MAD instead of mean and std
- More resistant to outliers in the calculation itself
- Threshold of 6.0 means values >6 standard deviations from median are flagged

**Why robust?**
- Regular z-score: Outliers affect mean/std, making detection less reliable
- Robust z-score: Outliers don't affect median/MAD, making detection more reliable

### Spike Detection

**Delta threshold:**
- Flags sudden changes between consecutive measurements
- Useful for detecting sensor glitches or movement artifacts
- Requires `--delta-thresh` argument to enable

**Example:**
- If B_total changes by >2.0 gauss between measurements, flag as spike
- Helps identify measurement errors or sensor issues

### Data Cleaning

**Missing values:**
- Rows with missing x, y, or B_total are dropped
- Other missing values are preserved (converted to NaN)

**Flagged rows:**
- Outliers and spikes are flagged but not necessarily removed
- Use `--drop-outliers` to remove them from output
- Flag columns added: `_flag_outlier`, `_flag_spike`, `_flag_any`

### Output Files

1. **`<stem>_clean.csv`**: Cleaned data with flag columns
2. **`<stem>_report.txt`**: Text report with statistics
3. **`<stem>_Btotal_vs_time.png`**: Time series plot
4. **`<stem>_Btotal_hist.png`**: Histogram
5. **`<stem>_scatter_xy_colored.png`**: Spatial scatter plot
6. **`<stem>_spike_deltas.png`**: Spike detection plot

---

## Summary: The Complete Workflow

1. **Parse arguments** → Get input file and options
2. **Load CSV** → Read data into DataFrame
3. **Validate** → Check required columns (x, y, B_total)
4. **Clean** → Remove missing values, compute B_total if needed
5. **Parse time** → Convert time column to datetime if present
6. **Compute statistics** → Min, max, mean, std, sample rate
7. **Detect outliers** → Robust z-score on B_total
8. **Detect spikes** → Delta threshold if provided
9. **Flag rows** → Add flag columns to DataFrame
10. **Optionally drop** → Remove flagged rows if `--drop-outliers` set
11. **Save cleaned CSV** → Write output with flags
12. **Generate plots** → Create diagnostic visualizations
13. **Write report** → Create text report with notes and stats

---

## Tips for Usage

1. **First run:** Use default settings to see what issues exist
2. **Adjust thresholds:** Tune `--z-thresh` based on your data distribution
3. **Spike detection:** Use `--delta-thresh` if you expect sudden changes
4. **Review flags:** Check `_flag_any` column before dropping rows
5. **Check plots:** Visual inspection helps understand data quality
6. **Read report:** Statistics help understand data characteristics

---

## Integration with Other Scripts

This script is typically used:
- **After data collection:** Run on raw CSV from `mag_to_csv.py`
- **Before analysis:** Clean data before `compute_local_anomaly_v1.py`
- **Quality control:** Regular validation during data collection

**Typical workflow:**
```
mag_to_csv.py → mag_data.csv → validate_and_diagnosticsV1.py → mag_data_clean.csv → compute_local_anomaly_v1.py
```

