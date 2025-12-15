# Complete Explanation of `mag_to_csv.py`

This document explains every part of the magnetometer data collection script, step by step.

---

## Overview

This script collects magnetic field measurements from an MMC5983MA magnetometer sensor and saves them to a CSV file. It operates in **auto-grid mode**, where the script automatically generates a grid of measurement points based on configuration settings. At each grid point, you move the sensor to the location and press Enter to capture. The script takes multiple samples, averages them for accuracy, and records the magnetic field components (Bx, By, Bz) along with the total magnitude (B_total).

---

## Section 1: Imports and Dependencies (Lines 1-9)

```python
#!/usr/bin/env python3
import os
import csv
import time
import math
from datetime import datetime, timezone

import qwiic_mmc5983ma
```

**What it does:**

1. **`#!/usr/bin/env python3`** (Shebang)
   - Makes the script executable directly from the command line

2. **Standard library imports:**
   - `os`: File system operations
   - `csv`: Reading and writing CSV files
   - `time`: Adding delays between sensor readings
   - `math`: Mathematical operations (square root for calculating B_total)
   - `datetime`: Generating timestamps

3. **External library:**
   - `qwiic_mmc5983ma`: SparkFun's Python library for the MMC5983MA magnetometer sensor

**Note:** Install with: `pip install sparkfun-qwiic-mmc5983ma`

---

## Section 2: Configuration Constants (Lines 11-25)

```python
CSV_PATH = "mag_data.csv"

# Point-capture settings (tweak anytime)
SAMPLES_PER_POINT = 100        # how many samples to average per grid point
SAMPLE_DELAY_S = 0.01          # delay between samples (0.01s = ~100 Hz loop)

# ---- GRID SETTINGS ----
# 5x5 ft ≈ 1.52 m. With 0.20 m spacing, use 9 points per side (~1.60 m span).
DX = 0.20   # meters between points in x
DY = 0.20   # meters between points in y
NX = 9      # number of points in x direction
NY = 9      # number of points in y direction
X0 = 0.0    # starting x (meters)
Y0 = 0.0    # starting y (meters)
# ------------------------------------------
```

**What it does:**

**Line-by-line breakdown:**

1. **`CSV_PATH = "mag_data.csv"`**
   - Output file where all measurements will be saved
   - If file doesn't exist, it will be created automatically
   - If file exists, new measurements will be appended

2. **`SAMPLES_PER_POINT = 100`**
   - Number of sensor readings to take at each measurement point
   - These readings are averaged together to reduce noise
   - More samples = better accuracy but slower collection

3. **`SAMPLE_DELAY_S = 0.01`**
   - Delay in seconds between each sample
   - `0.01` seconds = 10 milliseconds = ~100 samples per second
   - Gives the sensor time to take a new reading

4. **`DX = 0.20`**
   - Spacing between measurement points in the X direction (meters)
   - 0.20 m = 20 cm spacing

5. **`DY = 0.20`**
   - Spacing between measurement points in the Y direction (meters)
   - 0.20 m = 20 cm spacing

6. **`NX = 9`**
   - Number of points in the X direction
   - Total grid width = (NX - 1) × DX = 8 × 0.20 = 1.60 meters

7. **`NY = 9`**
   - Number of points in the Y direction
   - Total grid height = (NY - 1) × DY = 8 × 0.20 = 1.60 meters

8. **`X0 = 0.0`**
   - Starting X coordinate (meters)
   - First point will be at X0

9. **`Y0 = 0.0`**
   - Starting Y coordinate (meters)
   - First point will be at Y0

**Grid calculation:**
- Total points = NX × NY = 9 × 9 = 81 points
- Grid spans from (X0, Y0) to (X0 + (NX-1)×DX, Y0 + (NY-1)×DY)
- With default settings: (0.0, 0.0) to (1.60, 1.60) meters

**Why average?** Single readings can be noisy; averaging improves accuracy. More samples = better accuracy but slower collection.

---

## Section 3: Utility Functions

### Function 1: `utc_iso()` (Lines 27-28)

```python
def utc_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
```

**What it does:**

Generates a timestamp in ISO 8601 format with millisecond precision, using UTC timezone.

**Line-by-line breakdown:**

1. **`def utc_iso():`**
   - Defines function with no parameters

2. **`return datetime.now(timezone.utc).isoformat(timespec="milliseconds")`**
   - `datetime.now(timezone.utc)`: Gets current time in UTC timezone
   - `.isoformat(timespec="milliseconds")`: Formats as ISO 8601 string with milliseconds
   - Returns formatted timestamp string

**Example output:** `"2024-03-15T14:23:45.123Z"`

**Why UTC?**
- Consistent timezone regardless of where you run the script
- Important for data analysis across different locations/timezones
- Standard format for scientific data logging

**Why milliseconds?**
- Provides precise timing for each measurement
- Useful for analyzing data collection rate or sensor response time

---

### Function 2: `ensure_csv_header()` (Lines 31-57)

```python
def ensure_csv_header(path: str):
    """Create file + header if it doesn't exist or is empty."""
    try:
        needs_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    except (OSError, PermissionError) as e:
        raise RuntimeError(f"Cannot access file {path}: {e}")
    
    # Optional: Validate existing header
    if not needs_header:
        try:
            with open(path, "r") as f:
                first_line = f.readline().strip()
                expected = "time,x,y,Bx,By,Bz,B_total,units"
                if first_line != expected:
                    needs_header = True
        except (IOError, PermissionError) as e:
            raise RuntimeError(f"Cannot read file {path}: {e}")
    
    if needs_header:
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time", "x", "y", "Bx", "By", "Bz", "B_total", "units"])
        except (IOError, OSError, PermissionError) as e:
            raise RuntimeError(f"Cannot write to file {path}: {e}")
    return needs_header
```

**What it does:**

Ensures the CSV file exists and has the correct header row. If the file doesn't exist or is empty, it creates it with column headers.

**Line-by-line breakdown:**

1. **`def ensure_csv_header(path: str):`**
   - Defines function that takes file path as string parameter

2. **`try:`** (Line 33)
   - Starts error handling block for file access

3. **`needs_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)`**
   - `os.path.exists(path)`: Returns `True` if file exists, `False` otherwise
   - `not os.path.exists(path)`: `True` if file doesn't exist
   - `os.path.getsize(path)`: Returns file size in bytes
   - `(os.path.getsize(path) == 0)`: `True` if file is empty
   - `or`: Needs header if file doesn't exist OR file is empty

4. **`except (OSError, PermissionError) as e:`**
   - Catches file access errors (permission denied, etc.)
   - `as e`: Stores error in variable `e`

5. **`raise RuntimeError(f"Cannot access file {path}: {e}")`**
   - Raises new error with clear message including file path and original error

6. **`if not needs_header:`** (Line 39)
   - If header not needed (file exists and has content), validate it

7. **`with open(path, "r") as f:`**
   - Opens file in read mode
   - `with` statement ensures file is closed automatically

8. **`first_line = f.readline().strip()`**
   - Reads first line of file
   - `.strip()` removes leading/trailing whitespace

9. **`expected = "time,x,y,Bx,By,Bz,B_total,units"`**
   - Expected header format

10. **`if first_line != expected:`**
    - If header doesn't match, mark that header is needed

11. **`if needs_header:`** (Line 49)
    - If header is needed, create it

12. **`with open(path, "w", newline="") as f:`**
    - Opens file in write mode (overwrites if exists)
    - `newline=""`: Prevents extra line breaks (CSV best practice)

13. **`w = csv.writer(f)`**
    - Creates CSV writer object for the file

14. **`w.writerow(["time", "x", "y", "Bx", "By", "Bz", "B_total", "units"])`**
    - Writes header row with column names

15. **`return needs_header`**
    - Returns `True` if header was created, `False` otherwise

**CSV columns:**
- `time`: UTC timestamp
- `x`, `y`: Spatial coordinates
- `Bx`, `By`, `Bz`: Magnetic field components (gauss)
- `B_total`: Total magnitude = √(Bx² + By² + Bz²)
- `units`: Always "gauss"

**Example CSV file after header:**
```csv
time,x,y,Bx,By,Bz,B_total,units
```

**Example CSV file after measurements:**
```csv
time,x,y,Bx,By,Bz,B_total,units
2024-03-15T14:23:45.123Z,0.0,0.0,45.14,23.16,12.34,52.123456,gauss
2024-03-15T14:23:46.234Z,0.5,0.0,45.25,23.20,12.40,52.234567,gauss
2024-03-15T14:23:47.345Z,1.0,0.0,45.36,23.24,12.46,52.345678,gauss
```

---

### Function 3: `connect_sensor()` (Lines 60-67)

```python
def connect_sensor():
    mag = qwiic_mmc5983ma.QwiicMMC5983MA()
    if not mag.is_connected():
        raise RuntimeError(
            "MMC5983MA not detected on I2C. Check wiring/Qwiic HAT and that I2C is enabled."
        )
    mag.begin()
    return mag
```

**What it does:**

Initializes and connects to the MMC5983MA magnetometer sensor via I2C.

**Line-by-line breakdown:**

1. **`def connect_sensor():`**
   - Defines function with no parameters

2. **`mag = qwiic_mmc5983ma.QwiicMMC5983MA()`**
   - Creates sensor driver object (doesn't connect yet)

3. **`if not mag.is_connected():`**
   - `mag.is_connected()`: Checks if sensor is detected on I2C bus
   - Returns `True` if connected, `False` otherwise
   - `not`: Inverts the result

4. **`raise RuntimeError(...)`**
   - Raises error with troubleshooting message if sensor not found

5. **`mag.begin()`**
   - Initializes sensor for operation (configures settings)

6. **`return mag`**
   - Returns initialized sensor object for use in other functions

---

### Function 4: `read_avg_xyz_gauss()` (Lines 70-86)

```python
def read_avg_xyz_gauss(mag, n=SAMPLES_PER_POINT, delay_s=SAMPLE_DELAY_S):
    """Read N samples of (x,y,z) in gauss and return the averages."""
    sx = sy = sz = 0.0
    for i in range(n):
        try:
            x, y, z = mag.get_measurement_xyz_gauss()
            sx += x
            sy += y
            sz += z
        except Exception as e:
            raise RuntimeError(f"Failed to read sensor at sample {i+1}/{n}: {e}")
        if delay_s > 0:
            time.sleep(delay_s)
    ax = sx / n
    ay = sy / n
    az = sz / n
    return ax, ay, az
```

**What it does:**

Takes multiple sensor readings and returns the averaged values. This reduces noise and improves measurement accuracy.

**Line-by-line breakdown:**

1. **`def read_avg_xyz_gauss(mag, n=SAMPLES_PER_POINT, delay_s=SAMPLE_DELAY_S):`**
   - Function takes sensor object, number of samples (default 100), and delay between samples (default 0.01s)

2. **`sx = sy = sz = 0.0`** (Line 72)
   - Initializes accumulators for summing Bx, By, Bz values
   - All start at 0.0

3. **`for i in range(n):`** (Line 73)
   - Loops N times (once for each sample)
   - `i` is the loop counter (0 to n-1)

4. **`try:`** (Line 74)
   - Starts error handling for sensor read

5. **`x, y, z = mag.get_measurement_xyz_gauss()`** (Line 75)
   - Reads one measurement from sensor
   - Returns three values: (Bx, By, Bz) in gauss

6. **`sx += x`**, **`sy += y`**, **`sz += z`** (Lines 76-78)
   - Adds each component to its accumulator
   - `+=` is shorthand for `sx = sx + x`

7. **`except Exception as e:`** (Line 79)
   - Catches any error during sensor read

8. **`raise RuntimeError(f"Failed to read sensor at sample {i+1}/{n}: {e}")`** (Line 80)
   - Raises error with message showing which sample failed (1-indexed for user-friendly display)

9. **`if delay_s > 0:`** (Line 81)
   - Only sleep if delay is positive

10. **`time.sleep(delay_s)`** (Line 82)
    - Waits before next reading (gives sensor time to prepare)

11. **`ax = sx / n`**, **`ay = sy / n`**, **`az = sz / n`** (Lines 83-85)
    - Calculates averages by dividing sum by number of samples
    - Arithmetic mean

12. **`return ax, ay, az`** (Line 86)
    - Returns averaged (Bx, By, Bz) values as tuple

**Why average?** Sensor readings have random noise. Averaging reduces noise by √N (e.g., 100 samples = 10x less noise).

**Example:**
If `n = 5` and readings are:
- Sample 1: (45.1, 23.2, 12.3)
- Sample 2: (45.3, 23.0, 12.5)
- Sample 3: (45.0, 23.3, 12.2)
- Sample 4: (45.2, 23.1, 12.4)
- Sample 5: (45.1, 23.2, 12.3)

Then:
- `ax = (45.1 + 45.3 + 45.0 + 45.2 + 45.1) / 5 = 45.14`
- `ay = (23.2 + 23.0 + 23.3 + 23.1 + 23.2) / 5 = 23.16`
- `az = (12.3 + 12.5 + 12.2 + 12.4 + 12.3) / 5 = 12.34`

**Visual Example:**
```
Individual readings (noisy):
Sample 1: B_total = 52.3 gauss
Sample 2: B_total = 52.7 gauss
Sample 3: B_total = 51.9 gauss
Sample 4: B_total = 52.5 gauss
Sample 5: B_total = 52.1 gauss
... (95 more samples)

Averaged result: B_total = 52.3 gauss (much more stable!)
```

---

### Function 5: `append_row()` (Lines 89-94)

```python
def append_row(path, row):
    try:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)
    except (IOError, OSError, PermissionError) as e:
        raise RuntimeError(f"Cannot append to file {path}: {e}")
```

**What it does:**

Appends a single row of data to the CSV file.

**Line-by-line breakdown:**

1. **`def append_row(path, row):`**
   - Function takes file path and list of row data

2. **`try:`**
   - Starts error handling block

3. **`with open(path, "a", newline="") as f:`**
   - Opens file in append mode (`"a"` adds to end, doesn't overwrite)
   - `newline=""`: Prevents extra blank lines (CSV best practice)
   - `with` statement ensures file is closed automatically

4. **`csv.writer(f).writerow(row)`**
   - Creates CSV writer and writes one row
   - `row` should be a list matching header columns

5. **`except (IOError, OSError, PermissionError) as e:`**
   - Catches file write errors (disk full, permissions, etc.)

6. **`raise RuntimeError(f"Cannot append to file {path}: {e}")`**
   - Raises error with clear message including file path and original error

---

### Function 6: `beep()` (Lines 97-99)

```python
def beep():
    print("\a", end="", flush=True)  # ASCII bell
```

**What it does:**

Produces an audible beep sound to provide audio feedback after each measurement is saved. This is useful when you're focused on positioning the sensor and want confirmation that the data was successfully written without looking at the screen.

**Line-by-line breakdown:**

1. **`def beep():`**
   - Defines function with no parameters

2. **`print("\a", end="", flush=True)`**
   - `"\a"`: ASCII bell character (BEL, code 7) that triggers a beep sound
   - `end=""`: Prevents adding a newline after the beep
   - `flush=True`: Forces immediate output (ensures beep happens right away)
   - On Raspberry Pi, this typically produces a beep through the system speaker or audio output

**Usage:**
- Called after each successful measurement save
- Provides immediate audio feedback that data was written
- Helpful when you can't see the screen while positioning the sensor

**Note:** The beep sound depends on your system's audio configuration. On Raspberry Pi, you may need to enable the system beep or have audio output configured.

---

## Section 4: Main Function (Lines 101-157)

```python
def main():
    print("\n=== MMC5983MA -> CSV Logger (Point Capture Mode) ===")
    print(f"Output file: {CSV_PATH}")

    try:
        ensure_csv_header(CSV_PATH)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    try:
        mag = connect_sensor()
    except Exception as e:
        print(f"ERROR: {e}")
        return

    print(f"Auto-grid enabled: NX={NX}, NY={NY}, DX={DX} m, DY={DY} m")
    print("At each prompt, move the sensor to the point and press Enter.")
    print("Type 'q' then Enter to quit early.\n")

    for j in range(NY):
        for i in range(NX):
            x = X0 + i * DX
            y = Y0 + j * DY

            user = input(
                f"Point ({i+1}/{NX}, {j+1}/{NY}) -> x={x:.2f}, y={y:.2f}. "
                f"Press Enter to capture (or 'q' to quit): "
            ).strip()

            if user.lower() in ("q", "quit", "exit"):
                print("Done.")
                return

            print(f"  Sampling {SAMPLES_PER_POINT} readings...")
            try:
                bx, by, bz = read_avg_xyz_gauss(mag)
            except RuntimeError as e:
                print(f"  ERROR: {e}")
                print("  Skipping this measurement. You can re-run later for missing points.")
                continue

            b_total = math.sqrt(bx * bx + by * by + bz * bz)

            row = [utc_iso(), x, y, bx, by, bz, b_total, "gauss"]
            try:
                append_row(CSV_PATH, row)
            except RuntimeError as e:
                print(f"  ERROR: {e}")
                print("  Measurement taken but could not be saved!")
                print("  Check disk space and file permissions.")
                return

            print(f"  Saved: x={x:.2f}, y={y:.2f}, B_total={b_total:.6f} gauss\n")

    print("Grid complete. Done.")
```

**What it does:**

Runs the interactive data collection loop. Prompts user for coordinates, takes measurements, and saves to CSV.

**Line-by-line breakdown:**

**Initialization (Lines 102-115):**

1. **`print("\n=== MMC5983MA -> CSV Logger (Point Capture Mode) ===")`**
   - Prints welcome message with newline before it

2. **`print(f"Output file: {CSV_PATH}")`**
   - Shows output file path using f-string formatting

3. **`try:`** (Line 105)
   - Starts error handling for header creation

4. **`ensure_csv_header(CSV_PATH)`**
   - Ensures CSV file has proper header

5. **`except RuntimeError as e:`**
   - Catches file access errors

6. **`print(f"ERROR: {e}")`** and **`return`**
   - Prints error and exits function if header creation fails

7. **`try:`** (Line 111)
   - Starts error handling for sensor connection

8. **`mag = connect_sensor()`**
   - Connects to sensor and gets sensor object

9. **`except Exception as e:`** and **`return`**
   - If connection fails, prints error and exits

**Grid Setup (Lines 117-119):**

10. **`print(f"Auto-grid enabled: NX={NX}, NY={NY}, DX={DX} m, DY={DY} m")`**
    - Displays grid configuration to user
    - Shows number of points and spacing

11. **`print("At each prompt, move the sensor to the point and press Enter.")`**
    - Instructions for using auto-grid mode

12. **`print("Type 'q' then Enter to quit early.\n")`**
    - Instructions for early exit

**Grid Loop (Lines 121-155):**

13. **`for j in range(NY):`**
    - Outer loop: iterates through Y direction (rows)
    - `j` goes from 0 to NY-1

14. **`for i in range(NX):`**
    - Inner loop: iterates through X direction (columns)
    - `i` goes from 0 to NX-1
    - Processes points left-to-right, then moves to next row

15. **`x = X0 + i * DX`**
    - Calculates X coordinate for current grid point
    - Example: If X0=0.0, i=2, DX=0.20 → x = 0.0 + 2×0.20 = 0.40

16. **`y = Y0 + j * DY`**
    - Calculates Y coordinate for current grid point
    - Example: If Y0=0.0, j=3, DY=0.20 → y = 0.0 + 3×0.20 = 0.60

---

### Deep Dive: How the Nested For Loops Work

The nested for loops create a systematic grid traversal pattern. Let's break down exactly how they work with detailed examples and visual diagrams.

#### Understanding the Loop Structure

```python
for j in range(NY):        # Outer loop: rows (Y direction)
    for i in range(NX):    # Inner loop: columns (X direction)
        x = X0 + i * DX
        y = Y0 + j * DY
        # ... measurement code ...
```

**How nested loops work:**
- The **outer loop** (`j`) runs once for each row
- For each row, the **inner loop** (`i`) runs completely through all columns
- This creates a pattern: process all points in row 0, then all points in row 1, etc.

#### Example: Small 3×3 Grid

Let's trace through a small example with `NX=3`, `NY=3`, `DX=0.20`, `DY=0.20`, `X0=0.0`, `Y0=0.0`:

**Grid layout:**
```
Y
↑
0.40 ┌─────┬─────┬─────┐
     │ (2) │ (5) │ (8) │  j=2
0.20 ├─────┼─────┼─────┤
     │ (1) │ (4) │ (7) │  j=1
0.00 └─────┴─────┴─────┘
       (0)   (3)   (6)    j=0
     0.00  0.20  0.40    → X
          i=0   i=1   i=2
```

**Execution trace:**

| Step | j | i | x = X0 + i×DX | y = Y0 + j×DY | Point (x, y) | Order |
|------|---|---|---------------|---------------|--------------|-------|
| 1    | 0 | 0 | 0.0 + 0×0.20 = 0.00 | 0.0 + 0×0.20 = 0.00 | (0.00, 0.00) | 1st |
| 2    | 0 | 1 | 0.0 + 1×0.20 = 0.20 | 0.0 + 0×0.20 = 0.00 | (0.20, 0.00) | 2nd |
| 3    | 0 | 2 | 0.0 + 2×0.20 = 0.40 | 0.0 + 0×0.20 = 0.00 | (0.40, 0.00) | 3rd |
| 4    | 1 | 0 | 0.0 + 0×0.20 = 0.00 | 0.0 + 1×0.20 = 0.20 | (0.00, 0.20) | 4th |
| 5    | 1 | 1 | 0.0 + 1×0.20 = 0.20 | 0.0 + 1×0.20 = 0.20 | (0.20, 0.20) | 5th |
| 6    | 1 | 2 | 0.0 + 2×0.20 = 0.40 | 0.0 + 1×0.20 = 0.20 | (0.40, 0.20) | 6th |
| 7    | 2 | 0 | 0.0 + 0×0.20 = 0.00 | 0.0 + 2×0.20 = 0.40 | (0.00, 0.40) | 7th |
| 8    | 2 | 1 | 0.0 + 1×0.20 = 0.20 | 0.0 + 2×0.20 = 0.40 | (0.20, 0.40) | 8th |
| 9    | 2 | 2 | 0.0 + 2×0.20 = 0.40 | 0.0 + 2×0.20 = 0.40 | (0.40, 0.40) | 9th |

#### Why This Order?

**Row-by-row traversal (left-to-right, bottom-to-top):**
- **Systematic:** Easy to follow physically
- **Predictable:** Always know where you are
- **Efficient:** Minimizes movement if you're walking the grid
- **Progress tracking:** Easy to show "Point (i/NX, j/NY)"

**Alternative orders (not used in this script):**
- Column-by-column: Would require swapping the loops
- Spiral: Would require more complex logic
- Random: Would require shuffling

#### Visual Summary: 9×9 Grid Traversal

```
Final grid with measurement order (showing every 3rd point for clarity):

Y
↑
     ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
1.60 │ 73  │ 74  │ 75  │ 76  │ 77  │ 78  │ 79  │ 80  │ 81  │
     ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
1.40 │ 64  │ 65  │ 66  │ 67  │ 68  │ 69  │ 70  │ 71  │ 72  │
     ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
1.20 │ 55  │ 56  │ 57  │ 58  │ 59  │ 60  │ 61  │ 62  │ 63  │
     ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
1.00 │ 46  │ 47  │ 48  │ 49  │ 50  │ 51  │ 52  │ 53  │ 54  │
     ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
0.80 │ 37  │ 38  │ 39  │ 40  │ 41  │ 42  │ 43  │ 44  │ 45  │
     ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
0.60 │ 28  │ 29  │ 30  │ 31  │ 32  │ 33  │ 34  │ 35  │ 36  │
     ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
0.40 │ 19  │ 20  │ 21  │ 22  │ 23  │ 24  │ 25  │ 26  │ 27  │
     ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
0.20 │ 10  │ 11  │ 12  │ 13  │ 14  │ 15  │ 16  │ 17  │ 18  │
     |─────|─────|─────|─────|─────|─────|─────|─────|─────|
0.00 │  1  │  2  │  3  │  4  │  5  │  6  │  7  │  8  │  9  │
     └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
     0.00  0.20  0.40  0.60  0.80  1.00  1.20  1.40  1.60    → X
```

**Traversal path (arrows show direction):**
```
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9
                                ↓
10 ← 11 ← 12 ← 13 ← 14 ← 15 ← 16 ← 17 ← 18
↓
19 → 20 → 21 → 22 → 23 → 24 → 25 → 26 → 27
                                ↓
... (continues in this pattern)
                                ↓
73 → 74 → 75 → 76 → 77 → 78 → 79 → 80 → 81
```

#### Key Takeaways

1. **Outer loop (`j`) controls rows:** Each iteration of `j` processes one complete row
2. **Inner loop (`i`) controls columns:** Each iteration of `i` processes one point in the current row
3. **Coordinate calculation:** `x = X0 + i * DX`, `y = Y0 + j * DY`
4. **Traversal order:** Left-to-right within each row, bottom-to-top across rows
5. **Total iterations:** NX × NY (e.g., 9 × 9 = 81 points)
6. **Progress display:** Uses `i+1` and `j+1` for 1-indexed user-friendly display

17. **`user = input(f"Point ({i+1}/{NX}, {j+1}/{NY}) -> x={x:.2f}, y={y:.2f}. Press Enter to capture (or 'q' to quit): ").strip()`**
    - Prompts user with current grid position
    - Shows progress: `(i+1}/{NX}, {j+1}/{NY})` (1-indexed for user-friendly display)
    - Shows calculated coordinates formatted to 2 decimal places
    - `.strip()` removes leading/trailing whitespace

18. **`if user.lower() in ("q", "quit", "exit"):`**
    - Checks if user wants to quit early (case-insensitive)

19. **`print("Done.")`** and **`return`**
    - Prints message and exits function (not just loop)

20. **`print(f"  Sampling {SAMPLES_PER_POINT} readings...")`**
    - Status message showing how many samples will be taken

21. **`try:`** (Line 136)
    - Starts error handling for sensor reading

22. **`bx, by, bz = read_avg_xyz_gauss(mag)`**
    - Takes N samples and gets averaged magnetic field components

23. **`except RuntimeError as e:`** (Line 138)
    - Catches sensor read errors

24. **`print(f"  ERROR: {e}")`** and **`continue`**
    - Prints error and skips to next grid point (doesn't crash)
    - Note: User can re-run script later to fill in missing points

25. **`b_total = math.sqrt(bx * bx + by * by + bz * bz)`**
    - Calculates total magnitude using 3D Pythagorean theorem
    - `bx * bx` is Bx², etc.
    
    **Example:**
    - If `Bx = 45.0`, `By = 23.0`, `Bz = 12.0`
    - `B_total = √(45.0² + 23.0² + 12.0²)`
    - `B_total = √(2025 + 529 + 144)`
    - `B_total = √2698`
    - `B_total ≈ 51.94` gauss

26. **`row = [utc_iso(), x, y, bx, by, bz, b_total, "gauss"]`**
    - Creates list with all data: timestamp, coordinates, field components, total, units

27. **`try:`** (Line 146)
    - Starts error handling for file write

28. **`append_row(CSV_PATH, row)`**
    - Appends row to CSV file

29. **`except RuntimeError as e:`** (Line 148)
    - Catches file write errors (disk full, permissions, etc.)

30. **`print(...)`** and **`return`**
    - Warns user and exits function (critical error - can't save data)

31. **`beep()`** (Line 154)
    - Produces an audible beep sound to confirm data was saved
    - Provides audio feedback when you can't see the screen
    - Uses ASCII bell character (`\a`) to trigger system beep

32. **`print(f"  Saved: x={x:.2f}, y={y:.2f}, B_total={b_total:.6f} gauss\n")`**
    - Confirmation message with formatted values
    - `.2f` formats coordinates to 2 decimal places
    - `.6f` formats B_total to 6 decimal places

**Grid Completion (Line 157):**

33. **`print("Grid complete. Done.")`**
    - Message when all grid points have been processed

---

## Section 5: Script Entry Point (Lines 160-165)

```python
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
```

**What it does:**

Runs the main function when the script is executed directly. Handles Ctrl+C gracefully.

**Line-by-line breakdown:**

1. **`if __name__ == "__main__":`**
   - Only runs if script is executed directly (not imported as module)
   - Python best practice

2. **`try:`**
   - Starts error handling block

3. **`main()`**
   - Calls the main function

4. **`except KeyboardInterrupt:`**
   - Catches Ctrl+C interrupt signal

5. **`print("\nStopped.")`**
   - Prints graceful exit message instead of showing Python error

---

## Key Concepts

### Auto-Grid Mode

The script automatically generates a grid of measurement points based on configuration settings (DX, DY, NX, NY, X0, Y0). You move the sensor to each grid point and press Enter to capture.

**Grid traversal order:**
- Starts at (X0, Y0)
- Moves left-to-right in X direction
- Then moves to next row (increasing Y)
- Continues until all NX × NY points are processed

**Example grid (NX=9, NY=9, DX=0.20, DY=0.20, X0=0.0, Y0=0.0):**
```
(0.00,1.60) (0.20,1.60) ... (1.60,1.60)
(0.00,1.40) (0.20,1.40) ... (1.60,1.40)
...
(0.00,0.00) (0.20,0.00) ... (1.60,0.00)
```

**Benefits:**
- Systematic coverage of area
- No need to manually calculate coordinates
- Progress tracking (shows current position in grid)
- Can quit early and resume later (missing points can be filled in)

**Note:** The script still supports early exit with 'q', allowing you to resume data collection later for any missing points.

### Averaging for Noise Reduction

- Sensor readings have random electrical noise
- Averaging reduces noise by √N (where N = number of samples)
- Example: 100 samples = 10x less noise, 1000 samples = 31x less noise
- Trade-off: More samples = better accuracy but slower collection

### Magnetic Field Components

- **Bx, By, Bz**: Components in X, Y, Z directions (gauss)
- **B_total**: Total magnitude = √(Bx² + By² + Bz²)
- Direction-independent magnitude used by downstream scripts

**Why record all components?**
- Direction matters for some analyses
- Can detect orientation changes
- B_total is magnitude only (direction-independent)
- Individual components useful for vector analysis

**Example:**
- If sensor reads: Bx = 45.0, By = 23.0, Bz = 12.0 gauss
- B_total = √(45.0² + 23.0² + 12.0²) = √2698 ≈ 51.94 gauss
- This is the magnitude of the 3D magnetic field vector

---

## Tips for Best Results

1. **Sensor Setup:** Ensure I2C is enabled and sensor is detected
2. **Sampling:** Adjust `SAMPLES_PER_POINT` based on accuracy vs speed needs
3. **Data Collection:** Use consistent coordinate system and spacing
4. **File Management:** Script appends to existing files (delete to start fresh)
5. **Error Handling:** Script handles common errors gracefully and continues

---

## Tips for Tuning

### 1. Sampling Parameters

**`SAMPLES_PER_POINT` parameter:**
- **Too few** (10-20): Fast but noisy, may miss subtle variations
- **Balanced** (100): Good default, ~1 second per point
- **High accuracy** (200-500): Slower but very stable, good for critical measurements
- **Very high** (1000+): Maximum accuracy but very slow (~10+ seconds per point)

**`SAMPLE_DELAY_S` parameter:**
- **Default (0.01s)**: Works for most sensors, ~100 Hz sampling rate
- **Faster (0.005s)**: May work if sensor can handle it, but risk of incomplete readings
- **Slower (0.02s)**: More conservative, ensures sensor is ready, but slower overall

**Example scenarios:**
- **Quick survey:** `SAMPLES_PER_POINT = 50`, `SAMPLE_DELAY_S = 0.01` → ~0.5 seconds per point
- **Balanced (default):** `SAMPLES_PER_POINT = 100`, `SAMPLE_DELAY_S = 0.01` → ~1 second per point
- **High accuracy:** `SAMPLES_PER_POINT = 200`, `SAMPLE_DELAY_S = 0.01` → ~2 seconds per point
- **Maximum accuracy:** `SAMPLES_PER_POINT = 500`, `SAMPLE_DELAY_S = 0.01` → ~5 seconds per point

### 2. Grid Configuration

**Adjusting grid size:**
- **`NX`, `NY`**: Number of points in each direction
  - More points = better resolution but more time
  - Example: 9×9 = 81 points, 15×15 = 225 points
- **`DX`, `DY`**: Spacing between points
  - Smaller spacing = higher resolution but more points
  - Example: 0.10 m = 10 cm spacing (dense), 0.50 m = 50 cm spacing (sparse)
- **`X0`, `Y0`**: Starting coordinates
  - Set to match your physical reference point
  - Example: If your grid starts at corner of room, set X0, Y0 to that corner's coordinates

**Grid calculation:**
- Total points = NX × NY
- Grid width = (NX - 1) × DX
- Grid height = (NY - 1) × DY
- Last point at: (X0 + (NX-1)×DX, Y0 + (NY-1)×DY)

**Example configurations:**
- **Small area (1m×1m, dense):** NX=11, NY=11, DX=0.10, DY=0.10 → 121 points, 1.0m span
- **Medium area (1.6m×1.6m, balanced):** NX=9, NY=9, DX=0.20, DY=0.20 → 81 points, 1.6m span (default)
- **Large area (2m×2m, sparse):** NX=5, NY=5, DX=0.50, DY=0.50 → 25 points, 2.0m span

**Coordinate system:**
- All coordinates are in meters
- Grid starts at (X0, Y0) and extends in positive X and Y directions
- Document your coordinate system for later reference

### 3. File Management

- Script appends to existing files (resume capability)
- To start fresh, delete or rename `mag_data.csv`
- Consider using dated filenames: `mag_data_2024-03-15.csv`
- Check disk space before long collection sessions

### 4. Error Recovery

- If sensor disconnects mid-measurement, script skips that point and continues
- If file write fails, measurement is lost but script continues
- Always verify data was saved by checking CSV file after collection

---

## Integration with Other Scripts

This script creates CSV files used by:
- `compute_local_anomaly_v1.py`: Reads `x`, `y`, `B_total` columns
- `interpolate_to_heatmapV1.py`: Creates heatmaps from point measurements

**Data flow:**
```
mag_to_csv.py → mag_data.csv → compute_local_anomaly_v1.py → mag_data_with_anomaly.csv
                                                          ↓
                                    interpolate_to_heatmapV1.py → heatmap visualization
```

---

## Summary: The Complete Workflow

1. **Setup:** Connect MMC5983MA sensor via I2C
2. **Configure grid:** Adjust NX, NY, DX, DY, X0, Y0 in script if needed
3. **Run script:** `python mag_to_csv.py`
4. **For each grid point:**
   - Script displays current position: `Point (i/NX, j/NY) -> x=..., y=...`
   - Move sensor to that location
   - Press Enter to capture
   - Script takes N samples and averages them
   - Calculates B_total = √(Bx² + By² + Bz²)
   - Saves row to CSV with timestamp
5. **Continue:** Script automatically moves to next grid point
6. **Early exit:** Type `q` then Enter to quit early (can resume later)
7. **Completion:** When all points done, script prints "Grid complete. Done."
8. **Result:** CSV file with all measurements ready for analysis

---

## Next Steps

After collecting data:
1. **Analyze anomalies:** Run `compute_local_anomaly_v1.py`
2. **Create visualizations:** Use `interpolate_to_heatmapV1.py`
3. **Export to GIS:** Import CSV into QGIS or other mapping software
4. **Further processing:** Use pandas/numpy for custom analysis
