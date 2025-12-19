# Import Statements Explanation

This document explains the import statements used across Fluxspace Core scripts and provides examples of how they work.

## 1. `import pandas as pd`

**What it does:** Pandas is a powerful library for working with structured data (like CSV files, Excel files, databases). The `as pd` part creates an alias so you can type `pd` instead of `pandas` every time.

### What is `df`?

**Important:** `df` is NOT part of the import statement! It's a **variable name** that you create to hold a DataFrame object.

- `df` stands for **"DataFrame"** (a common naming convention)
- A **DataFrame** is pandas' main data structure - think of it as a table or spreadsheet
- You create `df` by calling `pd.read_csv()` which returns a DataFrame
- You could name it anything: `data`, `table`, `my_data`, etc. - `df` is just the convention

**In your script:**
```python
df = pd.read_csv(CSV_PATH)  # This line creates the variable 'df'
```

This line does two things:
1. `pd.read_csv(CSV_PATH)` - Reads the CSV file and creates a DataFrame object
2. `df = ...` - Stores that DataFrame in a variable named `df`

**Think of a DataFrame like this:**
```
     x      y    B_total
0  1.2   2.3      45.6
1  1.5   2.4      46.1
2  1.8   2.5      45.9
```

It has:
- **Rows** (each row is a data point)
- **Columns** (each column is a variable/feature)
- **Column names** (like "x", "y", "B_total")

**How it's used in the script:**
- `pd.read_csv(CSV_PATH)` - Reads a CSV file into a DataFrame (a table-like structure)
- `df[["x", "y"]]` - Selects specific columns from the DataFrame
- `df["B_total"]` - Selects a single column
- `df["local_anomaly"] = anomalies` - Adds a new column to the DataFrame
- `df.to_csv("output.csv", index=False)` - Saves the DataFrame to a CSV file

**Examples:**

```python
import pandas as pd

# Read a CSV file - THIS CREATES THE 'df' VARIABLE
df = pd.read_csv("data.csv")

# Inspect what's in df
print(df)           # Shows the entire DataFrame
print(df.head())    # Shows first 5 rows
print(df.shape)     # Shows (number_of_rows, number_of_columns)
print(df.columns)   # Shows column names: ['x', 'y', 'B_total']
print(df.info())    # Shows data types and memory usage

# Example: If your CSV looks like this:
# x,y,B_total
# 1.2,2.3,45.6
# 1.5,2.4,46.1
# 1.8,2.5,45.9
#
# Then df will be:
#      x    y  B_total
# 0  1.2  2.3     45.6
# 1  1.5  2.4     46.1
# 2  1.8  2.5     45.9

# Access columns
x_values = df["x"]           # Single column (returns a Series)
x_y = df[["x", "y"]]         # Multiple columns (returns a DataFrame, note double brackets)

# Access rows
first_row = df.iloc[0]       # First row by position
row_by_index = df.loc[0]     # Row by index label

# Filter data
filtered = df[df["B_total"] > 100]  # Rows where B_total > 100

# Add new column
df["new_column"] = df["x"] * 2

# Get basic statistics
print(df.describe())  # Mean, std, min, max for each numeric column

# Save to CSV
df.to_csv("output.csv", index=False)
```

---

## 2. `import numpy as np`

**What it does:** NumPy provides fast mathematical operations on arrays (lists of numbers). The `as np` creates a shorter alias. NumPy arrays are much faster than Python lists for mathematical operations.

**How it's used in the script:**
- `coords = df[["x", "y"]].values` - Converts DataFrame columns to NumPy array
- `B = df["B_total"].values` - Converts a column to NumPy array
- `np.zeros(N)` - Creates an array of zeros
- `np.sqrt(dx*dx + dy*dy)` - Square root function
- `np.any(neighbor_mask)` - Checks if any value in array is True
- `B.mean()` - Calculates mean of array values
- `np.abs(anomalies)` - Absolute value of each element
- `np.zeros_like(anomalies_for_color)` - Creates array of zeros with same shape

**Examples:**

```python
import numpy as np

# Create arrays
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros(10)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ones = np.ones(5)     # [1, 1, 1, 1, 1]

# Mathematical operations (element-wise)
arr * 2        # [2, 4, 6, 8, 10]
arr + 1        # [2, 3, 4, 5, 6]
np.sqrt(arr)   # [1.0, 1.41, 1.73, 2.0, 2.24]

# Array operations
arr.mean()     # 3.0 (average)
arr.max()      # 5
arr.min()      # 1
arr.sum()      # 15

# Boolean operations
mask = arr > 3  # [False, False, False, True, True]
np.any(mask)    # True (at least one True)
np.all(mask)    # False (not all True)

# Distance calculation (like in the script)
point1 = np.array([0, 0])
point2 = np.array([3, 4])
distance = np.sqrt(np.sum((point2 - point1)**2))  # 5.0

# Vectorized operations (very fast!)
coords = np.array([[1, 2], [3, 4], [5, 6]])
center = np.array([2, 3])
distances = np.sqrt(np.sum((coords - center)**2, axis=1))
# Calculates distance from center to each point in one operation!
```

---

## 3. `import matplotlib.pyplot as plt`

**What it does:** Matplotlib is a plotting library for creating visualizations. The `pyplot` module provides a simple interface for creating plots. The `as plt` creates a shorter alias.

> **Visual Examples:** For hands-on visual examples, run `scripts/plot_examples/matplotlib_examples.py`. This script creates 3 types of plots (colormaps, heatmap, and anomaly map) and saves them as PNG files so you can see exactly how matplotlib works!

**How it's used in the script:**
- `plt.figure(figsize=(6, 5))` - Creates a new figure with specified size
- `plt.scatter(...)` - Creates a scatter plot
- `plt.colorbar(...)` - Adds a color scale bar
- `plt.xlabel(...)` - Sets x-axis label
- `plt.ylabel(...)` - Sets y-axis label
- `plt.title(...)` - Sets plot title
- `plt.gca().set_aspect("equal", "box")` - Makes axes have equal scaling
- `plt.tight_layout()` - Adjusts spacing
- `plt.show()` - Displays the plot

**Examples:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Simple line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Sine Wave")
plt.show()

# Scatter plot with colors (like in the script)
x = np.random.rand(100)
y = np.random.rand(100)
colors = np.random.rand(100)  # Color for each point

plt.figure(figsize=(8, 6))
sc = plt.scatter(x, y, c=colors, cmap="viridis", s=50)
plt.colorbar(sc, label="Color values")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Colored Scatter Plot")
plt.show()

# Multiple plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(x, y)
ax2.scatter(x, y)
plt.show()
```

---

## Why Use Aliases (`as pd`, `as np`, `as plt`)?

1. **Shorter to type:** `pd.read_csv()` is faster than `pandas.read_csv()`
2. **Convention:** These are standard aliases used by the Python data science community
3. **Readability:** Everyone knows what `pd`, `np`, and `plt` mean

---

## How They Work Together in the Script

1. **Pandas** loads and manages the CSV data as a DataFrame
2. **NumPy** performs fast mathematical calculations (distances, means, etc.)
3. **Matplotlib** visualizes the results as a colored scatter plot

The workflow:
```
CSV file → pandas (read) → NumPy arrays (calculate) → pandas (store results) → matplotlib (visualize)
```

---

## 4. `import os`

**What it does:** The `os` module provides functions for interacting with the operating system, especially file and directory operations.

**How it's used in the scripts:**
- `os.path.exists(path)` - Checks if a file or directory exists
- `os.path.dirname(path)` - Gets the directory part of a file path
- `os.makedirs(path, exist_ok=True)` - Creates directories (won't error if they already exist)
- `os.path.join(path1, path2)` - Joins path components (handles different OS path separators)

**Examples:**

```python
import os

# Check if file exists
if os.path.exists("data.csv"):
    print("File exists!")

# Get directory name
file_path = "data/raw/mag_data.csv"
directory = os.path.dirname(file_path)  # "data/raw"

# Create directory (won't error if it already exists)
os.makedirs("data/processed", exist_ok=True)

# Join paths (works on Windows, Mac, Linux)
full_path = os.path.join("data", "raw", "mag_data.csv")  # "data/raw/mag_data.csv"

# Get current working directory
current_dir = os.getcwd()  # "/Users/username/fluxspace-core"

# List files in directory
files = os.listdir("data/raw")  # ['mag_data.csv', 'other_file.csv']
```

---

## 5. `import csv`

**What it does:** The `csv` module provides functions for reading and writing CSV (Comma-Separated Values) files. It handles proper formatting, escaping, and parsing of CSV data.

**How it's used in the scripts:**
- `csv.writer(file)` - Creates a CSV writer object
- `csv.reader(file)` - Creates a CSV reader object
- `writer.writerow(row)` - Writes a single row to the CSV file
- `reader.next()` or `next(reader)` - Reads the next row from CSV

**Important:** When opening files for CSV writing, use `newline=""` to prevent extra blank lines on Windows.

**Examples:**

```python
import csv

# Writing CSV
with open("output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "B_total"])  # Header row
    writer.writerow([1.2, 2.3, 45.6])       # Data row
    writer.writerow([1.5, 2.4, 46.1])       # Data row

# Reading CSV
with open("data.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)  # Read first row (header)
    print(header)  # ['x', 'y', 'B_total']
    
    for row in reader:
        x, y, b_total = float(row[0]), float(row[1]), float(row[2])
        print(f"Point: ({x}, {y}), B_total: {b_total}")

# Reading with DictReader (easier - uses column names)
with open("data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x = float(row["x"])
        y = float(row["y"])
        b_total = float(row["B_total"])
        print(f"Point: ({x}, {y}), B_total: {b_total}")
```

---

## 6. `import sys`

**What it does:** The `sys` module provides access to system-specific parameters and functions, including standard input/output streams and exit codes.

**How it's used in the scripts:**
- `sys.stderr` - Standard error stream (for error messages)
- `sys.stdout` - Standard output stream (for normal output)
- `sys.exit(code)` - Exits the program with an exit code
- `print(..., file=sys.stderr)` - Prints to error stream instead of normal output

**Why separate stdout and stderr?**
- **stdout** (`print()` by default): Normal program output, can be redirected to files
- **stderr** (`print(..., file=sys.stderr)`): Error messages, always shown to user even if output is redirected
- This allows separating normal output from errors, which is important for automation and logging

**Examples:**

```python
import sys

# Normal output (goes to stdout)
print("Processing data...")
print("Done!")

# Error output (goes to stderr)
print("ERROR: File not found!", file=sys.stderr)
print("ERROR: Could not connect to sensor!", file=sys.stderr)

# Exit codes
# 0 = success
# 1 = general error
# 2 = file error
# 130 = interrupted (Ctrl+C)
sys.exit(0)   # Exit successfully
sys.exit(1)   # Exit with error

# Example: In a script
try:
    data = load_data()
except FileNotFoundError:
    print("ERROR: Could not find data file", file=sys.stderr)
    sys.exit(1)  # Exit with error code 1
```

---

## 7. `import math`

**What it does:** The `math` module provides mathematical functions and constants. It's part of Python's standard library and works with regular Python numbers (not NumPy arrays).

**How it's used in the scripts:**
- `math.sqrt(x)` - Square root function
- `math.pow(x, y)` - x raised to the power of y
- `math.pi` - Mathematical constant π (3.14159...)
- `math.sqrt(bx*bx + by*by + bz*bz)` - Calculate magnitude of 3D vector

**Note:** For NumPy arrays, use `np.sqrt()` instead. `math.sqrt()` works with single numbers.

**Examples:**

```python
import math

# Square root
result = math.sqrt(16)  # 4.0
result = math.sqrt(2)   # 1.4142135623730951

# Power
result = math.pow(2, 3)  # 8.0 (2^3)
result = math.pow(5, 2)  # 25.0 (5^2)

# Constants
pi = math.pi  # 3.141592653589793

# Calculate B_total from components (like in mag_to_csv.py)
bx, by, bz = 0.5, 0.3, 0.4
b_total = math.sqrt(bx*bx + by*by + bz*bz)  # 0.7071...

# Distance between two points
x1, y1 = 0, 0
x2, y2 = 3, 4
distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)  # 5.0

# For arrays, use NumPy instead:
import numpy as np
arr = np.array([1, 4, 9, 16])
result = np.sqrt(arr)  # [1. 2. 3. 4.]
```

---

## 8. `import time`

**What it does:** The `time` module provides functions for working with time, including delays and time measurements.

**How it's used in the scripts:**
- `time.sleep(seconds)` - Pauses execution for a specified number of seconds
- `time.time()` - Returns current time as seconds since epoch (Unix timestamp)

**Examples:**

```python
import time

# Sleep (pause) for a specified duration
print("Starting...")
time.sleep(1.0)  # Pause for 1 second
print("Done!")

# Sleep with fractional seconds
time.sleep(0.01)  # Pause for 10 milliseconds (0.01 seconds)

# Example: Taking multiple sensor readings with delay
for i in range(10):
    reading = read_sensor()
    print(f"Reading {i+1}: {reading}")
    time.sleep(0.01)  # Wait 10ms between readings

# Get current time (Unix timestamp)
current_time = time.time()  # 1704067200.123 (seconds since Jan 1, 1970)

# Measure how long something takes
start = time.time()
# ... do some work ...
end = time.time()
elapsed = end - start
print(f"Took {elapsed:.2f} seconds")
```

---

## 9. `from datetime import datetime, timezone`

**What it does:** The `datetime` module provides classes for working with dates and times. We import `datetime` (the class) and `timezone` (for timezone-aware timestamps).

**How it's used in the scripts:**
- `datetime.now(timezone.utc)` - Gets current UTC time
- `.isoformat()` - Converts datetime to ISO 8601 string format
- UTC (Coordinated Universal Time) ensures timestamps are consistent regardless of local timezone

**Examples:**

```python
from datetime import datetime, timezone

# Get current UTC time
now = datetime.now(timezone.utc)
print(now)  # 2024-01-01 12:00:00.123456+00:00

# Convert to ISO format string
iso_string = now.isoformat()
print(iso_string)  # "2024-01-01T12:00:00.123456+00:00"

# With milliseconds precision (like in mag_to_csv.py)
iso_string = now.isoformat(timespec="milliseconds")
print(iso_string)  # "2024-01-01T12:00:00.123+00:00"

# Parse ISO string back to datetime
parsed = datetime.fromisoformat("2024-01-01T12:00:00.123+00:00")

# Compare times
time1 = datetime.now(timezone.utc)
time.sleep(1)
time2 = datetime.now(timezone.utc)
difference = time2 - time1  # timedelta object
print(difference.total_seconds())  # 1.0

# Format for display
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
print(formatted)  # "2024-01-01 12:00:00"
```

---

## 10. `import argparse`

**What it does:** The `argparse` module makes it easy to create command-line interfaces for scripts. It automatically generates help messages and handles argument parsing.

**How it's used in the scripts:**
- `argparse.ArgumentParser()` - Creates a parser object
- `.add_argument()` - Defines command-line arguments
- `.parse_args()` - Parses command-line arguments and returns them as an object

**Examples:**

```python
import argparse

# Create parser
parser = argparse.ArgumentParser(description="Process magnetometer data")

# Add arguments
parser.add_argument("--in", dest="infile", required=True, help="Input CSV file")
parser.add_argument("--out", dest="outfile", default="output.csv", help="Output file")
parser.add_argument("--radius", type=float, default=0.30, help="Neighborhood radius")
parser.add_argument("--plot", action="store_true", help="Generate plot")

# Parse arguments
args = parser.parse_args()

# Access parsed arguments
print(f"Input file: {args.infile}")
print(f"Output file: {args.outfile}")
print(f"Radius: {args.radius}")
if args.plot:
    print("Plotting enabled")

# Example command line:
# python3 script.py --in data.csv --radius 0.5 --plot
# 
# This creates:
# args.infile = "data.csv"
# args.outfile = "output.csv" (default)
# args.radius = 0.5
# args.plot = True
```

---

## 11. `from pathlib import Path`

**What it does:** `pathlib` provides an object-oriented way to work with file paths. It's more modern and easier to use than `os.path`, and works the same on all operating systems.

**How it's used in the scripts:**
- `Path("file.csv")` - Creates a Path object
- `.with_name("new_name.csv")` - Changes filename while keeping directory
- `.with_suffix(".png")` - Changes file extension
- `.parent` - Gets parent directory
- `.exists()` - Checks if path exists
- `.stem` - Gets filename without extension

**Examples:**

```python
from pathlib import Path

# Create Path objects
file_path = Path("data/raw/mag_data.csv")
output_dir = Path("data/processed")

# Get parts of path
print(file_path.name)      # "mag_data.csv"
print(file_path.stem)      # "mag_data"
print(file_path.suffix)    # ".csv"
print(file_path.parent)    # Path("data/raw")

# Change filename
new_path = file_path.with_name("mag_data_clean.csv")
# Path("data/raw/mag_data_clean.csv")

# Change extension
png_path = file_path.with_suffix(".png")
# Path("data/raw/mag_data.png")

# Join paths
full_path = Path("data") / "raw" / "mag_data.csv"
# Path("data/raw/mag_data.csv")

# Check if exists
if file_path.exists():
    print("File exists!")

# Create directory
output_dir.mkdir(parents=True, exist_ok=True)

# Read/write (works with open())
with open(file_path, "r") as f:
    data = f.read()

# Example: Generate output filename from input
input_file = Path("data/raw/mag_data.csv")
output_file = input_file.parent / f"{input_file.stem}_clean.csv"
# Path("data/raw/mag_data_clean.csv")
```

---

## 12. `from typing import ...`

**What it does:** The `typing` module provides type hints to make code more readable and help catch errors. Type hints tell you what types of values functions expect and return.

**Common imports:**
- `Optional[T]` - Value can be type T or None
- `Tuple[A, B]` - A tuple with two elements of types A and B
- `List[T]` - A list containing elements of type T
- `Dict[K, V]` - A dictionary with keys of type K and values of type V

**How it's used in the scripts:**
- Function parameters: `def func(x: float, name: str) -> int:`
- Return types: `-> Optional[pd.DataFrame]`
- Variable hints: `result: Tuple[Optional[pd.Series], str]`

**Examples:**

```python
from typing import Optional, Tuple, List, Dict

# Optional: Value can be the type or None
def find_item(items: List[str], target: str) -> Optional[str]:
    if target in items:
        return target
    return None  # Can return None

# Tuple: Multiple return values
def parse_data(line: str) -> Tuple[float, float, float]:
    parts = line.split(",")
    return float(parts[0]), float(parts[1]), float(parts[2])

# List: List of specific type
def process_numbers(numbers: List[float]) -> List[float]:
    return [n * 2 for n in numbers]

# Dict: Dictionary with specific key/value types
def create_lookup() -> Dict[str, int]:
    return {"x": 0, "y": 1, "z": 2}

# Combined example (like in validate_and_diagnosticsV1.py)
def coerce_time_series(s: pd.Series) -> Tuple[Optional[pd.Series], str]:
    if s is None:
        return None, "No time column"
    # ... process ...
    return time_series, "Parsed successfully"
```

---

## 13. `from __future__ import annotations`

**What it does:** This import enables "postponed evaluation of annotations" - it allows you to use type hints without quotes, even when referencing classes that haven't been defined yet.

**Why use it:**
- Makes type hints cleaner (no quotes needed)
- Allows forward references (referring to classes defined later)
- Required for some advanced type hinting features

**Examples:**

```python
# Without __future__ annotations:
from typing import Optional

class Node:
    def __init__(self, value: int, next_node: Optional['Node'] = None):
        # Note: 'Node' needs quotes because it's not defined yet
        self.value = value
        self.next_node = next_node

# With __future__ annotations:
from __future__ import annotations
from typing import Optional

class Node:
    def __init__(self, value: int, next_node: Optional[Node] = None):
        # No quotes needed!
        self.value = value
        self.next_node = next_node

# Also allows cleaner return type hints:
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    # Can use pd.DataFrame directly without importing it in quotes
    return df.copy()
```

**Note:** This is a "future" import that will become the default behavior in Python 3.11+. Using it now makes your code compatible with future Python versions.

---

## Summary: Standard Library vs External Libraries

**Standard Library (built into Python, no installation needed):**
- `os`, `csv`, `sys`, `math`, `time` - Basic operations
- `datetime` - Date/time handling
- `argparse` - Command-line arguments
- `pathlib` - Modern path handling
- `typing` - Type hints
- `__future__` - Future language features

**External Libraries (need to install with `pip`):**
- `pandas` - Data manipulation (`pip install pandas`)
- `numpy` - Numerical computing (`pip install numpy`)
- `matplotlib` - Plotting (`pip install matplotlib`)
- `qwiic_mmc5983ma` - Sensor library (`pip install sparkfun-qwiic-mmc5983ma`)

---

## Complete Import Example

Here's what a typical script might import:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Now you can use all these modules!
```

