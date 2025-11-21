# Import Statements Explanation

This document explains the three import statements used in `compute_local_anomaly_v1.py` and provides examples of how they work.

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

