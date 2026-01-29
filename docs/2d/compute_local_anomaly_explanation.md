# Complete Explanation of `compute_local_anomaly_v1.py`

This document explains every part of the local anomaly computation script, step by step.

---

## Overview

This script detects **local anomalies** in magnetic field data. Instead of comparing each point to the global average, it compares each point to its **local neighborhood** - nearby points within a certain radius. This helps identify small-scale variations that might be hidden by global trends.

---

## Section 1: Parameters (Lines 5-8)

```python
CSV_PATH = "mag_data.csv"      # input file
RADIUS = 0.3                   # neighborhood radius, in same units as x,y (cm or m)
```

**What it does:**
- `CSV_PATH``: Path to your input CSV file containing magnetic field measurements
- `RADIUS`: The distance threshold for finding "neighbors" around each point

**Example:**
- If `RADIUS = 0.3` meters, the script will look at all points within 0.3 meters of each point
- Points within this radius are considered "neighbors"
- This radius determines the scale of anomalies you're looking for:
  - **Small radius** (0.1 m): Detects very local, small-scale anomalies
  - **Large radius** (1.0 m): Detects broader regional anomalies

---

## Section 2: Loading Data (Lines 11-19)

```python
# 1) Load CSV
df = pd.read_csv(CSV_PATH)

# Expect columns: x, y, B_total
coords = df[["x", "y"]].values   # NumPy array, shape (N, 2)
B = df["B_total"].values         # NumPy array of length N

N = len(df)
anomalies = np.zeros(N)
```

**What it does:**

1. **`df = pd.read_csv(CSV_PATH)`**
   - Loads the CSV file into a pandas DataFrame
   - Expected format:
     ```
     x, y, B_total
     1.2, 2.3, 45.6
     1.5, 2.4, 46.1
     ...
     ```

2. **`coords = df[["x", "y"]].values`**
   - Extracts x and y coordinates into a NumPy array
   - Shape: `(N, 2)` where N is the number of data points
   - Example: `[[1.2, 2.3], [1.5, 2.4], [1.8, 2.5], ...]`
   - Each row is one point's (x, y) location

3. **`B = df["B_total"].values`**
   - Extracts magnetic field values into a NumPy array
   - Shape: `(N,)` - a 1D array of length N
   - Example: `[45.6, 46.1, 45.9, ...]`

4. **`N = len(df)`**
   - Total number of data points

5. **`anomalies = np.zeros(N)`**
   - Creates an empty array to store anomaly values
   - Will be filled in the loop below

---

## Section 3: Computing Local Anomalies (Lines 21-38)

This is the **core algorithm** - the most important part!

```python
# 2) For each point, find neighbors within RADIUS and compute local anomaly
for i in range(N):
    xi, yi = coords[i]
    # Compute distance from this point to all others
    dx = coords[:, 0] - xi
    dy = coords[:, 1] - yi
    dist = np.sqrt(dx*dx + dy*dy)

    # Boolean mask for neighbors; include everything within RADIUS, exclude self
    neighbor_mask = (dist <= RADIUS) & (dist > 0)

    # If no neighbors (shouldn't happen in a decent grid), fall back to global mean
    if not np.any(neighbor_mask):
        local_mean = B.mean()
    else:
        local_mean = B[neighbor_mask].mean()

    anomalies[i] = B[i] - local_mean
```

### Step-by-Step Breakdown:

#### For each point `i`:

1. **Get the point's location:**
   ```python
   xi, yi = coords[i]
   ```
   - Extracts the x and y coordinates of point `i`
   - Example: If `i=0` and `coords[0] = [1.2, 2.3]`, then `xi=1.2`, `yi=2.3`

2. **Calculate distances to ALL other points:**
   ```python
   dx = coords[:, 0] - xi
   dy = coords[:, 1] - yi
   dist = np.sqrt(dx*dx + dy*dy)
   ```
   - `coords[:, 0]` = all x-coordinates (entire first column)
   - `coords[:, 1]` = all y-coordinates (entire second column)
   - `dx` = array of x-differences: `[x0-xi, x1-xi, x2-xi, ..., xN-xi]`
   - `dy` = array of y-differences: `[y0-yi, y1-yi, y2-yi, ..., yN-yi]`
   - `dist` = array of distances using Pythagorean theorem: `[d0, d1, d2, ..., dN]`
   
   **Example:**
   - Point i is at (1.2, 2.3)
   - Point j is at (1.5, 2.4)
   - Distance = √[(1.5-1.2)² + (2.4-2.3)²] = √[0.09 + 0.01] = √0.1 ≈ 0.316

3. **Find neighbors (points within RADIUS):**
   ```python
   neighbor_mask = (dist <= RADIUS) & (dist > 0)
   ```
   - Creates a **boolean mask** (True/False array)
   - `dist <= RADIUS`: True for points within radius
   - `dist > 0`: True for points that are NOT the point itself (exclude self)
   - `&`: Both conditions must be True (AND operation)
   
   **Example:**
   - If `RADIUS = 0.3` and distances are `[0.0, 0.316, 0.5, 0.2, ...]`
   - Mask: `[False, False, False, True, ...]` (only point 3 is within radius)

4. **Calculate local mean:**
   ```python
   if not np.any(neighbor_mask):
       local_mean = B.mean()  # Fallback: use global mean
   else:
       local_mean = B[neighbor_mask].mean()  # Mean of neighbors only
   ```
   - `np.any(neighbor_mask)`: Checks if there are ANY neighbors
   - If no neighbors found: use global mean (shouldn't happen in practice)
   - If neighbors found: calculate mean of magnetic field values at those neighbor points
   
   **Example:**
   - Neighbors are at indices [3, 7, 12]
   - Their B values: `[45.6, 46.1, 45.9]`
   - `local_mean = (45.6 + 46.1 + 45.9) / 3 = 45.87`

5. **Compute anomaly:**
   ```python
   anomalies[i] = B[i] - local_mean
   ```
   - Anomaly = how different this point is from its local neighborhood
   - **Positive anomaly**: Point is higher than neighbors (hot spot)
   - **Negative anomaly**: Point is lower than neighbors (cold spot)
   - **Near zero**: Point is similar to neighbors (normal)

**Visual Example:**
```
Point locations and B values:
    45.0    45.5    46.0
    45.2  [46.5]   45.8    ← Point i with B=46.5
    45.1    45.3    45.4

If RADIUS includes the 8 surrounding points:
local_mean = (45.0 + 45.5 + 46.0 + 45.2 + 45.8 + 45.1 + 45.3 + 45.4) / 8
           = 45.4
anomaly[i] = 46.5 - 45.4 = +1.1  (positive anomaly - hot spot!)
```

---

## Section 4: Normalization (Lines 40-51)

```python
# 3) Normalize anomaly values for color mapping
anomalies_for_color = anomalies  # keep sign for now

amin = anomalies_for_color.min()
amax = anomalies_for_color.max()
# Avoid division by zero
if amax - amin < 1e-9:
    norm = np.zeros_like(anomalies_for_color)
else:
    norm = (anomalies_for_color - amin) / (amax - amin)
```

**What it does:**

Normalizes anomaly values to a range of 0 to 1 for color mapping in the visualization.

1. **Keep original anomalies:**
   ```python
   anomalies_for_color = anomalies
   ```
   - Could use `np.abs(anomalies)` if you only care about magnitude (not direction)

2. **Find min and max:**
   ```python
   amin = anomalies_for_color.min()  # Smallest anomaly value
   amax = anomalies_for_color.max()   # Largest anomaly value
   ```

3. **Normalize to 0-1 range:**
   ```python
   norm = (anomalies_for_color - amin) / (amax - amin)
   ```
   - Formula: `(value - minimum) / (maximum - minimum)` 
   - subtracts all min values from the array creating a new array
   - Maps the range `[amin, amax]` → `[0, 1]`
   
   **Example:**
   - Anomalies: `[-2.5, -1.0, 0.5, 1.5, 3.0]`
   - `amin = -2.5`, `amax = 3.0`
   - Normalized:
     - `-2.5` → `(-2.5 - (-2.5)) / (3.0 - (-2.5)) = 0.0`
     - `0.5` → `(0.5 - (-2.5)) / (3.0 - (-2.5)) = 0.545`
     - `3.0` → `(3.0 - (-2.5)) / (3.0 - (-2.5)) = 1.0`

4. **Safety check:**
   ```python
   if amax - amin < 1e-9:
       norm = np.zeros_like(anomalies_for_color)
   ```
   - Prevents division by zero if all anomalies are the same
   - If all values are identical, set normalized values to 0
   - Creates array of zeros

**Why normalize?**
- Color maps work best with values between 0 and 1
- Makes visualization consistent regardless of actual anomaly magnitudes
- Preserves the relative differences between points

---

## Section 5: Saving Results (Lines 53-58)

```python
# 4) Attach anomaly back into the DataFrame (optional, for exporting to QGIS later)
df["local_anomaly"] = anomalies
df["local_anomaly_norm"] = norm

# Save to CSV if you want to use in QGIS or other tools
df.to_csv("mag_data_with_anomaly.csv", index=False)
```

**What it does:**

1. **Add new columns to DataFrame:**
   - `df["local_anomaly"]`: Original anomaly values (can be positive or negative)
   - `df["local_anomaly_norm"]`: Normalized values (0 to 1)

2. **Save to CSV:**
   - Creates a new file with all original data plus the two new anomaly columns
   - `index=False`: Don't save the row numbers
   - Can be imported into QGIS or other GIS software for further analysis

**Output CSV format:**
```
x, y, B_total, local_anomaly, local_anomaly_norm
1.2, 2.3, 45.6, -0.5, 0.2
1.5, 2.4, 46.1, +1.1, 0.8
...
```

---

## Section 6: Visualization (Lines 60-76)

```python
# 5) Quick visualization (scatter "heatmap")
plt.figure(figsize=(6, 5))
sc = plt.scatter(
    df["x"], df["y"],
    c=df["local_anomaly_norm"],
    cmap="seismic",   # blue-white-red; you can try 'plasma', 'viridis', etc.
    s=120,
    edgecolor="k"
)

plt.colorbar(sc, label="Normalized local anomaly")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Magnetic Local Anomaly Map")
plt.gca().set_aspect("equal", "box")
plt.tight_layout()
plt.show()
```

**What it does:**

Creates a colored scatter plot where each point's color represents its anomaly value.

1. **Create figure:**
   ```python
   plt.figure(figsize=(6, 5))
   ```
   - Creates a new plot window
   - Size: 6 inches wide × 5 inches tall

2. **Create scatter plot:**
   ```python
   sc = plt.scatter(
       df["x"], df["y"],                    # X and Y positions
       c=df["local_anomaly_norm"],          # Color based on normalized anomaly
       cmap="seismic",                      # Color scheme: blue → white → red
       s=120,                               # Point size
       edgecolor="k"                        # Black edge around each point
   )
   ```
   - **`df["x"], df["y"]`**: Point locations
   - **`c=df["local_anomaly_norm"]`**: Color each point by its normalized anomaly
   - **`cmap="seismic"`**: 
     - Blue = negative anomalies (cold spots)
     - White = near zero (normal)
     - Red = positive anomalies (hot spots)
   - **`s=120`**: Size of each point (in square points)
   - **`edgecolor="k"`**: Black border around points for better visibility

3. **Add color bar:**
   ```python
   plt.colorbar(sc, label="Normalized local anomaly")
   ```
   - Shows the color scale
   - Label explains what the colors mean

4. **Add labels:**
   ```python
   plt.xlabel("x (m)")
   plt.ylabel("y (m)")
   plt.title("Magnetic Local Anomaly Map")
   ```
   - Axis labels and title

5. **Set equal aspect ratio:**
   ```python
   plt.gca().set_aspect("equal", "box")
   ```
   - Makes x and y axes use the same scale
   - Important for spatial data - prevents distortion
   - A circle in your data will look like a circle, not an ellipse

6. **Adjust layout and display:**
   ```python
   plt.tight_layout()  # Prevents labels from being cut off
   plt.show()          # Display the plot
   ```

---

## Summary: The Complete Workflow

1. **Load data** → Read CSV file with x, y, B_total columns
2. **For each point:**
   - Find all neighbors within RADIUS
   - Calculate mean of neighbors' B values
   - Compute anomaly = point's B - local mean
3. **Normalize** → Scale anomalies to 0-1 for visualization
4. **Save** → Add anomaly columns to DataFrame and export CSV
5. **Visualize** → Create colored scatter plot showing anomalies

---

## Key Concepts

### Local vs Global Anomaly

- **Global anomaly**: Compare each point to the overall average
  - Example: If global mean = 50, point with B=52 has +2 global anomaly
  - Problem: Misses local variations

- **Local anomaly**: Compare each point to its nearby neighbors
  - Example: If neighbors average 45, point with B=52 has +7 local anomaly
  - Advantage: Detects small-scale features that global analysis misses

### Why This Matters

In geophysics, you might have:
- A gradual regional trend (background field)
- Local anomalies (buried objects, geological features)

Local anomaly detection helps separate these:
- Regional trend is "normal" for the neighborhood
- Only deviations from the local pattern are flagged as anomalies

---

## Tips for Tuning

1. **RADIUS parameter:**
   - Too small: May miss broader anomalies, noisy results
   - Too large: May smooth out real local features
   - Start with ~10-20% of your data spacing

2. **Colormap choice:**
   - `"seismic"`: Good for positive/negative anomalies (blue-white-red)
   - `"viridis"`: Good for magnitude only (dark to bright)
   - `"plasma"`: Similar to viridis, different color scheme

3. **Point size (`s` parameter):**
   - Larger values: Better for sparse data
   - Smaller values: Better for dense data
   - Adjust based on your data density

