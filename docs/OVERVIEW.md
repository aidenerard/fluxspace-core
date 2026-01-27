# Fluxspace Core Scripts Overview

This document provides a high-level overview of all scripts in the Fluxspace Core pipeline, their purposes, and the recommended order of use.

---

## Pipeline Workflow

The typical data processing workflow follows this sequence:

```
1. Data Collection
   └─> mag_to_csv.py
       Output: data/raw/mag_data.csv

2. Validation & Cleaning
   └─> validate_and_diagnosticsV1.py
       Input: data/raw/mag_data.csv
       Output: data/processed/mag_data_clean.csv

3. Anomaly Detection
   └─> compute_local_anomaly_v2.py
       Input: data/processed/mag_data_clean.csv
       Output: data/processed/mag_data_anomaly.csv

4. Visualization (Two Approaches)
   ├─> interpolate_to_Btotal_heatmap.py (B_total field strength)
   │   Input: data/processed/mag_data_clean.csv
   │   Output: data/processed/mag_detection_grid.csv + heatmap.png
   └─> interpolate_to_heatmapV1.py (Anomaly detection)
       Input: data/processed/mag_data_anomaly.csv
       Output: data/exports/mag_data_grid.csv + heatmap.png
```

---

## Script Summaries

### 1. `mag_to_csv.py`
**Purpose:** Collect magnetic field measurements from an MMC5983MA magnetometer sensor and save them to CSV.

**What it does:**
- Connects to MMC5983MA sensor via I2C
- Operates in **auto-grid mode** - automatically generates a grid of measurement points
- At each grid point, prompts user to move sensor and press Enter
- Takes multiple samples per point and averages them for accuracy
- Records Bx, By, Bz components and computes B_total
- Saves data with UTC timestamps to CSV

**Key Features:**
- Configurable grid settings (NX, NY, DX, DY, X0, Y0)
- Error handling with specific exit codes
- Audio feedback (beep) after each measurement
- Automatic CSV header creation

**Output:**
- `data/raw/mag_data.csv` (or custom path)

**Detailed Documentation:** [`mag_to_csv_explanation.md`](./mag_to_csv_explanation.md)

**Example Usage:**
```bash
python3 scripts/mag_to_csv.py
```

---

### 2. `validate_and_diagnosticsV1.py`
**Purpose:** Validate, clean, and generate diagnostics for magnetometer CSV data.

**What it does:**
- Validates CSV structure and required columns (x, y, B_total)
- Cleans missing/invalid data (drops rows with NaN in critical columns)
- Detects outliers using robust z-score statistics (MAD-based)
- Detects spikes (sudden changes between consecutive measurements)
- Generates diagnostic plots:
  - B_total vs time
  - Histogram of B_total
  - XY scatter plot colored by B_total
  - Spike deltas plot
- Creates a text report with statistics and notes

**Key Features:**
- Automatic B_total computation if missing (from Bx, By, Bz)
- Time column detection and parsing
- Quality flag columns: `_flag_outlier`, `_flag_spike`, `_flag_any`
- Optional outlier removal with `--drop-outliers` flag
- Configurable thresholds for outlier and spike detection

**Input:**
- `data/raw/mag_data.csv` (or any magnetometer CSV)

**Outputs (in `data/processed/` by default):**
- `<stem>_clean.csv` - Cleaned data with flag columns
- `<stem>_report.txt` - Text report with statistics
- `<stem>_Btotal_vs_time.png` - Time series plot
- `<stem>_Btotal_hist.png` - Histogram
- `<stem>_scatter_xy_colored.png` - Spatial scatter plot
- `<stem>_spike_deltas.png` - Spike detection plot

**Detailed Documentation:** [`validate_and_diagnostics_explanation.md`](./validate_and_diagnostics_explanation.md)

**Example Usage:**
```bash
python3 scripts/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv
python3 scripts/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv --drop-outliers --z-thresh 5.0
```

---

### 3. `compute_local_anomaly_v2.py`
**Purpose:** Detect local magnetic anomalies by comparing each point to its neighborhood rather than the global average.

**What it does:**
- Reads cleaned CSV data (typically from `validate_and_diagnosticsV1.py`)
- For each point, finds all neighbors within a specified radius
- Computes local mean B_total from neighbors
- Calculates anomaly as: `local_anomaly = B_total - local_mean`
- Optionally filters out flagged rows (outliers/spikes)
- Adds three anomaly columns:
  - `local_anomaly` - Raw anomaly value
  - `local_anomaly_abs` - Absolute value
  - `local_anomaly_norm` - Normalized (0-1 scale)
- Optionally generates scatter plot visualization

**Key Features:**
- Command-line interface with flexible arguments
- Respects quality flags from validation step
- Configurable neighborhood radius
- Optional plotting for quick visualization
- Better error handling than v1

**Input:**
- `data/processed/mag_data_clean.csv` (or any CSV with x, y, B_total)

**Output:**
- `<input_stem>_anomaly.csv` - Original data + anomaly columns

**Detailed Documentation:** [`compute_local_anomaly_v2_explanation.md`](./compute_local_anomaly_v2_explanation.md)

**Example Usage:**
```bash
python3 scripts/compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --radius 0.30 --plot
```

---

### 4a. `interpolate_to_Btotal_heatmap.py`
**Purpose:** Interpolate B_total (magnetic field strength) onto a regular grid and generate heatmap visualizations.

**What it does:**
- Takes scattered (x, y, B_total) points from CSV
- Interpolates B_total values onto a regular grid using IDW (Inverse Distance Weighting)
- Exports grid data as CSV
- Generates heatmap PNG visualization
- Supports unit conversion (gauss ↔ microtesla)

**Key Features:**
- Vectorized IDW interpolator (efficient for large grids)
- Unit conversion support (gauss to microtesla)
- Configurable grid spacing and interpolation power
- Designed specifically for B_total visualization

**Input:**
- `data/processed/mag_data_clean.csv` (or any CSV with x, y, B_total)

**Outputs:**
- `<prefix>_grid.csv` - Regular grid with interpolated B_total values
- `<prefix>_heatmap.png` - Visual heatmap showing field strength

**Detailed Documentation:** [`interpolate_to_Btotal_heatmap_explanation.md`](./interpolate_to_Btotal_heatmap_explanation.md)

**Example Usage:**
```bash
python3 scripts/interpolate_to_Btotal_heatmap.py --in data/processed/mag_data_clean.csv --units uT
python3 scripts/interpolate_to_Btotal_heatmap.py --in data/processed/mag_data_clean.csv --grid-step 0.01 --units uT
```

**When to use:** Use this script when you want to visualize the **absolute magnetic field strength** across the area. This is useful for magnetic detection, field mapping, or understanding the overall field distribution.

---

### 4b. `interpolate_to_heatmapV1.py`
**Purpose:** Interpolate anomaly detection results onto a regular grid and generate heatmap visualizations.

**What it does:**
- Takes scattered (x, y, value) points from CSV (typically with `local_anomaly` column)
- Interpolates anomaly values onto a regular grid using IDW (Inverse Distance Weighting)
- Exports grid data as CSV
- Generates heatmap PNG visualization
- Configurable grid resolution and interpolation power
- Supports percentile clipping for better visualization

**Key Features:**
- Lightweight IDW interpolator (no SciPy required)
- Flexible grid spacing options
- Tunable interpolation power parameter
- Percentile-based color scale clipping
- Quick preview heatmap generation

**Input:**
- `data/processed/mag_data_anomaly.csv` (or any CSV with x, y, and value column like `local_anomaly`)

**Outputs:**
- `<stem>_grid.csv` - Regular grid with interpolated values
- `<stem>_heatmap.png` - Visual heatmap

**Detailed Documentation:** [`interpolate_to_heatmap_explanation.md`](./interpolate_to_heatmap_explanation.md)

**Example Usage:**
```bash
python3 scripts/interpolate_to_heatmapV1.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly
python3 scripts/interpolate_to_heatmapV1.py --in data/processed/mag_data_anomaly.csv --grid-step 0.05
```

**When to use:** Use this script when you want to visualize **local anomalies** (deviations from neighborhood). This is useful for anomaly detection, finding hot spots/cold spots, or identifying magnetic disturbances.

---

## Additional Scripts

### `compute_local_anomaly_v1.py`
**Purpose:** Original version of local anomaly computation (simpler, no CLI).

**Status:** Superseded by `compute_local_anomaly_v2.py` (recommended to use v2)

**Detailed Documentation:** [`compute_local_anomaly_explanation.md`](./compute_local_anomaly_explanation.md)

---

### `calibrate_magnetometerV1.py`
**Purpose:** (Placeholder - functionality to be implemented)

---

### `run_metadataV1.py`
**Purpose:** (Placeholder - functionality to be implemented)

---

## Complete Workflow Example

Here's a complete example of running the entire pipeline:

```bash
# Step 1: Collect data
python3 scripts/mag_to_csv.py
# Output: data/raw/mag_data.csv

# Step 2: Validate and clean
python3 scripts/validate_and_diagnosticsV1.py --in data/raw/mag_data.csv --drop-outliers
# Output: data/processed/mag_data_clean.csv + diagnostics

# Step 3: Compute anomalies
python3 scripts/compute_local_anomaly_v2.py --in data/processed/mag_data_clean.csv --radius 0.30 --plot
# Output: data/processed/mag_data_anomaly.csv

# Step 4a: Create B_total heatmap (field strength visualization)
python3 scripts/interpolate_to_Btotal_heatmap.py --in data/processed/mag_data_clean.csv --units uT --grid-step 0.05
# Output: data/processed/mag_detection_grid.csv + mag_detection_heatmap.png

# Step 4b: Create anomaly heatmap (anomaly detection visualization)
python3 scripts/interpolate_to_heatmapV1.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly --grid-step 0.05
# Output: data/exports/mag_data_grid.csv + mag_data_heatmap.png
```

---

## Data Directory Structure

The pipeline follows a clear data flow through organized directories:

```
data/
├── raw/              # Original sensor data (from mag_to_csv.py)
├── processed/        # Cleaned and analyzed data (from validate + anomaly scripts)
└── exports/          # Final outputs (grids, heatmaps)
```

**Flow:** `raw/` → `processed/` → `exports/`

---

## Key Concepts

### Auto-Grid Mode
`mag_to_csv.py` uses an auto-grid system where you configure:
- `NX`, `NY`: Number of points in X and Y directions
- `DX`, `DY`: Spacing between points (in meters)
- `X0`, `Y0`: Starting coordinates

The script automatically calculates each grid point and prompts you to move the sensor there.

### Quality Flags
`validate_and_diagnosticsV1.py` adds flag columns to identify problematic data:
- `_flag_outlier`: Points with extreme B_total values (robust z-score)
- `_flag_spike`: Points with sudden jumps between consecutive measurements
- `_flag_any`: Combined flag (outlier OR spike)

These flags can be used to filter data in subsequent steps.

### Local Anomalies
Unlike global anomalies (comparing to overall mean), local anomalies compare each point to its nearby neighbors. This helps detect:
- Small-scale variations hidden by global trends
- Regional magnetic field differences
- Localized sources of magnetic disturbance

### IDW Interpolation
Inverse Distance Weighting assigns values to grid points based on:
- Distance to nearby measurement points
- A power parameter (default: 2.0) that controls influence decay
- Closer points have more influence than distant ones

### Two Heatmap Approaches
The pipeline provides two different heatmap scripts for different visualization purposes:

1. **`interpolate_to_Btotal_heatmap.py`**: Visualizes **absolute magnetic field strength (B_total)**
   - Shows the actual field strength across the area
   - Supports unit conversion (gauss ↔ microtesla)
   - Use for: Field mapping, magnetic detection, understanding overall field distribution

2. **`interpolate_to_heatmapV1.py`**: Visualizes **local anomalies** (deviations from neighborhood)
   - Shows where the field differs from nearby areas
   - Highlights hot spots and cold spots
   - Use for: Anomaly detection, finding magnetic disturbances, identifying localized features

---

## Getting Help

For detailed explanations of each script, see the individual documentation files:
- [`mag_to_csv_explanation.md`](./mag_to_csv_explanation.md)
- [`validate_and_diagnostics_explanation.md`](./validate_and_diagnostics_explanation.md)
- [`compute_local_anomaly_v2_explanation.md`](./compute_local_anomaly_v2_explanation.md)
- [`compute_local_anomaly_explanation.md`](./compute_local_anomaly_explanation.md) (v1)
- [`interpolate_to_Btotal_heatmap_explanation.md`](./interpolate_to_Btotal_heatmap_explanation.md)
- [`interpolate_to_heatmap_explanation.md`](./interpolate_to_heatmap_explanation.md)

Each documentation file includes:
- Complete line-by-line code explanations
- Key concepts and algorithms
- Examples with sample data
- Tips for tuning parameters

---

## Notes

- All scripts use Python 3 and require various dependencies (pandas, numpy, matplotlib, etc.)
- Scripts are designed to be run from the command line
- Most scripts support `--help` flag for argument information
- Error handling includes specific exit codes for automation/scripting
- Output file naming follows consistent patterns (e.g., `<stem>_clean.csv`, `<stem>_anomaly.csv`)

