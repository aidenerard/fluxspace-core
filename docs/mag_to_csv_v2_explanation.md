## Complete Explanation of `mag_to_csv_v2.py`

This document explains the **continuous magnetometer logger** `mag_to_csv_v2.py` and how it fits into your 3D-fusion workflow.

---

### 1. Purpose and High-Level Behavior

- **Purpose**: Log a **continuous time series** of MMC5983MA readings for later fusion with a 3D trajectory (Polycam Raw Data, RTAB-Map, etc.).
- **Key differences from `mag_to_csv.py`:**
  - `mag_to_csv.py`: 2D **grid survey** (auto-grid, one averaged sample per grid point).
  - `mag_to_csv_v2.py`: **free-motion, continuous stream**, no grid assumptions.
- **Outputs**:
  - High-resolution timestamps (`t_unix_ns`, `t_rel_s`, `t_utc_iso`)
  - Full vector field + magnitude (`bx`, `by`, `bz`, `b_total`)
  - Row type + notes (`row_type`, `note`) for marks and info rows.

CSV columns (header is fixed and validated):

```text
t_unix_ns, t_utc_iso, t_rel_s, bx, by, bz, b_total, units, row_type, note
```

- **`row_type`**:
  - `SAMPLE` – normal magnetometer sample row
  - `MARK` – user marker row (Enter, or label + Enter)
  - `INFO` – informational row (start, read errors, etc.)

---

### 2. How to Run It

Basic run (while scanning with Polycam / RTAB-Map):

```bash
python3 scripts/mag_to_csv_v2.py --out data/raw/mag_run01.csv --hz 80 --units uT --samples 1
```

While it’s running:

- Press **Enter** to drop a marker row (`row_type = MARK`, `note = "MARK"`).
- Or type a label then Enter (e.g. `start`, `end`, `doorway`).
- Quit with **Ctrl+C**, or type `q` then Enter.

If reads are noisy, average multiple sensor readings per logged row:

```bash
python3 scripts/mag_to_csv_v2.py \
  --out data/raw/mag_run01.csv \
  --hz 50 \
  --units uT \
  --samples 5 \
  --sample-delay 0.002
```

Notes:
- Higher `--samples` and higher `--hz` both increase workload; if you increase `--samples`, consider reducing `--hz`.
- The script prints an **effective Hz** and an **estimated dropped tick count**; if `dropped_est` grows, reduce `--hz` or `--samples`.

Key arguments:

- `--out`: Output CSV path (directories are created if needed).
- `--hz`: Target logging rate in Hz.
- `--duration`: Optional duration in seconds (0 = run until Ctrl+C).
- `--samples`: Number of sensor reads per logged row (averaged).
- `--sample-delay`: Delay (seconds) between samples inside the average.
- `--units`: `"gauss"` or `"uT"` (1 gauss = 100 µT).
- `--flush-every`: Flush file every N rows (safer; default 1).
- `--beep-on-mark`: Emit ASCII bell on each `MARK` row.
- `--note`: Optional note string written into the first `INFO` row (`logger_start` if empty).
- `--force-header`: Overwrite existing file if header does not match the expected schema.

---

### 3. What the Script Logs

Each **SAMPLE** row contains:

- `t_unix_ns`: Wall-clock Unix time in nanoseconds (int, as string).
- `t_utc_iso`: UTC timestamp in ISO 8601 (millisecond resolution).
- `t_rel_s`: Relative time in seconds since logger start (float, string).
- `bx, by, bz`: Field components (in `units` chosen).
- `b_total`: Magnitude \\(\\sqrt{bx^2 + by^2 + bz^2}\\) in same units.
- `units`: `"gauss"` or `"uT"`.
- `row_type = "SAMPLE"`.
- `note`: Empty for samples.

Each **MARK** row:

- Uses current timestamps (`t_unix_ns`, `t_rel_s`, `t_utc_iso`).
- Leaves `bx/by/bz/b_total` empty.
- Sets `units` (so downstream code knows the file units).
- Sets `row_type = "MARK"`.
- Sets `note` to your label (or `"MARK"` if you just pressed Enter).

At start, an **INFO** row is written:

- `t_unix_ns` and `t_rel_s = 0.0` at logger start.
- `row_type = "INFO"`.
- `note = <your --note>` or `"logger_start"` if none.

If a sensor read fails, another **INFO** row is written with:

- `note` like `read_error: <exception>`.

The header is enforced via `EXPECTED_HEADER`. If the file already exists:

- If header matches exactly, rows are **appended**.
- If header differs:
  - Error by default.
  - If `--force-header` is set, the file is overwritten with the correct header.

---

### 4. Why This is 3D-Fusion-Ready

For 3D fusion, you will later combine:

- A **camera or LiDAR trajectory** (Polycam Raw Data / RTAB-Map pose log) with timestamps.
- This **magnetometer CSV** (`mag_run01.csv`).

Because this logger provides:

- High-resolution **wall-clock time** (`t_unix_ns`, `t_utc_iso`).
- A consistent **relative time base** (`t_rel_s`).
- Dense, continuous **Bx/By/Bz + magnitude** stream.
- Explicit **MARK rows** and **INFO rows** for alignment.

…you can:

1. Align the phone trajectory and magnetometer time bases using:
   - Wall-clock time (if clocks are reasonably in sync), **and/or**
   - Distinctive motions + `MARK` labels (e.g., a shake right after typing `start`).
2. Interpolate the phone pose trajectory to each magnetometer timestamp.
3. Transform magnetometer readings into the world frame and export `mag_world.csv`.

In contrast, `mag_to_csv.py` is designed for a **structured 2D grid survey**, not free-motion 3D fusion.

---

### 5. Typical Workflow with `mag_to_csv_v2.py`

On the Pi (sensor rig):

```bash
cd ~/fluxspace-core
source ~/fluxenv/bin/activate   # if using a venv

python3 scripts/mag_to_csv_v2.py \
  --out data/raw/mag_run01.csv \
  --hz 80 \
  --units uT \
  --samples 1
```

Suggested capture pattern:

1. **Mount** the magnetometer rigidly to the phone/camera rig.
2. **Start logger** (`mag_to_csv_v2.py`).
3. **Press Enter** (or type `start` + Enter) right before you start the 3D scan.
4. Perform the scan / trajectory capture as usual.
5. Optionally do a distinctive motion (e.g., small shake) near the start/end.
6. **Press Enter** again (e.g., `end`) when you finish.
7. Stop logger with **Ctrl+C**.

Result:

- `data/raw/mag_run01.csv` with continuous magnetometer data + marks suitable for offline time alignment and 3D fusion.

---

### 6. Integration with the Rest of the Pipeline

`mag_to_csv_v2.py` is meant to **augment**, not replace, your existing 2D-grid pipeline:

- Use **`mag_to_csv.py`** when:
  - You want a regular 2D grid for anomaly maps and B_total heatmaps.
  - You’re running the classic pipeline:
    - `mag_to_csv.py` → `validate_and_diagnosticsV1.py` → `compute_local_anomaly_v2.py` → heatmaps.

- Use **`mag_to_csv_v2.py`** when:
  - You’re doing a free-motion scan with a 3D trajectory.
  - You care about **pose-aligned 3D magnetometer maps** rather than a planar grid.

You can keep both styles in the same project and choose based on experiment type.

