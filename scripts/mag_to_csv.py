#!/usr/bin/env python3
import os
import csv
import time
import math
import sys
from datetime import datetime, timezone

import qwiic_mmc5983ma


CSV_PATH = "data/raw/mag_data.csv"

# Point-capture settings (tweak anytime)
SAMPLES_PER_POINT = 100        # how many samples to average per grid point
SAMPLE_DELAY_S = 0.01          # delay between samples (0.01s = ~100 Hz loop)

# ---- GRID SETTINGS ----
# 5x5 ft ≈ 1.52 m. With 0.20 m spacing, use 9 points per side (~1.60 m span).
DX = 0.05   # meters between points in x
DY = 0.05   # meters between points in y
NX = 5      # number of points in x direction
NY = 5      # number of points in y direction
X0 = 0.0    # starting x (meters)
Y0 = 0.0    # starting y (meters)
# ------------------------------------------

def utc_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


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
                    needs_header = True  # Recreate if header is wrong
        except (IOError, PermissionError) as e:
            raise RuntimeError(f"Cannot read file {path}: {e}")
    
    if needs_header:
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                # Keep x,y,B_total for your anomaly script; extra columns are fine.
                w.writerow(["time", "x", "y", "Bx", "By", "Bz", "B_total", "units"])
        except (IOError, OSError, PermissionError) as e:
            raise RuntimeError(f"Cannot write to file {path}: {e}")
    return needs_header


def connect_sensor():
    mag = qwiic_mmc5983ma.QwiicMMC5983MA()
    if not mag.is_connected():
        raise RuntimeError(
            "MMC5983MA not detected on I2C. Check wiring/Qwiic HAT and that I2C is enabled."
        )
    mag.begin()
    return mag


def read_avg_xyz_gauss(mag, n=SAMPLES_PER_POINT, delay_s=SAMPLE_DELAY_S):
    """Read N samples of (x,y,z) in gauss and return the averages."""
    sx = sy = sz = 0.0
    for i in range(n):
        try:
            x, y, z = mag.get_measurement_xyz_gauss()  # SparkFun API
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


def append_row(path, row):
    try:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)
    except (IOError, OSError, PermissionError) as e:
        raise RuntimeError(f"Cannot append to file {path}: {e}")


def beep():
    print("\a", end="", flush=True)  # ASCII bell


def main() -> int:
    print("\n=== MMC5983MA -> CSV Logger (Point Capture Mode) ===")
    print(f"Output file: {CSV_PATH}")

    try:
        ensure_csv_header(CSV_PATH)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    try:
        mag = connect_sensor()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

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
                return 0

            print(f"  Sampling {SAMPLES_PER_POINT} readings...")
            try:
                bx, by, bz = read_avg_xyz_gauss(mag)
            except RuntimeError as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                print("  Skipping this measurement. You can re-run later for missing points.")
                continue

            b_total = math.sqrt(bx * bx + by * by + bz * bz)

            row = [utc_iso(), x, y, bx, by, bz, b_total, "gauss"]
            try:
                append_row(CSV_PATH, row)
            except RuntimeError as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                print("  Measurement taken but could not be saved!", file=sys.stderr)
                print("  Check disk space and file permissions.", file=sys.stderr)
                return 3

            beep()
            print(f"  Saved: x={x:.2f}, y={y:.2f}, B_total={b_total:.6f} gauss\n")

    print("Grid complete. Done.")
    return 0


if __name__ == "__main__":
    # Only runs when this file is executed directly (not imported)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
