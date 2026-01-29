#!/usr/bin/env python3
import os
import csv
import time
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
import argparse

import qwiic_mmc5983ma

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="MMC5983MA -> CSV Logger (Point Capture Mode)")
    p.add_argument("--out", type=str, default="data/raw/mag_data.csv", help="Output CSV path")
    p.add_argument("--nx", type=int, default=5, help="Number of grid points in X")
    p.add_argument("--ny", type=int, default=5, help="Number of grid points in Y")
    p.add_argument("--dx", type=float, default=0.05, help="Grid spacing in X (meters)")
    p.add_argument("--dy", type=float, default=0.05, help="Grid spacing in Y (meters)")
    p.add_argument("--x0", type=float, default=0.0, help="Grid origin X (meters)")
    p.add_argument("--y0", type=float, default=0.0, help="Grid origin Y (meters)")
    p.add_argument("--samples", type=int, default=100, help="Samples to average per point")
    p.add_argument("--sample-delay", type=float, default=0.01, help="Delay between samples (seconds)")
    return p.parse_args()


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
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time", "x", "y", "Bx", "By", "Bz", "B_total", "units"])
        except (IOError, OSError, PermissionError) as e:
            raise RuntimeError(f"Cannot write to file {path}: {e}")
    return needs_header


def connect_sensor():
    mag = qwiic_mmc5983ma.QwiicMMC5983MA()
    if not mag.is_connected():
        raise RuntimeError(
            "MMC5983MA not detected on I2C. Check wiring/Qwiic adapter and that I2C is enabled."
        )
    mag.begin()
    return mag


def read_avg_xyz_gauss(mag, n: int, delay_s: float):
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
    return sx / n, sy / n, sz / n


def append_row(path, row):
    try:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)
    except (IOError, OSError, PermissionError) as e:
        raise RuntimeError(f"Cannot append to file {path}: {e}")


def beep():
    print("\a", end="", flush=True)  # ASCII bell


def main() -> int:
    args = parse_args()
    csv_path = args.out
    if not os.path.isabs(csv_path):
        csv_path = str(_REPO_ROOT / csv_path)
    nx, ny = args.nx, args.ny
    dx, dy = args.dx, args.dy
    x0, y0 = args.x0, args.y0
    samples_per_point = args.samples
    sample_delay_s = args.sample_delay

    print("\n=== MMC5983MA -> CSV Logger (Point Capture Mode) ===")
    print(f"Output file: {csv_path}")

    try:
        ensure_csv_header(csv_path)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    try:
        mag = connect_sensor()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Auto-grid enabled: NX={nx}, NY={ny}, DX={dx} m, DY={dy} m, X0={x0} m, Y0={y0} m")
    print(f"Samples/point: {samples_per_point}  |  Sample delay: {sample_delay_s}s")
    print("At each prompt, move the sensor to the point and press Enter.")
    print("Type 'q' then Enter to quit early.\n")

    for j in range(ny):
        for i in range(nx):
            x = x0 + i * dx
            y = y0 + j * dy

            user = input(
                f"Point ({i+1}/{nx}, {j+1}/{ny}) -> x={x:.4f}, y={y:.4f}. "
                f"Press Enter to capture (or 'q' to quit): "
            ).strip()

            if user.lower() in ("q", "quit", "exit"):
                print("Done.")
                return 0

            print(f"  Sampling {samples_per_point} readings...")
            try:
                bx, by, bz = read_avg_xyz_gauss(mag, n=samples_per_point, delay_s=sample_delay_s)
            except RuntimeError as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                print("  Skipping this measurement. You can re-run later for missing points.")
                continue

            b_total = math.sqrt(bx * bx + by * by + bz * bz)

            row = [utc_iso(), x, y, bx, by, bz, b_total, "gauss"]
            try:
                append_row(csv_path, row)
            except RuntimeError as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                print("  Measurement taken but could not be saved!", file=sys.stderr)
                print("  Check disk space and file permissions.", file=sys.stderr)
                return 3

            beep()
            print(f"  Saved: x={x:.4f}, y={y:.4f}, B_total={b_total:.6f} gauss\n")

    print("Grid complete. Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
