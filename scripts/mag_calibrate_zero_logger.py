#!/usr/bin/env python3
"""
mag_calibrate_zero_logger.py

Fluxspace: Live magnetometer calibration + "zero" (baseline) + continuous logging.

Designed for the exact use-case you're doing now:
- Phone provides pose (Polycam / RTAB-Map)
- Magnetometer provides field samples
- You need reliable timestamps and a stable baseline BEFORE the run begins.

What this script does:
  Phase 1) CALIBRATE (hard-iron + simple soft-iron diag scaling via min/max ranges)
    - You move/rotate the sensor (ideally the whole rigid rig) through many orientations
      for --calib-seconds. Keep the phone + any nearby metal in its normal "run" position.
  Phase 2) ZERO (baseline)
    - Hold the sensor still in the "start" pose for --zero-seconds. We record the mean corrected
      vector as baseline, and subtract it during logging.
  Phase 3) LOG
    - Continuous timestamped logging (Ctrl+C to stop).
    - Press Enter any time to add a MARK row (or type a label like "start"/"end" then Enter).

Important reality check:
- If the magnetometer is very close to the phone, the phone's own magnetic field can dominate.
  Calibration/zeroing can remove a *static* bias, but if your sensor-to-phone geometry changes during
  the run, the phone-induced field changes and will look like an anomaly. Rigid mounting matters.

CSV outputs both raw and corrected/zeroed values:
  t_unix_ns, t_utc_iso, t_rel_s,
  raw_bx, raw_by, raw_bz, raw_mag,
  corr_bx, corr_by, corr_bz, corr_mag,
  zero_bx, zero_by, zero_bz, zero_mag,
  units, row_type, note

JSON calibration (optional) includes offset, scale factors, baseline, and metadata.
"""

import os
import csv
import sys
import time
import math
import json
import argparse
import threading
import queue
from datetime import datetime, timezone

import qwiic_mmc5983ma


GAUSS_TO_UT = 100.0  # 1 gauss = 100 microtesla


def parse_args():
    p = argparse.ArgumentParser(
        description="MMC5983MA live calibrate + zero + continuous logger (for 3D fusion)."
    )
    p.add_argument("--out", type=str, default="data/raw/mag_run.csv", help="Output CSV path")
    p.add_argument("--hz", type=float, default=80.0, help="Target sample rate during LOG phase")
    p.add_argument("--units", type=str, default="uT", choices=["uT", "gauss"], help="Output units")
    p.add_argument("--samples", type=int, default=1, help="Average N sensor reads per logged sample")
    p.add_argument("--sample-delay", type=float, default=0.0, help="Delay between averaged reads (s)")

    p.add_argument("--calib-seconds", type=float, default=20.0, help="Calibration motion duration (s)")
    p.add_argument("--zero-seconds", type=float, default=3.0, help="Baseline averaging duration (s)")

    p.add_argument(
        "--no-softiron",
        action="store_true",
        help="Disable soft-iron scaling (only hard-iron offset).",
    )
    p.add_argument(
        "--save-cal",
        type=str,
        default="",
        help="Optional: save calibration+baseline JSON to this path",
    )

    p.add_argument(
        "--expected-field-ut",
        type=float,
        default=0.0,
        help="Optional: rescale corrected magnitude to this value (uT). 0 disables.",
    )
    return p.parse_args()


def ensure_parent_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def connect_sensor():
    mag = qwiic_mmc5983ma.QwiicMMC5983MA()
    if not mag.is_connected():
        raise RuntimeError(
            "MMC5983MA not detected on I2C. Check wiring/Qwiic adapter and that I2C is enabled."
        )
    mag.begin()
    return mag


def read_xyz_gauss(mag):
    # SparkFun API returns gauss
    x, y, z = mag.get_measurement_xyz_gauss()
    return float(x), float(y), float(z)


def convert_units(x_g, y_g, z_g, units: str):
    if units == "gauss":
        return x_g, y_g, z_g
    return x_g * GAUSS_TO_UT, y_g * GAUSS_TO_UT, z_g * GAUSS_TO_UT


def vec_mag(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


def read_avg_xyz(mag, n: int, delay_s: float, units: str):
    sx = sy = sz = 0.0
    for i in range(n):
        try:
            xg, yg, zg = read_xyz_gauss(mag)
        except Exception as e:
            raise RuntimeError(f"Failed to read sensor at sample {i+1}/{n}: {e}")
        x, y, z = convert_units(xg, yg, zg, units)
        sx += x
        sy += y
        sz += z
        if delay_s > 0:
            time.sleep(delay_s)
    return sx / n, sy / n, sz / n


def compute_calibration_from_minmax(min_v, max_v, use_softiron: bool):
    """
    Hard-iron offset: center of min/max box.
    Soft-iron (simple): diagonal scale so each axis range matches average range.
    """
    off = [(max_v[i] + min_v[i]) * 0.5 for i in range(3)]
    if not use_softiron:
        scales = [1.0, 1.0, 1.0]
        return off, scales

    radii = [(max_v[i] - min_v[i]) * 0.5 for i in range(3)]
    # Avoid divide-by-zero
    radii = [r if abs(r) > 1e-9 else 1e-9 for r in radii]
    avg_r = sum(radii) / 3.0
    scales = [avg_r / r for r in radii]
    return off, scales


def apply_calibration(x, y, z, offset, scales):
    # diag soft-iron
    xc = (x - offset[0]) * scales[0]
    yc = (y - offset[1]) * scales[1]
    zc = (z - offset[2]) * scales[2]
    return xc, yc, zc


def maybe_rescale_to_expected(xc, yc, zc, expected_field):
    if expected_field <= 0:
        return xc, yc, zc
    m = vec_mag(xc, yc, zc)
    if m <= 1e-9:
        return xc, yc, zc
    k = expected_field / m
    return xc * k, yc * k, zc * k


def write_csv_header(path: str):
    ensure_parent_dir(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t_unix_ns",
                "t_utc_iso",
                "t_rel_s",
                "raw_bx",
                "raw_by",
                "raw_bz",
                "raw_mag",
                "corr_bx",
                "corr_by",
                "corr_bz",
                "corr_mag",
                "zero_bx",
                "zero_by",
                "zero_bz",
                "zero_mag",
                "units",
                "row_type",
                "note",
            ]
        )


def append_row(path: str, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()


def start_marker_thread(marker_q: "queue.Queue[str]"):
    """
    Reads stdin lines to create MARK rows while logging.
    - Press Enter -> MARK with empty note
    - Type text then Enter -> MARK with that text
    - Type 'q' then Enter -> request stop
    """
    def _run():
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if line is None:
                continue
            note = line.strip()
            marker_q.put(note)
            if note.lower() == "q":
                break

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def main() -> int:
    args = parse_args()

    print("\n=== MMC5983MA Live Calibrate + Zero + Logger ===")
    print(f"Output CSV: {args.out}")
    print(f"Units: {args.units}")
    print(f"Calibrate seconds: {args.calib_seconds:.1f}")
    print(f"Zero seconds: {args.zero_seconds:.1f}")
    print(f"Log Hz target: {args.hz:.1f}")
    if args.expected_field_ut > 0:
        print(f"Expected field rescale: {args.expected_field_ut:.2f} uT (applied after calibration)")

    try:
        mag = connect_sensor()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Phase 1: CALIBRATE
    print("\n[1/3] CALIBRATION")
    print("Move/rotate the sensor through many orientations now.")
    print("Keep the phone + any nearby metal in the same configuration you'll use during the run.")
    print("Tip: slow figure-8 + roll/pitch/yaw coverage works well.")
    print("Starting in 3...2...1...\n")

    t0 = time.time()
    min_v = [float("inf"), float("inf"), float("inf")]
    max_v = [float("-inf"), float("-inf"), float("-inf")]
    n_cal = 0
    last_print = t0

    while True:
        t = time.time()
        if t - t0 >= args.calib_seconds:
            break
        try:
            x, y, z = read_avg_xyz(mag, args.samples, args.sample_delay, args.units)
        except Exception as e:
            print(f"WARNING: read failed during calibration: {e}", file=sys.stderr)
            continue

        min_v[0] = min(min_v[0], x); min_v[1] = min(min_v[1], y); min_v[2] = min(min_v[2], z)
        max_v[0] = max(max_v[0], x); max_v[1] = max(max_v[1], y); max_v[2] = max(max_v[2], z)
        n_cal += 1

        if t - last_print >= 1.0:
            pct = min(100.0, 100.0 * (t - t0) / max(args.calib_seconds, 1e-6))
            print(f"  collecting... {pct:5.1f}%  samples={n_cal}", end="\r", flush=True)
            last_print = t

    print("\nCalibration collection done.")
    use_softiron = not args.no_softiron
    offset, scales = compute_calibration_from_minmax(min_v, max_v, use_softiron)
    print("Calibration parameters:")
    print(f"  hard-iron offset ({args.units}): [{offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}]")
    if use_softiron:
        print(f"  soft-iron diag scales: [{scales[0]:.6f}, {scales[1]:.6f}, {scales[2]:.6f}]")
    else:
        print("  soft-iron: disabled")

    # Phase 2: ZERO baseline
    print("\n[2/3] ZERO / BASELINE")
    print("Hold the sensor still in your intended START pose.")
    print("Starting baseline averaging in 2 seconds...")
    time.sleep(2.0)

    t1 = time.time()
    bx = by = bz = 0.0
    n_zero = 0
    while True:
        t = time.time()
        if t - t1 >= args.zero_seconds:
            break
        try:
            xr, yr, zr = read_avg_xyz(mag, args.samples, args.sample_delay, args.units)
        except Exception as e:
            print(f"WARNING: read failed during zeroing: {e}", file=sys.stderr)
            continue
        xc, yc, zc = apply_calibration(xr, yr, zr, offset, scales)
        xc, yc, zc = maybe_rescale_to_expected(xc, yc, zc, args.expected_field_ut if args.units == "uT" else 0.0)
        bx += xc; by += yc; bz += zc
        n_zero += 1

    if n_zero == 0:
        print("ERROR: Could not collect baseline samples.", file=sys.stderr)
        return 3

    baseline = [bx / n_zero, by / n_zero, bz / n_zero]
    baseline_mag = vec_mag(baseline[0], baseline[1], baseline[2])
    print("Baseline (corrected) vector:")
    print(f"  baseline ({args.units}): [{baseline[0]:.3f}, {baseline[1]:.3f}, {baseline[2]:.3f}]  |B|={baseline_mag:.3f}")

    # Save calibration JSON if requested
    if args.save_cal:
        ensure_parent_dir(args.save_cal)
        cal = {
            "created_utc": now_utc_iso(),
            "units": args.units,
            "hard_iron_offset": {"x": offset[0], "y": offset[1], "z": offset[2]},
            "soft_iron_diag_scales": {"x": scales[0], "y": scales[1], "z": scales[2]},
            "expected_field_ut": args.expected_field_ut,
            "baseline_corrected": {"x": baseline[0], "y": baseline[1], "z": baseline[2]},
            "baseline_corrected_mag": baseline_mag,
            "notes": {
                "method": "minmax hard-iron + diag soft-iron, baseline mean after correction",
                "warning": "If sensor-to-phone geometry changes, phone-induced field changes won't be removed.",
            },
        }
        with open(args.save_cal, "w") as f:
            json.dump(cal, f, indent=2)
        print(f"Saved calibration JSON: {args.save_cal}")

    # Phase 3: LOG
    print("\n[3/3] LOGGING")
    print("Logging now. Press Enter to add a MARK row (or type a label + Enter). Type 'q' + Enter to stop.")
    print("Ctrl+C also stops.\n")

    write_csv_header(args.out)

    # write INFO row for calibration parameters (so the CSV is self-describing)
    t_unix_ns = time.time_ns()
    append_row(
        args.out,
        [
            t_unix_ns,
            now_utc_iso(),
            0.0,
            "", "", "", "",
            "", "", "", "",
            "", "", "", "",
            args.units,
            "INFO",
            f"offset=[{offset[0]:.6f},{offset[1]:.6f},{offset[2]:.6f}] scales=[{scales[0]:.6f},{scales[1]:.6f},{scales[2]:.6f}] baseline=[{baseline[0]:.6f},{baseline[1]:.6f},{baseline[2]:.6f}]",
        ],
    )

    start_ns = time.time_ns()
    marker_q: "queue.Queue[str]" = queue.Queue()
    start_marker_thread(marker_q)

    period_s = 1.0 / max(args.hz, 1e-6)
    next_t = time.perf_counter()
    sample_idx = 0

    try:
        while True:
            # handle markers (non-blocking)
            while not marker_q.empty():
                note = marker_q.get_nowait()
                if note.lower() == "q":
                    raise KeyboardInterrupt
                t_unix_ns = time.time_ns()
                t_rel_s = (t_unix_ns - start_ns) / 1e9
                append_row(
                    args.out,
                    [
                        t_unix_ns,
                        now_utc_iso(),
                        f"{t_rel_s:.6f}",
                        "", "", "", "",
                        "", "", "", "",
                        "", "", "", "",
                        args.units,
                        "MARK",
                        note,
                    ],
                )

            # sample
            t_unix_ns = time.time_ns()
            t_rel_s = (t_unix_ns - start_ns) / 1e9

            xr, yr, zr = read_avg_xyz(mag, args.samples, args.sample_delay, args.units)
            raw_mag = vec_mag(xr, yr, zr)

            xc, yc, zc = apply_calibration(xr, yr, zr, offset, scales)
            xc, yc, zc = maybe_rescale_to_expected(xc, yc, zc, args.expected_field_ut if args.units == "uT" else 0.0)
            corr_mag = vec_mag(xc, yc, zc)

            xz = xc - baseline[0]
            yz = yc - baseline[1]
            zz = zc - baseline[2]
            zero_mag = vec_mag(xz, yz, zz)

            append_row(
                args.out,
                [
                    t_unix_ns,
                    now_utc_iso(),
                    f"{t_rel_s:.6f}",
                    f"{xr:.6f}",
                    f"{yr:.6f}",
                    f"{zr:.6f}",
                    f"{raw_mag:.6f}",
                    f"{xc:.6f}",
                    f"{yc:.6f}",
                    f"{zc:.6f}",
                    f"{corr_mag:.6f}",
                    f"{xz:.6f}",
                    f"{yz:.6f}",
                    f"{zz:.6f}",
                    f"{zero_mag:.6f}",
                    args.units,
                    "SAMPLE",
                    "",
                ],
            )

            sample_idx += 1
            if sample_idx % int(max(args.hz, 1)) == 0:
                # ~once per second at requested hz
                print(f"  samples={sample_idx}  t={t_rel_s:6.1f}s", end="\r", flush=True)

            # timing
            next_t += period_s
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # we're behind; resync
                next_t = time.perf_counter()

    except KeyboardInterrupt:
        print("\nStopping logger.")
    except Exception as e:
        print(f"\nERROR during logging: {e}", file=sys.stderr)
        return 4

    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
