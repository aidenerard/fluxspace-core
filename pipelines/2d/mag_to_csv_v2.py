#!/usr/bin/env python3
"""mag_to_csv_v2.py

Continuous MMC5983MA magnetometer logger for 3D fusion.

Run this while you capture a 3D scan/trajectory on your phone (Polycam Raw Data / RTAB-Map, etc.).
This logger does NOT assume a 2D grid. It writes a timestamped stream of B-field samples.

Output CSV columns:
  t_unix_ns, t_utc_iso, t_rel_s, bx, by, bz, b_total, units, row_type, note

- row_type is either:
    SAMPLE -> a normal magnetometer sample row
    MARK   -> a user marker row (press Enter; you can type a label then Enter)
    INFO   -> an informational row (logged at start, and if a sensor read fails)

Recommended capture workflow:
1) Rigidly mount magnetometer to the phone.
2) Start this logger:
     python3 mag_to_csv_v2.py --out data/raw/mag_run01.csv --hz 80 --units uT
3) Press Enter to insert a MARK row (or type a label like 'start' then Enter), then start phone scan.
4) Optionally do a distinctive 1-2 second motion (small shake) to create an alignment signature.
5) Press Enter near end to insert a MARK row (e.g., 'end'), then stop both.

Notes on time alignment:
- t_unix_ns is wall-clock Unix time; your phone exports will have wall-clock time (or a relative time).
- If your devices aren't time-synced, use the MARK rows + motion signature to align.

Dependencies:
- qwiic_mmc5983ma (SparkFun) as used in your existing mag_to_csv.py
"""

import os
import csv
import time
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
import argparse
import threading
import queue

import qwiic_mmc5983ma

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


EXPECTED_HEADER = [
    "t_unix_ns",
    "t_utc_iso",
    "t_rel_s",
    "bx",
    "by",
    "bz",
    "b_total",
    "units",
    "row_type",
    "note",
]


def parse_args():
    p = argparse.ArgumentParser(description="MMC5983MA -> CSV Logger (Continuous Mode for 3D fusion)")
    p.add_argument("--out", type=str, default="data/raw/mag_log.csv", help="Output CSV path")
    p.add_argument("--hz", type=float, default=80.0, help="Target logging rate (Hz)")
    p.add_argument("--duration", type=float, default=0.0, help="Duration seconds (0 = run until Ctrl+C)")
    p.add_argument("--samples", type=int, default=1, help="Samples to average per logged row")
    p.add_argument("--sample-delay", type=float, default=0.0, help="Delay between samples inside an average (seconds)")
    p.add_argument("--units", choices=["gauss", "uT"], default="gauss", help="Output units for Bx/By/Bz/B_total")
    p.add_argument("--flush-every", type=int, default=1, help="Flush file every N rows (safer, slightly slower)")
    p.add_argument("--beep-on-mark", action="store_true", help="ASCII bell on MARK rows")
    p.add_argument("--note", type=str, default="", help="Optional note to write in an INFO row at start")
    p.add_argument("--force-header", action="store_true", help="Overwrite/replace file if header mismatch")
    return p.parse_args()


def utc_iso_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def ensure_csv_header(path: str, force_header: bool) -> None:
    """Create file + header if it doesn't exist or is empty.
    If file exists with different header, error unless --force-header is used.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(EXPECTED_HEADER)
        return

    # Validate existing header
    with open(path, "r", newline="") as f:
        first = f.readline().strip()

    if first != ",".join(EXPECTED_HEADER):
        if not force_header:
            raise RuntimeError(
                f"CSV header mismatch in {path}.\n"
                f"Expected: {','.join(EXPECTED_HEADER)}\n"
                f"Found:    {first}\n"
                f"Use --force-header to overwrite."
            )
        # Overwrite file with correct header
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(EXPECTED_HEADER)


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
            x, y, z = mag.get_measurement_xyz_gauss()
            sx += x
            sy += y
            sz += z
        except Exception as e:
            raise RuntimeError(f"Failed to read sensor at sample {i+1}/{n}: {e}")
        if delay_s > 0:
            time.sleep(delay_s)
    return sx / n, sy / n, sz / n


def gauss_to_microtesla(v_gauss: float) -> float:
    # 1 gauss = 100 microtesla
    return v_gauss * 100.0


class StdinMarkerListener(threading.Thread):
    """Listens for Enter presses (and optional label text) without blocking the main loop."""

    def __init__(self, out_queue: "queue.Queue[str]", stop_event: threading.Event):
        super().__init__(daemon=True)
        self._q = out_queue
        self._stop = stop_event

    def run(self):
        while not self._stop.is_set():
            try:
                line = sys.stdin.readline()
                if line == "":
                    # stdin closed
                    return
                label = line.strip()
                if label.lower() in ("q", "quit", "exit", "stop"):
                    self._q.put("__QUIT__")
                    return
                # Empty line becomes a generic marker
                self._q.put(label if label else "MARK")
            except Exception:
                return


def beep():
    print("\a", end="", flush=True)


def main() -> int:
    args = parse_args()
    if not os.path.isabs(args.out):
        args.out = str(_REPO_ROOT / args.out)

    if args.hz <= 0:
        print("ERROR: --hz must be > 0", file=sys.stderr)
        return 2
    if args.samples <= 0:
        print("ERROR: --samples must be > 0", file=sys.stderr)
        return 2

    try:
        ensure_csv_header(args.out, force_header=args.force_header)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    try:
        mag = connect_sensor()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    stop_event = threading.Event()
    marker_q: "queue.Queue[str]" = queue.Queue()
    listener = StdinMarkerListener(marker_q, stop_event)
    listener.start()

    print("\n=== MMC5983MA -> CSV Logger (Continuous Mode) ===")
    print(f"Output file: {args.out}")
    print(f"Target rate: {args.hz:.2f} Hz  |  Avg samples/row: {args.samples}  |  In-avg delay: {args.sample_delay}s")
    print(f"Units: {args.units}")
    print("Markers: press Enter to add a MARK row; or type a label then Enter (e.g., 'start', 'end').")
    print("Quit: Ctrl+C (or type 'q' then Enter).")
    print()

    t0_ns = time.time_ns()
    t0_monotonic = time.monotonic()

    rows_since_flush = 0
    total_rows = 0
    dropped_est = 0

    interval = 1.0 / args.hz
    next_tick = time.monotonic()

    # Write an INFO row at start
    try:
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            info_note = args.note if args.note else "logger_start"
            w.writerow([str(t0_ns), utc_iso_now(), "0.0", "", "", "", "", args.units, "INFO", info_note])
            f.flush()
    except Exception as e:
        print(f"ERROR: Could not write INFO row: {e}", file=sys.stderr)
        return 3

    start_wall = time.time()

    try:
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)

            while True:
                # Duration check
                if args.duration and (time.time() - start_wall) >= args.duration:
                    break

                # Handle queued markers
                while True:
                    try:
                        m = marker_q.get_nowait()
                    except queue.Empty:
                        break

                    if m == "__QUIT__":
                        stop_event.set()
                        return 0

                    now_ns = time.time_ns()
                    t_rel = time.monotonic() - t0_monotonic
                    w.writerow([str(now_ns), utc_iso_now(), f"{t_rel:.6f}", "", "", "", "", args.units, "MARK", m])
                    total_rows += 1
                    rows_since_flush += 1
                    if args.beep_on_mark:
                        beep()

                # Pace to target rate
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(min(0.005, next_tick - now))
                    continue

                lateness = now - next_tick
                if lateness > interval:
                    dropped_est += int(lateness / interval)

                next_tick += interval

                # Sample magnetometer
                try:
                    bx, by, bz = read_avg_xyz_gauss(mag, n=args.samples, delay_s=args.sample_delay)
                except Exception as e:
                    now_ns = time.time_ns()
                    t_rel = time.monotonic() - t0_monotonic
                    w.writerow([str(now_ns), utc_iso_now(), f"{t_rel:.6f}", "", "", "", "", args.units, "INFO", f"read_error: {e}"])
                    total_rows += 1
                    rows_since_flush += 1
                    continue

                if args.units == "uT":
                    bx = gauss_to_microtesla(bx)
                    by = gauss_to_microtesla(by)
                    bz = gauss_to_microtesla(bz)

                b_total = math.sqrt(bx * bx + by * by + bz * bz)

                now_ns = time.time_ns()
                t_rel = time.monotonic() - t0_monotonic

                w.writerow(
                    [
                        str(now_ns),
                        utc_iso_now(),
                        f"{t_rel:.6f}",
                        f"{bx:.9f}",
                        f"{by:.9f}",
                        f"{bz:.9f}",
                        f"{b_total:.9f}",
                        args.units,
                        "SAMPLE",
                        "",
                    ]
                )

                total_rows += 1
                rows_since_flush += 1

                if args.flush_every > 0 and rows_since_flush >= args.flush_every:
                    f.flush()
                    rows_since_flush = 0

                # Print status every ~2 seconds
                if total_rows % int(max(1, args.hz * 2)) == 0:
                    elapsed = time.monotonic() - t0_monotonic
                    eff_hz = (total_rows / elapsed) if elapsed > 0 else 0.0
                    print(
                        f"Rows: {total_rows} | elapsed: {elapsed:.1f}s | eff: {eff_hz:.1f} Hz | dropped_est: {dropped_est}",
                        end="\r",
                        flush=True,
                    )

    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    finally:
        stop_event.set()

    print(f"\nSaved {total_rows} rows to {args.out}")
    if dropped_est:
        print(
            f"Note: estimated dropped ticks due to lateness: {dropped_est} (if averaging is heavy, reduce --hz or --samples)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

