#!/usr/bin/env python3
"""
polycam_raw_to_trajectory.py

Extract a timestamped camera trajectory from a Polycam Raw Data export folder.
Input: folder containing cameras.json or corrected_cameras (or similar).
Output: trajectory.csv with columns t_rel_s, x, y, z, qx, qy, qz, qw.

Uses Polycam timestamps if present; otherwise derives relative t from frame order
and prints a warning. Matches fluxspace style: argparse, clear prints, sane defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Polycam Raw Data export -> trajectory.csv (t_rel_s, x, y, z, qx, qy, qz, qw)"
    )
    p.add_argument(
        "--in",
        dest="input_dir",
        required=True,
        help="Path to Polycam Raw Data export folder (contains cameras.json or corrected_cameras)",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output trajectory CSV path. Default: <input_dir>/../processed/trajectory.csv or <input_dir>/trajectory.csv",
    )
    return p.parse_args()


def find_cameras_json(folder: Path) -> Path | None:
    """Return path to cameras.json or corrected_cameras.json if present."""
    for name in ("cameras.json", "corrected_cameras.json", "CorrectedCameras.json"):
        p = folder / name
        if p.exists():
            return p
    # Check one level down
    for sub in folder.iterdir():
        if sub.is_dir():
            for name in ("cameras.json", "corrected_cameras.json"):
                q = sub / name
                if q.exists():
                    return q
    return None


def load_polycam_cameras(path: Path) -> list[dict]:
    """Load camera/frame list from JSON. Handle various Polycam export shapes."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Array of frames
    if isinstance(data, list):
        return data
    # Object with "cameras" or "frames" or "poses"
    if isinstance(data, dict):
        for key in ("cameras", "frames", "poses", "corrected_cameras"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # Single "poses" array at top level
        if "poses" in data:
            return data["poses"]
    return []


def frame_to_pose(frame: dict, index: int) -> tuple[float, float, float, float, float, float, float]:
    """Extract (x, y, z, qx, qy, qz, qw) from a frame dict. Return identity quat if missing."""
    # Position: various keys
    x = y = z = 0.0
    for key in ("position", "translation", "pos", "t"):
        if key in frame:
            v = frame[key]
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                x, y, z = float(v[0]), float(v[1]), float(v[2])
                break
            if isinstance(v, dict) and "x" in v:
                x = float(v.get("x", 0))
                y = float(v.get("y", 0))
                z = float(v.get("z", 0))
                break
    # Quaternion: various keys
    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    for key in ("rotation", "orientation", "quaternion", "quat", "q"):
        if key in frame:
            v = frame[key]
            if isinstance(v, (list, tuple)) and len(v) >= 4:
                qx, qy, qz, qw = float(v[0]), float(v[1]), float(v[2]), float(v[3])
                break
            if isinstance(v, dict):
                qx = float(v.get("x", 0))
                qy = float(v.get("y", 0))
                qz = float(v.get("z", 0))
                qw = float(v.get("w", 1))
                break
    return x, y, z, qx, qy, qz, qw


def get_timestamp(frame: dict, index: int) -> float | None:
    """Return Unix timestamp (seconds) if present, else None."""
    for key in ("timestamp", "time", "t", "timestamp_sec", "timestamp_ns"):
        if key not in frame:
            continue
        v = frame[key]
        if v is None:
            continue
        try:
            t = float(v)
            if t > 1e12:  # ns
                return t * 1e-9
            return t
        except (TypeError, ValueError):
            pass
    return None


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"ERROR: Not a directory: {input_dir}", file=sys.stderr)
        return 2

    cameras_path = find_cameras_json(input_dir)
    if cameras_path is None:
        print(f"ERROR: No cameras.json or corrected_cameras.json found under {input_dir}", file=sys.stderr)
        return 2

    frames = load_polycam_cameras(cameras_path)
    if not frames:
        print(f"ERROR: No camera frames in {cameras_path}", file=sys.stderr)
        return 2

    # Build trajectory rows
    t_rel_list: list[float] = []
    timestamps_sec: list[float] = []
    rows: list[dict] = []
    t0_sec: float | None = None
    has_timestamps = False

    for i, frame in enumerate(frames):
        ts = get_timestamp(frame, i)
        if ts is not None:
            has_timestamps = True
            if t0_sec is None:
                t0_sec = ts
            t_rel = ts - t0_sec
            timestamps_sec.append(ts)
        else:
            t_rel = float(i) * (1.0 / 30.0)  # assume 30 fps if no timestamp
        t_rel_list.append(t_rel)
        x, y, z, qx, qy, qz, qw = frame_to_pose(frame, i)
        rows.append({
            "t_rel_s": t_rel,
            "x": x, "y": y, "z": z,
            "qx": qx, "qy": qy, "qz": qz, "qw": qw,
        })

    if not has_timestamps:
        print("WARNING: No timestamps in camera data; using frame-order relative time (assume 30 fps).", file=sys.stderr)

    df = pd.DataFrame(rows)
    # Sort by t_rel_s for consistency
    df = df.sort_values("t_rel_s").reset_index(drop=True)
    # Normalize t_rel_s to start at 0
    t_min = df["t_rel_s"].min()
    df["t_rel_s"] = df["t_rel_s"] - t_min

    if args.out:
        out_path = Path(args.out)
    else:
        # Default: processed/trajectory.csv next to raw folder, or inside input_dir
        parent = input_dir.parent
        if parent.name == "raw" and (parent.parent / "processed").exists():
            out_path = parent.parent / "processed" / "trajectory.csv"
        else:
            out_path = input_dir / "trajectory.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f"Wrote trajectory: {out_path} ({len(df)} poses)")
    print(f"  t_rel_s range: {df['t_rel_s'].min():.3f} .. {df['t_rel_s'].max():.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
