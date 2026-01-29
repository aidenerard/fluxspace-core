#!/usr/bin/env python3
"""
rtabmap_poses_to_trajectory.py

Convert RTAB-Map exported poses to normalized trajectory.csv.
Input: RTAB-Map "Export poses" file (TUM or KITTI format).
Output: trajectory.csv with columns t_rel_s, x, y, z, qx, qy, qz, qw.

TUM format: timestamp tx ty tz qx qy qz qw (one line per pose).
KITTI format: 4x4 matrix rows (R|t) per pose, no quat; we convert to quat.

Matches fluxspace style: argparse, clear prints, sane defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RTAB-Map poses file -> trajectory.csv (t_rel_s, x, y, z, qx, qy, qz, qw)"
    )
    p.add_argument(
        "--in",
        dest="input_file",
        required=True,
        help="Path to RTAB-Map exported poses (TUM or KITTI format)",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output trajectory CSV. Default: <input_dir>/trajectory.csv or processed/trajectory.csv",
    )
    p.add_argument(
        "--format",
        choices=["TUM", "KITTI", "auto"],
        default="auto",
        help="Pose file format (auto = detect from content)",
    )
    return p.parse_args()


def rotation_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


def read_tum(path: Path) -> pd.DataFrame:
    """Read TUM format: timestamp tx ty tz qx qy qz qw."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                t = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                rows.append({"t_rel_s": t, "x": tx, "y": ty, "z": tz, "qx": qx, "qy": qy, "qz": qz, "qw": qw})
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


def read_kitti(path: Path) -> pd.DataFrame:
    """Read KITTI format: 4 lines per pose (3x4 matrix), then blank. No timestamp -> use index."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    t_rel = 0.0
    dt = 1.0 / 30.0
    while i + 3 <= len(lines):
        R = np.zeros((3, 3))
        t = np.zeros(3)
        for row in range(3):
            parts = lines[i + row].split()
            if len(parts) < 4:
                break
            for col in range(3):
                R[row, col] = float(parts[col])
            t[row] = float(parts[3])
        else:
            qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
            rows.append({
                "t_rel_s": t_rel,
                "x": float(t[0]), "y": float(t[1]), "z": float(t[2]),
                "qx": qx, "qy": qy, "qz": qz, "qw": qw,
            })
            t_rel += dt
        i += 4
        while i < len(lines) and lines[i].strip() == "":
            i += 1
    return pd.DataFrame(rows)


def detect_format(path: Path) -> str:
    """Detect TUM vs KITTI from first non-empty lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 8:
                try:
                    [float(x) for x in parts]
                    return "TUM"
                except ValueError:
                    pass
            if len(parts) >= 4 and len(parts) <= 12:
                return "KITTI"
            break
    return "TUM"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        return 2

    fmt = args.format
    if fmt == "auto":
        fmt = detect_format(input_path)
        print(f"Detected format: {fmt}")

    if fmt == "TUM":
        df = read_tum(input_path)
    else:
        df = read_kitti(input_path)

    if df.empty:
        print(f"ERROR: No valid poses in {input_path}", file=sys.stderr)
        return 2

    # Normalize t_rel_s to start at 0
    t_min = df["t_rel_s"].min()
    df["t_rel_s"] = df["t_rel_s"] - t_min

    if args.out:
        out_path = Path(args.out)
    else:
        parent = input_path.parent
        if parent.name == "raw" and (parent.parent / "processed").exists():
            out_path = parent.parent / "processed" / "trajectory.csv"
        else:
            out_path = input_path.parent / "trajectory.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f"Wrote trajectory: {out_path} ({len(df)} poses)")
    print(f"  t_rel_s range: {df['t_rel_s'].min():.3f} .. {df['t_rel_s'].max():.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
