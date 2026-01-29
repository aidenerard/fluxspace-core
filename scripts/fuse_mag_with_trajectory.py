#!/usr/bin/env python3
"""
fuse_mag_with_trajectory.py

Fuse timestamped magnetometer log with camera trajectory using extrinsics.
Inputs: trajectory.csv, mag_run.csv (from mag_to_csv_v2 or mag_calibrate_zero_logger), extrinsics.json.
Output: mag_world.csv with t_rel_s, x, y, z, value, value_type.

- Time alignment: nearest-neighbor by default; --interpolate for linear pose interpolation at mag timestamps.
- Extrinsics: translation (and optional quaternion rotation) from phone/camera frame to magnetometer frame.
  We apply: world_mag_position = pose_position + pose_rotation @ translation_m.
- value_type: zero_mag (baseline-subtracted magnitude), raw (b_total), or corr if present.

Matches fluxspace style: argparse, clear prints, sane defaults.
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
        description="Fuse magnetometer log with trajectory -> mag_world.csv (x, y, z, value)"
    )
    p.add_argument("--trajectory", required=True, help="Path to trajectory.csv (t_rel_s, x, y, z, qx, qy, qz, qw)")
    p.add_argument("--mag", required=True, help="Path to mag_run.csv (from mag_to_csv_v2 or mag_calibrate_zero_logger)")
    p.add_argument("--extrinsics", required=True, help="Path to extrinsics.json (translation_m, optional rotation_quat_xyzw)")
    p.add_argument("--out", default="", help="Output mag_world.csv path. Default: <trajectory_dir>/mag_world.csv")
    p.add_argument(
        "--value-type",
        choices=["zero_mag", "raw", "corr", "b_total"],
        default="zero_mag",
        help="Which value to write: zero_mag (baseline-subtracted), raw/b_total, or corr",
    )
    p.add_argument(
        "--interpolate",
        action="store_true",
        help="Linearly interpolate pose at mag timestamps (default: nearest-neighbor)",
    )
    return p.parse_args()


def load_extrinsics(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (translation_m [3], rotation_quat_xyzw [4] or None)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    t = data.get("translation_m")
    if t is None:
        raise ValueError("extrinsics.json must contain 'translation_m' [x, y, z]")
    trans = np.array([float(t[0]), float(t[1]), float(t[2])])
    q = data.get("rotation_quat_xyzw")  # optional: [x, y, z, w]
    if q is not None and len(q) >= 4:
        quat = np.array([float(q[0]), float(q[1]), float(q[2]), float(q[3])])
        quat = quat / np.linalg.norm(quat)
        return trans, quat
    return trans, None


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (x, y, z, w)."""
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    return np.array([
        (1 - 2*qy*qy - 2*qz*qz)*v[0] + (2*qx*qy - 2*qz*qw)*v[1] + (2*qx*qz + 2*qy*qw)*v[2],
        (2*qx*qy + 2*qz*qw)*v[0] + (1 - 2*qx*qx - 2*qz*qz)*v[1] + (2*qy*qz - 2*qx*qw)*v[2],
        (2*qx*qz - 2*qy*qw)*v[0] + (2*qy*qz + 2*qx*qw)*v[1] + (1 - 2*qx*qx - 2*qy*qy)*v[2],
    ])


def main() -> int:
    args = parse_args()
    traj_path = Path(args.trajectory)
    mag_path = Path(args.mag)
    ext_path = Path(args.extrinsics)

    for p, name in [(traj_path, "trajectory"), (mag_path, "mag"), (ext_path, "extrinsics")]:
        if not p.exists():
            print(f"ERROR: {name} file not found: {p}", file=sys.stderr)
            return 2

    traj = pd.read_csv(traj_path)
    mag = pd.read_csv(mag_path)
    for col in ("t_rel_s", "x", "y", "z"):
        if col not in traj.columns:
            print(f"ERROR: trajectory missing column '{col}'", file=sys.stderr)
            return 2
    for col in ("t_rel_s",):
        if col not in mag.columns:
            print(f"ERROR: mag CSV missing column 't_rel_s'", file=sys.stderr)
            return 2

    # Sample rows only
    sample_mask = mag.get("row_type", pd.Series(dtype=str)) == "SAMPLE"
    if "row_type" in mag.columns:
        mag = mag.loc[sample_mask].copy()
    if mag.empty:
        print("ERROR: No SAMPLE rows in mag CSV", file=sys.stderr)
        return 2

    # Value column: prefer mag_calibrate_zero_logger columns (zero_mag, corr_mag, raw_mag) when present
    value_col = None
    if args.value_type == "zero_mag":
        value_col = "zero_mag" if "zero_mag" in mag.columns else ("b_total" if "b_total" in mag.columns else "B_total")
    elif args.value_type in ("raw", "b_total"):
        value_col = "raw_mag" if "raw_mag" in mag.columns else ("b_total" if "b_total" in mag.columns else "B_total")
    elif args.value_type == "corr":
        value_col = "corr_mag" if "corr_mag" in mag.columns else ("b_total_cal" if "b_total_cal" in mag.columns else "B_total_cal")
    else:
        value_col = "zero_mag" if "zero_mag" in mag.columns else ("b_total" if "b_total" in mag.columns else "B_total")
    if value_col not in mag.columns:
        print(f"ERROR: mag CSV missing value column '{value_col}'", file=sys.stderr)
        return 2

    values = mag[value_col].astype(float).to_numpy()
    if args.value_type == "zero_mag" and value_col not in ("zero_mag",):
        # mag_to_csv_v2-style: subtract baseline
        baseline = np.nanmedian(values)
        values = values - baseline
    value_type_label = "zero_mag" if args.value_type == "zero_mag" else args.value_type

    trans, quat = load_extrinsics(ext_path)
    t_mag = mag["t_rel_s"].astype(float).to_numpy()
    t_traj = traj["t_rel_s"].astype(float).to_numpy()
    x_traj = traj["x"].astype(float).to_numpy()
    y_traj = traj["y"].astype(float).to_numpy()
    z_traj = traj["z"].astype(float).to_numpy()
    has_quat = "qx" in traj.columns and quat is not None
    if has_quat:
        qx = traj["qx"].astype(float).to_numpy()
        qy = traj["qy"].astype(float).to_numpy()
        qz = traj["qz"].astype(float).to_numpy()
        qw = traj["qw"].astype(float).to_numpy()

    n_mag = len(t_mag)
    x_world = np.empty(n_mag)
    y_world = np.empty(n_mag)
    z_world = np.empty(n_mag)

    if args.interpolate:
        for i in range(n_mag):
            t = t_mag[i]
            idx = np.searchsorted(t_traj, t, side="left")
            if idx <= 0:
                idx = 0
                w0, w1 = 1.0, 0.0
            elif idx >= len(t_traj):
                idx = len(t_traj) - 1
                w0, w1 = 0.0, 1.0
            else:
                t0, t1 = t_traj[idx - 1], t_traj[idx]
                denom = t1 - t0
                w1 = (t - t0) / denom if denom > 0 else 0.0
                w0 = 1.0 - w1
            i0, i1 = max(0, idx - 1), min(idx, len(t_traj) - 1)
            x0, y0, z0 = x_traj[i0], y_traj[i0], z_traj[i0]
            x1, y1, z1 = x_traj[i1], y_traj[i1], z_traj[i1]
            px = w0 * x0 + w1 * x1
            py = w0 * y0 + w1 * y1
            pz = w0 * z0 + w1 * z1
            if has_quat:
                q0 = np.array([qx[i0], qy[i0], qz[i0], qw[i0]])
                q1 = np.array([qx[i1], qy[i1], qz[i1], qw[i1]])
                q0 = q0 / np.linalg.norm(q0)
                q1 = q1 / np.linalg.norm(q1)
                # Spherical interpolation (slerp) simplified: use q0 for V1
                q_pose = q0
                trans_rotated = quat_rotate_vector(q_pose, trans)
            else:
                trans_rotated = trans
            x_world[i] = px + trans_rotated[0]
            y_world[i] = py + trans_rotated[1]
            z_world[i] = pz + trans_rotated[2]
    else:
        for i in range(n_mag):
            t = t_mag[i]
            idx = np.argmin(np.abs(t_traj - t))
            px, py, pz = x_traj[idx], y_traj[idx], z_traj[idx]
            if has_quat:
                q_pose = np.array([qx[idx], qy[idx], qz[idx], qw[idx]])
                q_pose = q_pose / np.linalg.norm(q_pose)
                trans_rotated = quat_rotate_vector(q_pose, trans)
            else:
                trans_rotated = trans
            x_world[i] = px + trans_rotated[0]
            y_world[i] = py + trans_rotated[1]
            z_world[i] = pz + trans_rotated[2]

    out_df = pd.DataFrame({
        "t_rel_s": t_mag,
        "x": x_world, "y": y_world, "z": z_world,
        "value": values,
        "value_type": value_type_label,
    })

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = traj_path.parent / "mag_world.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote mag_world: {out_path} ({len(out_df)} points)")
    print(f"  value_type: {value_type_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
