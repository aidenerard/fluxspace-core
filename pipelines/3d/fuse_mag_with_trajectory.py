#!/usr/bin/env python3
"""
fuse_mag_with_trajectory.py

Fuse timestamped magnetometer log with camera trajectory using extrinsics.

Inputs (all relative to RUN_DIR by default):
  processed/trajectory.csv   t_rel_s, x, y, z, qx, qy, qz, qw
  raw/mag_run.csv            magnetometer log
  raw/extrinsics.json        cam-to-mag rigid offset (optional — identity if missing)

Output:
  processed/mag_world.csv    t_rel_s, x, y, z, value, value_type

Extrinsics schemas accepted (auto-detected):
  1) {"translation_m": [x,y,z], "quaternion_xyzw": [x,y,z,w]}
  2) {"translation_cm": [x,y,z], "quaternion_xyzw": [...]}   → cm converted to m
  3) {"t": [x,y,z], "q": [x,y,z,w]}                         → short-form
  4) File missing or empty → identity (warns loudly)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_paths import resolve_run_dir, processed_dir, raw_dir, ensure_dirs  # noqa: E402


# ---------------------------------------------------------------------------
# Extrinsics
# ---------------------------------------------------------------------------
def _parse_default_extrinsics(spec: str) -> np.ndarray:
    """Parse a shorthand like 'behind_cm=2,down_cm=10'.

    Assumed camera frame:  +x right, +y down, +z forward.
    "behind" = -z, "down" = +y.
    Returns translation in metres.
    """
    t = np.zeros(3)
    for token in spec.split(","):
        key, _, val = token.strip().partition("=")
        val_f = float(val)
        key = key.strip().lower()
        if key == "behind_cm":
            t[2] -= val_f / 100.0
        elif key == "forward_cm":
            t[2] += val_f / 100.0
        elif key == "down_cm":
            t[1] += val_f / 100.0
        elif key == "up_cm":
            t[1] -= val_f / 100.0
        elif key == "right_cm":
            t[0] += val_f / 100.0
        elif key == "left_cm":
            t[0] -= val_f / 100.0
        else:
            raise ValueError(f"Unknown extrinsics key: '{key}'. "
                             f"Use behind_cm, down_cm, right_cm, etc.")
    return t


def load_extrinsics(path: Path | None) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (translation_m [3], quaternion_xyzw [4] | None).

    Supports multiple JSON schemas and gracefully defaults to identity
    if the file is missing.
    """
    if path is None or not path.exists():
        warnings.warn(
            f"Extrinsics file not found ({path}). "
            "Using IDENTITY (translation=[0,0,0], no rotation). "
            "This assumes the magnetometer is at the camera centre.",
            stacklevel=2,
        )
        return np.zeros(3), None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Translation ---
    t_raw = (
        data.get("translation_m")
        or data.get("translation_cm")
        or data.get("t")
    )
    if t_raw is None:
        warnings.warn(
            f"extrinsics.json has no recognised translation key "
            f"(tried translation_m, translation_cm, t). Using [0,0,0].",
            stacklevel=2,
        )
        trans = np.zeros(3)
    else:
        trans = np.array([float(t_raw[0]), float(t_raw[1]), float(t_raw[2])])
        # Convert cm → m if needed
        if "translation_cm" in data:
            trans /= 100.0
            print(f"  Extrinsics: converted translation_cm {t_raw} → m {trans.tolist()}")

    # --- Quaternion (optional) ---
    q_raw = data.get("quaternion_xyzw") or data.get("q")
    quat: np.ndarray | None = None
    if q_raw is not None and len(q_raw) >= 4:
        quat = np.array([float(q_raw[i]) for i in range(4)])
        quat = quat / np.linalg.norm(quat)

    print(f"  Extrinsics: translation_m={trans.tolist()}, quat={'none' if quat is None else quat.tolist()}")
    return trans, quat


# ---------------------------------------------------------------------------
# Quaternion helper
# ---------------------------------------------------------------------------
def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector *v* by quaternion *q* (x, y, z, w)."""
    qx, qy, qz, qw = q
    return np.array([
        (1 - 2*qy*qy - 2*qz*qz)*v[0] + (2*qx*qy - 2*qz*qw)*v[1] + (2*qx*qz + 2*qy*qw)*v[2],
        (2*qx*qy + 2*qz*qw)*v[0] + (1 - 2*qx*qx - 2*qz*qz)*v[1] + (2*qy*qz - 2*qx*qw)*v[2],
        (2*qx*qz - 2*qy*qw)*v[0] + (2*qy*qz + 2*qx*qw)*v[1] + (1 - 2*qx*qx - 2*qy*qy)*v[2],
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fuse magnetometer log with trajectory -> processed/mag_world.csv"
    )
    p.add_argument(
        "--run", default="",
        help="RUN_DIR. Derives default --trajectory, --mag, --extrinsics, --out paths.",
    )
    p.add_argument(
        "--trajectory", default="",
        help="Path to trajectory.csv.  Default: $RUN_DIR/processed/trajectory.csv",
    )
    p.add_argument(
        "--mag", default="",
        help="Path to mag_run.csv.  Default: $RUN_DIR/raw/mag_run.csv",
    )
    p.add_argument(
        "--extrinsics", default="",
        help="Path to extrinsics.json.  Default: $RUN_DIR/raw/extrinsics.json  (optional)",
    )
    p.add_argument(
        "--default-extrinsics", default="",
        help="Shorthand extrinsics, e.g. 'behind_cm=2,down_cm=10'. "
             "Camera frame: +x right, +y down, +z forward.  Overrides extrinsics.json.",
    )
    p.add_argument(
        "--out", default="",
        help="Output mag_world.csv path.  Default: $RUN_DIR/processed/mag_world.csv",
    )
    p.add_argument(
        "--value-type",
        choices=["zero_mag", "raw", "corr", "b_total"],
        default="zero_mag",
        help="Which value column to write (default: zero_mag)",
    )
    p.add_argument(
        "--interpolate", action="store_true",
        help="Linearly interpolate pose at mag timestamps (default: nearest-neighbor)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    # --- Resolve run-relative defaults ---
    run = None
    if args.run:
        run = Path(args.run).expanduser().resolve()
    elif "RUN_DIR" in __import__("os").environ:
        run = Path(__import__("os").environ["RUN_DIR"]).expanduser().resolve()

    traj_path = Path(args.trajectory) if args.trajectory else (
        processed_dir(run) / "trajectory.csv" if run else None
    )
    mag_path = Path(args.mag) if args.mag else (
        raw_dir(run) / "mag_run.csv" if run else None
    )
    ext_path = Path(args.extrinsics) if args.extrinsics else (
        raw_dir(run) / "extrinsics.json" if run else None
    )
    out_path = Path(args.out) if args.out else (
        processed_dir(run) / "mag_world.csv" if run else None
    )

    # Validate required paths
    if traj_path is None:
        print("ERROR: --trajectory or --run is required.", file=sys.stderr)
        return 2
    if mag_path is None:
        print("ERROR: --mag or --run is required.", file=sys.stderr)
        return 2
    if out_path is None:
        out_path = traj_path.parent / "mag_world.csv"

    for p, name in [(traj_path, "trajectory"), (mag_path, "mag")]:
        if not p.exists():
            print(f"ERROR: {name} file not found: {p}", file=sys.stderr)
            return 2

    print(f"Trajectory : {traj_path}")
    print(f"Mag input  : {mag_path}")
    print(f"Extrinsics : {ext_path or '(none)'}")

    # --- Load data ---
    traj = pd.read_csv(traj_path)
    mag = pd.read_csv(mag_path)

    for col in ("t_rel_s", "x", "y", "z"):
        if col not in traj.columns:
            print(f"ERROR: trajectory missing column '{col}'", file=sys.stderr)
            return 2
    if "t_rel_s" not in mag.columns:
        print(f"ERROR: mag CSV missing column 't_rel_s'", file=sys.stderr)
        return 2

    # Filter to SAMPLE rows if row_type column exists
    if "row_type" in mag.columns:
        mag = mag.loc[mag["row_type"] == "SAMPLE"].copy()
    if mag.empty:
        print("ERROR: No SAMPLE rows in mag CSV", file=sys.stderr)
        return 2

    # --- Value column ---
    value_col = None
    if args.value_type == "zero_mag":
        value_col = next((c for c in ("zero_mag", "b_total", "B_total") if c in mag.columns), None)
    elif args.value_type in ("raw", "b_total"):
        value_col = next((c for c in ("raw_mag", "b_total", "B_total") if c in mag.columns), None)
    elif args.value_type == "corr":
        value_col = next((c for c in ("corr_mag", "b_total_cal", "B_total_cal") if c in mag.columns), None)
    else:
        value_col = next((c for c in ("zero_mag", "b_total", "B_total") if c in mag.columns), None)

    if value_col is None or value_col not in mag.columns:
        print(f"ERROR: mag CSV missing value column for --value-type={args.value_type}", file=sys.stderr)
        return 2

    values = mag[value_col].astype(float).to_numpy()
    if args.value_type == "zero_mag" and value_col != "zero_mag":
        baseline = np.nanmedian(values)
        values = values - baseline
    value_type_label = "zero_mag" if args.value_type == "zero_mag" else args.value_type

    # --- Extrinsics ---
    if args.default_extrinsics:
        trans = _parse_default_extrinsics(args.default_extrinsics)
        quat = None
        print(f"  Using CLI extrinsics: translation_m={trans.tolist()}")
    else:
        trans, quat = load_extrinsics(ext_path)

    # --- Fuse ---
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
            px = w0 * x_traj[i0] + w1 * x_traj[i1]
            py = w0 * y_traj[i0] + w1 * y_traj[i1]
            pz = w0 * z_traj[i0] + w1 * z_traj[i1]
            if has_quat:
                q_pose = np.array([qx[i0], qy[i0], qz[i0], qw[i0]])
                q_pose /= np.linalg.norm(q_pose)
                trans_rotated = quat_rotate_vector(q_pose, trans)
            else:
                trans_rotated = trans
            x_world[i] = px + trans_rotated[0]
            y_world[i] = py + trans_rotated[1]
            z_world[i] = pz + trans_rotated[2]
    else:
        for i in range(n_mag):
            t = t_mag[i]
            idx = int(np.argmin(np.abs(t_traj - t)))
            px, py, pz = x_traj[idx], y_traj[idx], z_traj[idx]
            if has_quat:
                q_pose = np.array([qx[idx], qy[idx], qz[idx], qw[idx]])
                q_pose /= np.linalg.norm(q_pose)
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote mag_world  : {out_path}  ({len(out_df)} points)")
    print(f"  value_type     : {value_type_label}")
    print(f"  xyz range      : x [{x_world.min():.4f}, {x_world.max():.4f}]  "
          f"y [{y_world.min():.4f}, {y_world.max():.4f}]  "
          f"z [{z_world.min():.4f}, {z_world.max():.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
