#!/usr/bin/env python3
"""
validate_run.py — Smoke-test a FluxSpace 3D run directory.

Checks:
  1. Required raw files exist (color, depth, timestamps, intrinsics)
  2. trajectory.csv exists and contains valid numeric data
  3. Mesh and point cloud files are non-empty
  4. Clean outputs exist if cleaning was enabled
  5. Volume NPZ is loadable (skipped in --camera-only mode)
  6. Reports (reconstruction, cleaning) are parseable

Usage:
  python3 pipelines/3d/validate_run.py --run data/runs/run_20260210_1430
  python3 pipelines/3d/validate_run.py --run data/runs/run_20260210_1430 --require-clean
  python3 pipelines/3d/validate_run.py --run data/runs/run_cam_only_... --camera-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def _ok(msg: str):
    print(f"  OK   {msg}")

def _warn(msg: str):
    print(f"  WARN {msg}")

def _fail(msg: str):
    print(f"  FAIL {msg}")


def validate(run_dir: Path, require_clean: bool = False,
             camera_only: bool = False) -> bool:
    """Validate a run directory. Returns True if all checks pass."""
    passed = True
    warnings = 0

    mode_label = "camera-only" if camera_only else "full (camera + mag)"
    print(f"Validating: {run_dir}")
    print(f"Mode: {mode_label}\n")

    # --- Raw inputs ---
    print("[Raw inputs]")
    raw_oak = run_dir / "raw" / "oak_rgbd"

    for d in ["color", "depth"]:
        p = raw_oak / d
        if p.is_dir():
            ext = "*.jpg" if d == "color" else "*.png"
            n = len(list(p.glob(ext)))
            if n > 0:
                _ok(f"{d}/ — {n} files")
            else:
                _fail(f"{d}/ — directory exists but no {ext} files")
                passed = False
        else:
            _fail(f"{d}/ — directory missing")
            passed = False

    ts = raw_oak / "timestamps.csv"
    if ts.exists():
        _ok(f"timestamps.csv — {ts.stat().st_size} bytes")
    else:
        _fail("timestamps.csv — missing")
        passed = False

    intr = raw_oak / "intrinsics.json"
    if intr.exists():
        _ok("intrinsics.json — present")
    else:
        _warn("intrinsics.json — missing (approximate intrinsics will be used)")
        warnings += 1

    # Magnetometer file
    mag = run_dir / "raw" / "mag_run.csv"
    if camera_only:
        if mag.exists():
            _ok(f"mag_run.csv — {mag.stat().st_size} bytes (present but not required)")
        else:
            _ok("mag_run.csv — not required (camera-only)")
    else:
        if mag.exists():
            _ok(f"mag_run.csv — {mag.stat().st_size} bytes")
        else:
            _fail("mag_run.csv — missing")
            passed = False

    # Calibration + extrinsics (always optional, just informational)
    for name in ["calibration.json", "extrinsics.json"]:
        p = run_dir / "raw" / name
        if p.exists():
            _ok(f"{name} — present")
        else:
            if camera_only:
                _ok(f"{name} — not required (camera-only)")
            else:
                _warn(f"{name} — missing (optional)")
                warnings += 1

    # --- Processed outputs ---
    print("\n[Processed outputs]")
    proc = run_dir / "processed"

    # trajectory.csv
    traj = proc / "trajectory.csv"
    if traj.exists():
        try:
            with open(traj, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            n_rows = len(rows)
            if n_rows == 0:
                _fail("trajectory.csv — 0 rows")
                passed = False
            else:
                nan_count = 0
                for r in rows:
                    vals = [r.get("x", ""), r.get("y", ""), r.get("z", "")]
                    for v in vals:
                        try:
                            if not np.isfinite(float(v)):
                                nan_count += 1
                                break
                        except (ValueError, TypeError):
                            nan_count += 1
                            break
                if nan_count > 0:
                    _warn(f"trajectory.csv — {n_rows} rows, {nan_count} with NaN/invalid")
                    warnings += 1
                else:
                    _ok(f"trajectory.csv — {n_rows} rows, all valid")
        except Exception as exc:
            _fail(f"trajectory.csv — parse error: {exc}")
            passed = False
    else:
        _fail("trajectory.csv — missing")
        passed = False

    # Geometry files (accept both new _raw names and legacy names)
    for name_group in [
        ("open3d_pcd_raw.ply",),
        ("open3d_mesh_raw.ply", "open3d_mesh.ply"),
    ]:
        found = False
        for name in name_group:
            p = proc / name
            if p.exists():
                sz = p.stat().st_size
                if sz > 1000:
                    _ok(f"{name} — {sz:,} bytes")
                else:
                    _warn(f"{name} — suspiciously small ({sz} bytes)")
                    warnings += 1
                found = True
                break
        if not found:
            _warn(f"{name_group[0]} — missing")
            warnings += 1

    # Clean outputs
    clean_files = ["open3d_pcd_clean.ply", "open3d_mesh_clean.ply"]
    for name in clean_files:
        p = proc / name
        if p.exists():
            sz = p.stat().st_size
            if sz > 1000:
                _ok(f"{name} — {sz:,} bytes")
            else:
                _warn(f"{name} — suspiciously small ({sz} bytes)")
                warnings += 1
        elif require_clean:
            _fail(f"{name} — missing (--require-clean)")
            passed = False
        else:
            _warn(f"{name} — not present (cleaning may have been skipped)")
            warnings += 1

    # Reports
    for rpt in ["reconstruction_report.json", "cleaning_report.json"]:
        p = proc / rpt
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                w = data.get("warnings", [])
                if w:
                    _ok(f"{rpt} — valid, {len(w)} warning(s): {w}")
                else:
                    _ok(f"{rpt} — valid, no warnings")
            except Exception as exc:
                _warn(f"{rpt} — parse error: {exc}")
                warnings += 1
        else:
            _warn(f"{rpt} — not present")
            warnings += 1

    # Mag world + Volume (skip checks in camera-only mode)
    if camera_only:
        print("\n[Mag / Exports — skipped (camera-only)]")
        _ok("mag_world*.csv — not required")
        _ok("volume.npz — not required")
    else:
        # Mag world
        mw = proc / "mag_world.csv"
        mw_m = proc / "mag_world_m.csv"
        if mw.exists():
            _ok(f"mag_world.csv — {mw.stat().st_size:,} bytes")
        elif mw_m.exists():
            _ok(f"mag_world_m.csv — {mw_m.stat().st_size:,} bytes (auto-scaled)")
        else:
            _warn("mag_world*.csv — missing")
            warnings += 1

        # Exports
        print("\n[Exports]")
        vol = run_dir / "exports" / "volume.npz"
        if vol.exists():
            try:
                d = np.load(str(vol), allow_pickle=True)
                shape = d["volume"].shape if "volume" in d else d["grid"].shape
                _ok(f"volume.npz — shape={shape}")
            except Exception as exc:
                _fail(f"volume.npz — load error: {exc}")
                passed = False
        else:
            _warn("volume.npz — missing")
            warnings += 1

    # --- Summary ---
    print(f"\n{'PASSED' if passed else 'FAILED'} — {warnings} warning(s)")
    return passed


def main() -> int:
    p = argparse.ArgumentParser(description="Validate a FluxSpace 3D run directory")
    p.add_argument("--run", required=True, help="Path to run directory")
    p.add_argument("--require-clean", action="store_true",
                   help="Fail if clean geometry outputs are missing")
    p.add_argument("--camera-only", action="store_true",
                   help="Camera-only mode: do not require mag files, volume, or fusion outputs")
    args = p.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    if not run_dir.is_dir():
        print(f"ERROR: Not a directory: {run_dir}", file=sys.stderr)
        return 2

    ok = validate(run_dir, require_clean=args.require_clean,
                  camera_only=args.camera_only)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
