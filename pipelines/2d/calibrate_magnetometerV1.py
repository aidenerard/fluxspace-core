#!/usr/bin/env python3
"""
calibrate_magnetometerV1.py

Fluxspace Core: magnetometer calibration (hard-iron + optional soft-iron) with Earth-field scaling.

This script is designed to sit alongside the existing pipeline scripts (same CLI + error-handling style as
compute_local_anomaly_v2.py).

What it does (v1):
  1) Loads a CSV that contains raw magnetometer components (Bx, By, Bz) in microtesla (uT).
  2) Fits calibration parameters:
       - Hard-iron offset (3-vector, uT)
       - Soft-iron correction (3x3 matrix), optional via ellipsoid fit
  3) Optionally rescales so the median calibrated magnitude matches an expected local Earth field magnitude.
  4) Writes:
       - calibration JSON (offset + matrix + metadata)
       - optional calibrated CSV (adds Bx_cal, By_cal, Bz_cal, B_total_cal)

How to collect calibration data:
  - Move the sensor through as many orientations as possible (slowly rotate in 3D).
  - Collect at least a few hundred samples.
  - For best results on a drone: calibrate *with the sensor mounted on the airframe* and wiring routed the way
    you will fly (ESCs, PDB, battery, flight controller nearby). A 4-in-1 ESC + motors + PDB are common strong
    magnetic interference sources, so physical placement matters.

Notes on "Earth field":
  - Earth's magnetic field magnitude varies by location (~25–65 uT globally). This script lets you set a target
    magnitude (earth_field_ut) that can be changed and tested easily.
  - If you set earth_field_ut <= 0, Earth-field scaling is disabled (the calibration will output unit-sphere or
    relative scale depending on method).

Example:
  # Fit calibration from a rotation dataset
  python3 scripts/calibrate_magnetometerV1.py --in data/raw/mag_cal.csv --method ellipsoid --earth-field-ut 52

  # Fit + write a calibrated CSV + save plots
  python3 scripts/calibrate_magnetometerV1.py --in data/raw/mag_cal.csv --method minmax --earth-field-ut 50 \
      --write-calibrated --plot --no-show

"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Based on the current parts list for your drone build (frame, motors, 4-in-1 ESC, PDB, Pixhawk 6c, GPS, etc.).
# This is included in the calibration JSON as helpful context because these parts are typical magnetic
# interference sources near the magnetometer.
DEFAULT_PLATFORM_PARTS = {
    "frame": "ZMR250 / Readytosky 250 mm carbon frame",
    "motors": "Emax ECO II motors",
    "escs": "JHEMCU AM32A60 60A 4-in-1 ESC (3–6S) with current sensor",
    "propellers": "Gemfan 5 inch props",
    "battery": "Zeee 3S LiPo 2200mAh",
    "flight_controller": "Pixhawk 6c",
    "magnetometer": "QMC5883P",
    "raspberry_pi": "Vilros Raspberry Pi 4 4GB starter kit",
    "rx_tx": "Flysky FS-iA6B receiver",
    "gps": "u-blox NEO-M8N",
    "pdb": "QWinOut power distribution board",
}



# ------------------------
# Helpers / Core math
# ------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _robust_z(x: np.ndarray) -> np.ndarray:
    """
    Robust z-score using MAD (median absolute deviation).
    Returns z values; if MAD==0 -> zeros.
    """
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad <= 0:
        return np.zeros_like(x, dtype=float)
    return 0.6745 * (x - med) / mad


def _ensure_cols(df: pd.DataFrame, bx: str, by: str, bz: str) -> pd.DataFrame:
    missing = [c for c in (bx, by, bz) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Columns found: {list(df.columns)}")

    out = df.copy()
    out[bx] = pd.to_numeric(out[bx], errors="coerce")
    out[by] = pd.to_numeric(out[by], errors="coerce")
    out[bz] = pd.to_numeric(out[bz], errors="coerce")
    return out


def _vector_magnitude(B: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(B * B, axis=1))


@dataclass
class Calibration:
    method: str
    offset_ut: np.ndarray             # shape (3,)
    softiron_matrix: np.ndarray       # shape (3,3), applied as: B_cal = M @ (B_raw - offset)
    earth_field_ut: float
    gain: float                       # additional scalar gain applied after M (already folded into M if desired)


def fit_minmax(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple, robust v1 fit:
      offset = (max + min)/2
      scale  = diag( avg_radius / axis_radius )
    This corrects axis biases and per-axis scaling, but NOT cross-axis coupling.
    """
    mins = np.nanmin(B, axis=0)
    maxs = np.nanmax(B, axis=0)
    offset = 0.5 * (maxs + mins)

    radii = 0.5 * (maxs - mins)
    if np.any(~np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("Invalid min/max radii; need variation on all axes.")

    avg_r = float(np.mean(radii))
    scale = np.diag(avg_r / radii)  # maps ellipsoid-ish ranges to similar radii
    return offset, scale


def fit_ellipsoid(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ellipsoid fit (least squares) to estimate hard-iron offset + full 3x3 soft-iron correction.

    Fits quadratic form:
        x^T A x + b^T x + c = 0
    Derives center and shape matrix, then returns:
        offset (center) and M such that || M @ (x - offset) || ≈ 1.

    If the fit is ill-conditioned, raises ValueError.
    """
    x = B[:, 0]
    y = B[:, 1]
    z = B[:, 2]

    # Build design matrix for: [x^2, y^2, z^2, xy, xz, yz, x, y, z, 1]
    D = np.column_stack([
        x * x,
        y * y,
        z * z,
        x * y,
        x * z,
        y * z,
        x,
        y,
        z,
        np.ones_like(x),
    ])

    # Solve D @ v = 0 subject to ||v||=1.
    # Use SVD; solution is right singular vector corresponding to smallest singular value.
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    v = Vt[-1, :]

    # Unpack parameters
    # A is symmetric:
    # [a  d/2 e/2]
    # [d/2 b  f/2]
    # [e/2 f/2 c]
    a, b, c, d, e, f, g, h, i, j = v
    A = np.array([
        [a, d / 2.0, e / 2.0],
        [d / 2.0, b, f / 2.0],
        [e / 2.0, f / 2.0, c],
    ], dtype=float)
    bb = np.array([g, h, i], dtype=float)
    cc = float(j)

    if np.linalg.cond(A) > 1e12:
        raise ValueError("Ellipsoid fit ill-conditioned (A matrix). Try more diverse orientations or use --method minmax.")

    center = -0.5 * np.linalg.solve(A, bb)

    # Translate: x' = x - center.
    # Compute constant term for translated quadric:
    # k = cc + center^T A center + bb^T center
    k = cc + float(center.T @ A @ center) + float(bb.T @ center)

    if not np.isfinite(k) or abs(k) < 1e-12:
        raise ValueError("Ellipsoid fit failed (degenerate constant term).")

    # For an ellipsoid, we expect (x')^T A (x') = -k  (positive rhs)
    # Normalize shape matrix:
    Mshape = A / (-k)

    # Ensure positive definite
    evals, evecs = np.linalg.eigh(Mshape)
    if np.any(evals <= 0) or np.any(~np.isfinite(evals)):
        raise ValueError("Ellipsoid fit produced non-positive-definite shape matrix. Try --method minmax.")

    # We want transform T such that ||T x'||^2 = x'^T Mshape x' and thus ||T x'|| ≈ 1
    # If Mshape = R diag(l) R^T, then T = R diag(sqrt(l)) R^T
    T = (evecs @ np.diag(np.sqrt(evals)) @ evecs.T).astype(float)

    return center.astype(float), T


def apply_calibration(B: np.ndarray, cal: Calibration) -> np.ndarray:
    # B_cal = gain * (M @ (B - offset))
    return (cal.gain * (cal.softiron_matrix @ (B - cal.offset_ut).T)).T


def compute_gain_to_match_earth(B_cal: np.ndarray, earth_field_ut: float) -> float:
    if earth_field_ut <= 0:
        return 1.0
    mag = _vector_magnitude(B_cal)
    med = float(np.nanmedian(mag))
    if not np.isfinite(med) or med <= 0:
        return 1.0
    return float(earth_field_ut / med)


def write_calibration_json(out_json: Path, cal: Calibration, meta: dict) -> None:
    payload = {
        "created_utc": _utc_now_iso(),
        "method": cal.method,
        "earth_field_ut": float(cal.earth_field_ut),
        "offset_ut": [float(x) for x in cal.offset_ut.tolist()],
        "softiron_matrix": [[float(x) for x in row] for row in cal.softiron_matrix.tolist()],
        "gain": float(cal.gain),
        "notes": meta,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")


def load_calibration_json(path: Path) -> Calibration:
    obj = json.loads(path.read_text())
    offset = np.array(obj["offset_ut"], dtype=float).reshape(3)
    M = np.array(obj["softiron_matrix"], dtype=float).reshape(3, 3)
    earth = float(obj.get("earth_field_ut", 0.0))
    gain = float(obj.get("gain", 1.0))
    method = str(obj.get("method", "unknown"))
    return Calibration(method=method, offset_ut=offset, softiron_matrix=M, earth_field_ut=earth, gain=gain)


# ------------------------
# Plotting
# ------------------------

def plot_projections(B_raw: np.ndarray, B_cal: Optional[np.ndarray], out_png: Optional[Path], show: bool) -> None:
    """
    2D projections (xy, xz, yz) + magnitude histograms before/after.
    """
    fig = plt.figure(figsize=(10, 8))

    def _scatter(ax, X, Y, title):
        ax.scatter(X, Y, s=6)
        ax.set_title(title)
        ax.set_xlabel("uT")
        ax.set_ylabel("uT")
        ax.set_aspect("equal", "box")

    ax1 = fig.add_subplot(2, 2, 1)
    _scatter(ax1, B_raw[:, 0], B_raw[:, 1], "Raw XY")

    ax2 = fig.add_subplot(2, 2, 2)
    _scatter(ax2, B_raw[:, 0], B_raw[:, 2], "Raw XZ")

    ax3 = fig.add_subplot(2, 2, 3)
    _scatter(ax3, B_raw[:, 1], B_raw[:, 2], "Raw YZ")

    ax4 = fig.add_subplot(2, 2, 4)
    mag_raw = _vector_magnitude(B_raw)
    ax4.hist(mag_raw[np.isfinite(mag_raw)], bins=40, alpha=0.7, label="raw")
    if B_cal is not None:
        mag_cal = _vector_magnitude(B_cal)
        ax4.hist(mag_cal[np.isfinite(mag_cal)], bins=40, alpha=0.7, label="cal")
    ax4.set_title("|B| histogram")
    ax4.set_xlabel("uT")
    ax4.set_ylabel("count")
    ax4.legend()

    fig.tight_layout()

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=160)
        plt.close(fig)
        print(f"Wrote plot PNG: {out_png}")
    elif show:
        plt.show()
    else:
        plt.close(fig)


# ------------------------
# CLI
# ------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit/apply magnetometer calibration and write calibration JSON (+ optional calibrated CSV).")
    p.add_argument("--in", dest="infile", required=True, help="Input CSV path containing Bx/By/Bz (uT).")
    p.add_argument("--bx-col", default="Bx", help="Column name for Bx (default: Bx)")
    p.add_argument("--by-col", default="By", help="Column name for By (default: By)")
    p.add_argument("--bz-col", default="Bz", help="Column name for Bz (default: Bz)")

    p.add_argument("--method", choices=["minmax", "ellipsoid"], default="minmax", help="Calibration method (default: minmax).")
    p.add_argument("--earth-field-ut", type=float, default=50.0, help="Expected Earth field magnitude in uT. Set <=0 to disable scaling.")
    p.add_argument("--no-earth-scaling", action="store_true", help="Disable Earth-field scaling regardless of earth_field_ut value.")

    p.add_argument("--clip-mag-z", type=float, default=6.0, help="Drop rows whose |B| robust-z exceeds this threshold (default: 6.0).")
    p.add_argument("--min-samples", type=int, default=200, help="Minimum number of samples required after cleaning (default: 200).")

    p.add_argument("--out-json", default=None, help="Output calibration JSON path. Default: <input_stem>_calibration.json next to input.")
    p.add_argument("--write-calibrated", action="store_true", help="If set, also write a calibrated CSV.")
    p.add_argument("--out-csv", default=None, help="Calibrated CSV path if --write-calibrated. Default: <input_stem>_calibrated.csv")

    # Optional: quick interference delta reporting if you labeled data
    p.add_argument("--segment-col", default=None, help="Optional column to segment data (e.g., 'power_state' or 'throttle').")
    p.add_argument("--segment-a", default=None, help="Segment label A (e.g., 'OFF').")
    p.add_argument("--segment-b", default=None, help="Segment label B (e.g., 'ON').")

    p.add_argument("--plot", action="store_true", help="If set, show/save quick projection plots + magnitude hist.")
    p.add_argument("--no-show", action="store_true", help="If set with --plot, save plot PNG instead of displaying it.")
    p.add_argument("--plot-out", default=None, help="Output PNG path if using --plot --no-show. Default: <out_json_stem>_plot.png")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    infile = Path(args.infile)
    if not infile.is_absolute():
        infile = _REPO_ROOT / infile
    if not infile.exists():
        print(f"ERROR: input file not found: {infile}", file=sys.stderr)
        return 2

    try:
        df = pd.read_csv(infile)
    except Exception as e:
        print(f"ERROR: could not read CSV: {e}", file=sys.stderr)
        return 2

    try:
        df = _ensure_cols(df, args.bx_col, args.by_col, args.bz_col)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Drop NaNs in components
    before = len(df)
    df = df.dropna(subset=[args.bx_col, args.by_col, args.bz_col]).copy()
    if len(df) == 0:
        print("ERROR: no valid rows after dropping NaNs in Bx/By/Bz.", file=sys.stderr)
        return 2
    if len(df) != before:
        print(f"Note: dropped {before - len(df)} rows due to NaNs in Bx/By/Bz.")

    B_raw = df[[args.bx_col, args.by_col, args.bz_col]].to_numpy(dtype=float)

    # Optional magnitude clipping to remove obvious spikes / saturations
    mag = _vector_magnitude(B_raw)
    z = _robust_z(mag)
    keep = np.abs(z) <= float(args.clip_mag_z)
    if np.any(~keep):
        dropped = int(np.sum(~keep))
        df = df.loc[keep].copy()
        B_raw = df[[args.bx_col, args.by_col, args.bz_col]].to_numpy(dtype=float)
        print(f"Note: dropped {dropped} samples where |B| robust-z > {args.clip_mag_z}.")

    if len(df) < int(args.min_samples):
        print(f"ERROR: only {len(df)} samples after cleaning; need at least {args.min_samples}.", file=sys.stderr)
        return 2

    # Fit calibration
    try:
        if args.method == "ellipsoid":
            offset, M = fit_ellipsoid(B_raw)
        else:
            offset, M = fit_minmax(B_raw)
    except Exception as e:
        print(f"ERROR: calibration fit failed: {e}", file=sys.stderr)
        return 2

    earth_field_ut = float(args.earth_field_ut)
    if args.no_earth_scaling:
        earth_field_ut = 0.0

    cal0 = Calibration(method=args.method, offset_ut=offset, softiron_matrix=M, earth_field_ut=earth_field_ut, gain=1.0)
    B_cal0 = apply_calibration(B_raw, cal0)

    gain = compute_gain_to_match_earth(B_cal0, earth_field_ut=earth_field_ut)
    cal = Calibration(method=args.method, offset_ut=offset, softiron_matrix=M, earth_field_ut=earth_field_ut, gain=gain)
    B_cal = apply_calibration(B_raw, cal)

    # Basic stats
    mag_raw = _vector_magnitude(B_raw)
    mag_cal = _vector_magnitude(B_cal)
    print(f"Raw |B| median: {float(np.median(mag_raw)):.3f} uT   (min={float(np.min(mag_raw)):.3f}, max={float(np.max(mag_raw)):.3f})")
    print(f"Cal |B| median: {float(np.median(mag_cal)):.3f} uT   (min={float(np.min(mag_cal)):.3f}, max={float(np.max(mag_cal)):.3f})")
    if earth_field_ut > 0:
        print(f"Earth scaling target: {earth_field_ut:.3f} uT   -> applied gain: {gain:.6f}")

    # Optional interference delta report
    meta: dict = {"platform_parts": DEFAULT_PLATFORM_PARTS}
    if args.segment_col and (args.segment_col in df.columns) and (args.segment_a is not None) and (args.segment_b is not None):
        seg = df[args.segment_col].astype(str)
        mask_a = seg == str(args.segment_a)
        mask_b = seg == str(args.segment_b)
        if np.any(mask_a) and np.any(mask_b):
            a_med = np.median(_vector_magnitude(B_cal[mask_a.to_numpy()]))
            b_med = np.median(_vector_magnitude(B_cal[mask_b.to_numpy()]))
            meta["segment_col"] = args.segment_col
            meta["segment_a"] = str(args.segment_a)
            meta["segment_b"] = str(args.segment_b)
            meta["segment_a_median_B_ut"] = float(a_med)
            meta["segment_b_median_B_ut"] = float(b_med)
            meta["segment_delta_median_B_ut"] = float(b_med - a_med)
            print(f"Interference check ({args.segment_col}): median |B| {args.segment_a}={a_med:.3f} uT, {args.segment_b}={b_med:.3f} uT (delta={b_med-a_med:+.3f} uT)")
        else:
            print("Note: segment labels not found in data for interference check; skipping.", file=sys.stderr)

    # Write calibration JSON
    if args.out_json is None:
        out_json = infile.with_name(infile.stem + "_calibration.json")
    else:
        out_json = Path(args.out_json)
        if not out_json.is_absolute():
            out_json = _REPO_ROOT / out_json

    try:
        write_calibration_json(out_json, cal=cal, meta=meta)
    except Exception as e:
        print(f"ERROR: could not write calibration JSON: {e}", file=sys.stderr)
        return 3
    print(f"Wrote calibration JSON: {out_json}")

    # Optional calibrated CSV
    if args.write_calibrated:
        if args.out_csv is not None:
            out_csv = Path(args.out_csv)
            if not out_csv.is_absolute():
                out_csv = _REPO_ROOT / out_csv
        else:
            out_csv = infile.with_name(infile.stem + "_calibrated.csv")

        df_out = df.copy()
        df_out["Bx_cal"] = B_cal[:, 0]
        df_out["By_cal"] = B_cal[:, 1]
        df_out["Bz_cal"] = B_cal[:, 2]
        df_out["B_total_cal"] = mag_cal

        try:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(out_csv, index=False)
        except Exception as e:
            print(f"ERROR: could not write calibrated CSV: {e}", file=sys.stderr)
            return 3
        print(f"Wrote calibrated CSV: {out_csv}")

    # Plot (optional)
    if args.plot:
        if args.no_show:
            if args.plot_out is None:
                plot_out = out_json.with_suffix("").with_name(out_json.stem + "_plot.png")
            else:
                plot_out = Path(args.plot_out)
                if not plot_out.is_absolute():
                    plot_out = _REPO_ROOT / plot_out
            plot_projections(B_raw=B_raw, B_cal=B_cal, out_png=plot_out, show=False)
        else:
            plot_projections(B_raw=B_raw, B_cal=B_cal, out_png=None, show=True)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
