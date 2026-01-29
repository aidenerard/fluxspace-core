#!/usr/bin/env python3
"""
interpolate_to_heatmapv1.py

Takes scattered points (x, y, value) and interpolates them onto a regular grid,
then exports:
  - <stem>_grid.csv          (x,y,value on a grid, long-form table)
  - <stem>_heatmap.png       (quick preview)

Designed for your pipeline:
  validate_and_diagnostics.py  -> *_clean.csv
  compute_local_anomaly_v2.py  -> *_anomaly.csv
  interpolate_to_heatmapv1.py  -> grid + heatmap

Default interpolation: IDW (Inverse Distance Weighting) with a tunable power.

Example:
  python3 interpolate_to_heatmapv1.py --in data/processed/mag_data_anomaly.csv --value-col local_anomaly

If your x/y are in meters and you want 0.05 m grid spacing:
  python3 interpolate_to_heatmapv1.py --in ... --grid-step 0.05

Notes:
- Lightweight interpolator (no SciPy required).
- For dense grids over large datasets, naive IDW can be slow (O(N * grid_points)).
  For your 9x9 or 20x20 tests, it’s totally fine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interpolate scattered x,y,value points to a grid and save grid CSV + heatmap."
    )
    p.add_argument(
        "--in",
        dest="infile",
        required=True,
        help="Input CSV (e.g., data/processed/mag_data_anomaly.csv)",
    )
    p.add_argument(
        "--value-col",
        default="local_anomaly",
        help="Column to grid (default: local_anomaly)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. If omitted, defaults to data/exports when possible.",
    )
    p.add_argument(
        "--grid-step",
        type=float,
        default=None,
        help="Grid spacing in same units as x/y (e.g., 0.05). If omitted, uses --grid-n.",
    )
    p.add_argument(
        "--grid-n",
        type=int,
        default=200,
        help="Grid resolution per axis if --grid-step not given (default: 200).",
    )
    p.add_argument(
        "--power",
        type=float,
        default=2.0,
        help="IDW power parameter (default: 2.0). Higher = more local influence.",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Small epsilon to avoid divide-by-zero (default: 1e-12).",
    )
    p.add_argument(
        "--clip-percentile",
        type=float,
        default=99.0,
        help="Clip colormap to [100-clip, clip] percentiles for nicer plots (default: 99). "
        "Use 100 to disable clipping.",
    )
    p.add_argument(
        "--drop-flag-any",
        action="store_true",
        help="If set, drop rows where _flag_any is True (if present).",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="If set, do not generate the PNG heatmap (still writes the grid CSV).",
    )
    return p.parse_args()


# -----------------------------
# Grid helper
# -----------------------------
def make_grid_axes(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    grid_step: Optional[float],
    grid_n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns gx, gy 1D arrays defining the target grid axes.
    - If grid_step is provided, uses np.arange with a small inclusive buffer.
    - Otherwise uses np.linspace with grid_n points per axis.
    """
    if grid_step is not None and grid_step > 0:
        # Include the max edge (with a tiny buffer so endpoint isn't missed due to float rounding)
        gx = np.arange(xmin, xmax + grid_step * 0.5, grid_step)
        gy = np.arange(ymin, ymax + grid_step * 0.5, grid_step)
    else:
        gx = np.linspace(xmin, xmax, grid_n)
        gy = np.linspace(ymin, ymax, grid_n)
    return gx, gy


def resolve_outdir(infile: Path, out_dir_arg: Optional[str]) -> Path:
    """
    Resolve output directory.
    Priority:
      1) --out-dir (explicit)
      2) If path contains a 'data' directory, use <...>/data/exports
      3) Otherwise fallback to <input_dir>/exports
    """
    if out_dir_arg:
        return Path(out_dir_arg)

    parts = infile.resolve().parts
    if "data" in parts:
        data_root = Path(*parts[: parts.index("data") + 1])  # .../data
        return data_root / "exports"

    # fallback if not in a project layout
    return infile.parent / "exports"


# -----------------------------
# IDW interpolation
# -----------------------------
def idw_grid(
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    power: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Inverse Distance Weighting interpolation.
    Returns grid Z with shape (len(gy), len(gx)), where rows correspond to gy (y-axis).
    """
    Xg, Yg = np.meshgrid(gx, gy)  # (ny, nx)

    tx = Xg.ravel()
    ty = Yg.ravel()

    Z = np.empty(tx.size, dtype=float)

    # Note: d2 is squared distance; using power/2 makes it equivalent to 1/d^power.
    for i in range(tx.size):
        dx = tx[i] - x
        dy = ty[i] - y
        d2 = dx * dx + dy * dy

        # If target is (numerically) on top of a source point, copy that exact value.
        j0 = int(np.argmin(d2))
        if d2[j0] <= eps:
            Z[i] = v[j0]
            continue

        # weights: 1 / (d^power) but computed via squared distances:
        # d^power == (d^2)^(power/2) == d2^(power/2)
        w = 1.0 / (d2 ** (power / 2.0) + eps)
        Z[i] = float(np.sum(w * v) / np.sum(w))

    return Z.reshape(Yg.shape)


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    args = parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        print(f"ERROR: input file not found: {infile}", file=sys.stderr)
        return 2

    try:
        df = pd.read_csv(infile)
    except Exception as e:
        print(f"ERROR: could not read CSV: {e}", file=sys.stderr)
        return 2

    # Required columns
    needed = ["x", "y", args.value_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(
            f"ERROR: missing required column(s): {missing}. Columns found: {list(df.columns)}",
            file=sys.stderr,
        )
        return 2

    # Optional drop flagged rows
    if args.drop_flag_any and "_flag_any" in df.columns:
        before = len(df)
        # Make sure it's boolean-ish, then invert mask to keep unflagged
        mask = ~df["_flag_any"].astype(bool)
        df = df.loc[mask].copy()
        print(f"Note: dropped {before - len(df)} rows where _flag_any == True.")

    # Coerce numeric columns; bad values -> NaN
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df[args.value_col] = pd.to_numeric(df[args.value_col], errors="coerce")

    # Drop rows missing x/y/value
    df = df.dropna(subset=["x", "y", args.value_col]).copy()
    if len(df) == 0:
        print("ERROR: no valid rows after dropping NaNs in x/y/value.", file=sys.stderr)
        return 2

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    v = df[args.value_col].to_numpy(dtype=float)

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    # Output directory (exports by default)
    outdir = resolve_outdir(infile, args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    stem = infile.stem
    grid_csv = outdir / f"{stem}_grid.csv"
    heatmap_png = outdir / f"{stem}_heatmap.png"

    # Build grid axes
    gx, gy = make_grid_axes(xmin, xmax, ymin, ymax, args.grid_step, args.grid_n)

    print(
        f"Interpolating '{args.value_col}' onto grid: nx={len(gx)}, ny={len(gy)} "
        f"(IDW power={args.power})"
    )

    # Compute interpolated grid
    Z = idw_grid(x, y, v, gx, gy, power=args.power, eps=args.eps)

    # Export grid CSV as long-form table (x,y,value)
    # (This rebuilds Xg/Yg only for labeling the grid with coordinates.)
    Xg, Yg = np.meshgrid(gx, gy)
    out_df = pd.DataFrame(
        {
            "x": Xg.ravel(),
            "y": Yg.ravel(),
            args.value_col: Z.ravel(),
        }
    )

    try:
        out_df.to_csv(grid_csv, index=False)
    except Exception as e:
        print(f"ERROR: could not write grid CSV: {e}", file=sys.stderr)
        return 3

    # Heatmap plot (optional)
    if not args.no_plot:
        clip = float(args.clip_percentile)
        if 0 < clip < 100:
            lo = np.nanpercentile(Z, 100 - clip)
            hi = np.nanpercentile(Z, clip)
            vmin, vmax = float(lo), float(hi)
        else:
            vmin, vmax = None, None

        plt.figure()
        im = plt.imshow(
            Z,
            origin="lower",
            extent=[float(gx.min()), float(gx.max()), float(gy.min()), float(gy.max())],
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(im, label=args.value_col)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Heatmap (IDW) of {args.value_col}")
        plt.tight_layout()

        try:
            plt.savefig(heatmap_png, dpi=160)
            plt.close()
        except Exception as e:
            print(f"ERROR: could not save heatmap PNG: {e}", file=sys.stderr)
            return 3

    print(f"Wrote grid CSV: {grid_csv}")
    if not args.no_plot:
        print(f"Wrote heatmap:  {heatmap_png}")
    else:
        print("Note: --no-plot set, so no PNG was generated.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
