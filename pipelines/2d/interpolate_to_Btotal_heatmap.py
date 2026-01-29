#!/usr/bin/env python3
"""
mag_detection_heatmap.py

Creates a "magnetic detection" heatmap from B_total (gauss) using IDW interpolation.
- Input: cleaned CSV (default: data/processed/mag_data_clean.csv)
- Output (default): data/exports/
    - mag_detection_grid.csv
    - mag_detection_heatmap.png

Goal: help you see "where the metal is" by visualizing raw field magnitude (B_total).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interpolate B_total to a smooth 2D heatmap (IDW) and export results."
    )
    p.add_argument(
        "--in",
        dest="in_path",
        default="data/processed/mag_data_clean.csv",
        help="Input CSV (cleaned). Must contain x, y, and B_total columns.",
    )
    p.add_argument(
        "--out-dir",
        dest="out_dir",
        default="data/exports",
        help="Directory to write exported outputs (grid + png).",
    )
    p.add_argument(
        "--grid-step",
        dest="grid_step",
        type=float,
        default=0.003,
        help=(
            "Interpolation grid spacing in meters. Smaller = smoother image. "
            "Recommended: 0.003–0.005 for your ~0.01834m sample spacing."
        ),
    )
    p.add_argument(
        "--power",
        dest="power",
        type=float,
        default=2.0,
        help="IDW power parameter (typical: 1.5–3.0).",
    )
    p.add_argument(
        "--epsilon",
        dest="epsilon",
        type=float,
        default=1e-9,
        help="Small value to avoid divide-by-zero in IDW.",
    )
    p.add_argument(
        "--value-col",
        dest="value_col",
        default="B_total",
        help="Column to visualize. Default is B_total.",
    )
    p.add_argument(
        "--title",
        dest="title",
        default="Heatmap (IDW) of B_total",
        help="Plot title.",
    )
    p.add_argument(
        "--no-plot",
        dest="no_plot",
        action="store_true",
        help="If set, do not generate PNG (still writes grid CSV).",
    )
    return p.parse_args()


def _grid_bounds(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    return xmin, xmax, ymin, ymax


def idw_interpolate(
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
    power: float = 2.0,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    IDW interpolation onto meshgrid (xi, yi).
    x,y,v are 1D sample points.
    xi, yi are 2D meshgrid arrays.
    Returns 2D grid values.
    """
    # Flatten grid to vector for efficient broadcasting
    gx = xi.ravel()
    gy = yi.ravel()

    # distances: (G, N)
    dx = gx[:, None] - x[None, :]
    dy = gy[:, None] - y[None, :]
    d2 = dx * dx + dy * dy

    # If a grid point is exactly at a sample point, copy that value directly
    # Mask where distance is ~0
    zero_mask = d2 < eps
    out = np.empty(gx.shape[0], dtype=float)

    # For exact hits: assign directly from the first matching sample
    if np.any(zero_mask):
        hit_rows = np.where(np.any(zero_mask, axis=1))[0]
        for r in hit_rows:
            c = np.where(zero_mask[r])[0][0]
            out[r] = v[c]

        # For remaining: standard IDW
        rem = np.ones(gx.shape[0], dtype=bool)
        rem[hit_rows] = False
    else:
        rem = np.ones(gx.shape[0], dtype=bool)

    if np.any(rem):
        d = np.sqrt(d2[rem]) + eps
        w = 1.0 / (d**power)
        wsum = np.sum(w, axis=1)
        out[rem] = (w @ v) / wsum

    return out.reshape(xi.shape)


def main() -> int:
    args = parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_csv = out_dir / "mag_detection_grid.csv"
    heatmap_png = out_dir / "mag_detection_heatmap.png"

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)

    # Required columns
    for col in ["x", "y", args.value_col]:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column '{col}' in {in_path}. "
                f"Found columns: {list(df.columns)}"
            )

    # Extract arrays
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    v = df[args.value_col].to_numpy(dtype=float)

    xmin, xmax, ymin, ymax = _grid_bounds(x, y)

    # Build interpolation grid
    step = float(args.grid_step)
    if step <= 0:
        raise ValueError("--grid-step must be > 0")

    xs = np.arange(xmin, xmax + step * 0.5, step)
    ys = np.arange(ymin, ymax + step * 0.5, step)
    xi, yi = np.meshgrid(xs, ys)

    print(f"Interpolating '{args.value_col}' onto grid: nx={len(xs)}, ny={len(ys)} "
          f"(IDW power={args.power}, grid_step={step})")

    grid = idw_interpolate(
        x=x,
        y=y,
        v=v,
        xi=xi,
        yi=yi,
        power=float(args.power),
        eps=float(args.epsilon),
    )

    # Save grid CSV
    # (x, y, value) long-format is easiest to reuse later
    grid_df = pd.DataFrame(
        {
            "x": xi.ravel(),
            "y": yi.ravel(),
            args.value_col: grid.ravel(),
        }
    )
    grid_df.to_csv(grid_csv, index=False)
    print(f"Wrote grid CSV: {grid_csv}")

    if not args.no_plot:
        plt.figure()
        plt.title(args.title)

        # Key change for “not pixelated”: bilinear interpolation on render
        im = plt.imshow(
            grid,
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            aspect="equal",
            interpolation="bilinear",
        )
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        cbar = plt.colorbar(im)
        cbar.set_label(f"{args.value_col} (gauss)" if args.value_col.lower() == "b_total" else args.value_col)

        plt.tight_layout()
        plt.savefig(heatmap_png, dpi=200)
        plt.close()
        print(f"Wrote heatmap PNG: {heatmap_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
