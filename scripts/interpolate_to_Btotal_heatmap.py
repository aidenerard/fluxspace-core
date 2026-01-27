#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interpolate B_total onto a grid and save a magnetic-detection heatmap."
    )
    p.add_argument("--in", dest="inp", required=True, help="Input CSV (typically data/processed/*_clean.csv)")
    p.add_argument("--value-col", default="B_total", help="Column to map (default: B_total)")
    p.add_argument("--grid-step", type=float, default=0.01, help="Grid spacing in meters (default: 0.01)")
    p.add_argument("--power", type=float, default=2.0, help="IDW power parameter (default: 2.0)")
    p.add_argument("--eps", type=float, default=1e-12, help="Small epsilon to avoid div-by-zero (default: 1e-12)")
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory. If empty, uses the input file's directory.",
    )
    p.add_argument(
        "--out-prefix",
        default="mag_detection",
        help="Prefix for outputs (default: mag_detection)",
    )
    p.add_argument(
        "--units",
        choices=["gauss", "uT"],
        default="gauss",
        help="Display units. If uT, values are converted from gauss to microtesla (1 G = 100 uT).",
    )
    return p.parse_args()


def idw_interpolate(
    xs: np.ndarray,
    ys: np.ndarray,
    vs: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    power: float,
    eps: float,
) -> np.ndarray:
    """
    IDW interpolation onto a meshgrid defined by gx, gy (both 2D arrays).
    """
    # Flatten grid for vectorized compute
    gxf = gx.ravel()
    gyf = gy.ravel()

    # Distances from each grid point to each sample: shape (G, N)
    dx = gxf[:, None] - xs[None, :]
    dy = gyf[:, None] - ys[None, :]
    d2 = dx * dx + dy * dy

    # If any grid point exactly matches a sample point, take that sample value directly
    exact = d2 <= eps
    out = np.empty(gxf.shape[0], dtype=float)

    if np.any(exact):
        # For each grid point, if it matches one or more samples, pick the first match
        match_idx = np.argmax(exact, axis=1)  # index of first True
        has_match = np.any(exact, axis=1)
        out[has_match] = vs[match_idx[has_match]]

        # For non-matching points, do IDW
        idx_nomatch = np.where(~has_match)[0]
        if idx_nomatch.size > 0:
            d2_nm = d2[idx_nomatch, :]
            w = 1.0 / np.power(d2_nm + eps, power / 2.0)
            out[idx_nomatch] = (w @ vs) / np.sum(w, axis=1)
    else:
        w = 1.0 / np.power(d2 + eps, power / 2.0)
        out = (w @ vs) / np.sum(w, axis=1)

    return out.reshape(gx.shape)


def main() -> int:
    args = parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    out_dir = Path(args.out_dir) if args.out_dir else inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)

    # Required columns
    for col in ("x", "y", args.value_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {inp.name}. Columns: {list(df.columns)}")

    df = df.dropna(subset=["x", "y", args.value_col]).copy()

    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)
    vs = df[args.value_col].to_numpy(dtype=float)

    # Convert units if requested (your logger stores gauss)
    label_units = "gauss"
    if args.units == "uT":
        vs = vs * 100.0  # 1 gauss = 100 microtesla
        label_units = "µT"

    # Grid bounds
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    step = float(args.grid_step)
    if step <= 0:
        raise ValueError("--grid-step must be > 0")

    xi = np.arange(xmin, xmax + step * 0.5, step)
    yi = np.arange(ymin, ymax + step * 0.5, step)
    gx, gy = np.meshgrid(xi, yi)

    print(
        f"Interpolating '{args.value_col}' onto grid: nx={len(xi)}, ny={len(yi)} "
        f"(IDW power={args.power}, step={step})..."
    )

    grid = idw_interpolate(xs, ys, vs, gx, gy, power=float(args.power), eps=float(args.eps))

    # Save grid CSV
    grid_csv = out_dir / f"{args.out_prefix}_grid.csv"
    out_df = pd.DataFrame(
        {
            "x": gx.ravel(),
            "y": gy.ravel(),
            args.value_col: grid.ravel(),
            "units": label_units,
        }
    )
    out_df.to_csv(grid_csv, index=False)

    # Save heatmap PNG
    heatmap_png = out_dir / f"{args.out_prefix}_heatmap.png"
    plt.figure()
    im = plt.imshow(
        grid,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="auto",
    )
    plt.title(f"Heatmap (IDW) of {args.value_col}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    cbar = plt.colorbar(im)
    cbar.set_label(f"{args.value_col} ({label_units})")
    plt.tight_layout()
    plt.savefig(heatmap_png, dpi=200)
    plt.close()

    print(f"Wrote grid CSV:  {grid_csv}")
    print(f"Wrote heatmap:   {heatmap_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
